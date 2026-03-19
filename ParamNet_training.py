"""Training script for ParamNet with physics-informed losses and uncertainty estimation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
import os
import copy

from torch.nn.parameter import UninitializedParameter


@dataclass
class TrainingConfig:
    train_file: str = 'Training_data_batch.npz'
    val_file: str = 'Var_data_batch.npz'
    Test_file: str = 'Test_data_batch.npz'
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    hidden_dim: int = 256
    dropout: float = 0.2
    patience: int = 10
    save_dir: str = './checkpoints'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_interval: int = 10
    window_size: int = 200
    seq_overlap: int = 100
    physics_weight: float = 0.5
    physics_warmup_epochs: int = 30
    grad_accum_steps: int = 1
    lookahead_steps: int = 5
    lookahead_alpha: float = 0.5
    grad_clip: float = 1.0
    num_workers: int = 2
    include_pressure_aux: bool = True
    # Std equivalent of multiplicative uniform pressure jitter U[-0.3, 0.3].
    pressure_window_jitter_rel: float = 0.1732
    pressure_window_jitter_abs: float = 0.0
    pressure_jitter_std: float = 0.0
    pressure_dropout_prob: float = 0.0
    pressure_shuffle_prob: float = 0.0
    ts_noise_std: float = 0.01
    ts_gain_jitter: float = 0.10
    gamma_p_corr_weight: float = 0.08
    physics_peak_epoch_ratio: float = 0.35
    physics_min_ratio: float = 0.15
    
class BrownianDataset(Dataset):
    def __init__(
        self,
        npz_file,
        window_size=200,
        overlap=100,
        is_train=False,
        include_pressure_aux=False,
        pressure_window_jitter_rel=0.0,
        pressure_window_jitter_abs=0.0,
        pressure_jitter_std=0.0,
        pressure_dropout_prob=0.0,
        pressure_shuffle_prob=0.0,
        ts_noise_std=0.0,
        ts_gain_jitter=0.0,
    ):
        data = np.load(npz_file)
        self.is_train = bool(is_train)
        self.include_pressure_aux = bool(include_pressure_aux)
        self.pressure_window_jitter_rel = float(max(0.0, pressure_window_jitter_rel))
        self.pressure_window_jitter_abs = float(max(0.0, pressure_window_jitter_abs))
        self.pressure_jitter_std = float(max(0.0, pressure_jitter_std))
        self.pressure_dropout_prob = float(np.clip(pressure_dropout_prob, 0.0, 1.0))
        self.pressure_shuffle_prob = float(np.clip(pressure_shuffle_prob, 0.0, 1.0))
        self.ts_noise_std = float(max(0.0, ts_noise_std))
        self.ts_gain_jitter = float(max(0.0, ts_gain_jitter))

        self.position_raw = data["position"].astype(np.float32)
        self.k0_log = np.log10(data["k0"]).astype(np.float32)
        self.gamma_log = np.log10(data["gamma"]).astype(np.float32)
        self.D_lin = data["D"].astype(np.float32)
        self.D_log = np.log10(np.clip(self.D_lin, 1e-30, None)).astype(np.float32)
        self.P = data["P"].astype(np.float32)
        self.P_log = np.log10(np.clip(self.P, 1e-30, None)).astype(np.float32)
        self.m = data["m"].astype(np.float32)
        self.T = data["T"].astype(np.float32)
        self.T = self.T + np.random.normal(0, 0.05, size=self.T.shape).astype(np.float32) * self.T

        fs_raw = data["fs"]
        if np.ndim(fs_raw) == 0:
            self.fs_values = np.full(self.position_raw.shape[0], float(fs_raw), dtype=np.float32)
        else:
            self.fs_values = fs_raw.astype(np.float32)

        self.window_size = int(window_size)
        self.stride = max(1, self.window_size - int(overlap))
        self.num_samples, self.time_points = self.position_raw.shape
        self.windows_per_sample = max(1, (self.time_points - self.window_size) // self.stride + 1)
        self.total_windows = self.num_samples * self.windows_per_sample

        # Normalize each full trajectory once, then slice windows from it.
        self.position = np.empty_like(self.position_raw)
        self.velocity = np.empty_like(self.position_raw)
        for i in range(self.num_samples):
            dt = 1.0 / self.fs_values[i]
            vel_raw = np.gradient(self.position_raw[i], dt)

            pos_mu, pos_std = np.mean(self.position_raw[i]), np.std(self.position_raw[i])
            vel_mu, vel_std = np.mean(vel_raw), np.std(vel_raw)
            pos_std = max(float(pos_std), 1e-8)
            vel_std = max(float(vel_std), 1e-8)

            self.position[i] = (self.position_raw[i] - pos_mu) / pos_std
            self.velocity[i] = (vel_raw - vel_mu) / vel_std

        self.position = self.position.astype(np.float32)
        self.velocity = self.velocity.astype(np.float32)

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        sample_idx = idx // self.windows_per_sample
        window_idx = idx % self.windows_per_sample
        start = window_idx * self.stride
        end = start + self.window_size

        pos_norm = self.position[sample_idx, start:end]
        vel_norm = self.velocity[sample_idx, start:end]
        pos_raw = self.position_raw[sample_idx, start:end]
        dt = 1.0 / self.fs_values[sample_idx]
        vel_raw = np.gradient(pos_raw, dt)

        time_series = np.stack([pos_norm, vel_norm], axis=0).astype(np.float32)

        if self.is_train:
            if self.ts_gain_jitter > 0.0:
                gain = 1.0 + np.random.uniform(-self.ts_gain_jitter, self.ts_gain_jitter)
                time_series = time_series * np.float32(gain)
            if self.ts_noise_std > 0.0:
                time_series = time_series + np.random.normal(
                    0.0, self.ts_noise_std, size=time_series.shape
                ).astype(np.float32)

        # Pressure is used as an auxiliary prior, with optional anti-shortcut noise.
        p_log = np.float32(self.P_log[sample_idx])
        p_aux = np.float32(0.0)
        if self.include_pressure_aux:
            p_aux = p_log
            if self.is_train:
                p_true = float(self.P[sample_idx])
                p_sigma = abs(p_true) * self.pressure_window_jitter_rel + self.pressure_window_jitter_abs
                if p_sigma > 0.0:
                    p_noisy = p_true + np.random.normal(0.0, p_sigma)
                    p_noisy = float(np.clip(p_noisy, 1e-30, None))
                    p_aux = np.float32(np.log10(p_noisy))
                if np.random.rand() < self.pressure_shuffle_prob:
                    ridx = np.random.randint(0, len(self.P_log))
                    p_aux = np.float32(self.P_log[ridx])
                if self.pressure_jitter_std > 0.0:
                    p_aux = np.float32(p_aux + np.random.normal(0.0, self.pressure_jitter_std))
                if np.random.rand() < self.pressure_dropout_prob:
                    p_aux = np.float32(0.0)

        aux_features = np.array([
            np.log10(self.T[sample_idx]),
            np.log10(np.clip(np.var(pos_raw), 1e-30, None)),
            np.log10(self.m[sample_idx]),
            np.log10(np.clip(np.var(vel_raw), 1e-30, None)),
            np.log10(dt),
            p_aux,
        ], dtype=np.float32)

        targets = np.array([self.k0_log[sample_idx], self.gamma_log[sample_idx]], dtype=np.float32)

        return (
            torch.from_numpy(time_series),
            torch.from_numpy(aux_features),
            torch.from_numpy(targets),
            torch.from_numpy(pos_raw.astype(np.float32)),
            torch.tensor(self.m[sample_idx], dtype=torch.float32),
            torch.tensor(dt, dtype=torch.float32),
            torch.tensor(p_log, dtype=torch.float32),
        )
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, s=1, d=1):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=(k//2)*d,
                            dilation=d, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class SE1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        h = max(1, ch // r)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(ch, h, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(h, ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class TemporalMSResBlock(nn.Module):
    def __init__(self, ch, kernels=(3, 5, 7), dilation_base=1, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(ch)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                ch,
                ch,
                kernel_size=k,
                padding=(k // 2) * (dilation_base * i),
                dilation=dilation_base * i,
                groups=ch,
                bias=False
            )
            for i, k in enumerate(kernels, start=1)
        ])
        self.pointwise = nn.Conv1d(ch * len(kernels), ch, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        y = self.norm(x)
        y = torch.cat([conv(y) for conv in self.convs], dim=1)
        y = self.pointwise(y)
        y = self.dropout(self.act(y))
        return residual + y


class TemporalMSResStack(nn.Module):
    def __init__(self, in_ch=2, hidden_ch=128, depth=3, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            TemporalMSResBlock(hidden_ch, dilation_base=2 ** i, dropout=dropout)
            for i in range(depth)
        ])
        self.se = SE1D(hidden_ch, r=8)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.se(x)


class CrossDomainAttention(nn.Module):
    def __init__(self, time_dim, freq_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.joint = nn.Sequential(
            nn.Linear(time_dim + freq_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.t_gate = nn.Sequential(
            nn.Linear(hidden, time_dim),
            nn.Sigmoid()
        )
        self.f_gate = nn.Sequential(
            nn.Linear(hidden, freq_dim),
            nn.Sigmoid()
        )

    def forward(self, t_feat, f_feat):
        joint = self.joint(torch.cat([t_feat, f_feat], dim=-1))
        t_refined = t_feat * self.t_gate(joint)
        f_refined = f_feat * self.f_gate(joint)
        return t_refined, f_refined

class ParamNetV2(nn.Module):
    """
        Dual-branch regressor.
        - Time branch: multi-scale residual temporal encoder.
        - Frequency branch: FFT-derived summary features.
        Outputs means and heteroscedastic variances for log10(k) and log10(gamma).
    """
    def __init__(self, window_size=200, aux_dim=6, hid=192, dropout=0.1, topK=60):
        super().__init__()
        # Keep feature size stable across different window lengths.
        self.topK = min(int(topK), int(window_size) // 2 + 1)

        t_hidden = 128
        self.time_dim = t_hidden
        self.t_encoder = TemporalMSResStack(in_ch=2, hidden_ch=t_hidden, depth=3, dropout=dropout)
        self.t_pool = nn.AdaptiveAvgPool1d(1)

        self.freq_feature_dim = self.topK * 3 + 13
        self.freq_dim = hid // 2
        self.f_fc = nn.Sequential(
            nn.Linear(self.freq_feature_dim, hid),
            nn.LayerNorm(hid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, self.freq_dim),
            nn.SiLU()
        )

        lag_pool = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
        self.acf_lags = [l for l in lag_pool if l < window_size]
        if len(self.acf_lags) == 0:
            self.acf_lags = [1]
        self.acf_feature_dim = len(self.acf_lags) * 2 + 3
        self.acf_dim = hid // 3
        self.acf_fc = nn.Sequential(
            nn.Linear(self.acf_feature_dim, hid // 2),
            nn.LayerNorm(hid // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hid // 2, self.acf_dim),
            nn.SiLU()
        )

        self.cross_attention = CrossDomainAttention(self.time_dim, self.freq_dim, hidden=hid, dropout=dropout)

        self.head_pre = nn.Sequential(
            nn.Linear(self.time_dim + self.freq_dim + self.acf_dim + aux_dim, hid),
            nn.LayerNorm(hid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid)
        )

        self.mean_head = nn.Linear(hid, 2)
        self.logvar_head = nn.Linear(hid, 2)

    @staticmethod
    def _resample_lastdim(x, target_len):
        """Resample the last dimension to a fixed length."""
        if x.shape[-1] == target_len:
            return x
        if x.dim() == 3:
            return F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        if x.dim() == 2:
            return F.interpolate(x.unsqueeze(1), size=target_len, mode='linear', align_corners=False).squeeze(1)
        raise ValueError(f"Unsupported tensor rank for resample: {x.dim()}")

    def _freq_features(self, x):
        B, C, W = x.shape
        X = torch.fft.rfft(x, dim=-1)
        mag = torch.abs(X).clamp_min(1e-16)
        pow_spec = mag ** 2
        freqs = torch.linspace(0, 0.5, X.shape[-1], device=x.device, dtype=x.dtype)

        centroid = (pow_spec * freqs).sum(dim=-1) / (pow_spec.sum(dim=-1) + 1e-16)
        peak_idx = pow_spec.argmax(dim=-1)
        peak_freq = freqs[peak_idx]
        energy = pow_spec.sum(dim=-1)
        var = ((freqs - centroid.unsqueeze(-1)) ** 2 * pow_spec).sum(dim=-1) / (energy + 1e-16)

        cross = X[:, 0] * torch.conj(X[:, 1])
        cross_mag = torch.abs(cross)
        coherence = (cross_mag.pow(2) / (pow_spec[:, 0] * pow_spec[:, 1] + 1e-16)).clamp(0.0, 1.0)
        coh_mean = coherence.mean(dim=-1, keepdim=True)
        coh_max = coherence.max(dim=-1, keepdim=True)[0]
        phase = torch.angle(cross)
        phase_cos = torch.cos(phase).mean(dim=-1, keepdim=True)
        phase_sin = torch.sin(phase).mean(dim=-1, keepdim=True)
        cross_power = cross_mag.mean(dim=-1, keepdim=True)

        # Resample spectra so FC input size does not depend on window length.
        mag_k = self._resample_lastdim(mag, self.topK)
        coherence_k = self._resample_lastdim(coherence, self.topK)

        feat = torch.cat([
            mag_k.reshape(B, -1),
            coherence_k.reshape(B, -1),
            centroid, peak_freq, energy, var,
            coh_mean, coh_max, phase_cos, phase_sin, cross_power
        ], dim=-1)
        return feat

    def _acf_features(self, x):
        pos = x[:, 0, :]
        vel = x[:, 1, :]

        pos = pos - pos.mean(dim=1, keepdim=True)
        vel = vel - vel.mean(dim=1, keepdim=True)
        pos_var = pos.pow(2).mean(dim=1, keepdim=True) + 1e-8
        vel_var = vel.pow(2).mean(dim=1, keepdim=True) + 1e-8

        pos_acf = []
        vel_acf = []
        for lag in self.acf_lags:
            pos_acf.append((pos[:, :-lag] * pos[:, lag:]).mean(dim=1, keepdim=True) / pos_var)
            vel_acf.append((vel[:, :-lag] * vel[:, lag:]).mean(dim=1, keepdim=True) / vel_var)

        pos_acf = torch.cat(pos_acf, dim=1)
        vel_acf = torch.cat(vel_acf, dim=1)

        zcr_pos = (pos[:, 1:] * pos[:, :-1] < 0).float().mean(dim=1, keepdim=True)
        zcr_vel = (vel[:, 1:] * vel[:, :-1] < 0).float().mean(dim=1, keepdim=True)
        pv_corr = (pos * vel).mean(dim=1, keepdim=True) / torch.sqrt(pos_var * vel_var)

        return torch.cat([pos_acf, vel_acf, zcr_pos, zcr_vel, pv_corr], dim=1)

    def forward(self, x, aux):
        t_encoded = self.t_encoder(x)
        tfeat = self.t_pool(t_encoded).squeeze(-1)
        ffeat_raw = self._freq_features(x)
        ffeat = self.f_fc(ffeat_raw)
        acf_raw = self._acf_features(x)
        acf_feat = self.acf_fc(acf_raw)
        tfeat, ffeat = self.cross_attention(tfeat, ffeat)
        fused = torch.cat([tfeat, ffeat, acf_feat, aux], dim=1)
        z = self.head_pre(fused)
        mean = self.mean_head(z)
        log_k, log_gamma = mean[:, 0], mean[:, 1]
        logvar = self.logvar_head(z).clamp(min=-8.0, max=4.0)
        s2_k, s2_g = torch.exp(logvar[:, 0]), torch.exp(logvar[:, 1])
        return log_k, log_gamma, s2_k, s2_g


def _batch_corrcoef(x, y, eps=1e-8):
    x = x.float().view(-1)
    y = y.float().view(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).mean() + eps) * (y.pow(2).mean() + eps))
    return (x * y).mean() / denom


def gamma_pressure_decorrelation_loss(log_gamma_pred, p_log):
    corr = _batch_corrcoef(log_gamma_pred, p_log)
    return corr.pow(2), corr.detach()
class Lookahead:
    def __init__(self, optimizer, alpha=0.5, k=5):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("Lookahead alpha should be in (0, 1].")
        if k < 1:
            raise ValueError("Lookahead k should be >= 1.")
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self._step = 0
        self._slow_params = []
        for group in optimizer.param_groups:
            for p in group['params']:
                self._slow_params.append(p.detach().clone())
        self.reset()

    def reset(self):
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self._slow_params[idx].copy_(p.data)
                idx += 1

    def step(self):
        self._step += 1
        if self._step % self.k != 0:
            return
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    idx += 1
                    continue
                slow = self._slow_params[idx]
                slow.add_(p.data - slow, alpha=self.alpha)
                p.data.copy_(slow)
                idx += 1

    def state_dict(self):
        return {
            "slow_params": [sp.clone() for sp in self._slow_params],
            "step": self._step,
            "alpha": self.alpha,
            "k": self.k
        }

    def load_state_dict(self, state_dict):
        self.alpha = state_dict.get("alpha", self.alpha)
        self.k = state_dict.get("k", self.k)
        self._step = state_dict.get("step", 0)
        slow_params = state_dict.get("slow_params", [])
        if slow_params:
            if len(slow_params) != len(self._slow_params):
                raise ValueError("Lookahead state size mismatch.")
            for dst, src in zip(self._slow_params, slow_params):
                dst.copy_(src)

def ar2_physics_loss(log_k, log_gamma, pos_raw, mass, dt):
    log_k = log_k.float().clamp(-20, 15)
    log_gamma = log_gamma.float().clamp(-20, 15)
    k = torch.pow(10.0, log_k)
    g = torch.pow(10.0, log_gamma)
    m = mass
    dt = dt.clamp_min(1e-12)

    rho = torch.exp(-g * dt / (2.0 * m)).clamp(1e-6, 1-1e-6)
    damp = g / (2.0*m)
    omega2 = (k/m - damp**2).clamp(min=0.0)
    theta = torch.sqrt(omega2) * dt
    theta = theta.clamp(1e-6, math.pi-1e-6)

    x = pos_raw - pos_raw.mean(dim=1, keepdim=True)
    res = x[:,2:] - 2.0*rho[:,None]*torch.cos(theta)[:,None]*x[:,1:-1] + (rho**2)[:,None]*x[:,:-2]
    energy = x.pow(2).mean(dim=1) + 1e-18
    return (res.pow(2).mean(dim=1)/energy).mean()

def physics_loss_bundle(log_k, log_gamma, pos_raw, mass, temp, dt, w_ar=1.0):
    L_ar = ar2_physics_loss(log_k, log_gamma, pos_raw, mass, dt)
    return w_ar * L_ar, (L_ar.item(),)
def nll_gauss_2d(logk_pred, logg_pred, s2k, s2g, tgt_logk, tgt_logg):
    # Use fp32 for numerical stability under autocast.
    logk_pred, logg_pred = logk_pred.float(), logg_pred.float()
    s2k, s2g = s2k.float(), s2g.float()
    tgt_logk, tgt_logg = tgt_logk.float(), tgt_logg.float()

    term_k = 0.5*(torch.log(s2k + 1e-18) + (logk_pred - tgt_logk).pow(2)/(s2k + 1e-18))
    term_g = 0.5*(torch.log(s2g + 1e-18) + (logg_pred - tgt_logg).pow(2)/(s2g + 1e-18))
    return (term_k + term_g).mean()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        # Track EMA for floating-point tensors only.
        for k, v in model.state_dict().items():
            if isinstance(v, UninitializedParameter):
                continue
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k] = v.clone()

    @torch.no_grad()
    def _ensure_init(self, model):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                if isinstance(v, UninitializedParameter):
                    continue
                self.shadow[k] = v.detach().clone() if torch.is_floating_point(v) else v.clone()

    @torch.no_grad()
    def update(self, model):
        self._ensure_init(model)
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone() if torch.is_floating_point(v) else v.clone()
                continue

            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                if self.shadow[k].dtype != v.dtype or self.shadow[k].shape != v.shape:
                    self.shadow[k] = v.clone()
                else:
                    self.shadow[k].copy_(v)

    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

def train_epoch(model, loader, optimizer, device, physics_weight=0.0,
                scaler=None, ema=None, grad_accum_steps=1, grad_clip=1.0,
                lookahead=None, gamma_p_corr_weight=0.0):
    model.train()
    total = 0.0
    valid_batches = 0
    progress = tqdm(loader, desc="Train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = max(1, grad_accum_steps)
    has_valid_grad = False
    for step, batch in enumerate(progress):
        time_series, aux, targets, pos_raw, mass, dt, p_log = batch
        time_series, aux, targets = time_series.to(device), aux.to(device), targets.to(device)
        pos_raw, mass, dt = pos_raw.to(device), mass.to(device), dt.to(device)
        p_log = p_log.to(device)

        tgt_logk, tgt_logg = targets[:, 0], targets[:, 1]

        amp_device = 'cuda' if str(device).startswith('cuda') else 'cpu'
        with torch.amp.autocast(device_type=amp_device, enabled=(scaler is not None)):
            log_k, log_g, s2k, s2g = model(time_series, aux)
            loss_nll = nll_gauss_2d(log_k, log_g, s2k, s2g, tgt_logk, tgt_logg)

            loss_phys_value = 0.0
            if physics_weight > 0.0:
                temp_lin = torch.pow(10.0, aux[:, 0])
                loss_phys, _ = physics_loss_bundle(
                    log_k, log_g, pos_raw, mass, temp_lin, dt,
                    w_ar=1.0,
                )
                loss_phys_value = float(loss_phys.detach().cpu())
                loss = loss_nll + physics_weight * loss_phys
            else:
                loss = loss_nll

            loss_corr_value = 0.0
            if gamma_p_corr_weight > 0.0:
                loss_corr, corr_now = gamma_pressure_decorrelation_loss(log_g, p_log)
                loss_corr_value = float(loss_corr.detach().cpu())
                loss = loss + gamma_p_corr_weight * loss_corr

        if torch.isfinite(loss):
            loss_scaled = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            has_valid_grad = True
            total += loss.item()

        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
        if should_step and has_valid_grad:
            if scaler is not None:
                scaler.unscale_(optimizer)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if lookahead is not None:
                lookahead.step()

            if ema is not None:
                ema.update(model)

            optimizer.zero_grad(set_to_none=True)
            has_valid_grad = False

        if torch.isfinite(loss):
            valid_batches += 1
            progress.set_postfix({
                "nll": f"{loss_nll.item():.4f}",
                "loss": f"{loss.item():.4f}",
                "loss_phys": f"{loss_phys_value:.4e}",
                "loss_corr": f"{loss_corr_value:.4e}"
            })
        else:
            progress.set_postfix({
                "nll": "NaN",
                "loss": "NaN",
                "loss_phys": "NaN",
                "loss_corr": "NaN"
            })

    return total / max(1, valid_batches)

@torch.no_grad()
def validate(model, loader, device, physics_weight=0.0, gamma_p_corr_weight=0.0):
    model.eval()
    total = 0.0
    k_errs, g_errs = [], []
    corr_values = []
    for batch in loader:
        time_series, aux, targets, pos_raw, mass, dt, p_log = batch
        time_series, aux, targets = time_series.to(device), aux.to(device), targets.to(device)
        pos_raw, mass, dt = pos_raw.to(device), mass.to(device), dt.to(device)
        p_log = p_log.to(device)

        tgt_logk, tgt_logg = targets[:,0], targets[:,1]
        log_k, log_g, s2k, s2g = model(time_series, aux)
        loss_nll = nll_gauss_2d(log_k, log_g, s2k, s2g, tgt_logk, tgt_logg)

        if physics_weight > 0.0:
            temp_lin = torch.pow(10.0, aux[:,0])
            loss_phys, _ = physics_loss_bundle(log_k, log_g, pos_raw, mass, temp_lin, dt,
                                               w_ar=1.0)
            loss = loss_nll + physics_weight * loss_phys
        else:
            loss = loss_nll

        if gamma_p_corr_weight > 0.0:
            loss_corr, corr_now = gamma_pressure_decorrelation_loss(log_g, p_log)
            loss = loss + gamma_p_corr_weight * loss_corr
            corr_values.append(float(torch.abs(corr_now).detach().cpu()))

        total += loss.item()

        k_rel = (torch.abs(10**log_k - 10**tgt_logk) / (10**tgt_logk)).detach().cpu().numpy()
        g_rel = (torch.abs(10**log_g - 10**tgt_logg) / (10**tgt_logg)).detach().cpu().numpy()
        k_errs.append(k_rel); g_errs.append(g_rel)

    corr_mean = float(np.mean(corr_values)) if len(corr_values) > 0 else 0.0
    return total/len(loader), np.concatenate(k_errs).mean(), np.concatenate(g_errs).mean(), corr_mean
def train_model(config):

    print(f"Device: {config.device}")
    worker_count = max(0, config.num_workers)
    val_worker_count = max(0, config.num_workers // 2) if config.num_workers > 1 else worker_count
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    train_dataset = BrownianDataset(
        config.train_file, 
        window_size=config.window_size, 
        overlap=config.seq_overlap,
        is_train=True,
        include_pressure_aux=config.include_pressure_aux,
        pressure_window_jitter_rel=config.pressure_window_jitter_rel,
        pressure_window_jitter_abs=config.pressure_window_jitter_abs,
        pressure_jitter_std=config.pressure_jitter_std,
        pressure_dropout_prob=config.pressure_dropout_prob,
        pressure_shuffle_prob=config.pressure_shuffle_prob,
        ts_noise_std=config.ts_noise_std,
        ts_gain_jitter=config.ts_gain_jitter,
    )
    val_dataset = BrownianDataset(
        config.val_file, 
        window_size=config.window_size, 
        overlap=config.seq_overlap,
        is_train=False,
        include_pressure_aux=config.include_pressure_aux,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=(config.device == 'cuda'),
        persistent_workers=worker_count > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=val_worker_count,
        pin_memory=(config.device == 'cuda'),
        persistent_workers=val_worker_count > 0
    )

    model = ParamNetV2(window_size=config.window_size, aux_dim=6,
                   hid=config.hidden_dim, dropout=config.dropout).to(config.device)

    base_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lookahead = Lookahead(base_optimizer, alpha=config.lookahead_alpha, k=config.lookahead_steps) if config.lookahead_steps > 0 else None

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == 'cuda'))
    ema = EMA(model, decay=0.999)
    best_val = float('inf'); patience_counter = 0; train_losses = []; val_losses = []
    for epoch in range(1, config.epochs + 1):
        print(f"\n=== Epoch {epoch}/{config.epochs} ===")
        if config.physics_warmup_epochs <= 0:
            peak_epoch = max(1, int(config.epochs * config.physics_peak_epoch_ratio))
            if epoch <= peak_epoch:
                up = epoch / peak_epoch
                phys_w = 0.5 * config.physics_weight * (1.0 - math.cos(math.pi * up))
            else:
                down = (epoch - peak_epoch) / max(1, config.epochs - peak_epoch)
                tail = config.physics_min_ratio + (1.0 - config.physics_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * down))
                phys_w = config.physics_weight * tail
        else:
            warm_progress = min(1.0, epoch / max(1, config.physics_warmup_epochs))
            warm_w = 0.5 * config.physics_weight * (1.0 - math.cos(math.pi * warm_progress))
            decay_progress = min(1.0, max(0.0, (epoch - config.physics_warmup_epochs) / max(1, config.epochs - config.physics_warmup_epochs)))
            decay_tail = config.physics_min_ratio + (1.0 - config.physics_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            phys_w = min(warm_w, config.physics_weight * decay_tail)

        tr = train_epoch(
            model,
            train_loader,
            base_optimizer,
            config.device,
            physics_weight=phys_w,
            scaler=scaler,
            ema=ema,
            grad_accum_steps=config.grad_accum_steps,
            grad_clip=config.grad_clip,
            lookahead=lookahead,
            gamma_p_corr_weight=config.gamma_p_corr_weight,
        )

        ema_backup = None
        if ema is not None:
            ema_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)

        va, k_err, g_err, gp_corr = validate(
            model,
            val_loader,
            config.device,
            physics_weight=phys_w,
            gamma_p_corr_weight=config.gamma_p_corr_weight,
        )

        train_losses.append(tr); val_losses.append(va)
        scheduler.step(epoch)
        print(
            f"Train: {tr:.6f} | Val: {va:.6f} | phys_w:{phys_w:.3f} | "
            f"k:{k_err*100:.2f}% γ:{g_err*100:.2f}% | |corr(logγ,logP)|:{gp_corr:.3f}"
        )

        if va < best_val:
            best_val = va; patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': base_optimizer.state_dict(),
                'val_loss': va,
                'k_error': k_err,
                'gamma_error': g_err,
                'gamma_pressure_abs_corr': gp_corr,
                'ema_state_dict': copy.deepcopy(ema.shadow) if ema is not None else None,
                'lookahead_state_dict': lookahead.state_dict() if lookahead is not None else None,
                'scheduler_state_dict': scheduler.state_dict()
            }, f"{config.save_dir}/best_model.pth")
            print("Saved best checkpoint (EMA weights).")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping: no improvement for {config.patience} epochs.")
                break

        if ema_backup is not None:
            model.load_state_dict(ema_backup, strict=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{config.save_dir}/loss_curve.png")
    plt.show()
    
    checkpoint = torch.load(f"{config.save_dir}/best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def predict(model, loader, device):
    """Run inference and return predicted and target values in linear scale."""
    model.eval()
    k_pred, g_pred = [], []
    k_true, g_true = [], []
    
    with torch.no_grad():
        for time_series, aux, targets, _, _, _, _ in tqdm(loader, desc="Predict"):
            time_series = time_series.to(device)
            aux = aux.to(device)

            k_t, g_t = targets[:, 0], targets[:, 1]

            log_k, log_g, _, _ = model(time_series, aux)

            k_pred.extend(10**log_k.cpu().numpy())
            g_pred.extend(10**log_g.cpu().numpy())

            k_true.extend(10**k_t.numpy())
            g_true.extend(10**g_t.numpy())

    return {
        'k_pred': np.array(k_pred),
        'g_pred': np.array(g_pred),
        'k_true': np.array(k_true),
        'g_true': np.array(g_true),
    }

def visualize_predictions(predictions):
    """Plot prediction-vs-target scatter plots for k and gamma."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(predictions['k_true'], predictions['k_pred'], alpha=0.5)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].plot([min(predictions['k_true']), max(predictions['k_true'])],
                [min(predictions['k_true']), max(predictions['k_true'])],
                'r--')
    axs[0].set_xlabel('True k (N/m)')
    axs[0].set_ylabel('Predicted k (N/m)')
    axs[0].set_title('Elastic Constant (k)')
    axs[0].grid(True, which='both', ls='--', alpha=0.5)

    axs[1].scatter(predictions['g_true'], predictions['g_pred'], alpha=0.5)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].plot([min(predictions['g_true']), max(predictions['g_true'])],
                [min(predictions['g_true']), max(predictions['g_true'])],
                'r--')
    axs[1].set_xlabel('True γ (kg/s)')
    axs[1].set_ylabel('Predicted γ (kg/s)')
    axs[1].set_title('Damping Coefficient (γ)')
    axs[1].grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('./checkpoints/prediction_results.png')
    plt.show()

if __name__ == "__main__":
    config = TrainingConfig()

    model = train_model(config)

    val_dataset = BrownianDataset(
        config.val_file, 
        window_size=config.window_size, 
        overlap=0,
        is_train=False,
        include_pressure_aux=config.include_pressure_aux,
        pressure_window_jitter_rel=config.pressure_window_jitter_rel,
        pressure_window_jitter_abs=config.pressure_window_jitter_abs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(config.device == 'cuda')
    )

    predictions = predict(model, val_loader, config.device)

    visualize_predictions(predictions)

    print("Training and evaluation completed.")