# ParamNet

ParamNet is a Physics-Guided Deep Learning Framework for Parameters self-Inversion in Vacuum Optical Levitation System.

The current training pipeline predicts:
- Elastic constant: k
- Damping coefficient: gamma

The model combines:
- A temporal branch (multi-scale residual 1D convolution + squeeze-and-excitation)
- A frequency branch (FFT-derived features)
- A heteroscedastic Gaussian objective (mean + variance prediction)
- Physics-based regularization (AR(2) residual)

## Repository Structure

- ParamNet_training.py: End-to-end training, validation, inference, and plotting utilities.
- ParamNet_running.ipynb: Interactive experimentation notebook.
- Traditional Methods.ipynb: Baseline/traditional method experiments.

## Data Format

The training script expects NPZ files with these keys:
- position: shape [B, T], displacement sequences
- k0: shape [B], ground-truth elastic constants
- gamma: shape [B], ground-truth damping coefficients
- D: shape [B], diffusion coefficients (currently loaded but not used as a target)
- P: shape [B], pressure values
- m: shape [B], particle mass
- T: shape [B], temperature
- fs: scalar or shape [B], sampling frequency

Default file names in TrainingConfig:
- Training_data_batch.npz
- Var_data_batch.npz
- Test_data_batch.npz

## Environment

Recommended:
- Python 3.10+
- PyTorch 2.1+
- NumPy
- SciPy
- Matplotlib
- tqdm

Example install:

```bash
pip install torch numpy scipy matplotlib tqdm
```

## Quick Start

Run training with default configuration:

```bash
python ParamNet_training.py
```

The script will:
- Train ParamNet with mixed precision (when CUDA is available)
- Track validation loss and early stopping
- Save best checkpoint to checkpoints/best_model.pth
- Export loss and prediction plots under checkpoints/

## Core Training Features

- Sliding-window dataset with optional augmentation
- Pressure auxiliary prior with anti-shortcut jitter/dropout/shuffle
- CosineAnnealingWarmRestarts scheduler
- Gradient clipping
- Lookahead optimizer wrapper
- EMA model averaging for validation/checkpointing
- Optional gamma-pressure decorrelation loss

## Main Outputs

After running training, expected outputs are:
- checkpoints/best_model.pth
- checkpoints/loss_curve.png
- checkpoints/prediction_results.png

## Reproducibility Notes

This script currently does not force a global random seed. For strict reproducibility, set seeds for:
- Python random
- NumPy
- PyTorch CPU/CUDA

Also consider fixing deterministic backend options if exact repeatability is required.

## License

No license file is currently included. Add a license (for example MIT or Apache-2.0) before public release.

## Citation

If you use this project in academic work, please cite your corresponding paper or technical report.
