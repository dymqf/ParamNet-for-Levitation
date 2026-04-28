# ParamNet

ParamNet is a physics-guided deep learning framework for fast parameter inversion in vacuum optical levitation systems. It is designed to estimate the trap stiffness `kappa` and damping coefficient `gamma` directly from short Brownian-motion trajectories, especially in regimes where classical calibration methods become unreliable because of low sampling rate, short observation windows, aliasing, and underdamped dynamics.

This repository accompanies the manuscript:

`ParamNet: A Physics-Guided Deep Learning Framework for Intelligent Self-Inversion of Vacuum Optical Levitation Systems`

## Overview

Vacuum optical tweezers are widely used in precision metrology, force sensing, inertial sensing, and optomechanics. A central requirement in these systems is accurate calibration of the optical trap and the particle dynamics. In practice, this means estimating:

- `kappa`: optical trap stiffness
- `gamma`: viscous damping coefficient

Traditional estimators such as PSD fitting, ACF analysis, and FORMA are effective when long trajectories and sufficiently high sampling rates are available. Under short-window and aliasing-limited conditions, however, these methods can suffer from statistical bias, distorted spectral structure, and poor robustness. ParamNet addresses this setting by combining learned representations with explicit physical constraints from the underdamped Langevin model.

## Method Summary

ParamNet uses a dual-branch time-frequency architecture:

- A time-domain branch extracts local and multi-scale features from short trajectory windows.
- A frequency-domain branch captures oscillatory and damping-related structure from FFT-derived representations.
- A fusion head combines both feature streams and predicts `kappa` and `gamma`.
- A heteroscedastic Gaussian objective models both prediction means and uncertainties.
- Physics-guided regularization enforces consistency with underdamped dynamics through AR-based and ACF-based losses.

The current training pipeline also includes:

- progressive physics-weight scheduling during training
- mixed-precision training when CUDA is available
- early stopping and best-checkpoint selection
- EMA-based evaluation/checkpoint stabilization
- notebook-based demo inference and visualization

## Main Results Reported In The Paper

According to the manuscript, ParamNet achieves:

- about `3%` MAPE for `kappa`
- about `7%` MAPE for `gamma`
- strong performance using windows of only `100` samples (`0.01 s` at `10 kHz`)
- clear improvements over PSD, ACF, and FORMA under short-window conditions
- near-real-time inference suitable for feedback-oriented applications

The paper further positions ParamNet as a compact surrogate model for:

- intelligent self-inversion of levitated systems
- digital-twin-style monitoring
- adaptive control and fast calibration workflows

## Repository Structure

- `ParamNet_training.py`: main training, validation, inference, checkpointing, and plotting pipeline
- `ParamNet_running.ipynb`: interactive notebook for loading checkpoints, running demos, and visualizing predictions
- `Traditional Methods.ipynb`: baseline experiments with classical calibration methods
- `ParamNet.tex`: manuscript source describing the full method, experiments, and discussion
- `requirements.txt`: core Python dependencies

## Data Format

The training and evaluation scripts expect `.npz` files with keys consistent with the simulation pipeline. The main fields are:

- `position`: particle displacement sequence, shape `[B, T]`
- `k0`: ground-truth trap stiffness
- `gamma`: ground-truth damping coefficient
- `D`: diffusion-related coefficient
- `P`: pressure
- `m`: particle mass
- `T`: temperature
- `fs`: sampling frequency

The default file names in `TrainingConfig` are:

- `Training_data_batch.npz`
- `Var_data_batch.npz`
- `Test_data_batch.npz`

The notebook also supports running demo inference on local files such as `demo_new.npz`.

## Installation

Recommended environment:

- Python `3.10+`
- PyTorch `2.1+`
- NumPy `1.24+`
- SciPy `1.10+`
- Matplotlib `3.7+`
- tqdm `4.65+`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the model

```bash
python ParamNet_training.py
```

This script will:

- load the training and validation datasets
- train ParamNet with the configured loss and optimizer settings
- save the best checkpoint to `checkpoints/best_model.pth`
- export training and prediction figures to `checkpoints/`

### 2. Run the notebook demo

Open:

- `ParamNet_running.ipynb`

The notebook can be used to:

- load a trained checkpoint
- run inference on demo or test data
- generate diagnostic figures for `kappa` and `gamma`
- export prediction summaries to `.mat`

### 3. Compare against traditional methods

Open:

- `Traditional Methods.ipynb`

This notebook contains the baseline workflow used for comparison with conventional estimators.

## Training Notes

The manuscript emphasizes short-window inversion with `100` samples at `10 kHz`. In the codebase, the window length is configurable through `TrainingConfig`, so you can adapt the model to different experimental settings without changing the network definition.

The current physics-guided training setup in the main script uses:

- AR loss weight `W_AR = 1.76`
- ACF loss weight `W_ACF = 0.54`
- peak global physics coefficient `1.12`
- physics warm-up over the first `30` epochs

These settings reflect the current implementation used to align the repository with the manuscript description.

## Checkpoint

Pretrained checkpoint archive:

- Baidu Netdisk: `https://pan.baidu.com/s/1JC-Ow5_TYqADCJjunsgkQw?pwd=kvmu`
- extraction code: `kvmu`

## Scope And Limitations

ParamNet is intended for parameter inversion in regimes reasonably close to the underdamped harmonic Langevin model used during simulation and training. As discussed in the paper, performance can degrade when there is strong model mismatch, for example:

- strongly non-harmonic trapping potentials
- heavy baseline drift, transient spikes, or non-stationary noise outside the training distribution
- operating conditions far beyond the simulated range
- sim-to-real discrepancies not covered by the current dataset

In its current form, the framework should be viewed as a physics-guided and data-efficient inversion tool rather than a universally robust estimator for arbitrary optical trapping conditions.

## Citation

If you use this repository in academic work, please cite the ParamNet manuscript. A formal citation block can be added here once the paper metadata is finalized.
