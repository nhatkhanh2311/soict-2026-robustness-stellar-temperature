"""Utilities shared by the training and evaluation notebooks.

Every perturbation notebook loads the checkpoints, reads the test partition and
computes its metrics through the functions below, so that a perturbed evaluation
differs from the clean one (notebook 05.0) in the input tensor and in nothing else.
"""

import json
import os
import random

import h5py
import numpy as np
import pandas as pd
import torch

from models import build_model

# ---- default locations, relative to the repository root ----------------------
DATA_DIR = 'data'
RESULTS_DIR = 'results'
CKPT_DIR = 'checkpoints'

TENSORS_FILE = 'sdss_tensors.h5'
CATALOG_FILE = 'sdss_final_catalog.csv'
PREP_META_FILE = 'preprocessing_meta.json'

# ---- conventions used throughout the paper -----------------------------------
SEEDS = [0, 1, 2, 3, 4]
REGIME_EDGES_K = [3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000]
CATASTROPHIC_K = 300.0      # |dT| above this counts as a gross failure
EVAL_BATCH_SIZE = 256       # batch size of every evaluation, clean or perturbed


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """CUDA when available, with deterministic cuDNN kernels."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_prep_meta(data_dir=DATA_DIR):
    """Scaler constants, grid description and label-noise floor written by notebook 02."""
    return read_json(os.path.join(data_dir, PREP_META_FILE))


def load_test_split(data_dir=DATA_DIR, grid_n=4096):
    """Test rows of the catalogue, sorted, read from the master tensor file.

    This is the row selection of the training data loader (notebook 04a), so the
    order of the returned spectra matches the stored test predictions.

    Returns (test_idx, X_test, y_true_k) with X_test of shape (N, 1, grid_n),
    float32, and y_true_k the unscaled labels in Kelvin, float64.
    """
    catalog_df = pd.read_csv(os.path.join(data_dir, CATALOG_FILE))
    test_idx = np.sort(catalog_df.index[catalog_df['dataset_split'] == 'test'].values)
    with h5py.File(os.path.join(data_dir, TENSORS_FILE), 'r') as f:
        X_test = f['X'][test_idx]
        y_true_k = f['y_raw'][test_idx].astype(np.float64).ravel()
    X_test = X_test.reshape(len(test_idx), 1, grid_n).astype(np.float32)
    return test_idx, X_test, y_true_k


def wavelength_grid(grid):
    """The logarithmic wavelength grid (Angstrom) described by prep_meta['grid']."""
    n = grid['seq_length']
    lo, hi = grid['wave_start'], grid['wave_end']
    return lo * (hi / lo) ** (np.arange(n) / (n - 1))


def checkpoint_path(model_name, seed, ckpt_dir=CKPT_DIR):
    return os.path.join(ckpt_dir, f'{model_name}_seed{seed}_best.pth')


def load_model(model_name, seed, device, ckpt_dir=CKPT_DIR):
    """Rebuild a model from models.py, load its best checkpoint, set eval() mode."""
    state = torch.load(checkpoint_path(model_name, seed, ckpt_dir), map_location=device)
    model = build_model(model_name).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, teff_min, teff_range, batch_size=EVAL_BATCH_SIZE):
    """Forward a device-resident tensor; returns absolute temperatures in Kelvin."""
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for i in range(0, x.shape[0], batch_size):
        out[i:i + batch_size] = model(x[i:i + batch_size]).squeeze(-1)
    return out.double().cpu().numpy() * teff_range + teff_min


def metrics(pred_k, true_k, catastrophic_k=CATASTROPHIC_K):
    """The five reported error measures, plus n and R^2."""
    r = pred_k - true_k
    a = np.abs(r)
    ss_res = float(np.sum(r ** 2))
    ss_tot = float(np.sum((true_k - true_k.mean()) ** 2))
    return {'n': int(len(r)),
            'rmse_K': float(np.sqrt(np.mean(r ** 2))),
            'mae_K': float(np.mean(a)),
            'bias_K': float(np.mean(r)),
            'p99_abs_K': float(np.percentile(a, 99)),
            'gross_fail_frac': float(np.mean(a > catastrophic_k)),
            'r2': 1.0 - ss_res / ss_tot}


def free_cuda(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
