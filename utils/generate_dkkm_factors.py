"""
DKKM (Random Fourier Features) factor return computation.

ROOT REFERENCE: main_revised.py lines 133-141

This script implements the root's approach:
1. Generate ONE large W matrix of shape (max_features/2, nchars)
2. Compute factor returns for ALL max_features features at once

Portfolio statistics are computed separately by evaluate_sdfs.py

CRITICAL DIFFERENCE FROM noipca/utils/run_dkkm.py:
  noipca generates a SEPARATE W matrix for each nfeatures value.
  Root generates ONE W matrix and takes subsets (first num rows).
  This means in root, nfeatures=6 uses the SAME random features as
  the first 6 features of nfeatures=3600.

Usage:
    python generate_dkkm_factors.py [panel_id] [--config CONFIG_MODULE]
"""

import sys
import os

# Add current directory and parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import importlib
import pickle

def fmt(s):
    h, m, sec = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return f"{h}h {m}m {sec}s" if h else f"{m}m {sec}s"

def now():
    return datetime.now(ZoneInfo('America/Chicago')).strftime('%a %d %b %Y, %I:%M%p %Z')

# Parse optional --config argument
config_module_name = 'config'
if '--config' in sys.argv:
    config_idx = sys.argv.index('--config')
    config_module_name = sys.argv[config_idx + 1]
    sys.argv.pop(config_idx)
    sys.argv.pop(config_idx)

# Import config module
config = importlib.import_module(config_module_name)
sys.modules['config'] = config

# Import factor modules
try:
    from utils_factors import dkkm_functions as dkkm
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the noipca2 directory")
    sys.exit(1)


def compute(panel_id, model_name):
    """
    Compute DKKM factors and return results dict (no disk I/O).

    ROOT: main_revised.py lines 133-141 (generate_rff_panel)

    Returns:
        results dict with keys: dkkm_factors, weights, nmat, max_features, etc.
    """
    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']
    nchars = len(CHARS)
    max_features = CONFIG['max_features']
    NMAT = CONFIG['nmat']

    # Load panel data
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")
    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        panel = pickle.load(f)['panel']
    print(f"Loaded panel: shape={panel.shape}")

    # Prepare panel
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    # Generate W matrices and compute factor returns
    n_jobs = config.get_n_jobs_for_step('dkkm')
    print(f"\nComputing DKKM factor returns (max_features={max_features}, NMAT={NMAT}, n_jobs={n_jobs})...")
    t0 = time.time()

    half = int(max_features / 2)
    W_list = []
    frets_list = []

    for mat_idx in range(NMAT):
        if NMAT > 1:
            print(f"  Matrix {mat_idx + 1}/{NMAT}...", end=" ", flush=True)

        W_i = np.random.normal(
            size=(half, nchars + (MODEL == 'bgn'))
        )
        gamma_i = np.random.choice(
            config.GAMMA_GRID,
            size=(half, 1)
        )
        W_i = gamma_i * W_i

        frets_i = dkkm.factors(
            panel=panel, W=W_i, n_jobs=n_jobs,
            start=start, end=end, model=MODEL, chars=CHARS
        )

        W_list.append(W_i)
        frets_list.append(frets_i)

        if NMAT > 1:
            print(f"done (shape={frets_i.shape})")

    elapsed = time.time() - t0
    print(f"[OK] Factor returns computed in {fmt(elapsed)} at {now()}")
    print(f"  frets shape: {frets_list[0].shape}")
    print(f"  Months: {frets_list[0].index.min()} to {frets_list[0].index.max()}")

    return {
        'dkkm_factors': frets_list,
        'weights': W_list,
        'nmat': NMAT,
        'max_features': max_features,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'start': start,
        'end': end,
    }


def main():
    """Standalone entry point: compute and save to disk."""
    start_time = time.time()
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(script_name='run_dkkm')
    results = compute(panel_id, model_name)

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_dkkm.pkl")
    factor_utils.save_factor_results(results, output_file, verbose=True)

    print(f"\nTotal runtime: {fmt(time.time() - start_time)} at {now()}")


if __name__ == "__main__":
    main()
