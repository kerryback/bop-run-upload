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


def main():
    """
    Main execution function.

    ROOT: main_revised.py lines 133-141 (generate_rff_panel)

    Key logic:
    1. Generate ONE W matrix of (max_features/2, nchars) with gamma scaling
    2. Compute factor returns for max_features

    Portfolio stats are computed by run_portfolio_stats.py
    """
    start_time = time.time()

    # Parse arguments
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(script_name='run_dkkm')

    # Load config
    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']
    nchars = len(CHARS)
    max_features = CONFIG['max_features']
    NMAT = CONFIG['nmat']

    # Load panel data
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")
    if not os.path.exists(panel_path):
        print(f"ERROR: Panel file not found at: {panel_path}")
        sys.exit(1)

    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        arrays_data = pickle.load(f)
    panel = arrays_data['panel']
    print(f"Loaded panel: shape={panel.shape}")

    # Prepare panel
    # ROOT: main_revised.py lines 62-70
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    CONFIG['T'] = end - start + 1
    CONFIG['N'] = panel.groupby(level='month').size().max()

    # Print header
    factor_utils.print_script_header(
        title="DKKM (RANDOM FOURIER FEATURES) FACTORS",
        model=MODEL,
        panel_id=panel_id,
        config=CONFIG,
        additional_info={
            'Max features': max_features,
            'NMAT': NMAT,
        }
    )

    # =========================================================================
    # STEP 1: Generate W matrices and compute factor returns
    # ROOT: main_revised.py lines 133-141 (generate_rff_panel)
    #
    # For NMAT > 1: Generate NMAT different W matrices and compute
    # factor returns for each. Portfolio stats will average across them.
    # =========================================================================
    print(f"\n{'-'*70}")
    print(f"Step 1: Computing DKKM factor returns (max_features={max_features}, NMAT={NMAT})...")
    t0 = time.time()

    half = int(max_features / 2)
    W_list = []
    frets_list = []

    for mat_idx in range(NMAT):
        if NMAT > 1:
            print(f"  Matrix {mat_idx + 1}/{NMAT}...", end=" ", flush=True)

        # ROOT line 134: generate weight matrix
        # nchars + (model == 'bgn') adds one column for rf rate in BGN model
        W_i = np.random.normal(
            size=(half, nchars + (MODEL == 'bgn'))
        )

        # ROOT lines 135-136: scale by gamma (random bandwidth)
        gamma_i = np.random.choice(
            config.GAMMA_GRID,
            size=(half, 1)
        )
        W_i = gamma_i * W_i

        # Compute factor returns for all max_features
        # dkkm.factors returns rank-standardized factor returns
        frets_i = dkkm.factors(
            panel=panel, W=W_i, n_jobs=CONFIG['n_jobs'],
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

    # =========================================================================
    # STEP 2: Save results
    # Portfolio stats are computed by run_portfolio_stats.py
    # =========================================================================
    results = {
        'dkkm_factors': frets_list,  # List[DataFrame] of length NMAT
        'weights': W_list,           # List[ndarray] of length NMAT
        'nmat': NMAT,
        'max_features': max_features,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'start': start,
        'end': end,
    }

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_dkkm.pkl")
    factor_utils.save_factor_results(results, output_file, verbose=True)

    # Print runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {fmt(total_time)} at {now()}")

    factor_utils.print_script_footer(
        panel_id=panel_id,
        usage_examples=[
            "import pickle",
            f"with open('{output_file}', 'rb') as f:",
            "    results = pickle.load(f)",
            "frets_list = results['dkkm_factors']  # List of DataFrames",
            "W_list = results['weights']           # List of arrays",
            f"nmat = results['nmat']                # {NMAT}",
        ]
    )


if __name__ == "__main__":
    main()
