"""
Fama-French and Fama-MacBeth factor return computation.

ROOT REFERENCE: main_revised.py lines 122-130 (factor computation)

Computes:
1. FF factor returns (Fama-French sorted portfolios)
2. FM factor returns (Fama-MacBeth cross-sectional regression)

Portfolio statistics are computed separately by evaluate_sdfs.py

Usage:
    python generate_fama_factors.py [panel_id] [--config CONFIG_MODULE]
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
    from utils_factors import fama_functions as fama
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the noipca2 directory")
    sys.exit(1)


def compute(panel_id, model_name):
    """
    Compute Fama factors and return results dict (no disk I/O).

    ROOT: main_revised.py lines 122-130

    Returns:
        results dict with keys: ff_returns, fm_returns, panel_id, model, etc.
    """
    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']

    # Load panel data
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")
    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        panel = pickle.load(f)['panel']
    print(f"Loaded panel: shape={panel.shape}")

    # Prepare panel
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    # Compute factor returns
    n_jobs = config.get_n_jobs_for_step('fama')
    print(f"\nComputing Fama-French and Fama-MacBeth factors (n_jobs={n_jobs})...")
    t0 = time.time()

    ff_rets = fama.factors(
        fama.fama_french, panel,
        n_jobs=n_jobs, start=start, end=end, chars=CHARS
    )

    fm_rets = fama.factors(
        fama.fama_macbeth, panel,
        n_jobs=n_jobs, start=start, end=end, chars=CHARS
    )

    elapsed = time.time() - t0
    print(f"[OK] Fama factors computed in {fmt(elapsed)} at {now()}")
    print(f"  FF returns: {ff_rets.shape}")
    print(f"  FM returns: {fm_rets.shape}")

    return {
        'ff_returns': ff_rets,
        'fm_returns': fm_rets,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'start': start,
        'end': end,
    }


def main():
    """Standalone entry point: compute and save to disk."""
    start_time = time.time()
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(script_name='run_fama')
    results = compute(panel_id, model_name)

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_fama.pkl")
    factor_utils.save_factor_results(results, output_file, verbose=True)

    print(f"\nTotal runtime: {fmt(time.time() - start_time)} at {now()}")


if __name__ == "__main__":
    main()
