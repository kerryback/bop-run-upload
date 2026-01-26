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


def main():
    """
    Main execution function.

    ROOT: main_revised.py lines 122-130, 266-299
    """
    start_time = time.time()

    # Parse arguments
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(script_name='run_fama')

    # Load config
    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']

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
        title="FAMA-FRENCH & FAMA-MACBETH FACTORS",
        model=MODEL,
        panel_id=panel_id,
        config=CONFIG,
        additional_info={'Methods': 'Fama-French, Fama-MacBeth'}
    )

    # =========================================================================
    # STEP 1: Compute factor returns
    # ROOT: main_revised.py lines 122-123
    #   ff_rets = fama.factors(fama.fama_french, panel, n_jobs=n_jobs, start=start, end=end)
    #   fm_rets = fama.factors(fama.fama_macbeth, panel, n_jobs=n_jobs, start=start, end=end)
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Computing Fama-French and Fama-MacBeth factors...")
    t0 = time.time()

    # ROOT line 122: Fama-French factor returns
    ff_rets = fama.factors(
        fama.fama_french, panel,
        n_jobs=CONFIG['n_jobs'], start=start, end=end, chars=CHARS
    )

    # ROOT line 123: Fama-MacBeth factor returns
    fm_rets = fama.factors(
        fama.fama_macbeth, panel,
        n_jobs=CONFIG['n_jobs'], start=start, end=end, chars=CHARS
    )

    elapsed = time.time() - t0
    print(f"[OK] Fama factors computed in {fmt(elapsed)} at {now()}")
    print(f"  FF returns: {ff_rets.shape}")
    print(f"  FM returns: {fm_rets.shape}")

    # =========================================================================
    # STEP 2: Save results
    # Portfolio statistics are computed by run_portfolio_stats.py
    # =========================================================================
    results = {
        'ff_returns': ff_rets,
        'fm_returns': fm_rets,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'start': start,
        'end': end,
    }

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_fama.pkl")
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
            "ff = results['ff_returns']",
            "fm = results['fm_returns']",
        ]
    )


if __name__ == "__main__":
    main()
