"""
25 double-sorted portfolios (5x5 on market cap and book-to-market).

For each month, assigns firms to quintiles by mve and by bm,
then computes value-weighted excess returns for each of the 25 groups.

Output: {panel_id}_25_portfolios.pkl

Usage:
    python generate_25_portfolios.py [panel_id] [--config CONFIG_MODULE]
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
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the noipca2 directory")
    sys.exit(1)


def main():
    """Compute 25 double-sorted portfolios (mve x bm quintiles)."""
    start_time = time.time()

    # Parse arguments
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(
        script_name='generate_25_portfolios'
    )

    # Load config
    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']

    # Load panel
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")
    if not os.path.exists(panel_path):
        print(f"ERROR: Panel file not found at: {panel_path}")
        sys.exit(1)

    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        arrays_data = pickle.load(f)
    panel = arrays_data['panel']

    # Prepare panel (adds size=log(mve), sets MultiIndex, removes NaNs)
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    print("="*70)
    print("25 DOUBLE-SORTED PORTFOLIOS (MVE x BM)")
    print("="*70)
    print(f"Panel ID: {panel_id}")
    print(f"Model: {MODEL}")
    print(f"Months: {start} to {end}")
    print(f"Started at {now()}")
    print("="*70)

    # Compute portfolios
    print(f"\n{'-'*70}")
    print("Computing value-weighted excess returns...")
    t0 = time.time()

    all_results = []

    for month in range(start, end + 1):
        data = panel.loc[month]

        mve_q = pd.qcut(data['mve'], 5, labels=[1, 2, 3, 4, 5])
        bm_q = pd.qcut(data['bm'], 5, labels=[1, 2, 3, 4, 5])

        for (mq, bq), group in data.groupby([mve_q, bm_q], observed=True):
            weights = group['mve'] / group['mve'].sum()
            vw_xret = (weights * group['xret']).sum()
            all_results.append({
                'month': month,
                'mve_q': int(mq),
                'bm_q': int(bq),
                'vw_xret': vw_xret,
            })

    elapsed = time.time() - t0
    portfolios = pd.DataFrame(all_results)
    print(f"[OK] {len(portfolios)} portfolio-months computed in {fmt(elapsed)} at {now()}")

    # Save results
    results = {
        'portfolios': portfolios,
        'panel_id': panel_id,
        'model': MODEL,
        'start': start,
        'end': end,
    }

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_25_portfolios.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[OK] Results saved to: {output_file}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Total runtime: {fmt(total_time)}")


if __name__ == "__main__":
    main()
