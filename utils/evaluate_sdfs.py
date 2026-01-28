"""
SDF evaluation: compute portfolio statistics using pre-computed weights.

This script loads the stock weights from estimate_sdfs.py and the moments
from calculate_moments.py, then computes mean, stdev, xret for each method.

Output: {panel_id}_results.pkl containing:
  - fama_stats: DataFrame (month, method, alpha, stdev, mean, xret)
  - dkkm_stats: DataFrame (month, nfeatures, alpha, stdev, mean, xret)
  - returns: DataFrame (month, sdf_ret, mkt_rf)

Usage:
    python evaluate_sdfs.py [panel_id] [--config CONFIG_MODULE]
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
    from utils_factors import sdf_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the noipca2 directory")
    sys.exit(1)


def main():
    """Main execution function."""
    start_time = time.time()

    # Parse arguments
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(
        script_name='evaluate_sdf'
    )

    # Load config
    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']

    # =========================================================================
    # LOAD REQUIRED DATA
    # =========================================================================
    print("="*70)
    print("SDF EVALUATION (COMPUTE STATISTICS)")
    print("="*70)
    print(f"Panel ID: {panel_id}")
    print(f"Model: {MODEL}")
    print(f"Started at {now()}")
    print("="*70)

    # Load panel (needed for xret)
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")
    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        panel_data = pickle.load(f)
    panel = panel_data['panel']

    # Prepare panel
    panel, _, _ = factor_utils.prepare_panel(panel, CHARS)

    # Load Fama weights
    fama_weights_path = os.path.join(config.DATA_DIR, f"{panel_id}_stock_weights_fama.pkl")
    print(f"Loading Fama weights from {fama_weights_path}...")
    with open(fama_weights_path, 'rb') as f:
        fama_weights_data = pickle.load(f)

    # Load DKKM weights
    dkkm_weights_path = os.path.join(config.DATA_DIR, f"{panel_id}_stock_weights_dkkm.pkl")
    print(f"Loading DKKM weights from {dkkm_weights_path}...")
    with open(dkkm_weights_path, 'rb') as f:
        dkkm_weights_data = pickle.load(f)

    # Merge weights (combine Fama and DKKM per month)
    all_weights = {}
    for month in fama_weights_data['weights']:
        fama_month = fama_weights_data['weights'][month]
        dkkm_month = dkkm_weights_data['weights'][month]
        all_weights[month] = {
            'firm_ids': fama_month['firm_ids'],
            'fama': fama_month['fama'],
            'dkkm': dkkm_month['dkkm'],
            'mkt_rf': fama_month['mkt_rf'],
        }

    nfeatures_lst = dkkm_weights_data['nfeatures_lst']
    alpha_lst = dkkm_weights_data['alpha_lst']
    alpha_lst_fama = fama_weights_data['alpha_lst_fama']
    NMAT = dkkm_weights_data['nmat']
    eval_start = fama_weights_data['start']
    eval_end = fama_weights_data['end']

    # Load moments
    moments, N, moments_start, moments_end = sdf_utils.load_precomputed_moments(panel_id)
    print(f"  Moments range: {moments_start} to {moments_end}")

    # Determine evaluation range (intersection of weights and moments)
    eval_start = max(eval_start, moments_start)
    eval_end = min(eval_end, moments_end)

    print(f"\nConfiguration:")
    print(f"  NMAT: {NMAT}")
    print(f"  Features: {nfeatures_lst}")
    print(f"  Alphas (DKKM): {alpha_lst}")
    print(f"  Alphas (Fama): {alpha_lst_fama}")
    print(f"  Evaluation range: {eval_start} to {eval_end}")

    # =========================================================================
    # COMPUTE STATISTICS
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Computing portfolio statistics...")
    t0 = time.time()

    all_fama_results = []
    all_dkkm_results = []
    all_returns = []

    n_months = 0
    for month in range(eval_start, eval_end + 1):
        if month not in all_weights:
            continue
        if month not in moments:
            continue

        n_months += 1
        w = all_weights[month]
        m = moments[month]
        firm_ids = w['firm_ids']
        mkt_rf = w.get('mkt_rf', np.nan)

        # Get panel data for this month
        data = panel.loc[month]
        data_xret = data.xret.values

        # Subset moments to present firms
        rp = m['rp'][firm_ids]
        stock_cov = m['cond_var'][firm_ids, :][:, firm_ids]
        sdf_ret = m['sdf_ret']

        # Per-month returns (common across all methods)
        all_returns.append({
            'month': month,
            'sdf_ret': sdf_ret,
            'mkt_rf': mkt_rf,
        })

        # Fama stats (includes ff, fm, capm)
        for (method_name, alpha), weights_on_stocks in w['fama'].items():
            stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
            mean = weights_on_stocks @ rp
            xret = weights_on_stocks @ data_xret

            all_fama_results.append({
                'month': month,
                'method': method_name,
                'alpha': alpha,
                'stdev': stdev,
                'mean': mean,
                'xret': xret,
            })

        # DKKM stats
        for (nfeatures, alpha), weights_on_stocks in w['dkkm'].items():
            stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
            mean = weights_on_stocks @ rp
            xret = weights_on_stocks @ data_xret

            all_dkkm_results.append({
                'month': month,
                'nfeatures': nfeatures,
                'alpha': alpha,
                'stdev': stdev,
                'mean': mean,
                'xret': xret,
            })

    elapsed = time.time() - t0
    print(f"[OK] Statistics computed for {n_months} months in {fmt(elapsed)} at {now()}")

    # =========================================================================
    # CREATE OUTPUT DATAFRAMES
    # =========================================================================
    fama_stats = pd.DataFrame(all_fama_results)
    dkkm_stats = pd.DataFrame(all_dkkm_results)
    returns = pd.DataFrame(all_returns)

    print(f"\nResults:")
    print(f"  Fama stats: {len(fama_stats)} observations")
    print(f"  DKKM stats: {len(dkkm_stats)} observations")
    print(f"  Returns: {len(returns)} months")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'fama_stats': fama_stats,
        'dkkm_stats': dkkm_stats,
        'returns': returns,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'nfeatures_lst': nfeatures_lst,
        'alpha_lst': alpha_lst,
        'alpha_lst_fama': alpha_lst_fama,
        'nmat': NMAT,
        'start': eval_start,
        'end': eval_end,
    }

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_results.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[OK] Results saved to: {output_file}")

    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {now()}")
    print(f"Total runtime: {fmt(total_time)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
