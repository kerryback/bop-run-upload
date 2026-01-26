"""
SDF evaluation: compute portfolio statistics using pre-computed weights.

This script loads the stock weights from estimate_sdfs.py and the moments
from calculate_moments.py, then computes mean, stdev, xret for each method.

Output: {panel_id}_portfolio_stats.pkl containing fama_stats and dkkm_stats DataFrames.

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
import time
import importlib
import pickle

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
    print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print("="*70)

    # Load panel (needed for xret)
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")
    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        panel_data = pickle.load(f)
    panel = panel_data['panel']

    # Prepare panel
    panel, _, _ = factor_utils.prepare_panel(panel, CHARS)

    # Load weights
    weights_path = os.path.join(config.DATA_DIR, f"{panel_id}_weights.pkl")
    print(f"Loading weights from {weights_path}...")
    with open(weights_path, 'rb') as f:
        weights_data = pickle.load(f)

    all_weights = weights_data['weights']
    nfeatures_lst = weights_data['nfeatures_lst']
    alpha_lst = weights_data['alpha_lst']
    alpha_lst_fama = weights_data['alpha_lst_fama']
    NMAT = weights_data['nmat']
    eval_start = weights_data['start']
    eval_end = weights_data['end']

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
        mkt_rf = w.get('mkt_rf', np.nan)  # Market excess return for this month

        # Get panel data for this month
        data = panel.loc[month]
        data_xret = data.xret.values

        # Subset moments to present firms
        rp = m['rp'][firm_ids]
        stock_cov = m['cond_var'][firm_ids, :][:, firm_ids]
        sdf_ret = m['sdf_ret']

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
                'sdf_ret': sdf_ret,
                'mkt_rf': mkt_rf,
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
                'sdf_ret': sdf_ret,
                'mkt_rf': mkt_rf,
            })

    elapsed = time.time() - t0
    print(f"[OK] Statistics computed for {n_months} months in {elapsed:.1f}s")

    # =========================================================================
    # CREATE OUTPUT DATAFRAMES
    # =========================================================================
    fama_stats = pd.DataFrame(all_fama_results)
    dkkm_stats = pd.DataFrame(all_dkkm_results)

    print(f"\nResults:")
    print(f"  Fama stats: {len(fama_stats)} observations")
    print(f"  DKKM stats: {len(dkkm_stats)} observations")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'fama_stats': fama_stats,
        'dkkm_stats': dkkm_stats,
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

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_portfolio_stats.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[OK] Results saved to: {output_file}")

    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
