"""
Fama-French and Fama-MacBeth factor computation.

ROOT REFERENCE: main_revised.py lines 122-130 (factor computation)
               main_revised.py lines 266-299 (run_month fama section)

Computes:
1. FF factor returns (Fama-French sorted portfolios)
2. FM factor returns (Fama-MacBeth cross-sectional regression)
3. Portfolio statistics for each method

Usage:
    python run_fama.py [panel_id] [--config CONFIG_MODULE]
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
import scipy.linalg as linalg

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
    from utils_factors import sdf_utils
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
    print(f"[OK] Fama factors computed in {elapsed:.1f}s")
    print(f"  FF returns: {ff_rets.shape}")
    print(f"  FM returns: {fm_rets.shape}")

    # =========================================================================
    # STEP 2: Compute portfolio statistics
    # ROOT: main_revised.py lines 266-299 (fama section of run_month)
    #
    # Key root logic:
    #   for m in ["ff", "fm"]:
    #       frets = ff_rets if m=="ff" else fm_rets
    #       method = fama.fama_french if m=="ff" else fama.fama_macbeth
    #       factor_weights = method(data[chars], mve=data.mve)
    #       for alpha in alpha_lst_fama:
    #           port_of_factors = fama.mve_data(frets, month, alpha)
    #           weights_on_stocks = factor_weights @ port_of_factors
    #           ... compute stdev, mean, xret, hjd ...
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Computing portfolio statistics...")

    # Load pre-computed SDF moments
    moments, N, moments_start, moments_end = sdf_utils.load_precomputed_moments(panel_id)
    print(f"  Moments range: {moments_start} to {moments_end}")

    eval_start = moments_start
    eval_end = moments_end
    alpha_lst_fama = CONFIG['alpha_lst_fama']  # [0] — OLS only

    t0 = time.time()
    results_list = []

    for method_name in ["ff", "fm"]:
        print(f"\n  Processing {method_name}...")
        frets = ff_rets if method_name == "ff" else fm_rets
        method = fama.fama_french if method_name == "ff" else fama.fama_macbeth

        for month in range(eval_start, eval_end + 1):
            if month not in moments:
                continue

            # ROOT line 149: get SDF moments
            month_moments = moments[month]
            rp_full = month_moments['rp']
            cond_var_full = month_moments['cond_var']
            sdf_ret = month_moments['sdf_ret']

            # ROOT line 152: get data for this month
            data = panel.loc[month]

            # ROOT lines 153-158: subset to present firms
            firm_ids = data.index.to_numpy()
            rp = rp_full[firm_ids]
            stock_cov = cond_var_full[firm_ids, :][:, firm_ids]
            second_moment = stock_cov + np.outer(rp, rp)
            second_moment_inv = linalg.pinv(second_moment)

            # ROOT line 271: factor_weights = method(data[chars], mve=data.mve)
            factor_weights = method(data[CHARS], CHARS, mve=data.mve)

            for alpha in alpha_lst_fama:
                # ROOT line 274: port_of_factors = fama.mve_data(frets, month, alpha)
                port_of_factors = fama.mve_data(frets, month, alpha)

                # ROOT line 275: weights_on_stocks = factor_weights @ port_of_factors
                weights_on_stocks = factor_weights @ port_of_factors.values

                # ROOT lines 276-278: stdev, mean, xret
                stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
                mean = weights_on_stocks @ rp
                xret = weights_on_stocks @ data.xret.values

                # ROOT lines 280-281: HJD (squared — NO sqrt in root!)
                errs = rp - second_moment @ weights_on_stocks
                hjd = errs @ second_moment_inv @ errs  # ROOT: no sqrt!

                results_list.append({
                    'month': month,
                    'method': method_name,
                    'alpha': alpha,
                    'stdev': stdev,
                    'mean': mean,
                    'xret': xret,
                    'sdf_ret': sdf_ret,
                    'hjd': hjd
                })

            if month % 100 == 0:
                print(f"    Month {month} done")

    elapsed = time.time() - t0
    fama_stats = pd.DataFrame(results_list)
    print(f"\n[OK] Portfolio statistics computed in {elapsed:.1f}s")
    print(f"  Total observations: {len(fama_stats)}")

    # =========================================================================
    # STEP 3: Save results
    # =========================================================================
    results = {
        'ff_returns': ff_rets,
        'fm_returns': fm_rets,
        'fama_stats': fama_stats,
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
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    factor_utils.print_script_footer(
        panel_id=panel_id,
        usage_examples=[
            "import pickle",
            f"with open('{output_file}', 'rb') as f:",
            "    results = pickle.load(f)",
            "ff = results['ff_returns']",
            "stats = results['fama_stats']",
        ]
    )


if __name__ == "__main__":
    main()
