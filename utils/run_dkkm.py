"""
DKKM (Random Fourier Features) factor computation.

ROOT REFERENCE: main_revised.py lines 133-258

This script implements the root's approach:
1. Generate ONE large W matrix of shape (max_features/2, nchars)
2. Compute factor returns for ALL max_features features at once
3. For each nfeatures, SUBSET the factor returns and W matrix
4. Compute portfolio statistics for each (nfeatures, alpha) combination

CRITICAL DIFFERENCE FROM noipca/utils/run_dkkm.py:
  noipca generates a SEPARATE W matrix for each nfeatures value.
  Root generates ONE W matrix and takes subsets (first num rows).
  This means in root, nfeatures=6 uses the SAME random features as
  the first 6 features of nfeatures=3600.

Usage:
    python run_dkkm.py [panel_id] [--config CONFIG_MODULE]
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
    from utils_factors import dkkm_functions as dkkm
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

    ROOT: main_revised.py lines 133-258 (generate_rff_panel + run_month DKKM section)

    Key logic:
    1. Generate ONE W matrix of (max_features/2, nchars) with gamma scaling
    2. Compute factor returns for max_features
    3. For each nfeatures, subset and compute portfolio stats
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
    nfeatures_lst = CONFIG['n_dkkm_features_list']
    alpha_lst = CONFIG['alpha_lst']
    include_mkt = CONFIG['include_mkt']
    nmat = CONFIG['nmat']
    rank_standardize = CONFIG['dkkm_rank_standardize']

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
            'Feature list': str(nfeatures_lst),
            'Alpha list': str(alpha_lst),
            'Include market': include_mkt,
            'Rank standardize': rank_standardize,
        }
    )

    # =========================================================================
    # STEP 1: Generate W matrix and compute factor returns
    # ROOT: main_revised.py lines 133-141 (generate_rff_panel)
    #
    #   W = np.random.normal(size=(int(max_features/2), nchars + (model == 'bgn')))
    #   gamma = np.random.choice(gamma_grid, size=(int(max_features/2), 1))
    #   W = gamma*W
    #   _, (f_rs, f_nors) = generate_rff_panel(i)  # only for i=0 since nmat=1
    # =========================================================================
    print(f"\n{'-'*70}")
    print(f"Step 1: Computing DKKM factor returns (max_features={max_features})...")
    t0 = time.time()

    # ROOT line 134: generate weight matrix
    # nchars + (model == 'bgn') adds one column for rf rate in BGN model
    W = np.random.normal(
        size=(int(max_features/2), nchars + (MODEL == 'bgn'))
    )

    # ROOT lines 135-136: scale by gamma (random bandwidth)
    gamma = np.random.choice(
        config.GAMMA_GRID,
        size=(int(max_features/2), 1)
    )
    W = gamma * W

    # ROOT line 139: compute factor returns for all max_features
    # dkkm.factors returns (f_rs, f_nors)
    f_rs, f_nors = dkkm.factors(
        panel=panel, W=W, n_jobs=CONFIG['n_jobs'],
        start=start, end=end, model=MODEL, chars=CHARS
    )

    elapsed = time.time() - t0
    print(f"[OK] Factor returns computed in {elapsed:.1f}s")
    print(f"  f_rs shape: {f_rs.shape}")
    print(f"  Months: {f_rs.index.min()} to {f_rs.index.max()}")

    # Select which version to use for portfolio stats
    # ROOT uses rank-standardized (f_rs) in run_month
    frets = f_rs if rank_standardize else f_nors

    # =========================================================================
    # STEP 2: Load Fama returns (needed when include_mkt=True)
    # ROOT: main_revised.py line 122: ff_rets = fama.factors(...)
    # ROOT: main_revised.py line 228: ff_rets.iloc[:, -1] as market return
    # =========================================================================
    ff_rets = None
    if include_mkt:
        fama_file = os.path.join(config.DATA_DIR, f"{panel_id}_fama.pkl")
        if not os.path.exists(fama_file):
            raise FileNotFoundError(
                f"include_mkt=True but fama file not found at {fama_file}. "
                f"Run run_fama.py first."
            )
        with open(fama_file, 'rb') as f:
            fama_data = pickle.load(f)
        if 'ff_returns' not in fama_data:
            raise KeyError(f"'ff_returns' not in {fama_file}. Re-run run_fama.py.")
        ff_rets = fama_data['ff_returns']
        print(f"\n  Loaded FF returns for market factor: shape={ff_rets.shape}")

    # =========================================================================
    # STEP 3: Load pre-computed SDF moments
    # ROOT: main_revised.py line 149: sdf_ret, max_sr, rp, cond_var = sdf_loop(month-1)
    # In noipca2, moments are pre-computed by calculate_moments.py
    # =========================================================================
    moments, N, moments_start, moments_end = sdf_utils.load_precomputed_moments(panel_id)
    print(f"\n  Moments range: {moments_start} to {moments_end}")

    # Evaluation range: start at moments_start (which is burnin+360)
    eval_start = moments_start
    eval_end = moments_end

    # =========================================================================
    # STEP 4: Compute portfolio statistics for each (nfeatures, month, alpha)
    # ROOT: main_revised.py lines 206-258 (DKKM section of run_month)
    #
    # Key root logic:
    #   half = max_features // 2
    #   for nfeatures in nfeatures_lst:
    #       num = int(nfeatures/2)
    #       nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])
    #       factor_weights = dkkm.rff(data[chars], rf, W=W[:num,:], model=model)
    #       factor_weights.columns = [str(ind) for ind in nf_indx]
    #       if include_market:
    #           factor_weights['mkt_rf'] = fama.fama_french(data[chars], mve=data.mve)[:,-1]
    #       for alpha in alpha_lst:
    #           port_of_factors = dkkm.mve_data(frets.iloc[:,nf_indx], month,
    #                                            nfeatures*alpha, ff_rets.iloc[:,-1])
    #           weights_on_stocks = factor_weights @ port_of_factors
    #           ... compute stdev, mean, xret, hjd ...
    # =========================================================================
    print(f"\n{'-'*70}")
    print(f"Step 2: Computing portfolio statistics...")
    print(f"  Features: {nfeatures_lst}")
    print(f"  Alphas: {alpha_lst}")
    print(f"  Evaluation months: {eval_start} to {eval_end}")
    t0 = time.time()

    half = max_features // 2  # ROOT: implicit from arr structure
    results_list = []

    for nfeatures in nfeatures_lst:
        print(f"\n  Processing nfeatures={nfeatures}...")
        t_nf = time.time()

        # ROOT lines 212-213: compute subset indices
        num = int(nfeatures / 2)
        nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])

        # Subset factor returns to this nfeatures
        # ROOT line 228: frets.iloc[:, nf_indx]
        f_subset = frets.iloc[:, nf_indx]

        for month in range(eval_start, eval_end + 1):
            # Get SDF moments for this month
            # ROOT line 149: sdf_ret, max_sr, rp, cond_var = sdf_loop(month-1)
            if month not in moments:
                continue

            month_moments = moments[month]
            rp_full = month_moments['rp']
            cond_var_full = month_moments['cond_var']
            sdf_ret = month_moments['sdf_ret']

            # Get data for this month
            # ROOT line 152: data = panel.loc[month]
            data = panel.loc[month]

            # ROOT line 153-155: keep firms present this month, subset matrices
            # In noipca2, firm IDs from data index
            firm_ids = data.index.to_numpy()

            # ROOT lines 154-158: subset cov and rp to present firms
            rp = rp_full[firm_ids]
            stock_cov = cond_var_full[firm_ids, :][:, firm_ids]
            second_moment = stock_cov + np.outer(rp, rp)
            second_moment_inv = linalg.pinv(second_moment)

            # ROOT lines 214-218: compute factor weights (RFF loadings)
            # Uses W[:num, :] — first num rows of the big W matrix
            if MODEL == 'bgn':
                rf = data.rf_stand
            else:
                rf = None

            # ROOT line 218: factor_weights = dkkm.rff(data[chars], rf, W=W[:num,:], model)
            # Returns (rank_standardized, raw) — use rank_standardized
            factor_weights, _ = dkkm.rff(data[CHARS], rf, W=W[:num, :], model=MODEL)

            # ROOT line 224: rename columns to match nf_indx
            factor_weights.columns = [str(ind) for ind in nf_indx]

            # ROOT lines 225-226: add market weights if include_mkt
            if include_mkt:
                # ROOT line 226: fama.fama_french(data[chars], mve=data.mve)[:, -1]
                factor_weights['mkt_rf'] = fama.fama_french(
                    data[CHARS], CHARS, mve=data.mve
                )[:, -1]

            # ROOT lines 227-228: compute portfolio for all alphas at once
            # ROOT: dkkm.mve_data(frets.iloc[:,nf_indx], month, nfeatures*alpha, mkt_rf)
            # We pass the full alpha grid (pre-scaled by nfeatures) at once
            scaled_alphas = np.array([nfeatures * a for a in alpha_lst])
            mkt_rf = ff_rets.iloc[:, -1] if include_mkt else None

            port_of_factors_df = dkkm.mve_data(
                f_subset, month, scaled_alphas, mkt_rf
            )

            # ROOT lines 229-258: compute stats for each alpha
            for i, alpha in enumerate(alpha_lst):
                # ROOT line 229: weights_on_stocks = factor_weights @ port_of_factors
                port_of_factors = port_of_factors_df.iloc[:, i]
                weights_on_stocks = factor_weights.values @ port_of_factors.values

                # ROOT lines 252-253: stdev, mean
                stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
                mean = weights_on_stocks @ rp

                # ROOT line 254: realized return
                xret = weights_on_stocks @ data.xret.values

                # ROOT lines 256-258: HJD (squared — NO sqrt in root!)
                errs = rp - second_moment @ weights_on_stocks
                hjd = errs @ second_moment_inv @ errs  # ROOT: no sqrt!

                results_list.append({
                    'month': month,
                    'matrix': 0,  # ROOT: for i in range(nmat), here nmat=1
                    'nfeatures': nfeatures,
                    'alpha': alpha,
                    'include_mkt': include_mkt,
                    'stdev': stdev,
                    'mean': mean,
                    'xret': xret,
                    'sdf_ret': sdf_ret,
                    'hjd': hjd
                })

            if month % 100 == 0:
                print(f"    Month {month} done")

        elapsed_nf = time.time() - t_nf
        print(f"    [OK] nfeatures={nfeatures} done in {elapsed_nf:.1f}s")

    elapsed = time.time() - t0
    print(f"\n[OK] Portfolio statistics computed in {elapsed:.1f}s")

    # =========================================================================
    # STEP 5: Save results
    # =========================================================================
    dkkm_stats = pd.DataFrame(results_list)
    print(f"\nResults: {len(dkkm_stats)} observations")

    # Save factor returns (rank-standardized version)
    results = {
        'dkkm_factors': frets,
        'dkkm_stats': dkkm_stats,
        'weights': W,
        'max_features': max_features,
        'nfeatures_lst': nfeatures_lst,
        'nmat': nmat,
        'rank_standardize': rank_standardize,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'alpha_lst': alpha_lst,
        'include_mkt': include_mkt,
        'start': start,
        'end': end,
    }

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_dkkm.pkl")
    factor_utils.save_factor_results(results, output_file, verbose=True)

    # Also save per-nfeatures files for compatibility with analyze.py
    for nfeatures in nfeatures_lst:
        subset = dkkm_stats[dkkm_stats['nfeatures'] == nfeatures]
        nf_results = {
            'dkkm_factors': frets.iloc[:, :nfeatures],  # first nfeatures columns
            'dkkm_stats': subset,
            'weights': W[:int(nfeatures/2), :],
            'nfeatures': nfeatures,
            'rank_standardize': rank_standardize,
            'panel_id': panel_id,
            'model': MODEL,
            'chars': CHARS,
            'start': start,
            'end': end,
        }
        nf_file = os.path.join(config.DATA_DIR, f"{panel_id}_dkkm_{nfeatures}.pkl")
        with open(nf_file, 'wb') as f:
            pickle.dump(nf_results, f)
        print(f"  Saved {nf_file}")

    # Print runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    factor_utils.print_script_footer(
        panel_id=panel_id,
        usage_examples=[
            "import pickle",
            f"with open('{output_file}', 'rb') as f:",
            "    results = pickle.load(f)",
            "stats = results['dkkm_stats']",
        ]
    )


if __name__ == "__main__":
    main()
