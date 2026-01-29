"""
SDF estimation: compute stock weights for DKKM method.

This script computes the weights_on_stocks for DKKM method by running
ridge regression on factor returns and combining with factor weights.

Requires fama_weights.pkl for market weights (pre-computed by estimate_sdf_fama.py).

Output: {panel_id}_dkkm_weights.pkl containing stock weights per month for DKKM.

Usage:
    python estimate_sdf_dkkm.py [panel_id] [--config CONFIG_MODULE]
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
from joblib import Parallel, delayed
import gc

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
    from utils_factors import fama_functions as fama
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the noipca2 directory")
    sys.exit(1)


# Shared data for parallel workers. Set before Parallel() call; workers
# inherit via fork (copy-on-write) with backend='multiprocessing'.
_SHARED_DATA = {}


def compute_month_weights_dkkm(month_data, half, nfeatures_lst, alpha_lst, CHARS, MODEL, NMAT):
    """
    Compute DKKM stock weights for a single month.

    This function is called in parallel for each month.
    Large data (W_list, ff_rets, frets_list, market_weights) is read from
    module-level _SHARED_DATA to avoid serializing per worker.

    Args:
        month_data: Dict containing pre-extracted data for this month
        half: max_features // 2
        nfeatures_lst: List of feature counts [6, 36, 360, 3600]
        alpha_lst: DKKM alpha list
        CHARS: List of characteristic names
        MODEL: Model name (bgn, kp14, gs21)
        NMAT: Number of random W matrices

    Returns:
        Tuple of (month, firm_ids, dkkm_weights_dict)
    """
    month = month_data['month']
    data_chars = month_data['data_chars']  # numpy array
    firm_ids = month_data['firm_ids']
    rf_stand = month_data.get('rf_stand')  # For BGN model

    # Read large shared data (inherited via fork, not serialized)
    W_list = _SHARED_DATA['W_list']
    ff_rets = _SHARED_DATA['ff_rets']
    frets_list = _SHARED_DATA['frets_list']
    market_weights = _SHARED_DATA['market_weights']

    # Create DataFrame for characteristics (needed by dkkm.rff)
    data_chars_df = pd.DataFrame(data_chars, columns=CHARS, index=firm_ids)

    # Get market weight for this month (pre-computed by Fama step)
    market_weight = market_weights[month]

    # =========================================================================
    # DKKM WEIGHTS WITH NMAT AVERAGING
    # =========================================================================
    dkkm_weights = {}

    # Market returns (needed for mve_data)
    mkt_rf = ff_rets.iloc[:, -1]

    # rf for BGN model
    rf = rf_stand if MODEL == 'bgn' else None

    for nfeatures in nfeatures_lst:
        # Compute subset indices
        num = int(nfeatures / 2)
        nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])

        # Scaled alphas for ridge regression
        scaled_alphas = np.array([nfeatures * a for a in alpha_lst])

        # Accumulate factor_weights and port_of_factors across NMAT iterations
        factor_weights_sum = None
        port_of_factors_sum = None

        for mat_idx in range(NMAT):
            W_i = W_list[mat_idx]
            frets_i = frets_list[mat_idx]

            # Subset factor returns for this W matrix
            f_subset_i = frets_i.iloc[:, nf_indx]

            # Factor weights from rff (returns ndarray, wrap in DataFrame)
            fw_i = pd.DataFrame(
                dkkm.rff(data_chars_df, rf, W=W_i[:num, :], model=MODEL),
                columns=[str(ind) for ind in nf_indx],
                index=firm_ids,
            )

            # Always add market weights (reuse pre-computed)
            fw_i['mkt_rf'] = market_weight

            # Portfolio of factors from ridge regression
            pof_i = dkkm.mve_data(f_subset_i, month, scaled_alphas, mkt_rf)

            # Accumulate
            if factor_weights_sum is None:
                factor_weights_sum = fw_i.values.copy()
                port_of_factors_sum = pof_i.values.copy()
            else:
                factor_weights_sum += fw_i.values
                port_of_factors_sum += pof_i.values

        # Average across NMAT iterations
        factor_weights_avg = factor_weights_sum / NMAT
        port_of_factors_avg = port_of_factors_sum / NMAT

        # Compute weights for each alpha
        for i, alpha in enumerate(alpha_lst):
            pof_alpha = port_of_factors_avg[:, i]
            weights_on_stocks = factor_weights_avg @ pof_alpha

            key = (nfeatures, alpha)
            dkkm_weights[key] = weights_on_stocks.astype(np.float32)

        print(f"  Month {month}: DKKM nfeatures={nfeatures} complete at {now()}")

    return month, firm_ids, dkkm_weights


def run(panel_id, model_name):
    """
    Run DKKM SDF estimation.

    Loads panel, fama_weights (for market weights), and DKKM factors from disk.
    """
    start_time = time.time()

    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']
    max_features = CONFIG['max_features']
    nfeatures_lst = CONFIG['n_dkkm_features_list']
    alpha_lst = CONFIG['alpha_lst']

    half = max_features // 2

    # =========================================================================
    # LOAD REQUIRED DATA
    # =========================================================================
    print("="*70)
    print("DKKM SDF ESTIMATION (COMPUTE WEIGHTS)")
    print("="*70)
    print(f"Panel ID: {panel_id}")
    print(f"Model: {MODEL}")
    print(f"Started at {now()}")
    print("="*70)

    # Load panel
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")
    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        panel = pickle.load(f)['panel']

    # Prepare panel
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    # Load Fama weights (for market weights)
    fama_weights_path = os.path.join(config.DATA_DIR, f"{panel_id}_stock_weights_fama.pkl")
    print(f"Loading fama weights from {fama_weights_path}...")
    with open(fama_weights_path, 'rb') as f:
        fama_weights_data = pickle.load(f)

    # Extract market weights per month
    market_weights = {}
    for month, month_data in fama_weights_data['weights'].items():
        market_weights[month] = month_data['market_weight']

    # Load Fama factors (for mkt_rf)
    fama_path = os.path.join(config.DATA_DIR, f"{panel_id}_fama.pkl")
    print(f"Loading fama factors from {fama_path}...")
    with open(fama_path, 'rb') as f:
        fama_data = pickle.load(f)
    ff_rets = fama_data['ff_returns']

    # Load DKKM results
    dkkm_path = os.path.join(config.DATA_DIR, f"{panel_id}_dkkm.pkl")
    print(f"Loading dkkm factors from {dkkm_path}...")
    with open(dkkm_path, 'rb') as f:
        dkkm_data = pickle.load(f)
    frets_list = dkkm_data['dkkm_factors']
    W_list = dkkm_data['weights']
    NMAT = dkkm_data['nmat']

    # Evaluation range: need 360 months of history for ridge regression
    eval_start = start + 360
    eval_end = end

    print(f"\nConfiguration:")
    print(f"  N_JOBS: {config.N_JOBS}")
    print(f"  NMAT: {NMAT}")
    print(f"  Features: {nfeatures_lst}")
    print(f"  Alphas (DKKM): {alpha_lst}")
    print(f"  Evaluation range: {eval_start} to {eval_end}")

    # =========================================================================
    # SET SHARED DATA FOR PARALLEL WORKERS
    # =========================================================================
    _SHARED_DATA['W_list'] = W_list
    _SHARED_DATA['ff_rets'] = ff_rets
    _SHARED_DATA['frets_list'] = frets_list
    _SHARED_DATA['market_weights'] = market_weights

    # =========================================================================
    # PRE-EXTRACT PER-MONTH DATA (small per-month data only)
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Preparing per-month data...")
    t0 = time.time()

    month_data_list = []
    for month in range(eval_start, eval_end + 1):
        data = panel.loc[month]
        firm_ids = data.index.to_numpy()

        # Use to_numpy() for better performance and ensure C-contiguous layout
        data_chars = np.ascontiguousarray(data[CHARS].to_numpy())

        month_data_list.append({
            'month': month,
            'data_chars': data_chars,
            'firm_ids': firm_ids,
            'rf_stand': data.rf_stand if MODEL == 'bgn' and 'rf_stand' in data.columns else None,
        })

    print(f"  Prepared {len(month_data_list)} months in {time.time()-t0:.1f}s")

    # =========================================================================
    # PARALLEL WEIGHT COMPUTATION
    # =========================================================================
    print(f"\n{'-'*70}")
    n_jobs = config.get_n_jobs_for_step('sdf')
    print(f"Computing DKKM stock weights (n_jobs={n_jobs})...")
    t0 = time.time()

    chunk_size = 50
    n_chunks = (len(month_data_list) + chunk_size - 1) // chunk_size

    all_weights = {}

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(month_data_list))
        chunk = month_data_list[chunk_start:chunk_end]

        print(f"\n  Chunk {chunk_idx + 1}/{n_chunks}: months {chunk[0]['month']} to {chunk[-1]['month']}")

        with Parallel(n_jobs=n_jobs, verbose=5, backend='multiprocessing') as parallel:
            chunk_results = parallel(
                delayed(compute_month_weights_dkkm)(
                    md, half, nfeatures_lst, alpha_lst, CHARS, MODEL, NMAT
                )
                for md in chunk
            )

        for month, firm_ids, dkkm_weights in chunk_results:
            all_weights[month] = {
                'firm_ids': firm_ids,
                'dkkm': dkkm_weights,
            }

        del chunk_results
        gc.collect()

    # Clear shared data
    _SHARED_DATA.clear()

    elapsed = time.time() - t0
    print(f"\n[OK] DKKM weights computed in {fmt(elapsed)} at {now()}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'weights': all_weights,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'nfeatures_lst': nfeatures_lst,
        'alpha_lst': alpha_lst,
        'nmat': NMAT,
        'start': eval_start,
        'end': eval_end,
    }

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_stock_weights_dkkm.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[OK] DKKM weights saved to: {output_file}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("DKKM ESTIMATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {now()}")
    print(f"Total runtime: {fmt(total_time)}")
    print(f"{'='*70}")


def main():
    """Standalone entry point: parse args and run."""
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(
        script_name='estimate_sdf_dkkm'
    )
    run(panel_id, model_name)


if __name__ == "__main__":
    main()
