"""
SDF estimation: compute stock weights for Fama, CAPM, and DKKM methods.

This script computes the weights_on_stocks for each method/alpha combination
by running ridge regression on factor returns and combining with factor weights.

Does NOT require moments data - only needs panel, fama factors, and dkkm factors.

Output: {panel_id}_weights.pkl containing stock weights per month for each method.

Usage:
    python estimate_sdfs.py [panel_id] [--config CONFIG_MODULE]
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


def compute_month_weights(month_data, W_list, half, nfeatures_lst, alpha_lst,
                          alpha_lst_fama, CHARS, MODEL, NMAT):
    """
    Compute stock weights for a single month.

    This function is called in parallel for each month.
    Returns weights_on_stocks for each (method, alpha) combination.

    Args:
        month_data: Dict containing pre-extracted data for this month
        W_list: List of DKKM weight matrices (each max_features/2, nchars)
        half: max_features // 2
        nfeatures_lst: List of feature counts [6, 36, 360]
        alpha_lst: DKKM alpha list
        alpha_lst_fama: Fama alpha list
        CHARS: List of characteristic names
        MODEL: Model name (bgn, kp14, gs21)
        NMAT: Number of random W matrices

    Returns:
        Tuple of (month, firm_ids, fama_weights_dict, dkkm_weights_dict)
    """
    month = month_data['month']
    data_chars = month_data['data_chars']  # numpy array
    data_mve = month_data['data_mve']      # numpy array
    firm_ids = month_data['firm_ids']
    ff_rets = month_data['ff_rets']        # DataFrame (full history)
    fm_rets = month_data['fm_rets']        # DataFrame (full history)
    frets_list = month_data['frets_list']  # List of DKKM factor returns DataFrames
    rf_stand = month_data.get('rf_stand')  # For BGN model

    # Create DataFrame for characteristics (needed by dkkm.rff and fama methods)
    data_chars_df = pd.DataFrame(data_chars, columns=CHARS, index=firm_ids)

    # =========================================================================
    # FAMA WEIGHTS (ff, fm, capm)
    # =========================================================================
    fama_weights = {}

    # Get market factor and market weights (used by capm)
    mkt_rf = ff_rets.iloc[:, -1]  # Last column is market excess return
    market_weight = fama.fama_french(data_chars_df, CHARS, mve=data_mve)[:, -1]

    for method_name in ["ff", "fm"]:
        f_rets = ff_rets if method_name == "ff" else fm_rets
        method = fama.fama_french if method_name == "ff" else fama.fama_macbeth

        # Factor weights on stocks
        factor_weights = method(data_chars_df, CHARS, mve=data_mve)

        for alpha in alpha_lst_fama:
            # Portfolio of factors from ridge regression
            port_of_factors = fama.mve_data(f_rets, month, alpha)

            # Final weights on stocks
            weights_on_stocks = factor_weights @ port_of_factors.values

            key = (method_name, alpha)
            fama_weights[key] = weights_on_stocks.astype(np.float32)

    # CAPM: single-factor model using only market excess return
    mkt_rf_df = mkt_rf.to_frame(name='mkt_rf')  # Single-column DataFrame
    for alpha in alpha_lst_fama:
        # Portfolio of factors from ridge regression (single factor)
        port_of_factors = fama.mve_data(mkt_rf_df, month, alpha)

        # Final weights on stocks (market_weight * scalar)
        weights_on_stocks = market_weight * port_of_factors.values[0]

        key = ('capm', alpha)
        fama_weights[key] = weights_on_stocks.astype(np.float32)

    # Store mkt_rf value for this month (for evaluate_sdfs output)
    mkt_rf_value = mkt_rf.loc[month] if month in mkt_rf.index else np.nan

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

            # Factor weights from rff
            fw_i = dkkm.rff(data_chars_df, rf, W=W_i[:num, :], model=MODEL)
            fw_i.columns = [str(ind) for ind in nf_indx]

            # Always add market weights
            fw_i['mkt_rf'] = fama.fama_french(
                data_chars_df, CHARS, mve=data_mve
            )[:, -1]

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

    return month, firm_ids, fama_weights, dkkm_weights, mkt_rf_value


def main():
    """Main execution function."""
    start_time = time.time()

    # Parse arguments
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(
        script_name='estimate_sdf'
    )

    # Load config
    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']
    max_features = CONFIG['max_features']
    nfeatures_lst = CONFIG['n_dkkm_features_list']
    alpha_lst = CONFIG['alpha_lst']
    alpha_lst_fama = CONFIG['alpha_lst_fama']

    half = max_features // 2

    # =========================================================================
    # LOAD REQUIRED DATA
    # =========================================================================
    print("="*70)
    print("SDF ESTIMATION (COMPUTE WEIGHTS)")
    print("="*70)
    print(f"Panel ID: {panel_id}")
    print(f"Model: {MODEL}")
    print(f"Started at {now()}")
    print("="*70)

    # Load panel
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")
    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        panel_data = pickle.load(f)
    panel = panel_data['panel']

    # Prepare panel
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    # Load fama results
    fama_path = os.path.join(config.DATA_DIR, f"{panel_id}_fama.pkl")
    print(f"Loading fama factors from {fama_path}...")
    with open(fama_path, 'rb') as f:
        fama_data = pickle.load(f)
    ff_rets = fama_data['ff_returns']
    fm_rets = fama_data['fm_returns']

    # Load dkkm results
    dkkm_path = os.path.join(config.DATA_DIR, f"{panel_id}_dkkm.pkl")
    print(f"Loading dkkm factors from {dkkm_path}...")
    with open(dkkm_path, 'rb') as f:
        dkkm_data = pickle.load(f)
    frets_list = dkkm_data['dkkm_factors']  # List of DataFrames
    W_list = dkkm_data['weights']           # List of arrays
    NMAT = dkkm_data['nmat']

    # Evaluation range: need 360 months of history for ridge regression
    eval_start = start + 360
    eval_end = end

    print(f"\nConfiguration:")
    print(f"  N_JOBS: {config.N_JOBS}")
    print(f"  NMAT: {NMAT}")
    print(f"  Features: {nfeatures_lst}")
    print(f"  Alphas (DKKM): {alpha_lst}")
    print(f"  Alphas (Fama): {alpha_lst_fama}")
    print(f"  Evaluation range: {eval_start} to {eval_end}")

    # =========================================================================
    # PRE-EXTRACT PER-MONTH DATA
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Preparing per-month data...")
    t0 = time.time()

    month_data_list = []
    for month in range(eval_start, eval_end + 1):
        data = panel.loc[month]
        firm_ids = data.index.to_numpy()

        month_data_list.append({
            'month': month,
            'data_chars': data[CHARS].values,
            'data_mve': data.mve.values,
            'firm_ids': firm_ids,
            'ff_rets': ff_rets,
            'fm_rets': fm_rets,
            'frets_list': frets_list,
            'rf_stand': data.rf_stand.values if MODEL == 'bgn' and 'rf_stand' in data.columns else None,
        })

    print(f"  Prepared {len(month_data_list)} months in {time.time()-t0:.1f}s")

    # =========================================================================
    # PARALLEL WEIGHT COMPUTATION
    # =========================================================================
    print(f"\n{'-'*70}")
    print(f"Computing stock weights in parallel (n_jobs={config.N_JOBS})...")
    t0 = time.time()

    # Process in chunks to manage memory
    chunk_size = 50
    n_chunks = (len(month_data_list) + chunk_size - 1) // chunk_size

    all_weights = {}  # month -> {'firm_ids': ..., 'fama': ..., 'dkkm': ...}

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(month_data_list))
        chunk = month_data_list[chunk_start:chunk_end]

        print(f"\n  Chunk {chunk_idx + 1}/{n_chunks}: months {chunk[0]['month']} to {chunk[-1]['month']}")

        with Parallel(n_jobs=config.N_JOBS, verbose=5) as parallel:
            chunk_results = parallel(
                delayed(compute_month_weights)(
                    md, W_list, half, nfeatures_lst, alpha_lst,
                    alpha_lst_fama, CHARS, MODEL, NMAT
                )
                for md in chunk
            )

        # Collect results
        for month, firm_ids, fama_weights, dkkm_weights, mkt_rf_value in chunk_results:
            all_weights[month] = {
                'firm_ids': firm_ids,
                'fama': fama_weights,
                'dkkm': dkkm_weights,
                'mkt_rf': mkt_rf_value,
            }

        del chunk_results
        gc.collect()

    elapsed = time.time() - t0
    print(f"\n[OK] Weights computed in {fmt(elapsed)} at {now()}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'weights': all_weights,  # month -> {'firm_ids', 'fama', 'dkkm'}
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

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_weights.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[OK] Weights saved to: {output_file}")

    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("ESTIMATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {now()}")
    print(f"Total runtime: {fmt(total_time)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
