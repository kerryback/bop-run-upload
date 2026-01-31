"""
SDF estimation: compute stock weights for Fama-French, Fama-MacBeth, and CAPM methods.

This script computes the weights_on_stocks for FF/FM/CAPM methods by running
ridge regression on factor returns and combining with factor weights.

Output: {panel_id}_fama_weights.pkl containing stock weights per month for each method.

Usage:
    python estimate_sdf_fama.py [panel_id] [--config CONFIG_MODULE]
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
    from utils_factors import fama_functions as fama
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the noipca2 directory")
    sys.exit(1)


# Shared data for parallel workers. Set before Parallel() call; workers
# inherit via fork (copy-on-write) with backend='multiprocessing'.
_SHARED_DATA = {}


def compute_month_weights_fama(month_data, alpha_lst_fama, CHARS):
    """
    Compute Fama/CAPM stock weights for a single month.

    This function is called in parallel for each month.
    Large data (ff_rets, fm_rets) is read from module-level _SHARED_DATA
    to avoid serializing per worker.

    Args:
        month_data: Dict containing pre-extracted data for this month
        alpha_lst_fama: Fama alpha list
        CHARS: List of characteristic names

    Returns:
        Tuple of (month, firm_ids, fama_weights_dict, mkt_rf_value)
    """
    month = month_data['month']
    data_chars = month_data['data_chars']  # numpy array
    data_mve = month_data['data_mve']      # numpy array
    firm_ids = month_data['firm_ids']

    # Read large shared data (inherited via fork, not serialized)
    ff_rets = _SHARED_DATA['ff_rets']
    fm_rets = _SHARED_DATA['fm_rets']

    # Create DataFrame for characteristics (needed by fama methods)
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

    print(f"  Month {month}: Fama/CAPM complete at {now()}")

    return month, firm_ids, fama_weights, mkt_rf_value, market_weight


def run(panel_id, model_name):
    """
    Run Fama/CAPM SDF estimation.

    Loads panel and Fama factors from disk.
    """
    start_time = time.time()

    CONFIG = config.get_model_config(model_name)
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']
    alpha_lst_fama = CONFIG['alpha_lst_fama']

    # =========================================================================
    # LOAD REQUIRED DATA
    # =========================================================================
    print("="*70)
    print("FAMA/CAPM SDF ESTIMATION (COMPUTE WEIGHTS)")
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

    # Load Fama results
    fama_path = os.path.join(config.DATA_DIR, f"{panel_id}_fama.pkl")
    print(f"Loading fama factors from {fama_path}...")
    with open(fama_path, 'rb') as f:
        fama_data = pickle.load(f)
    ff_rets = fama_data['ff_returns']
    fm_rets = fama_data['fm_returns']

    # Evaluation range: need 360 months of history for ridge regression
    eval_start = start + 360
    eval_end = end

    print(f"\nConfiguration:")
    print(f"  N_JOBS (fama, {model_name}): {config.get_n_jobs_for_step('estimate_fama', model_name)}")
    print(f"  Alphas (Fama): {alpha_lst_fama}")
    print(f"  Evaluation range: {eval_start} to {eval_end}")

    # =========================================================================
    # SET SHARED DATA FOR PARALLEL WORKERS
    # =========================================================================
    _SHARED_DATA['ff_rets'] = ff_rets
    _SHARED_DATA['fm_rets'] = fm_rets

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
        data_mve = np.ascontiguousarray(data.mve.to_numpy())

        month_data_list.append({
            'month': month,
            'data_chars': data_chars,
            'data_mve': data_mve,
            'firm_ids': firm_ids,
        })

    print(f"  Prepared {len(month_data_list)} months in {time.time()-t0:.1f}s")

    # =========================================================================
    # PARALLEL WEIGHT COMPUTATION
    # =========================================================================
    print(f"\n{'-'*70}")
    n_jobs = config.get_n_jobs_for_step('estimate_fama', model_name)
    print(f"Computing Fama/CAPM stock weights (n_jobs={n_jobs})...")
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
                delayed(compute_month_weights_fama)(
                    md, alpha_lst_fama, CHARS
                )
                for md in chunk
            )

        for month, firm_ids, fama_weights, mkt_rf_value, market_weight in chunk_results:
            all_weights[month] = {
                'firm_ids': firm_ids,
                'fama': fama_weights,
                'mkt_rf': mkt_rf_value,
                'market_weight': market_weight,  # Save for DKKM
            }

        del chunk_results
        gc.collect()

    # Clear shared data
    _SHARED_DATA.clear()

    elapsed = time.time() - t0
    print(f"\n[OK] Fama/CAPM weights computed in {fmt(elapsed)} at {now()}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'weights': all_weights,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'alpha_lst_fama': alpha_lst_fama,
        'start': eval_start,
        'end': eval_end,
    }

    output_file = os.path.join(config.DATA_DIR, f"{panel_id}_stock_weights_fama.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[OK] Fama/CAPM weights saved to: {output_file}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("FAMA/CAPM ESTIMATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {now()}")
    print(f"Total runtime: {fmt(total_time)}")
    print(f"{'='*70}")


def main():
    """Standalone entry point: parse args and run."""
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(
        script_name='estimate_sdf_fama'
    )
    run(panel_id, model_name)


if __name__ == "__main__":
    main()
