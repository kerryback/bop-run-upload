"""
Fama-French and Fama-MacBeth factor computation.

This is a near-exact copy of the root codebase's fama_functions.py.
All functions match the root logic precisely.

ROOT FILE: c:\\Users\\kerry\\repos\\CodeNew\\fama_functions.py

KEY DIFFERENCES FROM noipca/utils_factors/fama_functions.py:
  1. fama_french uses kwargs["mve"] (root style) vs explicit mve parameter
  2. fama_macbeth has no standardization (root has it commented out)
  3. mve_data uses sklearn Ridge (root) vs eigendecomposition (noipca)
  4. factors() signature: method comes first, no stdz_fm parameter
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import scipy.linalg as linalg
from sklearn.linear_model import Ridge


# Shared data for parallel workers. Set before Parallel() call; workers
# inherit via fork (copy-on-write) with backend='multiprocessing'.
_SHARED_DATA = {}


# =============================================================================
# fama_french
# ROOT: fama_functions.py lines 9-71
#
# Constructs Fama-French factor portfolios using 2x3 sorts.
# Size median split x characteristic 30/40/30 tercile split.
# Value-weighted within each of the 6 portfolios.
# Returns (N, K+1) array including market portfolio.
#
# DEVIATION FROM ROOT: SMB is computed from a simple value-weighted size sort
# (long small-cap, short large-cap) rather than the traditional 2x3 BM sort.
# This decouples SMB from BM, so "bm" need not be in chars to get SMB.
# =============================================================================
def fama_french(data, chars, **kwargs):
    """
    Compute Fama-French factor portfolio weights for a single month.

    ROOT: fama_functions.py lines 9-71

    Always produces SMB (from a simple value-weighted size sort) and market
    (mkt_rf) regardless of which chars are supplied. Additional factors are
    produced for each non-size characteristic in chars.

    Args:
        data: DataFrame of characteristics (N firms). Must include "size".
        chars: List of characteristic names. "size" must be present.
        **kwargs: Must include 'mve' (market value of equity)

    Returns:
        (N, K+1) numpy array of portfolio weights
        Column order: smb, [one factor per non-size char], mkt_rf
    """
    factor_dct = {}

    # Map each non-size characteristic to its factor name.
    _CHAR_TO_FACTOR = {"bm": "hml", "agr": "cma", "roe": "rmw",
                       "mom": "umd", "mkt_lev": "mkt_lev"}
    name_dct = {c: _CHAR_TO_FACTOR[c] for c in chars if c in _CHAR_TO_FACTOR}

    # Size median split — used for SMB and all 2x3 sorts
    big = data["size"] > data["size"].median()
    small = 1 - big
    mve = kwargs["mve"]

    # SMB: simple value-weighted size sort (small minus big).
    # Does not require "bm" in chars; always produced.
    smb_small = mve * small
    smb_big = mve * big
    if smb_small.sum() != 0:
        smb_small = smb_small / smb_small.sum()
    if smb_big.sum() != 0:
        smb_big = smb_big / smb_big.sum()
    factor_dct["smb"] = (smb_small - smb_big).to_numpy()

    # For each non-size characteristic: 2x3 sort → long-short factor
    for char in [c for c in chars if c != "size"]:

        # Tercile split on characteristic
        low = data[char] <= data[char].quantile(0.3)
        high = data[char] > data[char].quantile(0.7)
        med = 1 - low - high

        # Six value-weighted portfolios
        high_big = mve * (high & big)
        high_small = mve * (high & small)
        low_big = mve * (low & big)
        low_small = mve * (low & small)
        med_big = mve * (med & big)
        med_small = mve * (med & small)

        for ser in [high_big, high_small, low_big, low_small, med_big, med_small]:
            if ser.sum() != 0:
                ser /= ser.sum()

        # Long-short factor (high minus low, averaged across size groups)
        factor = 0.5 * (high_big + high_small - low_big - low_small)
        factor_dct[name_dct[char]] = factor.to_numpy()

    # Create output DataFrame
    df = pd.DataFrame(factor_dct)
    df.index = data.index

    # CMA is investment factor: low investment minus high (flip sign)
    if "cma" in name_dct.values():
        df["cma"] *= -1

    # Value-weighted market portfolio (always last column)
    df['mkt_rf'] = mve / mve.sum()

    return df.to_numpy()


# =============================================================================
# fama_macbeth
# ROOT: fama_functions.py lines 74-97
#
# Cross-sectional regression portfolios (Fama-MacBeth).
# NO standardization of characteristics (commented out in root).
# Uses equal-weighted market portfolio.
# =============================================================================
def fama_macbeth(data, chars, **kwargs):
    """
    Compute Fama-MacBeth factor portfolio weights for a single month.

    ROOT: fama_functions.py lines 74-97

    Always produces SMB (from the "size" column in the cross-sectional
    regression) and market (mkt_rf) regardless of which chars are supplied.

    Note: Standardization is COMMENTED OUT in root (lines 78-81).
    This uses raw characteristics directly.

    Args:
        data: DataFrame of characteristics. Must include "size".
        chars: List of characteristic names. "size" must be present.
        **kwargs: Accepts additional arguments for compatibility

    Returns:
        (N, K+1) numpy array of portfolio weights
        Column order: smb, [one factor per non-size char], mkt_rf
    """
    # ROOT line 75: drop NaN values
    d = data.dropna()

    # ROOT lines 78-81: standardization COMMENTED OUT in root
    # d = d.apply(
    #     lambda x: x / x.std() if x.std() != 0 else 0,
    #     axis=0
    # )

    # ROOT lines 84-86: FM portfolios via pseudo-inverse
    X = d.to_numpy()
    X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis=1)
    P = X @ linalg.pinvh(X.T @ X)

    # ROOT lines 87-88: drop intercept column, normalize
    P = pd.DataFrame(P[:, 1:], index=d.index)
    P *= 2 / P.abs().sum()

    # ROOT lines 89-92: set column names dynamically from char→factor mapping.
    # FM uses all chars (including size) in the cross-sectional regression, so
    # P has len(chars) columns after dropping the intercept. "size" → "smb".
    _CHAR_TO_FACTOR = {"size": "smb", "bm": "hml", "agr": "cma", "roe": "rmw",
                       "mom": "umd", "mkt_lev": "mkt_lev"}
    P.columns = [_CHAR_TO_FACTOR.get(c, c) for c in chars]

    # ROOT line 95: equal-weighted market portfolio (always last column)
    P['mkt_rf'] = 1 / len(d)

    # Reindex to original data shape so dimensions match rets in caller
    P = P.reindex(data.index, fill_value=0)

    # ROOT line 97: return as numpy array
    return P.to_numpy()


# =============================================================================
# factors
# ROOT: fama_functions.py lines 103-121
#
# Parallel computation of factor returns across months.
# Uses _SHARED_DATA + backend='multiprocessing' (fork on Linux) so panel
# is inherited via copy-on-write instead of serialized to each worker.
# =============================================================================
def _compute_monthly_fama(month, method, chars):
    """
    Module-level worker function for parallel fama factor computation.

    Reads panel from _SHARED_DATA (inherited via fork/COW).
    """
    panel = _SHARED_DATA['panel']
    data = panel.loc[month]
    weights = method(data[chars], chars, mve=data.mve)
    rets = data.xret.to_numpy().reshape(-1, 1)
    wts = weights.T @ rets
    return pd.DataFrame(wts.T, index=[month])


def factors(method, panel, n_jobs, start, end, chars):
    """
    Compute panel of factor returns for a given method.

    ROOT: fama_functions.py lines 103-121

    Args:
        method: Function (fama_french or fama_macbeth)
        panel: Panel data with multi-index (month, firmid)
        n_jobs: Number of parallel jobs
        start: Start month
        end: End month
        chars: List of characteristics

    Returns:
        Factor returns DataFrame indexed by month.
        Columns: smb, [factor per non-size char], mkt_rf
    """
    # Set shared data for workers (inherited via fork/COW)
    _SHARED_DATA['panel'] = panel

    # ROOT lines 111-113: parallel computation
    lst = Parallel(n_jobs=n_jobs, verbose=0, backend='multiprocessing')(
        delayed(_compute_monthly_fama)(month, method, chars)
        for month in range(start, end+1)
    )

    _SHARED_DATA.clear()

    # ROOT lines 119-121: concatenate results
    f = pd.concat(lst)
    f.index.name = "month"
    return f


# =============================================================================
# mve_data (for Fama methods)
# ROOT: fama_functions.py lines 124-129
#
# Uses sklearn Ridge regression with penalty 360*alpha.
# For Fama methods, alpha is always 0 (OLS).
# =============================================================================
def mve_data(f, month, alpha):
    """
    Compute mean-variance efficient portfolio of Fama factors.

    ROOT: fama_functions.py lines 124-129

    Uses sklearn Ridge (matches root exactly).
    For Fama methods, alpha should always be 0 (OLS).

    Args:
        f: DataFrame of factor returns
        month: Current month
        alpha: Ridge penalty (0 = OLS for Fama methods)

    Returns:
        Portfolio weights as Series
    """
    # ROOT line 125: past 360 months
    X = f.loc[month-360:month-1].dropna().to_numpy()
    y = np.ones(len(X))

    # ROOT lines 127-128: sklearn Ridge with penalty 360*alpha
    ridge = Ridge(fit_intercept=False, alpha=360*alpha)
    pi = ridge.fit(X=X, y=y).coef_

    # ROOT line 129: return as Series
    return pd.Series(pi, index=f.columns)
