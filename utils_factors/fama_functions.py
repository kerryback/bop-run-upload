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


# =============================================================================
# fama_french
# ROOT: fama_functions.py lines 9-71
#
# Constructs Fama-French factor portfolios using 2x3 sorts.
# Size median split x characteristic 30/40/30 tercile split.
# Value-weighted within each of the 6 portfolios.
# Returns (N, K+1) array including market portfolio.
# =============================================================================
def fama_french(data, chars, **kwargs):
    """
    Compute Fama-French factor portfolio weights for a single month.

    ROOT: fama_functions.py lines 9-71

    Args:
        data: DataFrame of characteristics (N firms)
        chars: List of characteristic names
        **kwargs: Must include 'mve' (market value of equity)

    Returns:
        (N, K+1) numpy array of portfolio weights (factors + market)
    """
    factor_dct = {}

    # ROOT lines 13-16: determine factor names based on number of chars
    if len(chars) == 5:
        names = ["smb", "hml", "cma", "rmw", "umd"]
    else:
        names = ["smb", "hml", "umd"]
    name_dct = dict(zip(chars, names))

    # ROOT lines 20-21: sort on size (median split)
    big = data["size"] > data["size"].median()
    small = 1 - big
    mve = kwargs["mve"]

    # ROOT lines 24-51: for each non-size characteristic
    for char in [c for c in chars if c != "size"]:

        # ROOT lines 27-29: tercile split on characteristic
        low = data[char] <= data[char].quantile(0.3)
        high = data[char] > data[char].quantile(0.7)
        med = 1 - low - high

        # ROOT lines 32-37: form six portfolios (value-weighted)
        high_big = mve * (high & big)
        high_small = mve * (high & small)
        low_big = mve * (low & big)
        low_small = mve * (low & small)
        med_big = mve * (med & big)
        med_small = mve * (med & small)

        # ROOT lines 40-42: normalize by portfolio market cap
        for ser in [high_big, high_small, low_big, low_small, med_big, med_small]:
            if ser.sum() != 0:
                ser /= ser.sum()

        # ROOT lines 45-48: long-short factor portfolio
        factor = 0.5 * (
            high_big + high_small
            - low_big - low_small
        )

        # ROOT line 51: store factor weights
        factor_dct[name_dct[char]] = factor.to_numpy()

        # ROOT lines 54-58: define SMB using book-to-market sorts
        if char == "bm":
            smb = (
                high_small + med_small + low_small
                - high_big - med_big - low_big
            ) / 3
            factor_dct["smb"] = smb.to_numpy()

    # ROOT lines 61-62: create output DataFrame
    df = pd.DataFrame(factor_dct)
    df.index = data.index

    # ROOT lines 65-66: CMA is low minus high (flip sign)
    if "cma" in names:
        df["cma"] *= -1

    # ROOT lines 69: value-weighted market portfolio
    df['mkt_rf'] = mve / mve.sum()

    # ROOT line 71: return as numpy array
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

    Note: Standardization is COMMENTED OUT in root (lines 78-81).
    This uses raw characteristics directly.

    Args:
        data: DataFrame of characteristics
        chars: List of characteristic names
        **kwargs: Accepts additional arguments for compatibility

    Returns:
        (N, K+1) numpy array of portfolio weights (factors + market)
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
    P = pd.DataFrame(P[:, 1:])
    P *= 2 / P.abs().sum()

    # ROOT lines 89-92: set column names
    if len(chars) == 5:
        P.columns = ["smb", "hml", "cma", "rmw", "umd"]
    else:
        P.columns = ["smb", "hml", "umd"]

    # ROOT line 95: equal-weighted market portfolio
    P['mkt_rf'] = 1 / len(data)

    # ROOT line 97: return as numpy array
    return P.to_numpy()


# =============================================================================
# factors
# ROOT: fama_functions.py lines 103-121
#
# Parallel computation of factor returns across months.
# =============================================================================
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
        Factor returns DataFrame indexed by month
    """
    # ROOT lines 104-110: monthly return computation
    def monthly_rets(month):
        data = panel.loc[month]
        weights = method(data[chars], chars, mve=data.mve)
        rets = data.xret.to_numpy().reshape(-1, 1)
        wts = weights.T @ rets
        wts = pd.DataFrame(wts.T, index=[month])
        return wts

    # ROOT lines 111-113: parallel computation
    lst = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(monthly_rets)(month) for month in range(start, end+1)
    )

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
