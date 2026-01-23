"""
DKKM (Random Fourier Features) factor computation.

This is a near-exact copy of the root codebase's dkkm_functions.py.
All functions match the root logic precisely.

ROOT FILE: c:\\Users\\kerry\\repos\\CodeNew\\dkkm_functions.py

KEY DIFFERENCES FROM noipca/utils_factors/dkkm_functions.py:
  1. rank_standardize uses pandas .rank() with formula (ranks-0.5)/len(ranks)-0.5
     (noipca used scipy rankdata with (ranks-1)/(N-1)-0.5 — DIFFERENT VALUES)
  2. rff() operates entirely on DataFrames (noipca converted to numpy)
  3. ridge_regr() is inline here (noipca used separate ridge_utils.py)
  4. mve_data() takes alpha_lst (already scaled by nfeatures from caller)
     (noipca scaled internally — same net effect but different interface)
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# =============================================================================
# rank_standardize
# ROOT: dkkm_functions.py lines 10-13
#
# CRITICAL DIFFERENCE FROM noipca:
#   Root:   (ranks - 0.5) / len(ranks) - 0.5
#   noipca: (ranks - 1) / (N - 1) - 0.5
#
#   For N=1000: Root maps rank 1 to -0.4995, rank 1000 to 0.4995
#               noipca maps rank 1 to -0.5, rank 1000 to 0.5
# =============================================================================
def rank_standardize(arr):
    """
    DKKM rank-based standardization.

    ROOT: dkkm_functions.py lines 10-13
    Uses pandas .rank(axis=0) which handles DataFrames naturally.
    Formula: (ranks - 0.5) / N - 0.5

    Args:
        arr: DataFrame or Series to rank-standardize

    Returns:
        Rank-standardized DataFrame/Series (values in ~[-0.5, 0.5])
    """
    ranks = arr.rank(axis=0)
    ranks = (ranks - 0.5) / len(ranks) - 0.5
    return ranks


# =============================================================================
# rff (Random Fourier Features)
# ROOT: dkkm_functions.py lines 16-25
#
# CRITICAL: Operates entirely on DataFrames.
# W @ X.T where X is a DataFrame uses pandas __rmatmul__,
# preserving DataFrame structure throughout.
#
# Returns BOTH rank-standardized and raw versions.
# =============================================================================
def rff(data, rf, W, model):
    """
    Compute Random Fourier Features for a single month.

    ROOT: dkkm_functions.py lines 16-25

    Args:
        data: DataFrame of characteristics (N firms x L chars)
        rf: Risk-free rate Series (only used for model='bgn')
        W: (D/2, L) random weight matrix
        model: Model name ('bgn', 'kp14', 'gs21')

    Returns:
        (rank_standardized_features, raw_features) — both as DataFrames (N, D)
    """
    # ROOT line 17: rank-standardize characteristics
    X = rank_standardize(data)

    # ROOT lines 18-19: add risk-free rate for BGN model
    if model == 'bgn':
        X['rf'] = rf

    # ROOT line 20: W @ X.T — matrix multiply (W is numpy, X.T is DataFrame)
    # Result is a numpy array of shape (D/2, N)
    Z = W @ X.T

    # ROOT lines 21-22: sin and cos features
    Z1 = np.sin(Z)
    Z2 = np.cos(Z)

    # ROOT line 23: concatenate [sin; cos] then transpose to (N, D)
    arr = pd.concat([pd.DataFrame(Z1), pd.DataFrame(Z2)], axis=0).T
    arr.columns = [str(i) for i in range(arr.shape[1])]
    arr.index = data.index

    # ROOT line 25: return (rank_standardized, raw)
    return rank_standardize(arr), arr


# =============================================================================
# ridge_regr
# ROOT: dkkm_functions.py lines 118-157
#
# Ridge regression using eigenvalue decomposition for a grid of penalties.
# Uses kernel trick when P > T for efficiency.
#
# The penalty applied is 360*z where z comes from shrinkage_list.
# This means the caller must pre-scale alpha (e.g., nfeatures*alpha)
# to get the desired effective penalty of 360*nfeatures*alpha.
# =============================================================================
def ridge_regr(signals, labels, future_signals, shrinkage_list):
    """
    Ridge regression via eigendecomposition for grid of penalties.

    ROOT: dkkm_functions.py lines 118-157

    Regression: beta = (zI + S'S)^{-1}S'y = S' (zI+SS')^{-1}y
    Uses eigenvalue decomposition for efficiency:
    (zI+A)^{-1} = U (zI+D)^{-1} U'

    Args:
        signals: (T, P) design matrix
        labels: (T,) response vector
        future_signals: unused (kept for interface compatibility)
        shrinkage_list: array of ridge parameters (penalty = 360*z)

    Returns:
        betas: (P, len(shrinkage_list)) coefficient matrix
    """
    t_ = signals.shape[0]
    p_ = signals.shape[1]

    if p_ < t_:
        # ROOT lines 136-145: Standard regime (P < T)
        # Eigendecompose X'X
        eigenvalues, eigenvectors = np.linalg.eigh(signals.T @ signals)
        means = signals.T @ labels.reshape(-1, 1)
        multiplied = eigenvectors.T @ means

        # Compute for all shrinkage values at once
        # ROOT line 143: penalty is 360*z
        intermed = np.concatenate([
            (1 / (eigenvalues.reshape(-1, 1) + 360*z)) * multiplied
            for z in shrinkage_list
        ], axis=1)
        betas = eigenvectors @ intermed

    else:
        # ROOT lines 146-156: Over-parametrized regime (P >= T)
        # Use kernel trick: eigendecompose XX' instead of X'X
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T)
        means = labels.reshape(-1, 1)
        multiplied = eigenvectors.T @ means

        # ROOT line 153: same 360*z penalty
        intermed = np.concatenate([
            (1 / (eigenvalues.reshape(-1, 1) + 360*z)) * multiplied
            for z in shrinkage_list
        ], axis=1)
        tmp = eigenvectors.T @ signals
        betas = tmp.T @ intermed

    return betas


# =============================================================================
# factors
# ROOT: dkkm_functions.py lines 46-70
#
# Parallel computation of factor returns across months.
# Returns BOTH rank-standardized and non-rank-standardized factor returns.
# =============================================================================
def factors(panel, W, n_jobs, start, end, model, chars):
    """
    Compute panel of random Fourier factor returns.

    ROOT: dkkm_functions.py lines 46-70

    Args:
        panel: Panel data with multi-index (month, firmid)
        W: (D/2, L) RFF weight matrix
        n_jobs: Number of parallel jobs
        start: Start month
        end: End month
        model: Model name
        chars: List of characteristic names

    Returns:
        (f_rs, f_nors): Rank-standardized and non-rank-standardized factor returns
    """
    # ROOT lines 47-56: monthly factor return computation
    def monthly_rets(month):
        data = panel.loc[month]

        # ROOT lines 50-53: get risk-free rate for BGN
        if model == 'bgn':
            rf = data.rf_stand
        else:
            rf = None

        # ROOT line 55: compute RFF (returns both versions)
        weights_rs, weights_nors = rff(data[chars], rf, W=W, model=model)

        # ROOT line 56: factor returns = features' @ excess returns
        return (month,
                (weights_rs.T @ data.xret).astype(np.float32),
                (weights_nors.T @ data.xret).astype(np.float32))

    # ROOT lines 57-59: parallel computation
    lst = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(monthly_rets)(month) for month in range(start, end+1)
    )

    # ROOT lines 62-70: assemble DataFrames
    f_nors = pd.concat([x[2] for x in lst], axis=1).T
    f_rs = pd.concat([x[1] for x in lst], axis=1).T
    f_nors["month"] = [x[0] for x in lst]
    f_nors.sort_values(by="month", inplace=True)
    f_nors.set_index("month", inplace=True)
    f_rs["month"] = [x[0] for x in lst]
    f_rs.sort_values(by="month", inplace=True)
    f_rs.set_index("month", inplace=True)

    return f_rs, f_nors


# =============================================================================
# mve_data
# ROOT: dkkm_functions.py lines 161-194
#
# DKKM portfolio based on past 360 months of factor returns.
# Takes alpha_lst which is ALREADY SCALED by nfeatures from the caller.
# (Root main_revised.py line 228: dkkm.mve_data(..., nfeatures*alpha, ...))
#
# When include_mkt:
#   - For alpha=0: standard ridge on full X (including market column)
#   - For alpha>0: augment X to avoid penalizing market column
#     Augmentation: sqrt(360*alph) * I[:-1] — penalty only on non-market features
#
# The effective penalty is 360 * alpha_value, where alpha_value = nfeatures * alpha
# from the caller. So total = 360 * nfeatures * alpha.
# =============================================================================
def mve_data(f, month, alpha_lst, mkt_rf=None):
    """
    Compute mean-variance efficient portfolio from factor returns.

    ROOT: dkkm_functions.py lines 161-194

    Args:
        f: DataFrame of factor returns
        month: Current month
        alpha_lst: Array of ridge penalties (ALREADY SCALED by nfeatures from caller)
        mkt_rf: Market return Series (if including market)

    Returns:
        DataFrame of portfolio weights (columns = alpha values)
    """
    # ROOT line 163: past 360 months of factor returns
    X = f.loc[month-360:month-1].dropna().to_numpy()
    include_mkt = mkt_rf is not None

    # ROOT lines 166-167: append market returns column
    if include_mkt:
        X = np.column_stack((X, mkt_rf.loc[month-360:month-1].dropna().to_numpy()))

    # ROOT line 169: target = ones (maximize expected return)
    y = np.ones(len(X))
    index_cols = list(f.columns) + (['mkt_rf'] if include_mkt else [])

    # ROOT lines 172-193: compute betas for each alpha
    betas_list = []

    if include_mkt:
        # ROOT lines 174-184: handle market (don't penalize last variable)
        # For alpha=0: standard ridge
        beta = ridge_regr(X, y, None, np.array([0]))  # shape (p_, 1)
        betas_list.append(beta.reshape(-1))

        for alph in alpha_lst:
            if alph > 0:
                # ROOT lines 179-183: augment to avoid penalizing market
                # Augmentation adds sqrt(360*alph) penalty rows for all vars except last
                X_aug = np.concatenate(
                    (X, np.sqrt(360 * alph) * np.eye(X.shape[1])[:-1]),
                    axis=0
                )
                y_aug = np.concatenate([y, np.zeros((X.shape[1] - 1,))])

                beta = ridge_regr(X_aug, y_aug, None, np.array([0]))  # shape (p_, 1)
                betas_list.append(beta.reshape(-1))

        # ROOT line 187: assemble DataFrame
        betas_df = pd.DataFrame(
            np.column_stack(betas_list),
            index=index_cols,
            columns=alpha_lst
        )

    else:
        # ROOT lines 189-192: standard ridge for all alphas at once
        betas = ridge_regr(X, y, None, alpha_lst)  # shape (p_, len(alpha_lst))
        betas_df = pd.DataFrame(betas, index=index_cols, columns=alpha_lst)

    return betas_df
