"""
DKKM (Random Fourier Features) factor computation.

This is a near-exact copy of the root codebase's dkkm_functions.py.
All functions match the root logic precisely.

ROOT FILE: c:\\Users\\kerry\\repos\\CodeNew\\dkkm_functions.py

KEY DIFFERENCES FROM noipca/utils_factors/dkkm_functions.py:
  1. rank_standardize uses numpy argsort with formula (ranks-0.5)/N-0.5
     (noipca used scipy rankdata with (ranks-1)/(N-1)-0.5 — DIFFERENT VALUES)
  2. rff() uses pure numpy internally (avoids DataFrame overhead at large D)
  3. ridge_regr() is inline here (noipca used separate ridge_utils.py)
  4. mve_data() takes alpha_lst (already scaled by nfeatures from caller)
     (noipca scaled internally — same net effect but different interface)
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# Shared data for parallel workers. Set before Parallel() call; workers
# inherit via fork (copy-on-write) with backend='multiprocessing'.
_SHARED_DATA = {}


# =============================================================================
# rank_standardize (pure numpy)
# ROOT: dkkm_functions.py lines 10-13
#
# CRITICAL DIFFERENCE FROM noipca:
#   Root/noipca2: (ranks - 0.5) / N - 0.5
#   noipca:       (ranks - 1) / (N - 1) - 0.5
#
#   For N=1000: Root maps rank 1 to -0.4995, rank 1000 to 0.4995
#               noipca maps rank 1 to -0.5, rank 1000 to 0.5
# =============================================================================
def rank_standardize(values):
    """
    DKKM rank-based standardization.

    ROOT: dkkm_functions.py lines 10-13
    Formula: (ranks - 0.5) / N - 0.5

    Uses numpy argsort instead of pandas .rank() for speed.
    Produces ordinal ranks (ties broken by position), which is equivalent
    to pandas method='average' when data is continuous with no ties.

    Args:
        values: (N, P) numpy ndarray to rank-standardize

    Returns:
        (N, P) numpy ndarray with values in ~[-0.5, 0.5], C-contiguous
    """
    n = values.shape[0]
    order = np.argsort(values, axis=0, kind='quicksort')
    # Pre-allocate with C-contiguous layout
    ranks = np.empty(values.shape, dtype=np.float64, order='C')
    col_idx = np.arange(values.shape[1])
    ranks[order, col_idx] = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    result = (ranks - 0.5) / n - 0.5
    # Ensure result is C-contiguous for downstream operations
    return np.ascontiguousarray(result)


# =============================================================================
# rff (Random Fourier Features)
# ROOT: dkkm_functions.py lines 16-25
#
# Pure numpy implementation. Avoids DataFrame overhead for large feature
# counts (D=18000+) where pd.concat and DataFrame construction dominate.
#
# Returns rank-standardized features only (noipca2 simplification).
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
        rank_standardized_features as ndarray (N, D)
    """
    # ROOT line 17: rank-standardize characteristics
    X = rank_standardize(data.values)

    # ROOT lines 18-19: add risk-free rate for BGN model
    if model == 'bgn':
        X = np.column_stack([X, rf.values])

    # Ensure C-contiguous for optimal matrix multiplication performance
    X = np.ascontiguousarray(X)

    # ROOT line 20: W @ X.T — (D/2, L) @ (L, N) = (D/2, N)
    Z = W @ X.T

    # ROOT lines 21-22: sin and cos features
    # ROOT line 23: concatenate [sin; cos] then transpose to (N, D)
    # Use column_stack instead of vstack+transpose for better memory layout
    features = np.ascontiguousarray(np.vstack([np.sin(Z), np.cos(Z)]).T)

    # Return rank-standardized features only
    return rank_standardize(features)


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
    # Ensure C-contiguous arrays for optimal BLAS performance
    signals = np.ascontiguousarray(signals)
    labels = np.ascontiguousarray(labels)

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
        # Pre-allocate for better memory performance
        intermed = np.empty((multiplied.shape[0], len(shrinkage_list)), dtype=multiplied.dtype)
        for i, z in enumerate(shrinkage_list):
            intermed[:, i:i+1] = (1 / (eigenvalues.reshape(-1, 1) + 360*z)) * multiplied
        betas = eigenvectors @ intermed

    else:
        # ROOT lines 146-156: Over-parametrized regime (P >= T)
        # Use kernel trick: eigendecompose XX' instead of X'X
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T)
        means = labels.reshape(-1, 1)
        multiplied = eigenvectors.T @ means

        # ROOT line 153: same 360*z penalty
        # Pre-allocate for better memory performance
        intermed = np.empty((multiplied.shape[0], len(shrinkage_list)), dtype=multiplied.dtype)
        for i, z in enumerate(shrinkage_list):
            intermed[:, i:i+1] = (1 / (eigenvalues.reshape(-1, 1) + 360*z)) * multiplied
        tmp = eigenvectors.T @ signals
        betas = tmp.T @ intermed

    return betas


# =============================================================================
# factors
# ROOT: dkkm_functions.py lines 46-70
#
# Parallel computation of factor returns across months.
# Returns rank-standardized factor returns only (noipca2 simplification).
# Uses _SHARED_DATA + backend='multiprocessing' (fork on Linux) so panel
# and W are inherited via copy-on-write instead of serialized to each worker.
# =============================================================================
def _compute_monthly_dkkm(month, model, chars):
    """
    Module-level worker function for parallel DKKM factor computation.

    Reads panel and W from _SHARED_DATA (inherited via fork/COW).
    """
    panel = _SHARED_DATA['panel']
    W = _SHARED_DATA['W']
    data = panel.loc[month]

    if model == 'bgn':
        rf = data.rf_stand
    else:
        rf = None

    weights_rs = rff(data[chars], rf, W=W, model=model)
    return (month, (weights_rs.T @ data.xret.values).astype(np.float32))


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
        f_rs: Rank-standardized factor returns DataFrame
    """
    D = W.shape[0] * 2  # total features (sin + cos)

    # Set shared data for workers (inherited via fork/COW)
    _SHARED_DATA['panel'] = panel
    _SHARED_DATA['W'] = W

    # ROOT lines 57-59: parallel computation
    lst = Parallel(n_jobs=n_jobs, verbose=0, backend='multiprocessing')(
        delayed(_compute_monthly_dkkm)(month, model, chars)
        for month in range(start, end+1)
    )

    _SHARED_DATA.clear()

    # ROOT lines 62-70: assemble DataFrame
    months = [x[0] for x in lst]
    frets = np.vstack([x[1] for x in lst])
    f_rs = pd.DataFrame(frets,
                         columns=[str(i) for i in range(D)])
    f_rs.index = months
    f_rs.index.name = 'month'
    f_rs.sort_index(inplace=True)

    return f_rs


# =============================================================================
# mve_data
# ROOT: dkkm_functions.py lines 161-194
#
# DKKM portfolio based on past 360 months of factor returns.
# Takes alpha_lst which is ALREADY SCALED by nfeatures from the caller.
# (Root main_revised.py line 228: dkkm.mve_data(..., nfeatures*alpha, ...))
#
# Always includes market factor (noipca2 simplification):
#   - For alpha=0: standard ridge on full X (including market column)
#   - For alpha>0: augment X to avoid penalizing market column
#     Augmentation: sqrt(360*alph) * I[:-1] — penalty only on non-market features
#
# The effective penalty is 360 * alpha_value, where alpha_value = nfeatures * alpha
# from the caller. So total = 360 * nfeatures * alpha.
# =============================================================================
def mve_data(f, month, alpha_lst, mkt_rf):
    """
    Compute mean-variance efficient portfolio from factor returns.

    ROOT: dkkm_functions.py lines 161-194

    Args:
        f: DataFrame of factor returns
        month: Current month
        alpha_lst: Array of ridge penalties (ALREADY SCALED by nfeatures from caller)
        mkt_rf: Market return Series (always required - market always included)

    Returns:
        DataFrame of portfolio weights (columns = alpha values)
    """
    # ROOT line 163: past 360 months of factor returns
    X = f.loc[month-360:month-1].dropna().to_numpy()

    # ROOT lines 166-167: append market returns column (always)
    X = np.column_stack((X, mkt_rf.loc[month-360:month-1].dropna().to_numpy()))

    # ROOT line 169: target = ones (maximize expected return)
    y = np.ones(len(X))
    index_cols = list(f.columns) + ['mkt_rf']

    # ROOT lines 172-193: compute betas for each alpha
    # Handle market (don't penalize last variable)
    betas_list = []

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

    return betas_df
