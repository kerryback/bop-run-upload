"""
Oracle decomposition of the achievable Sharpe ratio.

Given the TRUE conditional moments of excess returns (mu_t, Sigma_t) each month and a matrix of
characteristic-based features Phi_t (N x P), the best portfolio whose weights are w = Phi_t theta is

    max_theta  (theta' a_t) / sqrt(theta' B_t theta),   a_t = Phi_t' mu_t,  B_t = Phi_t' Sigma_t Phi_t

with value SR_t(Phi)^2 = a_t' B_t^{-1} a_t  (the "conditional oracle" -- theta re-optimised every month),
while the object that a rolling "regress 1 on factor returns" (FMR / FFC / DKKM ridge) estimates is the
CONSTANT-theta portfolio

    theta* = E[f f']^{-1} E[f],   E[f] = mean_t a_t,  E[ff'] = mean_t (B_t + a_t a_t')

evaluated with the true conditional moments ("population" FMR / DKKM, i.e. no estimation noise).
Comparing bases (linear in characteristics vs. nonlinear / RFF) isolates the *approximation* gap that
a complexity method could in principle harvest, from the *estimation* gap.
"""
import numpy as np
import pandas as pd
import scipy.linalg as la


# ----------------------------------------------------------------------------- feature construction
def rank_standardize(arr):
    """DKKM cross-sectional rank standardisation to (-0.5, 0.5). Works on DataFrame or 2-d array."""
    df = pd.DataFrame(arr)
    ranks = df.rank(axis=0)
    ranks = (ranks - 0.5) / len(ranks) - 0.5
    return ranks.to_numpy()


def rff_features(Xrank, W, rf=None, restandardize=True):
    """Random Fourier features exactly as in Code/dkkm_functions.rff: W is (P/2 x d) already scaled by gamma,
    X is rank-standardised chars (N x d); if rf is given it is appended as an extra column (BGN)."""
    X = Xrank if rf is None else np.column_stack([Xrank] + [np.full((len(Xrank), 1), v) for v in np.atleast_1d(rf)])
    Z = X @ W.T                       # N x P/2
    F = np.column_stack([np.sin(Z), np.cos(Z)])
    return rank_standardize(F) if restandardize else F


def draw_W(P, d, gamma_grid, rng):
    W = rng.standard_normal(size=(P // 2, d))
    gamma = rng.choice(gamma_grid, size=(P // 2, 1))
    return gamma * W


def poly2_features(Xrank, rf=None):
    """ranks, squares, pairwise products (+ rf and rf x ranks)."""
    cols = [Xrank]
    d = Xrank.shape[1]
    cols.append(Xrank ** 2)
    cols.append(np.column_stack([Xrank[:, i] * Xrank[:, j] for i in range(d) for j in range(i + 1, d)]))
    if rf is not None:
        for v in np.atleast_1d(rf):
            cols.append(np.column_stack([np.full((len(Xrank), 1), v), Xrank * v]))
    return np.column_stack(cols)


def bin_features(Xraw, nbins=10, npair=5):
    """decile dummies per characteristic + (npair x npair) cell dummies for every pair of characteristics.
    Fully nonparametric 1-d and 2-d (interaction) effects."""
    N, d = Xraw.shape
    cols = []
    q = np.zeros((N, d), dtype=int)
    for j in range(d):
        q[:, j] = pd.qcut(pd.Series(Xraw[:, j]).rank(method="first"), nbins, labels=False)
        D = np.zeros((N, nbins)); D[np.arange(N), q[:, j]] = 1
        cols.append(D)
    qp = np.zeros((N, d), dtype=int)
    for j in range(d):
        qp[:, j] = pd.qcut(pd.Series(Xraw[:, j]).rank(method="first"), npair, labels=False)
    for i in range(d):
        for j in range(i + 1, d):
            D = np.zeros((N, npair * npair)); D[np.arange(N), qp[:, i] * npair + qp[:, j]] = 1
            cols.append(D)
    return np.column_stack(cols)


# ----------------------------------------------------------------------------- two-pass evaluation
def max_sr(mu, Sigma):
    return np.sqrt(mu @ la.solve(Sigma, mu, assume_a="pos"))


def evaluate_bases(months_data, feature_fn, names, zgrid_rel=(0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
                   cond_oracle_maxP=None, verbose=False):
    """months_data: list of dicts with keys mu, Sigma, X_raw, X_rank, rf.
    feature_fn(md) -> dict name -> Phi (N x P) (deterministic given md).
    Pass 1 accumulates E[f], E[ff'] per basis; pass 2 evaluates theta*(z) for every z in zgrid_rel
    (z relative to trace(E[ff'])/P) with the true conditional moments, plus the conditional oracle for small P.
    Returns dict name -> results."""
    Tm = len(months_data)
    Ef, Eff, P = {}, {}, {}
    for k, md in enumerate(months_data):
        feats = feature_fn(md)
        for name in names:
            Phi = feats[name]
            a = Phi.T @ md["mu"]
            if name not in Ef:
                P[name] = Phi.shape[1]; Ef[name] = np.zeros(P[name]); Eff[name] = np.zeros((P[name], P[name]))
            B = Phi.T @ (md["Sigma"] @ Phi)
            Ef[name] += a / Tm
            Eff[name] += (0.5 * (B + B.T) + np.outer(a, a)) / Tm
        if verbose and k % 100 == 0:
            print(f"    pass1 {k+1}/{Tm}", flush=True)
    thetas, scales = {}, {}
    for name in names:
        scales[name] = np.trace(Eff[name]) / P[name]
        thetas[name] = []
        for zr in zgrid_rel:
            M = Eff[name] + zr * scales[name] * np.eye(P[name])
            try:
                th = la.solve(M, Ef[name], assume_a="pos")
            except la.LinAlgError:
                th = la.pinvh(M) @ Ef[name]
            thetas[name].append(th)
    mean = {n: np.zeros((Tm, len(zgrid_rel))) for n in names}
    var = {n: np.zeros((Tm, len(zgrid_rel))) for n in names}
    cond = {n: np.full(Tm, np.nan) for n in names}
    for k, md in enumerate(months_data):
        feats = feature_fn(md)
        for name in names:
            Phi = feats[name]
            a = Phi.T @ md["mu"]
            SPhi = md["Sigma"] @ Phi
            TH = np.column_stack(thetas[name])            # P x nz
            mean[name][k] = a @ TH
            var[name][k] = np.einsum("pj,pj->j", TH, (Phi.T @ SPhi) @ TH)
            if cond_oracle_maxP is None or P[name] <= cond_oracle_maxP:
                if P[name] < 0.5 * len(md["mu"]):
                    B = Phi.T @ SPhi
                    B = 0.5 * (B + B.T) + 1e-10 * np.trace(B) / len(B) * np.eye(len(B))
                    cond[name][k] = np.sqrt(max(a @ la.solve(B, a, assume_a="pos"), 0))
        if verbose and k % 100 == 0:
            print(f"    pass2 {k+1}/{Tm}", flush=True)
    out = {}
    for name in names:
        sr = mean[name] / np.sqrt(np.maximum(var[name], 1e-300))      # Tm x nz conditional SR
        unc = mean[name].mean(0) / np.sqrt((var[name] + mean[name] ** 2).mean(0) - mean[name].mean(0) ** 2)
        out[name] = {"P": P[name], "zgrid_rel": list(zgrid_rel), "cond_sr_mean": sr.mean(0), "unc_sr": unc,
                     "cond_sr_ts": sr, "cond_oracle_ts": cond[name], "cond_oracle_mean": np.nanmean(cond[name]),
                     "scale": scales[name]}
    return out


def level_standardize(X_raw, med, iqr, clip=3.0):
    """levels standardized by FIXED (time-invariant) per-characteristic stats -> preserves level and
    aggregate-state information that per-month rank-standardization destroys."""
    return np.clip((X_raw - med) / iqr, -clip, clip)


def build_feature_sets(X_raw, X_rank, rf, Wdict, include_rf=True, X_lev=None, Wdict_lev=None,
                       nest_linear=True):
    """Returns dict name -> Phi (N x P).  X_raw/X_rank: N x d arrays of the 5 characteristics.
    rf: scalar standardised rate (BGN) or None.  Wdict: name -> W matrix for RFF bases."""
    N, d = X_rank.shape
    one = np.ones((N, 1))
    feats = {}
    feats["fmr_raw"] = np.column_stack([one, X_raw])                     # span of FMR (raw chars + const)
    feats["lin_rank"] = np.column_stack([one, X_rank])                   # linear in rank-chars (KPS-6 / BSV)
    if rf is not None and include_rf:
        _rfv = np.atleast_1d(rf)
        feats["lin_rank_rf"] = np.column_stack([one, X_rank] + [np.column_stack([np.full((N, 1), v), X_rank * v]) for v in _rfv])
    feats["poly2"] = np.column_stack([one, poly2_features(X_rank, rf if include_rf else None)])
    feats["bins"] = np.column_stack([one, bin_features(X_raw)])
    for name, W in Wdict.items():
        feats[name] = np.column_stack([one, rff_features(X_rank, W, rf if include_rf else None)])

    # ---- nested ceiling bases (2026-09-04, decision 6: "require nesting") ----
    # `room` is defined as nl_ceil - lin_ceil, i.e. what a NONLINEAR method can add
    # on top of any linear one. That reading is only valid if the nonlinear basis
    # SPANS the linear one. Measured (5 chars, N=500), the worst unexplained
    # fraction of a linear rank column is:
    #     poly2   3.4e-28  -> nests (it already contains X_rank)
    #     bins    6.1e-03  -> does NOT nest (decile dummies cannot reproduce a line)
    #     rff36   1.7e-02  -> does NOT nest
    #     rff360  4.1e-04  -> does NOT nest
    # so `bins` and `rff*` could score BELOW lin_rank and make `room` negative --
    # which is what the two GS gamma(x) rows in results/grid_summary.csv show
    # (room = -0.0004 / -0.0005 alongside gap = +0.038 / +0.041).
    #
    # These `*_n` bases append X_rank so nesting holds by construction and
    # room >= 0. The originals are kept so every published number stays
    # reproducible, and the difference (e.g. rff360_n - rff360) is itself the
    # interesting quantity: how much of the LINEAR signal pure RFF fails to span.
    if nest_linear:
        _lin_cols = [X_rank] if X_lev is None else [X_rank, X_lev]
        feats["bins_n"] = np.column_stack([one] + _lin_cols + [bin_features(X_raw)])
        for name, W in Wdict.items():
            feats[name + "_n"] = np.column_stack(
                [one] + _lin_cols + [rff_features(X_rank, W, rf if include_rf else None)])

    if X_lev is not None:
        feats["lin_rank_lev"] = np.column_stack([one, X_rank, X_lev])
        XX = np.column_stack([X_rank, X_lev])
        for name, W in (Wdict_lev or {}).items():
            feats[name] = np.column_stack([one, rff_features(XX, W, rf if include_rf else None)])
            if nest_linear:
                feats[name + "_n"] = np.column_stack(
                    [one, X_rank, X_lev, rff_features(XX, W, rf if include_rf else None)])
    return feats
