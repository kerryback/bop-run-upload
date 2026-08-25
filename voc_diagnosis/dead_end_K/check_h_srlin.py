"""
CHECK H: the decisive test. Does raising K actually lower SR_lin/SR*, the best Sharpe
attainable inside the LINEAR characteristic span (= exactly Fama-MacBeth's feasible set)?

SR_X = sqrt( rp' X (X' Sigma X)^{-1} X' rp )   and   SR* = sqrt( rp' Sigma^{-1} rp )

X = [1, log mve, bm] -- the basis the audit measured at 91.7% of max Sharpe in BGN.
If SR_lin/SR* does not fall as K rises, the construction does not bite and the channel
is wrong, however exact the invariance is.
"""
import sys, os, importlib; sys.path.insert(0, os.getcwd())
import numpy as np

SEED, N, T, TS = 20260825, 300, 100, [70, 90, 98]


def sr_in_span(X, Sig, rp):
    M = X.T @ Sig @ X
    v = X.T @ rp
    M = M + 1e-14 * np.trace(M) / len(M) * np.eye(len(M))
    return float(np.sqrt(max(v @ np.linalg.solve(M, v), 0.0)))


def run(k_extra, band, omega=0.75, decay=1.0):
    import config
    config.BGN_N_EXTRA_FACTORS = k_extra
    config.BGN_SIGMAJ_BAND = band
    config.BGN_EXTRA_OMEGA = omega
    config.BGN_EXTRA_DECAY = decay
    for m in ('utils_bgn.panel_functions_bgn', 'utils_bgn.sdf_compute_bgn', 'utils_bgn.vasicek'):
        sys.modules.pop(m, None)
    pf = importlib.import_module('utils_bgn.panel_functions_bgn')
    sc = importlib.import_module('utils_bgn.sdf_compute_bgn')
    np.random.seed(SEED)
    arr = pf.create_arrays(N, T)
    P, book = np.asarray(arr[9]), np.asarray(arr[11])
    loop = sc.sdf_compute(N, T, arr)
    rat, star = [], []
    for t in TS:
        _, _, rp, cv = loop(t, 0)
        rp, cv = np.asarray(rp), np.asarray(cv)
        mve = P[t + 1]; bk = book[t]
        keep = (bk > 0) & (mve > 0)
        rp_, cv_ = rp[keep], cv[np.ix_(keep, keep)]
        n = keep.sum()
        X = np.column_stack([np.ones(n), np.log(mve[keep]), bk[keep] / mve[keep]])
        s_star = float(np.sqrt(max(rp_ @ np.linalg.solve(
            cv_ + 1e-14 * np.trace(cv_) / n * np.eye(n), rp_), 0.0)))
        rat.append(sr_in_span(X, cv_, rp_) / s_star)
        star.append(s_star)
    return np.mean(rat), np.mean(star)


print(f"{'band':>8} | " + "  ".join(f"K={k:<3d}" for k in (0, 5, 20, 40)) + "   |  SR* (K=0 -> K=40)")
print("-" * 78)
for band in (0.03, 0.15, 0.30, 0.375):
    row, stars = [], []
    for k in (0, 5, 20, 40):
        r, s = run(k, band)
        row.append(r); stars.append(s)
    tag = f"{band:.3f}" + ("*" if abs(band - 0.03) < 1e-9 else "")
    print(f"{tag:>8} | " + "  ".join(f"{v:.4f}" for v in row)
          + f"   |  {stars[0]:.4f} -> {stars[-1]:.4f}")
print("\nSR_lin/SR* is the fraction of the maximum Sharpe a LINEAR characteristic model")
print("attains. Theory says it should fall like sqrt(L/K) with L=3 here.")
print("* = currently shipped sigmaj band.")
