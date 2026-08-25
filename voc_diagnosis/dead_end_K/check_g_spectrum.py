"""
CHECK G: does raising K actually raise the RANK of the systematic covariance, and how much
does that depend on the sigmaj band? Sweeps (K_extra, band) and reports the eigenvalue
spectrum of cond_var -- the object that determines SR_lin/SR* ~ sqrt(L/K).
"""
import sys, os, importlib; sys.path.insert(0, os.getcwd())
import numpy as np

SEED, N, T, TS = 20260825, 300, 100, [70, 90, 98]


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
    corr_zj = np.asarray(arr[6])
    loop = sc.sdf_compute(N, T, arr)
    eigs, effr, sr, offsd = [], [], [], []
    for t in TS:
        _, m_, rp, cv = loop(t, 0)
        cv = np.asarray(cv)
        w = np.linalg.eigvalsh(cv)[::-1]
        w = np.maximum(w, 0)
        eigs.append(w[:6] / w.sum())
        effr.append(w.sum()**2 / (w**2).sum())
        sr.append(float(m_))
        off = cv[~np.eye(N, dtype=bool)]
        offsd.append(off.std())
    return (np.mean(eigs, axis=0), np.mean(effr), np.mean(sr), np.mean(offsd),
            (1 - corr_zj**2).mean())


print(f"{'K':>4} {'band':>7} {'1-corr_zj^2':>12} | {'eig1':>7} {'eig2':>7} {'eig3':>7} "
      f"{'eig4':>7} {'eig5':>7} | {'effrank':>8} {'sd(offdiag)':>12} {'max_sr':>8}")
print("-" * 110)
for band in (0.03, 0.15, 0.30, 0.375):
    for k in (0, 5, 20, 40):
        e, er, sr, osd, ncz = run(k, band)
        tag = f"{band:.3f}" + ("*" if abs(band - 0.03) < 1e-9 else "")
        print(f"{k:>4} {tag:>7} {ncz:12.4f} | {e[0]:7.4f} {e[1]:7.4f} {e[2]:7.4f} "
              f"{e[3]:7.4f} {e[4]:7.4f} | {er:8.2f} {osd:12.3e} {sr:8.4f}")
    print()
print("* = current shipped value (0.1 * 0.3). 0.30 = the un-scaled 0.3; 0.375 = 1.25x it.")
