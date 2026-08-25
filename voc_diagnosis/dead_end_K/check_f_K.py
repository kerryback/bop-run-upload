"""
CHECK F (T5): specification tests for the extra-aggregate-factor construction.
These are not tests of the refactor, they are tests of the ECONOMICS. The first one is
the important one: because lambda'b_s is pinned, expected returns MUST be invariant in K.
If rp moves with K, the null-space construction is broken and nothing downstream matters.

Run:  python voc_diagnosis/refactor_checks/check_f_K.py [K_extra ...]
"""
import sys, os, importlib; sys.path.insert(0, os.getcwd())
import numpy as np

KS = [int(a) for a in sys.argv[1:]] or [0, 5, 20, 40]
SEED, N, T, TS = 20260825, 300, 100, [50, 70, 90, 98]


def run(k_extra):
    import config
    config.BGN_N_EXTRA_FACTORS = k_extra
    for m in ('utils_bgn.panel_functions_bgn', 'utils_bgn.sdf_compute_bgn'):
        if m in sys.modules:
            del sys.modules[m]
    pf = importlib.import_module('utils_bgn.panel_functions_bgn')
    sc = importlib.import_module('utils_bgn.sdf_compute_bgn')
    np.random.seed(SEED)
    arr = pf.create_arrays(N, T)
    loop = sc.sdf_compute(N, T, arr)
    out = {'beta': np.asarray(arr[5]), 'chi': np.asarray(arr[4]), 'P': np.asarray(arr[9])}
    for t in TS:
        s, m, rp, cv = loop(t, 0)
        out[f'rp{t}'] = np.asarray(rp); out[f'cv{t}'] = np.asarray(cv)
        out[f'sr{t}'] = float(m)
    return out


res = {k: run(k) for k in KS}
base = res[KS[0]]

print(f"\n{'K_extra':>8} {'beta':>10} {'chi':>10} {'P':>10} | {'max|rp-rp0|':>12} "
      f"{'diag ratio':>11} {'mean offdiag':>13} {'mean max_sr':>12}")
print("-" * 100)
for k in KS:
    r = res[k]
    bit = lambda nm: 'identical' if np.array_equal(r[nm], base[nm]) else 'DIFFERS'
    drp = max(np.abs(r[f'rp{t}'] - base[f'rp{t}']).max() for t in TS)
    dr = np.mean([np.diag(r[f'cv{t}']).mean() / np.diag(base[f'cv{t}']).mean() for t in TS])
    off = np.mean([(r[f'cv{t}'].sum() - np.trace(r[f'cv{t}'])) / (N * (N - 1)) for t in TS])
    sr = np.mean([r[f'sr{t}'] for t in TS])
    print(f"{k:>8} {bit('beta'):>10} {bit('chi'):>10} {bit('P'):>10} | {drp:12.3e} "
          f"{dr:11.4f} {off:13.3e} {sr:12.4f}")

print()
ok = True
for k in KS:
    d = max(np.abs(res[k][f'rp{t}'] - base[f'rp{t}']).max() for t in TS)
    if d > 1e-12:
        print(f"  FAIL: rp moved by {d:.2e} at K_extra={k} -- pinning is broken"); ok = False
for nm in ('beta', 'chi', 'P'):
    if not all(np.array_equal(res[k][nm], base[nm]) for k in KS):
        print(f"  FAIL: {nm} is not invariant in K -- the draws are not append-only"); ok = False
offs = [np.mean([(res[k][f'cv{t}'].sum() - np.trace(res[k][f'cv{t}'])) for t in TS]) for k in KS]
srs = [np.mean([res[k][f'sr{t}'] for t in TS]) for k in KS]
if len(KS) > 1:
    print(f"  off-diagonal covariance rises with K: {all(b >= a - 1e-15 for a, b in zip(offs, offs[1:]))}")
    print(f"  max_sr falls with K:                  {all(b <= a + 1e-15 for a, b in zip(srs, srs[1:]))}")
print("\n" + ("ALL SPECIFICATION TESTS PASS" if ok else "SPECIFICATION TESTS FAILED"))
