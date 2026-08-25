"""
Golden-values snapshot / comparison for the sdf_compute_bgn refactor.

  python voc_diagnosis/refactor_checks/golden.py save   -> writes golden_bgn.npz
  python voc_diagnosis/refactor_checks/golden.py check   -> compares current code to it

Also verifies that create_arrays is reproducible under np.random.seed (needed for T2).
"""
import sys, os, time; sys.path.insert(0, os.getcwd())
import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_bgn.npz")
SEED, N, T = 20260825, 300, 100
TS = [50, 70, 90, 98]


def build():
    from utils_bgn import panel_functions_bgn as pf
    from utils_bgn import sdf_compute_bgn as sc
    np.random.seed(SEED)
    arr = pf.create_arrays(N, T)
    sdf_loop = sc.sdf_compute(N, T, arr)
    out = {}
    for t in TS:
        sdf_ret, max_sr, rp, cond_var = sdf_loop(t, 0)
        out[f"sdf_ret_{t}"] = np.array(sdf_ret)
        out[f"max_sr_{t}"] = np.array(max_sr)
        out[f"rp_{t}"] = np.asarray(rp)
        out[f"cond_var_{t}"] = np.asarray(cond_var)
    # a few panel arrays, for the T2 bit-identity test
    r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, ocf = arr
    for nm, a in [("beta", beta), ("sigmaj", sigmaj), ("corr_zj", corr_zj),
                  ("eret", eret), ("ret", ret), ("P", P), ("book", book), ("r", r)]:
        out[f"panel_{nm}"] = np.asarray(a)
    out["panel_chi"] = np.asarray(chi if not hasattr(chi, "get_row_slice") else chi)
    return out


if MODE == "save":
    d = build()
    np.savez_compressed(OUT, **d)
    print(f"saved {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print("  reproducibility re-run...")
    d2 = build()
    bad = [k for k in d if not np.array_equal(d[k], d2[k])]
    print(f"  create_arrays reproducible under np.random.seed: "
          f"{'YES' if not bad else 'NO -> ' + str(bad)}")
else:
    g = np.load(OUT)
    d = build()
    print(f"{'key':22s} {'max abs':>12s} {'max rel':>12s}  {'scale':>12s}")
    worst = 0.0
    for k in sorted(g.files):
        a, b = g[k], d[k]
        if a.shape != b.shape:
            print(f"{k:22s}  SHAPE MISMATCH {a.shape} vs {b.shape}"); continue
        if k.startswith("panel_"):
            print(f"{k:22s}  {'bit-identical' if np.array_equal(a, b) else 'DIFFERS':>26s}")
            continue
        ad = np.abs(a - b)
        sc_ = max(np.abs(a).max(), 1e-300)
        rel = ad.max() / sc_
        worst = max(worst, rel)
        print(f"{k:22s} {ad.max():12.3e} {rel:12.3e}  {sc_:12.3e}")
    print(f"\nworst relative difference over moment objects: {worst:.3e}")
    print("PASS" if worst < 1e-9 else "FAIL (>1e-9)")
