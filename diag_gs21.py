#!/usr/bin/env python
"""
GS21 conditioning diagnostic. GS21 has NO zero-capital firms (capital floored at
alpha), so the KP14/BGN book>0 mask doesn't apply. Its ER[1:,1:] off-diagonal is a
rank-<=10 common-x' factor (n=10 GH nodes) plus an idiosyncratic diagonal, so high
condition numbers come from near-duplicate firm states / small idiosyncratic
variance, not singular degeneracy.

This measures, per month: (1) the condition number, (2) whether that conditioning
actually moves sdf_ret/max_sr (perturbation stability), (3) what drives the
smallest eigen-direction (default firms? duplicate states?), and (4) whether
excluding defaulted firms helps.

Run on Sol (bop env), repo root:
    BOP_SCRATCH_DIR=/scratch/sjpruitt/bop_gs21 \
    BOP_TEMP_DIR=/scratch/sjpruitt/bop_temp_gs21 \
    python diag_gs21.py gs21_0 560
(adjust dirs/panel/month to your GS21 outputs; trailing args = extra months)
"""
import sys, os, pickle, warnings
import numpy as np
import scipy.linalg

PANEL  = sys.argv[1] if len(sys.argv) > 1 else "gs21_0"
MONTHS = [int(x) for x in sys.argv[2:]]

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import config
config.init_from_env()
from utils.sparse_3d import load_sparse_3d, Sparse3D
from utils_gs21 import sdf_compute_gs21 as sdf_module

r = config.GS21_R
burnin = config.GS21_BURNIN
if not MONTHS:
    MONTHS = [burnin + 360]                     # first month calculate_moments computes

arr_dir = os.path.join(config.TEMP_DIR, f"{PANEL}_arr")
with open(os.path.join(arr_dir, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)
N, T = metadata["N"], metadata["T"]
sinfo = metadata.get("sparse_info", {})

arrs = []
for i in range(metadata["n_arrays"]):
    info = sinfo.get(i, {})
    if info.get("is_sparse", False):
        arrs.append(Sparse3D(load_sparse_3d(os.path.join(arr_dir, f"{i}_sparse"),
                                            info["n_slices"]), info["shape"]))
    else:
        arrs.append(np.load(os.path.join(arr_dir, f"{i}.npy"), mmap_mode="r"))
arr_tuple = tuple(arrs)
# arr order: k, b, x, z, cutoff, eta, P, P_ex, eret, ret, op_cashflow, default
b, z, cutoff, eta, P_ex, default = (arr_tuple[1], arr_tuple[3], arr_tuple[4],
                                    arr_tuple[5], arr_tuple[7], arr_tuple[11])
sdf_loop = sdf_module.sdf_compute(N, T + burnin, arr_tuple)
rng = np.random.default_rng(0)

print(f"panel={PANEL}  N={N}  r={r}")
for m in MONTHS:
    print("\n" + "=" * 70)
    print(f"MONTH {m}")
    print("=" * 70)

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        sdf_ret, max_sr, rp, cond_var = sdf_loop(m - 1, 0)
    rp = np.asarray(rp).ravel()
    cv = np.asarray(cond_var)
    one_plus_eret = rp + np.exp(r)

    # reconstruct ER exactly from (rp, cond_var)
    ER = np.zeros((N + 1, N + 1))
    ER[1:, 1:] = cv + np.outer(one_plus_eret, one_plus_eret)
    ER[0, 0] = np.exp(2 * r)
    ER[0, 1:] = one_plus_eret * np.exp(r)
    ER[1:, 0] = one_plus_eret * np.exp(r)
    sym = (ER + ER.T) / 2

    w = np.linalg.eigvalsh(sym)
    wmax = w.max()
    cond = wmax / w.min() if w.min() > 0 else np.inf
    n_tiny = int((w < 1e-8 * wmax).sum())
    n_small = int((w < 1e-6 * wmax).sum())
    print(f"  cond(ER) = {cond:.3e}   eig min/max = {w.min():.3e}/{wmax:.3e}   "
          f"#eig<1e-8*max={n_tiny}  #eig<1e-6*max={n_small}")
    print(f"  sdf_loop: sdf_ret={sdf_ret:.6f}  max_sr={max_sr:.6f}  "
          f"warnings={'; '.join(str(x.message) for x in wlist) or 'none'}")

    # (1) does the conditioning actually move the output? relative perturbation 1e-8
    def solve_port(M):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = scipy.linalg.solve(M, np.ones((N + 1, 1)), assume_a="pos").reshape(-1)
        p = p / p.sum()
        return p
    port = solve_port(ER)
    pert = ER * (1 + 1e-8 * rng.standard_normal(ER.shape))
    pert = (pert + pert.T) / 2
    port_p = solve_port(pert)
    sdf_p = -(port_p[1:] * rp).sum()      # using rp as a proxy excess-return vector
    sdf_0 = -(port[1:] * rp).sum()
    rel = abs(sdf_p - sdf_0) / (abs(sdf_0) + 1e-30)
    print(f"  stability: max|weight|={np.abs(port[1:]).max():.3e}   "
          f"rel change in (port.rp) under 1e-8 perturb = {rel:.2e}  "
          f"(garbage if ~1; fine if <<1)")

    # (2) what drives the smallest eigen-direction? top firms in min eigenvector
    _, V = np.linalg.eigh(sym)
    vmin = np.abs(V[1:, 0])               # risky-firm loadings of smallest eigenvector
    top = np.argsort(vmin)[::-1][:8]
    print(f"  smallest-eig direction concentrated on firms {top.tolist()}")
    print(f"    of those, #defaulted this month = {int(np.asarray(default[m])[top].sum())}")

    # (3) near-duplicate firm states (the documented mve/bm ties)
    state = np.column_stack([np.asarray(z[m]), np.asarray(b[m]), np.asarray(eta[m]),
                             np.asarray(cutoff[m]), np.asarray(P_ex[m])])
    uniq = np.unique(np.round(state, 10), axis=0).shape[0]
    print(f"  duplicate firm states (rounded): {N - uniq} firms share a state  "
          f"| defaulted this month: {int(np.asarray(default[m]).sum())}")

    # (4) does excluding defaulted firms reduce the condition number?
    deflt = np.asarray(default[m]).astype(bool)
    if deflt.any():
        keep = ~deflt
        idx = np.concatenate(([0], 1 + np.flatnonzero(keep)))
        sub = sym[np.ix_(idx, idx)]
        ws = np.linalg.eigvalsh(sub)
        cond_s = ws.max() / ws.min() if ws.min() > 0 else np.inf
        print(f"  cond after dropping {int(deflt.sum())} defaulted firms = {cond_s:.3e}  "
              f"(was {cond:.3e})")

print("\n[done]")
