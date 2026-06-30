#!/usr/bin/env python
"""
Diagnose the 'Singular matrix' crash in utils/calculate_moments.py for KP14.

Runs ON the remote node against the memory-mapped arr_tuple (does NOT load the
big EtA file into RAM). Reconstructs the exact `second_moment` for the first
computed month and tests whether zero-capital (book=0 / bm=0) firms are the
cause of the rank deficiency.

Usage (on Sol, in the `bop` conda env, from the repo root):
    BOP_SCRATCH_DIR=/scratch/sjpruitt/bop_kp14 \
    BOP_TEMP_DIR=/scratch/sjpruitt/bop_temp_kp14 \
    python diag_singular.py kp14_0 560

    # extra trailing args = more months to check, e.g. ... kp14_0 560 561 575
Output is a few hundred bytes of text — safe to copy back over slow wifi.
"""
import sys, os, pickle, warnings
import numpy as np
import scipy.linalg

PANEL  = sys.argv[1] if len(sys.argv) > 1 else "kp14_0"
MONTHS = [int(x) for x in sys.argv[2:]] or [560]   # 560 = burnin(200)+360 = first computed month

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import config
config.init_from_env()                  # reads BOP_SCRATCH_DIR / BOP_TEMP_DIR
from utils.sparse_3d import load_sparse_3d, Sparse3D
from utils_kp14 import sdf_compute_kp14 as sdf_module

arr_dir = os.path.join(config.TEMP_DIR, f"{PANEL}_arr")
print(f"[load] {arr_dir}")
with open(os.path.join(arr_dir, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)
N, T = metadata["N"], metadata["T"]
sparse_info = metadata.get("sparse_info", {})

arr_list = []
for i in range(metadata["n_arrays"]):
    info = sparse_info.get(i, {})
    if info.get("is_sparse", False):
        sl = load_sparse_3d(os.path.join(arr_dir, f"{i}_sparse"), info["n_slices"])
        arr_list.append(Sparse3D(sl, info["shape"]))
    else:
        arr_list.append(np.load(os.path.join(arr_dir, f"{i}.npy"), mmap_mode="r"))
arr_tuple = tuple(arr_list)
burnin = config.KP14_BURNIN
K, price, eret = arr_tuple[0], arr_tuple[12], arr_tuple[14]
print(f"[load] N={N} T={T} burnin={burnin}  (arr_tuple has {len(arr_tuple)} arrays)")

sdf_loop = sdf_module.sdf_compute(N, T + burnin, arr_tuple)


def capital_per_firm(month):
    """Per-firm capital at `month` = sum over project vintages of K[:month+1, month, :]."""
    if hasattr(K, "get_col_slice_dense"):
        Kcol = K.get_col_slice_dense(month, month + 1)   # (month+1, N), same slice sdf_loop uses
    else:
        Kcol = np.asarray(K[:month + 1, month, :])
    return np.asarray(Kcol).sum(axis=0)


for month in MONTHS:
    print("\n" + "=" * 70)
    print(f"MONTH {month}   panel={PANEL}")
    print("=" * 70)

    cap = capital_per_firm(month)
    degen = np.where(cap == 0)[0]
    prow = np.asarray(price[month])
    print(f"  zero-capital firms (book=0 -> bm=0): {len(degen)} / {N}   e.g. {degen[:12].tolist()}")
    print(f"  price[month] min/max = {prow.min():.4g} / {prow.max():.4g}   (#price<=0: {(prow <= 0).sum()})")

    # Reconstruct the exact failing computation (calculate_moments.py:79-84).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdf_ret, max_sr, rp, cond_var = sdf_loop(month - 1, 0)
    rp = np.asarray(rp).reshape(-1, 1)
    second = np.asarray(cond_var) + rp @ rp.T
    sym = (second + second.T) / 2

    print(f"  second_moment: shape={second.shape}  finite={np.isfinite(second).all()}  "
          f"#nan={int(np.isnan(second).sum())}  #inf={int(np.isinf(second).sum())}")
    zero_rows = np.where(np.abs(second).sum(axis=1) == 0)[0]
    print(f"  exactly-zero rows: {len(zero_rows)}")

    if np.isfinite(sym).all():
        w = np.linalg.eigvalsh(sym)
        wmax = np.abs(w).max()
        tol = 1e-8 * wmax
        rank = int((np.abs(w) > tol).sum())
        print(f"  eig min/max = {w.min():.4e} / {w.max():.4e}   rcond ~ {w.min()/wmax:.2e}")
        print(f"  numerical rank = {rank} / {N}   (deficiency = {N - rank}; #zero-capital = {len(degen)})")

    # Does it actually fail to invert?  And does dropping degenerate firms fix it?
    try:
        np.linalg.inv(second)
        print("  np.linalg.inv(full): OK")
    except np.linalg.LinAlgError as e:
        print(f"  np.linalg.inv(full): FAILED -> {e}")

    if len(degen):
        keep = np.setdiff1d(np.arange(N), degen)
        sub = second[np.ix_(keep, keep)]
        try:
            np.linalg.inv(sub)
            print(f"  np.linalg.inv(drop {len(degen)} zero-capital firms -> {len(keep)}x{len(keep)}): OK  "
                  f"<-- confirms degenerate firms are the cause")
        except np.linalg.LinAlgError as e:
            wsub = np.linalg.eigvalsh((sub + sub.T) / 2)
            print(f"  np.linalg.inv(drop zero-capital): STILL FAILS -> {e}  "
                  f"(submatrix eig min/max = {wsub.min():.2e}/{wsub.max():.2e})")

    # ------------------------------------------------------------------
    # VALIDATION: does excluding zero-capital firms change sdf_ret / max_sr?
    # Reconstructs ER exactly from (rp, cond_var) -- no heavy recompute -- then
    # re-solves the SDF portfolio (a) over ALL firms (current pipeline behavior)
    # and (b) over book>0 firms only (proposed fix), faithfully mirroring the
    # try/except in sdf_compute_kp14.py:265-274.
    # ------------------------------------------------------------------
    r, dt = config.KP14_R, config.KP14_DT
    erdt = np.exp(r * dt)
    rp_vec = np.asarray(rp).ravel()
    cv = np.asarray(cond_var)
    one_plus_eret = rp_vec + erdt
    ER = np.zeros((N + 1, N + 1))
    ER[1:, 1:] = cv + np.outer(one_plus_eret, one_plus_eret)
    ER[0, 0] = np.exp(2 * r * dt)
    ER[0, 1:] = one_plus_eret * erdt
    ER[1:, 0] = one_plus_eret * erdt

    ret_row = np.asarray(arr_tuple[13][month])     # realized returns this month (sdf_ret uses these)
    gross_xs_real = 1 + ret_row - erdt
    keep_mask = cap > 0                            # book>0 == has live projects

    def solve_sdf(idx):
        """Mirror sdf_compute's port solve on the submatrix ER[idx, idx]; scatter back to N+1."""
        sub_er = ER[np.ix_(idx, idx)].copy()
        ones = np.ones((len(idx), 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                p_sub = scipy.linalg.solve(sub_er, ones, assume_a="pos").reshape(-1)
                path = "pos"                       # Cholesky 'succeeded' (silently, even if singular)
            except Exception:
                sub_er = sub_er + np.eye(len(idx)) * 1e-6
                try:
                    p_sub = scipy.linalg.solve(sub_er, ones).reshape(-1)
                    path = "ridge1e-6"
                except Exception:
                    p_sub = np.full(len(idx), np.nan); path = "nan"
        port = np.zeros(N + 1)
        port[idx] = p_sub
        s = port.sum()
        port = port / s if s != 0 else port
        sdf_r = -(port[1:] * gross_xs_real).sum()
        denom = port[1:] @ (cv @ port[1:])
        sr = -(port[1:] * rp_vec).sum() / np.sqrt(denom) if denom > 0 else np.nan
        return port, sdf_r, sr, path

    port_f, sdf_f, sr_f, path_f = solve_sdf(np.arange(N + 1))
    port_k, sdf_k, sr_k, path_k = solve_sdf(np.concatenate(([0], 1 + np.flatnonzero(keep_mask))))
    wf = np.abs(port_f[1:])
    deg_w = wf[degen].sum() / wf.sum() if wf.sum() > 0 else 0.0

    print(f"  --- SDF-weight validation: current FULL solve vs book>0 KEEP solve ---")
    print(f"  reconstruction check |sdf_ret_full - sdf_ret from sdf_loop| = {abs(sdf_f - sdf_ret):.3e}  (should be ~0)")
    print(f"  solve path:   full={path_f!r}   keep={path_k!r}   "
          f"(full='pos' on a singular matrix == silent garbage)")
    print(f"  gross weight on the {len(degen)} zero-capital firms (full solve): {deg_w:.1%}")
    print(f"  max |port weight|:   full={wf.max():.3e}   keep={np.abs(port_k[1:]).max():.3e}")
    print(f"  sdf_ret:   full={sdf_f:.6f}   keep={sdf_k:.6f}   |diff|={abs(sdf_f - sdf_k):.3e}")
    print(f"  max_sr :   full={sr_f:.6f}   keep={sr_k:.6f}   |diff|={abs(sr_f - sr_k):.3e}")

print("\n[done]")
