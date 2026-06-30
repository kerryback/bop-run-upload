#!/usr/bin/env python
"""
Verify the zero-capital fix: the PATCHED sdf_compute_kp14 should now return sane,
finite sdf_ret / max_sr (and emit no ill-conditioning warning) for months that
previously produced garbage (max_sr ~1e57).

Run on Sol (bop env), repo root, AFTER pulling the fix:
    BOP_SCRATCH_DIR=/scratch/sjpruitt/bop_kp14 \
    BOP_TEMP_DIR=/scratch/sjpruitt/bop_temp_kp14 \
    python verify_fix.py kp14_0 560 561 575

Expected for kp14_0 month 560: sdf_ret ~ -0.0859, max_sr ~ 9.07, warnings=none.
"""
import sys, os, pickle, warnings
import numpy as np

PANEL  = sys.argv[1] if len(sys.argv) > 1 else "kp14_0"
MONTHS = [int(x) for x in sys.argv[2:]] or [560]

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import config
config.init_from_env()
from utils.sparse_3d import load_sparse_3d, Sparse3D
from utils_kp14 import sdf_compute_kp14 as sdf_module

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

sdf_loop = sdf_module.sdf_compute(N, T + config.KP14_BURNIN, tuple(arrs))

print(f"panel={PANEL}  N={N}")
all_ok = True
for m in MONTHS:
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        sdf_ret, max_sr, rp, cond_var = sdf_loop(m - 1, 0)
    finite = bool(np.isfinite(sdf_ret) and np.isfinite(max_sr))
    sane = finite and abs(max_sr) < 100            # garbage case was ~1e57
    all_ok = all_ok and sane and not wlist
    warns = "; ".join(str(w.message) for w in wlist) or "none"
    flag = "OK" if (sane and not wlist) else "**SUSPECT**"
    print(f"  month {m}: sdf_ret={sdf_ret: .6f}  max_sr={max_sr: .6f}  "
          f"finite={finite}  warnings={warns}  [{flag}]")

print("\n[done] " + ("ALL SANE" if all_ok else "-- CHECK FLAGGED MONTHS"))
