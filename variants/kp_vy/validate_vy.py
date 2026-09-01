"""kp_vy validation: realized vs expected by type, max_sr by y-tercile, PSD."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.seterr(all="ignore")
from parameters_kp14 import burnin, type_bv, r, dt, ntypes
import panel_functions_kp14 as pf, sdf_compute_kp14 as sc
np.random.seed(3)
N, T = 120, burnin + 260
arr = pf.create_arrays(N, T)
loop = sc.sdf_compute(N, T, arr)
rets, erets, yreg, ftype = arr[15], arr[16], arr[-2], arr[-1]
rows = []
for t in range(burnin + 5, T - 2, 2):
    _, max_sr, mu, Sig, _ = loop(t - 1)
    rows.append((t, yreg[t], max_sr, np.nanstd(mu), np.linalg.eigvalsh(Sig).min()))
rows = np.array(rows)
q = np.quantile(rows[:, 1], [1/3, 2/3]); terc = np.digitize(rows[:, 1], q)
print(f"type_bv={list(type_bv)}")
for k in range(3):
    m = rows[terc == k]
    print(f"y-terc {k}: months {len(m):3.0f}  max_sr {m[:,2].mean():.4f}  cs-sd(mu) {m[:,3].mean():.5f}  min eig {m[:,4].min():.2e}")
rf = np.exp(r*dt) - 1
mo = slice(burnin + 5, T - 2)
for f in range(ntypes):
    cols = ftype == f
    re = rets[mo][:, cols].mean() - rf
    ee = erets[mo][:, cols].mean() - rf
    se = rets[mo][:, cols].mean(1).std() / np.sqrt(T - 7 - burnin)
    print(f"type {f} (bv={type_bv[f]}): realized {re:.5f}  expected {ee:.5f}  (se {se:.5f})")
print("VY VALIDATION DONE")
