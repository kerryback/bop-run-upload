"""Stressed-calibration validation: realized vs expected returns and max_sr by regime."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.seterr(all="ignore")
import panel_functions as pf, sdf_compute as sc
from parameters import burnin, gmult
np.random.seed(3)
N, T = 100, burnin + 260
arr = pf.create_arrays(N, T)
loop = sc.sdf_compute(N, T, arr)
r, eret, ret, sreg = arr[0], arr[7], arr[8], arr[-1]
rows = []
for t in range(burnin + 5, T - 2):
    _, max_sr, mu, Sig, _ = loop(t - 1)
    rows.append((t, sreg[t], max_sr, np.nanmean(mu), np.nanstd(mu), np.linalg.eigvalsh(Sig).min()))
rows = np.array(rows)
print(f"gmult={list(gmult)}")
for g in (0, 1):
    m = rows[rows[:, 1] == g]
    print(f"regime {g}: months {len(m):3.0f}  max_sr {m[:,2].mean():.4f}  E[mu] {m[:,3].mean():.5f}  "
          f"cs-sd(mu) {m[:,4].mean():.5f}  min eig {m[:,5].min():.2e}")
mo = slice(burnin + 5, T - 2)
re = (ret[mo] - (np.exp(r[mo]) - 1)[:, None]).mean(1)
ee = (eret[mo] - (np.exp(r[mo]) - 1)[:, None]).mean(1)
sg = sreg[mo]
for g in (0, 1):
    n = (sg == g).sum()
    print(f"regime {g}: realized mean excess {re[sg==g].mean():.5f}  expected {ee[sg==g].mean():.5f}  "
          f"(n={n}, se {re[sg==g].std()/np.sqrt(n):.5f})")
print("BGN GAM VALIDATION DONE")
