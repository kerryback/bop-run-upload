"""gs_bx validation: Euler, premia by type and regime."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.seterr(all="ignore")
import gs_sim_bx as gs
np.random.seed(11)
N, T = 150, gs.burnin + 130
arr = gs.create_arrays(N, T)
loop = gs.sdf_compute(N, T, arr)
sreg, ftype = arr["sreg"], arr["ftype"]
eul, rows = [], []
for t in range(gs.burnin + 5, T - 2):
    m = gs.conditional_moments(arr, t)
    eul.append(np.abs(m["euler"] - 1).max())
    _, max_sr, mu, Sig, _ = loop(t - 1)
    rows.append((t, sreg[t], max_sr, np.nanstd(mu)))
rows = np.array(rows)
print(f"betas={gs._bx_list}  max |E[MR]-1|: {max(eul):.2e}")
for s in (0, 1):
    m = rows[rows[:, 1] == s]
    print(f"regime {s}: months {len(m):3.0f}  max_sr {m[:,2].mean():.4f}  cs-sd(mu) {m[:,3].mean():.5f}")
rf = np.exp(gs.r) - 1
mo = slice(gs.burnin + 5, T - 2)
for f in range(gs.ntypes):
    cols = ftype == f
    re = arr["rets"][mo][:, cols].mean() - rf
    print(f"type {f} (beta={gs._bx_list[f]}): mean realized excess {re:.5f}")
print("GSBX VALIDATION DONE")
