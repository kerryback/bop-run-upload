"""gs_vy validation, in the shape verify_solves.py parses: E[M R] = 1 on the typed economy, realised vs
expected excess returns by type, max_sr by y-tercile, PSD."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.seterr(all="ignore")
import gs_sim_vy as gs
np.random.seed(3)
N, T = 150, gs.burnin + 200
arr = gs.create_arrays(N, T)
loop = gs.sdf_compute(N, T, arr)
rows = []; worst = 0.0
for t in range(gs.burnin + 5, T - 2, 2):
    m = gs.conditional_moments(arr, t)
    worst = max(worst, float(np.abs(m["euler"] - 1).max()))
    _, max_sr, mu, Sig, _ = loop(t - 1)
    rows.append((t, gs.ygrid[arr["iy"][t]], max_sr, np.nanmean(mu), np.nanstd(mu), np.linalg.eigvalsh(Sig).min()))
rows = np.array(rows)
print(f"max |E[MR]-1|: {worst:.2e}")
q = np.quantile(rows[:, 1], [1 / 3, 2 / 3]); terc = np.digitize(rows[:, 1], q)
print(f"types gs_bv={gs._bv_list} gs_bc={gs._bc_list} gs_fc={gs.gs_fc} gs_bphi={gs.gs_bphi} gs_bi={gs._bi_list} gs_bk={gs._bk_list} gs_bzl={gs.gs_bzl} gs_bzi={gs.gs_bzi} gs_bzb={gs.gs_bzb} gs_bx2={gs.gs_bx2} gs_pj={gs.gs_pj} shares={gs._share} ashift={np.round(gs.ashift_ts, 5).tolist()} NY={gs.NY} NY2={gs.NY2} gs_bv2={gs._bv2_list}")
for k in range(3):
    m = rows[terc == k]
    if len(m) == 0:        # a coarse chain (few distinct y values) can leave a quantile tercile empty
        print(f"y-terc {k}: months   0  (empty: the chain has few distinct y values)"); continue
    print(f"y-terc {k}: months {len(m):3.0f}  max_sr {m[:,2].mean():.4f}  E[mu] {m[:,3].mean():.5f}  cs-sd(mu) {m[:,4].mean():.5f}  min eig {m[:,5].min():.2e}")
mo = slice(gs.burnin + 5, T - 2)
rf = np.exp(gs.r) - 1
rets = arr["rets"][mo]
erets = np.stack([gs.conditional_moments(arr, t, want_matrix=False)["eret"] for t in range(gs.burnin + 5, T - 2)])
for f in range(gs.ntypes):
    cols = arr["ftype"] == f
    re = rets[:, cols].mean() - rf; ee = erets[:, cols].mean() - rf
    se = rets[:, cols].mean(1).std() / np.sqrt(rets.shape[0])
    print(f"type {f} (bv={gs._bv_list[f]} bc={gs._bc_list[f]}): realized {re:.5f}  expected {ee:.5f}  (se {se:.5f})  premium {12*ee*100:.2f}%/yr  "
          f"mean log size {np.log(arr['k'][mo][:, cols] * arr['Vcont'][mo][:, cols]).mean():.3f}  default rate {arr['default'][mo][:, cols].mean():.4f}")
print("GSVY VALIDATION DONE")
