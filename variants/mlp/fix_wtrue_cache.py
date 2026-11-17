"""Replace w_true in the section-3 weight parquets by the exact conditional mean-variance weight on the
KEPT firms, solve(Sigma_keep, mu_keep), which is the oracle the estimators are scored against (sr_max).
The oracle npz stores the model's own 500-firm SDF weight restricted to the kept firms; identical in BGN
(nothing dropped), nearly so in GS21, but in KP14 (~50 firms dropped) the restriction halves the conditional
Sharpe. The old column is kept as w_true_oracle. On the same pass, cache what the implied-premium figure
needs (mu, chars, Sigma @ w for each method) so it can be redrawn without reloading Sigma.
usage: python fix_wtrue_cache.py <model>      (bgn_vy | gs_vy | kp_vy)
"""
import sys, glob, shutil, time
import numpy as np, pandas as pd
HERE = "/kellogg/proj/axa3203/bop_runs"; DATA = f"{HERE}/mlp/data"; RES = f"{HERE}/results"
CACHE = f"{HERE}/mlp/sec3/cache"; BAK = f"{DATA}/bak_wtrue_20261108"
TAGS = {"bgn_vy": ["bvy1ch", "bvychs35", "bvyhx4", "bvyuq2"], "gs_vy": ["gvylqs10", "gvyslzg6", "gvylqun10", "gvylf10", "gvylfs6i", "gvylqup5"], "kp_vy": ["vy1", "vygelbcs", "vyhx3", "vyg25s", "vyuq2"]}
model = sys.argv[1]
for tag in TAGS[model]:
    rows = []
    for f in sorted(glob.glob(f"{DATA}/wts_{model}_{tag}_s[0-9][0-9][0-9].parquet")):
        t0 = time.time(); seed = int(f[-11:-8]); w = pd.read_parquet(f)
        m = np.load(f"{RES}/{model}_moments_{tag}_s{seed:03d}.npz"); midx = {int(mo): i for i, mo in enumerate(m["months"])}
        SIG = m["Sigma"]; MU = m["mu"]; KEEP = m["keep"]
        if "w_true_oracle" not in w: w["w_true_oracle"] = w["w_true"]
        new = np.empty(len(w)); sr_old = []; sr_new = []
        for mo, idx in w.groupby("month").indices.items():
            i = midx[int(mo)]; n = len(idx); g = w.iloc[idx]
            assert np.array_equal(KEEP[i, :n], g.firmid.to_numpy()), (tag, seed, mo)
            S = SIG[i, :n, :n].astype(float); mu = MU[i, :n].astype(float)
            wm = np.linalg.solve(S, mu); new[idx] = wm
            wo = g.w_true_oracle.to_numpy(); sr_old.append(wo @ mu / np.sqrt(wo @ S @ wo)); sr_new.append(np.sqrt(mu @ wm))
            d = {"seed": seed, "month": int(mo), "firmid": g.firmid.to_numpy(), "mu": 100 * mu}
            for c in ["size", "lev"]:
                if c in g: d[c] = g[c].to_numpy()
            for mk in ["fm", "ff", "dkkm"]: d[f"sw_{mk}"] = 100 * (S @ g[f"w_{mk}"].to_numpy())
            rows.append(pd.DataFrame(d))
        w["w_true"] = new
        if not glob.glob(f"{BAK}/{f.split('/')[-1]}"): shutil.copy(f, BAK)
        w.to_parquet(f); del SIG, m
        print(f"{tag} s{seed} n={len(w)//len(midx)}  cond SR oracle-restricted {np.mean(sr_old):.3f} -> exact kept-set {np.mean(sr_new):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(f"{CACHE}/prem_{model}_{tag}.parquet"); print(f"cached prem_{model}_{tag}", flush=True)
