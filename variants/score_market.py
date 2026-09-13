"""Score the equal-weighted market on the TRUE conditional moments over the evaluation months of
every current seeded run, beside the SR_max split into the part the market spans and the part it
does not. Evidence behind docs/RESULTS.md cross-cutting finding 8.

Why it exists. run_estimators.py appends the EW market to the random-feature factors UNPENALISED
(--include_mkt), while linrank/linlev carry the same market as a PENALISED constant column. At its
largest ridge penalty DKKM therefore collapses onto a scaled market, and this script is what showed
that the collapsed value IS the market's Sharpe -- so a DKKM "gap" can be the market beating
linear estimates that were made to shrink it.

Month logic mirrors run_estimators.py exactly:
    start, end = months.min(), months.max();  eval = [start+W .. end] present in the moments file
    n = rows of panel at that month;  mu = MU[i,:n];  Sigma = SIG[i,:n,:n]
On the conditional frontier SR_max^2 = SR_ew^2 + SR_orth^2, so SR_orth is the Sharpe of the
tangency portfolio's part orthogonal to the market.

Needs the saved moments (*_moments_*.npz, ~400 MB each), which live on the cluster checkout, so
run it there. 80 seeds took 6.5 min and 30 GiB on one Sol htc node (2026-09-13).
    python variants/score_market.py [out.csv]      # default: results/market_sr.csv
Then summarise anywhere with variants/market_decomposition.py.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("BOP_RESULTS_DIR") or os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "market_sr.csv")
W = 360
ECON = [("kp_vy", "vyx"), ("bgn_gam", "g0235"), ("bgn_gam", "g0235f"), ("bgn_gam", "g0235s"),
        ("bgn_gam", "g0235r"), ("gs_bx", "g28"), ("gs_bx", "gx7"), ("gs_bx", "bx7")]
DK = {"rff", "rff_ens", "rff_lev", "rff_lev_ens"}

rows = []
for model, tag in ECON:
    for seed in range(10):
        st = lambda kind: os.path.join(R, f"{model}_{kind}_{tag}_s{seed:03d}")
        mom = np.load(st("moments") + ".npz")
        months_m, MU, SIG = mom["months"], mom["mu"], mom["Sigma"]
        midx = {int(x): i for i, x in enumerate(months_m)}
        nper = pd.read_parquet(st("panel") + ".parquet", columns=["month", "firmid"]).groupby("month").size()
        start, end = int(months_m.min()), int(months_m.max())
        ev = [m for m in range(start + W, end + 1) if m in midx]
        ts = pd.read_csv(st("oracle") + "_ts.csv").set_index("month")
        sr_ew, sr_orth, sr_max, ew_mu = [], [], [], []
        for m in ev:
            i, n = midx[m], int(nper.loc[m])
            mu, S = MU[i, :n].astype(float), SIG[i, :n, :n].astype(float)
            w = np.full(n, 1.0 / n)
            s = (w @ mu) / np.sqrt(max(w @ S @ w, 1e-300))
            smax = float(ts.loc[m, "sr_max"]) if m in ts.index else np.nan
            sr_ew.append(s); sr_max.append(smax); ew_mu.append(w @ mu)
            sr_orth.append(np.sqrt(max(smax ** 2 - s ** 2, 0.0)))
        summ = pd.read_csv(st("estimators") + "_w360_summary.csv")
        dk = summ[summ.method.isin(DK)]
        top = dk[dk.kappa == dk.kappa.max()]
        rows.append(dict(
            model=model, tag=tag, seed=seed, eval_months=len(ev),
            sr_ew=float(np.mean(sr_ew)), sr_max_eval=float(np.nanmean(sr_max)),
            sr_orth=float(np.nanmean(sr_orth)), ew_share_of_srmax=float(np.mean(sr_ew) / np.nanmean(sr_max)),
            ew_mu_min=float(np.min(ew_mu)),
            dkkm_top_kappa=float(dk.kappa.max()), dkkm_at_top_min=float(top.sharpe.min()), dkkm_at_top_max=float(top.sharpe.max()),
            dkkm_best=float(dk.sharpe.max()), lin_best=float(summ[~summ.method.isin(DK)].sharpe.max()),
            linrank_k0=float(summ[(summ.method == "linrank") & (summ.kappa == 0)].sharpe.max()),
        ))
        r = rows[-1]
        print(f"{model}/{tag} s{seed}: EW {r['sr_ew']:.4f}  DKKM@topk [{r['dkkm_at_top_min']:.4f},{r['dkkm_at_top_max']:.4f}]  "
              f"DKKM best {r['dkkm_best']:.4f}  lin best {r['lin_best']:.4f}  SRmax {r['sr_max_eval']:.4f}  orth {r['sr_orth']:.4f}",
              flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("wrote", OUT)
