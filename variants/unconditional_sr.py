"""Exact unconditional SR of each estimated portfolio from the TRUE conditional moments:
   SR_unc = E[m_t] / sqrt(E[s_t^2 + m_t^2] - E[m_t]^2),  m_t = w_t'mu_t, s_t^2 = w_t'Sigma_t w_t
(only the weights carry estimation noise), next to the mean conditional SR and the realised SR."""
import glob, os, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
rows = []
for f in sorted(glob.glob("results/*_estimators_*_w360.csv")):
    tag = os.path.basename(f).replace("_estimators_", " ").replace("_w360.csv", "")
    d = pd.read_csv(f)
    d = d[(d.method != "rff") | (d.P == 3600)]
    d = d[d.method != "rff_ens"] if "rff_ens" not in d.method.values else d[(d.method != "rff")]
    for (m, P, k), g in d.groupby(["method", "P", "kappa"]):
        g = g.groupby("month").agg(mn=("mn", "mean"), stdev=("stdev", "mean"), sharpe=("sharpe", "mean"), xret=("xret", "mean"))
        unc = g.mn.mean() / np.sqrt((g.stdev ** 2 + g.mn ** 2).mean() - g.mn.mean() ** 2)
        rows.append({"run": tag, "method": m, "P": P, "kappa": k, "cond_mean": g.sharpe.mean(), "uncond_exact": unc,
                     "realised": g.xret.mean() / g.xret.std(), "n": len(g)})
r = pd.DataFrame(rows)
best = r.loc[r.groupby(["run", "method", "P"]).uncond_exact.idxmax()]          # best kappa by the exact unconditional SR
best2 = r.loc[r.groupby(["run", "method", "P"]).cond_mean.idxmax()]
out = best.merge(best2[["run", "method", "P", "cond_mean", "kappa"]], on=["run", "method", "P"], suffixes=("", "_bestcond"))
print(out[["run", "method", "P", "kappa", "cond_mean_bestcond", "uncond_exact", "realised", "n"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
