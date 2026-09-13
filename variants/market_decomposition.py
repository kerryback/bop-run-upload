"""Where the measured DKKM gap comes from: the two tables of docs/RESULTS.md cross-cutting finding 8.

Reads results/market_sr.csv (from variants/score_market.py, run on the cluster) and
results/seed_table.csv (from aggregate_seeds.py). Runs anywhere; writes nothing.

Exact per-seed identity, every Sharpe ratio scored on the TRUE moments over the 125 eval months:
    gap = (DKKM - EW market) + (EW market - best linear)
Fair-gap bound: at DKKM's largest penalty its portfolio is a scaled market. A linear benchmark
given the same unpenalised market collapses to the same portfolio at full shrinkage, so it trails
DKKM by at most  DKKM best - DKKM at the top penalty  (up to incomplete shrinkage at that penalty;
uninformative for vyx, whose top penalty does not reach the market).
    python variants/market_decomposition.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("BOP_RESULTS_DIR") or os.path.join(HERE, "results")

d = pd.read_csv(os.path.join(R, "market_sr.csv"))
seed = pd.read_csv(os.path.join(R, "seed_table.csv"))
seed = seed[(seed["N"] == 500) & (seed["T"] == 500) & seed["dkkm"].notna()][["model", "tag", "seed", "dkkm", "lin", "gap", "room_eval"]]
d = d.merge(seed, on=["model", "tag", "seed"], how="left")
assert np.allclose(d.dkkm, d.dkkm_best, atol=1e-9) and np.allclose(d.lin, d.lin_best, atol=1e-9), \
    "market_sr.csv and seed_table.csv were built from different estimator summaries"
assert (d.dkkm_best <= d.sr_max_eval + 1e-9).all(), "a DKKM Sharpe exceeds SR_max over the same months"

d["dkkm_over_ew"] = d.dkkm_best - d.sr_ew
d["ew_over_lin"] = d.sr_ew - d.lin_best
d["fair_gap_bound"] = d.dkkm_best - d.dkkm_at_top_max
d["topk_minus_ew"] = d.dkkm_at_top_max - d.sr_ew
d["orth_share"] = d.sr_orth / d.sr_max_eval

order = ["vyx", "g0235", "g0235f", "g0235s", "g0235r", "g28", "gx7", "bx7"]
g = d.groupby("tag").agg(
    SRmax=("sr_max_eval", "mean"), EW=("sr_ew", "mean"), orth=("sr_orth", "mean"),
    DKKM=("dkkm_best", "mean"), lin=("lin_best", "mean"), gap=("gap", "mean"),
    dkkm_over_ew=("dkkm_over_ew", "mean"), ew_over_lin=("ew_over_lin", "mean"),
    fair_gap_bound=("fair_gap_bound", "mean"), room_eval=("room_eval", "mean"),
).reindex(order)
g["EW/SRmax"] = g.EW / g.SRmax
g["gap_share_linear_below_market"] = g.ew_over_lin / g.gap
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("=== per economy, means over 10 seeds ===")
print(g.round(4).to_string())

d["collapses_to_market"] = d.topk_minus_ew.abs() <= 0.001
print("\n=== seeds whose top-penalty DKKM Sharpe is the EW market's to within 0.001 ===")
print(d.groupby("tag").collapses_to_market.sum().reindex(order).to_string())
print("\nnon-vyx seeds where it is not (top penalty not fully shrinking, or the market weight's sign flips in-window):")
print(d.loc[~d.collapses_to_market & d.tag.ne("vyx"), ["tag", "seed", "sr_ew", "dkkm_at_top_min", "dkkm_at_top_max", "ew_mu_min"]]
      .round(4).to_string(index=False))

print("\n=== does DKKM leave the market when the market spans less? ===")
nv = d[d.tag != "vyx"].copy()
for x in ["sr_orth", "orth_share"]:
    print(f"  spearman({x}, DKKM - EW): all 80 {spearmanr(d[x], d.dkkm_over_ew)[0]:+.2f}   "
          f"without vyx (70) {spearmanr(nv[x], nv.dkkm_over_ew)[0]:+.2f}")
nv["non_market_sharpe"] = pd.cut(nv.sr_orth, [0, 0.05, 0.10, 0.20, 0.30, 10.0])
print(nv.groupby("non_market_sharpe", observed=True).agg(
    seeds=("seed", "size"), dkkm_over_ew=("dkkm_over_ew", "mean"), ew_over_lin=("ew_over_lin", "mean"),
    gap=("gap", "mean"), fair_gap_bound=("fair_gap_bound", "mean")).round(4).to_string())
print("vyx:", d[d.tag == "vyx"][["sr_orth", "dkkm_over_ew", "ew_over_lin", "gap"]].mean().round(4).to_dict())
