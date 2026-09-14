"""E1: the gap against linear methods given the equal-weighted market on the same terms as DKKM.

docs/RESULTS.md finding 8 showed the measured gap outside vyx is the market portfolio -- which DKKM
carries UNPENALISED -- against linear methods that shrink the same market or never see it. E1
re-scored the 80 saved panels with run_estimators.py --linear_only --fair_linear (SEED_STAGE=linear),
adding linrank_m and linlev_m (the market a separate unpenalised column), mkt_est (the market alone,
its weight estimated the same way) and ew (the always-long market). This script computes

    fair_gap = best DKKM (the committed summary)  -  best of ff, fm, linrank, linlev,
                                                   linrank_m, linlev_m, mkt_est (the E1 summary)

and grades each economy against the bound pre-registered in docs/RESULTS.md ("E1 in detail") at
commit 7f82711, BEFORE the re-scoring ran. `ew` is reported but excluded from fair_lin: an always-long
position assumes the premium's sign, which no estimator is given.

It refuses to proceed if the E1 re-score of the original four methods differs from the committed
summary, since then the two files are not scoring the same panel.

Economies launched after E1 (K4, X3, B4 in docs/RUNS.md) ran with --fair_linear, so their own
summary in variants/results already holds the fair methods beside DKKM. Those are read directly,
with nothing to cross-check, and carry source "own run"; E1's rows carry "E1 re-score".

    python variants/fair_gap.py          # writes results_e1/fair_gap_{seed,economy}_table.csv
"""
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMMITTED = os.path.join(HERE, "results")
E1 = os.environ.get("BOP_E1_DIR") or os.path.join(HERE, "results_e1")

RFF = ["rff", "rff_ens", "rff_lev", "rff_lev_ens"]
ORIG = ["linrank", "linlev", "fm", "ff"]
FAIR = ORIG + ["linrank_m", "linlev_m", "mkt_est"]
# Pre-registered 2026-09-13 at 7f82711, before E1 ran: finding 8's fair-gap bound plus 0.003.
BOUND = {"vyx": (">=", 0.08), "g0235": ("<=", 0.003), "g0235f": ("<=", 0.008), "g0235s": ("<=", 0.008),
         "g0235r": ("<=", 0.004), "g28": ("<=", 0.006), "gx7": ("<=", 0.004), "bx7": ("<=", 0.006)}
STEM = re.compile(r"^(?P<model>[a-z_]+)_estimators_(?P<tag>.+)_s(?P<seed>\d{3})_w(?P<w>\d+)_summary\.csv$")

def _row(m, c, e, drift, source):
    """DKKM from summary c, every linear method (the fair ones included) from summary e."""
    dk = c[c.method.isin(RFF)]
    g = e.groupby("method").sharpe.max()
    return dict(model=m["model"], tag=m["tag"], seed=int(m["seed"]), window=int(m["w"]), source=source,
                dkkm=dk.sharpe.max(), dkkm_method=dk.loc[dk.sharpe.idxmax(), "method"],
                lin=g[ORIG].max(), fair_lin=g[FAIR].max(), fair_method=g[FAIR].idxmax(),
                mkt_est=g["mkt_est"], ew=g["ew"], linrank_m=g["linrank_m"], linlev_m=g["linlev_m"],
                orig_drift=drift, prov_lin=e["prov"].iloc[0] if "prov" in e else None)


rows = []
e1_files = sorted(glob.glob(os.path.join(E1, "*_estimators_*_summary.csv")))
for f in e1_files:
    m = STEM.match(os.path.basename(f))
    if not m:
        continue
    c_path = os.path.join(COMMITTED, os.path.basename(f))
    if not os.path.exists(c_path):
        raise SystemExit(f"{os.path.basename(f)}: no committed summary to compare against")
    c, e = pd.read_csv(c_path), pd.read_csv(f)
    key = ["method", "P", "kappa"]
    both = c[c.method.isin(ORIG)].merge(e[e.method.isin(ORIG)], on=key, suffixes=("_c", "_e"))
    drift = float(np.abs(both.sharpe_c - both.sharpe_e).max())
    if len(both) != int(c.method.isin(ORIG).sum()) or drift > 1e-6:
        raise SystemExit(f"{os.path.basename(f)}: E1's original-method rows differ from the committed run "
                         f"(max |diff| {drift:.2e}); the two files are not scoring the same panel")
    rows.append(_row(m, c, e, drift, "E1 re-score"))

# A run made with --fair_linear scored the fair methods itself: one summary supplies DKKM and every
# linear method, so there is no second file to cross-check against.
e1_names = {os.path.basename(f) for f in e1_files}
for f in sorted(glob.glob(os.path.join(COMMITTED, "*_estimators_*_summary.csv"))):
    m = STEM.match(os.path.basename(f))
    if not m or os.path.basename(f) in e1_names:
        continue
    s = pd.read_csv(f)
    if set(FAIR) <= set(s.method):
        rows.append(_row(m, s, s, np.nan, "own run"))
d = pd.DataFrame(rows)
d["gap"] = d.dkkm - d.lin
d["fair_gap"] = d.dkkm - d.fair_lin
d["fair_gap_pct_fair_lin"] = np.where(d.fair_lin > 1e-6, 100 * d.fair_gap / d.fair_lin, np.nan)

out = []
for (model, tag, w), x in d.groupby(["model", "tag", "window"], sort=False):
    se = x.fair_gap.std() / np.sqrt(len(x))
    r = dict(model=model, tag=tag, window=w, source=";".join(sorted(set(x.source))), n=len(x),
             gap=x.gap.mean(), fair_gap=x.fair_gap.mean(),
             fair_gap_sd=x.fair_gap.std(), fair_gap_t=x.fair_gap.mean() / se if se > 0 else np.nan,
             seeds_fair_gap_positive=int((x.fair_gap > 0).sum()),
             dkkm=x.dkkm.mean(), lin=x.lin.mean(), fair_lin=x.fair_lin.mean(), mkt_est=x.mkt_est.mean(), ew=x.ew.mean(),
             linrank_m=x.linrank_m.mean(), linlev_m=x.linlev_m.mean(),
             fair_gap_over_fair_lin=100 * x.fair_gap.mean() / x.fair_lin.mean(),
             fair_winners=";".join(f"{k}:{v}" for k, v in x.fair_method.value_counts().items()))
    if tag in BOUND:
        op, b = BOUND[tag]
        r["preregistered"] = f"{op} {b:+.3f}"
        r["verdict"] = "PASS" if (r["fair_gap"] >= b if op == ">=" else r["fair_gap"] <= b) else "FAIL"
    out.append(r)
econ = pd.DataFrame(out)
econ.to_csv(os.path.join(E1, "fair_gap_economy_table.csv"), index=False)
d.to_csv(os.path.join(E1, "fair_gap_seed_table.csv"), index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["tag", "window", "source", "n", "gap", "fair_gap", "fair_gap_sd", "fair_gap_t", "seeds_fair_gap_positive", "dkkm",
        "lin", "fair_lin", "mkt_est", "ew", "fair_gap_over_fair_lin", "preregistered", "verdict", "fair_winners"]
print(econ[[c for c in cols if c in econ]].round(4).to_string(index=False))
n_e1 = int((d.source == "E1 re-score").sum())
print(f"\noriginal four methods, E1 vs committed: max |diff| {d.orig_drift.max():.2e} over {n_e1} seeds")
print(f"wrote {os.path.join(E1, 'fair_gap_economy_table.csv')} and fair_gap_seed_table.csv")
