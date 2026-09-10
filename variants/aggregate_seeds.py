"""Per-seed and per-economy tables from variants/results, every row traceable.

WORKING.md §41: the ten-seed means that are this project's headline numbers lived only in
_scratch and in prose, computed by a gitignored script that hardcoded one economy. This is
the tracked producer. It reads every SEEDED oracle JSON ({model}_oracle_{tag}_s{seed}.json)
with its estimator summary ({model}_estimators_{tag}_s{seed}_w{window}_summary.csv), writes
one row per seed, then aggregates per (model, tag, N, T, window). Rows carry the spec_id, the
consumed solve ids and both provenance tags, so every number in the economy table points
back to the files, the commit and the spec that produced it. Unseeded legacy files (no
_sNNN suffix; pre-2026-09-07 parameters) are never read.

Definitions (variants/common/oracle.py evaluate_bases; run_estimators.py):
  room_all  = best const-theta SR over the nonlinear bases (rff*, bins, poly2) minus
              lin_rank's const-theta SR, averaged over ALL months
  room_eval = the same restricted to the estimators' evaluation months (oracle --eval_window,
              2026-09-09+); NaN for oracles run before the flag existed
  sr_max_eval = mean of the oracle's per-month sr_max over those same evaluation months,
              read from the run's _ts.csv. Available for EVERY run, including oracles that
              predate --eval_window, because the per-month series was always saved. It is a
              hard per-month upper bound on any portfolio's conditional SR (Cauchy-Schwarz:
              w'mu / sqrt(w'Sigma w) <= sqrt(mu' Sigma^-1 mu)), so dkkm <= sr_max_eval must
              hold. Against the ALL-MONTH sr_max it does not: g28's DKKM is 0.2975 against
              0.2621 all-month and 0.3087 over the evaluation months. That apparent violation
              of an inequality is the month-sample confound (WORKING.md §41) made visible
  dkkm      = best sharpe over rff, rff_ens, rff_lev, rff_lev_ens (any P, any kappa)
  lin       = best sharpe over linrank, linlev, fm, ff
  gap       = dkkm - lin
  t         = max t_vs_fm over the rff methods
  gap/room  in the economy table is the RATIO OF MEANS (§40: not the mean of per-seed ratios)

usage:
  python variants/aggregate_seeds.py                 # every (model, tag, N, T, window)
  python variants/aggregate_seeds.py --flagship      # N=500, T=500 only
  python variants/aggregate_seeds.py --out DIR       # default: variants/results
writes {out}/seed_table.csv and {out}/economy_table.csv and prints the economy table.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RFF = ("rff", "rff_ens", "rff_lev", "rff_lev_ens")
LIN = ("linrank", "linlev", "fm", "ff")
SEEDED = re.compile(r"^(?P<model>[a-z_]+)_oracle_(?P<tag>.+)_s(?P<seed>\d{3})\.json$")


def _nonlin_keys(bases):
    return [k for k in bases if k.startswith("rff") or k in ("bins", "poly2")]


def seed_rows(results):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "*_oracle_*_s???.json"))):
        m = SEEDED.match(os.path.basename(f))
        if not m:
            continue
        model, tag, seed = m["model"], m["tag"], int(m["seed"])
        d = json.load(open(f))
        b = d["bases"]
        nl = _nonlin_keys(b)
        lin, lin_e = b["lin_rank"]["const_best"], b["lin_rank"].get("const_best_eval", np.nan)
        best_nl = max(b[k]["const_best"] for k in nl)
        best_nl_e = max(b[k].get("const_best_eval", np.nan) for k in nl) if nl else np.nan
        ts_path = os.path.join(results, f"{model}_oracle_{tag}_s{seed:03d}_ts.csv")
        ts_months = ts_sr = None
        if os.path.exists(ts_path):
            ts = pd.read_csv(ts_path)
            if {"month", "sr_max"} <= set(ts.columns):
                ts_months, ts_sr = ts["month"].to_numpy(), ts["sr_max"].to_numpy()
        base = dict(model=model, tag=tag, seed=seed, N=d.get("N"), T=d.get("T"), months=d.get("months"),
                    spec_id=d.get("spec_id"),
                    solves=",".join(s["solve_id"] or "UNREGISTERED" for s in (d.get("solves") or [])),
                    sr_max=d["sr_max_mean"], lin_ceiling_all=lin, nonlin_ceiling_all=best_nl,
                    room_all=best_nl - lin, lin_ceiling_eval=lin_e, nonlin_ceiling_eval=best_nl_e,
                    room_eval=best_nl_e - lin_e, eval_window=d.get("eval_window"),
                    eval_months_oracle=d.get("eval_months"), prov_oracle=d.get("prov"))

        def sr_max_eval(window):
            """Mean per-month sr_max over the months an estimator at `window` is scored on.
            Same rule as run_estimators.py: month >= first_month + window."""
            if ts_months is None or window is None or np.isnan(window):
                return np.nan
            mask = ts_months >= ts_months.min() + int(window)
            return float(ts_sr[mask].mean()) if mask.any() else np.nan
        ests = sorted(glob.glob(os.path.join(results, f"{model}_estimators_{tag}_s{seed:03d}_w*_summary.csv")))
        if not ests:
            rows.append(dict(base, window=np.nan, sr_max_eval=np.nan, dkkm=np.nan, lin=np.nan,
                             gap=np.nan, t=np.nan, dkkm_method=None, lin_method=None, prov_est=None))
            continue
        for e in ests:
            w = int(re.search(r"_w(\d+)_summary\.csv$", e).group(1))
            s = pd.read_csv(e)
            best = s.loc[s.groupby("method").sharpe.idxmax()].set_index("method")
            dk = best.loc[[k for k in RFF if k in best.index]].sharpe
            ln = best.loc[[k for k in LIN if k in best.index]].sharpe
            t = s.loc[s.method.str.startswith("rff"), "t_vs_fm"].max() if "t_vs_fm" in s else np.nan
            rows.append(dict(base, window=w, sr_max_eval=sr_max_eval(w),
                             dkkm=dk.max(), lin=ln.max(), gap=dk.max() - ln.max(), t=t,
                             dkkm_method=dk.idxmax(), lin_method=ln.idxmax(),
                             prov_est=s["prov"].iloc[0] if "prov" in s else None))
    return pd.DataFrame(rows)


def economy_table(seeds):
    keys = ["model", "tag", "N", "T", "window"]
    out = []
    for k, g in seeds.groupby(keys, dropna=False):
        rec = dict(zip(keys, k))
        rec["n_seeds"] = int(g.seed.nunique())
        rec["seeds"] = ",".join(str(s) for s in sorted(g.seed.unique()))
        specs, solves = sorted(set(g["spec_id"].dropna())), sorted(set(g.solves.dropna()))
        rec["spec_id"] = specs[0] if len(specs) == 1 else ("MIXED:" + "|".join(specs) if specs else None)
        rec["solves"] = solves[0] if len(solves) == 1 else ("MIXED:" + "|".join(solves) if solves else None)
        for col in ("sr_max", "sr_max_eval", "room_all", "room_eval", "dkkm", "lin", "gap", "t"):
            v = g[col].astype(float).dropna()
            rec[f"{col}_mean"] = v.mean() if len(v) else np.nan
            rec[f"{col}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            rec[f"{col}_se"] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            rec[f"{col}_n"] = int(len(v))
        rec["gap_over_room_all"] = rec["gap_mean"] / rec["room_all_mean"] if rec["room_all_mean"] else np.nan
        rec["gap_over_room_eval"] = rec["gap_mean"] / rec["room_eval_mean"] if rec["room_eval_mean"] else np.nan
        out.append(rec)
    return pd.DataFrame(out).sort_values(["model", "tag", "N", "T", "window"]).reset_index(drop=True)


def render(econ):
    lines = ["| economy | spec | n | SR_max all | SR_max eval | room all | room eval | DKKM | best lin | gap | t | gap/room all | gap/room eval |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def ms(r, c, sign=False):
        m, sd = r[f"{c}_mean"], r[f"{c}_sd"]
        if pd.isna(m):
            return "-"
        d = 1 if c == "t" else 4
        out = f"{m:+.{d}f}" if sign else f"{m:.{d}f}"
        return out + (f" ({sd:.{d}f})" if not pd.isna(sd) else "")

    def ratio(x):
        return "-" if pd.isna(x) else f"{x:.2f}"

    for _, r in econ.iterrows():
        w = "-" if pd.isna(r["window"]) else int(r["window"])
        econ_name = f"{r['model']}/{r['tag']} N={r['N']} T={r['T']} w={w}"
        spec = r["spec_id"] or "-"
        lines.append(f"| {econ_name} | {spec} | {r['n_seeds']} | {ms(r, 'sr_max')} | {ms(r, 'sr_max_eval')} | "
                     f"{ms(r, 'room_all', True)} | {ms(r, 'room_eval', True)} | {ms(r, 'dkkm')} | "
                     f"{ms(r, 'lin')} | {ms(r, 'gap', True)} | {ms(r, 't')} | "
                     f"{ratio(r['gap_over_room_all'])} | {ratio(r['gap_over_room_eval'])} |")
    lines.append("")
    lines.append("mean (sd across seeds). gap/room = ratio of means. room_eval is NaN for oracles "
                 "run before --eval_window (2026-09-09); SR_max eval is available for every run, "
                 "from the per-month series, and is the bound DKKM must respect.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=None, help="default: the results dir")
    ap.add_argument("--flagship", action="store_true", help="keep only N=500, T=500")
    a = ap.parse_args(argv)
    seeds = seed_rows(a.results)
    econ = economy_table(seeds)
    # The CSVs are always the COMPLETE set, so the tracked tables do not depend on which
    # flag the last person ran; --flagship narrows only what is printed.
    out = a.out or a.results
    seeds.to_csv(os.path.join(out, "seed_table.csv"), index=False)
    econ.to_csv(os.path.join(out, "economy_table.csv"), index=False)
    shown = econ[(econ["N"] == 500) & (econ["T"] == 500)] if a.flagship else econ   # NOT econ.T: transpose
    print(render(shown))
    print(f"\nwrote {os.path.join(out, 'seed_table.csv')} ({len(seeds)} rows) and economy_table.csv ({len(econ)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
