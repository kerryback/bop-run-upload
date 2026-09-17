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
  fair_lin  = best over those four PLUS linrank_m, linlev_m and mkt_est -- the market given
              to the linear methods on DKKM's terms (finding 8). `ew` is reported beside it
              and excluded from it: always-long assumes the premium's sign.
  fair_gap  = dkkm - fair_lin. THE complexity gap outside KP14, and the column RESULTS.md
              ranks by. Folded in here 2026-09-17; it used to come from variants/fair_gap.py
              and a second table in variants/results_e1, joined on (model, tag).
  gap       = dkkm - lin
  fm        = Fama-MacBeth's sharpe (the method has one row: no penalty grid)
  gap_fm    = dkkm - fm, the gap against Fama-MacBeth alone. RESULTS.md reports gap and room as
              percentages of fm (2026-09-15): gap_fm_over_fm = 100 * gap_fm_mean / fm_mean and
              room_*_over_fm = 100 * room_*_mean / fm_mean, ratios of means like every other ratio here.
              fm's Sharpe is near zero in some BGN regime economies (g0235r 0.0003), where these explode.
  t         = max t_vs_fm over the rff methods
  gap/room  in the economy table is the RATIO OF MEANS (§40: not the mean of per-seed ratios)
  *_pct_lin = room and gap as a percentage of the LINEAR SHARPE ACTUALLY ATTAINED (`lin`),
              which says whether a gap is a large or a small deal relative to what the
              classical methods deliver in that economy: vyx's +0.1046 is 16% of a linear
              0.6428, while g0235's smaller +0.0227 is 27% of a linear 0.0855. The per-seed
              ratio is in seed_table.csv; the economy table carries BOTH the mean and sd of
              those per-seed ratios (`*_pct_lin_mean/_sd`) and the RATIO OF MEANS
              (`room_all_over_lin`, `gap_over_lin`). They differ where `lin` is itself
              dispersed across seeds -- g0235's lin has sd 0.0359 on a mean of 0.0855, and
              its gap reads 31% as a mean of ratios against 27% as a ratio of means, because
              the low-denominator seeds dominate the first. The printed table and RESULTS.md
              quote the RATIO OF MEANS, for the same reason §40 gives for gap/room.

ON PROTOCOL. Reading the canonical results directory, every row must be at the measurement
protocol's N, T and window (variants/common/protocol.py), and this refuses to write a table
that mixes samples. It did mix them: scaling and smoke probes at N=60 to 500, T=80 to 200 and
windows 20/36/50 sat in economy_table.csv beside the real economies, and the one economy at
T=860 sat beside the twelve at T=500, so `--flagship` existed to filter the table down to the
rows that could be compared. The filter is gone because there is nothing to filter: an
off-protocol run must write to its own BOP_RESULTS_DIR (run_seeds_slurm.sh refuses otherwise),
and aggregating such a directory warns instead of refusing.

usage:
  python variants/aggregate_seeds.py                 # the canonical directory; every row on protocol
  python variants/aggregate_seeds.py --results DIR   # e.g. an off-protocol probe directory
  python variants/aggregate_seeds.py --out DIR       # default: the results dir
writes {out}/seed_table.csv and {out}/economy_table.csv and prints the economy table.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import protocol

HERE = os.path.dirname(os.path.abspath(__file__))
RFF = ("rff", "rff_ens", "rff_lev", "rff_lev_ens")
LIN = ("linrank", "linlev", "fm", "ff")
# The FAIR benchmark: the four linear methods plus the two ridge methods given the
# equal-weighted market as a separate unpenalised column, as --include_mkt gives it to DKKM,
# plus the market alone with its weight estimated. `ew`, the always-long market, is reported
# but kept OUT of the benchmark: an always-long position assumes the premium's sign, which no
# estimator is given. docs/RESULTS.md finding 8 is why this exists -- outside KP14 the measured
# gap is the market against methods not given it on the same terms.
FAIR = LIN + ("linrank_m", "linlev_m", "mkt_est")
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
                             gap=np.nan, t=np.nan, room_all_pct_lin=np.nan, room_eval_pct_lin=np.nan,
                             gap_pct_lin=np.nan, fm=np.nan, gap_fm=np.nan, room_all_pct_fm=np.nan,
                             room_eval_pct_fm=np.nan, gap_fm_pct_fm=np.nan,
                             fair_lin=np.nan, fair_gap=np.nan, ew=np.nan, fair_method=None,
                             dkkm_method=None, lin_method=None, prov_est=None))
            continue
        for e in ests:
            w = int(re.search(r"_w(\d+)_summary\.csv$", e).group(1))
            s = pd.read_csv(e)
            best = s.loc[s.groupby("method").sharpe.idxmax()].set_index("method")
            dk = best.loc[[k for k in RFF if k in best.index]].sharpe
            ln = best.loc[[k for k in LIN if k in best.index]].sharpe
            t = s.loc[s.method.str.startswith("rff"), "t_vs_fm"].max() if "t_vs_fm" in s else np.nan
            _lin = ln.max()
            # A percentage of a linear Sharpe near zero is not informative, and a negative
            # one is meaningless; guard rather than emit a huge or signed-wrong number.
            _pct = (lambda v: 100.0 * v / _lin if _lin > 1e-6 else np.nan)
            _fm = float(best.loc["fm", "sharpe"]) if "fm" in best.index else np.nan
            # The fair benchmark, from THIS run: every economy scores it in-run since the
            # protocol landed (estimation.fair_linear), so there is no second directory to
            # join against. Until 2026-09-17 these two columns came from variants/results_e1,
            # E1's separate re-scoring of the eight economies that predated --fair_linear, and
            # RESULTS.md had to be checked against two CSVs joined on (model, tag).
            _fair = best.loc[[k for k in FAIR if k in best.index]].sharpe
            _fair_lin = _fair.max() if len(_fair) else np.nan
            _ew = float(best.loc["ew", "sharpe"]) if "ew" in best.index else np.nan
            # Per-seed percentages of fm are recorded for completeness, but fm can be at or below zero in
            # a seed, so only the RATIO OF MEANS (economy table) is quoted; see the docstring.
            _pfm = (lambda v: 100.0 * v / _fm if (not np.isnan(_fm)) and _fm > 1e-6 else np.nan)
            rows.append(dict(base, window=w, sr_max_eval=sr_max_eval(w),
                             dkkm=dk.max(), lin=_lin, gap=dk.max() - _lin, t=t,
                             room_all_pct_lin=_pct(base["room_all"]),
                             room_eval_pct_lin=_pct(base["room_eval"]),
                             gap_pct_lin=_pct(dk.max() - _lin),
                             fm=_fm, gap_fm=dk.max() - _fm,
                             room_all_pct_fm=_pfm(base["room_all"]), room_eval_pct_fm=_pfm(base["room_eval"]),
                             gap_fm_pct_fm=_pfm(dk.max() - _fm),
                             fair_lin=_fair_lin, fair_gap=dk.max() - _fair_lin, ew=_ew,
                             fair_method=_fair.idxmax() if len(_fair) else None,
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
        for col in ("sr_max", "sr_max_eval", "room_all", "room_eval", "dkkm", "lin", "gap", "t",
                    "room_all_pct_lin", "room_eval_pct_lin", "gap_pct_lin",
                    "fm", "gap_fm", "room_all_pct_fm", "room_eval_pct_fm", "gap_fm_pct_fm",
                    "fair_lin", "fair_gap", "ew"):
            v = g[col].astype(float).dropna()
            rec[f"{col}_mean"] = v.mean() if len(v) else np.nan
            rec[f"{col}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            rec[f"{col}_se"] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            rec[f"{col}_n"] = int(len(v))
        rec["gap_over_room_all"] = rec["gap_mean"] / rec["room_all_mean"] if rec["room_all_mean"] else np.nan
        rec["gap_over_room_eval"] = rec["gap_mean"] / rec["room_eval_mean"] if rec["room_eval_mean"] else np.nan
        # Ratio of means, the figure the printed table and RESULTS.md quote. The mean and sd
        # of the per-seed ratios are the *_pct_lin_mean/_sd columns above; see the docstring.
        _lm = rec["lin_mean"]
        ok = _lm is not None and not np.isnan(_lm) and _lm > 1e-6
        rec["room_all_over_lin"] = 100.0 * rec["room_all_mean"] / _lm if ok else np.nan
        rec["room_eval_over_lin"] = 100.0 * rec["room_eval_mean"] / _lm if ok else np.nan
        rec["gap_over_lin"] = 100.0 * rec["gap_mean"] / _lm if ok else np.nan
        # The same ratios of means over Fama-MacBeth's Sharpe, the denominator RESULTS.md reports.
        _fmm = rec["fm_mean"]
        okf = _fmm is not None and not np.isnan(_fmm) and _fmm > 1e-6
        rec["gap_fm_over_fm"] = 100.0 * rec["gap_fm_mean"] / _fmm if okf else np.nan
        rec["room_all_over_fm"] = 100.0 * rec["room_all_mean"] / _fmm if okf else np.nan
        rec["room_eval_over_fm"] = 100.0 * rec["room_eval_mean"] / _fmm if okf else np.nan
        out.append(rec)
    return pd.DataFrame(out).sort_values(["model", "tag", "N", "T", "window"]).reset_index(drop=True)


def render(econ):
    lines = ["| economy | spec | n | SR_max all | SR_max eval | room all | room eval | room % of lin | "
             "DKKM | best lin | gap | gap % of lin | t | gap/room all | gap/room eval |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def ms(r, c, sign=False):
        m, sd = r[f"{c}_mean"], r[f"{c}_sd"]
        if pd.isna(m):
            return "-"
        d = 1 if c == "t" else 4
        out = f"{m:+.{d}f}" if sign else f"{m:.{d}f}"
        return out + (f" ({sd:.{d}f})" if not pd.isna(sd) else "")

    def ratio(x):
        return "-" if pd.isna(x) else f"{x:.2f}"

    def pct(x):
        return "-" if pd.isna(x) else f"{x:.1f}%"

    for _, r in econ.iterrows():
        w = "-" if pd.isna(r["window"]) else int(r["window"])
        econ_name = f"{r['model']}/{r['tag']} N={r['N']} T={r['T']} w={w}"
        spec = r["spec_id"] or "-"
        lines.append(f"| {econ_name} | {spec} | {r['n_seeds']} | {ms(r, 'sr_max')} | {ms(r, 'sr_max_eval')} | "
                     f"{ms(r, 'room_all', True)} | {ms(r, 'room_eval', True)} | {pct(r['room_all_over_lin'])} | "
                     f"{ms(r, 'dkkm')} | {ms(r, 'lin')} | {ms(r, 'gap', True)} | {pct(r['gap_over_lin'])} | "
                     f"{ms(r, 't')} | {ratio(r['gap_over_room_all'])} | {ratio(r['gap_over_room_eval'])} |")
    lines.append("")
    lines.append("mean (sd across seeds). gap/room and the two `% of lin` columns are RATIOS OF MEANS; "
                 "the mean and sd of the per-seed ratios are in economy_table.csv as *_pct_lin_mean/_sd. "
                 "room_eval is NaN for oracles run before --eval_window (2026-09-09); SR_max eval is "
                 "available for every run, from the per-month series, and is the bound DKKM must respect.")
    return "\n".join(lines)


def off_protocol(econ):
    """Rows whose sample is not the protocol's, as `model/tag N=.. T=.. w=..` strings."""
    bad = []
    for _, r in econ.iterrows():
        w = None if pd.isna(r["window"]) else int(r["window"])
        if (r["N"], r["T"], w) != (protocol.N, protocol.T, protocol.WINDOW):
            bad.append(f"{r['model']}/{r['tag']} N={r['N']} T={r['T']} w={w}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=None, help="default: the results dir")
    a = ap.parse_args(argv)
    seeds = seed_rows(a.results)
    econ = economy_table(seeds)
    off = off_protocol(econ)
    canonical = os.path.realpath(a.results) == os.path.realpath(os.path.join(HERE, "results"))
    if off and canonical:
        raise SystemExit(
            "REFUSED: the canonical results directory holds rows off the measurement "
            "protocol (N=%d T=%d window=%d):\n  %s\n"
            "Rows measured on different samples are not comparable, and a table that mixes "
            "them invites exactly the comparison it should prevent. An off-protocol run "
            "belongs in its own BOP_RESULTS_DIR." % (
                protocol.N, protocol.T, protocol.WINDOW, "\n  ".join(off)))
    if off:
        print("WARNING: off protocol, not reportable:\n  " + "\n  ".join(off) + "\n",
              file=sys.stderr)
    out = a.out or a.results
    seeds.to_csv(os.path.join(out, "seed_table.csv"), index=False)
    econ.to_csv(os.path.join(out, "economy_table.csv"), index=False)
    print(render(econ))
    print(f"\nwrote {os.path.join(out, 'seed_table.csv')} ({len(seeds)} rows) and economy_table.csv ({len(econ)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
