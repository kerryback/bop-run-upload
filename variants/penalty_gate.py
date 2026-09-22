"""Did the ridge grid bind? The first thing to read off any campaign.

WHY THIS IS A GATE AND NOT A NOTE. DKKM's reported Sharpe is a MAXIMUM over the ridge
penalty grid. If the winning penalty is at an EDGE of that grid, the number is censored:
it says where the search stopped, not what the economy affords. That is not hypothetical.
Before the 2026-09-15 protocol, the grid's floor won in 9 of 10 kp_vy/vyx seeds and 10 of
10 kp_vy/vyg25 seeds, and the one economy given an extra decade (the retired vyxT860)
gained +0.049 of DKKM Sharpe from the decade alone -- about a quarter of its headline gap.
Three different grids were in use, so the censoring point differed by economy and the rows
were not comparable.

The protocol's grid (variants/common/protocol.py KAPPAS) is wide on both sides so the
argmax CAN be interior. Whether it IS interior is a property of the results, which no test
can assert ahead of time. Hence this script, and hence the rule: an economy whose winning
penalty sits at an edge in more than two of ten seeds is FLAGGED, and asked the second
question below before its gap is quoted or built on.

POSITION IS NOT THE TEST, AND ON 2026-09-22 IT STOPPED BEING TREATED AS ONE. An argmax at
the ceiling means the grid binds only if the curve is still CLIMBING there. It need not be.
As kappa -> inf the ridge coefficient (X'X + kI)^-1 X'y -> X'y / k: the portfolio DIRECTION
stops depending on kappa, and a Sharpe ratio is scale-invariant, so sharpe(kappa) has a
HORIZONTAL ASYMPTOTE. Once the grid reaches it the argmax lands on whichever ceiling node
wins in the sixth decimal, and every further decade reproduces that exactly. Measured on the
protocol-v3 campaign: 16 of 130 seeds put the argmax at the ceiling 1000, and for every one
of them the ceiling gained at most 2.7e-05 of Sharpe over the best interior point -- three
orders of magnitude below the smallest gap this project reports. Widening the grid again
would have cost a 130-job campaign and changed nothing.

So the gate asks both questions and only the second one is decisive:

  1. is the argmax interior in at least `--min-interior` of ten seeds?   (position)
  2. for the seeds where it is not, what does the edge BUY over the best interior point?
     If that gain is at most TOL for all of them, the curve has flattened, another decade
     cannot move the reported number, and the row is NOT censored.        (materiality)

TOL is half the last digit docs/RESULTS.md prints, so a flat-tail pass certifies exactly
what the document needs: no number in it could change. A row that fails BOTH is censored and
the grid genuinely needs another decade on the binding side.

usage:
  python variants/penalty_gate.py                      # the canonical results directory
  python variants/penalty_gate.py --results DIR
  python variants/penalty_gate.py --min-interior 8     # default: 8 of 10 seeds
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
import protocol  # noqa: E402

# The random-feature estimators. `dkkm` in the economy table is the best Sharpe over these,
# any feature count, any penalty -- so the argmax over exactly this set is what was reported.
DKKM = ("rff", "rff_ens", "rff_lev", "rff_lev_ens")
# Half of RESULTS.md's last printed digit: an edge win that buys less than this cannot change
# a number the document shows, however many decades are added beyond it.
TOL = 5e-5
SUMMARY = re.compile(r"^(?P<model>[a-z_]+)_estimators_(?P<tag>.+)_s(?P<seed>\d{3})"
                     r"_w(?P<window>\d+)_summary\.csv$")


def winners(results):
    """One row per seed: the (P, kappa) at which its best DKKM portfolio was found.

    `edge_gain` is what that win is WORTH if it sits at an edge -- the best Sharpe at the two
    edge penalties minus the best over every interior one. It is the materiality question, and
    it is what decides the gate; `kappa` alone only says where the argmax landed.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(results, "*_summary.csv"))):
        m = SUMMARY.match(os.path.basename(path))
        if not m:
            continue
        d = pd.read_csv(path)
        d = d[d["method"].isin(DKKM)]
        if d.empty:
            continue
        best = d.loc[d["sharpe"].idxmax()]
        by_kappa = d.groupby("kappa")["sharpe"].max().sort_index()
        interior = by_kappa.iloc[1:-1]
        rows.append({"model": m["model"], "tag": m["tag"], "seed": int(m["seed"]),
                     "window": int(m["window"]), "P": int(best["P"]),
                     "kappa": float(best["kappa"]), "sharpe": float(best["sharpe"]),
                     "method": best["method"],
                     "n_kappas": d["kappa"].nunique(),
                     "edge_gain": (float(by_kappa.iloc[[0, -1]].max() - interior.max())
                                   if len(interior) else float("nan"))})
    return pd.DataFrame(rows)


def report(w, min_interior, grid, tol=TOL):
    lo, hi = min(grid), max(grid)
    lines, failed = [], []
    lines.append(f"ridge grid: {', '.join(repr(k) for k in grid)}")
    lines.append(f"gate: the winning penalty is interior (neither {lo!r} nor {hi!r}) "
                 f"in at least {min_interior} of 10 seeds,")
    lines.append(f"      OR an edge win buys at most {tol:g} of Sharpe over the best "
                 f"interior penalty, which means the curve has flattened\n")
    lines.append("%-9s %-9s %5s %9s %9s %9s %11s  %-28s %s"
                 % ("model", "tag", "seeds", "interior", f"at {lo!r}", f"at {hi!r}",
                    "edge buys", "winning penalties", "verdict"))
    for (model, tag), g in w.groupby(["model", "tag"]):
        n = len(g)
        at_lo = int((g["kappa"] == lo).sum())
        at_hi = int((g["kappa"] == hi).sum())
        interior = n - at_lo - at_hi
        edge = g.loc[g["kappa"].isin((lo, hi)), "edge_gain"]
        buys = float(edge.max()) if len(edge) else 0.0
        counts = g["kappa"].value_counts().sort_index()
        shown = " ".join(f"{k:g}x{v}" for k, v in counts.items())
        if interior >= min_interior:
            verdict = "PASS"
        elif buys <= tol:
            verdict = "PASS (flat tail)"
        else:
            verdict = "*** CENSORED ***"
            failed.append(f"{model}/{tag}: interior in {interior} of {n} "
                          f"({at_lo} at the floor {lo!r}, {at_hi} at the ceiling {hi!r}), "
                          f"and the edge buys {buys:.2e} over the best interior penalty")
        lines.append("%-9s %-9s %5d %9d %9d %9d %11.1e  %-28s %s"
                     % (model, tag, n, interior, at_lo, at_hi, buys, shown, verdict))
    return "\n".join(lines), failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--min-interior", type=int, default=8)
    a = ap.parse_args(argv)

    w = winners(a.results)
    if w.empty:
        raise SystemExit(f"no estimator summaries in {a.results}")

    grid = list(protocol.KAPPAS)
    # A seed scored on a different grid is not comparable and must not be averaged in.
    off = w[w["n_kappas"] != len(grid)]
    if len(off):
        print("WARNING: seeds scored on a grid of a different size -- these predate the "
              "protocol and their DKKM numbers are not comparable:", file=sys.stderr)
        for _, r in off.iterrows():
            print(f"  {r['model']}/{r['tag']} s{r['seed']:03d}: {r['n_kappas']} penalties, "
                  f"expected {len(grid)}", file=sys.stderr)
        print(file=sys.stderr)

    text, failed = report(w[w["n_kappas"] == len(grid)], a.min_interior, grid)
    print(text)
    print()
    if failed:
        print("GATE FAILED -- these economies' DKKM Sharpe is censored by the grid:")
        for f in failed:
            print("  " + f)
        print("\nThe grid needs another decade on the binding side before these gaps are "
              "quoted or built on (docs/RESULTS.md, 'The measurement protocol').")
        return 1
    flat = [l for l in text.splitlines() if l.endswith("PASS (flat tail)")]
    print("GATE PASSED: no economy's reported DKKM Sharpe is set by where the grid stops.")
    if flat:
        print(f"  {len(flat)} of them by the FLAT-TAIL clause rather than by position: the "
              f"argmax sits at an edge in more than {10 - a.min_interior} seeds, but the "
              f"curve there is flat, so another decade cannot move the number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
