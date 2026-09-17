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
penalty sits at an edge in more than two of ten seeds has a censored DKKM number, and the
grid needs another decade before that economy's gap is quoted or built on.

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
SUMMARY = re.compile(r"^(?P<model>[a-z_]+)_estimators_(?P<tag>.+)_s(?P<seed>\d{3})"
                     r"_w(?P<window>\d+)_summary\.csv$")


def winners(results):
    """One row per seed: the (P, kappa) at which its best DKKM portfolio was found."""
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
        rows.append({"model": m["model"], "tag": m["tag"], "seed": int(m["seed"]),
                     "window": int(m["window"]), "P": int(best["P"]),
                     "kappa": float(best["kappa"]), "sharpe": float(best["sharpe"]),
                     "method": best["method"],
                     "n_kappas": d["kappa"].nunique()})
    return pd.DataFrame(rows)


def report(w, min_interior, grid):
    lo, hi = min(grid), max(grid)
    lines, failed = [], []
    lines.append(f"ridge grid: {', '.join(repr(k) for k in grid)}")
    lines.append(f"gate: the winning penalty is interior (neither {lo!r} nor {hi!r}) "
                 f"in at least {min_interior} of 10 seeds\n")
    lines.append("%-9s %-9s %5s %9s %9s %9s  %-24s %s"
                 % ("model", "tag", "seeds", "interior", f"at {lo!r}", f"at {hi!r}",
                    "winning penalties", "verdict"))
    for (model, tag), g in w.groupby(["model", "tag"]):
        n = len(g)
        at_lo = int((g["kappa"] == lo).sum())
        at_hi = int((g["kappa"] == hi).sum())
        interior = n - at_lo - at_hi
        counts = g["kappa"].value_counts().sort_index()
        shown = " ".join(f"{k:g}x{v}" for k, v in counts.items())
        ok = interior >= min_interior
        if not ok:
            failed.append(f"{model}/{tag}: interior in {interior} of {n} "
                          f"({at_lo} at the floor {lo!r}, {at_hi} at the ceiling {hi!r})")
        lines.append("%-9s %-9s %5d %9d %9d %9d  %-24s %s"
                     % (model, tag, n, interior, at_lo, at_hi, shown,
                        "PASS" if ok else "*** CENSORED ***"))
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
    print("GATE PASSED: every economy's winning penalty is interior. The reported DKKM "
          "Sharpe is the economy's, not the grid's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
