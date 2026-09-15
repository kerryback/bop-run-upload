"""docs/RESULTS.md's current-results table must agree with variants/results/economy_table.csv.

Including the two `% of lin` columns, which are RATIOS OF MEANS (economy_table.csv's
`room_all_over_lin` / `gap_over_lin`) and not the mean of the per-seed ratios -- those differ
by 5 points for g0235 and live in the same CSV as `*_pct_lin_mean`. Pinning the ratio of means
here is what stops the two definitions being mixed in the document, which is the mistake
WORKING.md §40 records for gap/room.

RESULTS.md is the ongoing, human-written record of what each economy is and what it
produced; economy_table.csv is produced by aggregate_seeds.py from the result files. The
numbers in the prose table are copied by hand and go stale the moment a new seed or a new
economy lands. This pins them: every ten-seed flagship row of the CSV must appear in the table
with the same n, room, gap and t to display precision, and the table must not list an economy
the CSV does not have. A flagship row with fewer seeds is a screen and must instead have its
own SCREEN heading.

Run with: python -m pytest tests/ -k results_md
"""
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "docs", "RESULTS.md")
CSV = os.path.join(ROOT, "variants", "results", "economy_table.csv")
E1 = os.path.join(ROOT, "variants", "results_e1", "fair_gap_economy_table.csv")


def _md_rows():
    """Every `model/tag` row of the tables under "Current results", read by COLUMN HEADER, not position:
    RESULTS.md adds columns (2026-09-15: room all % of lin, fair gap % of fair lin), and a positional
    parser silently reads the wrong cell when one moves."""
    txt = open(MD).read()
    start = txt.index("## Current results")
    end = txt.index("\n## ", start + 5)
    block = txt[start:end]

    def num(c):
        mm = re.match(r"^([+-]?\d+\.\d+)", c or "")
        return float(mm.group(1)) if mm else None

    def pct(c):
        mm = re.match(r"^([+-]?\d+\.\d+)%", c or "")
        return float(mm.group(1)) if mm else None

    rows, header = {}, None
    for line in block.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "economy":
            header = cells
            continue
        m = re.match(r"^([a-z_]+)/([A-Za-z0-9]+)$", cells[0])
        if not (m and header):
            continue
        assert len(cells) == len(header), f"{cells[0]}: {len(cells)} cells under a {len(header)}-column header"
        c = dict(zip(header, cells))
        rows[(m.group(1), m.group(2))] = dict(
            spec=c["spec"], n=int(c["n"]), room_all=num(c["room all"]), room_all_pct=pct(c.get("room all % of lin")),
            room_eval=num(c["room eval"]), room_pct=pct(c["room eval % of lin"]),
            gap=num(c["gap"]), gap_pct=pct(c["gap % of lin"]), t=num(c["t"]),
            dkkm=num(c["DKKM"]), lin=num(c["best linear"]), sr_max_eval=num(c["SR_max eval"]),
            fair_gap=num(c.get("fair gap")), fair_gap_pct=pct(c.get("fair gap % of fair lin")))
    assert rows, "no economy rows parsed from the 'Current results' tables"
    return rows


def _csv_rows(current=True):
    """Flagship rows (N=500, T=500, window 360). The table ranks CURRENT economies, which have ten
    seeds (RESULTS.md, status labels); a flagship row with fewer is a screen, reported in its model's
    section and never ranked, so current=False returns those instead."""
    e = pd.read_csv(CSV)
    e = e[(e["N"] == 500) & (e["T"] == 500) & (e["window"] == 360)]
    e = e[e["n_seeds"] >= 10] if current else e[e["n_seeds"] < 10]
    return {(r["model"], r["tag"]): r for _, r in e.iterrows()}


def test_every_screen_is_reported_under_a_screen_heading():
    """A flagship economy with fewer than ten seeds can neither vanish from the document nor be
    ranked beside ten-seed economies: it needs its own `#### <model>/<tag> -- SCREEN` heading."""
    txt = open(MD).read()
    missing = [f"{m}/{t}" for (m, t) in _csv_rows(current=False)
               if not re.search(rf"^#### {re.escape(m)}/{re.escape(t)} -- SCREEN", txt, re.M)]
    assert not missing, f"economy_table.csv has screens RESULTS.md does not report under a SCREEN heading: {missing}"


def test_every_flagship_economy_in_the_csv_is_in_the_table():
    md, csv = _md_rows(), _csv_rows()
    missing = sorted(set(csv) - set(md))
    assert not missing, f"economy_table.csv has flagship rows RESULTS.md does not list: {missing}"


def test_the_table_lists_no_economy_the_csv_lacks():
    md, csv = _md_rows(), _csv_rows()
    extra = sorted(set(md) - set(csv))
    assert not extra, f"RESULTS.md lists economies with no flagship row in economy_table.csv: {extra}"


def test_every_number_in_the_table_matches_the_csv_to_display_precision():
    md, csv = _md_rows(), _csv_rows()
    bad = []
    for key, row in md.items():
        c = csv[key]
        checks = [("n", row["n"], int(c["n_seeds"]), 0),
                  ("spec", row["spec"], c["spec_id"], None),
                  ("room_all", row["room_all"], c["room_all_mean"], 5.1e-5),
                  ("gap", row["gap"], c["gap_mean"], 5.1e-5),
                  ("t", row["t"], c["t_mean"], 0.051),
                  ("dkkm", row["dkkm"], c["dkkm_mean"], 5.1e-5),
                  ("lin", row["lin"], c["lin_mean"], 5.1e-5),
                  ("sr_max_eval", row["sr_max_eval"], c["sr_max_eval_mean"], 5.1e-5),
                  # ratio of means, in percent -- NOT the mean of the per-seed ratios
                  ("room_all_pct", row["room_all_pct"], c["room_all_over_lin"], 0.051),
                  ("room_pct", row["room_pct"], c["room_eval_over_lin"], 0.051),
                  ("gap_pct", row["gap_pct"], c["gap_over_lin"], 0.051)]
        if row["room_eval"] is not None or not pd.isna(c["room_eval_mean"]):
            checks.append(("room_eval", row["room_eval"], c["room_eval_mean"], 5.1e-5))
        for name, got, want, tol in checks:
            if tol is None:
                ok = got == want
            elif got is None or (isinstance(want, float) and pd.isna(want)):
                ok = got is None and (want is None or pd.isna(want))
            else:
                ok = abs(float(got) - float(want)) <= tol
            if not ok:
                bad.append(f"{key[0]}/{key[1]} {name}: RESULTS.md {got!r} vs economy_table.csv {want!r}")
    assert not bad, "RESULTS.md is stale:\n  " + "\n  ".join(bad)


def test_the_fair_gap_column_matches_the_e1_table():
    """The last column is E1's fair gap (variants/fair_gap.py): DKKM against linear methods given the
    equal-weighted market on DKKM's terms. Outside vyx it is the only complexity gap the table shows
    (RESULTS.md finding 8), so it is pinned like every other cell. An economy whose own run scored the
    fair methods (--fair_linear) will need fair_gap.py to read that run as well as results_e1/."""
    md = _md_rows()
    e = pd.read_csv(E1)
    e = {(r["model"], r["tag"]): r for _, r in e.iterrows() if int(r["window"]) == 360}
    bad = []
    for key, row in md.items():
        if key not in e:
            bad.append(f"{key[0]}/{key[1]}: no row in {os.path.relpath(E1, ROOT)}")
        elif row["fair_gap"] is None or abs(row["fair_gap"] - e[key]["fair_gap"]) > 5.1e-5:
            bad.append(f"{key[0]}/{key[1]} fair gap: RESULTS.md {row['fair_gap']!r} vs {e[key]['fair_gap']!r}")
        elif row["fair_gap_pct"] is None or abs(row["fair_gap_pct"] - e[key]["fair_gap_over_fair_lin"]) > 0.051:
            # the fair gap over the fair linear Sharpe, a ratio of means in percent
            bad.append(f"{key[0]}/{key[1]} fair gap % of fair lin: RESULTS.md {row['fair_gap_pct']!r} "
                       f"vs {e[key]['fair_gap_over_fair_lin']!r}")
    assert not bad, "RESULTS.md fair-gap column is stale:\n  " + "\n  ".join(bad)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
