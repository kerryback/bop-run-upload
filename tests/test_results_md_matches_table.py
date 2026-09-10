"""docs/RESULTS.md's current-results table must agree with variants/results/economy_table.csv.

Including the two `% of lin` columns, which are RATIOS OF MEANS (economy_table.csv's
`room_all_over_lin` / `gap_over_lin`) and not the mean of the per-seed ratios -- those differ
by 5 points for g0235 and live in the same CSV as `*_pct_lin_mean`. Pinning the ratio of means
here is what stops the two definitions being mixed in the document, which is the mistake
WORKING.md §40 records for gap/room.

RESULTS.md is the ongoing, human-written record of what each economy is and what it
produced; economy_table.csv is produced by aggregate_seeds.py from the result files. The
numbers in the prose table are copied by hand and go stale the moment a new seed or a new
economy lands. This pins them: every flagship row of the CSV must appear in the table with
the same n, room, gap and t to display precision, and the table must not list an economy
the CSV does not have.

Run with: python -m pytest tests/ -k results_md
"""
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "docs", "RESULTS.md")
CSV = os.path.join(ROOT, "variants", "results", "economy_table.csv")


def _md_rows():
    txt = open(MD).read()
    start = txt.index("## Current results")
    end = txt.index("\n## ", start + 5)
    block = txt[start:end]
    rows = {}
    for line in block.splitlines():
        m = re.match(r"^\|\s*([a-z_]+)/([A-Za-z0-9]+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # economy | spec | n | room all | room eval | room eval % of lin | gap | gap % of lin |
        # t | DKKM | best linear | SR_max eval
        def num(c):
            mm = re.match(r"^([+-]?\d+\.\d+)", c)
            return float(mm.group(1)) if mm else None

        def pct(c):
            mm = re.match(r"^([+-]?\d+\.\d+)%", c)
            return float(mm.group(1)) if mm else None
        rows[(m.group(1), m.group(2))] = dict(spec=cells[1], n=int(cells[2]), room_all=num(cells[3]),
                                             room_eval=num(cells[4]), room_pct=pct(cells[5]),
                                             gap=num(cells[6]), gap_pct=pct(cells[7]), t=num(cells[8]),
                                             dkkm=num(cells[9]), lin=num(cells[10]), sr_max_eval=num(cells[11]))
    assert rows, "no economy rows parsed from the 'Current results' table"
    return rows


def _csv_rows():
    e = pd.read_csv(CSV)
    e = e[(e["N"] == 500) & (e["T"] == 500) & (e["window"] == 360)]
    return {(r["model"], r["tag"]): r for _, r in e.iterrows()}


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
