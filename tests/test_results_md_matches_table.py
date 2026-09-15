"""docs/RESULTS.md's economy tables must agree with the tables the pipeline produces.

Every economy-results table in RESULTS.md shares one column set (2026-09-15), defined once in "Reading
the tables" at the top. EVERY such table is pinned here, cell by cell, against
variants/results/economy_table.csv (aggregate_seeds.py) and variants/results_e1/fair_gap_economy_table.csv
(fair_gap.py): seeds, SR_max, the EW market, FMR, best linear and DKKM Sharpes, DKKM - FMR and its
percentage of FMR, DKKM - best linear, DKKM - best fair linear, room and its percentage of FMR, and t.

The two percentages are RATIOS OF MEANS over Fama-MacBeth's ten-seed mean Sharpe
(`gap_fm_over_fm`, `room_eval_over_fm`), not means of per-seed ratios: FMR's Sharpe is at or below zero
in some seeds, where a per-seed ratio is meaningless. Pinning the ratio of means stops the two being
mixed in the document, the mistake WORKING.md §40 records for gap/room.

The tables are read by COLUMN HEADER, not position, so adding or moving a column cannot make the parser
read the wrong cell. Every ten-seed row of the CSV must appear, and the tables must list no economy the CSV lacks.
Every row must be on the measurement protocol (variants/common/protocol.py). A row with fewer
than ten seeds is a screen and must instead have its own SCREEN heading -- a tier that exists
only until bgn_gam/g0235d reaches the protocol's ten seeds.

Run with: python -m pytest tests/ -k results_md
"""
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
import protocol  # noqa: E402
MD = os.path.join(ROOT, "docs", "RESULTS.md")
CSV = os.path.join(ROOT, "variants", "results", "economy_table.csv")
E1 = os.path.join(ROOT, "variants", "results_e1", "fair_gap_economy_table.csv")


COLS = {  # RESULTS.md header -> field; the unified column set of every economy-results table
    "economy": "economy", "seeds": "n", "SR_max": "sr_max_eval", "EW market SR": "ew", "FMR SR": "fm",
    "best linear SR": "lin", "DKKM SR": "dkkm", "DKKM - FMR": "gap_fm", "(DKKM - FMR) / FMR": "gap_fm_pct",
    "DKKM - best linear": "gap", "DKKM - best fair linear": "fair_gap", "room": "room_eval",
    "room / FMR": "room_pct", "t, DKKM vs FMR": "t",
}


# Tables that begin with `economy` but are not economy-results tables. Listed so an
# unrecognised shape fails loudly rather than being skipped.
OTHER_SHAPES = [
    ["economy", "quantity", "reference", "predicted", "result", "verdict"],
    ["economy", "spec", "what it is in the code", "conditioning columns", "solve"],
    ["economy", "spec", "what it is in the code", "solve"],
    ["economy", "model", "what differs"],
]


def _md_rows():
    """Every `model/tag` row of EVERY unified-column table in the file, read by column header.

    This used to parse one section, "Current results", and nothing checked the other tables:
    the headline table, the baselines table, the per-path ladders and finding 8's eleven-row
    split all carried the same quantities and could drift from the CSV silently. They are one
    document making one set of claims, so every table that uses the unified columns is pinned,
    and a row that appears in several must agree with the CSV in all of them.

    A row's first cell may carry a description after the tag (`kp_vy/vyx: the parent economy`),
    which is why the key is parsed off the front rather than matched whole.
    """
    txt = open(MD).read()
    block = txt

    def num(c):
        mm = re.match(r"^([+-]?\d+(?:,\d{3})*\.\d+)", c or "")
        return float(mm.group(1).replace(",", "")) if mm else None

    rows, header, tables = {}, None, 0
    for line in block.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "economy":
            # Three table shapes start with `economy`: the unified economy table, a
            # prediction table, and the descriptive "how each baseline is produced" one.
            # Only the first is pinned -- but an unrecognised shape is an error, so a new
            # table cannot quietly use near-miss columns and escape the check.
            if cells == list(COLS):
                header, tables = cells, tables + 1
            else:
                assert cells in OTHER_SHAPES, (
                    f"a table starting with `economy` uses columns that are neither the "
                    f"unified set nor a known descriptive shape: {cells}")
                header = None
            continue
        m = re.match(r"^([a-z_]+)/([A-Za-z0-9]+)(?::.*)?$", cells[0])
        if not (m and header):
            continue
        assert len(cells) == len(header), f"{cells[0]}: {len(cells)} cells under a {len(header)}-column header"
        c = {COLS[h]: v for h, v in zip(header, cells)}
        parsed = {k: (int(v) if k == "n" else num(v)) for k, v in c.items() if k != "economy"}
        key = (m.group(1), m.group(2))
        if key in rows:
            assert rows[key] == parsed, (
                f"{key[0]}/{key[1]} appears twice with different numbers; the tables of one "
                f"document must agree:\n  {rows[key]}\n  {parsed}")
        rows[key] = parsed
    assert rows and tables, "no unified-column economy tables parsed from RESULTS.md"
    return rows


def _csv_rows(current=True):
    """The economy rows, which are all on the measurement protocol by construction.

    This used to filter to N=500, T=500, window 360, because the CSV held rows that were not:
    smoke and scaling probes at N=60 to 500 and windows 20/36/50, and one economy at T=860.
    The filter is now an ASSERTION (test_the_csv_holds_nothing_off_protocol) -- silently
    dropping an off-protocol row is how one came to be published beside the others in the
    first place. The ten-seed split stays only while bgn_gam/g0235d has a single seed; the
    protocol's SEEDS is 10 for every economy, and the screen tier goes when it reaches that.
    """
    e = pd.read_csv(CSV)
    if current is not None:
        e = e[e["n_seeds"] >= protocol.SEEDS] if current else e[e["n_seeds"] < protocol.SEEDS]
    return {(r["model"], r["tag"]): r for _, r in e.iterrows()}


def test_the_csv_holds_nothing_off_protocol():
    """Every row of economy_table.csv is at the protocol's sample. aggregate_seeds.py refuses
    to write otherwise; this is the same claim, checked from the committed file."""
    e = pd.read_csv(CSV)
    bad = [f"{r['model']}/{r['tag']} N={r['N']} T={r['T']} w={r['window']}"
           for _, r in e.iterrows()
           if (r["N"], r["T"], r["window"]) != (protocol.N, protocol.T, protocol.WINDOW)]
    assert not bad, ("economy_table.csv holds rows off the protocol "
                     f"(N={protocol.N} T={protocol.T} window={protocol.WINDOW}):\n  "
                     + "\n  ".join(bad))


def test_every_screen_is_reported_under_a_screen_heading():
    """A flagship economy with fewer than ten seeds can neither vanish from the document nor be
    ranked beside ten-seed economies: it needs its own `#### <model>/<tag> -- SCREEN` heading."""
    txt = open(MD).read()
    missing = [f"{m}/{t}" for (m, t) in _csv_rows(current=False)
               if not re.search(rf"^#{{3,4}} {re.escape(m)}/{re.escape(t)} -- SCREEN", txt, re.M)]
    assert not missing, f"economy_table.csv has screens RESULTS.md does not report under a SCREEN heading: {missing}"


def test_every_flagship_economy_in_the_csv_is_in_the_table():
    md, csv = _md_rows(), _csv_rows()
    missing = sorted(set(csv) - set(md))
    assert not missing, f"economy_table.csv has flagship rows RESULTS.md does not list: {missing}"


def test_the_table_lists_no_economy_the_csv_lacks():
    md, csv = _md_rows(), _csv_rows(current=None)
    extra = sorted(set(md) - set(csv))
    assert not extra, f"RESULTS.md lists economies with no flagship row in economy_table.csv: {extra}"


def test_every_number_in_the_table_matches_the_csv_to_display_precision():
    # Every row the document shows, screens included: a screen is not RANKED, but the numbers
    # it does show must still be the ones the pipeline produced.
    md, csv = _md_rows(), _csv_rows(current=None)
    bad = []
    for key, row in md.items():
        c = csv[key]
        checks = [("seeds", row["n"], int(c["n_seeds"]), 0),
                  ("SR_max", row["sr_max_eval"], c["sr_max_eval_mean"], 5.1e-5),
                  ("FMR SR", row["fm"], c["fm_mean"], 5.1e-5),
                  ("best linear SR", row["lin"], c["lin_mean"], 5.1e-5),
                  ("DKKM SR", row["dkkm"], c["dkkm_mean"], 5.1e-5),
                  ("DKKM - FMR", row["gap_fm"], c["gap_fm_mean"], 5.1e-5),
                  ("DKKM - best linear", row["gap"], c["gap_mean"], 5.1e-5),
                  ("room", row["room_eval"], c["room_eval_mean"], 5.1e-5),
                  ("t, DKKM vs FMR", row["t"], c["t_mean"], 0.051),
                  # ratios of means over FMR's mean Sharpe, in percent -- NOT means of per-seed ratios
                  ("(DKKM - FMR) / FMR", row["gap_fm_pct"], c["gap_fm_over_fm"], 0.051),
                  ("room / FMR", row["room_pct"], c["room_eval_over_fm"], 0.051)]
        for name, got, want, tol in checks:
            if got is None or pd.isna(want):
                ok = got is None and pd.isna(want)
            else:
                ok = abs(float(got) - float(want)) <= tol
            if not ok:
                bad.append(f"{key[0]}/{key[1]} {name}: RESULTS.md {got!r} vs economy_table.csv {want!r}")
    assert not bad, "RESULTS.md is stale:\n  " + "\n  ".join(bad)


def test_the_market_and_fair_gap_columns_match_the_e1_table():
    """EW market SR and DKKM - best fair linear come from variants/fair_gap.py: the equal-weighted market's
    Sharpe, and DKKM against linear methods given that market on DKKM's terms. Outside KP14 the fair gap is
    the only complexity gap the tables show (RESULTS.md finding 8), so both are pinned like every other cell."""
    md = _md_rows()
    e = pd.read_csv(E1)
    e = {(r["model"], r["tag"]): r for _, r in e.iterrows()
         if int(r["window"]) == protocol.WINDOW}
    bad = []
    for key, row in md.items():
        if key not in e:
            bad.append(f"{key[0]}/{key[1]}: no row in {os.path.relpath(E1, ROOT)}")
            continue
        for name, field, want in [("EW market SR", "ew", e[key]["ew"]), ("DKKM - best fair linear", "fair_gap", e[key]["fair_gap"])]:
            if row[field] is None or abs(row[field] - want) > 5.1e-5:
                bad.append(f"{key[0]}/{key[1]} {name}: RESULTS.md {row[field]!r} vs {want!r}")
    assert not bad, "RESULTS.md market or fair-gap column is stale:\n  " + "\n  ".join(bad)


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
