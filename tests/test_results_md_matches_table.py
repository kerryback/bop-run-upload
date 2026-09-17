"""docs/RESULTS.md's economy tables must agree with the tables the pipeline produces.

Every economy-results table in RESULTS.md shares one column set (2026-09-15), defined once in "Reading
the tables" at the top. EVERY such table is pinned here, cell by cell, against
variants/results/economy_table.csv (aggregate_seeds.py), which since 2026-09-17 carries the
fair benchmark too: seeds, SR_max, the EW market, FMR, best linear and DKKM Sharpes, DKKM - FMR and its
percentage of FMR, DKKM - best linear, DKKM - best fair linear, room and its percentage of FMR, and t.

The two percentages are RATIOS OF MEANS over Fama-MacBeth's ten-seed mean Sharpe
(`gap_fm_over_fm`, `room_eval_over_fm`), not means of per-seed ratios: FMR's Sharpe is at or below zero
in some seeds, where a per-seed ratio is meaningless. Pinning the ratio of means stops the two being
mixed in the document, the mistake WORKING.md §40 records for gap/room.

The tables are read by COLUMN HEADER, not position, so adding or moving a column cannot make the parser
read the wrong cell. Every ten-seed row of the CSV must appear, and the tables must list no economy the CSV lacks.
Every row must be on the measurement protocol (variants/common/protocol.py), at its ten seeds.

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


COLS = {  # RESULTS.md header -> field; the unified column set of every economy-results table
    "economy": "economy", "seeds": "n", "SR_max": "sr_max_eval", "EW market SR": "ew", "FMR SR": "fm",
    "best linear SR": "lin", "DKKM SR": "dkkm", "DKKM - FMR": "gap_fm", "(DKKM - FMR) / FMR": "gap_fm_pct",
    "DKKM - best linear": "gap", "DKKM - best fair linear": "fair_gap", "room": "room_eval",
    "room / FMR": "room_pct", "t, DKKM vs FMR": "t",
}


def _is_near_miss(cells):
    """A header that is ALMOST the unified column set, which is what must fail loudly.

    Several tables in this file legitimately start with `economy` and are not economy-results
    tables: the prediction table, the baseline-production table, "what differs", the penalty
    gate. An allowlist of their exact shapes needed editing every time one was added, and an
    allowlist is the wrong instrument anyway -- what matters is not "is this shape known" but
    "is this a unified table that has DRIFTED". A header sharing most of the unified column
    names is drift; one sharing almost none is a different kind of table.
    """
    shared = len(set(cells) & set(COLS))
    return shared >= max(3, len(COLS) // 2) and cells != list(COLS)


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
            if cells == list(COLS):
                header, tables = cells, tables + 1
            else:
                # A descriptive table is fine; a DRIFTED unified table is not.
                assert not _is_near_miss(cells), (
                    f"a table shares most of the unified column set but is not it -- the "
                    f"columns have drifted: {cells}\nexpected: {list(COLS)}")
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


def _csv_rows(current=None):
    """Every economy row. All on protocol, all at the protocol's ten seeds.

    Two filters used to live here and both are now assertions instead. The sample filter
    (N=500, T=500, window 360) went when the off-protocol probe rows were deleted -- silently
    dropping such a row is how one came to be published beside the others. The seed filter
    went on 2026-09-17, when bgn_gam/g0235d got its ten seeds: it existed to keep a one-seed
    SCREEN out of a ranked table, and there is no longer any tier but CURRENT. `current` is
    kept as an accepted-and-ignored argument only so a stale caller fails loudly on the
    assertion below rather than silently filtering.
    """
    e = pd.read_csv(CSV)
    assert (e["n_seeds"] == protocol.SEEDS).all(), (
        f"economy_table.csv has rows away from the protocol's {protocol.SEEDS} seeds:\n"
        + e.loc[e["n_seeds"] != protocol.SEEDS, ["model", "tag", "n_seeds"]].to_string())
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


def test_no_economy_is_below_the_protocols_seed_count():
    """There is no SCREEN tier any more.

    A screen was one seed, run under a spec whose other seeds started only if it cleared a
    gate; it was reported under its own heading and never ranked, because cross-seed sd of a
    gap runs 15% to 63% of its mean (finding 4). The protocol sets ten seeds for every
    economy, so fewer seeds is a numerical shortcut rather than a tier, and bgn_gam/g0235d --
    the last screen -- got its ten on 2026-09-17. This is the assertion that keeps it that
    way; _csv_rows carries the same check so every other test in this file inherits it.
    """
    e = pd.read_csv(CSV)
    short = e.loc[e["n_seeds"] < protocol.SEEDS, ["model", "tag", "n_seeds"]]
    assert short.empty, ("economies below the protocol's seed count:\n" + short.to_string()
                         + "\nRun them to ten seeds; do not reintroduce a SCREEN heading.")


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


def test_the_market_and_fair_gap_columns_come_from_the_one_table():
    """EW market SR and DKKM - best fair linear used to live in a SECOND table.

    Until 2026-09-17 they came from variants/results_e1/fair_gap_economy_table.csv, E1's
    separate re-scoring of the eight economies that predated --fair_linear, and this test
    re-performed the join RESULTS.md performed by hand. Every economy now scores the fair
    benchmark in its own run (the protocol's estimation.fair_linear), so aggregate_seeds.py
    computes both columns and there is one canonical table. Verified bit-identical to
    fair_gap.py's arithmetic on all thirteen economies before that script was deleted.
    """
    md, csv = _md_rows(), _csv_rows(current=None)
    bad = []
    for key, row in md.items():
        c = csv[key]
        for name, field, want in (("EW market SR", "ew", c["ew_mean"]),
                                  ("DKKM - best fair linear", "fair_gap", c["fair_gap_mean"])):
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
