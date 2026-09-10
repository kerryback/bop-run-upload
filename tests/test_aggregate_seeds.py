"""The tracked aggregate producer must reproduce the numbers already in the record.

WORKING.md §37 and §40 report the ten-seed vyx and g0235 means from a gitignored script;
this pins variants/aggregate_seeds.py to them, so the tracked table cannot silently drift
from what was written down. Also pins: unseeded legacy files are never read, every flagship
row carries a spec_id, and gap/room is the ratio of means (§40's correction).

Run with: python -m pytest tests/ -k aggregate
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))
import aggregate_seeds  # noqa: E402

RESULTS = os.path.join(ROOT, "variants", "results")


def _flagship():
    seeds = aggregate_seeds.seed_rows(RESULTS)
    seeds = seeds[(seeds["N"] == 500) & (seeds["T"] == 500)]
    return seeds, aggregate_seeds.economy_table(seeds)


def _row(econ, model, tag, window=360):
    r = econ[(econ["model"] == model) & (econ["tag"] == tag) & (econ["window"] == window)]
    assert len(r) == 1, (model, tag, len(r))
    return r.iloc[0]


def test_vyx_ten_seeds_match_working_37():
    r = _row(_flagship()[1], "kp_vy", "vyx")
    assert r["n_seeds"] == 10
    assert abs(r.room_all_mean - 0.3491) < 5e-4 and abs(r.room_all_sd - 0.0365) < 5e-4
    assert abs(r.gap_mean - 0.1046) < 5e-4 and abs(r.gap_sd - 0.0155) < 5e-4


def test_g0235_ten_seeds_match_working_40():
    r = _row(_flagship()[1], "bgn_gam", "g0235")
    assert r["n_seeds"] == 10
    assert abs(r.room_all_mean - 0.0188) < 5e-4 and abs(r.room_all_sd - 0.0065) < 5e-4
    assert abs(r.gap_mean - 0.0227) < 5e-4 and abs(r.gap_sd - 0.0084) < 5e-4
    assert abs(r["gap_over_room_all"] - 1.21) < 0.02, "ratio of means, not mean of ratios (1.31)"


def test_gs_seed_zero_rows_carry_eval_room_and_spec():
    seeds, econ = _flagship()
    for tag, spec in (("g28", "var-gs_bx-g28-v2"), ("bx7", "var-gs_bx-bx7-v3")):
        r = _row(econ, "gs_bx", tag)
        assert r["n_seeds"] >= 1 and r["spec_id"] == spec
        assert r.room_eval_n >= 1, "gs oracles ran with --eval_window, so eval room must exist"
        s0 = seeds[(seeds["tag"] == tag) & (seeds.seed == 0)].iloc[0]
        assert s0.solves and "UNREGISTERED" not in s0.solves
        assert s0.prov_oracle and s0.prov_est


def test_every_flagship_row_names_its_spec_and_solves():
    seeds, _ = _flagship()
    assert seeds["spec_id"].notna().all(), seeds[seeds["spec_id"].isna()][["model", "tag", "seed"]]
    assert (seeds.solves.str.len() > 0).all()


def test_unseeded_legacy_files_are_never_read():
    seeds = aggregate_seeds.seed_rows(RESULTS)
    # kp_vy_oracle_vyx.json (no seed suffix) is the pre-lambda economy at SR_max 1.2607;
    # every seeded vyx oracle is below 1.24.
    assert (seeds[(seeds["model"] == "kp_vy") & (seeds["tag"] == "vyx")].sr_max < 1.25).all()


def test_no_estimator_beats_the_windowed_oracle_bound():
    """Cauchy-Schwarz: w'mu / sqrt(w' Sigma w) <= sqrt(mu' Sigma^-1 mu) = sr_max, every month.
    The estimators are scored against the TRUE moments, so no method can beat the oracle's
    sr_max averaged over the SAME months. Violating it means a real bug -- or, as happened
    on 2026-09-10, that the two sides were averaged over different months: g28's DKKM is
    0.2975 against an all-month sr_max of 0.2621 and an evaluation-window sr_max of 0.3087.
    """
    seeds, _ = _flagship()
    scored = seeds[seeds["sr_max_eval"].notna() & seeds["dkkm"].notna()]
    assert len(scored) >= 40, f"only {len(scored)} scored runs; the bound is not being checked"
    bad = scored[(scored["dkkm"] > scored["sr_max_eval"] + 1e-9)
                 | (scored["lin"] > scored["sr_max_eval"] + 1e-9)]
    assert bad.empty, ("estimators beating the windowed oracle bound:\n"
                       + bad[["model", "tag", "seed", "window", "dkkm", "lin", "sr_max_eval"]].to_string())


def test_sr_max_eval_exceeds_all_month_sr_max_on_these_panels():
    """Not a law, a fact about these four economies, and the reason the confound bites: the
    last 125 months of every flagship panel carry a higher tangency SR than the full 485."""
    _, econ = _flagship()
    for _, r in econ[econ["sr_max_eval_n"] > 0].iterrows():
        assert r["sr_max_eval_mean"] > r["sr_max_mean"], (r["model"], r["tag"])


def test_the_csvs_are_the_complete_set_regardless_of_flagship():
    """--flagship narrows the printout only; the tracked tables must not depend on it."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        for flag in ([], ["--flagship"]):
            subprocess.run([sys.executable, os.path.join(ROOT, "variants", "aggregate_seeds.py"),
                            "--out", td] + flag, check=True, capture_output=True)
            n = sum(1 for _ in open(os.path.join(td, "economy_table.csv"))) - 1
            assert n >= 5, f"economy_table.csv has {n} rows under {flag or 'no flag'}; probes at other N/T are missing"


def test_room_eval_is_nan_where_the_oracle_predates_the_flag():
    seeds, _ = _flagship()
    old = seeds[(seeds["tag"] == "vyx")]
    assert old.room_eval.isna().all(), "vyx oracles predate --eval_window; a value here is a bug"


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
