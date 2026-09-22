"""The tracked aggregate producer must reproduce the numbers already in the record.

WORKING.md §37 and §40 report the ten-seed vyx and g0235 means from a gitignored script;
this pins variants/aggregate_seeds.py to them, so the tracked table cannot silently drift
from what was written down. Also pins: unseeded legacy files are never read, every flagship
row carries a spec_id, and gap/room is the ratio of means (§40's correction).

Run with: python -m pytest tests/ -k aggregate
"""
import json
import os
import sys

import pandas as pd

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


def test_vyx_ten_seeds_match_the_published_row():
    """Pinned to the protocol-v3 campaign of 2026-09-21..22, the first with correct pricing.

    room here is a POPULATION quantity, and unlike the two earlier protocol changes this one
    MOVED it: +0.3491 to +0.0549. variants/kp_vy priced the OU state y as a constant addition
    to the discount rate, which over-discounts because an OU Girsanov adjustment saturates,
    and correcting it (merge 91095fb) cut the attainable Sharpe by roughly two thirds. The gap
    went with it, +0.1251 to +0.0064, and the FAIR gap to +0.0009 -- the falsification clause
    in var-kp_vy-vyx-v4. Nothing here is a lower bound any more: no seed wins at either edge
    of the ridge grid (variants/penalty_gate.py).
    """
    r = _row(_flagship()[1], "kp_vy", "vyx")
    assert r["n_seeds"] == 10
    assert abs(r.room_all_mean - 0.0549) < 5e-4 and abs(r.room_all_sd - 0.0045) < 5e-4
    assert abs(r.gap_mean - 0.0064) < 5e-4 and abs(r.gap_sd - 0.0161) < 5e-4


def test_g0235_ten_seeds_match_the_published_row():
    """Also protocol v3: BGN's bond recursion added the log-kernel/short-rate covariance to
    the cumulative variance once instead of twice, so every bgn_gam row moved too."""
    r = _row(_flagship()[1], "bgn_gam", "g0235")
    assert r["n_seeds"] == 10
    assert abs(r.room_all_mean - 0.0281) < 5e-4 and abs(r.room_all_sd - 0.0069) < 5e-4
    assert abs(r.gap_mean - 0.0206) < 5e-4 and abs(r.gap_sd - 0.0076) < 5e-4
    assert abs(r["gap_over_room_all"] - 0.733) < 0.02, "ratio of means, not mean of ratios"


def test_gs_seed_zero_rows_carry_eval_room_and_spec():
    seeds, econ = _flagship()
    for tag, spec in (("g28", "var-gs_bx-g28-v4"), ("bx7", "var-gs_bx-bx7-v5")):
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


def test_the_canonical_table_is_every_economy_and_only_on_protocol_rows():
    """There is one table and no filter on it.

    This checked that `--flagship` narrowed only the printout, because the CSV held rows the
    printout had to hide: probes at N=60 to 500, T=80 to 200 and windows 20/36/50, plus one
    economy at T=860. Hiding them is what let one be published beside rows it was not
    comparable to. The flag is gone and the guarantee is the other way round: every row the
    canonical directory produces is on the measurement protocol, and aggregate_seeds.py
    refuses to write the table otherwise.
    """
    import subprocess, tempfile
    sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
    import protocol
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, os.path.join(ROOT, "variants", "aggregate_seeds.py"),
                        "--out", td], check=True, capture_output=True)
        e = pd.read_csv(os.path.join(td, "economy_table.csv"))
    assert len(e) >= 12, f"economy_table.csv has {len(e)} rows; the economies are missing"
    off = e[(e["N"] != protocol.N) | (e["T"] != protocol.T) | (e["window"] != protocol.WINDOW)]
    assert off.empty, ("aggregate_seeds.py wrote off-protocol rows:\n"
                       + off[["model", "tag", "N", "T", "window"]].to_string())


def test_room_eval_is_present_exactly_where_the_oracle_computed_it():
    """room_eval must come from the oracle, never be inferred.

    It requires the per-basis conditional-SR series restricted to the evaluation months, which
    only run_oracle.py --eval_window produces; the series itself is never saved, so nothing
    downstream can reconstruct it. Until 2026-09-10 the vyx and g0235 oracles predated the flag
    and this asserted their room_eval was absent; the twenty re-runs supplied it, reproducing
    every all-month field bit-for-bit (WORKING.md §49). The durable invariant is the
    correspondence, not which side of it a given economy is on.
    """
    seeds, _ = _flagship()
    bad = []
    for _, r in seeds.iterrows():
        f = os.path.join(RESULTS, f"{r['model']}_oracle_{r['tag']}_s{int(r['seed']):03d}.json")
        computed = json.load(open(f)).get("eval_window") is not None
        have = not pd.isna(r["room_eval"])
        if computed != have:
            bad.append(f"{r['model']}/{r['tag']} s{int(r['seed']):03d}: oracle computed={computed}, table has={have}")
    assert not bad, "room_eval does not match what the oracle produced:\n  " + "\n  ".join(bad)


def test_all_four_flagship_economies_now_carry_eval_window_room():
    """The commensurable room exists for every current economy, so a gap/room ratio no longer
    has to be quoted against a different month sample from the gap."""
    _, econ = _flagship()
    missing = [f"{r['model']}/{r['tag']}" for _, r in econ.iterrows()
               if r["window"] == 360 and r["room_eval_n"] != r["room_all_n"]]
    assert not missing, f"flagship economies without eval-window room on every seed: {missing}"


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
