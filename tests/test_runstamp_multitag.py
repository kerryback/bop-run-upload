"""A multi-solve economy must be looked up as ONE economy.

gs_bx bx7 is five exposure types, five solution.npz, five manifests each under its own
soldir tag. runstamp.live_solves took a single tag until 2026-09-09, so for bx7
run_is_current compared the five ids in a run record against the one id of whichever tag
it was handed, and reported STALE forever: every resubmission of the seed array would
have re-run every finished seed. Verified before the fix:

    live_solves("gs_bx", "sol_reg") = ['63fa7ebbc2db49ea']      # one
    a bx7 run record carries five ids                             => never equal

Run with: python -m pytest tests/ -k multitag
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))
from common import runstamp  # noqa: E402

BX7 = "var-gs_bx-bx7-v3"


def _bx7():
    spec = runstamp.load_spec(BX7)
    return list(spec["expected_solves"]), sorted(spec["expected_solves"].values())


def _run_record(ids):
    fh = tempfile.NamedTemporaryFile("w", suffix="_run.json", delete=False)
    json.dump({"solves": [{"stage": f"s{i}", "solve_id": sid} for i, sid in enumerate(ids)]}, fh)
    fh.close()
    return fh.name


def test_solve_tags_parses_strings_and_lists():
    assert runstamp.solve_tags("vyx") == ["vyx"]
    assert runstamp.solve_tags("sol_reg, sol_b25c,,") == ["sol_reg", "sol_b25c"]
    assert runstamp.solve_tags(["a", " b "]) == ["a", "b"]


def test_one_tag_is_unchanged():
    assert runstamp.live_solves("bgn_gam", "Jstar_g0235") == ["be222462dd017b2c"]
    assert runstamp.live_solves("gs_bx", "sol_g28") == ["8b584c38614695ac"]


def test_five_tags_resolve_to_the_five_pinned_solves():
    tags, ids = _bx7()
    assert runstamp.live_solves("gs_bx", ",".join(tags)) == ids
    assert runstamp.live_solves("gs_bx", tags) == ids


def test_a_bx7_run_record_is_current_under_all_five_tags():
    tags, ids = _bx7()
    rec = _run_record(ids)
    try:
        ok, why = runstamp.run_is_current(rec, "gs_bx", ",".join(tags))
        assert ok is True, why
    finally:
        os.remove(rec)


def test_the_pre_fix_failure_is_pinned():
    """One tag against a five-id record: STALE, and the message says why."""
    tags, ids = _bx7()
    rec = _run_record(ids)
    try:
        ok, why = runstamp.run_is_current(rec, "gs_bx", tags[0])
        assert ok is False and "built from" in why
        ok, why = runstamp.run_is_current(rec, "gs_bx", ",".join(tags[:4]))
        assert ok is False
    finally:
        os.remove(rec)


def test_a_record_missing_one_solve_is_stale():
    tags, ids = _bx7()
    rec = _run_record(ids[:4])
    try:
        assert runstamp.run_is_current(rec, "gs_bx", ",".join(tags))[0] is False
    finally:
        os.remove(rec)


def test_the_cli_takes_the_comma_list_the_seed_array_passes():
    tags, ids = _bx7()
    out = subprocess.run([sys.executable, os.path.join(ROOT, "variants", "common", "runstamp.py"),
                          "current", "--model", "gs_bx", "--tag", ",".join(tags)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert sorted(out.stdout.split()) == ids


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
