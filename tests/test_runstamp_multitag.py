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


def _run_record(ids, **over):
    """A run record as run_estimators.py writes one: the solves AND the protocol.

    The protocol half is not decoration. run_is_current checks it (see
    test_a_pre_protocol_run_is_not_current_even_when_its_solve_is), because a protocol
    change can invalidate every seed without moving a solve_id -- which is exactly what
    happened on 2026-09-15 and skipped seventy cluster tasks.
    """
    sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
    import protocol
    d = {"solves": [{"stage": f"s{i}", "solve_id": sid} for i, sid in enumerate(ids)],
         "kappas": list(protocol.KAPPAS), "window": protocol.WINDOW,
         "fair_linear": True, "winsor": protocol.WINSOR}
    d.update(over)
    fh = tempfile.NamedTemporaryFile("w", suffix="_run.json", delete=False)
    json.dump(d, fh)
    fh.close()
    return fh.name


def test_solve_tags_parses_strings_and_lists():
    assert runstamp.solve_tags("vyx") == ["vyx"]
    assert runstamp.solve_tags("sol_reg, sol_b25c,,") == ["sol_reg", "sol_b25c"]
    assert runstamp.solve_tags(["a", " b "]) == ["a", "b"]


def test_one_tag_is_unchanged():
    # The bgn_gam id moved on 2026-09-15: the protocol's burn-in change re-keyed every BGN
    # jstar solve without changing a byte of its table. be222462dd017b2c is the same economy
    # at burn-in 300 and is marked superseded_by this id, which is why exactly one comes back.
    assert runstamp.live_solves("bgn_gam", "Jstar_g0235") == ["2d462ae0f4463f03"]
    assert runstamp.live_solves("gs_bx", "sol_g28") == ["8b584c38614695ac"]


def test_a_superseded_manifest_is_not_live_but_is_still_pinnable():
    """Two manifests for the same STAGE under one tag would make every run look STALE.

    `superseded_by` is what keeps them apart from retired ids. It has to be a separate flag:
    test_expected_solves_are_live_manifests refuses a spec that pins a RETIRED id, and the
    superseded v1/v2 BGN specs pin these ids as the record of what produced their results.
    """
    sys.path.insert(0, os.path.join(ROOT, "variants"))
    from common import solstamp
    old = solstamp.lookup("be222462dd017b2c")
    assert old is not None, "the superseded manifest must still be in the registry"
    assert not old.get("retired"), "superseded is not retired: this id is still reachable"
    sb = old.get("superseded_by")
    assert sb and sb.get("solve_id") == "2d462ae0f4463f03", sb
    assert (sb.get("reason") or "").strip(), "a supersession must say why"
    # and the tag it used to own now resolves to exactly one live solve
    for tag in ("Jstar_bgnbase", "Jstar_g0235", "Jstar_g0235f",
                "Jstar_g0235s", "Jstar_g0235r", "Jstar_g0235d"):
        assert len(runstamp.live_solves("bgn_gam", tag)) == 1, (
            f"{tag} has {runstamp.live_solves('bgn_gam', tag)}; two ids for one stage make "
            f"run_is_current report every seed STALE")


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


def test_a_pre_protocol_run_is_not_current_even_when_its_solve_is():
    """The seed checkpoint must see a protocol change, not only a solve change.

    THE INCIDENT, 2026-09-15. run_is_current keyed only on the solve, on the premise that
    "re-solving the economy invalidates every seed at once". The protocol change broke that
    premise from a direction it could not see: it moved the ridge grid, the burn-in and the
    conditioning columns while leaving KP14's and GS21's solve ids untouched. Seventy seeds
    therefore looked current against pre-protocol results, and seventy cluster tasks exited
    in three seconds each with "already complete and current for the recorded solve".
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
    import protocol

    def rec(**over):
        d = {"solves": [{"stage": "jstar", "solve_id": "x"}],
             "kappas": list(protocol.KAPPAS), "window": protocol.WINDOW,
             "fair_linear": True, "winsor": protocol.WINSOR}
        d.update(over)
        fh = tempfile.NamedTemporaryFile("w", suffix="_run.json", delete=False)
        json.dump(d, fh); fh.close()
        return fh.name

    ok, why = runstamp.run_matches_protocol(rec())
    assert ok, why

    # the four ways a record can be off protocol, each named in the reason
    for over, needle in (({"kappas": [0.001, 0.01, 0.1, 1.0]}, "kappas"),
                         ({"window": 720}, "window"),
                         ({"fair_linear": None}, "fair_linear"),
                         ({"winsor": 0.05}, "winsor")):
        ok, why = runstamp.run_matches_protocol(rec(**over))
        assert not ok, f"{over} was accepted"
        assert needle in why, f"{over} -> {why!r} does not name {needle}"

    # and run_is_current folds it in, so the checkpoint cannot skip an off-protocol seed
    # whose solve happens to still be live
    live = runstamp.live_solves("gs_bx", "sol_g28")
    assert len(live) == 1
    stale = rec(solves=[{"stage": "sol_g28", "solve_id": live[0]}],
                kappas=[0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ok, why = runstamp.run_is_current(stale, "gs_bx", "sol_g28")
    assert not ok, "a pre-protocol gs_bx record with a live solve was called current"
    assert "off protocol" in why, why
