"""A spec that cannot be verified must not silently stamp a run.

`run_oracle.py --spec X` records X's spec_id in the summary. Two paths let it record
a spec_id the run did not earn:

  1. verify_against_spec() returns None (not False) when the spec declares no
     expected_solves. run_oracle.py guarded with `if _ok is False`, and
     `None is False` is False -- so an UNVERIFIABLE spec passed as though verified.

  2. The superseded check sat AFTER the no-expected_solves early return, so it was
     unreachable for exactly the specs most likely to be superseded: the v1s, which
     predate expected_solves. Same ordering bug as retired-before-superseded in
     solfiles.cmd_check.

Concretely: `--spec var-kp_vy-vyx-v1` today builds the v2 economy (the lambda
regime-label fix was made by editing parameters_kp14.py in place, not behind a
`method` switch) and would stamp the result var-kp_vy-vyx-v1.

Run with: python -m pytest tests/ -k refusal   (or: python tests/test_spec_refusal.py)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "experiments", "specs")
sys.path.insert(0, os.path.join(ROOT, "variants"))
sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
from common import runstamp  # noqa: E402


def specs():
    for fn in sorted(os.listdir(SPECS)):
        if fn.endswith(".json"):
            with open(os.path.join(SPECS, fn)) as fh:
                yield fn[:-5], json.load(fh)


def superseded(d):
    return bool(d.get("lineage", {}).get("superseded_by"))


def test_superseded_specs_are_refused_not_noted():
    """A superseded spec must return ok=False, whatever its expected_solves says."""
    bad = []
    for sid, d in specs():
        if not superseded(d):
            continue
        ok, lines = runstamp.verify_against_spec(sid, [])
        if ok is not False:
            bad.append(f"{sid}: verify_against_spec returned {ok!r}, expected False")
    assert not bad, ("superseded specs that do not refuse:\n  " + "\n  ".join(bad))


def test_superseded_specs_say_so_in_their_lines():
    """The refusal must name the superseding spec, so the operator knows where to go."""
    bad = []
    for sid, d in specs():
        if not superseded(d):
            continue
        _ok, lines = runstamp.verify_against_spec(sid, [])
        target = d["lineage"]["superseded_by"]
        if not any(target in ln for ln in lines):
            bad.append(f"{sid}: no line mentions {target}; got {lines!r}")
    assert not bad, ("superseded specs whose refusal hides the successor:\n  "
                     + "\n  ".join(bad))


def test_unverifiable_is_not_the_same_value_as_verified():
    """None must never compare equal to a pass under the caller's guard.

    run_oracle.py's guard is `if _ok is False`. This asserts the contract that makes
    that guard sound: any non-True return must be falsy, so a caller that writes
    `if not _ok` and one that writes `if _ok is False` agree.
    """
    for sid, d in specs():
        ok, _lines = runstamp.verify_against_spec(sid, [])
        assert ok is not None or not superseded(d), (
            f"{sid} is superseded yet returned None -- ambiguous to the caller")
        if ok is not True:
            assert not ok, f"{sid} returned a truthy non-True {ok!r}"


def test_run_oracle_guard_rejects_every_non_true():
    """The abort in run_oracle.py must not be spelled `is False`.

    `is False` lets None through. The guard has to reject anything that is not
    an explicit pass.
    """
    with open(os.path.join(ROOT, "variants", "run_oracle.py")) as fh:
        src = fh.read()
    assert "if _ok is False:" not in src, (
        "run_oracle.py guards the spec abort with `if _ok is False`, which lets the "
        "unverifiable None through and stamps the summary with an unearned spec_id")
    assert "_ok is not True" in src or "if not _ok" in src, (
        "run_oracle.py must reject every non-True result of verify_against_spec")


def test_live_specs_still_verify_normally():
    """The refusal must not break the specs actually in use."""
    for sid, d in specs():
        if superseded(d) or not d.get("expected_solves"):
            continue
        ok, lines = runstamp.verify_against_spec(sid, [])
        # No solves passed, so it must MISMATCH -- but it must reach the comparison,
        # not short-circuit on supersession.
        assert ok is False, f"{sid} should have compared and failed, got {ok!r}"
        assert any("MISMATCH" in ln for ln in lines), (
            f"{sid} did not reach the stage comparison: {lines!r}")


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
