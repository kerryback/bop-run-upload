"""Every live spec must be invocable by variants/precommit_spec.py.

WHY: the tool drives the real producer to get its ids, so it has to know each producer's
CLI. That knowledge is a small explicit table (`_argv_for`), and the thing that rots is
the spec format drifting away from it. It already has: `vyg25` and `kpbase` declare ONE
stage named "vy_tables" with `args {"prefix": ...}`, while `vym3` declares TWO stages
named "G" and "integ" with `args {"KP_VY_PREFIX": ...}`. Both describe the same single
run of build_vy_tables.py. The first shape was the one the tool was written against and
the second raised KeyError on the first real spec it was pointed at.

So this asserts the cheap half -- that every live spec can be turned into a command --
without running anything. The expensive half, that the ids come back equal to the pins,
is what the tool prints when you use it, and is checked for real at build time by
`tests/test_specs_match_shell.py` and the manifests.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "experiments", "specs")
sys.path.insert(0, os.path.join(ROOT, "variants"))

import precommit_spec  # noqa: E402


def _live_specs():
    for fn in sorted(os.listdir(SPECS)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(SPECS, fn)) as fh:
            d = json.load(fh)
        lin = d.get("lineage") or {}
        if lin.get("superseded_by") or lin.get("retired"):
            continue
        yield d


def test_every_live_spec_stage_can_be_turned_into_a_command():
    bad = []
    for spec in _live_specs():
        for stage in (spec.get("solve") or {}).get("stages") or []:
            try:
                argv = precommit_spec._argv_for(stage["producer"], stage, spec)
            except SystemExit as e:
                bad.append(f"{spec['spec_id']} / {stage['name']}: {e}")
                continue
            except Exception as e:
                bad.append(f"{spec['spec_id']} / {stage['name']}: "
                           f"{type(e).__name__}: {e}")
                continue
            if not all(isinstance(a, str) for a in argv):
                bad.append(f"{spec['spec_id']} / {stage['name']}: non-str argv {argv!r}")
    assert not bad, ("precommit_spec cannot invoke these stages:\n  " + "\n  ".join(bad))


def test_kp_prefix_is_found_in_either_spec_shape():
    """The two live KP shapes must both resolve, and to the tag the spec runs under."""
    bad = []
    for spec in _live_specs():
        if spec["model"] != "kp_vy":
            continue
        want = (spec.get("env") or {}).get("KP_VY_PREFIX")
        for stage in (spec.get("solve") or {}).get("stages") or []:
            got = precommit_spec._argv_for(stage["producer"], stage, spec)[0]
            if want and got != want:
                bad.append(f"{spec['spec_id']} / {stage['name']}: prefix {got!r} "
                           f"but the spec runs under KP_VY_PREFIX={want!r}")
    assert not bad, "KP prefix resolution disagrees with the spec:\n  " + "\n  ".join(bad)


def test_chained_stages_are_declared_for_the_producer_that_has_one():
    """build_vy_tables.py is the only chained producer; if that changes, say so here.

    `integ`'s solve_id hashes the G tables' raw bytes
    (`build_vy_tables.py`, inputs=solstamp.artifact_digests(G_ARTIFACTS)), which is why
    it cannot be precommitted without building G first. Nothing else in the repo chains,
    and the tool's --build-chained warning is keyed off this table.
    """
    assert "build_vy_tables.py" in precommit_spec.CHAINED
    entry = precommit_spec.CHAINED["build_vy_tables.py"]
    assert entry["free"] == {"G"} and entry["chained"] == {"integ"}

    producers = {os.path.basename(s["producer"])
                 for spec in _live_specs()
                 for s in (spec.get("solve") or {}).get("stages") or []}
    unknown = producers - {"build_vy_tables.py", "rebuild_jstar_gam.py",
                           "gs_solve_reg.py", "gs_solve_gam.py"}
    assert not unknown, (f"live specs name producers precommit_spec has never been "
                         f"taught: {sorted(unknown)}")
