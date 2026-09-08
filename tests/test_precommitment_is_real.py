"""A spec that CLAIMS precommitment must actually be one, by git dates.

WHY: docs/refactor/DECISION-provenance-layers.md defers the "delete the precommitment
layer?" decision to a count of `extra.spec_check` across sidecars. ASU pointed out the
hole on 2026-09-08: a wall of `verified` is only evidence if the spec was written BEFORE
the solve. If the id was read off an existing manifest and pasted in, `verified` can
only confirm that the id you copied is the id that is there -- vacuous, and it would be
counted as proof the machinery works.

That is not hypothetical. Measured the same day, THREE pinned ids are retrofitted, and
they are the two economies actually in use:

    var-kp_vy-vyx-v2    G, integ   spec pinned 09:27:14, manifest tracked 08:38:19
    var-bgn_gam-g0235-v2  jstar    spec pinned 09:27:14, manifest tracked 08:38:19
    var-gs_bx-bx7-v3    all five   spec pinned 10:03:22, manifest tracked 18:09:05  OK

Neither retrofitted spec CLAIMS precommitment, so nothing is dishonest -- but the
distinction has to be mechanical, not remembered, or the Phase 3 count is worthless.

This is the same failure shape as test_registry_ids_are_still_reachable on 2026-09-07,
which passed vacuously because its filter excluded exactly what it was looking for.

Run with: python -m pytest tests/ -k precommitment
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "experiments", "specs")
SOLF = os.path.join(ROOT, "experiments", "registry")

CLAIM_MARKERS = ("PRECOMMITMENT", "WITHOUT solving", "computed from the current "
                 "parameters and code WITHOUT solving")


def _git(args):
    r = subprocess.run(["git"] + args, cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def spec_pinned_at(spec_file, solve_id):
    """When the spec first contained this id (author date of the earliest commit)."""
    out = _git(["log", "-S", solve_id, "--format=%aI", "--", spec_file])
    return out.splitlines()[-1] if out else None


def manifest_tracked_at(solve_id):
    """--follow, because the registry directory gets renamed.

    Without it, `experiments/solfiles` -> `experiments/registry` (2026-09-08) hid every
    manifest's original add, and all three known retrofits silently reclassified as
    PRECOMMITTED -- precisely the vacuous-evidence failure this module exists to
    prevent. test_the_known_retrofits_stay_labelled caught it on the rename's first run.
    """
    out = _git(["log", "--follow", "--diff-filter=A", "--format=%aI", "--",
                os.path.join("experiments", "registry", solve_id + ".json")])
    return out.splitlines()[-1] if out else None


def classify():
    """{spec_id: {stage: 'precommitted' | 'retrofitted' | 'unknown'}}"""
    out = {}
    for fn in sorted(os.listdir(SPECS)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(SPECS, fn)))
        stages = {}
        for stage, sid in (d.get("expected_solves") or {}).items():
            sp = spec_pinned_at(os.path.join("experiments", "specs", fn), sid)
            mp = manifest_tracked_at(sid)
            if sp is None:
                stages[stage] = "unknown"
            elif mp is None or sp < mp:
                stages[stage] = "precommitted"
            else:
                stages[stage] = "retrofitted"
        if stages:
            out[fn[:-5]] = stages
    return out


def claims_precommitment(spec):
    notes = (spec.get("notes") or "").upper()
    return any(m.upper() in notes for m in CLAIM_MARKERS)


def test_a_spec_claiming_precommitment_actually_is_one():
    cls = classify()
    bad = []
    for sid, stages in cls.items():
        spec = json.load(open(os.path.join(SPECS, sid + ".json")))
        if not claims_precommitment(spec):
            continue
        retro = [st for st, v in stages.items() if v == "retrofitted"]
        if retro:
            bad.append(f"{sid}: claims precommitment but {retro} were pinned AFTER "
                       f"their manifests were tracked")
    assert not bad, ("specs whose precommitment claim is not supported by git dates:\n  "
                     + "\n  ".join(bad))


def test_the_classification_is_computable_for_every_pinned_id():
    """If this cannot be computed, the Phase 3 count cannot be read."""
    cls = classify()
    assert cls, "no spec pins any solve_id -- the classifier found nothing to check"
    unknown = {s: [k for k, v in st.items() if v == "unknown"]
               for s, st in cls.items()}
    unknown = {s: v for s, v in unknown.items() if v}
    assert not unknown, (
        "pinned ids with no git history, so precommitment cannot be established: "
        f"{unknown}")


def test_the_known_retrofits_stay_labelled():
    """Guards the labelling itself: these three are retrofitted as a matter of record.

    If a future change makes them read as precommitted, the classifier has broken and
    the Phase 3 count would silently start counting vacuous passes as evidence.
    """
    cls = classify()
    for sid, stage in (("var-kp_vy-vyx-v2", "G"),
                       ("var-kp_vy-vyx-v2", "integ"),
                       ("var-bgn_gam-g0235-v2", "jstar")):
        if sid in cls and stage in cls[sid]:
            assert cls[sid][stage] == "retrofitted", (
                f"{sid}/{stage} was retrofitted on 2026-09-07 (spec pinned 09:27:14, "
                f"manifest tracked 08:38:19) but now classifies as "
                f"{cls[sid][stage]!r} -- the classifier has broken")


if __name__ == "__main__":
    import traceback
    print("classification:")
    for s, st in sorted(classify().items()):
        print(f"  {s:24s} " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))
    print()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
