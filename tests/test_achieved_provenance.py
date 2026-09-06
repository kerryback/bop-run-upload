"""A manifest must record what a solve ACHIEVED, not only what it was asked for.

History (2026-09-05). GS21's sol_reg recorded `tol: 1e-06` in its manifest. The loop
actually enforces `tol * 20` = 2e-5 (undocumented), and the solve exited by cycle-averaging
at its 5600-sweep cap having reached 3.4e-5 -- then printed "converged" through the same
code path a tolerance exit uses. It was 1.7x over, so the artifact is probably fine. That
is what makes it dangerous: the identical path writes the identical manifest at 1000x over.

So `tol` is hashed into the solve_id while being, for this economy, decorative: a different
tol gives a different solve_id and identical numerics.

The fix is asymmetric on purpose. Achieved quality is RECORDED but NEVER HASHED -- hashing
an outcome would put two runs of the same code on the same parameters into different
registry slots. solstamp.record(achieved=...) does exactly that.

Also pinned here: the false-provenance bug. Under GS_SOLVE_FORCE=1 / BGN_JSTAR_FORCE=1 the
cache branch is falsified by the env var rather than by a missing manifest, so control fell
to an `else` written for the unrecorded case and announced "provenance is unrecorded" about
a solve that was recorded and matching -- a false provenance claim from the provenance
system. Both producers had it; both were written by the primary session.

Run: python tests/test_achieved_provenance.py
"""
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))
from common import solstamp  # noqa: E402

PRODUCERS = {
    "gs": os.path.join(ROOT, "variants", "gs_bx", "gs_solve_reg.py"),
    "bgn": os.path.join(ROOT, "variants", "bgn_gam", "rebuild_jstar_gam.py"),
    "kp": os.path.join(ROOT, "variants", "kp_vy", "build_vy_tables.py"),
}


def _live(path):
    return "\n".join(l for l in open(path).read().splitlines()
                     if not l.lstrip().startswith("#"))


# ------------------------------------------------------------- behaviour ----

def test_achieved_is_recorded_but_does_not_change_identity():
    """The whole asymmetry, in one test."""
    d = tempfile.mkdtemp()
    art = os.path.join(d, "a.bin")
    open(art, "wb").write(b"x" * 32)
    src = os.path.join(d, "prod.py")
    open(src, "w").write("TOL = 1e-6\n")

    snap = solstamp.snapshot({"tol": 1e-6}, [src], model="probe")
    m1 = solstamp.record(snap, [art], achieved={"exit": "tolerance", "resid": 1e-9})
    m2 = solstamp.record(snap, [art], achieved={"exit": "cycle_capped", "resid": 1e-2})

    assert m1["solve_id"] == m2["solve_id"], \
        "achieved quality changed the solve_id; it must not"
    assert m2["achieved"]["exit"] == "cycle_capped", "achieved was not recorded"
    assert m1["achieved"]["resid"] != m2["achieved"]["resid"]
    os.unlink(solstamp.manifest_path(snap.solve_id))


def test_achieved_is_absent_rather_than_faked_when_not_supplied():
    d = tempfile.mkdtemp()
    art = os.path.join(d, "a.bin")
    open(art, "wb").write(b"y" * 16)
    src = os.path.join(d, "prod2.py")
    open(src, "w").write("A = 1\n")
    snap = solstamp.snapshot({"a": 1}, [src], model="probe2")
    m = solstamp.record(snap, [art])
    assert "achieved" not in m, "an unsupplied achieved block must be omitted, not invented"
    os.unlink(solstamp.manifest_path(snap.solve_id))


def test_the_real_gs_manifest_records_that_it_was_capped():
    m = solstamp.lookup("a8ef7a2522eda19d")
    if m is None:
        return  # artifact-dependent; skip if the registry entry is gone
    a = m.get("achieved")
    assert a, "the GS solve's achieved block is missing"
    assert a["exit"] == "cycle_capped", a["exit"]
    assert a["qerr_rel"] > a["threshold_enforced"], \
        "a capped solve that beat its threshold would not need this field"
    assert a["threshold_enforced"] > a["tol_requested"], \
        "the tol*20 discrepancy should be visible in the manifest"


# ---------------------------------------------------------------- source ----

def test_every_producer_records_what_it_achieved():
    for name, path in PRODUCERS.items():
        assert "achieved=" in _live(path), f"{name}: solve outcome is not recorded"


def test_gs_distinguishes_capped_from_converged():
    live = _live(PRODUCERS["gs"])
    assert '"cycle_capped"' in live and '"tolerance"' in live, \
        "the two exit paths must be distinguishable in the manifest"


def test_no_producer_claims_unrecorded_provenance_for_a_recorded_solve():
    """The forced-rebuild path must not announce 'provenance is unrecorded'."""
    for name in ("gs", "bgn"):
        # Join implicit string concatenation: the GS message is split mid-phrase, so
        # both the line break AND the intervening `" f"` have to go. Matching the raw
        # text was a test bug, not a code defect -- twice.
        live = " ".join(_live(PRODUCERS[name]).split())
        live = re.sub(r'"\s*f?"', "", live)
        assert "is recorded and matches solve_id" in live, \
            f"{name}: the forced-rebuild branch still falls through to the unrecorded message"
        i_elif = live.index("is recorded and matches solve_id")
        i_else = live.index("provenance is unrecorded")
        assert i_elif < i_else, f"{name}: the matching case must be handled before the else"


def test_solfiles_surfaces_a_capped_solve():
    src = open(os.path.join(ROOT, "variants", "solfiles.py")).read()
    assert "CAPPED" in src, "check does not flag solves that missed their tolerance"
    assert "did NOT exit on its tolerance test" in src, "show does not flag the exit path"


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
