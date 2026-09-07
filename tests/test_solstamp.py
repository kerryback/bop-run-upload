"""Content-addressed solve identity for the variant producers.

Every variant economy has an expensive solve stage, and all three producers used
to get staleness detection wrong: KP cached on a hand-written key that missed 7
parameters, BGN had no stamp at all, GS wrote a params array that omitted the very
parameters distinguishing one solve from another.

variants/common/solstamp.py replaces all three. The properties that matter:

  SENSITIVE   any parameter, environment setting or producer-source change that
              alters the output must change the solve_id.
  STABLE      cosmetic differences (output directory, filename prefix) must NOT
              change it, or identical solves stop being reusable.
  DURABLE     the manifest records enough to identify a solve after its artifacts
              are gone.

Run: python tests/test_solstamp.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))

from common import solstamp  # noqa: E402


class _NS(object):
    """Stand-in for a producer's parameter module."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _sources(tmp, text="print('producer')\n"):
    p = os.path.join(tmp, "producer.py")
    with open(p, "w") as f:
        f.write(text)
    return p


# ------------------------------------------------------------- sensitivity ---

def test_scalar_change_moves_the_id():
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        a = solstamp.snapshot(_NS(alpha=0.85, delta=0.1), [src])
        b = solstamp.snapshot(_NS(alpha=0.85, delta=0.2), [src])
        assert a.solve_id != b.solve_id


def test_float_precision_holds_above_the_hash_quantum():
    """Floats are hashed to HASH_SIG_DIGITS, so sensitivity has a stated floor.

    Until 2026-09-07 this asserted that 0.1 and 0.1+1e-16 are different solves.
    That bit-exactness was not free: derived parameters are computed by LAPACK
    and libm at import, they drift ~1e-14 between library versions, and the
    solve_id moved with them -- so the laptop and Sol could never agree on an id
    and a `conda update` orphaned every manifest. See solstamp.HASH_SIG_DIGITS.

    The contract is now explicit in both directions: differences above roughly
    1e-8 relative move the id; differences below it deliberately do not.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        base = solstamp.snapshot(_NS(x=0.1), [src])

        moved = solstamp.snapshot(_NS(x=0.1 * (1 + 1e-6)), [src])
        assert base.solve_id != moved.solve_id, (
            "a 1e-6 relative parameter change must still move the solve_id")

        drift = solstamp.snapshot(_NS(x=0.1 + 1e-16), [src])
        assert base.solve_id == drift.solve_id, (
            "sub-ULP library drift must NOT move the solve_id (that is the "
            "whole point of quantising the hash input)")

        assert base.params["x"] == repr(0.1), (
            "the manifest must still RECORD the exact value; only the hash is "
            "quantised")


def test_list_change_moves_the_id():
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        a = solstamp.snapshot(_NS(type_bv=[0.02, 0.07, 0.14]), [src])
        b = solstamp.snapshot(_NS(type_bv=[0.02, 0.07, 0.20]), [src])
        assert a.solve_id != b.solve_id


def test_producer_source_change_moves_the_id():
    """The half the old KP key missed entirely: editing the solver."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp, "print('v1')\n")
        a = solstamp.snapshot(_NS(alpha=0.85), [src])
        with open(src, "w") as f:
            f.write("print('v2')\n")
        b = solstamp.snapshot(_NS(alpha=0.85), [src])
        assert a.solve_id != b.solve_id


def test_env_params_are_hashed():
    """BGN's JSTAR_TOL changes the table but lives in the environment."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        a = solstamp.snapshot(_NS(g=1.0), [src], env_params={"JSTAR_TOL": 3e-4})
        b = solstamp.snapshot(_NS(g=1.0), [src], env_params={"JSTAR_TOL": 1e-4})
        assert a.solve_id != b.solve_id


def test_large_array_change_moves_the_id():
    """Arrays over 32 elements are hashed by bytes, not inlined -- still sensitive."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        x = np.linspace(0, 1, 200)
        y = x.copy()
        y[137] *= 1 + 1e-6                     # one element, above the hash quantum
        a = solstamp.snapshot(_NS(grid=x), [src])
        b = solstamp.snapshot(_NS(grid=y), [src])
        assert a.solve_id != b.solve_id, "a real change in one array element was missed"

        z = x.copy()
        z[137] = np.nextafter(z[137], 1.0)     # one ULP: library drift, not a new economy
        c = solstamp.snapshot(_NS(grid=z), [src])
        assert a.solve_id == c.solve_id, (
            "one ULP in one element still moves the id -- the array path is not "
            "quantised (see solstamp.HASH_SIG_DIGITS)")


def test_large_arrays_do_not_bloat_the_manifest():
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        snap = solstamp.snapshot(_NS(grid=np.linspace(0, 1, 100_000)), [src])
        blob = json.dumps(snap.as_dict())
        assert len(blob) < 4000, f"manifest payload is {len(blob)} B"
        assert "__array_sha256__" in blob


# ----------------------------------------------------------------- stability --

def test_extra_labels_do_not_move_the_id():
    """Same solve written to a different directory is the SAME solve."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        a = solstamp.snapshot(_NS(g=1.0), [src], extra={"outdir": "sol_reg"})
        b = solstamp.snapshot(_NS(g=1.0), [src], extra={"outdir": "sol_elsewhere"})
        assert a.solve_id == b.solve_id


def test_skip_excludes_path_plumbing():
    """GS keeps outdir/HERE in globals(); they must not make the id machine-dependent."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        a = solstamp.snapshot(_NS(g=1.0, outdir="/a/sol", HERE="/a"), [src],
                              skip=("outdir", "HERE"))
        b = solstamp.snapshot(_NS(g=1.0, outdir="/b/sol", HERE="/b"), [src],
                              skip=("outdir", "HERE"))
        assert a.solve_id == b.solve_id


def test_id_is_deterministic_across_processes():
    """Not a counter and not insertion-ordered: a re-run reproduces the id."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        a = solstamp.snapshot(_NS(a=1, b=2.5, c=[1, 2]), [src])
        b = solstamp.snapshot(_NS(c=[1, 2], b=2.5, a=1), [src])
        assert a.solve_id == b.solve_id


def test_functions_and_modules_are_not_parameters():
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        a = solstamp.snapshot(_NS(g=1.0), [src])
        b = solstamp.snapshot(_NS(g=1.0, helper=lambda x: x, mod=os), [src])
        assert a.solve_id == b.solve_id


# ------------------------------------------------------------------ registry --

def _isolated_registry(fn):
    """Run fn with REGISTRY_DIR pointed at a temp dir."""
    real = solstamp.REGISTRY_DIR
    tmp = tempfile.mkdtemp()
    solstamp.REGISTRY_DIR = tmp
    try:
        return fn(tmp)
    finally:
        solstamp.REGISTRY_DIR = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_record_and_lookup_round_trip():
    def body(reg):
        with tempfile.TemporaryDirectory() as tmp:
            src = _sources(tmp)
            art = os.path.join(tmp, "solution.bin")
            with open(art, "wb") as f:
                f.write(b"x" * 1024)
            snap = solstamp.snapshot(_NS(g=1.0), [src], model="gs_bx")
            solstamp.record(snap, [art], tag="sol_reg")
            got = solstamp.lookup(snap.solve_id)
            assert got is not None
            assert got["model"] == "gs_bx"
            assert got["tags"] == ["sol_reg"]
            assert got["total_bytes"] == 1024
            assert solstamp.artifacts_ok(got)
    _isolated_registry(body)


def test_artifact_tampering_is_detected():
    def body(reg):
        with tempfile.TemporaryDirectory() as tmp:
            src = _sources(tmp)
            art = os.path.join(tmp, "solution.bin")
            with open(art, "wb") as f:
                f.write(b"original")
            snap = solstamp.snapshot(_NS(g=1.0), [src])
            solstamp.record(snap, [art])
            with open(art, "wb") as f:
                f.write(b"tampered")
            problems = solstamp.artifact_problems(solstamp.lookup(snap.solve_id))
            assert problems and "content changed" in problems[0]
    _isolated_registry(body)


def test_missing_artifact_is_detected_and_still_identifiable():
    """The durability property: the manifest outlives the artifact."""
    def body(reg):
        with tempfile.TemporaryDirectory() as tmp:
            src = _sources(tmp)
            art = os.path.join(tmp, "solution.bin")
            with open(art, "wb") as f:
                f.write(b"y" * 64)
            snap = solstamp.snapshot(_NS(gamma_v=1.8), [src], model="kp_vy")
            solstamp.record(snap, [art], tag="vyx")
            os.remove(art)
            man = solstamp.lookup(snap.solve_id)
            assert man is not None, "manifest must survive artifact deletion"
            assert man["params"]["gamma_v"] == repr(1.8)
            problems = solstamp.artifact_problems(man)
            assert problems and "missing" in problems[0]
    _isolated_registry(body)


def test_large_artifact_is_flagged_not_committable():
    def body(reg):
        with tempfile.TemporaryDirectory() as tmp:
            src = _sources(tmp)
            art = os.path.join(tmp, "big.bin")
            with open(art, "wb") as f:
                f.seek(solstamp.SMALL_ARTIFACT_BYTES + 1)
                f.write(b"\0")
            snap = solstamp.snapshot(_NS(g=1.0), [src], model="gs_bx")
            man = solstamp.record(snap, [art])
            assert man["committable"] is False
    _isolated_registry(body)


def test_tags_and_specs_accumulate_across_records():
    """One solve reused by several experiments keeps every reference."""
    def body(reg):
        with tempfile.TemporaryDirectory() as tmp:
            src = _sources(tmp)
            art = os.path.join(tmp, "a.bin")
            with open(art, "wb") as f:
                f.write(b"z")
            snap = solstamp.snapshot(_NS(g=1.0), [src])
            solstamp.record(snap, [art], tag="vyx", spec_id="var-kp_vy-vyx-v1")
            man = solstamp.record(snap, [art], tag="vyz", spec_id="var-kp_vy-vyz-v1")
            assert man["tags"] == ["vyx", "vyz"]
            assert man["spec_ids"] == ["var-kp_vy-vyx-v1", "var-kp_vy-vyz-v1"]
    _isolated_registry(body)


def test_diff_params_names_what_changed():
    a = {"mu_H": "0.075", "delta": "0.1"}
    b = {"mu_H": "0.075", "delta": "0.2", "new": "1"}
    d = dict((k, (x, y)) for k, x, y in solstamp.diff_params(a, b))
    assert d["delta"] == ("0.1", "0.2")
    assert d["new"][0] == "<absent>"
    assert "mu_H" not in d


def test_ensure_raises_on_an_unrecorded_solve():
    def body(reg):
        with tempfile.TemporaryDirectory() as tmp:
            src = _sources(tmp)
            art = os.path.join(tmp, "a.bin")
            with open(art, "wb") as f:
                f.write(b"q")
            snap = solstamp.snapshot(_NS(g=1.0), [src])
            try:
                solstamp.ensure(snap, [art], "test", mode="error")
            except solstamp.SolveStaleError:
                return
            raise AssertionError("ensure() did not raise on an unrecorded solve")
    _isolated_registry(body)


# ------------------------------------------------------- staged solve ids -----

def test_downstream_stage_does_not_invalidate_upstream():
    """The reason staging exists.

    KP's G stage costs ~45 min per type; the integ stage is 63 short jobs. Editing
    the cheap producer must NOT force a re-solve of the expensive one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        up = os.path.join(tmp, "expensive.py")
        dn = os.path.join(tmp, "cheap.py")
        for path, txt in ((up, "v1\n"), (dn, "v1\n")):
            open(path, "w").write(txt)
        art = os.path.join(tmp, "G0.csv")
        open(art, "w").write("r,J\n0,1\n")

        def ids():
            g = solstamp.snapshot(_NS(a=1), [up], stage="G")
            i = solstamp.snapshot(_NS(a=1), [dn], stage="integ",
                                  inputs=solstamp.artifact_digests([art]))
            return g.solve_id, i.solve_id

        g0, i0 = ids()
        open(dn, "w").write("v2\n")                     # touch the CHEAP producer
        g1, i1 = ids()
        assert g1 == g0, "editing the cheap producer invalidated the expensive stage"
        assert i1 != i0, "editing the cheap producer did not move its own stage"


def test_upstream_artifact_change_propagates_downstream():
    """utils/solfile_stamp.py's 'mode 3: upstream moved, downstream did not'."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        art = os.path.join(tmp, "G0.csv")
        open(art, "w").write("r,J\n0,1\n")
        a = solstamp.snapshot(_NS(x=1), [src], stage="integ",
                              inputs=solstamp.artifact_digests([art]))
        open(art, "w").write("r,J\n0,2\n")
        b = solstamp.snapshot(_NS(x=1), [src], stage="integ",
                              inputs=solstamp.artifact_digests([art]))
        assert a.solve_id != b.solve_id


def test_stage_name_separates_otherwise_identical_solves():
    with tempfile.TemporaryDirectory() as tmp:
        src = _sources(tmp)
        a = solstamp.snapshot(_NS(x=1), [src], stage="G")
        b = solstamp.snapshot(_NS(x=1), [src], stage="integ")
        assert a.solve_id != b.solve_id


def test_kp_driver_excluded_from_stage_sources():
    """build_vy_tables.py only orchestrates; including it would recreate the
    coupling the split removes."""
    src = open(os.path.join(ROOT, "variants/kp_vy/build_vy_tables.py")).read()
    g_line = [l for l in src.splitlines() if l.startswith("G_SOURCES")][0]
    i_line = [l for l in src.splitlines() if l.startswith("I_SOURCES")][0]
    for line in (g_line, i_line):
        assert "build_vy_tables.py" not in line, f"driver is back in a stage: {line}"
    assert "kp14_fd_vy.py" in g_line and "integ_kp14.py" in i_line


# ------------------------------------------------- the real producers wire up --

def test_all_three_producers_import_solstamp():
    """Regression guard: none of the three may go back to a hand-rolled key."""
    checks = {
        "variants/kp_vy/build_vy_tables.py": "solstamp",
        "variants/bgn_gam/rebuild_jstar_gam.py": "solstamp",
        "variants/gs_bx/gs_solve_reg.py": "solstamp",
    }
    for path, needle in checks.items():
        src = open(os.path.join(ROOT, path)).read()
        assert needle in src, f"{path} no longer uses solstamp"


def test_kp_no_longer_uses_the_hand_rolled_meta_key():
    src = open(os.path.join(ROOT, "variants/kp_vy/build_vy_tables.py")).read()
    assert 'json.load(open(meta))' not in src, "the old 5-key cache is back"
    assert '"rho": [list(map(float, r)) for r in rho_ty]' not in src


def test_gs_shell_guard_is_not_a_bare_existence_check():
    """Only EXECUTABLE lines count -- the removal is documented in a comment that
    quotes the old guard verbatim, and that quote must not trip this test."""
    src = open(os.path.join(ROOT, "variants/gs_bx/run_gs_bx7.sh")).read()
    live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "[ -f sol_reg/solution.npz ] ||" not in live, (
        "the parameter-blind existence guard is back"
    )
    assert "gs_solve_reg.py" in live, "the solve invocation went missing"


def test_gs_solution_records_its_identifying_parameters():
    """A solution.npz must be able to say which exposure type it is."""
    src = open(os.path.join(ROOT, "variants/gs_bx/gs_solve_reg.py")).read()
    for key in ("solve_id=", "gs_bx=gs_bx", "gs_ashift=gs_ashift", "gmreg=gmreg"):
        assert key in src, f"solution.npz no longer records {key}"


def test_kp_streams_solver_progress():
    """A multi-hour solve must not be indistinguishable from a hung one."""
    src = open(os.path.join(ROOT, "variants/kp_vy/build_vy_tables.py")).read()
    g_solve = src.split("def one(")[0]
    assert "stdout=subprocess.DEVNULL" not in g_solve, (
        "the G solve's progress output is being discarded again"
    )


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
