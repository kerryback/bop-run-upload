"""A solve_id must be reproducible ACROSS PROCESSES AND MACHINES.

Two independent defects broke that, both found on 2026-09-07 while wiring the
seeded run arrays. Either one alone makes the registry useless: a
content-addressed cache whose address moves never hits, so the expensive solve
repeats forever, and a result computed on Sol cannot be traced to the manifest
committed from the laptop.

DEFECT 1 -- hash order (below).
DEFECT 2 -- float drift: derived parameters are computed by LAPACK and libm at
  import time and differ in the last bits between library versions. Measured on
  ONE laptop, identical parameters and source:

      numpy 2.4.2   A_0 = 3.851851173257416    G solve_id = b5d95b8eb1afc6d1
      numpy 2.4.6   A_0 = 3.8518511732574208   G solve_id = 3500d25aeeb64809

  10 of 66 kp_vy parameters differed, all derived, up to 45 ULPs. Fixed by
  quantising the HASH INPUT to solstamp.HASH_SIG_DIGITS while still recording
  exact values. An intermediate attempt cleared 10 mantissa bits; with ~200
  array elements and 45 ULPs of drift that still split every time, which is why
  the quantum has to sit several orders above the drift rather than just above.

DEFECT 1 in full.

History (2026-09-07). Commit b85629f added override-readback to
variants/kp_vy/parameters_kp14.py, which left two `set` objects in the module
namespace (`_pre_override_names`, 45 entries, and `_NORMALISED`). solstamp's
`_canon` had no set branch, so a set fell through to `{'__repr__': repr(value)}`
-- and Python randomises str hashing per interpreter, so a set's repr ordering
changes from run to run. Measured, identical parameters and identical code:

    PYTHONHASHSEED=0      G solve_id = fa54addb83e5bff2
    PYTHONHASHSEED=1      G solve_id = a8b4cdc0fb2cab7f
    PYTHONHASHSEED=2      G solve_id = da8081b77e6d5045
    PYTHONHASHSEED=12345  G solve_id = 6c7e520d50348425

This is worse than a wrong id. A content-addressed cache whose address moves
never hits: `build_vy_tables.py` would have re-run the multi-hour integ stage on
every invocation, and the manifests recorded under the old ids
(b0260fa9ca745db8, 8e4b5e820ad371da) were unreachable the moment the process
that wrote them exited. Nothing in the suite noticed, because every existing
solstamp test compares ids computed inside ONE interpreter, where hashing is
fixed. Checking reproducibility therefore REQUIRES subprocesses.

bgn_gam was unaffected (b80c6e516c132e13 under every seed) -- its namespace
happens to hold no unordered container. That is luck, not design, which is why
the guard below is generic rather than a fix pinned to the two names.

Run: python tests/test_solve_id_reproducible.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))
from common import solstamp  # noqa: E402

SEEDS = ["0", "1", "2", "12345"]


def _run(code, cwd, env_extra, seed):
    env = dict(os.environ, PYTHONHASHSEED=seed, **env_extra)
    out = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, f"probe failed under PYTHONHASHSEED={seed}:\n{out.stderr}"
    return out.stdout.strip()


def _ids_across_seeds(code, cwd, env_extra):
    return {s: _run(code, cwd, env_extra, s) for s in SEEDS}


# ------------------------------------------------------------ unit level ----

def test_canon_of_a_set_does_not_depend_on_hash_order():
    """The direct cause. A set must canonicalise to a sorted form."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from common import solstamp\n"
        "names = {'alpha','delta','gamma_v','kappa_y','rho','sigma_eps','theta_u',"
        "'mu_H','mu_L','lambda_H','lambda_L','type_bv','type_share','bv_comp'}\n"
        "print(solstamp.snapshot({'names': names}, []).solve_id)\n"
        % os.path.join(ROOT, "variants")
    )
    ids = _ids_across_seeds(code, ROOT, {})
    assert len(set(ids.values())) == 1, (
        "a set in the parameter namespace still moves the solve_id across "
        "processes: " + ", ".join(f"{s}->{v}" for s, v in ids.items()))


def test_canon_of_a_set_is_still_sensitive_to_membership():
    """Sorting must not flatten distinct sets onto one id."""
    a = solstamp.snapshot({"names": {"alpha", "delta"}}, [])
    b = solstamp.snapshot({"names": {"alpha", "gamma"}}, [])
    assert a.solve_id != b.solve_id, "different sets collide onto one solve_id"
    c = solstamp.snapshot({"names": {"delta", "alpha"}}, [])
    assert a.solve_id == c.solve_id, "the same set built in another order differs"


def test_frozenset_and_nested_sets_are_handled():
    a = solstamp.snapshot({"x": frozenset({"p", "q"})}, [])
    b = solstamp.snapshot({"x": frozenset({"q", "p"})}, [])
    assert a.solve_id == b.solve_id
    n1 = solstamp.snapshot({"x": {("a", 1), ("b", 2)}}, [])
    n2 = solstamp.snapshot({"x": {("b", 2), ("a", 1)}}, [])
    assert n1.solve_id == n2.solve_id, "a set of tuples is still order-sensitive"


def test_ulp_drift_in_derived_parameters_does_not_move_the_id():
    """The property, without needing a second interpreter installed.

    Perturb a kp_vy-shaped namespace by the drift actually measured between
    numpy 2.4.2 and 2.4.6 (up to 45 ULPs on derived quantities) and require the
    id to hold. This is the test that fails if HASH_SIG_DIGITS is ever tightened
    back toward bit-exactness.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 21)) * 3.0          # A0_ty-shaped, > 32 elements
    scal = {"A_0": 3.851851173257416, "rho": 0.0412, "C": 0.1584}
    small = np.array([1.0, 7.2830034370753225, 58.34012175987296])   # pm_tau-shaped

    def perturb(x, ulps):
        out = np.array(x, float, copy=True)
        flat = out.ravel()
        for i in range(flat.size):
            for _ in range(int(ulps)):
                flat[i] = np.nextafter(flat[i], np.inf if i % 2 else -np.inf)
        return out.reshape(np.shape(x))

    base = solstamp.snapshot(dict(A=A, small=small, **scal), [])
    drifted = solstamp.snapshot(
        dict(A=perturb(A, 45), small=perturb(small, 45),
             **{k: float(perturb(v, 45)) for k, v in scal.items()}), [])
    assert base.solve_id == drifted.solve_id, (
        "45 ULPs of library drift still moves the solve_id -- the laptop and "
        "Sol will not agree on an id and every manifest is orphaned by a numpy "
        "upgrade")

    real = solstamp.snapshot(dict(A=A * (1 + 1e-6), small=small, **scal), [])
    assert base.solve_id != real.solve_id, (
        "a 1e-6 relative parameter change is missed -- quantisation has gone "
        "too far and two different economies now share one id")


def _other_interpreter():
    """Any python on PATH that is not the one running the suite and has numpy."""
    here = os.path.realpath(sys.executable)
    seen = set()
    for name in ("python3", "python", "python3.11", "python3.12", "python3.13"):
        for d in os.environ.get("PATH", "").split(os.pathsep) + [
                "/usr/bin", "/usr/local/bin", "/Library/Frameworks/Python.framework/Versions/3.11/bin"]:
            cand = os.path.join(d, name)
            if not (os.path.isfile(cand) and os.access(cand, os.X_OK)):
                continue
            real = os.path.realpath(cand)
            if real == here or real in seen:
                continue
            seen.add(real)
            ok = subprocess.run([real, "-c", "import numpy"], capture_output=True)
            if ok.returncode == 0:
                return real
    return None


def test_a_second_interpreter_computes_the_same_id():
    """The end-to-end check. Skipped, loudly, when only one interpreter exists.

    This is the property that matters for the cluster: Sol runs a different
    numpy from the laptop, and a manifest committed from one must be reachable
    from the other.
    """
    other = _other_interpreter()
    if other is None:
        print("    (skipped: no second interpreter with numpy on this machine)")
        return
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import numpy as np\n"
        "from common import solstamp\n"
        "rng = np.random.default_rng(0)\n"
        "A = rng.normal(size=(3, 21)) * 3.0\n"
        "M = np.linalg.solve(np.diag(np.arange(1.0, 22.0)) + 0.01, np.ones(21))\n"
        "print(solstamp.snapshot({'A': A, 'M': M, 'p': float(M[3] ** 1.7)}, []).solve_id)\n"
        % os.path.join(ROOT, "variants")
    )
    mine = _run(code, ROOT, {}, "0")
    env = dict(os.environ, PYTHONHASHSEED="0")
    got = subprocess.run([other, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    theirs = got.stdout.strip()
    print(f"    ({os.path.basename(os.path.dirname(os.path.dirname(other)))} "
          f"agrees: {theirs})")
    assert mine == theirs, (
        f"{sys.executable} says {mine}, {other} says {theirs} -- "
        "a solve_id does not survive a change of interpreter")


# --------------------------------------------------- the real producers -----
# These are the tests that would have caught b85629f. They import the actual
# parameter modules under the actual overrides the run scripts export.

def test_kp_vy_solve_ids_reproduce_across_processes():
    code = (
        "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'..')\n"
        "import parameters_kp14 as P\n"
        "from common import solstamp\n"
        "g = solstamp.snapshot(P, ['kp14_fd_vy.py','parameters_kp14.py'],\n"
        "                      model='kp_vy', stage='G', extra={'prefix':'vyx'})\n"
        "i = solstamp.snapshot(P, ['integ_kp14.py','parameters_kp14.py'],\n"
        "                      model='kp_vy', stage='integ', extra={'prefix':'vyx'})\n"
        "print(g.solve_id, i.solve_id)\n"
    )
    env = {"KP_PARAM_OVERRIDES": '{"type_share":[0.34,0.33,0.33],'
                                 '"type_bv":[0.02,0.07,0.14],"gamma_v":1.8,"bv_comp":1.2}',
           "KP_VY_PREFIX": "vyx"}
    ids = _ids_across_seeds(code, os.path.join(ROOT, "variants", "kp_vy"), env)
    assert len(set(ids.values())) == 1, (
        "kp_vy solve_ids move across processes -- the cache can never hit: "
        + "; ".join(f"{s}->{v}" for s, v in ids.items()))


def test_bgn_gam_solve_id_reproduces_across_processes():
    code = (
        "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'..')\n"
        "import parameters as P\n"
        "from common import solstamp\n"
        "print(solstamp.snapshot(P, ['vasicek.py','parameters.py'], model='bgn_gam',\n"
        "                        env_params={'JSTAR_TOL': 3e-4},\n"
        "                        extra={'out': P.jstar_gam_file}).solve_id)\n"
    )
    env = {"BGN_PARAM_OVERRIDES": '{"gmult":[0.2,3.5],"jstar_gam_file":"Jstar_g0235.csv"}'}
    ids = _ids_across_seeds(code, os.path.join(ROOT, "variants", "bgn_gam"), env)
    assert len(set(ids.values())) == 1, (
        "bgn_gam solve_id moves across processes: "
        + "; ".join(f"{s}->{v}" for s, v in ids.items()))


def test_no_live_manifest_canonicalises_a_parameter_by_bare_repr():
    """`__repr__` is _canon's fall-through for types it does not understand.

    Any such type is a reproducibility hazard by construction -- nothing
    guarantees its repr is stable across processes, and the set case proved the
    hazard is real rather than theoretical. So the rule for LIVE manifests is
    absolute: no bare repr at all. A manifest that predates the fix is expected
    to violate it and must be marked `retired` (variants/solfiles.py retire),
    which is what keeps old experiments identifiable without pretending their
    ids can be recomputed.

    An earlier version of this test tried to recognise a set by its repr and
    excluded anything starting with `{'` -- which is exactly what a set of
    strings looks like. It passed with the bug present. Hence the blanket rule.
    """
    bad = []
    for man in solstamp.iter_manifests():
        if man.get("retired"):
            continue
        for section in ("params", "env_params"):
            for name, val in sorted((man.get(section) or {}).items()):
                if isinstance(val, dict) and "__repr__" in val:
                    bad.append(f"{man['solve_id']} ({man.get('model')}): "
                               f"{section}.{name} = {val['__repr__'][:60]}")
    assert not bad, (
        "live manifests canonicalising a parameter by bare repr, so their "
        "solve_id may not be recomputable:\n  " + "\n  ".join(bad)
        + "\nEither teach solstamp._canon that type, or retire the manifest.")


def test_retired_manifests_say_why():
    for man in solstamp.iter_manifests():
        r = man.get("retired")
        if r is None:
            continue
        assert (r.get("reason") or "").strip(), \
            f"{man['solve_id']} is retired with no reason recorded"


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
