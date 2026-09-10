"""The run's PARAMETERS must match its spec, not just the solve ids it consumed.

This is the last link in params -> solve -> result, and it was open until 2026-09-09.
The failure it closes is not hypothetical; it was reproduced end to end:

    BGN_PARAM_OVERRIDES='{"gmult":[0.2,3.0],"jstar_gam_file":"Jstar_g0235.csv"}' \
    python variants/run_oracle.py --model bgn_gam --spec var-bgn_gam-g0235-v2 ...

The J* table on disk really is the one var-bgn_gam-g0235-v2 pins, so
`verify_against_spec` PASSED and the summary was written with spec_id
var-bgn_gam-g0235-v2 and spec_check "verified" -- while the simulator ran gmult
[0.2, 3.0] instead of the spec's [0.2, 3.5]. Solve-id checking structurally cannot
see this: the artifact is innocent, the parameters layered on top of it are not.

Run with: python -m pytest tests/ -k spec_env
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))
from common import runstamp  # noqa: E402

BGN = "var-bgn_gam-g0235-v2"
KP = "var-kp_vy-vyx-v2"
GS = "var-gs_bx-bx7-v3"


def _bgn_env(**over):
    spec = runstamp.load_spec(BGN)
    p = dict(spec["params"]); p.update(over)
    return {"BGN_PARAM_OVERRIDES": json.dumps(p)}


# ----------------------------------------------------------- the happy path ----

def test_matching_overrides_verify():
    ok, lines = runstamp.verify_env_against_spec(BGN, "BGN_PARAM_OVERRIDES", _bgn_env())
    assert ok is True, lines


def test_whitespace_and_key_order_do_not_refuse():
    """A correct run must not be refused for spelling its JSON differently."""
    env = {"BGN_PARAM_OVERRIDES":
           '{ "jstar_gam_file" : "Jstar_g0235.csv" ,  "gmult" : [0.2 , 3.5] }'}
    ok, lines = runstamp.verify_env_against_spec(BGN, "BGN_PARAM_OVERRIDES", env)
    assert ok is True, lines


# ------------------------------------------------------- the reproduced bug ----

def test_the_reproduced_gmult_mismatch_is_refused():
    ok, lines = runstamp.verify_env_against_spec(
        BGN, "BGN_PARAM_OVERRIDES", _bgn_env(gmult=[0.2, 3.0]))
    assert ok is False
    blob = "\n".join(lines)
    assert "MISMATCH" in blob
    assert "gmult" in blob, "the refusal must name the parameter that differs"


def test_unset_overrides_are_refused_not_treated_as_defaults():
    ok, _ = runstamp.verify_env_against_spec(BGN, "BGN_PARAM_OVERRIDES", {})
    assert ok is False


def test_extra_undeclared_parameter_is_refused():
    """Adding a knob the spec does not mention is still a different economy."""
    ok, _ = runstamp.verify_env_against_spec(
        BGN, "BGN_PARAM_OVERRIDES", _bgn_env(sigma_r=0.004))
    assert ok is False


def test_malformed_json_is_refused():
    ok, lines = runstamp.verify_env_against_spec(
        BGN, "BGN_PARAM_OVERRIDES", {"BGN_PARAM_OVERRIDES": "{not json"})
    assert ok is False
    assert "not valid JSON" in "\n".join(lines)


# ------------------------------------------------------ literal env entries ----

def test_kp_prefix_must_match():
    spec = runstamp.load_spec(KP)
    good = {"KP_PARAM_OVERRIDES": json.dumps(spec["params"]), "KP_VY_PREFIX": "vyx"}
    assert runstamp.verify_env_against_spec(KP, "KP_PARAM_OVERRIDES", good)[0] is True
    bad = dict(good, KP_VY_PREFIX="vys")
    ok, lines = runstamp.verify_env_against_spec(KP, "KP_PARAM_OVERRIDES", bad)
    assert ok is False and "KP_VY_PREFIX" in "\n".join(lines)


def test_kp_prefix_unset_is_refused():
    """KP_VY_PREFIX defaults to 'vys' inside runstamp, so unset is a real economy swap."""
    spec = runstamp.load_spec(KP)
    ok, _ = runstamp.verify_env_against_spec(
        KP, "KP_PARAM_OVERRIDES", {"KP_PARAM_OVERRIDES": json.dumps(spec["params"])})
    assert ok is False


# ---------------------------------------------------------------- gs_bx -------
# gs_bx has no param_env: its `params` is GS_PARAM_OVERRIDES, read by the SOLVER and
# already covered by the solve_id. Its simulator takes structural parameters out of
# solution.npz. So the check must verify the three GS_BX_* selectors and refuse an
# undeclared GS_SIM_OVERRIDES -- and must NOT compare `params` to GS_SIM_OVERRIDES,
# which would refuse every correct gs_bx run.

def _gs_env(**over):
    e = dict(runstamp.load_spec(GS)["env"]); e.update(over)
    return e


def test_gs_correct_env_verifies_without_sim_overrides():
    ok, lines = runstamp.verify_env_against_spec(GS, "GS_SIM_OVERRIDES", _gs_env())
    assert ok is True, lines


def test_gs_empty_sim_overrides_is_equivalent_to_unset():
    ok, _ = runstamp.verify_env_against_spec(
        GS, "GS_SIM_OVERRIDES", _gs_env(GS_SIM_OVERRIDES="{}"))
    assert ok is True


def test_gs_reordered_soldirs_is_refused():
    """Reordering soldirs against a fixed beta ladder mis-assigns every exposure."""
    d = _gs_env()["GS_BX_SOLDIRS"].split(",")
    ok, _ = runstamp.verify_env_against_spec(
        GS, "GS_SIM_OVERRIDES", _gs_env(GS_BX_SOLDIRS=",".join(reversed(d))))
    assert ok is False


def test_gs_undeclared_sim_override_is_refused():
    """The escape hatch the solve_id cannot see: overwrite a value read from the npz."""
    ok, lines = runstamp.verify_env_against_spec(
        GS, "GS_SIM_OVERRIDES", _gs_env(GS_SIM_OVERRIDES='{"gamma_x": 0.9}'))
    assert ok is False
    assert "REFUSED" in "\n".join(lines)


# ------------------------------------------------------- three-valued result ----

def test_a_spec_declaring_nothing_is_unverifiable_not_a_pass():
    """None, never True. run_oracle.py aborts on anything that is not an explicit pass,
    for the same reason verify_against_spec does: `None is False` is False, and that
    once let an unverifiable spec stamp a real result."""
    import tempfile
    spec = {"schema_version": 1, "spec_id": "tmp-empty", "model": "bgn_gam"}
    path = os.path.join(runstamp.SPECS_DIR, "tmp-empty.json")
    with open(path, "w") as fh:
        json.dump(spec, fh)
    try:
        ok, lines = runstamp.verify_env_against_spec("tmp-empty", None, {})
        assert ok is None, lines
    finally:
        os.remove(path)
