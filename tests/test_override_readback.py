"""An override that the module silently discarded must be refused, not recorded.

`globals().update(ov)` accepts any key. A name re-derived after that line reverts; a
misspelled name is created and read by nothing. Both leave the run recording an override
the economy never used, and the spec/env check (tests/test_spec_env_refusal.py) cannot see
it: it verifies spec == environment, not environment == what the module actually used.

`parameters_kp14.py` guards this in-module. `bgn_gam/parameters.py` cannot, because it is
digest-bearing for be222462dd017b2c and all ten committed g0235 seeds; so the check is
`variants/common/readback.py`, applied after import. This pins its behaviour on the real
bgn module, imported fresh under each environment.

Run with: python -m pytest tests/ -k readback
"""
import contextlib
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))
from common import readback  # noqa: E402

BGN_PARAMS = os.path.join(ROOT, "variants", "bgn_gam", "parameters.py")
KP_PARAMS = os.path.join(ROOT, "variants", "kp_vy", "parameters_kp14.py")
GS_SIM = os.path.join(ROOT, "variants", "gs_bx", "gs_sim_bx.py")

_n = [0]


@contextlib.contextmanager
def _env(**vals):
    saved = {k: os.environ.get(k) for k in vals}
    os.environ.update(vals)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _import_fresh(path, env_var, overrides):
    """Import the parameter module under a unique name with the override applied."""
    _n[0] += 1
    name = f"_readback_probe_{_n[0]}"
    with _env(**{env_var: json.dumps(overrides)}):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


G0235 = {"gmult": [0.2, 3.5], "jstar_gam_file": "Jstar_g0235.csv"}


# ------------------------------------------------------------- the happy path ----

def test_the_g0235_overrides_take_effect():
    mod = _import_fresh(BGN_PARAMS, "BGN_PARAM_OVERRIDES", G0235)
    ok, lines = readback.overrides_took(mod, G0235)
    assert ok is True, lines


def test_nothing_requested_is_none_not_true():
    mod = _import_fresh(BGN_PARAMS, "BGN_PARAM_OVERRIDES", {})
    ok, _ = readback.overrides_took(mod, {})
    assert ok is None


def test_list_vs_array_coercion_is_not_a_mismatch():
    """parameters.py does gmult = np.array(gmult); the request was a list."""
    mod = _import_fresh(BGN_PARAMS, "BGN_PARAM_OVERRIDES", {"gmult": [0.2, 3.5]})
    ok, lines = readback.overrides_took(mod, {"gmult": [0.2, 3.5]})
    assert ok is True, lines


# -------------------------------------------------------- the two silent holes ----

def test_a_derived_parameter_override_is_refused():
    """prob_calm = p10/(p01+p10) is recomputed AFTER globals().update(_ov)."""
    ov = {"prob_calm": 0.9}
    mod = _import_fresh(BGN_PARAMS, "BGN_PARAM_OVERRIDES", ov)
    assert abs(mod.prob_calm - 0.9) > 1e-6, "the probe assumes the module re-derives prob_calm"
    ok, lines = readback.overrides_took(mod, ov)
    assert ok is False
    assert "DISCARDED" in "\n".join(lines) and "prob_calm" in "\n".join(lines)


def test_a_derived_array_override_is_refused():
    ov = {"Preg": [[0.5, 0.5], [0.5, 0.5]]}
    mod = _import_fresh(BGN_PARAMS, "BGN_PARAM_OVERRIDES", ov)
    ok, _ = readback.overrides_took(mod, ov)
    assert ok is False


def test_a_misspelled_parameter_is_refused():
    """hasattr() cannot catch this: globals().update() already created the key."""
    ov = {"gmulr": [0.2, 3.5]}
    mod = _import_fresh(BGN_PARAMS, "BGN_PARAM_OVERRIDES", ov)
    assert hasattr(mod, "gmulr"), "the probe assumes update() created the misspelled key"
    ok, lines = readback.overrides_took(mod, ov)
    assert ok is False
    assert "NOT A PARAMETER" in "\n".join(lines)


def test_one_bad_key_fails_the_whole_set_and_names_it():
    ov = dict(G0235, prob_calm=0.9)
    mod = _import_fresh(BGN_PARAMS, "BGN_PARAM_OVERRIDES", ov)
    ok, lines = readback.overrides_took(mod, ov)
    assert ok is False
    blob = "\n".join(lines)
    assert "prob_calm" in blob and "gmult" in blob


# ----------------------------------------------------------- the other models ----

def test_kp_normalised_type_share_is_compared_proportionally():
    ov = {"type_share": [0.34, 0.33, 0.33], "type_bv": [0.02, 0.07, 0.14],
          "gamma_v": 1.8, "bv_comp": 1.2}
    mod = _import_fresh(KP_PARAMS, "KP_PARAM_OVERRIDES", ov)
    ok, lines = readback.overrides_took(mod, ov)
    assert ok is True, lines


def test_gs_sim_module_scope_names_are_found_without_importing_it():
    """gs_sim_bx.py loads five 100 MB solutions at import, so its names are parsed
    from source. The GS_SIM_OVERRIDES site is at module scope after the tables load."""
    names = readback.assigned_names_in_source(GS_SIM)
    for n in ("burnin", "reg_seed", "chars", "gamma_grid", "sigma_m"):
        assert n in names, n
    assert "gamma_xx" not in names


def test_names_bound_inside_functions_are_not_parameters():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write("a = 1\nfor i in range(2):\n    b = i\ndef f():\n    c = 3\n    return c\n"
                 "class K:\n    d = 4\n")
        p = fh.name
    try:
        names = readback.assigned_names_in_source(p)
    finally:
        os.remove(p)
    assert {"a", "i", "b", "f", "K"} <= names
    assert "c" not in names and "d" not in names


def test_from_env_rejects_malformed_json():
    mod = _import_fresh(BGN_PARAMS, "BGN_PARAM_OVERRIDES", {})
    ok, lines = readback.overrides_took_from_env(mod, "X_OVERRIDES", {"X_OVERRIDES": "{nope"})
    assert ok is False and "not valid JSON" in "\n".join(lines)


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
