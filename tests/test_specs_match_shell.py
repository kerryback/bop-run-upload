"""Every experiment spec must still say what its shell script says.

The specs in experiments/specs/ were transcribed by hand from the variants run
scripts on 2026-09-04. While both representations exist, they can drift. These
tests re-parse the shell scripts and assert equality on every parameter.

This matters most for var-gs_bx-bx7-v1: run_gs_bx7.sh was the ONLY committed
record of that experiment (its own oracle JSON records overrides={}), so a
silent drift between script and spec would lose the flagship spec a second time.

Run with: python -m pytest tests/ -k specs   (or: python tests/test_specs_match_shell.py)
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "experiments", "specs")


def load_spec(spec_id):
    with open(os.path.join(SPECS, spec_id + ".json")) as fh:
        return json.load(fh)


def read_script(rel):
    with open(os.path.join(ROOT, rel)) as fh:
        return fh.read()


def shell_json_assignments(text, var):
    """Every `VAR='{...}'` in the script, parsed, in order of appearance.

    Matches both `export VAR='{...}'` and the inline `VAR='{...}' cmd` form that
    run_gs_bx7.sh uses to scope the override to a single solve invocation.
    """
    out = []
    for m in re.finditer(re.escape(var) + r"='(\{.*?\})'", text, re.DOTALL):
        out.append(json.loads(m.group(1)))
    return out


def shell_scalar(text, var):
    """`export VAR=value` or `VAR=value`, unquoted or single/double quoted."""
    m = re.search(re.escape(var) + r"=[\"']?([^\"'\s]+)[\"']?", text)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# BGN g0235
# --------------------------------------------------------------------------

def test_bgn_g0235_overrides_match():
    spec = load_spec("var-bgn_gam-g0235-v1")
    txt = read_script(spec["provenance"]["source_script"])
    found = shell_json_assignments(txt, "BGN_PARAM_OVERRIDES")
    assert len(found) == 1, f"expected 1 BGN_PARAM_OVERRIDES, found {len(found)}"
    assert found[0] == spec["params"], f"{found[0]} != {spec['params']}"


def test_bgn_g0235_estimator_flags_match():
    spec = load_spec("var-bgn_gam-g0235-v1")
    txt = read_script(spec["provenance"]["source_script"])
    assert "--window 360" in txt
    assert "--levels" in txt and "--include_mkt" in txt
    kap = re.search(r"--kappas\s+([\d.,]+)", txt).group(1)
    assert [float(x) for x in kap.split(",")] == spec["estimation"]["kappas"]
    n = int(re.search(r"--N\s+(\d+)", txt).group(1))
    t = int(re.search(r"--T\s+(\d+)", txt).group(1))
    assert (n, t) == (spec["panel"]["N"], spec["panel"]["T"])


# --------------------------------------------------------------------------
# KP vyx
# --------------------------------------------------------------------------

def test_kp_vyx_overrides_match():
    spec = load_spec("var-kp_vy-vyx-v1")
    txt = read_script(spec["provenance"]["source_script"])
    found = shell_json_assignments(txt, "KP_PARAM_OVERRIDES")
    assert len(found) == 1, f"expected 1 KP_PARAM_OVERRIDES, found {len(found)}"
    assert found[0] == spec["params"], f"{found[0]} != {spec['params']}"


def test_kp_vyx_prefix_matches():
    spec = load_spec("var-kp_vy-vyx-v1")
    txt = read_script(spec["provenance"]["source_script"])
    assert shell_scalar(txt, "KP_VY_PREFIX") == spec["env"]["KP_VY_PREFIX"]


def test_kp_vyx_estimator_flags_match():
    spec = load_spec("var-kp_vy-vyx-v1")
    txt = read_script(spec["provenance"]["source_script"])
    kap = re.search(r"--kappas\s+([\d.,]+)", txt).group(1)
    assert [float(x) for x in kap.split(",")] == spec["estimation"]["kappas"]
    n = int(re.search(r"--N\s+(\d+)", txt).group(1))
    t = int(re.search(r"--T\s+(\d+)", txt).group(1))
    assert (n, t) == (spec["panel"]["N"], spec["panel"]["T"])


# --------------------------------------------------------------------------
# GS bx7 -- the one that matters most
# --------------------------------------------------------------------------

def test_gs_bx7_has_five_solve_stages():
    spec = load_spec("var-gs_bx-bx7-v1")
    txt = read_script(spec["provenance"]["source_script"])
    found = shell_json_assignments(txt, "GS_PARAM_OVERRIDES")
    assert len(found) == 5, f"expected 5 GS_PARAM_OVERRIDES solves, found {len(found)}"
    assert len(spec["solve"]["stages"]) == 5


def test_gs_bx7_every_solve_stage_matches_in_order():
    """The soldir <-> (gs_bx, gs_ashift) pairing exists nowhere else."""
    spec = load_spec("var-gs_bx-bx7-v1")
    txt = read_script(spec["provenance"]["source_script"])
    found = shell_json_assignments(txt, "GS_PARAM_OVERRIDES")
    for i, (shell_params, stage) in enumerate(zip(found, spec["solve"]["stages"])):
        assert shell_params == stage["params"], (
            f"solve stage {i} ({stage['name']}): {shell_params} != {stage['params']}"
        )


def test_gs_bx7_soldir_names_and_order_match():
    spec = load_spec("var-gs_bx-bx7-v1")
    txt = read_script(spec["provenance"]["source_script"])
    # the sol_* directory is the last positional arg of each gs_solve_reg.py call
    dirs = re.findall(r"gs_solve_reg\.py\s+\d+\s+\S+\s+(sol_\w+)", txt)
    assert dirs == [s["name"] for s in spec["solve"]["stages"]], f"{dirs}"
    assert dirs == spec["types"]["soldirs"]
    assert shell_scalar(txt, "GS_BX_SOLDIRS") == ",".join(dirs)


def test_gs_bx7_betas_and_shares_match():
    spec = load_spec("var-gs_bx-bx7-v1")
    txt = read_script(spec["provenance"]["source_script"])
    betas = [float(x) for x in shell_scalar(txt, "GS_BX_BETAS").split(",")]
    shares = [float(x) for x in shell_scalar(txt, "GS_BX_SHARES").split(",")]
    assert betas == spec["types"]["exposure"]["gs_bx"]
    assert shares == spec["types"]["share"]
    assert abs(sum(shares) - 1.0) < 1e-12


def test_gs_bx7_beta_ladder_is_consistent_across_the_two_records():
    """GS_BX_BETAS and the per-solve gs_bx values must be the same ladder.

    The first solve (sol_reg) has no explicit gs_bx -- it is the beta=1 baseline
    -- so it is filled from the ladder's first entry.
    """
    spec = load_spec("var-gs_bx-bx7-v1")
    from_stages = [s["params"].get("gs_bx", 1.0) for s in spec["solve"]["stages"]]
    assert from_stages == spec["types"]["exposure"]["gs_bx"]


def test_gs_bx7_solver_args_match():
    spec = load_spec("var-gs_bx-bx7-v1")
    txt = read_script(spec["provenance"]["source_script"])
    calls = re.findall(r"gs_solve_reg\.py\s+(\d+)\s+(\S+)\s+sol_\w+", txt)
    assert len(calls) == 5
    for (xnum, tol), stage in zip(calls, spec["solve"]["stages"]):
        assert int(xnum) == stage["args"]["xnum"]
        assert float(tol) == stage["args"]["tol"]


# --------------------------------------------------------------------------
# Cross-cutting
# --------------------------------------------------------------------------

def test_every_spec_hash_is_reproducible():
    """spec_hash must be recomputable from the spec's own hashed view."""
    import hashlib

    excluded = {"title", "question", "notes", "lineage", "provenance", "spec_hash"}
    for fn in sorted(os.listdir(SPECS)):
        if not fn.endswith(".json"):
            continue
        spec = load_spec(fn[:-5])
        view = {k: v for k, v in spec.items() if k not in excluded}
        h = hashlib.sha256(
            json.dumps(view, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert h == spec["spec_hash"], f"{fn}: hash drifted"


def test_spec_id_matches_filename():
    for fn in sorted(os.listdir(SPECS)):
        if fn.endswith(".json"):
            assert load_spec(fn[:-5])["spec_id"] == fn[:-5]


def test_all_three_flagship_specs_exist():
    for sid in ("var-bgn_gam-g0235-v1", "var-kp_vy-vyx-v1", "var-gs_bx-bx7-v1"):
        assert os.path.exists(os.path.join(SPECS, sid + ".json")), sid


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
