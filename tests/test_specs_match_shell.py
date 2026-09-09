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

    # solves_pending is STATUS, not definition: it says whether the precommitted
    # solves have been recorded yet, and clearing it once they land must not change
    # the spec's identity. It was inside the hashed view until 2026-09-07, so
    # verifying var-gs_bx-bx7-v3's five solves on Sol would have "drifted" its hash.
    excluded = {"title", "question", "notes", "lineage", "provenance", "spec_hash",
                "solves_pending",
                # `precommitted` records HOW the spec was made, like provenance. It is
                # not part of the experiment's definition, so it must not move the id.
                "precommitted"}
    for fn in sorted(os.listdir(SPECS)):
        if not fn.endswith(".json"):
            continue
        spec = load_spec(fn[:-5])
        view = {k: v for k, v in spec.items() if k not in excluded}
        h = hashlib.sha256(
            json.dumps(view, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert h == spec["spec_hash"], f"{fn}: hash drifted"


def test_seed_array_overrides_match_the_specs():
    """variants/run_seeds_slurm.sh restates each economy's overrides; they must agree.

    The seed array is one script for every economy rather than one per economy, so it
    carries a `case` block with the override string for each. That is a second copy of
    the parameters, and a second copy is exactly what drifted between run_gs_bx7.sh and
    var-gs_bx-bx7-v1 -- the reason this file exists. Pinned here so the array cannot
    quietly replicate a different economy from the one the spec describes.
    """
    txt = read_script("variants/run_seeds_slurm.sh")
    for spec_id, var in (("var-bgn_gam-g0235-v1", "BGN_PARAM_OVERRIDES"),
                         ("var-kp_vy-vyx-v1", "KP_PARAM_OVERRIDES")):
        spec = load_spec(spec_id)
        found = shell_json_assignments(txt, var)
        assert len(found) == 1, (
            f"expected exactly 1 {var} in run_seeds_slurm.sh, found {len(found)}")
        assert found[0] == spec["params"], (
            f"run_seeds_slurm.sh {var} = {found[0]} but {spec_id} says {spec['params']}")


def test_seed_array_does_not_solve():
    """The array must consume a recorded solve, never produce one.

    kp_vy's builder takes a single-builder lock (ten tasks would serialise on a
    multi-hour stage); bgn_gam's takes none (ten tasks would race on one CSV). Both
    failure modes are silent in SLURM logs, so the discipline is pinned rather than
    documented.
    """
    txt = read_script("variants/run_seeds_slurm.sh")
    body = [l for l in txt.splitlines() if not l.lstrip().startswith("#")]
    # The producers ARE named on live lines, inside the SOLVE_HINT strings the abort
    # path prints. What must not appear is an INVOCATION: a command line that runs one.
    for producer in ("build_vy_tables.py", "rebuild_jstar_gam.py", "gs_solve_reg.py"):
        bad = [l for l in body
               if producer in l and re.search(r"(^|[;&|]|\bthen\b)\s*(python|\$PY)\b", l)
               and "SOLVE_HINT" not in l]
        assert not bad, (
            f"run_seeds_slurm.sh invokes {producer}: {bad} -- the seed array must "
            f"consume a recorded solve, never produce one")
    live = "\n".join(body)
    assert "runstamp.py current" in live, (
        "the array no longer verifies that a live solve exists before spending compute")
    assert "runstamp.py is-current" in live, (
        "the array no longer checkpoints per seed on the consumed solve_id")


def test_spec_id_matches_filename():
    for fn in sorted(os.listdir(SPECS)):
        if fn.endswith(".json"):
            assert load_spec(fn[:-5])["spec_id"] == fn[:-5]


def test_all_three_flagship_specs_exist():
    for sid in ("var-bgn_gam-g0235-v1", "var-kp_vy-vyx-v1", "var-gs_bx-bx7-v1"):
        assert os.path.exists(os.path.join(SPECS, sid + ".json")), sid


def test_expected_solves_are_live_manifests():
    """A spec that pins a solve_id must pin one that still exists and is reachable.

    expected_solves is what closes spec -> run -> solve: run_oracle.py --spec aborts
    when the tables it read are not the ones the spec names. That guard is only as good
    as the ids, and every id in this repo moved once already (2026-09-07, the
    canonicaliser fix), so a spec pinning a retired id would abort every correct run.
    """
    sys.path.insert(0, os.path.join(ROOT, "variants"))
    from common import solstamp

    live, retired = {}, {}
    for m in solstamp.iter_manifests():
        (retired if m.get("retired") else live)[m["solve_id"]] = m

    problems, pending = [], []
    pinned = 0
    for fn in sorted(os.listdir(SPECS)):
        if not fn.endswith(".json"):
            continue
        spec = load_spec(fn[:-5])
        for stage, sid in (spec.get("expected_solves") or {}).items():
            pinned += 1
            if sid in live:
                continue
            if sid in retired:
                problems.append(f"{spec['spec_id']}.{stage} pins RETIRED {sid}: "
                                f"{retired[sid]['retired'].get('reason', '')[:80]}")
            elif spec.get("solves_pending"):
                # A spec may legitimately PRECOMMIT to an id before the solve runs --
                # the id is a hash of inputs, so it is knowable in advance, and pinning
                # it is what lets the cluster's output be checked against the laptop's
                # prediction. That is only honest when declared, so it requires the flag:
                # otherwise a typo'd id in a solved economy would pass forever.
                pending.append(f"{spec['spec_id']}.{stage} -> {sid} (declared pending)")
            else:
                problems.append(f"{spec['spec_id']}.{stage} pins {sid}, which is in no "
                                f"manifest and the spec does not declare solves_pending")
    assert pinned >= 8, f"only {pinned} pinned solve_ids across all specs -- expected 8"
    assert not problems, "specs pinning unusable solve_ids:\n  " + "\n  ".join(problems)
    if pending:
        print("    (" + str(len(pending)) + " precommitted, not yet solved)")


def test_every_current_spec_pins_its_solves():
    """Every spec that is NOT superseded must pin expected_solves.

    v1 specs predate the field and are superseded, so they are exempt; anything that is
    still the current description of a runnable economy is not.
    """
    missing = []
    for fn in sorted(os.listdir(SPECS)):
        if not fn.endswith(".json"):
            continue
        spec = load_spec(fn[:-5])
        if spec.get("lineage", {}).get("superseded_by"):
            continue
        if not spec.get("expected_solves"):
            missing.append(spec["spec_id"])
    assert not missing, ("current specs with no expected_solves, so `run_oracle.py "
                         "--spec` cannot verify them: " + ", ".join(missing))



def test_the_seed_array_can_actually_find_each_economys_solve():
    """Every SEED_SPEC's solve lookup must resolve to a live solve.

    run_seeds_slurm.sh refuses to start until `runstamp.py current --model M --tag T`
    finds one. That guard is cheap and fails closed, which is right -- but it has to ask
    the RIGHT question. It looked up the array's output TAG, while bgn_gam's solve is
    registered under its producer's tag (the J* filename), so g0235 could never start:
    job 62876077 aborted in 44 s on 2026-09-08 while be222462dd017b2c sat in the
    registry the whole time. SOLVE_TAG now carries the lookup key separately.

    This test is what makes that discoverable here instead of on a compute node.
    """
    sys.path.insert(0, os.path.join(ROOT, "variants"))
    sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
    from common import runstamp

    body = read_script("variants/run_seeds_slurm.sh")
    specs = re.findall(
        r"^\s{2}(\w+)\)\s*$.*?MODEL=(\w+);\s*TAG=(\w+);.*?SOLVE_TAG=(\S+)",
        body, re.M | re.S)
    assert specs, "parsed no SEED_SPEC branches out of run_seeds_slurm.sh"
    missing = []
    for seed_spec, model, tag, solve_tag in specs:
        if not runstamp.live_solves(model, solve_tag.strip()):
            missing.append(f"SEED_SPEC={seed_spec}: no live solve for model={model} "
                           f"tag={solve_tag} -- the array would abort at startup")
    assert not missing, "\n  ".join([""] + missing)



def test_every_runstamp_lookup_in_the_seed_array_uses_SOLVE_TAG():
    """ALL of them, not just the first one found.

    e42a3a7 fixed the precondition's lookup to use SOLVE_TAG and added a test for it.
    The per-seed checkpoint and the post-run verification made the same lookup with
    the output TAG and were not touched. g0235 seed 0 then ran both stages to
    completion (3 h 14 m) and was marked FAILED by the post-run check -- "built from
    be222462dd017b2c; registry now has nothing" -- because it looked up g0235 while the
    solve is registered as Jstar_g0235. The checkpoint had the same bug, so a resubmit
    would have re-run the seed instead of skipping it.

    A fix that reaches one call site and a test that checks one call site are the same
    mistake. This asserts every runstamp invocation in the script passes the same key.
    """
    body = read_script("variants/run_seeds_slurm.sh")
    calls = re.findall(r"runstamp\.py\s+\S+.*?--tag\s+(\S+)", body)
    assert len(calls) >= 3, f"expected the precondition, checkpoint and post-run lookups; found {len(calls)}"
    wrong = [c for c in calls if c != '"${SOLVE_TAG:-$TAG}"']
    assert not wrong, (
        "runstamp lookups in run_seeds_slurm.sh not keyed on SOLVE_TAG (bgn_gam's solve "
        f"is registered as Jstar_g0235, not g0235): {wrong}")


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