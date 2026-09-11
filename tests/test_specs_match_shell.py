"""Every CURRENT experiment spec must still say what its scripts say.

The specs in experiments/specs/ were transcribed by hand from the variants run scripts
on 2026-09-04. While both representations exist they can drift, and the drift is silent:
the script runs, the spec reads plausibly, and the two describe different economies.

Two lessons shape this file:

  * Guard the CURRENT spec, not the one it superseded. Until 2026-09-09 the GS tests
    checked var-gs_bx-bx7-v1 against run_gs_bx7.sh; both were stale in the same way, so
    they agreed and the suite was green while v3 -- the spec actually pinned to the five
    live solves -- said two incompatible things about gs_ashift. Every test below loads
    the spec named in CURRENT and the script that spec's provenance names.
  * The seed array restates each economy's parameters in a `case` block. That is a
    second copy, and a second copy is exactly what drifted before. It is pinned against
    the spec each case names (test_seed_array_cases_match_the_specs_they_name), so
    adding an economy to the array means adding it correctly or not at all.

Run with: python -m pytest tests/ -k specs   (or: python tests/test_specs_match_shell.py)
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "experiments", "specs")

# The spec each economy currently runs under. When one is superseded, update this.
CURRENT = {
    "g0235": "var-bgn_gam-g0235-v2",
    "vyx": "var-kp_vy-vyx-v2",
    "bx7": "var-gs_bx-bx7-v3",
    "g28": "var-gs_bx-g28-v2",
    # proposed 2026-09-10 (docs/RESULTS.md); B1's three persistence points and G1
    "g0235f": "var-bgn_gam-g0235f-v1",
    "g0235s": "var-bgn_gam-g0235s-v1",
    "g0235r": "var-bgn_gam-g0235r-v1",
    "gx7": "var-gs_bx-gx7-v1",
}
SEED_ARRAY = "variants/run_seeds_slurm.sh"


def load_spec(spec_id):
    with open(os.path.join(SPECS, spec_id + ".json")) as fh:
        return json.load(fh)


def read_script(rel):
    with open(os.path.join(ROOT, rel)) as fh:
        return fh.read()


def source_script(spec):
    rel = spec["provenance"].get("source_script")
    assert rel, f"{spec['spec_id']} names no source_script; nothing to check it against"
    assert os.path.exists(os.path.join(ROOT, rel)), f"{spec['spec_id']} names {rel}, which does not exist"
    return read_script(rel)


def shell_json_assignments(text, var):
    """Every `VAR='{...}'` in the script, parsed, in order of appearance."""
    return [json.loads(m.group(1))
            for m in re.finditer(re.escape(var) + r"='(\{.*?\})'", text, re.DOTALL)]


def shell_scalar(text, var):
    """`export VAR=value` or `VAR=value`, unquoted or single/double quoted."""
    m = re.search(r"(?m)^\s*(?:export\s+)?" + re.escape(var) + r"=[\"']?([^\"'\s]+)[\"']?", text)
    return m.group(1) if m else None


def shell_bash_array(text, var):
    """`VAR=( a b 'c' )` possibly spanning lines -> list of the unquoted elements."""
    m = re.search(re.escape(var) + r"=\((.*?)\)", text, re.DOTALL)
    assert m, f"no bash array {var}=( ... ) in script"
    return [e[0] or e[1] or e[2] for e in re.findall(r"'([^']*)'|\"([^\"]*)\"|(\S+)", m.group(1))]


def seed_array_cases():
    """The seed array's `case` block, parsed: SEED_SPEC name -> what that branch sets."""
    txt = read_script(SEED_ARRAY)
    body = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    m = re.search(r"case \"\$SEED_SPEC\" in(.*?)^esac", body, re.DOTALL | re.M)
    assert m, "no case block on $SEED_SPEC"
    cases = {}
    for name, block in re.findall(r"^\s{2}(\w+)\)\s*$(.*?)^\s*;;", m.group(1), re.DOTALL | re.M):
        c = {"env": {}}
        for k in ("MODEL", "TAG", "SPEC", "SOLVE_TAG", "KAPPAS"):
            mm = re.search(r"(?m)(?:^|[;\s])" + k + r"=([^;\s]+)", block)
            c[k] = mm.group(1) if mm else None
        for mm in re.finditer(r"(?m)^\s*export\s+(\w+)=(?:'([^']*)'|\"([^\"]*)\"|(\S+))", block):
            c["env"][mm.group(1)] = mm.group(2) if mm.group(2) is not None else (mm.group(3) or mm.group(4))
        cases[name] = c
    assert cases, "parsed no SEED_SPEC branches"
    return cases


# --------------------------------------------------------------------------
# BGN g0235 and KP vyx: the laptop scripts each spec still names
# --------------------------------------------------------------------------

def test_bgn_g0235_overrides_match():
    spec = load_spec(CURRENT["g0235"])
    found = shell_json_assignments(source_script(spec), "BGN_PARAM_OVERRIDES")
    assert len(found) == 1, f"expected 1 BGN_PARAM_OVERRIDES, found {len(found)}"
    assert found[0] == spec["params"], f"{found[0]} != {spec['params']}"


def test_bgn_g0235_estimator_flags_match():
    spec = load_spec(CURRENT["g0235"])
    txt = source_script(spec)
    assert f"--window {spec['estimation']['window']}" in txt
    assert "--levels" in txt and "--include_mkt" in txt
    kap = re.search(r"--kappas\s+([\d.,]+)", txt).group(1)
    assert [float(x) for x in kap.split(",")] == spec["estimation"]["kappas"]
    n = int(re.search(r"--N\s+(\d+)", txt).group(1))
    t = int(re.search(r"--T\s+(\d+)", txt).group(1))
    assert (n, t) == (spec["panel"]["N"], spec["panel"]["T"])


def test_kp_vyx_overrides_match():
    spec = load_spec(CURRENT["vyx"])
    found = shell_json_assignments(source_script(spec), "KP_PARAM_OVERRIDES")
    assert len(found) == 1, f"expected 1 KP_PARAM_OVERRIDES, found {len(found)}"
    assert found[0] == spec["params"], f"{found[0]} != {spec['params']}"


def test_kp_vyx_prefix_matches():
    spec = load_spec(CURRENT["vyx"])
    assert shell_scalar(source_script(spec), "KP_VY_PREFIX") == spec["env"]["KP_VY_PREFIX"]


def test_kp_vyx_estimator_flags_match():
    spec = load_spec(CURRENT["vyx"])
    txt = source_script(spec)
    kap = re.search(r"--kappas\s+([\d.,]+)", txt).group(1)
    assert [float(x) for x in kap.split(",")] == spec["estimation"]["kappas"]
    n = int(re.search(r"--N\s+(\d+)", txt).group(1))
    t = int(re.search(r"--T\s+(\d+)", txt).group(1))
    assert (n, t) == (spec["panel"]["N"], spec["panel"]["T"])


# --------------------------------------------------------------------------
# GS bx7: five solves made by the SLURM array; the panel comes from the seed array
# --------------------------------------------------------------------------

def test_gs_bx7_solve_array_restates_all_five_stages_in_order():
    """run_gs_bx7_slurm.sh's OVERRIDES array is the record of how the five pinned
    solves were made. It must equal solve.stages, element for element."""
    spec = load_spec(CURRENT["bx7"])
    ovs = [json.loads(o) for o in shell_bash_array(source_script(spec), "OVERRIDES")]
    stages = spec["solve"]["stages"]
    assert len(ovs) == len(stages) == 5, (len(ovs), len(stages))
    for i, (shell_params, stage) in enumerate(zip(ovs, stages)):
        assert shell_params == stage["params"], (
            f"solve stage {i} ({stage['name']}): {shell_params} != {stage['params']}")


def test_gs_bx7_soldir_names_and_order_match_everywhere():
    spec = load_spec(CURRENT["bx7"])
    dirs = shell_bash_array(source_script(spec), "OUTDIRS")
    assert dirs == [s["name"] for s in spec["solve"]["stages"]], dirs
    assert dirs == spec["types"]["soldirs"]
    assert spec["env"]["GS_BX_SOLDIRS"] == ",".join(dirs)
    assert list(spec["expected_solves"]) == dirs, "expected_solves must be keyed by soldir, in order"


def test_gs_bx7_solver_args_match():
    spec = load_spec(CURRENT["bx7"])
    m = re.search(r"gs_solve_reg\.py\s+(\d+)\s+(\S+)\s+\"?\$OUTDIR", source_script(spec))
    assert m, "no gs_solve_reg.py invocation over $OUTDIR"
    for stage in spec["solve"]["stages"]:
        assert int(m.group(1)) == stage["args"]["xnum"]
        assert float(m.group(2)) == stage["args"]["tol"]


def test_gs_bx7_beta_ladder_is_consistent_across_the_records():
    """solve.stages, types.exposure and env must be the SAME ladder. sol_reg has no
    explicit gs_bx -- it is the beta=1 baseline -- so it is filled from the default."""
    spec = load_spec(CURRENT["bx7"])
    from_stages = [s["params"].get("gs_bx", 1.0) for s in spec["solve"]["stages"]]
    assert from_stages == spec["types"]["exposure"]["gs_bx"]
    assert [float(x) for x in spec["env"]["GS_BX_BETAS"].split(",")] == from_stages


def test_gs_bx7_ashift_ladder_is_consistent_across_the_records():
    """The 2026-09-09 drift: method said 'gs_ashift = 0 for every type' and every pinned
    solve recorded 0.0, while types.exposure still listed the v2 ladder."""
    spec = load_spec(CURRENT["bx7"])
    from_stages = [s["params"].get("gs_ashift", 0.0) for s in spec["solve"]["stages"]]
    assert from_stages == spec["types"]["exposure"]["gs_ashift"], (
        f"solve.stages say gs_ashift {from_stages}; types.exposure says "
        f"{spec['types']['exposure']['gs_ashift']}")
    assert all(a == 0.0 for a in from_stages), "v3 is the zero-ashift ladder by definition"


def test_gs_specs_env_block_agrees_with_types():
    """GS_BX_* are the only statement the simulator gets of which types exist."""
    for key in ("bx7", "g28"):
        spec = load_spec(CURRENT[key])
        env, types = spec["env"], spec["types"]
        assert env["GS_BX_SOLDIRS"].split(",") == types["soldirs"], key
        assert [float(x) for x in env["GS_BX_BETAS"].split(",")] == types["exposure"]["gs_bx"], key
        shares = [float(x) for x in env["GS_BX_SHARES"].split(",")]
        assert shares == types["share"], key
        assert abs(sum(shares) - 1.0) < 1e-12, key


# --------------------------------------------------------------------------
# GS g28: one solve, made by run_g28_slurm.sh
# --------------------------------------------------------------------------

def test_gs_g28_solve_script_restates_its_stage():
    spec = load_spec(CURRENT["g28"])
    rel = spec["provenance"]["solve_script"]
    txt = read_script(rel)
    stages = spec["solve"]["stages"]
    assert len(stages) == 1
    ov = shell_json_assignments(txt, "OV")
    assert len(ov) == 1 and ov[0] == stages[0]["params"], (ov, stages[0]["params"])
    assert shell_scalar(txt, "OUTDIR") == stages[0]["name"] == spec["types"]["soldirs"][0]
    m = re.search(r"gs_solve_gam\.py\s+(\d+)\s+(\S+)\s+\"?\$OUTDIR", txt)
    assert m and int(m.group(1)) == stages[0]["args"]["xnum"]
    assert float(m.group(2)) == stages[0]["args"]["tol"]
    assert stages[0]["producer"].endswith("gs_solve_gam.py")


# --------------------------------------------------------------------------
# The seed array: one script, every economy, each case pinned to the spec it names
# --------------------------------------------------------------------------

def test_seed_array_cases_match_the_specs_they_name():
    """Each SEED_SPEC branch restates its economy's parameters. They must equal the spec
    that branch's SPEC= names -- model, override JSON, literal env, and the kappa grid."""
    cases = seed_array_cases()
    for name, c in cases.items():
        assert c["SPEC"], f"{name}: no SPEC="
        spec = load_spec(c["SPEC"])
        assert not spec.get("lineage", {}).get("superseded_by"), (
            f"{name} runs {c['SPEC']}, which is superseded by "
            f"{spec['lineage']['superseded_by']}")
        assert c["MODEL"] == spec["model"], (name, c["MODEL"], spec["model"])
        pe = spec.get("param_env")
        if pe:
            assert pe in c["env"], f"{name}: {c['SPEC']} carries params in {pe}, branch does not export it"
            assert json.loads(c["env"][pe]) == spec["params"], (
                f"{name}: {pe} = {c['env'][pe]} but {c['SPEC']} says {spec['params']}")
        for k, v in (spec.get("env") or {}).items():
            assert c["env"].get(k) == str(v), (
                f"{name}: {k} = {c['env'].get(k)!r} but {c['SPEC']} says {v!r}")
        assert c["KAPPAS"], f"{name}: no KAPPAS= (the estimator grid must come from the spec)"
        assert [float(x) for x in c["KAPPAS"].split(",")] == spec["estimation"]["kappas"], (
            f"{name}: KAPPAS {c['KAPPAS']} but {c['SPEC']} says {spec['estimation']['kappas']}")


def test_seed_array_passes_the_case_kappas_to_the_estimators():
    body = "\n".join(l for l in read_script(SEED_ARRAY).splitlines()
                     if not l.lstrip().startswith("#"))
    assert re.search(r'--kappas\s+"\$KAPPAS"', body), (
        "run_estimators.py must be given --kappas \"$KAPPAS\"; a hardcoded grid scores "
        "the gs specs (eight kappas) on the wrong ridge grid")


def test_every_current_spec_has_a_seed_array_case():
    cases = seed_array_cases()
    named = {c["SPEC"] for c in cases.values()}
    missing = [sid for sid in CURRENT.values()
               if sid not in named and not load_spec(sid).get("lineage", {}).get("superseded_by")]
    # Every current spec must be runnable the checked way. (bx7 joined on 2026-09-09
    # once runstamp could look up a five-solve economy.)
    allowed_missing = set()
    assert set(missing) <= allowed_missing, f"current specs with no SEED_SPEC case: {missing}"


def test_seed_array_passes_its_window_to_the_oracle():
    """--eval_window must be the estimators' --window, or the oracle's window-restricted
    ceilings cover different months from the gap they are differenced against."""
    body = "\n".join(l for l in read_script(SEED_ARRAY).splitlines()
                     if not l.lstrip().startswith("#"))
    assert re.search(r'run_oracle\.py.*?--eval_window\s+"\$WINDOW"', body, re.DOTALL), (
        "run_oracle.py must be given --eval_window \"$WINDOW\"")
    assert re.search(r'run_estimators\.py.*?--window\s+"\$WINDOW"', body, re.DOTALL)


def test_seed_array_output_paths_all_derive_from_one_results_dir():
    """BOP_RESULTS_DIR moves the output; the log and the run record must move with it.

    The script hardcoded `results/` for both while run_oracle.py and run_estimators.py
    honoured the variable, so setting it split the output from the checkpoint that reads
    it: the per-seed guard would look for a run record that had been written somewhere
    else and re-run a finished seed, silently.
    """
    body = "\n".join(l for l in read_script(SEED_ARRAY).splitlines()
                     if not l.lstrip().startswith("#"))
    assert re.search(r'RESULTS=\$\{BOP_RESULTS_DIR:-results\}', body), (
        "the script must mirror BOP_RESULTS_DIR into one shell variable")
    for var in ("LOG=", "RUNJSON=", "ORACLEJSON="):
        m = re.search(r"(?m)^\s*" + var + r'"([^"]*)"', body)
        assert m, f"no {var} assignment"
        assert m.group(1).startswith("$RESULTS/"), (
            f"{var}{m.group(1)!r} does not live under $RESULTS")
    assert not re.search(r'(?m)^\s*(LOG|RUNJSON|ORACLEJSON)="results/', body)


def test_seed_array_oracle_stage_runs_the_oracle_and_nothing_else():
    """SEED_STAGE=oracle re-derives ceilings on an existing economy. It must not reach
    run_estimators.py (there is no panel: --save_panel is dropped in this mode) and must
    not be able to pass an unrecognised stage silently."""
    body = "\n".join(l for l in read_script(SEED_ARRAY).splitlines()
                     if not l.lstrip().startswith("#"))
    assert re.search(r'SEED_STAGE=\$\{SEED_STAGE:-both\}', body)
    assert re.search(r'case "\$SEED_STAGE" in both\|oracle\)', body), "unknown stages must be rejected"
    # the early exit sits between the oracle call and the estimator call
    o = body.index("run_oracle.py")
    e = body.index("run_estimators.py")
    guard = body[o:e]
    assert re.search(r'if \[ "\$SEED_STAGE" = oracle \]; then(.|\n)*?exit 0', guard), (
        "SEED_STAGE=oracle must exit before run_estimators.py")
    assert re.search(r'\[ "\$SEED_STAGE" = oracle \] && SAVE_PANEL=', body), (
        "oracle-only runs must drop --save_panel; nothing downstream reads the panel")
    assert "--levels $SAVE_PANEL" in body


def test_seed_array_does_not_solve():
    """The array must consume a recorded solve, never produce one."""
    txt = read_script(SEED_ARRAY)
    body = [l for l in txt.splitlines() if not l.lstrip().startswith("#")]
    for producer in ("build_vy_tables.py", "rebuild_jstar_gam.py", "gs_solve_reg.py", "gs_solve_gam.py"):
        bad = [l for l in body
               if producer in l and re.search(r"(^|[;&|]|\bthen\b)\s*(python|\$PY)\b", l)
               and "SOLVE_HINT" not in l]
        assert not bad, (
            f"run_seeds_slurm.sh invokes {producer}: {bad} -- the seed array must "
            f"consume a recorded solve, never produce one")
    live = "\n".join(body)
    assert "runstamp.py current" in live
    assert "runstamp.py is-current" in live


def test_gs_solve_hints_fetch_rather_than_resolve():
    """When a GS solve is missing the array must say how to get it, and for an economy
    whose solves ALREADY EXIST that means FETCH, never re-solve: they are published
    content-addressed and cost 3-6 h each to remake.

    An economy still declaring `solves_pending` has nothing to fetch -- nobody has built
    its solves yet -- so its hint is the job that builds them. The distinction is the
    point: a fetch hint on an unsolved economy would send someone looking for a file that
    does not exist, and a solve hint on a published one burns a day of cluster time.
    """
    for name, c in seed_array_cases().items():
        if c["MODEL"] != "gs_bx":
            continue
        block = re.search(r"^\s{2}" + name + r"\)\s*$(.*?)^\s*;;", read_script(SEED_ARRAY), re.DOTALL | re.M).group(1)
        hint = re.search(r"SOLVE_HINT=(['\"])(.*?)\1", block, re.DOTALL)
        assert hint, f"{name}: no SOLVE_HINT"
        body = hint.group(2)
        if load_spec(c["SPEC"]).get("solves_pending"):
            assert "sbatch" in body, (
                f"{name}: its solves are declared pending, so the hint must be the job that "
                f"builds them, not a fetch of something that does not exist yet")
        else:
            assert "fetch_solves.py" in body and c["SPEC"] in body, (
                f"{name}: SOLVE_HINT must be a fetch_solves.py --spec {c['SPEC']} command")


# --------------------------------------------------------------------------
# Cross-cutting
# --------------------------------------------------------------------------

def test_every_spec_hash_is_reproducible():
    """spec_hash must be recomputable from the spec's own hashed view."""
    import hashlib
    excluded = {"title", "question", "notes", "lineage", "provenance", "spec_hash",
                # solves_pending is STATUS; precommitted and reused_solves record HOW the
                # spec was made, not what economy it defines
                "solves_pending", "precommitted", "reused_solves"}
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


def test_current_specs_exist_and_are_not_superseded():
    for key, sid in CURRENT.items():
        assert os.path.exists(os.path.join(SPECS, sid + ".json")), sid
        assert not load_spec(sid).get("lineage", {}).get("superseded_by"), (
            f"CURRENT[{key}] = {sid} is superseded; update CURRENT")


def test_expected_solves_are_live_manifests():
    """A spec that pins a solve_id must pin one that still exists and is reachable."""
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
                pending.append(f"{spec['spec_id']}.{stage} -> {sid} (declared pending)")
            else:
                problems.append(f"{spec['spec_id']}.{stage} pins {sid}, which is in no "
                                f"manifest and the spec does not declare solves_pending")
    assert pinned >= 8, f"only {pinned} pinned solve_ids across all specs -- expected 8"
    assert not problems, "specs pinning unusable solve_ids:\n  " + "\n  ".join(problems)
    if pending:
        print("    (" + str(len(pending)) + " precommitted, not yet solved)")


def test_every_current_spec_pins_its_solves():
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

    The array refuses to start until `runstamp.py current --model M --tag T` finds one.
    It once asked for the output TAG while bgn_gam's solve is registered under the J*
    filename, so g0235 could never start (job 62876077, 44 s). SOLVE_TAG carries the
    lookup key; a multi-solve economy lists several, comma-separated, and every one
    must resolve.
    """
    sys.path.insert(0, os.path.join(ROOT, "variants"))
    from common import runstamp

    missing = []
    for name, c in seed_array_cases().items():
        # A spec may legitimately PRECOMMIT solve ids before the solve runs -- the id is a
        # hash of inputs, so it is knowable in advance, and the array's own precondition is
        # what stops it starting early. That state is declared by `solves_pending`, exactly
        # as test_expected_solves_are_live_manifests treats it; without the flag a case
        # pointing at a solve nobody ever built would pass forever.
        if c["SPEC"] and load_spec(c["SPEC"]).get("solves_pending"):
            continue
        for tag in (c["SOLVE_TAG"] or c["TAG"]).split(","):
            if not runstamp.live_solves(c["MODEL"], tag.strip()):
                missing.append(f"SEED_SPEC={name}: no live solve for model={c['MODEL']} "
                               f"tag={tag} -- the array would abort at startup")
    assert not missing, "\n  ".join([""] + missing)


def test_every_runstamp_lookup_in_the_seed_array_uses_SOLVE_TAG():
    """ALL of them. e42a3a7 fixed one call site; the checkpoint and the post-run check
    kept the old key and g0235 seed 0 was marked FAILED after 3 h 14 m of good work."""
    body = read_script(SEED_ARRAY)
    calls = re.findall(r"runstamp\.py\s+\S+.*?--tag\s+(\S+)", body)
    assert len(calls) >= 3, f"expected the precondition, checkpoint and post-run lookups; found {len(calls)}"
    wrong = [c for c in calls if c != '"${SOLVE_TAG:-$TAG}"']
    assert not wrong, f"runstamp lookups not keyed on SOLVE_TAG: {wrong}"


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
