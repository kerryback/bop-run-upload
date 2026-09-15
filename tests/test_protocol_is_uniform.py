"""Every reported economy is measured the same way, and the three copies of "the same way"
cannot drift apart.

WHY THIS FILE EXISTS. Results differ along two axes and only one is interesting: an economy
differs from another economy in its PARAMETERS and its DRIVING FORCES. It must not differ in
how many firms were simulated, how long the burn-in was, how many months were evaluated, or
how fine any grid was. When those drift, the rows of a table stop being comparable, and prose
does not fix it.

They had drifted, in four ways, none of which any test could see before 2026-09-15:

  burn-in     bgn_gam 300, kp_vy 400, gs_bx 300 in the CODE -- while twelve of nineteen specs
              DECLARED 200, which is config.py's value, i.e. the legacy main.py pipeline's,
              copied in from the wrong tree. Nothing read the spec field. run_oracle.py never
              receives a burn-in: it imports whatever its model module happens to say.
  ridge grid  Four kappas for bgn_gam and kp_vy, eight for gs_bx, five including 1e-4 for
              vyxT860 -- one per `case` branch of the seed array. DKKM's reported Sharpe is a
              MAX over that grid, and the floor won in 29 of 30 KP14 Path 1 seeds, so the
              number was censored at a point that differed by economy. A quarter of vyxT860's
              headline gap was that extra decade rather than its economy.
  sample      vyxT860 at T=860 and window 720 against everyone else's 500 and 360.
  rf_cols     the three baselines narrowed their feature bases to the state their paper has,
              so baseline and parameterization differed in their bases as well as their
              economics, and "what the route added" was not a difference in the economy alone.

variants/common/protocol.py now holds the numbers. They appear in three places -- that module,
the seed array's literals, and every spec's panel/estimation block -- because the seed array
runs on a cluster where importing the repo at job-start is a failure mode nobody wants on the
critical path, and because a spec must be a self-contained record of how its run was measured.
Three copies are fine if a test pins them, which is what this file is. What was missing before
was not a single source of truth; it was any check at all.

WHAT THIS FILE DOES NOT CHECK. Solve-side precision is per model -- the three papers have
different solvers -- so it cannot be shared. What matters is that it is uniform ACROSS each
model's economies, which is checked here against the registry manifests, not against protocol.py.

Run with: python -m pytest tests/ -k protocol
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "experiments", "specs")
REGISTRY = os.path.join(ROOT, "experiments", "registry")
SEED_ARRAY = os.path.join(ROOT, "variants", "run_seeds_slurm.sh")

sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
import protocol  # noqa: E402

# The economy -> spec map lives in one place; import it rather than keep a second copy.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_specs_match_shell import CURRENT  # noqa: E402

# Where each model's simulation burn-in actually lives. run_oracle.py imports `burnin` from
# these modules (`from parameters import *`, `from gs_sim_bx import burnin`), so these
# literals are what runs -- not anything a spec says.
BURNIN_SITES = {
    "bgn_gam": os.path.join(ROOT, "variants", "bgn_gam", "parameters.py"),
    "kp_vy": os.path.join(ROOT, "variants", "kp_vy", "parameters_kp14.py"),
    "gs_bx": os.path.join(ROOT, "variants", "gs_bx", "gs_sim_bx.py"),
}


def load_spec(spec_id):
    with open(os.path.join(SPECS, spec_id + ".json")) as fh:
        return json.load(fh)


def live_specs():
    """{tag: spec} for the economies that may be reported."""
    return {tag: load_spec(sid) for tag, sid in CURRENT.items()}


def module_burnin(path):
    """The module-scope `burnin = <int>` literal, read from source.

    Read rather than imported on purpose: gs_sim_bx.py loads ~500 MB of solutions at import
    (tests/test_override_readback.py parses its names from source for the same reason), and
    importing three parameter modules into one interpreter would collide on `from ... import *`.
    """
    with open(path) as fh:
        src = fh.read()
    m = re.search(r"(?m)^burnin\s*=\s*(\d+)", src)
    assert m, f"no module-scope `burnin = <int>` in {path}"
    return int(m.group(1))


def seed_array():
    with open(SEED_ARRAY) as fh:
        return fh.read()


# --------------------------------------------------------------------------
# protocol.py is self-consistent
# --------------------------------------------------------------------------

def test_evaluated_months_are_the_derived_count():
    """125 is not a free parameter: run_oracle.py keeps T - 15 months (`months >= burnin + 14`
    and `months <= T + burnin - 2`) and the rolling estimator consumes `window` more."""
    assert protocol.EVAL_MONTHS == protocol.T - protocol.EVAL_TRIM - protocol.WINDOW
    assert protocol.EVAL_MONTHS == 125, protocol.EVAL_MONTHS


def test_the_ridge_grid_spans_enough_decades_to_have_an_interior_argmax():
    """Not a check that the argmax IS interior -- that is a property of each economy's results
    and is checked there. This is the weaker structural claim: the grid has decades on both
    sides of where the winner has historically sat (1e-3 for every economy measured before
    2026-09-15), so an interior argmax is at least POSSIBLE."""
    k = sorted(protocol.KAPPAS)
    assert k == list(protocol.KAPPAS), "keep KAPPAS sorted; the summary CSVs are read by value"
    assert k[0] <= 1e-5 and k[-1] >= 10, k
    assert k.index(1e-3) >= 2, "at least two decades below the old grid floor"
    assert len(k) - 1 - k.index(1e-3) >= 3, "at least three decades above the old floor"


# --------------------------------------------------------------------------
# the code agrees with protocol.py
# --------------------------------------------------------------------------

def test_every_models_burnin_is_the_protocols():
    """The literal each model module runs, against BURNIN. These sat at 300 / 400 / 300."""
    bad = {m: module_burnin(p) for m, p in BURNIN_SITES.items()
           if module_burnin(p) != protocol.BURNIN}
    assert not bad, (f"burn-in differs from the protocol's {protocol.BURNIN}: {bad}. "
                     f"An economy burned in for a different number of months is measured "
                     f"differently, not modelled differently.")


def test_the_seed_array_runs_the_protocol():
    txt = seed_array()
    got = {
        "N": int(re.search(r"(?m)^N=\$\{SEED_N:-(\d+)\}", txt).group(1)),
        "T": int(re.search(r"(?m)^T=\$\{SEED_T:-(\d+)\}", txt).group(1)),
        "WINDOW": int(re.search(r"(?m)^WINDOW=\$\{SEED_WINDOW:-(\d+)\}", txt).group(1)),
    }
    want = {"N": protocol.N, "T": protocol.T, "WINDOW": protocol.WINDOW}
    assert got == want, f"{SEED_ARRAY} runs {got}, protocol.py says {want}"
    grid = re.search(r"(?m)^KAPPAS=(\S+)$", txt).group(1)
    assert [float(x) for x in grid.split(",")] == list(protocol.KAPPAS), (
        f"the array's ridge grid is {grid}, protocol.py says {protocol.kappas_csv()}")


def test_the_seed_array_sets_the_sample_and_the_grid_exactly_once():
    """One assignment, outside the `case` block. A second copy is what drifted: the grid used
    to be per branch, and three different grids reached the estimators."""
    txt = seed_array()
    body = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    for var in ("N", "T", "WINDOW", "KAPPAS"):
        n = len(re.findall(r"(?m)^" + var + r"=", body))
        assert n == 1, f"{var} is assigned {n} times in the seed array; expected exactly 1"
    assert not re.search(r"(?m)^\s+(KAPPAS|RF_COLS)=", body), (
        "a case branch sets KAPPAS or RF_COLS; both are the protocol's")
    assert not re.search(r"SEED_(T|WINDOW):=", body), (
        "a case branch pins its own sample length or window")


def test_an_off_protocol_run_may_not_write_into_the_reported_results():
    """SEED_N/SEED_T/SEED_WINDOW stay available for a smoke test, but such a run must name its
    own BOP_RESULTS_DIR. Without this, probes at N=60/T=80/window=20 landed in
    variants/results and were aggregated into economy_table.csv beside the real economies."""
    body = "\n".join(l for l in seed_array().splitlines() if not l.lstrip().startswith("#"))
    m = re.search(r'if \[ "\$N" != 500 \] \|\| \[ "\$T" != 500 \] \|\| \[ "\$WINDOW" != 360 \];'
                  r' then(.*?)\nfi', body, re.DOTALL)
    assert m, "no off-protocol guard on N/T/WINDOW in the seed array"
    guard = m.group(1)
    assert 'BOP_RESULTS_DIR' in guard and "exit 2" in guard, guard


# --------------------------------------------------------------------------
# every live spec agrees with protocol.py
# --------------------------------------------------------------------------

def test_every_live_spec_declares_the_protocol_sample():
    bad = []
    for tag, spec in live_specs().items():
        for key, want in (("N", protocol.N), ("T", protocol.T), ("burnin", protocol.BURNIN)):
            if spec["panel"].get(key) != want:
                bad.append(f"{spec['spec_id']}: panel.{key} = {spec['panel'].get(key)!r}, protocol {want}")
        if spec["estimation"].get("window") != protocol.WINDOW:
            bad.append(f"{spec['spec_id']}: estimation.window = "
                       f"{spec['estimation'].get('window')!r}, protocol {protocol.WINDOW}")
    assert not bad, "\n  " + "\n  ".join(bad)


def test_every_live_spec_declares_the_protocol_ridge_grid():
    bad = [f"{s['spec_id']}: {s['estimation'].get('kappas')}"
           for s in live_specs().values()
           if [float(k) for k in (s["estimation"].get("kappas") or [])] != list(protocol.KAPPAS)]
    assert not bad, (f"specs off the protocol grid {protocol.kappas_csv()}:\n  "
                     + "\n  ".join(bad))


def test_no_live_spec_narrows_the_conditioning_columns():
    """protocol.RF_COLS is None: the model's FULL set, for every economy including the
    baselines. A narrowed baseline is a different protocol, not a different economy."""
    bad = [f"{s['spec_id']}: rf_cols {s['estimation']['rf_cols']!r}"
           for s in live_specs().values() if "rf_cols" in s["estimation"]]
    assert not bad, ("these specs narrow the feature bases' conditioning columns:\n  "
                     + "\n  ".join(bad))


def test_every_live_spec_scores_the_fair_benchmark():
    """DKKM gets the equal-weighted market unpenalised (--include_mkt); the linear methods must
    be given it on the same terms, or the measured gap is partly the market (finding 8)."""
    bad = [s["spec_id"] for s in live_specs().values()
           if not (s["estimation"].get("fair_linear") and s["estimation"].get("include_mkt")
                   and s["estimation"].get("levels"))]
    assert not bad, f"specs missing fair_linear / include_mkt / levels: {bad}"


def test_the_retired_long_sample_spec_is_not_live():
    """kp_vy/vyxT860's economy IS vyx's: it differed only in T, the window and the ridge grid.
    It must not come back as an economy, and its spec must say why it went."""
    assert "vyxT860" not in CURRENT, "vyxT860 is back in CURRENT; it is not an economy"
    spec = load_spec("var-kp_vy-vyxT860-v1")
    assert spec.get("lineage", {}).get("retired"), (
        "var-kp_vy-vyxT860-v1 must record its retirement in lineage.retired")
    assert "KAPPAS" not in seed_array() or "vyxT860)" not in seed_array(), \
        "the seed array still has a vyxT860 case"


# --------------------------------------------------------------------------
# solve-side precision: uniform within each model
# --------------------------------------------------------------------------

# The numerical (not economic) entries of each model's solve params. A solve whose grid or
# tolerance differs from its siblings' is measured differently, whatever its economy.
PRECISION_KEYS = {
    "kp": ("NY", "_i0"),
    "gs": ("bnum", "znum", "xnum", "imin", "imax", "tol"),
    "bgn": ("I", "JSTAR_TOL"),
}


def _family(params):
    return "kp" if "NY" in params else ("gs" if "bnum" in params else "bgn")


def test_solve_precision_is_uniform_within_each_model():
    """Read from the manifests, which record what each solve was actually built from.

    This is the axis protocol.py cannot own: KP14's y-node count, GS21's grid sizes and
    fixed-point tolerance and BGN's J* tolerance are not comparable quantities. What is
    checkable, and what matters, is that every economy of a given model shares them.
    """
    seen = {}
    for fn in sorted(os.listdir(REGISTRY)):
        if not fn.endswith(".json"):
            continue
        man = json.load(open(os.path.join(REGISTRY, fn)))
        params = man.get("params") or {}
        if not params:
            continue
        fam = _family(params)
        env = man.get("env_params") or {}
        view = {}
        for k in PRECISION_KEYS[fam]:
            v = params.get(k, env.get(k))
            if v is not None:
                view[k] = str(v)
        seen.setdefault(fam, {}).setdefault(json.dumps(view, sort_keys=True), []).append(fn[:-5])
    bad = []
    for fam, groups in seen.items():
        if len(groups) > 1:
            bad.append(f"{fam}: {len(groups)} distinct precision settings across its solves")
            for view, ids in sorted(groups.items()):
                bad.append(f"    {view}  <- {', '.join(sorted(ids))}")
    assert not bad, "solve precision is not uniform within a model:\n  " + "\n  ".join(bad)


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
