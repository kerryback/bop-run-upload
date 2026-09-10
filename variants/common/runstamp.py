"""The run side of provenance: which solve produced the economy a result describes.

`solstamp.py` addresses the SOLVE -- parameters and code in, solve_id and manifest
out. That is half of the link. The other half was missing: `run_oracle.py` and
`run_estimators.py` read G tables, J* tables and solution.npz files, wrote a summary
JSON, and recorded nothing about which solve those artifacts came from. So a summary
could not be traced back to its economy once the panels were purged from scratch --
and at 33.8 MB per panel they will be purged.

The link is made by CONTENT, not by trust. This module hashes the artifacts the run
actually read and asks the registry which manifest describes those exact bytes. A run
that silently picked up a stale or foreign table therefore records the solve_id of
what it really read, or `null` if nothing in the registry matches -- which is itself
the finding.

    from common import runstamp

    solves = runstamp.consumed_solves("kp_vy")     # after the panel is built
    summary["solves"] = solves

Naming lives here too, because the seed has to reach the filenames: a replication
array writes one panel per seed, and `{model}_panel_{tag}.parquet` has room for
exactly one.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from common import solstamp  # noqa: E402

VARIANTS_DIR = solstamp.VARIANTS_DIR


# ------------------------------------------------------------------ naming ----

def stem(model, kind, tag, seed):
    """Result-file basename, seed included.

    The seed suffix is UNCONDITIONAL, including for seed 0. Suffixing only the
    non-zero seeds would make `..._vyx.parquet` ambiguous between "seed 0" and
    "written before seeds existed", and those are different things: everything
    written before 2026-09-07 came from the pre-correction parameters (see
    WORKING.md §25) and must not be silently adopted as a replication.
    """
    return f"{model}_{kind}_{tag}_s{int(seed):03d}"


# ------------------------------------------------- what the run just read ----

def _kp_paths(mod):
    prefix = os.environ.get("KP_VY_PREFIX", "vys")
    d = os.path.join(VARIANTS_DIR, "kp_vy")
    ntypes, NY = int(mod.ntypes), int(mod.NY)
    return [
        ("G", [os.path.join(d, f"G_{prefix}{f}.csv") for f in range(ntypes)]),
        ("integ", [os.path.join(d, f"integ_{prefix}{f}_{iy}.npz")
                   for f in range(ntypes) for iy in range(NY)]),
    ]


def _bgn_paths(mod):
    d = os.path.join(VARIANTS_DIR, "bgn_gam")
    return [("jstar", [os.path.join(d, mod.jstar_gam_file)])]


def _gs_paths(mod):
    d = os.path.join(VARIANTS_DIR, "gs_bx")
    dirs = os.environ.get("GS_BX_SOLDIRS", "sol_reg,sol_bx3,sol_bx5").split(",")
    # One manifest PER exposure type, so each solution.npz is its own group.
    return [(s.strip(), [os.path.join(d, s.strip(), "solution.npz")])
            for s in dirs if s.strip()]


_PATHS = {"kp_vy": _kp_paths, "bgn_gam": _bgn_paths, "gs_bx": _gs_paths}


def _match(paths):
    """The manifest describing these exact bytes, preferring a live one.

    A retired manifest still describes real artifacts truthfully -- it is only its
    ID that can no longer be recomputed -- so it is reported when nothing live
    matches, flagged rather than hidden.
    """
    present = [p for p in paths if os.path.exists(p)]
    if not present:
        return None, "artifacts absent"
    live, retired = None, None
    want = {}
    for p in present:
        ap = os.path.abspath(p)
        rel = (os.path.relpath(ap, solstamp.REPO_DIR)
               if ap.startswith(solstamp.REPO_DIR + os.sep) else ap)
        want[rel] = solstamp.file_digest(ap)
    for man in solstamp.iter_manifests():
        got = {a["path"]: a["sha256"] for a in man.get("artifacts", [])}
        if not got or any(got.get(k) != v for k, v in want.items()):
            continue
        if man.get("retired"):
            retired = retired or man
        else:
            live = live or man
    if live is not None:
        return live, None
    if retired is not None:
        return retired, ("matches only a RETIRED manifest: its solve_id cannot be "
                         "recomputed, so re-solving will not reproduce this id")
    return None, ("no manifest describes these bytes -- the artifacts were built by "
                  "an unrecorded parameter/code combination")


def consumed_solves(model, mod=None):
    """[{stage, solve_id, files, bytes, warning}] for the artifacts this run read.

    `mod` is the model's parameter namespace (kp_vy needs ntypes/NY, bgn_gam needs
    jstar_gam_file). Defaults to the already-imported module.
    """
    if mod is None:
        mod = sys.modules["__main__"]
    out = []
    for stage, paths in _PATHS[model](mod):
        man, warning = _match(paths)
        present = [p for p in paths if os.path.exists(p)]
        rec = {"stage": stage,
               "solve_id": man["solve_id"] if man else None,
               "files": len(present),
               "files_expected": len(paths),
               "bytes": sum(os.path.getsize(p) for p in present)}
        if man and man.get("retired"):
            rec["retired"] = True
        if warning:
            rec["warning"] = warning
        if len(present) != len(paths):
            rec["warning"] = ((rec.get("warning") + "; ") if rec.get("warning") else "") \
                + f"{len(paths) - len(present)} expected artifact(s) missing"
        out.append(rec)
    return out


def describe(solves):
    """One line per stage, for a run log."""
    lines = []
    for s in solves:
        sid = s["solve_id"] or "UNREGISTERED"
        flag = "  [RETIRED]" if s.get("retired") else ""
        lines.append(f"[runstamp] {s['stage']:8s} <- solve_id {sid}  "
                     f"({s['files']}/{s['files_expected']} files, "
                     f"{s['bytes']:,} B){flag}")
        if s.get("warning"):
            lines.append(f"[runstamp]          WARNING: {s['warning']}")
    return "\n".join(lines)


# ------------------------------------------------------------ checkpointing ----

def solve_tags(tag):
    """'sol_reg,sol_b25c' / ['sol_reg', 'sol_b25c'] / 'vyx' -> a list of tags.

    A multi-solve economy (gs_bx bx7: five exposure types, five solution.npz, five
    manifests each under its own soldir tag) is ONE economy and must be looked up as one.
    The seed array passes SOLVE_TAG through unchanged, so the comma list is the contract
    at the shell boundary.
    """
    if isinstance(tag, str):
        tag = tag.split(",")
    return [t.strip() for t in tag if t and t.strip()]


def live_solves(model, tag):
    """Non-retired solve_ids recorded for this model under ANY of `tag`'s tags, sorted.

    `tag` is a str (comma-separated for several) or a list; one tag reproduces the
    pre-2026-09-09 behaviour exactly. Until then this took a single tag, so for bx7 --
    whose run record names FIVE solve_ids -- run_is_current compared five ids against the
    one id of whichever tag it was handed and reported STALE forever: every resubmission
    re-ran every finished seed, the opposite of what a checkpoint is for.
    """
    want = set(solve_tags(tag))
    return sorted({m["solve_id"] for m in solstamp.iter_manifests()
                   if m.get("model") == model and not m.get("retired")
                   and want & set(m.get("tags") or [])})


def run_is_current(run_json, model, tag):
    """True when `run_json` was produced from the solves currently live for model+tag.

    This is what makes a seed array restartable without re-running finished seeds,
    and it is keyed on the SOLVE, not on file existence: re-solving the economy
    invalidates every seed at once, which is the behaviour a bare `[ -f ... ]` guard
    cannot give (see run_gs_bx7.sh's removed existence check).

    `tag` may name several tags (see live_solves); the run is current when the ids it
    consumed are exactly the union of the live ids under those tags.
    """
    import json as _json
    if not os.path.exists(run_json):
        return False, "no run record"
    try:
        with open(run_json) as f:
            rec = _json.load(f)
    except (OSError, ValueError) as e:
        return False, f"unreadable run record ({e})"
    got = sorted(s["solve_id"] for s in (rec.get("solves") or []) if s.get("solve_id"))
    want = live_solves(model, tag)
    if not got:
        return False, "run record names no solve_id"
    if got != want:
        return False, f"built from {', '.join(got)}; registry now has {', '.join(want) or 'nothing'}"
    return True, f"current for {', '.join(want)}"


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("current", help="live solve_ids for a model+tag")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True,
                   help="solve tag; comma-separated for a multi-solve economy")

    p = sub.add_parser("is-current", help="exit 0 if a run record matches those solves")
    p.add_argument("run_json")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True,
                   help="solve tag; comma-separated for a multi-solve economy")

    a = ap.parse_args(argv)
    if a.cmd == "current":
        ids = live_solves(a.model, a.tag)
        print("\n".join(ids) if ids else "")
        return 0 if ids else 1
    ok, why = run_is_current(a.run_json, a.model, a.tag)
    print(("CURRENT: " if ok else "STALE: ") + why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))


# -------------------------------------------------------- spec -> run link ----

SPECS_DIR = os.path.join(solstamp.REPO_DIR, "experiments", "specs")


def load_spec(spec_id):
    import json as _json
    path = os.path.join(SPECS_DIR, spec_id + ".json")
    if not os.path.exists(path):
        raise SystemExit(f"no spec {spec_id} in {SPECS_DIR}")
    with open(path) as f:
        return _json.load(f)


def verify_against_spec(spec_id, solves):
    """Do the solves this run consumed match the ones its spec declares?

    Completes the chain the registry only half-closed. `solstamp` says which
    parameters produced an artifact; `consumed_solves` says which artifact a run
    read; this says whether that is the economy the spec claims to describe.

    Returns (ok, lines). A spec with no `expected_solves` (every v1) cannot be
    checked and says so rather than passing silently -- a spec that pins nothing
    must not read as a spec that pins something and agrees.
    """
    spec = load_spec(spec_id)

    # Supersession is checked BEFORE expected_solves. It used to sit after the early
    # return below, which made it unreachable for every v1 -- exactly the specs most
    # likely to be superseded, since v1s predate expected_solves entirely. Same
    # ordering bug as retired-before-superseded in solfiles.cmd_check.
    #
    # And it REFUSES rather than notes. A superseded spec describes an economy this
    # code no longer builds: var-kp_vy-vyx-v1 declares kp_regime_labels
    # "legacy_swapped", but that fix was made by editing parameters_kp14.py in place
    # rather than behind a `method` switch, so running v1's spec today silently builds
    # v2's economy. Nothing in `method` is executable (no python file reads any of the
    # seven keys), so the spec cannot restore the economy it names.
    sup = spec.get("lineage", {}).get("superseded_by")
    if sup:
        return False, [f"[spec] {spec_id} is SUPERSEDED by {sup}, and describes an "
                       f"economy this code no longer builds. Use {sup}."]

    want = spec.get("expected_solves")
    if not want:
        return None, [f"[spec] {spec_id} declares no expected_solves -- nothing to "
                      f"verify against (v1 specs predate the field)"]
    got = {s["stage"]: s["solve_id"] for s in solves}
    lines, ok = [], True
    for stage in sorted(set(want) | set(got)):
        w, g = want.get(stage), got.get(stage)
        if w == g and w is not None:
            lines.append(f"[spec] {stage:8s} {g}  matches {spec_id}")
        else:
            ok = False
            lines.append(f"[spec] {stage:8s} MISMATCH: spec {spec_id} expects "
                         f"{w or '<not declared>'}, run consumed {g or '<nothing>'}")
    return ok, lines


# ------------------------------------------------- spec -> RUN PARAMETERS ----

def _env_expectations(spec, ov_env):
    """(literal_env, param_env_name) -- what the environment must look like.

    Two representations coexist in the specs and they mean different things:

      * `param_env` names an environment variable that carries `params` as JSON.
        bgn_gam and kp_vy use it: BGN_PARAM_OVERRIDES / KP_PARAM_OVERRIDES are
        read by the SIMULATOR, so `params` is a run-side quantity there.
      * `env` is a flat map of variable -> literal value (KP_VY_PREFIX,
        GS_BX_SOLDIRS, ...).

    gs_bx has NO `param_env`, and that is not an omission: its `params`
    ({"gmreg": [0.6, 3.0]}) is GS_PARAM_OVERRIDES, consumed by the SOLVER and
    already covered by the solve_id. Its simulator takes the structural
    parameters out of solution.npz instead. Comparing gs_bx's `params` against
    GS_SIM_OVERRIDES would refuse every correct run.
    """
    return dict(spec.get("env") or {}), spec.get("param_env")


def _same_json(a, b):
    """Compare two override blobs by VALUE, not by spelling.

    '{"gmult":[0.2,3.5]}' and '{"gmult": [0.2, 3.5]}' are the same economy;
    whitespace and key order must not be able to refuse a correct run.
    """
    import json as _json
    try:
        return _json.loads(a or "{}") == _json.loads(b or "{}")
    except ValueError:
        return False


def verify_env_against_spec(spec_id, ov_env, env=None):
    """Do this run's PARAMETERS match the ones its spec declares?

    `verify_against_spec` closes artifact -> spec. This closes parameters -> spec,
    which was the last open link in params -> solve -> result and the one the whole
    registry exists to protect.

    The hole it fills, reproduced 2026-09-09: simulate BGN at gmult [0.2, 3.0] while
    reading the J* table built at [0.2, 3.5], with --spec var-bgn_gam-g0235-v2. The
    table is genuinely the spec's table, so the solve check PASSES, and the run
    records spec_check "verified" on a summary describing a different economy. The
    wrong overrides did reach the sidecar, so the mistake was discoverable after the
    fact -- but nothing refused it, and a wrong number that has been written down is
    already the expensive kind.

    Returns (ok, lines) with the same three-valued convention as
    verify_against_spec: True verified, False refused, None nothing to check.
    """
    import json as _json
    if env is None:
        env = os.environ
    spec = load_spec(spec_id)
    literal, param_env = _env_expectations(spec, ov_env)
    lines, ok, checked = [], True, 0

    if param_env:
        want = spec.get("params")
        got_raw = env.get(param_env)
        checked += 1
        if want is None:
            lines.append(f"[env] {spec_id} names param_env {param_env} but declares no "
                         f"params -- cannot verify")
            ok = False
        elif _same_json(got_raw, _json.dumps(want)):
            lines.append(f"[env] {param_env:22s} matches {spec_id}")
        else:
            ok = False
            lines.append(f"[env] {param_env:22s} MISMATCH")
            lines.append(f"[env]   spec {spec_id} declares: {_json.dumps(want, sort_keys=True)}")
            lines.append(f"[env]   this run has:            {got_raw or '<unset>'}")
            try:
                g = _json.loads(got_raw or "{}")
                for k in sorted(set(g) | set(want)):
                    if g.get(k) != want.get(k):
                        lines.append(f"[env]     {k}: spec {want.get(k, '<absent>')!r} "
                                     f"vs run {g.get(k, '<absent>')!r}")
            except ValueError:
                lines.append("[env]     (run value is not valid JSON)")

    for k in sorted(literal):
        want_s, got_s = str(literal[k]), env.get(k)
        checked += 1
        # A declared value that looks like a JSON object is compared as one, so an
        # unset variable reads as {} rather than as a refusal. Everything else --
        # GS_BX_SOLDIRS, GS_BX_BETAS, KP_VY_PREFIX -- is an exact string, where
        # being unset IS the mismatch.
        hit = _same_json(got_s, want_s) if want_s.lstrip().startswith("{") else got_s == want_s
        if hit:
            lines.append(f"[env] {k:22s} matches {spec_id}")
        else:
            ok = False
            lines.append(f"[env] {k:22s} MISMATCH: spec expects {want_s!r}, "
                         f"run has {got_s if got_s is not None else '<unset>'!r}")

    # The run-side override variable must be ACCOUNTED FOR, not merely absent from
    # the spec. gs_bx reaches here with param_env None and GS_SIM_OVERRIDES not in
    # `env`; unset or {} is the correct state for it, because the simulator takes its
    # structural parameters out of solution.npz. But GS_SIM_OVERRIDES='{"gamma_x":0.9}'
    # would silently overwrite a value that came from the solution -- undeclared, and
    # invisible to both checks above. Refuse it rather than let it through as "the
    # spec said nothing about that".
    if ov_env and ov_env != param_env and ov_env not in literal:
        raw = env.get(ov_env)
        if not _same_json(raw, "{}"):
            ok = False
            lines.append(f"[env] {ov_env:22s} REFUSED: set to {raw!r}, but {spec_id} "
                         f"declares neither param_env nor env[{ov_env}] for it. An "
                         f"undeclared override changes the economy without changing "
                         f"the spec. Declare it in the spec or unset it.")
        checked += 1

    if not checked:
        return None, [f"[env] {spec_id} declares no params, param_env or env -- "
                      f"nothing to verify the run parameters against"]
    return ok, lines
