"""What code and what parameters produced this file.

THE GOAL, in Seth's words: (a) I see some results are good and think "let's pursue
this further"; (b) I know what code and parameters produced those results.

Nothing in this repo recorded (b) before 2026-09-08. `solstamp` content-addresses
SOLVES so a six-hour solve can be reused safely, and `runstamp` links a run to the
solves it consumed -- both real, both about caching and staleness. Neither records the
CODE VERSION, and the summary CSVs you actually read carried no link to anything at
all -- including the published `grid_summary.csv`, which was deleted on 2026-09-10
precisely because nothing it contained could be traced back to a commit or a spec.

This module is the whole of (b), and it is deliberately small. A sidecar next to every
summary:

    read the sidecar -> git checkout the sha -> set the env -> run the argv

No registry, no cross-referencing, no hash to verify.

WHY THIS DOES NOT REPLACE solstamp. A git sha identifies the whole repo, which is right
for reproducing a run and wrong for deciding whether a cached solve is stale. Measured
2026-09-08: 207 commits in this repo, SIX of which touched gs_solve_reg.py. A sha-based
staleness check refuses 207 times where content-addressing refuses 6 -- a 34x false
invalidation rate against a solve costing ~5 h per type. The two are not substitutes:
this answers "what made this result", solstamp answers "may I reuse those bytes".
With this file in place solstamp stops being the provenance story and becomes a cache
key nobody has to look at.
"""
import json
import os
import subprocess
import sys
import time

# Env vars that change what a run computes. ANCHORED, not substring: an unanchored
# "_PREFIX" captured HOMEBREW_PREFIX and GSETTINGS_SCHEMA_DIR, which say nothing about
# the run and bury the two lines that do.
ENV_SUFFIXES = ("_OVERRIDES", "_NUM_THREADS")
ENV_PREFIXES = ("GS_", "KP_", "BGN_", "SEED_")
ENV_EXACT = ("CONDA_ENV", "CONDA_DEFAULT_ENV", "VECLIB_MAXIMUM_THREADS",
             # PYTHONHASHSEED earns its place: str hashing randomisation is what made
             # kp_vy's solve_id non-reproducible across processes on 2026-09-07.
             "PYTHONHASHSEED",
             "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID",
             "SLURM_CPUS_PER_TASK", "SLURMD_NODENAME")
DIFF_CAP = 200_000
# Paths whose contents are never inlined into a sidecar's diff: they are outputs, and a
# diff of outputs cannot help reproduce a run. Their names still appear in `status`.
OUTPUT_DIRS = ("variants/results",)


def _git(args, cwd, default=None):
    """Run a git command; never raise. Provenance must not be able to fail a run."""
    try:
        out = subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else default
    except Exception:
        return default


def git_state(cwd=None):
    """sha + branch + working-tree state. A dirty tree stores its own diff inline.

    An untracked .py file is a genuine hole: it is not in the sha and not in the diff,
    so it is listed explicitly rather than passed over. `git checkout <sha>` does not
    restore it and the sidecar says so.
    """
    cwd = cwd or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = _git(["rev-parse", "--show-toplevel"], cwd)
    if root is None:
        return {"sha": None, "note": "not a git repository, or git unavailable"}
    st = {
        "sha": _git(["rev-parse", "HEAD"], root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "root": root,
    }
    porcelain = _git(["status", "--porcelain"], root, default="")
    st["dirty"] = bool(porcelain)
    if porcelain:
        st["status"] = porcelain.splitlines()[:200]
        # The inline diff exists so a dirty tree's CODE can be reproduced. Outputs are
        # not code: on 2026-09-08 the cluster tree was dirty only because the array had
        # overwritten three tracked result files, and every one of thirty sidecars
        # inlined a ~200 KB diff of OTHER runs' results. `status` still lists them, so
        # the fact of the dirty tree is recorded; only their contents are left out.
        diff = _git(["diff", "HEAD", "--"] + [f":(exclude){d}" for d in OUTPUT_DIRS],
                    root, default="")
        st["diff_truncated"] = len(diff) > DIFF_CAP
        st["diff"] = diff[:DIFF_CAP]
        untracked = [l[3:] for l in porcelain.splitlines() if l.startswith("??")]
        code = [u for u in untracked if u.endswith((".py", ".sh"))]
        if code:
            st["untracked_code"] = code
            st["untracked_code_warning"] = (
                "these files are NOT in the sha and NOT in the diff; `git checkout` "
                "will not restore them and this run is not fully reproducible from "
                "the sha alone")
    return st


def environment():
    import platform
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "hostname": platform.node()}
    for name in ("numpy", "scipy", "pandas", "sklearn", "pyarrow"):
        mod = sys.modules.get(name)
        if mod is not None:
            env[name] = getattr(mod, "__version__", "?")
    return env


def relevant_env():
    def keep(k):
        return (k in ENV_EXACT or k.startswith(ENV_PREFIXES) or k.endswith(ENV_SUFFIXES))
    return {k: v for k, v in sorted(os.environ.items()) if keep(k)}


def run_provenance(inputs=None, extra=None):
    """Everything needed to re-run this, as a plain dict."""
    g = git_state()
    # `run_from` is the MAIN SCRIPT's directory, not getcwd(). run_oracle.py chdirs into
    # the model directory (variants/kp_vy) partway through, so a cwd captured at write
    # time says "cd variants/kp_vy" for a run launched from variants/ -- and run_oracle.py
    # is not in kp_vy, so following it fails. argv[0] is resolved against the launch
    # directory, so the script's own location is what pairs with argv.
    main = sys.modules.get("__main__")
    mf = getattr(main, "__file__", None)
    run_from = os.path.dirname(os.path.abspath(mf)) if mf else os.getcwd()
    if g.get("root"):
        run_from = os.path.relpath(run_from, g["root"])
    prov = {
        "schema": 1,
        "git": g,
        "argv": list(sys.argv),
        "run_from": run_from,
        "cwd_at_write": (os.path.relpath(os.getcwd(), g["root"])
                         if g.get("root") else os.getcwd()),
        "env": relevant_env(),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment(),
        "how_to_reproduce": (
            "git checkout %s && cd %s && <env below> python %s"
            % (g.get("sha") or "<unknown>", run_from, " ".join(sys.argv))),
    }
    if inputs:
        prov["inputs"] = inputs
    if extra:
        prov["extra"] = extra
    return prov


def short_tag(prov, stem):
    """The value that goes in a summary's `prov` column.

    Every row carries it, which is redundant on purpose: the workflow is reading one
    good row and wanting its code, and a row copied into a notebook must not lose the
    pointer. `stem` names the sidecar, the sha names the code, `+dirty` says the sha
    alone is not the whole story.
    """
    sha = (prov.get("git", {}).get("sha") or "nogit")[:7]
    return "%s@%s%s" % (stem, sha, "+dirty" if prov.get("git", {}).get("dirty") else "")


def write_sidecar(target, inputs=None, extra=None):
    """Write `<target>.prov.json` beside a summary. Returns (prov, short_tag).

    `target` is the summary itself, so the sidecar sorts next to it and is obvious in
    a directory listing.
    """
    prov = run_provenance(inputs=inputs, extra=extra)
    stem = os.path.basename(target)
    for ext in (".csv", ".json", ".parquet"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    try:
        with open(target + ".prov.json", "w") as fh:
            json.dump(prov, fh, indent=1)
    except Exception as e:                      # never fail a six-hour run over this
        print("[prov] WARNING could not write sidecar: %s" % e, flush=True)
    return prov, short_tag(prov, stem)
