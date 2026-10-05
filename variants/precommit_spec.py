#!/usr/bin/env python3
"""Compute a spec's solve ids WITHOUT polluting the repo, so the spec can be committed first.

WHY THIS EXISTS. The discipline is: pin the ids a spec expects, commit the spec, THEN
build. `tests/test_precommitment_is_real.py` checks that order by git dates, because a
pinned id read off an existing manifest proves nothing. Step 2 of "Adding an experiment"
in docs/RESULTS.md says to compute the ids "WITHOUT solving", and for most stages that
is literally true -- a solve_id is sha256(parameters + source digests), and every
producer prints it with flush=True BEFORE it starts solving.

ONE STAGE IS NOT LIKE THE OTHERS. KP14's integral stage hashes the G tables' RAW BYTES
(`build_vy_tables.py`, `inputs=solstamp.artifact_digests(G_ARTIFACTS)`), so its id
cannot be known until the G tables exist. That is not an oversight -- it is what makes
the chain tamper-evident -- but it means precommitting a KP14 economy requires building
G first, and building G writes manifests into experiments/registry/ and tables into
variants/kp_vy/, which is exactly the pollution the precommitment order exists to avoid.

So: run the producer inside a THROWAWAY GIT WORKTREE. Its REPO_DIR is elsewhere, so
solstamp writes its registry and artifacts there, and the whole tree is deleted
afterwards. The ids are real -- same sources, same parameters, same arithmetic -- and
nothing reaches the working tree. This was done by hand with ad-hoc scripts under
_scratch/ for vym3 and vym3t3; this is that procedure, written down and repeatable.

usage:
    python variants/precommit_spec.py experiments/specs/var-kp_vy-foo-v1.json
        Every id that is free. Stops each producer the moment it has printed what it
        can, so nothing is actually solved. A chained stage is reported as PENDING.

    python variants/precommit_spec.py <spec> --build-chained
        Also pays for the upstream build a chained stage needs (KP14: ~20 min per G
        type) inside the throwaway tree, to get the downstream id.

    python variants/precommit_spec.py <spec> --build-chained --write
        ...and writes expected_solves into the spec, recomputing spec_hash.

Then: commit the spec, build for real, and check each printed id against its pin.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# A solve_id line, from any of the three producers:
#   [solstamp] kp_vy prefix=vym3 stage=G solve_id=dd23d7300cd54a0d
#   [solstamp] bgn_gam Jstar_bgnbase.csv solve_id=cb6340649eace098
#   [solstamp] gs_bx sol_reg solve_id=63fa7ebbc2db49ea
_ID = re.compile(r"solve_id=([0-9a-f]{16})\b")
_STAGE = re.compile(r"\bstage=(\w+)")

# How each producer takes its positional arguments. Adding a model means adding a row,
# which is deliberate: this file should not guess at a new producer's CLI.
#
# The spec format is not uniform here and both shapes are live: vyg25 and kpbase declare
# ONE stage "vy_tables" with args {"prefix": ...}, while vym3 declares TWO stages "G" and
# "integ" with args {"KP_VY_PREFIX": ...}. Both describe the same single invocation of
# build_vy_tables.py, which emits both ids. So the prefix is looked for in either place
# and in spec.env, and invocations are de-duplicated before anything is run.
def _argv_for(producer, stage, spec):
    args = stage.get("args") or {}
    base = os.path.basename(producer)
    if base == "build_vy_tables.py":
        prefix = (args.get("prefix") or args.get("KP_VY_PREFIX")
                  or (spec.get("env") or {}).get("KP_VY_PREFIX"))
        if not prefix:
            raise SystemExit(f"{spec['spec_id']}: cannot find the KP prefix in the "
                             f"stage's args or the spec's env")
        return [str(prefix)]
    if base in ("gs_solve_reg.py", "gs_solve_gam.py"):
        return [str(args.get("xnum", 161)), str(args.get("tol", 1e-6)), stage["name"]]
    if base == "rebuild_jstar_gam.py":
        return []
    raise SystemExit(f"precommit_spec does not know how to invoke {base}. Add it to "
                     f"_argv_for() rather than guessing.")


# Stages whose id depends on an upstream artifact's bytes, and so cannot be known until
# that artifact exists. Keyed by producer basename -> the expected_solves keys that are
# free, and the ones that are not.
CHAINED = {"build_vy_tables.py": {"free": {"G"}, "chained": {"integ"}}}


def _git(args, cwd=ROOT, check=True):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout.strip()


def _refuse_if_sources_dirty():
    """The id must be the one the COMMITTED sources produce, or the pin is a fiction.

    The worktree is created from HEAD, so an uncommitted edit to a producer would be
    invisible here and the id would be computed from code that is not what lands. That
    is the 2026-09-08 failure shape in reverse, so it is refused rather than warned.
    """
    lines = [ln for ln in _git(["status", "--porcelain", "--", "variants"]).splitlines()
             if ln.strip()]
    # Untracked files are allowed: a solve_id digests only the files a producer names in
    # its SOURCES list, all of which are tracked, so a new file beside them cannot move
    # an id. A MODIFIED tracked file can and does.
    dirty = "\n".join(ln for ln in lines if not ln.startswith("??"))
    if dirty:
        print("REFUSED: variants/ has uncommitted changes to tracked files, so the ids "
              "computed from HEAD would not be the ids the committed sources produce.\n",
              file=sys.stderr)
        print(dirty, file=sys.stderr)
        print("\nCommit the producer change first, then precommit the spec against it.",
              file=sys.stderr)
        raise SystemExit(2)


def _run_stage(wt, spec, stage, want, build_chained):
    """Run one stage's producer in the worktree; return {expected_solves key: solve_id}."""
    producer = os.path.join(wt, stage["producer"])
    cmd = [sys.executable, producer] + _argv_for(stage["producer"], stage, spec)

    env = dict(os.environ)
    env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
    if stage.get("param_env"):
        env[stage["param_env"]] = json.dumps(stage.get("params") or {})
    env["PYTHONUNBUFFERED"] = "1"

    base = os.path.basename(stage["producer"])
    chain = CHAINED.get(base)
    # what this invocation can yield without paying for a build
    free = chain["free"] if chain else {stage["name"]}
    target = set(want) if (build_chained or not chain) else set(free)

    print(f"  $ {stage['param_env'] or 'ENV'}=... {os.path.basename(producer)} "
          f"{' '.join(_argv_for(stage['producer'], stage, spec))}")
    found = {}
    p = subprocess.Popen(cmd, cwd=wt, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            m = _ID.search(line)
            if m:
                sm = _STAGE.search(line)
                key = sm.group(1) if sm else stage["name"]
                found[key] = m.group(1)
                print(f"      {key:10s} {m.group(1)}")
            elif build_chained and line.strip():
                print(f"      | {line.rstrip()[:100]}")
            if target and target <= set(found):
                break                      # got everything this run was asked for
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        if p.stdout:
            p.stdout.close()
    return found


def _spec_hash(spec):
    """Recompute exactly as tests/test_specs_match_shell.py does."""
    excluded = {"title", "question", "notes", "lineage", "provenance", "spec_hash",
                "solves_pending", "precommitted", "reused_solves"}
    view = {k: v for k, v in spec.items() if k not in excluded}
    return hashlib.sha256(
        json.dumps(view, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="path to the spec JSON, or a bare spec_id")
    ap.add_argument("--build-chained", action="store_true",
                    help="pay for the upstream build a chained stage needs (KP14 integ)")
    ap.add_argument("--write", action="store_true",
                    help="write expected_solves into the spec and recompute spec_hash")
    ap.add_argument("--keep-worktree", action="store_true",
                    help="leave the throwaway tree in place for inspection")
    args = ap.parse_args()

    path = args.spec
    if not os.path.exists(path):
        path = os.path.join(ROOT, "experiments", "specs", args.spec + ".json")
    if not os.path.exists(path):
        raise SystemExit(f"no such spec: {args.spec}")
    spec = json.load(open(path))
    stages = (spec.get("solve") or {}).get("stages") or []
    if not stages:
        raise SystemExit(f"{spec['spec_id']} declares no solve stages; nothing to pin")

    _refuse_if_sources_dirty()

    head = _git(["rev-parse", "--short", "HEAD"])
    wt = os.path.join(ROOT, "_scratch", f"precommit_wt_{spec['spec_id']}")
    if os.path.exists(wt):
        _git(["worktree", "remove", "--force", wt], check=False)
        shutil.rmtree(wt, ignore_errors=True)
    os.makedirs(os.path.join(ROOT, "_scratch"), exist_ok=True)

    print(f"{spec['spec_id']}  model={spec['model']}  sources at HEAD {head}")
    print(f"throwaway worktree: {os.path.relpath(wt, ROOT)}\n")
    _git(["worktree", "add", "--detach", wt, "HEAD"])

    ids, pending = {}, []
    try:
        seen = set()
        for stage in stages:
            base = os.path.basename(stage["producer"])
            sig = (stage["producer"],
                   json.dumps(stage.get("params") or {}, sort_keys=True),
                   json.dumps(_argv_for(stage["producer"], stage, spec)))
            if sig in seen:
                continue              # another stage name for the same producer run
            seen.add(sig)
            chain = CHAINED.get(base)
            want = ((chain["free"] | chain["chained"]) if chain else {stage["name"]})
            if chain and not args.build_chained:
                print(f"[{base}] {sorted(chain['chained'])} depends on the bytes this "
                      f"same run produces -- re-run with --build-chained to pay for it")
                pending.extend(sorted(chain["chained"]))
            ids.update(_run_stage(wt, spec, stage, want, args.build_chained))
    finally:
        if not args.keep_worktree:
            _git(["worktree", "remove", "--force", wt], check=False)
            shutil.rmtree(wt, ignore_errors=True)
            print(f"\nworktree discarded (its registry and tables went with it)")
        else:
            print(f"\nworktree kept at {wt}")

    print("\nexpected_solves:")
    print(json.dumps(ids, indent=2, sort_keys=True))
    still = [k for k in pending if k not in ids]
    if still:
        print(f"\nSTILL PENDING: {still} -- re-run with --build-chained")

    prior = spec.get("expected_solves") or {}
    if prior:
        drift = {k: (prior.get(k), v) for k, v in ids.items() if prior.get(k) not in (None, v)}
        if drift:
            print("\n!! the spec already pins different ids:")
            for k, (was, now) in drift.items():
                print(f"   {k}: spec says {was}, this run computes {now}")
            print("   That is a real disagreement -- do not overwrite it without "
                  "understanding why (variants/solve_impact.py).")

    if args.write:
        if still:
            raise SystemExit("\nrefusing --write while a chained id is still pending")
        spec["expected_solves"] = dict(sorted({**prior, **ids}.items()))
        spec["solves_pending"] = sorted(spec["expected_solves"])
        spec["precommitted"] = True
        spec["spec_hash"] = _spec_hash(spec)
        with open(path, "w") as fh:
            json.dump(spec, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {os.path.relpath(path, ROOT)}  spec_hash={spec['spec_hash'][:16]}...")
        print("NOW: commit this spec BEFORE building, or the precommitment is not one.")


if __name__ == "__main__":
    main()
