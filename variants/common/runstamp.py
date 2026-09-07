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

def live_solves(model, tag):
    """Non-retired solve_ids recorded for this model+tag, newest-registry order."""
    return sorted(m["solve_id"] for m in solstamp.iter_manifests()
                  if m.get("model") == model and not m.get("retired")
                  and tag in (m.get("tags") or []))


def run_is_current(run_json, model, tag):
    """True when `run_json` was produced from the solves currently live for model+tag.

    This is what makes a seed array restartable without re-running finished seeds,
    and it is keyed on the SOLVE, not on file existence: re-solving the economy
    invalidates every seed at once, which is the behaviour a bare `[ -f ... ]` guard
    cannot give (see run_gs_bx7.sh's removed existence check).
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
    p.add_argument("--tag", required=True)

    p = sub.add_parser("is-current", help="exit 0 if a run record matches those solves")
    p.add_argument("run_json")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)

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
