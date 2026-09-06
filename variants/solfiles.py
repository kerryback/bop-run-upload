#!/usr/bin/env python3
"""Browse the solve registry: what has been solved, with what parameters, and is it still here.

Every expensive solve (BGN J* tables, KP G/integral tables, GS solutions) records a
manifest in experiments/solfiles/<solve_id>.json keyed by a content hash of its
parameters and producer source. Manifests are small and committed, so the record of
"which parameters produced this artifact" survives even after the artifacts are
purged from scratch.

    python variants/solfiles.py list                 # every recorded solve
    python variants/solfiles.py list --model kp_vy   # filter
    python variants/solfiles.py show <solve_id>      # full parameters
    python variants/solfiles.py diff <id_a> <id_b>   # what changed between two solves
    python variants/solfiles.py check                # are the artifacts still on disk and intact
    python variants/solfiles.py gc --dry-run         # what could be deleted and reclaimed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import solstamp  # noqa: E402


def _hsize(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n/1:,.1f} {unit}"
        n /= 1024.0


def _size(n):
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n/div:.1f} {unit}"
    return f"{n} B"


def cmd_list(args):
    rows = [m for m in solstamp.iter_manifests()
            if not args.model or m.get("model") == args.model]
    if not rows:
        print("no solves recorded yet (experiments/solfiles/ is empty)")
        return 0
    print(f"{'solve_id':18s} {'model':10s} {'size':>10s} {'git':4s} {'files':>6s}  tags / specs")
    print("-" * 88)
    total = 0
    for m in sorted(rows, key=lambda m: (m.get("model") or "", m["solve_id"])):
        total += m.get("total_bytes", 0)
        label = ", ".join(m.get("tags", []) + m.get("spec_ids", [])) or "-"
        print(f"{m['solve_id']:18s} {(m.get('model') or '?'):10s} "
              f"{_size(m.get('total_bytes', 0)):>10s} "
              f"{'yes' if m.get('committable') else 'NO':4s} "
              f"{len(m.get('artifacts', [])):>6d}  {label}")
    print("-" * 88)
    print(f"{len(rows)} solve(s), {_size(total)} of artifacts")
    return 0


def cmd_show(args):
    m = solstamp.lookup(args.solve_id)
    if m is None:
        print(f"no manifest for {args.solve_id}")
        return 1
    print(f"solve_id   {m['solve_id']}")
    print(f"model      {m.get('model')}")
    print(f"tags       {', '.join(m.get('tags', [])) or '-'}")
    print(f"spec_ids   {', '.join(m.get('spec_ids', [])) or '-'}")
    print(f"total      {_size(m.get('total_bytes', 0))}  "
          f"(committable: {m.get('committable')})")
    print("\nproducer sources:")
    for path, d in sorted((m.get("sources") or {}).items()):
        print(f"  {solstamp.short(d)}  {path}")
    if m.get("env_params"):
        print("\nenvironment / CLI parameters (hashed):")
        for k, v in sorted(m["env_params"].items()):
            print(f"  {k} = {v}")
    ach = m.get("achieved")
    if ach:
        print("\nachieved (recorded, NOT hashed -- what the solve actually did):")
        exit_path = ach.get("exit")
        for k, v in sorted(ach.items()):
            flag = ""
            if k == "exit" and v not in ("tolerance", "direct_solve"):
                flag = "   <-- did NOT exit on its tolerance test"
            print(f"  {k} = {v}{flag}")
        req, got = ach.get("threshold_enforced"), ach.get("qerr_rel")
        if req and got:
            print(f"  -> overshoot {got/req:.2f}x the enforced threshold")
    else:
        print("\nachieved: not recorded (solve predates solstamp's `achieved` field)")
    if m.get("extra"):
        print("\nlabels (not hashed):")
        for k, v in sorted(m["extra"].items()):
            print(f"  {k} = {v}")
    print(f"\nparameters ({len(m.get('params', {}))}):")
    for k, v in sorted((m.get("params") or {}).items()):
        s = repr(v)
        print(f"  {k:22s} {s if len(s) <= 88 else s[:85] + '...'}")
    print("\nartifacts:")
    for a in m.get("artifacts", []):
        here = "ok" if os.path.exists(
            a["path"] if os.path.isabs(a["path"])
            else os.path.join(solstamp.REPO_DIR, a["path"])) else "MISSING"
        print(f"  {solstamp.short(a['sha256'])}  {_size(a['bytes']):>9s}  {here:8s} {a['path']}")
    return 0


def cmd_diff(args):
    a, b = solstamp.lookup(args.a), solstamp.lookup(args.b)
    for sid, m in ((args.a, a), (args.b, b)):
        if m is None:
            print(f"no manifest for {sid}")
            return 1
    print(f"{args.a}  ->  {args.b}")
    src = [(p, a["sources"].get(p), b["sources"].get(p))
           for p in sorted(set(a["sources"]) | set(b["sources"]))
           if a["sources"].get(p) != b["sources"].get(p)]
    if src:
        print("\nproducer source changed:")
        for p, x, y in src:
            print(f"  {p}: {solstamp.short(x)} -> {solstamp.short(y)}")
    d = solstamp.diff_params(a.get("params", {}), b.get("params", {}))
    de = solstamp.diff_params(a.get("env_params", {}), b.get("env_params", {}))
    if not d and not de and not src:
        print("\nidentical (this should be impossible for two different solve_ids)")
    for label, rows in (("parameters", d), ("environment", de)):
        if rows:
            print(f"\n{label} changed ({len(rows)}):")
            for k, x, y in rows:
                print(f"  {k:22s} {x!r} -> {y!r}")
    return 0


def cmd_check(args):
    bad = 0
    for m in solstamp.iter_manifests():
        if args.model and m.get("model") != args.model:
            continue
        problems = solstamp.artifact_problems(m)
        tag = ", ".join(m.get("tags", [])) or "-"
        if problems:
            bad += 1
            print(f"[STALE/MISSING] {m['solve_id']}  {m.get('model')}  {tag}")
            for p in problems[:4]:
                print(f"    - {p}")
            if len(problems) > 4:
                print(f"    ... and {len(problems) - 4} more")
        else:
            ach = m.get("achieved") or {}
            ex = ach.get("exit")
            if ex and ex not in ("tolerance", "direct_solve"):
                print(f"[ok, but CAPPED] {m['solve_id']}  {m.get('model')}  {tag}  "
                      f"exit={ex}"
                      + (f", sweeps={ach['sweeps']}" if "sweeps" in ach else "")
                      + (f", achieved {ach['qerr_rel']:.2e} vs threshold "
                         f"{ach['threshold_enforced']:.2e}"
                         if ach.get("qerr_rel") and ach.get("threshold_enforced") else ""))
            elif not args.quiet:
                print(f"[ok] {m['solve_id']}  {m.get('model')}  {tag}  "
                      f"{len(m.get('artifacts', []))} file(s), {_size(m.get('total_bytes', 0))}")
    if bad:
        print(f"\n{bad} solve(s) cannot be reused as recorded. "
              f"Re-run the producer; it will re-solve and re-record.")
    return 1 if bad else 0


def cmd_gc(args):
    """Report artifacts that could be deleted. The manifest is what must survive."""
    rows = []
    for m in solstamp.iter_manifests():
        if m.get("committable") and not args.include_small:
            continue
        present = [a for a in m.get("artifacts", [])
                   if os.path.exists(a["path"] if os.path.isabs(a["path"])
                                     else os.path.join(solstamp.REPO_DIR, a["path"]))]
        if present:
            rows.append((m, present, sum(a["bytes"] for a in present)))
    if not rows:
        print("nothing reclaimable")
        return 0
    total = 0
    for m, present, size in sorted(rows, key=lambda r: -r[2]):
        total += size
        tag = ", ".join(m.get("tags", [])) or "-"
        print(f"{_size(size):>10s}  {m['solve_id']}  {m.get('model'):8s}  {tag}  "
              f"({len(present)} file(s))")
    print(f"\n{_size(total)} reclaimable across {len(rows)} solve(s).")
    print("Manifests are kept regardless, so a deleted solve stays identifiable and "
          "re-derivable; re-running the producer reproduces it under the same solve_id.")
    if not args.dry_run:
        print("\n(refusing to delete: pass nothing but --dry-run is supported today; "
              "delete by hand once you are sure)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="every recorded solve")
    p.add_argument("--model")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="full parameters of one solve")
    p.add_argument("solve_id")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("diff", help="what differs between two solves")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("check", help="are recorded artifacts present and intact")
    p.add_argument("--model")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("gc", help="what could be deleted, and how much it reclaims")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--include-small", action="store_true",
                   help="also list solves small enough to keep in git")
    p.set_defaults(fn=cmd_gc)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
