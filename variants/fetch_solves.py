"""Get the solve artifacts a spec needs, without a cluster and without re-solving.

THE PROBLEM: a co-author has the repo but no access to Sol, Phoenix or /data/sjpruitt.
Regenerating is not a viable answer -- it is ~5 h per type on a cluster they do not
have, and until 2026-09-08 the byte-exact sha256 check would have reported their
CORRECT reproduction as corruption (see solstamp.artifact_problems).

MOST OF IT IS ALREADY IN THE CLONE. Of the 28 solves the live specs pin, 17 (81 MB --
every BGN and KP14 table) are committed to git and arrive with the repository. The
other 11 are the gs_bx solution.npz files, 92-105 MB each and 1,090 MB together, which
are too large to version and are published instead. The line is SMALL_ARTIFACT_BYTES in
common/solstamp.py and each manifest records which side it fell on in `committable`.
So on a fresh clone this usually has nothing to do unless you work on GS21.

WHERE THE PUBLISHED ONES LIVE IS NOT BAKED IN. It is the --from argument, or
BOP_SOLVES_DIR in the environment, because the same shared folder appears at a
different absolute path on every machine that syncs it. Set it once:

    export BOP_SOLVES_DIR="$HOME/<Your Org> Dropbox/<Your Name>/BGN and Kelly Malamud/solves"

WHY THIS IS ~100 LINES: the manifests already ARE the distribution index. Each records
every artifact's path, bytes and sha256; each spec names its expected_solves. So this
walks spec -> manifests -> artifacts, copies what is missing, and verifies what it
copied. Nothing new had to be built to make sharing work.

usage:
    python variants/fetch_solves.py --all               # fetch, using BOP_SOLVES_DIR
    python variants/fetch_solves.py --all --check       # report only, fetch nothing
    python variants/fetch_solves.py --all --from "/path/to/publish"   # explicit source
    python variants/fetch_solves.py --spec var-gs_bx-bx7-v4 --from "/path/to/publish"
    python variants/fetch_solves.py --all --publish "/path/to/publish" # push, don't pull

With neither --from nor BOP_SOLVES_DIR set it reports and fetches nothing, which is the
old check-only behaviour and still what you get on a machine that has never been told
where the folder is.

The publish layout is content-addressed, mirroring the registry:
    <publish>/<solve_id>/<basename of each artifact>
so a folder can hold many solves without collision and nothing is ever overwritten
with different bytes.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "common"))
import solstamp  # noqa: E402

SPECS = os.path.join(ROOT, "experiments", "specs")


def specs_wanted(args):
    if args.all:
        out = {}
        for fn in sorted(os.listdir(SPECS)):
            if not fn.endswith(".json"):
                continue
            d = json.load(open(os.path.join(SPECS, fn)))
            lin = d.get("lineage") or {}
            if lin.get("superseded_by"):
                continue                       # a superseded spec names a dead economy
            if lin.get("retired"):
                # RETIRED is the other way a spec stops being live, and it is NOT the
                # same as superseded: a superseded spec has a successor, a retired one
                # has none ("nothing supersedes this spec, because nothing replaces it"
                # -- var-kp_vy-vyxT860-v1). `runstamp.live_solves` has always excluded
                # both; this function checked only the first, so --all asked for
                # vyxT860's two solves and reported their 66 artifacts as problems on
                # every clean clone. They are not problems: vyxT860 reused vyx's tables
                # and vyx has re-solved since, so the paths hold newer bytes than the
                # retired manifest records.
                continue
            out.update(d.get("expected_solves") or {})
        return out
    d = json.load(open(os.path.join(SPECS, args.spec + ".json")))
    return d.get("expected_solves") or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec", help="spec_id, e.g. var-gs_bx-bx7-v4")
    g.add_argument("--all", action="store_true", help="every live (non-superseded) spec")
    ap.add_argument("--from", dest="src",
                    help="publish folder to copy artifacts FROM "
                         "(default: $BOP_SOLVES_DIR)")
    ap.add_argument("--publish", help="publish folder to copy artifacts INTO")
    ap.add_argument("--check", action="store_true",
                    help="report what is missing and fetch nothing, ignoring "
                         "$BOP_SOLVES_DIR")
    args = ap.parse_args()

    # The shared folder sits at a different absolute path on every machine that syncs
    # it, so the location is configuration, not code. --from wins; BOP_SOLVES_DIR is the
    # set-once fallback; --check suppresses both so a plain report is always reachable.
    env_src = os.environ.get("BOP_SOLVES_DIR")
    if args.check:
        args.src = None
    elif not args.src and not args.publish and env_src:
        args.src = env_src
    if args.src and not os.path.isdir(args.src):
        sys.exit(f"not a directory: {args.src}\n"
                 f"Set BOP_SOLVES_DIR, or pass --from, pointing at the shared "
                 f"'solves' folder. On a machine that syncs it the path usually ends "
                 f"'.../BGN and Kelly Malamud/solves'.")
    if args.src and args.src == env_src:
        print(f"source: {env_src}  (from BOP_SOLVES_DIR)")

    want = specs_wanted(args)
    if not want:
        sys.exit("that spec pins no expected_solves, so there is nothing to fetch")

    total = missing = fetched = published = bad = 0
    for stage, sid in sorted(want.items()):
        man = solstamp.lookup(sid)
        if man is None:
            print(f"  {stage:10s} {sid}  NO MANIFEST in experiments/registry/ -- "
                  f"cannot fetch what the repo cannot describe")
            bad += 1
            continue
        for art in man.get("artifacts", []):
            total += 1
            rel = art["path"]
            local = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
            store = os.path.join(args.publish or args.src or "", sid,
                                 os.path.basename(rel)) if (args.publish or args.src) else None

            if args.publish:
                if not os.path.exists(local):
                    print(f"  {stage:10s} {os.path.basename(rel):22s} NOT HERE, cannot publish")
                    missing += 1
                    continue
                os.makedirs(os.path.dirname(store), exist_ok=True)
                if os.path.exists(store) and solstamp.file_digest(store) == art["sha256"]:
                    print(f"  {stage:10s} {os.path.basename(rel):22s} already published")
                    continue
                shutil.copy2(local, store)
                ok = solstamp.file_digest(store) == art["sha256"]
                print(f"  {stage:10s} {os.path.basename(rel):22s} published "
                      f"{art['bytes']/1e6:7.1f} MB {'OK' if ok else 'DIGEST MISMATCH'}")
                published += 1
                bad += (not ok)
                continue

            # fetch / check
            probs = solstamp.artifact_problems({"solve_id": sid, "artifacts": [art]})
            if not probs:
                print(f"  {stage:10s} {os.path.basename(rel):22s} present and verified")
                continue
            missing += 1
            if not args.src:
                print(f"  {stage:10s} {os.path.basename(rel):22s} {probs[0].split(': ',1)[1][:60]}")
                continue
            if not os.path.exists(store):
                print(f"  {stage:10s} {os.path.basename(rel):22s} NOT IN THE PUBLISH FOLDER")
                bad += 1
                continue
            os.makedirs(os.path.dirname(local), exist_ok=True)
            shutil.copy2(store, local)
            ok = solstamp.file_digest(local) == art["sha256"]
            print(f"  {stage:10s} {os.path.basename(rel):22s} fetched "
                  f"{art['bytes']/1e6:7.1f} MB {'OK' if ok else 'DIGEST MISMATCH'}")
            fetched += 1
            bad += (not ok)

    print(f"\n{total} artifacts across {len(want)} solves: "
          f"{missing} were missing, {fetched} fetched, {published} published, {bad} bad")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
