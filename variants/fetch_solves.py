"""Get the solve artifacts a spec needs, without a cluster and without re-solving.

THE PROBLEM: a co-author has the repo but no access to Sol, Phoenix or /data/sjpruitt.
Regenerating is not a viable answer -- it is ~5 h per type on a cluster they do not
have, and until 2026-09-08 the byte-exact sha256 check would have reported their
CORRECT reproduction as corruption (see solstamp.artifact_problems).

THE PAYLOAD IS SMALL. All eight live solves are 504 MB total, of which 495 MB is the
five gs_bx solution.npz files. That is one Dropbox folder.

WHY THIS IS ~100 LINES: the manifests already ARE the distribution index. Each records
every artifact's path, bytes and sha256; each spec names its expected_solves. So this
walks spec -> manifests -> artifacts, copies what is missing, and verifies what it
copied. Nothing new had to be built to make sharing work.

usage:
    python variants/fetch_solves.py --spec var-gs_bx-bx7-v3 --from "/path/to/publish"
    python variants/fetch_solves.py --all --from "/path/to/publish"
    python variants/fetch_solves.py --spec var-gs_bx-bx7-v3            # check only
    python variants/fetch_solves.py --all --publish "/path/to/publish" # push, don't pull

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
            if d.get("lineage", {}).get("superseded_by"):
                continue                       # a superseded spec names a dead economy
            out.update(d.get("expected_solves") or {})
        return out
    d = json.load(open(os.path.join(SPECS, args.spec + ".json")))
    return d.get("expected_solves") or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec", help="spec_id, e.g. var-gs_bx-bx7-v3")
    g.add_argument("--all", action="store_true", help="every live (non-superseded) spec")
    ap.add_argument("--from", dest="src", help="publish folder to copy artifacts FROM")
    ap.add_argument("--publish", help="publish folder to copy artifacts INTO")
    args = ap.parse_args()

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
