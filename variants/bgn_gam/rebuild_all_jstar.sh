#!/bin/bash
# Rebuild every live BGN J* table from its SPEC's parameters, on one machine.
#
# WHY THIS EXISTS. Six tables, six parameter blobs, and the blobs live in the specs. Hand-copying
# one into a shell command is the exact hazard tests/test_specs_match_shell.py exists to catch, so
# this reads them instead: for each live bgn_gam spec it takes `params`, exports it as
# BGN_PARAM_OVERRIDES, and runs the producer. The producer prints its solve_id and refuses to
# proceed if a table on disk belongs to a different one.
#
# ONE MACHINE, ALL SIX. A J* solve_id does not depend on the table's bytes -- the jstar stage has no
# upstream artifacts, so the id is parameters plus source digests and is platform-independent. The
# MANIFEST records the artifact's sha256, and two machines write the last two digits differently:
# measured 2026-09-15, Jstar_g0235d.csv as built on Phoenix differs from a Mac build by 5e-15
# relative, on a bit-identical r grid. So the set must be built and committed from one machine or
# the manifests will not match the committed tables.
#
# usage:  bash variants/bgn_gam/rebuild_all_jstar.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
PY=${PYTHON:-python3}

SPECS=$($PY - <<'EOF'
import glob, json, os
out = []
for f in sorted(glob.glob("../../experiments/specs/var-bgn_gam-*.json")):
    d = json.load(open(f))
    if d.get("lineage", {}).get("superseded_by") or d.get("lineage", {}).get("retired"):
        continue
    if "jstar" not in (d.get("expected_solves") or {}):
        continue
    out.append("%s\t%s\t%s" % (d["spec_id"], d["expected_solves"]["jstar"],
                               json.dumps(d["params"], separators=(",", ":"))))
print("\n".join(out))
EOF
)

echo "$SPECS" | while IFS=$'\t' read -r spec want params; do
  [ -n "$spec" ] || continue
  echo "=== $spec  (expecting jstar $want)"
  BGN_PARAM_OVERRIDES="$params" $PY -W ignore rebuild_jstar_gam.py 2>&1 | grep -E "solstamp|Jstar_gam iter|building" || true
done

echo
echo "Now check every printed solve_id against the spec's expected_solves, clear solves_pending in"
echo "the six specs, and commit the tables and manifests -- AFTER the commit that pinned the ids."
