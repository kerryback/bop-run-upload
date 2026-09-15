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

LOG="$ROOT/_scratch/protocol/rebuild_all_jstar.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
echo "full output -> $LOG"

rc=0
echo "$SPECS" | while IFS=$'\t' read -r spec want params; do
  [ -n "$spec" ] || continue
  echo "=== $spec  (expecting jstar $want)"
  { echo "=== $spec  (expecting jstar $want)"; } >> "$LOG"
  BGN_PARAM_OVERRIDES="$params" $PY -W ignore rebuild_jstar_gam.py 2>&1 | tee -a "$LOG" \
    | grep -E "solve_id|recorded|^done|building" || true
  # The id the producer actually computed must be the one the spec precommitted. Checking here
  # rather than by eye: six ids, and the whole point of precommitting them is that a mismatch
  # stops the campaign instead of quietly redefining the economy.
  got=$(grep -oE "solve_id=[0-9a-f]{16}" "$LOG" | tail -1 | cut -d= -f2)
  if [ "$got" = "$want" ]; then
    echo "    ID OK: $got"
  else
    echo "    *** ID MISMATCH: precommitted $want, produced ${got:-<none>} ***"
    echo "MISMATCH $spec $want $got" >> "$LOG.mismatch"
  fi
done

echo
if [ -f "$LOG.mismatch" ]; then
  echo "AT LEAST ONE ID DID NOT REPRODUCE -- do not commit, chase the id first:"
  cat "$LOG.mismatch"
  exit 1
fi
echo "Every id reproduced. Next: clear solves_pending in the six specs, then commit the tables and"
echo "manifests -- in a LATER commit than the one that pinned the ids."
