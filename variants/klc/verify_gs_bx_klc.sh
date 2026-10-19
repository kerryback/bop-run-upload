#!/bin/bash
#SBATCH -J bop_verify_gs
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-02:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/verify_gs.%j.log
# gs_bx economies solved by solve_gs_klc.sh with GSBX_SPEC: write the printed ids into the spec, then the gate.
#   export GSBX_SPEC=var-gs_bx-gq25-v1; sbatch --export=ALL --dependency=afterok:<solve> variants/klc/verify_gs_bx_klc.sh
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
: "${GSBX_SPEC:?export GSBX_SPEC=<gs_bx spec id>}"
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
python - "$GSBX_SPEC" "$BOP_RUNS/logs" <<'PY'
import json, re, sys, os
sys.path.insert(0, "variants"); from precommit_spec import _spec_hash
sid, logs = sys.argv[1], sys.argv[2]; p = f"experiments/specs/{sid}.json"; s = json.load(open(p))
for st in s["solve"]["stages"]:
    n = st["name"]; m = re.findall(rf"\[solstamp\] gs_bx {re.escape(n)} solve_id=([0-9a-f]{{16}})", open(os.path.join(logs, f"solve_gs.detail.{n}.log")).read())
    assert m, f"no solve id for {n}"; s["expected_solves"][n] = m[-1]; print(n, m[-1])
s.pop("spec_hash", None); s["spec_hash"] = _spec_hash(s); json.dump(s, open(p, "w"), indent=1)
PY
BOP_VERIFY_GS_SPEC="$GSBX_SPEC" python variants/klc/verify_solves.py gs
