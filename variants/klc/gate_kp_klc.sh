#!/bin/bash
#SBATCH -J bop_gate_kp
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -t 1:00:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/gate_kp.%j.log
#
# Re-run ONLY the correctness gate (verify_solves.py kp / kpj) for kp_vy specs whose solves exist and whose
# ids are already written into the spec. For when the gate itself was wrong: on 2026-10-28 its type regex
# matched single digits only, so the 24-type economies vym6 / vym3s / vym6s "failed" with every solve check
# passing, and their afterok seed arrays went DependencyNeverSatisfied.
#   export KSPECS="var-kp_vy-vym6-v1 var-kp_vy-vym3s-v1"; sbatch --export=ALL variants/klc/gate_kp_klc.sh
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
: "${KSPECS:?export KSPECS='<space-separated kp_vy spec ids>'}"
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
echo "=== $(hostname) $(date '+%F %T') python $(which python) ==="
RC=0
for K in ${KSPECS//,/ }; do
  KMODEL=$(python -c "import json,sys; print(json.load(open('experiments/specs/'+sys.argv[1]+'.json'))['model'])" "$K")
  echo "--- gate $K  $(date '+%F %T')"
  if [ "$KMODEL" = kp_vj ]; then BOP_VERIFY_KPJ_SPEC="$K" python variants/klc/verify_solves.py kpj || RC=1
  else BOP_VERIFY_KP_SPEC="$K" python variants/klc/verify_solves.py kp || RC=1; fi
done
echo "=== gate exit $RC  $(date '+%F %T') ==="
exit $RC
