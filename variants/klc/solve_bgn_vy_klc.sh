#!/bin/bash
#SBATCH -J bop_solve_bvy
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 0-04:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/solve_bvy.%j.log
#
# bgn_vy solves FROM THEIR SPECS: per spec, the J*_f(r, y, s) dynamic programme (rebuild_jstar_vy.py,
# ~1-2 min), the printed id written into the spec, then the correctness gate (verify_solves.py bvy).
# BSPECS is SPACE-separated and must be exported in the environment (sbatch --export splits on commas):
#   export BSPECS="var-bgn_vy-bvy25-v1"; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -f "$REPO/variants/run_seeds_slurm.sh" ] || { echo "submit from the repo root" >&2; exit 2; }
: "${BSPECS:?export BSPECS='<space-separated bgn_vy spec ids>'}"
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
LOGDIR=/kellogg/proj/axa3203/bop_runs/logs
spec() { python -c "import json,sys; s=json.load(open('experiments/specs/'+sys.argv[1]+'.json')); print($2)" "$1"; }
echo "=== $(hostname) $(date '+%F %T') python $(which python) ==="
for B in ${BSPECS//,/ }; do
  echo "--- $B: expecting $(spec $B "s.get('expected_solves')")  $(date '+%F %T')"
  BOV=$(spec $B "json.dumps(s['params'])")
  BLOG="$LOGDIR/solve_bvy.${SLURM_JOB_ID}.$(spec $B "s['params']['jstar_vy_file']").build.log"
  BMODEL=$(spec $B "s['model']")
  ( cd "variants/$BMODEL" && BGN_PARAM_OVERRIDES="$BOV" python -W ignore rebuild_jstar_vy.py ) 2>&1 | tee "$BLOG"
  python variants/klc/fill_spec_ids.py "experiments/specs/$B.json" "$BLOG"
  echo "--- gate $B  $(date '+%F %T')"
  if [ "$BMODEL" = bgn_q ]; then BOP_VERIFY_BQ_SPEC="$B" python variants/klc/verify_solves.py bq; else BOP_VERIFY_BVY_SPEC="$B" python variants/klc/verify_solves.py bvy; fi
done
echo "=== END $(date '+%F %T') ==="
