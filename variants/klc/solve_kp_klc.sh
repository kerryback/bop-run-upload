#!/bin/bash
#SBATCH -J bop_solve_kp
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 0-06:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/solve_kp.%j.log
#
# KP14 solves FROM THEIR SPECS: per spec, one chain (G then integ), then write the printed ids into the
# spec (the integ id is chained and cannot be known before the build), then the correctness gate.
# Generalises solve_small_klc.sh's KP block to a list of specs. ~20-40 min per spec on Quest.
#
# KSPECS is SPACE-separated and must be exported in the environment -- sbatch --export splits on
# commas, so "--export=ALL,KSPECS=a,b" silently drops b (job 9026795, 2026-10-07):
#   export KSPECS="var-kp_vy-vyg25k15-v1 var-kp_vy-vyg25k70-v1"; sbatch --export=ALL variants/klc/solve_kp_klc.sh
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -f "$REPO/variants/run_seeds_slurm.sh" ] || { echo "submit from the repo root" >&2; exit 2; }
: "${KSPECS:?export KSPECS='<space-separated kp_vy spec ids>'}"
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
LOGDIR=$(dirname /kellogg/proj/axa3203/bop_runs/logs/x)
spec() { python -c "import json,sys; s=json.load(open('experiments/specs/'+sys.argv[1]+'.json')); print($2)" "$1"; }
echo "=== $(hostname) $(date '+%F %T') python $(which python) ==="
for K in ${KSPECS//,/ }; do
  echo "--- $K: expecting $(spec $K "s['expected_solves']")  $(date '+%F %T')"
  KOV=$(spec $K "json.dumps(s['params'])"); KPREFIX=$(spec $K "s['env']['KP_VY_PREFIX']"); KMODEL=$(spec $K "s['model']")
  BLOG="$LOGDIR/solve_kp.${SLURM_JOB_ID}.${KPREFIX}.build.log"
  # build_vy_tables.py fans the integ stage out over 6 worker processes; BLAS stays single-threaded
  # inside them. NOTE the prefix is argv[1], NOT KP_VY_PREFIX (docs/RUNS.md, fourth hazard).
  ( cd "variants/$KMODEL" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      KP_PARAM_OVERRIDES="$KOV" python -W ignore build_vy_tables.py "$KPREFIX" ) 2>&1 | tee "$BLOG"
  python variants/klc/fill_spec_ids.py "experiments/specs/$K.json" "$BLOG"
  echo "--- gate $K  $(date '+%F %T')"
  if [ "$KMODEL" = kp_vj ]; then BOP_VERIFY_KPJ_SPEC="$K" python variants/klc/verify_solves.py kpj; else BOP_VERIFY_KP_SPEC="$K" python variants/klc/verify_solves.py kp; fi
done
echo "=== END $(date '+%F %T') ==="
