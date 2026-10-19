#!/bin/bash
#SBATCH -J bop_solve_small
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 0-06:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/solve_small.%j.log
#
# The two cheap solves, each FROM ITS SPEC, then the correctness gate.
#   bgn_gam g0235df   one J* table, minutes
#   kp_vy   vyg30k70  three G types then 63 integral tables, built as ONE chain on this platform
# Submit from the repo root:  sbatch variants/klc/solve_small_klc.sh
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -f "$REPO/variants/run_seeds_slurm.sh" ] || { echo "submit from the repo root" >&2; exit 2; }
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
spec() { python -c "import json,sys; s=json.load(open('experiments/specs/'+sys.argv[1]+'.json')); print($2)" "$1"; }
echo "=== $(hostname) $(date '+%F %T') python $(which python) ==="

B=var-bgn_gam-g0235df-v1
echo "--- $B: expecting $(spec $B "s['expected_solves']")"
BOV=$(spec $B "json.dumps(s['params'])")       # read BEFORE the cd: spec() takes a repo-relative path
( cd variants/bgn_gam && BGN_PARAM_OVERRIDES="$BOV" python -W ignore rebuild_jstar_gam.py )

K=var-kp_vy-vyg30k70-v1
echo "--- $K: expecting $(spec $K "s['expected_solves']")"
# build_vy_tables.py fans the integ stage out over 6 worker processes; BLAS stays single-threaded
# inside them. NOTE the prefix is argv[1], NOT KP_VY_PREFIX (docs/RUNS.md, fourth hazard).
KOV=$(spec $K "json.dumps(s['params'])"); KPREFIX=$(spec $K "s['env']['KP_VY_PREFIX']")
( cd variants/kp_vy && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    KP_PARAM_OVERRIDES="$KOV" python -W ignore build_vy_tables.py "$KPREFIX" )

python variants/klc/verify_solves.py bgn kp
echo "=== END $(date '+%F %T') ==="
