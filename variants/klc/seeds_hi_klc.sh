#!/bin/bash
#SBATCH -J bop_seeds
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 1-16:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/seeds.%A.%a.log
# Extra seeds of ONE economy: the array index IS the seed (seeds_klc.sh maps i -> i % 10 over several economies).
#   SPEC=gvypk3 sbatch --export=ALL --array=10-19%5 variants/klc/seeds_hi_klc.sh
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
: "${SPEC:?set SPEC=<SEED_SPEC name>}"
source "$REPO/variants/klc/klc_env.sh"
export SEED_SPEC=$SPEC
echo "SEED_SPEC=$SEED_SPEC seed=$SLURM_ARRAY_TASK_ID results=$BOP_RESULTS_DIR"
cd "$REPO"
exec bash variants/run_seeds_slurm.sh
