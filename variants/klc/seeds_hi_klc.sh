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
# bash reads a script incrementally, so editing run_seeds_slurm.sh in the repo while a task
# is mid-run garbles that task (2026-10-26: bvychs seeds 0-2 died on "s: command not found"
# right after their 1.5 h oracle stage). Each task therefore runs its own snapshot; the
# runner locates the repo through SLURM_SUBMIT_DIR, not its own path.
_RUNNER=$(mktemp "${SLURM_TMPDIR:-/tmp}/run_seeds_slurm.XXXXXX")
cp variants/run_seeds_slurm.sh "$_RUNNER"
exec bash "$_RUNNER"
