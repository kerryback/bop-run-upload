#!/bin/bash
#SBATCH -J bop_seeds
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 1-16:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/seeds.%A.%a.log
#
# One array over SEVERAL economies, so a single %N throttle caps the cores they use together.
# Array index i -> economy SPECS[i / 10], seed i % 10. Then exec the one seed runner every
# economy uses; nothing about the economy or the protocol is restated here.
#
#   export SPECS="g0235df vyg30k70"; sbatch --export=ALL --array=0-19%5 variants/klc/seeds_klc.sh
#   sbatch --export=ALL,SPECS=gx7w --array=0-9%5 --mem=16G variants/klc/seeds_klc.sh
#
# SPECS is SPACE- or comma-separated, but a comma list must be EXPORTED in the environment, not
# written inside --export: sbatch splits --export on commas, so "--export=ALL,SPECS=a,b" silently
# drops b (job 8735360's tasks 10-19 died with "no economy in SPECS=g0235df" for this reason).
#
# Submit from the repo root. --array is REQUIRED on the command line (it carries the throttle).
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -f "$REPO/variants/run_seeds_slurm.sh" ] || { echo "submit from the repo root" >&2; exit 2; }
: "${SPECS:?set SPECS=<comma-separated SEED_SPEC names>}"
source "$REPO/variants/klc/klc_env.sh"
read -r -a _S <<< "${SPECS//,/ }"
i=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
[ "$i" -lt $(( ${#_S[@]} * 10 )) ] || { echo "array index $i has no economy in SPECS=$SPECS" >&2; exit 2; }
export SEED_SPEC=${_S[$(( i / 10 ))]}
export SLURM_ARRAY_TASK_ID=$(( i % 10 ))      # run_seeds_slurm.sh reads its seed from this
echo "array index $i -> SEED_SPEC=$SEED_SPEC seed=$SLURM_ARRAY_TASK_ID results=$BOP_RESULTS_DIR"
cd "$REPO"
# bash reads a script incrementally, so editing run_seeds_slurm.sh in the repo while a task
# is mid-run garbles that task (2026-10-26: bvychs seeds 0-2 died on "s: command not found"
# right after their 1.5 h oracle stage). Each task therefore runs its own snapshot; the
# runner locates the repo through SLURM_SUBMIT_DIR, not its own path.
_RUNNER=$(mktemp "${SLURM_TMPDIR:-/tmp}/run_seeds_slurm.XXXXXX")
cp variants/run_seeds_slurm.sh "$_RUNNER"
exec bash "$_RUNNER"
