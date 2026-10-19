#!/bin/bash
#SBATCH -J bop_screen
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 6:00:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/screen.%A.%a.log
#
# Reduced-form screen: one array task = one (config, seed). Array index i -> config CONFIGS[i / NSEEDS],
# seed i % NSEEDS. Off protocol by construction; writes to BOP_SCREEN_DIR, never to variants/results.
#
#   sbatch --export=ALL,CONFIGS=base,notype,levels --array=0-29%6 variants/klc/screen_klc.sh
#   sbatch --export=ALL,CONFIGS=hl6,hl24,NSEEDS=5,EXTRA="--full" --array=0-9%4 variants/klc/screen_klc.sh
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
source "$REPO/variants/klc/klc_env.sh"
: "${CONFIGS:?set CONFIGS=<comma-separated rf_configs names>}"
NSEEDS=${NSEEDS:-10}
SEED0=${SEED0:-0}          # first seed; NSEEDS=5,SEED0=5 runs seeds 5..9
EXTRA=${EXTRA:-}
export BOP_SCREEN_DIR=${BOP_SCREEN_DIR:-/kellogg/proj/axa3203/bop_runs/screen}
# CONFIGS may be a comma list of names, or @<ladder> for a name in rf_configs.LADDERS
if [[ "$CONFIGS" == @* ]]; then
  CONFIGS=$(python -c "import sys; sys.path.insert(0,'$REPO/variants/reduced'); import rf_configs; print(','.join(rf_configs.LADDERS['${CONFIGS#@}']))")
fi
IFS=, read -r -a _C <<< "$CONFIGS"
i=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
[ "$i" -lt $(( ${#_C[@]} * NSEEDS )) ] || { echo "array index $i has no config in CONFIGS=$CONFIGS" >&2; exit 2; }
CFG=${_C[$(( i / NSEEDS ))]}
SEED=$(( i % NSEEDS + SEED0 ))
echo "array index $i -> config=$CFG seed=$SEED out=$BOP_SCREEN_DIR extra='$EXTRA' threads=$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO"
exec python -W ignore variants/reduced/rf_screen.py --config "$CFG" --seed "$SEED" --out "$BOP_SCREEN_DIR" --n_jobs "$SLURM_CPUS_PER_TASK" $EXTRA
