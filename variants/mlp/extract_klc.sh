#!/bin/bash
#SBATCH -J bop_mlpx
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH -t 6:00:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/mlp/logs/mlpx.%A.%a.log
# One array task per seed:  sbatch --export=ALL,MODEL=kp_vy,TAG=vyg25,KAPPA=0.001 --array=0-9%5 extract_klc.sh
set -euo pipefail
source /kellogg/proj/axa3203/bop-run-upload/variants/klc/klc_env.sh
: "${MODEL:?}" "${TAG:?}" "${KAPPA:?}"
python -W ignore /kellogg/proj/axa3203/bop_runs/mlp/extract_weights.py --model "$MODEL" --tag "$TAG" \
       --seed "${SLURM_ARRAY_TASK_ID}" --kappa "$KAPPA" --n_jobs "${SLURM_CPUS_PER_TASK:-4}"
