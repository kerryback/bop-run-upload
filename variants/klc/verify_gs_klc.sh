#!/bin/bash
#SBATCH -J bop_verify_gs
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-01:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/verify_gs.%j.log
# The GS21 correctness gate. The GS seed array is submitted --dependency=afterok on THIS job, so
# no seed can start on a solve that capped out or fails E[M R] = 1.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
python variants/klc/verify_solves.py gs
