#!/bin/bash
#SBATCH -J bop_mlp
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/mlp/logs/mlp.%j.log
set -euo pipefail
source /kellogg/proj/axa3203/bop-run-upload/variants/klc/klc_env.sh
export LD_LIBRARY_PATH="$BOP_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # torch loads the system libstdc++ first otherwise
cd "/kellogg/proj/axa3203/external/MLP Analyze/BOP campaign"
python -W ignore run_bop.py "$@"
