#!/bin/bash
#SBATCH -J bop_sec3fig
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/mlp/logs/sec3fig.%j.log
set -euo pipefail
source /kellogg/proj/axa3203/bop-run-upload/variants/klc/klc_env.sh
export LD_LIBRARY_PATH="$BOP_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" OMP_NUM_THREADS=8
cd /kellogg/proj/axa3203/bop_runs/mlp/sec3
python -W ignore sec3_figs.py
python -W ignore sec3_figs.py --construction
echo SEC3_DONE
