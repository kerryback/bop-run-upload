#!/bin/bash
#SBATCH -J bop_roomraw
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/mlp/logs/roomraw.%A.%a.log
# array index -> (economy, seed): 12 tasks = 6 economies x seeds 0,1
set -euo pipefail
source /kellogg/proj/axa3203/bop-run-upload/variants/klc/klc_env.sh
E=(kp_vy:vy1 gs_vy:gvyz10 bgn_vy:bvy1ch kp_vy:vyg25 gs_vy:gvyp10 bgn_vy:bvych0)
i=${SLURM_ARRAY_TASK_ID}; e=${E[$((i / 2))]}; s=$((i % 2))
python -W ignore /kellogg/proj/axa3203/bop_runs/mlp/room_raw.py --model "${e%%:*}" --tag "${e##*:}" --seed $s
