#!/bin/bash
#SBATCH -J bop_solve_gsbase
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 0-12:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/solve_gsbase.%j.log
# The GS21 baseline solve on Quest (var-gs_bx-gsbase-v3: gs_solve_reg.py at gmreg [1,1]); the published solution.npz
# is not on Quest and the source is unchanged, so the rebuild must print the spec's id c6ae2d52428a7ce5.
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"; source "$REPO/variants/klc/klc_env.sh"; cd "$REPO/variants/gs_bx"
GS_PARAM_OVERRIDES='{"gmreg": [1.0, 1.0]}' python -W ignore gs_solve_reg.py 161 1e-06 sol_gsbase 2>&1 | tee "$BOP_RUNS/logs/solve_gs.detail.sol_gsbase.log"
grep -q "solve_id=c6ae2d52428a7ce5" "$BOP_RUNS/logs/solve_gs.detail.sol_gsbase.log" || { echo "FAIL: not the precommitted id"; exit 3; }
grep -q "cycle-averaged; stopping" "$BOP_RUNS/logs/solve_gs.detail.sol_gsbase.log" && { echo "FAIL: sweep cap"; exit 4; }
exit 0
