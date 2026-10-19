#!/bin/bash
#SBATCH -J bop_verify_gvy
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-02:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/verify_gvy.%j.log
# Fill the printed ids into the gs_vy spec, then the gate. The seed array is submitted afterok on this.
#   export GSPEC=var-gs_vy-gvy25-v1; sbatch --export=ALL --dependency=afterok:<solve array> variants/klc/verify_gs_vy_klc.sh
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
: "${GSPEC:?export GSPEC=<gs_vy spec id>}"
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
python variants/klc/fill_gs_vy_ids.py "experiments/specs/$GSPEC.json" "$BOP_RUNS/logs"
BOP_VERIFY_GVY_SPEC="$GSPEC" python variants/klc/verify_solves.py gvy
