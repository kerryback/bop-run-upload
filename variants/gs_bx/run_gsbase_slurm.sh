#!/bin/bash
#SBATCH -J gs_gsbase
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -o outslurm/gs_gsbase.log
#
# The GS21 BASELINE: ONE solve of the regime solver at unit multipliers, gmreg [1, 1], one
# exposure type -- the economy every GS path departs from (spec var-gs_bx-gsbase-v2, A1 in
# docs/NEXTUP.md). Expected solve_id c6ae2d52428a7ce5, precommitted on 2026-09-14 from these
# parameters and the current source without solving.
#
# Resources mirror run_g28_slurm.sh: the same solver family at the same xnum/tol ran ~5 h each in
# 8G. Threads are pinned so the solve does not spawn one thread per physical core.
#
# Submit from the REPO ROOT (outslurm/ must exist; SBATCH -o is resolved before the body runs),
# and chain the seed array on it so the array can only start on the precommitted economy:
#     mkdir -p outslurm && J=$(sbatch --parsable variants/gs_bx/run_gsbase_slurm.sh)
#     sbatch --export=ALL,SEED_SPEC=gsbase --dependency=afterok:$J variants/run_seeds_slurm.sh
set -u

# SLURM_SUBMIT_DIR, not BASH_SOURCE: sbatch stages a COPY of this script into /var/spool/slurmd/.
REPO="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO" ] || [ ! -d "$REPO/variants/gs_bx" ]; then
  echo "Submit from the repo root: cd <repo> && sbatch variants/gs_bx/run_gsbase_slurm.sh" >&2
  exit 2
fi
cd "$REPO/variants/gs_bx"

CONDA_ENV=${CONDA_ENV:-bop}
module load mamba/latest
source activate $CONDA_ENV
# `source activate` sets CONDA_PREFIX but does NOT prepend the env's bin/ under sbatch.
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "python: $(which python)"

NT=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
export VECLIB_MAXIMUM_THREADS=$NT

OUTDIR=sol_gsbase
OV='{"gmreg":[1.0,1.0]}'
EXPECT=c6ae2d52428a7ce5
LOG="$REPO/outslurm/gs_gsbase.detail.log"

echo "=== gsbase START $(date '+%F %T') on $(hostname) threads=$NT commit $(git -C "$REPO" rev-parse --short HEAD) ===" | tee "$LOG"
echo "    overrides: $OV" | tee -a "$LOG"
echo "    expecting solve_id $EXPECT (precommitted, see var-gs_bx-gsbase-v2)" | tee -a "$LOG"

# GS_SOLVE_FORCE deliberately unset: the solver exits early on a content-addressed cache hit.
S=$(date +%s)
GS_PARAM_OVERRIDES="$OV" python -W ignore gs_solve_reg.py 161 1e-6 "$OUTDIR" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
E=$(date +%s)
echo "=== gsbase END $(date '+%F %T') WALL=$((E-S))s rc=$RC ===" | tee -a "$LOG"

# The solver prints `converged:` on BOTH exit paths; the cycle-average marker says which.
grep -q "cycle-averaged; stopping" "$LOG" \
  && echo "NOTE: gsbase hit the sweep cap (cycle-averaged), did NOT meet tolerance" \
  || echo "NOTE: gsbase exited on the tolerance test"

# Fail unless the id printed is the precommitted one, so an afterok seed array never starts on
# an economy this spec does not name (the run_gx7_slurm.sh behaviour, not run_g28's).
if grep -q "solve_id=$EXPECT" "$LOG"; then
  echo "SOLVE OK: sol_gsbase=$EXPECT"
else
  echo "SOLVE MISMATCH: precommitted $EXPECT not seen in the log -- the dependent seed array must not start" >&2
  [ "$RC" = 0 ] && RC=3
fi
exit $RC
