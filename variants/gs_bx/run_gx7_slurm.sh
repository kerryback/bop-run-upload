#!/bin/bash
#SBATCH -J gs_gx7
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 0-12:00
#SBATCH -p general
#SBATCH -o outslurm/gx7.%A.%a.log
#
# The four NEW solves of var-gs_bx-gx7-v1: gamma(x) (slope 0.28, no regime) at exposure
# types gs_bx = 2.5, 4.0, 5.5, 7.0.  The FIFTH member of that ladder, gs_bx = 1.0, is
# sol_g28 (8b584c38614695ac) and is NOT re-solved: beta = 1 with this slope IS the g28
# economy, verified by recomputing its id from these parameters.
#
# Submit from the REPO ROOT, after `mkdir -p outslurm`:
#     cd <repo> && sbatch variants/gs_bx/run_gx7_slurm.sh
#
# Resources are the measured gs_solve_gam.py envelope, not guesses: g28 took 5.9 h at
# ~2.7 GiB peak on Phoenix and met the tolerance test at sweep 2974 of a 5600 cap
# (WORKING.md §37).  -t 0-12:00 covers the cap; 8G is ~3x the peak.  The `general`
# partition is used rather than `htc` because htc's limit is 4 h.
#
# Each task PRECOMMITS its id: the expected value is printed and checked against what the
# solver reports, so a laptop/cluster disagreement is caught here rather than after a
# panel has been built on top of it.
set -u
REPO="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO" ] || [ ! -d "$REPO/variants/gs_bx" ]; then
  echo "Submit from the repo root: cd <repo> && sbatch variants/gs_bx/run_gx7_slurm.sh" >&2
  exit 2
fi
cd "$REPO/variants/gs_bx"

BX=(2.5 4.0 5.5 7.0)
OUTDIRS=(sol_gx25 sol_gx40 sol_gx55 sol_gx70)
WANT=(a145001bf661632e bcc4e59f41f4a040 9bc863211011b5d9 89546b0b36dfd2d0)
i=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
b=${BX[$i]}; OUTDIR=${OUTDIRS[$i]}; EXPECT=${WANT[$i]}
OV="{\"gmreg\":[1.0,1.0],\"gs_gamma_slope\":0.28,\"gs_bx\":$b,\"gs_ashift\":0.0}"

CONDA_ENV=${CONDA_ENV:-bop}
module load mamba/latest
source activate "$CONDA_ENV"
# `source activate` sets CONDA_PREFIX but does not prepend the env's bin/ under sbatch.
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "python: $(which python)"
NT=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT VECLIB_MAXIMUM_THREADS=$NT

LOG="$REPO/outslurm/gx7.detail.$OUTDIR.log"
echo "=== gx7 task $i  $OUTDIR  gs_bx=$b  on $(hostname)  $(date '+%F %T')  threads=$NT ===" | tee "$LOG"
echo "    overrides: $OV" | tee -a "$LOG"
echo "    expecting solve_id $EXPECT (precommitted, see var-gs_bx-gx7-v1)" | tee -a "$LOG"
S=$(date +%s)
GS_PARAM_OVERRIDES="$OV" python -W ignore gs_solve_gam.py 161 1e-6 "$OUTDIR" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== $OUTDIR END $(date '+%F %T') WALL=$(( $(date +%s)-S ))s rc=$RC ===" | tee -a "$LOG"

# The solver prints "converged" on the sweep-capped exit as well as the tolerance exit
# (a known defect, WORKING.md §43), so trust the cycle-average marker, not the word.
grep -q "cycle-averaged; stopping" "$LOG" \
  && echo "NOTE: $OUTDIR hit the sweep cap (cycle-averaged), did NOT meet tolerance" \
  || echo "NOTE: $OUTDIR exited on the tolerance test"
if grep -q "solve_id=$EXPECT" "$LOG"; then
  echo "OK: $OUTDIR solve_id matches the precommitment $EXPECT"
else
  echo "WARNING: $OUTDIR did NOT produce $EXPECT -- CHECK BEFORE USING" >&2
  RC=${RC:-0}; [ "$RC" = 0 ] && RC=3
fi
exit $RC
