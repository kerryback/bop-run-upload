#!/bin/bash
#SBATCH -J gs_g28
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -o outslurm/gs_g28.log
#
# The gamma(x) economy: ONE solve, not an array. Reconstructed from REPORT.md:553
# (`sol_g28`), spec var-gs_bx-g28-v1, expected solve_id fa9032525bff9489 -- precommitted
# on 2026-09-07 and independently reproduced by the laptop and Phoenix before any solve.
#
# Resources mirror run_gs_bx7_slurm.sh, whose five solves of the same solver at the same
# xnum/tol ran ~5 h each in 8G. Not an array, so no inter-task BLAS contention, but the
# threads are still pinned: unpinned, this spawns one thread per PHYSICAL core and
# competes with whatever else shares the node.
#
# Submit from the REPO ROOT (outslurm/ must exist; SBATCH -o is resolved before the
# body runs):
#     mkdir -p outslurm && sbatch variants/gs_bx/run_g28_slurm.sh
set -u

# SLURM_SUBMIT_DIR, not BASH_SOURCE: sbatch stages a COPY of this script into
# /var/spool/slurmd/, so deriving the repo root from the script's own path lands in the
# spool directory. That bug cost job 62740438 on 2026-09-07.
REPO="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO" ] || [ ! -d "$REPO/variants/gs_bx" ]; then
  echo "Submit from the repo root: cd <repo> && sbatch variants/gs_bx/run_g28_slurm.sh" >&2
  exit 2
fi
cd "$REPO/variants/gs_bx"

CONDA_ENV=${CONDA_ENV:-bop}
module load mamba/latest
source activate $CONDA_ENV
# `source activate` sets CONDA_PREFIX but does NOT prepend the env's bin/ to PATH in the
# non-interactive shell sbatch provides; without this `python` is the mamba BASE
# interpreter and the job dies in ~1 s with ModuleNotFoundError: numpy.
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "python: $(which python)"

NT=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
export VECLIB_MAXIMUM_THREADS=$NT

OUTDIR=sol_g28
OV='{"gmreg":[1.0,1.0],"gs_gamma_slope":0.28}'
LOG="$REPO/outslurm/gs_g28.detail.log"

echo "=== g28 START $(date '+%F %T') on $(hostname) threads=$NT ===" | tee "$LOG"
echo "    overrides: $OV" | tee -a "$LOG"
echo "    expecting solve_id fa9032525bff9489 (precommitted, see var-gs_bx-g28-v1)" | tee -a "$LOG"

# GS_SOLVE_FORCE deliberately unset: the solver exits early on a content-addressed
# cache hit, so a resubmission after a failure is cheap and safe.
S=$(date +%s)
GS_PARAM_OVERRIDES="$OV" python -W ignore gs_solve_gam.py 161 1e-6 "$OUTDIR" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
E=$(date +%s)
echo "=== g28 END $(date '+%F %T') WALL=$((E-S))s rc=$RC ===" | tee -a "$LOG"

# The solve prints `converged:` on BOTH exit paths -- the tolerance test and the
# 5600-sweep cycle-average cap. The manifest records which; prefer
#     python variants/solfiles.py show fa9032525bff9489
# over trusting the word "converged" in this log.
grep -q "cycle-averaged; stopping" "$LOG" \
  && echo "NOTE: g28 hit the 5600-sweep cap (cycle-averaged), did NOT meet tolerance" \
  || echo "NOTE: g28 exited on the tolerance test"

grep -q "fa9032525bff9489" "$LOG" \
  && echo "OK: solve_id matches the precommitment" \
  || echo "WARNING: precommitted id fa9032525bff9489 not seen in the log -- CHECK BEFORE USING"

exit $RC
