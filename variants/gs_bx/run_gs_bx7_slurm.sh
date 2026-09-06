#!/bin/bash
#SBATCH -J gs_bx7
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 0-08:00
#SBATCH -p public
#SBATCH -o outslurm/gs_bx7.%a.log
#
# Five gs_bx exposure-type solves, one per array task. WRITTEN, NOT YET RUN.
#
# Submit from the REPO ROOT (not this directory), after `mkdir -p outslurm`:
#     mkdir -p outslurm && sbatch variants/gs_bx/run_gs_bx7_slurm.sh
# SBATCH -o is resolved before the script body runs, so outslurm/ must already exist.
#
# ---------------------------------------------------------------------------
# Why these resource requests -- all measured on 2026-09-05, not guessed.
# (variants/README.md said "~a minute each"; the measurement is in
#  docs/refactor/FINDINGS-gs21.md.)
#
#   -t 0-08:00      sol_reg measured 3 h 23 m 36 s wall at ~4.7 cores, riding the
#                   5600-sweep cap (2.18 s/sweep mean). The cap BOUNDS the work:
#                   `if it == 5600: break` is unconditional, so no solve can exceed
#                   ~5601 sweeps regardless of convergence. At the 4 cores requested
#                   here expect ~4-4.5 h; 8 h is ~1.8x margin. Do not budget from
#                   `range(60000)` in the source -- that bound is dead code.
#
#   --mem=8G        measured RSS 0.9-2.4 GB per solve. The driver is smooth(), which
#                   builds Pn = P[...,None] + mn, a (znum,xnum,bnum,161) =
#                   (200,161,20,161) array ~= 830 MB, 4x per sweep. 8 G leaves room
#                   for two live temporaries plus the per-regime state.
#
#   --cpus-per-task=4  GS is MEMORY-bandwidth bound, not core bound: running five
#                   solves concurrently on a 10-core laptop drove load to 16-24 and
#                   REDUCED aggregate throughput. Four cores per task is past the
#                   knee; more would mostly wait on memory.
#
# Threads are pinned below so N tasks on one node cannot oversubscribe: each task
# gets exactly the cores SLURM allocated it.
# ---------------------------------------------------------------------------

set -euo pipefail

CONDA_ENV=bop

module load mamba/latest
source activate $CONDA_ENV
# `source activate` sets CONDA_PREFIX but does NOT prepend the env's bin/ to PATH in
# the non-interactive shell sbatch provides. Without this, `python` resolves to the
# mamba BASE interpreter and every task dies in ~1 s with ModuleNotFoundError: numpy
# (the traceback names /etc/python/sitecustomize.py). Hit on Phoenix 2026-08-31 on
# all 11 array tasks. Same fix as run_bop_job.sh.
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "python: $(which python)"

# Pin BLAS to the allocated core count. gs_solve_reg.py's einsums go through BLAS;
# unpinned, each task would spawn one thread per PHYSICAL core on the node and the
# array tasks would fight each other.
NT=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$NT
export OPENBLAS_NUM_THREADS=$NT
export MKL_NUM_THREADS=$NT
export VECLIB_MAXIMUM_THREADS=$NT
export NUMEXPR_NUM_THREADS=$NT

# One entry per exposure type. Bash arrays -- no word-splitting games. (zsh does not
# split unquoted parameters at all, which is why this script is bash, not zsh like
# run_gs_bx7.sh.)
OUTDIRS=(sol_reg sol_b25c sol_b40c sol_b55c sol_b70c)
OVERRIDES=(
  '{"gmreg":[0.6,3.0]}'
  '{"gmreg":[0.6,3.0],"gs_bx":2.5,"gs_ashift":0.225}'
  '{"gmreg":[0.6,3.0],"gs_bx":4.0,"gs_ashift":0.450}'
  '{"gmreg":[0.6,3.0],"gs_bx":5.5,"gs_ashift":0.675}'
  '{"gmreg":[0.6,3.0],"gs_bx":7.0,"gs_ashift":0.900}'
)

i=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
OUTDIR=${OUTDIRS[$i]}
OV=${OVERRIDES[$i]}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p ../results/logs

LOG=../results/logs/log_gs_${OUTDIR}.txt
echo "=== task $i  $OUTDIR  on $(hostname)  $(date '+%F %T')  threads=$NT ===" | tee "$LOG"
echo "    overrides: $OV" | tee -a "$LOG"

# Each task writes its OWN outdir, so no two tasks ever touch one artifact.
# gs_solve_reg.py takes no lock (build_vy_tables.py does); never point two tasks at
# the same outdir.
#
# GS_SOLVE_FORCE is deliberately NOT set: gs_solve_reg.py exits early on a genuine
# content-addressed cache hit, so re-running the array after a partial failure
# re-solves only the missing types. Set GS_SOLVE_FORCE=1 to override.
S=$(date +%s)
GS_PARAM_OVERRIDES="$OV" python -W ignore gs_solve_reg.py 161 1e-6 "$OUTDIR" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
E=$(date +%s)

echo "=== task $i $OUTDIR END $(date '+%F %T') WALL=$((E-S))s rc=$RC ===" | tee -a "$LOG"

# The solve prints `converged:` on BOTH exit paths -- the tolerance test and the
# 5600-sweep cycle-average cap. Check which one this was; `cycle-averaged; stopping`
# immediately above `converged:` means it hit the cap and the residual is whatever it
# happened to be. As of 4db4536 the manifest records the achieved residual, so prefer
#     python variants/solfiles.py show <solve_id>
# over trusting the word "converged" in this log.
grep -q "cycle-averaged; stopping" "$LOG" \
  && echo "NOTE: $OUTDIR hit the 5600-sweep cap (cycle-averaged), did NOT meet tolerance" \
  || echo "NOTE: $OUTDIR exited on the tolerance test"

exit $RC
