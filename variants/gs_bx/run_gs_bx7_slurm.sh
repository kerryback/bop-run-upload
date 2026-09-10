#!/bin/bash
#SBATCH -J gs_bx7
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -o outslurm/gs_bx7.%a.log
#
# Five gs_bx exposure-type solves, one per array task. WRITTEN, NOT YET RUN.
#
# Calibration: Gomes & Schmid (2021) Table I, as corrected on 2026-09-06 -- delta =
# 0.02/3, rho_x = 0.95^(1/3), kappa_e = 0.025. The five solve_ids this array will
# produce are listed in docs/refactor/WORKING.md §43; anything computed
# under the earlier 0.96 / 0.02 / no-kappa_e parameters is a different economy.
#
# Submit from the REPO ROOT (not this directory), after `mkdir -p outslurm`:
#     mkdir -p outslurm && sbatch variants/gs_bx/run_gs_bx7_slurm.sh
# SBATCH -o is resolved before the script body runs, so outslurm/ must already exist.
#
# ---------------------------------------------------------------------------
# Why these resource requests -- all measured on 2026-09-05, not guessed.
# (variants/README.md said "~a minute each"; the measurement is in
#  docs/refactor/WORKING.md §43.)
#
#   -t 0-08:00      sol_reg measured 3 h 23 m 36 s wall at ~4.7 cores, riding the
#                   5600-sweep cap (2.18 s/sweep mean). The cap BOUNDS the work:
#                   `if it == 5600: break` is unconditional, so no solve can exceed
#                   ~5601 sweeps regardless of convergence. At the 4 cores requested
#                   here expect ~4-4.5 h; 8 h is ~1.8x margin. Do not budget from
#                   `range(60000)` in the source -- that bound is dead code.
#
#                   2026-09-06: kappa_e = 0.025 makes the b'-choice depend on current
#                   debt, so the re-optimisation sweeps now build a (z,x,b,b') array.
#                   Re-measured on the full grid, same box, same 4 threads:
#                       policy sweep (1 in 25)   1.86 s -> 2.13 s   (+14.5%)
#                       frozen sweep (24 in 25)  1.82 s -> 1.91 s   (+5.0%)
#                       steady-state mean                            +5.4%
#                   Projected 3 h 24 m -> ~3 h 35 m ON THE LAPTOP.
#
#                   2026-09-07, raised 8 h -> 24 h from measurement ON SOL. Every
#                   number above is an Apple-silicon laptop number and does not
#                   transfer. Sol node sc043 is an AMD EPYC 7713 at 2.88 GHz, and
#                   `ps` on the running job showed python at **114% CPU**, not the
#                   ~470% the laptop production run reported -- the solve is
#                   dominated by single-threaded elementwise numpy (smooth() and the
#                   broadcasts), so the 4 allocated cores buy little. Job 62740472
#                   had not printed sweep 200 after 625 s, i.e. > 3.13 s/sweep, and
#                   5600 sweeps in 8 h needs <= 5.13 s/sweep. Too close.
#
#                   The failure mode is total: a walltime kill happens before
#                   np.savez_compressed, so an over-run loses every sweep and writes
#                   no manifest. And it cannot be repaired in flight --
#                   `scontrol update jobid=... TimeLimit=...` is refused on Sol with
#                   "Modifications to existing jobs are not permitted". The partition
#                   allows 7 days, so a 24 h request costs nothing but reserves
#                   headroom to ~15 s/sweep. Raise the request, never the risk.
#
#   --mem=8G        peak RSS re-measured with /usr/bin/time -l on the full grid
#                   (xnum=161, znum=200, bnum=20): 2.61 GiB before the kappa_e change,
#                   2.74 GiB after (+135 MB, the (200,161,20,20) = 103 MB b-by-b'
#                   array plus one temporary). The earlier "0.9-2.4 GB" figure in
#                   WORKING.md §43 was sampled with ps mid-run, not a peak; 2.74 GiB
#                   is the number to size against. 8 G is 2.9x that.
#
#                   The dominant term is still smooth(), which builds
#                   Pn = P[...,None] + mn, a (znum,xnum,bnum,161) array ~= 830 MB,
#                   4x per sweep.
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
# 2026-09-07: gs_ashift zeroed (227f5e9, spec var-gs_bx-bx7-v3). The old ladder was
# gs_ashift = 0.15*(gs_bx-1), i.e. 0.225/0.450/0.675/0.900 -- a +146% level shift at
# gs_bx = 7 against a one-sd exposure swing of +-31%, so the cross-section was mostly a
# size sort wearing a beta label. Zeroing it also closes the solve/simulate gap for
# free: gs_solve_reg.py:152-153 applies gs_ashift, gs_sim_bx.py never did, and the two
# are identical only at 0. The key is kept EXPLICIT rather than dropped -- the module
# default is already 0.0 so the solve_id is the same either way (verified), but an
# explicit zero records that it was chosen, not forgotten.
OVERRIDES=(
  '{"gmreg":[0.6,3.0]}'
  '{"gmreg":[0.6,3.0],"gs_bx":2.5,"gs_ashift":0.0}'
  '{"gmreg":[0.6,3.0],"gs_bx":4.0,"gs_ashift":0.0}'
  '{"gmreg":[0.6,3.0],"gs_bx":5.5,"gs_ashift":0.0}'
  '{"gmreg":[0.6,3.0],"gs_bx":7.0,"gs_ashift":0.0}'
)

i=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
OUTDIR=${OUTDIRS[$i]}
OV=${OVERRIDES[$i]}

# sbatch stages a COPY of this script into the compute node's spool directory, so
# ${BASH_SOURCE[0]} is /var/spool/slurmd/job.../slurm_script -- NOT a path inside the
# repo. Deriving HERE from it lands in the spool dir, where the next line fails with
#     mkdir: cannot create directory '../results': Permission denied
# and every array task dies in ~2 s. Hit on Sol 2026-09-07, job 62740438, all five
# tasks, ~35 MB MaxRSS each -- i.e. before python ever started. The env activation was
# fine; this is a different failure from the PATH trap above and looks nothing like it.
# Under sbatch the job's working directory already IS the submit directory, and
# SLURM_SUBMIT_DIR names it. The BASH_SOURCE form is kept as the fallback for running
# this script directly outside SLURM, where it is correct.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HERE="$REPO/variants/gs_bx"
if [ ! -f "$HERE/gs_solve_reg.py" ]; then
  echo "ERROR: no gs_solve_reg.py under $HERE" >&2
  echo "       submit from the REPO ROOT: sbatch variants/gs_bx/run_gs_bx7_slurm.sh" >&2
  exit 2
fi
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
