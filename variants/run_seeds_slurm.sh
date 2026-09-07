#!/bin/bash
#SBATCH -J bop_seeds
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2-00:00
#SBATCH -p public
#SBATCH -o outslurm/seeds.%A.%a.log
#
# Seeded replication array. One REPLICATION per array task, seed = array index.
# Each task runs the oracle decomposition and then the estimators on its own panel.
# WRITTEN, NOT YET RUN.
#
# ONE script for every economy rather than one per economy: the two differ only in a
# model name, a tag and an override string, and a copied 140-line SLURM script is how
# variants/gs_bx/gs_solve_reg.py acquired a duplicate keyword argument that made it a
# hard SyntaxError for two days. The economy is selected by SEED_SPEC.
#
# Submit from the REPO ROOT, once per economy, after its solve is recorded:
#     mkdir -p outslurm
#     sbatch --export=ALL,SEED_SPEC=vyx   variants/run_seeds_slurm.sh
#     sbatch --export=ALL,SEED_SPEC=g0235 variants/run_seeds_slurm.sh
# SBATCH -o is resolved before the script body runs, so outslurm/ must already exist.
# Size and window can be overridden per submission, e.g.
#     sbatch --export=ALL,SEED_SPEC=vyx,SEED_N=200,SEED_T=300 variants/run_seeds_slurm.sh
#
# ---------------------------------------------------------------------------
# PRECONDITION: the solve must already be recorded, and this array must NOT solve.
#
# kp_vy's `build_vy_tables.py` takes a single-builder lock per prefix, so ten array
# tasks would serialise behind one another on a multi-hour integ stage while nine sat
# idle holding allocations. bgn_gam's `rebuild_jstar_gam.py` takes NO lock, so ten
# tasks would race on one CSV. Measured on 2026-09-05, two concurrent kp builders
# drove one process from 453% to 915% CPU and neither finished sooner.
#
# So each economy is solved ONCE, outside this array:
#     cd variants/kp_vy   && KP_PARAM_OVERRIDES=... KP_VY_PREFIX=vyx python build_vy_tables.py vyx
#     cd variants/bgn_gam && BGN_PARAM_OVERRIDES=... python rebuild_jstar_gam.py
# and every task below verifies by CONTENT that what it read is what the registry
# recorded, aborting before it spends compute if not. A task whose tables do not match
# a live manifest must fail loudly rather than quietly replicate a different economy.
#
# Live solve_ids as of 2026-09-07 (they ALL moved that day when solstamp's
# canonicaliser was made portable -- WORKING.md §26):
#     kp_vy   vyx    G f7be27e39d2b530f   integ 84e195172f091cd2
#     bgn_gam g0235  jstar be222462dd017b2c
#
# ---------------------------------------------------------------------------
# Why these resource requests.
#
#   -t 2-00:00      Was 0-12:00, sized from LAPTOP measurements. Raised 2026-09-07 on
#                   evidence from the gs_bx array: the same code that ran at ~470% CPU
#                   on this laptop ran at 114% on a Sol compute node -- roughly 4x less
#                   parallel work per second, on hardware the envelope assumed was
#                   comparable. A laptop-derived walltime is not a Sol walltime.
#
#                   The asymmetry is what settles the size. `scontrol update TimeLimit`
#                   is REFUSED on Sol ("Modifications to existing jobs are not
#                   permitted"), so a walltime kill cannot be rescued: the oracle has no
#                   mid-run checkpoint, so the task writes no panel and records nothing,
#                   and the seed is simply lost. Over-requesting on an empty queue in a
#                   partition with a 7-day limit costs approximately nothing. Ask for
#                   2 days and tighten from the first task's measured Elapsed.
#
#                   COST MODEL, MEASURED 2026-09-07 on this laptop, kp_vy, with the
#                   flags this script actually uses (--rff 36,360,3600 --levels,
#                   default --nmat). Six points: N = 100/200/300/500 at T = 200, and
#                   N = 100/200 at T = 500.
#
#                       wall   ~  30 + 0.58*T + 0.0070*N*T  seconds
#                       maxRSS ~  2421 + 0.0853*N*T         MiB
#
#                   Wall is LINEAR in N, not quadratic: the affine fit at T = 200
#                   predicted N = 500 at 822 s against 824 s measured (0.24%), and the
#                   bilinear form predicted N = 200 / T = 500 at 1038 s against 1036 s
#                   (0.2%). Flagship N = 500 / T = 500 projects to ~35 min HERE.
#
#                   THE ~T*N^2 CLAIM IS NOT REFUTED FOR bgn_gam. The measurement behind
#                   it (WORKING.md:1284) was bgn_gam at --nmat 2, per-month cost
#                   0.70/1.53/18.02 s at N = 100/200/500 -- a 12x jump over the last
#                   2.5x of N. This script runs BOTH economies via SEED_SPEC, and only
#                   kp_vy has been re-measured. Treat g0235 as unmeasured and assume
#                   the quadratic until a ladder exists for it.
#
#                   Laptop wall does not transfer: Sol ran the same code at 114% CPU
#                   against 470% here, so scale by ~4x. ~35 min here is ~2.3 h there.
#                   RE-MEASURE from the first completed task before widening the array.
#
#   --mem=64G       RAISED FROM 24G 2026-09-07 on measurement. Peak RSS tracks N*T,
#                   which is a structural result rather than a fit: N = 500/T = 200 and
#                   N = 200/T = 500 have the same N*T and measured 11209 and 10969 MiB
#                   -- 2.1% apart with the large dimension swapped. Flagship N*T =
#                   250000 therefore projects to
#
#                       2421 + 0.0853*250000 = 23753 MiB = 23.2 GiB
#
#                   against a 24G cap of 24576 MiB. The margin is +822 MiB (3.3%),
#                   and the fit's own max residual is 644 MiB -- so the headroom is
#                   1.3x the model's own error. 24G is not refuted; it is unresolved,
#                   which for an unrescuable job is the same decision.
#
#                   The projection is also a FLOOR for what this script does. It was
#                   measured WITHOUT --save_panel (which this script passes, and which
#                   holds an N*T-row panel), and the estimator stage that follows in
#                   the same task was not measured at all.
#
#                   The asymmetry decides the size, as with -t: the oracle has no
#                   mid-run checkpoint, so an OOM at hour 3 writes nothing and the seed
#                   is simply lost, while over-requesting on an empty queue costs
#                   approximately nothing. Tighten from the first task's real MaxRSS.
#
#   --cpus-per-task=8   The inner work is BLAS (N x N solves and products) plus the
#                   estimator stage's joblib fan-out. Threads are pinned below so
#                   array tasks cannot oversubscribe each other, and --n_jobs is tied
#                   to the same allocation.
#
#   --array=0-9     Ten replications. The seed drives BOTH the firm-level shocks and
#                   the AGGREGATE state path (run_oracle.py offsets gam_seed /
#                   reg_seed by the seed). Without that offset every replication would
#                   share one aggregate path and the cross-seed spread would measure
#                   firm noise only -- which, for a project about a regime-driven
#                   pricing channel, would understate the true sampling variability
#                   and overstate every t-stat built from it.
# ---------------------------------------------------------------------------

set -euo pipefail

: "${SEED_SPEC:?set SEED_SPEC (vyx | g0235) -- e.g. sbatch --export=ALL,SEED_SPEC=vyx ...}"

case "$SEED_SPEC" in
  vyx)
    MODEL=kp_vy; TAG=vyx; SPEC=var-kp_vy-vyx-v2
    export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":1.8,"bv_comp":1.2}'
    export KP_VY_PREFIX=vyx
    SOLVE_HINT='cd variants/kp_vy && KP_VY_PREFIX=vyx python build_vy_tables.py vyx'
    ;;
  g0235)
    MODEL=bgn_gam; TAG=g0235; SPEC=var-bgn_gam-g0235-v2
    export BGN_PARAM_OVERRIDES='{"gmult":[0.2,3.5],"jstar_gam_file":"Jstar_g0235.csv"}'
    SOLVE_HINT='cd variants/bgn_gam && python rebuild_jstar_gam.py'
    ;;
  *)
    echo "unknown SEED_SPEC '$SEED_SPEC' (expected vyx or g0235)" >&2; exit 2 ;;
esac

# The overrides above are transcribed from variants/{kp_vy/run_vyx.sh,bgn_gam/run_g0235.sh}
# and are checked against them by tests/test_specs_match_shell.py. It is the parameters,
# not the seed, that define the economy, so every task in an array exports the same ones.

N=${SEED_N:-500}
T=${SEED_T:-500}
WINDOW=${SEED_WINDOW:-360}

CONDA_ENV=${CONDA_ENV:-bop}
module load mamba/latest
source activate "$CONDA_ENV"
# `source activate` sets CONDA_PREFIX but does NOT prepend the env's bin/ to PATH in the
# non-interactive shell sbatch provides. Without this, `python` resolves to the mamba
# BASE interpreter and every task dies in ~1 s with ModuleNotFoundError: numpy (the
# traceback names /etc/python/sitecustomize.py). Hit on Phoenix 2026-08-31 on all 11
# array tasks. Same fix as run_bop_job.sh:33-40.
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "python: $(which python)"

NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NT
export OPENBLAS_NUM_THREADS=$NT
export MKL_NUM_THREADS=$NT
export VECLIB_MAXIMUM_THREADS=$NT
export NUMEXPR_NUM_THREADS=$NT

SEED=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
SS=$(printf %03d "$SEED")

# sbatch STAGES A COPY of this script into the compute node's spool directory, so
# ${BASH_SOURCE[0]} is /var/spool/slurmd/job.../slurm_script -- not a path in the repo.
# Deriving the repo from it lands in the spool dir, and `mkdir -p results/logs` there
# fails with a permission error before python ever starts. The ASU session hit exactly
# this on job 62740438: five tasks dead in 2-7 s at ~35 MB MaxRSS, which is the
# signature of a shell failure rather than a python one. This file carried the same
# idiom, and would have lost a ten-task array the same way.
#
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from -- the repo root, per the
# header above. The BASH_SOURCE form stays as the fallback for running this script
# directly outside SLURM.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "$REPO/variants/run_seeds_slurm.sh" ]; then
  echo "ABORT: \$REPO resolved to '$REPO', which is not the bop-run-upload root." >&2
  echo "Submit from the repo root:  cd <repo> && sbatch --export=ALL,SEED_SPEC=... variants/run_seeds_slurm.sh" >&2
  exit 2
fi
cd "$REPO/variants"
mkdir -p results/logs
LOG="results/logs/log_${TAG}_s${SS}.txt"

echo "=== $MODEL/$TAG seed $SEED on $(hostname) $(date '+%F %T') threads=$NT N=$N T=$T ===" | tee "$LOG"

# ---- the solve must exist and be registered, before any compute is spent -------
if ! python common/runstamp.py current --model "$MODEL" --tag "$TAG" | tee -a "$LOG"; then
  {
    echo "ABORT: no live solve recorded for $MODEL/$TAG."
    echo "Build it ONCE, outside this array, with the overrides exported above:"
    echo "    $SOLVE_HINT"
    echo "Then re-submit. Solving inside the array would serialise (kp) or race (bgn)."
  } | tee -a "$LOG"
  exit 2
fi

# ---- per-seed checkpoint, keyed on the SOLVE rather than on file existence -----
# Re-running the array after a partial failure redoes only the seeds that are not
# current. Re-solving the economy makes every seed stale at once -- which a bare
# `[ -f panel.parquet ]` guard cannot express, and which is exactly the hazard the
# existence check in run_gs_bx7.sh was removed for.
RUNJSON="results/${MODEL}_estimators_${TAG}_s${SS}_w${WINDOW}_run.json"
if python common/runstamp.py is-current "$RUNJSON" --model "$MODEL" --tag "$TAG" >>"$LOG" 2>&1; then
  echo "seed $SEED already complete and current for the recorded solve -- nothing to do" | tee -a "$LOG"
  exit 0
fi

S=$(date +%s)
# --spec makes the run VERIFY, before it records anything, that the tables it read are
# the ones the spec declares. Without it a summary could carry a spec_id for an economy
# it did not build -- which is the failure the whole registry exists to prevent.
python -W ignore run_oracle.py --model "$MODEL" --N "$N" --T "$T" --seed "$SEED" \
       --tag "$TAG" --spec "$SPEC" --levels --save_panel 2>&1 | tee -a "$LOG"
MID=$(date +%s)
echo "=== oracle done in $((MID-S))s ===" | tee -a "$LOG"

python -W ignore run_estimators.py --model "$MODEL" --tag "$TAG" --seed "$SEED" \
       --window "$WINDOW" --levels --include_mkt --kappas 0.001,0.01,0.1,1 \
       --n_jobs "$NT" 2>&1 | tee -a "$LOG"
E=$(date +%s)

echo "=== seed $SEED END $(date '+%F %T') oracle=$((MID-S))s estimators=$((E-MID))s total=$((E-S))s ===" | tee -a "$LOG"

# What this replication was actually built from. `python variants/solfiles.py show <id>`
# expands any of these ids into the full parameter set that produced it.
python common/runstamp.py is-current "$RUNJSON" --model "$MODEL" --tag "$TAG" | tee -a "$LOG"
