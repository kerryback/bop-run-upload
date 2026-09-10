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
#     sbatch --export=ALL,SEED_SPEC=g28   variants/run_seeds_slurm.sh
#     sbatch --export=ALL,SEED_SPEC=bx7   variants/run_seeds_slurm.sh
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
#     gs_bx   g28    sol_g28 8b584c38614695ac
#     gs_bx   bx7    sol_reg 63fa7ebbc2db49ea  sol_b25c 649bb384300faadf
#                    sol_b40c a3bb50b66287307c  sol_b55c 645262a8e72d944c
#                    sol_b70c 0818d7153d5708cc  (five tags -> one economy; SOLVE_TAG
#                    lists them all and runstamp unions them)
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
#
#                   THE ORACLE IS THE SMALLER HALF. Every task runs run_estimators.py
#                   after the oracle, under this same walltime, and that stage had
#                   NEVER been run to completion until 2026-09-07 -- no *_run.json
#                   existed anywhere in the repo. The first end-to-end run (kp_vy,
#                   N=60/T=80/window=36, 29 eval months) took 741 s against the
#                   oracle's 216 s: 3.4x LONGER. Its cost scales with eval_months
#                   (= T - window - 15) x P. MEASURED 2026-09-07, four points:
#
#                       estimator wall ~ 25.1 s * eval_months, intercept ~0
#
#                   at N=60, windows 20/36/50 (45/29/15 eval months -> 1132/741/378 s):
#                   25.2, 25.6, 25.2 s per eval month, linear through the origin. And it
#                   is essentially N-INDEPENDENT: N=60 -> 180 at fixed 29 eval months
#                   went 741 -> 751 s, +1.4% for 3x the cross-section, because the P x P
#                   ridge solve dominates and P does not depend on N. Peak RSS is ~1.4
#                   GiB and barely moves with N, so the estimator does NOT drive --mem.
#
#                   Flagship (T=500, window=360) is 125 eval months -> ~52 min here,
#                   ~3.5 h on Sol. With the oracle's ~35 min / ~2.3 h that is ~1.5 h
#                   here and ~6 h on Sol per task. The estimator is the larger half but
#                   not dominant: its cost grows only with eval_months (29 -> 125) while
#                   the oracle's grows with N*T (4800 -> 250000). Comfortable under
#                   -t 2-00:00. The N-independence is measured only to N=180.
#                   RE-MEASURE from the first completed task before widening the array.
#
#   --mem=64G       BOTH ECONOMIES NOW MEASURED AT FLAGSHIP (N=500/T=500/w=360, sacct
#                   MaxRSS of the whole task, oracle + estimators, Sol, 2026-09-08):
#
#                       kp_vy/vyx     30.6 GiB   (seed 0; seeds 1-9 all 29.9-30.6)
#                       bgn_gam/g0235 15.7-38.9 GiB (ten seeds on identical nodes; BIMODAL,
#                                     six near 16 and four at 27-39; cause not established,
#                                     WORKING.md §40 -- so do not size from seed 0 alone)
#                       gs_bx/g28     4.7 GiB   (seed 0, 2026-09-10; oracle 4616 s, estimators 9574 s)
#                       gs_bx/bx7     5.2 GiB   (seed 0, 2026-09-10; oracle 3414 s, estimators 10085 s)
#                                     gs is the LIGHTEST economy on memory by 3-7x, even with
#                                     five 100 MB solutions loaded; its estimator stage at
#                                     eight kappas runs 77-81 s per evaluation month on Sol
#                                     (WORKING.md §46). The gs arrays were submitted with
#                                     --mem=24G on the command line, 4.6x the measured peak.
#
#                   64G is 2.1x the larger. History of this line: 24G -> 64G on the
#                   kp_vy laptop ladder (the flagship task then used 30.6, so 24G would
#                   have OOMed) -> 128G on a bgn_gam ladder projecting ~68 GiB -> back to
#                   64G on the measurement above.
#
#                   THE 68 GiB PROJECTION WAS WRONG BY 3.5x, and the reason is NOT
#                   established. The laptop ladder had bgn_gam at 1.96x kp_vy's RSS at
#                   N=500/T=200; on Sol at T=500 it is 0.52x. The two measurements
#                   disagree in DIRECTION, so one of macOS max-RSS vs cgroup accounting,
#                   --save_panel, or something in how bgn_gam's memory scales in T is
#                   not what the ladder assumed. Both numbers are recorded; do not
#                   extrapolate bgn_gam memory from the laptop again.
#
#                   The asymmetry still decides ties: an OOM writes nothing and the seed
#                   is lost, over-requesting on an empty queue costs nothing. On a busy
#                   queue this now schedules where 128G would have waited.
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

: "${SEED_SPEC:?set SEED_SPEC (vyx | g0235 | g28 | bx7) -- e.g. sbatch --export=ALL,SEED_SPEC=vyx ...}"

case "$SEED_SPEC" in
  vyx)
    MODEL=kp_vy; TAG=vyx; SPEC=var-kp_vy-vyx-v2; SOLVE_TAG=vyx
    KAPPAS=0.001,0.01,0.1,1
    export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":1.8,"bv_comp":1.2}'
    export KP_VY_PREFIX=vyx
    SOLVE_HINT='cd variants/kp_vy && KP_VY_PREFIX=vyx python build_vy_tables.py vyx'
    ;;
  g0235)
    # SOLVE_TAG != TAG here, and it must reach ALL THREE runstamp lookups below -- the
    # precondition, the per-seed checkpoint, and the post-run verification. The first
    # fix (e42a3a7) reached only the precondition: g0235 seed 0 then ran both stages to
    # completion (3 h 14 m) and was marked FAILED by the post-run check, which looked up
    # g0235, found nothing, and reported the run STALE. The checkpoint had the same bug,
    # so a resubmission would have re-run the whole seed instead of skipping it.
    # The array's TAG names OUTPUT files, but bgn_gam's solve is
    # registered under the tag its PRODUCER used, which is the J* table's filename. The
    # precondition below looked up the spec tag and so could never pass for g0235: job
    # 62876077 aborted in 44 s on 2026-09-08 with "no live solve recorded" while
    # be222462dd017b2c sat in the registry the whole time. Failing closed and cheap is
    # the right direction for this check, but it was asking the wrong question.
    MODEL=bgn_gam; TAG=g0235; SPEC=var-bgn_gam-g0235-v2; SOLVE_TAG=Jstar_g0235
    KAPPAS=0.001,0.01,0.1,1
    export BGN_PARAM_OVERRIDES='{"gmult":[0.2,3.5],"jstar_gam_file":"Jstar_g0235.csv"}'
    SOLVE_HINT='cd variants/bgn_gam && python rebuild_jstar_gam.py'
    ;;
  bx7)
    # GS21 with five exposure types beta in {1, 2.5, 4, 5.5, 7}, one solve per type. FIVE
    # solve tags, so SOLVE_TAG lists them all and every runstamp lookup below unions them
    # (runstamp.live_solves takes a comma list; until 2026-09-09 it took one tag, and the
    # per-seed checkpoint would have reported STALE forever for this economy). The
    # simulator reads GS_BX_* for which types exist; the solutions carry their own gs_bx
    # and gs_sim_bx.py refuses a ladder that disagrees with them. GS_SIM_OVERRIDES stays
    # UNSET (WORKING.md §42). The kappa grid is the spec's eight values.
    MODEL=gs_bx; TAG=bx7; SPEC=var-gs_bx-bx7-v3; SOLVE_TAG=sol_reg,sol_b25c,sol_b40c,sol_b55c,sol_b70c
    export GS_BX_SOLDIRS=sol_reg,sol_b25c,sol_b40c,sol_b55c,sol_b70c
    export GS_BX_BETAS=1.0,2.5,4.0,5.5,7.0
    export GS_BX_SHARES=0.2,0.2,0.2,0.2,0.2
    unset GS_SIM_OVERRIDES
    KAPPAS=0.001,0.01,0.03,0.1,0.3,1,3,10
    SOLVE_HINT='python variants/fetch_solves.py --spec var-gs_bx-bx7-v3 --from "<the shared solves folder, e.g. the Dropbox solves/ dir>"'
    ;;
  g28)
    # GS21, one type, gamma(x) = clip(0.5 - 0.28 x/sd(x), 0.05, 1.0): the reconstruction of
    # REPORT §13g's economy, where the published table showed a gap with ZERO nonlinear
    # room. Its parameters are not an override blob: the solver read GS_PARAM_OVERRIDES
    # when it built sol_g28, and that is inside the solve_id the spec pins. The simulator
    # takes its structural parameters out of solution.npz; the three GS_BX_* variables are
    # its only statement of which types exist, and GS_SIM_OVERRIDES must stay UNSET --
    # run_oracle.py refuses an undeclared value there (WORKING.md §42).
    MODEL=gs_bx; TAG=g28; SPEC=var-gs_bx-g28-v2; SOLVE_TAG=sol_g28
    export GS_BX_SOLDIRS=sol_g28
    export GS_BX_BETAS=1.0
    export GS_BX_SHARES=1.0
    unset GS_SIM_OVERRIDES
    KAPPAS=0.001,0.01,0.03,0.1,0.3,1,3,10
    # The solve exists and is published content-addressed; 3-6 h to remake, seconds to
    # fetch. The hint says fetch, never re-solve.
    SOLVE_HINT='python variants/fetch_solves.py --spec var-gs_bx-g28-v2 --from "<the shared solves folder, e.g. the Dropbox solves/ dir>"'
    ;;
  *)
    echo "unknown SEED_SPEC '$SEED_SPEC' (expected vyx, g0235, g28 or bx7)" >&2; exit 2 ;;
esac

# Every case above restates its economy's parameters and kappa grid, and each is checked
# against the spec its SPEC= names by tests/test_specs_match_shell.py -- a second copy is
# what drifted between run_gs_bx7.sh and its spec. It is the parameters, not the seed,
# that define the economy, so every task in an array exports the same ones.

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
if ! python common/runstamp.py current --model "$MODEL" --tag "${SOLVE_TAG:-$TAG}" | tee -a "$LOG"; then
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
if python common/runstamp.py is-current "$RUNJSON" --model "$MODEL" --tag "${SOLVE_TAG:-$TAG}" >>"$LOG" 2>&1; then
  echo "seed $SEED already complete and current for the recorded solve -- nothing to do" | tee -a "$LOG"
  exit 0
fi

S=$(date +%s)
# --spec makes the run VERIFY, before it records anything, that the tables it read are
# the ones the spec declares. Without it a summary could carry a spec_id for an economy
# it did not build -- which is the failure the whole registry exists to prevent.
python -W ignore run_oracle.py --model "$MODEL" --N "$N" --T "$T" --seed "$SEED" \
       --tag "$TAG" --spec "$SPEC" --eval_window "$WINDOW" --levels --save_panel 2>&1 | tee -a "$LOG"
MID=$(date +%s)
echo "=== oracle done in $((MID-S))s ===" | tee -a "$LOG"

python -W ignore run_estimators.py --model "$MODEL" --tag "$TAG" --seed "$SEED" \
       --window "$WINDOW" --levels --include_mkt --kappas "$KAPPAS" \
       --n_jobs "$NT" 2>&1 | tee -a "$LOG"
E=$(date +%s)

echo "=== seed $SEED END $(date '+%F %T') oracle=$((MID-S))s estimators=$((E-MID))s total=$((E-S))s ===" | tee -a "$LOG"

# What this replication was actually built from. `python variants/solfiles.py show <id>`
# expands any of these ids into the full parameter set that produced it.
python common/runstamp.py is-current "$RUNJSON" --model "$MODEL" --tag "${SOLVE_TAG:-$TAG}" | tee -a "$LOG"
