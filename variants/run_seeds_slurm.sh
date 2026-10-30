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
#
#   Proposed economies added 2026-09-10 (docs/RESULTS.md, "Open proposals"):
#     g0235f g0235s g0235r   BGN regime-gamma at 4x faster / 5x slower / rare-and-short
#     gx7                    GS gamma(x) crossed with the five exposure types
#
#   Proposed 2026-09-13 (docs/RESULTS.md finding 8, "Open proposals"):
#     vyg25    K4  vyx with gamma_v 2.5 (its own tables, prefix vyg25)
#     g0235d   B4  g0235 stressed two thirds of the time
#
#   vyxT860 (X3) was removed 2026-09-15: its economy IS vyx's, and it differed only in T, the
#   window and the ridge grid -- numerical choices, which the protocol now fixes for everyone.
#
#   The three BASELINES, added 2026-09-14 (A1, docs/NEXTUP.md; docs/RESULTS.md "Baselines: the
#   anchor"): each paper's economy as published, run through the same pipeline as every path that
#   departs from it -- including, since 2026-09-15, the same conditioning columns. A baseline
#   used to narrow them (RF_COLS) to the state its paper has, which made baseline and
#   parameterization differ in their feature bases as well as their economics:
#     bgnbase  A1  bgn_gam at gmult [1,1], the regime inert
#     kpbase   A1  kp_vy with one type at beta 0, no priced state
#     gsbase   A1  gs_bx one type at gmreg [1,1], the regime inert
#
#   SEED_STAGE=linear (E1) re-scores a SAVED panel's linear methods, plus the fair ones, reading
#   panel/moments/oracle from BOP_INPUTS_DIR (default results/) and writing to BOP_RESULTS_DIR,
#   which must be a DIFFERENT directory -- the output names equal the recorded run's:
#     sbatch --export=ALL,SEED_SPEC=g28,SEED_STAGE=linear,BOP_RESULTS_DIR=/data/.../e1 --array=0-9 \
#            variants/run_seeds_slurm.sh
#
#   SEED_STAGE=oracle runs ONLY the oracle stage, and BOP_RESULTS_DIR sends its output
#   somewhere other than variants/results. Together they re-derive population ceilings
#   for panels that already exist without touching the committed record:
#
#     sbatch --export=ALL,SEED_SPEC=vyx,SEED_STAGE=oracle,BOP_RESULTS_DIR=/data/.../evalwin \
#            --array=0-9 variants/run_seeds_slurm.sh
#
#   The panel is a deterministic function of (spec, seed), so an oracle re-run reproduces
#   the one the estimators were scored on; the all-month figures must come back unchanged,
#   which is the check that says the re-run is the same economy (WORKING.md §48).
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
# canonicaliser was made portable -- WORKING.md §26).
#
# SUPERSEDED 2026-09-21 for the nine kp_vy and bgn_gam economies: the pricing fix of merge
# 91095fb re-keyed every one of them (WORKING.md §63). The list below is kept as the
# 2026-09-07 record; the LIVE ids are each spec's expected_solves, and until the tables are
# rebuilt those nine specs declare solves_pending. The five gs_bx ids are unchanged.
#     kp_vy   vyx    G f7be27e39d2b530f   integ 84e195172f091cd2
#     bgn_gam g0235  jstar be222462dd017b2c
#     gs_bx   g28    sol_g28 8b584c38614695ac
#     bgn_gam g0235f jstar 8662d7c1079f4b41   g0235s jstar 05ad848ab3779695
#     bgn_gam g0235r jstar 0c624174cf4b26fa
#     gs_bx   gx7    sol_g28 8b584c38614695ac  sol_gx25 a145001bf661632e
#                    sol_gx40 bcc4e59f41f4a040  sol_gx55 9bc863211011b5d9
#                    sol_gx70 89546b0b36dfd2d0
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

: "${SEED_SPEC:?set SEED_SPEC (vyx | g0235 | g0235f | g0235s | g0235r | g28 | bx7 | gx7 | vyg25 | vym3 | vym3t3 | g0235d | bgnbase | kpbase | gsbase | g0235df | vyg30k70 | gx7w | vyg25k15 | vyg25k70 | kpgam | vyj25 | vy1 | vym2t3 | vy1j | kpq | vy9t2 | vy9t2f | bvy25 | bvyc0 | gvy25 | gvyc25) -- e.g. sbatch --export=ALL,SEED_SPEC=vyx ...}"

case "$SEED_SPEC" in
  vyx)
    MODEL=kp_vy; TAG=vyx; SPEC=var-kp_vy-vyx-v5; SOLVE_TAG=vyx
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
    MODEL=bgn_gam; TAG=g0235; SPEC=var-bgn_gam-g0235-v4; SOLVE_TAG=Jstar_g0235
    export BGN_PARAM_OVERRIDES='{"gmult":[0.2,3.5],"jstar_gam_file":"Jstar_g0235.csv"}'
    SOLVE_HINT='cd variants/bgn_gam && python rebuild_jstar_gam.py'
    ;;
  g0235f)
    # B1, the regime-PERSISTENCE ladder (docs/RESULTS.md). gmult stays at g0235's
    # [0.2, 3.5]; only the switch probabilities move, and each point needs its own J*
    # table because p01/p10 enter the solve through Preg.
    # 4x FASTER than g0235: calm 12mo, stress 6mo, ~40 switches inside a 360-month window.
    # Stationary stress share held at EXACTLY 1/3, as g0235 has it, so this is the same
    # economy at a different speed -- which is what makes it a clean test of the
    # rolling-window mechanism behind gap > room (WORKING.md §40, §49).
    MODEL=bgn_gam; TAG=g0235f; SPEC=var-bgn_gam-g0235f-v3; SOLVE_TAG=Jstar_g0235f
    export BGN_PARAM_OVERRIDES='{"gmult": [0.2, 3.5], "p01": 0.08333333333333333, "p10": 0.16666666666666666, "jstar_gam_file": "Jstar_g0235f.csv"}'
    SOLVE_HINT='cd variants/bgn_gam && BGN_PARAM_OVERRIDES=<the blob above> python rebuild_jstar_gam.py   # ~4-8 min'
    ;;
  g0235s)
    # B1, the regime-PERSISTENCE ladder (docs/RESULTS.md). gmult stays at g0235's
    # [0.2, 3.5]; only the switch probabilities move, and each point needs its own J*
    # table because p01/p10 enter the solve through Preg.
    # 5x SLOWER: calm 240mo, stress 120mo, ~2 switches inside a 360-month window.
    # Stationary stress share also held at 1/3. Paired with g0235f as a 20x speed ladder:
    # if the gap is flat across it, the rolling-window explanation is refuted.
    MODEL=bgn_gam; TAG=g0235s; SPEC=var-bgn_gam-g0235s-v3; SOLVE_TAG=Jstar_g0235s
    export BGN_PARAM_OVERRIDES='{"gmult": [0.2, 3.5], "p01": 0.004166666666666667, "p10": 0.008333333333333333, "jstar_gam_file": "Jstar_g0235s.csv"}'
    SOLVE_HINT='cd variants/bgn_gam && BGN_PARAM_OVERRIDES=<the blob above> python rebuild_jstar_gam.py   # ~4-8 min'
    ;;
  g0235r)
    # B1, the regime-PERSISTENCE ladder (docs/RESULTS.md). gmult stays at g0235's
    # [0.2, 3.5]; only the switch probabilities move, and each point needs its own J*
    # table because p01/p10 enter the solve through Preg.
    # RARE AND SHORT: stress 10% of months, calm 240mo, stress 27mo.
    # This one moves the stationary MIX as well as the speed, so it is a different
    # economy rather than a different speed. It is the crash-like shape, and the question
    # is whether a window holding ~36 stress months can still learn the interaction.
    MODEL=bgn_gam; TAG=g0235r; SPEC=var-bgn_gam-g0235r-v3; SOLVE_TAG=Jstar_g0235r
    export BGN_PARAM_OVERRIDES='{"gmult": [0.2, 3.5], "p01": 0.004166666666666667, "p10": 0.0375, "jstar_gam_file": "Jstar_g0235r.csv"}'
    SOLVE_HINT='cd variants/bgn_gam && BGN_PARAM_OVERRIDES=<the blob above> python rebuild_jstar_gam.py   # ~4-8 min'
    ;;
  bx7)
    # GS21 with five exposure types beta in {1, 2.5, 4, 5.5, 7}, one solve per type. FIVE
    # solve tags, so SOLVE_TAG lists them all and every runstamp lookup below unions them
    # (runstamp.live_solves takes a comma list; until 2026-09-09 it took one tag, and the
    # per-seed checkpoint would have reported STALE forever for this economy). The
    # simulator reads GS_BX_* for which types exist; the solutions carry their own gs_bx
    # and gs_sim_bx.py refuses a ladder that disagrees with them. GS_SIM_OVERRIDES stays
    # UNSET (WORKING.md §42). The kappa grid is the spec's eight values.
    MODEL=gs_bx; TAG=bx7; SPEC=var-gs_bx-bx7-v5; SOLVE_TAG=sol_reg,sol_b25c,sol_b40c,sol_b55c,sol_b70c
    export GS_BX_SOLDIRS=sol_reg,sol_b25c,sol_b40c,sol_b55c,sol_b70c
    export GS_BX_BETAS=1.0,2.5,4.0,5.5,7.0
    export GS_BX_SHARES=0.2,0.2,0.2,0.2,0.2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='python variants/fetch_solves.py --spec var-gs_bx-bx7-v5 --from "<the shared solves folder, e.g. the Dropbox solves/ dir>"'
    ;;
  gx7)
    # G1: gamma(x) crossed with the five exposure types (docs/RESULTS.md). Type f's
    # premium is proportional to beta_f * gamma(x), a product that is nonlinear in
    # (beta_f, x) because gamma is clipped -- the state bends a heterogeneous exposure
    # map, which is the rule that produced vyx, here with GS21's own state.
    # FIVE solve tags, and the first of them is sol_g28 REUSED: beta = 1 at this slope
    # is the g28 economy, so only four solves are new (run_gx7_slurm.sh).
    MODEL=gs_bx; TAG=gx7; SPEC=var-gs_bx-gx7-v3
    SOLVE_TAG=sol_g28,sol_gx25,sol_gx40,sol_gx55,sol_gx70
    export GS_BX_SOLDIRS=sol_g28,sol_gx25,sol_gx40,sol_gx55,sol_gx70
    export GS_BX_BETAS=1.0,2.5,4.0,5.5,7.0
    export GS_BX_SHARES=0.2,0.2,0.2,0.2,0.2
    unset GS_SIM_OVERRIDES
    # Built 2026-09-10 on Phoenix, every id matching its precommitment, and published. So the
    # hint is FETCH: remaking them is four ~5 h solves (run_gx7_slurm.sh, if ever needed).
    SOLVE_HINT='python variants/fetch_solves.py --spec var-gs_bx-gx7-v3 --from "<the shared solves folder, e.g. the Dropbox solves/ dir>"'
    ;;
  g28)
    # GS21, one type, gamma(x) = clip(0.5 - 0.28 x/sd(x), 0.05, 1.0): the reconstruction of
    # REPORT §13g's economy, where the published table showed a gap with ZERO nonlinear
    # room. Its parameters are not an override blob: the solver read GS_PARAM_OVERRIDES
    # when it built sol_g28, and that is inside the solve_id the spec pins. The simulator
    # takes its structural parameters out of solution.npz; the three GS_BX_* variables are
    # its only statement of which types exist, and GS_SIM_OVERRIDES must stay UNSET --
    # run_oracle.py refuses an undeclared value there (WORKING.md §42).
    MODEL=gs_bx; TAG=g28; SPEC=var-gs_bx-g28-v4; SOLVE_TAG=sol_g28
    export GS_BX_SOLDIRS=sol_g28
    export GS_BX_BETAS=1.0
    export GS_BX_SHARES=1.0
    unset GS_SIM_OVERRIDES
    # The solve exists and is published content-addressed; 3-6 h to remake, seconds to
    # fetch. The hint says fetch, never re-solve.
    SOLVE_HINT='python variants/fetch_solves.py --spec var-gs_bx-g28-v4 --from "<the shared solves folder, e.g. the Dropbox solves/ dir>"'
    ;;
  vyg25)
    # K4: vyx with the price of the state's risk raised from 1.8 to 2.5 -- the dial finding 8
    # says matters, since the Sharpe the state's shock carries is the part the market misses.
    # Its own tables under prefix vyg25, built by variants/run_solve_slurm.sh from the spec.
    MODEL=kp_vy; TAG=vyg25; SPEC=var-kp_vy-vyg25-v4; SOLVE_TAG=vyg25
    export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":2.5,"bv_comp":1.2}'
    export KP_VY_PREFIX=vyg25
    SOLVE_HINT='sbatch --export=ALL,SOLVE_SPEC=var-kp_vy-vyg25-v4 variants/run_solve_slurm.sh'
    ;;
  vym3)
    # The first economy in this project that raises K, the number of priced aggregate shocks
    # (docs/RESULTS.md finding 11): THREE independent priced OU states instead of one, and twenty
    # types whose loading VECTORS span all three. The total price of y-risk is UNCHANGED --
    # ||gamma|| = 2.5, vyg25's scalar value, split as 2.5/sqrt(3) each -- so this isolates
    # DIMENSION from LEVEL rather than drifting back toward the calibration the pricing fix
    # removed. A type's premium tracks b_f . gamma while its exposure magnitude tracks ||b_f||;
    # with one state those are the same number times a constant, with three they are a PRODUCT,
    # which no linear map in the characteristics represents. Twenty types and not six because the
    # fair linear side carries up to eleven columns (linlev_m) and would otherwise interpolate.
    # No product grid: each type is the scalar 1-D problem at its own projection.
    MODEL=kp_vy; TAG=vym3; SPEC=var-kp_vy-vym3-v1; SOLVE_TAG=vym3
    export KP_PARAM_OVERRIDES='{"type_share":[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],"type_bv":[[0.0115470054,0.0115470054,0.0115470054],[-0.0005603547,0.0141365838,0.0141365838],[0.0069282032,-0.0043855053,0.0182419117],[-0.0103478274,0.0121021169,0.0121021169],[0.0023094011,-0.0115470054,0.0161658075],[0.0346410162,0.0346410162,0.0346410162],[-0.001681064,0.0424097514,0.0424097514],[0.0207846097,-0.0131565158,0.0547257352],[-0.0310434822,0.0363063508,0.0363063508],[0.0069282032,-0.0346410162,0.0484974226],[0.0577350269,0.0577350269,0.0577350269],[-0.0028017733,0.070682919,0.070682919],[0.0346410162,-0.0219275263,0.0912095586],[-0.051739137,0.0605105846,0.0605105846],[0.0115470054,-0.0577350269,0.0808290377],[0.0808290377,0.0808290377,0.0808290377],[-0.0039224826,0.0989560865,0.0989560865],[0.0484974226,-0.0306985369,0.1276933821],[-0.0724347918,0.0847148185,0.0847148185],[0.0161658075,-0.0808290377,0.1131606528]],"gamma_v":[1.443375673,1.443375673,1.443375673],"bv_comp":1.2}'
    export KP_VY_PREFIX=vym3
    SOLVE_HINT='sbatch --export=ALL,SOLVE_SPEC=var-kp_vy-vym3-v1 variants/run_solve_slurm.sh'
    ;;
  vym3t3)
    # vym3's THREE priced states with THREE types instead of twenty: the experiment that separates
    # DIMENSION from ESTIMATION DENSITY (docs/RESULTS.md finding 15, docs/NEXTUP.md item 1). vym3
    # raised K from 3 to 5 AND went from three types to twenty, and its gate was falsified; at
    # N=500 twenty equal types is 25 firms per distinct premium against vyg25's 167, and nothing in
    # that run says which change cost the gap. The types here are SELECTED, not redesigned -- types
    # 3, 10 and 19 of vym3's own twenty, so the gamma vector, the directions and the magnitudes are
    # all vym3's. Rank 3 AND genuinely three-dimensional: per-state loading ratio 2.17 against
    # vym3's own 2.28, unit-direction condition 1.60 against 2.25. That fourth bar is what rejected
    # v1's triple (0, 9, 16), whose ratio was 7.10. Premium mean 0.1133 against vym3's 0.1200, so
    # the level is matched. The inversion is sharp: the LARGEST magnitude earns nearly the smallest
    # premium -- ||b||=0.10 earns 0.250 while ||b||=0.14 earns 0.070. Shares are vyg25's.
    MODEL=kp_vy; TAG=vym3t3; SPEC=var-kp_vy-vym3t3-v2; SOLVE_TAG=vym3t3
    export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[[-0.0103478274,0.0121021169,0.0121021169],[0.0577350269,0.0577350269,0.0577350269],[0.0161658075,-0.0808290377,0.1131606528]],"gamma_v":[1.443375673,1.443375673,1.443375673],"bv_comp":1.2}'
    export KP_VY_PREFIX=vym3t3
    SOLVE_HINT='sbatch --export=ALL,SOLVE_SPEC=var-kp_vy-vym3t3-v2 variants/run_solve_slurm.sh'
    ;;
  g0235d)
    # B4: g0235's multipliers with the switch probabilities SWAPPED, so stress is two thirds of
    # months (48-month stress spells, 24-month calm). Seed 0 is a SCREEN: seeds 1-9 only if its
    # evaluation-window room >= +0.05 and non-market Sharpe >= 0.30 (the spec's notes).
    MODEL=bgn_gam; TAG=g0235d; SPEC=var-bgn_gam-g0235d-v3; SOLVE_TAG=Jstar_g0235d
    export BGN_PARAM_OVERRIDES='{"gmult": [0.2, 3.5], "p01": 0.041666666666666664, "p10": 0.020833333333333332, "jstar_gam_file": "Jstar_g0235d.csv"}'
    SOLVE_HINT='sbatch --export=ALL,SOLVE_SPEC=var-bgn_gam-g0235d-v3 variants/run_solve_slurm.sh'
    ;;
  bgnbase)
    # A1: the BGN baseline as published -- gmult [1, 1], so both regimes price the market shock
    # at sigma_z 0.4 and the regime is inert (docs/RESULTS.md, "The three models as published"). Its own
    # J* table (the solve's tag is the file name). Until 2026-09-15 this case narrowed the
    # bases to the rate alone; it now sees the full set, gam_stand included, as every
    # economy does -- an inert regime is a noise column, and paying for it is the price of
    # baseline and parameterization being scored the same way.
    MODEL=bgn_gam; TAG=bgnbase; SPEC=var-bgn_gam-bgnbase-v3; SOLVE_TAG=Jstar_bgnbase
    export BGN_PARAM_OVERRIDES='{"gmult":[1.0,1.0],"jstar_gam_file":"Jstar_bgnbase.csv"}'
    SOLVE_HINT='cd variants/bgn_gam && BGN_PARAM_OVERRIDES=<the blob above> python rebuild_jstar_gam.py   # ~4-8 min; the table is committed'
    ;;
  bgnzr)
    # The GATE on BGN's discount channel (docs/NEXTUP.md item 2), not a gap candidate. bgnbase
    # with beta_zr -0.00014 -> -0.00020: a ROTATION of the price of risk from the cash-flow
    # channel onto the rate channel at constant total (sigma_z 0.4), which moves the rate
    # channel's share of the price from 15.1% to 20.5%. It stops there and not further because
    # beta_zr is the covariance the 2026-09-18 pricing fix corrected, and the limiting term
    # spread is now about twice as sensitive to it: 3.79%/yr here against BGN (1999, p.21)'s
    # 2.4%, and 8.5%/yr at the -0.00040 an earlier plan proposed (docs/RESULTS.md finding 13).
    # What this run buys is d(room)/d(rate price) over a 43% price step, which decides whether
    # a multi-factor term structure earns a new solver. BGN stays K=2, so finding 11 caps the
    # ratio near 1.02 -- a large fair gap here would falsify the rotation argument, not confirm
    # the economy.
    MODEL=bgn_gam; TAG=bgnzr; SPEC=var-bgn_gam-bgnzr-v1; SOLVE_TAG=Jstar_bgnzr
    export BGN_PARAM_OVERRIDES='{"beta_zr":-0.0002,"jstar_gam_file":"Jstar_bgnzr.csv"}'
    SOLVE_HINT='cd variants/bgn_gam && BGN_PARAM_OVERRIDES=<the blob above> python rebuild_jstar_gam.py   # ~3-4 min; the table is committed'
    ;;
  kpbase)
    # A1: the KP14 baseline as published (r = 0.05, the repository's standing departure) -- one
    # firm type at beta 0, no priced state (gamma_v 0), no compensation. The y process is still
    # simulated and exported, and nothing loads on it -- its 21 integral tables agree across y
    # to 4e-12. The bases see it anyway, as they do in vyx: a column no firm loads on is noise,
    # and withholding it from the baseline alone would make the baseline a different protocol.
    MODEL=kp_vy; TAG=kpbase; SPEC=var-kp_vy-kpbase-v4; SOLVE_TAG=kpbase
    export KP_PARAM_OVERRIDES='{"type_share":[1.0],"type_bv":[0.0],"gamma_v":0.0,"bv_comp":0.0}'
    export KP_VY_PREFIX=kpbase
    SOLVE_HINT='cd variants/kp_vy && KP_VY_PREFIX=kpbase KP_PARAM_OVERRIDES=<the blob above> python build_vy_tables.py kpbase   # the tables are committed'
    ;;
  gsbase)
    # A1: the GS21 baseline as published at the corrected Table I -- one type, the regime solver
    # at gmreg [1, 1] so gamma_s = gamma_x 0.5 in both regimes and the regime is inert. The
    # bases see the full set, gam_stand included, as every economy's do. The solve is hours
    # (run_gsbase_slurm.sh, chained afterok); the hint is FETCH, as for every GS economy.
    MODEL=gs_bx; TAG=gsbase; SPEC=var-gs_bx-gsbase-v3; SOLVE_TAG=sol_gsbase
    export GS_BX_SOLDIRS=sol_gsbase
    export GS_BX_BETAS=1.0
    export GS_BX_SHARES=1.0
    unset GS_SIM_OVERRIDES
    # Built on Sol 2026-09-15 (job 63257244: tolerance exit at sweep 3024, the precommitted id) and
    # published content-addressed, so the hint is FETCH. Remaking it is one 5.6 h solve
    # (run_gsbase_slurm.sh).
    SOLVE_HINT='python variants/fetch_solves.py --spec var-gs_bx-gsbase-v3 --from "<the shared solves folder, e.g. the Dropbox solves/ dir>"'
    ;;
  g0235df)
    # g0235d's economy at g0235f's speed (2026-10-05, Quest): both switch probabilities x4, so the
    # stationary stress share stays at exactly 2/3 while stress spells go 48 -> 12 months and calm
    # 24 -> 6. Combines finding 12's speed lever with the mix that has BGN's largest DKKM-over-FMR/FF
    # gap. Its own J* table, because p01/p10 enter the solve through Preg.
    MODEL=bgn_gam; TAG=g0235df; SPEC=var-bgn_gam-g0235df-v1; SOLVE_TAG=Jstar_g0235df
    export BGN_PARAM_OVERRIDES='{"gmult": [0.2, 3.5], "p01": 0.16666666666666666, "p10": 0.08333333333333333, "jstar_gam_file": "Jstar_g0235df.csv"}'
    SOLVE_HINT='sbatch variants/klc/solve_small_klc.sh   # Quest; or cd variants/bgn_gam && BGN_PARAM_OVERRIDES=<the blob above> python rebuild_jstar_gam.py'
    ;;
  vyg30k70)
    # vyg25 with the priced state twice as fast (kappa_y 0.35 -> 0.70) and its price raised 2.5 -> 3.0
    # to hold the premium spread (2026-10-05, Quest; cf. K3 in docs/NEXTUP.md). NOT a one-parameter
    # experiment: built to be likely to show a gap, not to isolate kappa_y. Its own tables under
    # prefix vyg30k70, built as one chain on Quest so the integ id is the one those G bytes give.
    MODEL=kp_vy; TAG=vyg30k70; SPEC=var-kp_vy-vyg30k70-v1; SOLVE_TAG=vyg30k70
    export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":3.0,"kappa_y":0.7,"bv_comp":1.2}'
    export KP_VY_PREFIX=vyg30k70
    SOLVE_HINT='sbatch variants/klc/solve_small_klc.sh   # Quest; or cd variants/kp_vy && KP_PARAM_OVERRIDES=<the blob above> python build_vy_tables.py vyg30k70'
    ;;
  vyg25k15)
    # K3 (docs/NEXTUP.md item 1), 2026-10-07 on Quest: vyg25 with ONE parameter changed, kappa_y 0.35 -> 0.15
    # (half-life 24 -> 55 months; sigma_y = sqrt(2 kappa_y) moves with it). gamma_v stays 2.5. Paired with
    # its twin to test whether the fair gap moves with the speed of the priced state (finding 12) or is
    # flat in it (the reduced-form screen's prediction, docs/REDUCED_FORM.md). Own tables under prefix vyg25k15.
    MODEL=kp_vy; TAG=vyg25k15; SPEC=var-kp_vy-vyg25k15-v1; SOLVE_TAG=vyg25k15
    export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":2.5,"bv_comp":1.2,"kappa_y":0.15}'
    export KP_VY_PREFIX=vyg25k15
    SOLVE_HINT='sbatch --export=ALL,KSPECS=var-kp_vy-vyg25k15-v1 variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vyg25k70)
    # K3 (docs/NEXTUP.md item 1), 2026-10-07 on Quest: vyg25 with ONE parameter changed, kappa_y 0.35 -> 0.7
    # (half-life 24 -> 12 months; sigma_y = sqrt(2 kappa_y) moves with it). gamma_v stays 2.5. Paired with
    # its twin to test whether the fair gap moves with the speed of the priced state (finding 12) or is
    # flat in it (the reduced-form screen's prediction, docs/REDUCED_FORM.md). Own tables under prefix vyg25k70.
    MODEL=kp_vy; TAG=vyg25k70; SPEC=var-kp_vy-vyg25k70-v1; SOLVE_TAG=vyg25k70
    export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":2.5,"bv_comp":1.2,"kappa_y":0.7}'
    export KP_VY_PREFIX=vyg25k70
    SOLVE_HINT='sbatch --export=ALL,KSPECS=var-kp_vy-vyg25k70-v1 variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  gx7w)
    # gx7 with a steeper and wider price of risk: gamma(x) = clip(0.5 - 0.40 x/sd(x), 0.05, 1.5)
    # against gx7's slope 0.28 and ceiling 1.0 (2026-10-05, Quest). Same five-type ladder. GS21 is
    # K=1, so no FAIR gap is expected; this targets DKKM against FMR/FF, where gx7 holds the file's
    # largest absolute gap. FIVE new solves, none reused. GS_SIM_OVERRIDES stays UNSET.
    MODEL=gs_bx; TAG=gx7w; SPEC=var-gs_bx-gx7w-v1
    SOLVE_TAG=sol_gw10,sol_gw25,sol_gw40,sol_gw55,sol_gw70
    export GS_BX_SOLDIRS=sol_gw10,sol_gw25,sol_gw40,sol_gw55,sol_gw70
    export GS_BX_BETAS=1.0,2.5,4.0,5.5,7.0
    export GS_BX_SHARES=0.2,0.2,0.2,0.2,0.2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='python variants/fetch_solves.py --spec var-gs_bx-gx7w-v1 --from "<the shared solves folder>"   # or, on Quest, re-solve: sbatch variants/klc/solve_gs_klc.sh (five ~5 h solves)'
    ;;
  bvy25)
    # The KP14 construction in BGN (2026-10-07): a priced AR(1) state y, three exposure types, bv_comp 1.2,
    # gamma_v 2.5, kappa_y 0.35 -- vyg25's ingredients, nothing else changed from bgnbase. variants/bgn_vy/.
    MODEL=bgn_vy; TAG=bvy25; SPEC=var-bgn_vy-bvy25-v1; SOLVE_TAG=Jstar_vy_bvy25
    export BGN_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "jstar_vy_file": "Jstar_vy_bvy25.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvy25-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest, ~2 min + gate'
    ;;
  gvy25)
    # The KP14 construction in GS21 (2026-10-07): a priced 9-node y chain, three exposure types on it with
    # a derived level compensation, gamma_v 2.5, kappa_y 0.35. variants/gs_vy/. Three solves, hours each.
    MODEL=gs_vy; TAG=gvy25; SPEC=var-gs_vy-gvy25-v1
    SOLVE_TAG=sol_gvy25a,sol_gvy25b,sol_gvy25c
    export GS_VY_SOLDIRS=sol_gvy25a,sol_gvy25b,sol_gvy25c
    export GS_VY_BV=0.015,0.05,0.1
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy25-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyc25)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyc25; SPEC=var-gs_vy-gvyc25-v1
    SOLVE_TAG=sol_gvyc25a,sol_gvyc25b,sol_gvyc25c
    export GS_VY_SOLDIRS=sol_gvyc25a,sol_gvyc25b,sol_gvyc25c
    export GS_VY_BV=0.0,0.0,0.0
    export GS_VY_BC=-0.05,-0.2,-0.4
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyc25-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvys25)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvys25; SPEC=var-gs_vy-gvys25-v1
    SOLVE_TAG=sol_gvys25a,sol_gvys25b,sol_gvyc25c
    export GS_VY_SOLDIRS=sol_gvys25a,sol_gvys25b,sol_gvyc25c
    export GS_VY_BV=0.0,0.0,0.0
    export GS_VY_BC=0.4,0.0,-0.4
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvys25-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyi25)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyi25; SPEC=var-gs_vy-gvyi25-v1
    SOLVE_TAG=sol_gvyi25a,sol_gvyi25b,sol_gvyi25c
    export GS_VY_SOLDIRS=sol_gvyi25a,sol_gvyi25b,sol_gvyi25c
    export GS_VY_BV=0.0,0.0,0.0
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=-0.2,-0.6,-1.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyi25-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyk25)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyk25; SPEC=var-gs_vy-gvyk25-v1
    SOLVE_TAG=sol_gvyk25a,sol_gvyk25b,sol_gvyk25c
    export GS_VY_SOLDIRS=sol_gvyk25a,sol_gvyk25b,sol_gvyk25c
    export GS_VY_BV=0.0,0.0,0.0
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=-0.03,-0.1,-0.15
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyk25-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  bvyc0)
    # bgn_vy follow-up (2026-10-10): bvy25 with the type's level compensation OFF (bv_comp 0): the type is a plain value sort
    MODEL=bgn_vy; TAG=bvyc0; SPEC=var-bgn_vy-bvyc0-v1; SOLVE_TAG=Jstar_vy_bvyc0
    export BGN_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 0.0, "jstar_vy_file": "Jstar_vy_bvyc0.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvyc0-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvyc10)
    # bgn_vy follow-up (2026-10-10): bvy25 with exact level compensation (bv_comp 1.0): the type is invisible in value at y = 0
    MODEL=bgn_vy; TAG=bvyc10; SPEC=var-bgn_vy-bvyc10-v1; SOLVE_TAG=Jstar_vy_bvyc10
    export BGN_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.0, "jstar_vy_file": "Jstar_vy_bvyc10.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvyc10-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvysg)
    # bgn_vy follow-up (2026-10-10): BGN y-factor with a SIGNED exposure ladder (-0.07, 0, +0.07), no compensation
    MODEL=bgn_vy; TAG=bvysg; SPEC=var-bgn_vy-bvysg-v1; SOLVE_TAG=Jstar_vy_bvysg
    export BGN_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [-0.07, 0.0, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 0.0, "jstar_vy_file": "Jstar_vy_bvysg.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvysg-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  gvyj25)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyj25; SPEC=var-gs_vy-gvyj25-v1
    SOLVE_TAG=sol_gvyj25a,sol_gvyj25b,sol_gvyj25c
    export GS_VY_SOLDIRS=sol_gvyj25a,sol_gvyj25b,sol_gvyj25c
    export GS_VY_BV=0.0,0.0,0.0
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,-0.25,-0.5
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyj25-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyp20)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp20; SPEC=var-gs_vy-gvyp20-v1
    SOLVE_TAG=sol_gvyp20a
    export GS_VY_SOLDIRS=sol_gvyp20a
    export GS_VY_BV=0.2
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp20-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyp10)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp10; SPEC=var-gs_vy-gvyp10-v1
    SOLVE_TAG=sol_gvyp10a
    export GS_VY_SOLDIRS=sol_gvyp10a
    export GS_VY_BV=0.1
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp10-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyp05)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp05; SPEC=var-gs_vy-gvyp05-v1
    SOLVE_TAG=sol_gvyp05a
    export GS_VY_SOLDIRS=sol_gvyp05a
    export GS_VY_BV=0.05
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp05-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyp075)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp075; SPEC=var-gs_vy-gvyp075-v1
    SOLVE_TAG=sol_gvyp075a
    export GS_VY_SOLDIRS=sol_gvyp075a
    export GS_VY_BV=0.075
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp075-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyp15)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp15; SPEC=var-gs_vy-gvyp15-v1
    SOLVE_TAG=sol_gvyp15a
    export GS_VY_SOLDIRS=sol_gvyp15a
    export GS_VY_BV=0.15
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp15-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  kpgam)
    # kp_vy (2026-10-12): KP14 baseline with a COUNTER-CYCLICAL price of the productivity shock: gamma_x times a logistic of the state y, 1 to 3
    MODEL=kp_vy; TAG=kpgam; SPEC=var-kp_vy-kpgam-v1; SOLVE_TAG=kpgam
    export KP_PARAM_OVERRIDES='{"type_share": [1.0], "type_bv": [0.0], "gamma_v": 0.0, "bv_comp": 0.0, "g_lo": 1.0, "g_hi": 3.0, "g_steep": -2.0}'
    export KP_VY_PREFIX=kpgam
    SOLVE_HINT='export KSPECS=var-kp_vy-kpgam-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  gvyp10x)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp10x; SPEC=var-gs_vy-gvyp10x-v1
    SOLVE_TAG=sol_gvyp10xa
    export GS_VY_SOLDIRS=sol_gvyp10xa
    export GS_VY_BV=0.1
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp10x-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyp10j)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp10j; SPEC=var-gs_vy-gvyp10j-v1
    SOLVE_TAG=sol_gvyp10ja
    export GS_VY_SOLDIRS=sol_gvyp10ja
    export GS_VY_BV=0.1
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp10j-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  bvych0)
    # bgn_vy follow-up (2026-10-10): bvy25 with y as a 21-node Tauchen chain, no crash: the control for the crash economy
    MODEL=bgn_vy; TAG=bvych0; SPEC=var-bgn_vy-bvych0-v1; SOLVE_TAG=Jstar_vy_bvych0
    export BGN_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvych0.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvych0-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvyj25)
    # bgn_vy follow-up (2026-10-10): BGN y-factor economy with a rare priced CRASH in the state y: a different distribution of shocks
    MODEL=bgn_vy; TAG=bvyj25; SPEC=var-bgn_vy-bvyj25-v1; SOLVE_TAG=Jstar_vy_bvyj25
    export BGN_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "pj": 0.01, "gj": 2.0, "jstar_vy_file": "Jstar_vy_bvyj25.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvyj25-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  vyj25)
    # kp_vy (2026-10-12): KP14 vyg25 with a CRASH in the priced state: a jump of y to -y_max at rate 1% a month, priced at e^2
    MODEL=kp_vj; TAG=vyj25; SPEC=var-kp_vj-vyj25-v1; SOLVE_TAG=vyj25
    export KP_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.02, 0.07, 0.14], "gamma_v": 2.5, "bv_comp": 1.2, "pj": 0.01, "gj": 2.0}'
    export KP_VY_PREFIX=vyj25
    SOLVE_HINT='export KSPECS=var-kp_vj-vyj25-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  bvyk2)
    # bgn_vy follow-up (2026-10-10): BGN with TWO independent priced chains (K 1 -> 2 priced states beyond the rate) and three types whose loading vectors span them, at bvych0's loading sizes and total price
    MODEL=bgn_vy; TAG=bvyk2; SPEC=var-bgn_vy-bvyk2-v1; SOLVE_TAG=Jstar_vy_bvyk2
    export BGN_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.01, 0.0, 0.04949747], "gamma_v": 1.76776695, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "NY2": 21, "type_bv2": [0.0, 0.035, 0.04949747], "gamma_v2": 1.76776695, "kappa_y2": 0.35, "jstar_vy_file": "Jstar_vy_bvyk2.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvyk2-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  gvypk2)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvypk2; SPEC=var-gs_vy-gvypk2-v1
    SOLVE_TAG=sol_gvypk2a,sol_gvypk2b
    export GS_VY_SOLDIRS=sol_gvypk2a,sol_gvypk2b
    export GS_VY_BV=0.1,0.0
    export GS_VY_BC=0.0,0.0
    export GS_VY_BI=0.0,0.0
    export GS_VY_BK=0.0,0.0
    export GS_VY_SHARES=0.5,0.5
    export GS_VY_BV2=0.0,0.1
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvypk2-v1; sbatch --export=ALL --array=0-1 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  vy1)
    # kp_vy (2026-10-12): KP14 with the priced state y and ONE type, exposure 0.07 (group 3a: exposure through a single firm state)
    MODEL=kp_vy; TAG=vy1; SPEC=var-kp_vy-vy1-v1; SOLVE_TAG=vy1
    export KP_PARAM_OVERRIDES='{"type_share": [1.0], "type_bv": [0.07], "gamma_v": 2.5, "bv_comp": 1.2}'
    export KP_VY_PREFIX=vy1
    SOLVE_HINT='export KSPECS=var-kp_vy-vy1-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vym2t3)
    # kp_vy (2026-10-12): KP14 with TWO independent priced states and three types loading (0.02,0), (0,0.07), (0.099,0.099): vyg25's ladder in three directions, total price 2.5
    MODEL=kp_vy; TAG=vym2t3; SPEC=var-kp_vy-vym2t3-v1; SOLVE_TAG=vym2t3
    export KP_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [[0.02, 0.0], [0.0, 0.07], [0.0989949, 0.0989949]], "gamma_v": [1.76776695, 1.76776695], "bv_comp": 1.2}'
    export KP_VY_PREFIX=vym2t3
    SOLVE_HINT='export KSPECS=var-kp_vy-vym2t3-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  bvy1ch)
    # bgn_vy follow-up (2026-10-10): BGN with the priced y chain and ONE type, exposure 0.035 (group 3a: exposure through a single firm state)
    MODEL=bgn_vy; TAG=bvy1ch; SPEC=var-bgn_vy-bvy1ch-v1; SOLVE_TAG=Jstar_vy_bvy1ch
    export BGN_PARAM_OVERRIDES='{"type_share": [1.0], "type_bv": [0.035], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvy1ch.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvy1ch-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  gvy1)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvy1; SPEC=var-gs_vy-gvy1-v1
    SOLVE_TAG=sol_gvy1a
    export GS_VY_SOLDIRS=sol_gvy1a
    export GS_VY_BV=0.05
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy1-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvypk3)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvypk3; SPEC=var-gs_vy-gvypk3-v1
    SOLVE_TAG=sol_gvypk2a,sol_gvypk2b,sol_gvypk3c
    export GS_VY_SOLDIRS=sol_gvypk2a,sol_gvypk2b,sol_gvypk3c
    export GS_VY_BV=0.1,0.0,0.0707107
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    export GS_VY_BV2=0.0,0.1,0.0707107
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvypk3-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  vy1j)
    # kp_vy (2026-10-12): KP14 with a CRASH-ONLY cash-flow factor: one type, cash flows x e^{0.07 y}, y's Gaussian shocks unpriced, a priced crash of y to -3.5 sd at 1%/month (weight e^2)
    MODEL=kp_vj; TAG=vy1j; SPEC=var-kp_vj-vy1j-v1; SOLVE_TAG=vy1j
    export KP_PARAM_OVERRIDES='{"type_share": [1.0], "type_bv": [0.07], "gamma_v": 0.0, "bv_comp": 1.2, "pj": 0.01, "gj": 2.0}'
    export KP_VY_PREFIX=vy1j
    SOLVE_HINT='export KSPECS=var-kp_vj-vy1j-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  bvy1j)
    # bgn_vy follow-up (2026-10-10): BGN with a CRASH-ONLY cash-flow factor: one type, project cash flows x e^{0.035 y}, y's Gaussian shocks unpriced, a priced crash of y to -3.5 sd at 1%/month (weight e^2)
    MODEL=bgn_vy; TAG=bvy1j; SPEC=var-bgn_vy-bvy1j-v1; SOLVE_TAG=Jstar_vy_bvy1j
    export BGN_PARAM_OVERRIDES='{"type_share": [1.0], "type_bv": [0.035], "gamma_v": 0.0, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "pj": 0.01, "gj": 2.0, "jstar_vy_file": "Jstar_vy_bvy1j.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvy1j-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  gvy1j)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvy1j; SPEC=var-gs_vy-gvy1j-v1
    SOLVE_TAG=sol_gvy1ja
    export GS_VY_SOLDIRS=sol_gvy1ja
    export GS_VY_BV=0.05
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy1j-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  kpq)
    # kp_vy (2026-10-12): KP14 with a QUADRATIC price of productivity risk: gamma_x x min(1 + 0.25 y^2, 3), y an unpriced stationary state no firm is exposed to
    MODEL=kp_vy; TAG=kpq; SPEC=var-kp_vy-kpq-v1; SOLVE_TAG=kpq
    export KP_PARAM_OVERRIDES='{"type_share": [1.0], "type_bv": [0.0], "gamma_v": 0.0, "bv_comp": 0.0, "g_quad": 0.25}'
    export KP_VY_PREFIX=kpq
    SOLVE_HINT='export KSPECS=var-kp_vy-kpq-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  gq25)
    # GS21, one type, gamma(x) = min(0.5 (1 + 0.25 (x/sd(x))^2), 1.5): the group-2 quadratic price (2026-10-14)
    MODEL=gs_bx; TAG=gq25; SPEC=var-gs_bx-gq25-v1; SOLVE_TAG=sol_gq25
    export GS_BX_SOLDIRS=sol_gq25
    export GS_BX_BETAS=1.0
    export GS_BX_SHARES=1.0
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSBX_SPEC=var-gs_bx-gq25-v1; sbatch --export=ALL --array=0 variants/klc/solve_gs_klc.sh   # Quest'
    ;;
  gvyz10)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyz10; SPEC=var-gs_vy-gvyz10-v1
    SOLVE_TAG=sol_gvyz10a
    export GS_VY_SOLDIRS=sol_gvyz10a
    export GS_VY_BV=0.1
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyz10-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyz10j)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyz10j; SPEC=var-gs_vy-gvyz10j-v1
    SOLVE_TAG=sol_gvyz10ja
    export GS_VY_SOLDIRS=sol_gvyz10ja
    export GS_VY_BV=0.1
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyz10j-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  bvych41)
    # bgn_vy follow-up (2026-10-10): bvych0 on a 41-node chain: does the 21-node discretization halve bvy25's gap?
    MODEL=bgn_vy; TAG=bvych41; SPEC=var-bgn_vy-bvych41-v1; SOLVE_TAG=Jstar_vy_bvych41
    export BGN_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "NY": 41, "jstar_vy_file": "Jstar_vy_bvych41.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvych41-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bq25)
    # bgn_vy follow-up (2026-10-10): BGN with a QUADRATIC price of market risk in the rate: sigma_z x min(1 + 0.25 s^2, 3), s the standardized rate, binned to 7 levels (group 2, harmonized)
    MODEL=bgn_q; TAG=bq25; SPEC=var-bgn_q-bq25-v1; SOLVE_TAG=Jstar_vy_bq25
    export BGN_PARAM_OVERRIDES='{"type_share": [1.0], "type_bv": [0.0], "gamma_v": 0.0, "kappa_y": 0.35, "bv_comp": 0.0, "q_price": 0.25, "NB": 7, "jstar_vy_file": "Jstar_vy_bq25.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_q-bq25-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  vy9t2)
    # kp_vy (2026-10-12): KP14 with TWO independent priced states and NINE types: vyg25's ladder (0.02, 0.07, 0.14) on each state, the two type indices independent, total price 2.5 (2.5/sqrt(2) per state)
    MODEL=kp_vy; TAG=vy9t2; SPEC=var-kp_vy-vy9t2-v1; SOLVE_TAG=vy9t2
    export KP_PARAM_OVERRIDES='{"type_share": [0.1112, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111], "type_bv": [[0.02, 0.02], [0.02, 0.07], [0.02, 0.14], [0.07, 0.02], [0.07, 0.07], [0.07, 0.14], [0.14, 0.02], [0.14, 0.07], [0.14, 0.14]], "gamma_v": [1.76776695, 1.76776695], "bv_comp": 1.2}'
    export KP_VY_PREFIX=vy9t2
    SOLVE_HINT='export KSPECS=var-kp_vy-vy9t2-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vy9t2f)
    # kp_vy (2026-10-12): KP14 with TWO independent priced states and NINE types, each state priced at vyg25's 2.5: vy9t2 without the price split
    MODEL=kp_vy; TAG=vy9t2f; SPEC=var-kp_vy-vy9t2f-v1; SOLVE_TAG=vy9t2f
    export KP_PARAM_OVERRIDES='{"type_share": [0.1112, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111], "type_bv": [[0.02, 0.02], [0.02, 0.07], [0.02, 0.14], [0.07, 0.02], [0.07, 0.07], [0.07, 0.14], [0.14, 0.02], [0.14, 0.07], [0.14, 0.14]], "gamma_v": [2.5, 2.5], "bv_comp": 1.2}'
    export KP_VY_PREFIX=vy9t2f
    SOLVE_HINT='export KSPECS=var-kp_vy-vy9t2f-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  bvy9t2)
    # bgn_vy follow-up (2026-10-10): BGN with TWO independent priced chains and NINE types: bvych0's ladder (0.01, 0.035, 0.07) on each chain, the two type indices independent, total price 2.5 (2.5/sqrt(2) per chain)
    MODEL=bgn_vy; TAG=bvy9t2; SPEC=var-bgn_vy-bvy9t2-v1; SOLVE_TAG=Jstar_vy_bvy9t2
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1112, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111], "type_bv": [0.01, 0.01, 0.01, 0.035, 0.035, 0.035, 0.07, 0.07, 0.07], "gamma_v": 1.76776695, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "NY2": 21, "type_bv2": [0.01, 0.035, 0.07, 0.01, 0.035, 0.07, 0.01, 0.035, 0.07], "gamma_v2": 1.76776695, "kappa_y2": 0.35, "jstar_vy_file": "Jstar_vy_bvy9t2.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvy9t2-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  vyg25s)
    # kp_vy (2026-10-26): vyg25 with SIGN-SWITCHING types: the ladder mirrored, six types (-0.14, -0.07, -0.02, 0.02, 0.07, 0.14), E[b] = 0, level-compensated, price 2.5
    MODEL=kp_vy; TAG=vyg25s; SPEC=var-kp_vy-vyg25s-v1; SOLVE_TAG=vyg25s
    export KP_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.14, -0.07, -0.02, 0.02, 0.07, 0.14], "gamma_v": 2.5, "bv_comp": 1.2}'
    export KP_VY_PREFIX=vyg25s
    SOLVE_HINT='export KSPECS=var-kp_vy-vyg25s-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vyg25s18)
    # kp_vy (2026-10-28): vyg25s at bv_comp 1.8 (the level-compensation bracket)
    MODEL=kp_vy; TAG=vyg25s18; SPEC=var-kp_vy-vyg25s18-v1; SOLVE_TAG=vyg25s18
    export KP_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.14, -0.07, -0.02, 0.02, 0.07, 0.14], "gamma_v": 2.5, "bv_comp": 1.8}'
    export KP_VY_PREFIX=vyg25s18
    SOLVE_HINT='export KSPECS=var-kp_vy-vyg25s18-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vyg25sx2)
    # kp_vy (2026-10-29): vyg25s with the loadings DOUBLED (-0.28 .. 0.28), E[b] = 0, level-compensated, price 2.5
    MODEL=kp_vy; TAG=vyg25sx2; SPEC=var-kp_vy-vyg25sx2-v1; SOLVE_TAG=vyg25sx2
    export KP_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.28, -0.14, -0.04, 0.04, 0.14, 0.28], "gamma_v": 2.5, "bv_comp": 1.2}'
    export KP_VY_PREFIX=vyg25sx2
    SOLVE_HINT='export KSPECS=var-kp_vy-vyg25sx2-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vyg25sx3)
    # kp_vy (2026-10-30): vyg25s with the loadings TRIPLED (-0.42 .. 0.42), E[b] = 0, level-compensated, price 2.5
    MODEL=kp_vy; TAG=vyg25sx3; SPEC=var-kp_vy-vyg25sx3-v1; SOLVE_TAG=vyg25sx3
    export KP_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.42, -0.21, -0.06, 0.06, 0.21, 0.42], "gamma_v": 2.5, "bv_comp": 1.2}'
    export KP_VY_PREFIX=vyg25sx3
    SOLVE_HINT='export KSPECS=var-kp_vy-vyg25sx3-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vyg25s6)
    # kp_vy (2026-10-28): vyg25s at bv_comp 0.6 (the level-compensation bracket)
    MODEL=kp_vy; TAG=vyg25s6; SPEC=var-kp_vy-vyg25s6-v1; SOLVE_TAG=vyg25s6
    export KP_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.14, -0.07, -0.02, 0.02, 0.07, 0.14], "gamma_v": 2.5, "bv_comp": 0.6}'
    export KP_VY_PREFIX=vyg25s6
    SOLVE_HINT='export KSPECS=var-kp_vy-vyg25s6-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vyg25u)
    # kp_vy (2026-10-26): vyg25 WITHOUT the level compensation (bv_comp 0): the type shows in value at y = 0
    MODEL=kp_vy; TAG=vyg25u; SPEC=var-kp_vy-vyg25u-v1; SOLVE_TAG=vyg25u
    export KP_PARAM_OVERRIDES='{"type_share": [0.34, 0.33, 0.33], "type_bv": [0.02, 0.07, 0.14], "gamma_v": 2.5, "bv_comp": 0.0}'
    export KP_VY_PREFIX=vyg25u
    SOLVE_HINT='export KSPECS=var-kp_vy-vyg25u-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  bvychs)
    # bgn_vy (2026-10-26): bvych0 with SIGN-SWITCHING types: the ladder mirrored, six types (-0.07 .. 0.07), E[b] = 0, chain, level-compensated, price 2.5
    MODEL=bgn_vy; TAG=bvychs; SPEC=var-bgn_vy-bvychs-v1; SOLVE_TAG=Jstar_vy_bvychs
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.07, -0.035, -0.01, 0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychs.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychs-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvychs8)
    # bgn_vy (2026-10-28): bvychs at bv_comp 0.8 (the level-compensation ladder; bvychs = 1.2 over-compensates)
    MODEL=bgn_vy; TAG=bvychs8; SPEC=var-bgn_vy-bvychs8-v1; SOLVE_TAG=Jstar_vy_bvychs8
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.07, -0.035, -0.01, 0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 0.8, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychs8.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychs8-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvychsn5)
    # bgn_vy (2026-10-29): bvychs at bv_comp -0.5 (hump search: the compensation as a channel opposed to y)
    MODEL=bgn_vy; TAG=bvychsn5; SPEC=var-bgn_vy-bvychsn5-v1; SOLVE_TAG=Jstar_vy_bvychsn5
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.07, -0.035, -0.01, 0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": -0.5, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychsn5.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychsn5-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvychsx3)
    # bgn_vy (2026-10-29): bvychs with the loadings x 3 (-0.21 .. 0.21), E[b] = 0, chain, level-compensated, price 2.5
    MODEL=bgn_vy; TAG=bvychsx3; SPEC=var-bgn_vy-bvychsx3-v1; SOLVE_TAG=Jstar_vy_bvychsx3
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.21, -0.105, -0.03, 0.03, 0.105, 0.21], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychsx3.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychsx3-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvychsx2)
    # bgn_vy (2026-10-29): bvychs with the loadings x 2 (-0.14 .. 0.14), E[b] = 0, chain, level-compensated, price 2.5
    MODEL=bgn_vy; TAG=bvychsx2; SPEC=var-bgn_vy-bvychsx2-v1; SOLVE_TAG=Jstar_vy_bvychsx2
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.14, -0.07, -0.02, 0.02, 0.07, 0.14], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychsx2.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychsx2-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvyhump)
    # bgn_vy (2026-10-29): NON-MONOTONE exposure across an explicit level ladder: type_level [0.6, 0.8, 1.0, 1.2, 1.4], loadings [-0.07, 0.0, 0.07, 0.0, -0.07] (hump), exact compensation, chain, price 2.5
    MODEL=bgn_vy; TAG=bvyhump; SPEC=var-bgn_vy-bvyhump-v1; SOLVE_TAG=Jstar_vy_bvyhump
    export BGN_PARAM_OVERRIDES='{"type_share": [0.2, 0.2, 0.2, 0.2, 0.2], "type_bv": [-0.07, 0.0, 0.07, 0.0, -0.07], "type_level": [0.6, 0.8, 1.0, 1.2, 1.4], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.0, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvyhump.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvyhump-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvyvee)
    # bgn_vy (2026-10-29): NON-MONOTONE exposure across an explicit level ladder: type_level [0.6, 0.8, 1.0, 1.2, 1.4], loadings [0.07, 0.0, -0.07, 0.0, 0.07] (V), exact compensation, chain, price 2.5
    MODEL=bgn_vy; TAG=bvyvee; SPEC=var-bgn_vy-bvyvee-v1; SOLVE_TAG=Jstar_vy_bvyvee
    export BGN_PARAM_OVERRIDES='{"type_share": [0.2, 0.2, 0.2, 0.2, 0.2], "type_bv": [0.07, 0.0, -0.07, 0.0, 0.07], "type_level": [0.6, 0.8, 1.0, 1.2, 1.4], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.0, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvyvee.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvyvee-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvychs25)
    # bgn_vy (2026-10-29): bvychs at bv_comp 2.5 (hump search: the compensation as a channel opposed to y)
    MODEL=bgn_vy; TAG=bvychs25; SPEC=var-bgn_vy-bvychs25-v1; SOLVE_TAG=Jstar_vy_bvychs25
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.07, -0.035, -0.01, 0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 2.5, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychs25.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychs25-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvychs18)
    # bgn_vy (2026-10-29): bvychs at bv_comp 1.8 (hump search: the compensation as a channel opposed to y)
    MODEL=bgn_vy; TAG=bvychs18; SPEC=var-bgn_vy-bvychs18-v1; SOLVE_TAG=Jstar_vy_bvychs18
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.07, -0.035, -0.01, 0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.8, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychs18.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychs18-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvychs6)
    # bgn_vy (2026-10-28): bvychs at bv_comp 0.6 (the level-compensation ladder; bvychs = 1.2 over-compensates)
    MODEL=bgn_vy; TAG=bvychs6; SPEC=var-bgn_vy-bvychs6-v1; SOLVE_TAG=Jstar_vy_bvychs6
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.07, -0.035, -0.01, 0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 0.6, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychs6.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychs6-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  bvychs4)
    # bgn_vy (2026-10-28): bvychs at bv_comp 0.4 (the level-compensation ladder; bvychs = 1.2 over-compensates)
    MODEL=bgn_vy; TAG=bvychs4; SPEC=var-bgn_vy-bvychs4-v1; SOLVE_TAG=Jstar_vy_bvychs4
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1667, 0.1667, 0.1667, 0.1667, 0.1666, 0.1666], "type_bv": [-0.07, -0.035, -0.01, 0.01, 0.035, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 0.4, "y_kind": "chain", "jstar_vy_file": "Jstar_vy_bvychs4.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvychs4-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  gvy25s)
    # gs_vy (2026-10-27): gvy25 with SIGN-SWITCHING types: the ladder mirrored, six level-exposure types (-0.1 .. 0.1), E[b] = 0, level-compensated, price 2.5. Six solves.
    MODEL=gs_vy; TAG=gvy25s; SPEC=var-gs_vy-gvy25s-v1
    SOLVE_TAG=sol_gvy25sa,sol_gvy25sb,sol_gvy25sc,sol_gvy25sd,sol_gvy25se,sol_gvy25sf
    export GS_VY_SOLDIRS=sol_gvy25sa,sol_gvy25sb,sol_gvy25sc,sol_gvy25sd,sol_gvy25se,sol_gvy25sf
    export GS_VY_BV=-0.1,-0.05,-0.015,0.015,0.05,0.1
    export GS_VY_SHARES=0.1667,0.1667,0.1667,0.1667,0.1666,0.1666
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy25s-v1; sbatch --export=ALL --array=0-5 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvy2s)
    # gs_vy (2026-10-27): gvy25s's SIGNED ladder over TWO priced chains, exposure in cash flows only: twelve hidden types (2-vectors), 5 x 5 nodes, bv_comp 1.2. Twelve solves.
    MODEL=gs_vy; TAG=gvy2s; SPEC=var-gs_vy-gvy2s-v1
    SOLVE_TAG=sol_gvy2sa,sol_gvy2sb,sol_gvy2sc,sol_gvy2sd,sol_gvy2se,sol_gvy2sf,sol_gvy2sg,sol_gvy2sh,sol_gvy2si,sol_gvy2sj,sol_gvy2sk,sol_gvy2sl
    export GS_VY_SOLDIRS=sol_gvy2sa,sol_gvy2sb,sol_gvy2sc,sol_gvy2sd,sol_gvy2se,sol_gvy2sf,sol_gvy2sg,sol_gvy2sh,sol_gvy2si,sol_gvy2sj,sol_gvy2sk,sol_gvy2sl
    export GS_VY_BV=0.03535534,-0.03535534,0.04949747,-0.04949747,0.04171208,-0.04171208,0.07071068,-0.07071068,0.09899495,-0.09899495,0.08342417,-0.08342417
    export GS_VY_BV2=0.03535534,-0.03535534,-0.00707107,0.00707107,-0.02756995,0.02756995,0.07071068,-0.07071068,-0.01414214,0.01414214,-0.0551399,0.0551399
    export GS_VY_SHARES=0.083333,0.083333,0.083333,0.083333,0.083333,0.083333,0.083333,0.083333,0.083333,0.083333,0.083333,0.083337
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy2s-v1; sbatch --export=ALL --array=0-11 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvytls)
    # gs_vy (2026-10-28): gvytl with the exposure ladder MIRRORED (-0.10 .. 0.10, E[b] = 0, level-compensated) x three FIXED coupons 0.15 / 0.40 / 0.70: eighteen hidden types. Eighteen solves.
    MODEL=gs_vy; TAG=gvytls; SPEC=var-gs_vy-gvytls-v1
    # sol_gvytlsj..r are symlinks to sol_gvytla..i (identical parameters); their manifests carry gvytl's tags, so the staleness lookup (live_solves) uses those
    SOLVE_TAG=sol_gvytlsa,sol_gvytlsb,sol_gvytlsc,sol_gvytlsd,sol_gvytlse,sol_gvytlsf,sol_gvytlsg,sol_gvytlsh,sol_gvytlsi,sol_gvytla,sol_gvytlb,sol_gvytlc,sol_gvytld,sol_gvytle,sol_gvytlf,sol_gvytlg,sol_gvytlh,sol_gvytli
    export GS_VY_SOLDIRS=sol_gvytlsa,sol_gvytlsb,sol_gvytlsc,sol_gvytlsd,sol_gvytlse,sol_gvytlsf,sol_gvytlsg,sol_gvytlsh,sol_gvytlsi,sol_gvytlsj,sol_gvytlsk,sol_gvytlsl,sol_gvytlsm,sol_gvytlsn,sol_gvytlso,sol_gvytlsp,sol_gvytlsq,sol_gvytlsr
    export GS_VY_BV=-0.1,-0.1,-0.1,-0.05,-0.05,-0.05,-0.015,-0.015,-0.015,0.015,0.015,0.015,0.05,0.05,0.05,0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_SHARES=0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055556,0.055548
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvytls-v1; sbatch --export=ALL --array=0-17 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyib)
    # gs_vy (2026-10-28): investment cost x exp(-1.0 b~ y): the y exposure through the option is leverage x the option's value. One type, one solve.
    MODEL=gs_vy; TAG=gvyib; SPEC=var-gs_vy-gvyib-v1
    SOLVE_TAG=sol_gvyiba
    export GS_VY_SOLDIRS=sol_gvyiba
    export GS_VY_BV=0.0
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BIB=-1.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2 GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyib-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyib5)
    # gs_vy (2026-10-28, retry at half magnitude: -1.0 diverged): investment cost x exp(-0.5 b~ y): the y exposure through the option is leverage x the option's value. One type, one solve.
    MODEL=gs_vy; TAG=gvyib5; SPEC=var-gs_vy-gvyib5-v1
    SOLVE_TAG=sol_gvyib5a
    export GS_VY_SOLDIRS=sol_gvyib5a
    export GS_VY_BV=0.0
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BIB=-0.5
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2 GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyib5-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyib2)
    # gs_vy (2026-10-28, second retry: -1.0 and -0.5 diverged, 0.0/-0.01 converge): investment cost x exp(-0.2 b~ y): the y exposure through the option is leverage x the option's value. One type, one solve.
    MODEL=gs_vy; TAG=gvyib2; SPEC=var-gs_vy-gvyib2-v1
    SOLVE_TAG=sol_gvyib2a
    export GS_VY_SOLDIRS=sol_gvyib2a
    export GS_VY_BV=0.0
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BIB=-0.2
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2 GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyib2-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvy25sh)
    # gs_vy (2026-10-29): gvy25s at gamma_v 0.5, bv_comp 9.0: same level shifts, a smaller y price. Six solves.
    MODEL=gs_vy; TAG=gvy25sh; SPEC=var-gs_vy-gvy25sh-v1
    SOLVE_TAG=sol_gvy25sha,sol_gvy25shb,sol_gvy25shc,sol_gvy25shd,sol_gvy25she,sol_gvy25shf
    export GS_VY_SOLDIRS=sol_gvy25sha,sol_gvy25shb,sol_gvy25shc,sol_gvy25shd,sol_gvy25she,sol_gvy25shf
    export GS_VY_BV=-0.1,-0.05,-0.015,0.015,0.05,0.1
    export GS_VY_SHARES=0.1667,0.1667,0.1667,0.1667,0.1666,0.1666
    unset GS_VY_BV2 GS_VY_BIB GS_VY_BC GS_VY_BI GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy25sh-v1; sbatch --export=ALL --array=0-5 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvy25sx3)
    # gs_vy (2026-10-29): gvy25s with the loadings x 3 (-0.3 .. 0.3), E[b] = 0, level-compensated, price 2.5. Six solves.
    MODEL=gs_vy; TAG=gvy25sx3; SPEC=var-gs_vy-gvy25sx3-v1
    SOLVE_TAG=sol_gvy25sx3a,sol_gvy25sx3b,sol_gvy25sx3c,sol_gvy25sx3d,sol_gvy25sx3e,sol_gvy25sx3f
    export GS_VY_SOLDIRS=sol_gvy25sx3a,sol_gvy25sx3b,sol_gvy25sx3c,sol_gvy25sx3d,sol_gvy25sx3e,sol_gvy25sx3f
    export GS_VY_BV=-0.3,-0.15,-0.045,0.045,0.15,0.3
    export GS_VY_SHARES=0.1667,0.1667,0.1667,0.1667,0.1666,0.1666
    unset GS_VY_BV2 GS_VY_BIB GS_VY_BC GS_VY_BI GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy25sx3-v1; sbatch --export=ALL --array=0-5 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvy25sx2)
    # gs_vy (2026-10-29): gvy25s with the loadings x 2 (-0.2 .. 0.2), E[b] = 0, level-compensated, price 2.5. Six solves.
    MODEL=gs_vy; TAG=gvy25sx2; SPEC=var-gs_vy-gvy25sx2-v1
    SOLVE_TAG=sol_gvy25sx2a,sol_gvy25sx2b,sol_gvy25sx2c,sol_gvy25sx2d,sol_gvy25sx2e,sol_gvy25sx2f
    export GS_VY_SOLDIRS=sol_gvy25sx2a,sol_gvy25sx2b,sol_gvy25sx2c,sol_gvy25sx2d,sol_gvy25sx2e,sol_gvy25sx2f
    export GS_VY_BV=-0.2,-0.1,-0.03,0.03,0.1,0.2
    export GS_VY_SHARES=0.1667,0.1667,0.1667,0.1667,0.1666,0.1666
    unset GS_VY_BV2 GS_VY_BIB GS_VY_BC GS_VY_BI GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy25sx2-v1; sbatch --export=ALL --array=0-5 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvy25sq)
    # gs_vy (2026-10-29): gvy25s at gamma_v 0.25, bv_comp 12.0: same level shifts, a smaller y price. Six solves.
    MODEL=gs_vy; TAG=gvy25sq; SPEC=var-gs_vy-gvy25sq-v1
    SOLVE_TAG=sol_gvy25sqa,sol_gvy25sqb,sol_gvy25sqc,sol_gvy25sqd,sol_gvy25sqe,sol_gvy25sqf
    export GS_VY_SOLDIRS=sol_gvy25sqa,sol_gvy25sqb,sol_gvy25sqc,sol_gvy25sqd,sol_gvy25sqe,sol_gvy25sqf
    export GS_VY_BV=-0.1,-0.05,-0.015,0.015,0.05,0.1
    export GS_VY_SHARES=0.1667,0.1667,0.1667,0.1667,0.1666,0.1666
    unset GS_VY_BV2 GS_VY_BIB GS_VY_BC GS_VY_BI GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvy25sq-v1; sbatch --export=ALL --array=0-5 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyqh10)
    # gs_vy (2026-10-29): LEVERAGE-CURVATURE exposure gs_bv -0.1 x (b~^2 - 1) (hump in leverage), one type, price 2.5, no level shift
    MODEL=gs_vy; TAG=gvyqh10; SPEC=var-gs_vy-gvyqh10-v1
    SOLVE_TAG=sol_gvyqh10
    export GS_VY_SOLDIRS=sol_gvyqh10
    export GS_VY_BV=-0.1
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2 GS_VY_BIB GS_VY_BC GS_VY_BI GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyqh10-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyql20)
    # gs_vy (2026-10-29): LEVERAGE-CURVATURE exposure gs_bv 0.2 x (b~^2 - 1) (U in leverage), one type, price 2.5, no level shift
    MODEL=gs_vy; TAG=gvyql20; SPEC=var-gs_vy-gvyql20-v1
    SOLVE_TAG=sol_gvyql20
    export GS_VY_SOLDIRS=sol_gvyql20
    export GS_VY_BV=0.2
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2 GS_VY_BIB GS_VY_BC GS_VY_BI GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyql20-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyql10)
    # gs_vy (2026-10-29): LEVERAGE-CURVATURE exposure gs_bv 0.1 x (b~^2 - 1) (U in leverage), one type, price 2.5, no level shift
    MODEL=gs_vy; TAG=gvyql10; SPEC=var-gs_vy-gvyql10-v1
    SOLVE_TAG=sol_gvyql10
    export GS_VY_SOLDIRS=sol_gvyql10
    export GS_VY_BV=0.1
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2 GS_VY_BIB GS_VY_BC GS_VY_BI GS_VY_BK
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyql10-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gxq70)
    # gs_vy (2026-10-29): gsbase with a HUMP-shaped x loading in productivity, 1 + (-0.7) (z~^2 - 1)/sqrt2. One solve; no y.
    MODEL=gs_vy; TAG=gxq70; SPEC=var-gs_vy-gxq70-v1
    SOLVE_TAG=sol_gxq70a
    export GS_VY_SOLDIRS=sol_gxq70a
    export GS_VY_BV=0.0
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2 GS_VY_BIB
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gxq70-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gxq35)
    # gs_vy (2026-10-29): gsbase with a HUMP-shaped x loading in productivity, 1 + (-0.35) (z~^2 - 1)/sqrt2. One solve; no y.
    MODEL=gs_vy; TAG=gxq35; SPEC=var-gs_vy-gxq35-v1
    SOLVE_TAG=sol_gxq35a
    export GS_VY_SOLDIRS=sol_gxq35a
    export GS_VY_BV=0.0
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2 GS_VY_BIB
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gxq35-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  kpsgn)
    # kp_vy (2026-10-27): kpgam with the price of x SWITCHING SIGN with y: gamma_x x (2 logistic(-2y) - 1); y unpriced. A conditioning room (DKKM sees y as rf_stand).
    MODEL=kp_vy; TAG=kpsgn; SPEC=var-kp_vy-kpsgn-v1; SOLVE_TAG=kpsgn
    export KP_PARAM_OVERRIDES='{"type_share": [1.0], "type_bv": [0.0], "gamma_v": 0.0, "bv_comp": 0.0, "g_lo": -1.0, "g_hi": 1.0, "g_steep": -2.0}'
    export KP_VY_PREFIX=kpsgn
    SOLVE_HINT='export KSPECS=var-kp_vy-kpsgn-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  bgsgn)
    # bgn_gam (2026-10-27): the regime price of the market shock SWITCHING SIGN, sigma_z x (-1 / +1), 12-month spells, half the time each. A conditioning room (DKKM sees the regime as gam_stand).
    MODEL=bgn_gam; TAG=bgsgn; SPEC=var-bgn_gam-bgsgn-v1; SOLVE_TAG=Jstar_bgsgn
    export BGN_PARAM_OVERRIDES='{"gmult": [-1.0, 1.0], "p01": 0.08333333333333333, "p10": 0.08333333333333333, "jstar_gam_file": "Jstar_bgsgn.csv"}'
    SOLVE_HINT='export BSPECS=var-bgn_gam-bgsgn-v1; sbatch --export=ALL variants/klc/solve_bgn_gam_klc.sh   # Quest'
    ;;
  vym6)
    # kp_vy (2026-10-27): vym3's construction at K = 6: six priced OU states at 2.5/sqrt(6) each, 24 rank-6 types (six alignments x four magnitudes), bv_comp 1.2.
    MODEL=kp_vy; TAG=vym6; SPEC=var-kp_vy-vym6-v1; SOLVE_TAG=vym6
    export KP_PARAM_OVERRIDES='{"type_share":[0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041659],"type_bv":[[0.0081649658,0.0081649658,0.0081649658,0.0081649658,0.0081649658,0.0081649658],[0.0167647983,0.0048773259,0.0048773259,0.0048773259,0.0048773259,0.0048773259],[0.0055521768,0.0186682737,0.0022731525,0.0022731525,0.0022731525,0.0022731525],[0.0042457822,0.0042457822,0.0190403758,-0.000685749,-0.000685749,-0.000685749],[0.0029393877,0.0029393877,0.0029393877,0.018174435,-0.004678136,-0.004678136],[0.0016329932,0.0016329932,0.0016329932,0.0016329932,0.0154893996,-0.0122234133],[0.0244948974,0.0244948974,0.0244948974,0.0244948974,0.0244948974,0.0244948974],[0.0502943948,0.0146319776,0.0146319776,0.0146319776,0.0146319776,0.0146319776],[0.0166565303,0.0560048212,0.0068194575,0.0068194575,0.0068194575,0.0068194575],[0.0127373467,0.0127373467,0.0571211275,-0.0020572469,-0.0020572469,-0.0020572469],[0.0088181631,0.0088181631,0.0088181631,0.054523305,-0.0140344079,-0.0140344079],[0.0048989795,0.0048989795,0.0048989795,0.0048989795,0.0464681989,-0.0366702399],[0.040824829,0.040824829,0.040824829,0.040824829,0.040824829,0.040824829],[0.0838239914,0.0243866294,0.0243866294,0.0243866294,0.0243866294,0.0243866294],[0.0277608838,0.0933413687,0.0113657625,0.0113657625,0.0113657625,0.0113657625],[0.0212289111,0.0212289111,0.0952018791,-0.0034287449,-0.0034287449,-0.0034287449],[0.0146969385,0.0146969385,0.0146969385,0.090872175,-0.0233906798,-0.0233906798],[0.0081649658,0.0081649658,0.0081649658,0.0081649658,0.0774469981,-0.0611170665],[0.0571547607,0.0571547607,0.0571547607,0.0571547607,0.0571547607,0.0571547607],[0.1173535879,0.0341412812,0.0341412812,0.0341412812,0.0341412812,0.0341412812],[0.0388652373,0.1306779161,0.0159120675,0.0159120675,0.0159120675,0.0159120675],[0.0297204755,0.0297204755,0.1332826308,-0.0048002429,-0.0048002429,-0.0048002429],[0.0205757138,0.0205757138,0.0205757138,0.127221045,-0.0327469518,-0.0327469518],[0.0114309521,0.0114309521,0.0114309521,0.0114309521,0.1084257974,-0.0855638931]],"gamma_v":[1.0206207262,1.0206207262,1.0206207262,1.0206207262,1.0206207262,1.0206207262],"bv_comp":1.2}'
    export KP_VY_PREFIX=vym6
    SOLVE_HINT='export KSPECS=var-kp_vy-vym6-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vym3s)
    # kp_vy (2026-10-27): vyg25s's SIGNED type ladder over K = 3 priced states (2.5/sqrt(3) each): 24 hidden types, twelve mirrored alignments x magnitudes 0.07/0.14, bv_comp 1.2.
    MODEL=kp_vy; TAG=vym3s; SPEC=var-kp_vy-vym3s-v1; SOLVE_TAG=vym3s
    export KP_PARAM_OVERRIDES='{"type_share":[0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041659],"type_bv":[[0.0404145188,0.0404145188,0.0404145188],[-0.0404145188,-0.0404145188,-0.0404145188],[0.0649595916,0.018442498,0.018442498],[-0.0649595916,-0.018442498,-0.018442498],[0.0613849942,0.0318623014,-0.0108016772],[-0.0613849942,-0.0318623014,0.0108016772],[0.0361016569,0.053682284,-0.0267372915],[-0.0361016569,-0.053682284,0.0267372915],[-0.0019283831,0.0667066642,-0.0211306008],[0.0019283831,-0.0667066642,0.0211306008],[-0.0372220479,0.0592414494,0.0022293098],[0.0372220479,-0.0592414494,-0.0022293098],[0.0808290377,0.0808290377,0.0808290377],[-0.0808290377,-0.0808290377,-0.0808290377],[0.1299191832,0.0368849959,0.0368849959],[-0.1299191832,-0.0368849959,-0.0368849959],[0.1227699885,0.0637246028,-0.0216033544],[-0.1227699885,-0.0637246028,0.0216033544],[0.0722033138,0.1073645679,-0.053474583],[-0.0722033138,-0.1073645679,0.053474583],[-0.0038567661,0.1334133284,-0.0422612016],[0.0038567661,-0.1334133284,0.0422612016],[-0.0744440958,0.1184828988,0.0044586197],[0.0744440958,-0.1184828988,-0.0044586197]],"gamma_v":[1.443375673,1.443375673,1.443375673],"bv_comp":1.2}'
    export KP_VY_PREFIX=vym3s
    SOLVE_HINT='export KSPECS=var-kp_vy-vym3s-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  vym6s)
    # kp_vy (2026-10-27): vyg25s's SIGNED type ladder over K = 6 priced states (2.5/sqrt(6) each): 24 hidden types, twelve mirrored alignments x magnitudes 0.07/0.14, bv_comp 1.2.
    MODEL=kp_vy; TAG=vym6s; SPEC=var-kp_vy-vym6s-v1; SOLVE_TAG=vym6s
    export KP_PARAM_OVERRIDES='{"type_share":[0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041667,0.041659],"type_bv":[[0.0285773803,0.0285773803,0.0285773803,0.0285773803,0.0285773803,0.0285773803],[-0.0285773803,-0.0285773803,-0.0285773803,-0.0285773803,-0.0285773803,-0.0285773803],[0.058676794,0.0170706406,0.0170706406,0.0170706406,0.0170706406,0.0170706406],[-0.058676794,-0.0170706406,-0.0170706406,-0.0170706406,-0.0170706406,-0.0170706406],[0.0194326186,0.0653389581,0.0079560338,0.0079560338,0.0079560338,0.0079560338],[-0.0194326186,-0.0653389581,-0.0079560338,-0.0079560338,-0.0079560338,-0.0079560338],[0.0148602378,0.0148602378,0.0666413154,-0.0024001214,-0.0024001214,-0.0024001214],[-0.0148602378,-0.0148602378,-0.0666413154,0.0024001214,0.0024001214,0.0024001214],[0.0102878569,0.0102878569,0.0102878569,0.0636105225,-0.0163734759,-0.0163734759],[-0.0102878569,-0.0102878569,-0.0102878569,-0.0636105225,0.0163734759,0.0163734759],[0.0057154761,0.0057154761,0.0057154761,0.0057154761,0.0542128987,-0.0427819465],[-0.0057154761,-0.0057154761,-0.0057154761,-0.0057154761,-0.0542128987,0.0427819465],[0.0571547607,0.0571547607,0.0571547607,0.0571547607,0.0571547607,0.0571547607],[-0.0571547607,-0.0571547607,-0.0571547607,-0.0571547607,-0.0571547607,-0.0571547607],[0.1173535879,0.0341412812,0.0341412812,0.0341412812,0.0341412812,0.0341412812],[-0.1173535879,-0.0341412812,-0.0341412812,-0.0341412812,-0.0341412812,-0.0341412812],[0.0388652373,0.1306779161,0.0159120675,0.0159120675,0.0159120675,0.0159120675],[-0.0388652373,-0.1306779161,-0.0159120675,-0.0159120675,-0.0159120675,-0.0159120675],[0.0297204755,0.0297204755,0.1332826308,-0.0048002429,-0.0048002429,-0.0048002429],[-0.0297204755,-0.0297204755,-0.1332826308,0.0048002429,0.0048002429,0.0048002429],[0.0205757138,0.0205757138,0.0205757138,0.127221045,-0.0327469518,-0.0327469518],[-0.0205757138,-0.0205757138,-0.0205757138,-0.127221045,0.0327469518,0.0327469518],[0.0114309521,0.0114309521,0.0114309521,0.0114309521,0.1084257974,-0.0855638931],[-0.0114309521,-0.0114309521,-0.0114309521,-0.0114309521,-0.1084257974,0.0855638931]],"gamma_v":[1.0206207262,1.0206207262,1.0206207262,1.0206207262,1.0206207262,1.0206207262],"bv_comp":1.2}'
    export KP_VY_PREFIX=vym6s
    SOLVE_HINT='export KSPECS=var-kp_vy-vym6s-v1; sbatch --export=ALL variants/klc/solve_kp_klc.sh   # Quest'
    ;;
  gvyzl)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzl; SPEC=var-gs_vy-gvyzl-v1
    SOLVE_TAG=sol_gvyzla,sol_gvyzlb,sol_gvyzlc
    export GS_VY_SOLDIRS=sol_gvyzla,sol_gvyzlb,sol_gvyzlc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzl-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlx)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlx; SPEC=var-gs_vy-gvyzlx-v1
    SOLVE_TAG=sol_gvyzlxa,sol_gvyzlxb,sol_gvyzlxc
    export GS_VY_SOLDIRS=sol_gvyzlxa,sol_gvyzlxb,sol_gvyzlxc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlx-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlq)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlq; SPEC=var-gs_vy-gvyzlq-v1
    SOLVE_TAG=sol_gvyzlqa,sol_gvyzlqb,sol_gvyzlqc
    export GS_VY_SOLDIRS=sol_gvyzlqa,sol_gvyzlqb,sol_gvyzlqc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlq-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlk)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlk; SPEC=var-gs_vy-gvyzlk-v1
    SOLVE_TAG=sol_gvyzlka,sol_gvyzlkb,sol_gvyzlkc
    export GS_VY_SOLDIRS=sol_gvyzlka,sol_gvyzlkb,sol_gvyzlkc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlk-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlf)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlf; SPEC=var-gs_vy-gvyzlf-v1
    SOLVE_TAG=sol_gvyzlfa,sol_gvyzlfb,sol_gvyzlfc
    export GS_VY_SOLDIRS=sol_gvyzlfa,sol_gvyzlfb,sol_gvyzlfc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlf-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlw)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlw; SPEC=var-gs_vy-gvyzlw-v1
    SOLVE_TAG=sol_gvyzlwa,sol_gvyzlwb,sol_gvyzlwc
    export GS_VY_SOLDIRS=sol_gvyzlwa,sol_gvyzlwb,sol_gvyzlwc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlw-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzl5)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzl5; SPEC=var-gs_vy-gvyzl5-v1
    SOLVE_TAG=sol_gvyzl5a,sol_gvyzl5b,sol_gvyzl5c,sol_gvyzl5d,sol_gvyzl5e
    export GS_VY_SOLDIRS=sol_gvyzl5a,sol_gvyzl5b,sol_gvyzl5c,sol_gvyzl5d,sol_gvyzl5e
    export GS_VY_BV=0.1,0.1,0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0,0.0,0.0
    export GS_VY_SHARES=0.2,0.2,0.2,0.2,0.2
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzl5-v1; sbatch --export=ALL --array=0-4 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlb)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlb; SPEC=var-gs_vy-gvyzlb-v1
    SOLVE_TAG=sol_gvyzlba,sol_gvyzlbb,sol_gvyzlbc
    export GS_VY_SOLDIRS=sol_gvyzlba,sol_gvyzlbb,sol_gvyzlbc
    export GS_VY_BV=0.2,0.2,0.2
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlb-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlr)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlr; SPEC=var-gs_vy-gvyzlr-v1
    SOLVE_TAG=sol_gvyzlra,sol_gvyzlrb,sol_gvyzlrc
    export GS_VY_SOLDIRS=sol_gvyzlra,sol_gvyzlrb,sol_gvyzlrc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlr-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlp)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlp; SPEC=var-gs_vy-gvyzlp-v1
    SOLVE_TAG=sol_gvyzlpa,sol_gvyzlpb,sol_gvyzlpc
    export GS_VY_SOLDIRS=sol_gvyzlpa,sol_gvyzlpb,sol_gvyzlpc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlp-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlrp)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlrp; SPEC=var-gs_vy-gvyzlrp-v1
    SOLVE_TAG=sol_gvyzlrpa,sol_gvyzlrpb,sol_gvyzlrpc
    export GS_VY_SOLDIRS=sol_gvyzlrpa,sol_gvyzlrpb,sol_gvyzlrpc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlrp-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gsbg1)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gsbg1; SPEC=var-gs_vy-gsbg1-v1
    SOLVE_TAG=sol_gsbg1a
    export GS_VY_SOLDIRS=sol_gsbg1a
    export GS_VY_BV=0.0
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gsbg1-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyz10g)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyz10g; SPEC=var-gs_vy-gvyz10g-v1
    SOLVE_TAG=sol_gvyz10ga
    export GS_VY_SOLDIRS=sol_gvyz10ga
    export GS_VY_BV=0.1
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyz10g-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyzlg)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyzlg; SPEC=var-gs_vy-gvyzlg-v1
    SOLVE_TAG=sol_gvyzlga,sol_gvyzlgb,sol_gvyzlgc
    export GS_VY_SOLDIRS=sol_gvyzlga,sol_gvyzlgb,sol_gvyzlgc
    export GS_VY_BV=0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0
    export GS_VY_SHARES=0.34,0.33,0.33
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyzlg-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvyp2n)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp2n; SPEC=var-gs_vy-gvyp2n-v1
    SOLVE_TAG=sol_gvyp2na,sol_gvyp2nb
    export GS_VY_SOLDIRS=sol_gvyp2na,sol_gvyp2nb
    export GS_VY_BV=0.1,0.1
    export GS_VY_BC=0.0,0.0
    export GS_VY_BI=0.0,0.0
    export GS_VY_BK=0.0,0.0
    export GS_VY_SHARES=0.5,0.5
    export GS_VY_BV2=0.0,0.1
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp2n-v1; sbatch --export=ALL --array=0-1 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  bvy9t2f)
    # bgn_vy follow-up (2026-10-10): BGN group 4, NESTED in bvych0: bvych0's ladder on chain 1 at price 2.5 and an independent draw from the same ladder on a second 21-node chain at price 2.5, nine types
    MODEL=bgn_vy; TAG=bvy9t2f; SPEC=var-bgn_vy-bvy9t2f-v1; SOLVE_TAG=Jstar_vy_bvy9t2f
    export BGN_PARAM_OVERRIDES='{"type_share": [0.1112, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111], "type_bv": [0.01, 0.01, 0.01, 0.035, 0.035, 0.035, 0.07, 0.07, 0.07], "gamma_v": 2.5, "kappa_y": 0.35, "bv_comp": 1.2, "y_kind": "chain", "NY2": 21, "type_bv2": [0.01, 0.035, 0.07, 0.01, 0.035, 0.07, 0.01, 0.035, 0.07], "gamma_v2": 2.5, "kappa_y2": 0.35, "jstar_vy_file": "Jstar_vy_bvy9t2f.npz"}'
    SOLVE_HINT='export BSPECS=var-bgn_vy-bvy9t2f-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh   # Quest'
    ;;
  gvyp10c)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvyp10c; SPEC=var-gs_vy-gvyp10c-v1
    SOLVE_TAG=sol_gvyp10ca
    export GS_VY_SOLDIRS=sol_gvyp10ca
    export GS_VY_BV=0.1
    export GS_VY_BC=0.0
    export GS_VY_BI=0.0
    export GS_VY_BK=0.0
    export GS_VY_SHARES=1.0
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvyp10c-v1; sbatch --export=ALL --array=0-0 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvytl)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvytl; SPEC=var-gs_vy-gvytl-v1
    SOLVE_TAG=sol_gvytla,sol_gvytlb,sol_gvytlc,sol_gvytld,sol_gvytle,sol_gvytlf,sol_gvytlg,sol_gvytlh,sol_gvytli
    export GS_VY_SOLDIRS=sol_gvytla,sol_gvytlb,sol_gvytlc,sol_gvytld,sol_gvytle,sol_gvytlf,sol_gvytlg,sol_gvytlh,sol_gvytli
    export GS_VY_BV=0.015,0.015,0.015,0.05,0.05,0.05,0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_SHARES=0.1111,0.1111,0.1111,0.1111,0.1111,0.1111,0.1111,0.1111,0.1112
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvytl-v1; sbatch --export=ALL --array=0-8 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  gvytlg)
    # GS21, the priced y factor on a fixed OPERATING COST (2026-10-08): exposure runs through operating leverage,
    # convex in profitability within a type; at y = 0 every type pays the same cost. variants/gs_vy/.
    MODEL=gs_vy; TAG=gvytlg; SPEC=var-gs_vy-gvytlg-v1
    SOLVE_TAG=sol_gvytlga,sol_gvytlgb,sol_gvytlgc,sol_gvytlgd,sol_gvytlge,sol_gvytlgf,sol_gvytlgg,sol_gvytlgh,sol_gvytlgi
    export GS_VY_SOLDIRS=sol_gvytlga,sol_gvytlgb,sol_gvytlgc,sol_gvytlgd,sol_gvytlge,sol_gvytlgf,sol_gvytlgg,sol_gvytlgh,sol_gvytlgi
    export GS_VY_BV=0.015,0.015,0.015,0.05,0.05,0.05,0.1,0.1,0.1
    export GS_VY_BC=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_BI=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_BK=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    export GS_VY_SHARES=0.1111,0.1111,0.1111,0.1111,0.1111,0.1111,0.1111,0.1111,0.1112
    unset GS_VY_BV2
    unset GS_SIM_OVERRIDES
    SOLVE_HINT='export GSPEC=var-gs_vy-gvytlg-v1; sbatch --export=ALL --array=0-8 variants/klc/solve_gs_vy_klc.sh   # Quest'
    ;;
  *)
    echo "unknown SEED_SPEC '$SEED_SPEC' (expected vyx, g0235, g0235f, g0235s, g0235r, g28, bx7, gx7, vyg25, vym3, vym3t3, g0235d, bgnbase, kpbase, gsbase, g0235df, vyg30k70, gx7w, vyg25k15, vyg25k70, bvy25, gvy25 or gvyc25)" >&2; exit 2 ;;
esac

# Every case above restates its economy's parameters and kappa grid, and each is checked
# against the spec its SPEC= names by tests/test_specs_match_shell.py -- a second copy is
# what drifted between run_gs_bx7.sh and its spec. It is the parameters, not the seed,
# that define the economy, so every task in an array exports the same ones.

# ---- THE MEASUREMENT PROTOCOL ---------------------------------------------------------
# These four are the same for EVERY economy, so they are set once here rather than per
# case. variants/common/protocol.py defines them and tests/test_protocol_is_uniform.py
# pins these literals to it and to every spec's panel/estimation block, so the three
# copies cannot drift apart. They drifted before: each case carried its own KAPPAS= line,
# three different ridge grids reached the estimators, and DKKM's Sharpe -- a max over that
# grid, censored at its floor in 29 of 30 KP14 Path 1 seeds -- was not comparable across
# rows. docs/RESULTS.md, "The measurement protocol", says what each value is for.
#
# The grid is wide on BOTH sides so the argmax CAN be interior. That it IS interior is a
# check on each economy's results, not a property of the grid (docs/RESULTS.md).
#
# 2026-09-21: widened to eleven values, two decades each side, after the interior gate
# failed in five of thirteen economies in BOTH directions (vyx/vyg25 at the floor,
# gx7/bx7/g0235s at the ceiling). This literal must equal protocol.kappas_csv() --
# tests/test_protocol_is_uniform.py compares them -- so regenerate it rather than typing
# it:  python -c 'import sys; sys.path.insert(0,"variants/common"); import protocol; print(protocol.kappas_csv())'
N=${SEED_N:-500}
T=${SEED_T:-500}
WINDOW=${SEED_WINDOW:-360}
KAPPAS=1e-07,1e-06,1e-05,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0

# An off-protocol run is a smoke test and may not be reported, so it may not write where
# the reported results live. SEED_N/SEED_T/SEED_WINDOW stay available for one -- they are
# how the scaling probes were measured -- but such a run must name its own
# BOP_RESULTS_DIR. Without this guard, probes at N=60/T=80/window=20 landed in
# variants/results and were aggregated into economy_table.csv beside the real economies
# (the smoke, bigN, scal* and bscal* rows, removed 2026-09-15).
if [ "$N" != 500 ] || [ "$T" != 500 ] || [ "$WINDOW" != 360 ]; then
  if [ -z "${BOP_RESULTS_DIR:-}" ]; then
    echo "REFUSED: N=$N T=$T window=$WINDOW departs from the protocol (500/500/360)." >&2
    echo "  An off-protocol run may not write into the reported results. Point it" >&2
    echo "  elsewhere, e.g. BOP_RESULTS_DIR=/data/\$USER/probe." >&2
    exit 2
  fi
  echo "WARNING: off-protocol run (N=$N T=$T window=$WINDOW) -> $BOP_RESULTS_DIR;" \
       "not reportable (variants/common/protocol.py)" >&2
fi

# Conditioning columns for the feature bases. NO case narrows them: every economy runs the
# model's full set, which is what leaving RF_COLS unset means. The plumbing stays because
# run_oracle.py --rf_cols is still the way to ask the question, and because the list is
# recorded in every summary and run record -- run_estimators.py refuses a panel whose oracle
# used a different one. A case that sets it would be off protocol and is refused by
# tests/test_protocol_is_uniform.py.
RF_ARGS=()
[ -n "${RF_COLS:-}" ] && RF_ARGS=(--rf_cols "$RF_COLS")

# both  = oracle then estimators (the default; what every campaign run does)
# oracle = the oracle ONLY, and without --save_panel: nothing downstream consumes the
#          panel or the moments file in this mode, and skipping them saves ~35 MB/seed.
# linear = E1: no oracle; re-score the SAVED panel's linear methods and the fair ones
#          (run_estimators.py --linear_only --fair_linear) from $INPUTS into $RESULTS.
SEED_STAGE=${SEED_STAGE:-both}
# SEED_STAGE=estimators (2026-10-26) re-runs EVERY estimator on a saved panel (the linear stage without
# --linear_only): what a change to the estimators' inputs needs -- the GS21 leverage characteristic -- without
# re-simulating. Same inputs/results discipline as the linear stage.
case "$SEED_STAGE" in both|oracle|linear|estimators) ;; *)
  echo "unknown SEED_STAGE '$SEED_STAGE' (expected both, oracle, linear or estimators)" >&2; exit 2 ;;
esac

# run_oracle.py and run_estimators.py both honour BOP_RESULTS_DIR. This script used to
# hardcode `results/` for the log and the run record, so setting that variable moved the
# OUTPUT while the checkpoint kept looking in the old place -- it would have re-run a
# finished seed, or skipped an unfinished one, without saying so. Mirror it here.
RESULTS=${BOP_RESULTS_DIR:-results}
INPUTS=${BOP_INPUTS_DIR:-results}   # read by SEED_STAGE=linear only

CONDA_ENV=${CONDA_ENV:-bop}
# BOP_ENV_PREFIX: a site without ASU's `mamba/latest` module points this at an existing conda
# env's prefix instead (Quest: variants/klc/). Everything below is then identical.
if [ -n "${BOP_ENV_PREFIX:-}" ]; then
  export CONDA_PREFIX="$BOP_ENV_PREFIX"
else
  module load mamba/latest
  source activate "$CONDA_ENV"
fi
# `source activate` sets CONDA_PREFIX but does NOT prepend the env's bin/ to PATH in the
# non-interactive shell sbatch provides. Without this, `python` resolves to the mamba
# BASE interpreter and every task dies in ~1 s with ModuleNotFoundError: numpy (the
# traceback names /etc/python/sitecustomize.py). Hit on Phoenix 2026-08-31 on all 11
# array tasks. Same fix as legacy/run_bop_job.sh.
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
if [ "$SEED_STAGE" = linear ] || [ "$SEED_STAGE" = estimators ]; then
  mkdir -p "$RESULTS"
  if [ "$(cd "$INPUTS" && pwd -P)" = "$(cd "$RESULTS" && pwd -P)" ]; then
    echo "ABORT: SEED_STAGE=$SEED_STAGE writes *_estimators_* files under the SAME names as the run it re-scores." >&2
    echo "Set BOP_RESULTS_DIR to a directory other than its inputs ($INPUTS)." >&2
    exit 2
  fi
fi
mkdir -p "$RESULTS/logs"
LOG="$RESULTS/logs/log_${TAG}_s${SS}.txt"

echo "=== $MODEL/$TAG seed $SEED on $(hostname) $(date '+%F %T') threads=$NT N=$N T=$T stage=$SEED_STAGE results=$RESULTS ===" | tee "$LOG"

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
RUNJSON="$RESULTS/${MODEL}_estimators_${TAG}_s${SS}_w${WINDOW}_run.json"
ORACLEJSON="$RESULTS/${MODEL}_oracle_${TAG}_s${SS}.json"
if [ "$SEED_STAGE" = oracle ]; then
  # There is no run record in this mode, so the checkpoint asks the oracle's own output
  # whether it already covers this window. Makes the array restartable after a walltime
  # kill without re-deriving ceilings that are already there.
  if python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('eval_window')==int(sys.argv[2]) else 1)" \
       "$ORACLEJSON" "$WINDOW" 2>/dev/null; then
    echo "seed $SEED oracle already carries eval_window $WINDOW -- nothing to do" | tee -a "$LOG"
    exit 0
  fi
elif python common/runstamp.py is-current "$RUNJSON" --model "$MODEL" --tag "${SOLVE_TAG:-$TAG}" >>"$LOG" 2>&1; then
  echo "seed $SEED already complete and current for the recorded solve -- nothing to do" | tee -a "$LOG"
  exit 0
fi

S=$(date +%s)
if [ "$SEED_STAGE" = linear ] || [ "$SEED_STAGE" = estimators ]; then
  for f in "$INPUTS/${MODEL}_panel_${TAG}_s${SS}.parquet" "$INPUTS/${MODEL}_moments_${TAG}_s${SS}.npz" \
           "$INPUTS/${MODEL}_oracle_${TAG}_s${SS}.json"; do
    [ -f "$f" ] || { echo "ABORT: SEED_STAGE=$SEED_STAGE needs $f" | tee -a "$LOG"; exit 2; }
  done
  LIN_ONLY=--linear_only
  [ "$SEED_STAGE" = estimators ] && LIN_ONLY=
  python -W ignore run_estimators.py --model "$MODEL" --tag "$TAG" --seed "$SEED" \
         --window "$WINDOW" --levels --include_mkt --kappas "$KAPPAS" \
         $LIN_ONLY --fair_linear --inputs_from "$INPUTS" --n_jobs "$NT" ${RF_ARGS[@]+"${RF_ARGS[@]}"} 2>&1 | tee -a "$LOG"
  E=$(date +%s)
  echo "=== seed $SEED END $(date '+%F %T') stage=$SEED_STAGE estimators=$((E-S))s inputs=$INPUTS ===" | tee -a "$LOG"
  python common/runstamp.py is-current "$RUNJSON" --model "$MODEL" --tag "${SOLVE_TAG:-$TAG}" | tee -a "$LOG"
  exit 0
fi
SAVE_PANEL=--save_panel
[ "$SEED_STAGE" = oracle ] && SAVE_PANEL=
# --spec makes the run VERIFY, before it records anything, that the tables it read are
# the ones the spec declares. Without it a summary could carry a spec_id for an economy
# it did not build -- which is the failure the whole registry exists to prevent.
python -W ignore run_oracle.py --model "$MODEL" --N "$N" --T "$T" --seed "$SEED" \
       --tag "$TAG" --spec "$SPEC" --eval_window "$WINDOW" --levels $SAVE_PANEL ${RF_ARGS[@]+"${RF_ARGS[@]}"} 2>&1 | tee -a "$LOG"
MID=$(date +%s)
echo "=== oracle done in $((MID-S))s ===" | tee -a "$LOG"

if [ "$SEED_STAGE" = oracle ]; then
  echo "=== seed $SEED END $(date '+%F %T') oracle=$((MID-S))s (stage=oracle, estimators not run) ===" | tee -a "$LOG"
  python -c "import json,sys; d=json.load(open(sys.argv[1])); print('ORACLE OK: eval_window %s over %s of %s months' % (d.get('eval_window'), d.get('eval_months'), d.get('months')))" "$ORACLEJSON" | tee -a "$LOG"
  exit 0
fi

python -W ignore run_estimators.py --model "$MODEL" --tag "$TAG" --seed "$SEED" \
       --window "$WINDOW" --levels --include_mkt --kappas "$KAPPAS" --fair_linear \
       --n_jobs "$NT" ${RF_ARGS[@]+"${RF_ARGS[@]}"} 2>&1 | tee -a "$LOG"
E=$(date +%s)

echo "=== seed $SEED END $(date '+%F %T') oracle=$((MID-S))s estimators=$((E-MID))s total=$((E-S))s ===" | tee -a "$LOG"

# What this replication was actually built from. `python variants/solfiles.py show <id>`
# expands any of these ids into the full parameter set that produced it.
python common/runstamp.py is-current "$RUNJSON" --model "$MODEL" --tag "${SOLVE_TAG:-$TAG}" | tee -a "$LOG"
