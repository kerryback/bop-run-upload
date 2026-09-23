# Cluster operations and the run record

**What this file is for, and the two moments you open it.** It holds the operating rules for running
this repository on ASU's Sol and Phoenix -- where output goes, what may never be pulled while a job
is live, and the hazards that have cost real money -- plus the measured cost of every campaign, which
is what you size the next array from. Read it **before submitting anything**, and again **when a
result's provenance is in question**. It is not results (`docs/RESULTS.md`) and not a queue of work
(`docs/NEXTUP.md`); it is how the compute was run and what it cost.

## Where output goes, and why it is arranged this way

- **Result files are named by model, tag and seed only** (`variants/common/runstamp.py`, `stem`).
  Nothing in the name records T, the window or the cluster. This used to mean that a run at another
  sample needed its own tag; since 2026-09-15 there is no other sample -- the measurement protocol
  fixes N, T and the window for every economy (`variants/common/protocol.py`), and
  `run_seeds_slurm.sh` refuses to write an off-protocol run into `variants/results` at all. A
  re-scoring of saved panels still needs its own results directory; `SEED_STAGE=linear` refuses to
  write into the directory it reads.
- **`/data/sjpruitt` is shared by both clusters; `/scratch` is per-cluster and purged.** The
  pre-refactor `run_bop_job.sh` writes every run of a model into the same two scratch directory
  names, so each run overwrites the last.
- **SLURM logs carry the cluster**: `outslurm/sol.*` and `outslurm/phx.*`. The two clusters' job ids
  are independent sequences.
- **No `git pull` of the shared checkout while a job that imports from it is running** on either
  cluster. Every task reads its Python from that one tree.
- **Pull the shared checkout only with `bash variants/cluster_pull.sh`**, run inside it on the cluster.
  Jobs write their outputs into the checkout; the laptop copies, commits and pushes them; the cluster
  then holds untracked copies of files the pull brings in as tracked, and a plain `git pull` refuses to
  overwrite them, identical or not. The script clears those copies only if every one is byte-identical
  to what origin carries, then fast-forwards; any difference, or a checkout that cannot fast-forward,
  aborts with nothing touched (`tests/test_cluster_pull.py`).

## Three hazards worth carrying forward

Each of these has fired at least once and cost something. They are the reason for a test or a rule.

- **A chained solve id hashes its upstream tables' RAW FILE BYTES** (`solstamp.artifact_digests`);
  only parameters are quantised to 8 significant digits. The Mac and Sol write a solve's tables with
  different trailing digits, so an id computed on one platform is not reproducible on another. On
  2026-09-13 a KP14 integ solve on Sol printed `a2cc8d99c1bc474b` against the precommitted
  `8d1308e8f21723f8` and was cancelled at 11 minutes. The fix keeps the precommitment exact: build
  the whole chain on one platform, or ship the upstream tables to the other byte-identical with their
  manifest and build only the downstream stage there.
- **`solve_id` digests a producer's whole source file**, so an edit that cannot change a solve still
  re-keys it. Run `python variants/solve_impact.py --worktree` BEFORE paying for anything: it names
  the solves a change invalidates and whether the change is functional (AST differs) or only
  comments. A comment-only `sed` inside a `print()` moved a precommitted GS id on 2026-09-08, and the
  2026-09-15 burn-in edit re-keyed all six BGN J\* solves without changing a byte of their tables.
  The digest-bearing files are exactly: `kp14_fd_vy.py`, `parameters_kp14.py`, `integ_kp14.py`
  (`build_vy_tables.py:100-101`); `vasicek.py`, `parameters.py` (`rebuild_jstar_gam.py:29`);
  `gs_solve_reg.py`, `gs_solve_gam.py` (`gs_solve_reg.py:75-81`).
- **The per-seed checkpoint keys on the SOLVE, so a measurement-only change is invisible to it.** On
  2026-09-15 seven arrays exited in three seconds each, 10 of 10 tasks, with `seed N already complete
  and current for the recorded solve -- nothing to do`: the protocol change had moved the ridge grid,
  the burn-in and the conditioning columns while leaving KP14's and GS21's solve ids untouched, so
  **seventy seeds were silently skipped** and matched pre-protocol results still sitting in the shared
  results directory. Two fixes: the stale `*_run.json` records were deleted and the seven economies
  resubmitted, and `runstamp.run_matches_protocol` now checks the estimation half -- grid, window,
  fair benchmark, winsorisation -- against `variants/common/protocol.py`, with `run_is_current`
  folding it in (`tests/test_runstamp_multitag.py`). **Before submitting, confirm every id you expect
  to move has moved**; the 2026-09-21 campaign did, and all 130 tasks read STALE and re-ran.

## Choosing a cluster

Both clusters share `/data` and have independent job-id sequences, queues and fairshare.

- **Memory decides first and is not negotiable.** `run_oracle.py` allocates over `T + burnin`. Only
  `g0235r` and `g0235s` cannot sit on a 112-125 GB Phoenix `public` node; everything else can. The
  historical cost of getting this wrong is all ten `g0235r` seeds and three `g0235s` seeds killed at
  a 64 GiB cap.
- **On Sol, use `highmem`, not `public`.** Sol's own scheduler once put an identical task three days
  out on public and six hours out on highmem. The two BGN economies need the memory anyway.
- **Fairshare and backlog decide the rest, and they move.** Re-check with `sshare -U` and
  `squeue -h -p public -t PD | wc -l` on each, then `bash variants/submit_campaign.sh <sol|phx> --dry`,
  which is each scheduler's own answer on start time and fit. In 2026-09-15 Phoenix's fairshare was
  12.7x Sol's; by 2026-09-21 it was 5x. Both times eleven of thirteen economies went to Phoenix.
- **One seed first, unless the economy's memory is already measured.** The standing rule
  (`docs/RESULTS.md`, "Adding an experiment") is one seed then `sacct` before sizing an array. Both
  protocol campaigns skipped it deliberately, for re-runs of economies already measured; it stands for
  anything new.

## What it costs: the measured record

### Achieved 2026-09-22: all 130 COMPLETED, none failed, none OOM, none walltime-killed

Read off `sacct` on both clusters. **Every request was generous and the walltime estimate was the
one that was wrong** -- wrong HIGH, which cost nothing on an empty queue and is the trade this
campaign deliberately made.

| economy | cluster | request | achieved MaxRSS | achieved Elapsed, min to max | CPU-h |
|---|---|---|---|---|---|
| `g0235r` | Sol highmem | 8 cpu, 128G, 6 d | 76.3 GiB | 12.98 to 20.78 h | 510 |
| `g0235s` | Sol highmem | 8 cpu, 128G, 6 d | 70.5 GiB | 5.63 to 19.80 h | 460 |
| `g0235` | Phoenix | 8 cpu, 64G, 3 d | 36.0 GiB | 5.81 to 7.84 h | 391 |
| `vyx` | Phoenix | 8 cpu, 48G, 3 d | 29.2 GiB | 5.81 to 5.94 h | 380 |
| `vyg25` | Phoenix | 8 cpu, 48G, 3 d | 29.1 GiB | 5.77 to 6.19 h | 382 |
| `kpbase` | Phoenix | 8 cpu, 48G, 2 d | 29.1 GiB | 5.93 to 6.14 h | 382 |
| `bgnbase` | Phoenix | 8 cpu, 40G, 2 d | 19.9 GiB | 5.58 to 8.32 h | 427 |
| `g0235f` | Phoenix | 8 cpu, 40G, 2 d | 19.9 GiB | 5.55 to 6.67 h | 383 |
| `g0235d` | Phoenix | 8 cpu, 40G, 2 d | 19.9 GiB | 5.22 to 5.49 h | 372 |
| `bx7` | Phoenix | 8 cpu, 32G, 2 d | 4.3 GiB | 5.72 to 11.92 h | 486 |
| `gx7` | Phoenix | 8 cpu, 32G, 2 d | 4.1 GiB | 5.77 to 7.74 h | 444 |
| `gsbase` | Phoenix | 8 cpu, 32G, 2 d | 3.4 GiB | 6.00 to 8.82 h | 442 |
| `g28` | Phoenix | 8 cpu, 32G, 2 d | 3.3 GiB | 5.97 to 9.60 h | 467 |

**984 node-hours across the 130 tasks, 5,524 core-hours**, against a budget of about 900 node-hours
and the 2026-09-15 campaign's 650.

Three things this measurement settles, and they are the reason it is recorded:

1. **The memory prediction was right and it was right for the stated reason.** The sizing section
   above predicted memory would move DOWNWARD because BGN's corrected J\* falls 14-21% and BGN's peak
   tracks the J\* value scale. It did: `g0235` came in at 36.0 GiB against a ~44 GiB projection and
   `g0235r` at 76.3 against ~87. Every economy finished under 60% of its request.
2. **The +57% walltime estimate was wrong in the safe direction.** It assumed eleven penalties would
   cost 57% more than eight on a stage measured at ~2.8 h per seed. The slowest seed in the whole
   campaign ran 20.8 h against a six-day request, and every Phoenix row finished inside 12 h against
   a two- or three-day request. The next campaign can size from the table above: 1 day for the
   `gs_bx` and the light `bgn_gam` rows, 1 day for `kp_vy`, 2 days for `g0235s`/`g0235r`.
3. **`g0235s` and `g0235r` are the only rows whose seeds disagree by more than 2x in wall time**
   -- 5.6 to 19.8 h and 13.0 to 20.8 h -- which is the same per-seed dispersion finding 4 reports in
   their Sharpes, showing up in the scheduler.

### What the re-run had to get right, and did

**The re-run was ATOMIC.** `runstamp.stem` puts no spec version in a filename, so re-run files
overwrite the old ones in place; a partial re-run would have left `aggregate_seeds.py` emitting
`spec_id = "MIXED:v3|v4"` with pre- and post-fix seeds **averaged into one row** at `n_seeds = 10`,
and no test catches that. Nothing was aggregated until all 130 were in, and `economy_table.csv`
carries thirteen single-version rows at ten seeds each.

**Every task re-ran.** All 130 job logs record STALE on their solves and all 130 record `[spec]`,
`[env]` and `[readback]` verification. The 2026-09-15 campaign silently skipped seventy tasks that
exited in three seconds with "already complete and current", because a protocol change had left the
solve ids untouched; here all fifteen ids moved, which is what makes the skip impossible.

## The campaigns

| campaign | what it was | outcome |
|---|---|---|
| **bgnzr**, 2026-09-23, the gate on BGN's discount channel | One economy, ten seeds. `bgnbase` with `beta_zr` -0.00014 -> -0.00020, a rotation of the price of risk onto the rate channel at constant total, rate share 15.1% -> 20.5%. Stops there because the corrected covariance makes the limiting term spread twice as sensitive to `beta_zr` (finding 13): 3.79%/yr here against BGN's own 2.4%. J\* id `550412fe21ce97a3`, precommitted and reproduced exactly. Phoenix `21609884`, public, 40G, 1 day, sized from `bgnbase`'s achieved 19.9 GiB and 8.3 h | **SUBMITTED**, all ten RUNNING at 32 s. Reads `d(room)/d(rate price)` over a 43% price step; the registered gate is room rising by at least +0.005 over `bgnbase`'s +0.0274, below which the multi-factor term structure does not earn a new solver |
| **campaign 2026-09-21**, the corrected pricing and the amended ridge grid (ran to 2026-09-22) | All thirteen economies. Two changes: the pricing fix (merge `91095fb` -- `kp_vy`'s constant-rate treatment of the priced OU state, BGN's halved bond covariance; nine economies re-solved) and the ridge grid `1e-5 ... 10` -> `1e-7 ... 1000`, which is estimator-side and so moved all thirteen. Sol `63759851`-`2` (the two highmem BGN rows), Phoenix `21604189`-`21604199` | **COMPLETE, 130 of 130.** Costs above. The result is `docs/RESULTS.md`: `vyx`'s headline withdrawn, `vyg25` and `g0235f` surviving. The gate's criterion was found to be wrong and was amended -- see below |
| **campaign 2026-09-15**, the measurement protocol (ran to 2026-09-17) | All thirteen at one protocol, so that the only thing separating two rows of `docs/RESULTS.md` is the economy: burn-in 400 everywhere (was 300/400/300), one ridge grid (was three), the full conditioning set for the baselines too, ten seeds for `g0235d`. No economy's parameters changed; only BGN's six J\* tables re-solved. About 650 node-hours | **COMPLETE**, after the restart the third hazard above describes. Its own gate then reported five of thirteen rows censored at a grid edge, which is what made the 2026-09-21 campaign necessary |

### The gate's criterion was wrong, and this is the durable part

`python variants/penalty_gate.py` runs first on any campaign, before any number is read, then
`python variants/aggregate_seeds.py`. On 2026-09-22 the gate failed three economies for having the
winning ridge penalty at the grid ceiling (`g0235s` 6/10, `bx7` 7/10, `gx7` 5/10). Read literally
that demanded a third decade and another ~990 node-hours. **It would have bought nothing.** As kappa
grows the ridge direction `(X'X + kI)^-1 X'y -> X'y / k` stops depending on kappa and a Sharpe ratio
is scale-invariant, so the curve has a horizontal asymptote; across all 130 seeds the 16 at the
ceiling gained at most 2.7e-05 over their best interior penalty. The gate now tests materiality as
well as position and `tests/test_penalty_gate.py` pins both halves.

**The penalty-gate table in `docs/RESULTS.md` is pinned by NO test** and must be regenerated and
hand-transcribed after every campaign.

---

## Earlier campaigns

Their results are superseded by the protocol campaign above and are not quoted anywhere. Kept as the
job record; the hazards they taught are in "Two hazards worth carrying forward".

- **2026-09-14, the baselines (A1), commit `a5b5673`: COMPLETED 2026-09-15.** Each paper's economy as
  published, at ten seeds. Sol `63257244` (the GS solve, 5 h 34 min, tolerance exit at sweep 3024,
  peak 2.0 GiB) then `63257245` (gsbase seeds, 3.6 to 5.3 h, peak 4.5 GiB), `63257246` (bgnbase, 2.2
  to 2.9 h, peak 15.8 GiB) and `63257247` (kpbase, 2.3 to 3.5 h, peak 29.2 GiB). All 30 seeds CURRENT;
  211 files copied and checksum-matched. The GS solution was published content-addressed under
  `c6ae2d52428a7ce5` and is reused by the campaign above.
- **2026-09-13, commit `fef5802`: COMPLETED 2026-09-14.** K4 (`vyg25`, Sol `63188972` solve +
  `63188973` seeds, peak 29.2 GiB, longest 7 h 33 min), X3 (`vyxT860`, Sol `63188596`, peak 56.9 GiB
  -- the economy retired 2026-09-15), B4 (`g0235d` solve Phoenix `21571505` + seed 0 `21571506`, 2 h
  31 min, peak 15.6 GiB, screen missed both gates), and E1's eighty one-minute re-scorings on Phoenix
  `htc` (`21571507`-`21571532`, 52 to 76 s each, peak 1.5 to 1.7 GiB). K4's first solve attempt, Sol
  `63188594`, was cancelled at 11 minutes on the chained-id hazard. Pre-campaign archives: Sol
  `63188205`/`63188497` (80 panels and moments, 31 GB) and Phoenix `21571504` (58 GB of 2026-08-31
  KP14 scratch), both verified `ARCHIVE_OK`.
