# Cluster runs

Every job started on Sol or Phoenix for this repository: where it runs, what it waits on, what it
writes, and what became of it. Newest first.

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

## Two hazards worth carrying forward

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

---

## Campaign 2026-09-21 -- the corrected pricing and the amended ridge grid: COMPLETE (ran to 2026-09-22)

All thirteen economies again, and for the second time no economy's parameters change. Two things do:

1. **The pricing fix** (merge `91095fb`). `variants/kp_vy` priced the mean-reverting state `y` as a
   constant addition to the discount rate -- exact for KP14's GBM shocks, wrong for an OU state whose
   Girsanov adjustment saturates -- and BGN's `vasicek.py` added `sigma12 = -Cov(log z, r)` to the
   cumulative variance once instead of twice. Nine economies re-solve: all three `kp_vy`, all six
   `bgn_gam`. The four `gs_bx` solves are untouched and reused.
2. **The ridge grid** `1e-5 … 10` -> `1e-7 … 1000`, eleven values. The 2026-09-15 campaign's own gate
   failed in five of thirteen economies and in both directions (`vyx` 3/10 and `vyg25` 4/10 at the
   floor, `gx7` 2/10, `bx7` 7/10, `g0235s` 5/10 at the ceiling). Estimator-side, so it moves every
   row -- which is why all thirteen specs are bumped to v3-equivalent and all thirteen economies
   re-run, not only the nine.

### Before submission, in this order

| step | what | why this order |
|---|---|---|
| 1 | delete the two superseded KP14 pricing docs and fix their references FIRST | DONE, `028ec82`. Two of the ten references are in `parameters_kp14.py` and `kp14_fd_vy.py`, which are digested into every kp_vy solve_id. A comment-only edit re-keys them, so doing this after the ids are pinned means pinning them twice |
| 2 | `python variants/common/protocol.py`'s `KAPPAS`, the literal copy in `run_seeds_slurm.sh`, and the `--kappas` in `run_g0235.sh` / `run_vyx.sh` | DONE, `f2032b5`. Three tests compare them against each other; the fair benchmark's grid is DERIVED and needs no edit |
| 3 | compute all fifteen new solve_ids WITHOUT solving | DONE. BGN's six from `_scratch/precommit_id.sh bgn`; KP14's six needed a throwaway `git worktree`, because an integ id hashes the G tables' bytes and cannot be known until G exists. Seven G solves and all six ids: 21 s |
| 4 | commit the thirteen new specs pinning them, with `solves_pending` on the nine | DONE, `f2032b5`. Before any manifest exists, or `tests/test_precommitment_is_real.py` classifies them retrofitted and `precommitted: true` becomes a false claim. Verified after committing: the nine read `precommitted`, the four `gs_bx` read `retrofitted` but are exempt through `reused_solves` |
| 5 | `bash variants/bgn_gam/rebuild_all_jstar.sh` on the Mac, and `build_vy_tables.py` for `vyx`, `vyg25`, `kpbase` | one machine for all six BGN tables, for the reason the 2026-09-15 campaign records below. KP14 is REBUILT under prefix `vyx` rather than adopting the committed `vyxq` tables: the G solve_id is prefix-blind, because `extra` is never hashed, so `vyxq` and a post-fix `vyx` share an id and `solstamp.record` would silently rewrite one manifest over the other |
| 6 | check each printed solve_id against its spec's `expected_solves` | fifteen ids computed without solving; each must come back exactly |
| 7 | clear `solves_pending` in the nine specs; `git rm` the transitional `G_vyxq*` / `integ_vyxq*`; commit the tables and manifests; push | `solves_pending` is excluded from `spec_hash`, so clearing it does not disturb the pinning. **Until this lands the nine seed jobs abort in seconds** on the runner's "no live solve recorded" precondition |
| 8 | supersede the eighteen old manifests -- `supersede`, never `retire` | two live ids under one tag and stage make `runstamp.live_solves` return both, and then every seed of that economy reads STALE forever. Retiring instead breaks `tests/test_specs_match_shell.py`, which refuses a spec pinning a retired id -- and the old specs truthfully pin the old ids |
| 9 | `bash variants/cluster_pull.sh` in the shared checkout, **once**, with both queues empty | never a plain `git pull`; one tree serves both clusters |
| 10 | `bash variants/submit_campaign.sh sol --dry` and `... phx --dry`, then without `--dry` | the dry run is each scheduler's own answer on start time and fit |

### Sizing

**Memory is unchanged** and the 2026-09-15 peaks below still stand: the amendment is estimator-side,
the panels are the same size, and BGN's corrected J\* falls 14-21%, which moves memory DOWNWARD --
BGN's peak tracks the J\* value scale (`g0235f` 37 to 345 and 22 GiB, `g0235r` 390 to 2041 and 77 GiB).

**Walltime is what moves.** The estimator stage runs eleven penalties against seven. At the measured
77-81 s per evaluation month at eight penalties, that is about +57% on the DKKM stage, so every
one-day row goes to two days, the two-day rows to three, and the two Sol highmem rows from four days
to six. `scontrol update TimeLimit` is refused on Sol, so a walltime kill loses the seed outright and
over-requesting on an empty queue costs nothing. **Budget roughly 900 node-hours across 130
seed-jobs**, against the 2026-09-15 campaign's 650.

### Submitted 2026-09-21

Both queues empty and the shared checkout at `de109da` and clean before submission; both dry runs
accepted; every request fit its nodes. Phoenix took eleven of the thirteen because it is far less
contended -- 23 pending in `public` against Sol's 1435, fairshare 0.0241 against 0.0047 -- leaving
Sol only the two highmem economies.

| cluster | SEED_SPEC | partition | request | job |
|---|---|---|---|---|
| Sol | `g0235s` | highmem | 8 cpu, 128G, 6 d | 63759851 |
| Sol | `g0235r` | highmem | 8 cpu, 128G, 6 d | 63759852 |
| Phoenix | `bgnbase` | public | 8 cpu, 40G, 2 d | 21604189 |
| Phoenix | `kpbase` | public | 8 cpu, 48G, 2 d | 21604190 |
| Phoenix | `gsbase` | public | 8 cpu, 32G, 2 d | 21604191 |
| Phoenix | `vyx` | public | 8 cpu, 48G, 3 d | 21604192 |
| Phoenix | `vyg25` | public | 8 cpu, 48G, 3 d | 21604193 |
| Phoenix | `g28` | public | 8 cpu, 32G, 2 d | 21604194 |
| Phoenix | `gx7` | public | 8 cpu, 32G, 2 d | 21604195 |
| Phoenix | `bx7` | public | 8 cpu, 32G, 2 d | 21604196 |
| Phoenix | `g0235` | public | 8 cpu, 64G, 3 d | 21604197 |
| Phoenix | `g0235f` | public | 8 cpu, 40G, 2 d | 21604198 |
| Phoenix | `g0235d` | public | 8 cpu, 40G, 2 d | 21604199 |

**The checkpoint did not skip anything this time.** All 110 Phoenix tasks went straight to RUNNING
and none had finished 45 s later -- the 2026-09-15 failure was seven arrays exiting in three seconds
each with "already complete and current", and it happened because a protocol change left the solve
ids untouched. Here every one of the nine re-solved economies has a new id, so every seed reads
STALE and re-runs. Spot-checked in the logs: `vyx` seed 0 consumes `8b4702211a1f5efb` and
`c88b05407e4b377d` with `KP_PARAM_OVERRIDES` and `KP_VY_PREFIX` both matching
`var-kp_vy-vyx-v4`; `bgnbase` seed 6 consumes `cb6340649eace098`. Sol's two arrays were pending on
highmem nodes at submission.

**Do not pull the shared checkout until both queues are empty again.** Done 2026-09-22, with both
queues empty and after the laptop had committed and pushed the 910 result files.

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

### Reading the outcome

`python variants/penalty_gate.py` first, before any number, then `python variants/aggregate_seeds.py`.
Done 2026-09-22; the result is in `docs/RESULTS.md`. Two things are worth carrying forward from it:

- **The gate's criterion was wrong and has been amended.** It asked only whether the argmax was
  interior. Three rows failed that test (`g0235s` 6/10, `bx7` 7/10, `gx7` 5/10) and none of them is
  censored: the Sharpe-vs-penalty curve is FLAT at the ceiling, because as kappa grows the ridge
  direction stops depending on it and Sharpe is scale-invariant. Across all 130 seeds the 16 at the
  ceiling gained at most 2.7e-05 over their best interior penalty. Reading the gate literally would
  have bought a third decade and another ~990 node-hours to move three numbers by less than 3e-05.
  `penalty_gate.py` now tests materiality as well as position, and `tests/test_penalty_gate.py` pins
  both halves.
- **The penalty-gate table in `docs/RESULTS.md` is pinned by NO test** and was regenerated and
  hand-transcribed. That remains true for the next campaign.

---

## Campaign 2026-09-15 -- the measurement protocol: COMPLETE (ran to 2026-09-17)

All thirteen economies at one protocol, so that the only thing separating two rows of
`docs/RESULTS.md` is the economy (`docs/RESULTS.md`, "The measurement protocol"; `docs/NEXTUP.md`).
No economy's parameters change. What changes: burn-in 400 for all three models (it was 300 / 400 /
300), the ridge grid `1e-5 … 10` for every economy (it was three different grids), the model's full
conditioning set for the baselines too, and ten seeds for `g0235d`, which had one.

**Split across both clusters.** `bash variants/submit_campaign.sh <sol|phx>` holds the allocation and
the reason for each row; `--dry` asks each scheduler for its own start estimate without queueing
anything. Two economies on Sol, eleven on Phoenix. Why, measured on 2026-09-15 rather than assumed:

| | Sol | Phoenix |
|---|---|---|
| `public` node memory | ~515 GB | 112-125 GB, six nodes at 186 GB |
| `highmem` | 11 nodes at 2.0-2.3 TB, 3 idle | 2 nodes at 1.5 TB |
| `public` pending / idle nodes | 725 pending, 2 idle | 12 pending, 253 idle |
| fairshare | 0.0102 | 0.1303 |
| the scheduler's own estimate for one task | highmem today 17:53; **public Sept 18** | immediate at every size up to 112G |

- **Memory is the durable constraint and it decides first.** `run_oracle.py:152-154` allocates its
  arrays over `T + burnin`, so burn-in 300 -> 400 raises BGN's and GS21's peaks by 12.5%: g0235r's
  measured 77.0 GiB projects to ~87 and g0235s's 74.0 to ~83. Those two are the only economies that
  cannot sit comfortably on a 112-125 GB Phoenix node -- and the historical record is what happens
  when they do not fit, since all ten g0235r seeds and three g0235s seeds were once killed at a 64
  GiB cap. They go to Sol. KP14's peaks (29-31 GiB) are unchanged, its burn-in already being 400.
- **Fairshare and backlog are temporary and decide the rest.** Phoenix's fairshare is 12.7x Sol's and
  its public partition has 253 idle nodes against Sol's 2, so the eleven economies that fit anywhere
  go to Phoenix, where they start at once.
- **On Sol, use `highmem`, not `public`.** Sol's own scheduler put an identical task three days out on
  public and six hours out on highmem. The two BGN economies need the memory anyway, so highmem is
  both the right hardware and the shorter queue. 128G against a projected 87 GiB is 58% headroom on a
  2 TB node; if a seed still OOMs, 256G there costs nothing.
- **Re-check before submitting.** `sshare -U` and `squeue -h -p public -t PD | wc -l` on each, then
  `--dry`. Fairshare and backlog move; if they have flipped, move rows between the two lists. The
  memory column is the only part that is not negotiable.

**One seed first, or not.** The standing rule is one seed then `sacct` before sizing an array
(`docs/RESULTS.md`, "Adding an experiment"). It is deliberately skipped here, for these thirteen
only: they are re-runs of economies whose peak memory is already measured, the single change is a
known +12.5%, every request above carries at least 45% headroom on the projection, and Phoenix starts
immediately -- so a resubmission after an OOM costs hours, while thirteen serial probes cost days.
The rule stands for any economy whose memory has never been measured.

### Before submission, in this order

| step | what | why this order |
|---|---|---|
| 1 | `python variants/solve_impact.py --worktree` | names every solve the burn-in edit re-keys. Expect exactly BGN's six J\* ids; KP14's and GS21's producers did not change |
| 2 | commit the specs, the runner, `protocol.py` and the tests | DONE, `066e2d6`. The six new BGN jstar ids were computed WITHOUT solving and must be committed BEFORE their manifests, or `tests/test_precommitment_is_real.py` classifies them retrofitted and the specs' `precommitted: true` becomes a false claim |
| 3 | `bash variants/bgn_gam/rebuild_all_jstar.sh`, about 3 to 4 min per table | reads each live spec's `params` rather than taking a hand-copied blob, and builds all six on ONE machine -- see below for why that matters |
| 4 | check each printed solve_id against its spec's `expected_solves` | the six ids were computed without solving; each must come back exactly |
| 5 | clear `solves_pending` in the six BGN specs; commit the tables and manifests; push | `tests/test_specs_match_shell.py` pins the pairing. **Until this lands, the six BGN seed jobs abort in seconds** on the runner's "no live solve recorded" precondition, so there is no point submitting them earlier |
| 6 | `bash variants/cluster_pull.sh` in the shared checkout, **once**, with both queues empty | never a plain `git pull`. Both clusters import Python from this one tree, so this is the last moment anything may pull until both queues are empty again. It must come BEFORE anything is removed from `variants/results`: the 956 files there are TRACKED, the pull deletes 106 of them itself, and a checkout that cannot fast-forward aborts the script |
| 7 | decide what to do with the 51 GB of pre-protocol panels and moments in `variants/results` | They are the 398 GITIGNORED files -- panels 2.6 GB, moments 48 GB -- and are the only content there that is not in git; the 956 tracked summary files live in history at `066e2d6`. The campaign overwrites all of them except `kp_vy/vyxT860`'s twenty, whose economy is retired, so those are pure leftovers. Archiving is optional and `docs/RESULTS.md`'s own policy says superseded results are not retained; `/data` has 846 GB free either way, so nothing forces the decision |
| 8 | `bash variants/submit_campaign.sh sol --dry` and `... phx --dry`, then without `--dry` | the dry run is each scheduler's own answer on start time and whether the request fits its nodes |

GS21's four solutions are reused as published and all eleven `variants/gs_bx/sol_*/solution.npz`
(88-101 MB each) are already present in the shared checkout, so **one copy serves both clusters** and
no fetch is needed; KP14's 8 G tables and 148 integral tables are committed. No solve job is needed on
either cluster. `/data/sjpruitt` has 846 GB free against about 52 GB of new panels.

**Burn-in does not enter the BGN J\* solve, and the rebuild is a formality -- but it must happen on
one machine.** `vasicek.py` never reads `burnin`; it only arrives in the module namespace through
`from parameters import *`. Measured rather than argued on 2026-09-15: five of the six tables rebuilt
at burn-in 400 are BYTE-IDENTICAL to the committed ones, and `Jstar_g0235d.csv` -- the only one of the
six originally built on Phoenix (job `21571505`), the rest on the Mac -- differs by 5e-15 relative on
a bit-identical `r` grid. Rebuilding it at burn-in 300 and 400 on the same machine gives byte-identical
output, which isolates the variable: the difference is the platform, not the burn-in. A jstar
`solve_id` is parameters plus source digests, with no upstream artifacts, so it is
platform-independent; the MANIFEST records the table's sha256, so the committed table must come from
the same machine that recorded it. Hence one machine for all six. The old manifests
(`c6287d53674cc1ef` and the other five) stay in the registry describing tables no longer on disk;
nothing live references them.

### Submitted 2026-09-15, and the restart it needed

**The checkpoint skipped seventy tasks, and the reason is worth keeping.** The first submission put
all thirteen arrays up. The six BGN arrays ran; the seven KP14 and GS21 arrays exited in three
seconds each, 10 of 10 tasks, with `seed N already complete and current for the recorded solve --
nothing to do`.

The per-seed checkpoint keys on the SOLVE, on the premise that "re-solving the economy invalidates
every seed at once". The protocol change broke that premise from a direction the checkpoint could not
see: it moved the ridge grid, the burn-in and the conditioning columns while leaving KP14's and
GS21's solve ids untouched -- only BGN's jstar ids were re-keyed, which is why only BGN ran. So
seventy seeds matched their pre-protocol results, which were still sitting in the shared
`variants/results`, and were skipped.

Two fixes, one immediate and one durable:

- **Immediate**, because a code change could not be pulled with the BGN arrays running: the seventy
  stale `*_w360_run.json` records were removed from the shared results directory, which is the only
  thing the checkpoint reads, and the seven economies were resubmitted. They then ran.
- **Durable**: `runstamp.run_matches_protocol` checks the estimation half -- ridge grid, window, fair
  benchmark, winsorisation -- against `variants/common/protocol.py`, and `run_is_current` folds it in.
  A seed whose record predates a protocol change is no longer current whatever its solve says.
  `tests/test_runstamp_multitag.py` pins the failure. **This is committed but deliberately NOT
  pulled: no pull may happen while either queue is busy.** Pull it when both are empty, before any
  resubmission.

The lesson generalises past this campaign: a result depends on its solve AND on how it was measured,
and until 2026-09-15 only the first half was checked.

### Jobs

| cluster | experiment | SEED_SPEC | partition | request | status |
|---|---|---|---|---|---|
| Sol | BGN regime, slow | `g0235s` | highmem | 8 cpu, 128G, 4 d | Sol `63323405`, PENDING (highmem, est. 17:53) |
| Sol | BGN regime, rare | `g0235r` | highmem | 8 cpu, 128G, 4 d | Sol `63323406`, PENDING (highmem, est. 17:53) |
| Phoenix | BGN as published | `bgnbase` | public | 8 cpu, 40G, 1 d | Phoenix `21575311`, RUNNING |
| Phoenix | KP14 as published | `kpbase` | public | 8 cpu, 48G, 1 d | Phoenix `21575424` (resubmitted), RUNNING |
| Phoenix | GS21 as published | `gsbase` | public | 8 cpu, 32G, 1 d | Phoenix `21575427` (resubmitted), RUNNING |
| Phoenix | KP14 Path 1 parent | `vyx` | public | 8 cpu, 48G, 2 d | Phoenix `21575425` (resubmitted), RUNNING |
| Phoenix | KP14 higher price of risk | `vyg25` | public | 8 cpu, 48G, 2 d | Phoenix `21575426` (resubmitted), RUNNING |
| Phoenix | GS21 gamma(x) | `g28` | public | 8 cpu, 32G, 1 d | Phoenix `21575428` (resubmitted), RUNNING |
| Phoenix | GS21 gamma(x) x types | `gx7` | public | 8 cpu, 32G, 1 d | Phoenix `21575429` (resubmitted), RUNNING |
| Phoenix | GS21 types under a regime | `bx7` | public | 8 cpu, 32G, 1 d | Phoenix `21575430` (resubmitted), RUNNING |
| Phoenix | BGN regime | `g0235` | public | 8 cpu, 64G, 2 d | Phoenix `21575319`, RUNNING |
| Phoenix | BGN regime, fast | `g0235f` | public | 8 cpu, 40G, 1 d | Phoenix `21575320`, RUNNING |
| Phoenix | BGN regime, stress-dominant | `g0235d` | public | 8 cpu, 40G, 1 d | Phoenix `21575321`, RUNNING, **nine new seeds** |

**Memory follows calm spells**, which is why the two slow BGN economies are the ones that needed Sol:
longer calm spells price risk cheaply for longer, so firms accept more projects and the panel arrays
grow. The J\* value scale orders exactly as memory does (g0235f 37 to 345, g0235 57 to 483, g0235s
148 to 1262, g0235r 390 to 2041; peak memory 22, 39, 74 and 77 GiB), and within an economy the
longest calm spell in a seed's panel ranks with its peak memory at Spearman +0.84 (g0235s) and +0.80
(g0235r).

**The wider ridge grid is the added cost.** It is seven penalties against four for BGN and KP14 and
eight for GS21, and `run_seeds_slurm.sh`'s own measurement is 77 to 81 s per evaluation month at eight
penalties, so about 2.8 h of DKKM per seed on top of the panel build. Budget roughly 1.75x the old
estimator stage for the BGN and KP14 economies and about the same as before for GS21. Total on the
order of 650 node-hours across 130 seed-jobs.

### While it runs

- **Nothing may `git pull` on either cluster.** Both import from the one shared tree. The campaign is
  not finished until `squeue -u sjpruitt` is empty on BOTH.
- Both clusters write into the same shared `variants/results`. No two economies share a filename, so
  they cannot collide there; `outslurm/` is shared too, which is why `submit_campaign.sh` prefixes
  each log with its cluster (`outslurm/sol.seeds.*`, `outslurm/phx.seeds.*`) rather than relying on
  the two job-id sequences never meeting.
- Watch with `squeue -u sjpruitt -h -o "%.10i %.10P %.9j %.2t %.10M %.6D %R"` on each, and
  `sacct -u sjpruitt -o JobID,JobName%14,State,Elapsed,MaxRSS,ReqMem --units=G` for peaks as tasks
  land. **sacct reports MaxRSS in KiB unless told otherwise**; the first 2026-09-15 baseline report
  misread it by a factor of 1024.

### Reading the outcome

- **The gate that is not a test.** For each economy, read the winning penalty per seed out of
  `variants/results/<model>_estimators_<tag>_s<seed>_w360_summary.csv` (the `kappa` column of the
  best `rff*` row). It must be INTERIOR -- neither `1e-5` nor `10` -- in at least 8 of 10 seeds. If
  `1e-5` wins, DKKM's Sharpe is still censored and the grid needs another decade before any new
  economy is run. This is the whole point of the campaign, so it is checked first and reported beside
  the results.
- Every oracle sidecar must record `spec_check` verified, `env_check` verified, `burnin` 400 and the
  full `rf_cols`; every estimator `run.json` must record the seven-value `kappas` and `window` 360.
  Grep the sidecars, not the logs.
- Copy the oracle, sidecar, time-series and estimator files (not panels or moments) to the laptop,
  then `python variants/aggregate_seeds.py` and `python variants/fair_gap.py`. The aggregator refuses
  to write the canonical table if any row is off protocol.
- Once `g0235d` has ten seeds, delete the SCREEN tier: its heading in `docs/RESULTS.md`, and
  `test_every_screen_is_reported_under_a_screen_heading` plus the `current` argument of `_csv_rows`
  in `tests/test_results_md_matches_table.py`. The protocol's seed count is ten for every economy, so
  there is no tier left to define.
- Once every economy carries `--fair_linear` from its own run, `variants/results_e1/` is redundant:
  `fair_gap.py` will read all thirteen as "own run". Fold `ew` and `fair_gap` into
  `aggregate_seeds.py`'s `economy_table.csv`, delete the second directory and CSV, and drop the
  two-table join in `tests/test_results_md_matches_table.py`. Deferred until then deliberately --
  doing it now would mean writing an E1 fallback path in order to delete it.
- Then rewrite `docs/RESULTS.md`: delete the "Status: every number below predates the protocol"
  section, refill both tables, and grade each spec's registered prediction.

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
