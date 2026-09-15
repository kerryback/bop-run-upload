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

## Campaign 2026-09-15 -- the measurement protocol: RUNNING

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
