# Cluster runs

Every job started on Sol or Phoenix for this repository: where it runs, what it waits on, what it
writes, and what became of it. One section per campaign, newest first. The status columns are
updated when jobs end; between updates, `_scratch/watch_campaign.sh` polls both clusters and writes
`_scratch/CAMPAIGN_STATUS.md`.

## Where output goes, and why it is arranged this way

- **Result files are named by model, tag and seed only** (`variants/common/runstamp.py`, `stem`).
  Nothing in the name records T, the window or the cluster. A run that changes the sample needs its
  own tag -- X3 is `vyxT860`, not `vyx` -- and a re-scoring of saved panels needs its own results
  directory (E1). `SEED_STAGE=linear` refuses to write into the directory it reads.
- **`/data/sjpruitt` is shared by both clusters; `/scratch` is per-cluster and purged.** The
  pre-refactor `run_bop_job.sh` writes every run of a model into the same two scratch directory
  names, so each run overwrites the last.
- **SLURM logs carry the cluster**: `outslurm/sol.*` and `outslurm/phx.*`. The two clusters' job ids
  are independent sequences.
- **No `git pull` of the shared checkout while a job that imports from it is running** on either
  cluster. Every task reads its Python from that one tree.

---

## Campaign 2026-09-14 -- the baselines (A1), commit `a5b5673`

Each paper's economy as published, run through the pipeline exactly as every path that departs from
it (`docs/RESULTS.md`, "Baselines: the anchor"; `docs/NEXTUP.md`, A1). Specs precommitted in `3fcfe56`;
the BGN and KP14 solves built on the Mac and committed in `a5b5673`; the GS21 solve runs on Sol. Every
job runs from the shared checkout at `a5b5673`, pulled with both queues empty. Each case narrows the
conditioning columns to the state its paper has (`RF_COLS`, passed as `--rf_cols` to both stages).

### Solves built before submission

| stage | spec | id (precommitted, reproduced) | built | artifact | time |
|---|---|---|---|---|---|
| jstar | `var-bgn_gam-bgnbase-v1` | `c6287d53674cc1ef` | Mac | `variants/bgn_gam/Jstar_bgnbase.csv`, committed | 491 s |
| G | `var-kp_vy-kpbase-v1` | `7bc1f92a225c01b6` | Mac | `variants/kp_vy/G_kpbase0.csv`, committed | 1 s |
| integ | `var-kp_vy-kpbase-v1` | `f299747cd77fc4a1` | Mac | 21 `variants/kp_vy/integ_kpbase0_*.npz`, committed | 344 s |

The kp integ id hashes the G table's raw bytes, so both stages were built on the Mac and shipped through
git rather than rebuilt on Sol (the K4 hazard below). Checks before commit: the BGN table's two regime
columns coincide to 3e-13 at unit multipliers; the 21 KP14 integral tables are identical across the state
nodes to 4e-12, so the economy does not depend on y; every manifest's `recorded_at` is after the spec commit.

### Jobs

| experiment | SEED_SPEC / spec | cluster | job | partition and request | waits on | writes to | status |
|---|---|---|---|---|---|---|---|
| GS21 baseline solve | `var-gs_bx-gsbase-v1` via `variants/gs_bx/run_gsbase_slurm.sh` | Sol | `63257244` | public, 4 cpu, 8G, 1 d | -- | `variants/gs_bx/sol_gsbase/solution.npz` (untracked, about 100 MB), `experiments/registry/c6ae2d52428a7ce5.json` | PENDING at submission (Priority) |
| GS21 baseline seeds 0-9 | `gsbase` | Sol | `63257245` | public, 8 cpu, 64G, 2 d | afterok `63257244` | `variants/results/gs_bx_*_gsbase_*` | PENDING (Dependency) |
| BGN baseline seeds 0-9 | `bgnbase` | Sol | `63257246` | public, 8 cpu, 64G, 2 d | -- | `variants/results/bgn_gam_*_bgnbase_*` | PENDING at submission |
| KP14 baseline seeds 0-9 | `kpbase` | Sol | `63257247` | public, 8 cpu, 64G, 2 d | -- | `variants/results/kp_vy_*_kpbase_*` | PENDING at submission |

Expected: BGN seeds about 3 h each at well under g0235's memory (the unit-multiplier J* range is a third
of g0235's); KP14 seeds about 3 h at about 30 GiB; the GS solve 3.5 to 6 h, then its seeds about 3 h.
`_scratch/watch_baselines.sh` polls Sol every ten minutes and writes `_scratch/BASELINE_STATUS.md`.

### Reading the outcome

- The GS solve ends `SOLVE OK: sol_gsbase=c6ae2d52428a7ce5` or `SOLVE MISMATCH` in
  `outslurm/gs_gsbase.log`; on a mismatch `63257245` stays `DependencyNeverSatisfied` and the id must be
  chased before anything else. Its manifest is written into the shared checkout's registry and must be
  copied back to the laptop and committed, as K4's integ manifest was; `solution.npz` stays on `/data`
  (publish it with `variants/fetch_solves.py --publish` for the fetch hint).
- When the GS solve lands, switch the `gsbase` case's `SOLVE_HINT` in `variants/run_seeds_slurm.sh` to the
  fetch form and clear `solves_pending` in its spec; `tests/test_specs_match_shell.py` enforces the pairing.
- Results: copy the oracle, sidecar, time-series and estimator files (not panels or moments) to the
  laptop, `python variants/aggregate_seeds.py --flagship`, `python variants/fair_gap.py`, and add the three
  rows to RESULTS.md's current-results table; `tests/test_results_md_matches_table.py` fails until they are
  there. Grade each spec's registered prediction in the anchor section.
- No `git pull` of the shared checkout until all four jobs have finished.

---

## Campaign 2026-09-13 -- commit `fef5802`

The proposals ranked in `docs/RESULTS.md` after cross-cutting finding 8. Every job below runs from
the shared checkout `/data/sjpruitt/GitHub/bop-run-upload` at `fef5802`. K5 was withdrawn before
launch as infeasible (RESULTS.md, KP14 proposals).

### Earlier results pulled off first

| what | from | to (verified by checksum, then read-only) | size | job |
|---|---|---|---|---|
| panels and moments of all 80 current seeded runs | `variants/results` in the shared checkout | `/data/sjpruitt/projects/bop-run-upload/archive/2026-09-13_pre-campaign/variants_results/` | 31 GB | Sol `63188205`, rerun as `63188497` |
| run logs, SLURM logs, root `logs/`, `_evalwin/` | the shared checkout | `.../archive/2026-09-13_pre-campaign/{logs,_evalwin}/` | ~6 MB | Sol `63188497` |
| pre-refactor KP14 `main.py` output of 2026-08-31, runs 0-10 (88 + 22 pickles) | Phoenix `/scratch/sjpruitt/bop_kp14` and `bop_temp_kp14` | `/data/sjpruitt/projects/bop-run-upload/archive/phx-scratch-2026-08-31_kp14_main/` | 58 GB | Phoenix `21571504` |

Each archive job writes `SHA256SUMS` into its archive and logs beside it. Sol `63188205` copied all
160 panels and moments, then exited when rsync could not create the nested `logs/` parent directory;
it had not reached its read-only step, and `63188497` re-runs it with that directory created. The
2026-08-31 KP14 output predates the 2026-09-04 arrival-rate fix, so it is kept as a record of that
run, not as citable numbers. The scratch originals are left in place.

**Both verified.** Sol `63188497`: `ARCHIVE_OK` 16:03, 80 panels and 80 moments, no checksum
differences, 476 files in `SHA256SUMS`, none left writable. Phoenix `21571504`: `ARCHIVE_OK` 16:07,
88 + 22 files, no checksum differences, 110 in `SHA256SUMS`, none writable. All 22 files of
`bop_temp_kp14` are byte-identical copies of `bop_kp14`'s panels and moments, so that directory held
nothing unique.

### Jobs

| experiment | SEED_SPEC / spec | cluster | job | partition and request | waits on | writes to | status |
|---|---|---|---|---|---|---|---|
| K4 solve, first attempt | `var-kp_vy-vyg25-v1` | Sol | `63188594` | public, 8 cpu, 16G, 6 h | -- | moved to `/data/sjpruitt/projects/bop-run-upload/k4_sol_first_attempt_63188594/` | CANCELLED at 11 min: printed integ `a2cc8d99c1bc474b`, not the precommitted `8d1308e8f21723f8` |
| K4 solve | `var-kp_vy-vyg25-v1` | Sol | `63188972` | public, 8 cpu, 16G, 6 h | -- | integ tables and manifest; G shipped from the Mac, byte-identical | COMPLETED in 44 min: `SOLVE OK: G=f41d052f1f960c4c, integ=8d1308e8f21723f8`, both precommitted |
| K4 seeds 0-9, first chain | `vyg25` | Sol | `63188595` | public, 64G, 2 d | afterok `63188594` | -- | CANCELLED with its solve; never started |
| K4 seeds 0-9 | `vyg25` | Sol | `63188973` | public, 64G, 2 d | afterok `63188972` | `variants/results/kp_vy_*_vyg25_*` | COMPLETED 10/10, peak 29.2 GiB, longest 7 h 33 min; every run CURRENT for both solves |
| X3 seeds 0-9 | `vyxT860` | Sol | `63188596` | public, 96G, 2 d | -- | `variants/results/kp_vy_*_vyxT860_*` | COMPLETED 10/10, peak 56.9 GiB, longest 5 h 59 min; every run CURRENT for vyx's solves |
| B4 solve | `var-bgn_gam-g0235d-v1` | Phoenix | `21571505` | htc, 4 cpu, 8G, 2 h | -- | `variants/bgn_gam/Jstar_g0235d.csv`, `experiments/registry` | COMPLETED in 15 min: `SOLVE OK: jstar=e136e8b440bce761`, the precommitted id |
| B4 seed 0, the screen | `g0235d` | Phoenix | `21571506` | public, 64G, 2 d | afterok `21571505` | `variants/results/bgn_gam_*_g0235d_s000*` | COMPLETED in 2 h 31 min, peak 15.6 GiB, CURRENT. Screen MISSED: room_eval +0.0186, `sr_orth_eval` 0.1541 |
| B4 seeds 1-9 | `g0235d` | -- | never submitted | -- | seed 0: evaluation-window room >= +0.05 AND `sr_orth_eval` >= 0.30 | -- | closed: the screen missed both gates |
| E1 vyx | `vyx`, `SEED_STAGE=linear` | Phoenix | `21571507` | htc, 4 cpu, 16G, 2 h | -- | `/data/sjpruitt/projects/bop-run-upload/e1_fair_linear/` | COMPLETED 10/10 |
| E1 g0235 | `g0235`, linear | Phoenix | `21571508` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 g0235f | `g0235f`, linear | Phoenix | `21571527` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 g0235s | `g0235s`, linear | Phoenix | `21571528` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 g0235r | `g0235r`, linear | Phoenix | `21571529` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 g28 | `g28`, linear | Phoenix | `21571530` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 gx7 | `gx7`, linear | Phoenix | `21571531` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 bx7 | `bx7`, linear | Phoenix | `21571532` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |

**K4's first solve did not reproduce its precommitted integ id, and was cancelled.** Sol job `63188594`
rebuilt the G stage and printed the precommitted G id `f41d052f1f960c4c`, then integ `a2cc8d99c1bc474b`
instead of `8d1308e8f21723f8`. The integ stage's id hashes its upstream G tables' RAW FILE BYTES
(`solstamp.artifact_digests`); only parameters are quantised to 8 significant digits. The Mac and Sol direct
solves write G with different trailing digits, so a chained id computed on one platform is not reproducible
on another. The fix keeps the precommitment exact: the G tables built on the Mac, from which the integ id was
precommitted, were shipped to Sol byte-identical with their manifest (`solstamp.lookup` finds it and reports no
artifact problems), and `63188972` builds only the integ stage from those bytes. The first attempt's G, 36
integ tables, manifest and log are kept in `/data/sjpruitt/projects/bop-run-upload/k4_sol_first_attempt_63188594/`.

**E1 finished within minutes of submission**: 80 of 80 tasks COMPLETED in 52 to 76 s at 1.5 to 1.7
GiB peak, every run record CURRENT for its solves, and every economy's fair gap inside its registered
bound (`docs/RESULTS.md` finding 8). The 240 summary, sidecar and run-record files are committed in
`variants/results_e1/`; the 80 per-month CSVs (58 MB) stay in
`/data/sjpruitt/projects/bop-run-upload/e1_fair_linear/`. Their provenance tags read `fef5802+dirty`
because K4's solve had begun writing untracked tables into the checkout; the recorded code diff is empty.

**Results pulled, 2026-09-14.** Every job had finished and neither queue held anything. The 147 result
files of K4, X3 and B4 (everything except panels and moments) were copied to the laptop and match Sol's
sha256 sums. Sol's 63 K4 integral tables match the digests recorded in manifest `8d1308e8f21723f8`, and
they are the ones committed. They replace six integral tables that the Mac's G rebuild had left in the
laptop checkout; those differ in their bytes, as the chained-id hazard predicts, and are kept in
`_scratch/k4_mac_partial_integ/`. The first copy listed its panel and moments excludes after its
includes, and rsync applies the first rule that matches. Before it was stopped, B4's panel and moments
and K4 seed 0's moments (560 MB, gitignored) had reached the laptop. The originals stay on `/data`. What
the runs found is in `docs/RESULTS.md`, KP14 Path 1 and BGN Path 1.

Why the split: Sol's public nodes are 515 GB and X3's longer panel is expected to peak near 53 GiB,
so both KP14 jobs go there; Phoenix's `htc` partition starts short jobs at once, which suits E1's
eighty one-minute re-scorings and B4's light, stress-heavy BGN panel.

### Reading the outcome

- A solve job ends `SOLVE OK` or `SOLVE MISMATCH` in `outslurm/solve.<spec>.<jobid>.txt`. On a
  mismatch the dependent array stays pending as `DependencyNeverSatisfied`; cancel it and chase the id
  before anything else.
- E1's output is never copied into `variants/results`: its file names equal the committed runs'. It
  is aggregated beside them, against the pre-registered bound in RESULTS.md ("E1 in detail").
- Commit results from the laptop only once every job that imports the checkout has finished; then
  pull the shared checkout.
