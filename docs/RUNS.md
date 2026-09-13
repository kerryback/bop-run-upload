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
| K4 solve | `var-kp_vy-vyg25-v1` | Sol | `63188594` | public, 8 cpu, 16G, 6 h | -- | `variants/kp_vy/*vyg25*`, `experiments/registry` | running (16:06) |
| K4 seeds 0-9 | `vyg25` | Sol | `63188595` | public, 64G, 2 d | afterok `63188594` | `variants/results/kp_vy_*_vyg25_*` | pending on the solve |
| X3 seeds 0-9 | `vyxT860` | Sol | `63188596` | public, 96G, 2 d | -- | `variants/results/kp_vy_*_vyxT860_*` | 2 running, 8 pending (16:06) |
| B4 solve | `var-bgn_gam-g0235d-v1` | Phoenix | `21571505` | htc, 4 cpu, 8G, 2 h | -- | `variants/bgn_gam/Jstar_g0235d.csv`, `experiments/registry` | running (16:06) |
| B4 seed 0, the screen | `g0235d` | Phoenix | `21571506` | public, 64G, 2 d | afterok `21571505` | `variants/results/bgn_gam_*_g0235d_s000*` | pending on the solve |
| B4 seeds 1-9 | `g0235d` | -- | not submitted | -- | seed 0: evaluation-window room >= +0.05 AND `sr_orth_eval` >= 0.30 | -- | gated |
| E1 vyx | `vyx`, `SEED_STAGE=linear` | Phoenix | `21571507` | htc, 4 cpu, 16G, 2 h | -- | `/data/sjpruitt/projects/bop-run-upload/e1_fair_linear/` | COMPLETED 10/10 |
| E1 g0235 | `g0235`, linear | Phoenix | `21571508` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 g0235f | `g0235f`, linear | Phoenix | `21571527` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 g0235s | `g0235s`, linear | Phoenix | `21571528` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 g0235r | `g0235r`, linear | Phoenix | `21571529` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 g28 | `g28`, linear | Phoenix | `21571530` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 gx7 | `gx7`, linear | Phoenix | `21571531` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |
| E1 bx7 | `bx7`, linear | Phoenix | `21571532` | htc, 4 cpu, 16G, 2 h | -- | same | COMPLETED 10/10 |

**E1 finished within minutes of submission**: 80 of 80 tasks COMPLETED in 52 to 76 s at 1.5 to 1.7
GiB peak, every run record CURRENT for its solves, and every economy's fair gap inside its registered
bound (`docs/RESULTS.md` finding 8). The 240 summary, sidecar and run-record files are committed in
`variants/results_e1/`; the 80 per-month CSVs (58 MB) stay in
`/data/sjpruitt/projects/bop-run-upload/e1_fair_linear/`. Their provenance tags read `fef5802+dirty`
because K4's solve had begun writing untracked tables into the checkout; the recorded code diff is empty.

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
