# Task brief — get this week's work onto Sol, then run the GS solve array

For the ASU session. The GS work is yours; the SLURM script is yours; the cluster is where
it belongs. But there is a gate in front of it that nobody has hit yet.

## The gate: none of this week's work is on Sol

| | state |
|---|---|
| local `main` | `ee7344a`, **22 commits unpushed** |
| `origin/main` (**kerryback**/bop-run-upload) | `3fb94a1` "BIG PLAN" |
| Sol `~/GitHub/bop-run-upload` | `35bf8b5`, 8 behind origin, 2 files dirty |

So Sol is roughly **30 commits behind** and has none of: the corrected `config.py`
(delta/rho_x/sigma_x/kappa_e), the regenerated `utils_gs21` solfiles, the rebuilt `kp_vy`
tables, `experiments/registry/` manifests, `solstamp`, or your `gs_bx` work.

**Seth decides how to close that gap — ask before moving.** Two routes:

- **rsync the working tree** local → Sol. Keeps 22 commits of parameter corrections out of
  a co-author's repo until Seth is ready to present them. Probably right for now.
- **push to `origin/main`** then `git pull` on Sol. Simpler and gives Sol real git
  provenance, but it publishes to **Kerry's** repo. That is Seth's call, not ours.

Either way, do not `git push` without him saying so.

## Then, in order

1. **Verify the `bop` env on Sol covers the variants.** `~/.conda/envs/bop` exists. The
   variants need numpy, pandas, scipy, joblib, statsmodels, sklearn, pyarrow. The run env
   is deliberately minimal, so check rather than assume — and remember the documented trap
   in `run_bop_job.sh:33-40`: `source activate` does **not** prepend the env's `bin/` to
   `PATH` under sbatch, so `python` silently resolves to the mamba base and every task dies
   in ~1 s with `ModuleNotFoundError: numpy`. `export PATH="$CONDA_PREFIX/bin:$PATH"` is
   already in that script; make sure yours has it too.

2. **Create the scratch layout.** `/scratch/sjpruitt` exists (90% full, 385T free) but has
   no `bop_*` directories. `outslurm/` already exists in the repo on Sol.

3. **Submit `variants/gs_bx/run_gs_bx7_slurm.sh`** — 5 array tasks, one per exposure type,
   8 h / 8 G / 4 cores, BLAS pinned. Your measured envelope, ~3 h 35 m per task, all five
   concurrent.

4. **Validate and manifest.** Each task should record its own manifest via `solstamp`.
   When they land, confirm five distinct `solve_id`s recorded, `committable: false` on all
   (~78 MB each), and report the `achieved` blocks — `exit`, `sweeps`, `qerr_rel` — since
   that is exactly what tells us whether they converged or rode the 5600-sweep cap.

5. **Run `variants/gs_bx/validate_gs_bx.py`** before declaring them usable.

## Report back

The five recorded solve_ids, their `achieved` blocks, wall times against your ~3 h 35 m
projection, and whether the `kappa_e = 0.025` economy converged the same way `sol_reg` did
at `kappa_e = 0` (it rode the cap; a different exit path would be worth knowing).

## Boundaries — unchanged

Yours: `variants/gs_bx/**`, anything under `/scratch/sjpruitt/` on Sol, and your own
findings file. Not yours: `config.py`, `utils*/`, `tests/**`, `variants/kp_vy/**`,
`variants/bgn_gam/**`, `docs/refactor/WORKING.md`. **Nothing committed, nothing pushed.**

Note `variants/results/` is shared with the primary session's Phase 1 work, keyed by
`--tag`. Yours is `bx7`; theirs are `g0235` and `vyx`. No collision, but do not run the
oracle/estimator stages that `run_gs_bx7.sh` continues into — those are Phase 1 and belong
to the primary session's wrapper.
