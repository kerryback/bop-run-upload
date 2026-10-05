# legacy — the superseded pipeline

**Nothing in this directory produces a reportable number.** It is kept because it is where the
project started, because six tests still pin behaviour it got right, and because `docs/RESULTS.md`
and `variants/REPORT.md` refer back to results it produced. It is not where new work goes.

The live pipeline is [`variants/`](../variants/README.md). Start at
[`docs/quickstart.md`](../docs/quickstart.md).

## What is here

| path | what it is |
|---|---|
| `main.py` | the 7-step driver: panel → moments → portfolios → FF factors → DKKM factors → two SDF fits → evaluation, each step a separate subprocess |
| `utils/` | those step scripts, plus `sparse_3d.py` (sparse 3-D array storage) and `solfile_stamp.py` (the hand-written staleness stamp that `variants/common/solstamp.py` replaced with content addressing) |
| `run_bop_job.sh` | the SLURM array wrapper for `main.py`. Submit from the repo root — SLURM resolves `#SBATCH -o` against the submit directory |
| `deploy_koyeb.sh` | a Koyeb cloud deploy path with AWS S3 upload, last exercised 2026-02-02. Dead |

## Why it was superseded

Three reasons, all of them in `docs/RESULTS.md` and `docs/RUNS.md`:

1. **No measurement protocol.** Runs varied in N, T and window, so two numbers were rarely
   comparable. `variants/common/protocol.py` fixes all three, and `run_seeds_slurm.sh` refuses to
   write an off-protocol run into `variants/results` at all.
2. **No content addressing.** Staleness was a hand-written 5-key stamp that silently missed seven
   parameters and any edit to the solver itself. `variants/common/solstamp.py` digests the whole
   effective parameter namespace plus every source file that can change the output.
3. **Two solve-level pricing defects**, found 2026-09-18 and fixed in `91095fb`. Every number
   produced before that is withdrawn.

## What stayed at the repo root, and why

`config.py`. It is still the **parameter authority**: `tests/test_config_parity.py` asserts that
the private parameter copies under `variants/` have not drifted from it, and every model tree
(`utils_bgn/`, `utils_kp14/`, `utils_gs21/`, `utils_factors/`) imports it. Those trees stayed at the
root for the same reason — they are the published-model implementations, and five tests exercise
them.

## If you do run it

```bash
python legacy/main.py <model> [start] [end] [--chars char1,char2,...]
sbatch --array=0-9 legacy/run_bop_job.sh      # from the repo root
```

`main.py` resolves its step scripts and the repo root from its own location, so it runs from any
working directory. Output goes to `$BOP_SCRATCH_DIR` / `$BOP_TEMP_DIR`; `run_bop_job.sh` sets both.
Note that it writes every run of a model into the same two scratch directory names, so each run
overwrites the last (`docs/RUNS.md`, "Where output goes").
