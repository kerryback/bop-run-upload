# variants/klc — running on Northwestern Quest (the `kellogg` partition)

Site glue only. Nothing here defines an economy or a measurement: the economies are the
`SEED_SPEC` cases in `../run_seeds_slurm.sh`, pinned to `experiments/specs/`, and the protocol is
`../common/protocol.py`. See "Running this somewhere other than ASU" in `docs/RUNS.md`.

| file | what |
|---|---|
| `klc_env.sh` | sourced by every job: the conda env prefix, threads, where results and logs go |
| `solve_small_klc.sh` | one job: the BGN J* table and the KP14 G + integ chain, then `verify_solves.py` |
| `solve_gs_klc.sh` | a five-task array: one GS21 solve per exposure type |
| `verify_gs_klc.sh` | one job: `verify_solves.py gs`. The GS seed array is submitted `afterok` on it |
| `verify_solves.py` | the solve-correctness gate. Exits non-zero unless every check passes |
| `seeds_klc.sh` | the seed array. Maps one array index to (economy, seed) and execs `../run_seeds_slurm.sh` |

Results go to `$BOP_RESULTS_DIR` (default `/kellogg/proj/axa3203/bop_runs/results`), NOT to
`variants/results`, so the committed record and `docs/RESULTS.md` are untouched. Aggregate with
`python variants/aggregate_seeds.py --results $BOP_RESULTS_DIR` after
`python variants/penalty_gate.py --results $BOP_RESULTS_DIR`.

Core budget: every array carries a `%N` throttle. Check `squeue -u $USER` before raising one.
