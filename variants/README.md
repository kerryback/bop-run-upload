# variants — three engineered economies with room for a nonlinear SDF method

These are the experiments behind `REPORT.md` ("why does DKKM barely beat FMR/FFC in the simulated
economies, and what would change that"). Of the 24 economies in that study, the three flagships that give
a complexity method the most to find are kept here, each fully reproducible. The other 21 are gone;
their numbers survive in `results/grid_summary.csv` / `results/summary_grid.xlsx` and in `REPORT.md`.

Everything here is self-contained: nothing imports from the main pipeline (`utils_*`, `config.py`),
and nothing in the main pipeline imports from here. The simulators are modified copies of the Dropbox
code, not of `utils_bgn`/`utils_kp14`/`utils_gs21`. The estimators in `common/` follow the same
conventions as `utils_factors/` (Fama-French SMB from the 2x3 size x BM sort, average-rank
standardization, one evaluation per RFF draw).

Run everything from an environment with the repo requirements plus `pyarrow`
(`pip install -r ../requirements.txt pyarrow`). The shell scripts use `$PYTHON` if set, else `python3`.

## The three economies

| dir | economy | tag(s) | run |
|---|---|---|---|
| `bgn_gam/` | BGN with a 2-state regime multiplying the prices of risk, pushed to multipliers 0.2/3.5 near the closed-form frontier (REPORT §13, §19d) | `g0235` | `bgn_gam/run_g0235.sh` |
| `kp_vy/` | KP14 with a priced OU volatility factor y and three firm types with exposure e^{beta y}, gamma_v = 1.8 and beta in {0.02, 0.07, 0.14} (REPORT §19, §19d) | `vyx` | `kp_vy/run_vyx.sh` |
| `gs_bx/` | GS21 (2-regime, exact-kernel re-solve) with five exposure types beta in {1, 2.5, 4, 5.5, 7} (REPORT §19) | `bx7` | `gs_bx/run_gs_bx7.sh` |

Population numbers from `results/grid_summary.csv` (N=500, T=500; Sharpe ratios from the true conditional
moments; `room` = best nonlinear basis minus best linear-in-ranks basis):

| economy | SR_max | linear ceiling | nonlinear ceiling | room | FMR | FF | RFF (best) |
|---|---|---|---|---|---|---|---|
| bgn_gam g0235 | 0.186 | 0.135 | 0.158 | 0.023 | 0.067 | 0.090 | 0.175 |
| kp_vy vyx | 1.261 | 0.743 | 1.092 | 0.350 | 0.741 | 0.647 | 0.848 |
| gs_bx bx7 | 0.299 | 0.274 | 0.292 | 0.018 | 0.221 | 0.229 | 0.237 |

Realized gap = room x capture (REPORT §19d). BGN g0235 and KP vyx are the extreme versions of the earlier
`g0520`/`g0330` and `vys` economies; GS stays at bx7 because pushing it further (`bx9`) triples the room but
the estimators no longer capture it.

## Pipeline

Each `run_*.sh` runs the same three stages:

1. **Solve / tabulate the model.** `bgn_gam/rebuild_jstar_gam.py` (J*(r) tables per regime, `Jstar_g0235.csv`),
   `kp_vy/build_vy_tables.py vyx` (per-type G functions `G_vyx*.csv` and per-(type, y-node) integrals
   `integ_vyx*_*.npz`; cached while `meta_vyx.json` matches), `gs_bx/gs_solve_reg.py` (one solution per
   exposure type into `gs_bx/sol_*/solution.npz`, about a minute each). A `validate_*.py` in each directory
   checks Euler equations and the reduction to the baseline economy.
2. **Oracle.** `run_oracle.py --model {bgn_gam,kp_vy,gs_bx} --tag TAG --N 500 --T 500 --levels --save_panel`
   simulates a panel, computes the true conditional moments every month, and reports for each feature basis
   the conditional oracle and the constant-theta population portfolio. Writes
   `results/{model}_oracle_{tag}.json`, `..._ts.csv`, and with `--save_panel` also
   `results/{model}_panel_{tag}.parquet` and `results/{model}_moments_{tag}.npz`.
3. **Estimators.** `run_estimators.py --model M --tag TAG --window 360 --levels --include_mkt [--kappas ...]`
   runs FMR, FFC, linear-in-ranks and RFF+ridge (P = 36, 360, 3600) on the saved panel with rolling
   windows, scored with the true moments. Writes `results/{model}_estimators_{tag}_w360.csv` and
   `..._summary.csv` (columns: method, P, kappa, sharpe, hjd, real_sr, t_vs_fm).

Model parameters are overridden through JSON in an environment variable (`BGN_PARAM_OVERRIDES`,
`KP_PARAM_OVERRIDES`, `GS_PARAM_OVERRIDES` / `GS_SIM_OVERRIDES`); the run scripts set the winning values.

## What is (not) in git

Kept in git: all code, the solved tables that are small (`Jstar*.csv`, `G_*.csv`, `integ_*.npz`), the
oracle JSON/CSV and estimator CSV outputs, and the three summary files. Ignored (see `../.gitignore`):
`gs_bx/sol_*/` (about 85 MB each, rebuilt by `run_gs_bx7.sh`), `results/*.parquet` panels and
`results/*_moments_*.npz` (the moments files are 390 MB each and were deleted; `run_oracle.py --save_panel`
regenerates both), and `results/logs/`.

`run_estimators.py` needs the panel and moments files, so re-run the oracle stage first.

## Other files

| file | what |
|---|---|
| `common/oracle.py` | feature bases (FMR raw, linear-in-ranks, +-rf interactions, poly2, decile/pair bins, RFF) and the two-pass population evaluation |
| `common/dkkm_functions.py`, `common/fama_functions.py` | RFF+ridge and Fama-French / Fama-MacBeth estimators with rolling `WINDOW` and conditioning columns `RF_COLS` |
| `collect_results.py` | rebuilds `results/oracle_summary.csv` from every `*_oracle_*.json` on disk. With only the three winners present it would shrink the file to three rows, so do not run it unless the other oracle JSONs have been regenerated |
| `unconditional_sr.py` | exact unconditional Sharpe ratio of each estimated portfolio from the true conditional moments |
| `make_excel.py` | rebuilds `results/summary_grid.xlsx` (Headlines, Summary grid, Definitions sheets) from `results/grid_summary.csv` (needs openpyxl) |
| `REPORT.md` | the full narrative, sections 1-19d, including the 21 economies that were removed |
