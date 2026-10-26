# Neural-network fits of the portfolio weights (campaign economies)

Applies Seth Pruitt's `MLP Analyze` pipeline (repo `sjpsjpsjp/bop-analyze-remote`; `common.py` there holds the
MLP, rank-standardisation, partial-dependence and correlation helpers) to the Quest campaign economies. Nothing here
is imported by the protocol runs.

| file | what it does |
|---|---|
| `extract_weights.py` | rebuilds, for one seed of one economy, the per-firm weights of the true SDF, Fama-MacBeth, Fama-French and DKKM (`rff_ens`, P=3600, one kappa) from the saved `*_panel_*.parquet` / `*_moments_*.npz`, exactly as `run_estimators.py` forms them (same rng stream), and refuses to write unless the per-month conditional Sharpe ratios match the recorded `*_estimators_*_w360.csv` (they match to 1e-16). `--kappa auto` takes the kappa with the best mean `rff_ens` Sharpe over the economy's seeds. Output: `wts_<model>_<tag>_s###.parquet` (month, firmid, the five characteristics, extras, `w_true`, `w_fm`, `w_ff`, `w_dkkm`). |
| `extract_klc.sh` | SLURM array over seeds: `sbatch --export=ALL,MODEL=kp_vy,TAG=vyg25,KAPPA=0.001 --array=0-9 extract_klc.sh` |
| `run_bop.py` | lives next to `common.py` in the `MLP Analyze` checkout (`BOP campaign/run_bop.py`); fits a 5x32 MLP to each weight series per economy, computes the correlations with the truth, the linear/nonlinear split of every fitted surface, the interaction statistic per characteristic pair, and draws the 1x4 and benchmark-vs-3b partial-dependence figures. Reads `BOP_MLP_DATA` (default `bop_runs/mlp/data`). |
| `analyze_klc.sh` | runs `run_bop.py` on a compute node (sets `LD_LIBRARY_PATH` to the env's lib: torch must not load the system libstdc++ before numpy). |
| `mlp_report.py` | writes `docs/campaign_tables/mlp.{md,tex}` and `docs/mlp_figures/`, and fills the `TABLE:mlp` block of the campaign report. |

Economies: benchmarks `bgn_gam/bgnbase`, `gs_bx/gsbase`, `kp_vy/kpbase` (three seeds) and the 3b representatives
`bgn_vy/bvych0`, `gs_vy/gvyp10`, `kp_vy/vyg25` (ten seeds). Results and cached nets: `external/MLP Analyze/BOP campaign/{results,cache}`.

`room_raw.py` (+ `room_raw_klc.sh`): recomputes a saved panel's population ceilings with the random-Fourier basis built with
and without the rank standardization applied after the sines and cosines, next to the linear, quadratic and bin bases
(the check that the 3a economies' lack of room is not an artefact of that standardization; October 21, 2026).
