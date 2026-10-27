# The 2026-10 Quest campaign's files in this directory

Every economy run on Northwestern's Quest cluster between 2026-10-06 and 2026-10-19 (the `var-bgn_vy-*`,
`var-gs_vy-*`, `var-kp_vj-*`, `var-bgn_q-*`, `var-gs_bx-gq25` and the later `var-kp_vy-*` specs) is recorded here in
the same files, with the same names, as the sixteen published economies: the oracle summary
(`<model>_oracle_<tag>_s<seed>.json` and its `.prov.json`), the oracle's per-month series (`_ts.csv`), the estimators'
per-month-per-method file (`<model>_estimators_<tag>_s<seed>_w360.csv`), its best-per-method summary
(`_summary.csv` and `.prov.json`) and the run record (`_run.json`). None of the published economies' files were
touched, and `economy_table.csv` / `seed_table.csv` were not regenerated; aggregate the whole directory with
`python variants/aggregate_seeds.py --results variants/results --out <somewhere>` to get both tables with the
campaign rows added.

Left on Quest (`/kellogg/proj/axa3203/bop_runs/results`): the saved panels (`*_panel_*.parquet`, 6 GB) and the
moment files (`*_moments_*.npz`, 128 GB); nothing in the report reads them. The Quest re-run of `vyg25` (ten seeds,
fair gap +0.0136 against the committed +0.0148) is also left there, because its files would carry the same names
as the committed `vyg25` record.

The reduced-form screens (`variants/reduced`) are in `variants/reduced/screen/`: the per-seed json and summary csv
of every configuration, read by `python variants/reduced/rf_table.py --dir variants/reduced/screen --ladder <name>`.
The report's tables come from `python variants/campaign_tables.py`, which reads this directory and the screens.

The three benchmarks (`bgnbase`, `kpbase`, `gsbase`) were re-run on Quest on October 19, 2026 (seeds 0-2) only to
save their panels for the neural-network weight analysis of the report's Section 5 (`variants/mlp/`); their result
files stay in `bop_runs/results` because their names coincide with the committed baseline record.

Added October 26, 2026: the economies run on Quest after the October 19 staging, with the same file set per seed
(oracle json, oracle time series, per-month estimator csv, run json, summary csv, provenance json; panels and moment
files stay on Quest). They are the structural second-state economies `vy9t2`, `vy9t2f` (KP14) and `bvy9t2` (BGN);
the GS21 benchmark at gamma_x 0.1 (`gsbg1`) and its exposure wirings `gvyz10g`, `gvyzl`, `gvyzl5`, `gvyzlb`, `gvyzlf`,
`gvyzlg`, `gvyzlp`, `gvyzlr`, `gvyzlrp`, `gvyzlw`; and the product-sign and leverage-type wirings `gvyp10c`, `gvytl`,
`gvytlg`. Their specs, solve tables and registry manifests were committed with the report sections that cite them.

Added October 27, 2026, same file set: the sign-switching type ladders `vyg25s` (KP14) and `bvychs` (BGN), and
`vyg25u` (KP14 `vyg25` without the level compensation); report Section 3b, "The sign of the type loading". The
reduced-form sign ladder (`lsh_*`, `lrt_*`) that motivated them is in `variants/reduced/screen`.
