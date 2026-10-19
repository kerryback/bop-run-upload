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
