# Month-by-month estimator CSVs: not in git

`run_estimators.py` writes two files per economy, seed and window:

- `{model}_estimators_{tag}_s{seed}_w{window}_summary.csv`: about 27 KB, **committed**. This file is
  read by `aggregate_seeds.py`, `campaign_tables.py`, `penalty_gate.py`, `score_market.py` and the report.
- `{model}_estimators_{tag}_s{seed}_w{window}.csv`: every month, about 5.5 MB, **gitignored since
  2026-11-18**. Only `unconditional_sr.py`, a diagnostic, reads it.

Until 2026-11-18 the month-by-month files were committed too. They made up 1.5 GB of the repository's
2.4 GB, so they were removed from the history. Every version that was ever committed is kept
in one archive (1050 versions of 769 files, 5.09 GB uncompressed, 1.5 GB compressed), together with
`INDEX.tsv`. Each row of `INDEX.tsv` gives the repo path, the git blob id, the sha256, the size, and
the commit and date the version first appeared. The archive's location is Seth's Dropbox,
`bop-archive/monthly_csv/bop_monthly_estimator_csv_2026-11-18.tar.gz`; ask Seth for a copy.

Inside the archive, a file's version at the head of `quest-campaign-2026-10` sits at its repo path.
Earlier versions sit under `_versions/<path>.<first 12 characters of the blob id>`.

From now on, a cluster run leaves the month-by-month CSV in its results folder on that cluster.
To read them, point `BOP_RESULTS_DIR` at that folder, or at an unpacked copy of the archive's
`variants/results/`:

    BOP_RESULTS_DIR=/path/to/variants/results python unconditional_sr.py
