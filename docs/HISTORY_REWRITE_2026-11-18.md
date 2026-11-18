# History rewrite of 2026-11-18

## What changed

On 2026-11-18 the history of `main` and `quest-campaign-2026-10` was rewritten with
`git filter-repo`. Every month-by-month estimator csv was removed from every commit:
`variants/**/*_estimators_*.csv`, except the `_summary.csv` files. Nothing else changed, so
each rewritten commit has the same code, specs, manifests, summaries and solve tables as the
commit it replaces. One commit became empty and was dropped: `c8ed371a` of 2026-10-19, which
only added the quest campaign's month-by-month files.

**Why.** These files made up 1.5 GB of the repository's 2.4 GB, and no part of the pipeline
reads them. Every table and the report are built from the `_summary.csv` files; see
`variants/results/README_monthly.md`. The repository is now about 0.95 GB, and most of what
remains is the committed solve tables.

## What the rewrite does not change

- **Author dates.** `filter-repo` keeps them, so `tests/test_precommitment_is_real.py` dates
  every pin exactly as before.
- **File contents.** File contents were not edited, and no solve_id moved.
- **The data that was removed.** It is kept in full: every version of every file that was
  removed (1050 versions of 769 files) is in Seth's Dropbox. The archive is
  `bop-archive/monthly_csv/bop_monthly_estimator_csv_2026-11-18.tar.gz`, with `INDEX.tsv`.
  A git bundle of the complete pre-rewrite history is also there:
  `bop-archive/bop-run-upload_full-history_pre-rewrite_2026-11-18.bundle`.

## Old commit ids

All commit ids changed. Older text still uses them in three places:
- the `prov` column of the `_ts.csv` and `_summary.csv` files;
- the `git.sha` field of every `.prov.json` sidecar;
- commit references in docs, specs and comments.

`docs/commit_map_2026-11-18.tsv` maps each old commit to its new one. Each line is
`old new`, both as full 40-character ids. A dropped commit maps to forty zeros. To resolve a
short id:

    grep '^<old short id>' docs/commit_map_2026-11-18.tsv

`tests/test_solve_impact.py` does this lookup for its fixture commits.

## If you have a clone from before the rewrite

Delete it, or move it aside, and clone again:

    git clone https://github.com/kerryback/bop-run-upload.git

Do not pull into an old clone, and do not push from one: merging the old history back would
restore everything that was removed. If an old clone holds work that was never pushed, ask
Seth before doing anything with it.

A fresh clone has no GS21 solution.npz files, since those have never been in git. Fetch them
from the shared Dropbox folder. Your local path to that folder will differ from Seth's:

    python variants/fetch_solves.py --all --from "<your path to>/BGN and Kelly Malamud/solves"
