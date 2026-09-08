# Findings — merging `bop-analyze-remote` into `bop-run-upload`

Assessment only. Nothing was merged, nothing installed, no solve/oracle/estimator run.
Written 2026-09-05 while the primary session held the machine on KP integrals.

**Answer up front.** Merge, at prefix `analyze/`. Keep **two `environment.yml` files**.
The bloat worry is measurable and it is **false as stated** — installed-but-unimported
packages cost nothing at job startup — but two env files are still worth keeping, for
cluster disk quota and for one specific package (`torch`) that is imported by tracked code
and declared in neither file. The real argument for merging is not convenience: **the
interface between the two repos has already silently broken once**, and the break is
currently changing a headline number in the paper's treatment arm (§4.3).

---

## 1. Environment diff

### 1.1 What each side declares

| | `bop-run-upload` | `bop-analyze-remote` |
|---|---|---|
| conda env name | `bop` (cluster only — **not present on this laptop**) | `bop-analyze-remote` (local, 494 MB) |
| python | 3.12 | 3.12 |
| declared in `environment.yml` | numpy≥1.26, pandas≥2.0, scipy≥1.11, statsmodels≥0.14, scikit-learn≥1.3, joblib≥1.3, boto3≥1.34, requests≥2.31 | numpy≥1.20, pandas≥1.3, scipy≥1.7, scikit-learn≥1.2, matplotlib≥3.5, jinja2, lightgbm≥4.0 |
| `requirements.txt` | same set, looser floors (numpy≥1.20, pandas≥1.3, scipy≥1.7, sklearn≥1.0, statsmodels≥0.13) | same set as its `environment.yml` |

The run side's `requirements.txt` and `environment.yml` disagree with each other on every
floor. The `environment.yml` is the one the cluster uses (`run_bop_job.sh:  CONDA_ENV=bop`,
`module load mamba/latest`), so `requirements.txt` is already vestigial on the run side.

### 1.2 What the analysis side actually adds

Measured from the installed `bop-analyze-remote` env
(`/Users/sjpruitt/anaconda3/envs/bop-analyze-remote`, 494 MB, python 3.12, 17 conda
packages + 38 pip distributions). Sizes are `du -sm` of the installed package directory.

**Analysis-only closure — 10 packages, 81 MB:**

| package | MB | pulled in by |
|---|---:|---|
| matplotlib | 30 | figures |
| fonttools | 19 | matplotlib |
| pillow (`PIL`) | 15 | matplotlib |
| lightgbm | 8 | `fit_models.py`, `interpret_lightgbm.py` |
| contourpy | 2 | matplotlib |
| pyparsing | 2 | matplotlib |
| jinja2 | 2 | **nothing — see below** |
| cycler | 1 | matplotlib |
| kiwisolver | 1 | matplotlib |
| markupsafe | 1 | jinja2 |
| **total** | **81** | |

For scale, the numeric core the run side *already* installs, measured in the same env:
scipy 98 MB, pandas 71 MB, sklearn 46 MB, numpy 33 MB, joblib 3 MB — **251 MB**, before
statsmodels/boto3/requests, which are not in this env. So the analysis set is **+81 MB on
a ≥251 MB base, roughly +32% of the numeric core** and ~16% of a full env.

Everything else the two sides need is shared: `python_dateutil`, `six`, `packaging`,
`threadpoolctl`, `joblib` all arrive transitively via pandas/sklearn either way.

**`jinja2` is dead weight.** It is declared in both analyze env files and imported by no
script in the repo. Its only plausible purpose is `DataFrame.Styler.to_latex`, and all four
`to_latex` call sites (`analyze.py:310,326`, `compute_pricing_errors.py:271,320`) are plain
`DataFrame.to_latex`, which has no jinja2 dependency. Drop it and the added closure falls to
8 packages / 78 MB.

### 1.3 Two things declared nowhere

- **`torch`** — imported at `MLP Analyze/common.py` (`import torch`, `import torch.nn as nn`)
  and therefore required by `MLP Analyze/Full run/run.py` and `MLP Analyze/GS21 low vol/run.py`.
  It appears in neither `requirements.txt` nor `environment.yml`, and **it is not installed
  in any conda env on this laptop**, so that tier has never run in a declared environment
  here. On Linux the default wheel pulls the CUDA runtime and is the single largest thing
  either repo could ever install — it dwarfs the entire 81 MB above. This is the one package
  that genuinely justifies keeping the cluster env separate.
- **An IPython stack** (~51 MB: ipython, jedi 30 MB, pygments, prompt_toolkit, traitlets,
  and friends) is installed in the analyze env and declared in neither file. Harmless, but it
  means the env on disk is not reproducible from its own spec.

### 1.4 Version conflicts

**There are none, structurally.** Every pin on both sides is a lower bound; there is not a
single upper bound or `!=` in either repo. The union therefore resolves to
`max(floor_run, floor_analyze)` for each shared package, which is just the run side's floors.
A unified solve cannot fail on the declared constraints.

The risk that does exist is not in the pins but in the *drift* they permit. The analyze env
has already floated up to **pandas 3.0.2 and numpy 2.4.4** under `pandas>=1.3, numpy>=1.20`.
A unified env would put the run pipeline on those versions. I grepped the run path for the
usual pandas-3/numpy-2 removals and found **no blockers**: no `np.float`/`np.int`/`np.bool`
aliases, no `applymap`, no `iteritems`, no `.ix`, no `DataFrame.append`. There is heavy use
of `inplace=True` (11 sites in `utils_factors/`, `utils_bgn/`, `utils_kp14/`), which still
works in pandas 3 but is deprecated-and-noisy on some paths.

> **Not tested, deliberately.** I did not import anything — the box is saturated and the run
> env does not exist on this laptop to import into.
>
> **This matters only under the unified-env option, which §5 rejects.** Keeping two env files
> leaves `environment.yml` (run) untouched, so the merge introduces no pandas-3/numpy-2
> exposure on the run path at all — that drift is a pre-existing property of the *analyze*
> env. Under the recommendation, **there is no blocking test and no open item here.** If a
> unified env is ever adopted instead, the check is: build it once on the cluster, run
> `tests/` plus one single-panel `main.py` against the Phase-0 fixtures.

---

## 2. Does the bloat concern survive contact with the numbers?

No, not in the form stated — but a weaker version of it survives.

**Nothing in the run path imports anything in the analysis set.** I swept every run-path
module (`main.py`, `config.py`, `utils/`, `utils_bgn/`, `utils_factors/`, `utils_gs21/`,
`utils_kp14/`) and all of `variants/` for `matplotlib`, `lightgbm`, `jinja2`, `torch`,
`seaborn`, `plotly`, `jupyter`. **Zero hits in both sweeps.** The run path's entire
third-party surface is scipy, numpy, pandas, joblib, sklearn (one site), boto3, requests.

That settles the mechanism. Python only pays for what it imports; an unimported package is
inodes on disk, not work at startup. `main.py` runs seven steps as separate subprocesses,
each re-importing `config` — so import cost is paid seven times per panel and is worth caring
about — but that cost is a function of `numpy+scipy+pandas+sklearn+joblib`, which is
identical in both envs. WORKING.md §5 measured this directly (0.01 s bare interpreter
startup in a 494 MB env vs a 713 MB env; 0.95 s for the run pipeline's import set either
way). I did not re-run that measurement; nothing I found contradicts it, and the import
sweep above is the structural reason it came out that way.

**What does survive:**

1. **`torch`.** Not an import-cost argument — a "don't put a multi-GB CUDA wheel in the
   cluster env" argument. Real, and sufficient on its own to keep two env files.
2. **Cluster disk quota and env solve time.** 81 MB (or ~2 GB with torch) per env, times
   however many envs exist on Sol/Phoenix. Mamba solve time also grows with the spec.
3. **A wrong-env failure mode that is worse than bloat.** `run_bop_job.sh` already carries a
   scar from exactly this class of bug — the `export PATH="$CONDA_PREFIX/bin:$PATH"` fix,
   with the comment recording all 11 array tasks dying in ~1 s on Phoenix 2026-08-31. Two
   envs with clearly different names is a *defence* against that; one env with everything in
   it is also a defence. What is dangerous is two envs whose names are easy to confuse.

**Conclusion: the bloat concern as Seth phrased it — "module bloat slowing cluster compute" —
is measurably false.** Keep two envs anyway, for reasons 1 and 2, which are about disk and
provisioning rather than runtime.

---

## 3. Merge mechanics

### 3.1 Verified by actually doing it

Run in a throwaway clone under `_scratch/` (since deleted; ~20 s to reproduce):

```bash
git clone --no-hardlinks bop-run-upload run && cd run
git remote add analyze ../../bop-analyze-remote
git fetch analyze main
git subtree add --prefix=analyze analyze main
```

**Result: clean.** `Added dir 'analyze'`, `git status --porcelain` empty, no conflicts.

| measure | before | after |
|---|---:|---:|
| `.git` size | 68 M | **72 M** (+5.9%) |
| commits reachable from HEAD | 166 | **188** (= 166 + 21 + 1 merge) |

`git merge-base --is-ancestor FETCH_HEAD HEAD` → **true**: all 21 analyze commits are
genuine ancestors, not a squash.

**Prefix: `analyze/`.** It matches PLAN.md §3's directory layout, which already reserves
`analyze/  # subtree graft, Phase 4`.

> Do **not** run `git gc --aggressive` after the graft. I did, and `.git` went *up*, 72 M →
> 80 M, because the fresh clone's packing was already better than what aggressive repacking
> chose. Ordinary `git gc` or nothing at all is correct.

### 3.2 Corrections to the numbers PLAN.md §9 cites as evidence

PLAN.md §9 stages these figures as the mechanical evidence for the conversation with Kerry
("21 commits preserved, 4.4% growth, blame intact, four prefix-fixed collisions, one
rename"). Three need adjusting before they are presented:

| §9 / WORKING.md §5 | measured here |
|---|---|
| `.git` grows **4.4%** (68 M → 71 M) | **+5.9%** (68 M → 72 M) |
| `config`: **42** run sites vs 11 analyze | **41** vs 11 |
| "grafts all 21 commits … preserves blame" | 21 commits reachable ✓ and blame ✓ — but see below, path-limited `git log` does **not** survive |

None of these weakens the case. The growth figure is still trivial and the collision count and
rename are exactly as stated.

### 3.2a The history claim needs one qualifier

Both §9 and WORKING.md §5 say the graft "grafts all 21 analyze commits" and that
"`git blame` reaches through to the original 2026-03-25 commits". The blame half is
**confirmed** — blaming `analyze/analyze.py`
after the graft returns `7ccc0c8` (2026-03-25, "Initial commit"), `4b482f6` (2026-04-03),
`dc0d671` (2026-06-11), i.e. real original authorship.

But path-limited log does **not** work, and it is worth knowing before someone relies on it:

```
git log --oneline -- analyze/                    →  1 commit  (the graft itself)
git log --oneline --full-history -- analyze/     →  1 commit
git log --oneline --follow -- analyze/analyze.py →  0 commits
```

The historical commits touched `analyze.py` at the *root*, with no `analyze/` prefix, so
path-limited traversal cannot see them. The objects are all there and reachable; only the
`git log <path>` view is lost. `git blame`, `git log <sha>`, and `git show` all work.
`git subtree add` does not rewrite history to add the prefix — that would need
`--rejoin`-style rewriting or `filter-repo`, and it is not worth it.

### 3.3 Path collisions

Exactly **four** tracked-path collisions between the two roots:

```
.gitignore   config.py   environment.yml   requirements.txt
```

All four are resolved by the prefix — after the graft they are `analyze/.gitignore`,
`analyze/config.py`, etc., and the run repo's own copies are untouched. I checked the
untracked/generated dirs the brief asked about as well:

- `__pycache__/` — ignored by both repos' `.gitignore`; the analyze one lands as
  `analyze/.gitignore` and **still applies**, since gitignore is directory-scoped. No action.
- `figures/`, `tables/`, `slides/` — exist only on the analyze side, become
  `analyze/figures/` etc. The run repo has `outputs/`, `logs/`, `outslurm/`, `archive/`,
  `docs/`, `tests/`, `experiments/`, `variants/` — **no overlap at all**. Note the run
  `.gitignore` has `# figures/` and `# tables/` *commented out*, so `analyze/tables/*.tex`
  and `analyze/figures/*.pdf` would become trackable where today they are untracked. Today
  only the compiled `all_results.pdf` is tracked. Decide deliberately; I'd leave them
  untracked and let the summary layer (§4.4) be the durable artifact.
- **Per-machine symlinks** `wts_prediction` → Dropbox and `results_dir`. Untracked, ignored
  by `analyze/.gitignore`, and **must be recreated at `analyze/wts_prediction` after the
  graft** or `config.py` silently falls back to `/scratch/sjpruitt/bop`. One-line migration
  step, easy to forget, fails quietly.
- The 12 tracked `.pt` files and 12 `.png` under `MLP Analyze/` are 2.8 MiB total — the whole
  analyze tree is 2.83 MiB tracked. That is where the +4 M of `.git` comes from. Fine.

### 3.4 The one real semantic collision: the module name `config`

Both repos have a top-level `config.py`, and they are unrelated modules:

- run-side `config.py` — 17.8 KB, ~97 globals, `init_from_env()`, `get_model_config()`
- analyze-side `config.py` — 1.5 KB, exports exactly `RESULTS_DIR` and `WTS_DIR`

**41 files** in the run repo do `import config` / `from config import ...`; **11 sites** in
the analyze repo do `from config import RESULTS_DIR|WTS_DIR`. Today script-directory
resolution masks the clash — running `python analyze.py` from the analyze root puts that
root first on `sys.path`. After the graft it stops being masked, because the run repo
actively manipulates `sys.path`:

```
main.py:70-71                sys.path.insert(0, <repo_root>); sys.path.insert(0, <repo_root>/utils)
utils/evaluate_sdfs.py:21-22 sys.path.insert(0, <utils>); sys.path.insert(0, <repo_root>)
```

So `python -m`, pytest collection over a merged tree, a bare REPL at the root, or anything
that has imported `main.py` will resolve `config` to the *run* one, and `from config import
RESULTS_DIR` then raises `ImportError` — or worse, silently picks up a run-side name that
happens to exist.

**Fix: rename `analyze/config.py` → `analyze/paths.py`**, 11 one-line edits. This is what
PLAN.md Phase 4 already prescribes. It is mechanical and it should happen *in the same
commit as the graft*, because between the two the tree is broken.

---

## 4. The interface that matters

This is the contract a merged repo has to preserve. It is currently **undocumented, implicit
in filenames and dict keys, and already broken.**

### 4.1 `analyze.py` — consumes

- **Discovery:** `glob(f"{RESULTS_DIR}/{model}_*_results.pkl")` for `model` in a hardcoded
  `['bgn','kp14','gs21']` (`analyze.py:65,565`). Iteration index parsed out of `panel_id` by
  `int(panel_id.split('_')[-1])`.
- **`RESULTS_DIR`** resolved by `analyze/config.py`: env `BOP_RESULTS_DIR` → repo-root
  symlink `results_dir` → fallback `/scratch/sjpruitt/bop`. (`WTS_DIR` likewise, via
  `BOP_WTS_DIR` → `wts_prediction` → `RESULTS_DIR`.)
- **Payload** — the 12-key dict written by `utils/evaluate_sdfs.py:230-244`:
  `fama_results`, `dkkm_results`, `returns`, `panel_id`, `model`, `chars`, `nfeatures_lst`,
  `alpha_lst`, `alpha_lst_fama`, `nmat`, `start`, `end`. `analyze.py` reads only the first
  five, and merges `fama`/`dkkm` onto `returns` on `month` to get `sdf_ret`.

### 4.2 `analyze.py` — emits

`tables/{model}_fama.tex`, `tables/{model}_dkkm_sharpe.tex`, `tables/{model}_dkkm_hjd.tex`,
`figures/{model}_fama_sharpe_boxplot.pdf`, `figures/{model}_dkkm_sharpe_boxplot.pdf`, then
`all_results.tex` → `pdflatex` ×2 → `all_results.pdf`. **Only the PDF is tracked**; the
`.tex` and `.pdf` components are not. Metrics computed: `sharpe = mean/stdev`,
`hjd = sqrt(mean((xret - sdf_ret)^2))`, aggregated `groupby(method|nfeatures, alpha, iter)`
then `groupby(method|nfeatures, alpha)` for a cross-panel mean and std.

### 4.3 The contract is already broken — and it is changing a headline number

Run commit **`dc48c98`** (2026-09-01, Alexander Ober, "Four fixes: … per-draw DKKM …") added
a **`mat`** column to `dkkm_results` and rekeyed DKKM weights from `(nfeatures, alpha)` to
`(nfeatures, alpha, mat)` — `utils/estimate_sdf_dkkm.py:159`,
`utils/evaluate_sdfs.py:197-208`. `analyze.py` has never been updated; the string `mat` does
not appear in it.

**It does not crash. It silently mis-aggregates**, and the direction is knowable:

- `dkkm_results` now has `nmat` rows per `(month, nfeatures, alpha)` instead of 1
  (`nmat: 5` in the spec schema).
- `analyze.py:215` forms `sharpe = mean/stdev` **per row, before** the groupby, then
  `groupby(['alpha','nfeatures','iter']).mean()` at line 233 averages it.
- So the reported DKKM Sharpe is now the **mean of per-draw Sharpes**, not the Sharpe of the
  draw-averaged portfolio. Individual random-feature draws are noisier than their average, so
  this is **biased downward — against DKKM, the treatment arm** in the paper's central
  comparison. Magnitude unmeasured; I did not run anything.
- `evaluate_sdfs.py:9` says in its own docstring: *"one row per random-feature draw (mat);
  average across mat downstream."* Downstream does not.

**And the intended fix is not available from `results.pkl` as written.** Averaging weights
across draws gives a portfolio whose `mean` and `xret` *are* recoverable — both are linear in
`w`. Its `stdev` is **not**: `sqrt(w̄'Σw̄) ≠ mean_i sqrt(w_i'Σw_i)`, and `Σ` is not in
`results.pkl`. So "Sharpe of the draw-averaged DKKM portfolio" cannot be computed downstream
at all — the run side has to emit it.

That is a concrete requirement on the merged repo, and it is exactly what PLAN.md Phase 3's
`draw_pooling` spec field and `contract_version: 3` are for. **This finding is the strongest
argument for the merge**: a cross-repo, filename-and-dict-key contract with no version and no
CI produced a silent nine-month-old wrong number in four days, and nothing anywhere flagged it.

### 4.4 `build_sdfwts_chars.py` — the piece that actually needs the cluster

- **Consumes:** `{RESULTS_DIR}/{model}_{run}_moments.pkl` (≈2.9 GB each) **and**
  `{model}_{run}_panel.pkl`. Runs are discovered by regex
  `^{model}_(\d+)_(panel|moments)\.pkl$` and **intersected** — a run needs both.
- **Computes:** `np.linalg.solve(cond_var, rp)` per month key, NaN row on `LinAlgError`.
- **Emits:** `{WTS_DIR}/{model}_sdfwts_chars.pkl` =
  `{'sdfwts': {run: DataFrame}, 'chars': {run: DataFrame}}`. `sdfwts` is indexed by `month`
  with **positional integer columns `0..dim-1`**, which `fit_models.py:100-105` reads back as
  `firmid` via `.stack()`. That positional-column convention is load-bearing and undocumented
  on the run side.
  - *Minor:* its module docstring says output goes to `{RESULTS_DIR}/...`; the code writes to
    `WTS_DIR` (`build_sdfwts_chars.py:127`). Docstring is wrong.
- **Second tier:** `fit_models.py` → `{WTS_DIR}/{model}_preds_{tag}.pkl`;
  `interpret_lightgbm.py` → `{WTS_DIR}/{model}_interp_{tag}/`; `MLP Analyze/` → `.pt` caches
  and PNGs.

**Its whole third-party surface is `numpy` + `pandas` + stdlib.** Nothing else. So the one
analysis script that must run on the cluster needs **nothing the run env does not already
have** — which is what makes "two envs, one repo" cost essentially zero.

Six Mar–Apr 2026 scripts (`create_*.py`, `ttest_table.py`, `compute_pricing_errors.py`,
`analyze_gs21_panels.py`) are slide-deck legacy on the same `RESULTS_DIR` contract.

### 4.5 A latent bug worth fixing with the graft

`analyze.py:65`'s `glob(f"{model}_*_results.pkl")` matches `bgn_gam_3_results.pkl` under
`model='bgn'`. The moment a variant economy and its baseline land in one results directory,
**two different economies get pooled into one table, silently**, and
`int(panel_id.split('_')[-1])` will happily parse the index from either. PLAN.md Phase 4
already calls for replacing this glob with a ledger read filtered by `spec_id`. Until that
exists, an anchored regex (as `build_sdfwts_chars.py:39` already uses) closes it.

### 4.6 How this should extend the existing design, not duplicate it

The brief's framing is right: a 10-panel run is ~80 GB on non-retained scratch, so the
durable artifact must be the summary, linked to the spec that produced it.
`variants/common/solstamp.py` already implements exactly that pattern for solves, and it
generalizes without modification:

- `Snapshot` already takes **`inputs=`** (upstream artifact digests) and **`stage=`**
  precisely so a cheap downstream stage can be re-run without invalidating an expensive
  upstream one — `solstamp.py:225-230` describes this as "mode 3: upstream moved, downstream
  did not". Summarization *is* that downstream stage.
- So: `snapshot(spec_module, sources=['analyze/summary.py'], stage='summary',
  inputs=artifact_digests([...results.pkl]))` → `record(...)` writes
  `experiments/registry/<id>.json` with the summary's own bytes and sha256 alongside the
  digests of the results files it consumed. The manifest is a few KB and outlives the 80 GB,
  which is the stated design goal in `solstamp.py`'s docstring.
- `record()` already accepts **`spec_id=`** and accumulates `spec_ids` across calls. That is
  the spec↔summary link the merge is supposed to establish; it exists today and is unused
  for anything but solves.

The only thing genuinely missing is that summaries are computed in the *other repo* by
`analyze.py`, which cannot import `variants/common/solstamp.py`. Merging removes that
obstacle. **That is the concrete technical payoff of the merge** — not tidiness.

---

## 5. Recommendation

**Merge into one repo at `analyze/`. Keep two `environment.yml` files.**

Why this and not the alternatives:

- **Unified env** — rejected, but *only just*, and not for the stated reason. The bloat
  argument is false (§2). It fails on `torch`: tracked code imports it, and a multi-GB CUDA
  wheel does not belong in a cluster env that gets provisioned per machine. Also, the
  cluster-side analysis script needs nothing beyond numpy+pandas (§4.4), so unification buys
  nothing concrete.
- **Stay split** — rejected. §4.3 is the disqualifying evidence: the contract broke silently,
  is still broken, and is currently biasing DKKM Sharpe downward in the paper's main
  comparison. A shared repo lets one CI run assert the `results.pkl` contract against its
  only consumer.
- **Two envs in one repo** — recommended. `environment.yml` (run, unchanged) and
  `analyze/environment.yml` (analysis, plus a declared `torch` and minus the unused `jinja2`).

**Cost.** ~3 days of work, matching PLAN.md Phase 4's estimate:

1. `git subtree add --prefix=analyze analyze main` — minutes, verified clean, `.git` +4 M.
2. Rename `analyze/config.py` → `analyze/paths.py`, 11 edits, **same commit** as the graft.
3. Recreate the `analyze/wts_prediction` and `analyze/results_dir` symlinks (§3.3).
4. Fix `analyze.py`'s `mat` handling and the stale docstring; decide `draw_pooling`
   explicitly. **This one is not optional and not cosmetic** — and part of it needs the run
   side to emit a draw-averaged `stdev`, which `results.pkl` cannot currently support (§4.3).
5. Anchor the `_results.pkl` glob or replace it with a `spec_id` ledger read (§4.5).
6. Add `torch` to `analyze/environment.yml`; drop `jinja2`.

Items 1–3 are mechanical and safe. Item 4 is the actual work and is worth doing regardless of
whether the repos merge.

**Not on the critical path.** Nothing in Phase 0/1/2 is blocked by this. But item 4 *is*
urgent independent of the merge, because it is wrong right now.

**⚠ The blocker is governance, not technical — Seth's call.** `bop-run-upload` is owned by
**kerryback** with Seth at WRITE; 85 of 162 commits are Kerry Back's. `bop-analyze-remote` is
Seth's own repo at ADMIN. Merging pushes Seth's analysis code, including the `MLP Analyze/`
tier and 2.8 MiB of tracked model caches and PNGs, into a co-author's repository. **Ask
first.** (Carried forward from WORKING.md §5; I did not re-verify the ownership via the API.)

---

## 6. Consequences for the plan

Job 5 itself is unblocked and off the critical path. The findings that *do* change the plan
land elsewhere.

### 6.1 Open decision #5 (`draw_pooling`) is harder than §10.5 states

§10.5 reads: *"Current behavior is mean-of-ratios, undocumented. Recommendation: keep it,
state it in the spec."* Two problems, both from §4.3:

1. **"Current behavior" is ambiguous as of `dc48c98`.** Before it, mean-of-ratios meant over
   *months*. After it, over *months × draws*. These are different numbers, and the second is
   biased downward against DKKM. "Keep it" does not say which one, and the spec field cannot
   be written until it does.
2. **The alternative is not a free choice.** "Sharpe of the draw-averaged DKKM portfolio"
   is **not computable from `results.pkl`** — `mean` and `xret` are linear in `w` and
   recoverable, but `stdev` is quadratic and `Σ` is not stored. Choosing that definition
   forces `utils/evaluate_sdfs.py` to emit it. So #5 has a code branch, not just a
   documentation branch.

### 6.2 Phase 3's exit criterion is unreachable as written, for one metric

Phase 3 exits on *"`analyze.py` and `run_estimators.py` emit byte-identical summary rows for
the same input."* For DKKM Sharpe under the draw-averaged definition that cannot hold until
the run side emits a draw-averaged `stdev` (6.1.2). Either the run-side emit lands in Phase 3,
or the exit criterion carves out DKKM Sharpe explicitly.

### 6.3 The week-1 ask to Kerry is stronger than §9 frames it

§9 stages `dc48c98` as the concrete argument but describes `analyze.py` as surviving "only
because `mat` gets absorbed into a `groupby` mean" — which reads as a near-miss. It is not a
near-miss. It is live, and it is currently biasing the paper's treatment arm downward. That
is a materially better thing to walk in with, and it costs nothing to state accurately.

### 6.4 Item 4 of the §5 cost list is not merge-contingent

Fixing the `mat` handling is wrong *now*, in the split repos. It should not wait on the
governance conversation, and it is not a reason to rush the graft.

---

## Appendix — what was measured vs. inherited

**Measured here:** all env package inventories and sizes; the run-path import sweep; the
`git subtree add` trial and its `.git` growth, commit counts, ancestry, blame, and the
`git log <path>` limitation; the four tracked-path collisions; 41 vs 11 `config` import
sites; the `sys.path` insertion sites; the `mat` divergence and its aggregation consequence;
`jinja2` being unimported; `torch` being undeclared and uninstalled; the pandas-3/numpy-2
hazard grep.

**Inherited from WORKING.md §5, not re-verified:** the interpreter-startup timings
(0.01 s / 0.95 s), the repo-ancestry claim that these were one repo before the 2026-01-29
split, and the GitHub ownership/permission facts.

**Not tested, on purpose:** no imports, no env build, no solve/oracle/estimator. Under the
two-env recommendation this blocks nothing — the run `environment.yml` does not change. The
cluster check in §1.4 is required only if a unified env is adopted against this advice.
