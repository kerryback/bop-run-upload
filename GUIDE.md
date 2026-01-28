# noipca2 Guide

## Part 1: Quick Start

### Prerequisites

```
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, scikit-learn, joblib, numba, statsmodels, boto3, requests.

### Run locally (single panel)

```bash
cd noipca2
python main.py kp14          # Model kp14, index 0
python main.py bgn 0 3       # Model bgn, indices 0-2
python main.py gs21 5 6       # Model gs21, index 5 only
```

### Run on Koyeb (cloud)

```bash
cd noipca2
cp .env.example .env          # Fill in KOYEB_API_TOKEN, AWS keys
bash deploy_koyeb.sh kp14 0 1
```

### Output files

All outputs go to `noipca2/outputs/`. The final deliverable per panel is:

| File | Contents |
|------|----------|
| `{model}_{id}_results.pkl` | Fama, DKKM, and return statistics (DataFrames) |
| `{model}_{id}_25_portfolios.pkl` | 25 double-sorted portfolio returns |

Intermediate files (configurable via `KEEP_*` flags in config.py):

| File | Contents |
|------|----------|
| `{model}_{id}_panel.pkl` | Simulated panel data |
| `{model}_{id}_arr/` | Memory-mapped arrays for SDF moments |
| `{model}_{id}_fama.pkl` | Fama-French and Fama-MacBeth factor returns |
| `{model}_{id}_stock_weights.pkl` | Per-month stock weights for all methods |
| `{model}_{id}_moments.pkl` | SDF conditional moments (rp, cond_var, etc.) |

### Loading results

```python
import pickle

with open('outputs/kp14_0_results.pkl', 'rb') as f:
    results = pickle.load(f)

fama_stats = results['fama_stats']    # DataFrame: month, method, alpha, stdev, mean, xret
dkkm_stats = results['dkkm_stats']    # DataFrame: month, nfeatures, alpha, stdev, mean, xret
returns    = results['returns']        # DataFrame: month, sdf_ret, mkt_rf
```

---

## Part 2: Detailed Documentation

### Architecture

noipca2 is a modular reimplementation of a monolithic research codebase (`main_revised.py`). It simulates financial panel data under three economic models (BGN, KP14, GS21), computes factor portfolios via multiple methods (Fama-French, Fama-MacBeth, CAPM, DKKM random Fourier features), and evaluates portfolio performance using model-implied SDF moments.

The workflow is orchestrated by `main.py`, which calls six steps in sequence:

```
Step 1:  generate_panel.py         ──> _panel.pkl, _arr/
Step 1b: generate_25_portfolios.py ──> _25_portfolios.pkl
Step 2:  generate_fama_factors.py  ──> _fama.pkl
Step 3:  generate_dkkm_factors.py  ──> (in-memory dkkm_data)
Step 4:  estimate_sdfs.py          ──> _stock_weights.pkl
Step 5:  calculate_moments.py      ──> _moments.pkl
Step 6:  evaluate_sdfs.py          ──> _results.pkl
```

Steps 1, 1b, 2, 5, 6 run as subprocesses. Steps 3+4 run in-process so DKKM data (~2 GB) stays in memory and is never written to disk.

### Directory structure

```
noipca2/
├── main.py                  # Master orchestrator
├── config.py                # All parameters and model configs
├── deploy_koyeb.sh          # Koyeb cloud deployment script
├── requirements.txt
├── outputs/                 # All output files
├── logs/                    # Run logs ({model}_{start}_{end}.log)
├── utils/                   # Workflow step scripts
│   ├── generate_panel.py
│   ├── generate_25_portfolios.py
│   ├── generate_fama_factors.py
│   ├── generate_dkkm_factors.py
│   ├── estimate_sdfs.py
│   ├── calculate_moments.py
│   ├── evaluate_sdfs.py
│   ├── upload_to_aws.py
│   └── log_streamer.py
├── utils_factors/           # Shared factor computation library
│   ├── factor_utils.py      # Panel prep, argument parsing
│   ├── fama_functions.py    # Fama-French, Fama-MacBeth methods
│   ├── dkkm_functions.py    # Random Fourier Features, ridge regression
│   └── sdf_utils.py         # Load pre-computed moments
├── utils_bgn/               # BGN model: panel generation, SDF computation
├── utils_kp14/              # KP14 model: panel generation, SDF computation
└── utils_gs21/              # GS21 model: panel generation, SDF computation
```

### Configuration (config.py)

Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N` | 1000 | Number of firms per panel |
| `T` | 720 | Time periods (excluding burnin) |
| `BGN_BURNIN` / `KP14_BURNIN` / `GS21_BURNIN` | 500 | Burnin periods |
| `NMAT` | 20 | Number of random W matrices for DKKM |
| `N_DKKM_FEATURES_LIST` | [6, 36, 360, 3600, 18000] | Feature counts for DKKM |
| `MAX_FEATURES` | 18000 | Maximum features (determines W matrix size) |
| `ALPHA_LST` | [0, 0.001, 0.01, 0.05, 0.1, 1] | Ridge penalties for DKKM |
| `ALPHA_LST_FAMA` | [0] | Ridge penalties for Fama (OLS only) |
| `ALPHA_LST_GS` | [0, 1e-7, ..., 1] | Finer grid for GS21 model |
| `N_JOBS` | 24 | Parallel workers (default; jgsrc1 uses 10) |
| `GAMMA_GRID` | [0.5, 0.6, ..., 1.0] | RFF bandwidth grid |

Model characteristics:

| Model | Chars | Factor names |
|-------|-------|-------------|
| BGN, KP14 | size, bm, agr, roe, mom | smb, hml, cma, rmw, umd |
| GS21 | size, bm, agr, roe, mom, mkt_lev | smb, hml, cma, rmw, umd, mkt_lev |

### Step-by-step workflow detail

#### Step 1: generate_panel.py

Generates a simulated financial panel under the chosen economic model.

- **Input**: Model name, identifier index
- **Output**: `{model}_{id}_panel.pkl` (panel DataFrame), `{model}_{id}_arr/` (memory-mapped arrays)
- **Process**:
  1. Calls model-specific `create_arrays(N, T+burnin)` to generate raw economic state arrays
  2. Calls `create_panel(N, T+burnin, arr_tuple)` to build panel DataFrame with columns: month, firmid, mve, bm, agr, roe, mom, xret, rf_stand (BGN only), etc.
  3. Adds `size = log(mve)`
  4. Saves panel as pickle dict `{'panel': panel, 'N': N, 'T': T, ...}`
  5. Saves `arr_tuple` as individual `.npy` files for memory-mapped loading by `calculate_moments.py`

The `arr_tuple` contains all raw economic state variables (K, x, z, eps, uj, chi, rate, ...) needed by `sdf_compute` to calculate theoretical moments. These are ~36 GB total, hence the memory-mapped storage.

#### Step 1b: generate_25_portfolios.py

Forms 25 double-sorted portfolios (5x5 on market cap and book-to-market).

- **Input**: `_panel.pkl`
- **Output**: `_25_portfolios.pkl`
- **Process**: For each month, assigns firms to quintiles by mve and bm using `pd.qcut`, then computes value-weighted excess returns for each of the 25 groups.

#### Step 2: generate_fama_factors.py

Computes factor returns for Fama-French and Fama-MacBeth methods.

- **Input**: `_panel.pkl`
- **Output**: `_fama.pkl` containing `ff_returns` and `fm_returns` DataFrames
- **Process**:
  - **Fama-French**: 2x3 sorts (size median x characteristic 30/40/30 terciles), value-weighted within each of 6 portfolios, long-short factor construction. SMB defined using bm sorts. CMA sign-flipped. Market portfolio = value-weighted.
  - **Fama-MacBeth**: Cross-sectional regression via pseudo-inverse `P = X @ pinvh(X'X)`, normalized by `2 / abs(P).sum()`. No standardization of characteristics. Market portfolio = equal-weighted.
  - Both computed in parallel across months via joblib.

#### Step 3: generate_dkkm_factors.py (in-process)

Computes random Fourier feature (RFF) factor returns.

- **Input**: `_panel.pkl`
- **Output**: In-memory `dkkm_data` dict (not written to disk when called from main.py)
- **Process**:
  1. For each of `NMAT` iterations:
     - Generate random weight matrix `W` of shape `(MAX_FEATURES/2, nchars + (model=='bgn'))`, scaled by random `gamma` from `GAMMA_GRID`
     - Compute factor returns: for each month, compute `rff(data[chars], rf, W)` which applies rank standardization, then `sin(W @ X.T)` and `cos(W @ X.T)`, concatenates, and rank-standardizes again. Factor returns = `features.T @ xret`.
  2. Returns `{'dkkm_factors': frets_list, 'weights': W_list, 'nmat': NMAT, ...}`

The `compute()` function is called directly from `main.py` (not as a subprocess).

#### Step 4: estimate_sdfs.py (in-process)

Computes stock weights for every method/alpha combination for every month.

- **Input**: `_panel.pkl`, `_fama.pkl`, `dkkm_data` (from Step 3, in memory)
- **Output**: `_stock_weights.pkl`
- **Process**:
  For each month in the evaluation range (burnin+360 to end):

  **Fama weights (ff, fm, capm)**:
  1. Compute factor weights via `fama_french()` or `fama_macbeth()` for the current month
  2. Compute portfolio-of-factors via `fama.mve_data(f_rets, month, alpha)` — ridge regression on past 360 months of factor returns using sklearn Ridge with penalty `360*alpha`
  3. Stock weights = `factor_weights @ port_of_factors`
  4. CAPM: single-factor model using only market excess return

  **DKKM weights**:
  1. For each `nfeatures` in [6, 36, 360, 3600, 18000]:
     - Compute subset indices (`nf_indx`) selecting first `nfeatures/2` sin and cos features
     - Scale alphas: `scaled_alphas = nfeatures * alpha_lst`
     - For each of `NMAT` W matrices:
       - Factor weights from `dkkm.rff()` using first `nfeatures/2` rows of W
       - Portfolio-of-factors from `dkkm.mve_data()` using eigenvalue-based ridge regression with penalty `360 * nfeatures * alpha`, augmented design matrix to avoid penalizing the market column
     - Average factor weights and portfolio-of-factors across NMAT iterations
     - Stock weights = `avg_factor_weights @ avg_port_of_factors`

- **Parallelism**: Months processed in chunks of 50 via `joblib.Parallel(backend='multiprocessing')`. Large shared data (`frets_list`, `W_list`, `ff_rets`, `fm_rets`) stored in module-level `_SHARED_DATA` dict, inherited by workers via fork/copy-on-write (avoids serializing ~2 GB per worker).

#### Step 5: calculate_moments.py

Computes SDF conditional moments from the economic model.

- **Input**: `_arr/` directory (memory-mapped arrays)
- **Output**: `_moments.pkl`
- **Process**:
  1. Loads `arr_tuple` arrays via `np.load(mmap_mode='r')` — OS pages in only accessed slices (~40 MB/month)
  2. Creates `sdf_loop` function from model-specific `sdf_compute` module
  3. For each month: calls `sdf_loop(month-1, 0)` which returns `(sdf_ret, max_sr, rp, cond_var)`:
     - `rp`: (N,) expected excess returns vector
     - `cond_var`: (N, N) conditional variance matrix
     - `sdf_ret`: scalar SDF return
     - `max_sr`: scalar maximum Sharpe ratio
  4. Computes `second_moment = cond_var + rp @ rp.T` and its inverse
  5. Saves per-month moments dict
- **Parallelism**: Months in chunks of 50 via joblib. Each chunk is saved to a temporary pickle file on disk, then all chunks are consolidated. This prevents accumulating ~13+ GB of moment matrices in memory.

#### Step 6: evaluate_sdfs.py

Computes final portfolio statistics using weights and moments.

- **Input**: `_panel.pkl`, `_stock_weights.pkl`, `_moments.pkl`
- **Output**: `_results.pkl`
- **Process**:
  For each month, for each method/alpha combination:
  - `stdev = sqrt(weights @ stock_cov @ weights)` — conditional standard deviation
  - `mean = weights @ rp` — conditional expected excess return
  - `xret = weights @ data.xret` — realized excess return
- **Output DataFrames**:
  - `fama_stats`: (month, method, alpha, stdev, mean, xret) — methods: ff, fm, capm
  - `dkkm_stats`: (month, nfeatures, alpha, stdev, mean, xret)
  - `returns`: (month, sdf_ret, mkt_rf)

### Ridge regression details

Two different ridge implementations are used:

**Fama methods** (`fama_functions.mve_data`): sklearn Ridge with penalty `360*alpha`. For Fama methods, `alpha=0` (OLS). Fits on past 360 months of factor returns with target `y = ones(T)`.

**DKKM methods** (`dkkm_functions.ridge_regr`): Custom eigenvalue decomposition. Uses kernel trick when `P > T` (over-parametrized regime). Penalty is `360*z` where `z` is the passed shrinkage value. The caller passes `nfeatures*alpha`, so effective penalty = `360 * nfeatures * alpha`. Market column is protected from penalization via design matrix augmentation: `X_aug = [X; sqrt(360*alpha) * I[:-1]]`.

### Parallelism and memory management

| Step | Parallelism | Memory strategy |
|------|-------------|-----------------|
| 2 (Fama) | joblib across months | Standard (subprocess) |
| 3 (DKKM) | joblib within `dkkm.factors()` | In-process, returns dict to main.py |
| 4 (Estimate) | joblib multiprocessing backend, chunks of 50 | Fork/copy-on-write for ~2 GB shared data via `_SHARED_DATA` |
| 5 (Moments) | joblib, chunks of 50 | Memory-mapped arrays, chunk-to-disk-to-consolidate |

### Koyeb deployment

`deploy_koyeb.sh` creates a Koyeb worker service that:
1. Sends `/init-logs` to the koyeb-monitor app
2. Starts `log_streamer.py` in background (POSTs `run.log` every 60s)
3. Runs `main.py` with `--koyeb` flag (sets N_JOBS=24)
4. Uploads results to S3 incrementally as each step completes
5. Submits final logs, sends kill request

Instance type: `5xlarge` (default). S3 bucket: `bop-noipca`. Results stored at `s3://bop-noipca/koyeb-results/{workflow_id}/outputs/`.

Crash reporting: `main.py` wraps each panel's workflow in try/except. On failure, a crash report pickle containing the error, traceback, and last 100 lines of subprocess output is uploaded to S3.

### koyeb-monitor

A separate Flask app (`koyeb-monitor/app.py`) deployed on Koyeb that:
- Receives `/register`, `/init-logs`, `/submit-logs` from running jobs
- Stores logs in memory, serves them at `/logs/{app_name}`
- `/kill` endpoint is currently disabled (no-op)

---

## Part 3: Detailed Comparison to Root Codebase

### Structural differences

| Aspect | Root (`main_revised.py`) | noipca2 |
|--------|--------------------------|---------|
| Architecture | Single 460-line script, everything inline | 15+ modular scripts with shared library |
| Execution | Sequential loop: `run_panel()` → `run_month()` | 6-step pipeline, steps as subprocesses or in-process |
| Output format | CSV files appended row-by-row per month | Pickle files per step, DataFrames at end |
| Iteration | `for iter in range(startiter, numiters)` at bottom | `for idx in range(start_idx, end_idx)` in main.py |
| Config | Hardcoded at top of script + `parameters.py` | Centralized `config.py` with `get_model_config()` |
| IPCA | Included (lines 96-99, 301-332) | Not implemented |
| Latent factor model | Included (lines 72-78, 167-197) | Not implemented |
| Sorted portfolios | `sorted_portfolios.py` (inline call) | `generate_25_portfolios.py` (5x5 only) |
| Portfolio weights CSV | Written per-month per-method (port_*.csv) | Not written |
| HJD statistic | Computed for every method | Not computed |

### Parameter differences

| Parameter | Root | noipca2 | Notes |
|-----------|------|---------|-------|
| `burnin` | 300 (BGN) | 500 (all models) | noipca2 uses longer burnin |
| `nmat` | 1 | 20 | noipca2 averages over 20 random W matrices |
| `nfeatures_lst` | [6, 36, 360, 3600] | [6, 36, 360, 3600, 18000] | noipca2 adds 18000 |
| `alpha_lst` | [0, 0.01, 0.05, 0.1, 1] | [0, 0.001, 0.01, 0.05, 0.1, 1] | noipca2 adds 0.001 |
| `n_jobs` | 10 | 24 (default) / 10 (jgsrc1) | |
| `include_market` | False | Always True | noipca2 always includes market in DKKM |
| `max_features` | 3600 | 18000 | Driven by nfeatures_lst |

### Computational logic comparison

#### Panel preparation (identical)

Both perform the same steps:
```python
panel["size"] = np.log(panel.mve)
panel = panel[panel.month >= 2]
panel.replace([np.inf, -np.inf], np.nan, inplace=True)
panel.set_index(["month", "firmid"], inplace=True)
nans = panel[chars + ["mve", "xret"]].isnull().any(axis=1)
keep = nans[~nans].index
panel = panel.loc[keep]
```

This is in `factor_utils.prepare_panel()` (noipca2) vs inline in `run_panel()` (root).

#### Fama-French factor construction (identical)

Both use the same 2x3 sort methodology:
- Size: median split (big/small)
- Characteristic: 30/40/30 tercile split (low/med/high)
- Value-weighted within each of 6 portfolios
- Factor = 0.5 * (high_big + high_small - low_big - low_small)
- SMB from bm sorts: (high_small + med_small + low_small - high_big - med_big - low_big) / 3
- CMA sign-flipped
- Market = value-weighted

#### Fama-MacBeth (identical)

Both: `P = X @ pinvh(X'X)`, drop intercept, normalize by `2 / abs(P).sum()`, equal-weighted market. No standardization (commented out in root).

#### DKKM factor returns (identical)

Both: `rank_standardize(chars)` → `W @ X.T` → `sin/cos` → `rank_standardize(features)` → `features.T @ xret`.

Rank standardization formula is `(ranks - 0.5) / N - 0.5` in both (numpy argsort-based ordinal ranks).

#### DKKM ridge regression (identical)

Both use eigenvalue decomposition with kernel trick when P > T. Penalty = `360 * z`. The key formula:
```
beta = U @ diag(1 / (eigenvalues + 360*z)) @ U' @ X' @ y    (when P < T)
beta = X' @ V @ diag(1 / (eigenvalues + 360*z)) @ V' @ y    (when P >= T)
```

#### DKKM portfolio construction

**Root** (lines 206-230):
```python
for i, tpl in enumerate(dkkm_lst):       # nmat=1 iteration
    W, frets = tpl
    for nfeatures in nfeatures_lst:
        factor_weights = dkkm.rff(data[chars], rf, W[:num, :], model=model)
        factor_weights.columns = [str(ind) for ind in nf_indx]
        if include_market:                 # False by default
            factor_weights['mkt_rf'] = ...
        for alpha in alpha_lst:
            port_of_factors = dkkm.mve_data(frets[:, nf_indx], month, nfeatures*alpha, mkt_rf if include_market else None)
            weights_on_stocks = factor_weights @ port_of_factors
```

**noipca2** (estimate_sdfs.py lines 156-208):
```python
for nfeatures in nfeatures_lst:
    for mat_idx in range(NMAT):           # NMAT=20 iterations
        fw_i = pd.DataFrame(dkkm.rff(data_chars_df, rf, W[:num, :], model=MODEL), ...)
        fw_i['mkt_rf'] = fama.fama_french(...)[:, -1]   # ALWAYS included
        pof_i = dkkm.mve_data(f_subset_i, month, scaled_alphas, mkt_rf)
        factor_weights_sum += fw_i.values
        port_of_factors_sum += pof_i.values
    # Average across NMAT
    weights_on_stocks = (factor_weights_sum/NMAT) @ (port_of_factors_sum/NMAT)[:, i]
```

Key differences:
1. **NMAT averaging**: Root uses `nmat=1` (no averaging). noipca2 uses `nmat=20` and averages both factor weights and portfolio-of-factors across all W matrices before computing final weights.
2. **Market always included**: Root has `include_market=False` by default. noipca2 always appends the market portfolio column and always passes `mkt_rf` to `dkkm.mve_data()`.
3. **CAPM method**: noipca2 adds a CAPM single-factor model (market only). Root does not have this.

#### SDF moment computation

**Root** (lines 145-158):
```python
sdf_ret, max_sr, rp, cond_var = sdf_loop(month - 1)
keep_this_month = [tpl[1] for tpl in keep if tpl[0] == month]
stock_cov = cond_var[keep_this_month, :][:, keep_this_month]
rp = rp[keep_this_month]
cov_inv = linalg.pinv(stock_cov)
second_moment = stock_cov + np.outer(rp, rp)
second_moment_inv = linalg.pinv(second_moment)
```

**noipca2** (calculate_moments.py lines 57-64):
```python
sdf_ret, max_sr, rp, cond_var = sdf_loop(month - 1, 0)
rp_vec = rp.reshape(-1, 1)
second_moment = cond_var + (rp_vec @ rp_vec.T)
second_moment_inv = np.linalg.inv(second_moment)
```

Differences:
1. **Firm subsetting**: Root subsets to `keep_this_month` firms inline. noipca2 stores full N-dimensional moments, subsets later in `evaluate_sdfs.py` using `firm_ids`.
2. **Inverse method**: Root uses `scipy.linalg.pinv` (pseudo-inverse). noipca2 uses `np.linalg.inv` (exact inverse).
3. **Separation**: Root computes moments and portfolio stats in the same function. noipca2 pre-computes moments in Step 5, evaluates in Step 6.

#### Portfolio statistics (identical formulas)

Both compute:
- `stdev = sqrt(weights @ stock_cov @ weights)`
- `mean = weights @ rp`
- `xret = weights @ data.xret`

Root additionally computes HJD (Hansen-Jagannathan distance):
```python
errs = rp - second_moment @ weights_on_stocks
hjd = errs @ second_moment_inv @ errs
```
noipca2 does not compute HJD.

### Features in root but not in noipca2

1. **IPCA** (`ipca_functions.py`): Instrumented PCA factor model with 1-6 factors, using both base characteristics and RFF characteristics. Not implemented in noipca2.

2. **Latent factor model** ("taylor" and "proj" methods): Uses model-implied factor loadings (`A_1_taylor`, `A_2_proj`, etc.) from the DGP. Computes model-implied factor premia via OLS, then portfolio stats. Not implemented in noipca2.

3. **HJD statistic**: Hansen-Jagannathan distance computed for all methods. Not computed in noipca2.

4. **Per-month portfolio weight CSVs**: Root writes `port_model_{model}.csv`, `port_dkkm_{model}.csv`, `port_fama_{model}.csv`, `port_ipca_{model}.csv` with per-firm weights per month. noipca2 does not write these.

5. **Time series output**: Root builds a detailed `tseries` DataFrame with SDF returns, max Sharpe ratios, Fama factor returns, IPCA returns, and model premia. noipca2's `returns` DataFrame only has `sdf_ret` and `mkt_rf`.

### Features in noipca2 but not in root

1. **Cloud deployment**: Koyeb integration with `deploy_koyeb.sh`, S3 upload, crash reporting, log streaming.

2. **25 double-sorted portfolios**: 5x5 on mve x bm. Root has `sorted_portfolios.py` but with different sort methodology.

3. **Memory management**: Memory-mapped arrays, fork/copy-on-write parallelism, chunked moment computation with disk spillover.

4. **NMAT>1 averaging**: Averaging over 20 random W matrices for DKKM.

5. **CAPM method**: Single-factor market model evaluated alongside FF and FM.

6. **Extended feature/alpha grids**: 18000 features, alpha=0.001.
