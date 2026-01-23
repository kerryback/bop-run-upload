# noipca2 — Root-Faithful Reimplementation Plan

## Goal
Create `noipca2/` with the same file structure as `noipca/` but with code matching the root codebase (`c:\Users\kerry\repos\CodeNew\*.py`) as closely as possible. Extensive comments map each function/section to its root counterpart. No Numba.

---

## Critical Differences Between Root and noipca (Source of Result Divergence)

### 1. `rank_standardize` formula
- **Root** (`dkkm_functions.py:10-13`): Uses pandas `.rank(axis=0)`, formula `(ranks - 0.5) / len(ranks) - 0.5`
- **noipca** (`factor_utils.py:276-311`): Uses scipy `rankdata`, formula `(ranks - 1) / (N - 1) - 0.5`
- **Impact**: Different numerical values! Root maps to ~[-0.4995, 0.4995] for N=1000; noipca maps to exactly [-0.5, 0.5]

### 2. `rff()` — DataFrame vs numpy
- **Root** (`dkkm_functions.py:16-25`): Operates entirely on DataFrames. `W @ X.T` where X is a DataFrame preserves DataFrame structure through pandas `__rmatmul__`.
- **noipca** (`dkkm_functions.py:37-92`): Converts to numpy immediately, does all computation in numpy, wraps back.
- **Impact**: Subtle numerical differences from DataFrame alignment/indexing.

### 3. Max-features subsetting vs independent runs
- **Root** (`main_revised.py:133-141, 211-228`): Generates ONE W matrix of `(max_features/2, nchars)`. For each `nfeatures`, takes first `num=nfeatures/2` rows of W for loadings, and subsets factor return columns with `nf_indx`.
- **noipca** (`run_dkkm.py`): For each `nfeatures`, generates a SEPARATE W matrix of `(nfeatures/2, nchars)`.
- **Impact**: Different random draws across feature counts. Root ensures nfeatures=6 uses a SUBSET of nfeatures=3600's random features.

### 4. Penalty scaling in `mve_data()`
- **Root** (`main_revised.py:228`): Caller passes `nfeatures * alpha` to `mve_data()`. Inside `ridge_regr`, penalty is `360 * z`. So effective = `360 * nfeatures * alpha`.
- **noipca** (`dkkm_functions.py:240`): `mve_data()` internally scales `nfeatures * alpha_lst`, then `ridge_regression_grid` applies `360*z`. Same effective penalty.
- **Impact**: Equivalent scaling, but root passes pre-scaled alpha from outside while noipca scales inside.

### 5. `include_market` default
- **Root** (`main_revised.py:11`): `include_market = False`
- **noipca** (`config.py:88`): `INCLUDE_MKT = True`
- **noipca2**: Keep `INCLUDE_MKT = True`. Root code DOES support this path (lines 226, 228, 174-184 of main_revised.py and mve_data augmentation logic).
- **Impact**: When True, adds mkt_rf column to factor_weights and passes ff_rets market return to mve_data (which augments to avoid penalizing market).

### 6. HJD computation
- **Root** (`main_revised.py:257-258`): `hjd = errs @ second_moment_inv @ errs` (squared HJD)
- **noipca** (`dkkm_functions.py:375`): `hjd = np.sqrt(errs @ second_moment_inv @ errs)` (HJD)
- **Impact**: Root stores HJD², noipca stores HJD.

### 7. `alpha_lst` grid
- **Root** (`main_revised.py:16`): `[0, 0.01, 0.05, 0.1, 1]`
- **noipca** (`config.py:95`): `[0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]`
- **noipca2**: Use root grid `[0, 0.01, 0.05, 0.1, 1]`

---

## File Structure

```
noipca2/
├── PLAN.md                        # This file
├── config.py                      # Configuration (matches root parameters)
├── main.py                        # Workflow orchestrator
├── analyze.py                     # Results analysis (copy from noipca, adapt paths)
├── deploy_koyeb.sh                # Deployment (copy from noipca)
├── utils/
│   ├── __init__.py
│   ├── generate_panel.py          # Copy from noipca (panel generation orchestrator)
│   ├── calculate_moments.py       # Copy from noipca (SDF moments)
│   ├── run_fama.py                # Rewritten to match root fama logic
│   ├── run_dkkm.py                # Rewritten: max_features approach, subsetting
│   └── upload_to_aws.py           # Copy from noipca
├── utils_factors/
│   ├── __init__.py
│   ├── dkkm_functions.py          # Near-copy of root dkkm_functions.py
│   ├── fama_functions.py          # Near-copy of root fama_functions.py
│   ├── factor_utils.py            # Shared utilities (rank_standardize matches root)
│   └── sdf_utils.py               # Copy from noipca (moment loading)
├── utils_bgn/                     # Copy from noipca
├── utils_kp14/                    # Copy from noipca
└── utils_gs21/                    # Copy from noipca
```

**Removed from noipca:**
- `utils_factors/ridge_utils.py` — replaced by inline `ridge_regr()` inside dkkm_functions.py
- `utils_factors/dkkm_functions_numba.py` — no Numba
- `utils_factors/factor_utils_numba.py` — no Numba
- `utils_factors/portfolio_stats.py` — stats computed inline in run_dkkm.py
- `utils_factors/ipca_functions.py` — not needed

---

## File-by-File Mapping

### `utils_factors/dkkm_functions.py` ← Root `dkkm_functions.py`

| Function | Root Location | Notes |
|----------|--------------|-------|
| `rank_standardize(arr)` | lines 10-13 | Pandas `.rank(axis=0)`, formula `(ranks-0.5)/len(ranks)-0.5` |
| `rff(data, rf, W, model)` | lines 16-25 | DataFrame-based, returns `(rank_standardized, raw)` |
| `ridge_regr(signals, labels, future_signals, shrinkage_list)` | lines 118-157 | Eigendecomposition ridge, penalty `360*z` |
| `factors(panel, W, n_jobs, start, end, model, chars)` | lines 46-70 | Parallel factor returns |
| `mve_data(f, month, alpha_lst, mkt_rf=None)` | lines 161-194 | Grid ridge, augmentation for market |

### `utils_factors/fama_functions.py` ← Root `fama_functions.py`

| Function | Root Location | Notes |
|----------|--------------|-------|
| `fama_french(data, chars, **kwargs)` | lines 9-71 | 2x3 sorts, VW, CMA flip |
| `fama_macbeth(data, chars, **kwargs)` | lines 74-97 | No stdz, EW market |
| `factors(method, panel, n_jobs, start, end, chars)` | lines 103-121 | Parallel returns |
| `mve_data(f, month, alpha)` | lines 124-129 | sklearn Ridge, `alpha=360*alpha` |

### `utils/run_dkkm.py` ← Root `main_revised.py` lines 133-258

Key logic:
1. Load panel, prepare data
2. Generate ONE W matrix: shape `(max_features/2, nchars + (model=='bgn'))` with gamma scaling
3. Compute factor returns for ALL max_features: `f_rs, f_nors = dkkm.factors(panel, W, n_jobs, start, end, model, chars)`
4. For each nfeatures in N_DKKM_FEATURES_LIST:
   - `num = nfeatures // 2; half = max_features // 2`
   - `nf_indx = [0..num-1, half..half+num-1]`
   - For each month in evaluation range:
     - `factor_weights = dkkm.rff(data[chars], rf, W[:num,:], model)[0]`
     - Rename columns to match nf_indx
     - If include_mkt: `factor_weights['mkt_rf'] = fama.fama_french(data[chars], mve=data.mve)[:, -1]`
     - For each alpha: `port_of_factors = dkkm.mve_data(f_rs.iloc[:,nf_indx], month, nfeatures*np.array([alpha]), ff_rets.iloc[:,-1] if include_mkt else None)`
     - `weights_on_stocks = factor_weights @ port_of_factors`
     - Compute: stdev, mean, xret, hjd² (no sqrt)
5. Save results

### `config.py`

```python
INCLUDE_MKT = True             # Include market in DKKM (keep noipca behavior)
NMAT = 1                       # Root: nmat = 1
ALPHA_LST = [0, 0.01, 0.05, 0.1, 1]  # Match root: alpha_lst
ALPHA_LST_FAMA = [0]           # Root: alpha_lst_fama
N_DKKM_FEATURES_LIST = [6, 36, 360, 3600]  # Root: nfeatures_lst
GAMMA_GRID = np.arange(0.5, 1.1, 0.1)     # Root: gamma_grid
DKKM_RANK_STANDARDIZE = True   # Root uses f_rs (rank-standardized)
```

---

## Implementation Steps

1. Create directory structure and `__init__.py` files
2. Copy unchanged files from noipca (panel generation, moments, upload, sdf_utils, model-specific utils)
3. Write `config.py` with root-matching parameters
4. Write `utils_factors/dkkm_functions.py` — near-copy of root with extensive comments
5. Write `utils_factors/fama_functions.py` — near-copy of root with extensive comments
6. Write `utils_factors/factor_utils.py` — shared utilities with root rank_standardize
7. Write `utils/run_dkkm.py` — max_features approach matching root logic
8. Write `utils/run_fama.py` — matching root fama evaluation
9. Write `main.py` — orchestrator
10. Write `analyze.py` — adapted from noipca

---

## Verification

1. Generate a panel: `python main.py kp14 0 1`
2. Compare DKKM results against root by running root's `main_revised.py` with the same panel/seed
3. Check that rank_standardize output matches between root and noipca2
4. Check that rff() output matches
5. Check that mve_data() output matches for same inputs
6. Compare final portfolio stats (stdev, mean, xret, hjd²)
