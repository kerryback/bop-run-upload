# Plan: Implement NMAT > 1 Support in noipca2

## Overview

Currently noipca2 generates a single random W matrix (NMAT=1). With NMAT > 1:
1. Generate NMAT different random W matrices
2. Compute factor returns for each W
3. Run ridge regression for each W (in `mve_data`)
4. Average the ridge coefficient estimates across all NMAT iterations
5. Use averaged estimates for portfolio statistics

## Current Flow (NMAT=1)

```
run_dkkm.py:
  W = random(max_features/2, nchars)
  frets = dkkm.factors(panel, W, ...)  # Returns rank-standardized only
  Save: W, frets

run_portfolio_stats.py (per month):
  factor_weights = dkkm.rff(data, rf, W[:num,:], model)  # Returns rank-standardized only
  factor_weights['mkt_rf'] = fama.fama_french(...)       # Always include market
  mkt_rf = ff_rets.iloc[:, -1]                           # Market returns
  port_of_factors = dkkm.mve_data(frets, month, alpha_lst, mkt_rf)  # Ridge regression
  weights_on_stocks = factor_weights @ port_of_factors
  Compute: stdev, mean, xret
```

## Proposed Flow (NMAT > 1)

```
run_dkkm.py:
  W_list = []
  frets_list = []
  for mat_idx in range(NMAT):
    W_i = random(max_features/2, nchars)
    frets_i = dkkm.factors(panel, W_i, ...)  # Returns rank-standardized only
    W_list.append(W_i)
    frets_list.append(frets_i)
  Save: W_list, frets_list

run_portfolio_stats.py (per month):
  factor_weights_list = []
  port_of_factors_list = []
  mkt_rf = ff_rets.iloc[:, -1]  # Market returns (always included)

  for mat_idx in range(NMAT):
    fw_i = dkkm.rff(data, rf, W_list[mat_idx][:num,:], model)
    fw_i['mkt_rf'] = fama.fama_french(...)  # Always add market
    pof_i = dkkm.mve_data(frets_list[mat_idx], month, alpha_lst, mkt_rf)
    factor_weights_list.append(fw_i)
    port_of_factors_list.append(pof_i)

  # Average across NMAT iterations
  factor_weights_avg = mean(factor_weights_list)
  port_of_factors_avg = mean(port_of_factors_list)

  weights_on_stocks = factor_weights_avg @ port_of_factors_avg
  Compute: stdev, mean, xret
```

## Files to Modify

### 1. config.py
- `NMAT = 1` is already configurable
- `NMAT` is already in model config dict
- Note: `INCLUDE_MKT` and `DKKM_RANK_STANDARDIZE` have been removed (always True)

### 2. run_dkkm.py
**Changes:**
- Loop NMAT times to generate W matrices and factor returns
- Save list of W matrices and factor returns (instead of single)

```python
# Current:
W = np.random.normal(size=(half, nchars + (MODEL == 'bgn')))
W = gamma * W
frets = dkkm.factors(panel, W, ...)  # Returns rank-standardized only

# New:
W_list = []
frets_list = []
for mat_idx in range(NMAT):
    W_i = np.random.normal(size=(half, nchars + (MODEL == 'bgn')))
    gamma_i = np.random.choice(GAMMA_GRID, size=(half, 1))
    W_i = gamma_i * W_i

    frets_i = dkkm.factors(panel, W_i, ...)  # Returns rank-standardized only
    W_list.append(W_i)
    frets_list.append(frets_i)

# Save
results = {
    'dkkm_factors': frets_list,  # List of DataFrames
    'weights': W_list,           # List of arrays
    'nmat': NMAT,
    ...
}
```

### 3. run_portfolio_stats.py
**Changes:**
- Load list of W matrices and factor returns
- Loop over NMAT for each nfeatures/alpha combination
- Average the results before computing portfolio stats
- Always include market (no include_mkt conditional)

```python
def compute_month_stats(..., W_list, frets_list, NMAT, ...):
    """Compute stats with NMAT averaging."""

    mkt_rf = ff_rets.iloc[:, -1]  # Market returns (always needed)

    for nfeatures in nfeatures_lst:
        num = int(nfeatures / 2)
        nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])

        # Accumulate across NMAT iterations
        factor_weights_sum = None
        port_of_factors_sum = None

        for mat_idx in range(NMAT):
            W_i = W_list[mat_idx]
            frets_i = frets_list[mat_idx]

            # Factor weights from rff (returns rank-standardized only)
            fw_i = dkkm.rff(data_chars_df, rf, W=W_i[:num, :], model=MODEL)
            fw_i.columns = [str(ind) for ind in nf_indx]

            # Always add market
            fw_i['mkt_rf'] = fama.fama_french(data_chars_df, CHARS, mve=data_mve)[:, -1]

            # Portfolio of factors from ridge regression
            f_subset_i = frets_i.iloc[:, nf_indx]
            pof_i = dkkm.mve_data(f_subset_i, month, scaled_alphas, mkt_rf)

            # Accumulate
            if factor_weights_sum is None:
                factor_weights_sum = fw_i.values.copy()
                port_of_factors_sum = pof_i.values.copy()
            else:
                factor_weights_sum += fw_i.values
                port_of_factors_sum += pof_i.values

        # Average
        factor_weights_avg = factor_weights_sum / NMAT
        port_of_factors_avg = port_of_factors_sum / NMAT

        # Compute stats for each alpha using averaged weights
        for i, alpha in enumerate(alpha_lst):
            pof_alpha = port_of_factors_avg[:, i]
            weights_on_stocks = factor_weights_avg @ pof_alpha

            stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
            mean = weights_on_stocks @ rp
            xret = weights_on_stocks @ data_xret

            dkkm_results.append({...})
```

### 4. analyze.py
- No changes needed (already aggregates across panels/months)
- The `matrix` field in dkkm_stats can be removed or kept as 0
- Note: `include_mkt` field no longer in dkkm_stats (always True)

## Data Structure Changes

### run_dkkm.py output (`{panel_id}_dkkm.pkl`)

**Current:**
```python
{
    'dkkm_factors': frets,          # DataFrame
    'weights': W,                   # ndarray
    ...
}
```

**New:**
```python
{
    'dkkm_factors': frets_list,     # List[DataFrame] of length NMAT
    'weights': W_list,              # List[ndarray] of length NMAT
    'nmat': NMAT,
    ...
}
```

### Backward Compatibility

For NMAT=1, the code should work identically to before:
- `W_list = [W]`
- `frets_list = [frets]`
- Averaging a single item returns that item

## Memory Considerations

With NMAT > 1:
- W_list: NMAT × (max_features/2, nchars) float64 arrays
  - For max_features=360, nchars=5: 180 × 5 × 8 × NMAT = 7.2KB × NMAT
- frets_list: NMAT × (T, max_features) float32 DataFrames
  - For T=360, max_features=360: 360 × 360 × 4 × NMAT = 518KB × NMAT

For NMAT=10: ~5MB total — negligible.

## Testing Plan

1. Run with NMAT=1 and verify output matches current code
2. Run with NMAT=2 and verify:
   - Two W matrices are generated
   - Two sets of factor returns are computed
   - Averaging happens correctly
   - Output stats differ from single-W run (as expected)
3. Run with NMAT=10 for production use

## Implementation Order

1. Modify config.py (add NMAT parameter)
2. Modify run_dkkm.py (generate/save multiple W/frets)
3. Modify run_portfolio_stats.py (average across NMAT)
4. Test with NMAT=1 (should match current)
5. Test with NMAT>1
