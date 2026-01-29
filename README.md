# The Virtue of Complexity in Simple Economic Models

Code repository for Back, Ober, and Pruitt - "The Virtue of Complexity in Simple Economic Models"

*Last updated: January 29, 2026*

A computational finance research framework for estimating and evaluating Stochastic Discount Factors (SDFs) across multiple macroeconomic asset pricing models. The project compares traditional Fama-French factor methods with modern Random Fourier Features (DKKM) approaches.

## Overview

This framework:
- **Simulates** panel data from three structural asset pricing models (BGN, KP14, GS21)
- **Estimates** factor returns using Fama-French and Random Fourier Features methods
- **Computes** SDF weights via ridge regression
- **Evaluates** portfolio performance through Sharpe ratios and other statistics
- **Scales** to cloud deployment (Koyeb) with AWS S3 integration

## Models Implemented

| Model | Reference |
|-------|-----------|
| **BGN** | Berk-Green-Naik 1999 |
| **KP14** | Kogan-Papanikolaou 2014 |
| **GS21** | Gomes-Schmid 2021 |

## Directory Structure

```
├── config.py                 # Centralized configuration
├── main.py                   # Main script to run simulations
├── analyze.py                # Stand-alone script to analyze results after main.py completes
├── requirements.txt          # Python dependencies
├── deploy_koyeb.sh           # Wrapper to run main.py on Koyeb with AWS S3 storage
│
├── utils/                    # Core workflow scripts
│   ├── generate_panel.py         # Panel data generation
│   ├── generate_25_portfolios.py # Double-sorted portfolios
│   ├── generate_fama_factors.py  # Fama-French factors
│   ├── generate_dkkm_factors.py  # Random Fourier Features factors
│   ├── estimate_sdf_fama.py      # Fama/CAPM SDF estimation
│   ├── estimate_sdf_dkkm.py      # DKKM SDF estimation
│   ├── calculate_moments.py      # SDF conditional moments
│   ├── evaluate_sdfs.py          # Portfolio statistics
│   └── upload_to_aws.py          # S3 utilities
│
├── utils_factors/            # Factor computation utilities
│   ├── dkkm_functions.py         # Random Fourier Features
│   ├── fama_functions.py         # Fama-French construction
│   ├── factor_utils.py           # Common utilities
│   └── sdf_utils.py              # SDF-specific utilities
│
├── utils_bgn/                # BGN model implementation
├── utils_kp14/               # KP14 model implementation
├── utils_gs21/               # GS21 model implementation
│
├── outputs/                  # Generated panel data and results
├── tables/                   # LaTeX tables
├── figures/                  # Boxplot figures
└── logs/                     # Execution logs
```

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- numpy (>=1.20.0)
- pandas (>=1.3.0)
- scipy (>=1.7.0)
- statsmodels (>=0.13.0)
- scikit-learn (>=1.0.0)
- numba (>=0.56.0)
- joblib (>=1.1.0)
- boto3 (>=1.26.0)
- requests (>=2.31.0)

## Usage

### Running a Single Panel

```bash
python main.py <model> [start] [end]
```

**Examples:**
```bash
# Run KP14 model for panel index 0
python main.py kp14

# Run BGN model for panels 0-9
python main.py bgn 0 10

# Run GS21 model for panels 5-15
python main.py gs21 5 16
```

### Cloud Deployment (Koyeb)

```bash
./deploy_koyeb.sh <model> <start> <end> [instance_type] [git_repo] [upload_intermediate]
```

**Example:**
```bash
# Deploy KP14 panels 0-10 on 5xlarge instance
./deploy_koyeb.sh kp14 0 10 5xlarge
```

### Analyzing Results

```bash
python analyze.py
```

Generates LaTeX tables and boxplot figures from completed panel results.

## Workflow Pipeline

The framework executes a 6-step pipeline for each panel:

```
Step 1: Panel Generation
    └── Simulate N=1000 firms over T=720 months
    └── Uses: utils_bgn/panel_functions_bgn.py (BGN model simulation)
              utils_kp14/panel_functions_kp14.py (KP14 model simulation)
              utils_gs21/panel_functions_gs21.py (GS21 model simulation)
    └── Creates: {id}_panel.pkl, {id}_arr/ (memmap arrays)

Step 1b: Portfolio Construction
    └── Create 5×5 double-sorted portfolios (size × book-to-market)
    └── Uses: utils_factors/factor_utils.py (portfolio sorting utilities)
    └── Reads: {id}_panel.pkl
    └── Creates: {id}_25_portfolios.pkl

Step 2: Fama-French Factors
    └── Compute factor returns via 2×3 sorts and Fama-MacBeth
    └── Uses: utils_factors/fama_functions.py (Fama-French factor construction)
              utils_factors/factor_utils.py (common utilities)
    └── Reads: {id}_panel.pkl
    └── Creates: {id}_fama.pkl

Step 3: DKKM Factors
    └── Generate Random Fourier Features factors
    └── Uses: utils_factors/dkkm_functions.py (RFF computation)
              utils_factors/factor_utils.py (common utilities)
    └── Reads: {id}_panel.pkl
    └── Creates: {id}_dkkm.pkl

Step 4a: Fama SDF Estimation
    └── Ridge regression for Fama/CAPM SDF weights
    └── Uses: utils_factors/fama_functions.py (portfolio weight computation)
              utils_factors/factor_utils.py (common utilities)
    └── Reads: {id}_panel.pkl, {id}_fama.pkl
    └── Creates: {id}_stock_weights_fama.pkl

Step 4b: DKKM SDF Estimation
    └── Ridge regression for DKKM SDF weights
    └── Uses: utils_factors/dkkm_functions.py (ridge regression)
              utils_factors/factor_utils.py (common utilities)
    └── Reads: {id}_panel.pkl, {id}_fama.pkl, {id}_dkkm.pkl, {id}_stock_weights_fama.pkl
    └── Creates: {id}_stock_weights_dkkm.pkl

Step 5: Conditional Moments
    └── Compute expected returns and covariances
    └── Uses: utils_bgn/sdf_compute_bgn.py (BGN SDF computation)
              utils_kp14/sdf_compute_kp14.py (KP14 SDF computation)
              utils_gs21/sdf_compute_gs21.py (GS21 SDF computation)
    └── Reads: {id}_arr/ (memmap arrays)
    └── Creates: {id}_moments.pkl

Step 6: Evaluation
    └── Calculate Sharpe ratios and portfolio statistics
    └── Uses: utils_factors/factor_utils.py (common utilities)
              utils_factors/sdf_utils.py (SDF evaluation utilities)
    └── Reads: {id}_panel.pkl, {id}_stock_weights_fama.pkl, {id}_stock_weights_dkkm.pkl
    └── Creates: {id}_results.pkl
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 1000 | Number of firms |
| `T` | 720 | Time periods (months) |
| `N_JOBS` | 24 | Parallel workers |
| `N_DKKM_FEATURES_LIST` | [6, 36, 360, 3600] | DKKM feature counts |
| `ALPHA_LST` | [0, 0.001, 0.01, 0.05, 0.1, 1] | Ridge regularization (BGN, KP14) |
| `ALPHA_LST_GS` | [0, 1e-5, 1e-4, 5e-4, 1e-3, 0.01] | Ridge regularization (GS21) |
| `NMAT` | 5 | Independent DKKM weight matrices |

## Methods

### Factor Construction

**Fama-French Method:**
- 2×3 double sorts on size and characteristics
- Value-weighted portfolios
- Factors: MKT, SMB, HML, CMA, RMW, UMD
- GS21 includes an additional leverage factor (mkt_lev)

**Fama-MacBeth Method:**
- Cross-sectional regression each month
- Estimates risk premiums for each characteristic

**CAPM:**
- Market factor only

**Random Fourier Features (DKKM):**
- Rank standardization of characteristics
- Random weight matrix W ~ N(0, 1)
- RFF: φ(x) = [sin(Wx + b), cos(Wx + b)]
- Ridge regression on RFF features
- Results averaged over NMAT independent weight matrix draws

### SDF Estimation

For each month t:
```
min_β ||R_t - β'F_t||² + α||β||²
```

Where:
- R_t: Stock returns
- F_t: Factor returns
- α: Ridge penalty

## Environment Variables

For AWS S3 integration:
```bash
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export S3_BUCKET=<bucket-name>
export WORKFLOW_ID=<unique-id>
```

For Koyeb deployment:
```bash
export KOYEB_API_TOKEN=<token>
```

## Output Files

Each panel generates:

| File | Description |
|------|-------------|
| `{model}_{id}_panel.pkl` | Panel data (returns, characteristics) |
| `{model}_{id}_arr/` | Memory-mapped arrays |
| `{model}_{id}_fama.pkl` | Fama-French factor returns |
| `{model}_{id}_dkkm.pkl` | DKKM factor returns |
| `{model}_{id}_stock_weights_fama.pkl` | Fama SDF weights |
| `{model}_{id}_stock_weights_dkkm.pkl` | DKKM SDF weights |
| `{model}_{id}_moments.pkl` | Conditional moments |
| `{model}_{id}_results.pkl` | Final evaluation results |

## Performance Optimization

The framework includes several optimizations:
- **Memory mapping**: Large arrays stored as .npy files with mmap access
- **Parallel processing**: Joblib with multiprocessing backend
- **Chunked computation**: Reduces garbage collection pressure
- **Per-step worker tuning**: DKKM SDF estimation uses fewer workers for memory
- **Pre-allocated arrays**: Arrays pre-allocated with `np.empty()` to avoid repeated allocation
- **C-contiguous layout**: Arrays ensured to be C-contiguous for optimal BLAS performance in matrix operations


