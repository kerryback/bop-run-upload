"""
Centralized configuration for noipca2.

This config matches the root codebase (main_revised.py + parameters.py) as closely
as possible, while maintaining the modular file structure.

ROOT REFERENCES:
  - parameters.py: gamma_grid, chars, burnin, economic parameters
  - main_revised.py lines 1-20: N, T, nmat, nfeatures_lst, alpha_lst, n_jobs, include_market
"""

import numpy as np
import os

# =============================================================================
# PANEL DIMENSIONS
# ROOT: main_revised.py line 9: N, T, n_ipca_rff = 1000, 720, 36
# =============================================================================

N = 1000   # Number of firms
T = 720    # Number of time periods (excluding burnin)
BGN_BURNIN = 500   # ROOT: parameters.py line 12: burnin = 200
KP14_BURNIN = 500
GS21_BURNIN = 500
N_JOBS = 24   # ROOT: main_revised.py line 20: n_jobs = 10

# Per-step n_jobs configuration (overrides N_JOBS if set)
# Useful for memory management with large MAX_FEATURES
# Set to None to use the default N_JOBS value
N_JOBS_MOMENTS = 30      # Step 5: Moments calculation (not currently used - reserved for future)
N_JOBS_FAMA = None       # Step 2: Fama factors (uses N_JOBS=24)
N_JOBS_DKKM = None       # Step 3: DKKM factors (uses N_JOBS=24)
N_JOBS_SDF = 10          # Step 4: SDF estimation (memory-intensive with large MAX_FEATURES)

# =============================================================================
# DATA DIRECTORY CONFIGURATION
# =============================================================================

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_CONFIG_DIR, 'outputs')
TEMP_DIR = DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)


def set_temp_dir(temp_path):
    """Set temporary directory path."""
    global TEMP_DIR
    TEMP_DIR = temp_path
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"[CONFIG] TEMP_DIR set to: {TEMP_DIR}")


def set_n_jobs(n_jobs):
    """Set number of parallel jobs."""
    global N_JOBS
    N_JOBS = n_jobs
    print(f"[CONFIG] N_JOBS set to: {N_JOBS}")


def get_n_jobs_for_step(step_name):
    """
    Get the appropriate n_jobs for a specific step.

    Args:
        step_name: 'moments', 'fama', 'dkkm', or 'sdf'

    Returns:
        Number of parallel jobs to use
    """
    # Check for explicit per-step override
    overrides = {
        'moments': N_JOBS_MOMENTS,
        'fama': N_JOBS_FAMA,
        'dkkm': N_JOBS_DKKM,
        'sdf': N_JOBS_SDF,
    }

    if step_name in overrides and overrides[step_name] is not None:
        return overrides[step_name]

    # Default: use global N_JOBS
    return N_JOBS


def set_jgsrc1_config():
    """Configure for jgsrc1 server."""
    set_temp_dir('/opt/scratch/keb7')
    set_n_jobs(10)

# =============================================================================
# FILE MANAGEMENT FLAGS
# =============================================================================

KEEP_PANEL = True
KEEP_MOMENTS = True
KEEP_WEIGHTS = True

# =============================================================================
# DKKM AND FAMA PARAMETERS
# ROOT: main_revised.py lines 11-20
# =============================================================================

# ROOT: main_revised.py line 12: nmat = 1
NMAT = 20

# ROOT: main_revised.py line 13: nfeatures_lst = [6, 36, 360, 3600]
N_DKKM_FEATURES_LIST = [6, 36, 360, 3600, 18000]

# ROOT: main_revised.py line 14: max_features = max(nfeatures_lst)
MAX_FEATURES = 18000

# ROOT: main_revised.py line 15: alpha_lst_fama = [0]
ALPHA_LST_FAMA = [0]

# ROOT: main_revised.py line 16: alpha_lst = [0, 0.01, 0.05, 0.1, 1]
ALPHA_LST = [0, 0.001, 0.01, 0.05, 0.1, 1]

# ROOT: main_revised.py lines 17-18 (GS21 uses finer grid)
ALPHA_LST_GS = [x/100 for x in ALPHA_LST]

# =============================================================================
# ROOT: parameters.py — Economic parameters
# =============================================================================

# ROOT: parameters.py lines 3-13
PI = 0.99
RBAR = 0.006236
KAPPA = 0.95
SIGMA_R = 0.002
BETA_ZR = -0.00014
SIGMA_Z = 0.4
CBAR = -3.7
CHAT = np.exp(CBAR)
I = 1

# ROOT: parameters.py line 13: gamma_grid = np.arange(0.5, 1.1, 0.1)
GAMMA_GRID = np.arange(0.5, 1.1, 0.1)

# ROOT: parameters.py lines 14-15
CHARS_DEFAULT = ["size", "bm", "agr", "roe", "mom"]
FACTOR_NAMES_DEFAULT = ["smb", "hml", "cma", "rmw", "umd"]

# GS21 characteristics
CHARS_GS21 = ["size", "bm", "agr", "roe", "mom", "mkt_lev"]
FACTOR_NAMES_GS21 = ["smb", "hml", "cma", "rmw", "umd", "mkt_lev"]

# =============================================================================
# KP14 Model Parameters
# =============================================================================

KP14_DT = 1/12
KP14_MU_X = 0.01
KP14_MU_Z = 0.005
KP14_SIGMA_X = 0.13
KP14_SIGMA_Z = 0.035
KP14_THETA_EPS = 0.35
KP14_SIGMA_EPS = 0.2
KP14_THETA_U = 0.5
KP14_SIGMA_U = 1.5
KP14_DELTA = 0.1
KP14_MU_LAMBDA = 2.0
KP14_SIGMA_LAMBDA = 2.0
KP14_MU_H = 0.075
KP14_MU_L = 0.16
KP14_LAMBDA_H = 2.35
KP14_LAMBDA_L = (1 - KP14_MU_H/(KP14_MU_H + KP14_MU_L)*KP14_LAMBDA_H)/(1 - KP14_MU_H/(KP14_MU_H + KP14_MU_L))
KP14_R = 0.05
KP14_GAMMA_X = 0.69
KP14_GAMMA_Z = -0.35
KP14_ALPHA = 0.85

KP14_PROB_H = KP14_MU_L / (KP14_MU_H + KP14_MU_L)
KP14_CONST = KP14_R + KP14_GAMMA_X * KP14_SIGMA_X + KP14_DELTA - KP14_MU_X
KP14_A_0 = 1 / KP14_CONST
KP14_A_1 = 1 / (KP14_CONST + KP14_THETA_EPS)
KP14_A_2 = 1 / (KP14_CONST + KP14_THETA_U)
KP14_A_3 = 1 / (KP14_CONST + KP14_THETA_EPS + KP14_THETA_U)

KP14_RHO = (KP14_R + KP14_GAMMA_X * KP14_SIGMA_X - KP14_MU_X
            - KP14_ALPHA / (1 - KP14_ALPHA) * (KP14_MU_Z - KP14_GAMMA_Z * KP14_SIGMA_Z - 0.5 * KP14_SIGMA_Z**2)
            - 0.5 * (KP14_ALPHA / (1 - KP14_ALPHA))**2 * KP14_SIGMA_Z**2)
KP14_C = KP14_ALPHA**(1 / (1 - KP14_ALPHA)) * (KP14_ALPHA**(-1) - 1)

# =============================================================================
# GS21 Model Parameters
# =============================================================================

GS21_BETA = 0.994
GS21_PSI = 2
GS21_GAMMA = 10
GS21_G = 1.14
GS21_ALPHA = 0.2
GS21_DELTA = 0.02/3
GS21_RHO_X = 0.95**(1/3)
GS21_SIGMA_X = 0.012*np.sqrt((1 - 0.95**(3/2))/(1 - 0.95**2))
GS21_XBAR = 0
GS21_RHO_Z = 0.90**(1/3)
GS21_SIGMA_Z = 0.16*np.sqrt((1 - 0.9**(3/2))/(1 - 0.9**2))
GS21_ZBAR = 0
GS21_CHI = 1
GS21_TAU = 0.2/3
GS21_PHI = 0.4
GS21_KAPPA_E = 0
GS21_KAPPA_B = 0.004
GS21_ZETA = 0.03/3
GS21_IMIN = 0
GS21_IMAX = 2000
GS21_R = 0.074830/12

# =============================================================================
# MODEL MAPPINGS
# =============================================================================

LOADING_KEYS = {
    'bgn': ['A_1_', 'A_2_'],
    'kp14': ['A_1_', 'A_2_'],
    'gs21': ['A_1_']
}

FACTOR_KEYS = {
    'bgn': ['f_1_', 'f_2_'],
    'kp14': ['f_1_', 'f_2_'],
    'gs21': ['f_1_']
}

MODEL_CHARS = {
    'bgn': CHARS_DEFAULT,
    'kp14': CHARS_DEFAULT,
    'gs21': CHARS_GS21
}

MODEL_FACTOR_NAMES = {
    'bgn': FACTOR_NAMES_DEFAULT,
    'kp14': FACTOR_NAMES_DEFAULT,
    'gs21': FACTOR_NAMES_GS21
}

MODEL_ALPHA_LST = {
    'bgn': ALPHA_LST,
    'kp14': ALPHA_LST,
    'gs21': ALPHA_LST_GS
}


def get_model_config(model_name):
    """
    Get configuration dictionary for a specific model.

    ROOT: This combines parameters from main_revised.py and parameters.py
    into a single config dict used by all scripts.
    """
    if model_name not in MODEL_CHARS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Valid models: {list(MODEL_CHARS.keys())}")
        import sys
        sys.exit(1)

    burnin_map = {
        'bgn': BGN_BURNIN,
        'kp14': KP14_BURNIN,
        'gs21': GS21_BURNIN
    }

    return {
        'model': model_name,
        'N': N,
        'T': T,
        'burnin': burnin_map[model_name],
        'chars': MODEL_CHARS[model_name],
        'factor_names': MODEL_FACTOR_NAMES[model_name],
        'n_jobs': N_JOBS,
        'nmat': NMAT,
        'max_features': MAX_FEATURES,
        'n_dkkm_features_list': N_DKKM_FEATURES_LIST,
        'alpha_lst_fama': ALPHA_LST_FAMA,
        'alpha_lst': MODEL_ALPHA_LST[model_name],
    }


# =============================================================================
# CONFIGURATION EXAMPLES FOR DIFFERENT SCENARIOS
# =============================================================================
#
# For high feature counts (MAX_FEATURES > 1000), memory usage increases
# significantly in Step 4 (SDF estimation). Set N_JOBS_SDF to a lower value
# to prevent OOM errors:
#
# Example 1: MAX_FEATURES=18000 with memory optimization (current config)
#   N_DKKM_FEATURES_LIST = [6, 36, 360, 3600, 18000]
#   MAX_FEATURES = 18000
#   N_JOBS = 24             # Used by Steps 2 & 3 (Fama, DKKM)
#   N_JOBS_SDF = 10         # Reduce to 10 workers for Step 4
#   N_JOBS_MOMENTS = 30     # Reserved for Step 5 (not currently used)
#
# Example 2: Balanced configuration
#   MAX_FEATURES = 3600
#   N_JOBS = 24
#   N_JOBS_SDF = 16         # Moderate reduction for Step 4
#
# Note: N_JOBS_MOMENTS is reserved for future parallelization of Step 5
# (moments calculation in evaluate_sdfs.py), which is not currently parallel.
# =============================================================================
