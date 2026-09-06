"""
Centralized configuration for noipca2.

This config matches the root codebase (main_revised.py + parameters.py) as closely
as possible, while maintaining the modular file structure.

ROOT REFERENCES:
  - parameters.py: gamma_grid, chars, burnin, economic parameters
  - main_revised.py lines 1-20: N, T, nmat, nfeatures_lst, alpha_lst, n_jobs, include_market
"""

import os

# Set BLAS threads to 1 for better parallelization with many workers
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np

# =============================================================================
# PANEL DIMENSIONS
# ROOT: main_revised.py line 9: N, T, n_ipca_rff = 1000, 720, 36
# =============================================================================

N = 1000   # Number of firms
T = 720    # Number of time periods (excluding burnin)
BGN_BURNIN = 200
KP14_BURNIN = 200
GS21_BURNIN = 200

# =============================================================================
# N_JOBS CONFIGURATION BY MODEL AND STEP
# =============================================================================
# 4xlarge instances (32 vCPUs) with BLAS threads=1

MODEL_N_JOBS = {
    'bgn': {
        'moments': 1,   # n_jobs=1 runs in-process (no forking); OOM-killed workers cause indefinite hangs
        'generate_fama': 16,
        'generate_dkkm': 16,
        'estimate_fama': 16,
        'estimate_dkkm': 16,
    },
    'kp14': {
        'moments': 16,
        'generate_fama': 16,
        'generate_dkkm': 16,
        'estimate_fama': 16,
        'estimate_dkkm': 16,
    },
    'gs21': {
        'moments': 16,
        'generate_fama': 16,
        'generate_dkkm': 16,
        'estimate_fama': 16,
        'estimate_dkkm': 16,
    },
}

# Chunk sizes for moments calculation (per model)
MODEL_CHUNK_SIZE = {
    'bgn': 16,
    'kp14': 16,
    'gs21': 16,
}

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


def get_n_jobs_for_step(step_name, model_name='bgn'):
    """
    Get the appropriate n_jobs for a specific step and model.

    Args:
        step_name: 'moments', 'fama', or 'dkkm'
        model_name: 'bgn', 'kp14', or 'gs21'

    Returns:
        Number of parallel jobs to use
    """
    # Check for environment variable override (set by deploy_koyeb.sh)
    env_n_jobs = os.environ.get('N_JOBS')
    if env_n_jobs:
        return int(env_n_jobs)

    model_config = MODEL_N_JOBS.get(model_name, MODEL_N_JOBS['bgn'])
    return model_config[step_name]


def set_scratch_dir(scratch_dir, temp_dir=None, n_jobs_cap=None):
    """
    Route output and temporary files to scratch filesystems.

    Called automatically when the BOP_SCRATCH_DIR environment variable is set
    (e.g. from a SLURM sbatch script).  Optionally, BOP_TEMP_DIR can point to
    a separate directory for intermediate _arr/ files, keeping permanent pkl
    outputs in scratch_dir clean.  If temp_dir is omitted, both DATA_DIR and
    TEMP_DIR are set to scratch_dir (original behaviour).

    Args:
        scratch_dir: Path for permanent output files (panels, moments, results).
        temp_dir:    Path for intermediate _arr/ directories.  If None, falls
                     back to scratch_dir.
        n_jobs_cap:  If set, cap all MODEL_N_JOBS values to this number.
    """
    global DATA_DIR, TEMP_DIR
    DATA_DIR = scratch_dir
    TEMP_DIR = temp_dir if temp_dir else scratch_dir
    os.makedirs(DATA_DIR, exist_ok=True)
    if TEMP_DIR != DATA_DIR:
        os.makedirs(TEMP_DIR, exist_ok=True)
    if n_jobs_cap:
        for model in MODEL_N_JOBS:
            for step in MODEL_N_JOBS[model]:
                MODEL_N_JOBS[model][step] = min(MODEL_N_JOBS[model][step], n_jobs_cap)
    if TEMP_DIR != DATA_DIR:
        print(f"[CONFIG] DATA_DIR={DATA_DIR}")
        print(f"[CONFIG] TEMP_DIR={TEMP_DIR}")
    else:
        print(f"[CONFIG] DATA_DIR/TEMP_DIR set to: {scratch_dir}")


def init_from_env():
    """Read BOP_SCRATCH_DIR / BOP_TEMP_DIR env vars and configure dirs.

    Call at the top of every subprocess script so that cluster jobs route
    outputs to scratch rather than the default outputs/ directory.
    DATA_DIR  → BOP_SCRATCH_DIR (permanent pkl outputs)
    TEMP_DIR  → BOP_TEMP_DIR   (intermediate _arr/ directories)
    No-op when BOP_SCRATCH_DIR is not set (e.g. local runs, AWS).
    """
    scratch = os.environ.get('BOP_SCRATCH_DIR')
    if scratch:
        temp = os.environ.get('BOP_TEMP_DIR')
        set_scratch_dir(scratch, temp_dir=temp)


def set_jgsrc1_config():
    """Configure for jgsrc1 server with reduced worker counts."""
    set_temp_dir('/opt/scratch/keb7')
    # Cap all models to 10 workers max
    for model in MODEL_N_JOBS:
        for step in MODEL_N_JOBS[model]:
            MODEL_N_JOBS[model][step] = min(MODEL_N_JOBS[model][step], 10)
    print("[CONFIG] MODEL_N_JOBS capped to 10 workers")

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
NMAT = 5

# ROOT: main_revised.py line 13: nfeatures_lst = [6, 36, 360, 3600]
N_DKKM_FEATURES_LIST = [6, 36, 360, 3600]

# ROOT: main_revised.py line 14: max_features = max(nfeatures_lst)
MAX_FEATURES = max(N_DKKM_FEATURES_LIST)

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
# "size" is always the first char and is required — it drives the Big/Small
# sort in FF/FM and is the input for SMB. The --chars CLI flag always prepends
# "size" automatically, so users never need to specify it.
# SMB and market (mkt_rf) are always present as factors in FF and FM output,
# regardless of which subset of chars is chosen.
CHARS_DEFAULT = ["size", "bm", "agr", "roe", "mom"]
FACTOR_NAMES_DEFAULT = ["smb", "hml", "cma", "rmw", "umd"]

# GS21 characteristics (adds market leverage)
CHARS_GS21 = ["size", "bm", "agr", "roe", "mom", "mkt_lev"]
FACTOR_NAMES_GS21 = ["smb", "hml", "cma", "rmw", "umd", "mkt_lev"]

# Explicit characteristic → factor name mapping (non-size chars only).
# Used by fama_functions.py to assign factor names dynamically for any subset.
# Note: "smb" is derived from the "bm" sort, not a direct char column.
CHAR_TO_FACTOR = {
    "bm":      "hml",
    "agr":     "cma",
    "roe":     "rmw",
    "mom":     "umd",
    "mkt_lev": "mkt_lev",
}
FACTOR_TO_CHAR = {v: k for k, v in CHAR_TO_FACTOR.items()}

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
# 2026-09-06: Kogan-Papanikolaou (2014) Table II sets r = 0.025. This repo uses 0.05,
# a DELIBERATE departure flagged in variants/kp_vy/parameters_kp14.py:14 but not, until
# now, here. Every other Table II parameter matches the paper exactly (17 of 18).
KP14_R = 0.05          # paper: 0.025 -- see note above
# 2026-09-01: gamma_x, sigma_eps and sigma_u restored to Kogan-Papanikolaou (2014)
# Table 2 (0.69, 0.20, 1.50), matching the Dropbox code. The 0.1 factors on
# sigma_eps/sigma_u and the doubled gamma_x (1.38) were repo-only departures.
# Any change here MUST be followed by `python utils_kp14/regen_solfiles.py`;
# the stamp guard refuses to run on stale solution files (see
# kp14_crash_20260826.md for what happens otherwise).
KP14_GAMMA_X = 0.69
KP14_GAMMA_Z = -0.35
KP14_ALPHA = 0.85

# 2026-09-04: KP14_MU_H / KP14_MU_L are ENTRY rates, named for the state they
# lead TO -- MU_H is the low->high rate, MU_L is the high->low rate. Hence the
# stationary P(high) = MU_H/(MU_H+MU_L) = 0.3191.
#
# This was previously MU_L/(MU_H+MU_L) = 0.6809, which contradicted the two
# places that pin the same quantity independently:
#   * KP14_LAMBDA_L (line 257) solves w*LAMBDA_H + (1-w)*LAMBDA_L = 1 with
#     w = MU_H/(MU_H+MU_L), i.e. E[lambda] = 1 at P(high) = 0.3191.
#   * utils_kp14/kp14_fd.py:118-119 recombines the (mean, difference) basis as
#     G_up = Gbar + (1-P_H)*D and G_down = Gbar - P_H*D with coefficients
#     MU_L/(MU_L+MU_H) and MU_H/(MU_L+MU_H), i.e. again P(high) = 0.3191.
# At P(high) = 0.6809 the normalisation is infeasible: LAMBDA_L = -1.88, a
# negative arrival rate. Under the old value the simulated economy had
# E[lambda] = 1.7172 rather than the intended 1.0.
# See docs/kp14_regime_labels.md. No solfile depends on this constant, so the
# fix requires re-simulation but NOT re-solving.
KP14_PROB_H = KP14_MU_H / (KP14_MU_H + KP14_MU_L)

# Exit (hazard) rates, derived from the entry rates above. Consumers that ask
# "given I am in state s, at what rate do I leave?" must use these, not MU_H/MU_L.
KP14_EXIT_H = KP14_MU_L      # rate of leaving the HIGH state
KP14_EXIT_L = KP14_MU_H      # rate of leaving the LOW state
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

GS21_BETA = 0.994**(1/3)      # quarterly 0.994 -> monthly
GS21_PSI = 2
GS21_GAMMA = 10
GS21_G = 1.14
GS21_ALPHA = 0.2
# 2026-09-06: corrected against Gomes-Schmid (2021) Table I itself. delta is a
# PERIODIC cost: "maintenance of the existing capital stock entails periodic costs
# delta*k_jt, akin to depreciation" (p.287), set to "2% per quarter, consistent with
# standard estimates of capital depreciation rates" (p.792). A monthly model therefore
# needs 0.02/3. The previous justification here -- that it scales output and is not
# rescaled monthly -- was wrong on both counts: it scales CAPITAL, and it is periodic.
GS21_DELTA = 0.02/3
# 2026-09-06: Table I says rho_x = 0.95, and the body text repeats it verbatim
# ("rho_x = 0.95 and sigma_x = 0.012"). The comment previously here claimed Table 1
# said 0.96; that was false. GS21.m:22 had it right all along.
GS21_RHO_X = 0.95**(1/3)
# 2026-09-01: Gomes-Schmid (2021) Table I is quarterly (sigma_x = 0.012,
# rho_x = 0.95, sigma_z = 0.16, rho_z = 0.90). Monthly persistence is the cube
# root; the monthly innovation sd that reproduces the quarterly one under
# rho_m = rho_q**(1/3) is  sigma_q * sqrt((1 - rho_q**(2/3)) / (1 - rho_q**2)).
# (The 3/2 exponent in the Dropbox parameters_gs21.py is a typo, and the
# earlier 0.1 factor on sigma_z was a repo-only departure.) Any change here
# MUST be followed by `python utils_gs21/regen_solfiles.py`; the stamp guard
# refuses to run on stale solution files.
GS21_SIGMA_X = 0.012*np.sqrt((1 - 0.95**(2/3))/(1 - 0.95**2))   # 2026-09-06: base 0.95, per Table I
GS21_XBAR = 0
GS21_RHO_Z = 0.90**(1/3)
GS21_SIGMA_Z = 0.16*np.sqrt((1 - 0.9**(2/3))/(1 - 0.9**2))
GS21_ZBAR = 0
GS21_CHI = 1
GS21_TAU = 0.2      # a rate, not a flow: unchanged by the quarterly -> monthly conversion
GS21_PHI = 0.4
# 2026-09-06: Table I's benchmark is 0.025, chosen "to match the empirical frequency of
# equity issuances" (p.822). This was 0, matching GS21.m:33's `kappa_e = 0; %0.025` --
# but there the COMMENTED-OUT value is the paper's, the reverse of the sigma_m and r
# cases. The paper's kappa_e = 0 appears only as an expositional limit case for deriving
# the investment cutoff (p.407), not as a calibration. Seth's call: use the benchmark.
GS21_KAPPA_E = 0.025
GS21_KAPPA_B = 0.004
GS21_ZETA = 0.03/3
GS21_IMIN = 0
GS21_IMAX = 2000

# Grid construction. These lived only in GS21.m (lines 41-51) until 2026-08-26;
# the Python consumers inferred them from the solfile lengths (`zpts =
# len(zgrid)`), which is why nothing broke -- and why a grid-size change could
# not have been detected. Confirmed against the committed grid files to <=5e-16.
GS21_BMIN = 0.00
GS21_BMAX = 1.0
GS21_BNUM = 20
GS21_INUM = 40
# GS21_XNUM = 20 is LOAD-BEARING for solver stability, not just accuracy. The
# price operator reduces exactly to P = P0 + (PI-P0)^2/(2*imax), with gain
# E[M]*(1 + (PI-P0)*(g-1)/imax); it contracts only while i_cut << imax. A coarse
# x grid lets i_cut saturate, flipping the gain from E[M] = 0.992 to
# g*E[M] = 1.131, and the solve diverges outright (verified: xnum=5 overflows
# within one sweep, in both gs21_solve.py and GS21.m). GS21_ZNUM can be cut
# freely -- it is only cost.
GS21_XNUM = 20
GS21_ZNUM = 200
GS21_MNSTDEV = 4

# 2026-08-26: the last three GS21 parameters, identified rather than guessed.
# gs21_solve.py (the Python port of GS21.m) now consumes all three, so config.py
# is the single source of truth and the .m file is archival only.
#
# How they were identified: feed the COMMITTED solution files through the ported
# operator once. A faithful operator returns them nearly unchanged, so the
# parameter set that minimises that residual is the set that built them. On the
# core value functions:
#     r = 0.1/12,  xi = 0.01   ->  3.556e-04   <-- winner, 6x margin
#     r = config,  xi = 0.01   ->  2.131e-03
#     r = config,  xi = 0.03   ->  2.266e-03
#     r = 0.1/12,  xi = 0.03   ->  2.638e-03
# and sweeping gamma_x: 0.5 -> 9.5e-07, versus 0.4 -> 1.10e-02, 0.6 -> 1.52e-02.
#
# GS21_R was 0.074830/12, which is the COMMENTED-OUT alternative on GS21.m:27;
# the active value there is 0.1/12 and that is what built the solfiles.
# GS21_ZETA = 0.03/3 = 0.01 above is correct -- GS21.m:35's `0.03/3*3` is the
# erroneous one (the *3 should not be there).
GS21_R = 0.1/12
GS21_GAMMA_X = 0.5          # GS21.m:28, price of x risk
GS21_SIGMA_M = 5            # GS21.m:53, sd of the price-adjustment shock

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


# =============================================================================
# BOP_CHARS -- characteristic-subset override
# =============================================================================
# 2026-09-04: main.py's --chars flag used to work by mutating MODEL_CHARS in
# main.py's own interpreter. main.py runs all eight workflow steps as SEPARATE
# subprocesses (main.py:180), each of which re-imports this module from disk, so
# the mutation reached nothing: a --chars run produced byte-identical output to a
# full-set run, silently. It has to cross the process boundary the same way
# BOP_SCRATCH_DIR does -- through the environment.
#
# Value is a comma-separated list of FACTOR names (hml, cma, rmw, umd, mkt_lev),
# matching the --chars CLI surface. "size" is always prepended; "smb" is always
# produced.
def _apply_chars_env():
    raw = os.environ.get('BOP_CHARS')
    if not raw:
        return
    names = [f.strip().lower() for f in raw.split(',') if f.strip()]
    unknown = [f for f in names if f not in FACTOR_TO_CHAR]
    if unknown:
        raise ValueError(
            f"BOP_CHARS contains unknown factor name(s) {unknown}; "
            f"valid: {sorted(FACTOR_TO_CHAR)}"
        )
    chars = ["size"] + [FACTOR_TO_CHAR[f] for f in names]
    factors = ["smb"] + names
    for _m in MODEL_CHARS:
        MODEL_CHARS[_m] = list(chars)
        MODEL_FACTOR_NAMES[_m] = list(factors)


_apply_chars_env()


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
        'n_jobs': MODEL_N_JOBS[model_name],
        'nmat': NMAT,
        'max_features': MAX_FEATURES,
        'n_dkkm_features_list': N_DKKM_FEATURES_LIST,
        'alpha_lst_fama': ALPHA_LST_FAMA,
        'alpha_lst': MODEL_ALPHA_LST[model_name],
    }

