import numpy as np
import os, json

# ---- baseline BGN calibration (identical to Code/parameters.py) ----
pi = 0.99
rbar = 0.006236
kappa = 0.95
sigma_r = 0.002
beta_zr = - 0.00014
sigma_z = 0.4
Cbar = -3.7
I = 1
burnin = 300          # BGN paper: ~200 months suffice; Code/ uses 500
gamma_grid = np.arange(0.5, 1.1, 0.1)
chars = ["size", "bm", "agr", "roe", "mom"]
names = ["smb", "hml", "cma", "rmw", "umd"]

# ---- experiment knobs (baseline values reproduce the original code) ----
prob_in_money_targets = (0.10, 0.05)   # P(accept | r=0), P(accept | r=rbar) used to fit the beta distribution
sigmaj_width = 0.3                      # idiosyncratic-vol upper bound = |beta|/sigma_z + sigmaj_width*|Cbar|
sigmaj_scale_cap = 2.0                  # cap on that multiplier (keeps lognormal cash-flow vol finite for firms with no projects)
sigmaj_size_elast = 0.0                 # >0: projects of firms with few live projects (small firms) get more idiosyncratic cash-flow vol:
                                        #     width scaled by ((1+n_{t-1})/(1+nbar))^(-elast), nbar = cross-sectional mean
jstar_file = "Jstar.csv"

# ---- overrides via environment variable, e.g. BGN_PARAM_OVERRIDES='{"sigma_r": 0.004}' ----
_ov = json.loads(os.environ.get("BGN_PARAM_OVERRIDES", "{}"))
globals().update(_ov)

Chat = np.exp(Cbar)
nchars = len(chars)

# ---- state-dependent price of market risk (regime multiplies the z-shock price sigma_z) ----
gmult = [1.0, 1.0]                # sigma_z -> sigma_z * gmult[s]; [1,1] reproduces baseline exactly
p01, p10 = 0.25 / 12, 0.50 / 12   # monthly regime switch probs (calm->stress, stress->calm), as in kp_gam
gam_seed = 555
jstar_gam_file = "Jstar_gam.csv"
globals().update(_ov)             # allow overrides of the regime block too
gmult = np.array(gmult, float)
prob_calm = p10 / (p01 + p10)
Preg = np.array([[1 - p01, p01], [p10, 1 - p10]])
