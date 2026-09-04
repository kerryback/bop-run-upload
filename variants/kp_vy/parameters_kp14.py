burnin = 400
dt = 1/12
import os, json
import numpy as _np

mu_x, mu_z, sigma_x, sigma_z = 0.01, 0.005, 0.13, 0.035
theta_eps, sigma_eps = 0.35, 0.2
theta_u, sigma_u = 0.5, 1.5
delta, mu_lambda, sigma_lambda, mu_H, mu_L, lambda_H = 0.1, 2.0, 2.0, 0.075, 0.16, 2.35
lambda_L = (1 - mu_H/(mu_H + mu_L)*lambda_H)/(1 - mu_H/(mu_H + mu_L))
r, gamma_x, gamma_z = 0.05, 0.69, -0.35 ## NOTE: r is different from KP14
alpha = 0.85

# ---- kp_vy: a PRICED stationary OU factor y with heterogeneous firm exposures ----
# y ~ OU(kappa_y, stationary sd 1); the SDF prices dW_y at gamma_v; type-f cash flows load as
# e^{beta_f * y_t}.  Values of e^{beta y} flows carry y-dependent effective discounts (the OU pull
# kappa_y*y enters), so the A-coefficients are per-(type, y-node) ODE solutions: the state bends a
# heterogeneous exposure map while every level stays STATIONARY (harvestability clause iii).
# type_bv = [0] reproduces the kp_gamy unit economy exactly.
g_lo, g_hi, g_steep = 1.0, 1.0, 1.5      # optional gamma_x multiplier in y (1,1 = off)
kappa_y = 0.35
NY = 21
y_max = 3.5
gam_seed = 555
gamma_v = 0.12                            # price of y-risk
type_share = [1.0]
type_bv = [0.0]                           # exposure betas on y
bv_comp = 0.0                             # y=0 value compensation knob (omega analogue)
bx_seed = 777
g_file, integ_file = 'G_func.csv', 'integ_results.npz'

globals().update(json.loads(os.environ.get('KP_PARAM_OVERRIDES', '{}')))
lambda_L = (1 - mu_H/(mu_H + mu_L)*lambda_H)/(1 - mu_H/(mu_H + mu_L))
# 2026-09-04: mu_H/mu_L are ENTRY rates (named for the state they lead TO),
# so P(high) = mu_H/(mu_H+mu_L) = 0.3191, matching the lambda_L line above and
# the G recombination in kp14_fd.py:100-101. Was mu_L/(mu_H+mu_L) = 0.6809,
# which made E[lambda] = 1.7172 instead of the intended 1.0.
# See ../../docs/kp14_regime_labels.md.
prob_H = mu_H/(mu_H + mu_L)
exit_H, exit_L = mu_L, mu_H   # rates of LEAVING the high / low state

type_share = _np.array(type_share, float); type_share /= type_share.sum()
type_bv = _np.array(type_bv, float)
ntypes = len(type_share)
assert len(type_bv) == ntypes

y_grid = _np.linspace(-y_max, y_max, NY)
dy = y_grid[1] - y_grid[0]
sigma_y = _np.sqrt(2.0 * kappa_y)                      # stationary sd 1

def gmult_y(y):
    z = _np.clip(g_steep * _np.asarray(y, float), -40, 40)
    return g_lo + (g_hi - g_lo) / (1.0 + _np.exp(-z))

# OU generator on the y-grid (upwind drift, reflecting boundaries)
def _build_Qy():
    Q = _np.zeros((NY, NY))
    drift = -kappa_y * y_grid
    diff = 0.5 * sigma_y ** 2 / dy ** 2
    for i in range(NY):
        if drift[i] > 0 and i < NY - 1:
            Q[i, i + 1] += drift[i] / dy; Q[i, i] -= drift[i] / dy
        elif drift[i] < 0 and i > 0:
            Q[i, i - 1] += -drift[i] / dy; Q[i, i] -= -drift[i] / dy
        if 0 < i < NY - 1:
            Q[i, i - 1] += diff; Q[i, i + 1] += diff; Q[i, i] -= 2 * diff
        elif i == 0:
            Q[i, i + 1] += diff; Q[i, i] -= diff
        else:
            Q[i, i - 1] += diff; Q[i, i] -= diff
    return Q
Qy = _build_Qy()

gm_grid = gmult_y(y_grid)
const_base_y = r + gamma_x * gm_grid * sigma_x + delta - mu_x                       # (NY,)

def const_ty(f):
    """effective discount for type-f claims (flows ~ e^{beta_f y}):
    Ito on e^{beta y}: drift beta*(-kappa_y*y) + 0.5*beta^2*sigma_y^2; priced comp beta*gamma_v*sigma_y"""
    b = type_bv[f]
    return const_base_y + b * gamma_v * sigma_y + b * kappa_y * y_grid - 0.5 * b ** 2 * sigma_y ** 2

def _solve_coeff_ty(f, theta_c):
    return _np.linalg.solve(_np.diag(const_ty(f) + theta_c) - Qy, _np.ones(NY))
A0_ty = _np.stack([_solve_coeff_ty(f, 0.0) for f in range(ntypes)])                 # (ntypes, NY)
A1_ty = _np.stack([_solve_coeff_ty(f, theta_eps) for f in range(ntypes)])
A2_ty = _np.stack([_solve_coeff_ty(f, theta_u) for f in range(ntypes)])
A3_ty = _np.stack([_solve_coeff_ty(f, theta_eps + theta_u) for f in range(ntypes)])

# y=0 value compensation: theta multiplies the cash-flow level (and hence A's and all values)
_i0 = NY // 2
type_theta = (A0_ty[0, _i0] / A0_ty[:, _i0]) ** bv_comp
pm_tau = type_theta ** (1 / (1 - alpha))
A0_ty = A0_ty * type_theta[:, None]; A1_ty = A1_ty * type_theta[:, None]
A2_ty = A2_ty * type_theta[:, None]; A3_ty = A3_ty * type_theta[:, None]

def _coef_at(y, f):
    y = _np.asarray(y, float)
    return (_np.interp(y, y_grid, A0_ty[f]), _np.interp(y, y_grid, A1_ty[f]),
            _np.interp(y, y_grid, A2_ty[f]), _np.interp(y, y_grid, A3_ty[f]))

def A_y(ep, u, y, f):
    a0, a1, a2, a3 = _coef_at(y, f)
    return a0 + (ep - 1) * a1 + (u - 1) * a2 + (ep - 1) * (u - 1) * a3

rho_ty = _np.stack([(r + gamma_x * gm_grid * sigma_x - mu_x
                     + type_bv[f] * gamma_v * sigma_y + type_bv[f] * kappa_y * y_grid
                     - 0.5 * type_bv[f] ** 2 * sigma_y ** 2
                     - alpha / (1 - alpha) * (mu_z - gamma_z * sigma_z - 0.5 * sigma_z ** 2)
                     - 0.5 * (alpha / (1 - alpha)) ** 2 * sigma_z ** 2)
                    for f in range(ntypes)])                                        # (ntypes, NY)

# physical one-period E[e^{b*(y' - m(y))}]-style helpers for the OU factor
_ar_y = _np.exp(-kappa_y * dt)
_sd_c = _np.sqrt(1.0 - _ar_y ** 2)              # conditional sd of y' (stationary sd 1)
def Ey_exp(b, y):
    """E[e^{b y'} | y]  (physical, one period)"""
    return _np.exp(b * _ar_y * _np.asarray(y, float) + 0.5 * b ** 2 * _sd_c ** 2)

# legacy aliases (type 0, y = 0)
A_0, A_1, A_2, A_3 = float(A0_ty[0, _i0]), float(A1_ty[0, _i0]), float(A2_ty[0, _i0]), float(A3_ty[0, _i0])
def A(ep, u):
    return A_0 + (ep - 1) * A_1 + (u - 1) * A_2 + (ep - 1) * (u - 1) * A_3
rho = float(rho_ty[0, _i0])
C = alpha**(1 / (1 - alpha)) * (alpha**(-1) - 1)
