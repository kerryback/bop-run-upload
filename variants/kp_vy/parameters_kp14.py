burnin = 400
dt = 1/12
import os, json
import numpy as _np

mu_x, mu_z, sigma_x, sigma_z = 0.01, 0.005, 0.13, 0.035
theta_eps, sigma_eps = 0.35, 0.2
theta_u, sigma_u = 0.5, 1.5
delta, mu_lambda, sigma_lambda, mu_H, mu_L, lambda_H = 0.1, 2.0, 2.0, 0.075, 0.16, 2.35
# lambda_L is NOT set here: it is derived from the E[lambda]=1 normalisation below,
# after KP_PARAM_OVERRIDES is applied. An identical assignment used to sit on this
# line, which was dead (nothing reads lambda_L before the recomputation) and was the
# only reason lambda_L looked like a free parameter.
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
# 2026-09-18: price the e^{beta_f y} claims under the RISK-NEUTRAL OU dynamics (1), or the
# pre-fix way (0): physical generator plus a constant beta*gamma_v*sigma_y added to the
# discount rate. The constant-rate form is exact for KP14's GBM shocks and wrong for a
# mean-reverting state, whose cumulative Girsanov adjustment SATURATES at
# beta*gamma_v*sigma_y/kappa_y instead of growing linearly in the horizon. The corrected
# specification is _solve_coeff_ty below -- the substitution W = e^{by} A, solved under the
# Q-generator -- and its identities are asserted in tests/test_risk_neutral_pricing.py.
# 0 reproduces the pre-fix tables bit for bit and exists only so they can be rebuilt for
# comparison. At type_bv = [0] the two coincide.
y_risk_neutral = 1
g_file, integ_file = 'G_func.csv', 'integ_results.npz'

# Names defined before the override, so a misspelled key can be told from a real one:
# globals().update() CREATES whatever key it is given, so an unknown name would
# otherwise pass every later check while affecting nothing.
_pre_override_names = set(globals())
globals().update(json.loads(os.environ.get('KP_PARAM_OVERRIDES', '{}')))
lambda_L = (1 - mu_H/(mu_H + mu_L)*lambda_H)/(1 - mu_H/(mu_H + mu_L))
# 2026-09-04: mu_H/mu_L are ENTRY rates (named for the state they lead TO),
# so P(high) = mu_H/(mu_H+mu_L) = 0.3191, matching the lambda_L line above and
# the G recombination in kp14_fd.py:100-101. Was mu_L/(mu_H+mu_L) = 0.6809,
# which made E[lambda] = 1.7172 instead of the intended 1.0.
# See ../../docs/kp14_regime_labels.md.
prob_H = mu_H/(mu_H + mu_L)
exit_H, exit_L = mu_L, mu_H   # rates of LEAVING the high / low state


# --- overrides must actually take effect -------------------------------------------
# globals().update() above is silent: overriding a name that is RE-DERIVED below is
# accepted and discarded. `KP_PARAM_OVERRIDES={"lambda_L": 9.0}` used to run clean and
# leave lambda_L at 0.367. For a spec layer that hashes the requested overrides into a
# spec id, that means a spec could record a value the economy never used. Raise instead.
#
# `type_share` is renormalised to sum 1, so it is compared proportionally; everything
# else must match after list -> array coercion.
_NORMALISED = {'type_share'}


def _override_took(name, requested, current):
    a, b = _np.asarray(requested, float).ravel(), _np.asarray(current, float).ravel()
    if a.shape != b.shape:
        return False
    if name in _NORMALISED:
        s = a.sum()
        if s == 0:
            return False
        a = a / s
    return bool(_np.allclose(a, b, rtol=1e-12, atol=0.0))


_discarded = []
# 2026-09-21: _k and _v are DELETED after the loop. solstamp.param_namespace keeps any
# module-scope name that is not a callable and does not start with '__', so the loop
# variables were being hashed into the solve_id -- leaving it dependent on the KEY ORDER of
# KP_PARAM_OVERRIDES, since _k ends on whichever key came last. The same four parameters in
# three orders gave three different ids. runstamp._same_json compares specs by value and
# cannot catch it, so a spec authored with the keys in another order would silently claim a
# different solve. Initialised first so the del is safe when the override dict is empty.
_k = _v = None
for _k, _v in json.loads(os.environ.get('KP_PARAM_OVERRIDES', '{}')).items():
    if _k not in _pre_override_names:
        _discarded.append(f"{_k}: no such parameter (misspelled? it would affect nothing)")
    elif not _override_took(_k, _v, globals()[_k]):
        _discarded.append(f"{_k}: requested {_v!r}, module has {globals()[_k]!r}")
del _k, _v
if _discarded:
    raise ValueError(
        "KP_PARAM_OVERRIDES entries that had NO EFFECT. Either the name is derived "
        "below (so it is not a free parameter) or it does not exist. Silently ignoring "
        "them would let a spec record a value this economy never used:\n  "
        + "\n  ".join(_discarded))


type_share = _np.array(type_share, float); type_share /= type_share.sum()
type_bv = _np.array(type_bv, float)
ntypes = len(type_share)
assert len(type_bv) == ntypes

# ---- several priced states, at no extra grid cost ------------------------------------------
# type_bv may be (ntypes,) -- ONE priced state, the original economy -- or (ntypes, nstates), a
# loading VECTOR per type over independent unit-variance OU states that share kappa_y. The vector
# case needs no product grid and no new solve: the projection s_f = b_f . y is ITSELF a scalar OU,
# stationary sd ||b_f||, with its Q-drift shifted by -sigma_y (b_f . gamma). So type f is exactly
# the scalar problem below, at
#     b_eff(f) = ||b_f||        gamma_eff(f) = (b_f . gamma) / ||b_f||
# and everything downstream reads the type's own normalised projection in place of y.
#
# WHY THIS MATTERS ECONOMICALLY. The premium tracks b_f . gamma while the exposure MAGNITUDE
# tracks ||b_f||. With one state those are the same number times a constant, so any characteristic
# revealing the magnitude spans the whole premium cross-section exactly. With several they are a
# PRODUCT of magnitude and alignment, which no linear map in the characteristics reaches.
_bv = type_bv if type_bv.ndim == 2 else type_bv[:, None]
nstates = _bv.shape[1]
gamma_vec = _np.atleast_1d(_np.array(gamma_v, float))
assert len(gamma_vec) == nstates, (
    f"gamma_v has {len(gamma_vec)} entries but type_bv declares {nstates} priced state(s)")
if nstates == 1:
    # The scalar path is kept LITERALLY as it was, sign and all, so the three live kp_vy
    # economies rebuild byte-identical. The projection below normalises |b| and flips the grid
    # for b < 0 -- the same pricing in different coordinates, but not the same floating point.
    b_eff = _bv[:, 0].copy()
    gamma_eff = _np.full(ntypes, float(gamma_vec[0]))
else:
    b_eff = _np.linalg.norm(_bv, axis=1)
    _safe = _np.where(b_eff > 0, b_eff, 1.0)
    # at ||b_f|| = 0 the claim is on e^0 and A is 1/(rho0 + theta) whatever the generator, so
    # the value here is arbitrary; gamma_vec[0] keeps it continuous with the scalar path.
    gamma_eff = _np.where(b_eff > 0, (_bv @ gamma_vec) / _safe, float(gamma_vec[0]))
type_bv_mat = _bv          # (ntypes, nstates), star-exported for the panel's projection


def project_y(yreg):
    """Each type's normalised projection of a state path: ytil[f] = (b_f . y) / ||b_f||.

    That projection is a scalar OU with the same kappa_y and unit stationary sd, so every table
    lookup downstream is the scalar lookup it always was, at the type's own coordinate. At
    nstates = 1 it returns the raw path for every type, sign and all, which is what keeps the
    three live kp_vy economies byte-identical.
    """
    yreg = _np.asarray(yreg, float)
    if yreg.ndim == 1:
        yreg = yreg[:, None]
    if nstates == 1:
        return _np.repeat(yreg[:, 0][None, :], ntypes, axis=0)
    return (yreg @ type_bv_mat.T).T / b_eff[:, None]

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
    b, gv = b_eff[f], gamma_eff[f]
    return const_base_y + b * gv * sigma_y + b * kappa_y * y_grid - 0.5 * b ** 2 * sigma_y ** 2

# ---- risk-neutral pricing of the y-exposed claims -------------------------------------------
# Under the SDF  dM/M = ... - gamma_v dB_v  the state follows, under Q,
#     dy = (-kappa_y*y - gamma_v*sigma_y) dt + sigma_y dB_v^Q.
# The value of a type-f unit claim is W(y) = e^{b y} A(y) and solves, with rho0 = const_base_y,
#     (rho0 + theta_c) W - L^Q W = e^{b y}.
# Solving for W rather than A keeps the discount rate the POSITIVE rho0: the equivalent
# equation for A carries kappa_y*y*b in its discount bracket, which turns negative far below
# zero (y < -8.9 at b = 0.14) and would make a grid wide enough to hold the Q-measure ill-posed.
# In W-form there is no such wall, so the solve grid below simply spans the Q-stationary
# distribution (mean -gamma_v*sigma_y/kappa_y, sd 1) AND the table grid, finely; the table
# grid y_grid itself -- and with it NY and every downstream table -- is unchanged.
y_mean_Q = -float(_np.max(_np.abs(gamma_eff))) * sigma_y / kappa_y   # the widest type's Q-mean
_ys_lo = min(-y_max, y_mean_Q) - 8.0
_ys_hi = max(y_max, y_mean_Q) + 8.0
_sub = 20                                              # solve nodes per table interval
_n_lo = int(_np.ceil((-y_max - _ys_lo) / dy)); _n_hi = int(_np.ceil((_ys_hi - y_max) / dy))
y_solve = (-y_max - _n_lo * dy) + (dy / _sub) * _np.arange((_n_lo + _n_hi) * _sub + (NY - 1) * _sub + 1)
_i_tab = _n_lo * _sub + _sub * _np.arange(NY)          # y_solve[_i_tab] == y_grid
assert _np.allclose(y_solve[_i_tab], y_grid, atol=1e-10)

def build_generator(yv, drift):
    """generator of dy = drift dt + sigma_y dB on the uniform grid yv, reflecting ends.
    Central drift differences where they keep the matrix an M-matrix (|drift|*h <= sigma_y^2),
    upwind elsewhere."""
    n = len(yv); h = yv[1] - yv[0]
    Q = _np.zeros((n, n)); dif = 0.5 * sigma_y ** 2 / h ** 2
    for i in range(n):
        up = dif if i < n - 1 else 0.0; dn = dif if i > 0 else 0.0
        d = drift[i]
        if abs(d) * h <= sigma_y ** 2:
            if 0 < i < n - 1:
                up += d / (2 * h); dn -= d / (2 * h)
        elif d > 0 and i < n - 1:
            up += d / h
        elif d < 0 and i > 0:
            dn += -d / h
        if i < n - 1: Q[i, i + 1] = up
        if i > 0: Q[i, i - 1] = dn
        Q[i, i] = -(up + dn)
    return Q

# One generator per type: the Q-drift shift is gamma_eff(f)-dependent as soon as there is more
# than one priced state. At nstates = 1 every entry is the old single generator.
_QyQ_solve_ty = [build_generator(y_solve, -kappa_y * y_solve - gamma_eff[f] * sigma_y)
                 for f in range(ntypes)]
_const_base_solve = r + gamma_x * gmult_y(y_solve) * sigma_x + delta - mu_x
_A_solve = {}                                          # (f, theta_c) -> A on y_solve, pre-compensation

def _solve_coeff_ty(f, theta_c):
    if not y_risk_neutral:
        return _np.linalg.solve(_np.diag(const_ty(f) + theta_c) - Qy, _np.ones(NY))
    b = b_eff[f]
    W = _np.linalg.solve(_np.diag(_const_base_solve + theta_c) - _QyQ_solve_ty[f], _np.exp(b * y_solve))
    _A_solve[(f, theta_c)] = W * _np.exp(-b * y_solve)
    return _A_solve[(f, theta_c)][_i_tab]
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

def coef_on_solve_grid(f):
    """(A0, A1, A2, A3) for type f on y_solve, compensated like the tables. The G solve needs A
    where the Q-measure lives, which is outside y_grid; _coef_at would clamp it there."""
    return tuple(_A_solve[(f, th)] * type_theta[f] for th in (0.0, theta_eps, theta_u, theta_eps + theta_u))

def _coef_at(y, f):
    y = _np.asarray(y, float)
    return (_np.interp(y, y_grid, A0_ty[f]), _np.interp(y, y_grid, A1_ty[f]),
            _np.interp(y, y_grid, A2_ty[f]), _np.interp(y, y_grid, A3_ty[f]))

def A_y(ep, u, y, f):
    a0, a1, a2, a3 = _coef_at(y, f)
    return a0 + (ep - 1) * a1 + (u - 1) * a2 + (ep - 1) * (u - 1) * a3

# discount of the growth-option claim with the y-terms stripped: what the H = e^{b y} G form of
# the G equation uses under y_risk_neutral (kp14_fd_vy.py). rho_ty below is the pre-fix vector.
def rho0_at(yv):
    return (r + gamma_x * gmult_y(yv) * sigma_x - mu_x
            - alpha / (1 - alpha) * (mu_z - gamma_z * sigma_z - 0.5 * sigma_z ** 2)
            - 0.5 * (alpha / (1 - alpha)) ** 2 * sigma_z ** 2)

rho_ty = _np.stack([(r + gamma_x * gm_grid * sigma_x - mu_x
                     + b_eff[f] * gamma_eff[f] * sigma_y + b_eff[f] * kappa_y * y_grid
                     - 0.5 * b_eff[f] ** 2 * sigma_y ** 2
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
