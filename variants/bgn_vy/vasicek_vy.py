"""Valuation for bgn_vy: BGN's Vasicek bond pricing and project-beta distribution (verbatim from
bgn_gam/vasicek.py), the two regime bases generalised to the priced state y per exposure type, and
the growth-option value J*_f(r, y, s) solved as a dynamic programme on an (r, y) grid.

THE BASES. A type-f project's date-(t+k) cash flow is C e^{b_f y_{t+k}}, with C lognormal and
priced through beta as in BGN. The y-part is independent of (r, nu), so its price factorises:
    E^Q_t[e^{b y_{t+k}}] = exp(b rho_y^k y_t + m_k(b)),   m_k in parameters.py,
and the value per unit cost of a project with exposure beta accepted at date t is
    V_f(r, y, beta, s) = Chat_f [ e^{-beta g0} DA_f(r, y, s) + e^{-beta g1} DB_f(r, y, s) ],
    DA_f(r, y, s) = sum_k pi^k a_k(s) B(k, r) E_k(y; b_f),   DB_f with (1 - a_k(s)),
    Chat_f = type_theta[f] * Chat   (the y=0 level compensation, kp_vy's bv_comp).
At b_f = 0 and gamma_v = 0, E_k = 1 and DA_f, DB_f are bgn_gam's DA, DB exactly.

THE GROWTH OPTION. BGN's J*(r) is a sum over arrival dates of European options on coupon bonds,
priced by Jamshidian's decomposition, which needs a one-factor state. With (r, y) it is solved
instead as the fixed point of
    J_f(x) = e^{-r} E^Q_x[ OV_f(x') + J_f(x') ],   x = (r, y, s),
    OV_f(x) = E_beta[ max(V_f(x, beta) - 1, 0) ]   (closed form under the exponential beta density),
on a uniform (r, y) grid wide enough to hold the Q-stationary law of both states, with Gauss-Hermite
quadrature and bilinear interpolation. The Q-drift of r is shifted by -beta_zr (BGN's bond prices
say so: tests/test_risk_neutral_pricing.py) and that of y by -lam_y s_y. At b = 0 this reproduces
bgn_gam's closed-form J*(r) up to quadrature and its 950-period truncation (validate_bgn_vy.py).
"""
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, fsolve
from scipy.stats import norm
from scipy.special import roots_laguerre
from scipy.interpolate import RegularGridInterpolator
from parameters import *

# ---- bond pricing, as bgn_gam/vasicek.py ------------------------------------------------------
max_prds = 1500
alpha2 = kappa ** np.arange(max_prds + 1)
alpha1 = np.concatenate(([0], np.cumsum(alpha2[:-1])))
phi2 = rbar - rbar * kappa ** np.arange(max_prds + 1)
phi1 = np.concatenate(([0], np.cumsum(phi2[:-1] + 0.5 * sigma_z ** 2 * np.ones(max_prds))))
sigma2_sq = (sigma_r ** 2 / (1 - kappa ** 2)
             - (sigma_r ** 2 / (1 - kappa ** 2)) * kappa ** (2 * np.arange(max_prds + 1)))
sigma12 = np.zeros(max_prds + 1)
for i in range(max_prds):
    sigma12[i + 1] = kappa * (sigma12[i] + sigma2_sq[i]) + beta_zr
sigma1_sq = np.concatenate(
    ([0], np.cumsum(sigma2_sq[:-1] + 2 * sigma12[:-1] + sigma_z ** 2 * np.ones(max_prds))))


def B(k, r):
    return np.exp(- alpha1[k] * r - phi1[k] + 0.5 * sigma1_sq[k])


def D(r):
    b = np.exp(- alpha1[1:] * r - phi1[1:] + 0.5 * sigma1_sq[1:])
    return np.sum(pi ** np.arange(1, max_prds + 1) * b)


def beta_density(b, beta_star, scale):
    return np.exp((b - beta_star) / scale) / scale


def beta_cdf(b, beta_star, scale):
    return scale * beta_density(b, beta_star, scale)


def prob_in_money(r, beta_star, scale):
    return beta_cdf(Cbar + np.log(D(r)), beta_star, scale)


def fit_beta_params(x):
    beta_star = x[0]
    scale = np.exp(x[1])
    p1 = prob_in_money(0, beta_star, scale)
    p2 = prob_in_money(rbar, beta_star, scale)
    return np.array([p1 - prob_in_money_targets[0], p2 - prob_in_money_targets[1]])


_x = fsolve(fit_beta_params, x0=[1, 1])
beta_star = _x[0]
scale = np.exp(_x[1])

# ---- regime weights and the bond matrix (bgn_gam) ----------------------------------------------
_g0, _g1 = gmult
_ak = np.zeros((2, max_prds + 1))
_pk = np.eye(2)
for _k in range(1, max_prds + 1):
    _ak[:, _k] = _pk[:, 0]
    _pk = _pk @ Preg
_K = np.arange(1, max_prds + 1)
_piKfull = pi ** _K
_rhoK = rho_y ** _K                                             # (K,)


def _bond_all(r):
    r = np.asarray(r, float)
    return np.exp(-alpha1[1:] * r[..., None] - phi1[1:] + 0.5 * sigma1_sq[1:])


def Ek_mat(b, y):
    """E^Q_t[e^{b y_{t+k}} | y] for k = 1..max_prds: shape y.shape + (K,)"""
    y = np.asarray(y, float)
    return np.exp(b * _rhoK * y[..., None] + m_k(b, _K))


_EKQ = {}                                                       # chain mode: per type, (K+1, NY) E^Q[e^{b y_{t+k}}]
if CHAIN:
    for _f in range(ntypes):
        _EKQ[_f] = EkQ_chain(BY[_f], max_prds)


def _basis(r, y, s, f, which, chunk=4000):
    """DA_f (which='A') or DB_f ('B') at matching arrays r, y (any shape), regime s, type f.
    In chain mode y is the NODE INDEX (int array); the y factor is read from the chain's E^Q table."""
    r = np.asarray(r, float)
    wk = _piKfull * (_ak[s, 1:] if which == "A" else (1 - _ak[s, 1:]))
    if CHAIN:
        iy = np.asarray(y, int)
        shp = np.broadcast(r, iy).shape
        r = np.broadcast_to(r, shp).ravel(); iy = np.broadcast_to(iy, shp).ravel()
        out = np.empty(r.size)
        Ek = _EKQ[f][1:, :].T                                   # (NY, K): node x horizon
        for i in range(0, r.size, chunk):
            sl = slice(i, i + chunk)
            out[sl] = (wk * _bond_all(r[sl]) * Ek[iy[sl]]).sum(-1)
        return out.reshape(shp)
    y = np.asarray(y, float)
    shp = np.broadcast(r, y).shape
    r = np.broadcast_to(r, shp).ravel(); y = np.broadcast_to(y, shp).ravel()
    out = np.empty(r.size)
    b = float(type_bv[f])
    for i in range(0, r.size, chunk):
        sl = slice(i, i + chunk)
        out[sl] = (wk * _bond_all(r[sl]) * Ek_mat(b, y[sl])).sum(-1)
    return out.reshape(shp)


def DA(r, y, s, f):
    return _basis(r, y, s, f, "A")


def DB(r, y, s, f):
    return _basis(r, y, s, f, "B")


# ---- y=0 level compensation: type f's cash-flow level x type_theta[f] so that at (rbar, 0) the
# unit project's value matches type 0's (bv_comp = 1 exactly, 1.2 over-compensates as kp_vy) ----
_y0 = i_y0 if CHAIN else 0.0                                   # the y = 0 point: node i_y0 on the chain
_D0 = np.array([float(DA(rbar, _y0, 0, f) + DB(rbar, _y0, 0, f)) for f in range(ntypes)])
type_theta = (_D0[0] / _D0) ** bv_comp * type_level          # type_level = ones unless the spec sets it
Chat_f = Chat * type_theta                                      # (ntypes,)


def V_vy(r, y, beta, s, f):
    """project value per unit cost for exposure beta, state (r, y), regime s, type f"""
    return Chat_f[f] * (np.exp(-beta * _g0) * DA(r, y, s, f) + np.exp(-beta * _g1) * DB(r, y, s, f))


# ---- expected new-project NPV, exact under the exponential beta density (bgn_gam/sdf_compute.py)
_gmax = 2 * max(gmult)
assert _gmax * scale < 1, f"beta-density tail too fat for gmult={list(gmult)}: need 2*max(gmult)*scale < 1"


def _thresh(DAv, DBv, C):
    DAv = np.maximum(DAv, 0.0); DBv = np.maximum(DBv, 0.0)
    lo = np.full(DAv.shape, -30.0); hi = np.full(DAv.shape, 30.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        V = C * (DAv * np.exp(-mid * gmult[0]) + DBv * np.exp(-mid * gmult[1]))
        lo = np.where(V > 1, mid, lo)
        hi = np.where(V > 1, hi, mid)
    return 0.5 * (lo + hi)


def _I_of_g(g, x0):
    return np.exp(-g * beta_star) * np.exp(-(1 - g * scale) * x0) / (1 - g * scale)


def optval_core(DAv, DBv, C, sq=False):
    """E_beta[max(V - 1, 0)] (sq=False) or E_beta[max(V - 1, 0)^2] (sq=True) at cash-flow level C"""
    DAv = np.maximum(DAv, 0.0); DBv = np.maximum(DBv, 0.0)
    dead = (DAv + DBv) <= 0
    bb = _thresh(DAv, DBv, C)
    x0 = np.maximum(0.0, (beta_star - bb) / scale)
    g0, g1 = gmult
    if not sq:
        out = (C * DAv * _I_of_g(g0, x0) + C * DBv * _I_of_g(g1, x0) - np.exp(-x0))
    else:
        out = ((C * DAv) ** 2 * _I_of_g(2 * g0, x0) + (C * DBv) ** 2 * _I_of_g(2 * g1, x0)
               + 2 * C ** 2 * DAv * DBv * _I_of_g(g0 + g1, x0)
               - 2 * C * DAv * _I_of_g(g0, x0) - 2 * C * DBv * _I_of_g(g1, x0) + np.exp(-x0))
    return np.where(dead, 0.0, np.maximum(out, 0.0))


def optval(r, y, s, f, sq=False):
    return optval_core(DA(r, y, s, f), DB(r, y, s, f), Chat_f[f], sq=sq)


# ---- the (r, y) solve grid and the growth-option dynamic programme ------------------------------
_sd_r = sigma_r / np.sqrt(1 - kappa ** 2)
r_lo, r_hi = rbar - 4.5 * _sd_r, rbar + 4.5 * _sd_r
n_r = int(n_r_solve)
r_solve = np.linspace(r_lo, r_hi, n_r)
if CHAIN:
    y_solve = y_axis.copy(); _i_tab = np.arange(NY)     # the node index itself on a joint chain
else:
    dy = (y_grid[1] - y_grid[0]) / int(y_sub_solve)
    _n_lo = int(np.ceil((-y_max - (min(-y_max, y_mean_Q) - 6.0)) / dy))
    _n_hi = int(np.ceil(6.0 / dy))
    y_solve = (-y_max - _n_lo * dy) + dy * np.arange(_n_lo + (NY - 1) * int(y_sub_solve) + 1 + _n_hi)
    _i_tab = _n_lo + int(y_sub_solve) * np.arange(NY)
assert np.allclose(y_solve[_i_tab], y_axis if CHAIN else y_grid, atol=1e-10)
n_y = len(y_solve)


def interp_weights(grid, x):
    """linear interpolation (and linear extrapolation at the ends) of f(grid) at x: (len(x), len(grid))"""
    x = np.asarray(x, float).ravel()
    n = len(grid)
    j = np.clip(np.searchsorted(grid, x) - 1, 0, n - 2)
    w = (x - grid[j]) / (grid[j + 1] - grid[j])
    W = np.zeros((x.size, n))
    W[np.arange(x.size), j] = 1 - w
    W[np.arange(x.size), j + 1] = w
    return W


def _gh(n):
    h, w = np.polynomial.hermite.hermgauss(n)
    return h * np.sqrt(2.0), w / np.sqrt(np.pi)


def transition_Q(n_qr=40, n_qy=20):
    """W_r (n_r x n_r) and W_y (n_y x n_y): E^Q[f(x') | x] = W f for f on the solve grids."""
    hr, wr = _gh(n_qr); hy, wy = _gh(n_qy)
    Wr = np.zeros((n_r, n_r)); Wy = np.zeros((n_y, n_y))
    for i, r0 in enumerate(r_solve):
        rp = kappa * r0 + (1 - kappa) * rbar - beta_zr + sigma_r * hr
        Wr[i] = wr @ interp_weights(r_solve, rp)
    if CHAIN:
        return Wr, PQ_y.copy()                                  # exact on the chain
    for j, y0 in enumerate(y_solve):
        yp = rho_y * y0 - lam_y * s_y + s_y * hy
        Wy[j] = wy @ interp_weights(y_solve, yp)
    return Wr, Wy


def solve_jstar(tol=1e-7, max_iter=50000, verbose=True):
    """J_f(r, y, s) on (r_solve, y_solve) for every type and regime. Returns (J, OV, info)."""
    import time
    t0 = time.time()
    R, Y = np.meshgrid(r_solve, (np.arange(n_y) if CHAIN else y_solve), indexing="ij")
    OV = np.zeros((ntypes, 2, n_r, n_y))
    for f in range(ntypes):
        for s in (0, 1):
            OV[f, s] = optval(R, Y, s, f)
    if verbose:
        print(f"[jstar_vy] option values on the {n_r}x{n_y} grid in {time.time()-t0:.0f}s", flush=True)
    Wr, Wy = transition_Q()
    disc = np.exp(-r_solve)[:, None]
    J = np.zeros_like(OV)
    it = 0; rel = np.inf
    for it in range(1, max_iter + 1):
        Jn = np.empty_like(J)
        for f in range(ntypes):
            EJ = [Wr @ (OV[f, sp] + J[f, sp]) @ Wy.T for sp in (0, 1)]
            for s in (0, 1):
                Jn[f, s] = disc * (Preg[s, 0] * EJ[0] + Preg[s, 1] * EJ[1])
        rel = float(np.max(np.abs(Jn - J) / np.maximum(np.abs(Jn), 1e-12)))
        J = Jn
        if rel < tol:
            break
        if verbose and it % 500 == 0:
            print(f"[jstar_vy] iter {it}: max rel change {rel:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    info = {"iters": int(it), "rel_change": rel, "exit": "tolerance" if rel < tol else "iteration_cap",
            "seconds": time.time() - t0}
    return J, OV, info


def load_jstar(path):
    """RegularGridInterpolator per (type, regime) over (r, y), extrapolating linearly."""
    d = np.load(path)
    J = d["J"]
    assert J.shape == (ntypes, 2, len(d["r_solve"]), len(d["y_solve"])), "J* table does not match this economy's types"
    assert np.allclose(d["type_theta"], type_theta), "J* table was built at other type_theta"
    return [[RegularGridInterpolator((d["r_solve"], d["y_solve"]), J[f, s], bounds_error=False, fill_value=None)
             for s in (0, 1)] for f in range(ntypes)]
