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

# ---- the NB rate-bin bases (bgn_q) ----------------------------------------------------------------
# The date-k cash flow is priced at e^{-beta g(r_{t+k-1})}, so a project's value is Chat sum_j e^{-beta g_j} D_j with
#   D_j(r, y) = sum_k pi^k B(k, r) w_{k,j}(r) E^Q[e^{b y_{t+k}} | y],
# w_{k,j}(r) = E^Q[e^{-sum_{i<k} r_i} 1{bin(r_{t+k-1}) = j} | r] / B(k, r) the discount-weighted share of bin j at horizon k,
# computed by the recursion u_{1,j} = e^{-r} 1_j, u_{k,j} = e^{-r} E^Q[u_{k-1,j}(r') | r] on the r_solve grid (the J*
# programme's grid and Q-transition) and normalised to sum to one over j, so sum_j D_j is the closed-form D exactly
# and q_price = 0 reproduces bgn_vy.
_K = np.arange(1, max_prds + 1)
_sd_r0 = sigma_r / np.sqrt(1 - kappa ** 2)
_r_lo0, _r_hi0 = rbar - 4.5 * _sd_r0, rbar + 4.5 * _sd_r0
_nr0 = int(n_r_solve); _rgrid0 = np.linspace(_r_lo0, _r_hi0, _nr0)


def _interp_w0(grid, x):
    x = np.asarray(x, float).ravel(); n = len(grid)
    j = np.clip(np.searchsorted(grid, x) - 1, 0, n - 2); w = (x - grid[j]) / (grid[j + 1] - grid[j])
    W = np.zeros((x.size, n)); W[np.arange(x.size), j] = 1 - w; W[np.arange(x.size), j + 1] = w
    return W


_hr0, _wr0 = np.polynomial.hermite.hermgauss(40); _hr0 = _hr0 * np.sqrt(2.0); _wr0 = _wr0 / np.sqrt(np.pi)
_Wq0 = np.zeros((_nr0, _nr0))
for _i, _r0 in enumerate(_rgrid0):
    _Wq0[_i] = _wr0 @ _interp_w0(_rgrid0, kappa * _r0 + (1 - kappa) * rbar - beta_zr + sigma_r * _hr0)
_e_r0 = np.exp(-_rgrid0)
_Ushare = np.zeros((NB, max_prds, _nr0))                        # [j, k-1, r]: share of bin j at horizon k
_u = np.stack([_e_r0 * (bin_of_r(_rgrid0) == j) for j in range(NB)])
for _k in range(max_prds):
    _tot = _u.sum(0); _Ushare[:, _k, :] = _u / np.where(_tot > 0, _tot, 1.0)
    _u = _e_r0[None, :] * (_u @ _Wq0.T)
del _u, _tot


def _bin_shares(r, j):
    """w_{k,j}(r) for k = 1..max_prds at the points r (1-D): (len(r), K), linear in r on the grid, clamped outside"""
    r = np.asarray(r, float).ravel()
    idx = np.clip(np.searchsorted(_rgrid0, r) - 1, 0, _nr0 - 2)
    t = np.clip((r - _rgrid0[idx]) / (_rgrid0[idx + 1] - _rgrid0[idx]), 0.0, 1.0)
    return (_Ushare[j][:, idx] * (1 - t) + _Ushare[j][:, idx + 1] * t).T
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


def DJ(r, y, f, j, chunk=4000):
    """the bin-j basis D_j at matching arrays r, y (any shape), type f. In chain mode y is the NODE INDEX."""
    r = np.asarray(r, float)
    if CHAIN:
        iy = np.asarray(y, int)
        shp = np.broadcast(r, iy).shape
        r = np.broadcast_to(r, shp).ravel(); iy = np.broadcast_to(iy, shp).ravel()
        out = np.empty(r.size)
        Ek = _EKQ[f][1:, :].T                                   # (NY, K): node x horizon
        for i in range(0, r.size, chunk):
            sl = slice(i, i + chunk)
            out[sl] = (_piKfull * _bond_all(r[sl]) * _bin_shares(r[sl], j) * Ek[iy[sl]]).sum(-1)
        return out.reshape(shp)
    y = np.asarray(y, float)
    shp = np.broadcast(r, y).shape
    r = np.broadcast_to(r, shp).ravel(); y = np.broadcast_to(y, shp).ravel()
    out = np.empty(r.size)
    b = float(type_bv[f])
    for i in range(0, r.size, chunk):
        sl = slice(i, i + chunk)
        out[sl] = (_piKfull * _bond_all(r[sl]) * _bin_shares(r[sl], j) * Ek_mat(b, y[sl])).sum(-1)
    return out.reshape(shp)


def D_all(r, y, f):
    """[D_0, ..., D_{NB-1}] at matching r, y for type f"""
    return [DJ(r, y, f, j) for j in range(NB)]


EXPG = [np.exp(-g) for g in g_bins]                              # placeholders; the exponent is e^{-beta g_j} at each beta


# ---- y=0 level compensation: type f's cash-flow level x type_theta[f] so that at (rbar, 0) the
# unit project's value matches type 0's (bv_comp = 1 exactly, 1.2 over-compensates as kp_vy) ----
_y0 = i_y0 if CHAIN else 0.0                                   # the y = 0 point: node i_y0 on the chain
_D0 = np.array([float(sum(D_all(rbar, _y0, f))) for f in range(ntypes)])
type_theta = (_D0[0] / _D0) ** bv_comp
Chat_f = Chat * type_theta                                      # (ntypes,)


def V_vy(r, y, beta, s, f):
    """project value per unit cost for exposure beta, state (r, y), type f (s, the inert regime, is ignored)"""
    beta = np.asarray(beta, float)
    return Chat_f[f] * sum(np.exp(-beta * g_bins[j]) * DJ(r, y, f, j) for j in range(NB))


# ---- expected new-project NPV, exact under the exponential beta density (bgn_gam/sdf_compute.py)
_gmax = 2 * float(g_bins.max())
assert _gmax * scale < 1, f"beta-density tail too fat for g_bins max {g_bins.max():.3f}: need 2*max(g)*scale < 1 (scale {scale:.4f})"


def _thresh(Dv, C):
    """beta_bar solving C sum_j D_j e^{-b g_j} = 1 (V decreasing in b), vectorised bisection; Dv a list of NB arrays"""
    Dv = [np.maximum(d, 0.0) for d in Dv]
    lo = np.full(Dv[0].shape, -30.0); hi = np.full(Dv[0].shape, 30.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        V = C * sum(Dv[j] * np.exp(-mid * g_bins[j]) for j in range(NB))
        lo = np.where(V > 1, mid, lo)
        hi = np.where(V > 1, hi, mid)
    return 0.5 * (lo + hi)


def _I_of_g(g, x0):
    return np.exp(-g * beta_star) * np.exp(-(1 - g * scale) * x0) / (1 - g * scale)


def optval_core(Dv, C, sq=False):
    """E_beta[max(V - 1, 0)] (sq=False) or E_beta[max(V - 1, 0)^2] (sq=True) at cash-flow level C; Dv a list of NB arrays"""
    Dv = [np.maximum(d, 0.0) for d in Dv]
    dead = sum(Dv) <= 0
    bb = _thresh(Dv, C)
    x0 = np.maximum(0.0, (beta_star - bb) / scale)
    if not sq:
        out = sum(C * Dv[j] * _I_of_g(g_bins[j], x0) for j in range(NB)) - np.exp(-x0)
    else:
        out = (sum((C * Dv[j]) * (C * Dv[l]) * _I_of_g(g_bins[j] + g_bins[l], x0) for j in range(NB) for l in range(NB))
               - 2 * sum(C * Dv[j] * _I_of_g(g_bins[j], x0) for j in range(NB)) + np.exp(-x0))
    return np.where(dead, 0.0, np.maximum(out, 0.0))


def optval(r, y, s, f, sq=False):
    return optval_core(D_all(r, y, f), Chat_f[f], sq=sq)


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
