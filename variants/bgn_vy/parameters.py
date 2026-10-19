"""bgn_vy: BGN (1999) with a PRICED stationary AR(1) state y and heterogeneous firm exposures,
the kp_vy construction one model over.

    y_{t+1} = rho_y y_t + s_y eta_{t+1},   eta ~ N(0,1) i.i.d., stationary sd 1,
    rho_y = exp(-kappa_y dt), s_y = sqrt(1 - rho_y^2), dt = 1/12.
    log M_{t+1} gains  - lam_y eta_{t+1} - lam_y^2/2,   lam_y = gamma_v sqrt(dt),
so gamma_v is the price of y-risk per unit Brownian per sqrt(year), KP14's convention, and the
Q-mean of y is -lam_y s_y/(1 - rho_y), KP14's -gamma_v sigma_y/kappa_y to O(dt).
Type-f firms' project cash flows are multiplied by e^{b_f y_t}; type_theta compensates the
cash-flow LEVEL so that at y = 0 project values do not reveal the type (bv_comp as in kp_vy).

type_bv = [0] (or gamma_v = 0 and bv_comp = 0) reproduces bgn_gam's economy: every base
reduces to vasicek.py's, and the panel draws the same firm-level shocks from the global stream.

Every parameter below the override site is DERIVED; overriding one raises, as in parameters_kp14.py.
"""
import numpy as np
import os, json

# ---- baseline BGN calibration (identical to bgn_gam/parameters.py) ----
pi = 0.99
rbar = 0.006236
kappa = 0.95
sigma_r = 0.002
beta_zr = - 0.00014
sigma_z = 0.4
Cbar = -3.7
I = 1
burnin = 400          # the protocol's BURNIN (common/protocol.py)
gamma_grid = np.arange(0.5, 1.1, 0.1)
chars = ["size", "bm", "agr", "roe", "mom"]
names = ["smb", "hml", "cma", "rmw", "umd"]

prob_in_money_targets = (0.10, 0.05)
sigmaj_width = 0.3
sigmaj_scale_cap = 2.0
sigmaj_size_elast = 0.0
jstar_file = "Jstar.csv"

# ---- regime block (bgn_gam): kept so bgn_vy NESTS bgn_gam; [1,1] = inert ----
gmult = [1.0, 1.0]
p01, p10 = 0.25 / 12, 0.50 / 12
gam_seed = 555

# ---- the priced state y and the exposure types (kp_vy's knobs) ----
dt = 1.0 / 12
kappa_y = 0.35            # annual mean reversion of y
gamma_v = 0.0             # price of y-risk, per unit Brownian per sqrt(year) (kp_vy convention)
NY = 21                   # y-nodes of the reported tables
y_max = 3.5
type_share = [1.0]
type_bv = [0.0]           # exposure of each type's project cash flows: e^{b_f y}
bv_comp = 0.0             # y=0 level compensation exponent (kp_vy's bv_comp)
bx_seed = 777             # firm-type draw (run_oracle.py offsets it by the seed)
y_seed_tag = 1            # y path: default_rng([gam_seed, y_seed_tag]); the regime keeps gam_seed
jstar_vy_file = "Jstar_vy.npz"   # the solve artifact: J*_f(r, y, s) per type, written by rebuild_jstar_vy.py
jstar_tol = 1e-7          # DP fixed-point tolerance (relative), hashed into the solve_id via env_params
n_quad_y = 12             # Gauss-Hermite nodes for every y'-expectation
n_quad_r = 100            # Gauss-Hermite nodes for every r'-expectation (bgn_gam uses 100)
# ---- y as a DISCRETE CHAIN (2026-10-12, "different distributions"): y_kind "chain" replaces the Gaussian AR(1) by a
# Tauchen chain on NY nodes over +-y_max, optionally with a rare priced CRASH: with probability pj a month the chain
# jumps to its lowest node, and the jump carries its own kernel factor exp(gj). Every y-expectation (the bases' E^Q,
# the DP's transition, the moments' y'-sums, the panel's path) then runs on the chain, so no closed form is needed.
# "gauss" (the default) leaves every existing bgn_vy economy byte-identical.
y_kind = "gauss"
pj = 0.0
gj = 0.0
# ---- a SECOND independent priced chain (2026-10-12, "larger factor models"): NY2 > 1 (chain mode only) adds a chain
# y2 with its own exposures type_bv2 (None = zeros), price gamma_v2 (None = gamma_v) and mean reversion kappa_y2 (None =
# kappa_y). The state is then the JOINT node (y1, y2): P_y, M_y, PQ_y are Kronecker products, NY becomes NY1 * NY2
# (NY1 keeps the per-chain count), y_grid the first coordinate (what y_stand reports), and every type's exposure is the
# node table BY[f] = type_bv[f] y1 + type_bv2[f] y2. NY2 = 1 leaves every single-chain economy op-for-op unchanged.
NY2 = 1
type_bv2 = None
gamma_v2 = None
kappa_y2 = None
n_r_solve = 321           # r-nodes of the J* dynamic programme over rbar +- 4.5 sd (linear interpolation of a convex
                          # J biases E[J'] up by O(h^2): 81 nodes gave +1.0% vs the closed form at b=0, 321 gives ~1e-4)
y_sub_solve = 4           # J* y-nodes per reported y-grid interval (spacing dy/4 = 0.0875 against s_y 0.238)

_pre_override_names = set(globals())
_ov = json.loads(os.environ.get("BGN_PARAM_OVERRIDES", "{}"))
globals().update(_ov)

# ---- overrides must actually take effect (parameters_kp14.py's guard) ----
_NORMALISED = {"type_share"}


def _override_took(name, requested, current):
    if isinstance(requested, str) or isinstance(current, str):
        return requested == current
    a, b = np.asarray(requested, float).ravel(), np.asarray(current, float).ravel()
    if a.shape != b.shape:
        return False
    if name in _NORMALISED:
        s = a.sum()
        if s == 0:
            return False
        a = a / s
    return bool(np.allclose(a, b, rtol=1e-12, atol=0.0))


_discarded = []
_k = _v = None
for _k, _v in _ov.items():
    if _k not in _pre_override_names:
        _discarded.append(f"{_k}: no such parameter (misspelled? it would affect nothing)")
    elif not _override_took(_k, _v, globals()[_k]):
        _discarded.append(f"{_k}: requested {_v!r}, module has {globals()[_k]!r}")
del _k, _v
if _discarded:
    raise ValueError("BGN_PARAM_OVERRIDES entries that had NO EFFECT:\n  " + "\n  ".join(_discarded))

Chat = np.exp(Cbar)
nchars = len(chars)
gmult = np.array(gmult, float)
prob_calm = p10 / (p01 + p10)
Preg = np.array([[1 - p01, p01], [p10, 1 - p10]])

type_share = np.array(type_share, float); type_share /= type_share.sum()
type_bv = np.array(type_bv, float)
ntypes = len(type_share)
assert len(type_bv) == ntypes
NY2 = int(NY2)
type_bv2 = np.zeros(ntypes) if type_bv2 is None else np.array(type_bv2, float)
assert len(type_bv2) == ntypes
assert NY2 == 1 or y_kind == "chain", "a second chain (NY2 > 1) needs y_kind = 'chain'"
CHAIN2 = (y_kind == "chain" and NY2 > 1)
gamma_v2 = float(gamma_v if gamma_v2 is None else gamma_v2)
kappa_y2 = float(kappa_y if kappa_y2 is None else kappa_y2)

rho_y = float(np.exp(-kappa_y * dt))
s_y = float(np.sqrt(1.0 - rho_y ** 2))
lam_y = float(gamma_v * np.sqrt(dt))               # monthly price (Sharpe) of the y-innovation
y_mean_Q = -lam_y * s_y / (1.0 - rho_y)             # Q-stationary mean of y
y_grid = np.linspace(-y_max, y_max, NY)


def m_k(b, k):
    """log E^Q_t[e^{b y_{t+k}}] - b rho^k y_t: the Q-drift and variance terms, k >= 0 (array ok)."""
    k = np.asarray(k, float)
    return (-b * lam_y * s_y * (1.0 - rho_y ** k) / (1.0 - rho_y)
            + 0.5 * b * b * (1.0 - rho_y ** (2 * k)))


def Ek_y(b, k, y):
    """E^Q_t[e^{b y_{t+k}} | y_t = y] = exp(b rho^k y + m_k(b)); k and y broadcast."""
    k = np.asarray(k, float); y = np.asarray(y, float)
    return np.exp(b * rho_y ** k * y + m_k(b, k))


def Ephys_exp(b, y):
    """physical E_t[e^{b y_{t+1}} | y_t = y]"""
    return np.exp(b * rho_y * np.asarray(y, float) + 0.5 * b * b * s_y ** 2)


# ---- the chain, when y_kind == "chain": y_grid itself is the node set --------------------------------------------
CHAIN = (y_kind == "chain")
NY1 = NY                                                       # nodes of the first chain (NY is the JOINT count below)
if CHAIN:
    from scipy.special import erfc as _erfc

    def _tauchen_chain(grid, rho, s, lam, pj_, gj_):
        """one chain on `grid`: physical P and kernel M with E[M | i] = 1 on the grid, optional priced crash to node 0"""
        ginc = grid[1] - grid[0]
        cdf = lambda x, mu: 0.5 * _erfc(-((x - mu) / s) / np.sqrt(2))
        n = len(grid); P = np.zeros((n, n))
        for i in range(n):
            mu = rho * grid[i]
            P[i, 1:-1] = cdf(grid[1:-1] + ginc / 2, mu) - cdf(grid[1:-1] - ginc / 2, mu)
            P[i, 0] = cdf(grid[0] + ginc / 2, mu)
            P[i, -1] = 1 - cdf(grid[-1] - ginc / 2, mu)
            P[i] /= P[i].sum()
        ptau = P.copy()
        jump = np.zeros((n, n)); jump[:, 0] = 1.0
        if pj_ > 0:
            P = (1 - pj_) * ptau + pj_ * jump
        eta = (grid[None, :] - rho * grid[:, None]) / s
        M = np.exp(-lam * eta - 0.5 * lam ** 2)
        if pj_ > 0:
            M = ((1 - pj_) * ptau * M + pj_ * jump * np.exp(gj_)) / np.where(P > 0, P, 1.0)
        M = M / (M * P).sum(axis=1)[:, None]                  # E[M | i] = 1 exactly on the grid
        return P, M

    P_y, M_y = _tauchen_chain(y_grid, rho_y, s_y, lam_y, pj, gj)
    Y1, Y2 = y_grid.copy(), np.zeros(NY)
    if CHAIN2:
        rho_y2 = float(np.exp(-kappa_y2 * dt)); s_y2 = float(np.sqrt(1.0 - rho_y2 ** 2)); lam_y2 = float(gamma_v2 * np.sqrt(dt))
        y_grid2 = np.linspace(-y_max, y_max, NY2)
        P_y2, M_y2 = _tauchen_chain(y_grid2, rho_y2, s_y2, lam_y2, 0.0, 0.0)
        P_y = np.kron(P_y, P_y2); M_y = np.kron(M_y, M_y2)      # independent chains: joint node i = i1 * NY2 + i2
        M_y = M_y / (M_y * P_y).sum(axis=1)[:, None]
        Y1, Y2 = np.repeat(y_grid, NY2), np.tile(y_grid2, NY1)
        NY = NY1 * NY2
        y_grid = Y1                                               # the first coordinate: the reported "y value"
    PQ_y = M_y * P_y                                          # the risk-neutral transition
    pi_y = np.linalg.matrix_power(P_y.T, 2000) @ np.full(NY, 1.0 / NY)   # stationary law
    y_mean_Q = float(np.linalg.matrix_power(PQ_y.T, 2000) @ np.full(NY, 1.0 / NY) @ y_grid)
    BY = type_bv[:, None] * Y1[None, :] + type_bv2[:, None] * Y2[None, :]    # (ntypes, NY): each type's exponent per node
    i_y0 = int(np.argmin(np.abs(Y1) + np.abs(Y2)))             # the y = 0 node (NY // 2 on one chain)
    y_axis = (np.arange(NY, dtype=float) if CHAIN2 else y_grid)  # the y AXIS of every 2-D table: the node index on a joint chain

    def EkQ_chain(bexp, kmax):
        """(kmax+1, NY): E^Q[e^{B(y_{t+k})} | node i] for k = 0..kmax, B the node exponents (BY[f]), by repeated PQ_y products"""
        out = np.empty((kmax + 1, NY)); v = np.exp(np.asarray(bexp, float)); out[0] = v
        for k in range(1, kmax + 1):
            v = PQ_y @ v; out[k] = v
        return out

    def Ephys_exp_chain(bexp, iy):
        """physical E_t[e^{B(y_{t+1})} | node iy], B the node exponents (BY[f])"""
        return (P_y @ np.exp(np.asarray(bexp, float)))[np.asarray(iy, int)]
else:
    y_axis = y_grid
    i_y0 = NY // 2
