__metaclass__ = type

from datetime import datetime
import numpy as np 
import pandas as pd 
from scipy.optimize import root_scalar, fsolve
from scipy.stats import norm
from scipy.special import roots_laguerre
from parameters import *

print(f"started import of vasicek at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

# parameters for bond pricing, going out max_prds

max_prds = 1500

alpha2 = kappa**np.arange(max_prds+1)
alpha1 = np.concatenate(([0], np.cumsum(alpha2[:-1])))
phi2 = rbar - rbar * kappa**np.arange(max_prds+1)
phi1 = np.concatenate(
    ([0], np.cumsum(phi2[:-1] + 0.5*sigma_z**2 * np.ones(max_prds)))
)
sigma2_sq = (
    sigma_r**2 / (1-kappa**2) 
    - (sigma_r**2 / (1-kappa**2)) * kappa**(2*np.arange(max_prds+1))
)

sigma12 = np.zeros(max_prds+1)
sigma12[0] = 0
for i in range(max_prds):
    sigma12[i+1] = kappa * (sigma12[i] + sigma2_sq[i]) + beta_zr

sigma1_sq = np.concatenate(
    ([0], np.cumsum(sigma2_sq[:-1] + sigma12[:-1] + sigma_z**2 * np.ones(max_prds)))
)

# bond price at maturity k
def B(k, r):
    return np.exp(- alpha1[k]*r - phi1[k] + 0.5*sigma1_sq[k])

# price of consol bond with depreciating coupons, approximating infty = max_prds
def D(r):

    # bond prices at maturities 1, ..., n
    b = np.exp(- alpha1[1:]*r - phi1[1:] + 0.5*sigma1_sq[1:])

    # discounted sum
    return np.sum(pi**np.arange(1, max_prds+1) * b)

# beta distribution parameters

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
    return np.array([p1-prob_in_money_targets[0], p2-prob_in_money_targets[1]])

x = fsolve(fit_beta_params, x0 = [1, 1])
beta_star = x[0]
scale = np.exp(x[1])
# print(f"beta_star = {beta_star}")
# print(f"beta_bar = {beta_star - scale}")
# valuing growth options

# interest rate at which option is at the money
def rstar(beta):
    return root_scalar(
        lambda r: Cbar - beta + np.log(D(r)), x0 = rbar
    ).root


def J(r, option_mat, bond_mat, strike):
    num = (
        np.log(B(option_mat+bond_mat, r))
        - np.log(B(option_mat, r))
        - np.log(B(bond_mat, strike))
        + 0.5 * alpha1[bond_mat]**2 * sigma2_sq[option_mat]
    )
    d1 = num / (alpha1[bond_mat]*np.sqrt(sigma2_sq[option_mat]))
    d2 = d1 - alpha1[bond_mat]*np.sqrt(sigma2_sq[option_mat])
    return (
        B(option_mat+bond_mat, r) * norm.cdf(d1) 
        - B(option_mat, r) * B(bond_mat, strike) * norm.cdf(d2)
    )

## replace integralJ function by this code:
nodes, weights = roots_laguerre(100)

rstar_nodes = np.zeros((len(nodes), ))
beta_nodes = beta_star - scale*nodes
for n in range(len(nodes)):
    rstar_nodes[n] = rstar(beta_nodes[n])

def integralJ(r, option_mat, bond_mat):
    return np.sum(weights * J(r, option_mat, bond_mat, rstar_nodes) * np.exp(-beta_nodes))

def inner_sum(r, option_mat):
    integrals = np.array([integralJ(r, option_mat, k) for k in range(1, 401)])
    return np.sum(integrals * pi**np.arange(1, 401))

def Jstar(r):
    return np.sum([inner_sum(r, s) for s in range(1, 951)])

print(f"finished import of vasicek at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

# ---- vectorised J*(r): identical maths to inner_sum/Jstar above, evaluated as arrays over (s, k, node) ----
_S = np.arange(1, 951); _K = np.arange(1, 401)
_logB_strike = (- alpha1[_K][:, None] * rstar_nodes[None, :] - phi1[_K][:, None] + 0.5 * sigma1_sq[_K][:, None])   # (400, 100)
_piK = pi ** _K
_wexp = weights * np.exp(-beta_nodes)                                                                        # (100,)

def Jstar_fast(r, chunk=50):
    total = 0.0
    a1k = alpha1[_K]                                  # (400,)
    for s0 in range(0, len(_S), chunk):
        s = _S[s0:s0 + chunk]                          # (c,)
        sk = s[:, None] + _K[None, :]                  # (c, 400)
        logB_sk = - alpha1[sk] * r - phi1[sk] + 0.5 * sigma1_sq[sk]
        logB_s = - alpha1[s] * r - phi1[s] + 0.5 * sigma1_sq[s]
        sd = np.sqrt(sigma2_sq[s])                     # (c,)
        num = (logB_sk[:, :, None] - logB_s[:, None, None] - _logB_strike[None, :, :]
               + 0.5 * (a1k[None, :, None] ** 2) * sigma2_sq[s][:, None, None])
        denom = a1k[None, :, None] * sd[:, None, None]
        d1 = num / denom
        d2 = d1 - denom
        J = np.exp(logB_sk)[:, :, None] * norm.cdf(d1) - np.exp(logB_s)[:, None, None] * np.exp(_logB_strike)[None, :, :] * norm.cdf(d2)
        total += np.einsum("ckn,k,n->", J, _piK, _wexp)
    return total

def build_jstar_table(path, nsim=int(2e6), npts=41, tol=1e-4, verbose=True):
    """Tabulate J*(r) on an adaptively refined grid covering the stationary range of r (adapted from the
    commented-out block in Code/vasicek.py). Needed whenever rbar, kappa, sigma_r, beta_zr, pi, Cbar or the
    beta distribution change."""
    from joblib import Parallel, delayed
    sd = sigma_r/np.sqrt(1-kappa**2)
    rmin, rmax = rbar - 4.5*sd, rbar + 4.5*sd
    grid1 = np.linspace(rmin, rmax, npts)
    J1 = np.array(Parallel(n_jobs=8)(delayed(Jstar_fast)(r) for r in grid1))
    maxerr, it = 1, 0
    while maxerr > tol and it < 4:
        grid2 = 0.5*(grid1[:-1] + grid1[1:])
        J2 = np.array(Parallel(n_jobs=8)(delayed(Jstar_fast)(r) for r in grid2))
        J2hat = np.interp(grid2, grid1, J1)
        maxerr = np.max(np.abs(J2 - J2hat)/np.abs(J2))
        grid1 = np.concatenate((grid1, grid2)); J1 = np.concatenate((J1, J2))
        o = np.argsort(grid1); grid1, J1 = grid1[o], J1[o]
        it += 1
        if verbose: print(f"Jstar table iter {it}: {len(grid1)} pts, max interp err {maxerr:.2e}")
    pd.DataFrame({"r": grid1, "J": J1}).to_csv(path, index=False)
    return grid1, J1


'''
nsim = int(1.0e8)
r = np.zeros(nsim+1)
r[0] = rbar
xi = norm.rvs(size=nsim)
const = (1-kappa) * rbar
for i in range(nsim):
    r[i+1] = kappa*r[i] + const + sigma_r * xi[i]

r = pd.Series(r)
rmin = r.quantile(1-0.99995)
rmax = r.quantile(0.99995)
print(f"99.99% of the distribution is between {rmin} and {rmax}")

grid1 = np.linspace(rmin, rmax, 51)
J1 = np.array([Jstar(r) for r in grid1])

maxerr = 1
iter = 0
while maxerr > 0.0001:
    print(iter)

    grid2 = 0.5*(grid1[:-1] + grid1[1:])
    J2 = np.array([Jstar(r) for r in grid2])

    grid = np.concatenate((grid1, grid2))
    Jvals = np.concatenate((J1, J2))

    df = pd.DataFrame(
        {"r": grid, "J": Jvals}
    )
    df = df.sort_values(by="r")
    df.to_csv("Jstar.csv", index=False)

    J2hat = np.interp(grid2, grid1, J1)
    err = np.abs(np.array(J2) - J2hat) / J2
    maxerr = np.max(err) 

    grid1 = df.r.to_numpy()
    J1 = df.J.to_numpy()

    iter += 1 


'''
# ======================================================================================================
# Regime extension: price of the market (z) shock is sigma_z * gmult[s], s a 2-state chain (Preg).
# The date-k cash flow's premium factor is e^{-beta*gmult[s_{k-1}]} (priced by the regime at the start
# of its final period), independent of the r-path, so values decompose onto two bases:
#   V_s(r, beta) = Chat * [ e^{-beta*g0} * DA_s(r) + e^{-beta*g1} * DB_s(r) ],
#   DA_s(r) = sum_k pi^k a_k(s) B(k,r),  a_k(s) = [Preg^(k-1)](s,0),  DB_s uses (1 - a_k(s)).
# At gmult=[1,1]: DA_s + DB_s = D and everything reduces to the baseline closed forms exactly.
# ======================================================================================================
_g0, _g1 = gmult
_ak = np.zeros((2, max_prds + 1))          # _ak[s, k] = P^{k-1}(s, 0), k >= 1
_pk = np.eye(2)
for _k in range(1, max_prds + 1):
    _ak[:, _k] = _pk[:, 0]
    _pk = _pk @ Preg
_piKfull = pi ** np.arange(1, max_prds + 1)

def _bond_all(r):
    """B(k, r) for k = 1..max_prds; r scalar or array (returned shape r.shape + (max_prds,))"""
    r = np.asarray(r, float)
    return np.exp(-alpha1[1:] * r[..., None] - phi1[1:] + 0.5 * sigma1_sq[1:])

def DA(r, s):
    return (_piKfull * _ak[s, 1:] * _bond_all(r)).sum(-1)

def DB(r, s):
    return (_piKfull * (1 - _ak[s, 1:]) * _bond_all(r)).sum(-1)

def V_gam(r, beta, s):
    """project value (per unit cost) for exposure beta, current regime s"""
    return Chat_v * (np.exp(-beta * _g0) * DA(r, s) + np.exp(-beta * _g1) * DB(r, s))

Chat_v = np.exp(Cbar)

def rstar_gam(beta, s):
    """exercise threshold in r for exposure beta when the exercise-date regime is s"""
    return root_scalar(lambda r: V_gam(r, beta, s) - 1, x0=rbar, x1=rbar + 0.01).root

# ---- generalized J*_s0(r): sum over arrival dates s of European bond-option payoffs, regime-mixed ----
def Jstar_gam_fast(r, s0, chunk=50):
    """identical maths to Jstar_fast, but: exercise-date regime s'' ~ Preg^s(s0,·); strike and payoff
    weights are regime- and maturity-dependent: w_k(s'',beta) = pi^k (a_k e^{-b g0} + (1-a_k) e^{-b g1})."""
    total = 0.0
    a1k = alpha1[_K]
    # per-exercise-regime strike thresholds and log-strike-bond tables (400, 100) per s''
    rst = np.array([[rstar_gam(b, ss) for b in beta_nodes] for ss in (0, 1)])          # (2, 100)
    logB_strike = np.stack([- alpha1[_K][:, None] * rst[ss][None, :] - phi1[_K][:, None]
                            + 0.5 * sigma1_sq[_K][:, None] for ss in (0, 1)])          # (2, 400, 100)
    wk = np.stack([_piK[:, None] * (_ak[ss, _K][:, None] * np.exp(-beta_nodes[None, :] * _g0)
                                    + (1 - _ak[ss, _K][:, None]) * np.exp(-beta_nodes[None, :] * _g1))
                   for ss in (0, 1)])                                                   # (2, 400, 100)
    # regime-path weights to the exercise date: Preg^s(s0, s'')
    Ps = np.zeros((len(_S) + 1, 2))
    _pw = np.eye(2)[s0]
    for i, s in enumerate(range(1, len(_S) + 1)):
        _pw = _pw @ Preg
        Ps[i] = _pw
    for s0i in range(0, len(_S), chunk):
        s = _S[s0i:s0i + chunk]
        sk = s[:, None] + _K[None, :]
        logB_sk = - alpha1[sk] * r - phi1[sk] + 0.5 * sigma1_sq[sk]
        logB_s = - alpha1[s] * r - phi1[s] + 0.5 * sigma1_sq[s]
        sd = np.sqrt(sigma2_sq[s])
        for ss in (0, 1):
            num = (logB_sk[:, :, None] - logB_s[:, None, None] - logB_strike[ss][None, :, :]
                   + 0.5 * (a1k[None, :, None] ** 2) * sigma2_sq[s][:, None, None])
            denom = a1k[None, :, None] * sd[:, None, None]
            d1 = num / denom
            d2 = d1 - denom
            J = (np.exp(logB_sk)[:, :, None] * norm.cdf(d1)
                 - np.exp(logB_s)[:, None, None] * np.exp(logB_strike[ss])[None, :, :] * norm.cdf(d2))
            total += np.einsum("ckn,kn,n,c->", J, wk[ss], weights, Ps[s0i:s0i + len(s), ss])
    return total

def build_jstar_gam_table(path, npts=41, tol=3e-4, verbose=True):
    from joblib import Parallel, delayed
    sd = sigma_r / np.sqrt(1 - kappa ** 2)
    rmin, rmax = rbar - 4.5 * sd, rbar + 4.5 * sd
    grid1 = np.linspace(rmin, rmax, npts)
    J1 = {s: np.array(Parallel(n_jobs=8)(delayed(Jstar_gam_fast)(r, s) for r in grid1)) for s in (0, 1)}
    maxerr, it = 1, 0
    while maxerr > tol and it < 4:
        grid2 = 0.5 * (grid1[:-1] + grid1[1:])
        J2 = {s: np.array(Parallel(n_jobs=8)(delayed(Jstar_gam_fast)(r, s) for r in grid2)) for s in (0, 1)}
        maxerr = max(np.max(np.abs(J2[s] - np.interp(grid2, grid1, J1[s])) / np.abs(J2[s])) for s in (0, 1))
        grid1 = np.concatenate((grid1, grid2))
        for s in (0, 1):
            J1[s] = np.concatenate((J1[s], J2[s]))
        o = np.argsort(grid1); grid1 = grid1[o]
        for s in (0, 1):
            J1[s] = J1[s][o]
        it += 1
        if verbose: print(f"Jstar_gam iter {it}: {len(grid1)} pts, max err {maxerr:.2e}", flush=True)
    pd.DataFrame({"r": grid1, "J0": J1[0], "J1": J1[1]}).to_csv(path, index=False)
    return grid1, J1
