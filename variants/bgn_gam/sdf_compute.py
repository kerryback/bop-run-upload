"""Regime version of bgn/sdf_compute.py: the price of the market shock is sigma_z*gmult[s_t], s_t a
2-state chain.  Values decompose onto two bases (DA_s, DB_s; see vasicek.py); every next-period value
table is regime-indexed and all conditional moments mix the two switch branches as a COMMON shock:
the r'-integral tables are computed per destination regime s' and mixed with (1-p_switch, p_switch),
so cross-products always condition on a single common s' (mixtures of products, never products of
mixtures).  At gmult=[1,1] every table collapses to the baseline and the code reproduces bgn exactly.
E_t[R_i R_j] is a PHYSICAL expectation, so gmult never appears directly in the moments -- only through
the regime-indexed value tables (and through prices/erets computed in panel_functions)."""
import numpy as np
import pandas as pd
import scipy.special
import scipy.linalg
from scipy.sparse import csr_matrix, kron
from scipy.stats import expon
from vasicek import *
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from parameters import *

Chat = np.exp(Cbar)
approx = pd.read_csv(jstar_gam_file)
Jstar_g = [interpolate.interp1d(approx.r, approx[f"J{s}"], fill_value="extrapolate") for s in (0, 1)]
r_pts = np.array(approx.r).reshape(len(approx.r), 1)

# fine interpolants of the two basis values per current regime (fast evaluation everywhere below)
DA_i = [interpolate.interp1d(approx.r, DA(np.array(approx.r), s), fill_value="extrapolate") for s in (0, 1)]
DB_i = [interpolate.interp1d(approx.r, DB(np.array(approx.r), s), fill_value="extrapolate") for s in (0, 1)]
D = interpolate.interp1d(approx.r, DA_i[0](approx.r) + DB_i[0](approx.r), fill_value="extrapolate")  # legacy (loadings diagnostics)
Jstar = Jstar_g[0]                                                                                    # legacy (loadings diagnostics)

cond_mean = kappa * r_pts + (1 - kappa) * rbar
n_pts = 100
rshock_nodes, wgts = np.polynomial.hermite.hermgauss(n_pts)
r_adj = np.sqrt(2) * sigma_r * rshock_nodes.reshape(1, n_pts) + cond_mean

# per-regime tables on the (r-grid x GH-node) lattice
DAvec = [DA_i[s](r_adj) for s in (0, 1)]
DBvec = [DB_i[s](r_adj) for s in (0, 1)]
Jstarvec_g = [Jstar_g[s](r_adj) for s in (0, 1)]

# expected new-project NPV (and its square) per exercise regime -- EXACT under the exponential beta
# density (x = (beta*-beta)/scale ~ Exp(1)):  I(g) = e^{-g beta*} e^{-(1-g*scale)x0}/(1-g*scale), with
# x0 = max(0, (beta*-beta_bar)/scale) and beta_bar the acceptance threshold V_s(r,beta_bar)=1.
# Requires g*scale < 1 for each exponent used (checked at import; squares need 2*g*scale < 1).
_gmax = 2 * max(gmult)
assert _gmax * scale < 1, f"beta-density tail too fat for gmult={list(gmult)}: need 2*max(gmult)*scale < 1"

def _thresh(DAv, DBv):
    """beta_bar solving Chat(DA e^{-b g0} + DB e^{-b g1}) = 1, vectorized bisection (V decreasing in b)"""
    DAv = np.maximum(DAv, 0.0); DBv = np.maximum(DBv, 0.0)
    lo = np.full(DAv.shape, -30.0); hi = np.full(DAv.shape, 30.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        V = Chat * (DAv * np.exp(-mid * gmult[0]) + DBv * np.exp(-mid * gmult[1]))
        lo = np.where(V > 1, mid, lo)
        hi = np.where(V > 1, hi, mid)
    return 0.5 * (lo + hi)

def _I_of_g(g, x0):
    return np.exp(-g * beta_star) * np.exp(-(1 - g * scale) * x0) / (1 - g * scale)

def _optval_core(DAv, DBv, sq):
    DAv = np.maximum(DAv, 0.0); DBv = np.maximum(DBv, 0.0)
    dead = (DAv + DBv) <= 0
    bb = _thresh(DAv, DBv)
    x0 = np.maximum(0.0, (beta_star - bb) / scale)
    g0, g1 = gmult
    if not sq:
        out = (Chat * DAv * _I_of_g(g0, x0) + Chat * DBv * _I_of_g(g1, x0) - np.exp(-x0))
    else:
        out = ((Chat * DAv) ** 2 * _I_of_g(2 * g0, x0) + (Chat * DBv) ** 2 * _I_of_g(2 * g1, x0)
               + 2 * Chat ** 2 * DAv * DBv * _I_of_g(g0 + g1, x0)
               - 2 * Chat * DAv * _I_of_g(g0, x0) - 2 * Chat * DBv * _I_of_g(g1, x0) + np.exp(-x0))
    return np.where(dead, 0.0, np.maximum(out, 0.0))

def _optval_fn(s):
    def f(r):
        r = np.asarray(r, float)
        return _optval_core(DA_i[s](r), DB_i[s](r), sq=False)
    return f

def _optval_sq_fn(s):
    def f(r):
        r = np.asarray(r, float)
        return _optval_core(DA_i[s](r), DB_i[s](r), sq=True)
    return f

optval_fn = [_optval_fn(s) for s in (0, 1)]
optval_g = [optval_fn[s](r_adj) for s in (0, 1)]
optval_sq_g = [_optval_sq_fn(s)(r_adj) for s in (0, 1)]


def integrator(fpts, gpts, rdata):
    integ = (wgts.reshape(1, n_pts) * fpts * gpts).sum(axis=1) / np.sqrt(np.pi)
    interp = interpolate.interp1d(approx.r, integ, fill_value="extrapolate")
    return interp(rdata)


def integ_expy(f, y):
    r_adj3 = sigma_r * (np.sqrt(2) * rshock_nodes.reshape(1, n_pts, 1) + y[None, None, :]) + cond_mean.reshape(len(r_pts), 1, 1)
    fvec = f(r_adj3)
    integ = ((wgts[None, :, None] * fvec).sum(axis=1) * np.exp(y[None, :] ** 2 / 2) / np.sqrt(np.pi))
    return RegularGridInterpolator((r_pts.reshape(-1), y), integ, bounds_error=False, fill_value=None)

y_pts = np.linspace(-0.5, 0.5, 500)
interp_iDAexpy = [integ_expy(DA_i[s], y_pts) for s in (0, 1)]
interp_iDBexpy = [integ_expy(DB_i[s], y_pts) for s in (0, 1)]
interp_iJsexpy = [integ_expy(Jstar_g[s], y_pts) for s in (0, 1)]
interp_iOVexpy = [integ_expy(optval_fn[s], y_pts) for s in (0, 1)]

_psw = np.array([p01, p10])   # switch prob out of regime s


def _mix_series(tab, sreg_t):
    """tab: (2, T) per-destination-regime series; sreg_t: (T,) regime at the conditioning date.
    returns the branch-mixture E over s' for each t."""
    own = np.take_along_axis(tab, sreg_t[None, :], 0)[0]
    oth = np.take_along_axis(tab, (1 - sreg_t)[None, :], 0)[0]
    p = _psw[sreg_t]
    return (1 - p) * own + p * oth


def sdf_compute(N, T, arr_tuple):
    (r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow,
     loadings_mu_taylor, loadings_xi_taylor, loadings_mu_proj, loadings_xi_proj, sreg) = arr_tuple

    st = sreg[1:-1].astype(int)          # regime at the conditioning date for rows of r[1:-1]
    rr = r[1:-1]

    def series(f0, f1, g0=None, g1=None):
        """branch-mixed E_t[f_{s'} g_{s'}] series on the conditioning dates"""
        tab = np.stack([integrator(f0, g0 if g0 is not None else 1, rr),
                        integrator(f1, g1 if g1 is not None else 1, rr)])
        return _mix_series(tab, st)

    i_JJ = series(Jstarvec_g[0], Jstarvec_g[1], Jstarvec_g[0], Jstarvec_g[1])
    i_OVOV = series(optval_g[0], optval_g[1], optval_g[0], optval_g[1])
    i_OVsq = series(optval_sq_g[0], optval_sq_g[1])
    i_J_OV = series(Jstarvec_g[0], Jstarvec_g[1], optval_g[0], optval_g[1])
    i_AA = series(DAvec[0], DAvec[1], DAvec[0], DAvec[1])
    i_AB = series(DAvec[0], DAvec[1], DBvec[0], DBvec[1])
    i_BB = series(DBvec[0], DBvec[1], DBvec[0], DBvec[1])
    i_JA = series(Jstarvec_g[0], Jstarvec_g[1], DAvec[0], DAvec[1])
    i_JB = series(Jstarvec_g[0], Jstarvec_g[1], DBvec[0], DBvec[1])
    i_A_OV = series(DAvec[0], DAvec[1], optval_g[0], optval_g[1])
    i_B_OV = series(DBvec[0], DBvec[1], optval_g[0], optval_g[1])

    term1 = I ** 2 * Chat ** 2 * i_JJ
    term2 = I ** 2 * i_OVOV

    exp_betaA = np.exp(-beta * gmult[0])
    exp_betaB = np.exp(-beta * gmult[1])
    exp_corr_zr = np.exp(-0.5 * sigmaj ** 2 * corr_zj ** 2 * corr_zr ** 2)

    def sdf_loop(t, iter=0):
        s_now = int(sreg[t + 1])
        p_now = _psw[s_now]
        chisp = csr_matrix(chi[t, : t + 1, :])
        col1A = chisp.multiply(exp_betaA[: t + 1, :])
        col1B = chisp.multiply(exp_betaB[: t + 1, :])
        col2 = chisp.multiply(sigmaj[: t + 1, :] * corr_zj[: t + 1, :])
        col3 = chisp.multiply(exp_corr_zr[: t + 1, :])

        # E_t[f_{s'}(r') e^{y xi'}] on the sparse per-project data, branch-mixed at the data level
        integ_y = col2.copy() * corr_zr
        rdata = np.full_like(integ_y.data, r[t + 1])
        def mixed_expy(interp_pair):
            own = interp_pair[s_now]((rdata, integ_y.data))
            oth = interp_pair[1 - s_now]((rdata, integ_y.data))
            return (1 - p_now) * own + p_now * oth
        dataDA = mixed_expy(interp_iDAexpy)
        dataDB = mixed_expy(interp_iDBexpy)
        dataJs = mixed_expy(interp_iJsexpy)
        dataOV = mixed_expy(interp_iOVexpy)
        iDAexpy = integ_y.copy(); iDAexpy.data = dataDA
        iDBexpy = integ_y.copy(); iDBexpy.data = dataDB
        iJsexpy = integ_y.copy(); iJsexpy.data = dataJs
        iOVexpy = integ_y.copy(); iOVexpy.data = dataOV

        # value-cross term: E[V_i V_j] with V = Chat(e^{-b g0} DA_{s'} + e^{-b g1} DB_{s'}), common s'
        SA = np.asarray(col1A.sum(axis=0)).ravel()
        SB = np.asarray(col1B.sum(axis=0)).ravel()
        term3 = (Chat * I * pi) ** 2 * (np.outer(SA, SA) * i_AA[t] + (np.outer(SA, SB) + np.outer(SB, SA)) * i_AB[t]
                                        + np.outer(SB, SB) * i_BB[t])

        # cash-flow covariance cross term (physical, regime-free): identical to baseline
        result4 = kron(col2, col2).copy()
        result4.data = np.exp(result4.data)
        term4 = (Chat * I * pi) ** 2 * result4.sum(axis=0).reshape(N, N).A

        # value x cash-flow cross term: i-side basis a lives inside j's tilted integral
        S3A = np.asarray(col3.multiply(iDAexpy).sum(axis=0)).ravel()
        S3B = np.asarray(col3.multiply(iDBexpy).sum(axis=0)).ravel()
        term5 = (Chat * I * pi) ** 2 * (np.outer(SA, S3A) + np.outer(SB, S3B))

        term6 = 2 * I ** 2 * Chat * i_J_OV[t]
        term7 = I ** 2 * Chat * pi * (SA * (Chat * i_JA[t] + i_A_OV[t]) + SB * (Chat * i_JB[t] + i_B_OV[t]))[None, :]
        term8 = I ** 2 * Chat * pi * np.asarray(col3.multiply(Chat * iJsexpy + iOVexpy).sum(axis=0))

        # diagonal corrections
        term2t = term2[t] + np.diag(I ** 2 * (i_OVsq[t] - i_OVOV[t]) * np.ones((N,)))

        diag3_sub = np.diag(np.asarray(((Chat * I * pi) ** 2 * chi[t, : t + 1, :]
                        * (exp_betaA[: t + 1, :] ** 2 * i_AA[t] + 2 * exp_betaA[: t + 1, :] * exp_betaB[: t + 1, :] * i_AB[t]
                           + exp_betaB[: t + 1, :] ** 2 * i_BB[t])).sum(axis=0)).reshape(N,))
        term3 = term3 - diag3_sub + diag3_sub / pi

        diag4_sub = np.diag(np.asarray(((Chat * I * pi) ** 2 * chi[t, : t + 1, :]
                        * np.exp(sigmaj[: t + 1, :] ** 2 * corr_zj[: t + 1, :] ** 2)).sum(axis=0)).reshape(N,))
        diag4_add = np.diag(np.asarray(((Chat * I) ** 2 * pi * chi[t, : t + 1, :]
                        * np.exp(sigmaj[: t + 1, :] ** 2)).sum(axis=0)).reshape(N,))
        term4 = term4 - diag4_sub + diag4_add

        diag5_sub = (Chat * I * pi) ** 2 * np.diag(np.asarray(
            (col1A.multiply(col3).multiply(iDAexpy) + col1B.multiply(col3).multiply(iDBexpy)).sum(axis=0)).reshape(N,))
        term5 = term5 - diag5_sub + diag5_sub / pi

        ER = np.zeros((N + 1, N + 1))
        ER[1:, 1:] = (term1[t] + term2t + term3 + term4 + term5 + term5.T
                      + term6 + term7 + term7.T + term8 + term8.T)
        ER[1:, 1:] /= np.outer(P[t + 1], P[t + 1])
        ER[0, 0] = np.exp(2 * r[t + 1])
        ER[0, 1:] = (eret[t + 1, :] + 1) * np.exp(r[t + 1])
        ER[1:, 0] = (eret[t + 1, :] + 1) * np.exp(r[t + 1])

        try:
            port = scipy.linalg.solve(ER, np.ones((N + 1, 1)), assume_a="pos").reshape(-1)
        except Exception as e:
            print(f"An error occurred: {e}. Perturbing ER.")
            ER += np.eye(ER.shape[0]) * 1e-6
            try:
                port = scipy.linalg.solve(ER, np.ones((N + 1, 1)), assume_a="pos").reshape(-1)
            except Exception as e:
                print(f"Second attempt failed: {e}. Using fallback value for port.")
                port = np.full((N + 1,), np.nan)

        port /= port.sum()
        sdf_ret = -(port[1:] * (1 + ret[t + 1, :] - np.exp(r[t + 1]))).sum()
        cond_var = ER[1:, 1:] - np.outer(1 + eret[t + 1, :], 1 + eret[t + 1, :])
        max_sr = -(port[1:] * (1 + eret[t + 1, :] - np.exp(r[t + 1]))).sum() / np.sqrt(port[1:] @ (cond_var @ port[1:]))
        return sdf_ret, max_sr, 1 + eret[t + 1, :] - np.exp(r[t + 1]), cond_var, -port[1:]

    return sdf_loop


# ---- legacy aliases for the loadings diagnostics (regime-0 tables; loadings are candidate-factor
# diagnostics only and enter no pricing object) ----
Dvec = DAvec[0] + DBvec[0]
Jstarvec = Jstarvec_g[0]
optval = optval_g[0]
