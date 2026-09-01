import numpy as np
import pandas as pd
from scipy import interpolate
from parameters_kp14 import *
from parameters_kp14 import _coef_at
from joblib import Parallel, delayed

import scipy.special
from scipy.sparse import csr_matrix, diags, kron, vstack

# sdf_compute( ) computes second moment of returns after loading integrals computed in integ_kp14

### STOPPED FOR NOW -- should check everything carefully again after checking overleaf
## the top level code and sdf_loop run currently 

n_jobs = 7 # number of jobs in parallelized tasks

# per-y-node G functions and integral tables, interpolated in y
import os
_prefix = os.environ.get("KP_VY_PREFIX", "vys")
_names = {"A_mod_lst": "Et_A_mod", "G_up_lst": "Et_G_up", "G_down_lst": "Et_G_down",
          "ep_A_mod_lst": "Et_ep_A_mod", "ep_G_up_lst": "Et_ep_G_up", "ep_G_down_lst": "Et_ep_G_down",
          "A_mod2_lst": "Et_A_mod_sq", "G_up2_lst": "Et_G_up_sq", "G_down2_lst": "Et_G_down_sq",
          "A_mod_G_up_lst": "Et_A_mod_G_up", "A_mod_G_down_lst": "Et_A_mod_G_down"}
_tabs = []           # _tabs[type][y-node][name]
for _f in range(ntypes):
    eps_grid = pd.read_csv(f"G_{_prefix}{_f}.csv").eps.values
    row = []
    for _iy in range(NY):
        d = np.load(f"integ_{_prefix}{_f}_{_iy}.npz")
        row.append({dst: interpolate.interp1d(eps_grid, d[src], fill_value="extrapolate")
                    for src, dst in _names.items()})
    _tabs.append(row)

def _yw(yv):
    iy = int(np.clip(np.searchsorted(y_grid, yv) - 1, 0, NY - 2))
    w = float(np.clip((yv - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy]), 0, 1))
    return iy, w

_FTYPE = [None]      # set by sdf_compute; module-level for the helpers

def _tab_at(name, yv):
    iy, w = _yw(yv)
    ftype = _FTYPE[0]
    def f(e):
        out = np.empty_like(np.asarray(e, float))
        for ff in range(ntypes):
            cols = ftype == ff
            if cols.any():
                out[..., cols] = ((1 - w) * _tabs[ff][iy][name](e[..., cols])
                                  + w * _tabs[ff][iy + 1][name](e[..., cols]))
        return out
    return f

def _coef_vec(yv, scale=None):
    """per-firm A-coefficient vectors at y'=yv, optionally scaled (e.g. by e^{beta_f yv})"""
    ftype = _FTYPE[0]
    N = len(ftype)
    C = [np.empty(N) for _ in range(4)]
    for ff in range(ntypes):
        cols = ftype == ff
        a = _coef_at(yv, ff)
        for k in range(4):
            C[k][cols] = float(a[k])
    if scale is not None:
        C = [c * scale for c in C]
    return C

# compute more integrals analytically
Et_epsm1 = lambda eps: np.exp(-theta_eps*dt)*(eps - 1)
Et_um1 = lambda u: np.exp(-theta_u*dt)*(u - 1)
Et_epsm1_sq = lambda eps: eps*sigma_eps**2/theta_eps*(np.exp( - theta_eps*dt) - np.exp(-2*theta_eps*dt)) + sigma_eps**2/(2*theta_eps) *(1 - np.exp( - theta_eps*dt))**2 + Et_epsm1(eps)**2
Et_um1_sq = lambda u: u*sigma_u**2/theta_u*(np.exp( - theta_u*dt) - np.exp(-2*theta_u*dt)) + sigma_u**2/(2*theta_u) *(1 - np.exp( - theta_u*dt))**2 + Et_um1(u)**2

def _Et_A_sq(eps, u, coef):
    a0, a1, a2, a3 = coef
    return (a0**2 + a1**2*Et_epsm1_sq(eps) + a2**2 * Et_um1_sq(u) + a3**2 * Et_epsm1_sq(eps) * Et_um1_sq(u) +
        2*a0*a1*Et_epsm1(eps) + 2*a0*a2*Et_um1(u) + 2*a0*a3*Et_epsm1(eps)*Et_um1(u) +
        2*a1*a2*Et_epsm1(eps)*Et_um1(u) + 2*a1*a3*Et_epsm1_sq(eps)*Et_um1(u) + 2*a2*a3*Et_epsm1(eps)*Et_um1_sq(u))

def _cc(eps, coef):
    a0, a1, a2, a3 = coef
    Ee, Ee2 = Et_epsm1(eps), Et_epsm1_sq(eps)
    return (a0**2 + a1**2*Ee2 + 2*a0*a1*Ee,
            a0*a2 + a0*a3*Ee + a1*a2*Ee + a1*a3*Ee2,
            a2**2 + a3**2*Ee2 + 2*a2*a3*Ee)

def _Et_A_mod_A_s(eps, u, coef, yv, ebq):
    a0, a1, a2, a3 = coef
    return ebq*((a0 + a2*Et_um1(u))*_tab_at("Et_A_mod", yv)(eps) + (a1 + a3*Et_um1(u))*_tab_at("Et_ep_A_mod", yv)(eps))

def _Et_A_G_s(eps, u, coef, yv, which, ebq):
    a0, a1, a2, a3 = coef
    return ebq*((a0 + a2*Et_um1(u))*_tab_at(f"Et_G_{which}", yv)(eps) + (a1 + a3*Et_um1(u))*_tab_at(f"Et_ep_G_{which}", yv)(eps))

# Covariance computations 
def sdf_compute(N, T, arr_tuple):
    (K, book, op_cashflow, x, z, eps, uj, chi, rate, high, Et_G, EtA, alph, Et_z_alph, price, ret, eret, lambda_f,
     loadings_z_taylor, loadings_x_taylor, loadings_z_proj, loadings_x_proj, yreg, ftype) = arr_tuple
    _FTYPE[0] = ftype
    bvf = type_bv[ftype]
    tt = type_theta[ftype]
    _ghx, _ghw = np.polynomial.hermite.hermgauss(7)
    _ghw = _ghw / np.sqrt(np.pi)
    _ar = np.exp(-kappa_y * dt)
    _sdc = np.sqrt(1 - _ar ** 2)

    Et_z_alph2 = z**(2*alph)*np.exp(2*alph*mu_z*dt + alph*(3*alpha-1)/(1 - alpha)*sigma_z**2*dt)
    Et_x2 = x**2*np.exp(2*mu_x*dt + sigma_x**2*dt)

    def sdf_loop(t, iter=0):  # compute ER at date t+1 (to t+2)
        t = t + 1
        y_now = float(yreg[t])
        yq = _ar * y_now + _sdc * np.sqrt(2) * _ghx         # quadrature nodes for y' 
        Ktalpha = K[:t + 1, t, :]**alpha
        ujt = uj[:t + 1, t, :]
        eps_rep = np.repeat(eps[t, :].reshape((1, N)), t + 1, axis=0)
        _eE = 1 + (eps_rep - 1)*np.exp(-theta_eps*dt)
        _uE = 1 + (ujt - 1)*np.exp(-theta_u*dt)

        def branch(yv):
            """all A/G-dependent pieces of E_t[R_i R_j] * P_i P_j conditional on y' = yv; the
            per-firm value factor e^{beta_f yv} is absorbed into the coefficients and linear tables,
            so squared/cross terms carry the right powers automatically."""
            ebq = np.exp(bvf * yv)                        # (N,)
            coef = _coef_vec(yv, scale=ebq)               # e-scaled per-firm A-coefficients
            _t1 = _tab_at_s = lambda name: (lambda e: ebq * _tab_at(name, yv)(e))     # linear tables x e^{b y'}
            _t2 = lambda name: (lambda e: ebq**2 * _tab_at(name, yv)(e))              # squared/cross tables x e^{2 b y'}
            EtA_b = coef[0] + (_eE - 1)*coef[1] + (_uE - 1)*coef[2] + (_eE - 1)*(_uE - 1)*coef[3]
            part2 = (1 - delta*dt)*np.sum(chi[:t + 1, t, :]*EtA_b*Ktalpha, axis=0)
            EtGb = ((high[t, :] == 0)*lambda_f*((1 - mu_L*dt)*_t1("Et_G_down")(eps[t, :]) + mu_L*dt*_t1("Et_G_up")(eps[t, :])) +
                    (high[t, :] == 1)*lambda_f*((1 - mu_H*dt)*_t1("Et_G_up")(eps[t, :]) + mu_H*dt*_t1("Et_G_down")(eps[t, :])))
            part1 = C*rate[t, :]*dt*_t1("Et_A_mod")(eps[t, :]) + EtGb

            col = csr_matrix(Ktalpha).multiply(csr_matrix(EtA_b * (Ktalpha > 0)))
            result = kron(col, col)
            term1 = (1 - delta*dt)**2*result.sum(axis=0).reshape(N, N).A
            term2 = Et_z_alph[t]*np.outer(part1, part2)
            term2 = term2 + term2.T
            term3 = Et_z_alph2[t]*np.outer(part1, part1)

            # diagonal corrections
            term1_diag = ((1 - delta*dt)*Ktalpha**2*_Et_A_sq(eps_rep, ujt, coef)).sum(axis=0)
            c0, c1, c2 = _cc(eps[t, :], coef)
            v = Et_um1(ujt) * (Ktalpha > 0)
            S1v, S2v = Ktalpha.sum(axis=0), (Ktalpha**2).sum(axis=0)
            Sv, S2vv, S2v2 = (Ktalpha*v).sum(axis=0), (Ktalpha**2*v).sum(axis=0), (Ktalpha**2*v**2).sum(axis=0)
            term2_diag = (1 - delta*dt)**2*(c0*(S1v**2 - S2v) + 2*c1*(S1v*Sv - S2vv) + c2*(Sv**2 - S2v2))
            term3_diag = 2*Et_z_alph[t]*C*rate[t, :]*dt*(1 - delta*dt)*(_Et_A_mod_A_s(eps_rep, ujt, coef, yv, ebq)*Ktalpha).sum(axis=0)
            Et_A_Gb = ((high[t, :] == 0)*lambda_f*((1 - mu_L*dt)*_Et_A_G_s(eps_rep, ujt, coef, yv, "down", ebq) + mu_L*dt*_Et_A_G_s(eps_rep, ujt, coef, yv, "up", ebq)) +
                       (high[t, :] == 1)*lambda_f*((1 - mu_H*dt)*_Et_A_G_s(eps_rep, ujt, coef, yv, "up", ebq) + mu_H*dt*_Et_A_G_s(eps_rep, ujt, coef, yv, "down", ebq)))
            term4_diag = 2*Et_z_alph[t]*(1 - delta*dt)*(Et_A_Gb*Ktalpha).sum(axis=0)
            term5_diag = Et_z_alph2[t]*C**2*rate[t, :]*dt*_t2("Et_A_mod_sq")(eps[t, :])
            Et_G_sqb = ((high[t, :] == 0)*lambda_f**2*((1 - mu_L*dt)*_t2("Et_G_down_sq")(eps[t, :]) + mu_L*dt*_t2("Et_G_up_sq")(eps[t, :])) +
                        (high[t, :] == 1)*lambda_f**2*((1 - mu_H*dt)*_t2("Et_G_up_sq")(eps[t, :]) + mu_H*dt*_t2("Et_G_down_sq")(eps[t, :])))
            term6_diag = Et_z_alph2[t]*Et_G_sqb
            Et_lAGb = ((high[t, :] == 0)*lambda_f*lambda_L*dt*lambda_f*((1 - mu_L*dt)*_t2("Et_A_mod_G_down")(eps[t, :]) + mu_L*dt*_t2("Et_A_mod_G_up")(eps[t, :])) +
                       (high[t, :] == 1)*lambda_f*lambda_H*dt*lambda_f*((1 - mu_H*dt)*_t2("Et_A_mod_G_up")(eps[t, :]) + mu_H*dt*_t2("Et_A_mod_G_down")(eps[t, :])))
            term7_diag = Et_z_alph2[t]*2*C*Et_lAGb
            offdiag = term1 + term2 + term3
            diag = term1_diag + term2_diag + term3_diag + term4_diag + term5_diag + term6_diag + term7_diag
            M = offdiag
            M[np.arange(N), np.arange(N)] = diag
            return M

        Mmix = sum(_ghw[q] * branch(float(yq[q])) for q in range(len(_ghw)))

        # cash-flow adjustment terms (regime-free)
        cf = np.sum(eps[t, :]*ujt*x[t]*Ktalpha*dt, axis=0) * np.exp(bvf * y_now) * tt
        term4 = np.outer(cf/price[t, :], 1 + eret[t, :])
        term4 = term4 + term4.T - np.outer(cf/price[t, :], cf/price[t, :])
        term8_diag = 2*cf/price[t, :]*(1 + eret[t, :]) - (cf/price[t, :])**2
        term4[np.arange(N), np.arange(N)] = term8_diag

        ER = np.zeros((N + 1, N + 1))
        ER[1:, 1:] = Mmix*Et_x2[t]/np.outer(price[t, :], price[t, :]) + term4
        ER[0, 0] = np.exp(2 * r * dt)
        ER[0, 1:] = (eret[t, :] + 1) * np.exp(r * dt)
        ER[1:, 0] = (eret[t, :] + 1) * np.exp(r * dt)

        try:
            port = scipy.linalg.solve(ER, np.ones((N + 1, 1)), assume_a="pos").reshape(-1)
        except Exception as e:
            print(f"An error occurred: {e}. Perturbing ER.")
            ER += np.eye(ER.shape[0]) * 1e-6
            try:
                port = scipy.linalg.solve(ER, np.ones((N + 1, 1))).reshape(-1)
            except Exception as e:
                print(f"Second attempt failed: {e}. Using fallback value for port.")
                port = np.full((N + 1,), np.nan)

        port /= port.sum()
        sdf_ret = -(port[1:] * (1 + ret[t, :] - np.exp(r*dt))).sum()
        cond_var = ER[1:, 1:] - np.outer(1 + eret[t, :], 1 + eret[t, :])
        max_sr = -(port[1:] * (1 + eret[t, :] - np.exp(r*dt))).sum() / np.sqrt(port[1:] @ (cond_var @ port[1:]))
        return sdf_ret, max_sr, 1 + eret[t, :] - np.exp(r*dt), cond_var, -port[1:]

    return sdf_loop
