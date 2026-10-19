"""True conditional moments for bgn_vy: bgn_gam/sdf_compute.py with the priced state y and exposure
types. Every next-period value table is indexed by (type, destination regime) and every conditional
moment mixes the regime branches as a COMMON shock, as before. The y-part of each expectation is
taken FIRST, by Gauss-Hermite quadrature over y' (physical law, y' ~ N(rho_y y, s_y^2)), on a regular
fine (r', y') grid of the value functions; the r'-part SECOND, by Gauss-Hermite over the rate shock,
which turns every table into a function of (c, y) with c = E_t[r'] = kappa r + (1 - kappa) rbar, and
the cash-flow tilt e^{tau xi'} (bgn_gam's integ_expy) is just the shift c -> c + sigma_r tau. So the
tables are 2-D, cheap, and read by bilinear interpolation at run time.

Value functions per type f and regime s' (vasicek_vy.py): DA_f, DB_f (bases), J_f (growth options,
per unit I, Chat_f included), OV_f (expected new-project NPV per unit I), OVsq_f (its square).
Per (f, g) products give the value-cross terms, per (f, b_g) tilts E[F_f(x') e^{b_g y'} e^{tau xi'}]
give the value-x-cash-flow terms, and the cash-flow-x-cash-flow y-factor is closed form:
E[e^{(b_i + b_j) y'}] = g_i g_j e^{b_i b_j s_y^2}, g_i = E[e^{b_i y'}].

At type_bv = [0], gamma_v = 0 every table is bgn_gam's and sdf_loop returns its moments up to the
J* table's precision (validate_bgn_vy.py checks the reduction)."""
import os
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse import csr_matrix, kron
from scipy.interpolate import RegularGridInterpolator
from parameters import *
import vasicek_vy as vv
from vasicek_vy import type_theta, Chat_f, DJ, D_all, optval_core, Ek_mat

_HERE = os.path.dirname(os.path.abspath(__file__))
_Jint = vv.load_jstar(os.path.join(_HERE, jstar_vy_file))      # [f][s] interpolators over (r, y)

# ---- quadrature ------------------------------------------------------------------------------
_hr, _wr = np.polynomial.hermite.hermgauss(int(n_quad_r)); _hr = _hr * np.sqrt(2.0); _wr = _wr / np.sqrt(np.pi)
_hy, _wy = np.polynomial.hermite.hermgauss(int(n_quad_y)); _hy = _hy * np.sqrt(2.0); _wy = _wy / np.sqrt(np.pi)

# ---- the regular fine (r', y') grid of every value function ----------------------------------
_sd_r = sigma_r / np.sqrt(1 - kappa ** 2)
TILT_MAX = 0.5                                                   # |tau| range of the cash-flow tilt
r_fine = np.linspace(rbar - 4.5 * _sd_r * kappa - sigma_r * (_hr.max() + TILT_MAX) - 0.002,
                     rbar + 4.5 * _sd_r * kappa + sigma_r * (_hr.max() + TILT_MAX) + 0.002, 361)
y_fine = np.linspace(-(y_max * rho_y + s_y * _hy.max()) - 0.5, (y_max * rho_y + s_y * _hy.max()) + 0.5, 241)
if CHAIN:
    y_fine = y_axis.copy()                                           # the nodes themselves (their index on a joint chain)
_RF, _YF = np.meshgrid(r_fine, (np.arange(NY) if CHAIN else y_fine), indexing="ij")

DN = [f"D{j}" for j in range(NB)]                               # the NB rate-bin bases
VAL = {}                                                         # VAL[name][f][s'] : (n_rf, n_yf)
for _nm in DN + ["J", "OV", "OVsq"]:
    VAL[_nm] = [[None, None] for _ in range(ntypes)]
for _f in range(ntypes):
    Dv = D_all(_RF, _YF, _f)                                     # the bases do not depend on the (inert) regime
    _YJ = (y_axis[_YF.astype(int)] if CHAIN else _YF)
    for _s in (0, 1):
        for _j in range(NB):
            VAL[DN[_j]][_f][_s] = Dv[_j]
        VAL["J"][_f][_s] = _Jint[_f][_s](np.stack([_RF.ravel(), _YJ.ravel()], 1)).reshape(_RF.shape)
        VAL["OV"][_f][_s] = optval_core(Dv, Chat_f[_f], sq=False)
        VAL["OVsq"][_f][_s] = optval_core(Dv, Chat_f[_f], sq=True)

# ---- step 1: y'-expectations on the regular r' grid, at the 21 reported y nodes ----------------
_yq = rho_y * y_grid[:, None] + s_y * _hy[None, :]              # (NY, n_qy): physical y' nodes
_yq_w = np.clip(np.searchsorted(y_fine, _yq) - 1, 0, len(y_fine) - 2)
_yq_t = (_yq - y_fine[_yq_w]) / (y_fine[_yq_w + 1] - y_fine[_yq_w])


def _at_yq(X):
    """X (n_rf, n_yf) -> X at the y' quadrature nodes: (n_rf, NY, n_qy)"""
    return X[:, _yq_w] * (1 - _yq_t) + X[:, _yq_w + 1] * _yq_t


def ymean(X, tilt_b=0.0, tilt_v=None):
    """E_{y'|y}[X(r', y') e^{tilt_b y'}] on (r_fine, y_grid); on a chain the tilt is the node table tilt_v (BY[g])"""
    if CHAIN:
        return (X * np.exp(tilt_v if tilt_v is not None else tilt_b * y_grid)[None, :]) @ P_y.T   # (n_rf, NY): sum_j P[i,j] X[:, j] e^{B_j}
    w = _wy[None, None, :] * (np.exp(tilt_b * _yq)[None, :, :] if tilt_b else 1.0)
    return (_at_yq(X) * w).sum(-1)


def yprod(X, Y):
    if CHAIN:
        return (X * Y) @ P_y.T
    return (_at_yq(X) * _at_yq(Y) * _wy[None, None, :]).sum(-1)


# ---- step 2: r'-expectation as a function of c = E_t[r'] (+ sigma_r tau for a tilt) -----------
c_grid = np.linspace(kappa * (rbar - 4.5 * _sd_r) + (1 - kappa) * rbar - sigma_r * TILT_MAX - 0.001,
                     kappa * (rbar + 4.5 * _sd_r) + (1 - kappa) * rbar + sigma_r * TILT_MAX + 0.001, 361)
_cq = c_grid[:, None] + sigma_r * _hr[None, :]                  # (n_c, n_qr): r' nodes


def phi(X2):
    """X2 (n_rf, NY) on r_fine -> (n_c, NY): sum_q w_q X2(c + sigma_r h_q, y)"""
    out = np.empty((len(c_grid), NY))
    for j in range(NY):
        out[:, j] = (np.interp(_cq, r_fine, X2[:, j]) * _wr[None, :]).sum(-1)
    return out


class Tab2:
    """bilinear table on (c_grid, y_grid), linear extrapolation"""
    def __init__(self, Z):
        self.f = RegularGridInterpolator((c_grid, y_axis), Z, bounds_error=False, fill_value=None)

    def __call__(self, c, y):
        c = np.asarray(c, float); y = np.asarray(y, float)
        pts = np.stack(np.broadcast_arrays(c, y), -1)
        return self.f(pts.reshape(-1, 2)).reshape(pts.shape[:-1])


def _pair(name_a, name_b):
    return [[[Tab2(phi(yprod(VAL[name_a][f][s], VAL[name_b][g][s]))) for s in (0, 1)]
             for g in range(ntypes)] for f in range(ntypes)]


MEAN = {nm: [[Tab2(phi(ymean(VAL[nm][f][s]))) for s in (0, 1)] for f in range(ntypes)] for nm in DN + ["J", "OV", "OVsq"]}
PAIR = {"JJ": _pair("J", "J"), "OVOV": _pair("OV", "OV"), "J_OV": _pair("J", "OV")}
for _j in range(NB):
    for _l in range(_j, NB):
        PAIR[f"{DN[_j]}{DN[_l]}"] = _pair(DN[_j], DN[_l])         # D_j D_l, j <= l (symmetric in (f, g) transposed)
    PAIR[f"J{DN[_j]}"] = _pair("J", DN[_j]); PAIR[f"{DN[_j]}_OV"] = _pair("OV", DN[_j])
# tilted: TILT[nm][f][g][s'] = E[F_f(x') e^{b_g y'} e^{tau xi'}] as a function of (c + sigma_r tau, y), times e^{tau^2/2}
TILT = {nm: [[[Tab2(phi(ymean(VAL[nm][f][s], tilt_b=float(type_bv[g]), tilt_v=(BY[g] if CHAIN else None)))) for s in (0, 1)]
              for g in range(ntypes)] for f in range(ntypes)] for nm in DN + ["J", "OV"]}

_psw = np.array([p01, p10])


def mix_mean(nm, f, c, y, s):
    """branch-mixed physical E_t[F_f(x')] at conditioning (c, y, s); c, y, s arrays of one shape"""
    s = np.asarray(s, int); p = _psw[s]
    own = np.where(s == 0, MEAN[nm][f][0](c, y), MEAN[nm][f][1](c, y))
    oth = np.where(s == 0, MEAN[nm][f][1](c, y), MEAN[nm][f][0](c, y))
    return (1 - p) * own + p * oth


def _mix_tab(tabs, c, y, s):
    p = _psw[s]
    return (1 - p) * tabs[s](c, y) + p * tabs[1 - s](c, y)


def sdf_compute(N, T, arr_tuple):
    (r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow,
     l1, l2, l3, l4, sreg, yreg, ftype) = arr_tuple
    ft = np.asarray(ftype, int)
    th = Chat_f[ft]                                              # per-firm cash-flow level
    bvf = type_bv[ft]
    FT = np.ix_(ft, ft)
    exp_beta = [np.exp(-beta * g_bins[j]) for j in range(NB)]    # e^{-beta g_j} per project, (T, N) each
    exp_corr_zr = np.exp(-0.5 * sigmaj ** 2 * corr_zj ** 2 * corr_zr ** 2)
    pair_bb = np.exp(s_y ** 2 * np.outer(type_bv, type_bv))     # (ntypes, ntypes)
    if CHAIN:
        # E[e^{(b_i + b_j) y'} | node]: (NY, ntypes, ntypes), exact on the chain
        PAIRY = np.stack([(P_y @ np.exp(BY[fi] + BY[fj])) for fi in range(ntypes) for fj in range(ntypes)], 1).reshape(NY, ntypes, ntypes)
        IY = np.asarray(arr_tuple[-2], int) if np.asarray(arr_tuple[-2]).dtype.kind in "iu" else np.argmin(np.abs(np.asarray(arr_tuple[-2])[:, None] - y_grid[None, :]), 1)

    def sdf_loop(t, iter=0):
        s_now = int(sreg[t + 1]); p_now = _psw[s_now]
        rr = float(r[t + 1]); yy = float(y_axis[int(yreg[t + 1])] if CHAIN else yreg[t + 1])
        c0 = kappa * rr + (1 - kappa) * rbar
        # scalar tables at the conditioning state, (ntypes, ntypes) or (ntypes,)
        PR = {k: np.array([[_mix_tab(PAIR[k][f][g], c0, yy, s_now) for g in range(ntypes)] for f in range(ntypes)]) for k in PAIR}
        OVsq = np.array([_mix_tab(MEAN["OVsq"][f], c0, yy, s_now) for f in range(ntypes)])
        if CHAIN:
            Gy = PAIRY[int(IY[t + 1])][FT]                       # (N, N)
        else:
            gy = Ephys_exp(type_bv, yy)                          # (ntypes,) E[e^{b y'}]
            Gy = (np.outer(gy, gy) * pair_bb)[FT]                # (N, N)

        chisp = csr_matrix(chi[t, : t + 1, :])
        col1 = [chisp.multiply(exp_beta[j][: t + 1, :]).tocsr() for j in range(NB)]
        col2 = chisp.multiply(sigmaj[: t + 1, :] * corr_zj[: t + 1, :]).tocsr()
        col3 = chisp.multiply(exp_corr_zr[: t + 1, :]).tocsr()
        tau = col2.data * corr_zr                                # per live-project tilt
        colj = col2.indices                                      # the project's firm
        fj = ft[colj]
        cdat = c0 + sigma_r * tau
        etau = np.exp(0.5 * tau ** 2)

        def tilted(nm, fi):
            """sparse matrix (like col2) of E[F_fi(x') e^{b_fj y'} e^{tau xi'}] per live project"""
            vals = np.empty(len(tau))
            for g in range(ntypes):
                m = fj == g
                if m.any():
                    tabs = TILT[nm][fi][g]
                    vals[m] = ((1 - p_now) * tabs[s_now](cdat[m], yy) + p_now * tabs[1 - s_now](cdat[m], yy)) * etau[m]
            M = col2.copy(); M.data = vals
            return M

        S = [np.asarray(c.sum(axis=0)).ravel() for c in col1]
        onehot = np.stack([ft == f for f in range(ntypes)]).astype(float)   # (ntypes, N)

        term1 = I ** 2 * PR["JJ"][FT]
        term2 = I ** 2 * PR["OVOV"][FT] + np.diag(I ** 2 * (OVsq[ft] - np.diag(PR["OVOV"])[ft]))
        thth = np.outer(th, th)
        def PRD(j, l):
            return PR[f"{DN[j]}{DN[l]}"][FT] if j <= l else PR[f"{DN[l]}{DN[j]}"][FT].T
        term3 = np.zeros((N, N)); diag3 = np.zeros(N)
        for j in range(NB):
            for l in range(NB):
                term3 += np.outer(S[j], S[l]) * PRD(j, l)
                iDD = np.diag(PR[f"{DN[min(j, l)]}{DN[max(j, l)]}"])[ft]
                diag3 += np.asarray((chi[t, : t + 1, :] * exp_beta[j][: t + 1, :] * exp_beta[l][: t + 1, :] * iDD).sum(axis=0)).ravel()
        term3 = (I * pi) ** 2 * thth * term3
        term3 = term3 + np.diag((I * pi) ** 2 * th ** 2 * diag3 * (1 / pi - 1))

        result4 = kron(col2, col2).tocsr()
        result4.data = np.exp(result4.data)
        K4 = np.asarray(result4.sum(axis=0)).reshape(N, N)
        term4 = (I * pi) ** 2 * thth * Gy * K4
        diag4_sub = (I * pi) ** 2 * th ** 2 * np.diag(Gy) * np.asarray((chi[t, : t + 1, :] * np.exp(sigmaj[: t + 1, :] ** 2 * corr_zj[: t + 1, :] ** 2)).sum(axis=0)).ravel()
        diag4_add = I ** 2 * pi * th ** 2 * np.diag(Gy) * np.asarray((chi[t, : t + 1, :] * np.exp(sigmaj[: t + 1, :] ** 2)).sum(axis=0)).ravel()
        term4 = term4 + np.diag(diag4_add - diag4_sub)

        term5 = np.zeros((N, N)); diag5 = np.zeros(N); row8 = np.zeros((ntypes, N))
        for fi in range(ntypes):
            for j in range(NB):
                iD = tilted(DN[j], fi)
                S3 = np.asarray(col3.multiply(iD).sum(axis=0)).ravel()
                term5 += np.outer(S[j] * onehot[fi], S3)
                diag5 += onehot[fi] * np.asarray(col1[j].multiply(col3).multiply(iD).sum(axis=0)).ravel()
            iJ = tilted("J", fi); iOV = tilted("OV", fi)
            row8[fi] = np.asarray(col3.multiply(iJ + iOV).sum(axis=0)).ravel()
        term5 = (I * pi) ** 2 * thth * term5
        term5 = term5 + np.diag((I * pi) ** 2 * th ** 2 * diag5 * (1 / pi - 1))

        term6 = I ** 2 * (PR["J_OV"][FT] + PR["J_OV"][FT].T)
        term7 = I * (I * pi) * th[None, :] * sum(S[j][None, :] * (PR[f"J{DN[j]}"] + PR[f"{DN[j]}_OV"])[FT] for j in range(NB))
        term8 = I * (I * pi) * th[None, :] * row8[ft, :]

        ER = np.zeros((N + 1, N + 1))
        ER[1:, 1:] = (term1 + term2 + term3 + term4 + term5 + term5.T + term6 + term7 + term7.T + term8 + term8.T)
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
            except Exception as e2:
                print(f"Second attempt failed: {e2}. Using fallback value for port.")
                port = np.full((N + 1,), np.nan)
        port /= port.sum()
        sdf_ret = -(port[1:] * (1 + ret[t + 1, :] - np.exp(r[t + 1]))).sum()
        cond_var = ER[1:, 1:] - np.outer(1 + eret[t + 1, :], 1 + eret[t + 1, :])
        max_sr = -(port[1:] * (1 + eret[t + 1, :] - np.exp(r[t + 1]))).sum() / np.sqrt(port[1:] @ (cond_var @ port[1:]))
        return sdf_ret, max_sr, 1 + eret[t + 1, :] - np.exp(r[t + 1]), cond_var, -port[1:]

    return sdf_loop
