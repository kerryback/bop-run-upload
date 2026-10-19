"""Panel simulation for bgn_vy: bgn_gam/panel_functions.py with the priced AR(1) state y, firm
exposure types and the per-type (r, y, s) value tables. The firm-level shocks are drawn from the
global numpy stream in the SAME order as bgn_gam, the regime path from default_rng(gam_seed) as
before, the y path from default_rng([gam_seed, y_seed_tag]) and the types from default_rng(bx_seed)
-- so at type_bv = [0], gamma_v = 0 the panel is bgn_gam's up to the J* table's precision."""
import numpy as np
import pandas as pd
from scipy.stats import norm, expon
from parameters import *
import vasicek_vy as vv
from vasicek_vy import type_theta, Chat_f, beta_star, scale
import sdf_compute as sc


def create_arrays(N, T):
    # aggregate price-of-risk regime path s_t (0 = calm, 1 = stressed), separate RNG stream
    rng_gam = np.random.default_rng(gam_seed)
    sreg = np.zeros(T + 1, dtype=int)
    sreg[0] = int(rng_gam.random() > prob_calm)
    for t_ in range(T):
        ps = p01 if sreg[t_] == 0 else p10
        sreg[t_ + 1] = 1 - sreg[t_] if rng_gam.random() < ps else sreg[t_]
    # the priced state y: AR(1) with unit stationary sd, its own stream
    rng_y = np.random.default_rng([int(gam_seed), int(y_seed_tag)])
    yreg = np.zeros(T + 1)
    if CHAIN:
        iyp = np.zeros(T + 1, dtype=int)
        iyp[0] = rng_y.choice(NY, p=pi_y)
        for t_ in range(T):
            iyp[t_ + 1] = rng_y.choice(NY, p=P_y[iyp[t_]])
        yreg = y_grid[iyp]
    else:
        yreg[0] = rng_y.standard_normal()
        for t_ in range(T):
            yreg[t_ + 1] = rho_y * yreg[t_] + s_y * rng_y.standard_normal()
    ykey = iyp if CHAIN else yreg                               # what the bases take as their y argument
    # firm exposure types
    rng_bx = np.random.default_rng(bx_seed)
    ftype = rng_bx.choice(ntypes, size=N, p=type_share)
    bvf = type_bv[ftype]                       # (N,)
    thf = Chat_f[ftype]                        # (N,) cash-flow level per firm (Chat x type_theta)

    nu = norm.rvs(size=T)
    rate_shock = norm.rvs(size=T)
    corr_zr = beta_zr / (sigma_z * sigma_r)
    xi = corr_zr * nu + np.sqrt(1 - corr_zr ** 2) * rate_shock
    mu = np.sqrt(1 - corr_zr ** 2) * nu - corr_zr * rate_shock
    r = np.zeros(T + 1)
    r[0] = rbar
    const = (1 - kappa) * rbar
    for t in range(T):
        r[t + 1] = kappa * r[t] + const + sigma_r * xi[t]

    beta = -expon.rvs(size=(T, N), loc=-beta_star, scale=scale)
    sigmaj_low = np.abs(beta[:-1, :]) / sigma_z
    sigmaj_u = np.random.uniform(size=(T - 1, N))
    sigmaj = sigmaj_low + sigmaj_u * sigmaj_width * np.abs(Cbar)
    corr_zj = beta[:-1, :] / (sigma_z * sigmaj)
    cash_shocks = np.random.standard_normal((T - 1, T - 1, N))
    term1 = np.zeros_like(cash_shocks)
    for t in range(cash_shocks.shape[0]):
        term1[t, :, :] = nu[t + 1] * corr_zj
    eps = term1 + np.sqrt(1 - corr_zj ** 2) * cash_shocks

    # per-firm bases at dates 1..T: DA_f(r_t, y_t, s_t) for the firm's own type
    DAv = np.empty((T, N)); DBv = np.empty((T, N))
    for f in range(ntypes):
        cols = ftype == f
        if cols.any():
            da = np.where(sreg[1:] == 0, vv.DA(r[1:], ykey[1:], 0, f), vv.DA(r[1:], ykey[1:], 1, f))
            db = np.where(sreg[1:] == 0, vv.DB(r[1:], ykey[1:], 0, f), vv.DB(r[1:], ykey[1:], 1, f))
            DAv[:, cols] = da[:, None]; DBv[:, cols] = db[:, None]
    invest = 1 * (thf[None, :] * (np.exp(-beta * gmult[0]) * DAv + np.exp(-beta * gmult[1]) * DBv) > 1)

    chi = np.zeros((T, T, N))
    chi[range(T), range(T), :] = invest
    alive = np.random.binomial(n=1, size=(T - 1, T - 1, N), p=pi)
    for t in range(T - 1):
        chi[t + 1, : t + 1, :] = chi[t, : t + 1, :] * alive[t, : t + 1, :]

    # C[t, k, :] cash flows at date t+2 of projects that arrived at k+1: lognormal x type level x e^{b_f y_{t+2}}
    C = np.exp(Cbar - 0.5 * sigmaj ** 2 + eps * sigmaj)
    ylev = (np.exp(BY[ftype[None, :], iyp[2:, None]]) if CHAIN else np.exp(bvf[None, :] * yreg[2:, None])) * type_theta[ftype][None, :]   # (T-1, N) at dates 2..T
    op_cash_flow = np.zeros((T - 1, N))
    for t in range(T - 1):
        op_cash_flow[t, :] = I * (C[t, : t + 1, :] * chi[t + 1, : t + 1, :]).sum(axis=0) * ylev[t]
    free_cash_flow = -I * np.float64(invest)
    free_cash_flow[1:, :] += op_cash_flow
    num = chi.sum(axis=1)
    book = I * num
    if sigmaj_size_elast > 0:
        raise NotImplementedError("sigmaj_size_elast > 0 is not carried into bgn_vy")

    aA = (chi * np.exp(-beta * gmult[0])).sum(axis=1)
    aB = (chi * np.exp(-beta * gmult[1])).sum(axis=1)
    beta_avgA = np.divide(aA, num, out=np.zeros_like(aA), where=num != 0)
    beta_avgB = np.divide(aB, num, out=np.zeros_like(aB), where=num != 0)
    A = book * thf[None, :] * (beta_avgA * DAv + beta_avgB * DBv)                     # assets in place, dates 1..T

    # growth options per firm type at dates 0..T: G = I J_f(r, y, s)
    G = np.empty((T + 1, N))
    pts = np.stack([r, (y_axis[iyp] if CHAIN else yreg)], 1)
    for f in range(ntypes):
        cols = ftype == f
        if cols.any():
            j = np.where(sreg == 0, sc._Jint[f][0](pts), sc._Jint[f][1](pts))
            G[:, cols] = (I * j)[:, None]
    P = np.zeros((T + 1, N))
    P += G
    P[1:, :] += A
    ret = (P[1:, :] + free_cash_flow) / P[:-1, :] - 1

    # expected returns from branch-mixed physical expectations of next-period values
    c_all = kappa * r + (1 - kappa) * rbar
    yval_all = y_axis[iyp] if CHAIN else yreg                    # y on the tables' AXIS (its value, or its node index on a joint chain)
    def mean_by_type(nm, dates):
        out = np.empty((len(dates), N))
        for f in range(ntypes):
            cols = ftype == f
            if cols.any():
                out[:, cols] = sc.mix_mean(nm, f, c_all[dates], yval_all[dates], sreg[dates])[:, None]
        return out
    d1 = np.arange(1, T + 1)
    EJ = mean_by_type("J", d1); EOV = mean_by_type("OV", d1); EDA = mean_by_type("DA", d1); EDB = mean_by_type("DB", d1)
    gy = (np.stack([Ephys_exp_chain(BY[f], iyp[1:]) for f in range(ntypes)], 1)[:, ftype]
          if CHAIN else Ephys_exp(bvf[None, :], yreg[1:, None]))   # (T, N)
    term1 = I * EJ
    term2 = pi * book * thf[None, :] * (beta_avgA * EDA + beta_avgB * EDB)
    term3 = I * EOV
    term4 = thf[None, :] * pi * book * gy
    eret = np.zeros((T + 1, N))
    eret[0, :] = (I * mean_by_type("J", np.array([0])) + I * mean_by_type("OV", np.array([0])))[0] / P[0] - 1
    eret[1:, :] = (term1 + term2 + term3 + term4) / P[1:] - 1
    eret = eret[:T, :]

    zeros = np.zeros((T, N))                                     # the candidate-factor loadings are not carried
    return (r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow,
            zeros, zeros, zeros, zeros, sreg, (iyp if CHAIN else yreg), ftype)


def create_panel(N, T, arr_tuple):
    (r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow,
     l1, l2, l3, l4, sreg, yreg, ftype) = arr_tuple
    yval = y_grid[np.asarray(yreg, int)] if CHAIN else yreg
    rdf = pd.DataFrame({"month": range(1, T), "rf": r[1:-1],
                        "gam_stand": sreg[1:-1] - (1 - prob_calm),
                        "y_stand": yval[1:-1] / 4.0,
                        **({"y2_stand": Y2[np.asarray(yreg, int)][1:-1] / 4.0} if CHAIN2 else {}),
                        "f_mu_true": sigma_z * np.sqrt(1 - corr_zr ** 2) + mu[1:],
                        "f_xi_true": sigma_z * corr_zr + xi[1:]})
    df = pd.DataFrame({
        "firmid": np.repeat(range(N), T),
        "month": np.tile(range(T), N),
        "mve": np.reshape(P[:-1, :].T, (N * T,)),
        "book": np.reshape(np.concatenate((np.zeros((1, N)), book[:-1])).T, (N * T,)),
        "op_cash_flow": np.reshape(np.concatenate((np.zeros((1, N)), op_cash_flow)).T, (N * T,)),
        "ret": (ret.T).reshape(N * T),
        "ftype": np.repeat(np.asarray(ftype, int), T),
        "A_1_taylor": (l1.T).reshape(N * T), "A_2_taylor": (l2.T).reshape(N * T),
        "A_1_proj": (l3.T).reshape(N * T), "A_2_proj": (l4.T).reshape(N * T)})
    df.set_index(["firmid", "month"], inplace=True)
    df["roe"] = df.groupby("firmid", group_keys=False).apply(lambda d: (d.op_cash_flow / d.book).shift())
    df.loc[df.book <= 0, ["roe"]] = np.nan
    df["bm"] = (df.book / df.mve).where(df.book > 0)
    df["cumret"] = df.groupby("firmid", group_keys=False).ret.apply(lambda x: (1 + x).cumprod())
    df["mom"] = df.groupby("firmid", group_keys=False).cumret.apply(lambda x: x.shift(2) / x.shift(13) - 1)
    df["agr"] = df.groupby("firmid", group_keys=False).book.apply(lambda x: x.pct_change())
    df.index = df.index.swaplevel()
    df.sort_index(level=["month", "firmid"], inplace=True)
    df = df.drop(columns=["book", "cumret", "op_cash_flow"])
    df.reset_index(inplace=True)
    df = df.merge(rdf, on="month")
    df.ret -= (np.exp(df.rf) - 1)
    df = df.rename(columns={"ret": "xret"})
    rf = rdf[rdf.month < burnin].rf
    df["rf_stand"] = (df.rf - rf.mean()) / (4 * rf.std())
    df = df.drop(columns=["rf"])
    df = df[df.month > burnin - 1]
    return df
