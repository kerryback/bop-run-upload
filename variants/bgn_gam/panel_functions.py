# %%

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import expon
from vasicek import *
from sdf_compute import *
from loadings_compute import *
from parameters import *

def create_arrays(N, T):

    # per-regime growth-option values and basis-value interpolants come from sdf_compute (Jstar_g, DA_i, DB_i)
    # aggregate price-of-risk regime path s_t (0 = calm, 1 = stressed), separate RNG stream
    rng_gam = np.random.default_rng(gam_seed)
    sreg = np.zeros(T + 1, dtype=int)
    sreg[0] = int(rng_gam.random() > prob_calm)
    for t_ in range(T):
        ps = p01 if sreg[t_] == 0 else p10
        sreg[t_ + 1] = 1 - sreg[t_] if rng_gam.random() < ps else sreg[t_]

    # shocks to log SDF process at dates 1, ..., T
    nu = norm.rvs(size=T)
    # dct["nu"] = nu

    # interest rate at dates 0, ..., T
    rate_shock = norm.rvs(size=T)
    # dct["rate_shock"] = rate_shock

    corr_zr = beta_zr / (sigma_z * sigma_r)
    xi = corr_zr * nu + np.sqrt(1 - corr_zr**2) * rate_shock
    mu = np.sqrt(1 - corr_zr**2) * nu - corr_zr * rate_shock ## can check var(mu) = 1, corr(mu, xi) = 0, corr(mu, nu) = sqrt(1 - rho_zr^2)

    r = np.zeros(T + 1)
    r[0] = rbar
    const = (1 - kappa) * rbar
    for t in range(T):
        r[t + 1] = kappa * r[t] + const + sigma_r * xi[t]

    # beta is TxN
    # systematic risk of cashflows for projects that arrived at 1, ..., T
    beta = -expon.rvs(size=(T, N), loc=-beta_star, scale=scale)
    # dct["beta"] = beta

    # sigmaj is (T-1)xN
    # stdev of cashflows for projects that arrived at 1, ..., T-1
    sigmaj_low = np.abs(beta[:-1, :]) / sigma_z
    sigmaj_u = np.random.uniform(size=(T - 1, N))
    sigmaj = sigmaj_low + sigmaj_u * sigmaj_width * np.abs(Cbar)   # overwritten below if sigmaj_size_elast > 0
    # dct["sigmaj"] = sigmaj

    # corr_zj is (T-1)xN
    # implied correlation between SDF and cash flows
    # for projects that arrived at 1, ..., T-1
    corr_zj = beta[:-1, :] / (sigma_z * sigmaj)
    # dct["corr_zj"] = corr_zj

    # eps is (T-1)x(T-1)xN
    # eps[0, :1, :] is shocks at date 2 for projects that arrived at date 1
    # eps[1, :2, :] is shocks at date 3 for projects that arrived at dates 1 and 2
    # eps[T-2, :, :] is the (T-1)xN matrix of cash flow shocks at date T for
    # projects that arrived at dates 1, ..., T-1
    cash_shocks = np.random.standard_normal((T - 1, T - 1, N))
    # dct["cash_shocks"] = cash_shocks

    term1 = np.zeros_like(cash_shocks)
    for t in range(cash_shocks.shape[0]):
        term1[t, :, :] = nu[t + 1] * corr_zj
    eps = term1 + np.sqrt(1 - corr_zj**2) * cash_shocks
    # dct["eps"] = eps

    # invest[t, :] is N-dim array of investment decisions at date t+1; value uses the two regime bases
    DAv = DA_i[0](r[1:]) * (sreg[1:] == 0) + DA_i[1](r[1:]) * (sreg[1:] == 1)
    DBv = DB_i[0](r[1:]) * (sreg[1:] == 0) + DB_i[1](r[1:]) * (sreg[1:] == 1)
    DAv = DAv.reshape(T, 1); DBv = DBv.reshape(T, 1)
    invest = 1 * (Chat * (np.exp(-beta * gmult[0]) * DAv + np.exp(-beta * gmult[1]) * DBv) > 1)

    # chi is TxTxN
    # chi[t, :, :] is a TxN matrix
    # rows 0, ..., t of chi[t, :, :] specify which projects are alive at t+1 of those projects
    # that arrived at 1, ..., t+1
    # start with row t of chi[t, :, :] being investment decisions at t+1
    chi = np.zeros((T, T, N))
    chi[range(T), range(T), :] = invest

    # chi[0, 0, :] is investment decisions at date 1
    # chi[0, t, :] = 0 for t>0

    # chi[1, :, :] should have possibly nonzero rows 0 and 1.
    # row 1 = investment decisions at date 2
    # row 0 should be row 0 of chi[0, :, :] (i.e., date-0 investment decisions)
    # multplied by survive/die indicator from alive array

    # in general, chi[t, :t, :] tells us which of the projects that arrived at
    # dates 1, ..., t are alive at t+1
    # and chi[t, t, :] is investment decisions at t1

    # we need one row of survive/die indicators to compute chi[1, :, :] from chi[0, :, :]
    # we need T-1 rows of indicators to compute chi[T-1, : :] from chi[T-2, :, :]
    # we'll just generate T-1 rows T-1 times
    alive = np.random.binomial(n=1, size=(T - 1, T - 1, N), p=pi)
    # dct["alive"] = alive

    for t in range(T - 1):
        chi[t + 1, : t + 1, :] = chi[t, : t + 1, :] * alive[t, : t + 1, :]

    # C is (T-1)x(T-1)xN
    # C[0, :1, :] is cash flows at date 2 from projects that arrived at date 1,
    # ignoring whether investments were actually made

    # C[1, :2, :] is cash flows at date 3 from projects that arrived at dates 0 and 1,
    # ignoring whether investments were made and ignoring survival

    # C[T-2, :, :] is cash flows at T from projects that arrived at 1, ..., T-1
    C = np.exp(Cbar - 0.5 * sigmaj**2 + eps * sigmaj)

    # op_cash_flow is (T-1)xN
    # op_cash_flow[t, :] is N-dim array of project cash flows at date t+2
    # these are from projects that arrived at 1, ..., t
    op_cash_flow = np.zeros((T - 1, N))
    for t in range(T - 1):
        op_cash_flow[t, :] = I * (C[t, : t + 1, :] * chi[t + 1, : t + 1, :]).sum(axis=0)

    # free_cash_flow is TxN
    # free_cash_flow[t, :] is N-dim array of free cash flows at t+1
    # equal to cash flow from projects that arrived at 1, ..., t minus investments at t+1
    free_cash_flow = -I * np.float64(invest)
    free_cash_flow[1:, :] += op_cash_flow

    # num is TxN
    # num[t, :] = N-dim array of number of projects alive at t+1
    num = chi.sum(axis=1)

    # book is TxN
    # book[t, :] is N-dim array of book values at t+1
    # book = book value of equity = book value of assets
    # book value is zero for all firms at date 0
    book = I * num

    if sigmaj_size_elast > 0:
        # project arriving at date t+1 (row t of sigmaj, t=0..T-2) sees the firm's live-project count at date t+1
        # before investment, i.e. num[t-1] (zero at t=0); scale the idiosyncratic-vol width by relative firm size
        n_prev = np.zeros((T - 1, N)); n_prev[1:] = num[:T - 2]
        nbar = np.maximum(n_prev.mean(axis=1, keepdims=True), 1e-6)
        vol_scale = np.minimum(((1 + n_prev) / (1 + nbar)) ** (-sigmaj_size_elast), sigmaj_scale_cap)
        sigmaj = sigmaj_low + sigmaj_u * sigmaj_width * np.abs(Cbar) * vol_scale
        corr_zj = beta[:-1, :] / (sigma_z * sigmaj)
        eps = term1 * 0
        for t in range(cash_shocks.shape[0]):
            eps[t, :, :] = nu[t + 1] * corr_zj
        eps = eps + np.sqrt(1 - corr_zj**2) * cash_shocks
        C = np.exp(Cbar - 0.5 * sigmaj**2 + eps * sigmaj)
        op_cash_flow = np.zeros((T - 1, N))
        for t in range(T - 1):
            op_cash_flow[t, :] = I * (C[t, : t + 1, :] * chi[t + 1, : t + 1, :]).sum(axis=0)
        free_cash_flow = -I * np.float64(invest)
        free_cash_flow[1:, :] += op_cash_flow

    # per-basis averages of exp(-beta*g) over live projects
    aA = (chi * np.exp(-beta * gmult[0])).sum(axis=1)
    aB = (chi * np.exp(-beta * gmult[1])).sum(axis=1)
    beta_avgA = np.divide(aA, num, out=np.zeros_like(aA), where=num != 0)
    beta_avgB = np.divide(aB, num, out=np.zeros_like(aB), where=num != 0)

    # A = value of assets in place - TxN
    A = book * Chat * (beta_avgA * DAv + beta_avgB * DBv)

    # G is (T+1)x1: growth options valued with the current regime's J* table
    G = (I * Chat * (Jstar_g[0](r) * (sreg == 0) + Jstar_g[1](r) * (sreg == 1))).reshape(T + 1, 1)

    # P = price of firms at date - (T+1)xN - P[t] is price at date t
    # start with value of growth options which is same for every firm
    P = np.zeros((T + 1, N))
    P += G
    # then add assets in place at dates 1, ..., T
    P[1:, :] += A

    # returns - ret[t] is return from date t to t+1
    ret = (P[1:, :] + free_cash_flow) / P[:-1, :] - 1

    # expected returns - eret[t] is expected return from date t to t+1
    eret = np.zeros((T + 1, N))

    # branch-mixed physical expectations of next-period values (regime is a common shock)
    st = sreg[1:].astype(int)
    def mixed(tabs):
        ser = np.stack([integrator(tabs[0], 1, r[1:]), integrator(tabs[1], 1, r[1:])])
        own = np.take_along_axis(ser, st[None, :], 0)[0]
        oth = np.take_along_axis(ser, (1 - st)[None, :], 0)[0]
        p = np.where(st == 0, p01, p10)
        return ((1 - p) * own + p * oth).reshape(T, 1)
    term1 = I * Chat * mixed(Jstarvec_g)                              # expected growth opportunities
    term2 = pi * book * Chat * (beta_avgA * mixed(DAvec) + beta_avgB * mixed(DBvec))  # assets in place
    term3 = I * mixed(optval_g)                                       # expected value of option to invest
    term4 = Chat * pi * book                                          # expected cash flows of past projects

    s0 = int(sreg[0]); p0 = p01 if s0 == 0 else p10
    def mixed0(tabs):
        return ((1 - p0) * integrator(tabs[s0], 1, r[0]) + p0 * integrator(tabs[1 - s0], 1, r[0]))
    eret[0, :] = (I * Chat * mixed0(Jstarvec_g) + I * mixed0(optval_g)) / P[0] - 1
    eret[1:, :] = (term1 + term2 + term3 + term4) / P[1:] - 1
    eret = eret[:T, :]  # only keep first T rows

    # compute loadings for BGN latent factor model
    loadings_mu_taylor, loadings_xi_taylor = loadings_Taylor(r, P, A, chi, sigmaj, corr_zj, T, N)
    loadings_mu_proj, loadings_xi_proj = loadings_projection(r, P, A, chi, sigmaj, corr_zj, T, N)

    return (r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow,
            loadings_mu_taylor, loadings_xi_taylor, loadings_mu_proj, loadings_xi_proj, sreg)


def create_panel(N, T, arr_tuple):

    (r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow,
     loadings_mu_taylor, loadings_xi_taylor, loadings_mu_proj, loadings_xi_proj, sreg) = arr_tuple
    r = pd.DataFrame(
        {
            "month": range(1, T), 
            "rf": r[1:-1], 
            "gam_stand": sreg[1:-1] - (1 - prob_calm),
            "f_mu_true": sigma_z*np.sqrt(1 - corr_zr**2) + mu[1:], 
            "f_xi_true": sigma_z*corr_zr + xi[1:]
        }
    )

    df = pd.DataFrame(
        {
            "firmid": np.repeat(range(N), T),
            "month": np.tile(range(T), N),
          
            # market and book are at dates 0, ..., T-1
            "mve": np.reshape(P[:-1, :].T, (N * T,)),
            "book": np.reshape(np.concatenate((np.zeros((1, N)), book[:-1])).T, (N * T,)),
            "op_cash_flow": np.reshape(
                np.concatenate((np.zeros((1, N)), op_cash_flow)).T, (N * T,)
            ),
            "ret": (ret.T).reshape(
                N * T,
            ),
            "A_1_taylor": (loadings_mu_taylor.T).reshape(
                N * T,
            ),
            "A_2_taylor": (loadings_xi_taylor.T).reshape(
                N * T,
            ),
            "A_1_proj": (loadings_mu_proj.T).reshape(
                N * T,
            ),
            "A_2_proj": (loadings_xi_proj.T).reshape(
                N * T,
            )
        }
    )
    df.set_index(["firmid", "month"], inplace=True)

    # roe = cash flow over beginning of month book equity
    # roe at date 0 = NAN
    # roe at date 1 = cash flow from 0 to 1 divided by date 0 book equity
    df["roe"] = df.groupby("firmid", group_keys=False).apply(
        lambda d: (d.op_cash_flow / d.book).shift()
    )

    # firms with zero live projects have no meaningful characteristics: NaN them out so the
    # cross-sectional filter drops those firm-months (instead of keeping book=0 rows with bm=0, agr=-1)
    df.loc[df.book <= 0, ["roe"]] = np.nan
    df["bm"] = (df.book / df.mve).where(df.book > 0)
    df["cumret"] = df.groupby("firmid", group_keys=False).ret.apply(
        lambda x: (1 + x).cumprod()
    )
    df["mom"] = df.groupby("firmid", group_keys=False).cumret.apply(
        lambda x: x.shift(2) / x.shift(13) - 1
    )
    df["agr"] = df.groupby("firmid", group_keys=False).book.apply(lambda x: x.pct_change())
    df.index = df.index.swaplevel()
    df.sort_index(level=["month", "firmid"], inplace=True)
    df = df.drop(columns=["book", "cumret", "op_cash_flow"])  
    df.reset_index(inplace=True)
    df = df.merge(r, on="month")
    df.ret -= (np.exp(df.rf) - 1)
    df = df.rename(columns={"ret": "xret"})
    
    rf = r[r.month < burnin].rf
    df['rf_stand'] = (df.rf - rf.mean())/(4*rf.std())
    df = df.drop(columns = ["rf"])

    df = df[df.month > burnin - 1]
    return df

