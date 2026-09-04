
import numpy as np
import pandas as pd
from scipy import interpolate
from parameters_kp14 import *
from parameters_kp14 import _coef_at
from sdf_compute_kp14 import *
from loadings_compute_kp14 import *

# script to simulate panel for discretized Kogan Papanikolaou (2014) model

def create_arrays(N, T):
    # (2 x NY) G tables and NY per-node integral tables, interpolated in y
    import os
    prefix = os.environ.get("KP_VY_PREFIX", "vys")
    _tabnames = {"A_mod_lst": "Et_A_mod", "G_up_lst": "Et_G_up", "G_down_lst": "Et_G_down"}
    G_up_ty, G_down_ty, Et_ty = [], [], []
    for f in range(ntypes):
        G_in = pd.read_csv(f"G_{prefix}{f}.csv")
        eps_grid = G_in.eps.values
        G_up_ty.append([interpolate.interp1d(eps_grid, G_in[f"G_up_y{i}"].values, fill_value="extrapolate") for i in range(NY)])
        G_down_ty.append([interpolate.interp1d(eps_grid, G_in[f"G_down_y{i}"].values, fill_value="extrapolate") for i in range(NY)])
        row = []
        for i in range(NY):
            d = np.load(f"integ_{prefix}{f}_{i}.npz")
            row.append({dst: interpolate.interp1d(eps_grid, d[src], fill_value="extrapolate")
                        for src, dst in _tabnames.items()})
        Et_ty.append(row)

    def at_y(tabs, yv):
        """evaluate a per-node list of interp1d at state y = yv (linear interp between nodes): returns callable"""
        iy = int(np.clip(np.searchsorted(y_grid, yv) - 1, 0, NY - 2))
        w = float(np.clip((yv - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy]), 0, 1))
        return lambda e: (1 - w) * tabs[iy](e) + w * tabs[iy + 1](e)

    # OU path of the aggregate price-of-risk state (separate RNG stream)
    rng_gam = np.random.default_rng(gam_seed)
    ar = np.exp(-kappa_y * dt)
    sd_c = np.sqrt((1 - ar ** 2))                     # stationary sd is 1
    yreg = np.zeros(T + 1)
    yreg[0] = rng_gam.standard_normal()
    for t_ in range(T):
        yreg[t_ + 1] = ar * yreg[t_] + sd_c * rng_gam.standard_normal()

    # quadrature nodes/weights for E over y' (per current y)
    ghx, ghw = np.polynomial.hermite.hermgauss(7)
    ghw = ghw / np.sqrt(np.pi)
    yq = ar * yreg[:, None] + sd_c * np.sqrt(2) * ghx[None, :]      # (T+1, 7)


    # simulates length x N matrix of Cox-Ingersoll-Ross processes
    # each column is a separate process for each firm
    # dr = kappa(theta - r) dt + sigma dB
    def sim_cir(kappa, theta, sigma, length = T+1, init = None):

        r = np.zeros((length, N))
        # Draw r0 from stationary Gamma distribution
        if init == None:
            shape = 2 * kappa * theta / sigma ** 2
            scale = sigma ** 2 / (2 * kappa)
            r[0,:] = np.random.gamma(shape, scale, size = (N,))
        else:
            r[0,:] = 1

        c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
        d = 4 * kappa * theta / sigma ** 2
        for t in range(1, length):
            lambda_ = (4 * kappa * np.exp(-kappa * dt) * r[t - 1, :]) / (sigma ** 2 * (1 - np.exp(-kappa * dt)))
            z = np.random.noncentral_chisquare(df=d, nonc=lambda_)
            r[t, :] = c * z

        return r

    # simulates length T+1 vector of GBM
    # dS/S = mu dt + sigma dB
    def sim_gbm(mu, sigma, S0):
        S = np.zeros((T+1, ))
        S[0] = S0

        # Simulate Brownian increments
        dW = np.random.normal(0, np.sqrt(dt), size=T)
        W = np.cumsum(dW)  
        S[1:] = S0 * np.exp((mu - 0.5 * sigma**2) * np.arange(1, T+1)*dt + sigma * W)

        return S.reshape([-1, 1])

    # simulates (T+1)xN matrix of regime switching project rates
    # and simulates TXN matrix of project arrivals
    # each column corresponds to a firm n with average rate lambda_f[n]
    def sim_arrivals(lambda_f):

        state = np.zeros((T+1, N))
        state[0,:] = np.random.binomial(1, prob_H, size = (N, ))  # 1 = high, 0 = low
        for t in range(1, T + 1):
            curr = state[t - 1,:]
            switch_prob = np.where(curr == 1, mu_L * dt, mu_H * dt)
            state[t,:] =  np.where(np.random.random((N,)) < switch_prob, 1 - curr, curr)

        rate = lambda_L + state*(lambda_H - lambda_L)
        
        rate*= lambda_f

        arrivals = np.zeros((T+1, N))
        arrivals[0,:] = 0
        arrivals[1:,:] = (np.random.random((T, N)) < dt*rate[:-1,:]).astype(int)

        return arrivals, state

    # lambda_f[n] is (average) project arrival for firm n
    lambda_f = mu_lambda*delta - sigma_lambda*delta*np.log(np.random.uniform(size = (N,)))

    # eps[t, n] is value of eps shock at time t for firm n
    eps = sim_cir(theta_eps, 1, sigma_eps)

    # x[t], z[t] are x[t], z[t] shocks at time t
    x = sim_gbm(mu_x, sigma_x, 0.5) # not sure what initial value should be
    z = sim_gbm(mu_z, sigma_z, 0.5)

    # firm exposure types on the priced OU factor y
    rng_bx = np.random.default_rng(bx_seed)
    ftype = rng_bx.choice(ntypes, size=N, p=type_share)
    bvf = type_bv[ftype]                                  # (N,)
    tt = type_theta[ftype]                                # (N,)
    ebv = np.exp(bvf[None, :] * yreg[:, None])            # (T+1, N): e^{beta_f y_t}

    def _coefs_ty(yv_col):
        # per-(date, firm) A-coefficients at the y-path (yv_col: (T+1,) or scalar)
        C = [np.empty((np.size(yv_col), N)) for _ in range(4)]
        for f in range(ntypes):
            cols = ftype == f
            a = _coef_at(yv_col, f)
            for k in range(4):
                C[k][:, cols] = np.asarray(a[k]).reshape(-1, 1)
        return C
    def _Aval(C, ep, u):
        return C[0] + (ep - 1) * C[1] + (u - 1) * C[2] + (ep - 1) * (u - 1) * C[3]

    # arrivals[t, n] is whether a project arrives at date t for firm n
    # high[t, n] is whether the arrival rate of projects for firm n are in high state at date t
    arrivals, high = sim_arrivals(lambda_f)

    # chi[t1, t2, n] is whether project initiated at time t1 is alive at time t2 for firm n
    chi = np.zeros((T + 1, T + 1, N)) 
    chi[range(T+1), range(T+1),:] = arrivals
    chi = np.cumsum(chi, axis = 1)

    # alive[t1, t2, n] is equal to 0 if at time t2 the project which arrived at date t1 of firm n is dead 
    alive = np.ones((T + 1, T + 1, N))
    alive[chi == 1] = (np.random.random((T+1, T+1, N)) >= dt*delta)[chi == 1] 
    alive[:, 1:,:] = np.cumprod(alive[:, :-1, :], axis = 1)  # shift is because projects can't be dead in first period they exist

    chi = chi* alive # adjust chi for project death

    # Kj[t1, k] is capital allocated at time t1 by firm k if a project were to be initialized
    _Cown = _coefs_ty(yreg)
    Kj = ((alpha*z*_Aval(_Cown, eps, 1.0))**(1/(1 - alpha))) 

    # K[t1, t2, n] is capital allocated at t2 to project arriving at time t1 for firm n (0 if project is dead)
    K = chi*Kj[:, np.newaxis, :] 

    # book[t, n] is the book value of firm n at date t (see footnote 12 in KP14)
    book = x*ebv/z*K.sum(axis = 0)

    # uj[t1, t2, n] is uj process at time t2 for project arriving at time t1 for firm n
    uj = np.zeros((T+1, T+1, N)) 
    for t in range(T+1):
        uj[t, t:, :] = sim_cir(theta_u, 1, sigma_u, length = T +1 - t, init = 1)
        
    # cashflow[t1, t2, n] is cashflow at time t2 from project arriving at t1 for firm n
    # cashflow from t to t + 1 is flow_t dt, arriving at date t + 1
    cashflow = np.zeros((T+1, T+1, N))
    cashflow[:, 1:, :] = eps[np.newaxis, :-1, :]*(x[:-1]*ebv[:-1])[np.newaxis, :, :]*uj[:, :-1,:]*K[:, :-1,:]**alpha* dt * tt[None, None, :] 
    op_cashflow = cashflow # cashflow from operations
    op_cashflow = np.sum(op_cashflow, axis = 0)
    cashflow[range(T+1), range(T+1),:] = -K[range(T+1), range(T+1),:]*x*ebv/z

    # cashflow is overwritten by sum_t1 cashflow[t1, t2, n]
    # cashflow[t2, n] is total cashflows from all projects at time t2 for firm n
    cashflow = np.sum(cashflow, axis = 0)

    # VAP[t1, t2, n] is value at time t2 of project arriving at t1 for firm n
    VAP = np.zeros((T+1, T+1, N))
    VAP = (x*ebv)[np.newaxis, :, :]*K**alpha * _Aval([c[None, :, :] for c in _Cown], eps[np.newaxis, :, :], uj)

    # VAP is overwritten by sum_t1 VAP[t1, t2, n]
    # VAP[t2, n] is total value of assets in place at time t2 for firm n
    VAP = np.sum(VAP, axis = 0)

    # PVGO[t, n] is PVGO for firm n at date t
    # recall G_func.csv doesn't include lambda_f
    _Gd = np.empty_like(eps); _Gu = np.empty_like(eps)
    for f in range(ntypes):
        cols = ftype == f
        _Gd[:, cols] = np.stack([at_y(G_down_ty[f], yreg[t_])(eps[t_, cols]) for t_ in range(eps.shape[0])])
        _Gu[:, cols] = np.stack([at_y(G_up_ty[f], yreg[t_])(eps[t_, cols]) for t_ in range(eps.shape[0])])
    PVGO = z**(alpha/(1 - alpha))*x*ebv*lambda_f*(_Gd*(high == 0) + _Gu*(high == 1)) 

    # price[t] is price at date t
    price = VAP + PVGO

    # rets[t] is returns from date t to t+1
    rets = (price[1:,:] + cashflow[1:,:]) /price[:-1,:] - 1

    # erets[t, n] is E_t[R_{t+1}] for firm n
    # compute 4 terms from the KP14.tex overleaf
    Et_x = np.exp(mu_x*dt)*x
    # E_t[e^{beta_f y'} ...] handled inside the y'-quadrature with per-type exp factors
    ebq = np.exp(bvf[None, None, :] * yq[:, :, None])      # (T+1, 7, N): e^{beta_f y'_q}
    alph = alpha/(1 - alpha)
    Et_z_alph = z**alph*np.exp(alph*mu_z*dt + 0.5*alph*(2*alpha-1)/(1 - alpha)*sigma_z**2*dt)

    # E_t over the gamma-regime at t+1: mix the regime-s' A functions with the switching probabilities
    _eE = 1 + (eps[np.newaxis, :, :] - 1)*np.exp(-theta_eps*dt)
    _uE = 1 + (uj - 1)*np.exp(-theta_u*dt)
    # E over y' via Gauss-Hermite quadrature (7 nodes, common shock)
    EtA = 0.0
    for q in range(len(ghw)):
        Cq = _coefs_ty(yq[:, q])
        EtA = EtA + ghw[q] * ebq[None, :, q, :] * _Aval([c[None, :, :] for c in Cq], _eE, _uE)
    term1 = Et_x*(1 - delta*dt)*np.sum(chi*EtA*K**alpha, axis = 0)

    def _mixG(name, e):
        """E over y' of e^{beta_f y'} times the Et-table `name`, per firm type"""
        out = np.zeros_like(e)
        for f in range(ntypes):
            cols = ftype == f
            for t_ in range(e.shape[0]):
                row = 0.0
                for q in range(len(ghw)):
                    row = row + ghw[q] * np.exp(type_bv[f] * yq[t_, q]) * at_y([tb[name] for tb in Et_ty[f]], yq[t_, q])(e[t_, cols])
                out[t_, cols] = row
        return out
    Et_G = ((high == 0)*lambda_f*((1 - mu_H*dt)* _mixG("Et_G_down", eps) + mu_H*dt* _mixG("Et_G_up", eps)) + 
        (high == 1)*lambda_f*((1 - mu_L*dt)* _mixG("Et_G_up", eps) + mu_L*dt * _mixG("Et_G_down", eps)))

    term2 = Et_z_alph*Et_x*Et_G
    term3 = np.sum(chi*eps[np.newaxis, :, :]*uj*(x*ebv)[np.newaxis, :, :]*K**alpha*dt*tt[None, None, :], axis = 0)

    rate = lambda_f*(lambda_L + (lambda_H - lambda_L)*high)
    term4 = rate*dt*C*Et_z_alph*Et_x*_mixG("Et_A_mod", eps)

    erets = (term1 + term2  + term3 + term4) /price - 1

    # diagnostic loadings columns: evaluate with regime-selected tables (they enter no pricing objects)
    _mid = NY // 2   # diagnostics only: evaluate loadings at the median-y tables
    loadings_z_taylor, loadings_x_taylor = loadings_Taylor(K, x, z, eps, uj, rate, high, lambda_f, price, G_up_ty[0][_mid], G_down_ty[0][_mid])
    loadings_z_proj, loadings_x_proj = loadings_projection(K, x, z, eps, uj, rate, high, lambda_f, price, erets, Et_ty[0][_mid]["Et_G_up"], Et_ty[0][_mid]["Et_G_down"], Et_ty[0][_mid]["Et_A_mod"])


    arr_tuple = (K, book, op_cashflow, x, z, eps, uj, chi, rate, high, Et_G, EtA, alph, Et_z_alph, price,
                 rets, erets, lambda_f, loadings_z_taylor, loadings_x_taylor, loadings_z_proj, loadings_x_proj, yreg, ftype)
    return arr_tuple


def create_panel(N, T, arr_tuple):

    (K, book, op_cashflow, x, z, eps, uj, chi, rate, high, Et_G, EtA, alph, Et_z_alph, P, ret, eret, lambda_f,
     loadings_z_taylor, loadings_x_taylor, loadings_z_proj, loadings_x_proj, yreg, ftype) = arr_tuple
    
    df = pd.DataFrame(
        {
            "firmid": np.repeat(range(N), T),
            "month": np.tile(range(T), N),
          
            # market and book are at dates 0, ..., T-1
            "mve": np.reshape(P[:-1,:].T, (N * T,)),
            "book": np.reshape(book[:-1,:].T, (N * T)),
            "op_cash_flow": np.reshape(
                np.concatenate((np.zeros((1, N)), op_cashflow[:-2,:])).T, (N * T,)
            ),
            "ret": (ret.T).reshape(
                N * T,
            ),
            "A_1_taylor": (loadings_z_taylor[:-1,:].T).reshape(
                N * T,
            ),
            "A_2_taylor": (loadings_x_taylor[:-1,:].T).reshape(
                N * T,
            ),
            "A_1_proj": (loadings_z_proj[:-1,:].T).reshape(
                N * T,
            ),
            "A_2_proj": (loadings_x_proj[:-1,:].T).reshape(
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
    df.ret -= (np.exp(r*dt) - 1)
    df = df.rename(columns={"ret": "xret"})

    sser = pd.DataFrame({"month": range(T), "rf_stand": yreg[:T] / 4.0})
    df = df.merge(sser, on="month")
    df = df[df.month > burnin - 1]
    return df


#N, T = 100, burnin+10
#arr_tuple = create_arrays(N, T)
#print(create_panel(N, T, arr_tuple).xret.describe())