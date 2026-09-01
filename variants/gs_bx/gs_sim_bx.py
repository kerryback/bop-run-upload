"""Simulator + exact conditional moments for the 2-state-regime GS economy with EXPOSURE TYPES
(gs_solve_reg.py solutions per type: production loads as exp(beta_f x + z)).
Based on gs_sim_reg.py; firm types partition the panel and each firm uses its type's solution tables.

Regime s_t (0 calm, 1 stressed) is an exogenous monthly chain, unpriced.  All decisions and values at
date t use regime-s_t tables; the one-period expectation discounts with the current regime's kernel
M_{s_t} and mixes the two destination-regime payoff tables with (1-p, p) as a COMMON shock (the pair
matrix is the mixture of per-regime outer products, never products of mixtures).  At gmreg=[1,1] this
reproduces gs_sim.py on sol_g00chk exactly.

Interface mirrors gs_sim: create_arrays, create_panel, sdf_compute.  Panel exports TWO conditioning
features: rf_stand (aggregate x) and gam_stand (the regime), for the vector-rf plumbing.
"""
import numpy as np
import pandas as pd
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
_dirs = os.environ.get("GS_BX_SOLDIRS", "sol_reg,sol_bx3,sol_bx5").split(",")
_bx_list = [float(b) for b in os.environ.get("GS_BX_BETAS", "1.0,3.0,5.0").split(",")]
_share = [float(v) for v in os.environ.get("GS_BX_SHARES", "0.34,0.33,0.33").split(",")]
ntypes = len(_dirs)
_sols = [np.load(os.path.join(HERE, d.strip(), "solution.npz")) for d in _dirs]
_s = _sols[0]
xgrid, zgrid, bgrid = _s["xgrid"], _s["zgrid"], _s["bgrid"]
pr_x, pr_z = _s["pr_x"], _s["pr_z"]
Mx_s = _s["Mx"]                                     # (2, x, x')
psw = _s["psw"]                                     # switch prob out of regime s
P_up_ts = [d["P_up"] for d in _sols]; P_down_ts = [d["P_down"] for d in _sols]   # per type: (2, z, x, b)
Q0_ts = [d["Q0"] for d in _sols]; QI_ts = [d["QI"] for d in _sols]; QI_no_ts = [d["QI_no"] for d in _sols]
icut_up_ts = [d["icut_up"] for d in _sols]; icut_dn_ts = [d["icut_dn"] for d in _sols]
b_refin_0_ts = [d["b_refin_0"] for d in _sols]; b_refin_I_ts = [d["b_refin_I"] for d in _sols]
(g, delta, rho_x, sigma_x, rho_z, sigma_z, r, gamma_x, tau, phi,
 kappa_b, xi, imin, imax, sigma_m) = _s["params"]
znum, xnum, bnum = len(zgrid), len(xgrid), len(bgrid)
burnin = 300
alpha_e = 0.2
reg_seed = 909
chars = ["size", "bm", "agr", "roe", "mom", "lev"]
gamma_grid = np.arange(0.5, 1.1, 0.1)
ov = json.loads(os.environ.get("GS_SIM_OVERRIDES", "{}"))
globals().update(ov)

mn = np.linspace(-4 * sigma_m, 4 * sigma_m, 161)
mw = np.exp(-0.5 * (mn / sigma_m) ** 2)
mw /= mw.sum()

def smooth_moments(P):
    Pn = P[..., None] + mn
    pos = Pn > 0
    return (Pn * pos) @ mw, ((Pn * pos) ** 2) @ mw, pos @ mw

# per-(type, destination-regime) eta'-integrated payoff tables
S1_ts, S2_ts, EP0_ts = [], [], []
MP_s = [Mx_s[s] * pr_x for s in (0, 1)]
for f in range(ntypes):
    S1_s, S2_s = [], []
    for s in (0, 1):
        S1u, S2u, _ = smooth_moments(P_up_ts[f][s])
        S1d, S2d, _ = smooth_moments(P_down_ts[f][s])
        S1_s.append(xi * S1u + (1 - xi) * S1d)
        S2_s.append(xi * S2u + (1 - xi) * S2d)
    S1_ts.append(S1_s); S2_ts.append(S2_s)
    EP0_s = []
    for s in (0, 1):
        p = psw[s]
        mix = (1 - p) * S1_s[s] + p * S1_s[1 - s]
        EP0_s.append(np.einsum("xy,iyb->ixb", MP_s[s], np.einsum("ij,jxb->ixb", pr_z, mix)))
    EP0_ts.append(np.stack(EP0_s))
EPI_ts = [g * e for e in EP0_ts]


def _interp_b_at(V, iz, ix, b):
    jj = np.clip(np.searchsorted(bgrid, b) - 1, 0, bnum - 2)
    ww = np.clip((b - bgrid[jj]) / (bgrid[jj + 1] - bgrid[jj]), 0, 1)
    return V[iz, ix, jj] * (1 - ww) + V[iz, ix, jj + 1] * ww


def _interp_b(V, b):
    jj = np.clip(np.searchsorted(bgrid, b) - 1, 0, bnum - 2)
    ww = np.clip((b - bgrid[jj]) / (bgrid[jj + 1] - bgrid[jj]), 0, 1)
    return V[:, :, jj] * (1 - ww) + V[:, :, jj + 1] * ww


def _by_type(tabs_ts, s, zi, xt, b, ftype):
    """dispatch _interp_b_at over firm types: tabs_ts[f][s] is a (z,x,b) table"""
    out = np.empty(len(zi))
    for f in range(ntypes):
        cols = ftype == f
        if cols.any():
            out[cols] = _interp_b_at(tabs_ts[f][s], zi[cols],
                                     xt[cols] if np.ndim(xt) else np.full(cols.sum(), xt), b[cols])
    return out


def create_arrays(N, T, seed_offset=0):
    rng = np.random.default_rng(np.random.randint(0, 2 ** 31))
    rng_reg = np.random.default_rng(reg_seed)
    rng_bx = np.random.default_rng(909 + 1)
    ftype = rng_bx.choice(ntypes, size=N, p=np.array(_share) / np.sum(_share))
    bxf = np.array(_bx_list)[ftype]
    statx = np.linalg.matrix_power(pr_x.T, 500) @ np.full(xnum, 1 / xnum)
    statz = np.linalg.matrix_power(pr_z.T, 500) @ np.full(znum, 1 / znum)
    prob_calm = psw[1] / (psw[0] + psw[1])
    sreg = np.zeros(T + 1, dtype=int)
    sreg[0] = int(rng_reg.random() > prob_calm)
    for t in range(T):
        sreg[t + 1] = 1 - sreg[t] if rng_reg.random() < psw[sreg[t]] else sreg[t]

    ix = np.zeros(T + 1, dtype=int)
    iz = np.zeros((T + 1, N), dtype=int)
    ix[0] = rng.choice(xnum, p=statx)
    iz[0] = rng.choice(znum, size=N, p=statz)
    b = np.zeros((T + 1, N)); k = np.ones((T + 1, N))
    eta = rng.random((T + 1, N)) < xi
    icost = rng.uniform(imin, imax, size=(T + 1, N))
    mshock = rng.choice(mn, size=(T + 1, N), p=mw)
    scale = np.ones((T + 1, N)); bprime = np.zeros((T + 1, N))
    Vcont = np.zeros((T + 1, N)); payoff = np.zeros((T + 1, N))
    default = np.zeros((T + 1, N), dtype=bool)
    invest = np.zeros((T + 1, N), dtype=bool)
    div = np.zeros((T + 1, N))

    for t in range(T + 1):
        s = sreg[t]
        zi, xt = iz[t], ix[t]
        icu = _by_type(icut_up_ts, s, zi, xt, b[t], ftype); icd = _by_type(icut_dn_ts, s, zi, xt, b[t], ftype)
        cut = np.where(eta[t], icu, icd)
        invest[t] = icost[t] <= cut
        bI_star = np.empty(N); b0_star = np.empty(N)
        for f in range(ntypes):
            cols = ftype == f
            bI_star[cols] = b_refin_I_ts[f][s][zi[cols], xt]
            b0_star[cols] = b_refin_0_ts[f][s][zi[cols], xt]
        bp = np.where(eta[t], np.where(invest[t], bI_star, b0_star),
                      np.where(invest[t], b[t] / g, b[t]))
        bprime[t] = bp
        scale[t] = np.where(invest[t], g, 1.0)
        Vc = np.where(invest[t], _by_type(EPI_ts, s, zi, xt, bp, ftype), _by_type(EP0_ts, s, zi, xt, bp, ftype))
        Vcont[t] = Vc
        prod = (1 - tau) * (np.exp(bxf * xgrid[xt] + zgrid[zi]) - delta) - (1 - tau) * b[t]
        dflow = np.where(eta[t],
                         np.where(invest[t],
                                  (1 - kappa_b) * _by_type(QI_ts, s, zi, xt, bp, ftype) - _by_type(QI_no_ts, s, zi, xt, b[t], ftype),
                                  (1 - kappa_b) * _by_type(Q0_ts, s, zi, xt, bp, ftype) - _by_type(Q0_ts, s, zi, xt, b[t], ftype)),
                         0.0)
        div[t] = prod + dflow - invest[t] * icost[t]
        if t == T:
            break
        ix[t + 1] = rng.choice(xnum, p=pr_x[xt])
        for zz in np.unique(zi):
            sel = zi == zz
            iz[t + 1, sel] = rng.choice(znum, size=sel.sum(), p=pr_z[zz])
        s1 = sreg[t + 1]
        Pn = np.where(eta[t + 1],
                      _by_type(P_up_ts, s1, iz[t + 1], ix[t + 1], bp, ftype),
                      _by_type(P_down_ts, s1, iz[t + 1], ix[t + 1], bp, ftype))
        alive = (Pn + mshock[t + 1]) > 0
        default[t + 1] = ~alive
        payoff[t + 1] = scale[t] * np.where(alive, np.maximum(Pn + mshock[t + 1], 0.0), 0.0)
        b[t + 1] = np.where(alive, bp, 0.0)
        k[t + 1] = np.where(alive, k[t] * scale[t], np.maximum(alpha_e * k[t].mean(), 1e-3))
        iz[t + 1] = np.where(alive, iz[t + 1], rng.choice(znum, size=N, p=statz))

    rets = payoff[1:] / Vcont[:-1] - 1.0
    return dict(ix=ix, iz=iz, b=b, k=k, eta=eta, invest=invest, bprime=bprime, scale=scale,
                Vcont=Vcont, payoff=payoff, div=div, default=default, rets=rets, sreg=sreg, ftype=ftype)


def conditional_moments(arr, t, want_matrix=True):
    iz, ixt, bp, sc, Vc = arr["iz"][t], arr["ix"][t], arr["bprime"][t], arr["scale"][t], arr["Vcont"][t]
    s = int(arr["sreg"][t]); p = psw[s]
    N = len(iz)
    px = pr_x[ixt]
    mx = Mx_s[s][ixt]
    ftype = arr["ftype"]
    # per-destination-regime payoff tables at each firm's coupon, per firm type
    A_b, A2_b = [], []
    for sb in (s, 1 - s):
        A = np.empty((N, len(xgrid))); A2 = np.empty((N, len(xgrid)))
        for f in range(ntypes):
            cols = np.flatnonzero(ftype == f)
            if len(cols):
                S1b = _interp_b(S1_ts[f][sb], bp[cols])
                S2b = _interp_b(S2_ts[f][sb], bp[cols])
                A[cols] = np.einsum("fz,zxf->fx", pr_z[iz[cols]], S1b, optimize=True)
                A2[cols] = np.einsum("fz,zxf->fx", pr_z[iz[cols]], S2b, optimize=True)
        A_b.append(A); A2_b.append(A2)
    wts = (1 - p, p)
    ER1 = sc * sum(w * (A @ px) for w, A in zip(wts, A_b)) / Vc
    euler = sc * sum(w * (A @ (mx * px)) for w, A in zip(wts, A_b)) / Vc
    out = {"eret": ER1 - 1.0, "euler": euler}
    if want_matrix:
        ERR = np.zeros((N, N))
        diag = np.zeros(N)
        for w, A, A2 in zip(wts, A_b, A2_b):
            W = (sc[:, None] * A) / Vc[:, None]
            ERR += w * ((W * px[None, :]) @ W.T)
            diag += w * (sc ** 2) * (A2 @ px) / Vc ** 2
        np.fill_diagonal(ERR, diag)
        out["ERR"] = ERR
    Ex = px @ xgrid; Vx = px @ (xgrid - Ex) ** 2
    out["A_x"] = (sc * sum(w * (A @ (px * (xgrid - Ex))) for w, A in zip(wts, A_b))) / Vc / max(Vx, 1e-16)
    return out


def create_panel(N, T, arr):
    k, b, iz, ix = arr["k"], arr["b"], arr["iz"], arr["ix"]
    P_ex = arr["Vcont"]
    rets = arr["rets"]
    bxf = np.array(_bx_list)[arr["ftype"]]
    opcf = (1 - tau) * (np.exp(bxf[None, :] * xgrid[ix][:, None] + zgrid[iz]) - delta)
    Ax = np.zeros((T, N))
    for t in range(T):
        Ax[t] = conditional_moments(arr, t, want_matrix=False)["A_x"]
    df = pd.DataFrame({
        "firmid": np.repeat(range(N), T),
        "month": np.tile(range(T), N),
        "mve": (k[:-1] * P_ex[:-1]).T.reshape(N * T),
        "book": k[:-1].T.reshape(N * T),
        "op_cash_flow": (k[:-1] * opcf[:-1]).T.reshape(N * T),
        "ret": rets.T.reshape(N * T),
        "default": arr["default"][:-1].T.reshape(N * T),
        "lev": (b[:-1] / np.maximum(k[:-1] * P_ex[:-1], 1e-12)).T.reshape(N * T),
        "A_1_taylor": Ax.T.reshape(N * T),
        "A_1_proj": Ax.T.reshape(N * T),
    })
    df.set_index(["firmid", "month"], inplace=True)
    df["roe"] = df.groupby("firmid", group_keys=False).apply(lambda d: (d.op_cash_flow / d.book).shift())
    df["bm"] = df.book / df.mve
    df["cumret"] = df.groupby("firmid", group_keys=False).ret.apply(lambda x: (1 + x).cumprod())
    df["mom"] = df.groupby("firmid", group_keys=False).cumret.apply(lambda x: x.shift(2) / x.shift(13) - 1)
    df["agr"] = df.groupby("firmid", group_keys=False).book.apply(lambda x: x.pct_change(fill_method=None))
    df.index = df.index.swaplevel()
    df.sort_index(level=["month", "firmid"], inplace=True)
    df = df.drop(columns=["book", "cumret", "op_cash_flow"])
    df.reset_index(inplace=True)
    df["ret"] -= (np.exp(r) - 1)
    df = df.rename(columns={"ret": "xret"})
    prob_calm = psw[1] / (psw[0] + psw[1])
    xser = pd.DataFrame({"month": range(T),
                         "rf_stand": (xgrid[ix[:T]] - 0.0) / (4 * xgrid.std()),
                         "gam_stand": arr["sreg"][:T] - (1 - prob_calm)})
    df = df.merge(xser, on="month")
    df = df[df.month > burnin - 1]
    return df


def sdf_compute(N, T, arr):
    def sdf_loop(t, iter=0):
        m = conditional_moments(arr, t + 1)
        eret, ERRr = m["eret"], m["ERR"]
        ER = np.zeros((N + 1, N + 1))
        ER[1:, 1:] = ERRr
        ER[0, 0] = np.exp(2 * r)
        ER[0, 1:] = (1 + eret) * np.exp(r)
        ER[1:, 0] = (1 + eret) * np.exp(r)
        import scipy.linalg
        try:
            port = scipy.linalg.solve(ER, np.ones((N + 1, 1)), assume_a="pos").reshape(-1)
        except Exception:
            ER += np.eye(N + 1) * 1e-8 * np.trace(ER) / N
            port = scipy.linalg.solve(ER, np.ones((N + 1, 1))).reshape(-1)
        port /= port.sum()
        rets = arr["rets"][t + 1]
        sdf_ret = -(port[1:] * (1 + rets - np.exp(r))).sum()
        cond_var = ERRr - np.outer(1 + eret, 1 + eret)
        max_sr = -(port[1:] * (1 + eret - np.exp(r))).sum() / np.sqrt(port[1:] @ (cond_var @ port[1:]))
        return sdf_ret, max_sr, 1 + eret - np.exp(r), cond_var, -port[1:]
    return sdf_loop
