"""Simulator + exact conditional moments for gs_vy: GS21 with a priced NY-state chain for the
aggregate state y and exposure types (gs_solve_vy.py solutions per type; production loads as
exp(gs_bx x + z + gs_bv_f y + gs_ashift_f)). Based on gs_bx/gs_sim_bx.py: the regime s_t and its
(1-p, p) mixtures become the y-node iy_t and the chain's weights -- PHYSICAL pr_y[iy] for expected
payoffs and second moments, PRICED MPy[iy] for the Euler check -- as a COMMON shock (mixtures of
per-y' outer products, never products of mixtures). At NY = 1 this is gs_sim_bx.py at gmreg [1,1].

Interface mirrors gs_sim_bx: create_arrays, create_panel, sdf_compute, conditional_moments. The panel
exports TWO conditioning features: rf_stand (aggregate x) and y_stand (the priced state y).

Environment: GS_VY_SOLDIRS (comma list), GS_VY_BV (each type's gs_bv, must match the solves),
GS_VY_SHARES. GS_SIM_OVERRIDES as gs_sim_bx.
"""
import numpy as np
import pandas as pd
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
_dirs = os.environ.get("GS_VY_SOLDIRS", "sol_vy").split(",")
_bv_list = [float(b) for b in os.environ.get("GS_VY_BV", "0.0").split(",")]
_bc_list = [float(b) for b in os.environ.get("GS_VY_BC", ",".join(["0.0"] * len(_bv_list))).split(",")]
_bi_list = [float(b) for b in os.environ.get("GS_VY_BI", ",".join(["0.0"] * len(_bv_list))).split(",")]
_bk_list = [float(b) for b in os.environ.get("GS_VY_BK", ",".join(["0.0"] * len(_bv_list))).split(",")]
_bv2_list = [float(b) for b in os.environ.get("GS_VY_BV2", ",".join(["0.0"] * len(_bv_list))).split(",")]   # second chain
_share = [float(v) for v in os.environ.get("GS_VY_SHARES", "1.0").split(",")]
ntypes = len(_dirs)
_sols = [np.load(os.path.join(HERE, d.strip(), "solution.npz")) for d in _dirs]
_s = _sols[0]
xgrid, zgrid, bgrid = _s["xgrid"], _s["zgrid"], _s["bgrid"]
pr_x, pr_z = _s["pr_x"], _s["pr_z"]
Mx = _s["Mx"]                                       # (x, x')
ygrid, pr_y, MPy = _s["ygrid"], _s["pr_y"], _s["MPy"]
NY = len(ygrid)                                     # the JOINT node count on a two-chain solve
ygrid2 = _s["ygrid2"] if "ygrid2" in _s.files else np.zeros(NY)      # the second chain's coordinate (0 without one)
NY2 = int(_s["NY2"]) if "NY2" in _s.files else 1
P_up_ts = [d["P_up"] for d in _sols]; P_down_ts = [d["P_down"] for d in _sols]   # per type: (NY, z, x, b)
Q0_ts = [d["Q0"] for d in _sols]; QI_ts = [d["QI"] for d in _sols]; QI_no_ts = [d["QI_no"] for d in _sols]
icut_up_ts = [d["icut_up"] for d in _sols]; icut_dn_ts = [d["icut_dn"] for d in _sols]
b_refin_0_ts = [d["b_refin_0"] for d in _sols]; b_refin_I_ts = [d["b_refin_I"] for d in _sols]
(g, delta, rho_x, sigma_x, rho_z, sigma_z, r, gamma_x, tau, phi,
 kappa_b, xi, imin, imax, sigma_m) = _s["params"]
kappa_e = float(_s["kappa_e"])
gs_bx = float(_s["gs_bx"])
gs_fc = float(_s["gs_fc"]) if "gs_fc" in _s.files else 0.0
gs_bphi = float(_s["gs_bphi"]) if "gs_bphi" in _s.files else 0.0     # recovery exposure: solver-side only
gs_bzl = float(_s["gs_bzl"]) if "gs_bzl" in _s.files else 1.0
gs_bzi = float(_s["gs_bzi"]) if "gs_bzi" in _s.files else 0.0
gs_bzb = float(_s["gs_bzb"]) if "gs_bzb" in _s.files else 0.0
gs_bzc = float(_s["gs_bzc"]) if "gs_bzc" in _s.files else 0.0
gs_bx2 = float(_s["gs_bx2"]) if "gs_bx2" in _s.files else 0.0
gs_pj = float(_s["gs_pj"]) if "gs_pj" in _s.files else 0.0
ashift_ts = [float(d["gs_ashift"]) for d in _sols]  # applied in production below, as the solver does
gs_fixcs = int(_s["gs_fixcs"]) if "gs_fixcs" in _s.files else 0        # fixed capital structure: see gs_solve_vy.py
bfix_ts = [float(bgrid[np.argmin(np.abs(bgrid - float(d["gs_blo"])))]) if "gs_blo" in d.files else 0.0 for d in _sols]

_solved_bv = [float(d["gs_bv"]) for d in _sols]
if _solved_bv != _bv_list:
    raise ValueError(f"GS_VY_BV={_bv_list} does not match the exposures these solutions were solved at: "
                     f"{dict(zip(_dirs, _solved_bv))}. Fix GS_VY_BV, or point GS_VY_SOLDIRS at the solves you mean.")
_solved_bc = [float(d["gs_bc"]) if "gs_bc" in d.files else 0.0 for d in _sols]
_solved_bi = [float(d["gs_bi"]) if "gs_bi" in d.files else 0.0 for d in _sols]
_solved_bk = [float(d["gs_bk"]) if "gs_bk" in d.files else 0.0 for d in _sols]
_solved_bv2 = [float(d["gs_bv2"]) if "gs_bv2" in d.files else 0.0 for d in _sols]
if _solved_bv2 != _bv2_list:
    raise ValueError(f"GS_VY_BV2={_bv2_list} does not match the second-chain exposures these solutions were solved at: {dict(zip(_dirs, _solved_bv2))}.")
if _solved_bk != _bk_list:
    raise ValueError(f"GS_VY_BK={_bk_list} does not match the coupon exposures these solutions were solved at: {dict(zip(_dirs, _solved_bk))}.")
if _solved_bi != _bi_list:
    raise ValueError(f"GS_VY_BI={_bi_list} does not match the investment-cost exposures these solutions were solved at: "
                     f"{dict(zip(_dirs, _solved_bi))}.")
if _solved_bc != _bc_list:
    raise ValueError(f"GS_VY_BC={_bc_list} does not match the cost exposures these solutions were solved at: "
                     f"{dict(zip(_dirs, _solved_bc))}. Fix GS_VY_BC, or point GS_VY_SOLDIRS at the solves you mean.")
if len(_share) != ntypes:
    raise ValueError(f"GS_VY_SHARES has {len(_share)} entries for {ntypes} soldirs ({_dirs}); every type needs a share.")
for _f, (_d, _sol) in enumerate(zip(_dirs, _sols)):
    if _f == 0:
        continue
    for _k in ("xgrid", "zgrid", "bgrid", "pr_x", "pr_z", "Mx", "ygrid", "pr_y", "MPy", "params", "kappa_e", "gs_bx",
               "gamma_v", "kappa_y", "bv_comp", "gs_fc", "gs_bphi", "tau", "sigma_m", "gs_bzl", "gs_bzi", "gs_bzb", "gs_bzc", "gs_bx2", "gs_pj", "gs_gj",
               "ygrid2", "NY2", "gamma_v2", "kappa_y2"):
        if _k not in _s.files or _k not in _sol.files:
            continue
        if not np.array_equal(_s[_k], _sol[_k]):
            raise ValueError(f"solutions disagree on '{_k}': {_dirs[0].strip()} vs {_d.strip()}. The types must differ "
                             f"in gs_bv (and the derived gs_ashift) and NOTHING else; re-solve the ladder with one parameter set.")

znum, xnum, bnum = len(zgrid), len(xgrid), len(bgrid)
_ztil = zgrid / np.sqrt(sigma_z ** 2 / (1 - rho_z ** 2))
_zexpo = gs_bzl + gs_bzi * _ztil                     # (znum,) multiplies each type's gs_bv
_bxz = gs_bx + gs_bx2 * (_ztil ** 2 - 1.0) / np.sqrt(2.0)   # (znum,) x loading by productivity
def _expo_of(zi, b):
    """exposure multiplier at productivity node zi and coupon b (continuous): gs_bzl + gs_bzi z~ + gs_bzb z~ b~"""
    return _zexpo[zi] + (gs_bzb * _ztil[zi] + gs_bzc) * (b - 0.5) / 0.29
burnin = 400          # the protocol's BURNIN (common/protocol.py)
alpha_e = 0.2
reg_seed = 909
chars = ["size", "bm", "agr", "roe", "mom", "lev"]
gamma_grid = np.arange(0.5, 1.1, 0.1)
ov = json.loads(os.environ.get("GS_SIM_OVERRIDES", "{}"))
globals().update(ov)

mn = np.linspace(-4 * sigma_m, 4 * sigma_m, 161)
mw = np.exp(-0.5 * (mn / sigma_m) ** 2)
mw /= mw.sum()
# the smoothing moments as exact 1-D functions of P (right-cumulative sums; see gs_solve_vy.py)
_Cw = np.concatenate((np.cumsum(mw[::-1])[::-1], [0.0]))
_Cwm = np.concatenate((np.cumsum((mw * mn)[::-1])[::-1], [0.0]))
_Cwm2 = np.concatenate((np.cumsum((mw * mn * mn)[::-1])[::-1], [0.0]))


def smooth_moments(P):
    k = np.searchsorted(mn, -P, side="right")
    C, D = _Cw[k], _Cwm[k]
    return P * C + D, P * P * C + 2 * P * D + _Cwm2[k], C


# per-(type, destination y-node) eta'-integrated payoff tables, and the continuation values
S1_ts, S2_ts, EP0_ts = [], [], []
MPx = Mx * pr_x
for f in range(ntypes):
    S1_y, S2_y = [], []
    for iy in range(NY):
        S1u, S2u, _ = smooth_moments(P_up_ts[f][iy])
        S1d, S2d, _ = smooth_moments(P_down_ts[f][iy])
        S1_y.append(xi * S1u + (1 - xi) * S1d)
        S2_y.append(xi * S2u + (1 - xi) * S2d)
    S1_ts.append(S1_y); S2_ts.append(S2_y)
    disc = np.stack([np.einsum("xy,iyb->ixb", MPx, np.einsum("ij,jxb->ixb", pr_z, S1_y[iy])) for iy in range(NY)])
    EP0_ts.append(np.einsum("ab,bzxc->azxc", MPy, disc))        # (NY, z, x, b): priced y-mixture
EPI_ts = [g * e for e in EP0_ts]


def _interp_b_at(V, iz, ix, b):
    jj = np.clip(np.searchsorted(bgrid, b) - 1, 0, bnum - 2)
    ww = np.clip((b - bgrid[jj]) / (bgrid[jj + 1] - bgrid[jj]), 0, 1)
    return V[iz, ix, jj] * (1 - ww) + V[iz, ix, jj + 1] * ww


def _interp_b(V, b):
    jj = np.clip(np.searchsorted(bgrid, b) - 1, 0, bnum - 2)
    ww = np.clip((b - bgrid[jj]) / (bgrid[jj + 1] - bgrid[jj]), 0, 1)
    return V[:, :, jj] * (1 - ww) + V[:, :, jj + 1] * ww


def _by_type(tabs_ts, iy, zi, xt, b, ftype):
    """dispatch _interp_b_at over firm types: tabs_ts[f][iy] is a (z,x,b) table"""
    out = np.empty(len(zi))
    for f in range(ntypes):
        cols = ftype == f
        if cols.any():
            out[cols] = _interp_b_at(tabs_ts[f][iy], zi[cols],
                                     xt[cols] if np.ndim(xt) else np.full(cols.sum(), xt), b[cols])
    return out


def create_arrays(N, T, seed_offset=0):
    rng = np.random.default_rng(np.random.randint(0, 2 ** 31))
    _ss = np.random.SeedSequence(reg_seed)
    rng_reg, rng_bx = (np.random.default_rng(c) for c in _ss.spawn(2))
    ftype = rng_bx.choice(ntypes, size=N, p=np.array(_share) / np.sum(_share))
    bvf = np.array(_bv_list)[ftype]
    ashf = np.array(ashift_ts)[ftype]
    bcf = np.array(_bc_list)[ftype]
    bif = np.array(_bi_list)[ftype]
    bkf = np.array(_bk_list)[ftype]
    bvf2 = np.array(_bv2_list)[ftype]
    bfixf = np.array(bfix_ts)[ftype]                                  # the type's coupon per unit capital (gs_fixcs)
    statx = np.linalg.matrix_power(pr_x.T, 500) @ np.full(xnum, 1 / xnum)
    statz = np.linalg.matrix_power(pr_z.T, 500) @ np.full(znum, 1 / znum)
    staty = np.linalg.matrix_power(pr_y.T, 500) @ np.full(NY, 1 / NY)
    # the aggregate y path: its own stream (run_oracle.py offsets reg_seed by the seed)
    iy = np.zeros(T + 1, dtype=int)
    iy[0] = rng_reg.choice(NY, p=staty)
    for t in range(T):
        iy[t + 1] = rng_reg.choice(NY, p=pr_y[iy[t]])

    ix = np.zeros(T + 1, dtype=int)
    iz = np.zeros((T + 1, N), dtype=int)
    ix[0] = rng.choice(xnum, p=statx)
    iz[0] = rng.choice(znum, size=N, p=statz)
    b = np.zeros((T + 1, N)); k = np.ones((T + 1, N))
    if gs_fixcs:
        b[0] = bfixf                                                  # born with the type's debt
    eta = rng.random((T + 1, N)) < xi
    icost = rng.uniform(imin, imax, size=(T + 1, N))
    mshock = rng.choice(mn, size=(T + 1, N), p=mw)
    scale = np.ones((T + 1, N)); bprime = np.zeros((T + 1, N))
    Vcont = np.zeros((T + 1, N)); payoff = np.zeros((T + 1, N))
    default = np.zeros((T + 1, N), dtype=bool)
    invest = np.zeros((T + 1, N), dtype=bool)
    div = np.zeros((T + 1, N))

    for t in range(T + 1):
        s = iy[t]
        zi, xt = iz[t], ix[t]
        icu = _by_type(icut_up_ts, s, zi, xt, b[t], ftype); icd = _by_type(icut_dn_ts, s, zi, xt, b[t], ftype)
        cut = np.where(eta[t], icu, icd)
        invest[t] = icost[t] <= cut
        bI_star = _by_type(b_refin_I_ts, s, zi, xt, b[t], ftype)
        b0_star = _by_type(b_refin_0_ts, s, zi, xt, b[t], ftype)
        bp = np.where(eta[t], np.where(invest[t], bI_star, b0_star),
                      np.where(invest[t], b[t] / g, b[t]))
        if gs_fixcs:
            bp = b[t]                                                 # debt scales with capital: coupon per unit unchanged
        bprime[t] = bp
        scale[t] = np.where(invest[t], g, 1.0)
        Vc = np.where(invest[t], _by_type(EPI_ts, s, zi, xt, bp, ftype), _by_type(EP0_ts, s, zi, xt, bp, ftype))
        Vcont[t] = Vc
        prod = (1 - tau) * (np.exp(_bxz[zi] * xgrid[xt] + zgrid[zi] + (bvf * ygrid[s] + bvf2 * ygrid2[s]) * _expo_of(zi, b[t]) + ashf) - delta
                            - gs_fc * np.exp(bcf * ygrid[s])) - (1 - tau) * b[t] * np.exp(bkf * ygrid[s])
        dflow = np.where(eta[t],
                         np.where(invest[t],
                                  (1 - kappa_b) * _by_type(QI_ts, s, zi, xt, bp, ftype) - _by_type(QI_no_ts, s, zi, xt, b[t], ftype),
                                  (1 - kappa_b) * _by_type(Q0_ts, s, zi, xt, bp, ftype) - _by_type(Q0_ts, s, zi, xt, b[t], ftype)),
                         0.0)
        if gs_fixcs:                                                  # proceeds of the new debt issued pari passu at investment
            dflow = np.where(invest[t], (1 - kappa_b) * (g - 1) * _by_type(Q0_ts, s, zi, xt, b[t], ftype), 0.0)
        cf = prod + dflow
        div[t] = cf + kappa_e * np.minimum(cf, 0.0) - invest[t] * icost[t] * np.exp(bif * ygrid[s])
        if t == T:
            break
        ix[t + 1] = rng.choice(xnum, p=pr_x[xt])
        for zz in np.unique(zi):
            sel = zi == zz
            iz[t + 1, sel] = rng.choice(znum, size=sel.sum(), p=pr_z[zz])
        s1 = iy[t + 1]
        Pn = np.where(eta[t + 1],
                      _by_type(P_up_ts, s1, iz[t + 1], ix[t + 1], bp, ftype),
                      _by_type(P_down_ts, s1, iz[t + 1], ix[t + 1], bp, ftype))
        alive = (Pn + mshock[t + 1]) > 0
        default[t + 1] = ~alive
        payoff[t + 1] = scale[t] * np.where(alive, np.maximum(Pn + mshock[t + 1], 0.0), 0.0)
        b[t + 1] = np.where(alive, bp, bfixf if gs_fixcs else 0.0)
        k[t + 1] = np.where(alive, k[t] * scale[t], np.maximum(alpha_e * k[t].mean(), 1e-3))
        iz[t + 1] = np.where(alive, iz[t + 1], rng.choice(znum, size=N, p=statz))

    rets = payoff[1:] / Vcont[:-1] - 1.0
    return dict(ix=ix, iz=iz, b=b, k=k, eta=eta, invest=invest, bprime=bprime, scale=scale,
                Vcont=Vcont, payoff=payoff, div=div, default=default, rets=rets, iy=iy, ftype=ftype)


def conditional_moments(arr, t, want_matrix=True):
    iz, ixt, bp, sc, Vc = arr["iz"][t], arr["ix"][t], arr["bprime"][t], arr["scale"][t], arr["Vcont"][t]
    s = int(arr["iy"][t])
    N = len(iz)
    px = pr_x[ixt]
    mx = Mx[ixt]
    ftype = arr["ftype"]
    wts = pr_y[s]; wq = MPy[s]                       # physical and priced weights over y'
    A_b, A2_b = [], []
    for sb in range(NY):
        A = np.empty((N, len(xgrid))); A2 = np.empty((N, len(xgrid)))
        for f in range(ntypes):
            cols = np.flatnonzero(ftype == f)
            if len(cols):
                S1b = _interp_b(S1_ts[f][sb], bp[cols])
                S2b = _interp_b(S2_ts[f][sb], bp[cols])
                A[cols] = np.einsum("fz,zxf->fx", pr_z[iz[cols]], S1b, optimize=True)
                A2[cols] = np.einsum("fz,zxf->fx", pr_z[iz[cols]], S2b, optimize=True)
        A_b.append(A); A2_b.append(A2)
    ER1 = sc * sum(w * (A @ px) for w, A in zip(wts, A_b)) / Vc
    euler = sc * sum(w * (A @ (mx * px)) for w, A in zip(wq, A_b)) / Vc
    out = {"eret": ER1 - 1.0, "euler": euler}
    if want_matrix:
        ERR = np.zeros((N, N))
        diag = np.zeros(N)
        for w, A, A2 in zip(wts, A_b, A2_b):
            if w == 0.0:
                continue
            W = (sc[:, None] * A) / Vc[:, None]
            ERR += w * ((W * px[None, :]) @ W.T)
            diag += w * (sc ** 2) * (A2 @ px) / Vc ** 2
        np.fill_diagonal(ERR, diag)
        out["ERR"] = ERR
    Ex = px @ xgrid; Vx = px @ (xgrid - Ex) ** 2
    out["A_x"] = (sc * sum(w * (A @ (px * (xgrid - Ex))) for w, A in zip(wts, A_b))) / Vc / max(Vx, 1e-16)
    return out


def create_panel(N, T, arr):
    k, b, iz, ix, iy = arr["k"], arr["b"], arr["iz"], arr["ix"], arr["iy"]
    P_ex = arr["Vcont"]
    rets = arr["rets"]
    bvf = np.array(_bv_list)[arr["ftype"]]; ashf = np.array(ashift_ts)[arr["ftype"]]; bcf = np.array(_bc_list)[arr["ftype"]]
    bvf2 = np.array(_bv2_list)[arr["ftype"]]
    opcf = (1 - tau) * (np.exp(_bxz[iz] * xgrid[ix][:, None] + zgrid[iz] + (bvf[None, :] * ygrid[iy][:, None] + bvf2[None, :] * ygrid2[iy][:, None]) * _expo_of(iz, b) + ashf[None, :]) - delta
                        - gs_fc * np.exp(bcf[None, :] * ygrid[iy][:, None]))
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
    xser = pd.DataFrame({"month": range(T),
                         "rf_stand": (xgrid[ix[:T]] - 0.0) / (4 * xgrid.std()),
                         "y_stand": ygrid[iy[:T]] / 4.0,
                         **({"y2_stand": ygrid2[iy[:T]] / 4.0} if NY2 > 1 else {})})
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
