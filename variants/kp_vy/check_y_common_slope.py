"""Does ONE price of y-risk price all three vyx types? Step 0 of the disaster plan.

WHY THIS EXISTS. docs/kp14_y_risk_adjustment.md establishes analytically that the `vy`
route's A(y) charges the y-risk premium as a CONSTANT addition to the discount rate --
KP14's own eq. (11) shape, exact for its GBM shocks -- where a priced mean-reverting state
needs a Girsanov drift change whose cumulative effect SATURATES. The omitted term is
(gamma_v*sigma_y - sigma_y**2*b)*A'(y). Backing the price of y-risk out of the module's own
A gives 4.18 / 3.68 / 3.26 for b = 0.02 / 0.07 / 0.14 against a declared gamma_v = 1.8: not
merely the wrong level, but a DIFFERENT level per type. No single SDF prices all three.

That is an argument about a coefficient in one claim. It is not yet a statement about a
reported number. This script is the run that decides, and it is step 0 of
docs/plan-before-home-20260917.md, which gates every further KP14 experiment: vyx (+0.1251)
and vyg25 (+0.1636) are the only two economies in the project with a fair gap above +0.004,
and they are exactly the two the error lands on.

THE TEST, AND WHY IT IS EXACT RATHER THAN A DECOMPOSITION. KP14 as implemented states three
constant prices of risk -- gamma_x, gamma_z, gamma_v -- on three Brownians. Any pricing
function consistent with that SDF must satisfy, firm by firm and month by month,

    E^Q_t[ P_i,t+1 + CF_i,t+1 ] / P_i,t  =  exp(r*dt),

with Q the measure that shifts each Brownian by its price of risk. That is a statement about
the model's own output, so it needs no elasticity approximation, no linearisation of the SDF,
and no assumption about which characteristics span the premium. It is computed here by
rebuilding the module's own four expected-payoff terms (panel_functions_kp14.py:225-240) and
re-taking their expectation under Q:

    x   drift mu_x -> mu_x - gamma_x*sigma_x                   (GBM, exact)
    z   drift mu_z -> mu_z - gamma_z*sigma_z                   (GBM, exact)
    y   Gauss-Hermite nodes shifted by -sigma_y*gamma_v*(1 - e^{-kappa_y*dt})/kappa_y,
        the exact OU Girsanov displacement over one month -- which is the very thing the
        code replaces with a constant rate added to the discount.
    CF  unchanged: the month's cash flow is predetermined at t (it carries e^{b y_t} and
        x_t, not their t+1 values), so it takes no risk adjustment at all.

The rebuild is verified against the module's own `erets` before anything is read off it
(the [check] line below): if the four terms do not reproduce E^P[R] to machine precision,
the Q-version is measuring something else and the run aborts.

WHAT ISOLATES THE y CHANNEL. A nonzero mispricing would also pick up the model's ordinary
discretisation error -- the 21-node A resolvent, the G grid, the CIR steps -- which has
nothing to do with y. Two controls separate them, and neither costs a solve:

  * the b = 0.02 type is almost unexposed to y, so its mispricing is essentially the
    baseline discretisation error, and the SPREAD across the three types is the
    y-attributable part;
  * kpbase (type_bv = [0.0], gamma_v = 0) has no y exposure anywhere, so its mispricing is
    the baseline error outright. Run it as the control.

It then solves, per type, for the gamma_v that would set that type's mean mispricing to
zero. Under a correct model all three equal the declared gamma_v. This is the panel
counterpart of the implied-lambda table in docs/kp14_y_risk_adjustment.md.

Finally it runs the regression docs/plan-before-home-20260917.md names in step 0 -- each
type's conditional expected excess return on its y-exposure, testing for a common slope --
with beta^y differentiated through the module's own interpolants. The exact test above is
the headline; the regression is reported beside it because it is what the plan registered.

WHAT IT DOES NOT DO. It does not re-solve anything and it does not touch G_SOURCES or
I_SOURCES (build_vy_tables.py:100-101, explicit two-file lists), so no solve id moves.

Run, from variants/:
  KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],
                       "gamma_v":1.8,"bv_comp":1.2}' \
  KP_VY_PREFIX=vyx python kp_vy/check_y_common_slope.py --tag vyx --N 500 --T 500 --seed 0

The protocol panel (N 500, T 500, burn-in 400) peaks near 31 GiB while create_arrays runs;
run it where that fits. See docs/kp14_y_risk_adjustment.md.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from scipy import interpolate

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="vyx")
ap.add_argument("--N", type=int, default=500)
ap.add_argument("--T", type=int, default=500, help="months after burn-in, as the protocol sets it")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gamma_grid", default="0,0.5,1,1.5,1.8,2.5,3,3.5,4,4.5,5,6",
                help="gamma_v values at which the Q-mispricing is traced, so the value that "
                     "zeroes each type's mean can be read off by interpolation")
ap.add_argument("--h", type=float, default=None,
                help="central-difference step for beta^y in the registered regression. "
                     "Default dy/2; the run repeats at h = dy as a sensitivity.")
ap.add_argument("--out", default=None)
ap.add_argument("--save_misp", default=None,
                help="also dump the per-(month, firm) mispricing at the declared Q, padded to "
                     "(months, N) in run_oracle.py --save_panel's layout, so it lines up with a "
                     "saved *_moments_*.npz and the ceiling can be recomputed without its mu.")
args = ap.parse_args()

os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))

np.random.seed(args.seed)
import panel_functions_kp14 as mod
from parameters_kp14 import (burnin, dt, r, alpha, sigma_x, sigma_z, gamma_x, gamma_z,
                             gamma_v, sigma_y, kappa_y, mu_x, mu_z, y_grid, dy, NY, ntypes,
                             type_bv, type_theta, bv_comp, theta_eps, theta_u, delta,
                             lambda_H, lambda_L, mu_H, mu_L, C)
from parameters_kp14 import _coef_at

# The aggregate y path and the type draw move with the seed exactly as run_oracle.py does
# (its "aggregate-state generators" block), so seed s here is seed s there and nothing else.
for _name in ("gam_seed", "bx_seed"):
    setattr(mod, _name, int(getattr(mod, _name)) + args.seed)
print(f"[seed {args.seed}] gam_seed={mod.gam_seed} bx_seed={mod.bx_seed}; "
      f"global stream seeded {args.seed}", flush=True)
print(f"[params] gamma_v={gamma_v} type_bv={[float(v) for v in type_bv]} bv_comp={bv_comp} "
      f"sigma_y={sigma_y:.5f} kappa_y={kappa_y} NY={NY} y_max={y_grid[-1]}", flush=True)

alph = alpha / (1 - alpha)
h_reg = args.h if args.h is not None else dy / 2.0
ghx, ghw = np.polynomial.hermite.hermgauss(7)
ghw = ghw / np.sqrt(np.pi)
ar = np.exp(-kappa_y * dt)
sd_c = np.sqrt(1 - ar ** 2)
# Exact OU Girsanov displacement of y over one month per unit price of y-risk:
#   y' = ar*y + sd_c*eps  -  gamma_v * sigma_y*(1 - e^{-kappa*dt})/kappa
_OU_SHIFT = sigma_y * (1.0 - np.exp(-kappa_y * dt)) / kappa_y
print(f"[Q] OU one-month Girsanov displacement per unit gamma_v: {_OU_SHIFT:.6f} "
      f"(vs sigma_y*dt = {sigma_y*dt:.6f})", flush=True)

# ---- the panel, built exactly as run_oracle.py builds it -------------------------------
t0 = time.time()
N, T = args.N, args.T
arr = mod.create_arrays(N, T + burnin)
(K, book, op_cashflow, x, z, eps, uj, chi, rate, high, Et_G, EtA, _alph, Et_z_alph, price,
 rets, erets, lambda_f, lz_t, lx_t, lz_p, lx_p, yreg, ftype) = arr
panel = mod.create_panel(N, T + burnin, arr)
print(f"[{args.tag}] panel built in {time.time()-t0:.0f}s", flush=True)

# ---- collapse the project dimension ----------------------------------------------------
# chi is 0/1 and K = chi*Kj, so chi*K**alpha == K**alpha, and every project-indexed sum in
# the four expected-payoff terms is linear in u:
#   A(eps,u,y,f) = [a0 + (eps-1)a1] + (u-1)[a2 + (eps-1)a3] == Pa(y) + (u-1)*Pb(y)
# with Pa, Pb functions of the CURRENT date only. So the whole project dimension is carried
# by two (T+1, N) reductions that do not depend on y, which is what makes re-taking the
# expectation under Q -- and tracing it over a grid of gamma_v -- cost seconds instead of a
# second pass over a 3.25 GiB array.
Kalpha = K ** alpha
del K, EtA, Et_G, chi, book, op_cashflow, rets, lz_t, lx_t, lz_p, lx_p, arr
S0 = Kalpha.sum(axis=0)                                    # sum_t1 K^alpha
S1 = (Kalpha * (uj - 1.0)).sum(axis=0)                     # sum_t1 K^alpha (u-1)
del Kalpha, uj
print(f"[{args.tag}] project dimension collapsed at {time.time()-t0:.0f}s", flush=True)

# ---- tables, loaded as create_arrays and sdf_compute load them --------------------------
prefix = os.environ.get("KP_VY_PREFIX", "vys")
_tabnames = {"A_mod_lst": "Et_A_mod", "G_up_lst": "Et_G_up", "G_down_lst": "Et_G_down"}
G_up_ty, G_down_ty, Et_ty = [], [], []
for f in range(ntypes):
    G_in = pd.read_csv(f"G_{prefix}{f}.csv")
    eg = G_in.eps.values
    G_up_ty.append([interpolate.interp1d(eg, G_in[f"G_up_y{i}"].values, fill_value="extrapolate")
                    for i in range(NY)])
    G_down_ty.append([interpolate.interp1d(eg, G_in[f"G_down_y{i}"].values, fill_value="extrapolate")
                      for i in range(NY)])
    Et_ty.append([{dst: interpolate.interp1d(eg, np.load(f"integ_{prefix}{f}_{i}.npz")[src],
                                             fill_value="extrapolate")
                   for src, dst in _tabnames.items()} for i in range(NY)])


def at_y(tabs, yv):
    iy = int(np.clip(np.searchsorted(y_grid, yv) - 1, 0, NY - 2))
    w = float(np.clip((yv - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy]), 0, 1))
    return lambda e: (1 - w) * tabs[iy](e) + w * tabs[iy + 1](e)


TT = S0.shape[0]
COLS = [ftype == f for f in range(ntypes)]
tt = type_theta[ftype]
ebv_now = np.exp(type_bv[ftype][None, :] * yreg[:, None])
_eE = 1.0 + (eps - 1.0) * np.exp(-theta_eps * dt)
_uD = np.exp(-theta_u * dt)

# term3 is the month's cash flow. It carries x_t, e^{b y_t} and u_t -- all dated t -- so it
# is predetermined and is identical under P and Q (panel_functions_kp14.py:238).
term3 = x * ebv_now * dt * tt[None, :] * eps * (S0 + S1)


def payoff(gv, gx=None, gz=None):
    """E_t[P_{t+1} + CF_{t+1}] under the measure that prices x-, z- and y-risk at (gx, gz, gv).

    (0, 0, 0) is the physical measure and must reproduce the module's `erets`; the declared
    (gamma_x, gamma_z, gamma_v) is Q. Carrying the three separately is what lets the run say
    WHICH channel a mispricing sits in, rather than only that there is one."""
    gx = gamma_x if gx is None else gx
    gz = gamma_z if gz is None else gz
    yq = ar * yreg[:, None] - gv * _OU_SHIFT + sd_c * np.sqrt(2) * ghx[None, :]
    Etx = np.exp((mu_x - gx * sigma_x) * dt) * x
    Etza = z ** alph * np.exp(alph * (mu_z - gz * sigma_z) * dt
                              + 0.5 * alph * (2 * alpha - 1) / (1 - alpha) * sigma_z ** 2 * dt)
    SA = np.zeros_like(S0); MGd = np.zeros_like(S0); MGu = np.zeros_like(S0); MA = np.zeros_like(S0)
    for f in range(ntypes):
        cols = COLS[f]
        b = float(type_bv[f])
        tabs = Et_ty[f]
        gdt = [tb["Et_G_down"] for tb in tabs]
        gut = [tb["Et_G_up"] for tb in tabs]
        amt = [tb["Et_A_mod"] for tb in tabs]
        for t_ in range(TT):
            e_ = eps[t_, cols]; ee = _eE[t_, cols]
            s0 = S0[t_, cols]; s1 = S1[t_, cols]
            for q in range(len(ghw)):
                yv = float(yq[t_, q])
                a0, a1, a2, a3 = _coef_at(yv, f)
                wq = ghw[q] * np.exp(b * yv)
                SA[t_, cols] += wq * ((a0 + (ee - 1) * a1) * s0 + _uD * (a2 + (ee - 1) * a3) * s1)
                MGd[t_, cols] += wq * at_y(gdt, yv)(e_)
                MGu[t_, cols] += wq * at_y(gut, yv)(e_)
                MA[t_, cols] += wq * at_y(amt, yv)(e_)
    EtG = ((high == 0) * lambda_f[None, :] * ((1 - mu_H * dt) * MGd + mu_H * dt * MGu)
           + (high == 1) * lambda_f[None, :] * ((1 - mu_L * dt) * MGu + mu_L * dt * MGd))
    term1 = Etx * (1 - delta * dt) * SA
    term2 = Etza * Etx * EtG
    term4 = rate * dt * C * Etza * Etx * MA
    return term1 + term2 + term3 + term4


# ---- the rebuild must reproduce the module's own E^P[R] --------------------------------
_erets_rebuilt = payoff(0.0, gx=0.0, gz=0.0) / price - 1.0
_okm = np.isfinite(erets) & np.isfinite(_erets_rebuilt)
_relerr = float(np.abs(_erets_rebuilt[_okm] - erets[_okm]).max())
print(f"[check] max abs error rebuilding E^P[R] from the four terms: {_relerr:.3e}", flush=True)
if _relerr > 1e-10:
    raise SystemExit(
        f"ABORT: the rebuilt physical expected return differs from the module's by {_relerr:.3e}. "
        f"The Q-version below would then be the risk-neutral expectation of something other "
        f"than what KP14 prices.")

# ---- sample: run_oracle.py's month filter, verbatim ------------------------------------
panel["size"] = np.log(panel.mve)
panel = panel[panel.month >= 2]
panel.replace([np.inf, -np.inf], np.nan, inplace=True)
panel.set_index(["month", "firmid"], inplace=True)
chars = ["size", "bm", "agr", "roe", "mom"]
nans = panel[chars + ["mve", "xret"]].isnull().any(axis=1)
panel = panel.loc[nans[~nans].index]
months = panel.index.unique("month")
months = months[(months >= burnin + 14) & (months <= T + burnin - 2)]
ev_months = months[months >= months.min() + 360]
keep_idx = {m: panel.loc[m].index.to_numpy() for m in months}
print(f"[sample] {len(months)} months {months.min()}..{months.max()}; "
      f"evaluation window {len(ev_months)} months", flush=True)


def flatten(field, mlist):
    # A short smoke panel has no evaluation window; every reported panel does.
    if len(mlist) == 0:
        e = np.empty(0)
        return e, e.astype(int), e.astype(int)
    vals, typ, mon = [], [], []
    for m in mlist:
        k = keep_idx[m]
        vals.append(field[m, k]); typ.append(ftype[k]); mon.append(np.full(len(k), m))
    return np.concatenate(vals), np.concatenate(typ), np.concatenate(mon)


# ---- headline: the risk-neutral consistency test ---------------------------------------
Rf = float(np.exp(r * dt))

# ---- which channel is a mispricing in? -------------------------------------------------
# Switching the three prices of risk on one at a time says whether a residual is about y at
# all. kpbase (no y exposure) must come out near zero on the (gamma_x, gamma_z, 0) row, or
# the test has a zero point that is not about y and nothing below can be read as a y result.
_decomp = {}
for _nm, _gx, _gz, _gv in (("physical (0,0,0)", 0.0, 0.0, 0.0),
                           ("x only", gamma_x, 0.0, 0.0),
                           ("z only", 0.0, gamma_z, 0.0),
                           ("x+z", gamma_x, gamma_z, 0.0),
                           ("x+z+y (Q)", gamma_x, gamma_z, float(gamma_v))):
    _m = payoff(_gv, gx=_gx, gz=_gz) / price - Rf
    _v, _ty, _ = flatten(_m, months)
    _decomp[_nm] = {"mean": float(_v.mean()),
                    "by_type": [float(_v[_ty == f].mean()) for f in range(ntypes)]}
    print(f"[decomp] {_nm:<18} mean E[R]-Rf by type: "
          + "  ".join(f"{t:+.6f}" for t in _decomp[_nm]["by_type"]), flush=True)

grid = sorted(set([float(g) for g in args.gamma_grid.split(",")] + [float(gamma_v)]))
trace = {}
for gv in grid:
    misp = payoff(gv) / price - Rf
    v, ty, _ = flatten(misp, months)
    ve, tye, _ = flatten(misp, ev_months)
    trace[gv] = {"all": {"mean": float(v.mean()),
                         "by_type": [float(v[ty == f].mean()) for f in range(ntypes)],
                         "sd_by_type": [float(v[ty == f].std()) for f in range(ntypes)]},
                 "eval": ({"mean": float(ve.mean()),
                           "by_type": [float(ve[tye == f].mean()) for f in range(ntypes)]}
                          if len(ve) else None)}
    print(f"[Q] gamma_v={gv:<5g} mean mispricing/month by type: "
          + "  ".join(f"{t:+.6f}" for t in trace[gv]["all"]["by_type"]), flush=True)


def implied(f):
    """the gamma_v that zeroes type f's mean mispricing, by interpolation on the traced grid"""
    xs = np.array(grid); ys = np.array([trace[g]["all"]["by_type"][f] for g in grid])
    if ys.min() > 0 or ys.max() < 0:
        return None
    s = np.argsort(ys)
    return float(np.interp(0.0, ys[s], xs[s]))


implied_gv = [implied(f) for f in range(ntypes)]
print("\n[Q] gamma_v that prices each type exactly: "
      + "  ".join("n/a" if g is None else f"{g:.3f}" for g in implied_gv)
      + f"   (declared {gamma_v})", flush=True)

# ---- the registered regression: common slope on beta^y ---------------------------------
def VW(yvec):
    """assets in place and growth option at date-y `yvec`, stripped of the common x*e^{b y}."""
    V = np.empty_like(S0); Gs = np.empty_like(S0)
    for f in range(ntypes):
        cols = COLS[f]
        for t_ in range(TT):
            a0, a1, a2, a3 = _coef_at(yvec[t_], f)
            e_ = eps[t_, cols]
            V[t_, cols] = (a0 + (e_ - 1) * a1) * S0[t_, cols] + (a2 + (e_ - 1) * a3) * S1[t_, cols]
            gd = at_y(G_down_ty[f], yvec[t_])(e_); gu = at_y(G_up_ty[f], yvec[t_])(e_)
            Gs[t_, cols] = np.where(high[t_, cols] == 1, gu, gd)
    return V, (z ** alph) * lambda_f[None, :] * Gs


V0, W0 = VW(yreg)
_pr = float(np.abs(x * ebv_now * (V0 + W0) - price).max() / np.abs(price).max())
print(f"[check] max relative error rebuilding price from (V + W): {_pr:.3e}", flush=True)
if _pr > 1e-9:
    raise SystemExit("ABORT: the rebuilt price does not match the module's; beta^y would be wrong.")


def exposures(step):
    yp = np.clip(yreg + step, y_grid[0], y_grid[-1]); ym = np.clip(yreg - step, y_grid[0], y_grid[-1])
    Vp, Wp = VW(yp); Vm, Wm = VW(ym)
    by = type_bv[ftype][None, :] + (np.log(Vp + Wp) - np.log(Vm + Wm)) / (yp - ym)[:, None]
    return by, alph * W0 / (V0 + W0)


def ols_cluster(yv, X, cl):
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ yv); u = yv - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cl):
        s = cl == c; g = X[s].T @ u[s]; meat += np.outer(g, g)
    G = len(np.unique(cl))
    return b, XtX @ meat @ XtX * (G / max(G - 1, 1))


def wald_equal(b, V, idx):
    from scipy import stats
    R = np.zeros((len(idx) - 1, len(b)))
    for j in range(len(idx) - 1):
        R[j, idx[j]] = 1.0; R[j, idx[j + 1]] = -1.0
    Rb = R @ b
    chi2 = float(Rb @ np.linalg.pinv(R @ V @ R.T) @ Rb)
    return chi2, len(idx) - 1, float(1 - stats.chi2.cdf(chi2, len(idx) - 1))


mu_all = 1.0 + erets - Rf
reg = {}
for step, sname in ((h_reg, "h=dy/2"), (dy, "h=dy")):
    by_f, bz_f = exposures(step)
    reg[sname] = {}
    for mlist, lname in ((months, "all"), (ev_months, "eval")):
        if len(mlist) == 0:
            continue
        muv, ty, mo = flatten(mu_all, mlist)
        byv, _, _ = flatten(by_f, mlist)
        bzv, _, _ = flatten(bz_f, mlist)
        d = pd.DataFrame({"mu": muv, "by": byv, "bz": bzv, "m": mo})
        g = d.groupby("m")
        yv = (d.mu - g.mu.transform("mean")).to_numpy()
        byd = (d.by - g.by.transform("mean")).to_numpy()
        bzd = (d.bz - g.bz.transform("mean")).to_numpy()
        D = np.stack([(ty == f).astype(float) for f in range(ntypes)], axis=1)
        b2, V2 = ols_cluster(yv, np.column_stack([D * byd[:, None], bzd]), mo)
        chi2, dfree, p = wald_equal(b2, V2, list(range(ntypes)))
        b3, V3 = ols_cluster(yv, np.column_stack([D * byd[:, None], bzd, D[:, 1:]]), mo)
        chi3, _, p3 = wald_equal(b3, V3, list(range(ntypes)))
        reg[sname][lname] = {
            "nobs": int(len(d)), "nmonths": int(d.m.nunique()),
            "lam": [float(v) for v in b2[:ntypes]],
            "se": [float(np.sqrt(V2[i, i])) for i in range(ntypes)],
            "implied_gamma_v": [float(b2[i] / (sigma_y * dt)) for i in range(ntypes)],
            "wald_chi2": chi2, "df": dfree, "p": p,
            "within_type_implied_gamma_v": [float(b3[i] / (sigma_y * dt)) for i in range(ntypes)],
            "within_type_p": p3,
            "mean_beta_y_by_type": [float(byv[ty == f].mean()) for f in range(ntypes)],
            "mean_mu_by_type": [float(muv[ty == f].mean()) for f in range(ntypes)]}
        rr = reg[sname][lname]
        print(f"\n[reg {sname}/{lname}] {rr['nobs']} firm-months, {rr['nmonths']} months", flush=True)
        for f in range(ntypes):
            print(f"  type {f} (b={float(type_bv[f]):.2f})  lambda={rr['lam'][f]:+.6f} "
                  f"({rr['se'][f]:.6f})  implied gamma_v={rr['implied_gamma_v'][f]:.3f}", flush=True)
        print(f"  H0 common slope: chi2({rr['df']})={rr['wald_chi2']:.1f}  p={rr['p']:.3g}", flush=True)

summary = {"tag": args.tag, "seed": args.seed, "N": N, "T": T, "burnin": burnin,
           "gamma_v_declared": float(gamma_v), "sigma_y": float(sigma_y),
           "type_bv": [float(v) for v in type_bv], "bv_comp": float(bv_comp),
           "type_counts": [int((ftype == f).sum()) for f in range(ntypes)],
           "erets_rebuild_abserr": _relerr, "price_rebuild_relerr": _pr,
           "Rf_monthly": Rf, "ou_shift_per_gamma": float(_OU_SHIFT),
           "channel_decomposition": _decomp,
           "q_trace": {str(k): v for k, v in trace.items()},
           "implied_gamma_v_by_type": implied_gv,
           "regression": reg}
if args.save_misp:
    _m = payoff(float(gamma_v)) / price - Rf
    _nmax = max(len(keep_idx[m]) for m in months)
    _M = np.full((len(months), _nmax), np.nan)
    _KP = np.full((len(months), _nmax), -1, dtype=np.int64)
    for _i, _mo in enumerate(months):
        _k = keep_idx[_mo]
        _M[_i, :len(_k)] = _m[_mo, _k]; _KP[_i, :len(_k)] = _k
    np.savez_compressed(args.save_misp, months=np.asarray(months), misp=_M, keep=_KP,
                        ftype=ftype, type_bv=np.asarray(type_bv, float),
                        gamma_v=float(gamma_v))
    print(f"[misp] wrote {args.save_misp}  shape {_M.shape}", flush=True)

out = args.out or os.path.join(os.path.dirname(HERE), "results",
                               f"kp_vy_yslope_{args.tag}_s{args.seed:03d}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(summary, open(out, "w"), indent=1)
print(f"\n[done] {time.time()-t0:.0f}s -> {out}", flush=True)
