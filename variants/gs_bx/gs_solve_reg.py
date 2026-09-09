"""GS with an exogenous 2-state price-of-risk regime: gamma_s = gamma_x * gmreg[s], s a monthly
Markov chain (p01 calm->stress, p10 stress->calm), unpriced switches.  Two coupled Bellman systems:
every continuation expectation discounts with the CURRENT regime's kernel M_s and mixes next-regime
value functions with (1-p_s, p_s) -- the regime is a common shock.  Gauss-Seidel over regimes inside
each sweep; same damping / policy-freeze / cycle-average machinery as gs_solve.py.  At gmreg=[1,1]
both regimes' fixed points coincide with the single-regime solution (validated externally).

usage: python gs_solve_reg.py [xnum] [tol] [outdir]     (params via GS_PARAM_OVERRIDES)
"""
import numpy as np
from scipy.special import erfc
import os, sys, time, json

xnum = int(sys.argv[1]) if len(sys.argv) > 1 else 161
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6
outdir = sys.argv[3] if len(sys.argv) > 3 else "sol_reg"
HERE = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(HERE, outdir)
os.makedirs(outdir, exist_ok=True)

# ---- parameters (Gomes & Schmid 2021, Table I) ----
# 2026-09-06 (second pass): re-synced to config.py AFTER the paper itself was read.
# The paper calibrates QUARTERLY; this model runs MONTHLY, so per-period flows convert
# and dimensionless quantities do not.  Neither code tree was authoritative -- config.py
# was wrong on three parameters and GS21.m on two, overlapping on none.  Cite Table I,
# not either tree.  Evidence: docs/refactor/FINDINGS-gs21-table1.md.
#   delta   0.02 -> 0.02/3   : Table I 0.02 is a PERIODIC maintenance cost delta*k
#       "akin to depreciation" at 2% per quarter (p.287, p.792), so a monthly model
#       divides by 3.  The first pass here went the other way on the theory that delta
#       scales output; it scales CAPITAL, and it is explicitly per-period.
#   rho_x   0.96 -> 0.95     : Table I and the body text both say 0.95.  config.py:313's
#       old comment claiming "Table 1: rho_x = 0.96" was simply false; GS21.m:22 was
#       right.  sigma_x keeps the conversion sigma_q*sqrt((1-rho_q**(2/3))/(1-rho_q**2))
#       but rebases it on rho_q = 0.95.
#   kappa_e 0 (absent) -> 0.025 : the benchmark equity-issuance cost, charged only on a
#       negative current cash flow.  Both trees carried 0, because GS21.m:33 reads
#       `kappa_e = 0; %0.025` and here the COMMENTED-OUT value is the paper's -- the
#       reverse of the sigma_m and r cases.  Seth's call: run the benchmark.
#   tau 0.2 and sigma_m 5 are unchanged from the first pass; Table I confirms both
#       (tau is a RATE on a profit flow, so it does not rescale with period length).
# The one remaining deliberate difference from config.py is the x-grid size: xnum is
# passed as 161 here (run_gs_bx7*.sh) versus GS21_XNUM = 20 in config.py -- a
# cost/accuracy choice, not a calibration disagreement.
g = 1.14; delta = 0.02 / 3
rho_x = 0.95 ** (1 / 3); sigma_x = 0.012 * np.sqrt((1 - 0.95 ** (2 / 3)) / (1 - 0.95 ** 2))
rho_z = 0.9 ** (1 / 3); sigma_z = 0.16 * np.sqrt((1 - 0.9 ** (2 / 3)) / (1 - 0.9 ** 2))
r = 0.1 / 12; gamma_x = 0.5; x_bar = 0.0
tau = 0.2; phi = 0.4; kappa_e = 0.025; kappa_b = 0.004; xi = 0.03 / 3
bnum, znum = 20, 200
imin, imax = 0.0, 2000.0
sigma_m = 5
# regime block
gmreg = [1.0, 1.0]
p01, p10 = 0.25 / 12, 0.50 / 12
# exposure type: production loads as exp(gs_bx * x + z + gs_ashift); gs_bx = 1, gs_ashift = 0 is the baseline
gs_bx = 1.0
gs_ashift = 0.0
ov = json.loads(os.environ.get("GS_PARAM_OVERRIDES", "{}"))
globals().update(ov)
gmreg = np.array(gmreg, float)
psw = np.array([p01, p10])

# ---- provenance / content-addressing (variants/common/solstamp.py) ----
# Before 2026-09-04 a solution.npz recorded a 15-element `params` array that
# omitted gmreg, gs_bx, gs_ashift, p01/p10 and every grid size, so it could not
# identify its own exposure type; and run_gs_bx7.sh guarded sol_reg with a bare
# `[ -f solution.npz ]` existence check, which is parameter-blind. Both are fixed
# here: the snapshot below covers the WHOLE parameter namespace plus this file's
# source, and the manifest is the durable record.
sys.path.insert(0, os.path.dirname(HERE))          # variants/, for `common`
from common import solstamp                        # noqa: E402

# xnum/tol are module globals and so are already captured in the namespace;
# outdir/HERE are plumbing and must NOT make the identity directory-dependent.
_snap = solstamp.snapshot(
    {k: v for k, v in globals().items() if not k.startswith('_')},
    [os.path.join(HERE, "gs_solve_reg.py")],
    model="gs_bx",
    skip=("outdir", "HERE", "ov"),
    extra={"outdir": os.path.basename(outdir)},
)
_solution = os.path.join(outdir, "solution.npz")
print(f"[solstamp] gs_bx {os.path.basename(outdir)} solve_id={_snap.solve_id}", flush=True)

_hit = solstamp.lookup(_snap.solve_id)
if _hit and not os.environ.get("GS_SOLVE_FORCE"):
    _probs = solstamp.artifact_problems(_hit)
    if not _probs:
        print(f"cached: solve_id {_snap.solve_id} already solved "
              f"({_hit['total_bytes']:,} B) -- nothing to do")
        sys.exit(0)
    print("[solstamp] manifest exists but artifacts do not match; re-solving:")
    for _p in _probs[:5]:
        print(f"  - {_p}")
elif os.path.exists(_solution):
    _prior = solstamp._find_by_artifacts([_solution])
    if _prior and _prior["solve_id"] != _snap.solve_id:
        print(f"[solstamp] {os.path.basename(outdir)}/solution.npz belongs to solve_id "
              f"{_prior['solve_id']}; this run wants {_snap.solve_id}. Differences:")
        for _k, _a, _b in solstamp.diff_params(_prior["params"], _snap.params)[:10]:
            print(f"    {_k}: {_a!r} -> {_b!r}")
    elif _prior:
        # Reached under GS_SOLVE_FORCE=1: the first branch is falsified by the env
        # var, not by a missing manifest, so control lands here with a manifest that
        # MATCHES. Claiming it is unrecorded is a false provenance statement from the
        # provenance system itself.
        print(f"[solstamp] {os.path.basename(outdir)}/solution.npz is recorded and "
              f"matches solve_id {_snap.solve_id}; re-solving because GS_SOLVE_FORCE is set")
    else:
        print(f"[solstamp] {os.path.basename(outdir)}/solution.npz exists but its "
              f"provenance is unrecorded; re-solving")


def tauchen(sigma, rho, multiple, n):
    sdz = np.sqrt(sigma ** 2 / (1 - rho ** 2))
    z0 = np.linspace(-multiple * sdz, multiple * sdz, n)
    ginc = z0[1] - z0[0]
    cdf = lambda x, mu: 0.5 * erfc(-((x - mu) / sigma) / np.sqrt(2))
    P = np.zeros((n, n))
    for i in range(n):
        mu = rho * z0[i]
        P[i, 1:-1] = cdf(z0[1:-1] + ginc / 2, mu) - cdf(z0[1:-1] - ginc / 2, mu)
        P[i, 0] = cdf(z0[0] + ginc / 2, mu)
        P[i, -1] = 1 - cdf(z0[-1] - ginc / 2, mu)
        P[i] /= P[i].sum()
    return z0, P


xgrid, pr_x = tauchen(sigma_x, rho_x, 4, xnum)
zgrid, pr_z = tauchen(sigma_z, rho_z, 4, znum)
bgrid = np.linspace(0.0, 1.0, bnum)

# per-regime kernels, exactly renormalized (E[M_s|x] = e^{-r} in every state)
gam_s = gamma_x * gmreg
Mx_s, MP_s = [], []
for s in (0, 1):
    M = np.exp(-r - 0.5 * gam_s[s] ** 2
               - gam_s[s] * (xgrid[None, :] - (1 - rho_x) * x_bar - rho_x * xgrid[:, None]) / sigma_x)
    M = M * (np.exp(-r) / (M * pr_x).sum(axis=1))[:, None]
    Mx_s.append(M); MP_s.append(M * pr_x)
print(f"gamma per regime: {gam_s}; switch probs {psw}", flush=True)

mn = np.linspace(-4 * sigma_m, 4 * sigma_m, 161)
mw = np.exp(-0.5 * (mn / sigma_m) ** 2)
mw /= mw.sum()

def smooth(P):
    Pn = P[..., None] + mn
    pos = Pn > 0
    return (Pn * pos) @ mw, (pos @ mw)

pi_R = (1 - tau) * (np.exp(gs_bx * xgrid[None, :, None] + zgrid[:, None, None] + gs_ashift) - delta) - (1 - tau) * bgrid[None, None, :]
recov = phi * (1 - delta + np.exp(gs_bx * xgrid[None, :, None] + zgrid[:, None, None] + gs_ashift))


def issue(cf):
    """Equity-issuance cost: charge kappa_e on a NEGATIVE current cash flow only.

    Mirrors gs21_solve.py:309-311 and GS21.m:253-260, `(1 + (prof <= 0)*kappa_e)*prof`.
    `cf + kappa_e*min(cf, 0)` is the same function -- (1+kappa_e)*cf below zero, cf above,
    continuous at zero -- and avoids materialising the boolean mask, which matters because
    the array this is applied to below is (znum, xnum, bnum, bnum).

    Charged on the CURRENT cash flow only: not on the continuation value, and not on the
    realised investment cost, which both trees subtract outside this factor.
    """
    return cf if kappa_e == 0 else cf + kappa_e * np.minimum(cf, 0.0)


# Constant across sweeps: the no-refinancing branch carries no debt term, so its cash
# flow is pi_R alone -- gs21_solve.py:311's `down_R`.
pi_R_e = issue(pi_R)

from numpy import searchsorted
def _b_interp_weights(btarget):
    jj = np.clip(searchsorted(bgrid, btarget) - 1, 0, bnum - 2)
    ww = np.clip((btarget - bgrid[jj]) / (bgrid[jj + 1] - bgrid[jj]), 0, 1)
    return jj, ww
_jg, _wg = _b_interp_weights(bgrid / g)
def at_bg(V):
    return V[..., _jg] * (1 - _wg) + V[..., _jg + 1] * _wg

def discount(V, s):
    Vz = np.einsum("ij,jxb->ixb", pr_z, V, optimize=True)
    return np.einsum("xy,iyb->ixb", MP_s[s], Vz, optimize=True)

t0 = time.time()
# per-regime state
S = [dict(P_up=np.zeros((znum, xnum, bnum)), P_down=np.zeros((znum, xnum, bnum)),
          Q0=np.zeros((znum, xnum, bnum)), prob_up=np.zeros((znum, xnum, bnum)),
          prob_dn=np.zeros((znum, xnum, bnum)), b0idx=np.zeros((znum, xnum, bnum), dtype=int),
          bIidx=np.zeros((znum, xnum, bnum), dtype=int), icut_up=np.zeros((znum, xnum, bnum)),
          icut_dn=np.zeros((znum, xnum, bnum))) for _ in (0, 1)]
prob_delta = [1.0, 1.0]
acc = None
for it in range(60000):
    qerr = perr = 0.0
    # precompute both regimes' smoothed payoffs and debt inner values (previous iterate)
    pre = []
    for s in (0, 1):
        v = S[s]
        QI = g * v["Q0"]; QI_no = at_bg(QI)
        Qn_up = v["prob_up"] * QI_no + (1 - v["prob_up"]) * v["Q0"]
        Qn_dn = v["prob_dn"] * QI_no + (1 - v["prob_dn"]) * v["Q0"]
        Pu_pert, ind_up = smooth(v["P_up"])
        Pd_pert, ind_dn = smooth(v["P_down"])
        inner_up = (bgrid[None, None, :] + Qn_up) * ind_up + recov * (1 - ind_up)
        inner_dn = (bgrid[None, None, :] + Qn_dn) * ind_dn + recov * (1 - ind_dn)
        pre.append(dict(Pu=Pu_pert, Pd=Pd_pert, iu=inner_up, idn=inner_dn))
    for s in (0, 1):
        v = S[s]; p = psw[s]; o = 1 - s
        # ---- debt sweep: current-regime kernel, next-regime values mixed ----
        iu_mix_own = discount(pre[s]["iu"], s); iu_mix_oth = discount(pre[o]["iu"], s)
        idn_mix_own = discount(pre[s]["idn"], s); idn_mix_oth = discount(pre[o]["idn"], s)
        Q0_new = (xi * ((1 - p) * iu_mix_own + p * iu_mix_oth)
                  + (1 - xi) * ((1 - p) * idn_mix_own + p * idn_mix_oth))
        qerr = max(qerr, np.abs(Q0_new - v["Q0"]).max())
        v["Q0"] = 0.5 * Q0_new + 0.5 * v["Q0"]
        QI = g * v["Q0"]; QI_no = at_bg(QI)
        # ---- equity sweep ----
        EP0 = (xi * ((1 - p) * discount(pre[s]["Pu"], s) + p * discount(pre[o]["Pu"], s))
               + (1 - xi) * ((1 - p) * discount(pre[s]["Pd"], s) + p * discount(pre[o]["Pd"], s)))
        EPI = g * EP0
        # 2026-09-06: with kappa_e > 0 the issuance cost multiplies the WHOLE current
        # cash flow -- coupon, buyback and new issue together -- so the b'-choice can no
        # longer be factored out of the level term (pi_R[b] - Q0[b]).  The maximand is
        # obj0[b, b'] = issue(pi_R[b] - Q0[b] + (1-kappa_b)*Q0[b']) + EP0[b'], which is
        # genuinely 4-d, and the optimal b' now depends on current b.  That is the shape
        # gs21_solve.py:309 and GS21.m:253 have always had (their argmax runs over a
        # (state x b') matrix where state already includes b); the old 3-d form here was
        # an optimisation that only holds at kappa_e = 0.  b0idx/bIidx therefore widen
        # from (z, x) to (z, x, b), and so do the saved b_refin_* tables.
        if it % 25 == 0 or it < 50:
            obj0 = issue((pi_R - v["Q0"])[..., None] + (1 - kappa_b) * v["Q0"][..., None, :])
            obj0 += EP0[..., None, :]
            b0new = obj0.argmax(axis=3)
            P0_up = np.take_along_axis(obj0, b0new[..., None], 3)[..., 0]
            del obj0
            objI = issue((pi_R - QI_no)[..., None] + (1 - kappa_b) * QI[..., None, :])
            objI += EPI[..., None, :]
            bInew = objI.argmax(axis=3)
            PI_up = np.take_along_axis(objI, bInew[..., None], 3)[..., 0]
            del objI
            v["b0idx"], v["bIidx"] = b0new, bInew
            P0_dn = pi_R_e + EP0
            PI_dn = pi_R_e + at_bg(EPI)
            v["icut_up"] = np.clip(PI_up - P0_up, imin, imax)
            v["icut_dn"] = np.clip(PI_dn - P0_dn, imin, imax)
            pu_new = (v["icut_up"] - imin) / (imax - imin)
            pd_new = (v["icut_dn"] - imin) / (imax - imin)
            prob_delta[s] = max(np.abs(pu_new - v["prob_up"]).max(), np.abs(pd_new - v["prob_dn"]).max())
            v["prob_up"], v["prob_dn"] = pu_new, pd_new
        else:
            # Policy frozen: gather Q0/EP0 at the stored b' directly, so the 4-d array is
            # built only on the 1-in-25 sweeps that actually re-optimise.
            P0_up = (issue(pi_R - v["Q0"]
                           + (1 - kappa_b) * np.take_along_axis(v["Q0"], v["b0idx"], 2))
                     + np.take_along_axis(EP0, v["b0idx"], 2))
            PI_up = (issue(pi_R - QI_no
                           + (1 - kappa_b) * np.take_along_axis(QI, v["bIidx"], 2))
                     + np.take_along_axis(EPI, v["bIidx"], 2))
            P0_dn = pi_R_e + EP0
            PI_dn = pi_R_e + at_bg(EPI)
        P_up_new = v["prob_up"] * (PI_up - 0.5 * (v["icut_up"] + imin)) + (1 - v["prob_up"]) * P0_up
        P_dn_new = v["prob_dn"] * (PI_dn - 0.5 * (v["icut_dn"] + imin)) + (1 - v["prob_dn"]) * P0_dn
        perr = max(perr, np.abs(P_up_new - v["P_up"]).max(), np.abs(P_dn_new - v["P_down"]).max())
        v["P_up"] = 0.5 * P_up_new + 0.5 * v["P_up"]
        v["P_down"] = 0.5 * P_dn_new + 0.5 * v["P_down"]
        v["_P0_up"], v["_PI_up"], v["_P0_dn"], v["_PI_dn"] = P0_up, PI_up, P0_dn, PI_dn
    if not np.isfinite(perr):
        raise RuntimeError(f"diverged at sweep {it}")
    vscale = max(1.0, max(np.abs(S[s]["P_up"]).max() for s in (0, 1)))
    if it == 5000:
        acc = [dict(P_up=0.0, P_down=0.0, Q0=0.0, n=0) for _ in (0, 1)]
    if it >= 5000 and acc is not None:
        for s in (0, 1):
            acc[s]["P_up"] += S[s]["P_up"]; acc[s]["P_down"] += S[s]["P_down"]
            acc[s]["Q0"] += S[s]["Q0"]; acc[s]["n"] += 1
    if (qerr / vscale < tol * 20 and perr / vscale < tol * 20 and it > 100 and it % 25 == 24
            and max(prob_delta) / max(max(S[s]["prob_up"].max() for s in (0, 1)), 1e-9) < 1e-4):
        break
    if it == 5600:
        for s in (0, 1):
            S[s]["P_up"] = acc[s]["P_up"] / acc[s]["n"]
            S[s]["P_down"] = acc[s]["P_down"] / acc[s]["n"]
            S[s]["Q0"] = acc[s]["Q0"] / acc[s]["n"]
        # final policy pass per regime on the averaged values
        pre = []
        for s in (0, 1):
            v = S[s]
            Pu_pert, _ = smooth(v["P_up"]); Pd_pert, _ = smooth(v["P_down"])
            pre.append(dict(Pu=Pu_pert, Pd=Pd_pert))
        for s in (0, 1):
            v = S[s]; p = psw[s]; o = 1 - s
            QI = g * v["Q0"]; QI_no = at_bg(QI)
            EP0 = (xi * ((1 - p) * discount(pre[s]["Pu"], s) + p * discount(pre[o]["Pu"], s))
                   + (1 - xi) * ((1 - p) * discount(pre[s]["Pd"], s) + p * discount(pre[o]["Pd"], s)))
            EPI = g * EP0
            obj0 = issue((pi_R - v["Q0"])[..., None] + (1 - kappa_b) * v["Q0"][..., None, :])
            obj0 += EP0[..., None, :]
            v["b0idx"] = obj0.argmax(axis=3)
            v["_P0_up"] = np.take_along_axis(obj0, v["b0idx"][..., None], 3)[..., 0]
            del obj0
            objI = issue((pi_R - QI_no)[..., None] + (1 - kappa_b) * QI[..., None, :])
            objI += EPI[..., None, :]
            v["bIidx"] = objI.argmax(axis=3)
            v["_PI_up"] = np.take_along_axis(objI, v["bIidx"][..., None], 3)[..., 0]
            del objI
            v["_P0_dn"] = pi_R_e + EP0; v["_PI_dn"] = pi_R_e + at_bg(EPI)
            v["icut_up"] = np.clip(v["_PI_up"] - v["_P0_up"], imin, imax)
            v["icut_dn"] = np.clip(v["_PI_dn"] - v["_P0_dn"], imin, imax)
        print(f"cycle-averaged; stopping", flush=True)
        break
    if it % 200 == 0:
        print(f"sweep {it}: qerr {qerr:.3e} perr {perr:.3e}  ({time.time()-t0:.0f}s)", flush=True)

print(f"converged: sweep {it}, qerr {qerr:.2e}, perr {perr:.2e} in {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(os.path.join(outdir, "solution.npz"),
                    xgrid=xgrid, zgrid=zgrid, bgrid=bgrid, pr_x=pr_x, pr_z=pr_z,
                    Mx=np.stack(Mx_s), gmreg=gmreg, psw=psw,
                    P_up=np.stack([S[s]["P_up"] for s in (0, 1)]),
                    P_down=np.stack([S[s]["P_down"] for s in (0, 1)]),
                    P0_up=np.stack([S[s]["_P0_up"] for s in (0, 1)]),
                    PI_up=np.stack([S[s]["_PI_up"] for s in (0, 1)]),
                    P0_down=np.stack([S[s]["_P0_dn"] for s in (0, 1)]),
                    PI_down=np.stack([S[s]["_PI_dn"] for s in (0, 1)]),
                    Q0=np.stack([S[s]["Q0"] for s in (0, 1)]),
                    QI=np.stack([g * S[s]["Q0"] for s in (0, 1)]),
                    QI_no=np.stack([at_bg(g * S[s]["Q0"]) for s in (0, 1)]),
                    icut_up=np.stack([S[s]["icut_up"] for s in (0, 1)]),
                    icut_dn=np.stack([S[s]["icut_dn"] for s in (0, 1)]),
                    b_refin_0=np.stack([bgrid[S[s]["b0idx"]] for s in (0, 1)]),
                    b_refin_I=np.stack([bgrid[S[s]["bIidx"]] for s in (0, 1)]),
                    params=np.array([g, delta, rho_x, sigma_x, rho_z, sigma_z, r, gamma_x, tau, phi,
                                     kappa_b, xi, imin, imax, sigma_m]),
                    # 2026-09-04: the 15-element `params` array above omits every
                    # parameter that distinguishes one bx7 solve from another, so a
                    # solution could not identify its own exposure type. These do.
                    solve_id=_snap.solve_id,
                    gs_bx=gs_bx, gs_ashift=gs_ashift,   # gmreg already saved above
                    # kappa_e is NOT appended to `params` above: gs_sim_bx.py unpacks
                    # that array positionally as a 15-tuple, so growing it silently
                    # mis-assigns every field after the insertion point.
                    kappa_e=kappa_e,
                    p01=p01, p10=p10, xnum=xnum, bnum=bnum, znum=znum, tol=tol,
                    params_json=json.dumps(_snap.params, sort_keys=True, default=str))
print("saved", _solution)

_man = solstamp.record(
    _snap, [_solution], tag=os.path.basename(outdir),
    # Recorded, not hashed. `tol` alone cannot say whether this solve met its own
    # contract: the loop enforces tol*20, and the 5600-sweep cap exits through the
    # same print("converged") as the tolerance test.
    achieved={"exit": "cycle_capped" if it >= 5600 else "tolerance",
              "sweeps": int(it),
              "qerr_rel": float(qerr / vscale),
              "perr_rel": float(perr / vscale),
              "vscale": float(vscale),
              "threshold_enforced": float(tol * 20),
              "tol_requested": float(tol)})
print(f"[solstamp] recorded solve_id {_snap.solve_id} "
      f"({_man['total_bytes']:,} B, committable={_man['committable']})")
if not _man["committable"]:
    print(f"[solstamp] artifact exceeds {solstamp.SMALL_ARTIFACT_BYTES:,} B -- kept out of "
          f"git; the manifest in experiments/solfiles/ is the durable record")
