"""gs_vy: GS21 with a PRICED stationary AR(1) aggregate state y and firm exposure types -- the
kp_vy construction one model over. Derived from gs_bx/gs_solve_reg.py: the 2-state unpriced
price-of-risk regime is replaced by an NY-state Tauchen chain for y whose transitions ARE priced.

    y_{t+1} = rho_y y_t + s_y eta,  rho_y = exp(-kappa_y/12), s_y = sqrt(1 - rho_y^2)  (unit sd)
    M_{t+1} = Mx(x, x') * My(y, y'),  My = exp(-lam_y eta - lam_y^2/2),  lam_y = gamma_v sqrt(1/12),
each kernel renormalised exactly on its grid (sum_{x'} Mx pr_x = e^{-r}, sum_{y'} My pr_y = 1),
so E_t[M] = e^{-r} in every state. A type's production loads as exp(gs_bx x + z + gs_bv y + gs_ashift):
gs_bv is the exposure to y and gs_ashift the y=0 level compensation, DERIVED as
    gs_ashift = -bv_comp * log W(gs_bv),   W(b) = sum_k e^{-rk} e^{m_k(b)} / sum_k e^{-rk},
the y-premium discount of a level perpetuity (kp_vy's bv_comp on BGN/KP's value ratio). Unlike
gs_bx's old gs_ashift, the simulator (gs_sim_vy.py) applies it too and refuses a mismatch.

Every continuation expectation is  sum_{y'} MPy[y, y'] * discount_x(V[y'])  -- the regime code's
(1-p) own + p oth with the chain's priced weights; at NY = 1 the solve is gs_solve_reg.py's at
gmreg = [1, 1] (tests/test_gs_vy_parity.py). Same damping / policy-freeze / cycle-average
machinery; the per-regime dicts become per-y-node dicts.

usage: python gs_solve_vy.py [xnum] [tol] [outdir]     (params via GS_PARAM_OVERRIDES)
"""
import numpy as np
from scipy.special import erfc
import os, sys, time, json

xnum = int(sys.argv[1]) if len(sys.argv) > 1 else 161
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6
outdir = sys.argv[3] if len(sys.argv) > 3 else "sol_vy"
HERE = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(HERE, outdir)
os.makedirs(outdir, exist_ok=True)

# ---- parameters (Gomes & Schmid 2021, Table I; see gs_solve_reg.py for the provenance) ----
g = 1.14; delta = 0.02 / 3
rho_x = 0.95 ** (1 / 3); sigma_x = 0.012 * np.sqrt((1 - 0.95 ** (2 / 3)) / (1 - 0.95 ** 2))
rho_z = 0.9 ** (1 / 3); sigma_z = 0.16 * np.sqrt((1 - 0.9 ** (2 / 3)) / (1 - 0.9 ** 2))
r = 0.1 / 12; gamma_x = 0.5; x_bar = 0.0
tau = 0.2; phi = 0.4; kappa_e = 0.025; kappa_b = 0.004; xi = 0.03 / 3
bnum, znum = 20, 200
imin, imax = 0.0, 2000.0
sigma_m = 5
# exposure to x (gs_bx's ladder; 1 = the paper)
gs_bx = 1.0
# ---- the priced state y and this type's exposure to it ----
dt = 1.0 / 12
kappa_y = 0.35            # annual mean reversion of y
gamma_v = 0.0             # price of y-risk per unit Brownian per sqrt(year) (kp_vy's convention)
NY = 1                    # Tauchen nodes for y (1 = no state: reproduces gs_solve_reg at gmreg [1,1])
y_mult = 2.5              # Tauchen half-width in stationary sd
gs_bv = 0.0               # this type's exposure: production * exp(gs_bv * y)
bv_comp = 0.0             # y=0 level compensation exponent -> gs_ashift below
# ---- cost-side exposure (2026-10-08): a fixed operating cost gs_fc per unit capital whose level moves
# with y as exp(gs_bc * y). The firm's cash-flow exposure to y then runs through OPERATING LEVERAGE,
# -gs_fc gs_bc e^{gs_bc y} / (output - delta - cost): convex in profitability within a type, and at y = 0
# every type pays the same cost, so value does not reveal the type. gs_fc = 0 is the production-only model.
gs_fc = 0.0
gs_bc = 0.0
# ---- state-dependent recovery (2026-10-09, Gomes-Schmid's extension): creditors recover phi * exp(gs_bphi y)
# of the liquidation value on default, capped at 1; gs_bphi > 0 makes recovery LOW in bad states, which is
# priced through the debt value and so through equity's refinancing proceeds and leverage choice. Only the
# creditors' recovery term changes; equity's own payoff on default is zero as before.
gs_bphi = 0.0
# ---- investment-cost exposure (2026-10-09): a type's drawn investment cost is multiplied by exp(gs_bi y), so with
# gs_bi < 0 investing is dear in bad states and the firm's exposure to y is proportional to the value of its live
# investment option -- KP14's growth-option mechanism in GS21's own option. The cost draw u ~ U(imin, imax) is
# unchanged; the firm invests if cy*u <= gain, i.e. u <= gain/cy, and pays cy*u. icut_* stay in u-units.
gs_bi = 0.0
# ---- leverage-dependent investment cost (2026-10-28): the cost multiplier becomes exp((gs_bi + gs_bib * b~) y), b~ the
# standardized coupon (b - 0.5)/0.29 of the firm's CURRENT debt, so with gs_bib < 0 a levered firm's investment is dear
# in bad states and an unlevered firm's cheap: the firm's exposure to y through its option is the PRODUCT of its leverage
# and the option's value, with a cross-sectional mean near zero. gs_bib = 0 is the level cost exposure gs_bi alone.
gs_bib = 0.0
# ---- floating coupon (2026-10-09): the coupon paid by the firm AND received by creditors is b * exp(gs_bk y), so
# with gs_bk < 0 debt service rises in bad states (floating-rate debt, spreads widening on roll-over); the firm's
# exposure to y is then proportional to its leverage relative to profit, and the type is invisible at y = 0.
gs_bk = 0.0
# ---- productivity-interacted exposure (2026-10-10): the y exposure is gs_bv * (gs_bzl + gs_bzi * z~), z~ the firm's
# idiosyncratic productivity in stationary-sd units. gs_bzl 1, gs_bzi 0 is the level exposure of gvy25; gs_bzl 0,
# gs_bzi 1 makes the exposure a PRODUCT of the type and a within-firm state -- the KP14/BGN structure -- with zero
# cross-sectional mean, so the market carries none of it. The type then shows only in the slope of the premium in
# productivity, which no linear function of the ranks spans.
gs_bzl = 1.0
gs_bzi = 0.0
# gs_bzb: exposure term gs_bv * gs_bzb * z~ * b~, the PRODUCT of two within-firm states (productivity and the coupon,
# b~ = (b - 0.5)/0.29 on the unit grid): the premium is then bilinear in the two characteristics that reveal them (roe
# and leverage) with no additive part, and its cross-sectional mean is about zero -- the reduced-form latent-product
# factor, in GS21's own states, with no firm type needed.
gs_bzb = 0.0
# gs_bzc (2026-10-24): a leverage-LEVEL term gs_bv * gs_bzc * b~ beside the product, so the exposure is gs_bv b~ (z~ + gs_bzc) when
# gs_bzb = 1: it keeps the sign of leverage for every firm whose z~ > -gs_bzc (the market-share reading: levered firms lose
# demand in tight-credit states, in proportion to how productive the gainers are), and adds an additive leverage part.
gs_bzc = 0.0
# gs_bzq (2026-10-29, 'a firm with extreme leverage should be more exposed to cash-flow risk'): a leverage-CURVATURE term
# gs_bv * gs_bzq * (b~^2 - 1) in the exposure, b~ = (b - 0.5)/0.29 the standardized coupon, so with gs_bzl = 0 the exposure is
# U-shaped in leverage with mean about zero: firms at either leverage extreme load on y, mid-leverage firms load the other
# way. The premium is then non-monotone in the leverage characteristic. gs_bzq = 0 leaves every economy unchanged.
gs_bzq = 0.0
# ---- a SECOND nonlinear loading, on the existing priced productivity shock x (2026-10-12, "larger factor model"):
# the x loading becomes gs_bx + gs_bx2 * (z~^2 - 1)/sqrt(2), a curvature in productivity that is distinct from the
# y loading's product z~ b~. No second chain is needed: x is already priced at gamma_x.
gs_bx2 = 0.0
# ---- a rare priced CRASH in the y chain (2026-10-12, "different distribution"): with probability gs_pj per month the
# chain jumps to its lowest node instead of following the Tauchen transition; the jump is priced by its own factor
# exp(gs_gj) in the kernel (a Q-probability gs_pj * e^{gs_gj} before renormalisation), not by the Gaussian price
# extrapolated to a ten-sd move. gs_pj = 0 is the no-crash chain.
gs_pj = 0.0
gs_gj = 0.0
# ---- a SECOND independent priced chain (2026-10-12, "larger factor models"): NY2 > 1 adds a chain y2 with its own
# price gamma_v2 and mean reversion kappa_y2; this type's exposure to it is gs_bv2 on the SAME shape as gs_bv's
# (gs_bzl + gs_bzi z~ + gs_bzb z~ b~). The state is the joint node (y1, y2) with Kronecker transitions and kernels;
# ygrid (what the cost / coupon / recovery knobs and y_stand see) is the first coordinate, ygrid2 the second.
# NY2 = 1 is the single chain, unchanged.
NY2 = 1
gs_bv2 = 0.0
gamma_v2 = 0.0
kappa_y2 = 0.35
# ---- an EXOGENOUS capital structure (2026-10-21): the coupon the firm may choose at a refinancing event is restricted
# to the grid nodes in [gs_blo, gs_bhi]. gs_blo = gs_bhi = b_f fixes this type's coupon per unit capital at b_f (a
# debt capacity drawn independently of productivity, so leverage varies across firms independently of z); gs_bhi < 1
# alone is a debt-capacity cap with the choice free below it. (0, 1) is the unrestricted choice of GS21.
gs_blo = 0.0
gs_bhi = 1.0
# ---- a FIXED capital structure (2026-10-22): gs_fixcs = 1 removes refinancing altogether (xi must be 0) and makes debt
# scale with capital: at investment the firm issues new debt pari passu with the old at the same coupon per unit
# capital, so b' = b instead of b/g, equity receives (1 - kappa_b)(g - 1) Q(b) for the new debt, and the old debt's
# value per old unit of capital is unchanged (Q(b), not g Q(b/g)). The coupon per unit capital is then the type's
# gs_blo = gs_bhi for the firm's whole life. gs_fixcs = 0 is GS21's debt dynamics.
gs_fixcs = 0
# ---- RISKLESS debt (2026-10-23): gs_riskless = 1 values the debt claim at its riskless coupon value in every state,
# b / (e^r - 1) per unit coupon, as if creditors were made whole on default. Equity then bears the whole exposure of the
# assets to y and x: its premium is the asset premium times (assets / equity). gs_riskless = 0 is GS21's risky debt.
gs_riskless = 0
ov = json.loads(os.environ.get("GS_PARAM_OVERRIDES", "{}"))
globals().update(ov)
rho_y = float(np.exp(-kappa_y * dt))
s_y = float(np.sqrt(1.0 - rho_y ** 2))
lam_y = float(gamma_v * np.sqrt(dt))
_k = np.arange(1, 2001)
_mk = -gs_bv * lam_y * s_y * (1 - rho_y ** _k) / (1 - rho_y) + 0.5 * gs_bv ** 2 * (1 - rho_y ** (2 * _k))
rho_y2 = float(np.exp(-kappa_y2 * dt)); s_y2 = float(np.sqrt(1.0 - rho_y2 ** 2)); lam_y2 = float(gamma_v2 * np.sqrt(dt))
_mk2 = -gs_bv2 * lam_y2 * s_y2 * (1 - rho_y2 ** _k) / (1 - rho_y2) + 0.5 * gs_bv2 ** 2 * (1 - rho_y2 ** (2 * _k))   # 0 at gs_bv2 = 0
_W = float((np.exp(-r * _k) * np.exp(_mk + _mk2)).sum() / np.exp(-r * _k).sum())
gs_ashift = float(-bv_comp * np.log(_W))

sys.path.insert(0, os.path.dirname(HERE))          # variants/, for `common`
from common import solstamp                        # noqa: E402

_snap = solstamp.snapshot(
    {k: v for k, v in globals().items() if not k.startswith('_')},
    [os.path.join(HERE, "gs_solve_vy.py")],
    model="gs_vy",
    skip=("outdir", "HERE", "ov"),
    extra={"outdir": os.path.basename(outdir)},
)
_solution = os.path.join(outdir, "solution.npz")
print(f"[solstamp] gs_vy {os.path.basename(outdir)} solve_id={_snap.solve_id}", flush=True)

_hit = solstamp.lookup(_snap.solve_id)
if _hit and not os.environ.get("GS_SOLVE_FORCE"):
    _probs = solstamp.artifact_problems(_hit)
    if not _probs:
        print(f"cached: solve_id {_snap.solve_id} already solved ({_hit['total_bytes']:,} B) -- nothing to do")
        sys.exit(0)
    print("[solstamp] manifest exists but artifacts do not match; re-solving:")
    for _p in _probs[:5]:
        print(f"  - {_p}")
elif os.path.exists(_solution):
    _prior = solstamp._find_by_artifacts([_solution])
    if _prior and _prior["solve_id"] != _snap.solve_id:
        print(f"[solstamp] {os.path.basename(outdir)}/solution.npz belongs to solve_id {_prior['solve_id']}; this run wants {_snap.solve_id}. Differences:")
        for _k2, _a, _b in solstamp.diff_params(_prior["params"], _snap.params)[:10]:
            print(f"    {_k2}: {_a!r} -> {_b!r}")
    elif _prior:
        print(f"[solstamp] {os.path.basename(outdir)}/solution.npz is recorded and matches solve_id {_snap.solve_id}; re-solving because GS_SOLVE_FORCE is set")
    else:
        print(f"[solstamp] {os.path.basename(outdir)}/solution.npz exists but its provenance is unrecorded; re-solving")


def tauchen(sigma, rho, multiple, n):
    sdz = np.sqrt(sigma ** 2 / (1 - rho ** 2))
    if n == 1:
        return np.zeros(1), np.ones((1, 1))
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
_bh = 0.5 * (bgrid[1] - bgrid[0])                                      # half a grid step: [lo, hi] snaps to the nearest nodes
_bmask = (bgrid >= gs_blo - _bh) & (bgrid <= gs_bhi + _bh)            # coupons the firm may refinance to
assert _bmask.any(), f"no coupon node within half a step of [{gs_blo}, {gs_bhi}]"
def restrict_b(obj):
    """obj (z, x, b, b'): -inf outside the allowed b' nodes, so argmax picks within the capital-structure set"""
    if not _bmask.all():
        obj[..., ~_bmask] = -np.inf
    return obj
if gs_fixcs:
    assert xi == 0.0, "a fixed capital structure (gs_fixcs = 1) has no refinancing: set xi = 0"
    assert _bmask.sum() == 1, "a fixed capital structure needs one coupon node: gs_blo = gs_bhi"
def qi_no(Q0):
    """old debt's value per OLD unit of capital after an investment without refinancing"""
    return Q0 if gs_fixcs else at_bg(g * Q0)
def pi_dn(iy, Q0, EPI):
    """equity's payoff when investing without refinancing: cash flow (+ new-debt proceeds under gs_fixcs) + continuation"""
    return issue(pi_R[iy] + (1 - kappa_b) * (g - 1) * Q0) + EPI if gs_fixcs else pi_R_e[iy] + at_bg(EPI)
ygrid, pr_y = tauchen(s_y, rho_y, y_mult, NY)
if gs_pj > 0 and NY > 1:
    _jump = np.zeros((NY, NY)); _jump[:, 0] = 1.0
    pr_y = (1 - gs_pj) * pr_y + gs_pj * _jump

# kernels, exactly renormalized: E[Mx | x] = e^{-r}, E[My | y] = 1
Mx = np.exp(-r - 0.5 * gamma_x ** 2
            - gamma_x * (xgrid[None, :] - (1 - rho_x) * x_bar - rho_x * xgrid[:, None]) / sigma_x)
Mx = Mx * (np.exp(-r) / (Mx * pr_x).sum(axis=1))[:, None]
MPx = Mx * pr_x
if NY > 1:
    eta = (ygrid[None, :] - rho_y * ygrid[:, None]) / s_y
    My = np.exp(-lam_y * eta - 0.5 * lam_y ** 2)
    if gs_pj > 0:
        # kernel on the mixture: the Tauchen part keeps the Gaussian price, the jump part carries e^{gs_gj}
        _ptau = (pr_y - gs_pj * _jump) / (1 - gs_pj)
        _num = (1 - gs_pj) * _ptau * My + gs_pj * _jump * np.exp(gs_gj)
        My = _num / np.where(pr_y > 0, pr_y, 1.0)
    My = My / (My * pr_y).sum(axis=1)[:, None]
else:
    My = np.ones((1, 1))
NY1 = NY
if NY2 > 1:
    ygrid2, pr_y2 = tauchen(s_y2, rho_y2, y_mult, NY2)
    _eta2 = (ygrid2[None, :] - rho_y2 * ygrid2[:, None]) / s_y2
    My2 = np.exp(-lam_y2 * _eta2 - 0.5 * lam_y2 ** 2); My2 = My2 / (My2 * pr_y2).sum(axis=1)[:, None]
    pr_y = np.kron(pr_y, pr_y2); My = np.kron(My, My2)          # independent chains: joint node i = i1 * NY2 + i2
    My = My / (My * pr_y).sum(axis=1)[:, None]
    ygrid, ygrid2 = np.repeat(ygrid, NY2), np.tile(ygrid2, NY1)   # the joint nodes' two coordinates
    NY = NY1 * NY2
else:
    ygrid2 = np.zeros(NY)
MPy = My * pr_y                                             # (NY, NY): priced y-transition weights
print(f"y chain: {NY} nodes ({NY1} x {NY2}) over +-{y_mult} sd, rho_y {rho_y:.4f}, lam_y {lam_y:.4f}, lam_y2 {lam_y2:.4f}, gs_bv2 {gs_bv2}; gs_bv {gs_bv}, gs_ashift {gs_ashift:.5f}, gs_fc {gs_fc}, gs_bc {gs_bc}, gs_bphi {gs_bphi}, gs_bi {gs_bi}, gs_bib {gs_bib}, gs_bk {gs_bk}, gs_bzl {gs_bzl}, gs_bzi {gs_bzi}, gs_bzb {gs_bzb}, gs_bx2 {gs_bx2}, gs_pj {gs_pj}, gs_gj {gs_gj}, coupon set [{gs_blo}, {gs_bhi}] -> nodes {np.round(bgrid[_bmask], 4).tolist()}", flush=True)

mn = np.linspace(-4 * sigma_m, 4 * sigma_m, 161)
mw = np.exp(-0.5 * (mn / sigma_m) ** 2)
mw /= mw.sum()
# The smoothing shock enters only through sum_m mw[m] max(P + mn[m], 0) and sum_m mw[m] 1{P + mn[m] > 0},
# both 1-D functions of P: with k(P) the first node with mn[k] > -P they are P*C[k] + D[k] and C[k], for
# the right-cumulative sums C, D below. Exact (summation order aside, ~1e-13 against gs_solve_reg's
# (z, x, b, 161) broadcast, check_parity.py) and ~100x cheaper, which is what makes NY y-nodes affordable.
_Cw = np.concatenate((np.cumsum(mw[::-1])[::-1], [0.0]))
_Cwm = np.concatenate((np.cumsum((mw * mn)[::-1])[::-1], [0.0]))

def smooth(P):
    k = np.searchsorted(mn, -P, side="right")
    C = _Cw[k]
    return P * C + _Cwm[k], C

# per-y-node production: (NY, znum, xnum, bnum)
_ztil = zgrid / np.sqrt(sigma_z ** 2 / (1 - rho_z ** 2))                           # z in stationary-sd units
_btil = (bgrid - 0.5) / 0.29                                                       # coupon in grid-sd units
_expo = gs_bv * (gs_bzl + gs_bzi * _ztil[:, None] + gs_bzb * _ztil[:, None] * _btil[None, :] + gs_bzc * _btil[None, :] + gs_bzq * (_btil[None, :] ** 2 - 1.0))   # (znum, bnum)
_expo2 = gs_bv2 * (gs_bzl + gs_bzi * _ztil[:, None] + gs_bzb * _ztil[:, None] * _btil[None, :] + gs_bzc * _btil[None, :] + gs_bzq * (_btil[None, :] ** 2 - 1.0))  # the second chain's
_bxz = gs_bx + gs_bx2 * (_ztil ** 2 - 1.0) / np.sqrt(2.0)                          # (znum,) x loading by productivity
_prod = np.exp(_bxz[None, :, None, None] * xgrid[None, None, :, None] + zgrid[None, :, None, None] + _expo[None, :, None, :] * ygrid[:, None, None, None] + _expo2[None, :, None, :] * ygrid2[:, None, None, None] + gs_ashift)
_cost = delta + gs_fc * np.exp(gs_bc * ygrid)[:, None, None, None]            # (NY, 1, 1, 1)
cy = np.exp((gs_bi + gs_bib * _btil[None, :]) * ygrid[:, None])                 # (NY, bnum) investment-cost multiplier by y node and current coupon
ky = np.exp(gs_bk * ygrid)                                                      # (NY,) coupon multiplier
pi_R = (1 - tau) * (_prod - _cost) - (1 - tau) * (ky[:, None, None, None] * bgrid[None, None, None, :])
_phi_y = np.minimum(phi * np.exp(gs_bphi * ygrid), 1.0)[:, None, None, None]           # (NY, 1, 1, 1)
recov = _phi_y * (1 - delta + _prod) * np.ones((1, 1, 1, bnum))


def issue(cf):
    return cf if kappa_e == 0 else cf + kappa_e * np.minimum(cf, 0.0)


pi_R_e = issue(pi_R)

from numpy import searchsorted
def _b_interp_weights(btarget):
    jj = np.clip(searchsorted(bgrid, btarget) - 1, 0, bnum - 2)
    ww = np.clip((btarget - bgrid[jj]) / (bgrid[jj + 1] - bgrid[jj]), 0, 1)
    return jj, ww
_jg, _wg = _b_interp_weights(bgrid / g)
def at_bg(V):
    return V[..., _jg] * (1 - _wg) + V[..., _jg + 1] * _wg

def discount(V):
    """E[Mx V(z', x', b)] over (z', x') for one y-node table V (z, x, b)"""
    Vz = np.einsum("ij,jxb->ixb", pr_z, V, optimize=True)
    return np.einsum("xy,iyb->ixb", MPx, Vz, optimize=True)

def mix_y(tabs):
    """tabs: list over y' of (z, x, b) x-discounted tables -> list over y of the priced y-mixture"""
    stack = np.stack(tabs)                                   # (NY, z, x, b)
    return np.einsum("ab,bzxc->azxc", MPy, stack, optimize=True)

t0 = time.time()
_print_every = int(os.environ.get("GS_VY_PRINT_EVERY", "200"))   # progress cadence only; underscore: not in the id
S = [dict(P_up=np.zeros((znum, xnum, bnum)), P_down=np.zeros((znum, xnum, bnum)),
          Q0=np.zeros((znum, xnum, bnum)), prob_up=np.zeros((znum, xnum, bnum)),
          prob_dn=np.zeros((znum, xnum, bnum)), b0idx=np.zeros((znum, xnum, bnum), dtype=int),
          bIidx=np.zeros((znum, xnum, bnum), dtype=int), icut_up=np.zeros((znum, xnum, bnum)),
          icut_dn=np.zeros((znum, xnum, bnum))) for _ in range(NY)]
prob_delta = [1.0] * NY
acc = None
for it in range(60000):
    qerr = perr = 0.0
    pre = []
    for iy in range(NY):
        v = S[iy]
        QI = g * v["Q0"]; QI_no = qi_no(v["Q0"])
        Qn_up = v["prob_up"] * QI_no + (1 - v["prob_up"]) * v["Q0"]
        Qn_dn = v["prob_dn"] * QI_no + (1 - v["prob_dn"]) * v["Q0"]
        Pu_pert, ind_up = smooth(v["P_up"])
        Pd_pert, ind_dn = smooth(v["P_down"])
        inner_up = (ky[iy] * bgrid[None, None, :] + Qn_up) * ind_up + recov[iy] * (1 - ind_up)
        inner_dn = (ky[iy] * bgrid[None, None, :] + Qn_dn) * ind_dn + recov[iy] * (1 - ind_dn)
        pre.append(dict(Pu=Pu_pert, Pd=Pd_pert, iu=inner_up, idn=inner_dn))
    # ---- debt sweep: x-discount every y' table once, then the priced y-mixture ----
    iu_mix = mix_y([discount(p["iu"]) for p in pre])
    idn_mix = mix_y([discount(p["idn"]) for p in pre])
    for iy in range(NY):
        v = S[iy]
        Q0_new = xi * iu_mix[iy] + (1 - xi) * idn_mix[iy]
        if gs_riskless:
            Q0_new = np.broadcast_to((ky[iy] * bgrid / (np.exp(r) - 1.0))[None, None, :], Q0_new.shape).copy()
        qerr = max(qerr, np.abs(Q0_new - v["Q0"]).max())
        v["Q0"] = 0.5 * Q0_new + 0.5 * v["Q0"]
    # ---- equity sweep ----
    Pu_mix = mix_y([discount(p["Pu"]) for p in pre])
    Pd_mix = mix_y([discount(p["Pd"]) for p in pre])
    for iy in range(NY):
        v = S[iy]
        QI = g * v["Q0"]; QI_no = qi_no(v["Q0"])
        EP0 = xi * Pu_mix[iy] + (1 - xi) * Pd_mix[iy]
        EPI = g * EP0
        if it % 25 == 0 or it < 50:
            obj0 = issue((pi_R[iy] - v["Q0"])[..., None] + (1 - kappa_b) * v["Q0"][..., None, :])
            obj0 += EP0[..., None, :]
            b0new = restrict_b(obj0).argmax(axis=3)
            P0_up = np.take_along_axis(obj0, b0new[..., None], 3)[..., 0]
            del obj0
            objI = issue((pi_R[iy] - QI_no)[..., None] + (1 - kappa_b) * QI[..., None, :])
            objI += EPI[..., None, :]
            bInew = restrict_b(objI).argmax(axis=3)
            PI_up = np.take_along_axis(objI, bInew[..., None], 3)[..., 0]
            del objI
            v["b0idx"], v["bIidx"] = b0new, bInew
            P0_dn = pi_R_e[iy] + EP0
            PI_dn = pi_dn(iy, v["Q0"], EPI)
            v["icut_up"] = np.clip((PI_up - P0_up) / cy[iy], imin, imax)
            v["icut_dn"] = np.clip((PI_dn - P0_dn) / cy[iy], imin, imax)
            pu_new = (v["icut_up"] - imin) / (imax - imin)
            pd_new = (v["icut_dn"] - imin) / (imax - imin)
            prob_delta[iy] = max(np.abs(pu_new - v["prob_up"]).max(), np.abs(pd_new - v["prob_dn"]).max())
            v["prob_up"], v["prob_dn"] = pu_new, pd_new
        else:
            P0_up = (issue(pi_R[iy] - v["Q0"] + (1 - kappa_b) * np.take_along_axis(v["Q0"], v["b0idx"], 2))
                     + np.take_along_axis(EP0, v["b0idx"], 2))
            PI_up = (issue(pi_R[iy] - QI_no + (1 - kappa_b) * np.take_along_axis(QI, v["bIidx"], 2))
                     + np.take_along_axis(EPI, v["bIidx"], 2))
            P0_dn = pi_R_e[iy] + EP0
            PI_dn = pi_dn(iy, v["Q0"], EPI)
        P_up_new = v["prob_up"] * (PI_up - cy[iy] * 0.5 * (v["icut_up"] + imin)) + (1 - v["prob_up"]) * P0_up
        P_dn_new = v["prob_dn"] * (PI_dn - cy[iy] * 0.5 * (v["icut_dn"] + imin)) + (1 - v["prob_dn"]) * P0_dn
        perr = max(perr, np.abs(P_up_new - v["P_up"]).max(), np.abs(P_dn_new - v["P_down"]).max())
        v["P_up"] = 0.5 * P_up_new + 0.5 * v["P_up"]
        v["P_down"] = 0.5 * P_dn_new + 0.5 * v["P_down"]
        v["_P0_up"], v["_PI_up"], v["_P0_dn"], v["_PI_dn"] = P0_up, PI_up, P0_dn, PI_dn
    if not np.isfinite(perr):
        raise RuntimeError(f"diverged at sweep {it}")
    vscale = max(1.0, max(np.abs(S[iy]["P_up"]).max() for iy in range(NY)))
    if it == 5000:
        acc = [dict(P_up=0.0, P_down=0.0, Q0=0.0, n=0) for _ in range(NY)]
    if it >= 5000 and acc is not None:
        for iy in range(NY):
            acc[iy]["P_up"] += S[iy]["P_up"]; acc[iy]["P_down"] += S[iy]["P_down"]
            acc[iy]["Q0"] += S[iy]["Q0"]; acc[iy]["n"] += 1
    if (qerr / vscale < tol * 20 and perr / vscale < tol * 20 and it > 100 and it % 25 == 24
            and max(prob_delta) / max(max(S[iy]["prob_up"].max() for iy in range(NY)), 1e-9) < 1e-4):
        break
    if it == 5600:
        for iy in range(NY):
            S[iy]["P_up"] = acc[iy]["P_up"] / acc[iy]["n"]
            S[iy]["P_down"] = acc[iy]["P_down"] / acc[iy]["n"]
            S[iy]["Q0"] = acc[iy]["Q0"] / acc[iy]["n"]
        pre = []
        for iy in range(NY):
            v = S[iy]
            Pu_pert, _ = smooth(v["P_up"]); Pd_pert, _ = smooth(v["P_down"])
            pre.append(dict(Pu=Pu_pert, Pd=Pd_pert))
        Pu_mix = mix_y([discount(p["Pu"]) for p in pre])
        Pd_mix = mix_y([discount(p["Pd"]) for p in pre])
        for iy in range(NY):
            v = S[iy]
            QI = g * v["Q0"]; QI_no = qi_no(v["Q0"])
            EP0 = xi * Pu_mix[iy] + (1 - xi) * Pd_mix[iy]
            EPI = g * EP0
            obj0 = issue((pi_R[iy] - v["Q0"])[..., None] + (1 - kappa_b) * v["Q0"][..., None, :])
            obj0 += EP0[..., None, :]
            v["b0idx"] = restrict_b(obj0).argmax(axis=3)
            v["_P0_up"] = np.take_along_axis(obj0, v["b0idx"][..., None], 3)[..., 0]
            del obj0
            objI = issue((pi_R[iy] - QI_no)[..., None] + (1 - kappa_b) * QI[..., None, :])
            objI += EPI[..., None, :]
            v["bIidx"] = restrict_b(objI).argmax(axis=3)
            v["_PI_up"] = np.take_along_axis(objI, v["bIidx"][..., None], 3)[..., 0]
            del objI
            v["_P0_dn"] = pi_R_e[iy] + EP0; v["_PI_dn"] = pi_dn(iy, v["Q0"], EPI)
            v["icut_up"] = np.clip((v["_PI_up"] - v["_P0_up"]) / cy[iy], imin, imax)
            v["icut_dn"] = np.clip((v["_PI_dn"] - v["_P0_dn"]) / cy[iy], imin, imax)
        print(f"cycle-averaged; stopping", flush=True)
        break
    if it % _print_every == 0:
        print(f"sweep {it}: qerr {qerr:.3e} perr {perr:.3e}  ({time.time()-t0:.0f}s)", flush=True)

print(f"converged: sweep {it}, qerr {qerr:.2e}, perr {perr:.2e} in {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(os.path.join(outdir, "solution.npz"),
                    xgrid=xgrid, zgrid=zgrid, bgrid=bgrid, pr_x=pr_x, pr_z=pr_z,
                    Mx=Mx, ygrid=ygrid, pr_y=pr_y, My=My, MPy=MPy,
                    P_up=np.stack([S[iy]["P_up"] for iy in range(NY)]),
                    P_down=np.stack([S[iy]["P_down"] for iy in range(NY)]),
                    P0_up=np.stack([S[iy]["_P0_up"] for iy in range(NY)]),
                    PI_up=np.stack([S[iy]["_PI_up"] for iy in range(NY)]),
                    P0_down=np.stack([S[iy]["_P0_dn"] for iy in range(NY)]),
                    PI_down=np.stack([S[iy]["_PI_dn"] for iy in range(NY)]),
                    Q0=np.stack([S[iy]["Q0"] for iy in range(NY)]),
                    QI=np.stack([g * S[iy]["Q0"] for iy in range(NY)]),
                    QI_no=np.stack([qi_no(S[iy]["Q0"]) for iy in range(NY)]),
                    icut_up=np.stack([S[iy]["icut_up"] for iy in range(NY)]),
                    icut_dn=np.stack([S[iy]["icut_dn"] for iy in range(NY)]),
                    b_refin_0=np.stack([bgrid[S[iy]["b0idx"]] for iy in range(NY)]),
                    b_refin_I=np.stack([bgrid[S[iy]["bIidx"]] for iy in range(NY)]),
                    params=np.array([g, delta, rho_x, sigma_x, rho_z, sigma_z, r, gamma_x, tau, phi,
                                     kappa_b, xi, imin, imax, sigma_m]),
                    solve_id=_snap.solve_id,
                    gs_bx=gs_bx, gs_bv=gs_bv, gs_ashift=gs_ashift, bv_comp=bv_comp, gs_fc=gs_fc, gs_bc=gs_bc, gs_bphi=gs_bphi, gs_bi=gs_bi, gs_bib=gs_bib, cy=cy, gs_bk=gs_bk, ky=ky, gs_bzl=gs_bzl, gs_bzi=gs_bzi, gs_bzb=gs_bzb, gs_bzc=gs_bzc, gs_bzq=gs_bzq, gs_bx2=gs_bx2, gs_pj=gs_pj, gs_gj=gs_gj, gs_blo=gs_blo, gs_bhi=gs_bhi, gs_fixcs=gs_fixcs, gs_riskless=gs_riskless,
                    gamma_v=gamma_v, kappa_y=kappa_y, NY=NY, y_mult=y_mult, lam_y=lam_y, rho_y=rho_y, s_y=s_y,
                    ygrid2=ygrid2, NY1=NY1, NY2=NY2, gs_bv2=gs_bv2, gamma_v2=gamma_v2, kappa_y2=kappa_y2,
                    kappa_e=kappa_e, xnum=xnum, bnum=bnum, znum=znum, tol=tol,
                    params_json=json.dumps(_snap.params, sort_keys=True, default=str))
print("saved", _solution)

_man = solstamp.record(
    _snap, [_solution], tag=os.path.basename(outdir),
    achieved={"exit": "cycle_capped" if it >= 5600 else "tolerance",
              "sweeps": int(it),
              "qerr_rel": float(qerr / vscale),
              "perr_rel": float(perr / vscale),
              "vscale": float(vscale),
              "threshold_enforced": float(tol * 20),
              "tol_requested": float(tol)})
print(f"[solstamp] recorded solve_id {_snap.solve_id} ({_man['total_bytes']:,} B, committable={_man['committable']})")
if not _man["committable"]:
    print(f"[solstamp] artifact exceeds {solstamp.SMALL_ARTIFACT_BYTES:,} B -- kept out of git; the manifest in experiments/solfiles/ is the durable record")
