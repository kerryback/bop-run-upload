"""(2 lambda-regimes x NY y-nodes) G-function solve for kp_gamy: the gamma-regime pair of
kp14_fd_gam.py generalized to the OU price-of-risk state y (generator Qy from parameters).
Writes G_func_gamy.csv with columns G_up_y{i}, G_down_y{i} for i = 0..NY-1 (per unit lambda_bar_f).
At g_lo == g_hi == 1 all y-columns coincide with the baseline G_up/G_down.

Solved directly: (F + Q) G = -util is linear. See WORKING.md 17i.

2026-09-18, y_risk_neutral = 1 (the default): PVGO is e^{b y} G, a claim under Q, so the solve is
done for H = e^{b y} G under the RISK-NEUTRAL OU generator (drift -kappa_y*y - gamma_v*sigma_y),
with the y-free discount rho0 and the flow e^{b y} * util, on a y-grid wide enough to hold the
Q-stationary distribution; G = e^{-b y} H is then written at the y_grid table nodes only, so the
file layout is unchanged. y_risk_neutral = 0 is the pre-fix solve, bit for bit. The defect this
corrects: the pre-fix solve carried the y premium as a constant beta*gamma_v*sigma_y in the
discount rate, KP14's eq. (11) shape, which is exact for a GBM and wrong for a mean-reverting
state whose Girsanov adjustment saturates. See parameters_kp14.py and, for the identities,
tests/test_risk_neutral_pricing.py."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parameters_kp14 import *
from parameters_kp14 import _sub, _i_tab          # underscore names are not star-exported
_ft = int(os.environ.get("KP_VY_TYPE", "0"))
rho_y = rho_ty[_ft]

n = 1000
max_eps, min_eps = 5.0, 0.01
deps = (max_eps - min_eps) / (n - 1)
eps_pts = np.linspace(min_eps, max_eps, n)

_NY_TAB, _y_tab = NY, y_grid
if y_risk_neutral:
    # solve grid: every KP_VY_GSUB-th node of parameters' y_solve, trimmed to where the Q-measure
    # and the table grid live. It contains y_grid exactly.
    _gsub = int(os.environ.get("KP_VY_GSUB", "2"))           # solve nodes per table interval
    assert _sub % _gsub == 0
    _step = _sub // _gsub
    _lo = min(-y_max, y_mean_Q) - 6.0; _hi = y_max + 2.5
    _idx = np.arange(_i_tab[0] % _step, len(y_solve), _step)
    _idx = _idx[(y_solve[_idx] >= _lo - 1e-9) & (y_solve[_idx] <= _hi + 1e-9)]
    y_grid = y_solve[_idx]; NY = len(y_grid)
    _tab_pos = np.searchsorted(_idx, _i_tab); assert np.array_equal(_idx[_tab_pos], _i_tab)
    Qy = build_generator(y_grid, -kappa_y * y_grid - gamma_v * sigma_y)
    rho_y = rho0_at(y_grid)
    _b = type_bv[_ft]
    _Afine = [a[_idx] for a in coef_on_solve_grid(_ft)]
    def A_y(ep, u, yv, f):                                   # A at solve-grid nodes (not clamped)
        i = int(np.argmin(np.abs(y_grid - yv)))
        a0, a1, a2, a3 = (a[i] for a in _Afine)
        return a0 + (ep - 1) * a1 + (u - 1) * a2 + (ep - 1) * (u - 1) * a3
    print(f"risk-neutral G solve: y in [{y_grid[0]:.3f}, {y_grid[-1]:.3f}], {NY} nodes "
          f"(table has {_NY_TAB}); Q-mean of y = {y_mean_Q:.2f}", flush=True)

NS = 2 * NY                    # state j = 2*iy + (0 for lambda-H, 1 for lambda-L)
lam_rate = np.tile([lambda_H, lambda_L], NY)
rho_j = np.repeat(rho_y, 2)
Qs = np.zeros((NS, NS))
for iy in range(NY):
    Qs[2 * iy, 2 * iy + 1] += mu_L
    Qs[2 * iy + 1, 2 * iy] += mu_H
    for jy in range(NY):
        if jy != iy:
            Qs[2 * iy, 2 * jy] += Qy[iy, jy]
            Qs[2 * iy + 1, 2 * jy + 1] += Qy[iy, jy]
np.fill_diagonal(Qs, 0.0)
np.fill_diagonal(Qs, -Qs.sum(axis=1))

util = np.empty((n, NS))
for j in range(NS):
    iy = j // 2
    util[:, j] = lam_rate[j] * C * A_y(eps_pts, 1.0, y_grid[iy], _ft) ** (1 / (1 - alpha))
    if y_risk_neutral:
        util[:, j] *= np.exp(_b * y_grid[iy])                # flow of H = e^{b y} G

mu_epsF = np.maximum(-theta_eps * (eps_pts - 1), 0)
mu_epsB = -np.maximum(theta_eps * (eps_pts - 1), 0)
quad = 0.5 * sigma_eps ** 2 * eps_pts
I_F = (-theta_eps * (eps_pts - 1)) > 0
I_B = (-theta_eps * (eps_pts - 1)) < 0

def fd_matrix(rho_val):
    diag_m2 = quad / deps ** 2
    diag_m1 = I_B * (-mu_epsB) / deps + quad / deps ** 2
    diag_0 = -(rho_val) + I_B * mu_epsB / deps + I_F * (-mu_epsF) / deps - 2 * quad / deps ** 2
    diag_p1 = I_F * mu_epsF / deps + quad / deps ** 2
    diag_p2 = quad / deps ** 2
    diag_m2 = diag_m2.copy(); diag_p2 = diag_p2.copy(); diag_m1 = diag_m1.copy(); diag_0 = diag_0.copy(); diag_p1 = diag_p1.copy()
    diag_m2[:-1] = 0; diag_m2[-1] = quad[-1] / deps ** 2
    diag_m1[-1] = I_B[-1] * (-mu_epsB[-1]) / deps - 2 * quad[-1] / deps ** 2
    diag_0[-1] = -(rho_val) + I_B[-1] * mu_epsB[-1] / deps + quad[-1] / deps ** 2
    diag_p2[1:] = 0; diag_p2[0] = quad[0] / deps ** 2
    diag_p1[0] = I_F[0] * mu_epsF[0] / deps + -2 * quad[0] / deps ** 2
    diag_0[0] = -(rho_val) + I_F[0] * (-mu_epsF[0]) / deps + quad[0] / deps ** 2
    diags = [np.roll(diag_m2, -2), np.roll(diag_m1, -1), diag_0, diag_p1, diag_p2]
    return sp.diags(diags, [-2, -1, 0, 1, 2], shape=(n, n))

t0 = time.time()
# This was an implicit-Euler iteration  Mat @ G_new = G/dt + util  with Mat = I/dt - (F+Q).
# Its fixed point satisfies  [I/dt - (F+Q)] G = G/dt + util,  i.e.  (F + Q) G = -util,
# which is linear -- so solve it once rather than creep toward it.
#
# The iteration could not be salvaged by loosening its tolerance. It compared an
# ABSOLUTE Frobenius norm against 1e-8 while ||G|| here is 3e7 to 2e8, i.e. it demanded
# relative precision of 3e-16 (type 0), 1.4e-16 (type 1) and 5e-17 (type 2) -- at or
# below float64 eps (2.2e-16). It was unsatisfiable by construction; a 2026-09-04 run
# sat at its noise floor for 228,000 iterations, and would then have fallen out of the
# loop and written the table while printing "converged".
blocks = [[None] * NS for _ in range(NS)]
for j in range(NS):
    for k2 in range(NS):
        if k2 == j:
            blocks[j][k2] = fd_matrix(rho_j[j]) + Qs[j, j] * sp.eye(n)
        elif Qs[j, k2] != 0:
            blocks[j][k2] = Qs[j, k2] * sp.eye(n)
Mat = sp.bmat(blocks, format="csc")
rhs = (-util).T.reshape(-1)
G = spla.spsolve(Mat, rhs).reshape(NS, n).T

resid = np.linalg.norm(Mat @ G.T.reshape(-1) - rhs)
scale = max(float(np.linalg.norm(rhs)), 1.0)
rel = resid / scale
if not rel < 1e-6:
    raise RuntimeError(f"G solve did not converge: relative residual {rel:.3e} "
                       f"(absolute {resid:.3e}) exceeds 1e-6")
print(f"solved {NS * n}x{NS * n} system directly in {time.time()-t0:.1f}s; "
      f"relative residual {rel:.2e}", flush=True)

cols = {"eps": eps_pts}
if y_risk_neutral:
    for k, iy in enumerate(_tab_pos):                        # back to G, at the table nodes only
        cols[f"G_up_y{k}"] = G[:, 2 * iy] * np.exp(-_b * y_grid[iy])
        cols[f"G_down_y{k}"] = G[:, 2 * iy + 1] * np.exp(-_b * y_grid[iy])
else:
  for iy in range(NY):
    cols[f"G_up_y{iy}"] = G[:, 2 * iy]
    cols[f"G_down_y{iy}"] = G[:, 2 * iy + 1]
out = pd.DataFrame(cols)
gout = os.environ.get("KP_VY_GOUT", f"G_vy{_ft}.csv")
out.to_csv(gout)
print(f"saved {gout} (direct solve, relative residual {rel:.2e}, {time.time()-t0:.1f}s)")
