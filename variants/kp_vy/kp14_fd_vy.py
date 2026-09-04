"""(2 lambda-regimes x NY y-nodes) G-function solve for kp_gamy: the gamma-regime pair of
kp14_fd_gam.py generalized to the OU price-of-risk state y (generator Qy from parameters).
Writes G_func_gamy.csv with columns G_up_y{i}, G_down_y{i} for i = 0..NY-1 (per unit lambda_bar_f).
At g_lo == g_hi == 1 all y-columns coincide with the baseline G_up/G_down."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parameters_kp14 import *
_ft = int(os.environ.get("KP_VY_TYPE", "0"))
rho_y = rho_ty[_ft]

n = 1000
max_eps, min_eps = 5.0, 0.01
deps = (max_eps - min_eps) / (n - 1)
eps_pts = np.linspace(min_eps, max_eps, n)

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

dt_fd = 0.5
G = np.ones((n, NS))
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
blocks = [[None] * NS for _ in range(NS)]
for j in range(NS):
    Aj = sp.eye(n) / dt_fd - (fd_matrix(rho_j[j]) + Qs[j, j] * sp.eye(n))
    for k2 in range(NS):
        blocks[j][k2] = Aj if k2 == j else ((-Qs[j, k2]) * sp.eye(n) if Qs[j, k2] != 0 else None)
Mat = sp.bmat(blocks, format="csc")
lu = spla.splu(Mat)
print(f"factorized {NS * n}x{NS * n} system in {time.time()-t0:.0f}s", flush=True)

for it in range(1_000_000):
    rhs = (G / dt_fd + util).T.reshape(-1)
    Gn = lu.solve(rhs).reshape(NS, n).T
    err = np.linalg.norm(Gn - G)
    G = Gn
    if it % 20 == 0:
        print(f"iter {it}: err {err:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    if err < 1e-8:
        break

cols = {"eps": eps_pts}
for iy in range(NY):
    cols[f"G_up_y{iy}"] = G[:, 2 * iy]
    cols[f"G_down_y{iy}"] = G[:, 2 * iy + 1]
out = pd.DataFrame(cols)
gout = os.environ.get("KP_VY_GOUT", f"G_vy{_ft}.csv")
out.to_csv(gout)
print(f"saved {gout} (converged iter {it}, err {err:.2e}, {time.time()-t0:.0f}s)")
