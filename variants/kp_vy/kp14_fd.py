import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, vstack
import scipy.sparse.linalg as spla

from parameters_kp14 import *

# G value grid
n = 1000
max_eps = 5.0
min_eps = 0.01
deps = (max_eps - min_eps) / (n - 1)
eps_pts = np.linspace(min_eps, max_eps, n)


# Initialization
vec_prev = np.ones((n, 2)) # store G1 as first column, G2 as second
G = vec_prev.copy()
G_epsF = np.zeros((n, 2))
G_epsB = np.zeros((n, 2))
G_epsC = np.zeros((n, 2))
G_eps2 = np.zeros((n, 2))
rhs = np.zeros((n, 2))

# Main loop
n_iter = 1_000_000
dt = 0.5

for i in range(n_iter):
    # Compute derivatives
    G_epsF[:-1, :] = (G[1:, :] - G[:-1, :]) / deps
    G_epsF[-1, :] = G_epsF[-2, :]

    G_epsB[1:, :] = G_epsF[:-1, :]
    G_epsB[0, :] = G_epsB[1, :]

    G_epsC = 0.5 * G_epsF + 0.5 * G_epsB

    G_eps2[1:-1, :] = (G[2:, :] - 2 * G[1:-1, :] + G[:-2, :]) / deps**2
    G_eps2[0, :] = G_eps2[1, :]
    G_eps2[-1, :] = G_eps2[-2, :]

    I_F = (-theta_eps * (eps_pts - 1)) > 0
    I_B = (-theta_eps * (eps_pts - 1)) < 0

    ## build FD matrix
    mu_epsF = np.maximum(-theta_eps * (eps_pts - 1), 0)
    mu_epsB = -np.maximum(theta_eps * (eps_pts - 1), 0)
    quad = 0.5 * sigma_eps**2 * eps_pts

    # main diagonals
    diag_m2 = quad / deps**2
    diag_m1 = I_B*(-mu_epsB) / deps + quad / deps**2
    diag_0  = -(rho) + I_B*mu_epsB / deps +I_F*(-mu_epsF) / deps - 2 * quad / deps**2
    diag_p1 = I_F*mu_epsF / deps + quad / deps**2
    diag_p2 = quad / deps**2

    # endpoint adjustments
    diag_m2[:-1] = 0
    diag_m2[-1] = quad[-1] / deps**2
    diag_m1[-1] = I_B[-1]*(-mu_epsB[-1]) / deps -2*quad[-1] / deps**2
    diag_0[-1] = -(rho) + I_B[-1]*mu_epsB[-1] / deps + quad[-1] / deps**2

    diag_p2[1:] = 0
    diag_p2[0] = quad[0] / deps**2
    diag_p1[0] = I_F[0]*mu_epsF[0] / deps + -2*quad[0] / deps**2
    diag_0[0] = -(rho) + I_F[0]*(-mu_epsF[0]) / deps + quad[0] / deps**2

    diags = [np.roll(diag_m2, -2), np.roll(diag_m1, -1), diag_0, diag_p1, diag_p2] 
    offsets = [-2, -1, 0, 1, 2]

    # set up matrices
    mat = sp.diags(diags, offsets, shape=(n, n))
    mat = sp.eye(n) / dt - mat

    util = C * A(eps_pts, 1)**(1 / (1 - alpha))

    rhs1 = (vec_prev[:,0] / dt + util).reshape([-1, 1])
    rhs2 = (vec_prev[:,1] / dt + util).reshape([-1, 1])
    rhs = np.vstack([rhs1, rhs2])

    zero = csr_matrix((n,n))
    mat1 = hstack([mat,  zero])
    mat2 = hstack([zero, mat + (mu_H + mu_L)*sp.eye(n)])
    mat = vstack([mat1, mat2])

    # update
    vec = spla.spsolve(mat, rhs).reshape(n, 2, order='F')

    if i % 10 == 0:
        print(f'iter: {i} ; l2 error: {np.linalg.norm(vec - vec_prev)}')
    if np.linalg.norm(vec - vec_prev) < 1e-8:
        break

    G = vec.copy()
    vec_prev = vec.copy()

G_out = np.zeros((n, 2)) # outputted G won't have lambda_f, since it varies across firms
G_out[:, 0] = G[:, 0] + mu_L/(mu_L + mu_H) *(lambda_H - lambda_L) * G[:, 1]
G_out[:, 1] = G[:, 0] - mu_H/(mu_L + mu_H) *(lambda_H - lambda_L) * G[:, 1]

df_out = pd.DataFrame({
    'eps': eps_pts,
    'G_up' : G_out[:, 0],
    'G_down' : G_out[:, 1]
})
df_out.to_csv('G_func.csv')

