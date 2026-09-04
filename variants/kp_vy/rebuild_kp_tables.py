"""Rebuild G_func and integral tables for the current KP_PARAM_OVERRIDES (must set g_file and integ_file).
usage: KP_PARAM_OVERRIDES='{"gamma_z":-0.7,"r":0.12,"g_file":"G_func_gz7.csv","integ_file":"integ_gz7.npz"}' python rebuild_kp_tables.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
_t0 = time.time()
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

    if i % 200 == 0:
        print(f'iter: {i} ; l2 error: {np.linalg.norm(vec - vec_prev)}', flush=True)
    if np.linalg.norm(vec - vec_prev) < 1e-6 or i > 60000:
        print('G iteration stopped at', i, np.linalg.norm(vec - vec_prev)); break

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
df_out.to_csv(g_file)


print('G_func done', time.time()-_t0, flush=True)
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import ncx2
from scipy.integrate import quad
from parameters_kp14 import *
from joblib import Parallel, delayed

import scipy.special
from scipy.sparse import csr_matrix, diags, kron

# script to compute all numerical integrals needed
# see KP14.tex overleaf for list
# sdf_compute( ) computes second moment of returns after loading these integrals

n_jobs = 7 # number of jobs in parallelized tasks

# read in G functions estimated in kp14_fd.py 
# recall they don't include lambda_ft, which varies across firms and time
G_in = pd.read_csv(g_file)
eps_grid = G_in.eps.values
G_up = interpolate.interp1d(eps_grid, G_in.G_up.values, fill_value="extrapolate")
G_down = interpolate.interp1d(eps_grid, G_in.G_down.values, fill_value="extrapolate")

A_mod = lambda eps: A(eps, 1)**(1/(1-alpha))

funcs = [A_mod, G_up, G_down, 
        lambda ep: (ep - 1)*A_mod(ep), lambda ep: (ep - 1)*G_up(ep), lambda ep: (ep - 1)*G_down(ep), 
        lambda ep: A_mod(ep)**2, lambda ep: G_up(ep)**2, lambda ep: G_down(ep)**2,
        lambda ep: A_mod(ep)*G_up(ep), lambda ep: A_mod(ep)*G_down(ep)]

# Expected value of E[func(eps_t+dt)|eps_t] for func in funcs
def expected_f_eps(x0):
    c = (sigma_eps**2 * (1 - np.exp(-theta_eps * dt))) / (4 * theta_eps)
    d = 4 * theta_eps / sigma_eps**2
    lam = 4 * theta_eps * np.exp(-theta_eps * dt) * x0 / (sigma_eps**2 * (1 - np.exp(-theta_eps * dt)))

    def integrand(fun, eps):
        integr = fun(eps) * ncx2.pdf(eps / c, d, lam) / c # integrate over CIR transition density
        return integr
    
    # 2026-09-04: was a FIXED interval [0, 10] with a print-only check on the far
    # tail. That is the exact pattern that produced the 2026-06 all-zeros bug in
    # utils_kp14: at small sigma_eps the CIR density is a narrow spike, QUADPACK
    # samples only flat regions, returns 0 with a ~0 error estimate, and never
    # subdivides -- silently voiding every integral. The three checks above tested
    # eps_max = 10, i.e. the one end that cannot fail.
    #
    # Ported from utils_kp14/integ_kp14.py:104-105,120-127 (the technique, not the
    # file -- that version has no notion of the (type, y-node) structure this one
    # exists for). Using the density's own quantiles keeps the interval matched to
    # the spike for any (sigma_eps, theta_eps, dt).
    lo = c * ncx2.ppf(1e-13, d, lam)
    hi = c * ncx2.ppf(1 - 1e-13, d, lam)

    # epsabs carries the (eps-1)*f(eps) integrals, whose true value passes through
    # 0 near eps=1 where no relative tolerance is attainable; epsrel carries the rest.
    result = [quad(lambda ep: integrand(fun, ep), lo, hi, epsabs = 1e-8, epsrel = 1e-10, limit = 500)[0] for fun in funcs]

    # The density must integrate to 1 at every grid point. RAISE rather than print:
    # a silent zero is what caused the original damage, and these values are
    # unusable if the mass is lost.
    mass = quad(lambda ep: ncx2.pdf(ep / c, d, lam) / c, lo, hi,
                epsabs = 1e-12, epsrel = 1e-10, limit = 500)[0]
    if not abs(mass - 1) < 1e-8:
        raise RuntimeError(
            f'CIR transition density integrates to {mass!r}, not 1, at eps={x0!r} '
            f'(interval [{lo!r}, {hi!r}]). Integrals would be silently wrong.')

    return result

integ_lst = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(expected_f_eps)(ep) for ep in eps_grid
    )


A_mod_lst = np.array([integ_lst[i][0] for i in range(len(eps_grid))])
G_up_lst = np.array([integ_lst[i][1] for i in range(len(eps_grid))])
G_down_lst = np.array([integ_lst[i][2] for i in range(len(eps_grid))])
ep_A_mod_lst = np.array([integ_lst[i][3] for i in range(len(eps_grid))])
ep_G_up_lst = np.array([integ_lst[i][4] for i in range(len(eps_grid))])
ep_G_down_lst = np.array([integ_lst[i][5] for i in range(len(eps_grid))])
A_mod2_lst = np.array([integ_lst[i][6] for i in range(len(eps_grid))])
G_up2_lst = np.array([integ_lst[i][7] for i in range(len(eps_grid))])
G_down2_lst = np.array([integ_lst[i][8] for i in range(len(eps_grid))])
A_mod_G_up_lst = np.array([integ_lst[i][9] for i in range(len(eps_grid))])
A_mod_G_down_lst = np.array([integ_lst[i][10] for i in range(len(eps_grid))])

# Save all arrays to one .npz file
np.savez(integ_file,
         A_mod_lst=A_mod_lst,
         G_up_lst=G_up_lst,
         G_down_lst=G_down_lst,
         ep_A_mod_lst=ep_A_mod_lst,
         ep_G_up_lst=ep_G_up_lst,
         ep_G_down_lst=ep_G_down_lst,
         A_mod2_lst=A_mod2_lst,
         G_up2_lst=G_up2_lst,
         G_down2_lst=G_down2_lst,
         A_mod_G_up_lst=A_mod_G_up_lst,
         A_mod_G_down_lst=A_mod_G_down_lst)
print('integrals done', time.time()-_t0)
