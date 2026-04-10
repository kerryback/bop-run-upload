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
G_in = pd.read_csv('G_func.csv')
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
    
    eps_max = 10
    result = [quad(lambda ep: integrand(fun, ep), 0, eps_max, epsabs = 1e-10, epsrel = 1e-10, limit = 500)[0] for fun in funcs]

    if integrand(A_mod, eps_max) > 1e-12:
        print(f'error: eps = {x0}')
    if integrand(G_up, eps_max) > 1e-12:
        print(f'error: eps = {x0}')
    if integrand(G_down, eps_max) > 1e-12:
        print(f'error: eps = {x0}')

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
np.savez("integ_results.npz",
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