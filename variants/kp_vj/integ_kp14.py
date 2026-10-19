import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import ncx2
from scipy.integrate import quad
from parameters_kp14 import *
import os
_iy = int(os.environ.get("KP_GAMY_YIDX", "0"))
_ft = int(os.environ.get("KP_VY_TYPE", "0"))
_gin = os.environ.get("KP_GAMY_GIN", f"G_vy{_ft}.csv")
_iout = os.environ.get("KP_GAMY_IOUT", f"integ_vy{_ft}_{_iy}.npz")
A = lambda ep, u: (A0_ty[_ft, _iy] + (ep - 1) * A1_ty[_ft, _iy] + (u - 1) * A2_ty[_ft, _iy]
                   + (ep - 1) * (u - 1) * A3_ty[_ft, _iy])
from joblib import Parallel, delayed

import scipy.special
from scipy.sparse import csr_matrix, diags, kron

# script to compute all numerical integrals needed
# see KP14.tex overleaf for list
# sdf_compute( ) computes second moment of returns after loading these integrals

n_jobs = 7 # number of jobs in parallelized tasks

# read in G functions estimated in kp14_fd.py 
# recall they don't include lambda_ft, which varies across firms and time
G_in = pd.read_csv(_gin)
eps_grid = G_in.eps.values
G_up = interpolate.interp1d(eps_grid, G_in[f"G_up_y{_iy}"].values, fill_value="extrapolate")
G_down = interpolate.interp1d(eps_grid, G_in[f"G_down_y{_iy}"].values, fill_value="extrapolate")

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
    #
    # epsrel was 1e-10, which QUADPACK could not reach: it subdivided to limit=500
    # and returned 985 roundoff warnings per job saying its own error estimate was
    # unreliable. Measured 2026-09-06 on one (type, y-node) job:
    #     epsrel   wall   warnings   worst rel. diff vs 1e-10
    #     1e-10    110s        985   --
    #     1e-8      89s        863   8.3e-08
    #     1e-6      12s          0   3.8e-07
    # 1e-6 is 9.2x faster and moves the tables by 4e-7, which is four orders of
    # magnitude below the panel's own sampling noise (~2e-3 at N=500, T=500). It
    # also stops QUADPACK straining, so the returned error estimates mean something
    # again. quad stops at max(epsabs, epsrel*|I|), so the zero-crossing integrands
    # are still bounded by epsabs = 1e-8 exactly as before.
    result = [quad(lambda ep: integrand(fun, ep), lo, hi, epsabs = 1e-8, epsrel = 1e-6, limit = 500)[0] for fun in funcs]

    # The density must integrate to 1 at every grid point. RAISE rather than print:
    # a silent zero is what caused the original damage, and these values are
    # unusable if the mass is lost.
    mass = quad(lambda ep: ncx2.pdf(ep / c, d, lam) / c, lo, hi,
                epsabs = 1e-12, epsrel = 1e-6, limit = 500)[0]
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
np.savez(_iout,
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