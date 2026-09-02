import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import ncx2
from scipy.integrate import quad
import os
import sys
from joblib import Parallel, delayed

import scipy.special
from scipy.sparse import csr_matrix, diags, kron

# parameters_kp14.py was never committed to this repo; config.py is the single
# source of truth for KP14 parameters. Import the same names sdf_compute_kp14.py
# does (see sdf_compute_kp14.py:5-22) so this producer and its consumer cannot
# drift apart.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    KP14_DT as dt,
    KP14_THETA_EPS as theta_eps, KP14_SIGMA_EPS as sigma_eps,
    KP14_ALPHA as alpha,
    KP14_A_0 as A_0, KP14_A_1 as A_1, KP14_A_2 as A_2, KP14_A_3 as A_3,
)

A = lambda ep, u: (A_0 + (ep - 1) * A_1 + (u - 1) * A_2 + (ep - 1) * (u - 1) * A_3)

# script to compute all numerical integrals needed
# see KP14.tex overleaf for list
# sdf_compute( ) computes second moment of returns after loading these integrals

n_jobs = 7 # number of jobs in parallelized tasks

# Solution files live in KP14_solfiles/, not the CWD. Both this script and
# kp14_fd.py used to read and write bare filenames, which made regeneration a
# manual cd-into-the-right-directory ritual and let a stale G_func.csv be paired
# with fresh integrals without anything noticing.
_SOLFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'KP14_solfiles')
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else _SOLFILES_DIR   # override for dry runs

# read in G functions estimated in kp14_fd.py
# recall they don't include lambda_ft, which varies across firms and time
# Read G from OUT_DIR, not _SOLFILES_DIR: in a dry run (--out DIR) kp14_fd.py has
# just written the fresh G there, and reading the committed copy would silently
# pair these integrals with a stale G (the failure mode regen_solfiles.py exists
# to prevent). In a normal run the two directories coincide.
G_in = pd.read_csv(os.path.join(OUT_DIR, 'G_func.csv'))
eps_grid = G_in.eps.values
# Interpolation order, set 2026-08-31: cubic, NOT the interp1d default (linear).
#
# These are tabulated on a 1000-point uniform eps grid (h = 4.995e-3) and
# evaluated at FIRM-SPECIFIC eps, so every firm picks up an independent error.
# Linear was costing real accuracy on the three (eps-1)*f(eps) integrals, which
# cross zero at eps ~ 1 -- exactly where the CIR distribution for eps
# concentrates -- and which are the ones carrying the risk premium:
#
#     ep_A_mod_lst    RMS relative linear-interp error 7.8e-03 (max 3.7e-01)
#     ep_G_up_lst                                      2.5e-03
#     ep_G_down_lst                                    2.0e-03
#     the other eight                           1.6e-06 to 8.1e-05
#
# The driver is the curvature-to-level ratio |f''|max/|f|: 133, 35 and 18 for
# those three against 0.8-64 for the rest. Linear error is h^2*f''/8, so
# dividing by a level that passes through zero is what inflates it.
#
# Consequence: mu carried a diffuse per-firm error of ~2-4e-4 (1.5-2.8% of its
# mean) with no factor structure. Sigma^-1 at N=1000 levered that into
# sqrt(mu' Sigma^-1 mu) exceeding the analytic Hansen-Jagannathan bound in 770
# of 3,960 panel-months. See the note at max_sr in sdf_compute_kp14.py.
#
# Cubic, not pchip: all eleven integrals are strictly MONOTONE with zero turning
# points over the eps band, so there is no shape hazard for pchip to guard
# against, and measured cubic overshoot outside the data range is exactly zero.
# Held-out tests put cubic 2-6x ahead of pchip at the shipped spacing and
# 40-130x ahead as spacing coarsens; akima is worse than pchip throughout.
G_up = interpolate.interp1d(eps_grid, G_in.G_up.values, kind='cubic', fill_value="extrapolate")
G_down = interpolate.interp1d(eps_grid, G_in.G_down.values, kind='cubic', fill_value="extrapolate")

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
    
    # Integrate over the support the transition density actually occupies.
    # This used to be a fixed [0, 10]. The density is a spike of sd ~0.006
    # (sigma_eps=0.02, dt=1/12), so QUADPACK's initial 21-point Gauss-Kronrod
    # rule on [0, 10] had no node within ~17 sd of it: the estimate AND its
    # error estimate were both ~0, so quad returned 0 and never subdivided. That
    # zeroed these integrals over the eps band holding ~98% of firm-months and
    # made every KP14 expected return and SDF quantity void for two months.
    # Using the density's own quantiles keeps the interval matched to the spike
    # for any (sigma_eps, theta_eps, dt).
    lo = c * ncx2.ppf(1e-13, d, lam)
    hi = c * ncx2.ppf(1 - 1e-13, d, lam)

    # epsabs is what carries funcs[3:6] -- the (eps-1)*f(eps) integrals, whose true
    # value passes through 0 near eps=1, where no relative tolerance is attainable.
    # epsrel carries the rest, which peak around 1e8. Tightening either to 1e-12
    # only buys QUADPACK roundoff warnings and ~30% more time for the same answer
    # (verified: agrees to 1.7e-9). Accuracy is checked against a delta-method
    # reference in diag_integ_kp14.py, not assumed from the tolerance.
    result = [quad(lambda ep: integrand(fun, ep), lo, hi, epsabs = 1e-8, epsrel = 1e-10, limit = 500)[0] for fun in funcs]

    # The density must integrate to 1 at every grid point. This is the check
    # whose absence let the [0, 10] bug ship: the three checks that used to be
    # here tested the integrand at eps_max = 10, i.e. the far tail -- the one end
    # that cannot fail. Raise rather than print: a silent zero is what caused the
    # original damage, and these values are unusable if the mass is lost.
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
np.savez(os.path.join(OUT_DIR, "integ_results.npz"),
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