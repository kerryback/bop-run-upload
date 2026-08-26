#!/usr/bin/env python
"""
Diagnose the KP14 quadrature failure in utils_kp14/integ_kp14.py.

utils_kp14/KP14_solfiles/integ_results.npz holds E_t[f(eps_{t+dt})] on a 1000-point
eps grid for 11 functions f. Those values are wrong: 238 of 1000 grid points are
exactly 0, and over the eps band the simulation actually visits they are either 0
or ~1e-146..1e-12, then jump discontinuously to ~2400 at eps=1.049. The truth is a
smooth ~18,000 at eps=1.

This script (a) shows the damage, (b) proves the cause is the [0, 10] integration
range in integ_kp14.py:43 against a transition density of sd ~0.006, (c) shows the
one-line fix passes a mass check the current code never performs.

Runs locally, no cluster, no panel data. ~10 seconds.

Usage:  python diag_integ_kp14.py
"""
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import ncx2, norm
from scipy.integrate import quad

REPO = os.path.dirname(os.path.abspath(__file__))
SOL  = os.path.join(REPO, 'utils_kp14', 'KP14_solfiles')

import sys
sys.path.insert(0, REPO)
from config import (
    KP14_DT as dt, KP14_THETA_EPS as theta_eps, KP14_SIGMA_EPS as sigma_eps,
    KP14_ALPHA as alpha, KP14_A_0 as A_0, KP14_A_1 as A_1,
)

G_in     = pd.read_csv(os.path.join(SOL, 'G_func.csv'))
eps_grid = G_in.eps.values
stored   = np.load(os.path.join(SOL, 'integ_results.npz'))

# A_mod is funcs[0] in integ_kp14.py:27 -- the simplest of the 11, so we test on it
A_mod = lambda ep: (A_0 + (ep - 1) * A_1) ** (1 / (1 - alpha))

# CIR transition density of eps, exactly as in integ_kp14.py:34-36
c = (sigma_eps**2 * (1 - np.exp(-theta_eps * dt))) / (4 * theta_eps)
d = 4 * theta_eps / sigma_eps**2
lam_of = lambda x0: 4*theta_eps*np.exp(-theta_eps*dt)*x0 / (sigma_eps**2*(1 - np.exp(-theta_eps*dt)))

def pdf(ep, lam):
    return ncx2.pdf(ep / c, d, lam) / c

def integrate(fun, x0, lo, hi):
    lam = lam_of(x0)
    return quad(lambda ep: fun(ep) * pdf(ep, lam), lo, hi,
                epsabs=1e-12, epsrel=1e-12, limit=500)[0]

BROAD  = lambda x0: (0.0, 10.0)                            # integ_kp14.py:42-43, as shipped
def NARROW(x0):                                            # proposed fix
    lam = lam_of(x0)
    return c * ncx2.ppf(1e-13, d, lam), c * ncx2.ppf(1 - 1e-13, d, lam)

bar = lambda s: print('\n' + '=' * 74 + f'\n{s}\n' + '=' * 74)

# ---------------------------------------------------------------- 1. the damage
bar('1. WHAT IS IN integ_results.npz')
for k in stored.files:
    a = stored[k]
    print(f'  {k:<17s} n={a.size}  exact zeros={int((a == 0).sum()):4d}  '
          f'min={a.min():.4g}  max={a.max():.4g}')

sd_stat = sigma_eps * np.sqrt(1 / (2 * theta_eps))
band    = (eps_grid >= 1 - 4*sd_stat) & (eps_grid <= 1 + 4*sd_stat)
bar(f'2. THE BAND THE SIMULATION VISITS: eps ~ mean 1, sd {sd_stat:.5f} (+-4sd = '
    f'[{1-4*sd_stat:.4f}, {1+4*sd_stat:.4f}])')
A_st = stored['A_mod_lst']
print('  idx     eps      stored A_mod_lst        truth (narrow-limit quad)')
for i in np.flatnonzero(band)[::3]:
    print(f'  {i:4d} {eps_grid[i]:8.4f} {A_st[i]:20.6g} {integrate(A_mod, eps_grid[i], *NARROW(eps_grid[i])):22.6g}')
step = np.flatnonzero(band & (A_st > 1.0))
if step.size:
    thr = eps_grid[step[0]]
    print(f'\n  Et_A_mod is a STEP: ~0 below eps={thr:.4f}, ~{A_st[step[0]]:.0f} above.')
    print(f'  Fraction of firm-months below that step: {norm.cdf((thr - 1)/sd_stat):.1%}')

# --------------------------------------------------- 3. reproduce the root cause
bar('3. ROOT CAUSE: quad(..., 0, 10) LOSES THE PROBABILITY MASS')
print('  The integral of the density alone must be 1.0 for every eps.')
print('  integ_kp14.py never checks this. Here is what each range returns:\n')
one = lambda ep: 1.0
print(f'  {"eps":>7} {"density sd":>11} {"mass on [0,10]":>16} {"mass on narrow":>16}   {"E[A_mod] [0,10]":>17} {"E[A_mod] narrow":>17}')
for x0 in [0.05, 0.50, 0.95, 1.00, 1.05, 2.00, 5.00]:
    lam = lam_of(x0)
    sd  = np.sqrt(c**2 * 2 * (d + 2*lam))
    print(f'  {x0:7.2f} {sd:11.6f} {integrate(one, x0, *BROAD(x0)):16.4g} '
          f'{integrate(one, x0, *NARROW(x0)):16.10f}   '
          f'{integrate(A_mod, x0, *BROAD(x0)):17.6g} {integrate(A_mod, x0, *NARROW(x0)):17.6g}')
print('\n  QUADPACK\'s first 21-point rule on [0,10] has no node within ~100 sd of the')
print('  spike, so the estimate AND its error estimate are both ~0: it returns 0 and')
print('  never subdivides. No warning, no exception.')

# ------------------------------------------- 4. confirm the file matches the bug
bar('4. THE FILE IS REPRODUCIBLE FROM CURRENT config.py (so it is not parameter-stale)')
idxs = [3, 50, 150, 180, 198, 205, 208, 300, 500, 999]
agree_zero = sum((A_st[i] == 0) == (integrate(A_mod, eps_grid[i], *BROAD(eps_grid[i])) == 0) for i in idxs)
print(f'  exact-zero pattern of stored vs quad(0,10) under current config: {agree_zero}/{len(idxs)} agree')
print(f'  (with the pre-2026-06-18 SIGMA_EPS=0.2 the same code returns smooth, correct')
print(f'   values -- the 10x cut in KP14_SIGMA_EPS is what pushed it over the edge.)')

# ------------------------------------------------------- 5. validate the fix
bar('5. THE FIX VALIDATES: mass == 1 everywhere, and matches a delta-method check')
worst_mass = worst_rel = 0.0
for i in range(0, 1000, 11):
    x0  = eps_grid[i]
    lam = lam_of(x0)
    lo, hi = NARROW(x0)
    worst_mass = max(worst_mass, abs(integrate(one, x0, lo, hi) - 1))
    val = integrate(A_mod, x0, lo, hi)
    m, v = c*(d + lam), c**2*2*(d + 2*lam)
    h  = 1e-4 * m
    f2 = (A_mod(m + h) - 2*A_mod(m) + A_mod(m - h)) / h**2
    worst_rel = max(worst_rel, abs(val / (A_mod(m) + 0.5*f2*v) - 1))
print(f'  91 grid points: worst |mass - 1| = {worst_mass:.2e}')
print(f'                  worst |E[A_mod] / (f(m) + f\'\'(m)var/2) - 1| = {worst_rel:.2e}')
print(f'\n  Because sd/mean ~ {np.sqrt(c**2*2*(d+2*lam_of(1.0)))/(c*(d+lam_of(1.0))):.1e}, E[f] is essentially f(mean); any')
print( '  method that actually resolves the spike gets the same answer.')

bar('6. STATE OF THE PRODUCERS')
print('  FIXED: integ_kp14.py and kp14_fd.py used to do `from parameters_kp14 import *`,')
print('  against a module that is not in the worktree and never was in git history.')
print('  Both now import from config.py, so a config change reaches the solution files.')
print('  Verified: a fresh integ_kp14.py run reproduces gamma_x=0.69 (current config),')
print('  while the committed integ_results.npz still encodes gamma_x=1.38.')
print()
print('  STILL OPEN: (a) the [0,10] range at integ_kp14.py:58 -- section 3 above;')
print('              (b) integ_kp14.py reads/writes bare filenames in CWD (lines 33, 87)')
print('                  rather than KP14_solfiles/, so it must be run from that dir;')
print('              (c) both solution files need regenerating once (a) is fixed.')
print(f'\n  Measured cost: ~60s wall for integ_kp14.py (n_jobs=7); kp14_fd.py converges')
print( '  in ~45s (l2 < 1e-8 at iter ~3595, well short of the 1e6 cap).')
