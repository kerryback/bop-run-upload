#!/usr/bin/env python
"""tauchen.py vs tauchen.m (via Octave), and scipy's Gauss-Hermite vs GS21.m's.

The n=20000 production node set cannot go through Octave -- eig() on a full
20000x20000 needs ~7 GB -- so it is checked against a range-restricted
Golub-Welsch in Python instead, which is the same construction GS21.m uses.
Result when written: tauchen <=4.2e-15 at production dims; both rules keep
exactly the same 360 nodes, weights agreeing to 1.05e-15, with ~7e-3 of margin
to the |node| < mbar inclusion boundary.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_hermite
from scipy.stats import norm  # noqa: F401  (kept: handy when extending this file)

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(H))
sys.path.insert(0, REPO)
import config                                            # noqa: E402
from utils_gs21.tauchen import tauchen                    # noqa: E402

# tauchen.m must sit beside tch_run.m for Octave to find it
src = os.path.join(REPO, 'utils_gs21', 'tauchen.m')
if not os.path.exists(os.path.join(H, 'tauchen.m')):
    shutil.copy(src, H)

print('=== tauchen.py vs tauchen.m (Octave) ===')
cases = [(config.GS21_SIGMA_X, config.GS21_RHO_X, 4, 20, 'production x'),
         (config.GS21_SIGMA_Z, config.GS21_RHO_Z, 4, 200, 'production z'),
         (0.05, 0.9, 3, 7, 'small/odd'),
         (0.2, 0.5, 2.5, 4, 'wide/short'),
         (0.012, 0.99, 5, 51, 'persistent')]
ok = True
for sig, rho, mn, num, tag in cases:
    with open(f'{H}/tch_args.m', 'w') as f:
        f.write(f'sigma={float(sig):.17g}; rho={float(rho):.17g}; '
                f'mnstdev={float(mn):.17g}; num={num};\n')
    subprocess.run(['octave', '--no-gui', '--quiet', 'tch_run.m'], cwd=H,
                   check=True, capture_output=True)
    go = np.loadtxt(f'{H}/tch_g.csv', delimiter=',', ndmin=1).ravel()
    po = np.loadtxt(f'{H}/tch_pr.csv', delimiter=',', ndmin=2)
    gp, pp = tauchen(sig, rho, mn, num)
    dg, dp = np.abs(gp - go).max(), np.abs(pp - po).max()
    ok &= max(dg, dp) < 1e-14
    print(f'  {tag:14s} n={num:4d}  |dgrid|={dg:.3e}  |dP|={dp:.3e}')
print('  VERDICT:', 'PASS' if ok else '*** FAIL ***')

print('\n=== Gauss-Hermite: scipy vs Golub-Welsch (GS21.m:410-432) ===')
for n in (200, 1500):
    with open(f'{H}/gh_args.m', 'w') as f:
        f.write(f'n={n};\n')
    subprocess.run(['octave', '--no-gui', '--quiet', 'gh_run.m'], cwd=H,
                   check=True, capture_output=True)
    xo = np.loadtxt(f'{H}/gh_x.csv', delimiter=',').ravel()
    wo = np.loadtxt(f'{H}/gh_w.csv', delimiter=',').ravel()
    o = np.argsort(xo)
    xo, wo = xo[o], wo[o]
    xp, wp = roots_hermite(n)
    print(f'  octave n={n:5d}  |dnodes|={np.abs(xp - xo).max():.3e}  '
          f'|dweights|={np.abs(wp - wo).max():.3e}')

n, sigma_m = 20000, float(config.GS21_SIGMA_M)
cut = 4 * sigma_m / (np.sqrt(2) * sigma_m)
xp, wp = roots_hermite(n)
d = np.zeros(n)
e = np.sqrt(np.arange(1, n) / 2.0)
xg, Vg = eigh_tridiagonal(d, e, select='v', select_range=(-cut - 0.05, cut + 0.05))
wg = Vg[0, :] ** 2 * np.sqrt(np.pi)
kp, kg = np.abs(xp) < cut, np.abs(xg) < cut
print(f'  production n={n}: scipy keeps {kp.sum()}, Golub-Welsch keeps {kg.sum()}')
if kp.sum() == kg.sum():
    a, b = np.sort(xp[kp]), np.sort(xg[kg])
    wa = (wp[kp] / np.sqrt(np.pi))[np.argsort(xp[kp])]
    wb = (wg[kg] / np.sqrt(np.pi))[np.argsort(xg[kg])]
    print(f'    |dnodes| = {np.abs(a - b).max():.3e}   '
          f'|dweights| (renormalised) = '
          f'{np.abs(wa / wa.sum() - wb / wb.sum()).max():.3e}')
    print(f'    inclusion-boundary margin: last kept is '
          f'{cut - np.abs(xp[kp]).max():.3e} inside')
