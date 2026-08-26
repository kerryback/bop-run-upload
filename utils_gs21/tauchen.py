"""
Tauchen (1986) discretization of an AR(1) — 1:1 port of utils_gs21/tauchen.m.

Kept a faithful transcription rather than a tidier reimplementation, because the
committed GS21 solution files were produced by the MATLAB version and the grids
are the only thing that identifies which parameters built them. Any deviation
here would destroy that. Verified against the committed grids to 5.4e-16 (see
GS21_solfiles/zgrid.csv, xgrid.csv).

Conventions that matter, all inherited from the .m file:
  * grid bounds are +/- multiple * UNCONDITIONAL sd, sqrt(sigma^2/(1-rho^2))
  * the normal CDF inside the transition uses the CONDITIONAL sd, sigma
  * the two endpoint columns absorb the tails
  * rows are renormalized to sum to 1, then negatives clipped to 0
  * `sigma` is the innovation sd, NOT the stationary sd

This is plain Tauchen, not Tauchen-Hussey and not Rouwenhorst. At the
persistences GS21 uses (rho_x = 0.95^(1/3) ~ 0.983, rho_z = 0.9^(1/3) ~ 0.965)
Rouwenhorst would be the better approximation, but switching would change the
economy the committed solfiles encode. Treat any change here as a modeling
decision, not a numerical improvement.
"""

import numpy as np
from scipy.special import erfc


def _normcdfbase(x, mu, sigma):
    """Normal CDF, as tauchen.m:39 defines it (hand-rolled to avoid a toolbox)."""
    return 0.5 * erfc(-((x - mu) / sigma) / np.sqrt(2))


def tauchen(sigma, rho, multiple, znum):
    """Return (z0, pr_mat_z) for an AR(1) with innovation sd `sigma`.

    Args mirror tauchen.m's positional order: (sigma, rho, multiple, znum).

    z0        -- (znum,) grid, symmetric about 0
    pr_mat_z  -- (znum, znum) transition matrix, pr_mat_z[i, j] = P(z' = j | z = i)
    """
    sdz = np.sqrt(sigma**2 / (1 - rho**2))
    z0 = np.linspace(-multiple * sdz, multiple * sdz, znum)
    gridinc = z0[1] - z0[0]

    pr = np.zeros((znum, znum))
    for i in range(znum):
        meanval = rho * z0[i]
        interior = slice(1, znum - 1)
        pr[i, interior] = (_normcdfbase(z0[interior] + gridinc / 2, meanval, sigma)
                           - _normcdfbase(z0[interior] - gridinc / 2, meanval, sigma))
        pr[i, 0] = _normcdfbase(z0[0] + gridinc / 2, meanval, sigma)
        pr[i, -1] = 1 - _normcdfbase(z0[-1] - gridinc / 2, meanval, sigma)
        pr[i, :] /= pr[i, :].sum()

    pr[pr < 0] = 0
    return z0, pr


if __name__ == '__main__':
    import os
    import sys
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GS21_solfiles')
    for fname, sigma, rho, num in [('xgrid.csv', config.GS21_SIGMA_X, config.GS21_RHO_X, 20),
                                   ('zgrid.csv', config.GS21_SIGMA_Z, config.GS21_RHO_Z, 200)]:
        committed = np.sort(pd.read_csv(os.path.join(D, fname), header=None).values.ravel())
        grid, pr = tauchen(sigma, rho, 4, num)
        print(f'{fname:12s} max|config.py grid - committed| = {np.abs(grid - committed).max():.3e}'
              f'   rows sum to 1 to {np.abs(pr.sum(1) - 1).max():.2e}')
