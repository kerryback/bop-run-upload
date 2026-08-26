#!/usr/bin/env python
"""
Build GS21's four grid files from config.py -- no MATLAB required.

All 18 files in GS21_solfiles/ are now produced by utils_gs21/gs21_solve.py, so
none of them needs MATLAB. This module survives as the CHEAP check: the four grid
files (bgrid, igrid, xgrid, zgrid) are closed-form functions of the parameters,
so they can be re-derived and compared in milliseconds, whereas checking a policy
file means re-running the ~2.5 min solve.

That cheapness is why the grids did the historical work: being pure functions of
(sigma, rho, mnstdev, num) via tauchen, they PIN the parameters that built them,
which is how the sigma_x / sigma_z exponents were identified as 2/3 rather than
3/2 (and the committed files revealed as a hybrid -- GS21.m's exponents plus
config.py's 0.1 factor on sigma_z). The remaining three parameters, r / xi /
gamma_x, are invisible to the grids and needed the full operator; see the note
above GS21_R in config.py.

Default action is --check: compare against the committed files and report,
writing nothing. Regenerating is deliberate, because MATLAB's writematrix and
numpy's savetxt format floats differently -- the values would agree to 5e-16
while every byte changed.

Usage (from the repo root):
    python utils_gs21/grids.py              # check against committed files
    python utils_gs21/grids.py --write DIR  # write the four grids into DIR
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import config                                    # noqa: E402
from utils_gs21.tauchen import tauchen           # noqa: E402

SOLFILES_DIR = os.path.join(_HERE, 'GS21_solfiles')


def build():
    """The four grids, as {filename: 1-D array}, exactly as GS21.m constructs them."""
    # GS21.m:65  bgrid = linspace(bmin, bmax, bnum)'
    bgrid = np.linspace(config.GS21_BMIN, config.GS21_BMAX, config.GS21_BNUM)

    # GS21.m:67-68  igrid = linspace(imin, imax, inum+1)'; igrid = igrid(1:end-1)
    # Left-closed: inum points that stop one step short of imax. Do not "fix"
    # the missing top point -- the committed igrid tops out at 1950, not 2000.
    igrid = np.linspace(config.GS21_IMIN, config.GS21_IMAX, config.GS21_INUM + 1)[:-1]

    # GS21.m:73-74  [xgrid, pr_mat_x] = tauchen(sigma_x, rho_x, mnstdev, xnum)
    xgrid, _ = tauchen(config.GS21_SIGMA_X, config.GS21_RHO_X,
                       config.GS21_MNSTDEV, config.GS21_XNUM)
    zgrid, _ = tauchen(config.GS21_SIGMA_Z, config.GS21_RHO_Z,
                       config.GS21_MNSTDEV, config.GS21_ZNUM)

    return {'bgrid.csv': bgrid, 'igrid.csv': igrid,
            'xgrid.csv': xgrid, 'zgrid.csv': zgrid}


def check(solfiles_dir=None):
    """Compare the config-driven grids against the committed files.

    Returns {filename: max_abs_diff}. This is the parameter oracle: the grids are
    pure functions of (sigma, rho, mnstdev, num) and the b/i bounds, so a
    mismatch here identifies a parameter disagreement rather than a solve bug.
    """
    import pandas as pd
    solfiles_dir = solfiles_dir or SOLFILES_DIR
    out = {}
    for name, grid in build().items():
        path = os.path.join(solfiles_dir, name)
        if not os.path.exists(path):
            out[name] = float('nan')
            continue
        committed = np.sort(pd.read_csv(path, header=None).values.ravel())
        if len(committed) != len(grid):
            out[name] = float('inf')      # a length mismatch is not a tolerance question
            continue
        out[name] = float(np.abs(np.sort(grid) - committed).max())
    return out


def write(out_dir):
    """Write the four grids as single-column CSVs, matching writematrix's layout."""
    os.makedirs(out_dir, exist_ok=True)
    for name, grid in build().items():
        np.savetxt(os.path.join(out_dir, name), grid, delimiter=',')
    return sorted(build())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', metavar='DIR', default=None,
                    help='write the grids into DIR (default: check only, write nothing)')
    args = ap.parse_args()

    if args.write:
        for name in write(args.write):
            print(f'  wrote {os.path.join(args.write, name)}')
        return

    print('config.py-driven grids vs committed GS21_solfiles/:')
    worst = 0.0
    for name, d in check().items():
        print(f'  {name:12s} max|python - committed| = {d:.3e}')
        worst = max(worst, d if np.isfinite(d) else float("inf"))
    print(f'\n  worst = {worst:.3e}  ->  '
          f'{"grids agree; parameters that build them are reconciled" if worst < 1e-12 else "PARAMETER DISAGREEMENT (see utils_gs21/solfile_spec.py)"}')
    print('  All 18 GS21 solfiles are produced by utils_gs21/gs21_solve.py; no')
    print('  MATLAB is involved. This module remains the cheap check on the four')
    print('  grids, which are pure functions of config.py via tauchen.py.')


if __name__ == '__main__':
    main()
