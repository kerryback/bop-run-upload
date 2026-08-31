#!/usr/bin/env python
"""
Producer for utils_bgn/BGN_solfiles/Jstar.csv -- the value of BGN's growth options
on a grid of interest rates.

This code used to sit inside a triple-quoted string literal at the bottom of
vasicek.py, i.e. it was not executable. Regenerating meant editing the module by
hand, which is the same "no runnable producer" hazard that let KP14 ship a
corrupted integ_results.npz for two months. It is a real script now.

Two changes from the string-literal version, both making it reproducible:

  * The grid endpoints came from `r.quantile(...)` over an UNSEEDED 1e8-draw
    simulation of the AR(1). But r is exactly N(rbar, sigma_r^2/(1-kappa^2)), so
    the quantiles are analytic. The committed file's endpoints agree with the
    analytic values to 1.5e-6 and 1.8e-5 (2.4e-4 and 2.8e-3 in sd units) --
    i.e. they WERE these quantiles, plus Monte Carlo noise. Using the closed
    form makes regeneration deterministic; it also means a regenerated grid
    differs from the committed one in the r values by ~1e-5, which is expected.
  * n_jobs is a flag rather than a hardcoded 4.

KNOWN TRUNCATION BIAS, measured 2026-08-31, unchanged from the shipped file:
vasicek.py truncates the maturity sum at k <= 400 (weighted pi^k) and the
option-maturity sum at s <= 950 (UNWEIGHTED). Against extended sums:

    k <= 400 captures 99.8702% of k <= 1200      (missing 0.130%)
    s <= 950 captures 99.9280% of s <= 2000 at r = -0.0162  (missing 0.072%)
    s <= 950 captures 99.8761% of s <= 2000 at r = +0.0287  (missing 0.124%)

So Jstar is biased DOWN by roughly 0.2%, systematically, at every r. Jstar enters
firm value scaled by Chat = exp(-3.7) = 0.0247, so the effect on returns is
smaller again -- but it is a real bias, it was undocumented, and it is not
uniform in r (0.072% vs 0.124% across the grid), so it tilts the r-dependence
slightly. Raising the truncations is cheap in code and expensive in runtime;
left as-is deliberately so the file keeps matching the published numbers.

Runtime: ~19 s per grid point single-threaded. The adaptive refinement below
lands on 201 points, so budget ~65 min at n_jobs=1, ~32 min at n_jobs=2.

Usage (from the repo root):
    python utils_bgn/make_jstar.py                 # write into BGN_solfiles/
    python utils_bgn/make_jstar.py --out DIR       # dry run elsewhere
    python utils_bgn/make_jstar.py --n-jobs 2      # respect a core budget
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import RBAR as rbar, KAPPA as kappa, SIGMA_R as sigma_r   # noqa: E402

# Import the model functions WITHOUT going through utils_bgn/__init__.py, which
# imports the consumers, which verify provenance at import time -- so importing
# the package while the stamp is stale would stop the tool that repairs it.
sys.path.insert(0, _HERE)
import importlib.util as _ilu                                        # noqa: E402
_spec = _ilu.spec_from_file_location('_bgn_vasicek', os.path.join(_HERE, 'vasicek.py'))
_vas = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_vas)
Jstar = _vas.Jstar

SOLFILES_DIR = os.path.join(_HERE, 'BGN_solfiles')
QUANTILE = 0.99995          # as in the original: covers 99.99% of the r distribution
TOL = 1e-4                  # relative interpolation error at the midpoints


def grid_endpoints():
    """Analytic quantiles of the stationary distribution of r."""
    sd = np.sqrt(sigma_r**2 / (1.0 - kappa**2))
    return rbar + norm.ppf(1 - QUANTILE) * sd, rbar + norm.ppf(QUANTILE) * sd


def build(out_dir, n_jobs=2, tol=TOL, n0=51, max_rounds=8, verbose=True):
    """Adaptive bisection until midpoint interpolation error is below tol."""
    rmin, rmax = grid_endpoints()
    if verbose:
        print(f'grid endpoints (analytic {QUANTILE:.5f} quantiles): '
              f'[{rmin:.12f}, {rmax:.12f}]', flush=True)

    def evaluate(rs):
        return np.array(Parallel(n_jobs=n_jobs)(delayed(Jstar)(r) for r in rs))

    t0 = time.time()
    grid = np.linspace(rmin, rmax, n0)
    J = evaluate(grid)
    if verbose:
        print(f'round 0: {len(grid)} points ({time.time() - t0:.0f}s)', flush=True)

    for rnd in range(1, max_rounds + 1):
        mid = 0.5 * (grid[:-1] + grid[1:])
        Jmid = evaluate(mid)
        Jhat = np.interp(mid, grid, J)
        err = np.max(np.abs(Jmid - Jhat) / np.abs(Jmid))
        order = np.argsort(np.concatenate([grid, mid]))
        grid = np.concatenate([grid, mid])[order]
        J = np.concatenate([J, Jmid])[order]
        if verbose:
            print(f'round {rnd}: {len(grid)} points, max rel interp err = {err:.3e} '
                  f'({time.time() - t0:.0f}s)', flush=True)
        if err < tol:
            break
    else:
        raise RuntimeError(f'did not reach tol={tol:g} in {max_rounds} rounds '
                           f'(last err {err:.3e}); nothing written')

    if not np.all(np.diff(grid) > 0):
        raise RuntimeError('grid is not strictly increasing; nothing written')
    if not np.all(np.isfinite(J)):
        raise RuntimeError('Jstar produced non-finite values; nothing written')
    # J is monotone decreasing in r for this model; a violation means trouble.
    if not np.all(np.diff(J) < 0):
        n = int((np.diff(J) >= 0).sum())
        raise RuntimeError(f'Jstar is not monotone decreasing in r ({n} violations); '
                           f'nothing written')

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'Jstar.csv')
    pd.DataFrame({'r': grid, 'J': J}).to_csv(path, index=False)
    if verbose:
        print(f'\nwrote {path}: {len(grid)} points, '
              f'J in [{J.min():.4f}, {J.max():.4f}]', flush=True)
    return grid, J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default=SOLFILES_DIR)
    ap.add_argument('--out', dest='out_flag', default=None)
    ap.add_argument('--n-jobs', type=int, default=2)
    ap.add_argument('--tol', type=float, default=TOL)
    args = ap.parse_args()
    build(os.path.abspath(args.out_flag or args.out), n_jobs=args.n_jobs, tol=args.tol)


if __name__ == '__main__':
    main()
