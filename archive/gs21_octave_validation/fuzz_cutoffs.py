#!/usr/bin/env python
"""Fuzz update_cutoffs against GS21.m:395-408 via Octave.

The single-pass harness cannot reach cond0 (an all-nonpositive row) or exact
ties, because P_up there is dominated by E[max(P+node,0)] >= 0. This drives the
function directly with draws designed to hit every branch. Result when written:
bit-exact over 60 cases with cond0 202x, cond2 134x, interior 369x, the
non-finite weight guard 486x, and 1255 exact ties.
"""
import os
import subprocess
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(H))
sys.path.insert(0, REPO)
# bypass utils_gs21/__init__.py -- see py_ref.py for why
import importlib.util as _ilu                             # noqa: E402
_spec = _ilu.spec_from_file_location(
    '_gs21_solve_harness', os.path.join(REPO, 'utils_gs21', 'gs21_solve.py'))
_gs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gs)
update_cutoffs = _gs.update_cutoffs

rng = np.random.default_rng(4242)
worst_z = worst_w = 0.0
branch = {'cond0': 0, 'cond2': 0, 'interior': 0, 'nonfinite': 0, 'ties': 0}
NCASE = int(os.environ.get('T_NCASE', 60))
for case in range(NCASE):
    bnum = int(rng.integers(2, 6))
    xnum = int(rng.integers(2, 6))
    znum = int(rng.integers(2, 8))
    n = bnum * xnum * znum
    style = case % 6
    if style == 0:
        P = rng.normal(0, 1, n)
    elif style == 1:
        P = rng.normal(-3, 1, n)                      # mostly cond0
    elif style == 2:
        P = rng.normal(3, 1, n)                       # mostly cond2
    elif style == 3:
        P = rng.integers(-1, 2, n).astype(float)      # ties and exact zeros
    elif style == 4:
        P = np.where(rng.random(n) < 0.5, 0.0, rng.normal(0, 1, n))
    else:
        P = rng.choice([-1.0, 0.0, 1.0, 1.0], n)      # heavy exact ties
    zgrid = np.sort(rng.normal(0, 1, znum))

    pos = P.reshape(xnum * bnum, znum, order='F') > 0
    c0 = (pos.sum(1) == 0).sum()
    c2 = ((~pos).sum(1) == 0).sum()
    branch['cond0'] += int(c0)
    branch['cond2'] += int(c2)
    branch['interior'] += int(bnum * xnum - c0 - c2)
    P_re = P.reshape(bnum * xnum, znum, order='F')
    zi = np.argmax(pos, 1)
    r = np.arange(bnum * xnum)
    branch['nonfinite'] += int((P_re[r, zi] == P_re[r, np.maximum(0, zi - 1)]).sum())
    branch['ties'] += int(len(P) - len(np.unique(P)))

    np.savetxt(f'{H}/fz_P.csv', P[:, None], delimiter=',', fmt='%.17g')
    np.savetxt(f'{H}/fz_zgrid.csv', zgrid[:, None], delimiter=',', fmt='%.17g')
    with open(f'{H}/fz_dims.m', 'w') as f:
        f.write(f'bnum={bnum}; xnum={xnum}; znum={znum};\n')
    subprocess.run(['octave', '--no-gui', '--quiet', 'fz_run.m'], cwd=H,
                   check=True, capture_output=True)
    zo = np.loadtxt(f'{H}/fz_z.csv', delimiter=',', ndmin=1).ravel()
    wo = np.loadtxt(f'{H}/fz_w.csv', delimiter=',', ndmin=1).ravel()
    zp, wp = update_cutoffs(P, zgrid, bnum, xnum, znum)
    dz = np.abs(zp.ravel() - zo).max()
    dw = np.abs(wp.ravel() - wo).max()
    worst_z, worst_w = max(worst_z, dz), max(worst_w, dw)
    if max(dz, dw) > 1e-12:
        print(f'  case {case} dims {bnum}x{xnum}x{znum} style {style}: '
              f'dz={dz:.3e} dw={dw:.3e}')

print(f'{NCASE} random cases, dims 2-5 x 2-5 x 2-7')
print(f'  branch hits: {branch}')
print(f'  worst |z_cut diff|      = {worst_z:.3e}')
print(f'  worst |weight_cut diff| = {worst_w:.3e}')
ok = max(worst_z, worst_w) < 1e-12
print('  VERDICT:', 'PASS' if ok else '*** FAIL ***')
sys.exit(0 if ok else 1)
