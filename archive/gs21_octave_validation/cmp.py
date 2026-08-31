#!/usr/bin/env python
"""Compare py_ref.py's outputs against oct_ref.m's, intermediate by intermediate.

Usage:  python cmp.py [label]
Run py_ref.py then oct_ref.m first; run.sh does all three.
"""
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
NAMES = ['Q_up_old', 'Q_down_old', 'pay_up', 'pay_dn', 'EQ0_up', 'EQ0_down',
         'Q_0', 'Q_I_no', 'P_up_pert', 'P_down_pert', 'EPI', 'EP0',
         'P0_up', 'P0_down', 'PI_up', 'PI_down', 'i_cut_up', 'i_cut_down',
         'P_up', 'P_down', 'b_refin_0', 'b_refin_I',
         'z_cut_up', 'weight_cut_up', 'z_cut_down', 'weight_cut_down']

verbose = '-v' in sys.argv
label = next((a for a in sys.argv[1:] if not a.startswith('-')), '')
worst = []
if verbose:
    print(f'  {"quantity":16s} {"max abs diff":>13s} {"max rel diff":>13s} {"scale":>11s}')
for n in NAMES:
    p = np.loadtxt(f'{H}/py_{n}.csv', delimiter=',', ndmin=1).ravel()
    o = np.loadtxt(f'{H}/oct_{n}.csv', delimiter=',', ndmin=1).ravel()
    if p.shape != o.shape:
        print(f'  !! {n}: SHAPE {p.shape} vs {o.shape}')
        worst.append((9e9, n))
        continue
    ad = np.abs(p - o).max()
    sc = max(np.abs(o).max(), 1e-300)
    worst.append((ad / sc, n))
    if verbose:
        print(f'  {n:16s} {ad:13.4e} {ad / sc:13.4e} {sc:11.4e}')
worst.sort(reverse=True)
print(f'  {label:28s} worst rel = {worst[0][0]:.3e} ({worst[0][1]}) | next: '
      + ', '.join(f'{n}={r:.1e}' for r, n in worst[1:4]))
print(f'  {"":28s} VERDICT: ' + ('PASS' if worst[0][0] < 1e-11 else '*** FAIL ***'))
sys.exit(0 if worst[0][0] < 1e-11 else 1)
