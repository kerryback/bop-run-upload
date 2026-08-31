#!/usr/bin/env python
"""Small-grid, random-input reference for one debt pass + one price pass.

Dumps the shared inputs (so Octave runs on identical data) and this port's
outputs, for oct_ref.m to be compared against by cmp.py.

Two things make this test stronger than comparing against the committed
solfiles:

  * Random P takes BOTH signs, which exercises the default branches, cond0 /
    cond2 in update_cutoffs, and a non-degenerate z_cut. In the committed files
    P > 0 everywhere, so z_cut is pinned to zgrid[0] and those paths are
    unreachable -- the z_cut agreement there is vacuous.
  * A small grid makes GS21.m's dense expressions feasible at all. At the
    shipped size Mmat and pr_mat_re are 2.56 GB EACH; here they are ~30 KB.
    The algebra under test is size-independent.

Setup is constructed with quad='gh' because the point is to reproduce GS21.m's
360-node Gauss-Hermite rule. The shipped default is quad='exact' (closed-form
truncated-normal), which is deliberately NOT what GS21.m does -- see
gs21_solve.py's docstring.

Configure via environment variables:
    T_BNUM T_XNUM T_ZNUM   grid dims          (default 4, 5, 6)
    T_SEED                 RNG seed           (default 20260826)
    T_PSCALE T_PMEAN       sd/mean of random P (default 8.0, 0.0)
    T_SIGMA_M              shock sd           (default 5.0)
"""
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # archive/<this>/ -> repo root
sys.path.insert(0, REPO)

import config                                          # noqa: E402

BNUM = int(os.environ.get('T_BNUM', 4))
XNUM = int(os.environ.get('T_XNUM', 5))
ZNUM = int(os.environ.get('T_ZNUM', 6))
config.GS21_BNUM, config.GS21_XNUM, config.GS21_ZNUM = BNUM, XNUM, ZNUM

# Load gs21_solve WITHOUT going through utils_gs21/__init__.py, which imports
# the consumers, which verify(mode='error') at import time. This harness
# deliberately shrinks the grid dims in config for a small-grid test, so that
# check would (correctly!) fire and abort. Same bypass the producer uses.
import importlib.util as _ilu
_p = os.path.join(REPO, 'utils_gs21', 'gs21_solve.py')
_spec = _ilu.spec_from_file_location('_gs21_solve_harness', _p)
_gs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gs)
Setup, update_cutoffs = _gs.Setup, _gs.update_cutoffs

R, XI, GAMMA_X = 0.1 / 12, 0.01, 0.5
SIGMA_M = float(os.environ.get('T_SIGMA_M', 5.0))
su = Setup(gamma_x=GAMMA_X, sigma_m=SIGMA_M, r=R, xi=XI, quad='gh')
sn, xz, bnum, xnum, znum = su.statenum, su.xz, su.bnum, su.xnum, su.znum
g, xi, imin, imax = su.g, su.xi, su.imin, su.imax

rng = np.random.default_rng(int(os.environ.get('T_SEED', 20260826)))
PS = float(os.environ.get('T_PSCALE', 8.0))
PM = float(os.environ.get('T_PMEAN', 0.0))
P_up_old = rng.normal(PM, PS, sn)
P_down_old = rng.normal(PM, PS, sn)
Q_0_old = rng.normal(0.5, 1.0, sn)
Q_I_re_old = rng.normal(0.5, 1.0, sn)      # MATLAB's Q_I_re_old, = Q_I_no flattened
prob_i_up = rng.uniform(0, 1, sn)
prob_i_down = rng.uniform(0, 1, sn)


def dump(name, a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    np.savetxt(os.path.join(HERE, name + '.csv'), a, delimiter=',', fmt='%.17g')


for n, a in [('bgrid', su.bgrid), ('xgrid', su.xgrid), ('zgrid', su.zgrid),
             ('pr_x', su.pr_x), ('pr_z', su.pr_z),
             ('nodes', su.nodes), ('weights', su.weights),
             ('in_P_up_old', P_up_old), ('in_P_down_old', P_down_old),
             ('in_Q_0_old', Q_0_old), ('in_Q_I_re_old', Q_I_re_old),
             ('in_prob_i_up', prob_i_up), ('in_prob_i_down', prob_i_down)]:
    dump(n, a)
with open(os.path.join(HERE, 'params.m'), 'w') as f:
    for k, v in [('r', R), ('xi', XI), ('gamma_x', GAMMA_X), ('g', g),
                 ('delta', su.delta), ('tau', su.tau), ('phi', su.phi),
                 ('kappa_b', su.kappa_b), ('kappa_e', su.kappa_e),
                 ('rho_x', su.rho_x), ('sigma_x', su.sigma_x), ('x_bar', su.x_bar),
                 ('bnum', bnum), ('xnum', xnum), ('znum', znum),
                 ('imin', imin), ('imax', imax)]:
        f.write('%s = %.17g;\n' % (k, float(v)))

# ============================ one debt pass ============================
Q_up_old = prob_i_up * Q_I_re_old + (1 - prob_i_up) * Q_0_old
Q_down_old = prob_i_down * Q_I_re_old + (1 - prob_i_down) * Q_0_old

surv_up, surv_dn = su.shock_survive(P_up_old), su.shock_survive(P_down_old)
pay_up = (su.b_val + Q_up_old) * surv_up + su.def_Rmat_pre * (1 - surv_up)
pay_dn = (su.b_val + Q_down_old) * surv_dn + su.def_Rmat_pre * (1 - surv_dn)

EQ0_up_post, EQ0_dn_post = su.expectation(pay_up), su.expectation(pay_dn)
Q_0 = xi * EQ0_up_post + (1 - xi) * EQ0_dn_post
Q_I_re = (g * Q_0).reshape(bnum, xz, order='F')
Q_I_no = CubicSpline(su.bgrid, Q_I_re, axis=0,
                     extrapolate=True)(su.bgrid / g).reshape(-1, order='F')

for n, a in [('py_Q_up_old', Q_up_old), ('py_Q_down_old', Q_down_old),
             ('py_pay_up', pay_up), ('py_pay_dn', pay_dn),
             ('py_EQ0_up', EQ0_up_post), ('py_EQ0_down', EQ0_dn_post),
             ('py_Q_0', Q_0), ('py_Q_I_no', Q_I_no)]:
    dump(n, a)

# ========================== one price pass ==========================
Q0_by_bp, QI_by_bp = su.by_bprime(Q_0), su.by_bprime(g * Q_0)
prof0_up = su.pi_Rmat[:, None] + ((1 - su.kappa_b) * Q0_by_bp - Q_0[:, None])
P0_up_R = (1 + (prof0_up <= 0) * su.kappa_e) * prof0_up
profI_up = su.pi_Rmat[:, None] + ((1 - su.kappa_b) * QI_by_bp - Q_I_no[:, None])
PI_up_R = (1 + (profI_up <= 0) * su.kappa_e) * profI_up
down_R = ((1 + (su.pi_Rmat <= 0) * su.kappa_e) * su.pi_Rmat)[:, None]

P_up_pert, P_down_pert = su.shock_max(P_up_old), su.shock_max(P_down_old)
EPI = xi * su.expectation(g * P_up_pert) + (1 - xi) * su.expectation(g * P_down_pert)
EP0 = xi * su.expectation(P_up_pert) + (1 - xi) * su.expectation(P_down_pert)
EP0_m, EPI_m = su.by_bprime(EP0), su.by_bprime(EPI)

P0_up_full, PI_up_full = P0_up_R + EP0_m, PI_up_R + EPI_m
P0_down_full, PI_down_full = down_R + EP0_m, down_R + EPI_m

P0_up_v, no_up_bprime = P0_up_full.max(1), P0_up_full.argmax(1)
PI_up_v, I_up_bprime = PI_up_full.max(1), PI_up_full.argmax(1)
P0_down_v = P0_down_full[np.arange(sn), su.b_ind]
PI_dn_sp = CubicSpline(su.bgrid, PI_down_full.T, axis=0,
                       extrapolate=True)(su.bgrid / g).T
PI_down_v = PI_dn_sp[np.arange(sn), su.b_ind]

i_cut_up = np.minimum(imax, np.maximum(imin, PI_up_v - P0_up_v))
i_cut_down = np.minimum(imax, np.maximum(imin, PI_down_v - P0_down_v))
pi_up, pi_dn = (i_cut_up - imin) / (imax - imin), (i_cut_down - imin) / (imax - imin)
P_up_new = pi_up * (PI_up_v - 0.5 * (i_cut_up + imin)) + (1 - pi_up) * P0_up_v
P_down_new = pi_dn * (PI_down_v - 0.5 * (i_cut_down + imin)) + (1 - pi_dn) * P0_down_v

zc_up, wc_up = update_cutoffs(P_up_new, su.zgrid, bnum, xnum, znum)
zc_dn, wc_dn = update_cutoffs(P_down_new, su.zgrid, bnum, xnum, znum)

for n, a in [('py_P_up_pert', P_up_pert), ('py_P_down_pert', P_down_pert),
             ('py_EPI', EPI), ('py_EP0', EP0),
             ('py_P0_up', P0_up_v), ('py_P0_down', P0_down_v),
             ('py_PI_up', PI_up_v), ('py_PI_down', PI_down_v),
             ('py_i_cut_up', i_cut_up), ('py_i_cut_down', i_cut_down),
             ('py_P_up', P_up_new), ('py_P_down', P_down_new),
             ('py_b_refin_0', su.bgrid[no_up_bprime]),
             ('py_b_refin_I', su.bgrid[I_up_bprime]),
             ('py_z_cut_up', zc_up), ('py_weight_cut_up', wc_up),
             ('py_z_cut_down', zc_dn), ('py_weight_cut_down', wc_dn)]:
    dump(n, a)

# report how hard this draw actually works the branches
pos = P_up_new.reshape(xnum * bnum, znum, order='F') > 0
print(f'statenum={sn}  nodes={len(su.nodes)}')
print(f'P_down_new sign split: {(P_down_new > 0).sum()} pos / {(P_down_new <= 0).sum()} nonpos')
print(f'update_cutoffs branches: cond0={int((pos.sum(1) == 0).sum())} '
      f'cond2={int(((~pos).sum(1) == 0).sum())} '
      f'interior={int(((pos.sum(1) > 0) & ((~pos).sum(1) > 0)).sum())}')
print(f'prof0_up<=0 fraction: {(prof0_up <= 0).mean():.3f}')
print(f'i_cut interior (unclipped): up={int(((i_cut_up > imin) & (i_cut_up < imax)).sum())} '
      f'down={int(((i_cut_down > imin) & (i_cut_down < imax)).sum())} of {sn}')
