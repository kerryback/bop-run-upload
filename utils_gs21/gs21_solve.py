#!/usr/bin/env python
"""
Python port of utils_gs21/GS21.m -- the Gomes-Schmid (2021) firm pricing solve.

Why this exists
    GS21.m was the only producer in this repo that could not read config.py, so
    config.py and GS21.m were two competing sources of truth and they disagreed
    on five parameters. This removes that split at the root rather than detecting
    it after the fact: every parameter is read from config.py, the solfiles are
    provenance-stamped against it (utils_gs21/solfile_spec.py), and the solve
    runs in the same environment as the rest of the pipeline. GS21.m is now an
    archival reference that nothing reads.

    It also fixes two substantive defects, both documented below: GS21.m's
    iteration never converged (it sat on a period-2 orbit), and its shock
    quadrature was a step function wrong by up to 4.3e-3.

Faithfulness
    This is a 1:1 port of the algorithm, not a reimplementation. Three places
    depart from the MATLAB, all of them exact rather than approximate:

    1. The expectation operator. GS21.m forms `Mmat .* repmat(V,xz,1) .* pr_mat_re`
       -- three simultaneous statenum x (xnum*znum) arrays, 2.56 GB each. But
       Mmat[s,j] depends only on (x_s, x'_j), and pr_mat = kron(pr_mat_z,
       pr_mat_x) factorizes, so the whole thing is two small tensor contractions
       over an (bnum, xnum, znum) array -- 0.64 MB. Same arithmetic, ~4000x less
       memory. See expectation() and its brute-force test.
    2. The Gauss-Hermite shock integrals. `sum(max(P + nodes',0).*weights,2)` and
       the default indicator are step functions of the node value, so with nodes
       sorted they reduce to a searchsorted plus prefix sums -- exact, and O(log n)
       per state instead of O(n).
    3. The shock expectations, and hence termination. GS21.m takes E over a
       normal truncated to +-4*sigma_m using a 360-node Gauss-Hermite rule with
       a hard indicator, which makes the no-default probability a STEP function
       of P -- jumping by one node weight (~8.9e-3) at each crossing, and wrong
       by up to 4.3e-3. Both expectations are elementary in closed form, so this
       port evaluates them exactly (quad='exact'). That is a correction, not a
       reformulation, and it is what makes the solve converge: GS21.m's nested
       alternation lands on an exact period-2 orbit (|P_k - P_k-1| pinned at
       1.87e-01 while |P_k - P_k-2| -> 3.4e-13), which is why its outer loop has
       no break and why the committed solfiles are an interrupted run. With the
       exact expectations a single fused iteration converges to 9.1e-13.
       quad='gh' restores GS21.m's rule for comparison. See solve().
    4. Kernel row renormalisation (default on). E[M|x] = e^{-r} is an identity
       of the continuous model, but the tauchen chain truncates the tails the
       lognormal kernel weights, so the discrete row sums drift from e^{-r} --
       worst at the edge x rows, where risk-neutral mass piles up. Since the
       discrete economy's SDF is the discretised kernel itself, each row of K
       is rescaled so the identity holds exactly (also making the Bellman
       operator a contraction on any grid). The pre-fix error is printed at
       setup; --no-renorm restores the raw kernel of GS21.m and the committed
       solfiles.

MATLAB conventions preserved exactly
    * state ordering is b fastest, then x, then z  (GS21.m:95-97)
    * every reshape is column-major -- order='F' throughout. This is the single
      most dangerous thing about the port: C-order reshapes produce numerically
      valid garbage that converges and writes plausible files.
    * interp1(...,'spline','extrap') is a cubic spline with extrapolation, NOT
      linear. np.interp would be silently wrong.
    * igrid is left-closed: inum points stopping one step short of imax.

Usage (from the repo root):
    python utils_gs21/gs21_solve.py --check          # dims/setup only, no solve
    python utils_gs21/gs21_solve.py --maxouter 5     # short solve, writes nothing
    python utils_gs21/gs21_solve.py --write DIR      # solve and write 18 files
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_hermite
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import config                                    # noqa: E402

# Import tauchen directly, NOT as utils_gs21.tauchen: that would execute
# utils_gs21/__init__.py, which imports the consumers, which verify(mode='error')
# at import time -- so this producer could not run while the stamp it is meant to
# refresh was stale. A producer must not depend on the consumer package.
sys.path.insert(0, _HERE)
from tauchen import tauchen                      # noqa: E402


# ---------------------------------------------------------------- setup


class Setup:
    """Grids, transition matrices, payoffs and the SDF -- everything before the VFI."""

    def __init__(self, gamma_x=None, sigma_m=None, r=None, xi=None, quad='exact',
                 renorm=True):
        c = config
        self.beta, self.gamma = c.GS21_BETA, c.GS21_GAMMA
        self.g, self.alpha, self.delta = c.GS21_G, c.GS21_ALPHA, c.GS21_DELTA
        self.tau, self.phi = c.GS21_TAU, c.GS21_PHI
        self.kappa_e, self.kappa_b = c.GS21_KAPPA_E, c.GS21_KAPPA_B
        self.chi = c.GS21_CHI
        self.rho_x, self.sigma_x, self.x_bar = c.GS21_RHO_X, c.GS21_SIGMA_X, c.GS21_XBAR
        self.rho_z, self.sigma_z = c.GS21_RHO_Z, c.GS21_SIGMA_Z
        self.bmin, self.bmax, self.bnum = c.GS21_BMIN, c.GS21_BMAX, c.GS21_BNUM
        self.imin, self.imax, self.inum = c.GS21_IMIN, c.GS21_IMAX, c.GS21_INUM
        self.xnum, self.znum, self.mnstdev = c.GS21_XNUM, c.GS21_ZNUM, c.GS21_MNSTDEV

        # All four now come from config.py. They used to be split across
        # config.py and GS21.m, which disagreed; see the note above GS21_R.
        # The keyword overrides exist only for the identification experiments.
        self.gamma_x = c.GS21_GAMMA_X if gamma_x is None else gamma_x
        self.sigma_m = c.GS21_SIGMA_M if sigma_m is None else sigma_m
        self.r = c.GS21_R if r is None else r
        self.xi = c.GS21_ZETA if xi is None else xi
        self.quad = quad
        self.renorm = renorm

        self._build()

    def _build(self):
        bnum, xnum, znum, inum = self.bnum, self.xnum, self.znum, self.inum
        self.statenum = statenum = bnum * xnum * znum
        self.xz = xz = xnum * znum

        # --- grids (GS21.m:65-74) ---
        self.bgrid = np.linspace(self.bmin, self.bmax, bnum)
        self.igrid = np.linspace(self.imin, self.imax, inum + 1)[:-1]
        self.xgrid, self.pr_x = tauchen(self.sigma_x, self.rho_x, self.mnstdev, xnum)
        self.zgrid, self.pr_z = tauchen(self.sigma_z, self.rho_z, self.mnstdev, znum)

        # --- Gauss-Hermite shock nodes (GS21.m:55-60) ---
        x_gh, w_gh = roots_hermite(20000)
        w_gh = w_gh / np.sqrt(np.pi)
        nodes = x_gh * np.sqrt(2) * self.sigma_m
        mbar = 4 * self.sigma_m
        keep = (nodes > -mbar) & (nodes < mbar)
        nodes, wts = nodes[keep], w_gh[keep]
        order = np.argsort(nodes)                 # searchsorted below needs ascending
        self.nodes, self.weights = nodes[order], wts[order] / wts.sum()
        # prefix sums for the exact step-function reductions
        self._cw = np.concatenate(([0.0], np.cumsum(self.weights)))
        self._cwn = np.concatenate(([0.0], np.cumsum(self.weights * self.nodes)))
        # Closed forms for the same truncated-normal expectations. The shock is
        # N(0, sigma_m^2) truncated to (-mbar, mbar) and renormalised, so both
        # integrals are elementary -- exact, and smooth in P (see shock_survive).
        self._mbar = mbar
        self._Zt = norm.cdf(mbar / self.sigma_m) - norm.cdf(-mbar / self.sigma_m)

        # --- state space (GS21.m:95-102), b fastest then x then z ---
        b_v = np.tile(self.bgrid, xz)
        x_v = np.tile(np.repeat(self.xgrid, bnum), znum)
        z_v = np.repeat(self.zgrid, bnum * xnum)
        self.b_val, self.x_val, self.z_val = b_v, x_v, z_v
        self.b_ind = np.tile(np.arange(bnum), xz)
        self.x_ind = np.tile(np.repeat(np.arange(xnum), bnum), znum)

        # --- payoffs (GS21.m:127, 135) ---
        self.pi_Rmat = (np.exp(x_v + z_v) - self.delta) - (1 - self.tau) * b_v
        self.def_Rmat_pre = self.phi * (1 - self.delta + np.exp(x_v + z_v))

        # --- SDF kernel, collapsed to (xnum, xnum) (GS21.m:138) ---
        # Mmat[s, j] = M[x_s, x'_j], independent of z'. K folds in pr_x so the
        # expectation is a single contraction.
        M = np.exp(-self.r - 0.5 * self.gamma_x**2
                   - self.gamma_x * (self.xgrid[None, :]
                                     - (1 - self.rho_x) * self.x_bar
                                     - self.rho_x * self.xgrid[:, None]) / self.sigma_x)
        self.K = M * self.pr_x                    # (xnum, xnum), [x, x']
        # E[M|x] = e^{-r} is an identity of the continuous model, but on the
        # tauchen chain it holds only to discretization error: the lognormal
        # kernel weights the tail states pr_x truncates, so row sums drift from
        # e^{-r} -- worst at the edge x rows, exactly where risk-neutral mass
        # piles up. The discrete economy's SDF is this K itself, so renormalise
        # each row to make the identity exact (this also makes the Bellman
        # operator a contraction on any grid). --no-renorm restores the raw
        # kernel for comparison with the committed solfiles.
        self.kernel_rf_err = float(np.abs(self.K.sum(axis=1) - np.exp(-self.r)).max())
        if self.renorm:
            self.K = self.K * (np.exp(-self.r) / self.K.sum(axis=1))[:, None]

    # ---------------------------------------------------------- operators

    def expectation(self, V):
        """sum(Mmat .* repmat(reshape(V,bnum,xz), xz, 1) .* pr_mat_re, 2).

        V is (statenum,) in MATLAB's b-fastest order. Returns (statenum,).
        Exploits Mmat[s,j] = M[x_s,x'_j] and pr_mat = kron(pr_z, pr_x).
        """
        bnum, xnum, znum = self.bnum, self.xnum, self.znum
        V3 = V.reshape(bnum, xnum, znum, order='F')        # [b, x', z']
        W = V3 @ self.pr_z.T                               # [b, x', z]  contract z'
        # out[b,x,z] = sum_{x'} K[x,x'] * W[b,x',z]. The contracted index is K's
        # SECOND -- using K.T here silently transposes the kernel and costs ~8
        # digits without changing the shape, so it converges on a wrong answer.
        out = np.einsum('xy,byz->bxz', self.K, W)          # [b, x, z]   contract x'
        return out.reshape(-1, order='F')

    def _suffix_weight(self, thresh):
        """Total weight of nodes strictly greater than `thresh`, per element."""
        idx = np.searchsorted(self.nodes, thresh, side='right')
        return 1.0 - self._cw[idx], idx

    def shock_survive(self, P):
        """P(P + shock > 0) -- the no-default probability.

        quad='gh' reproduces GS21.m: the weight of the discrete nodes above -P.
        That is a STEP function of P, jumping by one node weight (~8.9e-3)
        whenever P crosses a node, and it is the reason the VFI cannot converge
        -- see solve(). quad='exact' uses the truncated-normal CDF instead:
        smooth, and exact rather than 4.3e-3 off.
        """
        if self.quad == 'gh':
            w, _ = self._suffix_weight(-P)
            return w
        a = np.clip(-P, -self._mbar, self._mbar)
        return (norm.cdf(self._mbar / self.sigma_m)
                - norm.cdf(a / self.sigma_m)) / self._Zt

    def shock_max(self, P):
        """E[max(P + shock, 0)] (GS21.m:262-263)."""
        if self.quad == 'gh':
            w, idx = self._suffix_weight(-P)
            return P * w + (self._cwn[-1] - self._cwn[idx])
        a = np.clip(-P, -self._mbar, self._mbar)
        return P * self.shock_survive(P) + self.sigma_m * (
            norm.pdf(a / self.sigma_m)
            - norm.pdf(self._mbar / self.sigma_m)) / self._Zt

    def by_bprime(self, V):
        """kron(reshape(V,bnum,xz)', ones(bnum,1)) -> (statenum, bnum).

        Row s gives V at every b' choice, holding (x,z) at state s's values.
        """
        V_re = V.reshape(self.bnum, self.xz, order='F')     # [b', xz]
        return np.repeat(V_re.T, self.bnum, axis=0)         # [(xz,b), b']


# ------------------------------------------------------------- the solve


def solve(su, maxiter=40000, tol=1e-11, verbose=True, damp=1.0, report_every=500):
    """Fused fixed-point iteration: one debt update and one price update per pass.

    Why not GS21.m's nested loops. GS21.m alternates two inner loops (solve Q
    given P, then P given Q) under tolerances that start at 1e-3/1e-4 and shrink
    by /1.2 whenever met, and its outer loop has no break. That alternation does
    not converge -- it lands on an exact period-2 orbit. Measured at 20x20x20:
    |P_k - P_k-1| pins to 1.8748e-01 while |P_k - P_k-2| falls to 3.4e-13. The
    committed MATLAB solfiles are one point of that orbit, which is why GS21.m
    was simply interrupted. Tightening the inner tolerances to 1e-8 reproduces
    the identical plateau, so the schedule was never the problem; and freezing
    the discrete b' policy does not help either, even though that policy is
    self-consistent (0 of 8000 states would switch).

    The actual cause was the quadrature: with quad='gh', shock_survive is a step
    function of P, so the map is genuinely discontinuous at the ~8.9e-3 scale of
    one node weight -- and the residual oscillation sat at exactly that scale.
    With quad='exact' the map is smooth and this iteration converges to 9.1e-13.

    The expectation operator itself is unchanged and is verified against GS21.m's
    dense expressions to 9.6e-15; only the iteration scheme differs.
    """
    statenum, bnum, g, xi = su.statenum, su.bnum, su.g, su.xi
    imin, imax = su.imin, su.imax
    idx = np.arange(statenum)
    spl_x = su.bgrid

    P_up = np.zeros(statenum)
    P_down = np.zeros(statenum)
    Q_0 = np.zeros(statenum)
    Q_I_no = np.zeros(statenum)
    prob_i_up = np.ones(statenum)
    prob_i_down = np.ones(statenum)
    converged = False

    for it in range(maxiter):
        P_up_prev, P_down_prev = P_up, P_down

        # ---- debt update (GS21.m:195-222) ----
        Q_up_old = prob_i_up * Q_I_no + (1 - prob_i_up) * Q_0
        Q_down_old = prob_i_down * Q_I_no + (1 - prob_i_down) * Q_0
        surv_up = su.shock_survive(P_up)
        surv_dn = su.shock_survive(P_down)
        pay_up = (su.b_val + Q_up_old) * surv_up + su.def_Rmat_pre * (1 - surv_up)
        pay_dn = (su.b_val + Q_down_old) * surv_dn + su.def_Rmat_pre * (1 - surv_dn)
        Q_0_new = xi * su.expectation(pay_up) + (1 - xi) * su.expectation(pay_dn)
        Q_0 = Q_0 + damp * (Q_0_new - Q_0)
        Q_I_re = (g * Q_0).reshape(bnum, su.xz, order='F')
        # interp1(bgrid, Q_I_re, bgrid/g, 'spline', 'extrap')  (GS21.m:222)
        Q_I_no = CubicSpline(spl_x, Q_I_re, axis=0,
                             extrapolate=True)(su.bgrid / g).reshape(-1, order='F')

        # ---- price update (GS21.m:252-341) ----
        # GS21.m:257 uses kron(Q_I_re', ...) where Q_I_re = reshape(g*Q_0) -- the
        # b'-CHOICE term -- while subtracting Q_I_no, the spline-shifted
        # current-debt term. Two different objects.
        prof0_up = su.pi_Rmat[:, None] + ((1 - su.kappa_b) * su.by_bprime(Q_0)
                                          - Q_0[:, None])
        profI_up = su.pi_Rmat[:, None] + ((1 - su.kappa_b) * su.by_bprime(g * Q_0)
                                          - Q_I_no[:, None])
        P0_up_R = (1 + (prof0_up <= 0) * su.kappa_e) * prof0_up
        PI_up_R = (1 + (profI_up <= 0) * su.kappa_e) * profI_up
        down_R = ((1 + (su.pi_Rmat <= 0) * su.kappa_e) * su.pi_Rmat)[:, None]

        P_up_pert = su.shock_max(P_up)
        P_down_pert = su.shock_max(P_down)
        EPI = xi * su.expectation(g * P_up_pert) + (1 - xi) * su.expectation(g * P_down_pert)
        EP0 = xi * su.expectation(P_up_pert) + (1 - xi) * su.expectation(P_down_pert)
        EP0_m, EPI_m = su.by_bprime(EP0), su.by_bprime(EPI)

        P0_up_full = P0_up_R + EP0_m
        PI_up_full = PI_up_R + EPI_m
        P0_down_full = down_R + EP0_m
        PI_down_full = down_R + EPI_m

        P0_up_v = P0_up_full.max(axis=1)
        no_up_bprime = P0_up_full.argmax(axis=1)
        PI_up_v = PI_up_full.max(axis=1)
        I_up_bprime = PI_up_full.argmax(axis=1)
        # interp1(bgrid, P0_down', bgrid, 'spline','extrap')' is the identity;
        # then pick the column b' = b (no refinancing). GS21.m:293-295.
        P0_down_v = P0_down_full[idx, su.b_ind]
        # PI_down evaluates the spline at bgrid/g, so it is NOT the identity.
        PI_down_v = CubicSpline(spl_x, PI_down_full.T, axis=0,
                                extrapolate=True)(su.bgrid / g).T[idx, su.b_ind]

        i_cut_up = np.minimum(imax, np.maximum(imin, PI_up_v - P0_up_v))
        i_cut_down = np.minimum(imax, np.maximum(imin, PI_down_v - P0_down_v))
        prob_i_up = (i_cut_up - imin) / (imax - imin)
        prob_i_down = (i_cut_down - imin) / (imax - imin)
        P_up_new = prob_i_up * (PI_up_v - 0.5 * (i_cut_up + imin)) + (1 - prob_i_up) * P0_up_v
        P_down_new = prob_i_down * (PI_down_v - 0.5 * (i_cut_down + imin)) + (1 - prob_i_down) * P0_down_v
        P_up = P_up + damp * (P_up_new - P_up)
        P_down = P_down + damp * (P_down_new - P_down)

        move = max(np.abs(P_up - P_up_prev).max(), np.abs(P_down - P_down_prev).max())
        if move < tol:
            converged = True
        if verbose and (it % report_every == 0 or converged):
            print(f'  pass {it + 1:6d}  |dP| = {move:.6e}', flush=True)
        if converged:
            if verbose:
                print(f'  converged: |dP| < {tol:g} after {it + 1} passes')
            break
    if not converged:
        print(f'  WARNING: did NOT converge -- |dP| = {move:.6e} after {maxiter} passes')

    z_cut_up, _ = update_cutoffs(P_up, su.zgrid, bnum, su.xnum, su.znum)
    z_cut_down, _ = update_cutoffs(P_down, su.zgrid, bnum, su.xnum, su.znum)

    return {
        'P_up.csv': P_up, 'P_down.csv': P_down,
        'PI_up.csv': PI_up_v, 'PI_down.csv': PI_down_v,
        'P0_up.csv': P0_up_v, 'P0_down.csv': P0_down_v,
        'Q_0.csv': Q_0, 'Q_I.csv': g * Q_0,
        'z_cut_up.csv': z_cut_up, 'z_cut_down.csv': z_cut_down,
        'i_cut_up.csv': i_cut_up, 'i_cut_down.csv': i_cut_down,
        'b_refin_0.csv': su.bgrid[no_up_bprime], 'b_refin_I.csv': su.bgrid[I_up_bprime],
        '_converged': converged, '_move': move, '_passes': it + 1,
    }


def update_cutoffs(P, zgrid, bnum, xnum, znum):
    """1:1 port of GS21.m:395. Returns (z_cut, weight_cut), each (xnum*bnum,)."""
    pos = P.reshape(xnum * bnum, znum, order='F') > 0
    z_ind = np.argmax(pos, axis=1)
    cond0 = pos.sum(1) == 0
    z_cut = np.where(cond0, zgrid[-1] + 1e-10, zgrid[z_ind])

    P_re = P.reshape(bnum * xnum, znum, order='F')
    rows = np.arange(bnum * xnum)
    P_plus = P_re[rows, z_ind]
    P_minus = P_re[rows, np.maximum(0, z_ind - 1)]
    cond2 = (1 - pos).sum(1) == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = P_plus / (P_plus - P_minus)
    weight[~np.isfinite(weight)] = 0.5
    weight_cut = cond0 * 0.5 + cond2 * 0.5 + (1 - cond0 - cond2) * weight
    return z_cut, weight_cut


# --------------------------------------------------------------------- CLI


def write_outputs(out_dir, su, pol):
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in pol.items():
        if name.startswith('_'):
            continue
        np.savetxt(os.path.join(out_dir, name), np.asarray(arr).ravel(), delimiter=',')
    for name, arr in [('bgrid.csv', su.bgrid), ('igrid.csv', su.igrid),
                      ('xgrid.csv', su.xgrid), ('zgrid.csv', su.zgrid)]:
        np.savetxt(os.path.join(out_dir, name), arr, delimiter=',')


def compare_committed(pol, solfiles_dir=None):
    """Report each policy against the committed MATLAB output.

    Exact agreement is NOT expected: GS21.m's outer loop has no break, so the
    committed files are an interactively-interrupted run at an unknown iteration.
    This measures how close the two fixed points are, not whether bytes match.
    """
    import pandas as pd
    solfiles_dir = solfiles_dir or os.path.join(_HERE, 'GS21_solfiles')
    rows = []
    for name, arr in sorted(pol.items()):
        if name.startswith('_'):
            continue
        path = os.path.join(solfiles_dir, name)
        if not os.path.exists(path):
            continue
        com = pd.read_csv(path, header=None).values.ravel()
        a = np.asarray(arr).ravel()
        if a.shape != com.shape:
            rows.append((name, float('nan'), float('nan'), f'shape {a.shape} vs {com.shape}'))
            continue
        scale = max(np.abs(com).max(), 1e-30)
        rows.append((name, np.abs(a - com).max(), np.abs(a - com).max() / scale, ''))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='set up only, no solve')
    ap.add_argument('--maxiter', type=int, default=40000)
    ap.add_argument('--tol', type=float, default=1e-11)
    ap.add_argument('--quad', choices=('exact', 'gh'), default='exact',
                    help="'exact' = closed-form truncated-normal shock "
                         "expectations (smooth, converges); 'gh' = GS21.m's "
                         "360-node rule (step function, does NOT converge)")
    ap.add_argument('--gamma-x', type=float, default=None,
                    help='override GS21_GAMMA_X (identification experiments only)')
    # r, xi and gamma_x were the three parameters config.py and GS21.m disagreed
    # on. They are settled (see config.py above GS21_R); these overrides remain
    # only so the identification experiment can be reproduced.
    ap.add_argument('--r', type=float, default=None, help='override GS21_R')
    ap.add_argument('--xi', type=float, default=None, help='override GS21_ZETA')
    ap.add_argument('--no-renorm', action='store_true',
                    help='skip the kernel row renormalisation (departure 4 in '
                         'the docstring); reproduces the raw GS21.m kernel')
    ap.add_argument('--label', default='', help='tag for the run, printed in output')
    ap.add_argument('--write', metavar='DIR', default=None)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--damp', type=float, default=1.0,
                    help='relaxation on the updates; <1 damps, fixed point unchanged')
    ap.add_argument('--report-every', type=int, default=500)
    args = ap.parse_args()

    t0 = time.time()
    su = Setup(gamma_x=args.gamma_x, r=args.r, xi=args.xi, quad=args.quad,
               renorm=not args.no_renorm)
    print(f'setup{" [" + args.label + "]" if args.label else ""}: '
          f'statenum={su.statenum:,}  xz={su.xz:,}  {len(su.nodes)} shock nodes  '
          f'({time.time() - t0:.2f}s)')
    print(f'  r={su.r!r}  xi={su.xi!r}  gamma_x={su.gamma_x!r}  quad={su.quad!r}  '
          f'renorm={su.renorm!r}')
    print(f'  raw kernel max |E[M|x] - e^-r| = {su.kernel_rf_err:.3e}'
          + ('  (corrected to 0 by renormalisation)' if su.renorm else '  (LEFT IN)'))
    if args.check:
        return

    pol = solve(su, maxiter=args.maxiter, tol=args.tol,
                verbose=not args.quiet, damp=args.damp,
                report_every=args.report_every)
    print(f'\nsolve finished in {time.time() - t0:.1f}s  '
          f'({pol["_passes"]} passes, |dP| = {pol["_move"]:.3e}, '
          f'converged={pol["_converged"]})')

    print('\nvs committed MATLAB output (exact match not expected -- see docstring):')
    print(f'  {"file":16s} {"max abs diff":>14s} {"rel to scale":>14s}')
    for name, ad, rd, note in compare_committed(pol):
        print(f'  {name:16s} {ad:14.6g} {rd:14.3e}  {note}')

    if args.write:
        # Never write a non-converged solve: the whole point of the exact
        # quadrature is that a fixed point now exists, so failing to reach one
        # means something is wrong, not that the answer is merely imprecise.
        if not pol['_converged']:
            sys.exit(f'\nREFUSING to write: solve did not converge '
                     f'(|dP| = {pol["_move"]:.3e} after {pol["_passes"]} passes). '
                     f'Nothing written to {args.write}.')
        write_outputs(args.write, su, pol)
        print(f'\nwrote 18 files to {args.write}')


if __name__ == '__main__':
    main()
