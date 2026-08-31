"""BGN audit: does Jstar.csv reproduce, and are its truncations adequate?

vasicek.py hardcodes three truncations:
    max_prds = 1500     the bond-price recursion horizon
    k <= 400            maturity sum inside inner_sum, weighted pi^k
    s <= 950            option-maturity sum in Jstar, UNWEIGHTED
plus roots_laguerre(100) for the beta-density integral.

With PI = 0.99, pi^400 = 1.8e-2, so the k-tail is not obviously negligible, and
the s-sum carries no discount factor at all. This reimplements the formulas with
those four as parameters so the truncation error can be measured rather than
assumed. Structure mirrors vasicek.py exactly.
"""
import sys, time
import numpy as np
from scipy.optimize import root_scalar, fsolve
from scipy.stats import norm
from scipy.special import roots_laguerre
sys.path.insert(0, '/Users/sjpruitt/GitHub/bop-run-upload')
from config import (PI as pi, RBAR as rbar, KAPPA as kappa, SIGMA_R as sigma_r,
                    BETA_ZR as beta_zr, SIGMA_Z as sigma_z, CBAR as Cbar)


class Model:
    def __init__(self, max_prds=1500, kmax=400, smax=950, nlag=100):
        self.kmax, self.smax = kmax, smax
        M = max_prds
        self.alpha2 = kappa**np.arange(M+1)
        self.alpha1 = np.concatenate(([0], np.cumsum(self.alpha2[:-1])))
        phi2 = rbar - rbar*kappa**np.arange(M+1)
        self.phi1 = np.concatenate(([0], np.cumsum(phi2[:-1] + 0.5*sigma_z**2*np.ones(M))))
        self.sigma2_sq = (sigma_r**2/(1-kappa**2)
                          - (sigma_r**2/(1-kappa**2))*kappa**(2*np.arange(M+1)))
        s12 = np.zeros(M+1)
        for i in range(M):
            s12[i+1] = kappa*(s12[i] + self.sigma2_sq[i]) + beta_zr
        self.sigma1_sq = np.concatenate(
            ([0], np.cumsum(self.sigma2_sq[:-1] + s12[:-1] + sigma_z**2*np.ones(M))))
        # beta params
        x = fsolve(self._fit, x0=[1, 1])
        self.beta_star, self.scale = x[0], np.exp(x[1])
        nodes, self.weights = roots_laguerre(nlag)
        self.beta_nodes = self.beta_star - self.scale*nodes
        self.rstar_nodes = np.array([self._rstar(b) for b in self.beta_nodes])

    def B(self, k, r):
        return np.exp(-self.alpha1[k]*r - self.phi1[k] + 0.5*self.sigma1_sq[k])

    def D(self, r, n=None):
        n = n or (len(self.alpha1)-1)
        b = np.exp(-self.alpha1[1:n+1]*r - self.phi1[1:n+1] + 0.5*self.sigma1_sq[1:n+1])
        return np.sum(pi**np.arange(1, n+1) * b)

    def _fit(self, x):
        bs, sc = x[0], np.exp(x[1])
        pim = lambda r: sc*np.exp((Cbar + np.log(self.D(r)) - bs)/sc)/sc
        return np.array([pim(0)-0.1, pim(rbar)-0.05])

    def _rstar(self, beta):
        return root_scalar(lambda r: Cbar - beta + np.log(self.D(r)), x0=rbar).root

    def J(self, r, om, bm, strike):
        num = (np.log(self.B(om+bm, r)) - np.log(self.B(om, r))
               - np.log(self.B(bm, strike)) + 0.5*self.alpha1[bm]**2*self.sigma2_sq[om])
        d1 = num/(self.alpha1[bm]*np.sqrt(self.sigma2_sq[om]))
        d2 = d1 - self.alpha1[bm]*np.sqrt(self.sigma2_sq[om])
        return (self.B(om+bm, r)*norm.cdf(d1)
                - self.B(om, r)*self.B(bm, strike)*norm.cdf(d2))

    def integralJ(self, r, om, bm):
        return np.sum(self.weights*self.J(r, om, bm, self.rstar_nodes)
                      * np.exp(-self.beta_nodes))

    def inner_sum(self, r, om, kmax=None):
        kmax = kmax or self.kmax
        ig = np.array([self.integralJ(r, om, k) for k in range(1, kmax+1)])
        return np.sum(ig * pi**np.arange(1, kmax+1)), ig

    def Jstar(self, r, smax=None, kmax=None):
        smax = smax or self.smax
        return np.sum([self.inner_sum(r, s, kmax)[0] for s in range(1, smax+1)])


import pandas as pd
tab = pd.read_csv('/Users/sjpruitt/GitHub/bop-run-upload/utils_bgn/BGN_solfiles/Jstar.csv')
print(f'Jstar.csv: {len(tab)} rows, r in [{tab.r.min():.6f}, {tab.r.max():.6f}]\n', flush=True)

m = Model()
print(f'beta_star = {m.beta_star!r}, scale = {m.scale!r}', flush=True)

# --- 1. reproduce shipped values (staleness / identification) ---
print('\n=== 1. does Jstar.csv reproduce from current config.py? ===', flush=True)
idx = [0, 40, 100, 160, 200]
for i in idx:
    r0, j0 = tab.r.iloc[i], tab.J.iloc[i]
    t = time.time(); jc = m.Jstar(r0)
    print(f'  r={r0:+.6f}  stored {j0:.10f}  recomputed {jc:.10f}  '
          f'rel {abs(jc-j0)/abs(j0):.2e}  ({time.time()-t:.0f}s)', flush=True)

# --- 2. truncation in k (weighted pi^k) ---
print('\n=== 2. k-truncation (shipped 400) ===', flush=True)
big = Model(max_prds=3000)
for r0 in (tab.r.iloc[10], tab.r.iloc[100], tab.r.iloc[190]):
    tot, ig = big.inner_sum(r0, 100, kmax=1200)
    w = pi**np.arange(1, 1201)
    cum = np.cumsum(ig*w)
    print(f'  r={r0:+.6f} om=100: k<=400 captures {cum[399]/cum[-1]:.6%} of k<=1200'
          f'   (missing {1-cum[399]/cum[-1]:.4%})', flush=True)

# --- 3. truncation in s (UNWEIGHTED) ---
print('\n=== 3. s-truncation (shipped 950, no discount factor) ===', flush=True)
for r0 in (tab.r.iloc[10], tab.r.iloc[190]):
    vals = np.array([big.inner_sum(r0, s, kmax=400)[0] for s in range(1, 2001)])
    cum = np.cumsum(vals)
    print(f'  r={r0:+.6f}: s<=950 captures {cum[949]/cum[-1]:.6%} of s<=2000'
          f'   (missing {1-cum[949]/cum[-1]:.4%})', flush=True)
    print(f'     inner_sum at s=950: {vals[949]:.3e}   at s=2000: {vals[-1]:.3e}', flush=True)

# --- 4. Gauss-Laguerre node count ---
print('\n=== 4. Laguerre nodes (shipped 100) ===', flush=True)
base = m.Jstar(tab.r.iloc[100], smax=120, kmax=200)
for n in (50, 200, 400):
    mm = Model(nlag=n)
    v = mm.Jstar(tab.r.iloc[100], smax=120, kmax=200)
    print(f'  nlag={n:4d}: rel diff vs 100 = {abs(v-base)/abs(base):.3e}', flush=True)
