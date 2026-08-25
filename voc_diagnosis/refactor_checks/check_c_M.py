import sys, os; sys.path.insert(0, os.getcwd())
"""
CHECK C: how big is the Gram (M x M) in production?  Decides whether blocking is needed.
BGN reaches steady state slowly, and the check_b run (T=80) was pre-steady-state.
KP14 accumulates projects for ~120 months, so it is the worry case.
Simulates ONLY the arrival/survival processes -- no solfiles, no valuation.
"""
import numpy as np

print("=" * 78)
print("BGN: survival pi=0.99/month => mean project life 100 months")
print("=" * 78)
from config import PI as pi, RBAR as rbar, KAPPA as kappa, SIGMA_R as sigma_r, CBAR as Cbar
from utils_bgn.vasicek import D, beta_star, scale
from scipy.stats import expon, norm

np.random.seed(7)
N, T = 1000, 1000
r = np.zeros(T + 1); r[0] = rbar
for t in range(T):
    r[t + 1] = kappa * r[t] + (1 - kappa) * rbar + sigma_r * norm.rvs()
beta = -expon.rvs(size=(T, N), loc=-beta_star, scale=scale)
bond = np.array([D(x) for x in r[1:]]).reshape(T, 1)
invest = (np.exp(Cbar - beta) * bond > 1).astype(float)
live = np.zeros(N)
print(f"  mean P(invest) = {invest.mean():.4f}   implied steady-state live/firm = {invest.mean()/(1-pi):.2f}")
for t in range(T):
    live = np.random.binomial(live.astype(int), pi) + invest[t]
    if t + 1 in (80, 200, 400, 700, 1000):
        M = live.sum()
        print(f"  t={t+1:5d}  live/firm mean={live.mean():6.2f} max={live.max():5.0f}"
              f"   M={M:8.0f}   Gram={M**2*8/1e9:6.3f} GB")

print()
print("=" * 78)
print("KP14: death delta*dt=0.1/12 per month => mean project life 120 months")
print("=" * 78)
from config import (KP14_DT as dt, KP14_DELTA as delta, KP14_MU_LAMBDA as mu_lambda,
                   KP14_SIGMA_LAMBDA as sigma_lambda, KP14_MU_H as mu_H, KP14_MU_L as mu_L,
                   KP14_LAMBDA_H as lambda_H, KP14_LAMBDA_L as lambda_L, KP14_PROB_H as prob_H)
np.random.seed(7)
T = 1000
lambda_f = mu_lambda * delta - sigma_lambda * delta * np.log(np.random.uniform(size=N))
state = (np.random.random(N) < prob_H).astype(float)
live = np.zeros(N)
print(f"  lambda_L={lambda_L:.4f} lambda_H={lambda_H:.2f} PROB_H(config)={prob_H:.4f}")
print(f"  lambda_f: mean={lambda_f.mean():.4f}  (mu_lambda*delta={mu_lambda*delta:.3f})")
for t in range(T):
    sw = np.where(state == 1, mu_H * dt, mu_L * dt)
    state = np.where(np.random.random(N) < sw, 1 - state, state)
    rate = lambda_f * (lambda_L + state * (lambda_H - lambda_L))
    live = np.random.binomial(live.astype(int), 1 - dt * delta) + (np.random.random(N) < dt * rate)
    if t + 1 in (200, 400, 700, 1000):
        M = live.sum()
        print(f"  t={t+1:5d}  live/firm mean={live.mean():6.2f} max={live.max():5.0f}"
              f"   M={M:8.0f}   Gram={M**2*8/1e9:6.3f} GB")
print(f"  simulated stationary P(high) = {state.mean():.4f}   (config PROB_H = {prob_H:.4f})")
print(f"  => E[rate] = {(lambda_f*(lambda_L+state*(lambda_H-lambda_L))).mean():.4f}/yr,"
      f"  steady-state live/firm = {(lambda_f*(lambda_L+state*(lambda_H-lambda_L))).mean()/delta:.2f}")

print()
print("=" * 78)
print("GS21: no project structure -- M = N")
print("=" * 78)
print(f"  M = 1000, Gram = {1000**2*8/1e9:.4f} GB   (trivial)")
