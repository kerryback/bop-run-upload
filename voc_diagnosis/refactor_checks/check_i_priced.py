"""
CHECK I: is the weak result in check_h because the extra factors are UNPRICED?
Isolates the mechanism synthetically, holding TOTAL systematic variance fixed:
  (a) PRICED:   lambda spread over all K factors  -> the premium cross-section is K-dim
  (b) UNPRICED: lambda on factor 1 only           -> the premium cross-section stays 1-dim
Same B, same Sigma, same SR* target. Only where lambda lives differs.
"""
import numpy as np
N, L = 1000, 3


def sr_span(X, Sig, rp):
    M = X.T @ Sig @ X + 1e-13 * np.eye(X.shape[1])
    v = X.T @ rp
    return float(np.sqrt(max(v @ np.linalg.solve(M, v), 0.0)))


def experiment(K, priced, seed=0):
    r = np.random.default_rng(seed)
    x = r.standard_normal((N, L - 1))
    X = np.column_stack([np.ones(N), x])
    # loadings: smooth Fourier modes of an observable coordinate tau (as in the BGN patch)
    tau = r.uniform(size=N)
    k = np.arange(1, K + 1)
    B = np.sqrt(2) * np.cos(2 * np.pi * np.outer(tau, k) + r.uniform(0, 2 * np.pi, K))
    B /= np.sqrt(K)                       # hold TOTAL systematic variance fixed in K
    sig_f = 0.20
    if priced:
        lam = r.standard_normal(K); lam = lam / np.linalg.norm(lam) * (0.30 * sig_f)
    else:
        lam = np.zeros(K); lam[0] = 0.30 * sig_f
    Om = np.eye(K) * sig_f**2
    d = (0.08**2) * np.ones(N)
    Sig = B @ Om @ B.T + np.diag(d)
    rp = B @ lam
    s_star = float(np.sqrt(rp @ np.linalg.solve(Sig, rp)))
    return sr_span(X, Sig, rp) / s_star, s_star


print(f"{'K':>4} | {'UNPRICED extras':>16} {'SR*':>8} | {'ALL PRICED':>12} {'SR*':>8} | {'sqrt(L/K)':>10}")
print("-" * 74)
for K in (1, 2, 3, 5, 10, 20, 40):
    u = np.mean([experiment(K, False, s)[0] for s in range(3)])
    us = np.mean([experiment(K, False, s)[1] for s in range(3)])
    p = np.mean([experiment(K, True, s)[0] for s in range(3)])
    ps = np.mean([experiment(K, True, s)[1] for s in range(3)])
    print(f"{K:>4} | {u:16.4f} {us:8.4f} | {p:12.4f} {ps:8.4f} | {min(1.0, (L/K)**0.5):10.3f}")
print("\nColumns are SR_lin/SR*. Lower = more room for a nonlinear/high-dim estimator.")
print("Implied best-case DKKM/FM = 1 / (SR_lin/SR*).")
