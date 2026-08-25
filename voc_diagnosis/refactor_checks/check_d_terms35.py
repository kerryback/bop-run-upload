"""
CHECK D: on REAL BGN slices, verify the term3 / term5 outer-product replacement
and time kron vs outer vs Gram. This is the "free speedup" half of the refactor.
"""
import sys, os, time; sys.path.insert(0, os.getcwd())
import numpy as np
from scipy.sparse import csr_matrix, kron
import config
from utils_bgn import panel_functions_bgn as pf
from config import PI as pi, CBAR as Cbar, SIGMA_Z as sigma_z, I

np.random.seed(999)
N, T = 500, 120
arr = pf.create_arrays(N, T)
r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow = arr
exp_beta = np.exp(-beta)
exp_corr_zr = np.exp(-0.5 * sigmaj**2 * corr_zj**2 * corr_zr**2)

print(f"real BGN slices: N={N}, T={T}")
print()
print("=" * 84)
print("D1. term3 and term5 as outer products of column sums (exact identity)")
print("=" * 84)
for t in [40, 80, T - 2]:
    chisp = csr_matrix(chi[t, :t + 1, :])
    col1 = chisp.multiply(exp_beta[:t + 1, :])
    col3 = chisp.multiply(exp_corr_zr[:t + 1, :])
    fake_integ = chisp.multiply(np.abs(np.random.default_rng(t).standard_normal((t + 1, N))))
    col35 = col3.multiply(fake_integ)                     # stands in for col3*integ_Dexpy

    t3_kron = kron(col1, col1).sum(axis=0).reshape(N, N).A
    s1 = np.asarray(col1.sum(0)).ravel()
    t3_out = np.outer(s1, s1)
    t5_kron = kron(col1, col35).sum(axis=0).reshape(N, N).A
    t5_out = np.outer(s1, np.asarray(col35.sum(0)).ravel())
    print(f"  t={t:3d}  M={chisp.nnz:5d}  term3 rel.err={np.abs(t3_kron-t3_out).max()/np.abs(t3_kron).max():.2e}"
          f"   term5 rel.err={np.abs(t5_kron-t5_out).max()/np.abs(t5_kron).max():.2e}")

print()
print("=" * 84)
print("D2. TIMING at the largest slice  (per-call, best of 3)")
print("=" * 84)
t = T - 2
chisp = csr_matrix(chi[t, :t + 1, :])
col1 = chisp.multiply(exp_beta[:t + 1, :])
col2 = chisp.multiply(sigmaj[:t + 1, :] * corr_zj[:t + 1, :])
live = chi[t, :t + 1, :] > 0
rows, cols = np.nonzero(live.T)
M = len(rows)
S_ind = np.zeros((M, N)); S_ind[np.arange(M), rows] = 1.0
bvals = (sigmaj[:t + 1, :] * corr_zj[:t + 1, :])[cols, rows]


def timeit(f, n=3):
    best = np.inf
    for _ in range(n):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best


def f_t3_kron(): return kron(col1, col1).sum(axis=0).reshape(N, N).A
def f_t3_outer():
    s = np.asarray(col1.sum(0)).ravel(); return np.outer(s, s)
def f_t4_kron():
    rk = kron(col2, col2).copy(); rk.data = np.exp(rk.data)
    return rk.sum(axis=0).reshape(N, N).A


def make_t4_gram(K):
    rng = np.random.default_rng(1)
    B_M = np.empty((M, K)); B_M[:, 0] = bvals
    if K > 1:
        B_M[:, 1:] = rng.standard_normal((M, K - 1)) * 0.3
    def f(): return S_ind.T @ np.exp(B_M @ B_M.T) @ S_ind
    return f


print(f"  M={M}, N={N}")
a = timeit(f_t3_kron); b = timeit(f_t3_outer)
print(f"  term3  kron  {a*1e3:9.2f} ms      outer {b*1e3:9.3f} ms      speedup {a/b:8.0f}x")
c = timeit(f_t4_kron)
print(f"  term4  kron  {c*1e3:9.2f} ms   (K=1 only; K factors would be ~K x this)")
for K in (1, 5, 20, 40):
    d = timeit(make_t4_gram(K))
    print(f"  term4  gram  {d*1e3:9.2f} ms   K={K:3d}      vs kron(K=1): {c/d:6.2f}x"
          f"      implied kron(K={K}): {c*K*1e3:9.0f} ms")
