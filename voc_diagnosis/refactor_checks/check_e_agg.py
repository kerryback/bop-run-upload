"""
CHECK E: why is the Gram path not faster at K=1, and can the aggregation be fixed?
Hypothesis: S_ind.T @ E @ S_ind with DENSE S_ind costs O(M^2 N), which dwarfs both the
GEMM O(M^2 K) and the elementwise exp O(M^2). Replace with a segment-sum: O(M^2).
"""
import sys, os, time; sys.path.insert(0, os.getcwd())
import numpy as np
from scipy.sparse import csr_matrix, kron
import config
from utils_bgn import panel_functions_bgn as pf

np.random.seed(999)
N, T = 500, 120
arr = pf.create_arrays(N, T)
r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow = arr

t = T - 2
chisp = csr_matrix(chi[t, :t + 1, :])
col2 = chisp.multiply(sigmaj[:t + 1, :] * corr_zj[:t + 1, :])
live = chi[t, :t + 1, :] > 0
rows, cols = np.nonzero(live.T)          # rows = firm (already sorted ascending)
M = len(rows)
bvals = (sigmaj[:t + 1, :] * corr_zj[:t + 1, :])[cols, rows]
assert np.all(np.diff(rows) >= 0), "rows must be sorted by firm for reduceat"

# firm boundaries in the project list, and which firms are present
present, starts, counts = np.unique(rows, return_index=True, return_counts=True)
S_dense = np.zeros((M, N)); S_dense[np.arange(M), rows] = 1.0
S_sp = csr_matrix((np.ones(M), (np.arange(M), rows)), shape=(M, N))


def B_of(K):
    rng = np.random.default_rng(1)
    B = np.empty((M, K)); B[:, 0] = bvals
    if K > 1:
        B[:, 1:] = rng.standard_normal((M, K - 1)) * 0.3
    return B


def agg_dense(E):
    return S_dense.T @ E @ S_dense


def agg_sparse(E):
    return (S_sp.T @ (S_sp.T @ E.T).T)      # sparse-dense both sides


def agg_reduceat(E):
    blk = np.add.reduceat(np.add.reduceat(E, starts, axis=0), starts, axis=1)
    out = np.zeros((N, N))
    out[np.ix_(present, present)] = blk
    return out


def timeit(f, n=3):
    best = np.inf
    for _ in range(n):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best


print(f"M={M}, N={N}   (M^2={M*M:,.0f},  M^2*N={M*M*N:,.0f})")
print()
print("--- correctness: all three aggregators agree, and match kron at K=1 ---")
B = B_of(1); E = np.exp(B @ B.T)
a_d, a_s, a_r = agg_dense(E), agg_sparse(E), agg_reduceat(E)
rk = kron(col2, col2).copy(); rk.data = np.exp(rk.data)
t4_kron = rk.sum(axis=0).reshape(N, N).A
sc = np.abs(t4_kron).max()
print(f"  dense vs reduceat  {np.abs(a_d-a_r).max()/sc:.2e}")
print(f"  sparse vs reduceat {np.abs(a_s-a_r).max()/sc:.2e}")
print(f"  kron  vs reduceat  {np.abs(t4_kron-a_r).max()/sc:.2e}")
print()
print("--- component timings ---")
for K in (1, 40):
    B = B_of(K)
    tg = timeit(lambda: B @ B.T)
    E = B @ B.T
    te = timeit(lambda: np.exp(E))
    print(f"  K={K:3d}:  GEMM {tg*1e3:7.2f} ms   exp {te*1e3:7.2f} ms")
E = np.exp(B_of(1) @ B_of(1).T)
for nm, f in (("dense S", agg_dense), ("sparse S", agg_sparse), ("reduceat", agg_reduceat)):
    print(f"  aggregate {nm:9s} {timeit(lambda: f(E))*1e3:8.2f} ms")
print()
print("--- END-TO-END term4, best aggregator, vs kron ---")
tk = timeit(lambda: (lambda rr: (rr.sum(axis=0).reshape(N, N).A))(
    (lambda r0: (setattr(r0, 'data', np.exp(r0.data)), r0)[1])(kron(col2, col2).copy())))
print(f"  kron (K=1 only)                {tk*1e3:8.2f} ms")
for K in (1, 5, 20, 40):
    B = B_of(K)
    f = lambda: agg_reduceat(np.exp(B @ B.T))
    tt = timeit(f)
    print(f"  gram+reduceat K={K:3d}            {tt*1e3:8.2f} ms    speedup vs kron(K=1) {tk/tt:5.2f}x"
          f"    implied kron(K={K}) {tk*K*1e3:8.0f} ms")
