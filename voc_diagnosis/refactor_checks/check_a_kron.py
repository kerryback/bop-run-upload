"""
CHECK A (T0 + T0b): pure-algebra verification of the proposed refactor.
No model dependencies. Answers three questions:
  A1. Does kron(A,B).sum(0).reshape(N,N) equal outer(colsum A, colsum B)?  (terms 3 and 5)
  A2. Does .multiply() retain EXPLICIT ZEROS?  (would make exp(0)=1 leak into term4)
  A3. Do three independent implementations of term4 agree?
      (i) existing kron+exp+sum   (ii) proposed Gram   (iii) brute-force triple loop
"""
import numpy as np
from scipy.sparse import csr_matrix, kron

rng = np.random.default_rng(0)
S, N, K = 8, 5, 1          # S project slots, N firms


def make_chi(seed):
    """Ragged sparsity on purpose: some firms many live projects, some one, some ZERO."""
    r = np.random.default_rng(seed)
    chi = (r.random((S, N)) < 0.45).astype(float)
    chi[:, 0] = 0.0                      # firm 0: no live projects
    chi[:, 1] = 0.0; chi[3, 1] = 1.0     # firm 1: exactly one
    chi[:, 2] = 1.0                      # firm 2: all slots live
    return chi


print("=" * 78)
print("A1. kron(A,B).sum(0).reshape(N,N)  ==  outer(colsum A, colsum B) ?")
print("=" * 78)
for seed in range(4):
    chi = make_chi(seed)
    chisp = csr_matrix(chi)
    v1 = rng.standard_normal((S, N)); v2 = rng.standard_normal((S, N))
    A = chisp.multiply(v1); B = chisp.multiply(v2)
    kron_way = kron(A, B).sum(axis=0).reshape(N, N).A
    outer_way = np.outer(np.asarray(A.sum(0)).ravel(), np.asarray(B.sum(0)).ravel())
    same_AA = kron(A, A).sum(axis=0).reshape(N, N).A
    outer_AA = np.outer(np.asarray(A.sum(0)).ravel(), np.asarray(A.sum(0)).ravel())
    print(f"  seed {seed}:  max|kron-outer| A,B = {np.abs(kron_way-outer_way).max():.3e}"
          f"   A,A = {np.abs(same_AA-outer_AA).max():.3e}"
          f"   symmetric={np.allclose(same_AA, same_AA.T)}")
print("  (also confirms reshape() C-order matches outer(), i.e. no transpose bug)")

print()
print("=" * 78)
print("A2. EXPLICIT-ZERO PROBE: does .multiply() store zeros?")
print("=" * 78)
chi = make_chi(0)
chisp = csr_matrix(chi)
print(f"  csr_matrix(chi).nnz                     = {chisp.nnz}   (chi.sum()={int(chi.sum())})")
print(f"  stored zeros in chisp                   = {(chisp.data == 0).sum()}")
vals = rng.standard_normal((S, N))
prod = chisp.multiply(vals)
print(f"  chisp.multiply(dense).nnz               = {prod.nnz}")
print(f"  stored zeros in that                    = {(prod.data == 0).sum()}")
vals_with_zero = vals.copy(); vals_with_zero[0, 2] = 0.0     # a genuine zero VALUE on a live slot
prod2 = chisp.multiply(vals_with_zero)
print(f"  with one true 0.0 value on a live slot: nnz={prod2.nnz}, stored zeros={(prod2.data==0).sum()}")
print("  => if stored zeros > 0, kron+exp turns them into exp(0)=1 and term4 is inflated.")

print()
print("=" * 78)
print("A3. THREE-WAY AGREEMENT ON term4 = sum_{s in i} sum_{s' in j} exp(b_s . b_s')")
print("=" * 78)


def term4_kron(col2):
    """(i) exactly what sdf_compute_bgn.py does today (K=1 only)."""
    r = kron(col2, col2).copy()
    r.data = np.exp(r.data)
    return r.sum(axis=0).reshape(N, N).A


def term4_gram(B_M, S_ind):
    """(ii) proposed: Gram -> elementwise exp -> aggregate. Any K."""
    G = B_M @ B_M.T
    E = np.exp(G)
    return S_ind.T @ E @ S_ind


def term4_brute(chi, B):
    """(iii) intended math, written independently. B is (S,N,K)."""
    out = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            tot = 0.0
            for s in range(chi.shape[0]):
                if chi[s, i] == 0:
                    continue
                for sp in range(chi.shape[0]):
                    if chi[sp, j] == 0:
                        continue
                    tot += np.exp(float(B[s, i] @ B[sp, j]))
            out[i, j] = tot
    return out


def stack_projects(chi, B):
    """Build B_M (M,K) and S_ind (M,N) over LIVE projects only."""
    rows, cols = np.nonzero(chi.T)          # rows=firm, cols=slot
    M = len(rows)
    B_M = np.array([B[c, r] for r, c in zip(rows, cols)])
    S_ind = np.zeros((M, N))
    S_ind[np.arange(M), rows] = 1.0
    return B_M, S_ind


for K in (1, 3, 7):
    print(f"\n  --- K = {K} ---")
    for seed in range(3):
        chi = make_chi(seed)
        r = np.random.default_rng(100 + seed)
        B = r.standard_normal((S, N, K)) * 0.7        # loading vectors
        B *= chi[:, :, None]                          # dead slots carry no loading
        B_M, S_ind = stack_projects(chi, B)
        g = term4_gram(B_M, S_ind)
        b = term4_brute(chi, B)
        line = f"    seed {seed}: M={B_M.shape[0]:3d}  max|gram-brute|={np.abs(g-b).max():.3e}"
        if K == 1:
            col2 = csr_matrix(chi).multiply(B[:, :, 0])
            k_ = term4_kron(col2)
            line += f"   max|kron-brute|={np.abs(k_-b).max():.3e}"
            line += f"   max|kron-gram|={np.abs(k_-g).max():.3e}"
        print(line)
