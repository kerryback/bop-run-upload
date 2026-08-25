"""
CHECK B: run the REAL BGN create_arrays at small scale and answer the open design questions.
  B1. Are there explicit stored zeros in col2 on real data? (the A2 risk, realised or not)
  B2. What is the actual magnitude of b_i.b_j = (beta_i/sigma_z)(beta_j/sigma_z)?
      -> settles whether a low-rank series expansion of exp() is viable
  B3. Distribution of M = live projects per cross-section -> Gram size / blocking
  B4. Does the kron path equal the Gram path on REAL slices at K=1?
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.getcwd())
from scipy.sparse import csr_matrix, kron
import config
from utils_bgn import panel_functions_bgn as pf
from config import SIGMA_Z as sigma_z

np.random.seed(12345)
N, T = 200, 80
arr = pf.create_arrays(N, T)
r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow = arr
print(f"BGN create_arrays: N={N} T={T}   chi{chi.shape}  beta{beta.shape}  sigmaj{sigmaj.shape}")

print()
print("=" * 78)
print("B1. EXPLICIT STORED ZEROS in col2 = chi * (sigmaj*corr_zj), on real slices")
print("=" * 78)
tot_z = 0
for t in [20, 40, 60, T - 2]:
    chisp = csr_matrix(chi[t, :t + 1, :])
    col2 = chisp.multiply(sigmaj[:t + 1, :] * corr_zj[:t + 1, :])
    col1 = chisp.multiply(np.exp(-beta)[:t + 1, :])
    nz = int((col2.data == 0).sum()); tot_z += nz
    print(f"  t={t:3d}  chi nnz={chisp.nnz:5d}  col2 nnz={col2.nnz:5d}  stored zeros={nz}"
          f"   col1 stored zeros={int((col1.data == 0).sum())}")
print(f"  TOTAL stored zeros across probed months: {tot_z}"
      f"   -> {'SAFE' if tot_z == 0 else 'PROBLEM: kron turns these into exp(0)=1'}")
print(f"  beta range [{beta.min():.4f}, {beta.max():.4f}];  P(beta<0) = {(beta<0).mean():.4%}"
      f"   exact zeros in beta = {(beta==0).sum()}")

print()
print("=" * 78)
print("B2. MAGNITUDE OF THE EXPONENT  b_i.b_j  (settles series-expansion viability)")
print("=" * 78)
for t in [40, T - 2]:
    live = chi[t, :t + 1, :] > 0
    b = (sigmaj[:t + 1, :] * corr_zj[:t + 1, :])[live]     # = beta/sigma_z on live projects
    G = np.outer(b, b)
    print(f"  t={t:3d}  M={b.size:5d}  b: mean={b.mean():+.4f} sd={b.std():.4f}"
          f" range[{b.min():+.4f},{b.max():+.4f}]")
    print(f"          b_i.b_j : mean={G.mean():+.4f}  p99={np.percentile(G,99):+.4f}"
          f"  max={G.max():+.4f}   exp(max)={np.exp(G.max()):.3f}")
    for order in (2, 3, 4, 6):
        approx = sum(G**n / math.factorial(n) for n in range(order + 1))
        rel = np.abs(approx - np.exp(G)).max() / np.abs(np.exp(G)).max()
        print(f"          Taylor order {order}: max rel err = {rel:.3e}")

print()
print("=" * 78)
print("B3. M = LIVE PROJECTS PER CROSS-SECTION (Gram is M x M)")
print("=" * 78)
for t in [20, 40, 60, T - 2]:
    live = chi[t, :t + 1, :] > 0
    per_firm = live.sum(axis=0)
    M = int(live.sum())
    print(f"  t={t:3d}  M={M:5d}  per firm: mean={per_firm.mean():.2f} max={per_firm.max()}"
          f"  zero-project firms={int((per_firm==0).sum())}"
          f"  |  scaled to N=1000: M~{int(M/N*1000)}, Gram={((M/N*1000)**2*8)/1e9:.2f} GB")

print()
print("=" * 78)
print("B4. KRON vs GRAM on REAL slices, K=1")
print("=" * 78)
for t in [20, 40, 60, T - 2]:
    chisp = csr_matrix(chi[t, :t + 1, :])
    col2 = chisp.multiply(sigmaj[:t + 1, :] * corr_zj[:t + 1, :])
    rk = kron(col2, col2).copy(); rk.data = np.exp(rk.data)
    t4_kron = rk.sum(axis=0).reshape(N, N).A
    live = chi[t, :t + 1, :] > 0
    rows, cols = np.nonzero(live.T)                       # rows=firm, cols=slot
    B_M = (sigmaj[:t + 1, :] * corr_zj[:t + 1, :])[cols, rows].reshape(-1, 1)
    S_ind = np.zeros((len(rows), N)); S_ind[np.arange(len(rows)), rows] = 1.0
    t4_gram = S_ind.T @ np.exp(B_M @ B_M.T) @ S_ind
    den = np.abs(t4_kron).max()
    print(f"  t={t:3d}  max|kron-gram|={np.abs(t4_kron-t4_gram).max():.3e}"
          f"   relative={np.abs(t4_kron-t4_gram).max()/den:.3e}   scale={den:.3e}")
