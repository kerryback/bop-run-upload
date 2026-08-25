# Refactor pre-checks — results

Run 2026-08-25 on this machine (`python` = conda `anaconda_env_20260224`, scipy 1.17, numpy 2.4).
Scripts: `check_a_kron.py`, `check_b_bgn.py`, `check_c_M.py`, `check_d_terms35.py`, `check_e_agg.py`.
All are read-only w.r.t. the models — nothing in `utils_*/` was modified.

## A. Algebra (synthetic, ragged sparsity incl. zero-project firms)

- **A1 kron ordering.** `kron(A,B).sum(0).reshape(N,N) == outer(colsum A, colsum B)` to 7e-15
  across 4 seeds, symmetric for A==B. **No transpose bug**; `reshape` C-order matches `outer`.
- **A2 explicit-zero probe.** `csr_matrix(dense)` stores no zeros, and `.multiply(dense)`
  introduces none — *unless* the multiplier has a true `0.0` on a live slot, in which case it
  **does** store it (verified: 1 stored zero). `kron`+`exp` would turn that into exp(0)=1.
  Mechanism is real; see B1 for whether it fires.
- **A3 three-way agreement on term4.** kron vs Gram vs brute-force triple loop:
  at K=1 kron==brute **exactly** (0.0e+00) and Gram==brute to 2.8e-14; at K=3 and K=7
  Gram==brute to 5e-12. The existing code implements the intended math exactly.

## B. Real BGN slices (N=200, T=80)

- **B1 explicit zeros: 0** in `col2` and `col1` at t = 20/40/60/78. beta range
  [-0.860, +0.772], P(beta<0) = 0.33%, zero exact zeros. **SAFE** — but add a cheap assertion,
  because the mechanism in A2 is real and a future calibration change could trip it.
- **B2 series expansion is DEAD.** b = beta/sigma_z on live projects: mean +0.60, sd 0.38,
  range [-2.15, +1.27] at t=78. The exponent b_i.b_j has mean +0.36, p99 +1.07,
  **max +4.62 (exp = 101.5)**. Taylor truncation max relative error:

  | order | t=40 | t=78 |
  |---|---|---|
  | 2 | 5.3e-01 | 8.4e-01 |
  | 3 | 3.0e-01 | 6.8e-01 |
  | 4 | 1.5e-01 | 4.9e-01 |
  | 6 | 2.3e-02 | 1.8e-01 |

  Keep elementwise `exp` on a materialised Gram. No low-rank trick.
- **B4 kron vs Gram on real slices, K=1:** relative difference 1.4e-16 to 3.9e-16 —
  **machine precision.**

## C. Gram size in production (N=1000, arrival/survival simulated only)

| model | live projects/firm | peak M | peak Gram |
|---|---|---|---|
| BGN  | 4.3–8.7 (r-path dependent; peaks ~t=400) | 8,671 | **0.60 GB** |
| KP14 | 6.9 steady state (max 45 for one firm)   | 7,063 | **0.40 GB** |
| GS21 | no project structure, M = N              | 1,000 | 0.008 GB |

**No blocking needed anywhere.** Caveat: BGN's moments step runs at `n_jobs: 1`; at
`n_jobs: 16` you would need 16 x 0.6 GB of Grams simultaneously, so either keep it serial
or block. Size for the *worst* month, not the mean — BGN's M swings 2x with the interest-rate path.

Bonus finding, independently confirming the audit: KP14's simulated stationary P(high) is
**0.683**, matching `KP14_PROB_H` = 0.6809, but `KP14_LAMBDA_L` is normalised assuming
P(high) = MU_H/(MU_H+MU_L) = 0.3191. Measured E[rate]/lambda_f = **1.710** against an
intended 1.0 — the audit's 1.7172 multiplier, reproduced from scratch.

## D/E. Timing, real BGN slice (M=3767, N=500, best of 3)

- **term3 / term5 outer-product identity**: relative error 4e-16 to 8e-16 on real slices.
  `kron` 74.7 ms -> `outer` 0.27 ms = **279x, free, K-independent.**
- **term4 component costs**: GEMM 19.5 ms (K=1) -> 20.8 ms (K=40) — **K is essentially free**.
  `exp` 40–56 ms. Aggregation is the swing factor:

  | aggregation of E into N x N | time |
  |---|---|
  | dense `S.T @ E @ S`  (O(M^2 N)) | 137.9 ms |
  | sparse S             (O(M^2))   | **46.1 ms** |
  | `np.add.reduceat` segment-sum   | 53.0 ms |

  All three agree to 3e-16. **Use sparse S** — as fast as reduceat and it handles
  zero-project firms automatically with no boundary bookkeeping.

- **End-to-end term4:**

  | path | time | vs kron(K=1) |
  |---|---|---|
  | kron (K=1 only) | 138.6 ms | 1.00x |
  | Gram, K=1  | 107.2 ms | **1.29x faster** |
  | Gram, K=5  | 122.3 ms | 1.13x |
  | Gram, K=20 | 129.3 ms | 1.07x |
  | Gram, K=40 | 135.7 ms | 1.02x |

  kron at K=40 would be ~5,542 ms, so the Gram path is **~41x better at K=40**.

## Corrections to the earlier plan

1. "The restructuring speeds up the K=1 case too" — **true only for terms 3/5** (279x).
   For term4 it depends entirely on the aggregator: with a dense S the Gram is *slower*
   than kron (0.87x). With sparse S it is 1.29x faster. Get the aggregator right or the
   refactor is a regression at K=1.
2. "exp(b'b) ~ 12 is too large for a series" — confirmed, and worse than I said (max 101.5).
3. Blocking: not needed. My earlier 288 MB estimate at M=6000 was right in magnitude;
   the true peak is 0.60 GB in BGN because M swings with the interest-rate path.

## Projected production cost

Scaling M^2 terms from M=3767 to BGN's peak M=8671 (5.3x): exp ~210 ms, aggregation ~245 ms,
GEMM at K=40 ~110 ms => **~0.6 s per month, ~3.6 min per panel** single-threaded, against
`estimate_sdf_dkkm` at 3h34m. K is not the bottleneck and never will be.
