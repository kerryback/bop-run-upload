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

---

# PART 2 — the refactor, as built and verified (2026-08-25)

## What changed

`utils_bgn/sdf_compute_bgn.py`
- `term3`, `term5`: `kron` -> `np.outer` of column sums (exact identity, proven in A1/D1).
- `term4`: new module-level `_term4_gram(col2, N, extra=None)` — Gram -> elementwise `exp`
  -> **sparse-S** aggregation. K enters only the GEMM's inner dimension.
- `diag4_sub` now reads a precomputed `bsq` = |b_s|^2 so the diagonal-correction site is
  K-agnostic.
- Added `assert col2.nnz == chisp.nnz` (the meaningful invariant — a dropped zero-loading
  project would vanish from term4 while still counting in term3 and the diagonal).

`utils_bgn/panel_functions_bgn.py`
- Extra-factor loadings `gload` (T-1, N, K-1): Fourier modes of an observable firm
  coordinate `tau`, amplitudes k^(-decay/2), scaled so |g_s|^2 = omega*sigmaj^2*(1-corr_zj^2)
  exactly, with the idiosyncratic remainder reduced to match. All draws **guarded on
  n_extra > 0 and placed after every pre-existing draw**, so K=0 consumes no RNG.
- `gload` appended LAST to `arr_tuple`; `create_panel` and `sdf_compute` accept either
  length, so existing 13-array `_arr/` directories still load.
- The hand-inserted `0.1*0.3` band is now `config.BGN_SIGMAJ_BAND` (default identical).

`config.py`: `BGN_N_EXTRA_FACTORS`, `BGN_EXTRA_OMEGA`, `BGN_EXTRA_DECAY`, `BGN_SIGMAJ_BAND`.

## Verification (all in `voc_diagnosis/refactor_checks/`)

| test | result |
|---|---|
| `golden.py` T1 at K=0 | **PASS**. `rp` bit-identical; `cond_var` 1.1e-13 rel; `max_sr` 3e-14; `sdf_ret` 6e-12 (amplified through `scipy.linalg.solve`, as predicted) |
| `golden.py` T2 panel bit-identity | **PASS** — beta, sigmaj, corr_zj, chi, eret, ret, P, book, r all bit-identical |
| `check_f_K.py` T5 invariance | **PASS** — beta/chi/P bit-identical and `max abs(rp - rp0) = 0.000e+00` at K = 5, 20, 40 |
| `timing.py` | **3.5x faster** end-to-end `sdf_loop` at K=0 (N=500, T=200, M=3366: 274 ms -> 76 ms) |
| T3 end-to-end | **NOT RUN** — needs a full pipeline pass (`estimate_sdf_dkkm` is ~3.5 h/panel) |

`create_arrays` is reproducible under `np.random.seed`, which is what makes T2 possible.

## THE NEGATIVE RESULT: this injection point is too weak in BGN

`check_h_srlin.py` measures the only thing that matters — SR_lin/SR*, the fraction of the
maximum conditional Sharpe attainable inside the linear characteristic span, which IS
Fama-MacBeth's feasible set. X = [1, log mve, bm].

| sigmaj band | K=0 | K=5 | K=20 | K=40 | SR* (K=0 -> 40) |
|---|---|---|---|---|---|
| 0.030 (shipped) | 0.9203 | 0.9199 | 0.9201 | 0.9202 | 0.2761 -> 0.2759 |
| 0.150 | 0.9218 | 0.9122 | 0.9159 | 0.9175 | 0.2742 -> 0.2728 |
| 0.300 | 0.9228 | 0.8739 | 0.8955 | 0.9030 | 0.2671 -> 0.2627 |
| 0.375 (critics' cap) | 0.9107 | 0.8255 | 0.8631 | 0.8761 | 0.2594 -> 0.2534 |

Three things to read off:

1. **At the shipped band, K does nothing at all** — 0.9203 to 0.9202. (This independently
   reproduces the audit's 91.7% figure for the {1, size, bm} span.)
2. Even at the critics' cap the best case is 0.826, i.e. **implied best-case DKKM/FM ~ 1.21x**,
   against a target of ~2.5x.
3. It is **non-monotonic in K** — K=5 beats K=40. With a fixed variance budget `omega` and a
   k^(-decay) spectrum, more modes means less variance each. **omega, not K, is binding.**

`check_g_spectrum.py` says the same in eigenvalue terms: participation-ratio effective rank
of `cond_var` goes 4.01 -> 4.03 at the shipped band, and only 7.50 -> 8.97 at band 0.375.
The extra factors add ~1.7% of total return variance. eig1 alone is 0.36-0.50.

## Why — and it is NOT that the extras are unpriced

`check_i_priced.py` isolates the mechanism synthetically at fixed total systematic variance:

| K | unpriced extras | all priced | sqrt(L/K) |
|---|---|---|---|
| 1 | 0.936 | 0.936 | 1.000 |
| 5 | 0.537 | 0.486 | 0.775 |
| 20 | **0.319** | **0.248** | 0.387 |
| 40 | 0.236 | 0.226 | 0.274 |

**Unpriced extras work nearly as well as priced ones** (0.319 vs 0.248 at K=20), and both
roughly track sqrt(L/K). So the construction is sound and the exact-invariance version is
NOT the thing holding it back — good news for defensibility.

The binding constraint is **where the variance is**. In the synthetic the common block is
~86% of return variance. In BGN the extra block is capped at omega*(1-corr_zj^2) of
*cash-flow* variance, and cash flows reach returns only through the dividend, with
Chat = exp(-3.7) = 0.0247. So it can never be more than a few percent of return variance.
Meanwhile eig1 — 36-50% of return variance — is the interest-rate/discount channel.

## Recommendation update

- **Keep the refactor.** It is 3.5x faster at K=0, exactly verified, and it makes K a
  one-line config change for whatever injection point turns out to be right.
- **Do not spend cluster time on `BGN_N_EXTRA_FACTORS` at the cash-flow shock.** The
  measured ceiling is DKKM/FM ~ 1.21x and only at a sigmaj band the critics rejected.
- **Move the injection to the discount-rate channel**, where 36-50% of BGN's return
  variance actually lives: multiple aggregate discount/term-structure factors with
  firm-specific loadings via project duration. That is the audit's "heterogeneous project
  durability" idea, which was rejected on cost (per-type `Jstar.csv` re-solves) — the
  cost/benefit now looks different, because this experiment shows the cash-flow route
  cannot substitute for it.
- Re-run `check_h_srlin.py` after any change. It is the whole answer in one table and it
  costs seconds.

---

# PART 3 — rollback (2026-08-25)

The extra-aggregate-factor route was abandoned. Everything except the speed-up was rolled
back; see `../README.md` ("Dead ends") for the reasoning and
`../dead_end_K/K_factors.patch` for the preserved implementation.

**Reverted in full:** `config.py` (all four knobs incl. `BGN_SIGMAJ_BAND` — the literal
`0.1*0.3` and its `# added a factor of 0.1` comment are back) and
`utils_bgn/panel_functions_bgn.py` (loading construction, `arr_tuple` 14th element, and the
eps-block relocation, which existed only to keep the extra draws append-only).

**Retained**, in `utils_bgn/sdf_compute_bgn.py` and nowhere else — 63 insertions, 7 deletions:
- `_term4_gram(col2, N)`: Gram -> elementwise `exp` -> **sparse-S** aggregation.
- `term3`, `term5`: `kron` -> `np.outer` of column sums.
- `assert col2.nnz == chisp.nnz`.

The `bsq` precompute was also dropped: on reflection it was never a real speed-up (the
`exp` over the slice runs either way), it existed only to make the diagonal-correction site
K-agnostic.

## Rollback verification

| gate | result |
|---|---|
| 1. diff scope outside `voc_diagnosis/` | only `utils_bgn/sdf_compute_bgn.py` (+63 / -7) |
| 2. `grep` for K machinery in `config.py`, `utils_bgn/`, `utils/`, `utils_factors/` | none found |
| 3. `golden.py check` against the fixture built by the ORIGINAL code | **PASS**, worst 5.98e-12 (`rp` bit-identical, `cond_var` 1.1e-13) |
| 4. `timing.py` vs the committed pre-refactor baseline | **1.8x - 4.0x**, rising with M; 3.66x at M=3366 |
| 5. `check_b_bgn.py` B1 | 0 stored zeros on real slices — the new assert cannot fire |

`baseline/sdf_compute_bgn_prerefactor.py` is the pre-refactor module (relative import
rewritten, plus its `Jstar.csv`), committed so `timing.py` no longer depends on `/tmp`.

## Still outstanding

**T3 has not been run.** A full-pipeline pass confirming BGN 3600 features / alpha=0.01 ->
0.4172 (findings.md section D) is the acceptance test before this goes near cluster jobs,
even though the change is numerically equivalent at N=300/T=100 and verified against three
independent implementations.

The same free `kron` -> `outer` win exists in KP14's `term1`
(`utils_kp14/sdf_compute_kp14.py`, `result = kron(col, col)` summed over axis 0). Not
applied — KP14's truth is corrupt (see `../README.md`), so there is no baseline to regress
against. GS21 uses quadrature, not `kron`, and is unaffected.
