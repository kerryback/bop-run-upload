# Why DKKM does not beat FF/FM in BGN / KP14 / GS21

Diagnosis run 2026-08-05 / 2026-08-07. Produced by Claude Code: my own numerical work
plus a 22-agent audit (5 diagnosis, 4 proposal sets, 12 adversarial critiques across three
lenses, 1 synthesis).

## Read in this order

1. **`findings.md`** — the numbers. Sections A–H:
   - A. The RFF basis as coded has participation-ratio effective rank **16.5** (L=5) /
     20.7 (L=6) vs ~800 in the real-data DKKM setup. sin features are 99% linear, cos 99.8%
     quadratic. DKKM as implemented = Fama-MacBeth + a quadratic overlay.
   - B. Population-headroom sweeps. Idiosyncratic-variance dispersion is **not** a lever
     (<1%); loading nonlinearity at K=2 is barely a lever (1.27x at an implausible extreme);
     **K, the number of priced aggregate shocks, is the lever** (1.00x at K=1, 1.01x at K=2,
     1.60x at K=10, 2.26x at K=20).
   - C. Algebra of the HJD metric.
   - D. The actual Sharpe/HJD numbers from 16 panels per model. **The complexity curve is
     flat** — 100x more features buys ≤0.5%, and GS21 peaks at 36 features and declines.
   - E. R² of the true SDF weight on linear / quadratic / wide-RFF bases. GS21's w* is
     90% linear; KP14 77%; only BGN has real nonlinear structure (35% → 73%).
   - F. Scale and tracking decomposition. DKKM tracks the true SDF best everywhere
     (corr +10–17% over FF), but modestly.
   - G. Reconciliation with the agent audit — including the two places my own initial
     hypotheses were **wrong** (the D⁻¹ tilt; widening GAMMA_GRID).
   - H. Which scripts are present.

2. **`agent_output/SYNTHESIS.json`** — the decision-ready program: root cause, an ordered
   10-step plan, per-model recommendations, the rejected list with killing arguments, and
   10 cheap diagnostics (D1–D10) runnable on pickles that already exist.

3. **`agent_output/diag_*.json`** — per-model and pipeline/theory diagnoses. These ran live
   simulations of the actual code, not code reading alone.

4. **`agent_output/prop_*.json`** and **`crit_*.json`** — the proposals and the three
   adversarial critique lenses (`structure` = economic-structure integrity, `mechanism` =
   would it actually produce the gap, `referee` = would it survive review).

## Headline

All three models are frictionless exact K-factor economies with K = 1 (GS21) or 2 (BGN,
KP14). Under exact K-factor pricing with N=1000, the maximum conditional Sharpe is attained
to within ~0.2% by any well-diversified portfolio with the right factor exposure, so six
FF/FM portfolios over-span the SDF. Measured population ceilings on DKKM-over-FM: **+0.7pp
in BGN, +3.0% in GS21.** No estimator change can exceed that.

Three correctness problems block interpretation of anything else:
- KP14's `integ_results.npz` returns exact zeros where 98% of firm-months live, so its
  "true" rp is −PVGO/P — wrong sign, sd 173x too large. Withdraw all KP14 numbers.
- GS21's solfiles solve a different economy than `config.py` simulates (ρ exponent 2/3 vs
  3/2, r, ζ, tax base), with silent off-grid extrapolation. Zero defaults in 2.16M firm-months.
- Three undisclosed 10x idiosyncratic-volatility cuts across the three models.

## Scripts

`scripts/` holds the four load-bearing analyses (`headroom3.py` is the K sweep — the key
experiment; `wstar2.py` is verified to reproduce exactly), plus `extract2.py`, which
regenerates `agent_output/` from `journal.jsonl`, and `workflow_definition.js`, the
workflow that produced the audit.

Paths in `surface.py` and `wstar2.py` point at
`~/ASU Dropbox/Seth Pruitt/BGN and Kelly Malamud/` — adjust on another machine.

## Dead ends

Recording these so nobody re-derives them. Each was implemented, verified correct, and
abandoned on measured grounds.

### Raising K by adding aggregate factors to BGN's cash-flow shock (abandoned 2026-08-25)

**The idea.** BGN's SDF is `log M = -r - 0.5|lambda|^2 - lambda'nu` and a project's price
depends on its loading `b_s` ONLY through the scalar `lambda'b_s`. So write
`b_s = (beta_s/sigma_z)*u + g_s` with `u'g_s = 0`, take the `|g_s|^2` budget out of the
idiosyncratic remainder, and every price, expected return, investment decision, `Jstar`,
`D(r)` and the `(beta_star, scale)` calibration is *exactly* unchanged while the systematic
covariance rank rises. `findings.md` section B3 predicted this should push
SR_lin/SR* down like sqrt(L/K) and open a 2-3x DKKM-over-FM gap.

**It was built and it was correct.** Panel bit-identical at K=0 (no extra RNG draws
consumed); `beta`, `chi`, `P` bit-identical and `max|rp - rp_0| = 0.000e+00` at K = 5, 20, 40,
confirming the null-space pinning. Implementation preserved at
`dead_end_K/K_factors.patch`.

**It does not move the objective.** SR_lin/SR* — the fraction of the maximum conditional
Sharpe attainable in the linear characteristic span, which *is* Fama-MacBeth's feasible set
(X = [1, log mve, bm]):

| sigmaj band | K=0 | K=5 | K=20 | K=40 |
|---|---|---|---|---|
| 0.030 (shipped) | 0.9203 | 0.9199 | 0.9201 | **0.9202** |
| 0.300 | 0.9228 | 0.8739 | 0.8955 | 0.9030 |
| 0.375 (critics' cap) | 0.9107 | **0.8255** | 0.8631 | 0.8761 |

At the shipped calibration K does *nothing* — 0.9203 to 0.9202. (This independently
reproduces the audit's 91.7% figure for the {1, size, bm} span.) Best case anywhere in the
grid is 0.826, i.e. **a ceiling of DKKM/FM ~ 1.21x** against a ~2.5x target. Participation-
ratio effective rank of `cond_var` moves 4.01 -> 4.03 at the shipped band, 7.50 -> 8.97 at
band 0.375.

**It is non-monotonic in K** — K=5 beats K=40. With a fixed variance budget `omega` and a
k^(-decay) mode spectrum, more modes means less variance each. **omega, not K, is binding.**

**The reason, and it is not what I first guessed.** My hypothesis was that the extra factors
being *unpriced* was the problem. `refactor_checks/check_i_priced.py` refutes that: at K=20,
unpriced extras give SR_lin/SR* = 0.319 and fully-priced extras 0.248 — both roughly
tracking sqrt(L/K). The construction is sound; the exact-invariance version is not what
holds it back.

The binding constraint is **where the variance is**. In the synthetic control the common
block is ~86% of return variance. In BGN it is capped at `omega*(1-corr_zj^2)` of *cash-flow*
variance, and cash flows reach returns only through the dividend, with `Chat = exp(-3.7) =
0.0247`. So it can never exceed a few percent of return variance. Meanwhile eig1 — 36-50% of
BGN's return variance — is the interest-rate/discount channel.

**Implication if anyone revives this.** Inject into the discount-rate channel, not the
dividend channel: multiple aggregate discount/term-structure factors loading through project
duration. That is the audit's "heterogeneous project durability" proposal, rejected on cost
(per-type `Jstar.csv` re-solves). This experiment shows the cheap cash-flow route cannot
substitute for it, so that cost/benefit deserves a second look.

**What was retained.** Only the speed-up found alongside it, in
`utils_bgn/sdf_compute_bgn.py`: `term3` and `term5` are rank-1 outer products of column
sums (the `kron` they used was materialising nnz^2 entries to build an N x N rank-1 matrix,
279x slower), and `term4` uses a Gram matrix with sparse aggregation instead of
`kron`+`exp`+`sum`. **~3.5x faster `sdf_loop`, numerically equivalent.** No K machinery
remains in the model code. See `refactor_checks/RESULTS.md`.

## Provenance

`journal.jsonl` (574 KB) is the raw workflow transcript: one `{"type":"result",...}` line
per completed agent with its full return value. `agent_output/` is derived from it.

