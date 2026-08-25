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

## Provenance

`journal.jsonl` (574 KB) is the raw workflow transcript: one `{"type":"result",...}` line
per completed agent with its full return value. `agent_output/` is derived from it.

