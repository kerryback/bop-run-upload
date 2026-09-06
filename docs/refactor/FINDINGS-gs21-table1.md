# GS21 parameters, settled against the paper

Source: Gomes & Schmid, *Equilibrium Asset Pricing with Leverage and Default*, **Journal of
Finance LXXVI(2), April 2021, Table I** — read 2026-09-06 from Seth's local copy. The paper
states plainly: **"The model is calibrated at a quarterly frequency."** This repo runs the
model **monthly**, so every per-period quantity needs converting and every dimensionless one
does not. That distinction is where both trees went wrong, in opposite directions.

## The verdict: neither tree was authoritative

| parameter | paper (quarterly) | correct monthly | `config.py` | `GS21.m` | right |
|---|---|---|---|---|---|
| β discount | 0.994 | `0.994^(1/3)` | ✅ 0.997996 | — | config |
| ψ EIS | 2 | 2 | ✅ | — | both |
| γ risk aversion | 10 | 10 | ✅ | — | both |
| g growth options | 1.14 | 1.14 | ✅ | ✅ | both |
| α entrant size | 0.2 | 0.2 | ✅ | ✅ | both |
| **δ maintenance** | **0.02** | **0.02/3** | ❌ 0.02 | ✅ 0.02/3 | **GS21.m** |
| **ρx persistence** | **0.95** | **0.95^(1/3)** | ❌ 0.96^(1/3) | ✅ 0.95^(1/3) | **GS21.m** |
| **σx volatility** | 0.012 | via ρx = 0.95 | ❌ (downstream of ρx) | ✅ | **GS21.m** |
| ρz persistence | 0.90 | `0.90^(1/3)` | ✅ 0.965489 | — | config |
| σz volatility | 0.16 | conversion | ✅ | ✅ | both |
| χ default cost | 1 | 1 | ✅ | ✅ | both |
| **τ tax rate** | 0.2 | **0.2** (a rate on profit) | ✅ 0.2 | ❌ 0.2/3 | **config** |
| φ bankruptcy cost | 0.4 | 0.4 | ✅ | ✅ | both |
| **κe equity issuance** | **0.025** | 0.025 | ❌ 0 | ❌ 0 | **neither** |
| κb bond issuance | 0.004 | 0.004 | ✅ | ✅ | both |
| ζ refi probability | 0.03 | 0.01 | ✅ | ✅ | both |

**`config.py` is wrong on three; `GS21.m` is wrong on two; they overlap on none.**

## What this overturns

1. **`rho_x` is 0.95, not 0.96.** `config.py:313`'s comment — *"GS21 Table 1: rho_x = 0.96
   quarterly"* — is simply false; Table I says 0.95, and the body text repeats it at
   `"ρx = 0.95 and σx = 0.012"`. The ASU session inferred 0.96 from config.py's track record
   against `GS21.m`, and the primary session accepted the reasoning while flagging that
   nobody had opened Table I. **The induction gave the wrong answer.** This is the single
   clearest argument in this whole effort for checking the source rather than the strongest
   available proxy.

2. **`delta` should be divided by 3.** The paper: *"maintenance of the existing capital stock
   entails periodic costs δk_jt, akin to depreciation"* and *"δ to 2% per quarter, consistent
   with standard estimates of capital depreciation rates."* It is a **per-period flow cost
   proportional to capital**, so a monthly model needs `0.02/3`. `config.py:311`'s
   justification — *"enters only as maintenance relative to per-period output, which is not
   rescaled monthly"* — is wrong on both counts: it scales **capital**, not output, and it is
   explicitly periodic. In `gs_solve_reg.py:130` it is subtracted from output-per-unit-capital
   each period, which confirms it.

   The primary session endorsed the opposite conclusion on 2026-09-05 (WORKING.md §20's
   evaluation), calling it "corroborated" because two independent readings agreed. Two
   readings of the *code* agreed; neither had read the paper. **Corroboration between
   proxies is not verification.**

3. **`tau` should NOT be divided.** Table I calls it the *effective corporate tax rate* — a
   rate applied to a profit flow, so it is invariant to period length. `config.py` was right
   here and `GS21.m` wrong. This is why the two cannot simply be ranked: the same commit
   `6c65bf4` that fixed `tau` broke `delta`.

4. **`kappa_e` is 0.025 in the benchmark, and both trees use 0.** `GS21.m:33` reads
   `kappa_e = 0; %0.025` — so here the **commented-out value is the paper's**, the reverse of
   the `sigma_m` and `r` cases. Note this defeats the earlier claim (WORKING.md §20) that
   "config.py transcribed all three commented-out alternatives correctly": it matched
   `GS21.m` on all three, but `GS21.m` itself departs from the paper on this one.

   The paper's κe = 0 appears only as an expositional limit case for deriving the investment
   cutoff, not as a calibration. It does say these costs *"only marginally improve our
   quantitative results and are not really essential"*, so running κe = 0 is defensible — but
   it is a **deliberate deviation that must be recorded as one**, not inherited silently.

## Immediate consequence

**Do not submit `run_gs_bx7_slurm.sh` as it stands.** The five solve_ids the ASU session
computed encode `rho_x = 0.96^(1/3)`, so all five tasks would spend ~3.4 h each solving an
economy the paper does not describe.

Also invalidated: the `utils_gs21` solfiles regenerated earlier today (commit `8abbdb2`) used
`GS21_DELTA = 0.02` and `GS21_RHO_X = 0.96^(1/3)`. They need regenerating again once
`config.py` is corrected — ~7 min, and cheap relative to catching it after the cluster run.

## The methodological point

Three sessions' worth of reasoning converged on 0.96 for `rho_x` from track record,
consistency, and code inspection. All of it was wrong, and one `grep` of Table I settled it
in under a minute. `GS21.m` and `config.py` are both *derived artifacts*; the paper is the
source. Every GS21 parameter should now be cited to Table I, not to either tree.

---

# Postscript: KP14 and BGN checked against their papers too (2026-09-06)

Seth supplied Kogan-Papanikolaou (JF 2014) and Berk-Green-Naik (JF 1999). Having found
`config.py` wrong on three GS21 parameters, the obvious question was whether the other two
models escaped. **They did.**

## KP14 — Table II: 17 of 18 match exactly

`mu_x`, `sigma_x`, `mu_z`, `sigma_z`, `theta_eps`, `sigma_eps`, `theta_u`, `sigma_u`,
`delta`, `mu_lambda`, `sigma_lambda`, `mu_H`, `mu_L`, `lambda_H`, `gamma_x`, `gamma_z`,
`alpha` — all exact.

The single difference is **`r`: paper 0.025, repo 0.05.** It is a deliberate departure,
flagged at `variants/kp_vy/parameters_kp14.py:14` ("NOTE: r is different from KP14") but
**not** in `config.py` until now. Documented there as well.

## The lambda regime-label fix is confirmed by the paper

The 2026-09-04 fix (WORKING.md, `docs/kp14_regime_labels.md`) was derived from three
in-code sites plus a feasibility argument, with no access to the paper. **The paper
confirms it three separate ways.**

1. **Equation (7), p.681** is literally the formula in `parameters_kp14.py`:
   `1 = lambda_L + [mu_H/(mu_H + mu_L)] * (lambda_H - lambda_L)`
   The weight on `(lambda_H - lambda_L)` is `mu_H/(mu_H+mu_L)` — i.e. **P(high) = 0.3191**,
   which is exactly what the fix asserts and the opposite of what the repo had.

2. **Equation (6), p.680**, with the text: *"mu_H dt and mu_L dt denote the instantaneous
   probability of ENTERING each state."* Entering rates, so exit-from-high = `mu_L` and
   exit-from-low = `mu_H`, giving a stationary `P(high) = mu_H/(mu_H+mu_L) = 0.3191`. That
   is the `KP14_EXIT_H = KP14_MU_L` swap, independently confirmed.

3. **The paper's own sanity sentence, p.689:** *"the firm grows at about twice the average
   rate in its high growth phase and at about a third of the average rate in the low growth
   phase."* Under the fix `lambda_L = 0.3672` — about a third. Under the old convention
   `lambda_L = -1.8800`, a **negative arrival rate**, which no descriptive sentence could
   have meant.

So the economy has been running at `E[lambda] = 1.7172` against the paper's explicit
normalisation of 1.0, and the fix restores it to exactly 1.0000.

## BGN — Table I: 11 of 11 match

`pi = 0.99`, `rbar = 0.006236` (Table I's rounding of `0.07483/12`), `kappa = 0.95`,
`sigma_r = 0.002`, `beta_zr = -0.00014`, `sigma_z = 0.4`, `Cbar = -3.7`, `I = 1`.

Two are computed rather than stored, and both are right:

- **`sigma = 0.3|Cbar|`.** `panel_functions_bgn.py:59` draws `sigmaj` uniform on
  `[|beta|/sigma_z, |beta|/sigma_z + 0.3|Cbar|]`, matching p.1574 exactly. Commit
  `6c65bf4` fixed a **10x error** here (an extra `0.1` factor made the range 0.111 instead
  of 1.11).
- **The two acceptance probabilities.** `vasicek.py:73` *solves* for `beta_star` and
  `scale` such that `Pr(accept | r=0) = 0.10` and `Pr(accept | r=rbar) = 0.05` — Table I's
  bottom panel — rather than hardcoding them. The translated-exponential density matches
  equation (48).

The `bgn_gam` variant agrees with `config.py` on all eight stored values.

## Scoreboard

| model | vs paper | notes |
|---|---|---|
| **KP14** | 17/18 | `r` = 0.05 vs 0.025, deliberate, now documented in both places |
| **BGN** | 11/11 | clean; `6c65bf4` had already fixed a 10x `sigmaj` error |
| **GS21** | 12/16 | `rho_x`, `delta`, `sigma_x`, `kappa_e` wrong — all corrected 2026-09-06 |

GS21 was the outlier, and the reason is visible: KP14 and BGN were transcribed from the
papers, while GS21 came through `GS21.m`, an intermediate artifact that was itself wrong in
several places. **Provenance through a derived artifact is where the errors entered.**
