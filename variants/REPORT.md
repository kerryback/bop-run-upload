# Why DKKM barely beats FMR/FFC in the simulated economies — and what would change that

All numbers are monthly Sharpe ratios (SR) of the estimated SDF-mimicking portfolio, evaluated with the *true*
conditional moments, as in the paper. Code and logs: `gap_experiments/` (see `README.md`). Nothing in `Code/` was modified.

## 1. The three solutions, as implemented

| | BGN (`panel_functions.py`, `sdf_compute.py`) | KP14 (`*_kp14.py`) | GS21 (`*_gs21.py`, `GS21.m`) |
|---|---|---|---|
| SDF | exogenous lognormal, one priced shock ν; Vasicek rate, corr(ν,ξ)=β_zr/(σ_zσ_r)=−0.175 | exogenous, two priced shocks (x: γ_x=0.69, z: γ_z=−0.35); constant prices of risk | exogenous, one shock ε_x, constant γ_x=0.5 (`Mmat` in `GS21.m`) — the GE/Epstein-Zin kernel with countercyclical price of risk is replaced |
| firm state | live projects {β_s, σ_s, χ_s}, r_t | live projects {K_j, u_j}, ε_f, λ_f, H/L regime, x_t, z_t | (k, b, z, η), x_t |
| E_t[R] | BGN eq. 44: **exactly affine in (book/P, 1/P)** with r-dependent coefficients | KP Prop. 3: **affine in one firm variable PVGO/V** | from interpolated VFI price functions + 10-pt Gauss–Hermite |
| true conditional moments | closed-form integrals over ξ (sparse kron over project pairs) | closed-form integrals (the O(t²N) same-firm term was replaced by an O(tN) identity, verified) | GH quadrature over (z′_i, z′_j, x′) |

**GS21 is not usable as an evaluation economy as it stands.** The exogenous SDF `exp(−r−½γ²−γε)` with γ_x=0.5 bounds every
monthly SR at √(e^{0.25}−1)=0.53, yet the saved results have mean "max conditional SR" 2.14, FMR 1.64, and even *realized*
SRs of 1.1–2.0 (`new_results/results_tseries_gs.csv`). `gs/check_gs.py` on a fresh N=200 panel: E_t[R] between 1.7% and 6.5%
per month, idiosyncratic vol 16%/month, and E[M(1+R)]=0.96≠1 — the simulated prices do not satisfy the Euler equation of the
SDF that is used to evaluate them (the r in `parameters_gs21.py` also differs from the r in `GS21.m`). Any DKKM-vs-FMR gap
in GS is an artifact until that is fixed; the rest of this note is about BGN and KP.

## 2. Diagnosis: how much Sharpe ratio is there for a *nonlinear* method to find?

`run_oracle.py` strips out estimation noise. For a feature basis Φ_t (N×P), the best constant-coefficient portfolio
w_t=Φ_tθ is θ*=E[ff′]⁻¹E[f] computed from the true moments (this is the population version of "regress 1 on factor
returns"); it is then scored with the true conditional moments. Bases: FMR (raw chars), linear in rank-standardised chars,
quadratic, decile/pair bins, and RFF with P=36/360/3600 (DKKM's construction). N=300, 345 months, baseline calibrations:

| population ceiling | BGN | KP |
|---|---|---|
| SR_max (all information) | 0.277 | 0.228 |
| best **nonlinear** function of the 5 chars (RFF-3600) | 0.262 | 0.218 |
| best **linear** function of rank-chars | 0.246 | 0.212 |
| ± interest-rate interactions | 0.246 | – |
| FMR basis (raw chars, constant θ) | 0.156 | 0.167 |

With no estimation noise at all, nonlinearity in the five characteristics is worth **+0.016 (BGN)** and **+0.006 (KP)**.
At the paper's N=1000 the BGN numbers are SR_max 0.333, linear 0.251, bins 0.278, RFF-3600 0.303 → **+0.03 to +0.05**
(the alpha part of SR_max grows with √N); the paper's DKKM realises +0.017 of it, the rest is estimation noise.

Why so little? DKKM's theory needs priced information that is *diffuse across many nonlinear directions* of the
characteristics. Here the true SDF weights are Σ⁻¹μ ∝ D⁻¹B·c: expected returns and betas are affine in one or two firm
state variables (BGN: book/P and 1/P; KP: PVGO/V) that the rank-characteristics capture linearly, and the SDF has one or
two priced shocks with constant prices of risk. The true weights *are* nonlinear in characteristics (cross-sectional R² of
Σ⁻¹μ on rank-chars 0.46, on bins 0.90 — `analyze_true_weights.py`), but that nonlinearity sits in directions with almost no
expected return per unit of risk, so it is worth nothing in SR terms. Conditioning on r_t (which RFF receives and FMR
does not) is worth nothing at σ_r=0.002.

Estimator-level confirmation (`run_estimators.py`, N=300, 120-month windows, baseline): FMR 0.163, FFC 0.168,
**OLS on the 6 rank-chars 0.190**, RFF-3600 + ridge 0.194. I.e. the DKKM edge over FMR in the paper is the rank
transformation (FMR's raw-characteristic factor weights are unstable over time — its constant-θ population SR is only
0.156 vs 0.249 if θ could be re-optimised each month), not complexity. KP at this scale is un-estimable for any method
(individual returns with >100%/month outliers; RFF collapses to zero weights) and is judged on the oracle layer only.

## 3. What moves the gap (population nonlinear gain, N=300; baseline +0.016)

| change | SR_max | linear | nonlinear (RFF-3600) | gain | comment |
|---|---|---|---|---|---|
| **idiosyncratic vol decreasing in firm size** (`sigmaj_size_elast=1`, multiplier capped at 2×) | 0.267 | 0.171 | 0.224 | **+0.053** | at N=1000: 0.199 → 0.268, **+0.069** |
| same, elasticity 2 | 0.261 | 0.157 | 0.201 | +0.044 | cap binds |
| same but only *lower* vol for large firms (cap 1) | 0.278 | 0.248 | 0.264 | +0.016 | no effect: the room comes from extreme-vol small firms |
| moderate version (elasticity 0.5, cap 1.5) | 0.273 | 0.236 | 0.254 | +0.018 | ≈ baseline; estimators: FMR 0.135, rank-ridge 0.158, RFF 0.159 |
| priced rate risk β_zr=−0.0004 (corr −0.5) | 0.306 | 0.277 | 0.297 | +0.019 | raises all ceilings ~equally |
| σ_r=0.004 and β_zr=−0.0008 | 0.334 | 0.278 | 0.307 | +0.029 | rates go deeply negative |
| σ_r=0.004 alone | 0.233 | 0.158 | 0.179 | +0.022 | mostly adds systematic variance; SR_max falls |
| idio vol level ×⅓ / ×2 (`sigmaj_width` 0.1 / 0.6) | 0.288 / 0.221 | 0.256 / 0.128 | 0.273 / 0.143 | +0.018 / +0.015 | level of noise is irrelevant to the gain |
| KP: γ_z=−0.7 (r raised to keep ρ>0), σ_ε=0.4, σ_z=0.07, σ_λ=4 | 0.22–0.28 | | | **≤ +0.008** | KP is affine in PVGO/V; nothing in its parameter space helps |

The only lever that materially widens the *room* is making the optimal weight b_i/σ²_i a product of two characteristic
functions — here by letting idiosyncratic cash-flow volatility fall with the number of live projects, so that very small
firms have several times the idiosyncratic variance of large ones (the empirical micro-cap pattern that also drives DKKM's
own results). A linear rule cannot assign ≈0 weight to the small, noisy firms while keeping a positive loading on the
signal; RFF/bins can.

The cost: the same fat-tailed small-firm returns destroy sample second moments. At N=300 with 120-month windows every
estimator collapses on the sel10 panel (FMR 0.074, rank-OLS 0.069, RFF-3600 0.072 vs population 0.171/0.224), so the
extra room is not realised at small scale. The full-scale check (N=1000, 360-month windows, baseline vs sel10) is in
`results/log_est_bgn_*_N1000_T500.txt` (see §5).

## 4. Recommendations

1. **Fix GS before using it.** Check the price functions against the Euler equation (`gs/check_gs.py` prints E[M(1+R)]);
   align `r` between `GS21.m` and `parameters_gs21.py`; GS is the one economy whose SDF (countercyclical price of risk,
   default clustering) could generate genuinely nonlinear cross-sectional premia, but only with the GE kernel, not with a
   constant-γ lognormal stand-in.
2. **Don't expect KP to deliver a gap** under any parameterisation — it is affine in one firm variable.
3. **In BGN, the gap can be widened only by breaking the affine structure of Σ⁻¹μ in the characteristics.** The cheapest
   way is size-dependent idiosyncratic volatility (`sigmaj_size_elast`), which triples the population room — but §5 shows
   that with BGN's lognormal cash flows the DKKM estimator cannot harvest it even at N=1000 / 360 months, because the same
   change fattens already extreme return tails. To make the room *realisable* the tails must be tamed at the same time
   (bound σ_j, or a thinner-tailed cash-flow distribution), or the heterogeneity must come from diversification (large
   firms holding many small projects) rather than from project-level σ. A second, independent lever is a larger priced
   rate-risk premium (β_zr ≈ −0.0004 to −0.0008; rebuilt J* tables are in `bgn/`), which raises SR_max and the gain modestly.
4. **Change the benchmark, not only the economy.** The right linear benchmark for "is complexity useful" is ridge/OLS on
   rank-standardised characteristics (KPS-6); against it the baseline RFF gain is ≈0 at every scale tried. If the paper's
   point is DKKM vs *FMR as practitioners run it*, the existing +0.017 is real but it is a statement about rank-standardisation.
5. **Metric.** The mean of monthly conditional SRs penalises portfolios whose conditional variance varies over time
   (in sel10 realised SRs are 2× the conditional means); report the unconditional SR of the SDF-portfolio return series
   alongside it.

## 5. Full-scale results (N=1000, 360-month rolling windows, 125 evaluation months, one RFF draw)

| | FMR | FFC | ridge on 6 rank-chars (best κ) | RFF-3600 (best κ) | t(RFF−FMR) | population: linear / RFF-3600 |
|---|---|---|---|---|---|---|
| baseline | 0.255 | 0.233 | 0.261 | **0.265** | 10.4 | 0.260 / 0.294 |
| size-dependent idio vol (sel10) | 0.201 | 0.165 | 0.199 | 0.195 | −2.8 | 0.214 / 0.260 |

(`results/bgn_estimators_*_N1000_T500_w360_summary.csv`; realised unconditional SRs are in the same files.)

* Baseline: the gap RFF−FMR is +0.010 (the paper: +0.017 with 5 RFF draws and 10 panels) and RFF−(rank ridge) is +0.004.
  With 360 months and N=1000 the estimators sit close to their population ceilings for the *linear* part (0.261 vs 0.260)
  but harvest only a quarter of the nonlinear room (0.265 vs 0.294).
* sel10: the room for nonlinearity is 2.5× larger (+0.046) but none of it is realised; RFF is even slightly below FMR.
  Reason: BGN's lognormal project cash flows already give monthly excess returns with kurtosis ≈1,300 and a maximum of
  +950% in the baseline panel; sel10 pushes kurtosis to ≈2,400. Rolling 360-month second moments of 3,600 factor returns
  are dominated by these outliers, and ridge shrinkage cannot separate the (now larger) signal from them. The realised
  unconditional SRs tell the same story (RFF 0.238 vs FMR 0.231).

**Bottom line.** In BGN/KP the "no gap" result is not a calibration accident that a parameter change fixes. The economies
place essentially all priced information in one or two affine directions of the characteristics (nothing for complexity
to add), and the one structural change that creates nonlinear room — size-dependent idiosyncratic risk — also fattens the
tails that make the DKKM ridge estimator fail at 360 months. A convincing virtue-of-complexity demonstration in this
framework needs (i) an economy whose optimal weights are nonlinear in *observable* characteristics for reasons other than
extreme idiosyncratic variance (several priced shocks with loadings that depend on different characteristics and their
interactions — e.g. a properly solved GS21 with its countercyclical GE kernel and default, or BGN with priced rate risk
*and* firm types), and (ii) a return distribution with realistic tails (bounding σ_j, or replacing lognormal cash flows
by a thinner-tailed specification), so that sample second moments are informative.

## 6. Follow-up: implementing "anomaly strength decreasing in size" and "thin-tailed, size-dependent idiosyncratic risk"

### 6a. Tails first (why the earlier room was not realised)

BGN's project cash flows are lognormal with σ_j ~ U[|β|/σ_m, |β|/σ_m + 0.3|C̄|], i.e. up to σ_j≈2.4 — monthly excess returns with
kurtosis ≈500–1300. Two ways to thin them while keeping the cross-section of expected returns *exactly* unchanged:

| | return kurtosis | max monthly return | E[μ], cs-sd(μ) | SR_max (N=1000) | comment |
|---|---|---|---|---|---|
| baseline (σ_m=0.4, width 0.3) | 534 | +380% | 0.0092 / 0.00121 | 0.333 | |
| **σ_m = 1.0** | 7 | +90% | 0.0088 / 0.00118 | **0.674** | prices, J*, β-distribution invariant (verified), but the systematic cash-flow covariance scales with 1/σ_m² so SR levels double — changes the economy |
| **width 0.1** (σ_j band 0.37 instead of 1.11) | 6 | +80% | 0.0092 / 0.00121 | 0.339 | **preferred**: same SR level, same premia |

(`check_sigmam.py`; at σ_m=1 the HJ bound is 1.3/month, so that economy is not a sensible calibration.)

### 6b. Full-scale estimator results with thin tails (N=1000, 360-month windows, 125 evaluation months)

| economy | FMR | FFC | rank-ridge (best κ) | RFF-3600 (best κ) | t(RFF−FMR) | ceilings: linear / RFF-3600 |
|---|---|---|---|---|---|---|
| width 0.1 | 0.260 | 0.252 | 0.270 | **0.274** | 6.6 | 0.267 / 0.309 |
| width 0.1 + size-dependent vol (elast. 1, cap 2) | 0.264 | 0.251 | 0.269 | 0.273 | 7.7 | 0.267 / 0.308 |
| σ_m = 1.0 | 0.620 | 0.575 | 0.640 | 0.632* | 13.8 | 0.648 / 0.665 |
| σ_m = 1.0 + size-dependent vol | 0.567 | 0.541 | 0.607 | 0.600* | 20.6 | 0.608 / 0.638 |

(*best κ at the boundary of the grid used in that run.)  Two lessons. (i) With thin tails the *linear* estimator reaches its
population ceiling (0.270 vs 0.267), so tails were indeed what had been blocking it. (ii) RFF-3600 still realises only
~⅙ of its extra room (0.274 of 0.309): with P/T=10 the ridge's implicit shrinkage ("complexity risk" in DKKM's theory)
costs ≈0.035 of SR at this signal-to-noise. (iii) Once the tail is removed, size-dependent idiosyncratic vol via the σ_j
band does nothing — its entire earlier effect was the fat tail, not the cross-sectional vol pattern.

### 6c. Firm types: small firms take dispersed projects, large firms take standard ones (`bgn_types/`)

Each firm has a permanent type τ with its own translated-exponential distribution of project β: scale s_τ (dispersion of
project risk) and upper support β*_τ, the latter backed out from a target acceptance probability p_τ at r̄ — so type sets
both firm size (steady-state number of live projects ≈ p_τ/(1−π): 5 / 15 / 45) and the within-firm dispersion of risk
(0.137 / 0.082 / 0.048). Because the accepted β is threshold − s_τ·Exp(1), its *mean* would also differ across types; a
type-specific C̄_τ = C̄ + (s_τ − s_0) shifts the acceptance threshold so that mean accepted β (hence the average risk premium)
is equal across types, leaving only the dispersion and size differences. Growth-option values J*_τ(r), the value of the
investment option, expected returns, the full N×N conditional second-moment matrix and the Taylor/projection loadings
are all type-indexed (`vasicek_types.py`, `panel_functions_types.py`, `sdf_compute_types.py`, `loadings_compute_types.py`).

Validation: with one type the code reproduces `Code/` to machine precision (`test_types_equivalence.py`); and since firms
do not interact in BGN, every type-k firm in a mixed economy must equal its counterpart in a pure type-k economy with the
same draws — it does, to exactly zero error in prices, expected returns, loadings and the covariance block
(`test_types_mixed.py`, after making all types share one interest-rate grid).

What it does to the cross-section (N=600 check): within-type cross-sectional sd of expected returns 0.0016 / 0.0006 /
0.0002 per month for the small / medium / large type — the "anomaly lives in small stocks" pattern — with no type-level
premium after the C̄ adjustment. The oracle at N=300: SR_max 0.268, linear-in-ranks 0.246, RFF-3600 0.260 → nonlinear
room **+0.014, no larger than baseline (+0.016)**. Making the bm-premium exist only among small firms does not, by itself,
create Sharpe ratio that a linear rule in rank-characteristics misses: the linear rule simply loads bm where it works and
the large firms, having little dispersion, contribute little either way. Full-scale results (N=1000, thin tails, and a
more data-like 60/30/10 configuration with stronger dispersion contrast) are in §6d.

### 6d. Type economies at scale (N=500, T=500, 360-month windows, thin tails; all SRs from true moments)

| economy | SR_max | ceilings: linear / RFF-3600 | FMR | FFC | rank-OLS/ridge | RFF-3600 (best κ) | RFF ens. (5 draws) |
|---|---|---|---|---|---|---|---|
| 3 equal types (5/15/45 projects; s = .137/.082/.048) | 0.294 | 0.267 / 0.284 | 0.230 | 0.201 | **0.252** | 0.204 | 0.202 |
| data-like: 60/30/10 shares, s = .20/.08/.03, 5/20/60 projects | 0.352 | 0.297 / **0.341** | 0.266 | 0.271 | **0.304** | 0.286 | 0.286 |

The data-like configuration is the economy with the most nonlinear room found anywhere in this project (+0.044 at N=500,
i.e. the best nonlinear rule beats the best linear rule by 15%), and it comes from exactly the intended mechanism: the
cross-sectional dispersion of expected returns is concentrated in the 60% small firms, so a linear rule in ranks cannot be
aggressive on bm among small firms and flat among large ones. But the estimator ranking is unchanged: a plain OLS on the six
rank-standardised characteristics (0.304) beats RFF-3600 (0.286, t≈15 either way), both beat FMR/FFC, and RFF realises
about a quarter of its own room (not a ridge-grid artifact: on a fresh panel RFF peaks at κ≈0.003 with 0.285 and falls to
0.106 ridgeless — `results/log_est_bgn_types3v2_w01b.txt`). Ensembling over RFF draws is immaterial (0.2748 vs 0.2747 on the thin-tail baseline
panel; identical here) — at P=3600 the random draw does not matter, the 360-month sample moments do.

On metrics: the realised (sample) Sharpe ratios of the RFF portfolios in these two economies are 0.31–0.39, far above
their conditional means (0.20–0.29), which looks like RFF timing its exposure. It is not: the *exact* unconditional SR
computed from the true conditional moments of each estimated portfolio (`unconditional_sr.py`) agrees with the mean
conditional SR to within 0.005 for every method in every run, while realised SRs over 125 months have a standard error of
≈0.09 (FMR's realised SR on the baseline panel is 0.169 against an exact 0.255). Conclusions drawn from 125-month realised
Sharpe ratios are noise; the paper's true-moment metrics are the right ones, and they do not hide a gap.

### 6e. Where this leaves items 1 and 2

* **Item 2 (thin tails + size-dependent idiosyncratic risk).** Thin tails matter and are essentially free (`sigmaj_width`
  0.1 leaves every expected return unchanged): with them the linear estimators sit on their population ceilings and
  RFF-3600 gains +0.014 over FMR at N=1000 (t≈7). Size-dependent idiosyncratic *variance* adds nothing once the tails are
  thin; its earlier large effect was the fat tail. (Don't use σ_m for this: it scales the systematic cash-flow covariance and
  doubles SR_max.)
* **Item 1 (anomaly strength decreasing in size).** Implemented as permanent firm types with exact tests; it produces the
  intended "anomalies live in small stocks" cross-section and, in the data-like configuration, the largest nonlinear room
  seen (+0.044 at N=500, +15%). It still does not translate into a DKKM-vs-linear gap at 360 months: RFF realises ~¼ of
  the room and ends below a rank-linear OLS.
* **Net.** Across ~25 BGN variants (parameters, tails, priced rate risk, size-dependent vol, firm types) the realised gap
  between DKKM-RFF and FMR with true-moment metrics is +0.01 to +0.02, and between DKKM-RFF and a rank-linear OLS it is
  ≤ +0.005 or negative. The population room for nonlinearity in five characteristics reaches 15% of SR at best, against
  ~100% in DKKM's data; and what room exists is mostly lost to the P/T=10 ridge at T=360. A convincing virtue-of-complexity
  demonstration therefore needs either a different economy (several priced shocks with loadings that are *different*
  nonlinear functions of characteristics — e.g. a properly solved GS21 with default and the GE kernel) or much longer
  estimation samples than 360 months, or both.

## 7. Summary table: what each modification added

All monthly SRs from true conditional moments; estimators use 360-month rolling windows. "Room" = nonlinear ceiling −
linear ceiling (what only a nonlinear method could add with zero estimation noise). Rows differ in N; compare within rows.

| # | modification | scale | SR_max | linear ceil. | RFF-3600 ceil. | room | FMR | rank-OLS | RFF-3600 | RFF−FMR | RFF−rankOLS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline BGN | N=1000 | 0.328 | 0.260 | 0.294 | +0.034 | 0.255 | 0.261 | 0.265 | +0.010 | +0.004 |
| 1 | thin tails (width 0.3→0.1) | N=1000 | 0.339 | 0.267 | 0.309 | +0.042 | 0.260 | 0.270 | 0.274 | +0.014 | +0.004 |
| 2 | size-dep. idio vol, fat tails | N=1000 | 0.319 | 0.214 | 0.260 | +0.046 | 0.201 | 0.199 | 0.195 | −0.006 | −0.004 |
| 3 | thin tails + size-dep. vol | N=1000 | 0.339 | 0.267 | 0.308 | +0.041 | 0.264 | 0.269 | 0.273 | +0.009 | +0.004 |
| 4 | σ_m 0.4→1.0 (rejected: rescales systematic risk) | N=1000 | 0.674 | 0.648 | 0.665 | +0.017 | 0.620 | 0.640 | 0.632 | +0.012 | −0.008 |
| 5 | firm types, equal shares + thin tails | N=500 | 0.294 | 0.267 | 0.284 | +0.017 | 0.230 | 0.252 | 0.204 | −0.026 | −0.048 |
| 6 | firm types, data-like (60/30/10) + thin tails | N=500 | 0.352 | 0.297 | 0.341 | +0.044 | 0.266 | 0.304 | 0.286 | +0.020 | −0.018 |
| 7 | priced rate risk β_zr=−0.0004 (oracle only) | N=300 | 0.306 | 0.277 | 0.297 | +0.019 | – | – | – | – | – |
| 8 | any KP parameter change | N=300 | 0.22–0.28 | – | – | ≤+0.008 | – | – | – | – | – |
| 9 | micro-cap spike (10% of firms carry the anomaly) | N=500 | 0.293 | 0.260 | 0.276 | +0.016 | – | – | – | – | – |
| 10 | U-shape via β_zr=+0.0002 (offsetting premia; capped by r∞>0) | N=500 | 0.269 | 0.198 | 0.239 | +0.041 | 0.159 | 0.177 | 0.165 | +0.006 | −0.012 |
| 11 | disaster SDF ω=0.8 (aligned channels) | N=500 | 0.398 | 0.333 | 0.379 | +0.046 | 0.340 | 0.349 | 0.318 | −0.022 | −0.031 |
| 12 | disaster SDF ω=1.3 (offsetting channels) | N=500 | 0.368 | 0.263 | 0.328 | +0.065 | 0.248 | 0.250 | 0.255 | +0.007 | +0.005 |
| 13 | disaster SDF ω=1.5 (ratio peak 1.32) | N=500 | 0.329 | 0.228 | 0.300 | +0.072 | 0.220 | 0.156 | 0.185 | −0.035 | +0.029 |
| 14 | ω=1.3 + level features (`rff_lev`) | N=500 | 0.368 | 0.263 | 0.327 | +0.064 | 0.248 | 0.255 | **0.260** | **+0.012 (t≈9)** | +0.005 |

Reading: items 1–2 raise the *room* to ~15% of total SR (#2, #6); the shape experiments (#10–13, §8–9) push the
nonlinear/linear ratio to 1.32 via offsetting premium channels, with the rare-disaster SDF the only mechanism that both
enlarges the room without a structural cap and yields a significant realised DKKM win (#12, #14: +0.012 over FMR, t≈9,
beating the rank-linear benchmark too once level features are added). Everything else leaves the realised RFF edge at
≈ +0.01 or negative; a low-dimensional rolling regression (rank-OLS, or raw FMR when types separate on levels, #13)
remains remarkably hard to beat at T=360.

## 8. Shape experiments: can the nonlinear/linear *ratio* be enlarged substantially? (all N=500, T=500, thin tails)

The room's ratio to the linear ceiling is scale-invariant — signal boosts move both ceilings together — so only the *shape*
of the premium function in rank space matters. Three orthogonal-to-linear shapes were tested:

| shape | economy | SR_max | linear | RFF-3600 | ratio | estimators (FMR / rank-OLS / RFF) |
|---|---|---|---|---|---|---|
| spike in bottom size decile | micro10: 10% micro-caps carry all β-dispersion | 0.293 | 0.260 | 0.276 | 1.06 | – |
| U-shape in bm via offsetting premia | bzrp2: β_zr=+0.0002 (growth options earn a *positive* rate premium) | 0.269 | **0.198** | **0.239** | **1.21** | 0.159 / 0.177 / 0.165 |
| both + data-like types | types3v2_bzrp2 | 0.326 | 0.283 | 0.314 | 1.11 | 0.249 / 0.276 / 0.272 |

Findings:
1. **The spike fails**: 50 micro-caps carry too little aggregate alpha, and a linear rule still rides the within-segment bm slope.
2. **Offsetting premia is the one shape lever that works**: flipping the sign of the rate-risk price makes growth (low-bm)
   firms earn a rate premium while value firms earn the cash-flow premium — the premium is U-ish in bm, the linear ceiling
   *falls* from 0.267 to 0.198 while the nonlinear one stays at 0.239. Ratio 1.21, the largest found in any BGN variant.
   But it is **capped by the term structure**: the Vasicek limiting yield r∞ = r̄ + (σ_r²/2)[(β_zr/σ_r²)² − (β_zr/σ_r² + 1/(1−κ))²]
   turns negative for β_zr > +0.00027, at which point long-bond prices and J* explode (the β_zr=+0.0005 attempt produced
   J* 5–19× assets in place and a singular return covariance). +0.0002 is already near the cap.
3. **Even there the estimators collect nothing**: at T=360, RFF-3600 (0.165) does not beat rank-OLS (0.177); the whole
   +0.041 of room is lost to sample-moment noise because this economy's premia are smaller relative to noise.
4. Stacking with the data-like types dilutes the ratio back to 1.11 (the types' linear premium re-enters).

**Answer to "how can the room be enlarged substantially":** within a BGN-class economy — firm value a linear aggregator
of project values under a lognormal SDF with an affine term structure — it essentially cannot. The three structural exits:
(i) **equity as an option** (leverage + default → premia genuinely non-monotone in distress; the fixed-GS route);
(ii) **several priced shocks whose loadings are different nonlinear functions of characteristics** (sector shocks with
mix-dependent exposures), which requires new valuation machinery, not parameters; (iii) a **non-lognormal SDF**
(state-dependent price of risk, disasters) so that premia depend on higher moments of cash flows that characteristics map
to nonlinearly. Each breaks one of the two linearity pillars; nothing inside the current pillars gets the ratio past ~1.2,
against ~2–4 in DKKM's data.

## 9. Non-lognormal SDF: rare disasters (`bgn_dis/`)

Specification. m_{t+1} = lognormal(ν) × κ^{D_{t+1}} / ((1−p) + pκ), with D iid Bernoulli(p) independent of (ν, ξ).
In a disaster month, each live project of a type-τ firm dies with probability L_τ (common trigger, idiosyncratic kills).
All closed forms survive because a priced disaster death-rate is mathematically BGN's own depreciation: the consol and
the growth options use the priced survival π·φ_τ, φ_τ = ((1−p)+pκ(1−L_τ))/((1−p)+pκ), while physical expectations use
π·(1−pL_τ); survival cross-moments add a common priced crash factor to the covariance matrix
(E[Y_iY_j] = π²[1−p(L_i+L_j)+pL_iL_j]).  Two design choices matter:
* all types share one β-distribution (`type_bstar_policy="common"`) — holding acceptance *probabilities* fixed instead
  forces high-L types into deep-hedge (negative-β) projects and flips the premium ordering;
* C̄ is compensated by a weight ω for the disaster value discount (`Cbar_comp`): ω<1 leaves the disaster premium loading
  into bm in the *same* direction as the β-channel; ω>1 overcompensates, so high-L firms look like growth firms — bm then
  carries two *offsetting* premium channels.

Validation (`results/log_dis_check2.txt`): p=0 and L=0 reproduce `bgn_types` to machine precision; with p=3%, κ=2.5,
L=(0,.15,.30): expected returns increase in L (1.31/1.67/1.75%/month), realized means match, disaster-month returns are
−10%/−15%, and the model's per-type conditional volatilities match realized ones (.064/.088/.114 vs .059/.084/.110) —
i.e. the crash-covariance algebra is confirmed in simulation.

Results (N=500, T=500, 360-month windows):

| ω | channels | SR_max | linear ceil. | RFF-3600 ceil. | ratio | FMR | rank-OLS | RFF-3600 | t(RFF−FMR) |
|---|---|---|---|---|---|---|---|---|---|
| 0.8 | aligned | 0.398 | 0.333 | 0.379 | 1.14 | 0.340 | 0.349 | 0.318 | −10 |
| 1.3 | offsetting | 0.368 | 0.263 | 0.328 | **1.25** | 0.248 | 0.250 | **0.255** | +4.8 |

ω=1.3 is the only configuration in the entire project where the DKKM estimator significantly beats *both* FMR and the
rank-linear OLS — exactly on the predicted mechanism: bm mixes an increasing (β) and a decreasing (disaster) premium
channel, so no linear rule can sign it, while bins/RFF disentangle the channels through interactions with roe (which
partially reveals the type via the ω-compensated cash-flow level).  The realized edge is still small (+0.007) because the
estimator again collects only ~78% of its ceiling at T=360, but the *ratio* 1.25 is the largest achieved, has no hard cap
(unlike β_zr's term-structure bound), and is tunable through (p, κ, L-spread, ω).

ω-sweep (oracle, N=500; ratio = RFF-3600 ceiling / linear ceiling):

| ω | 0.8 | 1.0 | 1.1 | 1.3 | 1.5 | 1.8 |
|---|---|---|---|---|---|---|
| linear ceiling | 0.333 | 0.334 | 0.332 | 0.263 | 0.228 | 0.186 |
| RFF-3600 ceiling | 0.379 | 0.376 | 0.349 | 0.328 | 0.300 | 0.237 |
| **ratio** | 1.14 | 1.13 | 1.05 | 1.25 | **1.32** | 1.28 |

The ratio dips at ω≈1.1 (both channels almost cancel, killing linear *and* nonlinear premia together), then peaks at
ω≈1.5 where the across-type (disaster) slope in bm is roughly minus the within-type (β) slope: room +0.072, the largest
in the project. Estimators at the peak (ω=1.5, best κ each): **FMR-raw 0.220 > RFF-3600 0.185 > rank-OLS 0.156** — two lessons.
RFF now beats the rank-linear benchmark by +0.03 (the first sizeable realized nonlinearity gain), but plain FMR on *raw*
characteristics wins outright: the L-types separate on characteristic *levels* (bm differs across types by multiples),
information the DKKM rank transform destroys, and rolling estimation lets FMR adapt beyond any constant-θ ceiling.  So a
disaster economy tuned for maximum nonlinear room also creates level information that favours raw-characteristic methods;
at ω=1.3 (types overlapping more) the ordering is instead RFF > FMR ≈ rank-OLS.  A fair horse race in these economies
should include both raw and rank versions of the linear benchmark.

Conclusion for the non-lognormal route: it works, for the reason the theory says it should — the disaster premium is a
second priced channel whose loading maps into the same characteristics with the opposite sign, which is a *shape* change,
not a scale change. Within one afternoon of tuning it already produces the best nonlinear/linear ratio and the first
significant RFF-over-linear win; pushed harder (larger L-dispersion with overlapping types, ω near the exact-cancellation
point, disasters also hitting the SDF's lognormal part), this is the most promising way to widen the DKKM gap without
leaving the BGN framework.

### 9b. Level features (`--levels`): giving RFF the information the rank transform destroys

Fixed-stats standardised raw characteristics (median/IQR from the first estimation window, clipped at ±3) are appended to
the rank features, both as extra RFF inputs (`rff_lev`) and in a linear benchmark (`linlev`).  Findings (N=500, W=360):

* Constant-θ population ceilings are unchanged (rffL-3600 ≈ rff-3600) — unconditionally, rank functions already span the
  type structure.  But the *monthly-reoptimised* linear oracle rises sharply with levels (0.245→0.279 at ω=1.5,
  0.273→0.300 at ω=1.3): levels carry conditioning information, which is exactly what rolling raw-characteristic FMR
  exploits and per-month ranks erase.
* ω=1.3: `rff_lev` (2-draw ensemble) reaches **0.260 with t vs FMR ≈ 8.9** — the strongest estimated result in the
  project, beating plain RFF (0.256, t 5.1), rank-OLS (0.255) and FMR (0.248) simultaneously.
* ω=1.5: levels halve RFF's HJ distance (0.86→0.47) and raise its realized SR, but rolling raw-FMR (0.220) still wins the
  conditional-SR ranking; the best RFF κ sits at the top of the grid — that economy wants heavy shrinkage plus
  conditioning, which a 6-parameter rolling regression supplies more cheaply than 3,600 shrunk features.

Net message of §9: the rare-disaster SDF plus level-augmented RFF produces the largest room (ratio 1.32), the most
significant DKKM win (t≈9), and a clean diagnosis of the remaining bottlenecks — rare-event learning (11 disasters per
window) and the state-blindness of rank-only features.

### 9c. Disaster frequency: the peso problem cannot be diluted for free

Attacking the rare-event learning problem by making disasters more frequent but smaller requires care with scaling.
Holding only the *premium* fixed (p·(κ−1)·L constant; p 3%→10%, κ 2.5→1.45, L unchanged) triples the disaster
*variance* while keeping its compensation — the price of crash risk per unit variance falls 3× and every method
deteriorates (FMR 0.165, rff_lev 0.154 vs 0.248 / 0.260 at p=3%), even though the ceiling ratio survives (1.27).
The scaling that holds the disaster claim's Sharpe ratio fixed while multiplying learning events by f is
L → L/√f and (κ−1) → (κ−1)/√f: at p=10% that is κ=1.82, L=(0, .082, .164) (tag `dis_p10b`).  Result: ceilings
0.231 / 0.307 (ratio **1.33**, tying the project peak; the largest absolute room, +0.075) — but the estimators do
*not* improve: FMR 0.211, rank-OLS 0.212, RFF 0.221 (t=1.9), rff_lev 0.212.  RFF's ceiling-capture rate is ~72%,
essentially the same as with 11 disasters per window (~78%).  Conclusion: the rare-event (peso) channel was **not**
the binding constraint — the generic P/T=10 ridge cost at T=360 is; and fewer, larger crashes identify the disaster
direction *better* per unit of sample than many small ones (rff_lev 0.260, t≈8.9 at p=3% vs 0.212, t≈0.4 at p=10%).
The recommended configuration therefore remains p=3%, κ=2.5, ω=1.3 with level-augmented RFF.

## 10. Final experiments: longer windows and diffuse multi-channel premia — and the synthesis

**Longer windows (`dis_long`: the ω=1.3 disaster economy, N=300, 985 months; ceilings 0.251 linear / 0.300 RFF-3600).**
Estimators at window 360 vs 840: FMR 0.232→0.240, rank-OLS 0.233→0.240, RFF-3600 0.232→0.238, rff_lev 0.235→0.237.
Longer windows lift every method, but the *linear* ones converge onto their ceiling first, while RFF's capture rate of its
larger ceiling barely moves (78%→79%; P/T falls only 10→4.3). At window 840 RFF is significantly *behind* FMR. Extrapolating
DKKM's own theory, harvesting the surplus would need P/T ≪ 1, i.e. thousands of months. **The virtue-of-complexity surplus
in these economies is real in population but un-harvestable at any plausible sample length.**

**Nine crossed sectors (`sect9`: 3 disaster exposures × 3 project-risk dispersions, uniform shares, equalised mean β,
N=500).** This implements DKKM's diffuse-λ misspecification story with existing machinery: premia live on 9 channels that
5 linear coefficients cannot span. Ceilings 0.272 linear / 0.350 RFF-3600+levels (room +0.077, ratio 1.28, SR_max 0.380);
estimators FMR 0.255, FFC 0.262, rank-OLS 0.257, RFF 0.261, **rff_lev 0.265 (t≈5)**. The room-side mechanism works exactly
as predicted; the realized edge is once more ≈+0.01 at the ~76% capture wall.

**Synthesis.** Across ~35 economies and every lever tried (parameters, tails, size-dependent risk, firm types, priced rate
risk, non-lognormal disaster SDF with offsetting channels, level features, ensembling, disaster frequency, window length,
nine-channel diffuse premia), the results obey one equation: realized gap = room × capture. Room can be pushed from ~5% to
~30% of SR by shape-based modifications (offsetting premium channels; many channels vs few characteristics) — but not to
the data's ~100–300%. Capture is pinned at ~75–80% by the P/T ridge at any realistic window, and longer windows help the
linear benchmark first. Hence the realized DKKM−FMR gap is bounded near +0.01–0.02 monthly SR in this entire model class,
with the best documented configuration being the disaster economy (p=3%, κ=2.5, ω=1.3) with level-augmented RFF
(t≈5–9 across variants). A data-sized gap requires an economy outside this class — equity-as-option with a countercyclical
GE kernel (fixed GS21) remains the one untested candidate.

## 11. Fixing GS, and the KP analogues

### 11a. What was wrong with GS — and what fixing it revealed

Diagnosis (`gs_fixed/check_bellman.py`): (i) the 20-point Tauchen x-grid spans ±4 *unconditional* sd, so the conditional
one-month transition has ~2–3 reachable points and the discrete kernel's one-period discount E[M|x] ranges 0.896–1.074
instead of e^{−r}=0.9917 — ±8%/month of phantom risk-free variation; (ii) the VFI's self-loosening tolerances leave
Bellman residuals of 0.5–1% of value (~1%/month of spurious alpha, matching E[m(1+R)]≈0.96); (iii) the Python simulator
contradicts the solution it consumes (r=0.0748/12 vs 0.1/12; dividends (1−τ)e^{x+z}−δ vs e^{x+z}−δ; continuous AR(1)s +
quadrature vs Tauchen chains; hard default vs the m∼N(0,2.5²) smoothing).  The identities among the saved files hold
exactly, so the CSVs are a coherent snapshot of an inaccurate solution.

Fix (`gs_fixed/`): a Python re-solve (`gs_solve.py`, flat joint iteration from a zero start — large starting values push
the endogenous investment probability toward 1 where the g-scaling makes the operator transiently expansive) with
xnum=161 (kernel error ~1e-4), tol 1e-6 (converges in ~350 sweeps / 2 min), one consistent dividend definition; plus a
discrete-state simulator (`gs_sim.py`) whose conditional moments use the same transition matrices and default smoothing,
so the Euler identity holds to machine precision.  Validation: Euler identity 1e-15; realized-kernel check 1.02 (BGN-level
small-sample drift); coarse-grid re-solve correlates 0.994 with the shipped MATLAB values; leverage interior and
state-dependent (coupon 0.05–0.58, market leverage ≈0.59); **max conditional SR 0.48 vs HJ bound 0.535** — inside the
bound at last (the original evaluation had 2.14).

Result (N=500): SR_max 0.459 and **every characteristic basis — 6-parameter linear included — attains 0.459**; estimated
FFC 0.449, FMR 0.447; RFF without a market factor collapses to 0.24 (see 11b).  Under the constant-γ one-shock SDF that
`GS21.m` actually implements, the cross-section carries **zero learnable structure**: premium dispersion is 0.16%/month
against a 1.1%/month market premium, and any market-exposed portfolio spans the kernel.  The leverage/default machinery
contributes nothing to premium dispersion when the price of risk is constant.  Conclusion: the GS simplification did not
merely have numerical bugs — it removed the economy's entire cross-sectional content.  GS becomes informative for the gap
question only with the GE kernel's countercyclical price of risk (a state-dependent γ(x) would be the minimal version,
now easy to add to `gs_solve.py`).

### 11b. KP analogues of the BGN modifications

* **Thin tails** (σ_u 1.5→0.7): room unchanged (+0.004 vs +0.005) — KP's premium surface is affine in PVGO/V and stays so.
* **A mechanical discovery that matters beyond KP**: rank-standardised RFF factor weights sum to zero, so every DKKM
  portfolio is zero-net-investment with no market exposure.  In alpha-driven economies (BGN) that is harmless; in
  level-driven economies (KP: cs-sd(μ)=0.09% vs E[μ]=0.50%; fixed GS even more so) market-less RFF is capped at the
  long-short Sharpe ratio — KP RFF ≈0.086 vs linear ≈0.20, t≈−135, at *any* ridge penalty, and winsorisation does not
  help because it is not an outlier problem.  This reproduces a pattern visible in the paper's own saved results
  (`rs` without `include_mkt` = 0.086 in KP).  With the unpenalised market factor appended (the paper's own device),
  RFF recovers to 0.2005 (κ=0.03) — parity with FMR 0.2052 and rank-OLS 0.1969, exactly where KP's +0.004 ceiling room
  says it must land.  Benchmarking implication: DKKM results in level-dominated economies are meaningless without the
  market factor.
* **Verdict on KP analogues**: thin tails (no effect — the premium surface stays affine in PVGO/V), level features (no
  effect), market-factor correction (restores RFF to parity).  The remaining analogue — priced disaster types with a
  λ-multiplier compensation — was deliberately not built: KP shares the affine, level-dominated structure that made the
  same machinery inert in simplified GS, its second-moment code is the costliest to modify, and the outcome (ratio ≤~1.1)
  is predictable from the BGN evidence.  KP cannot deliver a gap under any parameterisation or these modifications.

### 11c. Fixed GS with the market factor, and the two models' common lesson

RFF + unpenalised market on fixed GS: **0.4578 (t=+34 vs FMR 0.4470)**, above FFC 0.4488 and rank-OLS 0.4436, HJ distance
0.056 vs 0.095–0.147 for the linear methods.  Since every basis's population ceiling is 0.459, this is pure *estimation
efficiency* — the heavily shrunk RFF+market portfolio tracks the (market-dominated) optimum with less rolling-window noise
than small OLS — not a nonlinearity gap.  Common lesson of KP and simplified GS: both are level-dominated economies with
affine cross-sections; in such economies complexity can at best win on estimation efficiency by a Sharpe centile, and
only after being handed the market factor.  The one modification §11a leaves open with real upside is a state-dependent
price of risk γ(x) in `gs_solve.py` (the minimal stand-in for the GE kernel GS dropped) — a small change to the solver
now that it exists in Python.

## 12. Parity: every model gets the full alternative set, cleanly evaluated

The BGN alternatives (clean baseline; thin tails; exposure types; rare-disaster SDF with crash types and a
compensation knob; level features / market factor / zero-project guard in the estimators) now exist for all three
models. New machinery built for parity:

**KP crash (`kp_dis/`)** — disasters kill a fraction L_τ of a firm's projects; the priced kill intensity h^Q_τ enters
the A-coefficients per type; per-type G-function and integral tables; survival factors F2/sv on the VAP-continuation
terms of E[R_iR_j] (KP's timing lets cash flows accrue before deaths, so cash-flow terms carry none); a type output
multiplier θ_τ=(const_τ/const_0)^ω as the compensation knob (θ must also scale cash flows — values, K*, and PVGO all
scale by θ^{1/(1−α)}); λ-multipliers as a second knob making high-L firms growth-looking.  Three implementation bugs
were caught by the validation battery: a one-month kill-timing shift (crashes must coincide with the κ-weighted SDF
months), kills applied to unborn projects (cumulative products executed them at birth, collapsing high-L firms to
0.1 projects), and θ missing from simulated cash flows.  Final validation: p=0 reproduces `kp/` to 1e-12 and L=0 with
disasters on is exactly zero difference; premia increase in L (0.95→1.17→1.22%/month) with matching realized means;
crashes of −5%/−8.6% land in disaster months; per-type conditional vols match realized; max_sr 0.335.

**GS crash (in `gs_solve.py` + `gs_sim_dis.py`)** — disasters destroy a fraction L_τ of capital while the coupon is
unchanged, so the leverage state jumps to b/(1−L) on a crash: distress amplification inside the Bellman equations
(disaster branches in both the equity continuation and the debt recursion; per-type re-solves; `a_shift` available as
the compensation knob).  The discrete-argmax two-cycle that appears at coarse x-grids is damped (0.5 update weight);
note the Bellman operator is only a proper contraction when the x-grid is fine enough that E[M|x] ≤ 1 in all states —
another reason the original 20-point solution could not converge.  Validation: damped p=0 re-solve matches the
production solution to 1e-4 absolute; Euler identity 5e-4 (VFI tolerance); realized κ-kernel check 1.017;
premia increase in L (2.1→2.5→3.0%/month, realized matches); disaster-month returns −17%/−33%; high-L types
endogenously deleverage (mean coupon 0.41→0.13) and the highest-exposure type produces actual defaults (0.3%/yr);
max_sr 0.506 vs the disaster-inclusive HJ bound 0.603.

Results of the three crash economies (kpd_al, kpd_off, gsd) under the standard clean protocol are appended below.

### 12b. Results: the crash alternatives across all three models (N=500, T=500, clean protocol)

| economy | SR_max | linear ceil. | best nonlinear ceil. | room | ratio | FMR | FFC | rank-OLS | RFF-3600(+mkt,+lev best) |
|---|---|---|---|---|---|---|---|---|---|
| BGN crash ω=1.3 (`bgn_dis`) | 0.368 | 0.263 | 0.328 | +0.065 | 1.25 | 0.248 | 0.242 | 0.250 | **0.260** (t≈9) |
| BGN crash ω=1.5 (ratio peak) | 0.329 | 0.228 | 0.300 | +0.072 | 1.32 | **0.220** | 0.155 | 0.156 | 0.185 |
| KP crash, aligned (`kpd_al`) | 0.350 | 0.326 | 0.334 | +0.008 | 1.02 | 0.301 | 0.295 | **0.304** | 0.298 |
| KP crash, offsetting λ (`kpd_off`) | 0.358 | 0.330 | 0.338 | +0.008 | 1.02 | 0.299 | 0.301 | **0.307** | 0.306 |
| GS crash, distress amp. (`gsd`) | 0.506 | 0.502 | 0.504 | +0.002 | 1.004 | 0.481 | 0.483 | 0.479 | **0.487** (t=4.5, efficiency only) |

**The cross-model lesson of the parity program.** The identical crash recipe (κ^D kernel, per-type exposure L, priced
survival, compensation knob) produces radically different amounts of *learnable nonlinearity*: large in BGN, nil in KP
and GS.  The difference is where the exposure information lands in characteristic space.  In BGN, bm is a *ratio* whose
numerator and denominator carry premium channels of opposite sign, so the crash channel bends the premium surface
non-monotonically — a shape linear rules cannot span.  In KP and GS, each channel (PVGO share, disaster exposure,
leverage) maps monotonically into its own characteristic combination, so every new premium channel — however large
(both models' SR_max rose by 0.12–0.13) — lands inside the linear span.  Crash risk per se does not create a DKKM gap;
non-monotone characteristic-mixing does.  This sharpens §10's conclusion: the data-like gap requires premium surfaces
whose *shape* in characteristic space is non-monotone/interactive, and of the three models only BGN's valuation
structure supplies the necessary mixing — KP and GS would need state-dependent prices of risk (the GE kernel) to
break their monotone maps.

> **Errata for §11–12 GS numbers.** All GS results above (`gs_fixed` baseline `sol_x161`, the `gs_dis` crash
> rows in §12, and the §11 fix narrative's levels) were computed on solutions that predate the exact kernel
> renormalization — see §13b. The qualitative conclusions survive (each comparison was internally consistent,
> and the re-based crash room is +0.007 vs the stale +0.002), but for quotable levels use the §13g re-based
> table: constant-γ SR_max 0.401 (stale: 0.48), crash SR_max 0.413 (stale §12 row: 0.506).

## 13. State-dependent prices of risk (in progress) — and two structural findings about the original solutions

§12 ended with the diagnosis that KP and GS need state-dependent prices of risk to break their monotone
characteristic maps. Building those versions surfaced two errors in the *existing* solution files — both worth
knowing about independently of the γ-experiments.

### 13a. The original KP `G_func.csv` uses inconsistent stationary weights

KP's growth-option value G solves a two-regime (λ high/low) ODE system. The shipped `Code/.../G_func.csv` was built
from two component functions combined as `G_up = G1 + 1.35·G2`, with λ̃ normalized so that the *code's* λ_L = 1.
Solving the coupled system exactly (direct block solve, `kp_gam/kp14_fd_gam.py`, converges to 1e-8) shows the
correct stationary-consistent combination is `G_up = 1.717·G1 + 0.633·G2`: the code's normalization implies a
stationary mean E[λ̃] = 1.717 ≠ 1. Consequence: the original KP PVGO is mis-scaled by ≈40–60% depending on ε.
The price process is still *internally* consistent (the panel and SDF use the same tables), so none of the
§1–12 comparative statements about KP change — but the level of the KP PVGO share differs from the true KP14
equilibrium. All `kp_gam` runs use the exact 4-state tables.

### 13b. The GS baseline (`sol_x161`) predates the kernel renormalization — re-based

The discrete GS kernel must satisfy E[M(x,·)|x] = e^{-r} in every state x (else the model's bond is mispriced and
the Bellman operator need not be a contraction). Exact row renormalization was added to `gs_solve.py` *after*
`sol_x161` and the §12 disaster solves were produced: their kernels misprice the bond by up to 17% at edge-x states
(0.6% even in the interior). Because the risk adjustment concentrates risk-neutral mass exactly at the low-x edge
and GS valuations are near-unit-root (duration ~100+ months), those 11 edge rows move *interior* values by ~100%
(corr(Q0) between conventions is only 0.75). Both old and new solutions are exact fixed points of their own
operators (one-sweep residuals ≤1e-5), and each §12 comparison was internally consistent — but the correct
convention is the renormalized one. Everything GS is being re-based on it: `sol_g00chk` (constant γ, exact kernel)
replaces `sol_x161` as baseline; the three disaster types re-solved; oracle + estimator runs re-executed (chain
`gs_fixed/run_gs_rebase.sh`, results tagged `g00`, `g28`, `dis_rebase`).

### 13c. Design: state-dependent γ in all three models

* **GS** (`sol_g28`): γ(x) = clip(0.5 − 0.28·x/sd(x), 0.05, 1.0) — countercyclical price of risk in the solved
  kernel; solved with cycle-average termination. Slope 0 reproduces the (re-based) baseline by construction.
* **KP** (`kp_gam/`): a 2-state aggregate regime s multiplies both prices of risk (γ_x·gmult[s], γ_z·gmult[s]),
  ν01 = 0.25/yr (calm→stress), ν10 = 0.50/yr. The A-coefficients solve coupled 2×2 systems; G solves a 4-state
  (λ×γ) ODE; per-regime integral tables; the panel simulates the regime chain and every conditional moment mixes
  the two switch branches *as a common shock* (cross-firm second moments are mixtures of per-branch outer
  products, not products of mixtures). At gmult=[1,1] the economy is regime-independent to machine precision
  (panel 1e-14, μ/Σ bitwise across regime seeds). Stressed calibration under test: gmult=[0.5, 2.0]
  (regime discount spread ρ_s: 0.56%/mo calm vs 3.6%/mo stressed); the regime state is exported as the panel's
  conditioning feature (`rf_stand`), so RFF and linear-rf bases can condition on it.
* **BGN**: σ_m(r) or regime-γ analogue — not started; requires numerical repricing on the r-grid.

### 13d. KP regime-γ oracle (N=500, T=500, gmult=[0.5, 2.0]) — and why binary regimes cannot create room

The stressed economy behaves exactly as designed in the time series (validated: calm months max_sr 0.10,
E[μ]=0.19%/mo; stressed months max_sr 0.34, E[μ]=0.76%/mo, cross-sectional dispersion 3.7×; realized returns
match expected within well under 1 SE in both regimes; at gmult=[1,1] the economy is regime-independent to
machine precision). The oracle:

| basis | const-θ cond. SR | unc SR |
|---|---|---|
| FMR-raw (6) | 0.161 | 0.151 |
| lin_rank (6) | 0.219 | 0.223 |
| **lin_rank_rf (12)** | 0.220 | **0.244** |
| poly2 (27) | 0.223 | 0.246 |
| RFF-3600 | 0.222 | 0.237 |
| rffL-3600 | 0.223 | 0.238 |

SR_max 0.251. Nonlinear room: **+0.004** — the same nil as every other KP variant. The regime *does* add
+0.02 of unconditional SR through regime-timing, but the linear basis with rf(=regime) interactions captures
all of it; RFF even trails poly2 slightly.

**Why this is a theorem-shaped negative result.** With a binary aggregate state s and premium surfaces that are
affine in characteristics within each regime (true in KP: every channel map stays monotone/affine per regime),
the conditional premium is premium(X,s) = a_s + b_s'X, which is exactly spanned by (1, X, s, X·s) — the
lin_rank_rf basis. So *any* two-state price-of-risk process — however violent, and whether it scales or rotates
the premia — lands inside the linear-with-interactions span when the state is observable. And if the state is
hidden, it produces unspanned time-variation that no characteristic basis (linear or RFF) can capture either.
State-dependence can only create learnable nonlinearity if the state enters the *within-regime cross-sectional
map* non-monotonically — which is BGN's ratio-mixing mechanism again, not a regime overlay.

### 13e. BGN regime version (`bgn_gam/`) — built, closed-form

Making the market-shock price σ_z·gmult[s] regime-dependent keeps BGN fully closed-form because cash flows
are i.i.d. lognormal (each cash-flow date's premium is priced once, by the regime at the start of its final
period): project values decompose onto two bases, V_s(r,β) = Chat·[e^{-βg₀}·DA_s(r) + e^{-βg₁}·DB_s(r)] with
DA_s(r) = Σ_k π^k a_k(s) B(k,r), a_k(s) = [P^{k-1}](s,0); J* generalizes to per-regime bond-option sums
(reduces to the baseline at unit multipliers to 1e-12); all conditional moments branch-mix per destination
regime as a common shock. Interesting twist vs KP: e^{-βg₀} and e^{-βg₁} are two different monotone transforms
of the *same* project β, so a firm's channel pair lies on a curve, not a plane — within-regime curvature is
possible here, unlike in KP. Both conditioning features (r and s) are exported to the feature bases
(vector-rf plumbing added to oracle/dkkm/estimators).

Validation: with the J* table held fixed, `bgn_gam` at gmult=[1,1] reproduces the baseline `bgn` economy to
machine precision (prices 6e-16, conditional μ 1e-13, Σ 7e-13 relative), and is regime-path-independent to
1e-14; the expected-new-project-NPV closed form (exponential-density integrals of the two-basis payoff, with
a vectorized-bisection acceptance threshold) matches the baseline's closed form to 1e-14 at unit multipliers.
The β-density tail bounds the stress multiplier: 2·max(gmult)·scale < 1 (scale = 0.137 → gmult up to ≈3.6).

GS re-based sanity (N=150 sim on the exact-kernel solutions): constant-γ baseline mean max_sr 0.276
(cs-sd(μ) 0.0010) vs γ(x)-slope-0.28 economy 0.332 with double the time-variation (sd 0.034 vs 0.018) and
cs-sd(μ) 0.0012 — the countercyclical kernel is alive in both dimensions before the oracle runs.

### 13f. BGN regime-γ oracle (N=500, T=500, gmult=[0.5, 2.0]) — the within-regime curvature mechanism works

| basis | const-θ cond. SR | unc SR |
|---|---|---|
| FMR-raw (6) | 0.147 | 0.148 |
| lin_rank (6) | 0.193 | 0.196 |
| lin_rank_rf (18: +r, s interactions) | 0.196 | 0.213 |
| poly2 (33) | 0.198 | 0.218 |
| bins (301) | 0.201 | 0.208 |
| RFF-3600 | 0.216 | 0.239 |
| **rffL-3600** | **0.217** | **0.241** |

SR_max 0.245. **Room = +0.021 (ratio 1.11) — the largest non-crash room found in any variant of any model**,
and the unconditional gap over the best linear-with-interactions basis is +0.028. This is exactly the §13d
loophole: the spanning theorem kills regime overlays only when premia are affine within each regime. Here the
regime enters the *valuation map itself* — firm value mixes e^{-βg₀} and e^{-βg₁}, two different monotone
transforms of the same project β — so characteristics reveal the two exposure channels only jointly and the
within-regime premium surface is curved. Consistently, poly2 captures only a sliver of it (0.198): the
curvature is high-order, which is precisely RFF's comparative advantage. Same lesson as §12 from a new
direction: what creates a DKKM gap is never the state process per se but non-monotone channel mixing in the
characteristic map; a state-dependent price of risk delivers it only in BGN, whose ratio-valuation supplies
the mixing.

### 13g. GS re-based oracles (N=500, T=500, exact kernel) — γ(x) adds no room; §12 survives re-basing

| economy | SR_max | best linear ceil. | best nonlinear ceil. | room |
|---|---|---|---|---|
| GS constant γ (`g00`) | 0.401 | 0.400 | 0.401 | ≈0 |
| GS γ(x) slope 0.28 (`g28`) | 0.361 | 0.359 | 0.358 (RFF *trails*) | ≈0 |
| GS crash, exact kernel (`dis_rebase`) | 0.413 | 0.406 | 0.412 | +0.007 (ratio 1.016) |

Constant-γ GS still has zero learnable cross-section on the correct kernel. The γ(x) economy moves SR_max and
its state-dependence exactly as designed (max_sr 0.25→0.44 across x-states), but the cross-sectional surface
stays in the linear span: γ(x) scales near-affine premia by a common observable state, so the §13d spanning
logic applies with x in place of s — RFF-3600 even needs its heaviest ridge and still trails lin_rank. The
re-based crash economy's room (+0.007, ratio 1.016) is slightly above the stale-kernel estimate (+0.002) but
confirms §12: no ratio-mixing, no gap.

**Bottom line of the state-dependent-γ program**: BGN +0.021 (ratio 1.11) vs KP +0.002 vs GS ≈0. A
state-dependent price of risk creates learnable nonlinearity if and only if it enters a valuation map that
mixes exposure channels non-monotonically into the characteristics — which of the three models only BGN has.

### 13h. Estimators (w=360, N=500, best κ per method)

**KP regime-γ [0.5, 2.0]**: rff_lev 0.175 (t vs FMR ≈ 5) > FF 0.171 > linrank 0.168 > FMR 0.162, against
const-θ ceilings ≈0.22 (the usual ~78% capture). The +0.007 edge over linrank matches the oracle: a sliver of
regime-timing plus estimation efficiency — no nonlinearity to harvest, exactly as the spanning argument says.

**BGN regime-γ [0.5, 2.0] — the strongest estimated gap of the project:**

| method (best κ) | cond. Sharpe | t vs FMR |
|---|---|---|
| **RFF-3600 (+mkt)** | **0.2419** | **21.5** |
| rff_lev-3600 | 0.2415 | 19.0 |
| linrank | 0.2200 | 2.2 |
| FF | 0.2190 | 1.3 |
| FMR | 0.2164 | — |
| linlev | 0.1713 | −6.2 |

DKKM lands AT its unconditional-oracle ceiling (0.241): the room is 100% harvested, unlike the crash
economies' ~75% capture. Gap over the best linear method +0.022 (over FMR +0.025, 12% relative), at plain
T=360 / N=500 with the standard method set. Why full capture here: the nonlinearity is a smooth
regime-interaction whose conditioning variable is in the RFF feature set, the rolling window adapts θ to the
regime, and there are no disaster tails to poison the ridge. The two working levers now rank: regime-γ in BGN
(gap +0.022–0.025, t≈21, fully harvested) > rare disasters in BGN (room +0.065–0.072 but capture-limited to
gap ≈ +0.01, t≈9). Mechanism-wise they are the same lesson — non-monotone channel mixing in bm — but the
regime version puts the curvature where the estimator can actually reach it.

**GS constant-γ, exact kernel (`g00`)**: rff 0.399 (κ=0.1, P=360) vs linrank 0.383, FM 0.379, FF 0.372.
The +0.016 edge (t≈22) is pure estimation efficiency — the oracle gives every basis the same 0.400 ceiling,
which ridge-RFF-with-market attains and unshrunk linear OLS does not. Same conclusion as the stale-kernel run,
now on the correct solution.

**GS γ(x) slope 0.28 (`g28`)**: rff 0.358 (κ=1, P=36 — at its ceiling) vs FF 0.324 > linrank 0.320 > FM 0.316.
A +0.038 gap (t≈18) with *zero* nonlinear room — this is a **conditioning gap**: γ(x) puts large common
time-variation into premia, and the rf(=x)-featured ridge times it while the unconditioned rolling linear
methods cannot. A third distinct source of DKKM outperformance alongside efficiency and curvature — and a
caution for interpreting empirical gaps: a large realized DKKM edge need not indicate cross-sectional
nonlinearity at all.

**GS crash, exact kernel (`dis_rebase`)**: rff 0.388 vs FM 0.381 ≈ linrank 0.380 > FF 0.373 (+0.008, t≈5),
matching the small +0.007 oracle room.

## 14. The complete symmetric grid — overall results

Every cell run with the identical configuration (oracle N=500, T=500, `--levels`; estimators w=360,
`--levels --include_mkt`; RFF numbers are averages over independent draws, best fixed κ per method;
machine-readable copy was `results/grid_summary.csv`, deleted 2026-09-10 and recoverable at
23f9380; see docs/RESULTS.md for what is current):

| model | variant | SR_max | linear ceil. | nonlin ceil. | room | FMR | FF | lin | RFF | gap RFF−lin | t vs FMR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | baseline | 0.293 | 0.235 | 0.271 | +0.036 | 0.220 | 0.220 | 0.233 | 0.239 | +0.006 | 14.0 |
| BGN | crash ω=1.3 | 0.368 | 0.263 | 0.329 | +0.066 | 0.248 | 0.242 | 0.255 | 0.260 | +0.005 | 8.9 |
| BGN | regime-γ [.5,2] | 0.245 | 0.196 | 0.217 | +0.021 | 0.216 | 0.219 | 0.220 | **0.242** | **+0.022** | 21.5 |
| BGN | regime-γ [.3,3] | 0.212 | 0.154 | 0.180 | +0.025 | 0.105 | 0.132 | 0.174 | **0.198** | **+0.024** | 31.7 |
| KP | baseline | 0.240 | 0.223 | 0.226 | +0.004 | 0.205 | 0.193 | 0.197 | 0.201 | +0.004 | −8.5 |
| KP | crash (aligned) | 0.350 | 0.328 | 0.334 | +0.006 | 0.301 | 0.295 | 0.304 | 0.298 | −0.006 | −2.1 |
| KP | regime-γ [.5,2] | 0.251 | 0.222 | 0.223 | +0.002 | 0.162 | 0.171 | 0.168 | 0.175 | +0.007 | 5.1 |
| GS | baseline (exact kernel) | 0.401 | 0.400 | 0.401 | ≈0 | 0.379 | 0.372 | 0.383 | 0.399 | +0.016 | 21.8 |
| GS | crash (exact kernel) | 0.413 | 0.406 | 0.412 | +0.007 | 0.381 | 0.373 | 0.380 | 0.388 | +0.009 | 5.1 |
| GS | γ(x) slope 0.28 | 0.360 | 0.359 | 0.358 | ≈0 | 0.316 | 0.324 | 0.320 | 0.358 | +0.038 | 17.9 |
| BGN | γ(r) nonlin 0.5→3 | 0.557 | 0.439 | 0.533 | **+0.093** | **0.470** | 0.415 | 0.458 | 0.464 | +0.006 | −2.7 |
| GS | γ(x) nonlin logistic | 0.398 | 0.395 | 0.395 | ≈0 | 0.349 | 0.358 | 0.351 | **0.392** | **+0.041** | 14.1 |
| KP | γ(y) common 0.5→2.5 | 0.366 | 0.340 | 0.344 | +0.003 | 0.279 | 0.277 | 0.274 | 0.281 | +0.007 | 3.0 |
| KP | γ(y) rotation 0.85→3 | 0.430 | 0.414 | 0.418 | +0.004 | 0.363 | 0.345 | 0.364 | 0.367 | +0.002 | 4.7 |

(The last four rows are the §17 continuous nonlinear-state economies, added after the original grid.)

Reading the grid by column: **room** (population nonlinearity) exists only in BGN — crashes make the most
of it (+0.066) but resist harvesting; regime-γ makes less (+0.021–0.025) but yields it fully. **gap RFF−lin**
(what an econometrician sees) is large in exactly two places: BGN regime-γ (true curvature, fully captured)
and GS γ(x) (pure conditioning/timing — zero room). KP never produces a gap through any channel, and its
crash economy even puts rank-OLS ahead. The wide BGN regime economy adds classical-method breakdown on top
(FMR 0.105 vs DKKM 0.198). Intentional asymmetries: only BGN has a wide-regime variant (KP's is provably
inert by §13d spanning; GS's cross-section is flat), and GS's γ-state is its natural continuous x rather
than a 2-state chain.

## 15. SDF-weight analysis: neural-net fit of weights on characteristics (GomesSchmid protocol, all cells)

The `Documents/GomesSchmid` methodology — MLP (5×32 ReLU, 30 epochs) fit of each method's portfolio weights
on within-month rank-standardized characteristics, per-month cross-sectional correlation with the true
conditional MVE weights (both raw and MLP-fitted, winsorized 1/99), LOO variable importance on the truth
MLP, and 2×2 partial-dependence plots — implemented natively in `sdfweights/` and run on all 10 grid cells
(same seeds/windows/κ as the grid estimators; extraction stage reproduces `run_estimators`' weights exactly).
Outputs: `sdfweights/results/` (correlations_grid.csv, per-cell corr/loo CSVs, 20 PD figures).

**Cross-sectional correlation of method weights with the true SDF weights** (raw weight | MLP-fitted):

| model / variant | chars ceiling | FMR | FF | linrank | DKKM |
|---|---|---|---|---|---|
| BGN baseline | 0.77 | 0.39 \| 0.42 | 0.37 \| 0.43 | 0.39 \| 0.41 | **0.48 \| 0.49** |
| BGN crash ω=1.3 | 0.67 | 0.09 \| 0.11 | −0.07 \| −0.12 | −0.01 \| −0.08 | 0.07 \| 0.07 |
| BGN regime-γ [.5,2] | 0.69 | 0.41 \| 0.44 | 0.37 \| 0.49 | 0.28 \| 0.34 | **0.48 \| 0.51** |
| BGN regime-γ [.3,3] | 0.58 | 0.27 \| 0.28 | 0.31 \| 0.35 | 0.20 \| 0.24 | **0.38 \| 0.41** |
| KP baseline | 0.58 | **0.41 \| 0.45** | 0.13 \| 0.15 | 0.24 \| 0.25 | 0.21 \| 0.22 |
| KP crash aligned | 0.50 | **0.38 \| 0.40** | 0.29 \| 0.34 | 0.35 \| 0.38 | 0.31 \| 0.31 |
| KP regime-γ [.5,2] | 0.61 | 0.49 \| 0.52 | 0.47 \| 0.53 | 0.50 \| 0.52 | 0.42 \| 0.48 |
| GS baseline | 0.95 | 0.59 \| 0.62 | 0.57 \| 0.60 | 0.56 \| 0.62 | **0.68 \| 0.71** |
| GS crash | 0.96 | **0.81 \| 0.84** | 0.81 \| 0.84 | 0.72 \| 0.73 | 0.75 \| 0.76 |
| GS γ(x) | 0.93 | **0.48 \| 0.48** | 0.33 \| 0.35 | 0.35 \| 0.35 | 0.44 \| 0.46 |

Findings:
1. **The curvature signature shows directly in weight space.** In both BGN regime-γ economies DKKM's
   weights track the truth far better than linrank's (0.48 vs 0.28; 0.38 vs 0.20) — the linear method loses
   the weight *function* exactly where the premium surface curves. LOO confirms the mechanism: the truth MLP
   is dominated by bm (R² drop 0.27 in g0520; every other char ≤ 0.01), and the PD plots show the true and
   DKKM weight curves in bm bending together where the FM/FF/linrank panels stay affine.
2. **Crash weights are nearly unrecoverable.** BGN crash: every method's weight correlation collapses to
   ≈0 (FM 0.09, FF −0.07, DKKM 0.07) even though all earn 0.24–0.26 Sharpe — SR flows through a few priced
   directions while weight variance is dominated by (unlearned) idiosyncratic-hedging demands. This
   reproduces the GomesSchmid *Full run* pattern (raw correlations ≈0 in the original-vol calibrations) and
   explains it: weight-space correlation and SR capture can decouple completely.
3. **Weight correlation is blind to the conditioning channel.** GS γ(x): FM's weights correlate best with
   truth (0.48 > DKKM 0.44) even though DKKM's Sharpe is far higher (0.358 vs 0.316) — the γ(x) gap lives in
   time-varying scale, invisible to per-month cross-sectional correlation. Corollary for empirical work:
   weight-correlation diagnostics understate DKKM's edge wherever that edge is timing.
4. **KP: FMR's raw-characteristic levels track the true weights best in every variant** (PVGO information
   that rank transforms discard), yet earn no extra Sharpe — the mirror image of finding 3.
5. **Sign check vs the GomesSchmid low-vol run**: our KP correlations are positive (0.2–0.5) in all
   variants; the GomesSchmid `low vol` kp14 results show *large negative* correlations (−0.57 to −0.81)
   for all methods. Given §13a (the original `G_func.csv` used in that pipeline's `programs/` carries the
   mis-normalized stationary weights), the sign/scale of that pipeline's KP true weights deserves a re-check.

## 16. The complete gap taxonomy

The program ends with three distinct, separately-measured sources of DKKM−classical gaps:
1. **Estimation efficiency** (everywhere; ≈ +0.01–0.02, needs the unpenalized market in level-dominated
   economies): ridge shrinkage + rank transforms reach a shared ceiling that unshrunk OLS/FMR miss.
2. **Conditioning/timing** (state-dependent prices of risk; up to +0.04 in GS γ(x), +0.02 in KP regimes):
   observable-state premium variation that rf-featured RFF times; linear-with-interactions would too, but the
   *classical* methods don't condition.
3. **Cross-sectional curvature** (the only "true" nonlinearity gap; BGN only): ratio-characteristics mixing
   opposite-signed or differently-transformed exposure channels — crashes (+0.065–0.072 room, ~75% capture)
   or regime-γ (+0.021–0.025 room, ~92–100% capture, the harvestable version). Peak demonstration:
   `bgn_gam` gmult=[0.3,3.0], where all three sources stack and FMR additionally breaks on regime-shifting
   characteristic levels: DKKM 0.199 vs FMR 0.105 (+89% relative — data magnitude).

*Robustness of the g0520 gap to κ selection* (all methods get identical treatment; "rff/linrank" numbers are
averages over the two independent RFF draws, never a best-draw): with a single pre-registered κ for every
method the gap rff−linrank is +0.013/+0.022/+0.023 at κ=0.01/0.05/0.1 — no hindsight needed; with fully
out-of-sample κ selection (expanding-window trailing realized SR) rff_lev delivers 0.2446 vs linrank 0.2210
(+0.024). Plain rff degrades under OOS selection (0.217) only because its κ=0.001 column is a
high-variance trap for the trailing-SR rule; the levels variant has a flat κ profile and selects cleanly.

**Scaling the spread (`gmult=[0.3, 3.0]`, tag g0330)**: the room grows with the regime gap. Oracle:
SR_max 0.212, best linear 0.154 (lin_rank_rf), best nonlinear 0.180 (rffL3600) → room **+0.025, ratio 1.17**;
unconditionally rffL3600 0.217 vs lin_rank_rf 0.180 → **+0.037, ratio 1.21** — approaching the crash-economy
ratios (1.25–1.33) but, unlike them, in harvestable smooth-interaction form (validated: calm/stressed max_sr
0.069/0.290, realized ≈ expected per regime).

Estimators (w=360): **rff_lev-3600 0.1985 vs linrank 0.1745 (gap +0.024, t vs FMR ≈ 32) vs FF 0.132 vs FMR
0.105.** The DKKM–linear gap grows with the spread as the room does (capture ≈92% of the 0.217 unconditional
ceiling), and the classical raw-characteristic methods now *break down outright* — regime shifts move the raw
characteristic levels, wrecking rolling raw-coefficient estimates — so the DKKM–FMR gap reaches +0.093, i.e.
nearly double FMR's Sharpe (0.199 vs 0.105, an 89% relative gain, the first data-magnitude relative gap of the
project). Ranking across the two calibrations: mild regimes → gap +0.022 with FMR merely trailing; violent
regimes → gap +0.024 over the best linear method and a collapse of FMR/FF. If the goal is a demonstration
economy where DKKM decisively dominates *all* classical methods at realistic T, `bgn_gam` gmult=[0.3,3.0]
is it.

## 17. Nonlinear continuous-state prices of risk (the spanning theorem's loophole, all three models)

§13d showed a *binary* regime can never create RFF-only room. The loophole: make γ a **nonlinear function of
a continuous observable aggregate state** — then premium coefficients b(state) are nonlinear in a variable the
linear basis can only interact with linearly. Implemented in all three models with logistic (convex,
countercyclical) multipliers:

* **BGN `bgn_gamr/`**: gmult(r) = 0.5→3.0 logistic in the interest rate (the model's own state, already the
  panel's conditioning feature). Closed forms break because e^{-β·gmult(r_{k−1})} covaries with the discount
  path; solved by a calibrated one-period-operator chain on an r-grid (D(r,β) exact to 6e-5, option value to
  1e-3, J* to 0.15% vs the closed forms at constant γ; unit-γ equivalence to baseline at table precision),
  with a rank-7 SVD factorization D(r,β)=Σφ_m(r)w_m(β) powering the pair-moment engine (rank 1 exactly at
  constant γ). Validated stressed: max_sr 0.156/0.333/0.513 by r-tercile, realized ≈ expected.
  **Oracle (N=500): SR_max 0.557, best linear 0.439 (lin_rank_rf), best nonlinear 0.533 (rffL-3600) —
  room +0.094, ratio 1.21: the largest room of the entire project**, ~4.5× the binary-regime version, and it
  compounds both channels (r-nonlinearity × bm-curvature). Estimators pending.
* **GS `sol_gnl`**: γ(x) = 0.10→1.20 logistic (solver's kernel handles arbitrary γ(x)). Oracle: cross-sectional
  room still ≈0 (lin_rank 0.3955 ≈ oracle 0.3956, RFF trails) — GS's flat cross-section is immune however
  nonlinear the state-dependence — but the conditioning channel now shows plainly in unconditional SR:
  poly2 0.381 > lin_rank_rf 0.364 > lin_rank 0.329, with rff-3600 (0.347) losing timing signal to shrinkage.
  Estimators pending.
* **KP `kp_gamy/`**: exogenous OU state y (half-life 2y, unit variance) with gmult(y) = 0.5→2.5 logistic on
  both prices of risk. Full generalization of the kp_gam machinery from 2 regimes to a 21-node y-grid:
  A-coefficients solve (diag(const(y)+θ) − Q_y)A = 1 with the OU generator; G solves a (2λ × 21y) coupled
  ε-ODE system (42k unknowns, direct factorization); 21 per-node integral tables; the panel and all
  conditional moments replace the 2-branch common-shock mixing with 7-node Gauss–Hermite quadrature over y′;
  y is exported as the conditioning feature. This is the decisive test: KP's binary regime provably added
  nothing — if the *nonlinear-continuous* version creates room, the conditioning-nonlinearity channel is
  model-generic. Tables built; validation + runs in flight.

### 17b. KP continuous-y result — and the direction-invariance refinement of the spanning theorem

kp_gamy oracle (gmult(y) = 0.5→2.5 logistic on BOTH prices of risk, N=500): SR_max 0.366, best linear 0.3405,
best nonlinear 0.3437 — **room +0.003 (ratio 1.01): still nothing**, despite validated max_sr swinging
0.27→0.43 across y-terciles. This pins the theory down completely: because the multiplier is common to both
priced shocks, μ_t = gmult(y_t)·μ̄ — the cross-sectional premium **direction is constant**; only its magnitude
moves. Conditional Sharpe rankings across portfolios are invariant to a common premium scale, so *no* common
multiplier — however nonlinear, in however continuous a state — can create cross-sectional room in any model.
The full statement:

> State-dependent prices of risk create learnable cross-sectional nonlinearity **iff** the state changes the
> cross-sectional *shape* of premia: either by bending the exposure→characteristic map (BGN's e^{−βγ(r)},
> room +0.094) or by **rotating** the premium direction across priced channels nonlinearly in the state.
> Common scaling (KP regime, KP continuous-y, GS γ(x)) moves only the timing channel.

The remaining cell — a genuine rotation — is running: γ_x's multiplier swings 0.85→3.0 in y while γ_z's is
held fixed (the x/z premium-direction ratio moves 3.5×, nonlinearly). Counter-rotations (γ_z falling) turn the
growth-option discount rate ρ(y) negative at low y (the γ_z σ_z term carries ~±0.06 of ρ) and are infeasible
without violating perpetuity convergence — itself a nice economics constraint: KP's structure caps how far the
premium direction can swing.

### 17c. BGN γ(r) estimators — large room, and FMR (not DKKM) harvests the conditioning

Realized (w=360, best κ): **FMR 0.470 > RFF-3600 0.464 > linrank 0.458 > rff_lev 0.449 > FF 0.415.**
Three lessons. (1) The +0.094 room does not translate into a DKKM gap at T=360: the RFF−linrank edge is just
+0.006, and DKKM reaches only ~88% of its const-θ ceiling. (2) Rolling FMR *exceeds even the conditional
oracle of the affine-in-raw-chars span* (0.470 vs 0.455) — its month-by-month re-fit of raw-characteristic
coefficients is itself a state-adaptive nonlinear mechanism, and because raw levels co-move with r, six
re-estimated parameters track γ(r) better than 3600 shrunk features. (3) rff_lev now *hurts* (0.449):
level features double down on exactly the channel FMR already owns. Together with §13h this brackets the
empirical question sharply: DKKM's decisive wins come from *interaction curvature* (regime-γ economies,
+0.022–0.024, t≈21–32), while smooth continuous-state conditioning — however much room it creates in
population — is contestable by cheap classical conditioning at realistic T.

### 17d. The rotation test — KP's cross-section is irreducibly one-directional

kp_gamy/grot (γ_x multiplier 0.85→3.0 logistic in y, γ_z fixed — the premium direction ratio moves 3.5×,
nonlinearly): SR_max 0.430, best linear 0.4141, best nonlinear 0.4176 — **room +0.0035. The rotation is
inert too.** Diagnosis in the moments: the rotation doubles the *level* of premia (E[μ] 1.42%/mo) while
cross-sectional dispersion stays at 0.10%/mo — KP firms load on the two priced channels almost in parallel
(z-exposure is the common α-leverage on aggregate TFP; heterogeneity enters only through the monotone PVGO
share), so rotating between nearly-parallel directions moves nothing across firms. Combined with §17b's
direction-invariance result, KP is now closed under every price-of-risk channel: common scaling can't work
(theorem), and rotation has nothing to rotate. **Cross-sectional room requires firm-level exposure
heterogeneity that the state can bend — of the three models only BGN (project-β distributions) has it.**

### 17e. Continuous nonlinear-state summary (all runs N=500, T=500; estimators w=360, best fixed κ)

| economy | SR_max | room | FMR | FF | lin | RFF | RFF−lin | what the gap is |
|---|---|---|---|---|---|---|---|---|
| BGN γ(r) 0.5→3.0 | 0.557 | **+0.094** | **0.470** | 0.415 | 0.458 | 0.464 | +0.006 | room huge but conditioning-dominated; rolling raw-FMR harvests it |
| GS γ(x) logistic .10→1.20 | 0.398 | ≈0 | 0.349 | 0.358 | 0.351 | **0.392** | **+0.040** (t=14) | pure conditioning: rf-featured ridge times γ(x), classics can't |
| KP y, common scale 0.5→2.5 | 0.366 | +0.003 | 0.279 | 0.277 | 0.274 | 0.281 | +0.007 | theorem-null (direction invariance) |
| KP y, rotation γx 0.85→3.0 | 0.430 | +0.004 | 0.363 | — | 0.364 | 0.367 | +0.002 | exposure-parallel null: nothing to rotate |

The §16 taxonomy survives fully intact and gains precision: continuous nonlinear states are the richest
*conditioning* device (GS: +0.040 realized, t=14, with zero room; BGN: +0.094 of population room), but the
conditioning channel is contestable — in BGN, where raw characteristic levels co-move with the state, rolling
FMR beats the shrunk 3600-feature ridge to it. The only economies in the whole program where DKKM decisively
dominates *every* classical method remain the regime-γ BGN economies (§13f–h): interaction curvature, fully
harvested, +0.022–0.024 over the best linear method at t≈21–32 and up to +0.09 over FMR.

### 17f. KP uneven binary regime — the spanning theorem confirmed in its sharpest form

kp_gam/gux (gmult_x = [0.6, 2.4], gmult_z = [1, 1]: the premium direction across the two priced channels
swings 4× between regimes — the strongest uneven pattern with a stable coupled perpetuity; counter-rotating
the z-price drives the 2×2 (ρ_s + switching) system near-singular): SR_max 0.294, best linear 0.2676, best
nonlinear 0.2701 — **room +0.0025**. The winning nonlinear basis is poly2, i.e. exactly the (s, X·s)
interactions the §13d spanning theorem says suffice: with an observable binary state and per-regime-affine
premia, premium(X,s) = a_s + b_s′X is spanned by (1, X, s, X·s) *for any* b_s — uneven scaling included.
The regime's whole value is timing (unconditional poly2 0.288 vs lin_rank 0.264). Validation: regime split
works mechanically (max_sr 0.14/0.38, E[μ] 0.14%/1.50%) with cross-sectional dispersion unmoved (0.0010 in
both regimes) — the uneven price rotation has nothing cross-sectional to act on, consistent with §17d.
Estimators: rff_lev 0.226 (t≈9.6) vs FF 0.212 ≈ FMR 0.207 ≈ linrank 0.207 — a +0.019 realized gap with zero
room: pure regime-timing through the rf feature, per the taxonomy.

### 17g. GS binary regime (γ 0.3 calm / 1.5 stressed) — the default-optionality channel stays dormant

`gs_solve_reg.py` / `gs_sim_reg.py`: two coupled Bellman systems (equity + debt per regime), continuation
expectations discounting with the current regime's exactly-renormalized kernel and mixing next-regime values
as a common shock; both regime and x exported as conditioning features. Machine-exact at gmreg=[1,1]
(1–2e-15 vs `sol_g00chk`); stressed Euler identity 1.3e-15; realized ≈ expected per regime.

The economics: stress *triples* cross-sectional premium dispersion (0.0006→0.0016) — the precondition KP
lacked — via leverage-dependent value sensitivity. But the oracle returns **room = 0.000**: the conditional
oracle of plain lin_rank (0.2159) is within 0.4% of SR_max (0.2167) — the true MVE portfolio is linear in
rank-characteristics even in stress. Diagnosis: the equity-option convexity that could bend the map lives at
the default boundary, and the boundary is unpopulated — zero realized defaults even at γ=1.5, because firms
delever endogenously and the default-smoothing shock (σ_m = 2.5) flattens the kink. Away from the kink, the
leverage→premium map is smooth and monotone → spanned. The channel is real but equilibrium suppresses it;
activating it would require calibrations that force a standing population of near-default firms (higher
leverage targets, harsher or longer stress, thinner smoothing) — at which point GS starts resembling a
distress-economy rather than its published calibration. Estimators confirm the timing-only reading:
rff 0.195 (t≈20) vs linrank 0.176 ≈ FMR 0.175 — a +0.019 realized conditioning gap on zero room.

**Program close.** Every price-of-risk channel in every model is now tested: KP null four ways (common
binary, common continuous, continuous rotation, uneven binary — two by theorem, two by parallel exposures),
GS null three ways (γ(x) linear, γ(x) logistic, binary regime — flat/monotone cross-section, dormant default
channel), BGN positive three ways (binary regime +0.021, wide +0.025, continuous nonlinear +0.093). The
general law stands: **state-dependent prices of risk create learnable cross-sectional nonlinearity only where
the state bends a heterogeneous firm-level exposure map — and among BGN, KP14, and GS21, only BGN's
project-β distributions supply that heterogeneity.**

## 18. Constructive proof: exposure heterogeneity turns KP's room on (`kp_bx/`)

§17's law said room requires firm-level exposure heterogeneity that the state can bend. The constructive
test: give KP firm types whose cash flows load as x^{β_f} on the aggregate (β ∈ {1.0, 1.8, 3.0} — GBM powers
stay GBM, so everything remains closed-form: per-(type, regime) A-coefficients from coupled 2×2 solves,
per-(type, regime) G tables, and the pairwise conditional x-moments as an exact E[x'^{β_i+β_j}] matrix),
crossed with the 2-state regime-γ (×0.5/×2.0) and an ω-style calm-value compensation (1.2) so values do not
reveal types monotonically. Now the regime bends a heterogeneous map — type-1.0's discount moves ×1.7 across
regimes while type-3.0's moves ×3.0 — and bm mixes the β-channel against the λ_f growth-option channel.
(The spread is bounded below at β=1: lower exposures push the growth-option perpetuity's 4-state principal
eigenvalue negative.)

Validated: exact reduction to the kp_gam baseline at a single β=1 type; stressed economy has dispersion
quadrupling in stress and per-type premia ordered by exposure, realized ≈ expected.

**Oracle (N=500): best linear 0.2394 (lin_rank_rf), best nonlinear 0.2581 (rffL-3600) — room +0.019
(ratio 1.08), the first genuine room in KP after four null price-of-risk designs.** Unconditional gap
+0.035; RFF beats poly2 by +0.013 in the ceiling (high-order curvature). The law is now proven in both
directions: remove exposure heterogeneity (KP, GS as published) and no state process creates room; add it
(bgn's project-βs natively, kp_bx by construction) and regime-γ generates it immediately. Estimators pending.

**§18 estimators (w=360):** FMR 0.181 > rff 0.177 ≈ linrank 0.176 >> rff_lev 0.157 > FF 0.098 — the +0.019
room is NOT harvested, and the reason completes the recipe: x^{β_f} exposure makes raw characteristic LEVELS
co-move strongly with the state per type, handing rolling raw-FMR a free conditioning channel that rank
transforms strip and the P≫T ridge shrinks (rff_lev doubles down on FMR's channel and loses badly, t=−16).
Contrast bgn_gam, where the win was clean and total: its regime-curvature lives in rank-space interactions
while the r-state barely moves levels. Final form of the design rule: **a harvestable DKKM gap needs (i)
heterogeneous firm-level exposures, (ii) a state that bends them nonlinearly, and (iii) the bending confined
to rank/interaction space — if the exposure channel also shows in raw levels, cheap rolling classical
conditioning contests it.** BGN's regime economies satisfy all three; kp_bx satisfies (i)+(ii) only.

## 19. The engineered economies: room by design in KP and GS (campaign in progress)

Standing objective: substantial room in both population and estimation for all three models. BGN is the
achieved benchmark (§13f–h). The design rule (§18) engineered into the other two:

**KP (`kp_vy/`)** — a priced stationary OU factor y (κ=0.35, unit sd; price of risk γ_v = 1.2) with firm
types loading cash flows as e^{β_f y} (β ∈ {0.02, 0.06, 0.12} → premium spread 2–12%/yr; the feasibility
frontier is β·κ_y·y_max < const: larger exposures make the growth-option perpetuity non-contractive), plus
y=0 over-compensation 1.2. All closed-form: per-(type, y-node) A-coefficients from (diag(const_f(y)+θ)−Q_y)
solves, per-type (2λ×21y) G systems, and the e^{β_f y'} factors absorbed into coefficient/table scalings so
every squared and cross moment carries the right exponents automatically. Validated: realized ≈ expected per
type (first moments) and realized portfolio variances match predicted conditional variances to 1–3% on the
exposure-spread and neutral portfolios (second moments).

**Oracle (N=500): SR_max 0.961; linear ceiling 0.538; RFF-3600 ceiling 0.818 — room +0.280, ratio 1.52,
by far the largest of the project — with the empirical DKKM signature: a steep complexity ladder
(P=36: 0.670 → 360: 0.784 → 3600: 0.818) and bins/poly2 far below RFF.** Levels are stationary by
construction (clause iii), so unlike kp_bx the harvest has no FMR level-channel to lose to.

**§19a kp_vy estimators (w=360, κ up to 10): KP harvests.** rff_ens 0.5707 (P=360, κ=0.001) vs FMR 0.5449
and linrank 0.5417 — realized gap **+0.026 over FMR (t=5.2)**, +0.029 over the best linear method, with the
unconditional SR agreeing (0.588 vs 0.579). This is the first significant DKKM win in KP after five designs,
at the same absolute scale as BGN's flagship regime economies, and rff_lev *loses* (0.516, t=−4) — confirming
clause (iii): the level channel is dead by construction, so classical conditioning has nothing to contest
with. Two structural notes: (a) capture is ~10% of the +0.280 room — the ceiling needs P=3600 curvature, but
with a 360-month window θ-estimation noise caps the realized winner at P=360 (0.568 at P=3600 vs 0.571 at
P=360's ensemble); the room is real but a 500-month panel cannot buy the complexity that spans it, which is
itself the virtue-of-complexity point — the gap should *grow* with T. (b) The rolling estimators (0.571)
exceed the const-θ *linear* ceiling (0.538): rolling θ adds a timing harvest on top of the cross-sectional
one, consistent with §16's conditioning channel (lin_rank's conditional oracle is 0.727).

**GS (`gs_bx` iterations)** — exposure types e^{β_f x} on the stationary AR(1) aggregate: room trajectory
+0.004 (uncompensated) → +0.006 (compensated) → +0.009 (+ leverage characteristic). Crucially the harvest
mechanics already work in GS: the round-1 estimators put DKKM on top (0.2225, t=12) with no FMR takeover, and
the realized gap matches the room one-for-one — whatever room GS gains should be collected. Iteration 3
(five types, β ∈ {1, 2.5, 4, 5.5, 7}, compensation dosed by the measured value-gap slope) **doubles the room
again: +0.018** (rff-3600 0.2921 vs lin_rank_rf 0.2738; unconditional +0.024; poly2 trails RFF by 0.010 —
genuine high-order curvature). Trajectory: +0.004 → +0.006 → +0.009 → +0.018, now at the BGN/kp_bx scale.
Estimators chained (GS's demonstrated ~1:1 harvest makes this the likely second full success).

Also settled: the kp_bx harvest failure was NOT κ-selection — with the grid extended to κ=10 the best RFF is
unchanged (0.177, P=36) and FMR still wins; the failure is the level channel, as designed out in kp_vy.

**§19b gs_bx bx7 estimators (w=360, κ up to 10): GS harvests against every classical benchmark.**
rff_ens 0.2369 (P=36, κ=0.1) vs FMR 0.2213 (**gap +0.016, t=10.4**), FF 0.2291, linrank 0.2175 — the
gap over linrank (+0.019) again matches the +0.018 room one-for-one, GS's now-thrice-demonstrated ~1:1
harvest. Two caveats that complete the picture: (a) linlev (linear ranks+levels, heavily shrunk at κ=10)
climbs to 0.2356, within 0.0013 of DKKM — in GS the e^{β_f x} exposure shows partly in stationary raw
levels, so a levels-augmented linear ridge collects most of the curvature premium; unlike kp_bx this does
NOT hand the win to rolling FMR (the AR(1) x keeps levels stationary, clause iii holds), but it does mean
GS's gap is thinner against the strongest linear benchmark than against the classical ones. (b) β=7 raw
levels are extreme enough that FMR's cross-sectional regression overflows in 0.8% of months (now scored
NaN by a hardened run_estimators rather than crashing the run) — a small live demonstration of the
level-poison mechanism.

**§19c campaign verdict (final).** Standing objective met in all three models, with honest asterisks:

| model | room | realized DKKM gap | vs |
|---|---|---|---|
| BGN (g0520/g0330) | +0.021 / +0.025 | **+0.022 / +0.024** (t=21.5 / 31.7) — 100% captured | all methods |
| KP (kp_vy) | **+0.280** (ratio 1.52) | **+0.026** (t=5.2) — ~10% captured, T-limited | all methods |
| GS (bx7) | +0.018 | **+0.016** (t=10.4) vs FMR; +0.001 vs linlev | classical methods |

The three harvest regimes are the taxonomy of §16 made concrete: BGN's regime-curvature is rank-space
interaction (full capture), kp_vy's is high-P curvature (capture bounded by P/T, grows with T — the
virtue-of-complexity comparative static itself), gs_bx's is partly linear-in-levels (capture split with
linlev). Getting all three to show room required, in every case, the same structural ingredient the
published models lack: heterogeneous firm-level exposures that an aggregate state bends nonlinearly.

## 19d. The extreme variants: how large can the gap get? (2026-09-02)

Each winner pushed to its frontier — three different answers:

**BGN `g0235` (gmult [0.2, 3.5], near the closed-form bound gmult < 3.65): the room channel saturates but
the gap channel does not.** Population room is unchanged (+0.023, ratio 1.17 — same as [0.3, 3.0]; the
wilder stress regime adds variance as fast as premium). But the realized gap GROWS on both margins:
rff_ens 0.1752 vs linrank 0.1455 (+0.030, more than g0330's +0.024) and vs FMR 0.0674 — **+0.108, a 160%
relative gain at t = 29.1**, with linlev collapsing outright (0.034, negative t). Mechanism: regime-whipsawed
raw levels destroy every method that touches them, while DKKM's rank features are invariant; and rolling θ
lifts DKKM above its own constant-θ ceiling (0.175 > 0.158). The largest *relative* gap of the project.

**KP `vyx` (γ_v = 1.8, β = [0.02, 0.07, 0.14] → premia ~6/15/28%/yr): everything scales — the
data-magnitude KP win.** Room +0.350 (SR_max 1.26, RFF-3600 ceiling 1.092 vs linear 0.743, ratio 1.47),
and, decisively, capture jumps from ~10% to ~29%: **rff_ens 0.8484 vs FMR 0.7408 — gap +0.108 (t = 21.6),
+0.101 over the best linear method.** The 70% stronger cross-sectional signal (cs-sd(μ) 0.0064 vs 0.0038)
raises the signal-to-noise of θ-estimation at fixed T, so the realized gap grows FASTER than the room —
the vys bottleneck was noise, not room. Clause-(iii) signature intact: rff_lev < rff, linlev ≈ linrank.

**GS `bx9` (β up to 9, regime [0.5, 4.0]): the room triples, the harvest collapses — the capture
frontier found.** Room +0.054 (three × bx7; poly2 trails RFF by 0.031, deep curvature), but realized:
linrank 0.2097 ≈ rff_ens 0.2090 — the DKKM edge over linear is GONE (both beat FMR by ~+0.022, t≈5).
The harsher economy inflates conditional-moment noise faster than it adds room; at T=360 the P≫T ridge
gives the surplus back. GS's harvestable optimum is the bx7 calibration (+0.016 vs FMR, t=10.4).

**The design law, final form:** the realized gap = room × capture, and the extreme variants show the two
factors respond to "more extreme" in opposite ways depending on WHERE the extremity lands. Premium-side
extremity (KP: bigger priced spreads) raises both room and capture. Volatility-side extremity (BGN wide
regimes: variance grows with premium; GS β=9: exploding conditional moments) leaves room flat-to-up but
degrades classical methods faster than DKKM (BGN — relative gap explodes) or degrades DKKM's own
θ-estimation (GS — gap vanishes). The empirically-relevant frontier is premium-side.
