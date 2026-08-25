# Independent numerical findings — why DKKM doesn't beat FF/FM in BGN/KP14/GS21


## A. The DKKM feature basis as coded is not complex at all

Share of RFF feature variance spanned by polynomials in the characteristics
(N=3000 firms, 300 features, gamma ~ U{0.5..1.0}, chars rank-standardized to [-0.5,0.5]):

| config | sd(Wx) | sin: R2 linear | sin: R2 quad | cos: R2 linear | cos: R2 quad |
|---|---|---|---|---|---|
| L=5  gamma .5-1.0 (bgn/kp14, as coded) | 0.50 | **0.992** | 0.992 | 0.001 | **0.998** |
| L=6  gamma .5-1.0 (gs21, as coded)     | 0.60 | **0.981** | 0.982 | 0.001 | **0.994** |
| L=130 gamma .5-1.0 (real-data DKKM)    | 2.72 | 0.120 | -- | 0.043 | -- |
| L=5  gamma x2                          | 1.06 | 0.854 | 0.855 | 0.001 | 0.938 |
| L=5  gamma x3                          | 1.64 | 0.564 | 0.566 | 0.001 | 0.748 |
| L=5  gamma x5.1 (bandwidth-matched)    | 2.71 | 0.215 | 0.218 | 0.001 | 0.370 |

sd(Wx) = gamma_rms * sqrt(L/12); gamma_rms(U{.5..1}) = 0.769, so L=5 gives 0.496.
Matching the real-data effective bandwidth would need gamma x sqrt(130/5) = 5.1.

Effective dimension of the full 3600-feature basis exactly as coded (including the second
rank-standardization applied to the sin/cos features at dkkm_functions.py:110):

| config | eff. rank (participation ratio) | #PCs for 90% | top-21 share |
|---|---|---|---|
| L=5  gamma .5-1.0 (bgn/kp14, as coded) | **16.5** | 21 | **90.2%** |
| L=6  gamma .5-1.0 (gs21, as coded)     | **20.7** | 36 | 81.5% |
| L=5  gamma x3                          | 32.8 | 86 | 68.3% |
| L=5  gamma x5.1                        | 115.9 | 306 | 35.4% |
| L=130 gamma .5-1.0 (real-data DKKM)    | **809.0** | 829 | 5.0% |

The "3600-feature complex model" is functionally ~17-21 dimensional = {const, 5 linear,
15 quadratic}. FM already spans 6 of those 21. sin features are ~99% linear; cos features
are ~0% linear and ~99.8% quadratic. **DKKM as implemented = Fama-MacBeth + a quadratic
overlay.**

CAVEAT added after the agent audit: this is the CROSS-SECTIONAL feature rank. The object
that governs ridge is EffRank(E[F F']), a time-series quantity, and DKKM's Theorem 3 is
stated on that. The flat complexity curve in section D is the direct evidence that the
managed-portfolio spectrum is degenerate too — but the CAUSE is the DGP (section B), not
the bandwidth. See section G.

---

## B. Population headroom: what actually creates room for a nonlinear estimator

Best attainable CONDITIONAL Sharpe within each estimator's function class, zero estimation
error. N=1000, idio vol 8%/mo, total factor Sharpe held fixed at 3.2.

### B1. Idiosyncratic-variance dispersion (the D^{-1} tilt) — NOT a lever
sd(log idio var) swept 0 -> 2.0, K=2 factors, loadings LINEAR in chars:
FM 3.178 -> 3.180 against an oracle of 3.182 -> 3.201. Gap never exceeds **1%**.
At N=1000 with K=2 idiosyncratic risk is diversifiable, so max Sharpe -> the factor Sharpe
lam' Omega^{-1} lam and any portfolio with the right factor exposures attains it. The
w* = b(x)/d(x) ratio nonlinearity is real but economically inert.
(The agents put sharper numbers on this: the tilt is worth 1.81% of SR* at N=100, 0.186%
at N=1000, 0.019% at N=10000, and stays ~0.2% even with a 20x spread in d.)

### B2. Nonlinearity of the loadings, K=2 — barely a lever
theta (nonlinear share of loading variation) 0 -> 0.95 moves DKKM/FM from 1.01x to only
**1.27x**, and only at theta=0.95. At theta <= 0.6 the ratio is <= 1.04x.

### B3. K = number of priced aggregate shocks — THE lever
theta=0.7 fixed, total factor Sharpe fixed:

| K | FM | FF | DKKM(coded bw) | DKKM(wide bw) | ORACLE | DKKM/FM |
|---|---|---|---|---|---|---|
| 1  (GS21)      | 3.195 | 3.195 | 3.197 | 3.197 | 3.197 | **1.00x** |
| 2  (BGN, KP14) | 3.172 | 3.168 | 3.194 | 3.194 | 3.194 | **1.01x** |
| 3  | 3.096 | 3.066 | 3.187 | 3.186 | 3.187 | 1.03x |
| 5  | 2.886 | 2.873 | 3.184 | 3.183 | 3.184 | 1.10x |
| 10 | 1.965 | 1.917 | 3.153 | 3.153 | 3.157 | **1.60x** |
| 20 | 1.213 | 1.379 | 2.744 | 2.745 | 2.752 | **2.26x** |
| 40 | 1.169 | 1.150 | 2.059 | 2.060 | 2.062 | 1.76x |

Mechanism: with K priced factors and N large, ANY set of >= K portfolios with linearly
independent factor exposures and diversified idiosyncratic risk attains the full maximum
Sharpe. FF/FM supply 6-7. For K <= ~5 the empirical factor models OVER-SPAN the SDF and no
function class can beat them. The gap opens only when K exceeds the number of empirical
factors, peaking around K = 10-20.

The three models have K = 1 (GS21), 2 (BGN), 2 (KP14).
**=> The absence of a DKKM advantage is a theorem, not a calibration accident.**

Caveat: population span only. "coded" and "wide" tie here because with 5 chars a
400-column basis nests almost everything in population; the bandwidth defect in section A
is an ESTIMATION-side problem, not a population-span problem.

---

## C. Analytics on the HJD metric
For a mean-variance efficient portfolio with weights Sigma_2^{-1} mu — which is what BOTH
the true HJ portfolio and every method's ridge-with-y=1 MVE produce — the realized excess
return has
    mean = a = SR^2/(1+SR^2)        sd = SR/(1+SR^2)
So the two legs are on a COMPARABLE gross scale (the crude version of "HJD is a units
artifact" is refuted). But
    HJD^2 ~ (a_true - a_method)^2 + sd_true^2 + sd_method^2 - 2cov
with a_true = SR_max^2/(1+SR_max^2). When SR_max is large, a_true -> 1 and the MEAN-LEVEL
term dominates, so HJD ~ a_true - a_method, nearly constant across methods. Moving a
method from SR 0.2 to 0.4 moves a_method 0.038 -> 0.138, i.e. HJD ~0.92 -> ~0.82: HJD does
respond, but weakly, nonlinearly, and dominated by how unattainable SR_max is.

---

## D. THE ACTUAL NUMBERS
16 panels per model, from `~/ASU Dropbox/Seth Pruitt/BGN and Kelly Malamud/Code/aws_results/*_results.pkl`

### Conditional Sharpe: rows = ridge alpha, cols = # RFF features

BGN                                  KP14                                GS21
       6      36     360    3600            6     36    360   3600           6     36    360   3600
0.000 .3936 .3902  .0203  .1854     0.000 .1970 .1385 .0036 .0152   0e0   1.636 1.839 .194  .218
0.001 .3930 .4085  .3914  .3892     0.001 .1981 .1701 .1560 .1540   1e-5  1.636 1.838 1.677 1.593
0.010 .3881 .4156  .4172  .4172     0.010 .2028 .1970 .1946 .1944   1e-4  1.636 1.825 1.768 1.745
0.050 .3724 .4007  .4054  .4055     0.050 .2063 .2081 .2082 .2081   5e-4  1.636 1.801 1.787 1.782
0.100 .3587 .3841  .3891  .3892     0.100 .2059 .2082 .2084 .2083   1e-3  1.635 1.789 1.784 1.782
1.000 .2758 .2876  .2900  .2899     1.000 .1857 .1882 .1881 .1879   1e-2  1.624 1.764 1.773 1.773

ff .3513  fm .3617  capm .1969      ff .1917 fm .1845 capm .1205    ff 1.568 fm 1.573 capm .0295
true SDF (realized) 0.651           true SDF 0.244                  true SDF 2.236

**THE COMPLEXITY CURVE IS FLAT.** Best-alpha Sharpe by feature count:
  BGN  : 6->.3965  36->.4170  360->.4178  3600->.4179   (+0.2% from 36 to 3600)
  KP14 : 6->.2071  36->.2090  360->.2092  3600->.2091   (+0.0%)
  GS21 : 6->1.639  36->1.840  360->1.796  3600->1.792   (**-2.6%**, peaks at 36)

Exactly what an effective-rank-17-to-21 basis predicts: ~36 random features already span
everything the basis can span; the other 3564 are noise. The alpha=0 row blows up at
nf=360 (= T, the interpolation boundary): HJD 29.8 / 21.7 / 8.5. The double-descent
machinery is present and behaving — only the ASCENDING branch is missing.

### Gaps and ceilings
                     DKKM/FF   DKKM/FM   ceiling(trueSDF)/FF
  BGN                1.19x     1.16x     1.85x
  KP14               1.09x     1.13x     1.27x
  GS21               1.14x     1.14x     1.43x
Even a PERFECT estimator beats FF by only 1.3x-1.9x. Real-data DKKM beats FF by ~2.6-3.4x.
The shortfall cannot be closed by fixing the estimator — the models must change.

### HJD
  BGN  capm .483  ff .420  fm **.879**  dkkm .371   sd(sdf_ret)=.430
  KP14 capm .229  ff .186  fm .198      dkkm .161   sd(sdf_ret)=.245
  GS21 capm .911  ff .438  fm .468      dkkm .337   sd(sdf_ret)=.374

---

## E. DECISIVE DIAGNOSTIC on the saved true SDF weights
`wts_prediction/{model}_sdfwts_chars.pkl`, 3 panels x 30 cross-sections each, chars
rank-standardized, w* winsorized 1/99 as the MLP code does.

R2 of the TRUE SDF stock weight w* on:

| model | LINEAR chars (= what FM spans) | + QUADRATIC (= what DKKM AS CODED spans) | wide-bw RFF | nonlinear headroom |
|---|---|---|---|---|
| BGN  | 0.346 | 0.564 | 0.727 | **+0.381** |
| KP14 | 0.766 | 0.808 | 0.896 | +0.130 |
| GS21 | **0.898** | 0.925 | 0.934 | +0.037 |

GS21's true SDF weight is 90% LINEAR in the characteristics; FM spans it almost exactly.
KP14 is 77% linear. Only BGN has large nonlinear structure, and DKKM as coded already
captures over half of it (0.346 -> 0.564).

NOTE the key tension with section B: BGN's w* is only 35% linear, yet FM still attains
0.362 of a maximum 0.651. **Weight-space accuracy and Sharpe are decoupled when K is
small** — you can get w* badly wrong and still capture the whole premium.

---

## F. Scale / tracking decomposition of the realized returns
                      mean            sd            corr with true SDF return
  BGN   sdf .281 | capm .031 ff .139 fm .205 dkkm .173
        sd  .430 | capm .166 ff .387 fm **.913** dkkm .433
        corr     | capm .297 ff .558 fm .503 dkkm **.653**
  KP14  corr     | capm .428 ff .703 fm .670 dkkm **.763**
  GS21  corr     | capm .015 ff .616 fm .586 dkkm **.712**

- DKKM DOES track the true SDF best in every model (corr +10% to +17% over FF). Real, modest.
- BGN's FM portfolio has sd 0.913 vs the true SDF's 0.430 — over-levered ~2x. That, not
  worse tracking (corr .503 vs ff .558), is why FM's raw HJD is 0.879. HJD as computed is
  partly measuring a leverage mismatch.
- GS21 CAPM corr with the true SDF = 0.015 in a ONE-SHOCK economy: the value-weighted
  market is orthogonal to the only priced factor. A defect, not a finding.

---

## G. SYNTHESIS, reconciled with the agent audit

Two independent causes, only one of which is binding:

**(A) THE ESTIMATOR IS NOT COMPLEX.** Effective rank 16.5-20.7 vs ~800 in the real-data
DKKM setup; the complexity curve is flat above 36 features. Real, and it means the paper
is not currently testing the virtue of complexity at all. BUT it is NOT the cause of the
null, and the obvious fix backfires:
  - DKKM use the IDENTICAL gamma grid, the identical rank-standardization, and the SAME
    FIVE CHARACTERISTICS in real data, and still get 0.74 -> 1.93-2.50.
  - Widening GAMMA_GRID measurably HURTS where the truth is low-order (GS21 36-feature
    span 0.879 -> 0.799 at gamma x4.65, -> 0.376 at x10).
  - Simply DELETING the second rank_standardize makes it worse (effective rank 14.4 -> 5.1):
    it is silently equalizing the 3.1x sin/cos amplitude asymmetry. Z-score instead, if anything.
  => Raise L, don't raise gamma. Treat the bandwidth as a robustness axis, never a fix.

**(B) THE ECONOMY HAS K <= 2 PRICED AGGREGATE SHOCKS. This is binding.** Six empirical
factors over-span a K<=2 SDF, so the population ceiling on DKKM-over-FM is 1.01-1.03x.
Agent-measured ceilings agree: BGN {1,size,bm} raw = 91.7% of max Sharpe, actual FM span
91.8%, perfect degree-3 polynomial 92.5% (headroom 0.7pp). GS21 FM = 96.5%, RFF-360 = 99.4%
(headroom 3.0%).

What is actually missing, ranked:
1. An expected-return component NOT proportional to the covariance loadings (DKKM's AIPT
   near-arbitrage). Without it the RFF-over-linear ceiling is 1.00 and the exercise is over.
2. A priced-risk space of dimension far above L: SR_lin/SR* ~ sqrt(L/K), which holds even
   when loadings are linear in chars. K = 1-2 -> 30 takes SR_FM/SR* from 1.00 to ~0.45.
3. Characteristics that are noisy, lagged, partial proxies rather than exact inverses of a
   2-3 dimensional state (GS21: bm = 1/P_ex to machine precision; BGN: the cross-section is
   a verified bijection of two numbers).

Ordering matters: (2) and (3) are defensible inside a frictionless "simple economic models"
framing; (1) is not, unless the wedge carries an aggregate priced component so it is
compensation for noise-trader risk rather than an assumed free lunch.

---

## H. Scripts that produced these numbers (regenerate as needed)
rff2.py       section A, feature linearity vs bandwidth
rff3.py       section A, participation-ratio effective rank
headroom.py   section B1, idio-variance dispersion sweep
headroom2.py  section B2, loading-nonlinearity sweep
headroom3.py  section B3, K sweep  <- the important one
realnums.py   section D, Sharpe/HJD from the results pickles
surface.py    section D, full complexity x shrinkage surface
scalecheck.py section F, scale and correlation decomposition
wstar2.py     section E, R2 of true w* on linear / quadratic / wide-RFF bases
