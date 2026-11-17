# What produces a complexity gap: one economy per model and per change

Alexander Ober with Claude. First version October 11, 2026; final numbers November 12, 2026; plain-language edit November 16, 2026.

**What this report is.** Three asset-pricing models from the literature are simulated. One change is made to each at a time, the same change in all three, and every resulting economy is scored on one question: does a flexible, nonlinear portfolio method beat the best linear one out of sample? The main tables hold one economy per model per change. Everything else that was run is in section 5. Every economy ran ten simulations (seeds) unless stated. The tables are built by `variants/campaign_tables.py` from the saved results.

**Terms used throughout.**

- *Characteristics.* The five firm variables every method sees: size, book-to-market, asset growth, profitability and momentum (plus market leverage in GS21). Methods see their ranks within each month.
- *Sharpe ratio.* A portfolio's mean excess return divided by its standard deviation. Monthly throughout; multiply by 3.5 for annual. A difference of 0.01 a month is 0.035 a year.
- *DKKM.* The nonlinear method under test (Didisheim, Kelly, Kozak and Malamud): a penalised portfolio built on thousands of random nonlinear transformations of the characteristics.
- *Classical methods.* Fama-MacBeth (FM); Fama-French factors (FF); a linear portfolio in the ranks of the characteristics, with or without the market (linear); and the estimated market line (CAPM).
- *Gap.* DKKM's out-of-sample Sharpe ratio minus the best of the four classical methods, averaged over seeds, with the t-statistic of that mean across seeds. Everything is estimated on a 360-month window. The bar for an economy to count is a gap of +0.029 a month (0.1 a year) with a clear t-statistic.
- *Best Sharpe, best linear, room.* Three numbers computed by an investor who knows the model's true conditional means and covariances. The best Sharpe is the highest attainable. The best linear is the highest a portfolio linear in the ranks attains. The room is the best nonlinear portfolio (random features, bins, quadratic terms) minus the best linear: the scope for a gap before any estimation. Room is necessary for a gap, not sufficient.
- *Type and loading.* A type is a label a firm draws at birth and keeps for life; it sets how the firm's cash flows load on the added risk factor. The loading $b_f$ is the size of that exposure.
- *Compensation.* Each type's cash-flow level is scaled so that, with the risk factor at its mean, a project's value does not depend on the type. The scaling is raised to an exponent: at 1 the correction is exact; above 1 it over-corrects, so the exposed types come out larger; below 1 it under-corrects.

**Table columns.** Model; economy (its tag and a one-line description); best Sharpe; market share (the equal-weighted market's Sharpe as a share of the best); best linear; room; room % (room as a percentage of the best linear); interaction (the part of the best nonlinear Sharpe beyond the best linear, $\sqrt{\text{nonlinear}^2 - \text{linear}^2}$); DKKM, FM, FF, linear and CAPM (each method's out-of-sample Sharpe); gap (with its t-statistic in parentheses); DKKM extra (DKKM's Sharpe beyond the best classical method, with the sign of the gap); and + (the number of seeds with a positive gap). A dagger marks a true-moments calculation that predates the current evaluation window. GS21 methods are scored on six characteristics (the five plus leverage); a double dagger marks a GS21 economy still scored on the first five, whose method columns are not comparable.

## The answer

Three models, BGN (Berk, Green and Naik 1999), GS21 (Gomes and Schmid 2021) and KP14 (Kogan and Papanikolaou 2014), are each simulated as 500 firms over 500 months. Each starts from its published calibration, and each section makes one change to it, the same change in all three.

- **A risk factor whose only priced event is a crash** (section 1): no gap. Room is 3 to 12% of the best linear Sharpe.
- **A price of risk that rises with the square of the aggregate state** (section 2): no gap. Room 0 to 14%.
- **A priced cash-flow factor that every firm bears through one of its own states** (section 3a): no gap. Room 5 to 19%. An exposure that only rises or only falls in one firm state is captured by the rank of the characteristic that reveals that state, however curved the exposure is.
- **An exposure that is the product of two firm attributes**, one of them a sign the characteristics do not carry directly (section 3b): a gap in all three. BGN +0.071, KP14 +0.181, GS21 +0.023 (at the bar).
- **An exposure that is a U in one attribute**, the smallest and the largest firms exposed and the mid-sized not (section 3c): a gap in all three. BGN +0.030, KP14 +0.025, GS21 +0.038 (a U in leverage: the least and the most levered firms exposed).
- **More priced factors** (section 4) help only when each factor's loading is itself a product of partly hidden states or the square of a characteristic. Factors that load linearly on the characteristics give nothing at any number of factors.

All ten seeds are positive in each of the six economies with a gap.

**Why these and not the others.** Every factor model is a firm's exposure times an aggregate shock. A linear portfolio in the ranks of the characteristics holds any exposure that is a sum of functions of single characteristics. A gap therefore needs an exposure with no such part: a product of two attributes that are each mean zero across firms, or a function that falls and then rises in one attribute. With one priced factor it is all or nothing. A linear portfolio with any correlation with the exposure mimics the factor, diversifies away the rest and takes the factor's whole Sharpe ratio. So the product economies need the sign hidden from the characteristics, and the U economies need the premium to rise and fall in what the characteristics reveal. Where the premium only rises or only falls in something revealed (GS21's production levels, leverage fixed by type, the KP14 U whose exposure made both exposed types the largest; section 5), the linear portfolio holds it however curved the exposure.

The baselines every section is read against:

<!-- TABLE:g0 -->
<!-- /TABLE:g0 -->

**The three models.**

- *BGN.* A firm is a portfolio of projects. Each project costs 1, pays a lognormal cash flow $C_{jt}$ each month with its own correlation to the market shock, and dies with probability 1% a month. One new project arrives each month and is taken if its value exceeds its cost. Two things are priced: the market shock and a mean-reverting short rate. Expected returns come from the projects' market exposure and from the growth option's exposure to the rate.
- *GS21.* Output is $Y_{it} = e^{x_t + z_{it}} k_{it}$, with aggregate productivity $x_t$ and firm productivity $z_{it}$ both mean-reverting. Cash flow to equity is $(1-\tau)(Y_{it} - (\delta + b_{it}) k_{it})$ less investment, $b_{it}$ the coupon per unit of capital. Firms invest in lumps at a random cost and may default. Aggregate productivity shocks are priced at $\gamma_x = 0.5$.
- *KP14.* Firm value is assets in place plus growth options. Project $j$ of firm $i$ pays $C_{ijt} = x_t \varepsilon_{it} u_{jt} K_j^{\alpha}$ per unit time, with $x_t$ aggregate productivity, $\varepsilon_{it}$ the firm's profitability and $u_{jt}$ the project's own, both mean-reverting around 1, and $\alpha = 0.85$. Projects arrive at a firm-specific rate and die at rate $\delta$. Two shocks are priced: aggregate productivity and embodied technology.
- *The added risk factor $y$* (sections 1 and 3). A mean-reverting process with unit variance and mean reversion 0.35 a year, on a grid of 21 points in BGN and 9 in GS21 and continuous in KP14; its shocks are priced at $\gamma_v = 2.5$ unless stated. A firm's cash flow is the model's own cash flow (the project cash flow $C_{jt}$ in BGN and $C_{ijt}$ in KP14, output $Y_{it}$ in GS21) times $e^{b y_t}$, with $b$ the firm's loading, and growth options are valued on the same $y$. In GS21 the coupon per unit of capital $b_i$ is drawn at birth, uniformly from 16 grid points in [0.1, 0.9], and kept for life, so cash flow to equity is $(1-\tau)(Y_{it} e^{b y_t} - (\delta + b_i) k_{it})$ less investment; $\tilde b_i = (b_i - 0.5)/0.29$ is the coupon standardised across firms. How much of the shock reaches a firm's value then depends on one state the model already has: in BGN and KP14 the share of value in assets in place rather than growth options, which book-to-market reads; in GS21 output over equity cash flow, which rises with the coupon and market leverage reads.
- *Baselines.* `bgnbase`, `gsbase` (the paper's Table I calibration) and `kpbase`: each model as published, one type, no added factor.

## 1. A different distribution of shocks: a crash-only factor

One type with one loading per model (`bvy1j`, `gvyz10j` and `vy1j`: the 3a economies with the price on ordinary shocks set to zero): 0.035 in BGN, $0.1\,\tilde z_{it}$ in GS21 ($\tilde z$ the firm's standardised productivity) and 0.07 in KP14. The factor's only priced event is a crash. Each month, with probability 1% whatever $y_t$, $y$ jumps to $-3.5$ standard deviations (the lowest grid point; in KP14 a Poisson event at rate 0.12 a year), and the pricing rule weights that event by $e^2$, about 7% a month under the pricing probabilities; $y$ then recovers at its ordinary speed.

<!-- TABLE:g1 -->
<!-- /TABLE:g1 -->

**Verdict: no gap.**

- A crash changes the distribution of a shock that every firm bears in the same proportion. The cross-section of premia it creates is a sort on the one firm state that scales the exposure, and the linear portfolio takes nearly all of it.
- Room is 12% of the best linear in BGN, 3% in GS21 and 4% in KP14. Gaps +0.000, −0.005, −0.006.
- Adding the same crash to the product economies instead (section 5) leaves room and gap where they were in BGN and GS21 and halves KP14's.

## 2. A different price of risk: quadratic in the model's own state

The price of each model's own aggregate risk is multiplied by $1 + 0.25 s^2$, capped at 3, with $s$ a standardised aggregate state:

- `gq25` (GS21): $s$ is productivity $x$; the cap is 1.5.
- `kpq` (KP14): the model's own states are random walks, so $s$ is an unpriced mean-reverting state (mean reversion 0.35, unit variance) that no firm is exposed to.
- `bq25` (BGN): $s$ is the short rate, with the multiplier binned to seven equally likely rate levels (1.00, 1.03, 1.17 and 1.68 from the middle outward) so that project values keep a closed form; the cash flow at date $k$ is priced at the rate level one period before it pays.

The multiplier never falls below 1 because KP14's growth-option discount turns negative once the price of productivity risk drops below about 0.82 of its value.

<!-- TABLE:g2 -->
<!-- /TABLE:g2 -->

**Verdict: no gap.**

- A price of risk that moves with the aggregate state rotates the premium across states and leaves it linear in the ranks within each state.
- Room is under 1% of the best linear in GS21 and KP14, and 14% in BGN, where the rate is itself priced, so a price that rises with its distance from the mean rotates the premium with it. Gaps +0.003, −0.004, −0.002.
- The regime-switching prices tried earlier (section 5) do the same. BGN's fast regime gave the group's one small positive number (+0.008), which a linear portfolio in the ranks times the state takes.

## 3. Different exposures to aggregate risk

Every economy in this section is the baseline plus the priced factor $y$ of the front matter, with the loading set by a type.

- **Types.** Each firm draws a type $f$ at birth, independently of everything else, with equal shares across the types listed in each setup. The type fixes the firm's loading $b_f$ and its cash-flow level for life. The cash flow carries only $e^{b_f y_t}$; the within-firm state does the rest.
- **The three designs.** 3a: one type, the control. 3b: types with loadings of both signs, mean zero across firms. 3c: the loading is a U in the type's position on a grid of scales, zero in the middle and positive at both ends.
- **Compensation.** Where types differ, type $f$'s cash-flow level is multiplied by $\theta_f = (V_0/V_f)^{\omega}$, with $V_f$ the value at $y = 0$ of a unit project of type $f$. At $\omega = 1$ value at $y = 0$ does not depend on the type. $V_f$ falls in $b_f$ (under the pricing probabilities $y$ is expected to be negative), so at $\omega > 1$ size rises with $b_f$ and at $\omega < 1$ it falls. The exponent sets how visible the type is through size; 3b reports the gap across exponents.

### 3a. Exposure that a single characteristic captures

One type, so exposure differs across firms only through the within-firm state: the assets-in-place share in BGN and KP14, the coupon in GS21.

<!-- TABLE:g3a -->
<!-- /TABLE:g3a -->

**Verdict: no gap.**

- An exposure that only rises or only falls in one firm state, however curved, is captured by the rank of the characteristic that reveals it.
- Room is 10% of the best linear in BGN, 19% in GS21 and 5% in KP14. The gap is zero or negative in all three.
- The same holds for every other single-characteristic economy in section 5, including ones with far more room. Room is necessary, not sufficient.
- All three estimators track the truth (figure).

![Portfolio weights along the 3a state: book-to-market in BGN and KP14, market leverage in GS21. Each method's per-firm weight is fitted by a small neural network on the ranked characteristics, pooled over ten seeds and the 125 evaluation months and standardised within each month; each curve is the average fitted weight at each value of the characteristic on the axis. The truth is the weight of the investor who knows the true moments; DKKM is shown at its best penalty. The true weight moves steadily along the state in BGN and GS21, and FF and DKKM sit on it in both, FM in BGN only; in KP14 the true weight is flat in book-to-market and the three estimated ones tilt toward value anyway.](mlp_figures/sec3/fig_3a.png)

- **BGN** (`bvy1ch`). *Model.* Project $j$ pays $C_{jt}\,e^{0.035\,y_t}$ each month it is alive. A firm's exposure is 0.035 times the share of its value in live projects. *Result.* Best linear 0.28 to 0.33, room 10%. DKKM 0.283 against 0.282: gap +0.002 (t 0.8, seven of ten). *Reading.* Book-to-market reads the share of value in live projects, so a sort on it holds the premium.
- **GS21** (`gvylf10`). *Model.* Cash flow to equity is $(1-\tau)\big(Y_{it}\,e^{0.1\,y_t} - (\delta + b_i)\,k_{it}\big)$ less investment: one loading of 0.1 on the whole of output. Equity's exposure is $0.1\,Y_{it}$ over equity cash flow, which rises with the coupon. *Result.* Best linear 0.42 to 0.95 (leverage carries a premium of its own), room 0%. DKKM 0.752 against 0.760: gap −0.007 (t −5.8, none of ten). *Reading.* Market leverage reads the coupon. With the coupon written into the exponent instead, $0.1 \tilde b_i y_t$ (`gvylqs10`, section 5), the exposure is convex in the coupon, room is 11 to 29%, and the gap is still +0.000.
- **KP14** (`vy1`). *Model.* Project $j$ of firm $i$ pays $C_{ijt}\,e^{0.07\,y_t}$ per unit time. A firm's exposure is 0.07 times a weight that falls with its growth-option share. *Result.* Best linear 0.35 to 0.39, room 5%. DKKM 0.355 against 0.360: gap −0.005 (t −2.9, one of ten). *Reading.* As in BGN; the growth-option share is one number per firm that size and book-to-market read.

### 3b. Exposure as a product of two attributes

Types of equal share with loadings $\pm b_1, \pm b_2, \pm b_3$ ($\pm b_1, \pm b_2$ in GS21), mean zero across firms. The sign of $b_f$ is an industry attribute that no characteristic carries directly; how far size reveals it is set by the compensation exponent. The one-sided versions and the other levers tried are in section 5.

<!-- TABLE:g3b -->
<!-- /TABLE:g3b -->

**Why a product has room.**

- A linear portfolio in the ranks holds any sum of functions of single characteristics. The product of a loading that is mean zero across firms and a within-firm state has no such part, so no linear portfolio is long the exposure. DKKM's random features span products, and DKKM recovers part of the difference as a gap.
- The sign does not have to be hidden. With $\omega \neq 1$ size reveals $b_f$, and the gap grows with the exponent: BGN +0.026 at 1.2 and +0.071 at 3.5; KP14 +0.116 at 0, +0.142 at 0.6, +0.181 at 1.2, +0.185 at 1.8 (section 5). The product then reads as size times book-to-market, a product of two observed characteristics that the random features span and the ranks do not.
- What is needed is the mean-zero loading. One-sided versions, in which every type loads with the same sign, are a sort on the state's characteristic with a type-dependent slope: room 10 to 27% and no gap (section 5).
- In the estimated weights (figure) only DKKM's curves cross.

![Weights as a function of the within-firm state at five size quantiles (size reveals the type): book-to-market in BGN and KP14, market leverage in GS21. Rows are models; columns are the true weight, FM, FF and DKKM, fitted as in the 3a figure. A product of loading and state shows as curves of different slope for small and large firms. In BGN the large firm's weight falls in book-to-market while the small firm's rises; in GS21 the large firms (the positive loadings) carry a high weight that falls in leverage while the small firms' is negative and flat; in KP14 the large firms' weight rises steeply from the low end of book-to-market and the small firms' falls at the high end. DKKM reproduces the pattern in all three; FM's curves are parallel or flat, so it holds the state's premium but not its product with the loading; FF's size-sorted steps carry most of it in GS21, a little in KP14 and none in BGN.](mlp_figures/sec3/fig_3b.png)

- **BGN** (`bvychs35`). *Model.* Project $j$ of a type-$f$ firm pays $C_{jt}\,\theta_f\,e^{b_f y_t}$, with $b_f$ = ±0.01, ±0.035, ±0.07 (six types) and $\theta_f = (V_0/V_f)^{3.5}$; price 2.5. Size ranks by type run 0.09 to 0.93 up the loadings and book-to-market ranks 0.90 down to 0.07, so both characteristics reveal $b_f$. *Result.* Best linear 0.22 to 0.38, room 67% of it. DKKM 0.403 against 0.332: gap +0.071 (t 5.8, ten of ten). Exponents 2.5 / 3.0 / 3.5 / 4.0 give +0.048 / +0.063 / +0.071 / +0.059. *Reading.* Industries on opposite sides of a priced shock; within an industry the firm whose value sits in operating assets moves with it and the firm whose value sits in growth options does not. Book-to-market reads the second factor and size part of the first; no single characteristic reads the product.
- **KP14** (`vyg25s`). *Model.* Project $j$ of a type-$f$ firm pays $C_{ijt}\,\theta_f\,e^{b_f y_t}$ per unit time, with $b_f$ = ±0.02, ±0.07, ±0.14 (six types, BGN's loadings doubled as in 3a) and $\theta_f = (V_0/V_f)^{1.2}$; price 2.5. Size ranks by type run 0.16 to 0.93 up the loadings and book-to-market ranks 0.84 down to 0.16. *Result.* Best linear 0.11 to 0.28, room 97 to 382% of it. DKKM 0.373 against 0.192 (Fama-French in eight seeds): gap +0.181 (t 17.2, ten of ten; +0.143 to +0.230). Without compensation +0.116 (t 11.1); at exponent 0.6 +0.142, at 1.8 +0.185; loadings doubled +0.245, tripled +0.218. Premium by type, in the order above: 0.56, 0.76, 0.86, 0.86, 0.84, 0.74% a month. *Reading.* As in BGN. The profitability version that held this slot (`vygelbcs`, +0.091) is in section 5.
- **GS21** (`gvylfs6i`). *Model.* Cash flow to equity of a type-$f$ firm is $(1-\tau)\big(Y_{it}\,\theta_f\,e^{b_f y_t} - (\delta + b_i)\,k_{it}\big)$ less investment, with $b_f$ = −0.1, −0.03, +0.03, +0.1 (four types) and $\theta_f = (V_0/V_f)^{1.2}$; price 6. Size ranks by type run 0.02, 0.13, 0.41, 0.80 up the loadings and leverage ranks 0.78, 0.77, 0.62, 0.23 down them (seed 0); the −0.1 type is thin at this price (8 firms on seed 0 against 63, 131 and 130). *Result.* Best linear 0.62 to 1.51 (the leverage premium, which the linear portfolio holds), room 12 to 18%. DKKM 1.282 against 1.259 (linear in the leverage levels in every seed): gap +0.023 (t 4.1, nine of ten; −0.002 to +0.065). At price 2.5 room is 3 to 5% and the gap +0.009 (t 2.1, five seeds); at price 9 room is 13 to 20% and the gap −0.004. Six types with ±0.2 at price 6, and the four types at exponent 3.5, did not solve (section 5). Premium by type: −0.54, +0.01, +0.34, +0.61% a month. *Reading.* Two industries on opposite sides of the shock; within each the firm with the larger fixed payment moves more with it. Size and leverage read the industry and leverage the amplification; the product is read by neither. The gap is an eighth of KP14's because the leverage premium puts the best linear at three to five times KP14's, so the same room is a small fraction of it and DKKM recovers less. The version with the coupon in the exponent (`gvyslzg6`, +0.028) is in section 5.

GS21's product without types, productivity times coupon written into production (`gvyp15`, +0.078), is the same mechanism without the industry sign. Leverage fixed by type does not work: with a few perfectly observed groups the linear portfolio prices the exposure directly (section 5).

### 3c. Exposure that is a U in one attribute

An exposure that only rises or only falls in a firm state is captured by that state's rank (3a). One that falls and then rises is not: a linear portfolio fits a line through the U and misses the curvature, which the random features see. The construction is the same in all three models.

- Each firm draws a type at birth, with equal shares, from five positions on a grid of scales (sixteen in GS21). $\tilde z_f$ is the type's position, standardised to mean zero and unit variance across types.
- The loading is $b_f = c\,\tilde z_f^2$: zero for the middle type and largest at the two ends, so the firms at both extremes carry the exposure and the firms in the middle none.
- The scale is what the characteristic reads. In BGN it is the firm scale (project cost, cash flows, book and value all scaled by the type, so only size moves). In KP14 it is the cash-flow level, a pure scale there as well: investment adjusts, and book-to-market and profitability do not move. In GS21 it is the coupon per unit of capital, which market leverage reads.
- In BGN and KP14 the five positions give $b_f$ = 0.35, 0.09, 0, 0.09, 0.35. The exposure itself adds to value (0.6 log units of size per unit loading in BGN, 14 in KP14, where investment amplifies it), so the scales are set so that the five types occupy the five fifths of the size distribution after that effect. In GS21 the coupon is drawn from 16 points and $b_f = 0.5\,\tilde b_f^2$ runs from 0 at the middle coupon to 0.95 at the ends.
- Compensation at exponent 1, price 2.5.
- A loading of one sign at both ends is what the earlier hump ($-c, 0, +c, 0, -c$) and the signed U ($\tilde b^2 - 1$) did not have; those are in section 5.

<!-- TABLE:g3c -->
<!-- /TABLE:g3c -->

![Weights along the attribute the U is written on: size in BGN and KP14 (five types, the exposed ones at both ends), market leverage in GS21, fitted as in the 3a figure. The true weight is a U in size in BGN, short the unexposed middle type and long both ends; in KP14 a hump with a dip at the middle, short both exposed ends; in GS21 a peak at low leverage that falls through the rest of the range. DKKM follows the shape in all three. FM fits a line through the hump in KP14 and through the leverage profile in GS21, and in BGN a shallower U through book-to-market, which the exposed types share.](mlp_figures/sec3/fig_3c.png)

- **BGN** (`bvyuq2`). *Model.* A type-$f$ firm's projects cost $s_f$ and pay $s_f\,C_{jt}\,\theta_f\,e^{b_f y_t}$, with the scale $s_f$ = 0.33, 0.61, 1.0, 1.49, 1.99, $b_f = 0.35\,\tilde z_f^2$ = 0.35, 0.09, 0, 0.09, 0.35 in the same order, and $\theta_f = V_0/V_f$; price 2.5. Size rank by type 0.18, 0.30, 0.52, 0.69, 0.86: the exposed types are the smallest and the largest. *Result.* Best linear 0.35 to 0.72, room 8 to 51%. DKKM 0.554 against 0.524: gap +0.030 (t 6.1, ten of ten; +0.010 to +0.062). Premium by type, in the order of size: 4.6, 1.2, 0.6, 1.2, 4.6% a month, a U in size. The exposed types have market betas of 1.5 against 0.65 to 0.75 and alphas of +1.0 against −0.4 to −0.8% a month. Exposure also raises book-to-market (0.97 against 0.78 to 0.81), profitability and momentum, so the ranks of the five characteristics explain 67% of the loading's variance across firms, book-to-market most; what they leave is the pairing of the smallest firms with the largest. The true weight by type, in standard-deviation units and in the order of size, is +0.32, +0.30, −1.18, +0.30, +0.30: short the unexposed middle and long every other type. DKKM's is +0.74, −0.05, −0.76, −0.31, +0.49 and FM's +0.55, −0.13, −0.36, −0.28, +0.24, a shallower U reached through book-to-market. *Reading.* Small firms exposed through financing constraints, the largest through operating leverage, the mid-sized through neither. The premium is a U in size, and a portfolio linear in the size rank holds a slope through it.
- **KP14** (`vyuq2`). *Model.* Project $j$ of a type-$f$ firm pays $C_{ijt}\,\theta_f\,\ell_f\,e^{b_f y_t}$ per unit time, with the cash-flow level $\ell_f$ = 0.38, 0.74, 1.0, 0.92, 0.58 (in the order of size; the exposed types' set below the middle type's to offset the size the exposure adds), $b_f$ = 0.35, 0.09, 0, 0.09, 0.35 and $\theta_f = V_0/V_f$; price 2.5. Size rank by type 0.14, 0.29, 0.53, 0.69, 0.89. The level moves neither book-to-market nor profitability. *Result.* Best linear 0.29 to 0.33, room 26 to 41%. DKKM 0.318 against 0.294: gap +0.025 (t 9.2, ten of ten; +0.015 to +0.042). Premium by type, in the order of size: 0.17, 0.41, 0.51, 0.41, 0.18% a month. The exposed types earn less, a hump in size: their market beta is higher (1.11 against 0.83 in seed 0) and their alpha lower (−0.21 against +0.23% a month). This is KP14's pricing, not the construction. In every KP14 type economy the premium peaks near a loading of zero and falls on either side (in 3b it runs 0.56 up to 0.86 and back to 0.74 across the loadings), while in BGN the exposed types earn the most. Exposure lowers book-to-market (0.006 against 0.34), so book-to-market reveals who is exposed and size which end. The true weight by type is −0.28, +0.38, −0.24, +0.38, −0.26: short both exposed types and long the lightly exposed ones beside them, because exposure adds variance faster than premium; along size a hump with a dip in the middle. DKKM's is −0.82, +0.67, +0.62, +0.51, −1.05, a hump, and FM's +0.29, +0.24, +0.31, −0.22, −0.63, a line down size. *Reading.* The smallest and largest firms carry the shock and the mid-sized do not. In KP14 the shock's premium is small next to the market premium, so the exposed firms, with the higher beta but the lower alpha, earn less. The exposure is a U in size and the premium a hump, and neither is a line in the size rank.
- **GS21** (`gvylqup5`). *Model.* Cash flow to equity of a type-$f$ firm is $(1-\tau)\big(Y_{it}\,\theta_f\,e^{b_f y_t} - (\delta + c_f)\,k_{it}\big)$ less investment, with $c_f$ the $f$-th of the 16 coupon points in [0.1, 0.9] (sixteen types), $\tilde b_f = (c_f - 0.5)/0.29$, $b_f = 0.5\,\tilde b_f^2$ (0 at the middle coupon, 0.95 at the ends) and $\theta_f = 1/W(b_f)$, where $W(e) = \sum_k e^{-rk}\,\mathbb{E}^Q[e^{e y_k}] / \sum_k e^{-rk}$ is the value under the pricing probabilities of the exposure stream per unit of a riskless annuity, so that value at $y = 0$ does not reveal $b_f$; price 2.5. *Result.* Best linear 0.65 to 1.34 (the leverage premium), room 7 to 18%. DKKM 1.138 against 1.100: gap +0.038 (t 4.3, ten of ten; +0.004 to +0.085). Premium across leverage quintiles, averaged over seeds: 9.7, 2.7, 1.5, 2.6, 6.6% a month, a U in leverage, with market betas 2.3, 0.55, 0.28, 0.51, 1.33. At twice the loading (`gvylqup10`) the gap is +0.072 (t 6.1) but the least levered quintile earns 28% a month and is twelve times the size of the next, which rules it out. The true weight across the quintiles is +0.15, +0.60, −0.16, −0.34, −0.25: long the second quintile most, while the least levered firms, whose premium is the largest, carry little weight because their beta (2.3) and volatility are the largest too; a peak at low leverage rather than a U. DKKM's is −0.35, +0.71, +0.19, −0.28, −0.27 and FM's −0.37, −0.16, −0.02, +0.16, +0.39, a line up leverage. *Reading.* Unlevered firms are credit rationed and highly levered firms in distress; both are more sensitive to the shock than the firm of middling leverage, and the premium is a U in leverage. The signed U (`gvylqun10`, +0.027), the humps, the U in a distance-to-distress index (`gvylqzu5g4`, +0.055) and the productivity hump (`gvyzzh10`, +0.080) are in section 5.

In each of the three the exposure is a U in the attribute the characteristic reads, and the ranks of the characteristics span a slope through it and not the curvature. The premium follows the exposure in BGN and GS21 and is a hump in KP14; either way it rises and falls in what the characteristics reveal.

## 4. Larger factor models

The three sections above change one feature of a structural model. This one asks whether more priced factors help, in a reduced form whose structure is fully controlled: a panel of firms with five characteristics, a market factor, and $K$ extra priced factors whose total price of risk is held fixed as $K$ grows. The only thing that varies across the rows is what the extra factors load on: a linear function of the observed characteristics, a product of two partly hidden firm states (section 3b's design), or the square of one firm state (section 3c's). The structural version, a second priced state in each model, is in section 5.

**The reduced form.** 500 firms over 500 months after a 100-month burn-in, five seeds per cell, scored exactly as the structural economies; its true conditional moments are known in closed form. $i$ indexes firms, $t$ months, $j$ the five characteristics and $k$ the extra factors.

- *Firm states.* Each firm carries five independent mean-reverting processes $z_{ijt}$ with persistence (0.99, 0.98, 0.90, 0.95, 0.70) and unit variance.
- *Observed characteristics.* $X_{ijt} = z_{ijt}$ in the kfh and kfo families. In the lfh family each characteristic is its state plus observation noise with standard deviation $\sigma_\nu$: 0 (rows n0) or 1, as large as the state itself (rows n1). Methods see the ranks of $X$, as everywhere in the report.
- *Returns.* $r_{i,t+1} = B_{it}\lambda + B_{it} f_{t+1} + \varepsilon_{i,t+1}$, with factor shocks of standard deviation 0.04 and firm noise 0.10, all independent; $B_{it}$ holds the firm's loadings and $\lambda$ the prices of risk.
- *The market factor.* Loading $1 + 0.3\,(\tilde{\mathrm{bm}} + \tilde{\mathrm{roe}} + \tilde{\mathrm{mom}})$ on the standardised characteristics, price 0.002 (a Sharpe of 0.05). It is linear in the characteristics, so every method holds it.
- *The $K$ extra factors.* Price $0.024/\sqrt{K}$ each, so the total price of risk is the same at every $K$. Every extra loading is centred across firms each month, so the equal-weighted market carries none of the extra premia. The families differ only in the loading:
  - *kfh*: a function of a fixed random direction in the characteristics ($h(u) = u$ in the main table; $\tanh$ and the square in section 5). Exposure is a function of what the characteristics show directly.
  - *kfo* (section 5): kfh with the size coordinate removed, so no extra factor loads on size.
  - *lfh*: the product of two of the five states, one pair per factor (so $K \le 10$), which the characteristics reveal only up to the observation noise: section 3b's design.
  - *lsq, lsqb, lsqr*: the square of one state (size, book-to-market or profitability), $z^2 - 1$, observed exactly (n0) or with noise (n1): section 3c's design.
- In the kfh families the best Sharpe falls at $K = 16$ because sixteen directions over five characteristics are collinear and partly cancel; in the lfh family it stays near 0.6 at every $K$.

<!-- TABLE:rf -->
<!-- /TABLE:rf -->

**Findings.**

- Factors that load linearly on observed characteristics leave no room at any $K$ and a negative gap.
- Factors that load on products of partly hidden states give room and a gap that grow with $K$ and with the noise that hides the states, to +0.29 at ten factors with noise. A product of two mean-zero states has no part a linear portfolio can hold, and each extra factor adds another product.
- A square is a U. Squaring book-to-market (lsqb) gives room +0.40 and a gap of +0.37 (t 11) with the state observed exactly, and +0.34 with noise. Squaring size (lsq) gives the same room and no gap: market value is $e^{\mathrm{size}}$, so the value-weighted market inside the Fama-French set holds size squared mechanically, and FF reaches 0.57 of a best Sharpe of 0.60. Squaring profitability (lsqr) gives the largest of the three: room +0.48, about four times the best linear, and a gap of +0.446 (t 11.9) with the state observed exactly and +0.457 (t 17.5) with noise, five of five each. Shifting the mean of the squared state, $(z + m)^2 - 1$, adds a linear part and sends the room to near zero at $m$ of 1 or more (section 5), as the shifted products do for the interaction.

**The structural version** (`vym3s`, `vym6s`; rows in the section 5 table): one set of mean-zero types spread over $K$ priced states. $K$ independent mean-reverting states, each priced $2.5/\sqrt{K}$; 24 mean-zero types of equal share with loading vectors of size 0.07 or 0.14 at cosines ±1, ±0.84, ±0.68, ±0.52, ±0.36, ±0.20 to the price vector; compensation exponent 1.2. Both are `vyg25s` with its types spread over $K$ states.

- The best linear is unchanged (0.191 and 0.192 against 0.174), but the best Sharpe falls from 0.575 to 0.503 and 0.507: a type whose loading is misaligned with the price vector carries the full risk of its size and earns only the cosine of the premium.
- Room is 0.224 and 0.225 (110% of the best linear against 201%), and DKKM takes the same 71% of the best nonlinear Sharpe as in `vyg25s`: 0.296 against FM's 0.134 and the linear portfolio's 0.196, a gap of +0.100 (t 21) and +0.099 (t 15), ten of ten at either $K$, the second largest of the campaign and half of `vyg25s`'s.
- $K = 3$ and $K = 6$ agree to the third digit. In the structural models, unlike the reduced form, more priced states do not multiply interactions, because every state acts through the one growth-option channel. The one-signed construction at $K = 6$ (`vym6`) reproduces `vym3` (room 19%, gap +0.000): $K$ itself is not a lever.

## 5. Additional economies

Everything else that was run. The table has the main tables' columns, and the bullets describe each economy against its parent. Three groups answer side questions: the crash added to the product economies (`bvyj25`, `gvyp10j`, `vyj25`), the GS21 product family (`gvyp05` to `gvyp20`, `gvyp10x`), and the earlier single-characteristic and price-of-risk designs that the harmonised cells replace.

<!-- TABLE:xa -->
<!-- /TABLE:xa -->

<!-- ITEMS -->
<!-- /ITEMS -->

**What they add to the main results.**

- *The grid's resolution is not what separates `bvych0` from `bvy25`.* On 41 points the gap is +0.009 against +0.008 on 21, and the difference from the Gaussian economy's +0.015 (sd 0.016) is within noise.
- *A level exposure of GS21 output to $y$* (`gvy1`, `gvy1j`) has no room at all, like every other level design in GS21; the productivity exposure (`gvyz10`) has 5% and the product with the coupon 43%.
- *The crash on a product economy* leaves room and gap at the control's in BGN (+0.007 against +0.008) and GS21 (+0.030 against +0.020, within noise) and halves KP14's (+0.008 against +0.014). The crash months are a state that a linear portfolio in the ranks times the state reads as well as DKKM does.
- *The GS21 product family.* Room is hump-shaped in the product's size (45 / 111 / 47 / 22 / 6% of the best linear at 0.05 / 0.075 / 0.10 / 0.15 / 0.20) while the gap is +0.019 / +0.077 / +0.020 / +0.078 / +0.022; above 0.10 the market becomes a hedge. A second nonlinear loading on $x$ (`gvyp10x`) gives +0.049: a second interaction factor adds. `bvy25`, the Gaussian version of `bvych0`, has twice its gap at the same room, and the visibility sequence (`bvy25`, `bvyc0`, and `bvyc10` at exponent 1.0: +0.015 / +0.010 / +0.010) is flat within noise.
- *Two apparent failures of the mechanism are estimation cost, not structure.* `vyx` is `vyg25` at a lower price of $y$: its room (0.054) is about what DKKM gives up to estimation relative to the linear portfolio (0.053), so the gap is zero, and the higher price lifts the room more than the cost. `bvysg` has twice `bvy25`'s room but DKKM's estimation loss is two and a half times larger (0.16 against 0.06): with signed exposures the nonlinear part of the best portfolio is a direct bet on the persistent state $y$, whose sample mean a 360-month window barely pins down. `bvychs`, six signed types with compensation, has a positive gap of +0.026 at room 20%, so the loss in `bvysg` came with the visible type, as in `vyg25u`. What matters is room net of DKKM's estimation cost. Four more GS21 designs (investment cost, floating coupon, a larger option, state-dependent recovery) were stopped at the true-moments stage with room below +0.01; `gvyib2`, the investment cost made to depend on leverage times $y$, has room 0.001 and a gap of −0.003 (t −11, zero of ten); and GS21's default channel is dormant at every calibration tried.
- *Knobs that are not mechanisms.* The speed of the priced state (mean reversion 0.15 / 0.35 / 0.70 in KP14: +0.007 / +0.014 / +0.009, a mild hump within noise), the visibility of the type, the number of types (`vym3` against `vym3t3`), the sample size (`vym3` at 1000 firms: room up, gap flat), and the penalty grid (DKKM +0.0001). The price of the interacted component is the one scaling lever (`vyx` to `vyg25`).

### Two priced states

A second independent priced state, with the total price held at the one-state value ($2.5/\sqrt{2}$ on each). In BGN and GS21 three types span the two states with loading vectors in the pattern (1, 0), (0, 1), (1, 1)/$\sqrt{2}$, at the 3b exposure sizes; in GS21 each type carries the 0.10 product exposure to its state. In KP14 the cell is the nested extension of 3b: every firm draws a type for each state independently from the 3b loadings (0.02, 0.07, 0.14), nine types in all, so exposure to each state is its own type times the growth-option share, and zeroing the second set of loadings recovers `vyg25`. The three-direction KP14 version (`vym2t3`), the nested version with each state priced at the full 2.5 (`vy9t2f`) and the nested BGN version (`bvy9t2`) are in the table above.

<!-- TABLE:g4 -->
<!-- /TABLE:g4 -->

**What changes.** Total price fixed, $2.5/\sqrt{2}$ on each state.

- `bvyk2`: `bvych0` with a second independent 21-point grid $y^{(2)}$; project cash flows $C_j e^{b_f y_t + b_f^{(2)} y^{(2)}_t}$, $(b_f, b_f^{(2)}) \in \{(0.01, 0), (0, 0.035), (0.049, 0.049)\}$.
- `gvypk3`: `gvyp10` with a second independent grid (5 points each, 25 joint) and three types of equal share, loading $0.10\,\tilde z \tilde b$ on $y$, on $y^{(2)}$, and $0.071\,\tilde z \tilde b$ on both.
- `vy9t2`: `vyg25` with two independent mean-reverting states; a firm's type is a pair $(f_1, f_2)$ drawn independently from \{0.02, 0.07, 0.14\} for each state, nine types of equal share, cash flows $e^{b_{f_1} y_t + b_{f_2} y^{(2)}_t}$.

**Verdict: more factors are a lever only where the second state brings a second interaction.**

- In BGN the second grid raises room by a quarter and doubles the gap (+0.015, ten of ten).
- In GS21 it raises the product economy's gap from +0.020 to +0.056 (nineteen of twenty seeds), as the two-type version did.
- In KP14 it does not help however it is wired. The nested nine-type cell lowers room from 24% to 15% of the best linear and the gap from +0.015 to +0.005; the three-direction version (`vym2t3`) keeps the room (28%) but halves the gap (+0.008); pricing each state at the full 2.5 (`vy9t2f`) only returns the gap to 3b's (+0.015, nine of ten) at lower room (19%). The reason is in the ceilings: with an independent second type a firm's total exposure is the sum of two loading draws, six evenly spread values that size and momentum track almost linearly, so the best linear rises more than the best nonlinear (0.346 to 0.387). Both states act through the same channel, the growth-option share, so the second state adds premium and a factor but not a second interaction.
- BGN's nine-type version (`bvy9t2`, +0.012, room 16%) sits between 3b and the three-direction cell. Six states with `vym3`'s construction (`vym6`) reproduce `vym3` to the third digit, and the signed types over $K = 3$ and $K = 6$ states (`vym3s`, `vym6s`, section 4) give the same room and gap at either $K$. The reduced-form table of section 4 says the same with the structure controlled.

### Reduced form: bounded and squared loadings, the sign construction and the shifted squares

The RF rows of the section 5 table are the kfh family with bounded ($\tanh$) and squared loadings on a random index of the characteristics, the same squares with the size coordinate removed (kfo), the sign construction, and the shifted squares of one state. The squares of one state at mean zero are in the main reduced-form table of section 4.

- *Bounded loadings behave like linear ones.* No room at $K$ = 1 and 4, a little at $K$ = 16 (12%), and a negative gap throughout, because $\tanh$ of a single index only rises in that index and is nearly linear in its rank.
- *The square of a random index* gives room of about five times the best linear and a gap of about +0.5 at $K = 1$. In the unconstrained square rows the gap then falls to +0.12 and +0.14 at $K$ = 4 and 16 while the room does not move: the squared index contains size squared, Fama-French's legs are value-weighted and market value is $e^{\mathrm{size}}$, so those legs hold a size-squared exposure mechanically and Fama-French's Sharpe jumps from 0.05 to 0.46. The kfo rows remove the size coordinate from every direction, and the gap returns to about +0.5 at every $K$.
- *The sign construction* (lsh rows) shifts the mean of one or both of the two latent factors in the $K = 1$ product by 0, 1, 2 or 3 standard deviations, and lrt replaces the product by a one-signed ratio of the two (a debt-service coverage ratio with a floor). The mean-zero product has room of +0.39 (no noise) and +0.42 (noise 1.0) and gaps of +0.031 and +0.093, the $K = 1$ latent-product cell of the main reduced-form table. Every shifted cell and the ratio have room of 0.000 to 0.005 and a gap of −0.005 to −0.014, at the same best Sharpe of 0.598. The share of the loading's variance that a linear function of the latent states explains goes from 0.01 at no shift to 0.79 at a two-sd shift of one factor and 0.89 at two-sd shifts of both. Room in a one-factor economy is therefore all or nothing: a linear portfolio with any correlation with the exposure mimics the factor, diversifies away the rest and takes its whole Sharpe, and only an exposure uncorrelated with every linear function of the characteristics leaves the factor to the nonlinear method. This is what the sign-switching economies of section 3b test.
