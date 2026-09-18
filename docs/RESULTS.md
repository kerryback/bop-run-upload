# Experimental results

Every economy run through the oracle-and-estimator pipeline: what it was built to test, what it
produced, and what it decided. One measurement protocol governs all of them, stated once below, so
that the only thing separating two rows of a table is the economy. Every number here comes from the
protocol campaign of 2026-09-15 to 2026-09-17: thirteen economies, ten seeds each, 130 tasks across
Sol and Phoenix (`docs/RUNS.md`).

The tables are checked cell by cell against `variants/results/economy_table.csv`
(`variants/aggregate_seeds.py`, which carries the fair benchmark too since 2026-09-17) by
`tests/test_results_md_matches_table.py` -- every table that uses the column set below, not just one
section -- so this file cannot fall behind the numbers without the suite saying so. Where each job
ran: `docs/RUNS.md`. What to run next, and why: `docs/NEXTUP.md`.

## The measurement protocol

Results differ along two axes and only one of them is interesting. An economy differs from another
economy in its **parameters** and its **driving forces** -- a price of risk, a regime, a priced
factor, a set of exposure types. It must not differ in how many firms were simulated, how long the
burn-in was, how many months were evaluated, or how fine any grid was. Those choices are fixed here,
once, for every economy in this file:

| quantity | value | where |
|---|---|---|
| firms, N | 500 | `variants/common/protocol.py` |
| retained months, T | 500 | same |
| burn-in | 400 months, all three models | same; the literals live in each model's parameter module |
| estimation and evaluation window | 360 months | same |
| **evaluated months** | **125** = T - 15 - window | derived, never set: `run_oracle.py` keeps T - 15 months and the rolling estimator consumes `window` more |
| seeds | 0-9, every economy | same |
| ridge-penalty grid, DKKM and the four classical linear methods | `1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10` | same |
| ridge-penalty grid, the market-augmented fair methods (`linrank_m`, `linlev_m`) | the same seven plus an unpenalised column and `100`, `1000` -- deliberately two decades past DKKM's, so the fair benchmark is never the one that runs out of shrinkage | `variants/run_estimators.py` |
| random-feature counts | 36, 360, 3600, over 2 independent draws | same |
| bandwidth grid | `np.arange(0.5, 1.1, 0.1)`, 7 values | same |
| conditioning columns | the model's **full** set, every economy including the baselines | same |
| benchmark | `--fair_linear --include_mkt --levels`, no winsorisation | each spec's `estimation` block |

Solve-side precision cannot be shared -- the three papers have different solvers -- so what is
required of it is that it be identical across every economy of a given model. It is, and every
manifest in `experiments/registry/` records it:

| model | solve precision, identical across its economies |
|---|---|
| KP14 (`kp_vy`) | 21 y-nodes at `y_max` 3.5 with a byte-identical transition matrix; the G stage a direct sparse solve on a 1000-point epsilon grid, gated at a relative residual of 1e-6 and achieving about 1e-12; CIR integrals by adaptive quadrature at `epsrel` 1e-6 with the density's mass verified to 1e-8 |
| GS21 (`gs_bx`) | value-function iteration on a 161-point x grid, 200-point z grid, 20-point debt grid, exact-Gaussian Tauchen at +-4 sd, fixed-point `tol` 1e-6, 161-node quadrature for the default-smoothing shock |
| BGN (`bgn_gam`) | 100-node Gauss-Laguerre for the project-beta integral and 100-node Gauss-Hermite for the rate shock; the J* table by adaptive bisection to `tol` 3e-4, converging to 161 grid points in all six |

Every claim above is pinned by `tests/test_protocol_is_uniform.py`: each model's burn-in literal, the
seed array's sample and grid, every live spec's `panel` and `estimation` block, and the uniformity of
solve precision within each model. All 130 seeds of the campaign record `spec_check`, `env_check` and
`readback` verified (`readback` reads "not requested" for the four GS21 economies, which deliberately
leave `GS_SIM_OVERRIDES` unset), the protocol-v2 spec id, burn-in 400 recovered from the month
indices, 485 retained months and `eval_window` 360.

## Reading the tables

**The protocol.** Each estimator is fit on a 360-month rolling window of simulated data, and the
portfolio it produces is scored against the economy's TRUE conditional moments, month by month, over
the 125 evaluation months of a 500-month panel of 500 firms. An estimator is never judged on its own
realized returns, only on what its weights were worth. Ten seeds per economy; every figure is mean
(sd across seeds) unless the entry says otherwise.

**Units.** Every Sharpe ratio in this file is monthly, not annualised. In each evaluation month a
portfolio's Sharpe is its conditional expected excess return divided by its conditional volatility,
both from the economy's true moments. A table reports the mean over the evaluation months, then the
mean over seeds. A difference of two Sharpe ratios is in the same units. A percentage is the ratio of
two ten-seed means, times 100.

**The columns of every economy table.** Each table of economy results in this file has these fourteen
columns, in this order.

- **economy** -- the model directory and run tag, `model/tag`, sometimes followed by what sets the
  economy apart.
- **seeds** -- how many independently simulated panels the row averages. Always 10.
- **SR_max** -- the largest conditional Sharpe any portfolio of the firms could attain, sqrt(mu'
  Sigma^-1 mu), averaged over the evaluation months. An upper bound on every other Sharpe in the row.
- **EW market SR** -- the Sharpe of the equal-weighted market portfolio.
- **FMR SR** -- the Sharpe of the rolling Fama-MacBeth regression portfolio: five zero-net
  characteristic legs plus the equal-weighted market, combined by an unshrunk MVE on the estimation
  window. It holds a market but must estimate its weight (finding 8, finding 10). The benchmark of both
  percentage columns.
- **best linear SR** -- in each seed, the Sharpe of the best of four linear methods: Fama-MacBeth,
  Fama-French, and ridge regression on rank-standardised characteristics with and without level
  features.
- **DKKM SR** -- in each seed, the Sharpe of the best random-feature ridge estimator, over 36, 360 and
  3600 features, the penalty grid, and its four variants. A maximum over that grid: see the gate above.
- **DKKM - FMR** -- DKKM SR minus FMR SR. An absolute difference in Sharpe units; the standard
  deviation across seeds is in parentheses.
- **(DKKM - FMR) / FMR** -- [(DKKM SR) - (FMR SR)] / (FMR SR), in percent. Where Fama-MacBeth's Sharpe
  is near zero, as in the slow and rare BGN regime economies (g0235s, g0235r), this is very large and
  says nothing about DKKM.
- **DKKM - best linear** -- DKKM SR minus best linear SR. Absolute, sd in parentheses. The "gap" in
  this file's prose.
- **DKKM - best fair linear** -- DKKM SR minus the Sharpe of the best of seven linear methods: the
  four above, the two ridge methods given the equal-weighted market as a separate unpenalised column
  as DKKM has it, and the market alone with its weight estimated. Absolute, sd in parentheses. The
  "fair gap" in the prose, and the complexity gap outside KP14 (finding 8). The always-long market
  (`ew`) is reported beside it and excluded from it: always-long assumes the premium's sign, which no
  estimator is given.
- **room** -- the population headroom for nonlinearity over the evaluation months: the best Sharpe a
  nonlinear feature basis reaches with one fixed coefficient vector, minus the best a rank-linear
  basis reaches the same way, both computed from the true moments. No estimation involved. Absolute,
  sd in parentheses.
- **room / FMR** -- (room) / (FMR SR), in percent.
- **t, DKKM vs FMR** -- the largest paired t-statistic, across the evaluation months, of any
  random-feature estimator's monthly Sharpe against Fama-MacBeth's, averaged over seeds.

**Two cautions.**

1. **Room is not a ceiling on the gap and not a screen for it.** The gap exceeds the room in every
   BGN and GS economy, because there it is the market shortfall; in KP14 it is 35% (vyx) and 40%
   (vyg25) of the room. Ranking candidates by room does not rank them by gap.
2. **Every percentage is a RATIO OF MEANS, not a mean of per-seed ratios.** In g0235, (DKKM - FMR) /
   FMR is 49.4% as the ratio of the ten-seed means and rather larger as the mean of per-seed ratios,
   because Fama-MacBeth's Sharpe is small and dispersed. In g0235r and g0235s it is below zero in some
   seeds, where a per-seed ratio has no meaning at all. Both versions live in `economy_table.csv`
   (`gap_fm_over_fm`, `gap_fm_pct_fm_mean`).

## The answer so far

**A complexity gap exists in one family of economies, and only there.** In Kogan and Papanikolaou
(2014) with a priced, mean-reverting aggregate state that heterogeneous firm types load on, the
random-feature ridge estimator of Didisheim, Kelly, Kozak and Malamud beats the best linear method by
+0.125 to +0.164 of Sharpe, and Fama-MacBeth by 20% to 22% of Fama-MacBeth's own Sharpe, at t of 36
to 43, in every seed of both economies. Those two figures are LOWER bounds: the ridge grid's floor
wins in 7 and 6 seeds of ten. The gap survives the fair benchmark unchanged, because on this path the
linear methods beat the equal-weighted market by 0.28 and 0.33 and DKKM beats them by a further 0.125
to 0.164.

**In the other eleven economies the fair gap is between -0.008 and +0.004.** That includes all three
models as published, whose fair gaps are +0.0019 (BGN), -0.0016 (KP14) and -0.0028 (GS21).

| economy | seeds | SR_max | EW market SR | FMR SR | best linear SR | DKKM SR | DKKM - FMR | (DKKM - FMR) / FMR | DKKM - best linear | DKKM - best fair linear | room | room / FMR | t, DKKM vs FMR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bgn_gam/g0235r | 10 | 0.0554 | 0.0466 | 0.0121 | 0.0230 | 0.0352 | +0.0232 (0.0183) | 191.6% | +0.0123 (0.0268) | -0.0062 (0.0143) | +0.0030 (0.0018) | 24.8% | 43.9 |
| bgn_gam/g0235s | 10 | 0.0700 | 0.0573 | 0.0223 | 0.0397 | 0.0495 | +0.0272 (0.0163) | 121.8% | +0.0097 (0.0328) | -0.0080 (0.0261) | +0.0042 (0.0059) | 18.9% | 63.0 |
| bgn_gam/g0235 | 10 | 0.1509 | 0.1097 | 0.0741 | 0.0865 | 0.1107 | +0.0366 (0.0153) | 49.4% | +0.0243 (0.0056) | +0.0008 (0.0013) | +0.0203 (0.0127) | 27.4% | 31.8 |
| bgn_gam/g0235d | 10 | 0.3257 | 0.0818 | 0.2321 | 0.2922 | 0.2923 | +0.0602 (0.0390) | 25.9% | +0.0000 (0.0065) | -0.0003 (0.0061) | +0.0241 (0.0049) | 10.4% | 19.7 |
| bgn_gam/g0235f | 10 | 0.1950 | 0.1201 | 0.1060 | 0.1114 | 0.1312 | +0.0252 (0.0116) | 23.8% | +0.0199 (0.0080) | +0.0034 (0.0033) | +0.0303 (0.0131) | 28.6% | 15.0 |
| kp_vy/vyg25 | 10 | 1.3887 | 0.4694 | 0.7911 | 0.8004 | 0.9640 | +0.1729 (0.0252) | 21.9% | +0.1636 (0.0252) | +0.1636 (0.0252) | +0.4108 (0.0446) | 51.9% | 43.2 |
| gs_bx/gx7 | 10 | 0.4851 | 0.4725 | 0.3928 | 0.4497 | 0.4731 | +0.0803 (0.0949) | 20.4% | +0.0234 (0.0179) | +0.0003 (0.0007) | +0.0082 (0.0066) | 2.1% | 19.1 |
| kp_vy/vyx | 10 | 1.1778 | 0.3675 | 0.6389 | 0.6432 | 0.7683 | +0.1295 (0.0306) | 20.3% | +0.1251 (0.0302) | +0.1251 (0.0301) | +0.3606 (0.0394) | 56.4% | 36.2 |
| gs_bx/g28 | 10 | 0.3144 | 0.2978 | 0.2544 | 0.2694 | 0.3028 | +0.0484 (0.0415) | 19.0% | +0.0334 (0.0261) | -0.0024 (0.0018) | +0.0006 (0.0005) | 0.2% | 27.6 |
| gs_bx/bx7 | 10 | 0.3270 | 0.3061 | 0.2822 | 0.2920 | 0.3091 | +0.0268 (0.0267) | 9.5% | +0.0170 (0.0151) | -0.0018 (0.0019) | +0.0073 (0.0050) | 2.6% | 17.1 |

Ten seeds each, N=500, T=500, burn-in 400, window 360, 125 evaluation months; columns as defined
above. **Ranked by (DKKM - FMR) / FMR**, the ratio of the ten-seed means. Read the caution below the
table before drawing anything from the top of it.

**Why the KP14 route works.** Three things are true of it at once, and of no other economy here.
First, firms differ in their exposure to a priced shock: three types load their cash flows on the
state y as exp(beta_f y), with beta 0.02 / 0.07 / 0.14, and y's innovations carry a price of risk.
Second, the state bends that exposure map rather than shifting it: y is mean-reverting, so the value
of an exp(beta y) stream depends on where y is, and a compensation term keeps firm value from
revealing the type monotonically at y = 0. Third, the bending is confined to RANK AND INTERACTION
space: the exposure channel does not also show in raw levels, where a cheap rolling regression would
contest it. The result is a population room of +0.36 to +0.41 over the evaluation months that
only a nonlinear basis can reach, and an equal-weighted market that reaches only 31% to 34% of the
attainable Sharpe. Neither half is enough alone, and the baselines show it: the KP14 baseline's market
also carries only 35%, but with one firm type its room is +0.0044 and the linear methods take what
the market leaves. BGN as published has native room, +0.0267, and a market at 51%, and its fair gap is
+0.0019: room of under three hundredths is not enough.

**What the gap is not.** It is not the market portfolio, which DKKM appends to its features
unpenalised: here the market has Sharpe 0.37 and 0.47 against an attainable 1.18 and 1.39, the linear
methods beat it by 0.28 and 0.33, and giving them the market exactly as DKKM has it leaves the gap
within 0.0001 of itself. It is also not a calibration. The oracle's mean expected return is 18.2% a
year with a cross-sectional sd of 7.7% in vyx, and 22.9% and 10.6% in vyg25; these economies locate
where the mechanism is, not where a referee would accept it. Every other economy here sits between
3.3% and 12.6% a year.

## The three models as published

Each paper's economy, before any parameterization was built on it. Every "what the parameterization
added" statement in this file is a difference against these rows.

| economy | seeds | SR_max | EW market SR | FMR SR | best linear SR | DKKM SR | DKKM - FMR | (DKKM - FMR) / FMR | DKKM - best linear | DKKM - best fair linear | room | room / FMR | t, DKKM vs FMR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bgn_gam/bgnbase: BGN as published | 10 | 0.2830 | 0.1449 | 0.1950 | 0.2109 | 0.2175 | +0.0226 (0.0285) | 11.6% | +0.0066 (0.0096) | +0.0019 (0.0043) | +0.0267 (0.0059) | 13.7% | 14.5 |
| kp_vy/kpbase: KP14 as published | 10 | 0.2292 | 0.0797 | 0.1930 | 0.2020 | 0.2081 | +0.0151 (0.0105) | 7.8% | +0.0061 (0.0081) | -0.0016 (0.0060) | +0.0044 (0.0011) | 2.3% | 15.8 |
| gs_bx/gsbase: GS21 as published | 10 | 0.2926 | 0.2785 | 0.2650 | 0.2689 | 0.2835 | +0.0185 (0.0106) | 7.0% | +0.0146 (0.0079) | -0.0028 (0.0019) | +0.0012 (0.0001) | 0.4% | 24.4 |

What the anchor says:

- **No model as published has a complexity gap.** The fair gaps are +0.0019, -0.0016 and -0.0028. In
  GS21 a linear method given the market beats DKKM in eight seeds of ten; in KP14 in four; in BGN
  DKKM is ahead in seven by amounts averaging +0.0019. The measured gaps, +0.006 to +0.015, go once
  the linear methods hold the market as DKKM does. All three rows are clean on the penalty gate, so
  none of this is a grid artifact.
- **DKKM leaves the market wherever the market leaves Sharpe, and the linear methods follow it.** In
  the BGN and KP14 baselines the market carries 51% and 35% of the attainable Sharpe and DKKM is
  0.073 and 0.128 above it; in GS21, where the market carries 95%, it is 0.005 above. Non-market
  Sharpe of 0.21 to 0.24 without nonlinear room gives no gap.
- **Room alone is not enough and a low market share alone is not enough.** BGN as published has room,
  +0.0267, and a market at only 51%, and a fair gap of +0.0019. KP14 as published has a market at 35%
  and room of +0.0044. Only an economy with both -- KP14 Path 1's +0.36 of room and a market at 31% --
  produces a gap.

**How each baseline is produced.**

| economy | spec | what it is in the code | solve |
|---|---|---|---|
| bgn_gam/bgnbase | var-bgn_gam-bgnbase-v2 | `bgn_gam` at gmult [1, 1]: both regimes price the market shock at sigma_z 0.4 and the regime is inert; reproduces the paper's economy to machine precision. Table I: 11 of 11 parameters match | jstar `510d48d43c4763cb`, Mac, table committed |
| kp_vy/kpbase | var-kp_vy-kpbase-v2 | `kp_vy` with one type at beta 0, gamma_v 0, bv_comp 0: nothing depends on the state y (the 21 integral tables agree across y to 4e-12); r = 0.05 as in every KP14 economy here, the paper's 0.025 being the one standing departure. Table II: 17 of 18 match | G `7bc1f92a225c01b6` and integ `f299747cd77fc4a1`, Mac, tables committed |
| gs_bx/gsbase | var-gs_bx-gsbase-v2 | `gs_bx` with one type under the regime solver at gmreg [1, 1]: gamma_x 0.5 in both regimes, the paper's general-equilibrium kernel replaced by that stand-in | sol_gsbase `c6ae2d52428a7ce5`, Sol, 5.6 h, published content-addressed |

## What each model can express

The three conditions above are not a KP14 fact, and neither half of the reasoning needs an economy
that no longer exists.

**The negative half is an argument, so state it as one.** Conditional Sharpe rankings are invariant
to a common premium scale, so a price of risk that moves every firm's premium by the same factor --
however nonlinear the state it moves with -- changes the timing of the cross-section and not its
shape. And if an observable state s enters only by making premia affine within each state, the
conditional premium is a_s + b_s'X, which is exactly spanned by (1, X, s, X.s): a linear basis WITH
state interactions reaches it, and `linrank_m` carries those interactions
(`variants/run_estimators.py`). Neither argument is measured; both are checkable by reading them.
Nine of the thirteen economies here are instances of one or both -- every economy except the four
with exposure heterogeneity (`vyx`, `vyg25`, `gx7`, `bx7`) -- and their fair gaps span -0.0080 to
+0.0034.

**The positive half is live in two directions.** Remove exposure heterogeneity and no state process
creates room: `kpbase` room +0.0044, `gsbase` +0.0012. Add it and room appears at once: `gx7` is
`g28`'s gamma(x) PLUS a five-type exposure ladder, and its room is +0.0083 against `g28`'s +0.0006,
fourteen times larger; `bx7` is `gsbase`'s economy plus the same ladder under a regime, +0.0073
against +0.0012. BGN has the heterogeneity natively, in its project-beta distribution, and has room
as published, +0.0267.

**Clause (iii) decides portability, and it is easy to state too weakly.** Until 2026-09-17 this file
said "every level in the economy is stationary". That is the CONSTRUCTION `kp_vy` used, not the
condition: GS21's `bx7` also has stationary levels -- its x is AR(1) -- and its exposure premium is
still collected by a levels-augmented linear ridge. Measured on the live ten seeds, and the
comparator matters: DKKM beats plain `linlev` by +0.0244 in `bx7`, but beats `linlev_m` -- the same
ridge given the equal-weighted market unpenalised, as DKKM has it -- by only **+0.0015**, and in
`gx7` by +0.0009, `g28` +0.0004, `gsbase` -0.0025. So the levels channel is real but it is not
separable from finding 8: `linlev_m`'s winning penalty is the TOP of its grid in 6 of 10 `bx7` seeds
and 10 of 10 `gx7` seeds, where a ridge on eleven columns plus an unpenalised market column is
essentially the market.

The cleaner live statement of the levels channel is about which linear method wins. Where a beta
ladder exists, `linlev` beats `linrank` and becomes the linear winner: `gx7` 0.4457 against 0.3972
and `linlev` winning 6 of 10 seeds, `bx7` 0.2846 against 0.2825 and 4 of 10. Where it does not,
`linlev` loses: `g28` 0.2472 against 0.2527 and 2 of 10, `gsbase` 0.2438 against 0.2648 and 0 of 10.
That is the exposure showing up in stationary raw levels, at ten seeds, in live code.

**A fourth condition, which the protocol campaign added.** Room is not a gap. Finding 8 below: the
market must not already span the economy's Sharpe, or whatever DKKM finds is absorbed by a benchmark
that holds the market on the same terms. That condition is met in one of the three models.

**What each model can express, verified from the live specs and parameter modules.** This is why a
proposal that works in one model is usually not a template for another:

| lever | kp_vy | bgn_gam | gs_bx |
|---|---|---|---|
| price of a priced state's risk | `gamma_v` | `gmult` x `sigma_z`, capped at 1/(2 x 0.137) = 3.65 (`sdf_compute.py:45`) | `gamma_x` x `gmreg`, `gs_gamma_slope` |
| exposure heterogeneity | `type_bv` / `type_share`, any count, cheap (`parameters_kp14.py:89`) | a native project-beta continuum, but the SAME distribution for every firm: firm-level differences are sampling noise over live projects (`panel_functions.py:43`, `:161`). Discrete types need four modules that are no longer in the tree | `GS_BX_BETAS`, any count, but **one 3.5 to 5.6 h solve per type** |
| direct control of the exposure shape | `type_bv` | only `prob_in_money_targets` (`parameters.py:20`), which refits the distribution at `vasicek.py:61-70`. **A `scale` or `beta_star` override is silently discarded** there | `gs_bx`, per solve |
| compensation offset | `bv_comp` | none, and vacuous: there is no firm type to reveal | `gs_ashift` exists but the simulator **refuses any nonzero value** (`gs_sim_bx.py:49-56`), because the old ladder mispriced four of five types |
| persistence of the priced state | `kappa_y`, but **locked to `sigma_y` = sqrt(2 `kappa_y`)** (`parameters_kp14.py:94`), so it moves persistence and innovation size together | `p01`/`p10` (regime), `kappa` (rate) | `rho_x`, frozen by `tests/test_config_parity.py:160` as "genuinely open; needs the paper" |
| default / crash channel | none | none | **built in full and never exercised at protocol**: a debt grid, `kappa_e`, recovery and a `sigma_m` = 5 smoothing shock are all in `gs_solve_reg.py` and `gs_sim_bx.py`, which exports a `default` column -- but no GS panel is committed, so this file reports no default rate. `bx7`'s stress regime puts the price of risk at 0.5 x 3.0 = 1.5 and its fair gap is -0.0018. Settling dormancy is what proposal G3 is for |

And the structural bound each model puts on any proposal -- the first two are the papers' own
propositions, which is what makes them binding rather than measured:

| model | the bound | what it forbids |
|---|---|---|
| BGN | E[R] exactly affine in book-to-price and 1/price with rate-dependent coefficients | nothing in the cross-section: book-to-price is a RATIO, so two premium channels of opposite sign enter one characteristic and can bend the surface non-monotonically. This is the permissive structure, and the project-beta distribution is the only NATIVE exposure heterogeneity in any of the three models. Its bounds are elsewhere: `gmult` at 3.65, `beta_zr` at +0.00027 |
| KP14 | E[R] affine in a SINGLE firm variable, the share of value in growth options, and the two priced channels are nearly parallel across firms | both common scaling (direction invariance, above) and rotation: with premia affine in one firm variable and the two priced channels nearly parallel across firms, there is nothing to rotate between. Among the live economies the only KP14 room above `kpbase`'s +0.0044 belongs to `vyx` and `vyg25`, both built on the `exp(beta_f y)` exposure route |
| GS21 | one priced shock at a constant price, every firm loading on it in the same direction; leverage-to-premium smooth and monotone away from an unpopulated default boundary | cross-sectional room from the state, however nonlinear that state-dependence is: `g28` is a clipped-nonlinear gamma(x) with one exposure type and has the smallest room in this file, +0.0006. And the market reaches 93.6% to 97.4% of SR_max across the four GS21 economies, so the fourth condition fails even where room does appear (`gx7`, +0.0083) |

**The controlled experiment that IS live says the same thing.** The exposure ladder was built in two
models. In KP14 it produced room of +0.3606 and +0.4108 and a fair gap of +0.1251 and +0.1636. In
GS21 the same construction produced room of +0.0083 (`gx7`) and +0.0073 (`bx7`) and fair gaps of
+0.0003 and -0.0018 -- fourteen and six times their no-ladder parents' room, and no gap. BGN cannot
run it at all: there is no add-a-type knob.

And GS21 shows where its version of the room went. Adding the ladder to `g28` multiplies room by
fourteen, yet RAISES the best linear method's share of the attainable Sharpe from 85.7% to 92.7%,
while the measured gap FALLS from +0.0334 to +0.0234 and the fair gap stays at zero. A beta ladder is
close to a linear sort on the characteristics that reveal it. A proposal's value is a property of the
valuation structure it lands in, not of the proposal.

A third model would have made this cleaner. The crash program of the pre-protocol study built one
identical recipe in all three and is the reason that claim was first made, but its five economies have
no code and cannot be rebuilt from this repository's history; nothing in it is citable here. See
"Routes with no code".

## What differs from the baseline, in economic terms

| economy | model | what differs |
|---|---|---|
| kp_vy/vyx | KP14 | A new aggregate state y: stationary Ornstein-Uhlenbeck, mean reversion kappa_y 0.35, unit stationary sd, innovations priced at gamma_v 1.8. Three firm types in shares 0.34 / 0.33 / 0.33 load project cash flows on it as exp(beta_f y), beta 0.02 / 0.07 / 0.14, with a compensation term bv_comp 1.2 so that at y = 0 value does not reveal the type monotonically |
| kp_vy/vyg25 | KP14 | vyx with one parameter changed: gamma_v 1.8 to 2.5, the price of the state's risk |
| bgn_gam/g0235 | BGN | A two-state Markov regime multiplies the price of the market shock, gmult [0.2, 3.5], at the closed-form bound 1/(2 x 0.137) = 3.65; switches unpriced at 0.25/12 and 0.50/12, stress one third of months |
| bgn_gam/g0235f | BGN | g0235 with both switch probabilities times four: calm spells 12 months, stress 6, about forty switches per window; stationary stress share unchanged |
| bgn_gam/g0235s | BGN | g0235 with both times 0.2: calm spells 240 months, stress 120, about two switches per window |
| bgn_gam/g0235r | BGN | Calm-to-stress 0.05/12 and stress-to-calm 0.45/12: stress in 10% of months, in spells of about 27 months. A different economy from g0235, not the same one at another speed -- the mix moves too |
| bgn_gam/g0235d | BGN | g0235's multipliers with the switch probabilities swapped: stress occupies two thirds of months, in 48-month spells |
| gs_bx/g28 | GS21 | A countercyclical price of risk, gamma(x) = clip(0.5 - 0.28 x / sd(x), 0.05, 1.0): the baseline 0.5 at mean productivity, toward 0.05 in booms and 1.0 in busts |
| gs_bx/gx7 | GS21 | g28's gamma(x) crossed with five firm types in equal shares loading on the state as exp(beta_f x + z), beta 1 / 2.5 / 4 / 5.5 / 7 |
| gs_bx/bx7 | GS21 | The same five-type ladder under a two-state regime on the price of risk, gmreg [0.6, 3.0], switches 0.25/12 and 0.50/12, no level compensation |

**The ranking is by (DKKM - FMR) / FMR; the fair-gap column is what says whether a row means
anything.** The two are not close to each other, and the top of this ranking is where they diverge
most. `g0235r` leads at 191.6% on a Fama-MacBeth Sharpe of **0.0121** and `g0235s` follows at 121.8%
on **0.0223**; a DKKM improvement of +0.023 and +0.027 of Sharpe over a denominator that small is a
large percentage of very little, and both rows' fair gaps are NEGATIVE (-0.0062 and -0.0080). By
contrast the two KP14 economies sit 6th and 8th at 21.9% and 20.3% while carrying fair gaps of +0.1636
and +0.1251 -- the only genuine complexity gaps in the file -- because their Fama-MacBeth Sharpes are
0.7911 and 0.6389, thirty to sixty times g0235r's.

Two reasons the proportional column and the fair gap disagree, both already established here. In the
eight BGN and GS rows the winning DKKM portfolio is, to within half a hundredth of Sharpe, the
equal-weighted market (finding 8), so DKKM - FMR is mostly the market's advantage over Fama-MacBeth
rather than anything complexity found; the fair gap is what remains once the linear methods hold that
market on DKKM's terms, and it is at most +0.0034 and negative in five rows. And a ratio is only
informative where its denominator is not near zero (the caution in "Reading the tables").

Ranked by DKKM - best fair linear the order would be vyg25, vyx, g0235f, g0235, gx7, g0235d, bx7,
g28, g0235r, g0235s -- the two KP14 economies first and the two near-zero-denominator BGN regimes
last, i.e. almost the reverse of the table above at both ends.

**bgn_gam/g0235d is the clearest illustration of finding 4 in the file.** Its per-seed DKKM Sharpe
spans 0.1490 to 0.4731 -- a factor of 3.2 -- around a ten-seed mean of 0.2923, and its per-seed gap
runs -0.0114 to +0.0083 around a mean of +0.0000, so its seed 0 alone reads -0.0071: the wrong sign
for the economy. The screen it was run as missed both of its gates on that one seed, and the
ten-seed room, +0.0241, is indeed below the +0.05 gate, so the decision it drove was right; the levels
it was read off were not. It also has the largest DKKM-minus-market of any economy here, +0.2104, and
plain `linrank` follows it to within 0.00003, which is finding 8 at its sharpest.

## Findings

Numbered as they were found, and ORDERED by what reads together: the numbers are cited from specs
and from `docs/refactor/WORKING.md`, so they are kept even where a superseded reading was deleted,
which leaves gaps. Finding 10 sits beside finding 8 because it is the same mechanism one method over.

8. **Outside KP14 Path 1, the measured gap is the market portfolio against linear methods not given
   it on the same terms.** DKKM appends the equal-weighted market to its random features UNPENALISED
   while shrinking the features (`--include_mkt`, as in the paper): it gets the market for free and
   pays only for what it adds. Every linear method carries a market, so the asymmetry is not about who
   HOLDS it -- it is about SHRINKAGE, which is the correction made 2026-09-17 (this clause previously
   read "Fama-MacBeth none", which `variants/common/fama_functions.py` contradicts). Two different
   failures:

   - `linrank` and `linlev` carry the market as their constant column and **penalise it with
     everything else**, so heavy shrinkage takes the market away along with the noise.
   - Fama-French and Fama-MacBeth each carry a market -- value-weighted (`fama_functions.py`, `mve /
     mve.sum()`) and equal-weighted (`P['mkt_rf'] = 1 / len(data)`) -- but combine their factors by an
     **unshrunk** MVE (`run_estimators.py`, `mve_data(fr, month, 0)`), so the market's weight is
     ESTIMATED against noisy competitors with nothing protecting it. Fama-MacBeth's five
     characteristic legs are exactly zero-net-investment by construction -- with a `ones` column in
     the design matrix, the OLS slope weights satisfy `1'X(X'X)^-1 e_j = 0` for every slope `j` -- so
     all of its market exposure sits in that one appended leg, at an estimated weight.

   The fair methods (`linrank_m`, `linlev_m`, `mkt_est`) are what put the linear side on DKKM's terms:
   the market as a separate unpenalised column, the rest shrunk. The split reads off the Sharpe
   columns of any table above: DKKM SR minus EW
   market SR is what DKKM adds over the market, and EW market SR minus best linear SR is what the
   market has over the best linear method. In the four GS21 economies the market reaches 94% to 97% of
   SR_max and DKKM sits 0.0006 to 0.0050 above it. In four of the five BGN regime economies it
   reaches 62% to 84%; the exception is g0235d at 25%, which is why that row has the largest
   DKKM-minus-market in the file. The
   `gx7` row makes the mechanism explicit: its fair benchmark is `mkt_est` -- the market alone -- in 8
   of 10 seeds, and DKKM's winning penalty is the grid's ceiling in 8 of 10. DKKM is asking to be the
   market, and being measured against it.

   **When does DKKM leave the market?** When there is Sharpe the market does not span, as a LEVEL.
   The per-seed non-market Sharpe sqrt(SR_max^2 - SR_ew^2) is 0.09 to 0.12 in the GS21 economies,
   where DKKM stays within 0.005 of the market; 0.03 to 0.24 across BGN, where it ranges from 0.011
   below to 0.073 above; 0.32 in g0235d, where it is 0.210 above; and 1.12 and 1.31 in vyx and vyg25,
   where it is 0.40 and 0.49 above. Leaving the market is necessary for a complexity gap and not
   sufficient: the BGN and KP14 baselines have non-market Sharpe of 0.24 and 0.21 and room of +0.0267
   and +0.0044, against vyx's +0.3606.

10. **Fama-MacBeth beats the equal-weighted market exactly where the market is a minority of the
    attainable Sharpe, and the split is clean.** FMR holds the market (finding 8) yet loses to it in
    eight of the thirteen economies. Sorting all thirteen by the market's share of SR_max separates
    the two outcomes with no overlap:

    | | market share of SR_max | FMR - EW market |
    |---|---|---|
    | FMR wins, 5 economies | 25%, 31%, 34%, 35%, 51% | +0.0500 to +0.3217 |
    | the EW market wins, 8 | 62%, 73%, 82%, 84%, 94%, 95%, 95%, 97% | -0.0135 to -0.0797 |

    Every win is at or below 51% and every loss at or above 62%. The mechanism is dilution, not
    absence: FMR's market leg competes with five zero-net characteristic legs for weight in an
    unshrunk six-factor MVE. Where the market is a minority of the attainable Sharpe there is real
    cross-sectional signal for those legs to earn, and FMR beats holding the market by up to +0.32.
    Where the market is nearly all of it -- the four GS21 economies at 94% to 97% -- the legs are
    competing for a few thousandths, and an unshrunk tilt away from the market costs more than it
    finds. This is finding 8 one method over: DKKM can collapse ONTO the market because it gets it
    unpenalised, and FMR cannot because it has to estimate its weight.

    Two consequences for reading the tables. The (DKKM - FMR) / FMR column that ranks them is
    measured against a benchmark that is itself below the market in eight economies, which is part of
    why it reaches 191.6% in g0235r. And "DKKM beats Fama-MacBeth" is a weaker statement than it
    sounds wherever the market alone also beats Fama-MacBeth.

4. **Single seeds mislead**, and the live ten seeds show it without appealing to any earlier run.
   `g0235d`'s own per-seed DKKM Sharpe spans 0.1490 to 0.4731 -- a factor of 3.2 -- around a mean of
   0.2923, and its per-seed gap runs -0.0114 to +0.0083 around a mean of +0.0000: its seed 0 alone
   reads -0.0071, the wrong sign for the economy. `g28`'s per-seed gap runs +0.0025 to +0.0759, a
   factor of 29, around +0.0334, with seed 0 at +0.0655 -- nearly double the truth. And `g28`'s seed 0
   has NEGATIVE all-month room, -0.00024, against a ten-seed +0.00034 and its own evaluation-window
   +0.00055: one seed can put a population quantity on the wrong side of zero. Cross-seed sd of the
   gap runs 15% of its mean (`vyg25`) to over 300% (`g0235s`) across this file. The protocol's seed
   count is ten for every economy, and there is no longer any tier below it.

5. **Absolute and proportional gaps rank the economies differently, and the tables here are ranked
   proportionally.** (DKKM - FMR) / FMR answers "how much does DKKM improve on Fama-MacBeth, relative
   to what Fama-MacBeth achieves", which is the comparison a reader of the DKKM paper expects. Its
   cost is that it is a ratio: it puts g0235r first at 191.6% on a Fama-MacBeth Sharpe of 0.0121 and
   g0235s second at 121.8% on 0.0223, both of which have negative fair gaps, while the two economies
   with real complexity gaps sit 6th and 8th. The orderings are near-reverses of each other at both
   ends. So the ranking is proportional and the VERDICT is the fair gap; every table carries both
   columns, and the note under the parameterizations table states the divergence rather than leaving
   it to be discovered.

6. **Pre-registration record.** Thirteen specs carried written predictions into the protocol campaign.
   The pattern is sharp and worth stating plainly: **every prediction about a population quantity or a
   fair gap held; most predictions about DKKM's LEVEL were wrong; and the prediction that the widened
   grid would stop binding failed in five of thirteen economies.**

   | clause | economies | outcome |
   |---|---|---|
   | fair gap within its stated bound of zero | 13 of 13 | right; no falsification clause fired anywhere |
   | room and SR_max unchanged within seed noise | 13 of 13 | right. KP14's are bit-identical, its burn-in having already been 400; BGN's and GS21's moved within their seed sd |
   | DKKM's level, predicted range | 4 of 13 right | wrong LOW for vyx (+0.021 against +0.03 to +0.09) and vyg25 (+0.019 against +0.04 to +0.11); wrong HIGH for gx7 (+0.034) and bx7 (+0.013) against "within 0.005 either way"; wrong DOWN for bgnbase, gsbase and g0235s, which fell |
   | the winning penalty is interior in at least 8 of 10 seeds | 8 of 13 | **wrong for vyx, vyg25, gx7, bx7 and g0235s** -- the gate above |
   | g0235d's ten-seed room between +0.010 and +0.030 | 1 of 1 | right, +0.0241 |
   | g0235d leaves the market by more than any other BGN mean, `linrank` following within 0.005 | 1 of 1 | right: +0.2104 above the market, and `linrank` follows to +0.00003 |

   The lesson the DKKM rows teach is that a gain-per-decade extrapolation does not carry across
   economies: vyx's own flattening decades at this window (0.154, 0.100, 0.035) suggested the two new
   decades below the old floor were worth little, and they were worth +0.021 -- less than the naive
   sum but in the right direction, while the ceiling side moved in a direction nothing had predicted
   because nobody had looked above 10.

9. **Where the gap is genuine, it is bounded by DKKM's estimation shortfall.** In both KP14 Path 1
   economies the linear methods reach 99% to 100% of their population ceiling and DKKM 76% and 80% of
   its. Raising the price of the state's risk from 1.8 to 2.5 raised DKKM's share of its ceiling from
   76% to 80% and the fair gap by 31%. That the grid floor still binds in both is the open end of this
   finding: the shortfall has not been measured to its bottom.

## Open proposals

| proposal | scope | what it would decide | status |
|---|---|---|---|
| **A wider ridge grid** | **universal** | whether the five censored rows' DKKM levels are materially higher. KP14 needs decades below `1e-5`; GS21 and the slow BGN regime need decades above `10` | OPEN and declined for now: censoring is accepted and disclosed instead. A protocol amendment, so it changes every row |
| K6, gamma_v 1.2 with vyx's exposures | KP14; lever universal, closed in BGN and GS21 | whether a defensible calibration shows a gap: the price ladder's low end, E[mu] about 14% a year against vyx's 18.2% and vyg25's 22.9% | OPEN, the leading candidate. Its prediction should be registered against vyx's +0.1251, not the retired pre-protocol +0.1046 |
| K1, a continuum of exposures | KP14; ~10x the cost in GS21, no knob in BGN | fifteen types over [0, 0.14], shares right-skewed. A smooth exposure map suits random features and lowers the market's average exposure | OPEN, about 7.5 h of integrals then 30 h of seeds |
| K3, persistence of the priced state | KP14; answered in BGN as B1, frozen in GS21 | kappa_y 0.15 and 0.70. Informative about mechanism, not about the size of the gap: slower reversion bends values more but gives a window fewer cycles | OPEN, two solves of about 90 min |
| K2, a rare extreme type | KP14 and GS21 only | shares (0.45, 0.45, 0.10), top loading 0.14 to 0.20 | OPEN, low priority; the discount check that withdrew K5 applies |
| G3, the GS21 default-channel probe | **GS21 only** | whether GS21 has any non-market Sharpe: equity near default is a convex claim on the same shock, so its loading rises as the state worsens | OPEN, 20 min, oracle only. Its gate needs an evaluation-window room clause of at least +0.05 as well as a market share below 85% -- KP14's baseline meets the market clause at 35% and has no gap |

**On the scope column.** Only the ridge grid is universal, and necessarily so: it is estimator-side
(`variants/common/protocol.py`), so it applies to all thirteen rows at once. Four of the remaining
five are model-specific PARAMETERS of a universal LEVER -- every model has a price-of-risk knob and a
persistence knob -- but the lever is exhausted or frozen elsewhere: BGN climbed the price ladder to
its closed-form bound and its persistence ladder is B1, four ten-seed points that never moved the
fair gap; GS21's slope is fixed and its `rho_x` is pinned by a parity test. K1 is portable to GS21 in
principle and costs one solve per type there, so fifteen types is about 75 h of solves against 7.5 h
of integrals in KP14, and BGN has no add-a-type knob at all. G3 is the only strictly single-model
proposal: neither BGN's real-option projects nor KP14's projects carry debt, so there is no channel
to wake. See "What each model can express" for why porting is mostly pointless rather than merely
expensive.

Closed, with what closed them: BGN's regime path (five ten-seed points, room moved by a factor of ten
and the fair gap never left -0.008 to +0.004); GS21's exposure path (gx7's pre-registered negative
fired, and the protocol campaign confirms it at +0.0003); K5, signed exposures (the economy does not
exist -- the G solve discounts growth options at rho_ty, which goes negative for a loading of -0.06,
and a negative discount removes the operator's dissipation); the window and penalty ladders (absorbed
into the protocol, which fixes both for every economy).


### A numerical choice that binds: the ridge grid

DKKM's reported Sharpe is a **maximum over the ridge-penalty grid**, so where the winning penalty
sits at an EDGE of that grid the number is censored -- it says where the search stopped, not what the
economy affords. `variants/penalty_gate.py` reads the winning penalty per seed and is the first thing
to run on any campaign. It is a DISCLOSURE, not a block: censoring is accepted here, and the price of
accepting it is that every affected row says so.

| economy | interior | at the floor `1e-5` | at the ceiling `10` | winning penalties | reading |
|---|---|---|---|---|---|
| bgn_gam/bgnbase | 10/10 | 0 | 0 | 0.01x8 0.1x2 | clean |
| bgn_gam/g0235d | 10/10 | 0 | 0 | 0.01x4 0.1x6 | clean |
| gs_bx/gsbase | 10/10 | 0 | 0 | 0.1x4 1x6 | clean |
| kp_vy/kpbase | 10/10 | 0 | 0 | 0.01x7 0.1x3 | clean |
| bgn_gam/g0235f | 9/10 | 0 | 1 | 0.01x1 0.1x8 10x1 | clean |
| gs_bx/g28 | 9/10 | 0 | 1 | 0.1x2 1x7 10x1 | clean |
| bgn_gam/g0235 | 8/10 | 0 | 2 | 0.01x2 0.1x4 1x2 10x2 | clean |
| bgn_gam/g0235r | 8/10 | 2 | 0 | 1e-05x2 0.01x3 0.1x5 | clean |
| **gs_bx/bx7** | 7/10 | 0 | 3 | 0.1x2 1x5 10x3 | **censored at the ceiling** |
| **bgn_gam/g0235s** | 5/10 | 0 | 5 | 0.001x1 0.01x3 0.1x1 10x5 | **censored at the ceiling** |
| **kp_vy/vyg25** | 4/10 | 6 | 0 | 1e-05x6 0.0001x2 0.001x2 | **censored at the floor** |
| **kp_vy/vyx** | 3/10 | 7 | 0 | 1e-05x7 0.0001x2 0.001x1 | **censored at the floor** |
| **gs_bx/gx7** | 2/10 | 0 | 8 | 1x2 10x8 | **censored at the ceiling** |

**The censoring is directional, and the direction decides what it costs.** All five censored rows'
DKKM Sharpes are LOWER bounds, but they bind from opposite ends and that changes the reading:

- **KP14 (`vyx`, `vyg25`) binds at the FLOOR while its benchmark does not.** DKKM wants less
  shrinkage than the grid allows; the fair side is won by `linlev` and `fm`, mostly interior. Widening
  the floor would raise DKKM and leave the benchmark roughly put, so **+0.1251 and +0.1636 are lower
  bounds on the fair gap** -- the headline understates itself.
- **GS21 `gx7` and `bx7`, and BGN `g0235s`, bind at the CEILING, and there the benchmark is the
  market.** `gx7`'s fair benchmark is `mkt_est` in 8 of 10 seeds: DKKM at maximum shrinkage is trying
  to become the market, and the market IS what it is measured against. More decades push DKKM toward
  the market's own Sharpe, so the fair gap stays near zero either way. **The "no complexity gap
  outside KP14" conclusion is robust to this censoring**; only the level is soft.

What is NOT affected: every population quantity (room, SR_max, the ceilings), which involves no
estimation; the ranking by fair gap, since DKKM and its benchmark share the grid; and the eight clean
rows, which include all three baselines.

**Retired with the protocol, and the rule it establishes.** One KP14 economy was retired rather than
re-run because it was not an economy: same parameters, same solves as `kp_vy/vyx`, differing only in
T, the window and the ridge grid. Reporting it beside `vyx` made a sample length look like an economic
finding. The rule: an economy that wants a different sample, window or grid is a numerical experiment,
not another economy, and changing any of them is a protocol amendment that moves every row in this
file. Its spec records the retirement in `lineage.retired` and
`tests/test_protocol_is_uniform.py` refuses its return; its pre-protocol numbers are not quoted here
because they no longer exist. Detail in "Routes with no code".

## Routes with no code

**No claim in this section is citable, and nothing here can be rebuilt from this repository.** These
are the routes of the pre-protocol study. Their code was never in this repository's git history: the
import at `bba735f` ADDS 130 files under `variants/` and deletes none, `variants/` did not exist
before it, and across every commit reachable from `--all` the number of paths matching `kp_gam`,
`kp_gamy`, `kp_bx`, `kp_dis`, `bgn_types`, `bgn_dis`, `bgn_gamr`, `gs_fixed` or `gs_dis` is zero. The
narrative is `archive/REPORT.md`, which carries its own banner; every figure in it is a single seed,
its KP14 levels ran at growth-option arrival rate 1.72 and its GS21 levels on three wrong Table I
parameters. They are listed here so that a route is not re-proposed in ignorance, and for no other
purpose.

| route | model | what it was | why it cannot be cited |
|---|---|---|---|
| `kp_bx` | KP14 | exposure types loading as x^beta_f on the aggregate state, crossed with a regime | no code. It is the origin of clause (iii) -- room appeared and rolling Fama-MacBeth took it, because x^beta_f made raw levels co-move with the state -- but that claim now rests on live `bx7`/`gx7` instead |
| `kp_gam`, `kp_gamy` | KP14 | a two-state and then a continuous common multiplier on both prices of risk | no code. `kp_gamy` additionally needs `gz_lo`/`gz_hi`, which the live module does not have. The claim they supported is now the spanning argument, stated above |
| `kpd_al`, `kpd_off` | KP14 | type-specific disaster probabilities with a priced kill intensity and a compensating output multiplier | no code |
| `bgn_types` | BGN | discrete exposure types grafted onto BGN, no state | no code; needs four modules (`vasicek_types.py`, `sdf_compute_types.py`, `panel_functions_types.py`, `loadings_compute_types.py`) |
| `bgn_dis` | BGN | a disaster indicator in the kernel with three firm types and a value compensation | no code |
| `bgn_gamr` | BGN | a continuous logistic price of risk in the short rate | no code. It held the largest population room of the study and rolling Fama-MacBeth beat DKKM on it; that is exactly the shape clause (iii) forbids, and no live economy reproduces it |
| `gsd`, `dis_rebase` | GS21 | disasters destroying capital with the coupon unchanged | no code |
| `sol_gnl` | GS21 | a genuinely logistic gamma(x), 0.10 to 1.20 | no code for the SHAPE: the live solver has `gs_gamma_lo`/`gs_gamma_hi`/`gs_gamma_slope` but implements a CLIPPED LINEAR gamma(x), and `gamma_form`/`gnl_steep` appear nowhere under `variants/` |
| `grot` | KP14 | a rotation: the multiplier on gamma_x only, so the premium direction swings across the state | **expressible**, unusually: `g_lo`/`g_hi`/`g_steep` are live in `parameters_kp14.py` and wired into the solve. Never run at protocol, so it has no citable numbers either |
| `bx9` | GS21 | the exposure ladder to beta 9 under a wider regime | **expressible only at `gs_ashift` = 0**; as published it needs the compensation ladder, which `gs_sim_bx.py` refuses. Five fresh solves, ~18-28 h, before a seed |
| `g0520`, `g0330` | BGN | the regime ladder at gmult [0.5, 2.0] and [0.3, 3.0] | **expressible** -- just `gmult` plus a J* rebuild -- and superseded anyway: `g0235` sits at the closed-form bound and is live. Note `g0520` is also a legacy KP14 tag at the same multipliers, which has no code; key on `model/tag`, never the bare tag |
| `vyxT860` | KP14 | `vyx`'s economy at T=860, window 720 | not a separate economy: same parameters, same solves. Retired 2026-09-15 and enforced by `tests/test_protocol_is_uniform.py` |

**The crash program is the one loss that matters.** `kpd_al`/`kpd_off`, `bgn_dis` and `gsd` were one
identical recipe built in all three models -- the only controlled cross-model experiment in the record
-- and the portability argument above would be stronger for having it. Its code exists outside this
repository, un-versioned, and importing it means bringing in about 3,900 lines written against the
pre-correction calibrations and reopening `run_oracle.py`'s three-model whitelist. Until that is done
and re-measured at protocol, the two-model exposure-ladder comparison above is what the argument
rests on.

## Adding an experiment

1. Write `experiments/specs/var-<model>-<tag>-v1.json`: title, question, the parameter override,
   `env`, `estimation`, a written prediction with a falsification clause, and `expected_solves`
   computed WITHOUT solving. Say in economic terms what differs from the baseline. The `panel` and
   `estimation` blocks are the protocol's -- `tests/test_protocol_is_uniform.py` refuses any other
   sample, window, ridge grid or conditioning-column list. **A new economy is a parameter override and
   nothing else.** If what you want to change is the sample, the window or the grid, you are proposing
   a change to the protocol, which changes every row in this file and is not an experiment.
2. Solve with the model's producer (`rebuild_jstar_gam.py` -- or `rebuild_all_jstar.sh` for the whole
   BGN set, on ONE machine -- `build_vy_tables.py`, `gs_solve_reg.py` / `gs_solve_gam.py`); confirm
   the registry id matches the spec; publish with `variants/fetch_solves.py --publish`. A chained
   stage's id hashes its upstream tables' raw bytes, so build the chain on one platform or ship the
   upstream tables byte-identical (`docs/RUNS.md`).
3. Add a `SEED_SPEC` case to `variants/run_seeds_slurm.sh` (`tests/test_specs_match_shell.py` pins it
   to the spec). The case carries the economy's parameters and nothing else: no `KAPPAS`, no
   `RF_COLS`, no `SEED_T`. Add a row to `variants/submit_campaign.sh` with its memory and the reason.
4. Ten seeds, or the economy does not go in a table. If its peak memory has never been measured, run
   one seed and read `sacct` before sizing the array.
5. `python variants/aggregate_seeds.py`, then **`python variants/penalty_gate.py` before reading any
   number**: if the winning penalty sits at a grid edge, that economy's DKKM Sharpe is censored and
   must be reported as a lower bound, with the direction, in the gate table above. Commit the result
   files and the table; add the economy here with what differs from the baseline in economic terms
   first and what the result decided against the prediction.
   `tests/test_results_md_matches_table.py` fails until the row is added.
6. Push, then pull the shared checkout on the cluster with `bash variants/cluster_pull.sh`, never a
   plain `git pull`: both clusters import their Python from that one tree
   (`docs/RUNS.md`, "Where output goes").
