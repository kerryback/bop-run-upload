# Experimental results

Every economy run through the oracle-and-estimator pipeline: what it was built to test, what it
produced, and what it decided. One measurement protocol governs all of them, stated once below, so
that the only thing separating two rows of a table is the economy. Every number here comes from the
protocol-v3 campaign of 2026-09-21 to 2026-09-22: thirteen economies, ten seeds each, 130 tasks
across Sol and Phoenix, all COMPLETED. Where each job ran and what it cost: `docs/RUNS.md`. What to
run next, and why: `docs/NEXTUP.md`.

Every table below is checked cell by cell against `variants/results/economy_table.csv`
(`variants/aggregate_seeds.py`) by `tests/test_results_md_matches_table.py`, so this file cannot
fall behind the numbers without the suite saying so.

**This campaign corrected two solve-level pricing defects and widened the ridge grid, and every
number moved.** The defects were found on 2026-09-18 and fixed in merge `91095fb`: `kp_vy` carried
the price of the mean-reverting state `y` as a constant added to the discount rate -- KP14's eq. (11)
shape, exact for the paper's GBM shocks and wrong for an OU state, whose Girsanov adjustment
saturates -- and BGN's bond recursion added the log-kernel/short-rate covariance to the cumulative
variance once instead of twice. The corrected specification is `variants/kp_vy/parameters_kp14.py`
(`y_risk_neutral = 1`, the substitution `W = e^{by} A` solved under the Q-generator), and all three
models' `E[M R] = 1` identities are asserted in `tests/test_risk_neutral_pricing.py`. Nine economies
re-solved; all thirteen re-ran, because the grid amendment is estimator-side. **No pre-fix figure is
reported below as a result.** Where one appears it is on the left of an explicit before-and-after,
labelled as such.

**It took the headline with it.** `vyx`'s fair gap falls from +0.1251 to **+0.0009** and `vyg25`'s
from +0.1636 to +0.0148. The pre-fix `vyx` row was the largest single claim in this file and it does
not survive. What does is below, and it is an order of magnitude smaller.

## The results

**Ranked by the verdict column, DKKM minus the best FAIR linear method.** `t` is the ten-seed mean
over its standard error, which is the right test for a quantity whose cross-seed sd runs 37% to 251%
of its mean (finding 4).

| economy | fair gap | cross-seed sd | t | seeds with a positive gap |
|---|---|---|---|---|
| kp_vy/vyg25 | **+0.0148** | 0.0090 | 5.2 | **10 of 10** |
| bgn_gam/g0235f | **+0.0079** | 0.0052 | 4.8 | **10 of 10** |
| bgn_gam/g0235 | +0.0025 | 0.0030 | 2.7 | 8 of 10 |
| bgn_gam/g0235d | +0.0025 | 0.0072 | 1.1 | 5 of 10 |
| kp_vy/vyx | +0.0009 | 0.0087 | 0.3 | 7 of 10 |
| gs_bx/gx7 | +0.0003 | 0.0007 | 1.6 | 5 of 10 |
| bgn_gam/bgnbase | -0.0001 | 0.0061 | -0.0 | 6 of 10 |
| bgn_gam/g0235r | -0.0005 | 0.0027 | -0.6 | 5 of 10 |
| kp_vy/kpbase | -0.0010 | 0.0058 | -0.6 | 6 of 10 |
| gs_bx/bx7 | -0.0017 | 0.0020 | -2.8 | 0 of 10 |
| gs_bx/g28 | -0.0024 | 0.0018 | -4.3 | 0 of 10 |
| gs_bx/gsbase | -0.0028 | 0.0019 | -4.7 | 2 of 10 |
| bgn_gam/g0235s | -0.0038 | 0.0145 | -0.8 | 4 of 10 |

**Only `vyg25` and `g0235f` are positive in every seed AND several standard errors from zero**, at
+0.015 and +0.008 of monthly Sharpe against attainable Sharpes of 0.50 and 0.24. The other eleven lie
between -0.0038 and +0.0025, and that includes all three models as published: -0.0001 (BGN), -0.0010
(KP14), -0.0028 (GS21).

### Every column, all thirteen economies

**Ranked by (DKKM - FMR) / FMR**, the ratio of the ten-seed means, which is the comparison a reader
of the DKKM paper expects. It is not the verdict, and the two orderings are near-reverses at the
bottom: `g0235r` and `g0235s` lead here on Fama-MacBeth Sharpes of 0.0276 and 0.0376 while both have
NEGATIVE fair gaps. Read this table for the levels and the one above for the answer (finding 5).

| economy | seeds | SR_max | EW market SR | FMR SR | best linear SR | DKKM SR | DKKM - FMR | (DKKM - FMR) / FMR | DKKM - best linear | DKKM - best fair linear | room | room / FMR | t, DKKM vs FMR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bgn_gam/g0235r | 10 | 0.0757 | 0.0637 | 0.0276 | 0.0411 | 0.0580 | +0.0304 (0.0141) | 110.4% | +0.0169 (0.0139) | -0.0005 (0.0027) | +0.0047 (0.0028) | 17.0% | 28.4 |
| bgn_gam/g0235s | 10 | 0.0912 | 0.0746 | 0.0376 | 0.0548 | 0.0713 | +0.0337 (0.0243) | 89.5% | +0.0165 (0.0248) | -0.0038 (0.0145) | +0.0070 (0.0073) | 18.5% | 49.5 |
| bgn_gam/g0235 | 10 | 0.1800 | 0.1350 | 0.1027 | 0.1177 | 0.1383 | +0.0356 (0.0125) | 34.6% | +0.0206 (0.0076) | +0.0025 (0.0030) | +0.0276 (0.0153) | 26.9% | 27.8 |
| bgn_gam/g0235d | 10 | 0.3435 | 0.0894 | 0.2467 | 0.3048 | 0.3085 | +0.0618 (0.0389) | 25.0% | +0.0037 (0.0085) | +0.0025 (0.0072) | +0.0311 (0.0041) | 12.6% | 22.2 |
| gs_bx/gx7 | 10 | 0.4851 | 0.4725 | 0.3928 | 0.4499 | 0.4732 | +0.0803 (0.0949) | 20.5% | +0.0232 (0.0179) | +0.0003 (0.0007) | +0.0082 (0.0066) | 2.1% | 19.2 |
| bgn_gam/g0235f | 10 | 0.2375 | 0.1465 | 0.1406 | 0.1537 | 0.1690 | +0.0285 (0.0225) | 20.3% | +0.0154 (0.0085) | +0.0079 (0.0052) | +0.0447 (0.0118) | 31.8% | 14.9 |
| kp_vy/vyg25 | 10 | 0.4952 | 0.0529 | 0.2921 | 0.3283 | 0.3503 | +0.0582 (0.0264) | 19.9% | +0.0219 (0.0164) | +0.0148 (0.0090) | +0.0820 (0.0085) | 28.1% | 19.6 |
| gs_bx/g28 | 10 | 0.3144 | 0.2978 | 0.2544 | 0.2694 | 0.3028 | +0.0484 (0.0415) | 19.0% | +0.0334 (0.0261) | -0.0024 (0.0018) | +0.0006 (0.0005) | 0.2% | 27.6 |
| bgn_gam/bgnbase: BGN as published | 10 | 0.3013 | 0.1618 | 0.2133 | 0.2318 | 0.2382 | +0.0249 (0.0387) | 11.7% | +0.0063 (0.0148) | -0.0001 (0.0061) | +0.0274 (0.0109) | 12.8% | 13.6 |
| kp_vy/vyx | 10 | 0.4214 | 0.0598 | 0.2840 | 0.3069 | 0.3133 | +0.0293 (0.0245) | 10.3% | +0.0064 (0.0161) | +0.0009 (0.0087) | +0.0545 (0.0057) | 19.2% | 11.4 |
| gs_bx/bx7 | 10 | 0.3270 | 0.3061 | 0.2822 | 0.2930 | 0.3091 | +0.0269 (0.0267) | 9.5% | +0.0161 (0.0136) | -0.0017 (0.0020) | +0.0073 (0.0050) | 2.6% | 17.2 |
| kp_vy/kpbase: KP14 as published | 10 | 0.2292 | 0.0797 | 0.1930 | 0.2015 | 0.2079 | +0.0150 (0.0109) | 7.7% | +0.0064 (0.0087) | -0.0010 (0.0058) | +0.0044 (0.0011) | 2.3% | 15.4 |
| gs_bx/gsbase: GS21 as published | 10 | 0.2926 | 0.2785 | 0.2650 | 0.2689 | 0.2835 | +0.0185 (0.0106) | 7.0% | +0.0146 (0.0079) | -0.0028 (0.0019) | +0.0012 (0.0001) | 0.4% | 24.4 |

Ten seeds each, N=500, T=500, burn-in 400, window 360, 125 evaluation months. Mean (sd across seeds).

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
  3600 features, the penalty grid, and its four variants. A maximum over that grid: see "A numerical
  choice that turned out not to bind" below.
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

1. **Room is not a ceiling on the gap and not a screen for it.** The gap exceeds the room in six of
   the thirteen economies -- all four GS21 rows and two BGN ones -- because there it is the market
   shortfall rather than anything nonlinear; in KP14 it is 12% (vyx) and 27% (vyg25) of the room.
   Ranking candidates by room does not rank them by gap, and the corrected campaign makes that
   sharper: `vyx` has the file's second-largest room and its fifth-largest fair gap, which is +0.0009.
2. **Every percentage is a RATIO OF MEANS, not a mean of per-seed ratios.** In g0235, (DKKM - FMR) /
   FMR is 34.6% as the ratio of the ten-seed means and 43.7% as the mean of per-seed ratios,
   because Fama-MacBeth's Sharpe is small and dispersed. In g0235r and g0235s it is below zero in some
   seeds, where a per-seed ratio has no meaning at all. Both versions live in `economy_table.csv`
   (`gap_fm_over_fm`, `gap_fm_pct_fm_mean`).

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
| ridge-penalty grid, DKKM and the four classical linear methods | `1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000` -- eleven values, widened two decades each side on 2026-09-21 | same |
| ridge-penalty grid, the market-augmented fair methods (`linrank_m`, `linlev_m`) | the same eleven plus an unpenalised column and `1e4`, `1e5` -- DERIVED as `sorted(set(kappas) \| {10*max, 100*max})`, so the fair benchmark keeps its deliberate two-decade margin over DKKM's automatically and is never the one that runs out of shrinkage | `variants/run_estimators.py` |
| random-feature counts | 36, 360, 3600, over 2 independent draws | same |
| bandwidth grid | `np.arange(0.5, 1.1, 0.1)`, 7 values | same |
| conditioning columns | the model's **full** set, every economy including the baselines | same |
| benchmark | `--fair_linear --include_mkt --levels`, no winsorisation | each spec's `estimation` block |

Solve-side precision cannot be shared -- the three papers have different solvers -- so what is
required of it is that it be identical across every economy of a given model. It is, and every
manifest in `experiments/registry/` records it:

| model | solve precision, identical across its economies |
|---|---|
| KP14 (`kp_vy`) | a 21-node `y` TABLE grid at `y_max` 3.5, unchanged; since 2026-09-21 the A and G claims are solved under the risk-neutral measure on an INTERNAL grid wide enough to hold the Q-stationary distribution (89 to 103 nodes at these prices of risk) and sampled back to the 21 table nodes, so `NY` and the table layout are the same and the transition matrix is no longer what the coefficients come from; the G stage a direct sparse solve on a 1000-point epsilon grid, gated at a relative residual of 1e-6 and achieving about 1e-12; CIR integrals by adaptive quadrature at `epsrel` 1e-6 with the density's mass verified to 1e-8 |
| GS21 (`gs_bx`) | value-function iteration on a 161-point x grid, 200-point z grid, 20-point debt grid, exact-Gaussian Tauchen at +-4 sd, fixed-point `tol` 1e-6, 161-node quadrature for the default-smoothing shock |
| BGN (`bgn_gam`) | 100-node Gauss-Laguerre for the project-beta integral and 100-node Gauss-Hermite for the rate shock; the J* table by adaptive bisection to `tol` 3e-4, converging to 161 grid points in all six |

The internal Q-grid's span and subdivision are precision knobs that no manifest records --
`PRECISION_KEYS["kp"]` is still `("NY", "_i0")` -- which is a gap worth closing before another KP14
economy is added.

Every claim above is pinned by `tests/test_protocol_is_uniform.py`: each model's burn-in literal, the
seed array's sample and grid, every live spec's `panel` and `estimation` block, and the uniformity of
solve precision within each model. All 130 seeds of the campaign record `spec_check`, `env_check` and
`readback` verified (`readback` reads "not requested" for the four GS21 economies, which deliberately
leave `GS_SIM_OVERRIDES` unset), the protocol-v3 spec id, burn-in 400 recovered from the month
indices, 485 retained months and `eval_window` 360. All 130 also read STALE on their solves and
therefore re-ran: the 2026-09-15 campaign silently skipped seventy tasks because a protocol change
left the solve ids untouched, and confirming that every id moved is the check that catches it
(`docs/RUNS.md`).

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

## The answer so far

**The complexity gap is real, it is much smaller than this file claimed before 2026-09-22, and it
survives in two economies out of thirteen.** The numbers are in "The results" at the top of this
file; what follows is what they mean.

**What broke: the two conditions no longer separate the rows.** Before the pricing fix this file
argued that a gap needs both nonlinear room and a market that spans little of the attainable Sharpe,
and the two KP14 Path 1 rows had both and a gap of +0.125 to +0.164. They still have both, by a wide
margin -- `vyx` has the file's second-largest room, +0.0545, and its second-smallest market share,
14.2% -- and `vyx`'s gap is +0.0009. Meanwhile `g0235f`, whose market carries 62% of SR_max, has a
gap positive in ten seeds of ten. Room and market share still describe where DKKM LEAVES the market
(finding 8); they no longer predict whether anything is left over once the linear side is given the
market on DKKM's terms.

**What the KP14 correction did.** Pricing the OU state's risk as a saturating Girsanov adjustment
rather than a constant addition to the discount rate cuts the attainable Sharpe by roughly two
thirds and the population room by six to eight times: `vyx` SR_max 1.1778 to 0.4214 and room +0.3606
to +0.0545, `vyg25` 1.3887 to 0.4952 and +0.4108 to +0.0820. The gap falls faster still, by 99% in
`vyx` and 91% in `vyg25`. The pre-fix economies were also implausibly calibrated -- oracle mean
expected excess return 18.2% and 22.9% a year -- and the corrected ones are not: 4.2% and 3.9%,
inside the 3.3% to 12.6% band every other economy occupies. The calibration objection and the size
of the gap turned out to be the same problem, which is what proposal K6 was written to test; K6 is
answered by the correction rather than by a new economy.

*The pre-fix `vyx` and `vyg25` figures (+0.1251 and +0.1636) are superseded. They appear later in
this file only as the left-hand side of a before-and-after, never as a current number; the specs
that produced them are retained in `experiments/specs/` under `lineage.superseded_by`.*

**In the other eleven economies the fair gap is between -0.0038 and +0.0025.** That includes all
three models as published, whose fair gaps are -0.0001 (BGN), -0.0010 (KP14) and -0.0028 (GS21).

**Why the KP14 route still has the largest room, and why that is no longer the story.** Three
things are true of it at once, and of no other economy here. First, firms differ in their exposure
to a priced shock: three types load their cash flows on the state y as exp(beta_f y), with beta 0.02
/ 0.07 / 0.14, and y's innovations carry a price of risk. Second, the state bends that exposure map
rather than shifting it: y is mean-reverting, so the value of an exp(beta y) stream depends on where
y is, and a compensation term keeps firm value from revealing the type monotonically at y = 0.
Third, the bending is confined to RANK AND INTERACTION space: the exposure channel does not also
show in raw levels, where a cheap rolling regression would contest it. Under correct pricing that
construction still produces the file's two largest rooms, +0.0545 and +0.0820 over the evaluation
months, and the file's two smallest market shares, 14.2% and 10.7% of the attainable Sharpe -- and
in `vyx` it produces no gap at all. The mechanism is intact; its magnitude was eight times smaller
than the mispriced solve implied, and at the corrected magnitude a rank-and-interaction linear ridge
holding the market takes essentially all of it in `vyx` and 82% of it in `vyg25`.

**What the surviving gaps are not.** They are not the market portfolio, which DKKM appends to its
features unpenalised. `vyg25`'s market has Sharpe 0.053 against an attainable 0.495 and `g0235f`'s
0.147 against 0.238; in both the gap is measured after the linear side is given that same market as
a separate unpenalised column, which is what "fair" means here. Nor are they a calibration artifact,
which is the objection the pre-fix rows could not answer: the oracle's mean expected excess return
is 3.9% a year with a cross-sectional sd of 2.0% in `vyg25` and 9.9% with 1.7% in `g0235f`, against
3.3% to 12.6% across the file. What they ARE is small -- +0.015 and +0.008 of monthly Sharpe -- and
the question the project now faces is whether a gap of that size is worth a paper's central claim.

## The three models as published

Each paper's economy, before any parameterization was built on it. Every "what the parameterization
added" statement in this file is a difference against these rows.

Their rows are the three labelled `as published` in "The results" above.

*`kpbase` is the campaign's control: it has `beta_f = 0` and `gamma_v = 0`, where the KP14
correction is identically zero, and it came back with SR_max, the market and room unchanged to four
decimals while its `solve_id` moved. `gsbase` is untouched by either defect and reproduced every
column. `bgnbase` moved with the corrected bond covariance -- SR_max 0.2830 to 0.3013 -- which is
the fix working, not noise.*

What the anchor says:

- **No model as published has a complexity gap, and the correction did not change that.** The fair
  gaps are -0.0001 (BGN), -0.0010 (KP14) and -0.0028 (GS21). In GS21 a linear method given the
  market beats DKKM in eight seeds of ten; in BGN and KP14 in four each, with the ten-seed mean on
  the wrong side of zero in both. The measured gaps, +0.006 to +0.015, go once the linear methods
  hold the market as DKKM does. All three rows are clean on the penalty gate on POSITION -- their
  winning penalty is interior in ten seeds of ten -- so none of this is a grid artifact.
- **DKKM leaves the market wherever the market leaves Sharpe, and the linear methods follow it.** In
  the BGN and KP14 baselines the market carries 54% and 35% of the attainable Sharpe and DKKM is
  0.076 and 0.128 above it; in GS21, where the market carries 95%, it is 0.005 above. Non-market
  Sharpe of 0.21 to 0.25 without nonlinear room gives no gap.
- **Room and a small market share are necessary and NOT sufficient, and the corrected campaign is
  what shows the gap between the two.** BGN as published has room, +0.0274, and a market at only
  54%, and a fair gap of -0.0001. KP14 as published has a market at 35% and room of +0.0044. But
  `vyx` has the file's second-largest room, +0.0545, and its second-smallest market share, 14.2%,
  and its fair gap is +0.0009 -- so having both is not enough either. Before the pricing fix this
  bullet read "only an economy with both produces a gap", on the strength of two rows that no longer
  exist. What separates `vyg25` (+0.0148) from `vyx` (+0.0009) is one parameter, the price of the
  state's risk, 2.5 against 1.8; finding 9 is where that is taken up.

**How each baseline is produced.**

| economy | spec | what it is in the code | solve |
|---|---|---|---|
| bgn_gam/bgnbase | var-bgn_gam-bgnbase-v3 | `bgn_gam` at gmult [1, 1]: both regimes price the market shock at sigma_z 0.4 and the regime is inert; reproduces the paper's economy to machine precision. Table I: 11 of 11 parameters match | jstar `cb6340649eace098`, Mac, table committed (the corrected bond covariance re-solved it) |
| kp_vy/kpbase | var-kp_vy-kpbase-v3 | `kp_vy` with one type at beta 0, gamma_v 0, bv_comp 0: nothing depends on the state y (the 21 integral tables agree across y to 4e-12); r = 0.05 as in every KP14 economy here, the paper's 0.025 being the one standing departure. Table II: 17 of 18 match | G `57d8ab3fd1e683a8` and integ `eeea76dd678bb9eb`, Mac, tables committed; the ids moved with the producer source and the tables rebuilt at a relative norm of 3.5e-12, which is the different solve path and not an economic difference |
| gs_bx/gsbase | var-gs_bx-gsbase-v3 | `gs_bx` with one type under the regime solver at gmreg [1, 1]: gamma_x 0.5 in both regimes, the paper's general-equilibrium kernel replaced by that stand-in | sol_gsbase `c6ae2d52428a7ce5`, Sol, 5.6 h, published content-addressed |

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
with exposure heterogeneity (`vyx`, `vyg25`, `gx7`, `bx7`) -- and their fair gaps span -0.0038 to
+0.0079. That upper end is `g0235f` and it is new: under corrected BGN pricing one economy with no
exposure heterogeneity at all carries a gap positive in ten seeds of ten, which the argument above
does not forbid but did not anticipate either.

**The positive half is live in two directions.** Remove exposure heterogeneity and no state process
creates room: `kpbase` room +0.0044, `gsbase` +0.0012. Add it and room appears at once: `gx7` is
`g28`'s gamma(x) PLUS a five-type exposure ladder, and its room is +0.0082 against `g28`'s +0.0006,
fourteen times larger; `bx7` is `gsbase`'s economy plus the same ladder under a regime, +0.0073
against +0.0012. BGN has the heterogeneity natively, in its project-beta distribution, and has room
as published, +0.0274.

**Clause (iii) decides portability, and it is easy to state too weakly.** Until 2026-09-17 this file
said "every level in the economy is stationary". That is the CONSTRUCTION `kp_vy` used, not the
condition: GS21's `bx7` also has stationary levels -- its x is AR(1) -- and its exposure premium is
still collected by a levels-augmented linear ridge. Measured on the live ten seeds, and the
comparator matters: DKKM beats plain `linlev` by +0.0235 in `bx7`, but beats `linlev_m` -- the same
ridge given the equal-weighted market unpenalised, as DKKM has it -- by only **+0.0015**, and in
`gx7` by +0.0006, `g28` +0.0004, `gsbase` -0.0025. So the levels channel is real but it is not
separable from finding 8: `linlev_m`'s winning penalty is the TOP of its grid (`1e5`) in 6 of 10
`bx7` seeds and 10 of 10 `gx7` seeds, where a ridge on eleven columns plus an unpenalised market
column is essentially the market.

The cleaner live statement of the levels channel is about which linear method wins. Where a beta
ladder exists, `linlev` beats `linrank` and becomes the linear winner: `gx7` 0.4462 against 0.3972
and `linlev` winning 7 of 10 seeds, `bx7` 0.2856 against 0.2826 and 5 of 10. Where it does not,
`linlev` loses: `g28` 0.2483 against 0.2527 and 4 of 10, `gsbase` 0.2453 against 0.2648 and 0 of 10.
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
| KP14 | E[R] affine in a SINGLE firm variable, the share of value in growth options, and the two priced channels are nearly parallel across firms | both common scaling (direction invariance, above) and rotation: with premia affine in one firm variable and the two priced channels nearly parallel across firms, there is nothing to rotate between. Among the live economies the only KP14 room above `kpbase`'s +0.0044 belongs to `vyx` (+0.0545) and `vyg25` (+0.0820), both built on the `exp(beta_f y)` exposure route |
| GS21 | one priced shock at a constant price, every firm loading on it in the same direction; leverage-to-premium smooth and monotone away from an unpopulated default boundary | cross-sectional room from the state, however nonlinear that state-dependence is: `g28` is a clipped-nonlinear gamma(x) with one exposure type and has the smallest room in this file, +0.0006. And the market reaches 93.6% to 97.4% of SR_max across the four GS21 economies, so the fourth condition fails even where room does appear (`gx7`, +0.0082) |

**The controlled experiment that IS live says something weaker than it used to.** The exposure
ladder was built in two models. In KP14, under corrected pricing, it produces room of +0.0545 and
+0.0820 and fair gaps of +0.0009 and +0.0148. In GS21 the same construction produces room of +0.0082
(`gx7`) and +0.0073 (`bx7`) and fair gaps of +0.0003 and -0.0017 -- fourteen and six times their
no-ladder parents' room, and no gap. So KP14's ladder still yields six to eleven times GS21's room,
but only one of its two economies converts that into a gap, and the converting one does it at
+0.0148. Before the pricing fix the same comparison was +0.36 against +0.008 of room and +0.125
against +0.000 of gap, which is why this paragraph used to be stated as a clean contrast. BGN cannot
run the ladder at all: there is no add-a-type knob.

And GS21 shows where its version of the room went. Adding the ladder to `g28` multiplies room by
fourteen, yet RAISES the best linear method's share of the attainable Sharpe from 85.7% to 92.7%,
while the measured gap FALLS from +0.0334 to +0.0232 and the fair gap stays at zero. A beta ladder is
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
most. `g0235r` leads at 110.4% on a Fama-MacBeth Sharpe of **0.0276** and `g0235s` follows at 89.5%
on **0.0376**; a DKKM improvement of +0.030 and +0.034 of Sharpe over a denominator that small is a
large percentage of very little, and both rows' fair gaps are NEGATIVE (-0.0005 and -0.0038). By
contrast `vyg25` sits 7th at 19.9% and `g0235f` 6th at 20.3% while carrying the file's only two
fair gaps that are positive in every seed, +0.0148 and +0.0079.

Two reasons the proportional column and the fair gap disagree, both already established here. In the
eight BGN and GS rows where the market spans most of the economy the winning DKKM portfolio is, to
within half a hundredth of Sharpe, the equal-weighted market (finding 8), so DKKM - FMR is mostly the
market's advantage over Fama-MacBeth rather than anything complexity found; the fair gap is what
remains once the linear methods hold that market on DKKM's terms. And a ratio is only informative
where its denominator is not near zero (the caution in "Reading the tables").

Ranked by DKKM - best fair linear the order would be vyg25, g0235f, g0235, g0235d, vyx, gx7, g0235r,
bx7, g28, g0235s -- almost the reverse of the table above at the bottom, where the two near-zero-
denominator BGN regimes lead the proportional ranking and trail this one.

**bgn_gam/g0235d is the clearest illustration of finding 4 in the file.** Its per-seed DKKM Sharpe
spans 0.1624 to 0.4919 -- a factor of 3.0 -- around a ten-seed mean of 0.3085, and its per-seed gap
runs -0.0077 to +0.0166 around a mean of +0.0037, so its seed 0 alone reads -0.0022: the wrong sign
for the economy. The screen it was run as missed both of its gates on that one seed, and the
ten-seed room, +0.0311, is indeed below the +0.05 gate, so the decision it drove was right; the
levels it was read off were not. It also has the largest DKKM-minus-market of any BGN economy,
+0.2190, and plain `linrank` follows it to within 0.004, which is finding 8 at its sharpest.

## Findings

Numbered as they were found, and ORDERED by what reads together: the numbers are cited from specs
and from `docs/refactor/WORKING.md`, so they are kept even where a superseded reading was deleted,
which leaves gaps. Finding 10 sits beside finding 8 because it is the same mechanism one method over.

11. **The ceiling is structural, it was predicted in 2026-08-07, and this campaign confirms it.** All
    three models are frictionless exact K-factor economies: `mu_t = -cov_t(R, m)` holds by
    construction, so with K priced aggregate shocks and N large, ANY set of at least K
    well-diversified portfolios with independent factor exposures attains the maximum Sharpe. The
    fair benchmark supplies seven. A 22-agent audit measured the resulting population ceiling on
    DKKM-over-linear by sweeping K with the total factor Sharpe held fixed (`voc_diagnosis/`,
    `findings.md` B3; the sweep is `voc_diagnosis/scripts/headroom3.py` and it reproduces exactly):

    | K | 1 | 2 | 3 | 5 | 10 | 20 |
    |---|---|---|---|---|---|---|
    | ceiling on DKKM / linear | 1.00x | 1.01x | 1.03x | 1.10x | 1.60x | 2.26x |

    **K read off each model's own log-SDF**, which is the part to check before trusting the rest. A
    state-dependent PRICE is not another shock, and a regime whose switch carries no kernel value is
    not another shock:

    | model | K | why |
    |---|---|---|
    | GS21 | **1** | the kernel is affine in the x innovation alone and is renormalised per regime, so `gmreg` and `gamma(x)` scale the price and the switch carries no kernel value (`gs_solve_reg.py:137-138`) |
    | BGN | **2** | `log M = -r - 0.5 sigma_z^2 - sigma_z nu`, and exactly two true factors are exported (`panel_functions.py:222-223`): cash flow and the rate. The `gmult` switch is unpriced |
    | KP14 | **2**, or **3** on the `vy` route | constant prices on three Brownians, but the baseline runs `gamma_v = 0` with one type at `beta_f = 0`, so nothing loads on the third |

    Measured against it, on the corrected campaign:

    | K | economies | predicted ceiling | measured DKKM / best fair linear |
    |---|---|---|---|
    | 1 | the four GS21 rows | 1.00x | 0.990 to 1.001, mean **0.994** |
    | 2 | five BGN regimes, both BGN and KP14 baselines | 1.01x | 0.949 to 1.049, mean **1.002** |
    | 3 | `vyx`, `vyg25` | 1.03x | 1.003 to 1.044, mean **1.023** |

    Monotone in K, and no economy exceeds its own K's ceiling even at the most extreme loading
    nonlinearity the sweep allows. The level sits about a hundredth below the population prediction,
    which is the right sign: the sweep is zero-estimation-error, and in live estimation DKKM reaches
    81% to 92% of its own ceiling against the linear side's 92% to 95%.

    **Three things follow, and they reframe this whole file.**

    - **"No model as published has a complexity gap" is a PREDICTION, not just a measurement.** At
      K = 1 and 2 the ceiling is 1.00x-1.01x, and the three baselines come in at 0.990, 0.995 and
      1.000. The negative result is what the structure requires.
    - **The pre-fix `vyx` and `vyg25` were the anomaly.** At K = 3 they reported 1.19x and 1.20x
      against a 1.03x ceiling. That should have been unreachable, and it was: the pricing defect is
      why. The corrected rows sit at 1.003 and 1.044.
    - **At K = 1 the ceiling is 1.000 at EVERY degree of loading nonlinearity**, which is a cell the
      published sweep did not cover and which this file's GS21 rows illustrate: an exposure ladder, a
      countercyclical price of risk and a two-state regime move the ratio by 0.011 in total. With one
      priced shock the shape of the exposure map cannot matter, because any portfolio with the right
      exposure already attains the maximum.

    **What DOES vary within a K is the loading nonlinearity**, and that is where the two surviving
    gaps come from. Reading each economy's implied nonlinear share off the same surface: `g0235f`
    about 0.91 and `vyg25` about 0.81, against roughly zero for the other eleven. So the surviving
    gaps are a loading-shape story inside a fixed K, not a dimension story -- which is also why the
    ceiling at the project's reach is a few hundredths of Sharpe and not a multiple.

8. **Where the market spans most of the economy, the measured gap is the market portfolio against
   linear methods not given it on the same terms.** DKKM appends the equal-weighted market to its random features UNPENALISED
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
   SR_max and DKKM sits 0.0006 to 0.0051 above it. In four of the five BGN regime economies it
   reaches 62% to 84%; the exception is g0235d at 26%, which is why that row has the largest
   DKKM-minus-market of any BGN economy, +0.2190 -- third in the file, behind vyg25 at +0.2974 and
   vyx at +0.2535, whose markets carry 11% and 14%. The
   `gx7` row makes the mechanism explicit: its fair benchmark is `mkt_est` -- the market alone -- in 8
   of 10 seeds, and DKKM's winning penalty sits at the grid's ceiling in 5 of 10 even after the
   ceiling moved from 10 to 1000. DKKM is asking to be the market, and being measured against it.
   The two extra decades bought it +0.0001 of Sharpe, which is the direct measurement behind the
   flat-tail clause in "A numerical choice that binds" below.

   **When does DKKM leave the market?** When there is Sharpe the market does not span, as a LEVEL.
   The non-market Sharpe sqrt(SR_max^2 - SR_ew^2) is 0.09 to 0.12 in the GS21 economies,
   where DKKM stays within 0.006 of the market; 0.04 to 0.25 across the BGN economies other than
   g0235d, where it ranges from 0.006 below to 0.076 above; 0.33 in g0235d, where it is 0.219 above;
   and 0.42 and 0.49 in vyx and vyg25, where it is 0.254 and 0.297 above. Leaving the market is
   necessary for a complexity gap and not sufficient, and after the pricing fix it is not nearly
   sufficient: vyx is SECOND in the file on DKKM-minus-market, at +0.2535, and its fair gap is
   +0.0009.

12. **BGN's regime path was closed on the wrong dimension: the stationary stress share is
    exhausted, the switching SPEED is not.** The path was closed on five ten-seed points "whose fair
    gap never left -0.008 to +0.004". Under corrected pricing that band is broken, and the way it
    breaks is systematic rather than a single outlier. Three of the five vary ONLY the switch
    probabilities, both scaled together, so the stationary stress share is 33.3% in all three and
    switching speed is the only thing that moves:

    | switch probabilities | spells, calm / stress | switches per 360-month window | fair gap | t | seeds positive | room |
    |---|---|---|---|---|---|---|
    | x0.2 | 240 / 120 mo | 4 | -0.0038 | -0.83 | 4 of 10 | +0.0070 |
    | x1 | 48 / 24 mo | 20 | +0.0025 | 2.69 | 8 of 10 | +0.0276 |
    | x4 | 12 / 6 mo | 80 | **+0.0079** | **4.78** | **10 of 10** | +0.0447 |

    Monotone in every column. The other dimension of the family is flat: holding the speed fixed and
    moving the stationary stress share 10% / 33% / 67% gives fair gaps of -0.0005, +0.0025, +0.0025.
    **So the share is exhausted and the speed is not**, and the closure confused the two.

    **It is a loading-nonlinearity result, not a dimension one, which finding 11 requires.** The
    regime switch is unpriced, so every BGN economy is K=2 whatever the regime does, and the ceiling
    is the same for all six. What moves is the SHAPE: a firm's conditional premium is its cash-flow
    beta times `gmult` at the current regime, a characteristic-times-observable-state product. At 4
    switches per window the estimation window holds essentially one regime, the product is locally a
    constant times beta, and a linear map suffices; at 80 switches both regimes are amply
    represented in every window and the interaction has to be carried. The test is against the
    linear method that DOES carry interactions and the market, `linrank_m`: DKKM beats it by
    **0.0001, 0.0026, 0.0079** along the same ladder, winning in 4, 9 and 10 seeds of ten. So this
    is not the finding-8 story of a benchmark denied the market.

    One more 4x step is available -- monthly probabilities 0.333 and 0.667, spells of 3 and 1.5
    months -- and the mechanism predicts saturation rather than a turnover, because the regime stays
    observable however fast it switches. Either answer is informative. The caveat is economic rather
    than numerical: at 1.5-month spells a "regime" has stopped being a business-cycle object and
    become a high-frequency shock to the price of risk.

10. **Fama-MacBeth beats the equal-weighted market exactly where the market is a minority of the
    attainable Sharpe, and the split is clean.** FMR holds the market (finding 8) yet loses to it in
    eight of the thirteen economies. Sorting all thirteen by the market's share of SR_max separates
    the two outcomes with no overlap:

    | | market share of SR_max | FMR - EW market |
    |---|---|---|
    | FMR wins, 5 economies | 11%, 14%, 26%, 35%, 54% | +0.0515 to +0.2392 |
    | the EW market wins, 8 | 62%, 75%, 82%, 84%, 94%, 95%, 95%, 97% | -0.0059 to -0.0797 |

    Every win is at or below 54% and every loss at or above 62%. The split survived the pricing fix
    intact: it is the same five economies on the same side, at market shares that moved by up to 20
    points. The mechanism is dilution, not
    absence: FMR's market leg competes with five zero-net characteristic legs for weight in an
    unshrunk six-factor MVE. Where the market is a minority of the attainable Sharpe there is real
    cross-sectional signal for those legs to earn, and FMR beats holding the market by up to +0.24.
    Where the market is nearly all of it -- the four GS21 economies at 94% to 97% -- the legs are
    competing for a few thousandths, and an unshrunk tilt away from the market costs more than it
    finds. This is finding 8 one method over: DKKM can collapse ONTO the market because it gets it
    unpenalised, and FMR cannot because it has to estimate its weight.

    Two consequences for reading the tables. The (DKKM - FMR) / FMR column that ranks them is
    measured against a benchmark that is itself below the market in eight economies, which is part of
    why it reaches 110.4% in g0235r. And "DKKM beats Fama-MacBeth" is a weaker statement than it
    sounds wherever the market alone also beats Fama-MacBeth.

4. **Single seeds mislead**, and the live ten seeds show it without appealing to any earlier run.
   `g0235d`'s own per-seed DKKM Sharpe spans 0.1624 to 0.4919 -- a factor of 3.0 -- around a mean of
   0.3085, and its per-seed gap runs -0.0077 to +0.0166 around a mean of +0.0037: its seed 0 alone
   reads -0.0022, the wrong sign for the economy. `g28`'s per-seed gap runs +0.0025 to +0.0756, a
   factor of 30, around +0.0334, with seed 0 at +0.0655 -- nearly double the truth. And `g28`'s seed 0
   has NEGATIVE all-month room, -0.00024, against a ten-seed +0.00034 and its own evaluation-window
   +0.00055: one seed can put a population quantity on the wrong side of zero. Cross-seed sd of the
   gap runs 37% of its mean (`g0235`) to 251% (`vyx`) across this file. The protocol's seed count is
   ten for every economy, and there is no longer any tier below it. **This finding is why the two
   surviving fair gaps are reported with a t and a per-seed sign count** rather than a mean alone: at
   +0.0148 and +0.0079 the cross-seed sd is 61% and 66% of the mean, so a three-seed run of either
   economy could have come back at zero.

5. **Absolute and proportional gaps rank the economies differently, and the tables here are ranked
   proportionally.** (DKKM - FMR) / FMR answers "how much does DKKM improve on Fama-MacBeth, relative
   to what Fama-MacBeth achieves", which is the comparison a reader of the DKKM paper expects. Its
   cost is that it is a ratio: it puts g0235r first at 110.4% on a Fama-MacBeth Sharpe of 0.0276 and
   g0235s second at 89.5% on 0.0376, both of which have negative fair gaps, while the two economies
   with a fair gap positive in every seed sit 6th and 7th. The orderings are near-reverses of each
   other at the bottom. So the ranking is proportional and the VERDICT is the fair gap; every table carries both
   columns, and the note under the parameterizations table states the divergence rather than leaving
   it to be discovered.

6. **Pre-registration record.** Thirteen specs carried written predictions into the protocol-v3
   campaign, committed before a single solve was built (`tests/test_precommitment_is_real.py`
   verifies that by git dates). The pattern this time: **every prediction about a population
   quantity held, every prediction about DKKM's LEVEL held, one fair-gap prediction was falsified
   and it was the one the project's headline rested on, and the prediction that the widened grid
   would stop binding failed again in three economies.**

   | clause | economies | outcome |
   |---|---|---|
   | population quantities: unchanged where the correction is zero, moved where it is not | 13 of 13 | right. `kpbase`, `gsbase`, `g28`, `gx7`, `bx7` reproduced SR_max, the market and room to four decimals; all six BGN rows' SR_max rose with the corrected J\*; `vyx` and `vyg25` fell below their predicted ceilings of 0.55 and 0.65, to 0.4214 and 0.4952 |
   | DKKM's level, predicted range | 9 of 9 | right, having been 4 of 13 in protocol v2. The reversal is not skill: v2's ranges were guesses about a grid that genuinely bound, v3's were mostly "the new decades will be inert", and they were |
   | fair gap within its stated bound | 12 of 13 | **`vyx` FALSIFIED**: predicted +0.01 to +0.06 and positive at ten seeds, came back +0.0009, positive in 7 of 10, t 0.3. Its spec's stated consequence was that the headline be withdrawn rather than reduced, and this file withdraws it |
   | `g0235f` "stays inside the BGN band" | 0 of 1, threshold not crossed | it was named in advance as the row that would breach first, and it did: +0.0034 to +0.0079, outside the band's +0.0034 top and positive in ten seeds of ten. Its FALSIFICATION threshold was +0.01 and that was not reached, so nothing fired automatically -- see "Open proposals" |
   | the winning penalty is interior in at least 8 of 10 seeds | 8 of 11 | **wrong for g0235s (6), bx7 (7) and gx7 (5)** -- and this is the clause that turned out to be badly posed; see the gate section below |
   | oracle mean expected excess return in its predicted band | 1 of 2 | `vyx` right at 4.2% against 3% to 8%; **`vyg25` wrong at 3.9% against 4% to 11%**, missing the floor by 0.14 of a percentage point |
   | `g0235d`'s per-seed DKKM spread stays wide | 1 of 1 | right: 0.1624 to 0.4919, a factor of 3.0, against a predicted "factor of 3.2 in v2" |
   | `g0235s`'s (DKKM - FMR) / FMR stays uninformative | 1 of 1 | right: 89.5% on an FMR Sharpe of 0.0376 |

   **The one clause worth dwelling on is `vyg25`'s, because it was the only prediction that tied two
   economies together and it was half wrong in an informative way.** `vyx`'s spec said the two rows
   "carry the project's only complexity gap" and that falsifying `vyx` would mean "this project has
   no economy with a gap". `vyx` was falsified and `vyg25` was not: at gamma_v 2.5 against 1.8, and
   nothing else different, the gap survives at +0.0148 in ten seeds of ten while at 1.8 it does not
   survive at all. The conditional attached to the falsification was wrong, and the thing that made
   it wrong is the price of the state's risk -- which is finding 9.

9. **The gap is a function of the price of the state's risk, and at the corrected prices it is
   nearly all of what is left.** `vyx` and `vyg25` differ in one parameter: `gamma_v`, 1.8 against
   2.5. That 39% increase in the price of the OU state's risk takes the fair gap from +0.0009 (t
   0.3, positive in 7 seeds) to +0.0148 (t 5.2, positive in 10), the room from +0.0545 to +0.0820,
   and DKKM's share of its own population ceiling from 83.3% to 81.7% -- so the gap does not come
   from DKKM estimating better, it comes from there being more to estimate. The linear side reaches
   95.3% and 94.7% of ITS ceiling in the two economies, essentially unchanged, and the extra room a
   higher price of risk creates is room a rank-and-interaction linear ridge cannot reach.

   Two cautions on reading this as a ladder. It is two points, and the pre-fix version of the same
   comparison (+0.1251 against +0.1636, a 31% rise for the same 39% parameter change) was measured
   on solves that are now known to be wrong, so it is not a third point. And `gamma_v` 2.5 is
   already near the top of what this calibration tolerates: at 2.5 the oracle's mean expected excess
   return is 3.9% a year, LOWER than at 1.8 (4.2%), because a higher price of risk lowers claim
   values faster than it raises premia in the corrected solve. A ladder in `gamma_v` is therefore
   not a ladder in plausibility, and the economy that would settle this is a third point between
   1.8 and 2.5 rather than one above it.

## Open proposals

| proposal | scope | what it would decide | status |
|---|---|---|---|
| ~~A wider ridge grid~~ | universal | whether the censored rows' DKKM levels were materially higher | **CLOSED 2026-09-22.** Done: `1e-5 ... 10` became `1e-7 ... 1000`, all thirteen re-ran, and the answer is no. In the four `gs_bx` economies, where the effect can be isolated because no solve changed, two extra decades above the old ceiling moved DKKM by at most +0.0001 -- including in `gx7`, which had been censored at the ceiling in 8 of 10 seeds. See the gate section |
| **BGN's regime SPEED ladder** | BGN | how much further the fair gap climbs with switching frequency. **The closure was decided 2026-09-23 and does not stand** -- see finding 12. The share dimension is exhausted; the speed dimension is monotone and unexhausted, with roughly one more 4x step available before the monthly switch probability hits its bound | OPEN, and the cheapest live lever in the file: a pure parameter override, one 3-minute J\* rebuild, ten seeds at 40G |
| ~~K6, gamma_v 1.2 with vyx's exposures~~ | KP14 | whether a defensible calibration shows a gap | **ANSWERED 2026-09-22 by the pricing fix, not by an economy.** K6 existed because `vyx` and `vyg25` sat at 18.2% and 22.9% expected excess return a year and a referee would not accept them. Corrected, they sit at 4.2% and 3.9%, inside the file's 3.3%-to-12.6% band, and the gap at that calibration is +0.0009 and +0.0148. Lowering `gamma_v` further would lower the gap, not defend it |
| **K7, a third point in gamma_v** | KP14; lever universal, closed in BGN and GS21 | where between 1.8 and 2.5 the fair gap becomes distinguishable from zero. Finding 9 is two points and the interesting structure is between them, not above 2.5 | OPEN, the leading candidate now: one G solve plus one set of integrals, about 25 min, then ten seeds |
| K1, a continuum of exposures | KP14; ~10x the cost in GS21, no knob in BGN | fifteen types over [0, 0.14], shares right-skewed. A smooth exposure map suits random features and lowers the market's average exposure | OPEN, and much cheaper than this row used to say: the integral stage is about 20 min for THREE economies, not 90 min for one, so fifteen types is roughly 1.7 h of integrals then 30 h of seeds |
| K3, persistence of the priced state | KP14; answered in BGN as B1, frozen in GS21 | kappa_y 0.15 and 0.70. Now more interesting than it was: the corrected solve prices the state through a Girsanov adjustment that SATURATES at `b gamma_v sigma_y / kappa_y`, so `kappa_y` scales the whole correction and is no longer just a timing knob | OPEN, two solves of about 15 min each |
| K2, a rare extreme type | KP14 and GS21 only | shares (0.45, 0.45, 0.10), top loading 0.14 to 0.20 | OPEN, low priority; the discount check that withdrew K5 applies |
| G3, the GS21 default-channel probe | **GS21 only** | whether GS21 has any non-market Sharpe: equity near default is a convex claim on the same shock, so its loading rises as the state worsens | OPEN, 20 min, oracle only. Its gate needs an evaluation-window room clause of at least +0.05 as well as a market share below 85% -- KP14's baseline meets the market clause at 35% and has no gap |

**On the scope column.** The ridge grid was the one universal proposal and it is now closed. Of
what remains, three are model-specific PARAMETERS of a universal LEVER -- every model has a price-of-risk knob and a
persistence knob -- but the lever is exhausted or frozen elsewhere: BGN climbed the price ladder to
its closed-form bound and its persistence ladder is B1, four ten-seed points that never moved the
fair gap; GS21's slope is fixed and its `rho_x` is pinned by a parity test; the BGN regime question is not a
new lever but a re-reading of five points already measured. K1 is portable to GS21 in
principle and costs one solve per type there, so fifteen types is about 75 h of solves against 7.5 h
of integrals in KP14, and BGN has no add-a-type knob at all. G3 is the only strictly single-model
proposal: neither BGN's real-option projects nor KP14's projects carry debt, so there is no channel
to wake. See "What each model can express" for why porting is mostly pointless rather than merely
expensive.

Closed, with what closed them: GS21's exposure path (gx7's pre-registered negative
fired, and the corrected campaign confirms it at +0.0003, unchanged to four decimals); K6, which the
pricing fix answered; the ridge grid, which was done and found inert; K5, signed exposures (the economy does not
exist -- the G solve discounts growth options at rho_ty, which goes negative for a loading of -0.06,
and a negative discount removes the operator's dissipation); the window and penalty ladders (absorbed
into the protocol, which fixes both for every economy).


### A numerical choice that turned out not to bind: the ridge grid

DKKM's reported Sharpe is a **maximum over the ridge-penalty grid**, so where the winning penalty
sits at an EDGE of that grid the number may be censored -- it may say where the search stopped rather
than what the economy affords. `variants/penalty_gate.py` reads the winning penalty per seed and is
the first thing to run on any campaign, before any number in this file is read.

**The 2026-09-21 amendment did what it was for, and what it found is that the ceiling was never the
problem.** The grid went from seven values, `1e-5 ... 10`, to eleven, `1e-7 ... 1000`. On the FLOOR
side it worked exactly as intended: the v2 campaign had `vyx` winning at the floor in 7 seeds of ten
and `vyg25` in 6, and after the amendment neither has a single seed at the floor -- they win at
0.001 and 0.01, two to four decades above the old floor. On the CEILING side it bought nothing
measurable, and the four `gs_bx` economies are the clean test, because no solve of theirs changed
and the grid is the only difference between their v2 and v3 rows: two extra decades above the old
ceiling moved DKKM by at most **+0.0001** of Sharpe, including in `gx7`, which the v2 gate reported
as censored at the ceiling in 8 of 10 seeds.

| economy | interior | at the floor `1e-7` | at the ceiling `1000` | winning penalties | what the ceiling buys | reading |
|---|---|---|---|---|---|---|
| bgn_gam/bgnbase | 10/10 | 0 | 0 | 0.01x10 | 0.0e+00 | clean |
| bgn_gam/g0235d | 10/10 | 0 | 0 | 0.01x3 0.1x7 | 0.0e+00 | clean |
| bgn_gam/g0235f | 10/10 | 0 | 0 | 0.01x3 0.1x7 | 0.0e+00 | clean |
| gs_bx/gsbase | 10/10 | 0 | 0 | 0.1x4 1x6 | 0.0e+00 | clean |
| kp_vy/kpbase | 10/10 | 0 | 0 | 0.01x7 0.1x3 | 0.0e+00 | clean |
| kp_vy/vyg25 | 10/10 | 0 | 0 | 0.001x9 0.01x1 | 0.0e+00 | clean |
| kp_vy/vyx | 10/10 | 0 | 0 | 0.001x8 0.01x2 | 0.0e+00 | clean |
| bgn_gam/g0235 | 9/10 | 0 | 1 | 0.01x3 0.1x4 1x1 10x1 1000x1 | 2.0e-06 | clean |
| gs_bx/g28 | 9/10 | 0 | 1 | 0.1x2 1x7 1000x1 | 1.5e-06 | clean |
| bgn_gam/g0235r | 8/10 | 0 | 2 | 0.01x7 1x1 1000x2 | 1.5e-06 | clean |
| **gs_bx/bx7** | 7/10 | 0 | 3 | 0.1x2 1x5 1000x3 | 2.7e-05 | **flat tail** -- the argmax is at the ceiling, the curve is not |
| **bgn_gam/g0235s** | 6/10 | 0 | 4 | 0.01x6 1000x4 | 2.2e-06 | **flat tail** -- the argmax is at the ceiling, the curve is not |
| **gs_bx/gx7** | 5/10 | 0 | 5 | 1x2 10x3 1000x5 | 1.2e-05 | **flat tail** -- the argmax is at the ceiling, the curve is not |

**Why an argmax at the ceiling is not censoring, which is the gate's own correction.** As kappa goes
to infinity the ridge coefficient `(X'X + kI)^-1 X'y` tends to `X'y / k`: the portfolio DIRECTION
stops depending on kappa, and a Sharpe ratio is scale-invariant. `sharpe(kappa)` therefore has a
HORIZONTAL ASYMPTOTE, and once the grid reaches it the argmax lands on whichever ceiling node wins
in the sixth decimal while every further decade reproduces that exactly. Measured across all 130
seeds of this campaign: 16 put the argmax at `1000`, and not one of them gained more than
**2.7e-05** of Sharpe over its best interior penalty -- three orders of magnitude below the smallest
gap this file reports, and below half of the last digit it prints.

So the gate now asks two questions and only the second is decisive: is the argmax interior in at
least eight of ten seeds, and if not, what does the edge BUY over the best interior penalty? A row
that fails the first and passes the second is reported as a **flat tail** and its number is not
censored. Three rows are in that state and all three are at the ceiling; nothing is at the floor
anywhere in the file. `tests/test_penalty_gate.py` pins both halves: a tail that is still climbing
must still fail, and the tolerance must stay below what this document prints.

**What this cost and what it is worth knowing.** Reading the v2 gate literally would have bought a
third grid decade and another 130-job campaign -- about 990 node-hours -- to move three numbers by
less than 3e-05. The general lesson is the one the gate's docstring now carries: an argmax at an
edge is a question, not an answer, and the question is about the derivative there, not the position.

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
4. Ten seeds, or the economy does not go in a table, and all ten before ANY of them is aggregated:
   `runstamp.stem` puts no spec version in a filename, so a re-run overwrites in place and a partial
   one makes `aggregate_seeds.py` average two spec versions into one `MIXED:` row at `n_seeds = 10`,
   which no test catches. If the economy's peak memory has never been measured, run one seed and read
   `sacct` before sizing the array.
5. **`python variants/penalty_gate.py` before reading any number**, then
   `python variants/aggregate_seeds.py`. The gate fails only if a winning penalty sits at a grid edge
   AND the edge buys more than `penalty_gate.TOL` over the best interior penalty; a row that fails on
   position but passes on materiality is a flat tail and goes in the gate table above as one. A row
   that fails both has a censored DKKM Sharpe and must be reported as a lower bound, with the
   direction. Commit the result files and the table; add the economy here with what differs from the
   baseline in economic terms first and what the result decided against the prediction.
   `tests/test_results_md_matches_table.py` fails until the row is added.
6. Push, then pull the shared checkout on the cluster with `bash variants/cluster_pull.sh`, never a
   plain `git pull`: both clusters import their Python from that one tree
   (`docs/RUNS.md`, "Where output goes").
