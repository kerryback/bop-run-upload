# Experimental results

Every economy run through the oracle-and-estimator pipeline: what it was built to test, what it
produced, and what it decided. One measurement protocol governs all of them, stated once below, so
that the only thing separating two rows of a table is the economy. Last restructured 2026-09-15, when
the protocol landed and everything that did not adhere to it was deleted rather than kept as
"legacy".

The tables here are checked cell by cell against `variants/results/economy_table.csv`
(`variants/aggregate_seeds.py`) and `variants/results_e1/fair_gap_economy_table.csv`
(`variants/fair_gap.py`) by `tests/test_results_md_matches_table.py` -- every table that uses the
column set below, not just one section -- so this file cannot fall behind the numbers without the
suite saying so. Where each job ran: `docs/RUNS.md`. What to run next, and why: `docs/NEXTUP.md`.

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
| ridge-penalty grid | `1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10` | same |
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

`tests/test_protocol_is_uniform.py` pins every one of these claims: each model's burn-in literal, the
seed array's sample and grid, every live spec's `panel` and `estimation` block, and the uniformity of
solve precision within each model. It exists because none of it was checked before, and all four of
the panel-side choices had drifted -- burn-in sat at 300 / 400 / 300 while twelve of nineteen specs
declared 200 (`config.py`'s value, from the legacy `main.py` tree); the ridge grid was set per `case`
branch of the seed array, so four values reached KP14 and BGN, eight reached GS21, and five reached
one KP14 economy; that economy also ran T=860 and a 720-month window; and the three baselines
narrowed their feature bases to the state their own paper has.

**Two things that decision costs, stated plainly.** Widening the ridge grid is not free in
interpretation: DKKM's reported Sharpe is a **maximum over that grid**, so a grid whose edge wins
reports the grid rather than the economy. The grid is now wide on both sides so the argmax can be
interior, and that it *is* interior is checked per economy, not assumed. And giving the baselines
their model's full conditioning set hands each one a column its paper's model does not drive -- an
inert regime for BGN and GS21, a state no firm loads on for KP14. That is a noise column, and it
costs the baseline some estimation precision. It is the price of baseline and parameterization being
measured the same way, which is the only reading under which "what the route added" is a difference
in the economy alone.

**Retired with the protocol.** `kp_vy/vyxT860` is gone, not re-run. Its economy *is* `kp_vy/vyx` --
the same parameters and the same solves -- and it differed only in T, the window and the ridge grid.
Reporting it beside `vyx` made a sample length look like an economic finding, and it did: a quarter
of its +0.1875 gap was the extra penalty decade its siblings never saw. The question it asked,
whether DKKM's shortfall from its own ceiling is data, is now asked on protocol, because the shared
wide grid removes the censoring that reading rested on. Its spec records the retirement in
`lineage.retired`.

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
columns, in this order. A cell reads "--" where the quantity was not measured.

- **economy** -- the model directory and run tag, `model/tag`, sometimes followed by what sets the
  economy apart.
- **seeds** -- how many independently simulated panels the row averages.
- **SR_max** -- the largest conditional Sharpe any portfolio of the firms could attain, sqrt(mu'
  Sigma^-1 mu), averaged over the evaluation months. An upper bound on every other Sharpe in the row.
- **EW market SR** -- the Sharpe of the equal-weighted market portfolio.
- **FMR SR** -- the Sharpe of the rolling Fama-MacBeth regression portfolio. The benchmark of both
  percentage columns.
- **best linear SR** -- in each seed, the Sharpe of the best of four linear methods: Fama-MacBeth,
  Fama-French, and ridge regression on rank-standardised characteristics with and without level
  features.
- **DKKM SR** -- in each seed, the Sharpe of the best random-feature ridge estimator, over 36, 360 and
  3600 features, the penalty grid, and its four variants.
- **DKKM - FMR** -- DKKM SR minus FMR SR. An absolute difference in Sharpe units; the standard
  deviation across seeds is in parentheses.
- **(DKKM - FMR) / FMR** -- [(DKKM SR) - (FMR SR)] / (FMR SR), in percent. How much DKKM improves on
  Fama-MacBeth, relative to what Fama-MacBeth achieves. Where Fama-MacBeth's Sharpe is near zero, as
  in the slow and rare BGN regime economies (g0235s, g0235r), this is very large and says nothing
  about DKKM.
- **DKKM - best linear** -- DKKM SR minus best linear SR. Absolute, sd in parentheses. The "gap" in
  this file's prose.
- **DKKM - best fair linear** -- DKKM SR minus the Sharpe of the best of seven linear methods: the
  four above, the two ridge methods given the equal-weighted market as a separate unpenalised column
  as DKKM has it, and the market alone with its weight estimated. Absolute, sd in parentheses. The
  "fair gap" in the prose, and the complexity gap outside KP14 (finding 8).
- **room** -- the population headroom for nonlinearity over the evaluation months: the best Sharpe a
  nonlinear feature basis reaches with one fixed coefficient vector, minus the best a rank-linear
  basis reaches the same way, both computed from the true moments. No estimation involved. Absolute,
  sd in parentheses.
- **room / FMR** -- (room) / (FMR SR), in percent: the headroom relative to what Fama-MacBeth
  achieves.
- **t, DKKM vs FMR** -- the largest paired t-statistic, across the evaluation months, of any
  random-feature estimator's monthly Sharpe against Fama-MacBeth's, averaged over seeds.

**The columns of every prediction table.**

- **economy** -- the economy the spec made the prediction for.
- **quantity** -- what was predicted, named as in the economy tables where it is one of their columns.
- **reference** -- the parent economy's value the prediction started from, or "--".
- **predicted** -- the range written into the spec before the run, in the quantity's own units.
- **result** -- the measured value; se is its standard error across seeds. After a room or a gap, its
  percentage of FMR SR.
- **verdict** -- right, or wrong with the direction; and whether a falsification line or a gate was
  crossed.

**Two cautions.**

1. **Room is not a ceiling on the gap and not a screen for it.** The gap exceeds the room in every
   BGN and GS economy, because there it is the market shortfall; in KP14 it is 29% (vyx) and 35%
   (vyg25) of the room. Ranking candidates by room does not rank them by gap.
2. **Every percentage is a RATIO OF MEANS, not a mean of per-seed ratios.** In g0235, (DKKM - FMR) /
   FMR is 94.2% as the ratio of the ten-seed means and 110.3% (sd 71.0) as the mean of per-seed
   ratios, because Fama-MacBeth's Sharpe has sd 0.0208 on a mean of 0.0557. In g0235r and g0235s it
   is below zero in some seeds, where a per-seed ratio has no meaning at all. Both versions live in
   `economy_table.csv` (`gap_fm_over_fm`, `gap_fm_pct_fm_mean`).

## Status: every number below predates the protocol

The protocol landed 2026-09-15. The numbers in this file were measured before it, and they do not
adhere to it in two respects: BGN's panels burned in 300 months and GS21's 300 against the protocol's
400, and the ridge grid was `1e-3 … 1` for BGN and KP14 and `1e-3 … 10` for GS21 rather than the
shared `1e-5 … 10`. The campaign that brings all thirteen economies onto the protocol is
`docs/RUNS.md`, campaign 2026-09-15; every row here is replaced by it.

Two consequences worth knowing before quoting anything below:

- **The KP14 Path 1 rows are lower bounds.** In `vyx` the winning penalty was the smallest on its
  grid in 9 of 10 seeds and in `vyg25` in 10 of 10, so DKKM's Sharpe there is censored. `vyx`'s
  measured gains per penalty decade at this window were 0.154, 0.100 and then 0.035, so the two
  decades the protocol adds below the old floor should raise DKKM, and the gap with it.
- **The baselines are measured on narrowed feature bases.** They saw only the state their paper has;
  the parameterizations saw the full set. So "what the route added" currently carries one protocol
  difference alongside the economic one, which is the difference the protocol removes.

`bgn_gam/g0235d` also has one seed rather than ten, reported under its own SCREEN heading; the
campaign gives it the protocol's ten.

## The three models as published

Each paper's economy, before any parameterization was built on it. Every "what the parameterization
added" statement in this file is a difference against these rows.

| economy | seeds | SR_max | EW market SR | FMR SR | best linear SR | DKKM SR | DKKM - FMR | (DKKM - FMR) / FMR | DKKM - best linear | DKKM - best fair linear | room | room / FMR | t, DKKM vs FMR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bgn_gam/bgnbase: BGN as published | 10 | 0.2805 | 0.1405 | 0.2008 | 0.2148 | 0.2214 | +0.0206 (0.0170) | 10.2% | +0.0066 (0.0099) | +0.0012 (0.0071) | +0.0225 (0.0052) | 11.2% | 15.9 |
| kp_vy/kpbase: KP14 as published | 10 | 0.2292 | 0.0797 | 0.1930 | 0.2020 | 0.2073 | +0.0144 (0.0099) | 7.4% | +0.0053 (0.0076) | -0.0024 (0.0046) | +0.0044 (0.0011) | 2.3% | 17.6 |
| gs_bx/gsbase: GS21 as published | 10 | 0.2980 | 0.2838 | 0.2728 | 0.2782 | 0.2908 | +0.0179 (0.0182) | 6.6% | +0.0125 (0.0120) | -0.0022 (0.0014) | +0.0012 (0.0001) | 0.5% | 25.2 |

What the anchor says:

- **No model as published has a complexity gap.** The fair gaps are +0.0012, -0.0024 and -0.0022. In
  GS21 a linear method given the market beats DKKM in all ten seeds (t -5.0); in KP14 in seven; in
  BGN DKKM is ahead in six seeds by amounts that average +0.0012. The measured gaps, +0.005 to
  +0.013, go once the linear methods hold the market as DKKM does.
- **DKKM leaves the market wherever the market leaves Sharpe, and the linear methods follow it.** In
  the BGN and KP14 baselines the market carries 50% and 35% of the attainable Sharpe; DKKM is 0.081
  and 0.128 above it, the best linear method 0.074 and 0.122 above it, and in no seed of either does
  DKKM's most-shrunk portfolio equal the market. In GS21, where the market carries 95%, it does in
  six seeds of ten, as in g28. Non-market Sharpe of 0.21 to 0.24 without nonlinear room gives no gap.
- **Room alone is not enough and a low market share alone is not enough.** BGN as published has room,
  +0.0225, and a market at only 50%, and a fair gap of +0.0012. KP14 as published has a market at
  35% and room of +0.0044. Only an economy with both -- KP14 Path 1's +0.36 of room and a market at
  29% -- produces a gap.

**How each baseline is produced.**

| economy | spec | what it is in the code | solve |
|---|---|---|---|
| bgn_gam/bgnbase | var-bgn_gam-bgnbase-v2 | `bgn_gam` at gmult [1, 1]: both regimes price the market shock at sigma_z 0.4 and the regime is inert; reproduces the paper's economy to machine precision. Table I: 11 of 11 parameters match | jstar, Mac, about 8 min, table committed |
| kp_vy/kpbase | var-kp_vy-kpbase-v2 | `kp_vy` with one type at beta 0, gamma_v 0, bv_comp 0: nothing depends on the state y (the 21 integral tables agree across y to 4e-12); r = 0.05 as in every KP14 economy here, the paper's 0.025 being the one standing departure. Table II: 17 of 18 match | G and integ, Mac, 6 min, tables committed |
| gs_bx/gsbase | var-gs_bx-gsbase-v2 | `gs_bx` with one type under the regime solver at gmreg [1, 1]: gamma_x 0.5 in both regimes, the paper's general-equilibrium kernel replaced by that stand-in | sol_gsbase, Sol, 5.6 h, published content-addressed |

## Parameterizations

The nine ten-seed economies built on those three, ranked by DKKM - best fair linear;
`bgn_gam/g0235d`, which has one seed, is reported below rather than ranked among them. Each differs from its
model's baseline in parameters or driving forces only; nothing in the columns below differs in how it
was measured.

| economy | seeds | SR_max | EW market SR | FMR SR | best linear SR | DKKM SR | DKKM - FMR | (DKKM - FMR) / FMR | DKKM - best linear | DKKM - best fair linear | room | room / FMR | t, DKKM vs FMR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kp_vy/vyg25 | 10 | 1.3887 | 0.4694 | 0.7911 | 0.8000 | 0.9450 | +0.1540 (0.0189) | 19.5% | +0.1450 (0.0193) | +0.1450 (0.0193) | +0.4108 (0.0446) | 51.9% | 39.2 |
| kp_vy/vyx | 10 | 1.1778 | 0.3675 | 0.6389 | 0.6428 | 0.7475 | +0.1086 (0.0165) | 17.0% | +0.1046 (0.0155) | +0.1046 (0.0154) | +0.3606 (0.0394) | 56.4% | 30.2 |
| bgn_gam/g0235f | 10 | 0.2043 | 0.1209 | 0.0847 | 0.1115 | 0.1325 | +0.0479 (0.0320) | 56.5% | +0.0211 (0.0154) | +0.0025 (0.0029) | +0.0350 (0.0188) | 41.3% | 16.6 |
| bgn_gam/g0235s | 10 | 0.0946 | 0.0516 | 0.0145 | 0.0436 | 0.0616 | +0.0472 (0.0271) | 325.7% | +0.0180 (0.0127) | +0.0012 (0.0034) | +0.0119 (0.0267) | 82.0% | 43.8 |
| bgn_gam/g0235 | 10 | 0.1461 | 0.1061 | 0.0557 | 0.0855 | 0.1081 | +0.0525 (0.0342) | 94.2% | +0.0227 (0.0084) | +0.0006 (0.0009) | +0.0176 (0.0119) | 31.7% | 33.4 |
| gs_bx/gx7 | 10 | 0.4473 | 0.4385 | 0.3738 | 0.4163 | 0.4393 | +0.0655 (0.0396) | 17.5% | +0.0229 (0.0102) | +0.0001 (0.0006) | +0.0053 (0.0044) | 1.4% | 15.0 |
| gs_bx/bx7 | 10 | 0.3121 | 0.2926 | 0.2741 | 0.2887 | 0.2964 | +0.0223 (0.0160) | 8.1% | +0.0078 (0.0049) | -0.0009 (0.0028) | +0.0066 (0.0041) | 2.4% | 16.0 |
| gs_bx/g28 | 10 | 0.3087 | 0.2944 | 0.2518 | 0.2692 | 0.2975 | +0.0457 (0.0221) | 18.2% | +0.0283 (0.0136) | -0.0022 (0.0023) | +0.0004 (0.0003) | 0.2% | 24.6 |
| bgn_gam/g0235r | 10 | 0.0559 | 0.0472 | 0.0003 | 0.0109 | 0.0318 | +0.0315 (0.0199) | 9,675.8% | +0.0209 (0.0166) | -0.0025 (0.0069) | +0.0026 (0.0020) | 813.4% | 80.0 |

**What differs from the baseline, in economic terms.**

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

**Read DKKM - best fair linear, not DKKM - best linear or DKKM - FMR.** In the seven BGN and GS rows
the winning DKKM portfolio is, to within half a hundredth of Sharpe, the equal-weighted market
(finding 8); the fair gap is what remains once the linear methods are given that market as DKKM has
it, and it is at most +0.0025 and negative in three rows. The two KP14 rows' linear methods beat the
market by 0.28 and 0.33, so their fair gap is their gap. Ranked by DKKM - best linear the order would
be vyg25, vyx, g28, gx7, g0235, g0235f, g0235r, g0235s, bx7. Ranked by (DKKM - FMR) / FMR it would be
led by g0235r at 9,675.8% and g0235s at 325.7%, where Fama-MacBeth's mean Sharpe is 0.0003 and
0.0145: a percentage of a Sharpe near zero says nothing about DKKM.

### bgn_gam/g0235d -- SCREEN, one seed

| economy | seeds | SR_max | EW market SR | FMR SR | best linear SR | DKKM SR | DKKM - FMR | (DKKM - FMR) / FMR | DKKM - best linear | DKKM - best fair linear | room | room / FMR | t, DKKM vs FMR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bgn_gam/g0235d | 1 | 0.1812 | 0.0672 | 0.1307 | 0.1480 | 0.1484 | +0.0177 | 13.5% | +0.0004 | +0.0004 | +0.0186 | 14.2% | 8.1 |

The one economy above that has not had ten seeds. It ran as a pre-registered screen whose gates were
evaluation-window room of at least +0.05 AND non-market Sharpe of at least 0.30; it missed both
(+0.0186 and 0.1541), which closed BGN's regime path under the spec's own rule. Its one seed put DKKM
0.081 above the market -- more than any BGN ten-seed mean -- with plain `linrank`, whose market column
is penalised, following to within 0.0004. A single seed ranks nothing (finding 4), which is why this
row sits under its own heading. The campaign gives it ten.

## Findings

Numbered as they were found; the numbers are cited from specs and from
`docs/refactor/WORKING.md`, so they are kept even where the gaps show that a superseded reading was
deleted.

8. **Outside KP14 Path 1, the measured gap is the market portfolio against linear methods not given
   it on the same terms.** DKKM appends the equal-weighted market to its random features UNPENALISED
   (`--include_mkt`, as in the paper). `linrank` and `linlev` carry the same market as their constant
   column but penalise it with everything else (in g28 linrank's Sharpe goes from 0.256 unpenalised
   to 0.065 at the smallest penalty on the grid); Fama-French carries a value-weighted market;
   Fama-MacBeth none. At its largest penalty DKKM collapses onto the market: in 54 of 70 BGN and GS
   seeds its Sharpe there equals the market's to within 0.001, identically across 36, 360 and 3600
   features, and its winning portfolio is barely off that point. The split reads off the Sharpe
   columns of any table above: DKKM SR minus EW market SR is what DKKM adds over the market, and EW
   market SR minus best linear SR is what the market has over the best linear method.

   **When does DKKM leave the market?** When there is Sharpe the market does not span, as a LEVEL:
   the per-seed non-market Sharpe sqrt(SR_max^2 - SR_ew^2) ranks with DKKM's margin over the market
   at Spearman +0.76 across the 80 seeds (+0.65 without vyx); its share of SR_max does not (+0.06).
   Over the 70 BGN and GS seeds, seeds with non-market Sharpe up to 0.05 put DKKM 0.008 below the
   market, 0.10 to 0.20 put it 0.005 above, the two above 0.30 put it 0.089 above; vyx's ten, at
   1.12, put it 0.38 above. Leaving the market is necessary for a complexity gap and not sufficient:
   the BGN and KP14 baselines have non-market Sharpe of 0.24 and 0.21 and evaluation-window room of
   +0.0225 and +0.0044, against vyx's +0.3606. Evidence: `variants/score_market.py`,
   `variants/market_decomposition.py`, `variants/results/market_sr.csv`.

4. **Single seeds mislead.** g28's seed 0 gave +0.0119 against a ten-seed +0.0283; bx7's seed 0 put
   its evaluation-window room above its all-month room and ten seeds reversed it; vyx's seed 0 read
   "stronger than published" and ten seeds read "unchanged". Cross-seed sd of the gap is 15% of the
   mean for vyx, 37% for g0235, 48% for g28, 63% for bx7. No ranking below the top of a single-seed
   table means anything, which is why the protocol's seed count is ten for every economy and why
   g0235d is not ranked.

5. **Absolute and proportional gaps rank the economies differently**, and the proportional ordering
   is the one finding 8 broke: it put g0235r first at 192% on a linear Sharpe of 0.011. The
   proportional column is kept for the economies where the linear methods are far from zero; the
   ranking in this file is by fair gap.

6. **Pre-registration record.** Of the eleven specs run with written predictions, the two
   pre-registered decisions fired as written (gx7's negative; g0235d's gate), and E1's eight bounds
   and the three baselines' no-gap predictions all held. The level predictions mostly missed on the
   parameterizations and mostly held on the baselines: g0235f's on both halves, g0235s's and
   g0235r's on their second, vyg25's three level predictions low, BGN's baseline room low, none
   falsified. The lessons, each learned once: when a parameter enters the SOLVE, check whether the
   solved table moved before assuming the cross-section did not (g0235f); decompose a gap against the
   market before explaining it (finding 8); a range built on one legacy seed inherits that seed's
   draw (the BGN baseline's room); and a gain-per-decade extrapolation at one window does not carry
   to another -- which is now a protocol matter rather than an experiment, since every economy
   shares one window and one grid.

9. **Where the gap is genuine, it is bounded by DKKM's estimation shortfall.** In both KP14 Path 1
   economies the linear methods reach 99% to 100% of their population ceiling and DKKM 74% and 78% of
   its, with the smallest penalty on the grid winning in 19 of 20 seeds. Raising the price of the
   state's risk from 1.8 to 2.5 raised DKKM's share of its ceiling from 74% to 78% and the gap by
   39%. That the grid floor bound at all is what the protocol's wider grid is for: until the argmax
   is interior, the reported DKKM Sharpe is a statement about the grid.

## Open proposals

| proposal | what it would decide | status |
|---|---|---|
| K6, gamma_v 1.2 with vyx's exposures | whether a defensible calibration shows a gap: the price ladder's low end, E[mu] about 14% a year against vyx's 18.2% and vyg25's 22.9% | OPEN, and now the leading candidate: the referee-facing number |
| K1, a continuum of exposures | fifteen types over [0, 0.14], shares right-skewed. A smooth exposure map suits random features and lowers the market's average exposure | OPEN, about 7.5 h of integrals then 30 h of seeds |
| K3, persistence of the priced state | kappa_y 0.15 and 0.70. Informative about mechanism, not about the size of the gap: slower reversion bends values more but gives a window fewer cycles | OPEN, two solves of about 90 min |
| K2, a rare extreme type | shares (0.45, 0.45, 0.10), top loading 0.14 to 0.20 | OPEN, low priority; the discount check that withdrew K5 applies |
| G3, the GS21 default-channel probe | whether GS21 has any non-market Sharpe: equity near default is a convex claim on the same shock, so its loading rises as the state worsens | OPEN, 20 min, oracle only. Its gate needs an evaluation-window room clause of at least +0.05 as well as a market share below 85% -- KP14's baseline meets the market clause at 35% and has no gap |

Closed, with what closed them: BGN's regime path (five ten-seed points and a screen moved room by a
factor of thirteen and never moved the fair gap past +0.0025); GS21's exposure path (gx7's
pre-registered negative fired); K5, signed exposures (the economy does not exist -- the G solve
discounts growth options at rho_ty, which goes negative for a loading of -0.06, and a negative
discount removes the operator's dissipation); the window and penalty ladders (absorbed into the
protocol, which fixes both for every economy).

## Adding an experiment

1. Write `experiments/specs/var-<model>-<tag>-v1.json`: title, question, the parameter override,
   `env`, `estimation`, a written prediction with a falsification clause, and `expected_solves`
   computed WITHOUT solving. Say in economic terms what differs from the baseline. The `panel` and
   `estimation` blocks are the protocol's -- `tests/test_protocol_is_uniform.py` refuses any other
   sample, window, ridge grid or conditioning-column list. **A new economy is a parameter override
   and nothing else.** If what you want to change is the sample, the window or the grid, you are
   proposing a change to the protocol, which changes every row in this file and is not an experiment.
2. Solve with the model's producer (`rebuild_jstar_gam.py`, `build_vy_tables.py`, `gs_solve_reg.py` /
   `gs_solve_gam.py`); confirm the registry id matches the spec; publish with
   `variants/fetch_solves.py --publish`. A chained stage's id hashes its upstream tables' raw bytes,
   so build the chain on one platform or ship the upstream tables byte-identical (`docs/RUNS.md`).
3. Add a `SEED_SPEC` case to `variants/run_seeds_slurm.sh` (`tests/test_specs_match_shell.py` pins it
   to the spec). The case carries the economy's parameters and nothing else: no `KAPPAS`, no
   `RF_COLS`, no `SEED_T`.
4. Run ONE seed and read `sacct` before sizing an array. Then seeds 1-9. Ten seeds or the economy
   does not go in a ranked table.
5. `python variants/aggregate_seeds.py` and `python variants/fair_gap.py`; commit the result files
   and the tables; add the economy here with what differs from the baseline in economic terms first
   and what the result decided against the prediction. Report the winning penalty and feature count
   per seed: if the penalty grid's edge won, the DKKM number is censored and says less than it looks.
   `tests/test_results_md_matches_table.py` fails until the row is added.
6. Push, then pull the shared checkout on the cluster with `bash variants/cluster_pull.sh`, never a
   plain `git pull`: the cluster still holds its own untracked copies of the files just committed
   (`docs/RUNS.md`, "Where output goes").
