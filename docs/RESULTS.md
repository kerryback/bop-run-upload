# Experimental results

The record of every economy run through the oracle-and-estimator pipeline: what each was built to
test, what it produced, and what it decided. Last updated 2026-09-14, when the K4, X3 and B4 campaign
landed and every current economy was scored against E1's fair linear benchmark. The current-results
table is checked against `variants/results/economy_table.csv` and
`variants/results_e1/fair_gap_economy_table.csv` by `tests/test_results_md_matches_table.py`, so this
file cannot fall behind the numbers without the suite saying so. Where each job ran: `docs/RUNS.md`.
What to run next, and why: `docs/NEXTUP.md`.

## The answer so far

**A complexity gap exists in one family of economies, and only there.** In Kogan and Papanikolaou
(2014) with a priced, mean-reverting aggregate state that heterogeneous firm types load on (KP14
Path 1), the random-feature ridge estimator of Didisheim, Kelly, Kozak and Malamud (DKKM) beats the
best linear method by +0.10 to +0.19 of Sharpe, at t of 30 to 55, in every seed of every economy on
the path. The gap survives the fair benchmark of finding 8 unchanged, because on this path the linear
methods themselves beat the equal-weighted market by 0.28 to 0.33 and DKKM beats them by a further
0.10 to 0.19. In the seven Berk, Green and Naik (1999) and Gomes and Schmid (2021) economies the fair
gap is between -0.0025 and +0.0025.

| economy | what it is | window | gap | fair gap | t | DKKM | best linear | EW market | SR_max eval | room eval | gap / room |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kp_vy/vyxT860 | vyx on an 860-month panel, a 720-month window, penalty grid down to 1e-4 | 720 | +0.1875 (0.0393) | +0.1872 | 54.6 | 0.8452 | 0.6577 | 0.3577 | 1.2294 | +0.3864 | 0.49 |
| kp_vy/vyg25 | vyx with the price of the state's risk raised from 1.8 to 2.5 | 360 | +0.1450 (0.0193) | +0.1450 | 39.2 | 0.9450 | 0.8000 | 0.4694 | 1.3887 | +0.4108 | 0.35 |
| kp_vy/vyx | the parent economy | 360 | +0.1046 (0.0155) | +0.1046 | 30.2 | 0.7475 | 0.6428 | 0.3675 | 1.1778 | +0.3606 | 0.29 |

Mean (sd) over ten seeds, N=500 firms, 125 evaluation months scored on the true conditional moments.
The two window-360 rows also appear, with every other current economy, in "Current results" below.

**Why this route works.** Three things are true of it at once, and of no other economy in the
repository. First, firms differ in their exposure to a priced shock: three types load their cash
flows on the state y as exp(beta_f y), with beta 0.02 / 0.07 / 0.14, and y's innovations carry a
price of risk gamma_v. Second, the state bends that exposure map rather than shifting it: y is a
mean-reverting Ornstein-Uhlenbeck process, so the value of an exp(beta y) cash-flow stream depends on
where y is, and a compensation term keeps firm value from revealing the type monotonically at y = 0.
Third, every level in the economy is stationary, so raw characteristics do not trend with the state
and a rolling regression on raw levels cannot condition on it for free (the failure of KP14 Path 3 and
BGN Path 2). The result is a large population room, +0.36 to +0.41 over the evaluation months, that
only a nonlinear basis can reach, and an equal-weighted market that reaches only 29% to 34% of the
attainable Sharpe. Every other model either has no exposure heterogeneity for a state to bend (KP14 and
GS21 as published: room below +0.004) or a market that already carries 55% to 98% of the attainable
Sharpe (all BGN and GS economies).

**Why it widened, twice, in the 2026-09-13 campaign.** The linear methods sit at 99% to 100% of
their population ceiling in every KP14 economy; DKKM sits at 74% of its in vyx, and its winning ridge
penalty is the smallest on the grid in nine seeds of ten. So the gap is bounded by DKKM's estimation
shortfall, and two dials that raise DKKM's signal-to-noise each widened it. K4 raised the price of the
state's risk from 1.8 to 2.5: the cross-section of expected returns spread out, the gap rose 39% on
14% more room, and DKKM reached 78% of its ceiling. X3 kept the economy and doubled the estimation
window, with one more decade of penalty: the window alone took the gap to +0.138, the 1e-4 penalty
to +0.1875, DKKM reached 81% of its ceiling, and the smallest penalty won in every seed. Neither dial
is exhausted: in all three economies the bottom of the penalty grid still binds.

**What the gap is not.** It is not the market portfolio, which DKKM appends to its features
unpenalised: here the market has Sharpe 0.36 to 0.47 against an attainable 1.18 to 1.39, the linear
methods beat it by 0.28 to 0.33, and E1's benchmark, which gives them the market exactly as DKKM has
it, leaves the gap within 0.0003 of itself. It is also not a calibration. The oracle's mean expected
return is 18.2% a year with a cross-sectional sd of 7.7% in vyx, and 22.9% and 10.6% in vyg25; these
economies locate where the mechanism is, not where a referee would accept it. Every proposal that
turns a dial up reports the annualised premia beside the gap.

**Why nothing else worked.** Outside this path every measured gap was the equal-weighted market
against linear methods that were not given it on the same terms (finding 8). DKKM carries the market
unpenalised; `linrank` and `linlev` penalise it with everything else, Fama-French holds a different
market, Fama-MacBeth none. In 54 of the 70 BGN and GS seeds DKKM's most-shrunk portfolio IS the
market to within 0.001 of Sharpe, and its winning portfolio is barely off that point. E1 re-scored all
80 saved panels with the linear methods given the market unpenalised: every fair gap outside vyx came
in under the bound registered before the run, at most +0.0025 and negative in three economies. That
closed BGN's regime path (four persistence points moved room by a factor of thirteen and never moved
the fair gap; the stress-dominant screen B4 then missed both of its gates) and retired GS21's exposure
path (gx7's pre-registered negative fired). The baselines below say why those two models had nowhere
to go: neither has nonlinear room to begin with.

## Baselines: the anchor

The three models as published, before any path was built on them. Every path in this file is a
departure from one of these rows, and the design rule behind the winning path came from reading them:
BGN has native nonlinear room and the other two have none.

| model | baseline economy | SR_max | linear ceiling | nonlinear ceiling | room | FMR | FF | best linear | DKKM | gap | t | the economy the current code builds? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | one priced market shock at sigma_z 0.4, project betas from a translated exponential, Vasicek rate | 0.2926 | 0.2347 | 0.2712 | +0.0365 | 0.2198 | 0.2202 | 0.2334 | 0.2393 | +0.0059 | 14.0 | yes |
| KP14 | constant prices of risk gamma_x 0.69 and gamma_z -0.35, growth-option arrival regime, one firm type | 0.2396 | 0.2226 | 0.2263 | +0.0037 | 0.2052 | 0.1926 | 0.1969 | 0.2005 | +0.0036 | -8.5 | no: mean arrival rate 1.72, not 1 |
| GS21 | constant price of risk gamma_x 0.5 on productivity, one firm type, exact kernel | 0.4012 | 0.3996 | 0.4006 | +0.0010 | 0.3785 | 0.3716 | 0.3825 | 0.3988 | +0.0163 | 21.8 | no: rho_x, delta and kappa_e differ |

Single seed each, N=500, T=500, 360-month window, levels and the unpenalised market included: the
pre-refactor grid's protocol, which is the current protocol without the seeds and without E1's fair
methods. Recovered from `23f9380:variants/results/grid_summary.csv`; the narrative is `variants/REPORT.md`
§14. Room here is the all-month figure.

What the anchor says:

- **No unmodified model has a complexity gap.** The three baseline gaps are +0.004 to +0.016, and
  read through finding 8 the GS21 figure, +0.016 with zero room, has the shape of g28: DKKM on the
  market, the linear methods below it. None of the three was ever re-scored against the fair
  benchmark.
- **Only BGN starts with nonlinear room**, +0.036, from the one source of firm-level exposure
  heterogeneity any of the three models has natively, its project-beta distribution. KP14 and GS21
  start with +0.004 and +0.001: every path in those models had to build heterogeneity in, and the one
  that did so with a priced state and stationary levels is the winning route above.
- **Against the anchor, the winning route added +0.10 to +0.18 of gap to a KP14 baseline of
  +0.004.** BGN's best route moved its measured gap from +0.006 to +0.023 and its fair gap to +0.0006;
  GS21's from +0.016 to +0.028, fair -0.0022.

**What the anchor lacks.** Ten seeds, the current code, and the fair benchmark. The BGN row is the
economy the current code builds, so its single-seed levels are citable (WORKING.md §39 reproduced
the neighbouring g0235 row to the displayed digit). The KP14 row was simulated with the growth-option
arrival regime mislabelled (mean arrival rate 1.72 instead of 1, fixed 2026-09-04), and the GS21 row
with three Table I parameters wrong (fixed 2026-09-06); their levels describe different economies from
the ones the paths below are compared with, and only the direction of each finding survives. Each
baseline is one solve from a current-code run: `bgn_gam` at gmult = [1, 1] reproduces the baseline BGN
economy to machine precision (REPORT.md §13e), `kp_vy` with `type_bv = [0]` reduces to the unit KP14
economy (`variants/kp_vy/parameters_kp14.py`), and the GS21 regime solver at unit multipliers is the
constant-gamma baseline (one solve of 3.5 to 6 h). `docs/NEXTUP.md` carries this as the anchor run.

## How to read this

**The protocol behind every current number.** Each estimator is fit on a 360-month rolling window
of simulated data (720 for X3), and the portfolio it produces is scored against the economy's TRUE
conditional moments, month by month, over the 125 evaluation months of a 500-month panel (860 for X3)
of 500 firms. An estimator is never judged on its own realized returns, only on what its weights were
worth. Ten seeds per economy; every figure is mean (sd across seeds) unless the entry says otherwise.

### The columns

- **economy** -- `model/tag`: the model directory under `variants/` and the run tag naming its
  result files.
- **spec** -- the file in `experiments/specs/` defining the economy: parameters, estimator settings,
  and the `expected_solves` a run must consume or abort.
- **n** -- seeds.
- **room all / room eval** -- the population headroom for nonlinearity: the best Sharpe a nonlinear
  feature basis reaches with ONE fixed coefficient vector, minus the best a linear-in-ranks basis
  reaches the same way, both from the true moments. Over all 485 panel months, and over the 125
  evaluation months. Only the second is commensurable with `gap`; the two differ by up to 23%.
- **room eval % of lin** -- `room eval` over the linear Sharpe attained, in percent.
- **gap** -- the Sharpe of the winning random-feature estimator (DKKM: `rff`, `rff_ens`, `rff_lev`,
  `rff_lev_ens`, best feature count and best penalty) minus the best linear method (`linrank`,
  `linlev`, Fama-MacBeth `fm`, Fama-French `ff`). **Outside KP14 this measures the linear methods
  against the equal-weighted market DKKM holds unpenalised; the fair gap is the complexity gap.**
- **gap % of lin** -- `gap` over the linear Sharpe attained. Reorders the economies (finding 5) and
  misleads once the market is accounted for: g0235r's 192% is a linear Sharpe of 0.011 in months where
  the market earns 0.047.
- **t** -- the paired t-statistic of DKKM against Fama-MacBeth across evaluation months, averaged
  over seeds.
- **DKKM**, **best linear** -- the two Sharpes whose difference is `gap`.
- **SR_max eval** -- the oracle's maximum attainable conditional Sharpe over the evaluation months, a
  bound on every estimator by Cauchy-Schwarz. It holds on all eighty runs.
- **fair gap (E1)** -- DKKM minus the best of seven linear methods, the original four plus
  `linrank_m` and `linlev_m` (the market a separate unpenalised column, penalties two decades past the
  DKKM grid) and `mkt_est` (the market alone, its weight estimated as DKKM's is). From
  `variants/results_e1/fair_gap_economy_table.csv`, produced by `variants/fair_gap.py`.

### Two cautions

1. **Room is not a ceiling on the gap and not a screen for it.** The gap exceeds the room in every
   BGN and GS economy, because there it is the market shortfall; in KP14 it is 29% to 49% of the room.
   Ranking candidates by room does not rank them by gap.
2. **Every ratio here is a RATIO OF MEANS, not a mean of per-seed ratios.** g0235's gap reads 26.5%
   of the linear Sharpe one way and 31.2% (sd 17.5) the other, because its denominator has sd 0.036
   on a mean of 0.086. Both live in `economy_table.csv` (`gap_over_lin`, `gap_pct_lin_mean`).

### Status labels

- **CURRENT** -- ten seeds at N=500, T=500, window 360, under a spec that pins the solve ids the run
  must consume; every result file carries the spec id, the solve ids and a provenance tag naming the
  commit. X3, at T=860 and window 720, meets the standard at its own sample size and stays out of the
  ranked table, whose rows share one.
- **SCREEN** -- one seed, run under a spec whose other seeds start only if it clears a gate written
  into the spec first. Reported under its own `-- SCREEN` heading and never ranked (finding 4).
- **LEGACY, single seed** -- an economy of the pre-refactor grid, run once by code that is gone or
  since corrected. Quoted for the path it documents, never for a ranking: cross-seed sd of a gap runs
  15% to 63% of its mean. The grid's tables were deleted 2026-09-10 (WORKING.md §50) and are
  recoverable at `23f9380`; the narrative is `variants/REPORT.md`.
- **SUPERSEDED** -- an economy the current code no longer builds, kept as the record behind a
  published number.

Whether a model's legacy rows describe the economy the code still builds: yes for BGN, no for KP14
(arrival rate) and no for GS21 (Table I), as the anchor section says.

### Provenance of every current number

`experiments/specs/<spec>.json` names the parameters and `expected_solves`;
`experiments/registry/<solve_id>.json` records what each solve was built from;
`variants/results/<model>_oracle_<tag>_s<seed>.json` and
`<model>_estimators_<tag>_s<seed>_w<window>_summary.csv` carry the per-seed numbers, with `.prov.json`
sidecars naming the commit. `variants/aggregate_seeds.py` produces the tables.

### Ordering

Model sections run KP14, BGN, GS21: by best fair gap, the quantity the project is after. Within a
model, paths run by their best fair gap, then by best legacy gap for paths with no current economy.
Each model section ends with the status of the parameterizations proposed for it; the route log near
the end collects them.

## Current results

All nine CURRENT economies at N=500, T=500, window 360, ranked by fair gap. Mean (sd) over ten
seeds.

| economy | spec | n | room all | room eval | room eval % of lin | gap | gap % of lin | t | DKKM | best linear | SR_max eval | fair gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kp_vy/vyg25 | var-kp_vy-vyg25-v1 | 10 | +0.3993 (0.0425) | +0.4108 (0.0446) | 51.4% | +0.1450 (0.0193) | 18.1% | 39.2 | 0.9450 | 0.8000 | 1.3887 | +0.1450 (0.0193) |
| kp_vy/vyx | var-kp_vy-vyx-v2 | 10 | +0.3491 (0.0365) | +0.3606 (0.0394) | 56.1% | +0.1046 (0.0155) | 16.3% | 30.2 | 0.7475 | 0.6428 | 1.1778 | +0.1046 (0.0154) |
| bgn_gam/g0235f | var-bgn_gam-g0235f-v1 | 10 | +0.0342 (0.0091) | +0.0350 (0.0188) | 31.4% | +0.0211 (0.0154) | 18.9% | 16.6 | 0.1325 | 0.1115 | 0.2043 | +0.0025 (0.0029) |
| bgn_gam/g0235s | var-bgn_gam-g0235s-v1 | 10 | +0.0194 (0.0183) | +0.0119 (0.0267) | 27.2% | +0.0180 (0.0127) | 41.3% | 43.8 | 0.0616 | 0.0436 | 0.0946 | +0.0012 (0.0034) |
| bgn_gam/g0235 | var-bgn_gam-g0235-v2 | 10 | +0.0188 (0.0065) | +0.0176 (0.0119) | 20.6% | +0.0227 (0.0084) | 26.5% | 33.4 | 0.1081 | 0.0855 | 0.1461 | +0.0006 (0.0009) |
| gs_bx/gx7 | var-gs_bx-gx7-v1 | 10 | +0.0019 (0.0024) | +0.0053 (0.0044) | 1.3% | +0.0229 (0.0102) | 5.5% | 15.0 | 0.4393 | 0.4163 | 0.4473 | +0.0001 (0.0006) |
| gs_bx/bx7 | var-gs_bx-bx7-v3 | 10 | +0.0086 (0.0035) | +0.0066 (0.0041) | 2.3% | +0.0078 (0.0049) | 2.7% | 16.0 | 0.2964 | 0.2887 | 0.3121 | -0.0009 (0.0028) |
| gs_bx/g28 | var-gs_bx-g28-v2 | 10 | +0.0001 (0.0003) | +0.0004 (0.0003) | 0.2% | +0.0283 (0.0136) | 10.5% | 24.6 | 0.2975 | 0.2692 | 0.3087 | -0.0022 (0.0023) |
| bgn_gam/g0235r | var-bgn_gam-g0235r-v1 | 10 | +0.0043 (0.0038) | +0.0026 (0.0020) | 24.3% | +0.0209 (0.0166) | 192.1% | 80.0 | 0.0318 | 0.0109 | 0.0559 | -0.0025 (0.0069) |

**Read the last column, not the gap column, outside KP14.** In the seven BGN and GS rows the winning
DKKM portfolio is, to within half a hundredth of Sharpe, the equal-weighted market (finding 8); the
fair gap is what remains once the linear methods are given that market as DKKM has it, and it is at
most +0.0025 and negative in three rows, each inside the bound registered before E1 ran. The two KP14
rows scored the fair methods in their own runs; their linear methods beat the market by 0.28 and
0.33, so their fair gap is their gap. By measured gap the order would be vyg25, vyx, g28, gx7, g0235,
g0235f, g0235r, g0235s, bx7; by gap as a share of the linear Sharpe, g0235r first at 192%.

Not in the table: **X3**, vyx's economy on an 860-month panel with a 720-month window (gap +0.1875,
fair +0.1872; KP14 Path 1), and **B4's one-seed screen** g0235d, which missed its gates (fair gap
+0.0004; BGN Path 1).

Regenerate with `python variants/aggregate_seeds.py --flagship` (also writes the per-seed
`variants/results/seed_table.csv`); the last column with `python variants/fair_gap.py`.

---

## KP14 -- Kogan and Papanikolaou (2014)

### Baseline

A firm is a collection of live projects plus the growth opportunities it expects. Projects arrive to
firm f at a rate lambda_f that switches between a high regime (lambda_H = 2.35) and a low one
(lambda_L set so the mean arrival rate is exactly 1; entry intensities mu_H = 0.075 into the high
state and mu_L = 0.16 into the low). Output depends on firm profitability eps_f (theta_eps 0.35,
sigma_eps 0.2), project-level u (theta_u 0.5, sigma_u 1.5), aggregate productivity z (mu_z 0.005,
sigma_z 0.035) and the investment-specific technology state x (mu_x 0.01, sigma_x 0.13); alpha 0.85,
depreciation 0.1. The pricing kernel is exogenous with CONSTANT prices of risk gamma_x = 0.69 and
gamma_z = -0.35; r = 0.05 (the paper's 0.025, a deliberate departure). Table II: 17 of 18 parameters
match, r the exception.

Expected returns are affine in a single firm variable, the share of value in growth options, so the
cross-section is one-directional and no state-dependent price of risk can create curvature by scaling
alone. That is the finding behind every inert path below.

**Legacy rows are a different economy.** Until 2026-09-04 the regime probability was read with the
wrong label and the mean arrival rate simulated was 1.72 (`docs/kp14_regime_labels.md`). Legacy KP14
levels, the baseline row included, are not citable for the current code; the direction of each
path's finding survives.

### Path 1 -- a priced, mean-reverting aggregate state with heterogeneous firm exposure: vy, vyx, then a higher price and a longer sample

The one route that produced a complexity gap. Its ladder, in the order it was climbed:

| economy | gamma_v | window | grid floor | room eval | gap | fair gap | DKKM / its ceiling | status |
|---|---|---|---|---|---|---|---|---|
| kp_vy/vy | 1.2 | 360 | 0.001 | +0.278 (all-month) | +0.0289 | -- | about 10% of the room | LEGACY, single seed, pre-fix economy |
| kp_vy/vyx | 1.8 | 360 | 0.001 | +0.3606 | +0.1046 | +0.1046 | 74% | CURRENT |
| kp_vy/vyg25 | 2.5 | 360 | 0.001 | +0.4108 | +0.1450 | +0.1450 | 78% | CURRENT |
| kp_vy/vyxT860 | 1.8 | 720 | 0.0001 | +0.3864 | +0.1875 | +0.1872 | 81% | CURRENT at T=860 |

The linear methods sit at 99% to 100% of their own ceiling on every row; DKKM's winning penalty is
the smallest offered in 9, 10 and 10 seeds of ten.

#### kp_vy/vyx -- CURRENT, gap +0.1046 (sd 0.0155), t 30.2

**What differs from the baseline.** A new aggregate state y: stationary Ornstein-Uhlenbeck,
mean-reversion kappa_y = 0.35, unit stationary sd, innovations priced at gamma_v = 1.8. Three firm
types in shares 0.34 / 0.33 / 0.33 load project cash flows on it as exp(beta_f y), beta = 0.02 / 0.07
/ 0.14. Because y is priced each type earns a premium in its beta; because y mean-reverts, the VALUE
of an exp(beta y) stream depends on where y is, so the map from type to observable characteristics is
bent by the state rather than shifted. A compensation term (bv_comp = 1.2) rescales each type's
cash-flow level so that at y = 0 value does not reveal the type monotonically. Every level stays
stationary, which is what keeps rolling Fama-MacBeth from conditioning on the state through trending
levels (Path 3's failure).

**Why it was tried.** REPORT.md §18-§19's design rule: a harvestable gap needs heterogeneous
firm-level exposures, a state that bends them nonlinearly, and the bending confined to rank and
interaction space. vy was the first KP14 economy to satisfy all three; vyx pushed its two free knobs
(gamma_v 1.2 to 1.8, top beta 0.12 to 0.14) toward premium-side extremity.

**Result.** DKKM 0.7475 against best linear 0.6428 (linrank); SR_max 1.1778 over the evaluation
months, so DKKM reaches 63% of the attainable and 74% of its own nonlinear ceiling. The
mis-normalised arrival rate did not change this economy's numbers: the v1 spec published room +0.3496
and gap +0.1009 from one seed, the corrected economy at ten seeds gives +0.3491 and +0.1046
(WORKING.md §37).

**Provenance.** Spec `var-kp_vy-vyx-v2`; solves `f7be27e39d2b530f` (the G function) and
`84e195172f091cd2` (the type-by-state integrals); `SEED_SPEC=vyx`; peak 30.6 GiB.

#### kp_vy/vyg25 -- CURRENT, gap +0.1450 (sd 0.0193), t 39.2

**What differs from vyx.** One parameter: gamma_v from 1.8 to 2.5, with its own G and integral
tables under prefix `vyg25`. Proposal K4.

**Why it was tried.** vyx is the one economy where DKKM leaves the market, and the Sharpe the market
does not span there is carried by the state's shock, whose Sharpe scales with its price. vy to vyx had
raised gamma_v from 1.2 to 1.8 and the gap rose with the room.

**Result: the largest window-360 gap in the repository, just below its predicted range and not
falsified.** DKKM 0.9450, best linear 0.8000, market 0.4694. Against vyx the gap rose +0.0404 (se
0.0078), or 39%, on room that rose 14% and SR_max that rose 18%; DKKM captured more of what the
higher price created, 78% of its nonlinear ceiling against 74%, while the linear methods stayed at
100% of theirs.

| | vyx | predicted | vyg25 | verdict |
|---|---|---|---|---|
| room, evaluation window | +0.3606 | +0.45 to +0.50 | +0.4108 | wrong, low |
| SR_max, evaluation window | 1.1778 | up about 40% | 1.3887, up 18% | wrong, low |
| non-market Sharpe | 1.1178 | up about 40% | 1.3064, up 17% | wrong, low |
| DKKM's winning penalty | smallest on the grid in 9 of 10 | still the smallest | smallest in 10 of 10 | right |
| gap | +0.1046 | +0.15 to +0.20; falsified below +0.12 | +0.1450 | below the range by 0.005; not falsified |

The last decade of penalty, 0.01 to 0.001, still added 0.046 of Sharpe, so this economy is as
shrinkage-bound as vyx. Distance from a calibration: mean expected return 22.9% a year,
cross-sectional sd 10.6% (vyx 18.2%, 7.7%).

**Provenance.** Spec `var-kp_vy-vyg25-v1` (`precommitted: true`); solves `f41d052f1f960c4c` (G,
built on the Mac and shipped to Sol byte-identical) and `8d1308e8f21723f8` (integrals, Sol
`63188972`), both the precommitted ids, after a first Sol attempt failed its integral id because the
chained id hashes the upstream G tables' raw bytes (`docs/RUNS.md`); ten seeds on Sol array
`63188973` at `fef5802`, peak 29.2 GiB, longest 7.6 h.

#### kp_vy/vyxT860 -- CURRENT at T=860, window 720: gap +0.1875 (sd 0.0393), t 54.6

**What differs from vyx.** Nothing in the economy: the same parameters and solves, simulated for 860
months and estimated on a 720-month window, so the evaluation months again number 125. The ridge grid
gains 1e-4 below vyx's smallest penalty. Its own tag, because result files are named by model, tag and
seed only. Proposal X3.

**Why it was tried.** If DKKM's shortfall from its ceiling in vyx is data, a longer window closes part
of it and leaves the linear methods where they are.

**Result: the shortfall is data, and more of it than predicted.** DKKM 0.8452 against 0.7475; linear
0.6577 against 0.6428; room unchanged within seed noise.

| | vyx | predicted | vyxT860 | verdict |
|---|---|---|---|---|
| DKKM | 0.7475 | +0.03 to +0.08; falsified below +0.02 | 0.8452, +0.0977 (se 0.0244) | wrong, high; not falsified |
| best linear | 0.6428 | within +0.01 | 0.6577, +0.0149 (se 0.0159) | outside the band, not distinguishable from it |
| gap | +0.1046 | +0.13 to +0.17 | +0.1875 | wrong, high |
| room, evaluation window | +0.3606 | unchanged within seed noise | +0.3864, +0.0258 (se 0.0206) | right |

**The window and the grid, separated.** On vyx's grid (penalties 0.001 and up) DKKM reaches 0.7958
and the gap +0.1382: the window alone lands inside both predicted ranges. The 1e-4 penalty adds a
further 0.0494 of DKKM Sharpe and wins in all ten seeds. The spec had argued the extension alone was
worth little, from vyx's flattening gains per decade of penalty at window 360 (0.154, 0.100, 0.035);
at window 720 the decades give 0.157, 0.116, 0.073 and then 0.049. The longer window is what makes a
smaller penalty pay, and the bottom of the grid still binds. The winning feature count moved down to
360 in eight seeds (vyx: 3600 in six). The market's role is unchanged, 29% of SR_max with the linear
methods 0.30 above it. The ceilings are measured over different calendar months of a longer
simulation (SR_max 1.2294 against 1.1778), so level comparisons carry that difference.

**Provenance.** Spec `var-kp_vy-vyxT860-v1`, reusing vyx's solves (`reused_solves`); ten seeds on
Sol array `63188596` at `fef5802`, peak 56.9 GiB, longest 6.0 h.

#### kp_vy/vy -- LEGACY, single seed, gap +0.0289, t 5.2

gamma_v = 1.2 and beta = 0.02 / 0.06 / 0.12. Room +0.278, the first significant DKKM win in KP14
after five state-dependent-price designs, but capture near 10% and limited by T: the best realized fit
was at 360 features, not 3600. REPORT.md §19a. Pre-fix economy.

### Paths 2 to 4 -- the inert routes that led to Path 1

All LEGACY single-seed rows, all pre-fix, none with room above +0.019.

- **Path 2, state-dependent prices of risk on the baseline cross-section, four ways** (room below
  +0.004 in each). A two-state regime multiplying both prices by 0.5 / 2.0 (`kp_gam`, gap +0.0067);
  a continuous state y with a common logistic multiplier 0.5 to 2.5 (`kp_gamy`, +0.0074); the
  multiplier on gamma_x only, so the premium direction rotates 3.5x across the state (+0.0022); an
  uneven regime on gamma_x alone (+0.0185, all of it regime timing through the exported regime
  feature). Why inert: with premia affine in the firm variable within each state, the conditional
  premium is spanned by a linear basis with a state interaction, and a common multiplier gives mu_t =
  g(y_t) mu-bar, a constant cross-sectional direction. "KP's cross-section is irreducibly
  one-directional" (REPORT.md §13c-d, §17b, §17d, §17f).
- **Path 3, exposure types crossed with the regime** (`kp_bx`, room +0.0187, gap +0.0016). Cash flows
  load on the technology state as x^{beta_f}, beta in {1.0, 1.8, 3.0}, with a calm-state value
  compensation of 1.2. The first genuine room in KP14, and rolling Fama-MacBeth on raw levels
  harvested it (FMR 0.181 against DKKM 0.177) because x^{beta_f} makes raw characteristics co-move
  with the state. This failure produced the third design clause, stationary levels, and Path 1.
  REPORT.md §18.
- **Path 4, crash risk** (`kpd_al`, room +0.0058, gap -0.0064). Type-specific disaster probabilities,
  a priced kill intensity, a compensating output multiplier; the rank-linear method came out ahead.
  "Crash risk per se does not create a DKKM gap" (§12b).

### Proposals -- KP14

- **K4, gamma_v 2.5 -- DONE** (vyg25). Gap +0.1450; room, SR_max and the non-market Sharpe rose 14%
  to 18% against a predicted 25% to 40%; the winning penalty stayed at the grid bottom in every seed.
- **X3, vyx at T=860, window 720, grid to 1e-4 -- DONE** (vyxT860). DKKM +0.0977 and gap +0.1875,
  both above their predicted ranges; the linear methods +0.0149; room unchanged. The window alone gives
  +0.1382 and the 1e-4 penalty the rest, winning in every seed.
- **K5, signed exposures -- WITHDRAWN: the economy does not exist.** Type loadings (-0.06, +0.04,
  +0.14) were to leave the market carrying little of the state's risk. The G solve discounts growth
  options at rho_ty, which is -0.149 at the top of the state grid for a loading of -0.06 and -0.076
  even at y = 0; a negative discount removes the G operator's dissipation. The most negative loading
  with rho_ty positive everywhere is about -0.005, no exposure at all.
- **K1, a continuum of exposures -- OPEN.** Fifteen types on a grid over [0, 0.14], shares right-skewed
  so most firms sit low: less of the state's Sharpe spanned by the market, and a smooth exposure map
  for random features. Prediction: room and gap up modestly, less confidently than K4, since the
  three-type structure was never shown to be what the linear methods exploit. About 7.5 h of
  integrals, then 30 h of seeds.
- **K3, persistence of the priced state -- OPEN.** kappa_y 0.35 to 0.15 and to 0.70. Slower reversion
  bends values more but gives a window fewer independent cycles; with X3's finding that DKKM is
  data-limited, the gap should FALL at 0.15 and hold or rise at 0.70. Two solves of about 90 min, then
  two 30 h arrays.
- **K2, a rare extreme type -- OPEN, low priority.** Shares (0.45, 0.45, 0.10), top loading 0.14 to
  0.20; the discount check applies.

Ranked, with the case for each, in `docs/NEXTUP.md`.

---

## BGN -- Berk, Green and Naik (1999)

### Baseline

A firm is a collection of live projects, each a real option exercised when its present value crossed
a threshold. Projects carry a market-shock loading beta_s from a translated exponential (scale 0.137),
an idiosyncratic cash-flow volatility, and a cash-flow level around C-bar = -3.7; new projects arrive
and are exercised optimally against a threshold J*(r) that depends on the short rate, which is Vasicek
(monthly persistence 0.95, mean 0.006236, innovation sd 0.002). The pricing kernel is lognormal with
ONE priced shock, the market shock, at price sigma_z = 0.4, correlated -0.175 with the rate
innovation. Expected returns are exactly affine in book-to-price and 1/price with rate-dependent
coefficients. Table I: 11 of 11 parameters match.

The project-beta distribution is the only native source of firm-level exposure heterogeneity in any
of the three models, which is why the regime path created room here and not in KP14.

**Legacy rows are the same economy.** The calibration survived every audit unchanged, and the current
code reproduced the legacy g0235 row to the displayed digit on a different machine (WORKING.md §39).

### Path 1 -- a two-state regime on the price of the market shock: CLOSED 2026-09-14

A two-state Markov regime (calm, stress) multiplies the price of the market shock, sigma_z x gmult[s];
switches are unpriced; the exercise threshold J* is re-solved per economy; the regime and the rate are
exported as conditioning features. A project's value decomposes onto two regime bases, so the two
regimes apply two DIFFERENT monotone transforms of the same project beta: curvature in the map from
beta to the premium within a regime, which a common scaling of the price of risk cannot produce. The
path ran gmult = [0.5, 2.0] and [0.3, 3.0] (legacy), then [0.2, 3.5] at the closed-form bound
1/(2 x 0.137) = 3.65, then a persistence ladder on that point, then a stress-dominant screen.

What it decided: the room is real and the gap was not. Across the four ten-seed points the
evaluation-window room ranged from +0.0026 to +0.0350, a factor of thirteen, and the measured gap
stayed between +0.018 and +0.023 throughout, because DKKM held the equal-weighted market and the
linear methods fell below it (finding 8). The fair gap is +0.0025 or less at every point. The one
seed with both a lot of room and a lot of non-market Sharpe (g0235s seed 2) put DKKM 0.116 above the
market and only 0.011 above the linear methods, which followed most of it; the screen built on that
seed (B4) put DKKM 0.081 above the market and plain `linrank` followed all of it. The spec's rule ends
the path on the miss.

| economy | switch probabilities per month, calm to stress / stress to calm | stress share | room eval | gap | fair gap | DKKM - market | market - best linear |
|---|---|---|---|---|---|---|---|
| g0235f | 1/12, 2/12 (fast) | 1/3 | +0.0350 | +0.0211 | +0.0025 | +0.0116 | +0.0094 |
| g0235 | 0.25/12, 0.50/12 | 1/3 | +0.0176 | +0.0227 | +0.0006 | +0.0020 | +0.0206 |
| g0235s | 0.05/12, 0.10/12 (slow) | 1/3 | +0.0119 | +0.0180 | +0.0012 | +0.0100 | +0.0080 |
| g0235r | 0.05/12, 0.45/12 (rare) | 1/10 | +0.0026 | +0.0209 | -0.0025 | -0.0154 | +0.0363 |
| g0235d, SCREEN | 0.50/12, 0.25/12 (stress-dominant) | 2/3 | +0.0186 | +0.0004 | +0.0004 | +0.0812 | -0.0808 |

#### bgn_gam/g0235 -- CURRENT, gap +0.0227 (sd 0.0084), fair gap +0.0006, t 33.4

gmult = [0.2, 3.5], switches 0.25/12 and 0.50/12. DKKM 0.1081 against linrank 0.0855; Fama-MacBeth
collapses to 0.067 and the level-linear method to 0.034 because regime shifts move raw levels. The
published single-seed row (room +0.0233, gap +0.0297) is reproduced exactly by seed 0 and sits 0.8 sd
above the ten-seed mean. Through finding 8: DKKM is the market's 0.1061 plus 0.0020, and the gap is
the linear methods' 0.0206 shortfall below that market. Spec `var-bgn_gam-g0235-v2`; solve
`be222462dd017b2c` (`Jstar_g0235.csv`); `SEED_SPEC=g0235`; 15.7 to 38.9 GiB on Sol.

#### bgn_gam/g0235f -- CURRENT, gap +0.0211 (sd 0.0154), fair gap +0.0025, t 16.6

Both switch probabilities times four (calm spells 12 months, stress 6; about forty switches per
window); the stationary stress share unchanged at one third. Proposal B1's fast point. The
pre-registered prediction failed on both halves: room was to be unchanged and the proportional gap to
rise; room nearly doubled (+0.0176 to +0.0350) and the proportional gap fell (26.5% to 18.9%). Room
moved because switching speed enters the SOLVE: how long a firm expects cheap risk changes which
projects it takes, and the J* table (value range 37 to 345 against g0235's 57 to 483) said so before a
seed ran. Lesson: when a parameter enters the solve, check whether the solved table moved before
assuming the cross-section did not. Spec `var-bgn_gam-g0235f-v1`; solve `8662d7c1079f4b41`; Phoenix
array `21564268`, 3.0 to 3.3 h per seed.

#### bgn_gam/g0235s -- CURRENT, gap +0.0180 (sd 0.0127), fair gap +0.0012, t 43.8

Both switch probabilities times 0.2 (calm spells 240 months, stress 120; about two switches per
window). B1's slow point. Predicted: room unchanged, gap falling toward or below room; the gap stayed
above the evaluation-window room in nine of ten seeds. The evaluation window is one draw of a slow
regime path: six of ten seeds have no stress month in it, one (seed 2) has 74%, and every level
statistic follows that occupancy (stress share correlates 0.98 with SR_max, 0.99 with DKKM, 0.97 with
room) while the gap does not (-0.11). Spec `var-bgn_gam-g0235s-v1`; solve `05ad848ab3779695`; seeds
0, 2, 5 to 9 on Phoenix `21564269` at `c3d60c8`, seeds 1, 3, 4 on Sol `63033314` at `1fb6f44` after
the 64 GiB cap killed them on Phoenix (no Python differs between the commits).

#### bgn_gam/g0235r -- CURRENT, gap +0.0209 (sd 0.0166), fair gap -0.0025, t 80.0

Calm-to-stress 0.05/12 and stress-to-calm 0.45/12: stress in 10% of months, in spells of about 27
months between calm spells of 240. A different economy from g0235, not the same one at another speed.
Predicted: room falls (it did, to +0.0026) and the gap falls further (it did not: +0.0209). The 192%
proportional gap is a denominator: the best linear Sharpe averages 0.0109 and is below zero in four
seeds, in months where the market earns 0.047. DKKM itself is 0.015 BELOW the market: in seeds 0, 5,
6 and 7 its market weight turns negative in part of the evaluation months, against a true premium
that is positive in every one of them. Spec `var-bgn_gam-g0235r-v1`; solve `0c624174cf4b26fa`; six
seeds on Phoenix highmem `21564312`, four on Sol `63033315`. Seed 3's oracle ran at `c3d60c8` and its
estimators at `1fb6f44`, because the shared checkout was pulled while that 24.5-hour task sat between
stages: harmless here, a hazard in general.

#### bgn_gam/g0235d -- SCREEN, one seed: missed both gates, fair gap +0.0004

The switch probabilities swapped (0.50/12 calm-to-stress, 0.25/12 stress-to-calm), so stress occupies
two thirds of months in 48-month spells. Proposal B4, built on the one BGN seed with a lot of
non-market Sharpe (g0235s seed 2, stressed in 84% of its months). Seeds 1 to 9 were gated on
evaluation-window room of at least +0.05 AND the oracle's non-market Sharpe (`sr_orth_eval`) of at
least 0.30.

| | predicted | seed 0 |
|---|---|---|
| room, evaluation window | above +0.04 (gate +0.05) | +0.0186 |
| non-market Sharpe | 0.2 to 0.3 (gate 0.30) | 0.1541 |
| DKKM minus the equal-weighted market | +0.03 to +0.06 | +0.0812 |
| fair gap | +0.005 to +0.015 | +0.0004 |

DKKM (0.1484) left the market by more than any BGN ten-seed mean does, and plain `linrank`, whose
market column is penalised, reached 0.1480. The spec's own reason for a small fair gap, a
within-regime premium close to affine in book-to-price and 1/price, held more strongly than it
predicted. Spec `var-bgn_gam-g0235d-v1` (`precommitted: true`); solve `e136e8b440bce761`, reproduced
by Phoenix `21571505`; seed 0 on Phoenix `21571506` at `fef5802`, 15.6 GiB, 2.5 h.

#### g0520 and g0330 -- LEGACY, single seed

gmult = [0.5, 2.0] (room +0.0214, gap +0.0219, t 21.5) and [0.3, 3.0] (room +0.0254, gap +0.0240,
t 31.7). Room saturated near +0.021 to +0.025 while the gap rose with the spread, because the
widening damaged the raw-level methods (FMR 0.105, FF 0.132 at [0.3, 3.0]) faster than it added
room. REPORT.md §13f-h, §16.

#### Two things the ladder established besides finding 8

- **Memory follows calm spells.** Longer calm spells price risk cheaply for longer, so firms accept
  more projects and the panel arrays grow. The J* value scale orders exactly as memory does (g0235f 37
  to 345, g0235 57 to 483, g0235s 148 to 1262, g0235r 390 to 2041; peak memory 22, 39, 74 and 77 GiB),
  and within an economy the longest calm spell in a seed's panel ranks with its peak memory at
  Spearman +0.84 (g0235s) and +0.80 (g0235r). The panels that never enter stress are the heaviest and
  slowest (g0235r seed 3: 77.0 GiB, 24.5 h). All ten g0235r seeds and three g0235s seeds were first
  killed at a 64 GiB cap.
- **Partial means were biased toward the early seeds.** The late seeds were the calm-heavy ones; an
  early-seed snapshot overstated g0235r's gap by 22% and g0235s's room by 25%. Holding the partial
  means out of the table was right.

### Paths 2 to 4 -- LEGACY, single seed, same economy family

- **Path 2, a continuous price of risk in the rate, gamma(r)** (`bgn_gamr`, room +0.0934, gap +0.0060,
  t -2.7). A logistic multiplier 0.5 to 3.0 on the market-shock price as a function of the short rate.
  The largest room of the whole project, and rolling Fama-MacBeth harvests it (FMR 0.470 against DKKM
  0.464), because raw levels co-move with r and a rolling raw-level regression conditions on the rate
  for free. REPORT.md §17c.
- **Path 3, crash risk with firm types** (`bgn_dis`, room +0.0658, gap +0.0053, t 8.9). A disaster
  indicator in the kernel (p = 3%, kappa = 2.5), three firm types whose projects die in a disaster
  with probability 0 / 0.15 / 0.30, a value compensation 1.3 so exposed firms look like growth firms.
  Three quarters of the room captured. §9, §14.
- **Path 4, the oracle-only exploration** (parameter sweeps, tail shapes, size-dependent volatility,
  firm types, a U-shaped rate-shock correlation). Nothing above +0.008 of room except the two paths
  above. §3-§8.

### Proposals -- BGN: none open

- **B1, regime persistence -- DONE.** Four points; room moved by a factor of thirteen and the fair gap
  never left +0.0025.
- **B4, a stress-dominant regime -- DONE, the screen missed both gates**, and the spec's rule closes the
  regime path.
- **B2, along the frontier (a thinner beta tail admitting gmult up to 5) and B3, a live interest rate
  under the regime -- CLOSED by B4.** Both sat on the path B4 closed; B3's own prediction was already
  that its screen would miss, because the rate shock is priced at almost nothing (beta_zr = -0.00014).

A BGN economy earns estimator time only with room of at least +0.05 AND a non-market Sharpe of at
least 0.30 on a one-seed oracle screen (two points, vyx and g0235s seed 2: a heuristic). No proposed
BGN economy is predicted to reach it.

---

## GS21 -- Gomes and Schmid (2021)

### Baseline

A firm holds capital and one-period debt, produces from an aggregate productivity state x (quarterly
persistence 0.95, innovation sd 0.012) and an idiosyncratic state z (0.90, 0.16), pays maintenance
delta = 0.02 per quarter, taxes at 0.2, and chooses investment, borrowing and default each period;
equity issuance costs 0.025, debt issuance 0.004, lenders recover 0.4, default is smoothed by a shock of
sd 5. The pricing kernel is exogenous, exp(-r - gamma^2/2 - gamma eps_x), with a CONSTANT price of
risk gamma_x = 0.5 and r = 0.10 a year: the paper's general-equilibrium kernel with its countercyclical
price of risk is replaced by this stand-in, which Path 1 puts back. Value-function iteration on a
161-point x grid, about 3.5 h a solve.

With a constant price of risk every feature basis reaches the same population ceiling: GS21 has no
learnable nonlinear cross-section, and its gaps come from somewhere else.

**Legacy rows are a different economy.** Until 2026-09-06 the solver ran with rho_x = 0.96^(1/3),
delta per month, and kappa_e = 0; levels are not citable, mechanism findings survive.

### Path 1 -- a countercyclical price of risk gamma(x), alone and crossed with exposure types

#### gs_bx/g28 -- CURRENT, gap +0.0283 (sd 0.0136), fair gap -0.0022, t 24.6

gamma(x) = clip(0.5 - 0.28 x / sd(x), 0.05, 1.0): the baseline 0.5 at mean productivity, toward 0.05
in booms and 1.0 in busts. A RECONSTRUCTION from the formula in REPORT.md §13g, verified to reproduce
the regime solver byte-for-byte at slope zero. The spec's question: can a gap open with ZERO nonlinear
room? The measured gap said yes (room +0.0004, gap +0.0283, once the second-largest in this file) and
finding 8 said what it was: the market reaches 95% of SR_max here, DKKM is 0.0032 above it and the
linear methods 0.0251 below. Under the fair benchmark a linear method given the market beats DKKM in
nine seeds of ten (t -3.0). Seed 0 alone had read +0.0119. Spec `var-gs_bx-g28-v2`; solve
`8b584c38614695ac` (`gs_solve_gam.py`, 5.9 h on Phoenix); `SEED_SPEC=g28`.

#### gs_bx/gx7 -- CURRENT, gap +0.0229 (sd 0.0102), fair gap +0.0001, t 15.0

Five firm types in equal shares loading on the state as exp(beta_f x + z), beta = 1 / 2.5 / 4 / 5.5 /
7, under g28's gamma(x): the design rule that produced vyx, with GS21's own state. Proposal G1. **The
pre-registered negative fired**: if the gap did not beat g28's, exposure heterogeneity contributes
nothing in GS even when the state prices it, and the path retires. Gap +0.0229 against +0.0283; room
+0.0053. The exposure ladder mostly raised what the LINEAR methods reach (attainable Sharpe 0.3087 to
0.4473, the linear methods from 87% to 93% of it): a beta ladder is close to a linear sort on the
characteristics that reveal it. The four new solve ids were committed before the solves ran and each
came back exactly. Spec `var-gs_bx-gx7-v1`; solves `8b584c38614695ac` (reused), `a145001bf661632e`,
`bcc4e59f41f4a040`, `9bc863211011b5d9`, `89546b0b36dfd2d0`; Phoenix arrays `21564271` (solves) and
`21564272` (seeds).

#### gamma(x) nonlin -- LEGACY, single seed, gap +0.0406, t 14.1

A logistic gamma(x) from 0.10 to 1.20; room -0.0005. "GS's flat cross-section is immune however
nonlinear the state-dependence" (REPORT.md §17e). Old calibration, no code.

### Path 2 -- exposure types under a price-of-risk regime: bx7, RETIRED 2026-09-11

#### gs_bx/bx7 -- CURRENT, gap +0.0078 (sd 0.0049), fair gap -0.0009, t 16.0

The five-type ladder above under a two-state regime on the price of risk, gmreg = [0.6, 3.0] (0.3 calm,
1.5 stress), switches 0.25/12 and 0.50/12; no level compensation, because the earlier 0.15 (beta - 1)
shift was up to 3.3 times the exposure swing it accompanied and made the cross-section "a size sort
wearing a beta label". Room +0.0086 all-month, +0.0066 evaluation window; the market reaches 94% of
SR_max. Retired because gx7 gave these types the strongest state they could have and the gap fell.
Spec `var-gs_bx-bx7-v3`; five solves `63fa7ebbc2db49ea` (beta 1), `649bb384300faadf`,
`a3bb50b66287307c`, `645262a8e72d944c`, `0818d7153d5708cc`; `SEED_SPEC=bx7`.

**The legacy ladder** (single seed, old calibration): round 1 uncompensated (room +0.0039, gap
+0.0040), round 2 compensated (+0.0064, +0.0022), bx7 v1 with the level shift (+0.0182, +0.0013), and
bx9 with beta to 9 and the regime to [0.5, 4.0] (room +0.0540, gap -0.0007): the room tripled and the
gap vanished, REPORT.md §19d's capture frontier.

### Path 3 -- regime only, and crash risk: LEGACY, single seed, old calibration

The two-state regime alone (room +0.0001, gap +0.0187: regime timing; the default channel never
fires, zero realized defaults even at a stressed price of 1.5, because firms delever and the
smoothing shock flattens the kink; §17g) and disasters destroying capital with the coupon unchanged
(`dis_rebase`, room +0.0065, gap +0.0086; §13g).

### Proposals -- GS21

Finding 8 is decisive here. In all three economies the market reaches 94% to 98% of SR_max, so there
is almost no Sharpe outside it for any method to find. GS21 prices one shock every firm loads on in
the same direction, and nothing in the gamma(x) or exposure-type families changes that: they move the
market's Sharpe, not the part it misses.

- **G1, gamma(x) times exposure types -- DONE; the negative fired** (gx7). G2, its control, dropped.
- **G3, wake the default channel -- the only GS proposal left.** Equity near default is a convex claim
  on the same shock, so its loading rises as the state worsens: heterogeneous, state-dependent
  exposures the market does not replicate. Levers: sigma_z 0.16 to 0.25, or the tax advantage of debt
  0.2 to 0.3. Probe first, oracle only at N=200, T=200 (about 20 min); proceed only if defaults reach
  half a percent a year AND the market falls below 85% of SR_max.
- **G4, a wider gamma(x), and G5, re-verifying the capture frontier -- DROPPED.** G4 raises the
  market's Sharpe and the linear shortfall, not anything a complexity method learns; G5 served the
  retired path.

---

## Cross-cutting findings

Numbered as they were found; the numbers are referenced from specs and from WORKING.md, so they are
kept. Finding 8 reorganised the file and comes first.

8. **Outside KP14 Path 1, the measured gap is the market portfolio against linear methods not given
   it on the same terms.** DKKM appends the equal-weighted market to its random features UNPENALISED
   (`--include_mkt`, as in the paper). `linrank` and `linlev` carry the same market as their constant
   column but penalise it with everything else (in g28 linrank's Sharpe goes from 0.256 unpenalised to
   0.065 at the smallest penalty on the grid); Fama-French carries a value-weighted market; Fama-MacBeth
   none. At its largest penalty DKKM collapses onto the market: in 54 of 70 BGN and GS seeds its Sharpe
   there equals the market's to within 0.001, identically across 36, 360 and 3600 features, and its
   winning portfolio is barely off that point. Splitting each gap exactly into what DKKM adds over the
   market and what the market has over the best linear method:

   | economy | SR_max eval | EW market | market / SR_max | DKKM - market | market - best linear | gap | fair gap (E1) | registered bound |
   |---|---|---|---|---|---|---|---|---|
   | kp_vy/vyx | 1.1778 | 0.3675 | 0.31 | +0.3800 | -0.2753 | +0.1046 | +0.1046 | at least +0.08: PASS |
   | bgn_gam/g0235 | 0.1461 | 0.1061 | 0.73 | +0.0020 | +0.0206 | +0.0227 | +0.0006 | at most +0.003: PASS |
   | bgn_gam/g0235f | 0.2043 | 0.1209 | 0.59 | +0.0116 | +0.0094 | +0.0211 | +0.0025 | at most +0.008: PASS |
   | bgn_gam/g0235s | 0.0946 | 0.0516 | 0.55 | +0.0100 | +0.0080 | +0.0180 | +0.0012 | at most +0.008: PASS |
   | bgn_gam/g0235r | 0.0559 | 0.0472 | 0.84 | -0.0154 | +0.0363 | +0.0209 | -0.0025 | at most +0.004: PASS |
   | gs_bx/g28 | 0.3087 | 0.2944 | 0.95 | +0.0032 | +0.0251 | +0.0283 | -0.0022 | at most +0.006: PASS |
   | gs_bx/gx7 | 0.4473 | 0.4385 | 0.98 | +0.0007 | +0.0222 | +0.0229 | +0.0001 | at most +0.004: PASS |
   | gs_bx/bx7 | 0.3121 | 0.2926 | 0.94 | +0.0038 | +0.0039 | +0.0078 | -0.0009 | at most +0.006: PASS |

   The bound was DKKM's winning Sharpe minus its Sharpe at the largest penalty, plus 0.003 for
   incomplete shrinkage; it says nothing for vyx, whose largest penalty does not reach the market. E1
   re-scored all 80 saved panels with `linrank_m`, `linlev_m` and `mkt_est`, reproducing the original
   four methods to 3e-9. In g0235r the always-long market (0.047) beats every estimator, DKKM
   included; it is reported but kept out of the benchmark because it assumes the premium's sign.

   **When does DKKM leave the market?** When there is Sharpe the market does not span, as a LEVEL:
   the per-seed non-market Sharpe sqrt(SR_max^2 - SR_ew^2) ranks with DKKM's margin over the market at
   Spearman +0.76 across the 80 seeds (+0.65 without vyx); its share of SR_max does not (+0.06). Over
   the 70 BGN and GS seeds, seeds with non-market Sharpe up to 0.05 put DKKM 0.008 below the market,
   0.10 to 0.20 put it 0.005 above, the two above 0.30 put it 0.089 above; vyx's ten, at 1.12, put it
   0.38 above. The 2026-09-13 campaign extended both halves: vyg25 and vyxT860 widened a gap the fair
   benchmark leaves within 0.0003 of itself, and g0235d put DKKM 0.081 above the market with `linrank`
   following to within 0.0004. Evidence: `variants/score_market.py`,
   `variants/market_decomposition.py`, `variants/results/market_sr.csv`; WORKING.md §53.

4. **Single seeds mislead.** g28's seed 0 gave +0.0119 against a ten-seed +0.0283; bx7's seed 0 put
   its evaluation-window room above its all-month room and ten seeds reversed it; vyx's seed 0 read
   "stronger than published" and ten seeds read "unchanged". Cross-seed sd of the gap is 15% of the
   mean for vyx, 37% for g0235, 48% for g28, 63% for bx7. No ranking below the top of a single-seed
   table means anything, which is why screens are never ranked.

5. **Absolute and proportional gaps rank the economies differently**, and the proportional ordering
   is the one finding 8 broke: it put g0235r first at 192% on a linear Sharpe of 0.011. The
   proportional column is kept for the economies where the linear methods are far from zero; the
   ranking in this file is by fair gap.

6. **Pre-registration record.** Of the eight specs run with written predictions, the two
   pre-registered decisions fired as written (gx7's negative; B4's gate) and E1's eight bounds all
   held, while the level predictions mostly missed: g0235f's on both halves, g0235s's and g0235r's on
   their second, K4's three level predictions low and X3's two high, none falsified. The lessons, each learned once: when a parameter enters the
   SOLVE, check whether the solved table moved before assuming the cross-section did not (g0235f);
   decompose a gap against the market before explaining it (finding 8); and a gain-per-decade
   extrapolation at one window does not carry to another (X3).

9. **Where the gap is genuine, it is bounded by DKKM's estimation shortfall, and two dials reduce
   it.** In all three Path 1 economies the linear methods reach 99% to 100% of their population
   ceiling and DKKM 74% to 81% of its, with the smallest penalty on the grid winning in 29 of 30
   seeds. Raising the price of the state's risk (K4) raised DKKM's share of its ceiling from 74% to
   78% and the gap by 39%; doubling the window with one more decade of penalty (X3) raised the share
   to 81% and the gap by 79%, with the window alone worth about 40% of that and the penalty the rest.
   Both leave the grid bottom binding.

Superseded readings, kept so their numbers stay findable: **1** (room does not screen for the gap;
true, and outside KP14 because the gap was the market shortfall), **2** (gap = room x capture with
capture at most one is wrong; gap/room was 1.20 in g0235 and undefined in g28, which finding 8
explains), **3** (the different-month confound was real, measured, and not the explanation for gap
exceeding room: correcting the month sample moved gap/room the wrong way, WORKING.md §49), and **7**
(the eight gaps are one mechanism in seven economies and a second in vyx: the first draft of finding
8).

## The route log: every proposal, and what became of it

| proposal | what it decided | predicted | outcome |
|---|---|---|---|
| E1 fair linear benchmark | whether the BGN and GS gaps exist | fair gap within the registered bounds; vyx at least +0.08 | DONE 2026-09-13: every bound held; fair gap at most +0.0025 outside vyx |
| K4 gamma_v 2.5 | whether more non-market Sharpe widens a genuine gap | room +0.45 to +0.50, gap +0.15 to +0.20 | DONE 2026-09-14: gap +0.1450, room +0.41; below the range, not falsified |
| X3 vyx at T=860, window 720 | whether DKKM's shortfall is data | gap +0.13 to +0.17, room unchanged | DONE 2026-09-14: gap +0.1875; the 1e-4 penalty wins every seed |
| B4 stress-dominant BGN | whether BGN has non-market Sharpe to find | room above +0.04, fair gap +0.005 to +0.015 | DONE 2026-09-14: screen missed both gates; fair gap +0.0004; regime path closed |
| B1 persistence ladder | whether spell length moves the gap at fixed room | room unchanged | DONE 2026-09-13: room moved 13x, fair gap did not |
| G1 gamma(x) times exposure types | whether exposure heterogeneity adds anything in GS | gap above g28's, or retire | DONE 2026-09-11: negative fired, path retired |
| K1 continuum of exposures | smooth exposure maps | room and gap up modestly | OPEN |
| K3 kappa_y ladder | persistence against data | gap falls at 0.15, holds or rises at 0.70 | OPEN |
| G3 default-channel probe | whether GS has any non-market Sharpe | proceed only if the market is below 85% of SR_max | OPEN, 20 min |
| K2 rare extreme type | bounded rank premia | -- | OPEN, low priority |
| K5 signed exposures | -- | -- | WITHDRAWN: infeasible, rho_ty negative |
| B2 along the frontier, B3 live rate | -- | -- | CLOSED by B4 |
| X1 window ladder on g0235 | gap over room | -- | RETIRED: answered by finding 8 |
| X2 more seeds for g28 and bx7 | -- | -- | MOOT: E1 says those gaps do not exist |
| G4 wider gamma(x), G5 capture frontier | -- | -- | DROPPED |

The next experiment, with its prediction and the case for it over the open items above:
`docs/NEXTUP.md`.

**On pushing further.** Every proposal that turns a dial up reports the annualised premia the oracle
prints (E[mu] and its cross-sectional sd are in every oracle JSON) beside the gap, so a reader can see
when an economy has left the empirically defensible range. The largest gap in a laboratory economy is
worth less than a moderate gap in one a referee will accept.

## Superseded and retired

- `var-kp_vy-vyx-v1`: the vyx parameters under the mis-normalised arrival rate. Its published row is
  the vyx line of the pre-refactor grid; do not quote for v2.
- `var-gs_bx-bx7-v1` and `-v2`: old calibration and the level-shift ladder; v2 was superseded before
  it ran. Retired manifest `a8ef7a2522eda19d` is the pre-kappa_e `sol_reg`.
- `var-gs_bx-g28-v1`: the v2 economy; its precommitted solve id was invalidated by a comment-only
  edit to the solver source.
- The pre-refactor tables (`grid_summary.csv`, `oracle_summary.csv`, `summary_grid.xlsx`) and the
  unseeded run files beside them were deleted on 2026-09-10 (WORKING.md §50); recoverable at
  `23f9380`. For every economy still in use the ten-seed measurement dominates the single-seed one.

## Adding an experiment

1. Write `experiments/specs/var-<model>-<tag>-v1.json`: title, question, the parameter override,
   `env`, `estimation`, a written prediction with a falsification clause, and `expected_solves`
   computed WITHOUT solving. Say in economic terms what differs from the baseline.
2. Solve with the model's producer (`rebuild_jstar_gam.py`, `build_vy_tables.py`, `gs_solve_reg.py`
   / `gs_solve_gam.py`); confirm the registry id matches the spec; publish with
   `variants/fetch_solves.py --publish`. A chained stage's id hashes its upstream tables' raw bytes,
   so build the chain on one platform or ship the upstream tables byte-identical (`docs/RUNS.md`).
3. Add a `SEED_SPEC` case to `variants/run_seeds_slurm.sh` (`tests/test_specs_match_shell.py` pins it
   to the spec). A new sample size needs its own tag: result files are named by model, tag and seed
   only.
4. Run ONE seed at N=500, T=500 and read `sacct` before sizing an array. Then seeds 1-9.
5. `python variants/aggregate_seeds.py --flagship` and `python variants/fair_gap.py`; commit the
   result files and the tables; add the economy here, to the current table and under its path, with
   what differs from the baseline in economic terms first and what the result decided against the
   prediction. Move its proposal into the route log. `tests/test_results_md_matches_table.py` fails
   until the table row is added.
