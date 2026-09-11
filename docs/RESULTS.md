# Experimental results

An ongoing record of every economy we have run through the oracle-and-estimator pipeline,
organised by model, with the experiments inside each model grouped into the paths that
produced them. Updated as new paths are tried. Last updated 2026-09-10 at commit `132cbc8`
(the ten-seed GS results); the current-results table below is checked against
`variants/results/economy_table.csv` by `tests/test_results_md_matches_table.py`, so this file
cannot fall behind the numbers without the suite saying so.

## How to read this

**The protocol behind every current number.** Each estimator is fit on a 360-month rolling
window of simulated data, and the portfolio it produces is then scored against the economy's
TRUE conditional moments, month by month, over the 125 evaluation months of a 500-month panel
of 500 firms. So an estimator is never judged on its own realized returns; it is judged on
what its weights were actually worth. Ten seeds per economy; every figure is mean (sd across
seeds) unless the entry says otherwise.

### The columns

- **economy** -- `model/tag`. The model directory under `variants/` and the run tag that names
  its result files.
- **spec** -- the file in `experiments/specs/` that defines this economy: its parameters, its
  estimator settings, and the `expected_solves` a run must consume or abort.
- **n** -- seeds. Ten for every current economy.
- **room all** -- the population headroom for nonlinearity, over all 485 panel months. The best
  Sharpe a nonlinear feature basis reaches with ONE fixed coefficient vector for the whole
  sample, minus the best a linear-in-ranks basis reaches the same way, both computed by the
  oracle from the true moments. No estimation, no sampling error: a statement about the
  economy's geometry.
- **room eval** -- the same quantity restricted to the 125 evaluation months. This is the one
  commensurable with `gap`, because it covers the months the estimators are scored on.
  Available for every current economy since 2026-09-10; the two figures differ by up to 23%
  (see caution 2).
- **room eval % of lin** -- `room eval` divided by the linear Sharpe attained, in percent.
  Puts the headroom on the scale of what the classical methods actually deliver here. The
  all-month version is in `variants/results/economy_table.csv` as `room_all_over_lin`; it
  differs modestly (vyx 54.3% against 56.1%, g0235 22.0% against 20.6%, bx7 3.0% against 2.3%,
  g28 0.0% against 0.2%).
- **gap** -- the quantity the project is after. The Sharpe a random-feature ridge estimator
  achieves (DKKM: `rff`, `rff_ens`, `rff_lev`, `rff_lev_ens`, best number of features and best
  ridge penalty) minus the best any linear method achieves (`linrank`, `linlev`, Fama-MacBeth
  `fm`, Fama-French `ff`).
- **gap % of lin** -- `gap` divided by the linear Sharpe attained, in percent. An absolute gap
  of +0.10 means something different against a linear Sharpe of 0.64 than against one of 0.09,
  and this column is what separates the two cases. It reorders the economies; see finding 5.
- **t** -- the paired t-statistic of DKKM against Fama-MacBeth across the evaluation months,
  averaged over seeds.
- **DKKM** -- the Sharpe the winning random-feature estimator attained.
- **best linear** -- the Sharpe the winning linear method attained. The denominator of both
  percentage columns.
- **SR_max eval** -- the oracle's maximum attainable conditional Sharpe, averaged over the
  evaluation months. A hard upper bound on every estimator by Cauchy-Schwarz, so `DKKM` and
  `best linear` must both sit under it. They do, on all forty runs.

### Two cautions on these quantities

1. **Room is not a ceiling on the gap, and not a reliable screen for it.** A rolling-window
   estimator can beat any fixed-coefficient rule when the conditional tangency portfolio moves,
   so `gap` exceeding `room` is not a contradiction and happens in two of the four economies.
   Ranking candidate economies by room does not rank them by gap. Findings 1 and 2.
2. **Every ratio in this file is a RATIO OF MEANS, not the mean of the per-seed ratios.** The
   two differ where the denominator is itself dispersed across seeds: g0235's linear Sharpe has
   sd 0.0359 on a mean of 0.0855, so its gap reads 26.5% as a ratio of means and 31.2% (sd
   17.5) as a mean of ratios. `WORKING.md` §40 records getting this wrong once. Both live in
   `variants/results/economy_table.csv`, as `gap_over_lin` and `gap_pct_lin_mean`/`_sd`.

### Status labels

- **CURRENT** -- ten seeds at N=500, T=500, window 360, run under a spec that pins the solve
  ids the run must consume; every result file carries the spec id, the solve ids and a
  provenance tag naming the commit.
- **LEGACY, single seed** -- one economy of the pre-refactor grid, run once, by code that is
  either gone (21 of its 24) or since corrected. Quoted for the path it documents, never for a
  ranking: cross-seed sd of a gap runs 15% to 63% of its mean, so single-seed rows are not
  distinguishable below the top. **The tables themselves were deleted on 2026-09-10** for the
  reasons in `WORKING.md` §50; the figures quoted below come from `variants/REPORT.md`, which
  is the narrative record of that study and stays, and the deleted tables are recoverable in
  full at `23f9380:variants/results/grid_summary.csv` and `:oracle_summary.csv`.
- **SUPERSEDED** -- an economy the current code no longer builds, kept as the record behind a
  published number.

### Whether a model's legacy rows describe the economy the code still builds

Differs by model, and is stated in each section: **yes for BGN**, **no for KP14** (the
growth-option arrival rate was mis-normalised until 2026-09-04) and **no for GS21** (three
calibration parameters were corrected against the paper's Table I on 2026-09-06).

### Provenance of every current number

`experiments/specs/<spec>.json` names the parameters and `expected_solves`;
`experiments/registry/<solve_id>.json` records what each solve was built from;
`variants/results/<model>_oracle_<tag>_s<seed>.json` and
`<model>_estimators_<tag>_s<seed>_w360_summary.csv` carry the per-seed numbers, and their
`.prov.json` sidecars name the commit. `variants/aggregate_seeds.py` produces the tables.

### Ordering

Model sections are ordered by their best current PROPORTIONAL gap (gap as a share of the linear
Sharpe attained), and within a model the paths are ordered the same way; paths with no current
economy come after those with one, ordered by their best legacy gap. Each model section ends
with the parameterizations proposed next for it, and those are collected and ranked in
"Proposed next, ranked" near the end.

## Current results

All six CURRENT economies, ranked by gap as a share of the linear Sharpe attained. Mean (sd)
over ten seeds.

| economy | spec | n | room all | room eval | room eval % of lin | gap | gap % of lin | t | DKKM | best linear | SR_max eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bgn_gam/g0235 | var-bgn_gam-g0235-v2 | 10 | +0.0188 (0.0065) | +0.0176 (0.0119) | 20.6% | +0.0227 (0.0084) | 26.5% | 33.4 | 0.1081 | 0.0855 | 0.1461 |
| bgn_gam/g0235f | var-bgn_gam-g0235f-v1 | 10 | +0.0342 (0.0091) | +0.0350 (0.0188) | 31.4% | +0.0211 (0.0154) | 18.9% | 16.6 | 0.1325 | 0.1115 | 0.2043 |
| kp_vy/vyx | var-kp_vy-vyx-v2 | 10 | +0.3491 (0.0365) | +0.3606 (0.0394) | 56.1% | +0.1046 (0.0155) | 16.3% | 30.2 | 0.7475 | 0.6428 | 1.1778 |
| gs_bx/g28 | var-gs_bx-g28-v2 | 10 | +0.0001 (0.0003) | +0.0004 (0.0003) | 0.2% | +0.0283 (0.0136) | 10.5% | 24.6 | 0.2975 | 0.2692 | 0.3087 |
| gs_bx/gx7 | var-gs_bx-gx7-v1 | 10 | +0.0019 (0.0024) | +0.0053 (0.0044) | 1.3% | +0.0229 (0.0102) | 5.5% | 15.0 | 0.4393 | 0.4163 | 0.4473 |
| gs_bx/bx7 | var-gs_bx-bx7-v3 | 10 | +0.0086 (0.0035) | +0.0066 (0.0041) | 2.3% | +0.0078 (0.0049) | 2.7% | 16.0 | 0.2964 | 0.2887 | 0.3121 |
Rows are ordered by PROPORTIONAL gap. **By absolute gap the order is different**: vyx first at
+0.1046, then g28 +0.0283, gx7 +0.0229, g0235 +0.0227, g0235f +0.0211, bx7 +0.0078.

**Two more are running** and are deliberately not in this table: `g0235s` and `g0235r`, the slow
and rare points of the persistence ladder. Seven of ten and five of ten seeds have finished; the
rest were killed for memory and resubmitted (see BGN, Path 1). The seeds still missing are not a
random subset -- they are the memory-heavy ones, which by the mechanism in that section are the
seeds whose aggregate path spent longest in the calm regime -- so a mean over the finished seeds
would be biased, and a biased mean in a results table is worse than a blank row. Which ordering matters depends on the
question -- see cross-cutting finding 5.

Regenerate with `python variants/aggregate_seeds.py --flagship`, which also writes the per-seed
rows to `variants/results/seed_table.csv`.

---

## BGN -- Berk, Green and Naik (1999)

### Baseline

A firm is a collection of live projects, each a real option that was exercised when its
present value crossed a threshold. Projects carry a market-shock loading beta_s drawn from a
translated exponential distribution (scale 0.137), an idiosyncratic cash-flow volatility, and
a project-specific cash-flow level around C-bar = -3.7; new projects arrive and are exercised
optimally against a threshold J*(r) that depends on the short rate. The short rate is Vasicek
(monthly persistence kappa = 0.95, mean 0.006236, innovation sd sigma_r = 0.002). The pricing
kernel is exogenous and lognormal with ONE priced shock, the market shock nu, at price
sigma_z = 0.4, correlated -0.175 with the rate innovation (beta_zr = -0.00014). Expected
returns are exactly affine in book-to-price and 1/price with rate-dependent coefficients.
Calibration checked against the paper's Table I: 11 of 11 parameters match, including two the
code derives rather than stores.

What this implies for the cross-section: the project-beta distribution is the only source of
firm-level exposure heterogeneity in any of the three models. REPORT.md §17's closing
sentence: "only BGN's project-beta distributions supply that heterogeneity."

**Legacy rows are the same economy.** The BGN calibration survived every audit unchanged, and
the current code reproduced the legacy g0235 row to the displayed digit on a different machine
and Python version (WORKING.md §39). Legacy BGN levels are citable, single-seed caveat only.
Baseline row: room +0.0365, gap +0.0059, t 14.0.

### Path 1 -- a two-state regime on the price of the market shock: g0520, then g0330, then g0235, then its persistence

#### bgn_gam/g0235 -- CURRENT, gap +0.0227 (sd 0.0084), t 33.4

**What differs from the baseline.** A two-state Markov regime s_t (calm, stress) multiplies
the price of the market shock: sigma_z becomes sigma_z x gmult[s] with gmult = [0.2, 3.5] --
one fifth of the baseline price in calm months, three and a half times it in stress. Monthly
switch probabilities 0.25/12 calm-to-stress and 0.50/12 stress-to-calm; switches are
unpriced (conditional moments are physical expectations, and the regime enters only through
the regime-indexed value tables). The upper multiplier sits just under the bound 1/(2 x scale)
= 3.65 that the exponential beta-density tail imposes on the closed-form project values. The
project-exercise threshold J* is re-solved for the regime economy (the pinned solve). The
regime and the rate are exported as conditioning features.

The economic content that makes this path different from KP's Path 2: a project's value
decomposes onto two regime bases, V_s(r, beta) = C-hat [exp(-beta g0) DA_s(r) + exp(-beta g1)
DB_s(r)], so the two regimes apply two DIFFERENT monotone transforms of the same project
beta. That produces curvature in the map from beta to the premium WITHIN a regime, which a
common scaling of the price of risk cannot. This is why regime-gamma creates room in BGN and
not in KP14.

**Why it was tried.** The KP regime (Path 2 there) was built first and proved inert; the BGN
version was then built in closed form to test the same mechanism where the cross-section had
heterogeneity to bend. g0520 worked; g0330 widened it; g0235 pushed to the frontier.

**Result.** The third-largest gap, with the tightest relative error bar of the four (se
0.0027). DKKM 0.1081 against best linear 0.0855 (linrank; Fama-MacBeth collapses to 0.067 and
the level-linear method to 0.034, because regime shifts move raw levels). Room +0.0188
all-month; gap/room 1.20, so the realized gap exceeds the constant-coefficient room in seven
of ten seeds -- the finding that first showed room is not a ceiling (WORKING.md §40). The
published single-seed row (room +0.0233, gap +0.0297, t 29.1) is reproduced exactly by seed 0
and sits 0.8 sd above the ten-seed mean.

**Provenance.** Spec `var-bgn_gam-g0235-v2`; solve `be222462dd017b2c` (the J* table
`Jstar_g0235.csv`); seed array `SEED_SPEC=g0235`.

#### bgn_gam/g0235f -- CURRENT, gap +0.0211 (sd 0.0154), 18.9% of the linear Sharpe, t 16.6

**What differs from g0235.** Only the two regime switch probabilities, both scaled by four:
calm-to-stress 0.25/12 to 1/12 and stress-to-calm 0.50/12 to 2/12. Calm spells average 12 months
instead of 48, stress spells 6 instead of 24, so a 360-month estimation window holds about forty
regime switches instead of ten. The stationary share of stress months stays at exactly one third
and the multipliers stay at [0.2, 3.5]. The exercise-threshold table J* is re-solved, because the
switch probabilities enter it through the regime transition matrix.

**Why it was tried.** Proposal B1, the regime-persistence ladder. Spell length relative to window
length is the dial on the mechanism behind gap exceeding room (`WORKING.md` §40, §49), and it had
never been turned.

**What it decided: the pre-registered prediction failed on both halves.** The spec predicted room
unchanged, since the stationary mix is held fixed, and a RISING proportional gap, since a window
average blends two regimes more finely when they switch faster. Room nearly doubled instead
(evaluation window +0.0176 to +0.0350) and the proportional gap FELL (26.5% to 18.9%).

Room moved because switching speed is not a path parameter. How long a firm expects to stay in the
cheap-risk calm regime changes what its growth options are worth, so it changes which projects firms
take and therefore the cross-section itself. The J* table said so before a single seed ran: its
value range is 37 to 345, against g0235's 57 to 483, and memory fell with it (15.7 to 21.6 GiB, the
lightest BGN point). I read that table as bookkeeping rather than as evidence the economies
differed, and the spec and the commit that launched it called g0235f "the same economy at a
different speed". It is not. So B1 cannot run the test it was designed for, a gap level moving at
constant room.

**A post-hoc reading, labelled as one.** Measured as gap over room -- how far the estimators get
past or short of the best constant-coefficient rule -- fast switching gives 0.60 against g0235's
1.28. That is the direction the rolling-window mechanism predicts: with forty switches inside a
window, the window cannot track the conditional tangency, the estimators collapse toward a
constant-coefficient rule, and they finish short of it. It was not the pre-registered reading, so it
is an observation that sharpens what the slow point has to show, not a confirmation. `g0235s`, where
the window should track and gap over room should be HIGHEST, is the half that would make it a test.

**Provenance.** Spec `var-bgn_gam-g0235f-v1` (`precommitted: false` -- the table was built before
the spec was committed; the id was nonetheless computed first and matched); solve
`8662d7c1079f4b41`; Phoenix array `21564268`, all ten seeds CURRENT, 3.0 to 3.3 h each.

#### g0235s and g0235r -- RUNNING, the slow and rare points

`g0235s` switches five times more slowly than g0235 at the same one-third stress share (calm 240
months, stress 120, about two switches per window); `g0235r` makes stress rare and short (10% of
months, spells of 27 months). Both finished some seeds -- seven of ten and five of ten -- and are held
out of the table above until the rest land, because the missing seeds are systematically the heavy
ones.

**Memory is the finding so far, and it is the mechanism above made visible.** Longer calm spells
price risk cheaply for longer, so options are worth more, firms accept more projects and carry
larger inventories, and the panel arrays grow with them. The exercise-threshold value scale orders
exactly as memory does:

| economy | J* value range | peak memory (Phoenix) | wall per seed |
|---|---|---|---|
| g0235f | 37 to 345 | 15.7 to 21.6 GiB | 3.0 to 3.3 h |
| g0235 | 57 to 483 | 15.7 to 38.9 GiB (Sol) | about 3.2 h |
| g0235s | 148 to 1262 | 31.9 to 63.9 GiB on the seeds that fit | 4 to 12.4 h |
| g0235r | 390 to 2041 | 71.8 to 74.0 GiB | 14 to 20 h |

All ten `g0235r` seeds and three `g0235s` seeds were killed at the 64 GiB cap; one `g0235s` seed
finished 200 MB under it. Six `g0235r` seeds were re-run on the highmem partition at 200 GiB to
MEASURE the peak rather than guess it again, and the rest were resubmitted at 110 GiB. This also
supplies an explanation for the unexplained bimodality in g0235's memory (15.7 to 38.9 GiB across
seeds on identical nodes, `WORKING.md` §40): whether a seed's aggregate path spends long stretches
in calm decides how large the inventories grow.

#### g0520 and g0330 -- LEGACY, single seed, same economy family

- **regime-g** (`g0520`, gmult = [0.5, 2.0]; room +0.0214, gap +0.0219, t 21.5): the first
  build. DKKM landed at its unconditional ceiling, 100% of the room harvested; at the time
  "the strongest estimated gap of the project". REPORT.md §13f/h.
- **regime-g wide** (`g0330`, gmult = [0.3, 3.0]; room +0.0254, gap +0.0240, t 31.7): only the
  spread changed. Room grows with the regime gap; the classical raw-level methods break
  outright (FMR 0.105, FF 0.132) because regime shifts move raw characteristic levels; DKKM
  beats FMR by 89% relative. §16.

Read as a path: room saturates around +0.021 to +0.025 across the three while the gap keeps
rising with the spread, from +0.022 to +0.030 (single-seed) -- the widening damages the
linear methods faster than it adds population room.

### Path 2 -- a continuous state-dependent price of risk in the rate, gamma(r)

**gamma(r) nonlin** (`bgn_gamr`) -- LEGACY, single seed, room +0.0934, gap +0.0060, t -2.7.
The regime replaced by a logistic multiplier 0.5 to 3.0 on the market-shock price as a
function of the short rate, the model's own continuous state. Closed forms break; solved on
an r grid with a rank-7 factorisation of the value function. The largest room of the entire
project, and rolling Fama-MacBeth harvests it: FMR 0.470 against DKKM 0.464, because raw
characteristic levels co-move with r and a rolling raw-level regression conditions on the
rate for free. The conditioning gap is contestable by cheap classical conditioning. REPORT.md
§17c.

### Path 3 -- crash risk with firm types

**crash** (`bgn_dis`) -- LEGACY, single seed, room +0.0658, gap +0.0053, t 8.9. The kernel is
multiplied by kappa^D / ((1-p) + p kappa) with D a monthly disaster indicator, p = 3% and kappa
= 2.5; three permanent firm types whose live projects die in a disaster month with
probability 0, 0.15 or 0.30; a value compensation omega = 1.3 so that disaster-exposed firms
look like growth firms rather than being revealed by value. Motivation: book-to-market then
carries two offsetting premium channels (the beta channel rising, the disaster channel
falling), a shape no linear rule can sign. The largest BGN room short of gamma(r), only
three-quarters captured. REPORT.md §9, §14.

### Path 4 -- the earlier exploration (oracle only)

Parameter sweeps, thin versus fat cash-flow tails, size-dependent idiosyncratic volatility,
firm types with dispersed versus standard projects, and a U-shaped rate-shock correlation
were run through the oracle only (no estimators) before the grid. None produced room above
+0.008 except the disaster and regime paths above. REPORT.md §3-§8 is the record; the
57-row table that held their numbers spanned seven model families with no code in this
repository and was deleted (`23f9380:variants/results/oracle_summary.csv`).

### Proposed next parameterizations -- BGN

Each entry says what would differ from g0235, why the number is worth having, what result would
mean, and what it costs. The regime block sits inside the J* solve, so every point needs a J*
rebuild (minutes), then ten seeds at about 3.5 h each on Sol.

- **B1. Regime persistence -- RUNNING; the fast point is in, and its prediction failed.**
  Three points on g0235's multipliers: fast (4x), slow (5x slower), and rare-and-short stress.
  The design assumed that holding the stationary stress share fixed would hold room fixed, so a
  gap moving across the speed ladder would isolate the rolling-window mechanism. The fast point
  refuted that assumption: room nearly doubled, because switching speed changes option values and
  therefore the cross-section (Path 1, g0235f). What survives is a post-hoc reading in terms of gap
  over room -- 0.60 fast against 1.28 at the baseline speed -- which the slow point, still running,
  would have to extend to count. The slow and rare points also turned out to be the memory-heavy
  end of the whole project, 64 to 74 GiB and up to 20 hours a seed.

- **B2. Along the frontier.** The multiplier is capped near 3.65 by the closed-form requirement
  2 x gmult x scale < 1, where `scale` is the exponential tail of the project-beta distribution
  and is fitted from the two acceptance-probability targets (0.10, 0.05). The path from g0520 to
  g0235 walked the gmult axis at fixed scale and found room saturating while the gap rose. The
  frontier is a curve, not a point: a thinner tail (targets lowered so scale is near 0.10)
  admits gmult up to 5, a fatter one (scale near 0.20) caps it at 2.5 but gives the two regime
  transforms a more dispersed beta cross-section to bend. **Why.** Whether the gap is about the
  size of the multiplier or the dispersion of the exposures it acts on is the open design
  question for this path, and two points on the frontier settle it. **Prediction.** Not
  confident either way, which is the point. **Caveat.** The acceptance targets also move the
  baseline cross-section, so each frontier point wants its gmult = [1, 1] control, oracle only,
  to separate the two effects.

- **B3. A live interest rate under the regime.** sigma_r = 0.002 makes the short rate nearly
  constant, so the model's own continuous state does nothing. The legacy gamma(r) economy showed
  the rate is a powerful state and that rolling raw-level regressions harvest it because levels
  co-move with r. With the discrete regime doing the pricing and sigma_r raised to 0.004 or
  0.006 supplying a second, continuous state, the question is whether DKKM's rate-by-regime
  interactions gain more than the linear methods' level timing. **Prediction.** The linear
  methods improve, since raw book-to-price levels time the rate; DKKM improves more only if the
  rate-by-regime interaction is where the premium moves. The proportional gap could go either
  way. Lower priority than B1 and B2.

---

## KP14 -- Kogan and Papanikolaou (2014)

### Baseline

A firm is a collection of live projects (assets in place) plus the growth opportunities it
expects to receive. Projects arrive to firm f at a rate lambda_f that switches between a high
regime (lambda_H = 2.35) and a low one (lambda_L, set so that the mean arrival rate is exactly 1;
switching intensities mu_H = 0.075 into the high state and mu_L = 0.16 into the low state).
A project's output depends on firm-level profitability eps_f (mean-reverting, theta_eps = 0.35,
sigma_eps = 0.2), project-level u (theta_u = 0.5, sigma_u = 1.5), aggregate productivity z
(mu_z = 0.005, sigma_z = 0.035) and the investment-specific technology state x (mu_x = 0.01,
sigma_x = 0.13); alpha = 0.85; depreciation delta = 0.1. The pricing kernel is exogenous with
CONSTANT prices of risk: gamma_x = 0.69 on the technology shock and gamma_z = -0.35 on
productivity; r = 0.05 (the paper's 0.025, a deliberate departure). Calibration checked
against the paper's Table II: 17 of 18 parameters match, r being the exception.

What this implies for the cross-section: expected returns are affine in a single firm
variable, the share of value that is growth options (PVGO/V). The cross-section is
one-directional, and no state-dependent price of risk can create curvature in it by scaling
alone -- the finding behind every inert path below.

**Legacy rows are a different economy.** Until 2026-09-04 the regime probability was read
with the wrong label, so the mean arrival rate simulated was 1.72 instead of 1. Every KP row
of the pre-refactor grid, including its baseline row (room +0.0037, gap +0.0036, t -8.5), was
produced at that rate. Levels of legacy KP numbers are not citable for the current code;
the direction of each path's finding is what survives.

### Path 1 -- a priced, mean-reverting aggregate state with heterogeneous firm exposure: vy, then vyx

The best path in the repository, and the only one in KP14 that ever produced a gap.

#### kp_vy/vyx -- CURRENT, gap +0.1046 (sd 0.0155), t 30.2

**What differs from the baseline.** A new aggregate state y is added: a stationary
Ornstein-Uhlenbeck process with mean-reversion kappa_y = 0.35 and unit stationary standard
deviation (REPORT.md calls it the priced-volatility factor; in the code it is a generic priced
state). Its innovations are PRICED, at gamma_v = 1.8. Firms come in three types, in shares
0.34 / 0.33 / 0.33, whose project cash flows load on the state as exp(beta_f y) with
beta = 0.02 / 0.07 / 0.14. Because y is priced, each type earns a premium proportional to
its beta_f; because y mean-reverts, the VALUE of an exp(beta y) cash-flow stream depends on
where y is, so the map from a firm's type to its observable characteristics is bent by the
state rather than shifted. A compensation term (bv_comp = 1.2) rescales each type's cash-flow
level so that at y = 0 firm values do not reveal the type monotonically -- otherwise a linear
sort on value would recover the exposure ladder for free. Every level in the economy stays
stationary, so raw characteristics do not trend with the state; that is the design clause
that distinguishes this path from Path 3, where rolling Fama-MacBeth harvested the room
through trending levels.

**Why it was tried.** REPORT.md §18-§19's design rule: a harvestable gap needs heterogeneous
firm-level exposures, a state that bends them nonlinearly, and the bending confined to rank
and interaction space. vy was the first KP economy to satisfy all three; vyx pushes its two
free knobs (gamma_v 1.2 to 1.8, top beta 0.12 to 0.14) toward "premium-side extremity",
REPORT.md's conjecture for where capture rises with room.

**Result.** By far the largest gap in the repository, at four times the next economy, and the
largest room. DKKM 0.7475 against best linear 0.6428 (linrank). SR_max over the evaluation
months 1.1778, so DKKM reaches 63% of the attainable maximum. Ten seeds; gap/room 0.30, the
one economy where capture below one holds cleanly.

**The lambda fix did not change this economy's numbers.** The v1 spec ran at the
mis-normalised arrival rate and published room +0.3496, gap +0.1009, t 21.6 from one seed;
the corrected economy at ten seeds gives +0.3491 and +0.1046. WORKING.md §37.

**Provenance.** Spec `var-kp_vy-vyx-v2`; solves `f7be27e39d2b530f` (the G function) and
`84e195172f091cd2` (the type-by-state integrals); seed array `SEED_SPEC=vyx`.

#### kp_vy/vy -- LEGACY, single seed, gap +0.0289, t 5.2

Same construction with gamma_v = 1.2 and beta = 0.02 / 0.06 / 0.12. Room +0.278, the first
significant DKKM win in KP14 after five state-dependent-price designs, but capture near 10%
and limited by T: the best realized fit was at P = 360 features, not 3600. REPORT.md §19a.
Pre-lambda-fix economy.

### Path 2 -- state-dependent prices of risk on the baseline cross-section: inert, four ways

All LEGACY single-seed rows, all pre-lambda-fix, all with room below +0.004. They are the
negative results that led to Path 1.

- **regime-g** (`kp_gam`, room +0.0015, gap +0.0067, t 5.1): a two-state Markov regime
  multiplies BOTH prices of risk by gmult[s] = 0.5 in calm and 2.0 in stress; switching
  intensities 0.25/yr calm-to-stress and 0.50/yr stress-to-calm; switches unpriced. The first
  state-dependent-price build. REPORT.md §13c-d proves why it is inert: with premia affine in
  the firm variable within each regime, the conditional premium is spanned by (1, X, s, X s),
  which a linear basis with a regime interaction already contains.
- **gamma(y) common** (`kp_gamy`, room +0.0033, gap +0.0074, t 3.0): the regime replaced by a
  continuous OU state y with a logistic multiplier 0.5 to 2.5 on both prices of risk. §17b's
  refinement: a common multiplier gives mu_t = g(y_t) mu-bar, a constant cross-sectional
  direction, so no common scaling can create room in any model.
- **gamma(y) rotation** (room +0.0035, gap +0.0022, t 4.7): the multiplier 0.85 to 3.0 on
  gamma_x only, gamma_z fixed, so the premium direction rotates 3.5x across the state. Inert:
  KP firms load on the two shocks almost in parallel, and the cross-sectional dispersion of
  expected returns does not move while its mean doubles. §17d: "KP's cross-section is
  irreducibly one-directional."
- **regime uneven** (room +0.0025, gap +0.0185, t 9.6): gmult_x = [0.6, 2.4] with gmult_z
  fixed at 1. The +0.019 realized is regime timing through the exported regime feature,
  which a linear basis with the interaction captures; room stays nil. §17f.

### Path 3 -- exposure types crossed with a regime: room without harvest

**kp_bx** -- LEGACY, single seed, room +0.0187, gap +0.0016, t -1.8. Firm types whose cash
flows load on the technology state as x^{beta_f}, beta in {1.0, 1.8, 3.0}, crossed with the
Path 2 regime (0.5 / 2.0) and a calm-state value compensation of 1.2. The first genuine room
in KP14 -- and rolling Fama-MacBeth on raw levels harvested it (FMR 0.181 against DKKM 0.177)
because x^{beta_f} makes raw characteristics co-move with the state. This is the failure that
produced the third design clause and the stationary-level construction of Path 1. REPORT.md
§18.

### Path 4 -- crash risk

**kp crash** (`kpd_al`) -- LEGACY, single seed, room +0.0058, gap -0.0064, t -2.1. Rare
disasters destroy a fraction of each firm's projects with a type-specific probability, the
kill intensity is priced, and a type output multiplier compensates values. REPORT.md gives no
numeric disaster parameters for the KP version. The rank-linear method came out ahead of
DKKM. §12b: "crash risk per se does not create a DKKM gap."

### Proposed next parameterizations -- KP14

vyx already has the largest absolute gap and the largest room, and its linear methods reach
0.64, so the proportional gap is only 16%. Every proposal below tries to lower what the linear
methods can do without lowering what DKKM can. Each point needs the G solve (seconds per type)
and the type-by-state integrals (about 30 min per type), then ten seeds at about 2.8 h.

- **K1. A continuum of exposures.** Replace the three point-mass types with fifteen on a grid
  over [0, 0.14], shares right-skewed so most firms sit low. **Why.** Three types can be nearly
  spanned by a linear rule keyed on any characteristic that reveals type, which is exactly what
  the compensation knob (bv_comp = 1.2) is there to hide. A continuum has no three points to key
  on: the exposure map is smooth in a latent variable that the characteristics reveal only
  nonlinearly. **Prediction.** The linear Sharpe falls, room rises with the added dispersion,
  DKKM holds, so the proportional gap rises. If the linear methods keep 0.64 the type structure
  was never what they were exploiting. The machinery takes any number of types; the integral
  stage is the cost, about 7.5 h once for fifteen types.

- **K2. A rare extreme type.** Three types with shares (0.45, 0.45, 0.10) and the top exposure
  raised from 0.14 to 0.20. **Why.** This targets the rank transform specifically. A 10% type at
  the top gets ranks 0.9 to 1.0 whatever its exposure, so a linear-in-rank premium is bounded
  while the true premium is not; a random-feature basis can place a bump on those ranks, a
  linear rule can only tilt. **Prediction.** Room and proportional gap both rise; the
  level-linear method may hold better than the rank-linear one, which would itself be
  informative. **Check first.** The per-type value coefficients need a positive effective
  discount across the state grid; the table builder refuses if not, and beta = 0.20 sits closer
  to that bound than anything run so far.

- **K3. Persistence of the priced state.** kappa_y = 0.35 gives y a two-year half-life. The
  bending comes from the state entering the value of an exp(beta y) stream through the
  mean-reversion pull, so slower reversion (0.15, half-life 4.6 years) bends more but gives a
  360-month window fewer independent cycles to learn from; faster (0.70) does the reverse.
  **Why.** The mechanism has a dial that trades room against capture and it has never been
  turned. **Prediction.** Room rises as kappa_y falls; capture falls; the gap has an interior
  optimum, which may or may not be at 0.35.

- **K4. One more premium-side step.** gamma_v from 1.8 to 2.5. The path from vy to vyx raised
  capture from 0.10 to 0.29 along with room, the basis for the conjecture that premium-side
  extremity raises both. One more step says whether that continues. **Caveat, and the reason
  this is last.** The type premia were already near 6, 15 and 28 percent a year under the old
  labels (not re-derived since the arrival-rate fix); at gamma_v = 2.5 they approach 40 percent.
  Worth knowing where the mechanism ends; not a calibration anyone would defend.

---

## GS21 -- Gomes and Schmid (2021)

### Baseline

A firm holds capital k and one-period debt b, produces exp(x + z) k^alpha-style output from
an aggregate productivity state x (AR(1), quarterly persistence 0.95 and innovation sd 0.012,
converted to a monthly step) and an idiosyncratic state z (quarterly persistence 0.90, sd
0.16), pays a maintenance cost delta = 0.02 per quarter on capital, taxes at tau = 0.2, and
chooses investment, borrowing and default each period. Equity issuance costs kappa_e = 0.025
on a negative cash flow, debt issuance costs kappa_b = 0.004, lenders recover phi = 0.4 in
default, and default is smoothed by a shock of sd sigma_m = 5 so the kink is differentiable.
The pricing kernel is exogenous, exp(-r - gamma^2/2 - gamma eps_x) with a CONSTANT price of
risk gamma_x = 0.5 on the productivity shock and r = 0.10/yr: the paper's general-equilibrium
Epstein-Zin kernel with its countercyclical price of risk is replaced by this constant-price
stand-in, which is what Path 1 puts back. Solved by value-function iteration on a 161-point
x grid to tolerance 1e-6 per solve, roughly 3.5 h each.

What this implies for the cross-section: with a constant price of risk every feature basis
reaches the same population ceiling. GS21 has essentially no learnable nonlinear
cross-section, and the gap it produces comes from somewhere else.

**Legacy rows are a different economy.** Until 2026-09-06 the solver ran with rho_x =
0.96^(1/3) (the paper says 0.95), delta = 0.02 per month (the paper's 0.02 is per quarter),
and kappa_e = 0 (the paper's benchmark is 0.025). Every GS row of the pre-refactor grid,
including its baseline row (room +0.0010, gap +0.0163, t 21.8), was solved under those values.
Levels are not citable for the current code; the mechanism findings survive.

### Path 1 -- a countercyclical, continuous price of risk gamma(x): g28, then crossed with exposure types

#### gs_bx/g28 -- CURRENT, gap +0.0283 (sd 0.0136), t 24.6

**What differs from the baseline.** One firm type and no regime. The price of risk becomes a
function of the aggregate state: gamma(x) = clip(0.5 - 0.28 x / sd(x), 0.05, 1.0), with sd(x)
the stationary standard deviation of x. So gamma is the baseline 0.5 when productivity is at
its mean, falls toward 0.05 in booms and rises toward 1.0 in busts -- the minimal stand-in
for the paper's general-equilibrium kernel, whose price of risk is countercyclical. The
solver's kernel is re-derived on the x grid with the state-dependent gamma; the simulator and
the firm's problem are otherwise the baseline (corrected Table I calibration, kappa_e =
0.025). This is a RECONSTRUCTION: the original `sol_g28` code was never in the repository and
the economy was rebuilt from the formula in REPORT.md §13g, verified to reproduce the regime
solver byte-for-byte at slope zero.

**Why it was tried.** The published grid had this economy at rank 3 of 24 on gap with room of
essentially zero, the sole evidence that a DKKM advantage can arise from estimation
efficiency rather than from any nonlinearity in the cross-section. That conclusion rested on
deleted code. The spec's question: "can a gap open with ZERO nonlinear room?"

**Result.** Yes, and it is the second-largest gap in the repository. Room is +0.0001 all-month
and +0.0004 over the evaluation months -- the nonlinear and linear population ceilings are
the same number -- while DKKM beats the best linear method by +0.0283. Decomposed against the
evaluation-window ceilings: the best linear estimator lands 0.020 below its own ceiling, DKKM
0.008 below its. The whole advantage is the linear estimators' inefficiency, and the ridge
random-feature estimator's ability to use the exported state feature. Ranked by room this
economy is last of four; ranked by gap it is second. The cross-seed sd (0.0136) is half the
mean, the noisiest gap of the four.

**The seed-0 reading was misleading.** Seed 0 alone gave +0.0119; the ten-seed mean is 2.4
times that. WORKING.md §48.

**Provenance.** Spec `var-gs_bx-g28-v2`; solve `8b584c38614695ac` (producer
`gs_solve_gam.py`, one 5.9 h solve on Phoenix); seed array `SEED_SPEC=g28`. The legacy
single-seed row (room -0.0004, gap +0.0383, t 17.9) was the old calibration and a different
implementation; only the signature is expected to match, and it does.

#### gs_bx/gx7 -- CURRENT, gap +0.0229 (sd 0.0102), 5.5% of the linear Sharpe, t 15.0

**What differs from g28.** Five firm types instead of one, in equal shares, loading on the aggregate
state as exp(beta_f x + z) with beta = 1 / 2.5 / 4 / 5.5 / 7. The countercyclical price of risk
gamma(x) = clip(0.5 - 0.28 x / sd(x), 0.05, 1.0) is unchanged and applies to every type; there is no
regime. So a type's premium is proportional to beta_f times gamma(x), which is nonlinear in the pair
because gamma is clipped: the state bends a heterogeneous exposure map. That is the design rule that
produced vyx, used here for the first time with GS21's own aggregate state. The beta = 1 member IS
sol_g28, reused; the other four types are new solves.

**Why it was tried.** Proposal G1: the two GS economies crossed, g28's state-dependent price with
bx7's exposure heterogeneity.

**What it decided: the pre-registered negative fired, and the GS exposure path is retired.** The
spec said that if the gap did not beat g28's alone, exposure heterogeneity contributes nothing in
GS even when the state prices it, and the path retires rather than being pushed to bx9. The gap is
+0.0229 against g28's +0.0283 and 5.5% of the linear Sharpe against 10.5%: lower on both. Room did
not appear either, +0.0053 over the evaluation window against bx7's +0.0066. This rests on a genuine
precommitment: the four new solve ids were committed before the solves ran and each came back
exactly, so nothing here was tuned after the fact.

Why it failed: the exposure ladder mostly raised what the LINEAR methods could reach. The attainable
Sharpe over the evaluation months rose from g28's 0.3087 to 0.4473, and the linear methods' from
0.2692 to 0.4163 -- 93% of the attainable, against 87% in g28. DKKM reaches 98%. A beta ladder is
close to a linear sort on the characteristics that reveal it, so almost none of the new
cross-sectional variation was beyond the linear methods, and there was little left for a
complexity method to win.

**Provenance.** Spec `var-gs_bx-gx7-v1` (`precommitted: true`, with `reused_solves` naming sol_g28
as inherited from `var-gs_bx-g28-v2`); solves `8b584c38614695ac` (reused), `a145001bf661632e`,
`bcc4e59f41f4a040`, `9bc863211011b5d9`, `89546b0b36dfd2d0`, all on the tolerance exit, 2199 to 2799
sweeps, one at qerr 1.99e-5 just inside the 2e-5 threshold; produced by
`variants/gs_bx/run_gx7_slurm.sh` on Phoenix (array `21564271`); ten seeds by array `21564272`,
chained `afterok` behind the solves, all CURRENT.

#### gamma(x) nonlin -- LEGACY, single seed, gap +0.0406, t 14.1

Same idea with a logistic gamma(x) from 0.10 to 1.20 instead of the clipped line. Room
-0.0005. REPORT.md §17e: "GS's flat cross-section is immune however nonlinear the
state-dependence"; the realized gap is conditioning. Old calibration; no code.

### Path 2 -- heterogeneous exposure to the productivity state under a price-of-risk regime: bx7 (RETIRED)

**Retired 2026-09-11.** gx7 (Path 1) gave these exposure types the strongest state they could have
-- a continuous, countercyclical price of risk rather than a two-value regime -- and the gap fell
rather than rose, because the ladder mostly raised what the linear methods reach. bx9 is therefore
not worth re-verifying.

#### gs_bx/bx7 -- CURRENT, gap +0.0078 (sd 0.0049), t 16.0

**What differs from the baseline.** Two changes. First, five firm types in equal shares whose
production loads on the aggregate state as exp(beta_f x + z) with beta = 1 / 2.5 / 4 / 5.5 /
7 -- the baseline is beta = 1 for every firm -- so firms differ in how strongly the aggregate
state moves their output, and each type is a separate solve of the firm's problem. Second, a
two-state Markov regime multiplies the price of risk: gamma_s = 0.5 x gmreg[s] with gmreg =
[0.6, 3.0], i.e. 0.3 in calm and 1.5 in stress; monthly switch probabilities 0.25/12
calm-to-stress and 0.50/12 stress-to-calm; switches unpriced. No level compensation
(gs_ashift = 0): earlier versions shifted each type's productivity level by 0.15 (beta - 1)
to keep values comparable across types, and that shift was up to 3.3 times the exposure
swing it accompanied, so the cross-section was "mostly a size sort wearing a beta label"; v3
removes it so the five types differ in exposure and nothing else. Six characteristics
(leverage added) and an eight-value kappa grid.

**Why it was tried.** The GS analogue of KP's Path 3 and Path 1: give the state something
heterogeneous to bend. Built up through three rounds (below).

**Result.** A real but small gap, six times the number that demoted this economy to rank 22 of
24 in the published grid. Room +0.0086 all-month, +0.0066 evaluation-window; gap/room 0.90
against all-month room and 1.17 against the evaluation window. bx7 is the one CURRENT economy
where a seed-0 reading of the eval-window direction reversed at ten seeds.

**Provenance.** Spec `var-gs_bx-bx7-v3`; five solves `63fa7ebbc2db49ea` (sol_reg, beta 1),
`649bb384300faadf`, `a3bb50b66287307c`, `645262a8e72d944c`, `0818d7153d5708cc`; produced by
`variants/gs_bx/run_gs_bx7_slurm.sh` on Sol; seed array `SEED_SPEC=bx7`.

#### The legacy ladder: rounds 1-2, bx7 v1, bx9 -- all single seed, old calibration

- **exposure-types x regime, round 1** (room +0.0039, gap +0.0040, t 10.8): the first
  uncompensated build; REPORT.md §19 records the room and no parameters.
- **exposure-types compensated, round 2** (room +0.0064, gap +0.0022, t 16.8): a level
  compensation added.
- **bx7 v1** (room +0.0182, gap +0.0013, t 10.4): the five-type ladder with the 0.15 (beta-1)
  level shift, under the old calibration. Its published gap was measured against `linlev`,
  the strongest linear method for it, which is the current definition too. SUPERSEDED twice:
  by the Table I corrections and by dropping the level shift.
- **bx9** (room +0.0540, gap -0.0007, t 5.1): beta pushed to 9 and the regime to [0.5, 4.0].
  Room tripled and the gap over linear vanished: REPORT.md §19d's "capture frontier". The
  reason the README stopped GS at bx7.

### Path 3 -- regime only, and crash risk

Both LEGACY, single seed, old calibration.

- **regime-g** (room +0.0001, gap +0.0187, t 19.9): the two-state regime alone, gamma 0.3 calm
  / 1.5 stress, no exposure types. Room exactly zero: the default option, the channel that
  might have awakened under stress, never fires (zero realized defaults even at gamma 1.5,
  because firms delever endogenously and the smoothing shock flattens the kink). The realized
  +0.019 is regime timing. REPORT.md §17g.
- **crash** (`dis_rebase`, room +0.0065, gap +0.0086, t 5.1): disasters destroy a fraction of
  capital with the coupon unchanged, so leverage jumps inside both Bellman recursions;
  per-type re-solves. No room. §13g.

### Proposed next parameterizations -- GS21

Two current economies with opposite profiles: g28 has a gap and no room, bx7 has a little of
each. The natural first move combines them. Every GS point is expensive at the solve stage
(3.5 to 6 h per type on Sol, one solve per type), so the order matters.

- **G1. gamma(x) times exposure types -- DONE, and the pre-registered negative fired.** Gap
  +0.0229 against g28's +0.0283, 5.5% of the linear Sharpe against 10.5%. The GS exposure path is
  retired (Path 1, gx7). **G2**, the no-regime control, was to run only if G1 was ambiguous; it
  was not, so G2 is dropped.

- **G3. Wake the default channel.** GS21's native nonlinearity is equity as a levered claim
  near default, and it has never fired: zero realized defaults even at a stressed price of risk
  of 1.5 (legacy §17g), because firms delever. Levers that push firms toward the boundary:
  higher idiosyncratic volatility (sigma_z from 0.16 to 0.25 at the quarterly scale) or a larger
  tax advantage of debt (tau from 0.2 to 0.3) so optimal leverage rises. **Probe first.** The
  simulator exports a per-firm-month default flag in the panel, so one oracle-only run at
  N=200, T=200 (about 20 min) reports the realized default rate before any estimator time is
  spent; proceed only if defaults reach half a percent a year. **Prediction if it fires.** The
  leverage characteristic, already exported, acquires a convex premium map near the boundary
  that a rank-linear rule cannot follow, and GS shows room from a source of its own for the
  first time.

- **G4. A wider gamma(x).** Slope 0.5 with the clip widened to [0.02, 2.0]. **Why.** g28's DKKM
  already reaches 96% of the attainable Sharpe and its linear methods 87%, so g28's gap is
  capped by the linear shortfall unless the attainable Sharpe itself grows; larger price-of-risk
  swings raise it and enlarge the common-premium variation the linear methods fail to time.
  **Prediction.** The absolute gap grows; the proportional gap may not, because the linear
  Sharpe grows too. **Risk.** Convergence at gamma = 2: g28 met tolerance at sweep 2974 of a
  5600 cap, and the manifest records which exit a solve took.

- **G5. Re-verify the capture frontier -- DROPPED.** It existed to decide whether bx7 was the
  end of the exposure path. gx7 settled that from the other direction: the path is retired, so
  there is no frontier to locate.

---

## Cross-cutting findings

1. **Room is not a screen for the gap.** Ranked by constant-coefficient room the four current
   economies run vyx, g0235, bx7, g28; ranked by realized gap they run vyx, g28, g0235, bx7.
   g28 has the second-largest gap and no room at all. The published grid's finding that
   corr(room, gap) turns negative once the two KP priced-vol rows are dropped is
   unverifiable (the rows are gone) but its direction holds on the current four.

2. **The decomposition realized gap = room x capture with capture at most one is wrong.**
   gap/room is 0.30 for vyx, 1.20 for g0235, 0.90 (all-month) or 1.17 (evaluation window) for
   bx7, and undefined for g28. Estimators refit on a rolling window beat a
   constant-coefficient rule when the conditional tangency moves; the room measures the
   wrong thing for that. What survives of REPORT.md's taxonomy (§16) is the three sources:
   estimation efficiency (g28, everywhere at +0.01 to +0.03), conditioning (g28, the
   regime-only rows), and cross-sectional curvature (g0235, vyx).

3. **The different-month confound was real, measured, and is NOT the explanation for gap
   exceeding room.** Until 2026-09-09 the oracle averaged over all 485 months and the
   estimators over the last 125, and the last 125 carry a higher attainable Sharpe on every
   panel (vyx 1.1531 to 1.1778, g28 0.2621 to 0.3087). The symptom was g28's DKKM exceeding
   the all-month SR_max, which Cauchy-Schwarz forbids within a month; against the
   evaluation-window SR_max the bound holds on all forty runs. Twenty oracle re-runs on
   2026-09-10 supplied the commensurable room for vyx and g0235, reproducing every all-month
   field bit-for-bit. **Correcting the month sample does not rescue the ceiling reading**: for
   g0235 gap/room goes from 1.20 to 1.28, and for bx7 from 0.90 to 1.17. Both move the wrong
   way. So the realized gap genuinely exceeds the constant-coefficient room, and the
   explanation is the rolling window, not the month sample. `WORKING.md` §49.

4. **Single seeds mislead.** g28's seed 0 gave +0.0119 against a ten-seed +0.0283; bx7's seed 0
   put its evaluation-window room above its all-month room and ten seeds reversed the order;
   vyx's seed 0 read "stronger than published" and ten seeds read "unchanged". Cross-seed sd
   of the gap is 15% of the mean for vyx, 37% for g0235, 48% for g28, 63% for bx7. No ranking
   below the top of a single-seed table means anything.

5. **Absolute and proportional gaps rank the economies differently.** By absolute gap the
   order is vyx (+0.1046), g28 (+0.0283), g0235 (+0.0227), bx7 (+0.0078); as a share of the
   linear Sharpe attained it is g0235 (26.5%), vyx (16.3%), g28 (10.5%), bx7 (2.7%). vyx has
   four times g0235's absolute gap because everything in vyx is large -- its linear methods
   already reach 0.64 where g0235's reach 0.086. g0235 is the economy where a complexity
   method most changes what an investor gets, vyx the one where it adds the most Sharpe. Room
   splits the same way: 54.3% of the linear Sharpe in vyx against 22.0% in g0235, and 0.0% in
   g28, which is the room-free economy stated on this scale.

6. **Two proposals, two falsified predictions, and one was pre-registered.** gx7's decisive
   negative was written into its spec before any solve ran, and it fired: exposure heterogeneity
   adds nothing in GS even under a state-dependent price of risk. g0235f's prediction failed on
   both halves, and the reason -- switching speed changes option values -- was visible in the J*
   table before a seed ran. The practical lesson for the next proposal: when a parameter enters
   the SOLVE, check whether the solved table moved before assuming the cross-section did not.

7. **The four gaps are three mechanisms.** vyx: a priced state bends heterogeneous exposures
   with stationary levels (curvature, capture 0.30). g0235: two regimes apply different
   transforms to the same project beta (curvature, capture above one). g28: a countercyclical
   price of risk on a flat cross-section (efficiency plus conditioning, no room). bx7:
   exposure heterogeneity under a regime, small room, small gap. The largest gap by a factor
   of four comes from the premium-side design, as REPORT.md §19d conjectured.

## Proposed next, ranked

Ranked by information per node-hour against the question this file exists for: where is the
proportional gap large, and why. Each is described in its model's section.

| rank | proposal | what it decides | new solves | cluster time |
|---|---|---|---|---|
| - | ~~G1 gamma(x) x exposure types~~ | DONE: pre-registered negative; the GS exposure path is retired | 4 solves, done | done |
| - | B1 regime persistence ladder | RUNNING: fast point in (prediction failed; post-hoc gap/room reading); slow and rare pending | done | 7 retries at 110-200 GiB, up to 20 h each |
| 1 | K1 continuum of exposures | whether vyx's linear methods live off three type points | 15 integrals, ~8 h | 28 h |
| 2 | X1 window ladder on g0235 | the rolling-window mechanism from the estimator side, at FIXED room -- which B1 turned out unable to hold | none; panels exist | 2 windows x 38 h |
| 3 | K3 kappa_y ladder | the room-versus-capture trade-off in the priced state | 2 x ~90 min | 2 x 28 h |
| 4 | G3 default-channel probe | whether GS's own nonlinearity can be made to fire; now the only live GS direction | 1 oracle probe, 20 min | decide after |
| 5 | B2 frontier ladder | multiplier versus dispersion | J* rebuilds | 2 points x 35 h, plus controls |
| 6 | K2 rare extreme type | the rank transform's tail compression | 3 integrals | 28 h |
| 7 | G4, K4 | wider gamma(x); one more gamma_v step | 1 to 3 solves each | 40 h each |

**X1 moved up.** B1 was meant to test the rolling-window mechanism by moving the switching speed
at constant room, and the fast point showed that switching speed moves room. X1 moves the WINDOW
instead, on panels that already exist, so the economy -- and its room -- is fixed by construction.
It is now the clean version of the test B1 was designed to be.

Two estimation-side items belong alongside whatever runs first:

- **X1. Window ladder on the existing panels.** All forty seeded panels and their moment files
  are on Sol. Re-scoring at windows of 240 and 480 (the oracle re-run at the matching
  `--eval_window`, about 1 h, plus estimators at about 2.7 h per seed) tests the rolling-window
  mechanism with no new economy: if gap/room above one is the window tracking a moving tangency,
  a shorter window raises DKKM's gap in g0235 and bx7 at the cost of noise, and a longer one
  lowers it toward the constant-coefficient room. Start with g0235.
- **X2. Ten more seeds for g28 and bx7.** Their cross-seed sd is 48% and 63% of the mean, so
  the GS ordering is the least certain thing in the current table. About 80 node-hours halves
  both standard errors. Not a parameterization, but the cheapest way to make the GS rows mean
  something before building on them.

**On pushing further.** vyx's type premia were already near 6, 15 and 28 percent a year under
the pre-fix labels. Every proposal that turns a dial up should report the annualised premia the
oracle prints (E[mu] and its cross-sectional sd are in every oracle JSON) next to the gap, so a
reader can see when an economy has left the empirically defensible range. The largest gap in a
laboratory economy is worth less than a moderate gap in one a referee will accept.

## Superseded and retired

- `var-kp_vy-vyx-v1`: the vyx parameters under the mis-normalised arrival rate (mean 1.72).
  Its published row is the vyx line of the pre-refactor grid. Numbers describe a different
  economy from v2; do not quote.
- `var-gs_bx-bx7-v1` and `-v2`: old calibration and the level-shift ladder; v2 was superseded
  before it ran. The retired manifest `a8ef7a2522eda19d` is the pre-kappa_e `sol_reg`.
- `var-gs_bx-g28-v1`: identical economy to v2; its precommitted solve id was invalidated by a
  comment-only edit to the solver source.
- The pre-refactor tables (`grid_summary.csv`, 24 economies; `oracle_summary.csv`, 57 rows
  over seven model families; `summary_grid.xlsx`) and the two scripts that maintained them
  were deleted on 2026-09-10. Nothing in them was reproducible, and for every economy still
  in use the ten-seed measurement strictly dominates the single-seed one -- for g0235,
  demonstrably the same economy. `WORKING.md` §50. Recoverable at `23f9380`.
- The unseeded run files that produced the three surviving grid rows
  (`kp_vy_oracle_vyx.json` and the rest, no seed suffix) went with them. They sat beside the
  seeded files distinguished only by that suffix, and two of the three described superseded
  economies: `kp_vy_oracle_vyx.json` carried SR_max 1.2607 where the current economy gives
  1.1945.

## Adding an experiment

1. Write `experiments/specs/var-<model>-<tag>-v1.json`: title, question, the parameter
   override, `env`, `estimation`, and `expected_solves` computed WITHOUT solving (the id is a
   hash of inputs). Say in economic terms what differs from the baseline, here and in the
   spec's title.
2. Solve with the model's producer (`rebuild_jstar_gam.py`, `build_vy_tables.py`,
   `gs_solve_reg.py` / `gs_solve_gam.py`); confirm the registry id matches the spec; publish
   the artifact with `variants/fetch_solves.py --publish`.
3. Add a `SEED_SPEC` case to `variants/run_seeds_slurm.sh` with the spec's parameters and
   kappa grid (`tests/test_specs_match_shell.py` pins it to the spec).
4. Run ONE seed at N=500, T=500 and read `sacct` before sizing an array. Then seeds 1-9.
5. `python variants/aggregate_seeds.py --flagship`, commit the result files and the tables,
   and add the economy here: to the current table, and as a subsection under the path it
   extends (or a new path), with what differs from the baseline in economic terms first.
   If it was one of the proposals, move it up out of "Proposed next" into its path and record
   what the result decided. `tests/test_results_md_matches_table.py` will fail until the
   table row is added.
