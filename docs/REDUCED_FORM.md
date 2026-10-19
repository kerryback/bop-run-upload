# The reduced-form screen

A simulated conditional factor economy whose true moments are known in closed form, run through
the project's own estimators, so that the question "what has to be true of an economy for DKKM to
beat a fair linear benchmark" can be asked in minutes per economy instead of hours of equilibrium
solving per type. Code: `variants/reduced/`. Written 2026-10-06 on Quest; nothing here is in a
results table and nothing here is on protocol. It is a screen.

## Why a screen, and why this one

Sixteen economies at protocol plus two on Quest say the following (docs/RESULTS.md, findings 8-17):

- The classical gap, DKKM minus max(FF, FM), is mostly two mechanical things: DKKM holds the EW
  market unpenalised while FF and FM estimate its weight in an unshrunk MVE (finding 8), and a
  shrunk ridge beats an unshrunk six-factor MVE wherever there is real cross-sectional signal
  (finding 10). Neither needs nonlinearity.
- The fair gap, DKKM minus a linear side given the market on the same terms, survives in two
  economies at about +0.015 and +0.008 of monthly Sharpe and nowhere else. Where it survives the
  economy has firm types with heterogeneous exposure to a priced state, a high price on that
  state's risk (finding 9), a state that moves fast relative to the 360-month window (finding 12),
  and enough firms per type (finding 16).
- `gap = room - excess estimation loss` is an identity, and across the file `excess_loss = 0.0024 +
  0.887 x room` at correlation 0.99. Room converts to gap at about 11%. Raising K (findings 15,
  16) and raising N (finding 17) raised room and not the gap.
- No measured statistic of the premium predicts which economy has a gap. The time-variation share
  of the linear map tracks room; nothing tracks the gap.

The one tool the record already had for population questions, `voc_diagnosis/scripts/headroom3.py`,
computed population Sharpes only. Its ceilings failed as predictors three times, and the reason is
the identity above: the gap is an estimation-variance phenomenon, so a screen without estimation
measures room, and room is not the prize. This screen therefore runs the real estimators
(`variants/common/dkkm_functions.py`, `fama_functions.py`), method by method as
`variants/run_estimators.py` does, on every simulated panel.

## The model

Monthly. N firms, T months after a burn-in. All symbols are in `variants/reduced/rf_model.py`.

**Aggregate state** s_t. Either an OU process with unit stationary variance and persistence
rho = 2^(-1/halflife), or a two-state Markov chain with switch probabilities p01, p10 (standardised
to mean zero, unit variance for the estimators). The estimators see s_t as `rf_stand`, exactly as
they see KP14's y, BGN's rate and GS21's productivity.

**Priced factors** f_{t+1}, K = 2, independent, sd sigma_f: a market/cash-flow shock and the
state's own innovation. So the state's risk is a priced factor, as KP14's y and GS21's x are. In the
regime variant the switch itself is unpriced, as BGN's is (finding 11's K accounting).

**Prices of risk** lambda_t = (lam_m, lam_s) x g(s_t), with g = 1 (KP14), a clipped linear slope in
the state (GS21's gamma(x)), or a regime multiplier gmult[s_t] (BGN's gmult). lam / sigma_f is the
factor's Sharpe.

**Firm type** tau_i in J types with shares pi, exposure ladder b_tau to the state factor
(KP14's type_bv, GS21's GS_BX_BETAS). Centred by default so the EW market carries only lam_m.

**Characteristics** X_it, five of them (size, bm, agr, roe, mom): persistent latent AR(1) draws
z_it with unit stationary variance, PLUS the type's footprint in value:

    size += r b~_tau (s~_t - s0) + lev b~_tau,      bm -= the same.

`r` is state-dependent visibility: at s = s0 the type is invisible in value and the ordering of
firms by type flips sign across the state. That is KP14's compensation term, `bv_comp`, which
"keeps firm value from revealing the type monotonically at y = 0". `lev` is level visibility: the
type shows in size at every date, which is GS21's ladder, and which a rank on size reveals.

**Loadings and premium.**

    beta_it = [ 1 + a . X~_it ,  b_tau u(s_t) ],   u(s) = 1 + u1 s~
    mu_it   = beta_it' lambda_t                      (exact factor pricing, mu = -cov(R, m))
    xret_it = mu_it + beta_it' f_{t+1} + eps_it,     eps ~ N(0, sig_eps^2)
    Sigma_t = B_t Omega B_t' + diag(sig_eps^2)

So the premium has an always-visible linear part, a . X on the market factor (the Fama-French
world), and a type part whose map from characteristics to premium is

    premium   = lam_s g(s) u(s) b_tau
    footprint = r b~_tau (s~ - s0) + lev b~_tau.

The regression coefficient of premium on footprint is lam_s g(s) u(s) / (r (s~ - s0)) up to the
level term: it flips sign across the state and is unbounded at s0. A rolling linear fit holds one
coefficient vector per 360-month window. The faster the state moves within the window, the less of
that map a window-constant linear fit carries, and the more a method that conditions on the state
has to gain. That is the mechanism findings 9, 12 and 16 point at, with nothing else in the way.

**What it leaves out on purpose.** No value function, so size and bm are not prices; no leverage,
default, investment or growth-option channel; no regime-dependent kernel renormalisation; one
priced state. Every one of those is a thing a structural model can refuse to deliver (BGN has no
add-a-type knob, GS21's ladder leaks into levels, a price of risk cannot switch every six weeks),
and that refusal is exactly what the screen cannot see. The workflow is: screen first, find the
region of loading-map space where the fair gap is robust, then ask which structural model can be
put inside it.

## What is measured

Definitions follow `variants/aggregate_seeds.py` so a screen row reads like an economy row.

| quantity | definition |
|---|---|
| SR_max | sqrt(mu' Sigma^-1 mu) by Woodbury, mean over the evaluation months |
| mkt_share | EW market Sharpe over SR_max |
| room | best const-theta nonlinear ceiling (rff, bins, poly2) minus lin_rank's, evaluation months, from `variants/common/oracle.py` |
| **room_s** | the same against **lin_rank_rf**, the linear basis given the state and its interactions. `room - room_s` is the share of the record's "room" that is linear in (characteristics, state) |
| theta, time_variation | `premium_shape.py`'s statistics on the true premia |
| dkkm | best Sharpe over rff / rff_ens, any P, any kappa, market unpenalised (`--include_mkt`) |
| classical_gap | dkkm - max(fm, ff) |
| fair_gap | dkkm - max(fm, ff, linrank, linrank_m, mkt_est), finding 8's benchmark |
| **fair_s_gap** | dkkm - max(the above, **linrank_s**): ranks, state, ranks x state, unpenalised market |
| excess_loss | room - fair_gap, the identity's other term |

**The two bold rows are the screen's own contribution, and they come from reading the estimator
code.** In `run_estimators.py` the conditioning column goes into DKKM's random features and into
no linear method: `lin_rank_weights` is the constant plus five ranked characteristics, and
`linrank_m` is the same with the market as a separate column. Meanwhile `room` is measured against
`lin_rank`, not `lin_rank_rf`, so it includes whatever is linear in (characteristics x state).
Finding 12's text says `linrank_m` "carries interactions"; the code says it does not. If the
surviving gaps are state-conditioning gaps rather than nonlinearity gaps, `fair_s_gap` will be
near zero where `fair_gap` is positive, and `room_s` will be near zero where `room` is positive.
The smoke test on the first reduced-form economy gave room +0.0195 and room_s +0.0005. That is one
seed of a toy and proves nothing; it is the reason the two rows exist.

## Validation before exploration

The screen is trusted only where it reproduces what the structural economies measured. Each row is
a config in `variants/reduced/rf_configs.py`; the expected sign is from the record.

| config | moves | expected, and where it comes from |
|---|---|---|
| notype | J = 1 | room ~ 0, fair gap ~ 0 (kpbase +0.004, gsbase +0.001 room) |
| invisible | r = 0, lev = 0 | room ~ 0: the ceilings are over characteristic bases and the type leaves none |
| levels | r = 0, lev = 1 | fair gap ~ 0: a rank on size reveals the ladder (clause iii, gx7, bx7) |
| hl120 ... hl6 | OU half-life 120 to 6 months | fair gap rising with speed (finding 12: -0.004, +0.003, +0.008) |
| ps02 ... ps075 | lam_s / sigma_f 0.25 to 0.75 | fair gap rising with the price (finding 9: +0.001 at 1.8, +0.015 at 2.5) |
| bigmkt | lam_m up, lam_s down | large classical gap, fair gap ~ 0 (GS21: market at 94-97% of SR_max) |
| dense20 | J = 20 at N = 500 | fair gap below base (finding 16: 25 vs 167 firms per type) |
| n1000 | N = 1000 | room up, gap flat (finding 17: slope 0.968 along N) |
| reg_slow / reg_mid / reg_fast | BGN regime on the price, x0.2 / x1 / x4 | fair gap monotone in speed (finding 12) |

Across all seed-rows, `excess_loss` against `room` should come back with a slope near 0.89 and a
correlation near 0.99. If it does not, the screen's estimation variance is on a different scale
from the structural economies' and its gap levels are not comparable, only its orderings.

## Running it

    # one economy, one seed, reduced grid (rff 36,360; one feature draw; protocol kappas)
    python variants/reduced/rf_screen.py --config base --seed 0 --out /kellogg/proj/axa3203/bop_runs/screen
    # the validation ladder on Quest, five seeds each, six at a time
    sbatch --export=ALL,CONFIGS=notype,invisible,levels,hl120,hl48,hl24,hl12,hl6,NSEEDS=5 --array=0-39%6 variants/klc/screen_klc.sh
    # read it
    python variants/reduced/rf_table.py --dir /kellogg/proj/axa3203/bop_runs/screen --ladder speed

`--full` restores the protocol grid (rff 36,360,3600, two draws) for a run meant to be compared
level for level with a structural economy. `--stride k` scores every k-th evaluation month for a
coarser, faster screen. Ad-hoc economies take `--params '{"halflife": 9, "r": 2}'`.

## What the first evening found

**The state-conditioning hypothesis is false for KP14 and partly true for BGN and GS21.** Read
off every oracle JSON on disk (the 160 committed seeds plus the two Quest economies), the share of
`room` that is linear in characteristics x state, `1 - room_s / room`:

| economy | room | state share of room |
|---|---|---|
| kp_vy vyg30k70, vym3t3, vyg25, vym3, vyx | 0.055 to 0.102 | 2% to 4% |
| bgn_gam g0235f | 0.045 | 25% |
| bgn_gam g0235df, g0235d | 0.036, 0.031 | 47%, 68% |
| bgn_gam g0235s, g0235r | 0.007, 0.005 | 82%, 79% |
| gs_bx gx7 | 0.008 | 91% |

And `rescore_state.py` on the saved Quest panels: vyg30k70's fair gap goes +0.0161 to +0.0149 over
five seeds when `linrank_s` joins the benchmark; g0235df's -0.0051 to -0.0060. So KP14's room and
gap are genuinely nonlinear in the ranks GIVEN the state, and that is what makes KP14 the model
that works. BGN's and GS21's "room" is mostly a window-constant linear map failing to track a
state-dependent slope, which a linear method given the state would carry.

**Why KP14, in the reduced form's terms.** A type-f claim on exp(b_f y) has log value convex in b
with a compensation offset, so at y near zero the extreme types look alike in value while earning
opposite premia, and as y moves the ordering rotates. In the model above that is `lev` (persistent
monotone footprint) plus `q` (curvature) plus a modest `r`. On the toy at N=300: with lev 1 and
r 0, room is +0.003 at q 0, +0.005 at q 0.5, +0.023 at q 1, and room_s equals room throughout;
with r 1 and no lev or q (the original default), room is +0.12 and room_s +0.007, with Fama-MacBeth
negative. The base economy was therefore recentred on lev 1, q 1, r 0.5. BGN cannot run this
(no type knob; its premium is affine in book-to-price and 1/price by proposition) and GS21 cannot
convert it (K = 1, ceiling 1.000), which is why the screen is where the mechanism gets tested.

## Validation ladder, seeds 0 to 4 (2026-10-07, 00:08)

Reduced grid (rff 36 and 360, one draw), so DKKM is understated by roughly 0.01 against the
protocol grid (vyg30k70's rff 3600 and ensemble add that much); signs within 0.01 of zero are
provisional until the `fullcheck` array reports. Fair gap and the state-aware fair gap are
five-seed means with (t, seeds positive).

| config | room | room_s | fair gap | fair_s gap | expected |
|---|---|---|---|---|---|
| notype | 0.000 | 0.000 | +0.0003 | +0.0001 | zero: held |
| levels_only | +0.002 | +0.002 | -0.024 (-2.5, 0) | -0.024 | linear takes it: held |
| bigmkt (market 91% of SR_max) | +0.003 | +0.003 | -0.001 | -0.002 | classical +0.010, fair zero: held |
| speed hl120 -> hl6 | +0.016 -> +0.019 | | -0.018 -> -0.006 | | rising with speed: held in sign, never positive |
| price 0.25 -> 0.75 | +0.009 -> +0.026 | | -0.018 -> -0.010 | | rising with price: held in sign, never positive |
| dense20 vs base | +0.018 | | -0.022 vs -0.017 | | thinner types worse: held, weakly |
| n1000 vs base | +0.011 vs +0.018 | | -0.012 vs -0.017 | | room FELL; the structural record had it rise 61% |
| curvature q0 -> q2 | +0.006 -> **+0.070** | +0.002 -> +0.070 | -0.022 -> **+0.017 (2.2, 4)** | same | the one knob that makes a gap at base settings |
| state r0 -> r2 | +0.022 -> +0.073 | +0.022 -> +0.021 | -0.019 -> **+0.106** | -0.019 -> +0.015 (1.4, 2) | the BGN/GS signature: a state-aware linear side takes 86% |
| state_only (lev 0, q 0, r 1) | +0.246 | +0.015 | +0.221 (7.2, 5) | +0.028 (2.7, 5) | same, at the extreme |
| regime x curvature, slow / mid / fast | +0.047 / +0.060 / +0.055 | same | +0.020 / +0.011 / +0.012 | +0.019 / +0.008 / +0.008 | positive, noisy (t 0.6 to 1.3), not monotone |
| invisible | +0.046 | +0.043 | -0.006 | | expected zero room; see below |

**What it says.** Three of the structural record's patterns reproduce in sign: no types means no
room, a footprint visible in levels is taken by the linear side, a dominant market gives a
classical gap and no fair gap. Speed and price move the fair gap in the recorded direction but
from a base that is too small to cross zero. The knob that crosses zero at base settings is
curvature in value, and it converts: room +0.006 to +0.070 moved the fair gap by +0.039, a slope
of 0.6 against the record's 0.11. State dependence of the footprint makes a large fair gap that is
mostly a window-constant linear fit failing, which a linear side given the state recovers, exactly
as the BGN and GS21 decomposition said it would.

**Two caveats the ladder exposed.** The `invisible` row has room +0.046 where the type leaves no
footprint at all. With fixed types and characteristics persistent at 0.99, a fine basis (decile
cells, 360 random features) can hold a fixed portfolio that memorises which sample firms carry
which type, so the population ceiling is partly a finite-sample object. That N=1000 lowered the
toy's room is consistent with it. The structural economies also have fixed types and persistent
characteristics, and their room ROSE with N, so the two are not the same effect; it is a reason to
report `room` at two N before reading it as headroom. Second, the toy's premium is far more
nonlinear within a month (theta 0.5) than KP14's (0.19) and varies far less across months
(time-variation 0.01 against 0.10). The calibration that matches KP14 is still open and is what
the grid and cross are for.

## Validation ladder at ten seeds, and the cross (2026-10-07, 10:30)

Same reduced grid. Ten seeds per config (`u1` five). Fair gap and the state-aware fair gap are
means with (t, seeds positive). Read with `rf_table.py --ladder <name>`.

| config | room | room_s | fair gap | fair_s gap | at 5 seeds it read |
|---|---|---|---|---|---|
| notype | 0.000 | 0.000 | +0.0009 (1.2, 7) | +0.0008 | zero: held |
| levels_only | +0.002 | +0.002 | -0.0232 (-3.6, 0) | -0.0233 | held |
| bigmkt (market 91%) | +0.005 | +0.004 | +0.0002 (0.2, 6) | -0.0005 | classical +0.011: held |
| speed hl120 -> hl6 | +0.029 -> +0.029 | | -0.0057 -> +0.0006 | same | direction held, never positive |
| price ps02 -> ps075 | +0.017 -> +0.056 | | -0.0093 -> +0.0110 (0.7, 5) | +0.0102 | direction held, not significant |
| dense20 vs base | +0.032 vs +0.037 | | -0.0088 vs -0.0023 | | thinner types worse: held |
| n1000 vs base | +0.034 vs +0.037 | | +0.0012 vs -0.0023 | | room FLAT along N (was "fell" at 5 seeds) |
| curvature q0 -> q2 | +0.024 -> **+0.088** | +0.018 -> +0.082 | -0.0054 -> **+0.0290 (4.2, 9)** | same | the knob: +0.017 at 5 seeds, firmer now |
| state r0 -> r2 | +0.021 -> +0.118 | +0.021 -> +0.030 | -0.0168 -> +0.1079 (3.3, 10) | -0.0168 -> +0.0153 (1.9, 6) | a state-aware linear side takes 86% |
| state_only | +0.184 | +0.020 | +0.1850 (6.0, 10) | +0.0299 (3.9, 9) | same, 84% taken |
| s0 = 1 (type invisible off-centre) | +0.066 | +0.042 | +0.0227 (1.4, 6) | +0.0055 (1.1, 6) | mostly linear-given-state |
| u1 = 0.5 (rotating slope), 5 seeds | +0.017 | +0.016 | -0.0103 (-2.0, 0) | same | a rotating loading does NOT help |
| regime slow / mid / fast | +0.082 / +0.060 / +0.053 | same | **+0.0400 (1.9, 6) / +0.0205 (2.0, 6) / +0.0082 (1.4, 7)** | +0.0385 / +0.0144 / +0.0065 | REVERSED vs finding 12; SR_max 0.70 / 0.58 / 0.49 not held fixed |
| noise eps05 / eps20 | +0.035 / +0.034 | | +0.0027 / -0.0055 | | less idiosyncratic noise helps, weakly |
| invisible | +0.047 | +0.044 | -0.0030 (-0.9, 3) | | room +0.047 with no footprint; 5 of 10 seeds at the kappa grid edge |

**The cross** (`@cross`: half-life 6/12/24/48/120 x price 0.10/0.15/0.20/0.30 x curvature q 0/1/2,
three seeds, 60 cells). Every cell with a positive fair gap has q = 2, and in every one of those
`room_s` equals `room`, so the gap is nonlinear in the ranks GIVEN the state, KP14's signature.
At q = 2 the gap scales with the price of the state's risk and is flat in the half-life:

| q = 2, fair gap (3-seed mean) | h6 | h12 | h24 | h48 | h120 |
|---|---|---|---|---|---|
| p10 | +0.007 | +0.009 | -0.005 | -0.002 | +0.011 |
| p15 | +0.016 | +0.018 | +0.003 | +0.005 | +0.019 |
| p20 | +0.027 | +0.030 | +0.014 | +0.016 | +0.030 |
| p30 | +0.050 | +0.053 | +0.036 | +0.038 | +0.054 |

At q = 1 the best cell is +0.003 and at q = 0 every cell is negative. Time-variation of the premium
is 0.01 to 0.02 in every q = 2 cell against KP14's 0.10, so the toy's gap is made by a STATIC
curvature in the value footprint, and the state's speed is not what converts it. That is the
opposite of what the structural speed ladder (finding 12) says and the same as what the cross-economy
`room_s` decomposition says: KP14's room is rank-nonlinear given the state, BGN's and GS21's is not.

**The one-seed footprint grid** (`@grid`: lev 0.5/1/2 x q 0/0.5/1/2 x r 0/0.3/0.6/1). The best
cells are lev 0.5 to 1, q 2, r 0 to 0.3 (+0.020 to +0.036 fair_s); lev 2 kills the gap at every q
and r (the level footprint is taken by a rank on size). Seeds 1 to 4 are queued.

**Caveat on levels.** Across 513 seed-rows `excess_loss = 0.017 + 0.36 x room` at correlation
0.58, against the record's 0.0024 + 0.887 x room at 0.99. The screen's estimation loss is on a
different scale from the structural economies', so its gap LEVELS are not comparable to an economy
row, only its orderings and signs. The `fullcheck` array (protocol grid, seven configs, three seeds)
is the level-for-level check.

## The grid at five seeds, round two, and the protocol-grid check (2026-10-07, 18:30)

Every array is in. What follows is what the screen says once all of it is read together, and
then the tables it is read from. Nothing here is a level to quote against an economy row; the
protocol-grid check at the end says how far off the levels are.

**The one lever.** Across the 10-seed validation ladder, the 60-cell cross, the 48-cell footprint
grid at five seeds and the round-two configs below, a positive fair gap appears in exactly one
situation: the premium is CURVED in the ranked characteristics (q = 2), and then its size scales
with the price of the curved component. Nothing else produces one, and most other knobs subtract:

| knob | moved | fair gap | reading |
|---|---|---|---|
| curvature q | 0 -> 2 (10 seeds) | -0.005 -> **+0.029 (4.2, 9)** | the lever |
| price of the curved component | 0.10 -> 0.30 at q = 2 (cross) | +0.01 -> +0.05 | linear in the price |
| OU half-life | 6 -> 120 months at q = 2 (cross) | flat | speed does nothing |
| regime speed x q | x0.2 / x1 / x4, q 0 / 1 / 2 (3 seeds) | q2: +0.067 / +0.036 / +0.039; q <= 1: all <= +0.000 | q again; regime timing nothing |
| level footprint lev | 0.5 / 1 / 2 at q = 2 (grid) | best +0.037 / +0.023 / **-0.021** | lev 2 kills it: a rank on size takes a steep level sort |
| state r | 0 -> 1 at lev 0.5, q 2 (grid, fair_s) | +0.014 / +0.026 / +0.037 / +0.025 | a modest r helps fair_s; large r is taken by `linrank_s` |
| market share | 5% -> 66% (`mkt1`..`mkt16`) | -0.016 -> -0.007, room 0.019 -> 0.014 | keep the market small; it never helps |
| type density J | 2 / 3 / 5 / 10 / 20 | -0.016 / -0.001 / -0.020 / **-0.024 (-2.7)** / -0.009 | more types subtracts (finding 16, sharper) |
| KP-style exposure ladder | b 0.2 / 0.7 / 1.4 (`kpladder`) | **-0.013 (-4.3, 0)** | spacing the exposures does not help |
| noise dispersion | sd of idiosyncratic vol 0.4 (`epsdisp`) | -0.015 | subtracts |

In every positive cell `room_s` equals `room` to the third decimal, and the premium's
time-variation share sits at 0.01 to 0.02 against KP14's 0.10. So the gap the screen makes is a
STATIC rank-nonlinearity, and the state's only job is to keep that nonlinearity from being spanned
by the market. That is the same thing the measured footprint of `vyg30k70` says about KP14
(`docs/NEXTUP.md`, 2026-10-07 addendum): type is a level sort in size, ranks x state adds nothing
to the premium's R^2, and within type corr(mu, bm) is 0.70.

**What this says about the structural queue.** `kappa_y` (K3, running), regime speed, a continuum of
types (K1) and the shape of the exposure ladder are not levers; the screen struck all four. The
price of the curved component is the lever, and in KP14 the curved component is the growth-option
share within a type, whose premium is set by `gamma_z` and `gamma_x`, two prices the project has
never moved. The oracle-only room screen on `vyg25` (`variants/klc/kp_room_klc.sh`, six points) is
the screen's recommendation carried into the structural model.

**The footprint grid at five seeds** (`@grid`, lev 0.5 / 1 / 2 x q 0 / 0.5 / 1 / 2 x r 0 / 0.3 / 0.6
/ 1, `fair_s` with (t, seeds positive)):

| lev \ r | 0 | 0.3 | 0.6 | 1 |
|---|---|---|---|---|
| 0.5, q 2 | +0.014 (1.0, 4) | +0.026 (2.7, 4) | **+0.037 (4.0, 5)** | +0.025 (2.0, 4); fair +0.102 |
| 1, q 2 | +0.023 (2.7, 5) | +0.021 (1.9, 4) | +0.016 (2.3, 4) | +0.014 (1.6, 4) |
| 2, q 2 | -0.021 (-2.4, 1) | -0.015 | -0.013 | -0.013 |
| any lev, q <= 1 | negative in 30 of 36 cells; best +0.014 (lev 0.5, q 1, r 1, fair +0.086) | | | |

`lev 0.5, q 2, r 1` is the clearest `linrank_s` case in the file: fair gap +0.102 against a
state-blind benchmark, +0.025 once the benchmark sees the state. The paper's protocol is
state-blind for every linear method (only DKKM's features see the conditioning column in
`run_estimators.py`), so this is the size of the overstatement a referee would find.

**Round two** (3 seeds each). Regime x curvature:

| | q 0 | q 1 | q 2 |
|---|---|---|---|
| regime x0.2 (slow) | -0.012 | +0.000 | +0.067 (0.9, 2) |
| regime x1 (mid) | -0.013 | -0.009 | +0.036 (1.8, 3) |
| regime x4 (fast) | -0.009 | -0.003 | +0.039 (5.5, 3) |

Room is 0.000 at q 0, 0.024 at q 1 and 0.100 at q 2 whatever the speed; the q 2 column's
time-variation share runs 0.014 / 0.21 / 0.22 and the gap does not follow it. Market share
(`lam_m` 0.001 / base / 0.004 / 0.008 / 0.016, market 5 / 11 / 21 / 40 / 66% of SR_max): fair gap
-0.016 / -0.001 / -0.013 / -0.011 / -0.007, room 0.019 / 0.038 / 0.019 / 0.018 / 0.014 (base is at
ten seeds, the others at three, which is why base sits above its neighbours). Density: in the table
above. `kpladder` and `epsdisp`: in the table above.

**The protocol-grid check** (`fullcheck`: base, q0, q2, state_only, curvature_only, hl6, ps075 at
`--full`, rff 36 / 360 / 3600 with two feature draws, three seeds each, about 95 minutes a seed
against 2.3 for the reduced grid). Paired on identical seeds, the protocol grid gives:

| full minus reduced, 21 pairs | mean | sd | range |
|---|---|---|---|
| DKKM | +0.0030 | 0.0029 | -0.0054 to +0.0079 |
| fair gap | +0.0030 | | sign agrees in 19 of 21 |
| room (oracle, rff3600 added to the ceiling set) | +0.0042 | | |

Three-seed means, reduced / full: base -0.014 / -0.010, q0 -0.020 / -0.017, q2 +0.014 / +0.016,
curvature_only -0.017 / -0.014, hl6 -0.002 / +0.000, ps075 -0.007 / -0.003, state_only's `fair_s`
+0.013 / +0.017. So the reduced grid understates DKKM by about 0.003, a third of what this file
assumed when it was written, and every ordering and every conclusion above survives the protocol
grid. The two sign flips are seeds whose fair gap was within 0.005 of zero either way.

**Caveat on levels, updated.** Over 764 seed-rows `excess_loss = 0.018 + 0.40 x room` at
correlation 0.55, against the record's 0.0024 + 0.887 x room at 0.99. The screen's estimation loss
is still on a different scale from the structural economies', so a screen gap is an ordering, not
a prediction of an economy row's gap. The structural conversion that matters is KP14's own:
`room_s` to fair gap at 0.16 to 0.18 in `vyg25` and `vyg30k70`.

**A file hazard, fixed.** `rf_screen.py --full` wrote under the same stem as the reduced run of the
same config, so the fullcheck array overwrote the 21 reduced-grid records of its seven configs.
The overwritten files were protocol-grid runs (verified from their recorded `rff` and `nmat`) and
were renamed to `screen_<config>_full_s<seed>` with the record's name field rewritten; the 21
reduced seeds were re-run (job 9055069, 2026-10-07 18:13 to 18:30). `--full` now writes
`<name>_full`, and `rf_table.py --ladder fullcheck_pairs` prints the pairs side by side.

## Status

2026-10-06 evening. Smoke-tested; one full-size seed of the first base took 137 s on a 4-core
kellogg task (oracle 76 s, estimators 60 s). Submitted: the validation ladder (29 configs x 5
seeds, `@validation`) and the one-seed footprint grid (48 cells, `@grid`), to
`/kellogg/proj/axa3203/bop_runs/screen`. No number from the screen is to be quoted until the
ladder's expected signs are checked.

2026-10-07 10:30. Ladder at ten seeds and the 60-cell cross are in (section above). Running: grid
seeds 1 to 4 (192 tasks), round2 (regime x q, market, density, kpladder; 57 tasks), and
`fullcheck` (21 tasks, about 90 min each). The dead gx7w re-solve chain is unrelated to the screen.

2026-10-07 18:30. Everything submitted on 2026-10-06 and 2026-10-07 is complete and read (section
above): validation at ten seeds, the cross, the grid at five seeds, round two, the protocol-grid
check, and the reduced re-run after the stem fix. The screen's conclusion is one lever, the price of
a static rank-curvature; its structural follow-ups are K3 (running, `kappa_y`, predicted flat) and the
six-point oracle-only room screen on `vyg25` (`variants/klc/kp_room_klc.sh`, job 9055168), read with
`variants/klc/kp_room_table.py`.
