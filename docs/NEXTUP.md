# Next up

Rewritten 2026-09-23, folding in the disaster-shock plan that was
`docs/plan-before-home-20260917.md` (deleted in the same commit; `git log` has it). What the last campaign decided,
then the ranked queue, then the design work behind the disaster items, then what is not worth running.

## What the protocol-v3 campaign decided

The campaign is **done** (`docs/RUNS.md`, "The campaigns"; results in `docs/RESULTS.md`). All 130 tasks
COMPLETED, nothing failed, and the answers reshape what is worth running next.


1. **The `vyx` headline is withdrawn.** Its registered prediction was that the fair gap would land
   between +0.01 and +0.06 and stay positive at ten seeds, FALSIFIED below +0.01. It came back
   **+0.0009**, positive in 7 of 10 seeds, t 0.3. The pricing fix removed 99% of it.
2. **`vyg25` survived, and it is now the project's only strong result.** +0.0148, positive in 10 of
   10 seeds, t 5.2. `bgn_gam/g0235f` is second at +0.0079, also 10 of 10, t 4.8. Every other economy
   is between -0.0038 and +0.0025.
3. **The calibration objection died with the gap.** `vyx` and `vyg25` sat at 18.2% and 22.9% oracle
   expected excess return a year under the mispriced solve; corrected, they sit at 4.2% and 3.9%,
   inside the 3.3%-to-12.6% band every other economy occupies. **That answers K6**, which existed to
   find a defensible calibration: this IS one, and the gap at it is +0.0148 at best.
4. **The ridge-grid amendment was worth running and found nothing at the ceiling.** The floor side
   worked -- `vyx` and `vyg25` went from 7 and 6 seeds at the floor to none -- but two extra decades
   above the old ceiling moved DKKM by at most +0.0001 in the four `gs_bx` economies, where the
   effect can be isolated. The gate's criterion was wrong and has been amended; see below.

### The gate, amended

`variants/penalty_gate.py` asked only whether the winning penalty was interior. Three rows failed
that (`g0235s` 6/10, `bx7` 7/10, `gx7` 5/10) and none is censored: as `kappa` grows, the ridge
direction `(X'X + kI)^-1 X'y -> X'y / k` stops depending on `kappa`, and Sharpe is scale-invariant,
so the curve has a horizontal asymptote. All 16 ceiling-winning seeds in the campaign gained at most
2.7e-05 over their best interior penalty. The gate now tests materiality as well as position and
`tests/test_penalty_gate.py` pins both halves. Reading it literally would have bought a third decade
and another ~990 node-hours to move three numbers in the fifth decimal.

### Still open, unclaimed by any of the above

- `zero_book_in_sdf_solve: true` is declared in all thirteen specs and **read by nothing** in
  `variants/` or `tests/` -- the same shape as the `burnin` field that silently said 200 while the
  code ran 400. `variants/kp_vy/sdf_compute_kp14.py` still solves `ER` over all N firms with a ridge
  fallback that fires only on an exception, which is the construction that produces unreliable
  `sdf_ret` / `max_sr` -- and `max_sr` is `docs/RESULTS.md`'s `SR_max` column.
- `PRECISION_KEYS["kp"]` is still `("NY", "_i0")` and does not record the internal Q-grid span or
  subdivision, which the corrected solve introduced. Close it before another KP14 economy is added,
  which K7 would be.

## The queue

Ranked on the corrected evidence. Costs are measured, not projected, unless marked.

1. **Decide the BGN regime closure. No compute.** That path was closed on five ten-seed points
   "whose fair gap never left -0.008 to +0.004". Under corrected pricing the top of that band is
   `g0235f` at **+0.0079, positive in ten seeds of ten at t 4.8** -- more than double the old top, in
   an economy with no exposure heterogeneity at all. No spec's +0.01 falsification threshold was
   crossed, so nothing fired automatically, and no new economy is needed to ask the question: the
   five points are already measured. It comes first because it changes what everything below is
   testing -- whether the gap is a KP14 phenomenon or a small general one.
2. **K7, a third point in `gamma_v`.** `vyx`'s parameters at `gamma_v` 2.1 or 2.2, everything else
   identical. One G solve plus one set of integrals is minutes on the Mac -- the 2026-09-21 rebuild
   did three economies, six solves and 154 tables in about 20 minutes -- then ten Phoenix seeds of
   ~6 h, about 60 node-hours. Decides whether the gap is a threshold in the price of the state's risk
   or smooth in it. Register the falsification clause first.
3. **`gs_bx/gsdis`, GS21 capital destruction.** Promoted above `bgnzr` on 2026-09-23: it is the only
   disaster design whose premise the corrected campaign CONFIRMED rather than disturbed -- no `gs_bx`
   population quantity moved, to four decimals -- it re-keys nothing, and it is the cheapest entry.
   New module `gs_solve_dis.py`, a parity test beside `tests/test_gs_solver_parity.py`, a precommitted
   solve id, one solve, ten seeds at 32G.
4. **`bgn_gam/bgnzr`, rotate BGN's price of risk onto the rate channel.** Not a disaster: a pure
   parameter override, one 3-minute J\* rebuild, ten seeds at 40G, zero code and zero re-key. Still
   the cheapest untried lever, and still the precondition for a disaster in `r` -- but demoted below
   `gsdis` because all its reference numbers moved and its live region now looks
   calibration-strained. Re-anchor the prediction and restate the falsification line before running
   it. Its value is as a GATE on whether the rate channel carries enough premium, stated on the rate
   channel's share of SR_max, not on the gap.
5. **K1, a continuum of exposures.** Fifteen types over [0, 0.14], shares right-skewed. Much cheaper
   than this file used to say: the integral stage is ~20 min for three economies, so fifteen types is
   roughly 1.7 h, then 30 h of seeds. The case strengthened with the correction -- at `vyx` the
   linear side now reaches 95.3% of its ceiling, so what is left is a feature-basis question rather
   than a shrinkage one.
6. **K3, persistence of the priced state.** `kappa_y` 0.15 and 0.70, two solves of about 15 min each.
   More interesting after the fix: the corrected Girsanov adjustment SATURATES at
   `b gamma_v sigma_y / kappa_y`, so `kappa_y` now scales the whole correction rather than only the
   state's timing. Note `sigma_y` is locked to `sqrt(2 kappa_y)`, so this moves persistence and
   innovation size together.
7. **`kp_vy/vydis`, a jump in `y` -- on `vyg25`, not `vyx`.** Cheap now that the re-solve is done:
   `Qy` lives in `parameters_kp14.py`, so editing it re-keys all six KP14 solve ids, but at disaster
   rate 0 the tables must come back byte-identical, making it a rebuild-and-verify. Expect `y` to buy
   dispersion and not drama.
8. **G3, the GS21 default-channel probe.** Twenty minutes, oracle only. Largely subsumed by `gsdis`,
   which is G3 given a mechanism strong enough to reach the boundary. Its gate as written -- the
   market below 85% of SR_max -- is too weak: the KP14 baseline meets it at 35% and has no gap. It
   needs an evaluation-window room clause of at least +0.05 as well.
9. **`bgn_gam/bgndis`, a disaster in `r`. Conditional on two gates**: `bgnzr` showing the rate channel
   is alive, and a derivation showing the closed forms survive. A compensated Poisson jump keeps the
   Vasicek affine, but the OPTION value breaks -- `norm.cdf(d1)`, `norm.cdf(d2)` need `r` at the
   exercise date Gaussian, and Gaussian-plus-decayed-jumps is not a finite Gaussian mixture. One
   escape route worth thirty minutes of algebra first: if the disaster's `r` is deep enough that
   acceptance is numerically zero there is no option exercise in the disaster state, so the
   disaster-state option value is the discounted normal-regime `J*` at a geometrically distributed
   exit date -- machinery already in `vasicek.py`. If that fails, this is a new BGN solver and should
   not be paid for.
10. **K2, a rare extreme type.** Shares (0.45, 0.45, 0.10), top loading 0.14 to 0.20. Low priority;
    the discount check that withdrew K5 applies at 0.20.

**~~K6, the low end of the price ladder~~ -- ANSWERED, not by an economy.** K6 existed because `vyx`
and `vyg25` sat at 18.2% and 22.9% oracle expected excess return a year and no referee would accept
them. The pricing fix put them at 4.2% and 3.9% without changing a parameter, so the defensible
calibration is the one already measured and the gap at it is +0.0009 and +0.0148. Lowering `gamma_v`
would lower the gap, which is why K7 goes UP from 1.8 rather than down.

Each new economy is a spec with a registered prediction, a precommitted solve id and a falsification
line, per `docs/RESULTS.md` "Adding an experiment". None of this changes the protocol: a disaster is
a parameter and a driving force, which is what a new economy is allowed to be.

## The disaster-shock programme

Folded in on 2026-09-23 from `docs/plan-before-home-20260917.md`, deleted in the same commit,
and pared to what is measured.
**No economy here has been run**; `gsdis`, `bgnzr`, `vydis` and `bgndis` are proposed tags. The
plan's step 0 -- settle the KP14 `y` risk adjustment before any disaster work -- is CLOSED by the
protocol-v3 campaign, so the gate is lifted.

### Why a disaster at all

Every economy in `docs/RESULTS.md` varies a price of risk or an exposure ladder. **None varies the
level of output.** A rare disaster is the first proposal that would, and the first that could break
the two structural bounds in `docs/RESULTS.md`'s capability table, because those bounds are
statements about LOCAL exposure: a diffusion shock sees only the first derivative of the value
function, which each paper summarises with one firm variable, while a large jump sees the function
over a wide range and that non-local response can differ across firms in ways the variable does not
capture.

### Three facts that constrain every design

**1. Hold the VALUE drop constant across models, not the output drop.** A transitory output loss of
duration D transmits to value by roughly `D / (D + asset duration)`, and asset duration differs by
model: GS21's equity elasticity to a transitory `x` drop is 0.017 at two months and 0.25 at four
years; BGN's projects have effective duration ~62 months, so a 24-month disaster transmits ~28%. So
"output drops 33%" means a 0.6% value drop in one model and 33% in another. The calibration held
fixed is hazard `p = 1/480` per month (2.5%/yr, once in 40 years) and a **25% aggregate equity value
loss** on arrival; report the implied output drop per model.

**2. The realized channel is nearly unmeasurable at this protocol, and the price channel is not.**
Over 900 simulated months E[disasters] = 1.9, but over the 125 EVALUATED months E = 0.26, so about
**2.3 seeds of ten** ever see one. The hazard is on in every month, however, and `room` and `SR_max`
come from true conditional moments, so they are measured at full power with zero realizations.
**The primary outcome must therefore be `room` and the fair gap**; the realized gap is secondary and
is reported split by whether a disaster arrived. Pre-register that split and record per seed the
disaster count in the evaluation window.

**3. Rarity is not the content; the LOADING MAP is.** The pipeline scores everything by `mu_t` and
`Sigma_t`. To it a disaster is one extra priced factor with a specific loading map plus fat-tailed
realizations; its skewness is invisible. A disaster with a COMMON exposure is a pure market shock,
and since a disaster is an aggregate shock it adds to the market's own Sharpe -- pushing against the
condition that the market must not already span the economy. It helps only through cross-sectional
dispersion in exposure.

### Where each model can host one

**GS21 -- capital destruction. The strongest mechanism and the cheapest entry.** The value function
is `(znum, xnum, bnum)` with no capital dimension: the model is homogeneous of degree 1 in `k` and
`bgrid` is debt PER UNIT of capital, so destroying a fraction `kappa_D` of the stock while nominal
debt is untouched is exactly `b -> b/(1 - kappa_D)` -- an operation the code already performs in the
other direction with its interpolation weights precomputed (`gs_solve_reg.py:179`,
`gs_sim_bx.py:205`). Output falls one-for-one and permanently; exposure is violently non-affine,
because every firm's `b` rises by the same proportion but the consequence runs through
`alive = (Pn + mshock) > 0`, so high-`b` low-`z` firms default and low-`b` firms barely notice; and
it populates the default boundary, which is the explicit escape clause in GS21's own structural
bound. It re-keys nothing: a new `gs_solve_dis.py` is the sanctioned pattern, as `gs_solve_gam.py`
already is. Open items: the `b` grid must extend above 1.0, and the arrival is an unpriced regime
switch so it needs pricing. Cost: one solve at 3.5-5.9 h, then ten seeds in the campaign's cheapest
class (32G; the four `gs_bx` economies peaked at 3.3-4.3 GiB in 2026-09-21).

*Two GS21 designs were rejected on measurement.* Appending a disaster node to the Tauchen `x` grid
fails because the kernel multiplies `gamma` by the NORMALISED innovation and `sigma_x = 0.007046`:
computed on the committed `sol_gsbase` kernel, a node at `x_d = -0.10` is -14.2 innovation sd and
takes **69%** of the kernel mass after renormalisation, and `x_d = -0.20` takes **99.96%**. The
disaster's price would be set by the grid, not by a parameter. A regime-indexed output level is
cheap but would not bite: a short `x` disaster moves equity 0.5-2.7% and default probability averages
0.002% at the solved leverage.

**KP14 -- the obvious candidate is provably null; `y` is the one worth running.** Re-verified on live
source 2026-09-23: `book`, `VAP` and `PVGO` all carry `x*ebv` linearly and `z^{alpha/(1-alpha)}`
(`panel_functions_kp14.py:162,183,196`), so price and book scale identically, every ratio
characteristic is invariant, and a jump in `x` or `z` gives a return constant across firms. It is
perfectly spanned by the market and **shrinks every gap in the model**. Unaffected by the pricing
fix. `y` is the only KP14 shock with cross-sectional content, because exposure is
`ebv = exp(beta_f y)` -- and `y` is already a 21-node state with a generator, so the disaster is
extra entries in a transition matrix that exists.

> **Corrected 2026-09-23 against the post-fix solve.** Measured on the live corrected tables, a -2.1
> jump in `y` moves claim value -1.7% / -5.9% / -12.1% by type, a top-minus-bottom spread of
> **10.4 pp** against the pre-fix **12.9 pp**. So the correction took ~20% off the jump's
> cross-sectional content, against 85% off the room -- the room collapse was a LEVEL effect, while
> the jump's dispersion is mostly `beta_f`, which did not change. The plan's verdict stands and is
> firmer: **`y` buys dispersion, not drama**, and the 25% value-loss target is unreachable in KP14
> with cross-sectional content. **But `vydis` must ride `vyg25`, not `vyx`.** The plan justified it
> as "the only model with a live gap to add to" citing `vyx` +0.1251; `vyx`'s gap is now +0.0009 and
> `vyg25`'s is +0.0148, where the same jump spreads 10.95 pp.

**BGN -- `beta_zr` is a rotation knob, and `r` is the state a disaster can live in.** The two
exported factors are priced at `sigma_z*sqrt(1 - corr_zr^2)` and `sigma_z*corr_zr` with
`corr_zr = beta_zr/(sigma_z*sigma_r)`, so they are the legs of a right triangle with hypotenuse
`sigma_z = 0.4`: moving `beta_zr` moves price BETWEEN two differently-shaped loading maps -- `A_mu`
a project cash-flow-beta map, `A_xi` a duration and option-share map -- without changing the total.
That is why it escapes the argument that closed the `gmult` family, which scales both channels
together. A disaster in `r` then rides a real price: project acceptance halves over one `rbar` and is
numerically nil at `rbar + 3 sd`, so investment stops and the project stock decays at 1%/month,
giving an endogenous output path of 0.786 of trend at 24 months and 0.669 at 40. Honest caveat: this
is a Volcker-type discount-rate disaster with a lagged investment collapse, not a productivity
disaster.

> **Recomputed 2026-09-23 on the corrected `vasicek.py`.** The fix was to `sigma12 = -Cov(log z, r)`
> entering the cumulative variance twice instead of once -- and `sigma12` is `beta_zr` accumulated,
> so **the defect lived inside the channel this proposal wants to use.** The factor prices are
> untouched (pure `beta_zr`/`sigma_z`/`sigma_r` arithmetic), but the rotation's effect on VALUES is
> about a third larger than the plan costed:
>
> | `beta_zr` | `corr_zr` | price, CF | price, rate | net consol discount, plan | **corrected** | limiting spread |
> |---|---|---|---|---|---|---|
> | -0.00014 (published) | -0.175 | +0.394 | -0.070 | 8.67%/yr | **9.35%/yr** | 2.37%/yr |
> | -0.00040 | -0.500 | +0.346 | -0.200 | 11.79% | **13.56%** | 8.52% |
> | -0.00048 | -0.600 | +0.320 | -0.240 | 12.75% | **14.77%** | **10.42%** |
> | -0.00056 | -0.700 | +0.286 | -0.280 | 13.71% | **15.93%** | 12.31% |
>
> **And it walks into a calibration problem the plan did not face.** A 10.4%/yr limiting term spread
> at `beta_zr = -0.00048`, against the 2.37% the published value gives and that BGN (1999, p.21)
> report -- which is the same objection that just killed pre-fix `vyx`. Tighten the live region
> toward -0.0003. Two further stale premises: `bgnzr`'s prediction is anchored on `bgnbase` figures
> that all moved (SR_max 0.2830 -> 0.3013, market 51.2% -> 53.7%, room +0.0267 -> +0.0274, fair gap
> +0.0019 -> -0.0001), and its falsification line -- a fair gap above +0.01, against "the established
> -0.008 to +0.004 band" -- is compromised, because `g0235f` now sits at +0.0079 at t 4.8.

**A productivity disaster in BGN is declined**, and the pricing fix does not reopen it. Its
heterogeneous version is indistinguishable from a `gmult` regime, i.e. `g0235r`, already run and
still negative at -0.0005; the output-channel version breaks three closed forms at once (affine bond
prices, the Gaussian MGF tilt, the log-normal cash-flow cross-moment) with `Chat` used as a scalar in
~20 places; and its seeds are the campaign's most expensive.

### Where to put the disaster in each model: the hazard-versus-direct principle

A state-dependent hazard `lambda(state)` does **not** price the jump -- conditional on date-t
information the indicator is independent of the SDF innovation, so the conditional covariance is
zero. What it delivers instead is a premium for disaster-risk NEWS, priced by the EXISTING price of
the state's risk, present in every month whether or not a disaster arrives. That is the channel
worth buying, and it is free. A jump premium proper still needs `lambda_Q != lambda_P`.

Writing the hazard term in log value as `D_eff * lambda_bar * J`, its exposure to the state is that
divided by `sd(state)`:

| model | state | hazard exposure (`phi=1`) | DIRECT exposure | reading |
|---|---|---|---|---|
| `gs_bx` | `x`, transitory productivity | **0.80** | 0.17-0.22 (measured) | 4x -- **put the disaster in the HAZARD** |
| `kp_vy` | `y`, priced OU | 0.018 | 0.006 / 0.020 / 0.040 by type | same order -- either; the hazard is the cheaper rotation |
| `bgn_gam` | `r`, Vasicek discount rate | 1.63 | ~17 (duration) | 0.1x -- **put the disaster in the STATE** |

**The principle: a state-dependent hazard is a strong lever exactly where the priced state's DIRECT
effect is weak.** A discount-rate state moves value enormously, so disaster-risk news is a rounding
error beside it; a transitory productivity state barely moves value, so the news dominates. This
inverts the obvious intuition and it is why GS21's near-total absence of room with a market at 94-97%
is *because* its priced state has almost no direct effect.

> **Caution, 2026-09-23.** The `kp_vy` and `bgn_gam` rows of that table are derived arithmetic on
> PRE-FIX solves (`D_eff` 34 and 20 months, BGN's ~17 duration). A recomputation on the corrected
> code gives a much shorter BGN consol duration, 14-15 months. The ORDERING spans a factor of 40 and
> is almost certainly robust; the numbers are not. Recompute before costing anything on them.

## History: the two protocol campaigns, and what each was for

Kept short, because both have run and both are recorded in `docs/RUNS.md`.

**Protocol v2, 2026-09-15 to 2026-09-17.** Made the thirteen rows comparable: one N, T, burn-in,
window, seed count, ridge grid and conditioning set for every economy, replacing three different
grids and two different burn-ins, with the baselines no longer scored on narrowed feature bases. It
asked no new question. Its own gate then reported five of thirteen rows censored at a grid edge,
which is what made protocol v3 necessary.

**Protocol v3, 2026-09-21 to 2026-09-22.** Two changes at once: the pricing fix (merge `91095fb`,
nine economies re-solved) and the ridge grid from `1e-5 ... 10` to `1e-7 ... 1000` (estimator-side,
so all thirteen re-ran). Outcome above. Two things about it are worth carrying into the next
campaign rather than rediscovering:

- **The precommitment discipline paid.** All fifteen solve ids were computed WITHOUT solving and
  committed before any manifest existed, so when `vyx`'s prediction was falsified the falsification
  was clean rather than arguable. `tests/test_precommitment_is_real.py` verifies the ordering by git
  dates; the integ ids needed a throwaway `git worktree`, because a chained id hashes its upstream
  tables' raw bytes and cannot be known until those tables exist.
- **Confounding two changes in one campaign cost information.** Because the pricing fix and the grid
  amendment landed together, the grid's effect can only be isolated in the four `gs_bx` economies,
  whose solves did not change -- and there it was +0.0001. On the KP14 floor side, where the
  amendment mattered most, its effect is not separable from the pricing fix and never will be. It
  was still the right call, since both changes invalidated the same 130 jobs; but a protocol change
  landed alone is a measurement, and landed with an economic change it is not.

## Not worth running

- **Anything at a different sample, window or ridge grid.** That is a change to the protocol, which
  changes every row in `docs/RESULTS.md`, not an experiment. X4 -- `vyg25` at T=860 and window 720
  with the grid extended to `1e-5` -- was the recommendation here until 2026-09-15 and is deleted for
  exactly this reason. Two thirds of it was the grid, which every economy now gets; the remaining
  third was a longer sample, which would have made one row incomparable to the other twelve, as its
  predecessor `vyxT860` did.
- **Another step up the price ladder (gamma_v 3 or more).** The corrected solve makes this worse,
  not better: at gamma_v 2.5 the oracle's mean expected excess return is 3.9% a year, BELOW gamma_v
  1.8's 4.2%, because a higher price of risk cuts claim values faster than it raises premia. So
  climbing the ladder buys a less plausible economy AND a smaller expected return. K7 goes between
  the two measured points, not above them.
- **Any NEW economy in BGN's regime family, or anything in GS21's exposure family.** GS21's is
  closed: `gx7`'s pre-registered negative fired and the corrected campaign confirms it at +0.0003,
  unchanged to four decimals. BGN's regime family needs a DECISION about the five points already
  measured (item 1 of the queue), not a sixth point.
