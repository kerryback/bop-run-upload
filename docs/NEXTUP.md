# Next up

Rewritten 2026-09-22, after the protocol-v3 campaign (`docs/RESULTS.md`). One recommendation, then
the ranked alternatives, then what is not worth running.

## Now: K7, a third point in the price of the state's risk

The protocol-v3 campaign is **done** (`docs/RUNS.md`, "Campaign 2026-09-21"; results in
`docs/RESULTS.md`). All 130 tasks COMPLETED, nothing failed, and the answers it came back with
reshape what is worth running next.

### What it decided

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

### The recommendation: K7, one economy between gamma_v 1.8 and 2.5

`vyx` and `vyg25` differ in exactly one parameter, the price of the OU state's risk, and that one
parameter is the difference between no gap and the only gap in the file. Two points do not locate a
threshold, and the pre-fix pair that used to be the third point was measured on wrong solves.

**What.** `vyx`'s parameters with `gamma_v` 2.1 or 2.2, everything else identical: `type_share`
[0.34, 0.33, 0.33], `type_bv` [0.02, 0.07, 0.14], `bv_comp` 1.2. A new spec
`var-kp_vy-vyK7-v1.json`, ids precommitted without solving, ten seeds.

**Cost.** Small, and much smaller than this file used to say. One G solve plus one set of integrals
is a few minutes on the Mac -- the 2026-09-21 rebuild did three economies, six solves and 154
tables in about 20 minutes total -- then ten Phoenix seed-jobs of about 6 h each, ~60 node-hours.

**What it would decide.** Whether the gap is a threshold in `gamma_v` or smooth in it. If +0.0009 at
1.8 and +0.0148 at 2.5 are joined by something near +0.007 at 2.1, the gap is smooth and small and
the honest statement is that this class of economy produces a complexity gap of a hundredth of
Sharpe at best. If 2.1 comes back near zero, there is a threshold, and locating it is a result.

**Register the falsification clause before building anything**, as the last campaign did: the
project's precommitment discipline is the reason the `vyx` withdrawal above is clean rather than
arguable.

### The other thing that changed, and it is not an economy

**`bgn_gam/g0235f` breaks the stated basis for closing BGN's regime path.** That path was closed on
five ten-seed points "whose fair gap never left -0.008 to +0.004". Under corrected pricing the top
of that band is `g0235f` at +0.0079, positive in ten seeds of ten at t 4.8 -- more than double the
old top, in an economy with no exposure heterogeneity at all. No spec's +0.01 falsification
threshold was crossed, so nothing fired automatically, and no new economy is needed to ask the
question: the five points are already measured. **Someone should decide whether the closure stands**
before K7 or anything else is built on the assumption that BGN has no gap.

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

## History: the two protocol campaigns, and what each was for

Kept short, because both have run and both are recorded in full in `docs/RUNS.md`.

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

## Ranked alternatives, all of them after K7

1. **~~K6, the low end of the price ladder~~ -- ANSWERED, not by an economy.** K6 existed because
   `vyx` and `vyg25` sat at 18.2% and 22.9% oracle expected excess return a year and no referee
   would accept them. The pricing fix put them at 4.2% and 3.9% without changing a parameter, so
   the defensible calibration is the one already measured, and the gap at it is +0.0009 and +0.0148.
   Lowering `gamma_v` to 1.2 would lower the gap further, which is why K7 goes UP from 1.8 rather
   than down.
2. **Decide the BGN regime closure.** No compute at all: five ten-seed points are already measured
   and `g0235f`'s fair gap is now +0.0079 at t 4.8, above the band the closure was stated on. This
   is a reading, not an experiment, and it should happen before K7 because it changes what K7 is
   testing -- whether the gap is a KP14 phenomenon or a small general one.
3. **K1, a continuum of exposures.** Fifteen types over [0, 0.14], shares right-skewed so most firms
   sit low: less of the state's Sharpe spanned by the market, and a smooth exposure map for random
   features. Much cheaper than this file used to say -- the integral stage is about 20 min for three
   economies, so fifteen types is roughly 1.7 h, then 30 h of seeds. The case for it strengthened
   with the correction: at `vyx` the linear side now reaches 95.3% of its ceiling, so what is left
   is a feature-basis question rather than a shrinkage one.
4. **K3, persistence of the priced state.** kappa_y 0.15 and 0.70, two solves of about 15 min each.
   More interesting after the fix than before it: the corrected Girsanov adjustment SATURATES at
   `b gamma_v sigma_y / kappa_y`, so `kappa_y` now scales the whole correction rather than only the
   timing of the state. Note the standing constraint that `sigma_y` is locked to `sqrt(2 kappa_y)`
   (`parameters_kp14.py`), so this moves persistence and innovation size together.
5. **G3, the GS21 default-channel probe.** Twenty minutes, oracle only. Equity near default is a
   convex claim on the same shock, so its loading rises as the state worsens -- heterogeneous,
   state-dependent exposures the market does not replicate. Levers: sigma_z 0.16 to 0.25, or the tax
   advantage of debt 0.2 to 0.3. Its gate as written, "the market below 85% of SR_max", is too weak:
   the KP14 baseline meets it at 35% and has no gap. It needs an evaluation-window room clause of at
   least +0.05 as well. Not predicted to pass either, which is why it is cheap to settle.
6. **K2, a rare extreme type.** Shares (0.45, 0.45, 0.10), top loading 0.14 to 0.20. Low priority:
   the bounded-rank argument is real but the discount check that withdrew K5 applies at a top loading
   of 0.20.

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
  measured (item 2 above), not a sixth point.
