# Next up

Written 2026-09-14, against `docs/RESULTS.md` as restructured the same day. One recommendation with
its pre-registered prediction, then the ranked alternatives, then what is owed to the record and what
is not worth running. Labels continue RESULTS.md's route log (K1 to K5, X1 to X3, B1 to B4, G1 to G5,
E1 are taken).

## Recommendation: X4, the vyg25 economy on the long sample

**What.** The K4 economy (`kp_vy/vyg25`: vyx with the price of the state's risk at 2.5) simulated
for 860 months and estimated on a 720-month window, exactly as X3 did for vyx, with the ridge grid
extended one more decade to 1e-5. No new solve: vyg25's G and integral tables are reused. Its own tag,
`vyg25T860`, because result files are named by model, tag and seed only.

**Why this is the experiment the results point at.**

1. **The only genuine gap is on KP14 Path 1, and every other route is closed.** E1 put the fair gap
   between -0.0025 and +0.0025 in all seven BGN and GS economies; B4's screen closed BGN's regime
   path and gx7's pre-registered negative retired GS21's exposure path. The open items outside Path 1
   are a 20-minute GS probe (G3) whose own gate it is not predicted to pass. Whatever widens the gap
   from here widens it on Path 1.
2. **Two dials widened it, and both act on the same bottleneck.** In vyx, vyg25 and vyxT860 the
   linear methods reach 99% to 100% of their population ceiling and DKKM 74%, 78% and 81% of its,
   with the smallest penalty on the grid winning in 29 of 30 seeds. The gap is DKKM's estimation
   shortfall closing. K4 raised the signal at fixed T (cross-sectional sd of expected returns 7.7% to
   10.6% a year) and lifted DKKM's share of its ceiling by 4 points; X3 raised T at fixed signal and
   lifted it by 7 points. Neither has been run on the other's terms.
3. **The mechanism says they compound rather than substitute.** Ridge estimation error falls with
   signal strength and with sample length separately; each dial moved DKKM's share of its ceiling
   from a different starting point by a similar amount, and the linear side has nothing left to gain
   on either. If the two are substitutes, X4 is the run that shows it, and that would be news.
4. **The grid floor is still binding, and the window is what makes it pay.** X3's decades of penalty
   gave 0.157, 0.116, 0.073 and 0.049 at window 720 against 0.154, 0.100 and 0.035 at window 360;
   1e-4 won in all ten seeds and the winning feature count fell to 360 or below in nine of them. The
   next decade should still pay at 720. 1e-5 covers it and finds the floor if it is there.
5. **No calibration cost beyond K4's.** Same economy, so the oracle's mean expected return stays at
   22.9% a year with cross-sectional sd 10.6%. The added cost is the window, sixty years, which X3
   already accepted. This is the largest gap the repository can show without moving the economy
   further from a calibration.
6. **Cheap and hazard-free.** No solve stage, so the chained-id hazard that cost K4 a cancelled solve and a
   resubmission cannot fire. Ten seeds at roughly 8 to 10 h on Sol public (X3 6.0 h at T=860 for vyx; K4 7.6 h at T=500),
   peak memory near X3's 56.9 GiB.

**Prediction, registered here before the spec is written.**

| | vyg25 (window 360) | vyxT860 (gamma_v 1.8) | X4 predicted | falsified if |
|---|---|---|---|---|
| gap | +0.1450 | +0.1875 | +0.20 to +0.27, central +0.23 | below +0.19 |
| fair gap | +0.1450 | +0.1872 | within 0.001 of the gap | -- |
| DKKM | 0.9450 | 0.8452 | 1.00 to 1.07 | -- |
| best linear | 0.8000 | 0.6577 | 0.80 to 0.83 | -- |
| room, evaluation window | +0.4108 | +0.3864 | +0.38 to +0.44 (same economy; month sample only) | -- |
| SR_max, evaluation window | 1.3887 | 1.2294 | 1.36 to 1.46 | -- |
| EW market, share of SR_max | 0.4694, 34% | 0.3577, 29% | about 0.47, 31% to 35% | -- |
| winning penalty | 0.001 in 10 of 10 | 0.0001 in 10 of 10 | 1e-5 or 1e-4 in at least 9 of 10; 1e-5 in at least 5 | -- |
| DKKM gain from the 1e-5 decade | -- | (1e-4 decade: +0.049) | +0.01 to +0.04 | -- |
| winning feature count | 3600 in 7 of 10 | 360 or below in 9 of 10 | 360 or below in at least 7 of 10 | -- |

How the gap range was formed, three ways that agree: adding the two increments to vyx (+0.0404 from
K4 and +0.0829 from X3) gives +0.228; applying X3's 6.5-point lift in DKKM's share of its ceiling to
vyg25's ceiling of 1.21, plus 0.02 for the new decade and +0.015 for the linear side, gives +0.23;
scaling vyg25's gap by X3's factor of 1.79 gives +0.26. The falsification line, +0.19, is X3's own
level: below it the higher price adds nothing at the longer window.

**Report** the gap on three grids, penalties of 0.001 and up (vyg25's), 0.0001 and up (X3's) and the
full grid, so K4, X3 and the new decade are separable in one run; the winning penalty and feature
count per seed; the oracle's E[mu] beside the gap.

**What each outcome decides.**

- **In range.** The dials compound; the gap is DKKM's estimation shortfall closing, and the program's
  headline is a complexity gap of about +0.2 of Sharpe in an economy whose premia are stated. The next
  lever is then data and shrinkage, not a more extreme economy.
- **Below +0.19.** The dials are substitutes: at window 720 the price of risk adds nothing, so DKKM's
  remaining shortfall at 720 is not signal-to-noise but the feature basis, which points at K1 (a
  smooth exposure map) rather than at any further dial.
- **Above +0.27.** Capture accelerates with signal at fixed T, REPORT.md §19d's premium-side
  conjecture in its strong form; then K6 below matters more, since it asks how far down the price
  ladder that holds.
- **1e-5 loses to 1e-4 in most seeds.** The grid floor is found for the first time, and the window,
  not the penalty, is the remaining data lever.

**How to run it.**

1. `experiments/specs/var-kp_vy-vyg25T860-v1.json`: parent `var-kp_vy-vyg25-v1`; `reused_solves` G
   `f41d052f1f960c4c` and integ `8d1308e8f21723f8`; panel N 500, T 860, burnin 200; estimation window
   720, kappas [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], `fair_linear` true; env `KP_VY_PREFIX` vyg25;
   the prediction table above in `notes`, with the falsification clause.
2. A `vyg25T860` case in `variants/run_seeds_slurm.sh`, copied from `vyxT860` with `SOLVE_TAG=vyg25`,
   `KP_VY_PREFIX=vyg25`, gamma_v 2.5 in `KP_PARAM_OVERRIDES` and the six-value grid;
   `tests/test_specs_match_shell.py` pins it to the spec.
3. Sol public, `--mem=96G`, 2 days, seed 0 first and `sacct` before the array, then seeds 1 to 9. Archive
   nothing first: the tag is new and collides with nothing. Add the row to `docs/RUNS.md` at submission.
4. `aggregate_seeds.py --flagship` and `fair_gap.py`; the economy joins RESULTS.md beside vyxT860, out
   of the ranked table (its own sample size), with the prediction graded.

## Ranked alternatives

2. **K6, the low end of the price ladder: gamma_v 1.2 with vyx's exposures, ten seeds, at window
   360 and at 720.** The referee-facing number. The ladder in RESULTS.md has three current points at
   gamma_v 1.8 and 2.5 and a legacy single seed at 1.2 in a different economy (vy, gap +0.029,
   capture near 10%, "limited by T"). X3 says that limit is data, so the question is whether a
   defensible economy shows a gap once it has the window. Prediction: E[mu] about 14% a year; at
   window 360 gap +0.03 to +0.06; at 720 with the 1e-4 grid +0.07 to +0.11; falsified if the fair gap
   at 720 is below +0.03. Cost: one G solve (seconds) and three types of integrals (about 90 min),
   then ten seeds at each window, about 90 node-hours. Run this second, or first if the goal shifts
   from the largest gap to the most defensible one.
3. **K1, a continuum of exposures.** Fifteen types over [0, 0.14], shares right-skewed. The case is
   that a smooth exposure map suits random features and lowers the market's average exposure. The
   prediction in RESULTS.md is modest and less confident than K4's, and X4's "below +0.19" outcome is
   what would promote it. About 7.5 h of integrals, then 30 h of seeds.
4. **K3, persistence of the priced state.** kappa_y 0.15 and 0.70. Informative about mechanism
   (slower reversion bends values more but gives a window fewer cycles), not about the size of the
   gap: the prediction is that it FALLS at 0.15 and holds at 0.70. Two solves of about 90 min and two
   30 h arrays.
5. **G3, the GS21 default-channel probe.** Twenty minutes, oracle only. Its gate (defaults at half a
   percent a year AND the market below 85% of SR_max, from 94% to 98% today and 95% in the GS21 baseline; A1 below shows it also needs a room clause) is not predicted to pass,
   which is exactly why it is cheap to settle. Queue it on Phoenix `htc` beside X4; it competes with
   nothing.

## A1, the anchor under current code -- DONE 2026-09-15

Each model as published ran as a first-class economy: specs `var-bgn_gam-bgnbase-v1`,
`var-kp_vy-kpbase-v1` and `var-gs_bx-gsbase-v1`, solve ids precommitted and reproduced, ten seeds each
with the fair benchmark, conditioning columns narrowed to the state each paper has. Fair gaps +0.0012
(BGN), -0.0024 (KP14) and -0.0022 (GS21): no model as published has a complexity gap, and twelve of the
thirteen registered predictions held (BGN's all-month room came in low, +0.0251 against +0.030 to
+0.045). Numbers and grading: RESULTS.md, "Baselines: the anchor"; jobs: `docs/RUNS.md`.

What the anchor changes for the choices above:

- **Every Path 1 number now has a measured zero.** X4's and K6's gaps are differences against kpbase's
  fair gap of -0.0024 and room of +0.0044, produced by the same pipeline.
- **Leaving the market is necessary and not sufficient.** In the BGN and KP14 baselines the market
  carries 50% and 35% of the attainable Sharpe, DKKM leaves it by 0.08 to 0.13, and the linear methods
  follow to within 0.007. What separates vyx is room, +0.36 against +0.02 and +0.004. That favours dials
  that raise room together with the non-market Sharpe (X4, K6, K1) over dials that only lower the
  market's share.
- **G3's gate is too weak as written.** "The market below 85% of SR_max" is met by the KP14 baseline at
  35%, which has no gap. If G3 is run, its gate should also require evaluation-window room of at least
  +0.05, as B4's did.

## Not worth running

- **Another step up the price ladder (gamma_v 3 or more) at window 360.** K4's room and SR_max rose
  by less than half what was predicted, the economy is at 22.9% a year already, and X4 gets a larger
  gap from the same economy.
- **Anything in BGN's regime family, G4, G5, X1, X2.** Closed, dropped or moot for the reasons in the
  route log.
- **Re-scoring vyx and vyg25 at window 360 with a 1e-4 penalty.** The window-360 headline numbers are
  lower bounds, since the grid floor won in 19 of 20 seeds, but vyx's flattening decades (0.154, 0.100,
  0.035) predict a gain of +0.01 to +0.02, the runner has no DKKM-only stage so it costs a full
  re-run, and X4 answers the grid question at the window where it matters.
- **K2, the rare extreme type.** The bounded-rank argument is real but the discount check that
  withdrew K5 applies at a top loading of 0.20, and nothing in the results ranks it above K1.
