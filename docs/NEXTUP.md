# Next up

Rewritten 2026-09-15, when the measurement protocol landed (`docs/RESULTS.md`, "The measurement
protocol"). One recommendation, then the ranked alternatives, then what is not worth running.

## Recommendation: the protocol campaign

**What.** All thirteen economies, ten seeds each, at the protocol: N 500, T 500, burn-in 400, window
360, 125 evaluation months, the ridge grid `1e-5 … 10`, the model's full conditioning set, the fair
linear benchmark. Nothing else changes -- no economy's parameters move, no solve is redone except
BGN's six J\* tables, which the burn-in edit re-keyed without changing a byte of their contents.

**Why this and not another economy.** Every number in `docs/RESULTS.md` today was measured off
protocol in at least one respect, and two of those respects are known to move the answer:

1. **DKKM's Sharpe is censored, differently in different economies.** It is a maximum over the ridge
   grid, and the grid's floor won in 9 of 10 `vyx` seeds and 10 of 10 `vyg25` seeds. The floor was
   `1e-3` for BGN and KP14 and `1e-3` for GS21 with interior points BGN and KP14 never had. So the
   published KP14 gaps are lower bounds by an unknown amount, and the BGN-to-GS21 comparison
   included a grid difference. `vyx`'s own measured gains per decade at this window were 0.154,
   0.100, then 0.035, so the two new decades should be worth something and the flattening says not
   much -- but "not much" from an extrapolation is exactly what the retired X3 run showed can be
   wrong, and it is cheaper to measure than to argue.
2. **The baselines were scored on narrowed feature bases.** They saw only the state their paper has;
   the parameterizations saw the full set. Every "what the route added" statement therefore carries a
   protocol difference alongside the economic one. That is the comparison the whole file is built on.

Burn-in is the third change and is not expected to move anything: 300 or 400 months both reach the
stationary distribution, and the run that shows it is this one. It is worth doing because 300 / 400 /
300 across three models is a difference with no reason behind it, and because the spec field that was
supposed to record it said 200.

**No new question is being asked.** This is the run that makes the existing answers comparable. The
next economy comes after it, and which economy that should be depends on what it finds -- see the
alternatives.

**Predictions** are registered per economy, in each spec's `notes` (the thirteen protocol-v2 specs,
`experiments/specs/var-*-v{2,3,4}.json`). The common clauses: population quantities (room, SR_max)
unchanged within seed noise everywhere, since neither burn-in nor the grid touches them; DKKM up 0 to
0.010 in the BGN and GS21 economies whose floor did not bind; DKKM up +0.03 to +0.09 in `vyx` and
+0.04 to +0.11 in `vyg25`; every fair gap outside KP14 within its old bound; and **the winning
penalty interior -- neither `1e-5` nor `10` -- in at least 8 of 10 seeds of every economy**.

**What each outcome decides.**

- **Interior argmax everywhere, KP14 gaps up, the rest unmoved.** The protocol has done its job: the
  numbers are comparable and the KP14 gap is the economy's rather than the grid's. Go to K6.
- **`1e-5` still wins in KP14.** The grid needs another decade and KP14's DKKM is still censored.
  That is a protocol amendment, not an experiment, and it is the one outcome that must be fixed
  before any new economy is run -- otherwise the same problem is being rebuilt.
- **A fair gap outside KP14 rises above +0.01.** A path this file records as closed is not closed,
  and it was the narrowed bases or the grid that closed it. BGN's regime path or GS21's exposure path
  reopens, and `g0235d`'s ten seeds are the first place to look, since its screen is what closed BGN.
- **Burn-in moves a level statistic.** Then 300 months was not past stationarity for that model, and
  the slow BGN regimes (`g0235s`, `g0235r`, calm spells of 240 months) are where it would show.

**Cost.** 130 seed-jobs on Sol, roughly 650 node-hours; the sizing, ordering and memory are in
`docs/RUNS.md`, campaign 2026-09-15. The BGN J\* rebuild is six tables of about 8 minutes on the Mac.

## Ranked alternatives, all of them after the campaign

1. **K6, the low end of the price ladder: gamma_v 1.2 with vyx's exposures.** The referee-facing
   number, and the strongest candidate once the campaign lands. `docs/RESULTS.md` has two current
   points on this ladder, gamma_v 1.8 and 2.5, whose oracle mean expected returns are 18.2% and
   22.9% a year -- laboratory calibrations, as their specs say. gamma_v 1.2 should give about 14%.
   The question is whether a defensible economy shows a gap at all. Prediction to be registered
   against the campaign's `vyx` number, not today's: the gap should fall roughly with the price of
   the state's risk, so below `vyx`'s, and the falsification line is a fair gap below +0.03. Cost:
   one G solve (seconds) and three types of integrals (about 90 min), then ten seeds.
2. **K1, a continuum of exposures.** Fifteen types over [0, 0.14], shares right-skewed so most firms
   sit low: less of the state's Sharpe spanned by the market, and a smooth exposure map for random
   features. The case is weaker than K6's because the three-type structure was never shown to be
   what the linear methods exploit. About 7.5 h of integrals, then 30 h of seeds. Promoted over K6 if
   the campaign shows the KP14 gap is NOT signal-limited -- i.e. if the wider grid closes most of it,
   so what remains is the feature basis rather than shrinkage.
3. **K3, persistence of the priced state.** kappa_y 0.15 and 0.70. Informative about mechanism, not
   about the size of the gap: slower reversion bends values more but gives a window fewer independent
   cycles, so the prediction is that the gap FALLS at 0.15 and holds or rises at 0.70. Two solves of
   about 90 min, then two arrays.
4. **G3, the GS21 default-channel probe.** Twenty minutes, oracle only. Equity near default is a
   convex claim on the same shock, so its loading rises as the state worsens -- heterogeneous,
   state-dependent exposures the market does not replicate. Levers: sigma_z 0.16 to 0.25, or the tax
   advantage of debt 0.2 to 0.3. Its gate as written, "the market below 85% of SR_max", is too weak:
   the KP14 baseline meets it at 35% and has no gap. It needs an evaluation-window room clause of at
   least +0.05 as well. Not predicted to pass either, which is why it is cheap to settle. Queue it on
   Phoenix `htc` beside the campaign; it competes with nothing.
5. **K2, a rare extreme type.** Shares (0.45, 0.45, 0.10), top loading 0.14 to 0.20. Low priority:
   the bounded-rank argument is real but the discount check that withdrew K5 applies at a top loading
   of 0.20.

## Not worth running

- **Anything at a different sample, window or ridge grid.** That is a change to the protocol, which
  changes every row in `docs/RESULTS.md`, not an experiment. X4 -- `vyg25` at T=860 and window 720
  with the grid extended to `1e-5` -- was the recommendation here until 2026-09-15 and is deleted for
  exactly this reason. Two thirds of it was the grid, which every economy now gets; the remaining
  third was a longer sample, which would have made one row incomparable to the other twelve, as its
  predecessor `vyxT860` did.
- **Another step up the price ladder (gamma_v 3 or more).** `vyg25`'s room and SR_max rose by less
  than half what was predicted, the economy is already at 22.9% a year, and the direction of travel
  is toward a defensible calibration (K6), not away from one.
- **Anything in BGN's regime family beyond `g0235d`'s ten seeds, or GS21's exposure family.** Closed
  and retired for the reasons in `docs/RESULTS.md`, "Open proposals" -- unless the campaign reopens
  them, which is one of its registered outcomes.
