# Next up

Rewritten 2026-09-15, when the measurement protocol landed (`docs/RESULTS.md`, "The measurement
protocol"). One recommendation, then the ranked alternatives, then what is not worth running.

## Now: the corrected pricing and the amended grid, in one campaign

Two changes land together, and because the second is a protocol amendment **all thirteen economies
re-run**, not just the nine the pricing fix touched.

1. **The pricing fix**, merge `91095fb` (2026-09-21): `kp_vy`'s constant-rate treatment of the priced
   OU state, and BGN's halved bond covariance. Specified in `variants/kp_vy/parameters_kp14.py` and
   `variants/bgn_gam/vasicek.py`, asserted in `tests/test_risk_neutral_pricing.py`. It invalidates
   the solves behind all three `kp_vy` and all six `bgn_gam` economies -- 25 MB of cached artifacts
   across three producers, every change FUNCTIONAL (`python variants/solve_impact.py 91095fb^1 91095fb`).
2. **The ridge grid widens from `1e-5 ... 10` to `1e-7 ... 1000`**, eleven values. This is the one
   open *universal* proposal in `docs/RESULTS.md`, "Open proposals", and the gate table there shows
   five of thirteen rows censored: `vyx` 3/10 and `vyg25` 4/10 at the floor, `gx7` 2/10, `bx7` 7/10
   and `g0235s` 5/10 at the ceiling. It is estimator-side, so it moves every row -- including the
   four `gs_bx` rows the pricing fix leaves alone. Doing it now costs those four economies on top of
   the nine that must re-run regardless, instead of a second 130-job campaign later.

### A. Documentation -- DONE 2026-09-21

`docs/OU-process-question.md` and `docs/kp14_y_risk_adjustment.md` deleted: both prescribe a fix that
is not the one that landed (a per-type drift `-kappa_y*y - gamma_v*sigma_y + sigma_y^2*b`, plus a
grid widening to `y_max` ~7 and `NY` 21 -> 41 that never happened -- the `W = e^{by} A` substitution
removes the cross term and the ill-posedness wall, so `NY` is still 21 and 63 integral tables stayed
63). Their live content survives in `parameters_kp14.py`, `kp14_fd_vy.py` and
`tests/test_risk_neutral_pricing.py`.

In the same pass, a defect found while auditing: `parameters_kp14.py` leaked its override loop
variables `_k`/`_v` into the module namespace, which `solstamp.param_namespace` hashes -- so the
kp_vy solve_id **depended on the key order of `KP_PARAM_OVERRIDES`**. The same four parameters in
three orders gave three different ids, and `runstamp._same_json` compares by value and cannot catch
it. Fixed by deleting the loop variables; all orders now agree.

### B. The protocol amendment

Three constants, pinned against each other by `tests/test_protocol_is_uniform.py`, so they move in
one commit: `variants/common/protocol.py` `KAPPAS`, the literal second copy at
`variants/run_seeds_slurm.sh`, and nothing else -- the fair benchmark's grid is **derived**
(`variants/run_estimators.py`, `fair_kappas = sorted(set(kappas) | {10 * max(kappas), 100 * max(kappas)})`),
so it keeps its deliberate two-decade margin over DKKM's automatically.

All thirteen live specs then get a new version carrying the new `estimation.kappas`.

### C. Specs and solves, in the order precommitment requires

`rebuild_all_jstar.sh` exits 1 on any id mismatch and the repo refuses a spec claiming a
precommitment it did not earn, so the order is fixed: **compute the new ids without solving -> write
and commit the specs that pin them -> build -> commit the tables and manifests in a LATER commit.**

- **Compute.** BGN: `_scratch/precommit_id.sh bgn '<params>'`. KP14 has no case there; build the
  `solstamp.Snapshot` directly under `python -B`, as `tests/test_solve_id_reproducible.py` does. The
  **integ** id hashes the upstream G tables' raw bytes, so it cannot be computed until the G tables
  exist -- use a throwaway `git worktree` whose registry is discarded, the precedent at
  `docs/refactor/WORKING.md` section 61.
- **BGN, six J\* tables.** `bash variants/bgn_gam/rebuild_all_jstar.sh`, on the Mac and only the Mac
  (a Phoenix build differs at 5e-15 relative and the manifests record the artifact sha256). J\* falls
  14-21% across the rate range.
- **KP14: rebuild through `build_vy_tables.py`, do not adopt the `vyxq` tables.** `KP_VY_ADOPT=1`
  verifies file presence only, no kp_vy artifact carries an embedded `solve_id`, and -- decisively --
  the G solve_id is **prefix-blind**, because `extra` is not hashed. `vyxq` and a post-fix `vyx`
  produce the same id, and `solstamp.record` replaces a manifest's artifact list wholesale, so
  adopting one would silently destroy the other. Rebuilding under the single prefix `vyx` removes
  the collision. Cost: G is seconds to under a minute per type, integrals about 90 min per economy,
  so roughly 4.5 h for `vyx`, `vyg25` and `kpbase` together. Then delete the pre-fix `G_vyx*` /
  `integ_vyx*` and the transitional `G_vyxq*` / `integ_vyxq*`.
- **Supersede the old manifests -- `supersede`, never `retire`.** Eighteen of them (12 BGN + 6 KP14).
  Two live ids under one tag and stage make `runstamp.live_solves` return both, and then every seed
  of that economy reads STALE forever; retiring instead breaks the specs that truthfully pin the old
  ids.
- **Confirm every id actually moved** before submitting. `kpbase` is the one to watch: at
  `beta_f = 0` the KP14 correction is identically zero, so its tables may rebuild byte-identical
  while only the id moves. If an id did *not* move, that economy's ten tasks exit in three seconds
  with "already complete and current" -- the 2026-09-15 incident that skipped seventy tasks.

### D. The campaign

Both queues empty, then `bash variants/cluster_pull.sh` inside the shared checkout -- one tree, one
pull, never a plain `git pull`. All thirteen rows stay in `variants/submit_campaign.sh`.

**Memory is unchanged** (the amendment is estimator-side, the panels are the same size, and BGN's
smaller J\* points memory downward). **Walltime moves**: `docs/RUNS.md` measures about 2.8 h of DKKM
per seed at eight penalties, so eleven is roughly +57% on that stage. The one-day rows want two days
and `g0235s`'s 24.5 h class wants real headroom. Budget **~900 node-hours over 130 jobs** against the
2026-09-15 campaign's 650.

**The re-run must be atomic.** `runstamp.stem` puts no spec version in a filename, so re-run files
overwrite the old ones in place -- and a partial re-run leaves `aggregate_seeds.py` emitting
`spec_id = "MIXED:v3|v4"` with pre- and post-fix seeds **averaged into one row** at `n_seeds = 10`,
which no test catches. Do not aggregate or commit until all 130 tasks are in. Then
`python variants/penalty_gate.py` **before reading any number**, then
`python variants/aggregate_seeds.py`.

`docs/RESULTS.md` is then rewritten: the staleness block and both italic markers come out, the
protocol table's two ridge-grid rows and the KP14 solve-precision cell change, both economy tables
refill, and the **penalty-gate table is regenerated by hand** -- it is pinned by no test, and
uncensoring it is the whole point of the amendment.

### What this decides

The complexity gap is claimed on `vyx` and `vyg25`, both affected. An off-protocol smoke test
(N=200, T=360, window 240, one seed) put `vyx`'s DKKM-minus-best-linear at +0.007 (t 1.7) after the
fix against +0.019 (t 3.6) before, and mean expected excess return at 3.5%/yr against 17.7%. If the
protocol run agrees in direction, K6 below -- whether a defensible calibration shows a gap at all --
is being asked of a much smaller gap, and the ladder items (K1, K3, K2) need re-ranking against the
corrected `vyx` rather than the published one. The amendment settles the second question at the same
time: whether the KP14 gap was the economy's or the grid's floor.

## Recommendation: the protocol campaign

**Ran 2026-09-15 to 2026-09-17**; its results are `docs/RESULTS.md`. Kept here for the predictions
it registered and what each outcome was to decide. Nine of its thirteen economies are now superseded
by the section above.

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
