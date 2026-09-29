# Next up

A running to-do list. Experiments first, ranked by what each one settles; everything
already decided, and all history, is at the bottom. Numbers and their provenance are
`docs/RESULTS.md`; cost and cluster procedure are `docs/RUNS.md`.

**What used to set the ranking, and why it no longer does.** All three models are frictionless
exact K-factor economies, so `mu_t = -cov_t(R, m)` holds by construction and the population ceiling
on DKKM-over-linear is set by **K, the number of priced aggregate shocks** (finding 11). This list
was ordered by that ceiling. **On 2026-09-25 `vym3` tested it directly -- the first economy in the
project to raise K, from 3 to 5 -- and the gap went from +0.0148 to -0.0032** (finding 15). The
ceiling survives as an upper bound and fails as a guide to where to build.

Nor does the loading nonlinearity `theta` replace it. Measured on the panels rather than inverted
from the synthetic surface, `vyg25` and `vym3` are indistinguishable -- per-month theta 0.196 vs
0.197, and 0.293 vs 0.301 pooled, so the same share of premium variance sits in month-to-month
movement of the linear map -- while their fair gaps are +0.0148 and -0.0032
(`variants/diagnostics/premium_shape.py`). **This project has no
measured statistic of the premium that predicts which economy has a gap.**

So the queue below is no longer ranked by a predicted ceiling. Predicted ceilings have now failed
twice, on `bgnzr` and on `vym3`, and in both cases the number was read off a row the file had
already qualified. **It is ranked by what each experiment DISCRIMINATES, cheapest first**, with
priority to the ones that are the same experiment in more than one model.

What is actually known: two economies have a gap, `vyg25` (+0.0148, KP14, one priced state at
`gamma_v` 2.5, three types) and `g0235f` (+0.0079, BGN, a fast regime on the price of risk). Both
are LOADING-SHAPE results at fixed K. Everything below either tests what separates them from the
thirteen that have no gap, or is parked.

---

## The queue

| # | experiment | models | what it settles | cost | status |
|---|---|---|---|---|---|
| **1** | **K3 — `kappa_y` at 0.15 and 0.70** | **KP14** | **whether BGN's switching-speed ladder (finding 12) — the only loading-shape effect that has ever produced a gap — replicates in a second model** | **2 × 15 min + 20 seeds** | **ready; the cross-model twin, and now the only live item with a mechanism behind it** |
| 2 | K7 — a third `gamma_v` point | KP14 | whether the gap is smooth or a threshold between `vyx` (+0.0009) and `vyg25` (+0.0148) | 20 min + 10 seeds | ready |
| 3 | K1 — a continuum of exposure types | KP14 | — | 1.7 h + 30 h seeds | **parked — finding 16 shows density is real but not sufficient, so a continuum buys the density problem back for nothing** |
| 4 | A second priced shock in GS21 | GS21 | — | new solver | **withdrawn — a pure K lever, and K is what findings 15 and 16 closed** |
| ~~1~~ | ~~`vym3t3` — three priced states at three types~~ | KP14 | — | done | **RUN 2026-09-26 — GATE FALSIFIED (finding 16), but room is the largest in the file** |
| ~~1~~ | ~~Decide the BGN regime closure~~ | BGN | — | none | **DONE 2026-09-23 — it does not stand** |
| ~~1a~~ | ~~`g0235ff` — one more 4x on the switch speed~~ | BGN | — | — | **DROPPED — a price of risk cannot switch on a 1.5-month spell** |
| ~~2~~ | ~~`bgnzr` — GATE on the discount channel~~ | BGN | — | done | **RUN 2026-09-23 — GATE FALSIFIED** |
| ~~4~~ | ~~Multiple priced states in KP14~~ | KP14 | — | done | **RUN 2026-09-25 — GATE FALSIFIED (finding 15)** |
| ~~—~~ | ~~Multi-factor term structure~~ | BGN | — | — | **WITHDRAWN — its premise is what `bgnzr` falsified** |

### 0. PROBE — `vym3` at N=1000: is the density channel worth anything on its own?
**OFF-PROTOCOL and not reportable.** N=1000 departs from 500/500/360, so the runner refuses it
unless it names its own `BOP_RESULTS_DIR` (`variants/run_seeds_slurm.sh:448`). It writes to
`/data/sjpruitt/probe_n1000`, never to `variants/results`, and produces no economy row. N touches
nothing in the solve, so `vym3`'s committed G and integ tables are reused unchanged — no spec, no
new solve_id.

**Why `vym3` and not `vym3t3`.** Finding 16 read its +0.0068 as the density channel, but `vym3t3`
changed the type COUNT (20 → 3), the geometry and the premium mean all at once. N=1000 on `vym3`
holds every one of those fixed and moves ONLY firms per distinct premium, 25 → 50. **It is the
cleaner density test, and it is the one finding 16 could not run.**

**What the decomposition predicts.** `gap = room − excess estimation loss` is an identity, and
across the sixteen economies `excess_loss = 0.0024 + 0.887·room` (corr 0.99), so a room rise
returns only 11% of itself. N can only touch the DIVERSIFIABLE part of the loss, and that is
already spent: idiosyncratic risk is 0.26% of a diversified portfolio's variance at N=500 and 0.13%
at N=1000, a 0.06% reduction in portfolio volatility. So N should be nearly inert EXCEPT through
type resolution, which is a separate channel.

**REGISTERED PREDICTION (2026-09-28), before the run.** References: `vym3` at N=500 — SR_max
0.4178, room +0.0558, DKKM 0.2856, fair 0.2888, fair gap **-0.0032** (t -1.65, 5 of 10 positive),
DKKM 80.7% of its own ceiling.
- (a) fair gap rises by **+0.001 to +0.004**, point estimate +0.0025, landing near -0.0007 — still
  negative, still not significant. The band is wide because the +0.0025 is log-interpolated from
  finding 16's 25 → 167 move, which was NOT a clean density change.
- (b) room rises by **under 5%**, to below +0.0586. Room is a fixed-coefficient population
  quantity and N only removes idio drag.
- (c) SR_max rises by **under 1%**, to below 0.4220.
- (d) DKKM's share of its own ceiling rises by **under 2 points**, to below 82.7%.
- (e) the cross-seed sd of the gap does NOT fall materially (stays above 0.005). Seed variance is
  dominated by the aggregate state path, which is one path per seed at any N.

**THE FALSIFIER, and it is the point of running this.** If DKKM's share of its own ceiling rises by
**more than 5 points**, or the gap rises by more than +0.006, then the estimation loss is NOT mostly
time-series and N IS a lever. That would reopen raising N across the whole file and would mean the
0.887 slope is an artifact of N=500 rather than a structural relation. Under the predictions above
it is neither, and N is closed as a route.

**Cost.** Wall time scales 4-8x (the oracle is O(N^2 P) in `Phi'Sigma Phi` and O(N^3) in `max_sr`,
per month) and the per-month Sigma block of `months_data` goes ~1 GB to ~3.9 GB. Phoenix `public`
allows 7 days and ~112 GB, so no highmem is needed. **Seed 0 runs ALONE first as a timing and
memory probe**; the other nine are sized from what it achieves, because at 6 h per seed today a 6x
factor lands near the 2-day wall the campaign has been using.

**RESULT (2026-09-29). GRADED: the falsifier did NOT trip, and N is closed as a route to the gap.**
Ten seeds, `21639406_0` + `21659003_1..9`, 7:34-7:42 each, 58.2 GiB peak, all exit 0. Penalty gate
PASSES 10/10 interior. Paired by seed, which is exact here: the aggregate state path is
bit-identical across N (`rf` matches to 0.000e+00), so each seed is its own control.

| | N=500 | N=1000 | paired d | t |
|---|---|---|---|---|
| SR_max | 0.4261 | 0.5145 | **+0.0884** | +9.03 |
| room | 0.0594 | 0.0954 | **+0.0360** | +9.45 |
| DKKM | 0.2856 | 0.3143 | +0.0287 | +1.62 |
| fair linear | 0.2888 | 0.3163 | +0.0275 | +1.22 |
| **fair gap** | -0.0032 | -0.0021 | **+0.0012** | **+0.17** |

- (a) **HELD.** +0.0012, inside the +0.001 to +0.004 band; still negative, still not significant.
- (b) **FAILED.** Room rose **61%**, not under 5%.
- (c) **FAILED.** SR_max rose **21%**, not under 1%.
- (d) **HELD.** DKKM's share of its ceiling **fell** 3.9 points (t -1.33), against "rises under 2".
- (e) **HELD.** Cross-seed sd of the gap rose 0.0062 to 0.0192, above the 0.005 floor.

**The premise behind (b) and (c) was wrong: diversification is NOT exhausted at N=500.** The
"0.26% of a diversified portfolio's variance" argument above is withdrawn. Measured directly by
subsampling the N=1000 moments back to its own first 457 firms -- same realization, same premium
level -- SR_max goes 0.4507 to 0.5424 on the firm count alone.

**The error did not propagate, and that is the finding.** Room rose in **10 of 10** seeds; the gap
rose in **5 of 10**. On the identity `gap = room - excess loss` (exact to 0.00e+00 every seed):
d(room) **+0.0360** (t +9.45), d(excess loss) **+0.0349** (t +4.19), d(gap) **+0.0012** (t +0.17).
**The slope along N is 0.968**, against the 0.887 cross-economy slope: doubling the cross-section
buys 61% more room and the estimators hand back 97% of it. This is the sharpest confirmation of
room-is-not-gap in the file, precisely because it moves room by a large unambiguous 9-sigma amount
and the gap does not follow. The 0.887 slope is not an artifact of N=500, and raising N does not
reopen anything.

**One caveat for any future N experiment.** N does not nest: `ftype = rng_bx.choice(ntypes, size=N)`
consumes N draws before the idio shocks, so changing N offsets the whole firm-level stream and
redraws the run's premium level. At seed 0 that moved `mean_mu` 21%. It averages out -- paired over
ten seeds `mean_mu` is -0.0000 (t -0.04) -- but a single-seed N comparison is not interpretable, and
per-firm substreams would be needed to make one so.

Grading harness: `_scratch/probe_n1000/grade.py`, run as
`python variants/penalty_gate.py --results _scratch/probe_n1000` then `python _scratch/probe_n1000/grade.py`.

### 1. K3 — `kappa_y` at 0.15 and 0.70
**Now the best-motivated experiment in the file, and it is a cross-model replication.** Finding 12
established the only loading-shape result that has ever produced a gap: in BGN, holding the
stationary stress share fixed and scaling both switch probabilities together, the fair gap is
MONOTONE in switching speed — -0.0038 → +0.0025 → **+0.0079**, seeds positive 4 → 8 → **10 of 10**.
`kappa_y` is the same knob one model over: the speed at which KP14's priced state mean-reverts. If
the ladder replicates, the project has a mechanism that holds across two models with different
plumbing, which is worth more than either result alone; if it does not, `g0235f` is a BGN artifact
and the file should say so.

More interesting after the pricing fix, not less: the corrected Girsanov adjustment saturates at
`b gamma_v sigma_y / kappa_y`, so `kappa_y` scales the whole correction. Note `sigma_y` is locked to
`sqrt(2 kappa_y)`. Two solves of about 15 minutes, then twenty seeds.

### 2. K7 — a third point in `gamma_v`
`vyx`'s parameters at `gamma_v` 2.1 or 2.2, nothing else changed. One G solve plus integrals is
about 20 minutes on the Mac; then ten Phoenix seeds, ~60 node-hours. `vyx` and `vyg25` differ ONLY
in `gamma_v` and their fair gaps are +0.0009 and +0.0148, so a middle point says whether the gap is
smooth in the price of risk or a threshold — which is the shape of the one KP14 effect that has
survived. **Its old framing as a theta probe is dead**: the implied thetas it was to map (roughly 0
and 0.81) were inverted from the synthetic surface and withdrawn in finding 11; measured directly
both economies sit near 0.19. Register the falsification clause before building.

### 3-4. Parked and withdrawn
**K1, a continuum of exposure types** (fifteen types over [0, 0.14]; ~1.7 h of integrals then 30 h of
seeds): PARKED. It was ranked on raising theta, and `vym3` has now shown that more types at fixed N
does not raise measured theta while it does thin the cross-section — fifteen types is 33 firms each.
It is item 1's confound taken further, so item 1 comes first and this is re-ranked on the answer.

**A second priced shock in GS21** (K 1 → 2): WITHDRAWN. It was the only way GS21 gets off a ceiling
of exactly 1.000, and it is a pure K lever, which is what finding 15 falsified. There was never a
natural candidate shock either. Reopening it needs a reason that is not the ceiling table.

## Struck: run, dropped or withdrawn

### ~~1. `vym3t3`~~ — RUN 2026-09-26, GATE FALSIFIED (and the most informative failure yet)
Finding 16 has the tables. `vym3`'s three priced states with THREE types instead of twenty — types
3, 10 and 19 of `vym3`'s own, at `vyg25`'s shares, so firms per distinct premium went 25 → 167 with
the geometry held. **Fair gap +0.0036 at t 0.76, positive in 6 of 10, against a registered gate of
positive at t ≥ 2.**

Four of six clauses held. **Clause (d) failed and it is the finding**: room was predicted within 25%
of `vym3`'s +0.0558 and came in at **+0.0869 — the largest in the file**, above `vyg25`'s +0.0820.
Room is a POPULATION quantity, so three priced states at adequate density genuinely do leave more
for a nonlinear method. DKKM realised **74.9% of its own ceiling, the worst of the sixteen**, while
the fair linear side realised 96.4%, its usual number. The cross-seed sd of the gap tripled to
0.0151.

**Density was real and insufficient.** The sign flipped, -0.0032 → +0.0036, a move the size of
`g0235f`'s whole gap, so `vym3`'s thin types were part of its falsification. But at `vyg25`'s exact
density the K=5 economy still returns a quarter of `vyg25`'s gap.

**The registered dichotomy was too coarse.** This spec committed to reading a falsified gate as
"dimension does not pay at any density". That does not survive its own clause (d): dimension pays in
the population and fails in the estimator. Raising K enlarges the headroom and enlarges the variance
of capturing it, and for DKKM the second effect is larger.

**One thing to carry forward.** `vym3t3` is the first economy here whose design change moved a
measured statistic of the premium — the time-variation share, +0.1641 against +0.097 and +0.104.
It tracks ROOM and not the GAP. Any future proposal has to argue it raises room WITHOUT raising the
variance of estimating it, and nothing in this file yet shows how. 62 node-hours.



### ~~1. Decide the BGN regime closure~~ — DONE 2026-09-23, and it does not stand
Finding 12 in `docs/RESULTS.md` has the table. The closure confused two dimensions of the regime
family. Holding the stationary stress share at 33.3% and scaling both switch probabilities together,
the fair gap is **monotone in switching speed**: -0.0038 (4 switches per window) → +0.0025 (20) →
**+0.0079** (80), with t of -0.83 → 2.69 → 4.78 and seeds positive 4 → 8 → **10 of 10**. Room moves
the same way. Holding speed fixed and moving the stress share instead, 10% / 33% / 67% gives -0.0005,
+0.0025, +0.0025 — flat. **The share is exhausted; the speed is not.**

It is a loading-shape result, not a dimension one, exactly as finding 11 requires: the switch is
unpriced so every BGN economy is K=2, and what grows is DKKM's edge over `linrank_m` — the linear
method that DOES carry interactions and the market — 0.0001 → 0.0026 → **0.0079** along the ladder.

### ~~1a. `g0235ff`~~ — DROPPED 2026-09-23 on economics
Arithmetically available — monthly probabilities 0.333 and 0.667 are both under 1 — but it means
calm spells of 3 months and stress spells of 1.5. **A price of risk does not switch on that
timescale.** A volatility process can; a price of risk cannot. `g0235f` at 12 and 6 months is the
fastest defensible point in the family, so the ladder's top is where it was measured, not where the
parameter space ends.

### ~~2. `bgnzr`~~ — RUN 2026-09-23, GATE FALSIFIED
Finding 14 has the table. A 43% increase in the rate channel's price moved room from **+0.0274 to
+0.0292** — a rise of +0.0018 against a registered threshold of +0.005, and against a cross-seed
standard error on room of 0.0055, so not distinguishable from zero. Extrapolated to the 3–5x a
multi-factor structure could carry, room reaches +0.036 to +0.044 against `vyg25`'s +0.0820.

Four of five registered clauses held: SR_max rose, the market's share of it rose (condition 4 moving
the wrong way, registered as the standing risk), the fair gap stayed in band. The gate did not.

**And the direction of the miss is informative.** The fair gap went -0.0001 → **-0.0019** and
DKKM/fair 0.9997 → **0.9925**: rotating onto the rate channel made DKKM relatively WORSE. The rate
channel's dispersion is duration and the assets-in-place/growth-option mix — characteristic-linked
and persistent, which is exactly what a linear sort on book-to-price and 1/price is built to find.
BGN's own bound says `E[R]` is affine in those characteristics with rate-dependent coefficients. A
persistent, characteristic-linked loading map is good for the linear methods, not for complexity.

Cost: ~60 node-hours to close a direction that would have cost a new solver. Peak 19.8 GiB against
40G, 5.5–5.9 h per seed.

### ~~4. Multi-factor term structure in BGN~~ — WITHDRAWN 2026-09-23
Its premise was that 36–50% of BGN return variance sits in the rate channel, so injecting K there
would pay. **Item 2 tested the premise and it failed.** The variance is there and it is the wrong
kind: the rate channel's loading map is characteristic-linked, so more term-structure factors give
more of exactly what the linear side already captures. Reopening this needs a reason to believe a
slope or curvature factor loads differently across firms than the level factor does — which the
`bgnzr` result gives no support for.

### ~~4. Multiple priced states in KP14~~ — RUN 2026-09-25, GATE FALSIFIED
Finding 15 has the tables. `vym3` — three priced OU states, twenty rank-3 types, total price of
y-risk held at `||gamma||` = 2.5 — raised K from 3 to 5 and the fair gap went **+0.0148 → -0.0032**,
positive in 5 of 10 seeds. Room FELL, +0.0820 → +0.0558, against a registered prediction that it
would rise. Three of five clauses held, including every one about the design's mechanics: SR_max
within 15% of `vyg25`'s, expected excess return 4.33%/yr inside the band, ridge penalty interior
10 of 10.

**Two things from the probe are worth keeping, and one thing the probe got wrong.**

*Kept: the cost model.* A vector state needs no product grid. A type-f claim is on `e^{b_f . y}`, and
with independent OU components sharing `kappa_y` the projection `s_f = b_f . y` is itself a scalar
OU, so **type f IS the scalar problem the repository already solves**, at `b_eff = ||b_f||` and
`gamma_eff = (b_f . gamma)/||b_f||`. Cost scales with the number of TYPES, not states. The old text
here claimed a product grid — 21^3 nodes and 63 tables becoming ~27,800 — and was wrong by two
orders of magnitude. The generalisation is committed (`bbbb384`) and is the identity at
`nstates = 1`, so items 1 and 3 above inherit it free.

*Kept: the mechanism is real as stated.* Premium tracks `b_f . gamma`, magnitude tracks `||b_f||`;
under one state they are the same number times a constant, under three they come apart, and the
design does produce premium inversions. All of that is true and none of it helped.

*Wrong: the premise that it would not be spannable.* The probe measured the premium against maps on
MAGNITUDE and ALIGNMENT (R^2 0.52 and 0.885) and concluded a linear method could not represent it.
The estimators do not see magnitude or alignment; they see the five characteristics. Measured
against those, `vym3`'s theta is **0.187** against `vyg25`'s 0.200 — the product structure changed
spannability not at all (`variants/diagnostics/premium_shape.py`). **The lesson is to probe against the basis the estimator actually uses**,
which is the same error in a different costume as reading a ceiling off a withdrawn theta row.

*Also wrong: the type count.* Twenty types were chosen to over-determine `linlev_m`'s eleven
columns. The fair winner was `linrank_m` in 6 of 10 seeds and Fama-French in 3 — six columns — and
it beat DKKM anyway. The 25-firms-per-type note in the old text ("worth checking before
committing") was the right worry and was not acted on; it is now item 1.

## Housekeeping, not experiments
- `zero_book_in_sdf_solve: true` is declared in all sixteen live specs and **read by nothing** — the same
  shape as the `burnin` field that said 200 while the code ran 400. `sdf_compute_kp14.py` still
  solves `ER` over all N firms with a ridge fallback that fires only on an exception, which is the
  construction that makes `sdf_ret` / `max_sr` unreliable — and `max_sr` is RESULTS.md's `SR_max`.
- `PRECISION_KEYS["kp"]` is still `("NY", "_i0")` and does not record the internal Q-grid the
  corrected solve introduced. **Close it before another KP14 economy is added, which items 1 and 2
  both are** — every live item on the queue is now a KP14 economy, so this is on the critical path
  rather than beside it.

---

## Closed, declined, and not worth running

- **K6, a defensible calibration** — ANSWERED by the pricing fix, not by an economy. `vyx` and
  `vyg25` sat at 18.2% and 22.9% oracle expected excess return a year; corrected they sit at 4.2% and
  3.9%, inside the band every other economy occupies. The gap at that calibration is +0.0009 and
  +0.0148. Lowering `gamma_v` lowers it further, which is why K7 goes up from 1.8. (`vym3` sits at
  4.33%, so the band survived the three-state split too.)
- **GS21 capital destruction (`gsdis`)** — DECLINED by measurement, 2026-09-23. It needs both K 1→2
  and an extreme theta, because at K=1 the ceiling is 1.000 at *every* theta. Measured directly off
  `sol_gsbase/solution.npz`: theta 0.004 to 0.074, predicted ceiling **1.003**, below `vyg25`'s
  existing 1.044. And the default boundary it was built to populate is never reached — minimum equity
  per unit capital after the shock is 7.7 to 24.6 against a smoothing shock of sd 5, with
  `P + 5 <= 0` at 0.00% of occupancy weight. Unresolved: `kappa_D = 0.40` needs the `b` grid past
  1.0, though 0.25 already meets the 25% value-loss calibration. Probe in `_scratch/kceiling/`.
- **Disasters generally** — DECLINED 2026-09-25 on the same evidence that closed the K programme. The
  whole case for a disaster was that it adds ONE priced factor and so buys one step up the ceiling
  ladder: `vydis` on `vyg25` to about 1.10, a disaster in BGN's `r` about 1.07, `gsdis` 1.003.
  **`vym3` bought TWO steps and the gap went negative** (finding 15), so a ladder step is not a
  reason to build anything. A disaster is worth reopening only on an argument about loading SHAPE —
  that a rare, large, common shock makes the premium's dependence on characteristics change in a way
  a rolling linear window cannot track — and that argument has to be probed against the five
  characteristics the estimators actually see, not against the shock. The design work is preserved
  below and it is still measured and correct; it is the motivation that lapsed.
- **The ridge grid** — CLOSED 2026-09-22. Widened `1e-5 ... 10` → `1e-7 ... 1000`; in the four
  `gs_bx` economies, where the effect is isolable, two extra decades moved DKKM by +0.0001.
- **GS21's exposure path** — closed. `gx7`'s pre-registered negative fired and the corrected campaign
  confirms it at +0.0003, unchanged to four decimals.
- **Another rung up the price ladder (`gamma_v` 3+)** — the corrected solve makes it worse: at 2.5
  the oracle's mean excess return is 3.9%/yr, BELOW 1.8's 4.2%, because a higher price of risk cuts
  claim values faster than it raises premia.
- **Anything at a different sample, window or ridge grid** — a protocol amendment, which moves every
  row of `docs/RESULTS.md`, not an experiment.

---

## History

### The two protocol campaigns
**v2, 2026-09-15 to 17.** Made the thirteen rows comparable: one sample, burn-in, window, seed count,
ridge grid and conditioning set. Asked no new question. Its gate then reported five rows censored,
which forced v3.

**v3, 2026-09-21 to 22.** The pricing fix (merge `91095fb`; nine economies re-solved) and the ridge
grid, together. Withdrew the `vyx` headline: predicted fair gap +0.01 to +0.06, came back **+0.0009**,
positive in 7 of 10 seeds. `vyg25` survived at +0.0148 and `bgn_gam/g0235f` at +0.0079, both in ten
of ten seeds. Two lessons worth carrying: **precommitment paid** — all fifteen ids were computed
without solving and committed before any manifest, so the falsification was clean rather than
arguable; and **bundling two changes cost information** — the grid's effect is isolable only in the
four `gs_bx` economies, and on the KP14 floor side it never will be.

### The disaster-shock programme, preserved
Design work from `docs/plan-before-home-20260917.md` (deleted 2026-09-23), kept because it is
measured and not cheaply re-derivable. **Calibration:** hold the 25% VALUE drop, not the output drop —
a transitory loss of duration D transmits as `D / (D + asset duration)`, and GS21's equity elasticity
to a transitory `x` drop is 0.017 at two months against BGN's ~28% at 24. **Power:** over 125
evaluated months E[disasters] = 0.26, so about 2.3 seeds of ten ever see one — the primary outcome
must be `room` and the fair gap, never the realized gap. **Per model:** KP14's `x`/`z` jump is
provably null by exact homogeneity, re-verified on live source (`panel_functions_kp14.py:162,183,196`);
its `y` jump spreads 10.4 pp top-to-bottom under corrected pricing against 12.9 pre-fix, so the
mechanism survives at ~80% while the room fell 85%. GS21's Tauchen-node design is rejected because a
node at `x_d = -0.10` takes 69% of the kernel mass after renormalisation. BGN's `r` route needs the
option value re-derived; one escape is that at a deep enough `r` there is no exercise, so the
disaster-state value is the discounted normal-regime `J*` at a geometric exit date. **Hazard vs
direct:** a state-dependent hazard prices disaster-risk NEWS through the existing kernel, for free,
and is strong exactly where the priced state's direct effect is weak — 4x the direct channel in
GS21, a tenth of it in BGN. That table's KP14 and BGN rows are pre-fix arithmetic and need
recomputing before use.
