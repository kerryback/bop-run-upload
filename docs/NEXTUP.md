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
| **1** | **`vym3t3` — `vym3`'s three priced states with THREE types, not twenty** | **KP14** | **dimension vs ESTIMATION DENSITY — the two things finding 15 could not separate** | **3 solves + 10 seeds, ~60 node-h** | **ready; informative whichever way it comes out** |
| **2** | **K3 — `kappa_y` at 0.15 and 0.70** | **KP14** | **whether BGN's switching-speed ladder (finding 12) — the only loading-shape effect that has ever produced a gap — replicates in a second model** | **2 × 15 min + 20 seeds** | **ready; the cross-model twin** |
| 3 | K7 — a third `gamma_v` point | KP14 | whether the gap is smooth or a threshold between `vyx` (+0.0009) and `vyg25` (+0.0148) | 20 min + 10 seeds | ready |
| 4 | K1 — a continuum of exposure types | KP14 | — | 1.7 h + 30 h seeds | **parked — a continuum is item 1's density problem taken to the limit; run item 1 first** |
| 5 | A second priced shock in GS21 | GS21 | — | new solver | **withdrawn — a pure K lever, and K is what finding 15 falsified** |
| ~~1~~ | ~~Decide the BGN regime closure~~ | BGN | — | none | **DONE 2026-09-23 — it does not stand** |
| ~~1a~~ | ~~`g0235ff` — one more 4x on the switch speed~~ | BGN | — | — | **DROPPED — a price of risk cannot switch on a 1.5-month spell** |
| ~~2~~ | ~~`bgnzr` — GATE on the discount channel~~ | BGN | — | done | **RUN 2026-09-23 — GATE FALSIFIED** |
| ~~4~~ | ~~Multiple priced states in KP14~~ | KP14 | — | done | **RUN 2026-09-25 — GATE FALSIFIED (finding 15)** |
| ~~—~~ | ~~Multi-factor term structure~~ | BGN | — | — | **WITHDRAWN — its premise is what `bgnzr` falsified** |

### 1. `vym3t3` — the same three priced states, three types instead of twenty
**The one experiment that settles finding 15.** `vym3` changed two things at once: it raised K from
3 to 5 AND went from three types to twenty. At N=500 and equal shares twenty types is **25 firms per
distinct premium value against `vyg25`'s 167**, and DKKM is the higher-variance of the two
estimators, so it should pay for thin types first. Run `vym3`'s exact three-state geometry with three
types — the three magnitudes that span its range, at three alignments — and the density confound is
gone.

- If the gap comes back positive, the K=5 economy was fine and **twenty types was the defect**;
  every future design is then bounded by firms-per-type, which is a protocol constraint the file
  does not yet record.
- If it stays negative, **dimension itself is what does not pay**, finding 15 stands unqualified,
  and the K programme is closed for good rather than provisionally.

Cost is three G solves plus 63 integral tables (about 20 minutes on the Mac) and ten Phoenix seeds.
Register the falsification clause before building. It reuses the generalised chain committed in
`bbbb384`, so there is no new code.

### 2. K3 — `kappa_y` at 0.15 and 0.70
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

### 3. K7 — a third point in `gamma_v`
`vyx`'s parameters at `gamma_v` 2.1 or 2.2, nothing else changed. One G solve plus integrals is
about 20 minutes on the Mac; then ten Phoenix seeds, ~60 node-hours. `vyx` and `vyg25` differ ONLY
in `gamma_v` and their fair gaps are +0.0009 and +0.0148, so a middle point says whether the gap is
smooth in the price of risk or a threshold — which is the shape of the one KP14 effect that has
survived. **Its old framing as a theta probe is dead**: the implied thetas it was to map (roughly 0
and 0.81) were inverted from the synthetic surface and withdrawn in finding 11; measured directly
both economies sit near 0.19. Register the falsification clause before building.

### 4-5. Parked and withdrawn
**K1, a continuum of exposure types** (fifteen types over [0, 0.14]; ~1.7 h of integrals then 30 h of
seeds): PARKED. It was ranked on raising theta, and `vym3` has now shown that more types at fixed N
does not raise measured theta while it does thin the cross-section — fifteen types is 33 firms each.
It is item 1's confound taken further, so item 1 comes first and this is re-ranked on the answer.

**A second priced shock in GS21** (K 1 → 2): WITHDRAWN. It was the only way GS21 gets off a ceiling
of exactly 1.000, and it is a pure K lever, which is what finding 15 falsified. There was never a
natural candidate shock either. Reopening it needs a reason that is not the ceiling table.

## Struck: run, dropped or withdrawn

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
- `zero_book_in_sdf_solve: true` is declared in all fifteen live specs and **read by nothing** — the same
  shape as the `burnin` field that said 200 while the code ran 400. `sdf_compute_kp14.py` still
  solves `ER` over all N firms with a ridge fallback that fires only on an exception, which is the
  construction that makes `sdf_ret` / `max_sr` unreliable — and `max_sr` is RESULTS.md's `SR_max`.
- `PRECISION_KEYS["kp"]` is still `("NY", "_i0")` and does not record the internal Q-grid the
  corrected solve introduced. **Close it before another KP14 economy is added, which items 1, 2 and
  3 all are** — every live item on the queue is now a KP14 economy, so this is on the critical path
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
