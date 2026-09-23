# Next up

A running to-do list. Experiments first, ranked by how much room or gap they can buy; everything
already decided, and all history, is at the bottom. Numbers and their provenance are
`docs/RESULTS.md`; cost and cluster procedure are `docs/RUNS.md`.

**The one thing that sets the ranking.** All three models are frictionless exact K-factor economies,
so `mu_t = -cov_t(R, m)` holds by construction and the population ceiling on DKKM-over-linear is set
by **K, the number of priced aggregate shocks** (finding 11). Measured at the loading nonlinearity
the two gap-producing economies actually have (theta about 0.85):

| K | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| ceiling on DKKM / fair linear | 1.020 | 1.051 | 1.084 | 1.158 | 1.179 | 1.306 | **1.711** |

Today's best is **1.044** (`vyg25`, K=3), a fair gap of +0.0148. The models sit at K = 1 (GS21),
2 (BGN, KP14 baseline) and 3 (the KP14 `vy` route). **Nothing that leaves K alone can move the
answer by more than a few hundredths**, and that is why the list below is ordered the way it is.

---

## The queue

| # | experiment | models | K | predicted ceiling | cost | status |
|---|---|---|---|---|---|---|
| 1 | ~~Decide the BGN regime closure~~ | BGN | — | — | none | **DONE 2026-09-23 — it does not stand** |
| 1a | **`g0235ff` — one more 4x on the switch speed** | **BGN** | 2 | 1.02 | **3 min + 10 seeds** | **ready, do this first** |
| 2 | `bgnzr` — rotate price onto the rate channel | BGN | 2 | 1.02 | 3 min + 10 seeds | ready, gate for #4 |
| 3 | K7 — a third `gamma_v` point | KP14 | 3 | 1.05 | 20 min + 10 seeds | ready |
| 4 | **Multi-factor term structure** | **BGN** | **2 → 5** | **1.16** | new solver | scoping |
| 5 | **Multiple priced states** | **KP14** | **3 → 6** | **1.18** | new solve chain | scoping |
| 6 | A second priced shock | GS21 | 1 → 2 | 1.02 | new solver | low priority |
| 7 | K1 — a continuum of exposures | KP14 | 3 | 1.05 | 1.7 h + 30 h seeds | ready |
| 8 | K3 — persistence of the priced state | KP14 | 3 | 1.05 | 2 × 15 min | ready |

**Items 4 and 5 are the same experiment in two models, and they are the only ones that can move the
answer by more than a hundredth.** Everything above them is cheap and settles what they should test;
everything below them is bounded by its model's current K.

### 1. ~~Decide the BGN regime closure~~ — DONE 2026-09-23, and it does not stand
Finding 12 in `docs/RESULTS.md` has the table. The closure confused two dimensions of the regime
family. Holding the stationary stress share at 33.3% and scaling both switch probabilities together,
the fair gap is **monotone in switching speed**: -0.0038 (4 switches per window) → +0.0025 (20) →
**+0.0079** (80), with t of -0.83 → 2.69 → 4.78 and seeds positive 4 → 8 → **10 of 10**. Room moves
the same way. Holding speed fixed and moving the stress share instead, 10% / 33% / 67% gives -0.0005,
+0.0025, +0.0025 — flat. **The share is exhausted; the speed is not.**

It is a loading-shape result, not a dimension one, exactly as finding 11 requires: the switch is
unpriced so every BGN economy is K=2, and what grows is DKKM's edge over `linrank_m` — the linear
method that DOES carry interactions and the market — 0.0001 → 0.0026 → **0.0079** along the ladder.

### 1a. `g0235ff` — one more 4x on the switch speed
The follow-on the decision implies, and the cheapest live lever in the file: `g0235f`'s parameters
with both switch probabilities 4x again, `p01` 0.333 and `p10` 0.667, spells of 3 and 1.5 months,
about 320 switches per window. Both stay under 1, so it is feasible. **A pure parameter override** —
one 3-minute J\* rebuild with a precommitted id, ten seeds at 40G, no code and no re-key, the same
shape as `bgnzr`.

Predict before building: the mechanism says **saturation, not a turnover**, because the regime stays
observable however fast it switches, so the interaction never disappears from the conditional
premium — it only becomes better identified within a window, and by 80 switches it already is.
Falsification: a fair gap below `g0235f`'s +0.0079 means the ladder has turned over and the family
really is exhausted; above about +0.012 means it is still climbing and the speed dimension deserves
a third point. Caveat to state in the spec: at 1.5-month spells a "regime" has stopped being a
business-cycle object and is a high-frequency shock to the price of risk.

### 2. `bgnzr` — rotate BGN's price of risk onto the rate channel
Pure parameter override, `beta_zr` -0.00014 → about -0.0003, one 3-minute J\* rebuild, ten seeds at
40G. Zero code, zero re-key. Its value is as the **gate on item 4**: if rotating the available price
onto the rate channel produces no room, a term-structure programme there has nothing to amplify.
State the gate on the rate channel's share of SR_max, not on the gap. Re-anchor the prediction first
— every `bgnbase` reference number moved (SR_max 0.2830 → 0.3013, room +0.0267 → +0.0274, fair gap
+0.0019 → -0.0001), and the old falsification line is void because `g0235f` already sits above it.
**Tighten the live region to about -0.0003:** at -0.00048 the corrected code gives a 10.4%/yr
limiting term spread against BGN's own 2.4%, which is the calibration objection that killed pre-fix
`vyx`.

### 3. K7 — a third point in `gamma_v`
`vyx`'s parameters at `gamma_v` 2.1 or 2.2, nothing else changed. One G solve plus integrals is
about 20 minutes on the Mac; then ten Phoenix seeds, ~60 node-hours. It is now a **theta probe**:
`vyx` and `vyg25` differ only in `gamma_v` and imply theta of roughly 0 and 0.81, so this maps the
price of risk onto the loading nonlinearity and says whether the gap is a threshold or smooth.
Register the falsification clause before building.

### 4. Multi-factor term structure in BGN — K 2 → 5
**The audit already named this**: *"any revival must inject into the discount channel (project
duration × multiple term-structure factors), not the dividend channel"*, after raising K at BGN's
cash-flow shock was built, verified and found inert (`voc_diagnosis/dead_end_K/`) because the
variance is not there. 36-50% of BGN return variance IS in the rate channel. Replace the one-factor
Vasicek `r` with a 3-4 factor term structure (level, slope, curvature), priced, with project duration
as the loading map — which is characteristic-linked and persistent rather than sampling noise, the
one genuinely promising feature the `bgnzr` analysis found. Affine structure survives, so `B(k, r)`
generalises; what needs work is the option value, which is where a disaster in `r` also broke.

### 5. Multiple priced states in KP14 — K 3 → 6
The same experiment, and the cheaper implementation: the `vy` route already added one priced OU
state with a per-type exposure ladder, and the machinery is parameter-driven. Add `y2`, `y3` with
their own `beta_f` ladders and prices. Each new state multiplies the integral-table count, so scope
the cost before committing — but the 2026-09-21 rebuild did three economies, six solves and 154
tables in about 20 minutes, so the old "90 min per economy" figure is 13x out.

> **Both of these leave the published models.** BGN with four term-structure factors is not BGN
> (1999), and KP14 with three priced states is not KP14 (2014). That is a decision about what the
> paper is, not a technical one. The honest alternative is to report that within the three models as
> published, faithfully implemented and correctly priced, the complexity gap is at most +0.015 of
> monthly Sharpe — which finding 11 says is what the structure requires.

### 6-8. Bounded by their model's current K
**GS21 second priced shock** (K 1 → 2, ceiling 1.02): the only way GS21 gets off a ceiling of exactly
1.000, but the payoff is the smallest and there is no natural candidate shock. **K1**, fifteen
exposure types over [0, 0.14]: raises theta, not K, so bounded at 1.05; about 1.7 h of integrals then
30 h of seeds. **K3**, `kappa_y` 0.15 and 0.70: more interesting after the fix, since the corrected
Girsanov adjustment saturates at `b gamma_v sigma_y / kappa_y` so `kappa_y` scales the whole
correction; note `sigma_y` is locked to `sqrt(2 kappa_y)`. Two solves of about 15 minutes.

### Housekeeping, not experiments
- `zero_book_in_sdf_solve: true` is declared in all thirteen specs and **read by nothing** — the same
  shape as the `burnin` field that said 200 while the code ran 400. `sdf_compute_kp14.py` still
  solves `ER` over all N firms with a ridge fallback that fires only on an exception, which is the
  construction that makes `sdf_ret` / `max_sr` unreliable — and `max_sr` is RESULTS.md's `SR_max`.
- `PRECISION_KEYS["kp"]` is still `("NY", "_i0")` and does not record the internal Q-grid the
  corrected solve introduced. Close it before another KP14 economy is added, which items 3, 5, 7 and
  8 all are.

---

## Closed, declined, and not worth running

- **K6, a defensible calibration** — ANSWERED by the pricing fix, not by an economy. `vyx` and
  `vyg25` sat at 18.2% and 22.9% oracle expected excess return a year; corrected they sit at 4.2% and
  3.9%, inside the band every other economy occupies. The gap at that calibration is +0.0009 and
  +0.0148. Lowering `gamma_v` lowers it further, which is why K7 goes up from 1.8.
- **GS21 capital destruction (`gsdis`)** — DECLINED by measurement, 2026-09-23. It needs both K 1→2
  and an extreme theta, because at K=1 the ceiling is 1.000 at *every* theta. Measured directly off
  `sol_gsbase/solution.npz`: theta 0.004 to 0.074, predicted ceiling **1.003**, below `vyg25`'s
  existing 1.044. And the default boundary it was built to populate is never reached — minimum equity
  per unit capital after the shock is 7.7 to 24.6 against a smoothing shock of sd 5, with
  `P + 5 <= 0` at 0.00% of occupancy weight. Unresolved: `kappa_D = 0.40` needs the `b` grid past
  1.0, though 0.25 already meets the 25% value-loss calibration. Probe in `_scratch/kceiling/`.
- **Disasters generally** — a disaster is ONE more priced factor, so it buys one step on the ladder
  above: `vydis` on `vyg25` would reach about 1.10, a disaster in BGN's `r` about 1.07, `gsdis`
  1.003. Worth having only if items 4 and 5 are ruled out; the design work is preserved below.
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
