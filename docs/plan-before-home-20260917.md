# A disaster state in the productivity process: design, feasibility, and what it can decide

**STATUS: NO ECONOMY IN THIS DOCUMENT HAS BEEN RUN.** `bgnzr`, `gsdis`, `vydis` and `bgndis` are
proposed tags, not results, and every prediction about them is registered rather than measured.
**Step 0 is the one exception and it is now run** (2026-09-17, Sol 63535705): it bites, hard, and
its result is in `docs/kp14_y_risk_adjustment.md` and summarised under step 0 below. Numbers fall into exactly three kinds and are marked as such: (a) measured,
from `variants/results/economy_table.csv` or a named log; (b) read from live code, with a
`file:line`; (c) arithmetic derived here from (a) and (b), which is an argument and can be wrong.
Nothing in this file is citable as a result, and `docs/RESULTS.md` remains the only place results
live.

---

## HANDOFF -- read this section first

Written 2026-09-17 for a fresh session picking this up. Seth is away; nothing here has his approval
to execute beyond what "Decisions already made" records.

**State of the tree.** Last commit `66272f6` ("asuair sbatch rewrite"). Both cluster queues empty
(`squeue -u sjpruitt` returns 0 on Sol and on Phoenix), so `bash variants/cluster_pull.sh` is safe
once the uncommitted work is committed. `python -m pytest tests/ -q` is **264 passed**;
`python variants/solve_impact.py --worktree` reports **nothing**, so no solve is currently invalidated.

Uncommitted, and none of it is mine except the last three:

| path | state |
|---|---|
| `docs/RESULTS.md` | modified -- findings 8 and 10, the `(DKKM-FMR)/FMR` ranking, the FMR-vs-market separation |
| `docs/refactor/WORKING.md` | modified |
| `variants/aggregate_seeds.py` | modified -- the one-table fold, the `off_protocol()` guard |
| `archive/REPORT.md` | **untracked** -- see the loose end below |
| `tests/test_live_docs_cite_live_economies.py` | untracked -- the live-citation fence |
| `docs/kp14_y_risk_adjustment.md` | untracked -- **the most important file here** |
| `variants/kp_vy/check_y_risk_adjustment.py` | untracked -- its reproducible harness |
| `docs/plan-before-home-20260917.md` | untracked -- this file |

**The one thing that matters.** Step 0 below. `variants/kp_vy/parameters_kp14.py:122-133` values a
claim on `e^{beta_f*y}` flows with the physical generator and a premium folded into the discount
rate, which is KP14's own construction (the paper's eq. 11) and is exact for its GBM shocks but not
for a priced mean-reverting state. The omitted term is `(gamma_v*sigma_y - sigma_y^2*b)*A'(y)`.
Established numerically three ways in `docs/kp14_y_risk_adjustment.md`; reproduce with
`python variants/kp_vy/check_y_risk_adjustment.py`. Its footprint is exactly `vyx` and `vyg25`,
which are the only two economies in the project with a fair gap above +0.004.

**UPDATE 2026-09-17: the check is run and the effect is large.** At each economy's own declared SDF
the three exposure types earn +2.59 / +7.85 / +13.33 %/yr (`vyx`) and +3.53 / +10.34 / +16.97 %/yr
(`vyg25`) more than that SDF prices, against a `kpbase` control of +0.016 %/yr, and no single
`gamma_v` removes the spread. Stripping `vyg25`'s expected returns of the unpriced part cuts
`SR_max` on the evaluation window from 1.4697 to 0.5996 and `SR_orth` from 1.4026 to 0.5527.
**No reported number is withdrawn yet**, because how much of the CURRENT ceiling is spurious is not
the same question as what the CORRECTED economy reports; only the re-solve answers that, and it is
Seth's call to pay for it.

**Decisions already made. Do not reopen these without being asked.**

1. **The measurement protocol is fixed** (`variants/common/protocol.py`): N 500, T 500, burn-in 400,
   window 360, 125 evaluated months, ten seeds, ridge grid `1e-5..10`, full conditioning set. A new
   economy is a parameter override and nothing else. Wanting a different sample, window or grid is a
   protocol amendment that moves every row of `docs/RESULTS.md`, not an experiment.
2. **Ridge-grid censoring is accepted and disclosed**, not fixed. Five rows are censored; the
   disclosure table is in `docs/RESULTS.md`. `python variants/penalty_gate.py` is the first thing to
   run on any campaign.
3. **The legacy crash modules are not to be restored.** They were never in this repository's git
   history -- they exist only in an un-versioned Dropbox tree at pre-correction parameters. The
   experiment in this plan is the live-code replacement for that program.
4. **A live document may cite only live economies.** Enforced by
   `tests/test_live_docs_cite_live_economies.py` against `docs/RESULTS.md`; dead routes are named
   only in its "Routes with no code" appendix, which states that nothing in it is citable.
5. **Tables rank by `(DKKM - FMR) / FMR`**, with the fair gap as the column that says whether a row
   means anything.
6. **A productivity disaster in BGN is declined** (reasons in its section below). The `r` route is
   NOT declined and is step 2.

**Loose ends I did not clear.**

- `archive/REPORT.md` is untracked while `variants/REPORT.md` -- the five-line stub pointing at it --
  **is committed**. So the stub is currently a dangling pointer for anyone who clones. Commit the
  archive copy. Do not edit `variants/gs_bx/gs_solve_gam.py`, which cites the stub in comments and
  whose bytes are digested into five GS21 solve ids.
- `docs/NEXTUP.md` still argues K6 against the retired pre-protocol `+0.1046`; it should be
  re-registered against `vyx`'s live `+0.1251`. And its "six tables of about 8 minutes" for the BGN
  J\* rebuild contradicts the log, which says 185-201 s each; trust the log.

**What in this document is verified and what is not.** Numbers are marked (a) measured, from
`variants/results/economy_table.csv` or a named log, (b) read from live code with a `file:line`, or
(c) derived arithmetic, which is an argument and can be wrong. Every prediction is registered, not
measured. The KP14 finding in step 0 is category (a)/(b) and independently confirmed; the
per-model feasibility claims are mostly (b) plus (c).

---

Supersedes the previous plan in this file. Items 1-3 of that plan (move `REPORT.md`, re-ground the
legacy-dependent claims, the fenced appendix + `tests/test_live_docs_cite_live_economies.py`) are
DONE. Its item 4, "restore and re-run the crash program", is **withdrawn and replaced by this**: the
legacy crash modules are un-versioned pre-correction code that would have to be imported, not
recovered, and the experiment below is the same question asked with live code and the protocol.

## Context

`docs/RESULTS.md` reports a complexity gap in exactly one family (KP14's `vyx` / `vyg25`, fair gap
+0.1251 / +0.1636) and fair gaps between -0.0080 and +0.0034 in the other eleven economies. The
conditions it identifies are heterogeneous exposure to a priced shock, a state that BENDS the
exposure map rather than shifting it, bending confined to rank/interaction space, and a market that
does not already span the attainable Sharpe. Every existing economy varies a **price of risk** or an
**exposure ladder**. None varies the **level of output**.

A rare disaster is the first proposal that would. It is also the first that could break the two
propositions in the structural-bound table, because those bounds are statements about LOCAL
exposure: a diffusion shock sees only the first derivative of the value function, which each paper
summarises with one firm variable, whereas a large jump sees the value function over a wide range,
and that non-local response can differ across firms in ways the single variable does not capture.
That is the mechanism worth paying for. Everything below is about which model can actually deliver
it.

## Calibration, held constant across models

| quantity | value | why |
|---|---|---|
| hazard | `p = 1/480` per month (2.5%/yr) | "once every 40 years"; all three models are monthly (`kp_vy` `dt=1/12`, BGN `p01=0.25/12`, GS21 `r=0.1/12`) |
| impact target | aggregate equity value falls **25%** on arrival | the invariant that IS comparable; see "matching on output is not comparable" below |
| premium target | the disaster adds **+2%/yr** to the market premium | fixes the free jump-risk price, which no model derives |
| everything else | `protocol.py` unchanged | N 500, T 500, burn-in 400, window 360, 125 eval months, 10 seeds, grid `1e-5..10` |

**Matching on the output drop is not comparable across these models, and matching on the value drop
is.** A transitory output loss of duration D transmits to value by roughly `D / (D + asset
duration)`, and asset duration differs by model. Measured/derived: GS21 equity elasticity to a
transitory x drop is **0.017** at 2-month duration, **0.091** at 12 months, **0.25** at 4 years
(against `d log P/dx` of 0.17-0.22 measured on `sol_gsbase`); BGN projects have effective duration
~62 months (`pi=0.99`, `rbar=0.006236`), so a 24-month disaster transmits ~28%; KP14's `y` reverts at
`kappa_y=0.35`/yr against a ~0.14/yr effective discount, so transmission is ~0.29 per unit of
exposure. So "output drops 33%" means a 0.6% value drop in one model and a 33% drop in another. Hold
the value drop; report the implied output drop per model.

## The statistical shape of a 1-in-40-year event at this protocol

This decides what the experiment's primary outcome can be, and it is not negotiable, because
`docs/RESULTS.md` "Not worth running" already rules out a longer T.

- 900 simulated months per seed (400 burn-in + 500): **E[disasters] = 1.9**.
- 500 retained months: E = 1.04, **P(none) = 35%**.
- 125 **evaluated** months: E = 0.26, **P(at least one) = 23%** -> about **2.3 of 10 seeds**.

So the realized-return channel is visible in roughly two seeds of ten, and cross-seed sd of any
realized gap will be dominated by a binary "did one arrive". **But the hazard is on in every month,
so the PRICE channel is present in all 1,250 evaluated months.** `room` and `SR_max` are population
quantities computed from true conditional moments (`variants/common/oracle.py`), so they are
measured at full power with zero realizations.

**Therefore: the primary outcome is `room` and the fair gap; the realized gap is secondary and is
reported split by whether a disaster arrived.** Pre-register that split, and record per seed the
disaster count in the evaluation window and in the union of estimation windows (a new field in the
oracle JSON). This is the only honest way to run a rare-event experiment at a fixed protocol.

Two further consequences to pre-register. `WINSOR = 0.0`: a single -25% month inside a 360-month
window moves the sample mean by -0.07%/month and adds 4% to the sample sd of the market leg, so the
linear methods' estimates degrade in the seeds that realize one. And the penalty gate must run
before any number is read -- a fat-tailed window plausibly pushes the ridge argmax to the grid
ceiling, which is where three economies already sit.

## What the measurement can and cannot see

The apparatus scores everything by `mu_t` and `Sigma_t` only. As far as it is concerned a disaster IS
one extra priced factor with Sharpe `gamma_D * sqrt(p)` and a specific loading map, plus fat-tailed
realizations. Its skewness is invisible. **So the experiment's entire value lies in the SHAPE of the
disaster loading map across firms, not in the rarity.** Design accordingly: heterogeneous, non-affine
exposure is the whole content, and a disaster with a common exposure is a pure market shock.

That last point cuts hard, and it is the tension at the centre of this proposal: a disaster is an
AGGREGATE shock, so it adds to the market's own Sharpe, which pushes against the fourth condition
(the market must not span the economy). It helps only through cross-sectional dispersion in exposure
and hurts through the aggregate premium it adds.

## Per-model feasibility, and which shock

### KP14 -- the obvious candidate is provably null; `y` is the one worth running

`x` is aggregate disembodied productivity (output and investment cost, priced at `gamma_x = 0.69`,
premium 8.97%/yr); `z` is investment-specific technology (`gamma_z = -0.35`). `x` is the output
shock. **And a jump in it cannot create cross-sectional room, by exact homogeneity:**

- `VAP = (x*ebv) * K**alpha * A(...)` with `K = (alpha*z*A)**(1/(1-alpha))`, no `x`
  (`panel_functions_kp14.py:182`)
- `PVGO = z**(alpha/(1-alpha)) * x*ebv * lambda_f * G(eps)`, and `G` is solved over `(eps, y,
  regime)` only (`:193`)
- `book = x*ebv/z * K.sum()` (`:163`)

So `price = x * ebv * f(z, eps, uj, lambda_f, y)` exactly: **P is homogeneous of degree 1 in `x`.** A
multiplicative jump scales every firm's price by the same factor, leaves every ratio characteristic
(book-to-market, leverage) unchanged, and is perfectly spanned by the market. Both `VAP` and `PVGO`
carry `z**(alpha/(1-alpha))`, so the same argument kills a `z` jump. **Prediction, registered before
any solve: an `x` or `z` disaster raises `SR_max` and the market's share of it and shrinks every gap
in the model.** This is checkable off an existing panel in minutes -- the jump return must be
constant across firms -- and it is why the obvious experiment is not the one to run.

`y`, the priced OU state the `vyx` route added, is the only KP14 shock whose jump has cross-sectional
content, because exposure is `ebv = exp(beta_f * y)` with `beta_f` in {0.02, 0.07, 0.14}
(`parameters_kp14.py:29-31`). A downward jump in `y` is heterogeneous by construction, non-affine in
the exposure (an exponential), priced by the existing `gamma_v`, and -- the decisive practical point
-- **`y` is already a 21-node discrete state with a generator `Qy` (`parameters_kp14.py:101-117`), so
the disaster is extra entries in a transition matrix that already exists.** No new dimension, no new
`integ` files, no grid growth.

Its limit is the value effect: transmission ~0.29 per unit, so a -2 jump moves the top type -8% and
the bottom type -1.1%; at the grid edge (`y_max = 3.5`) the top type reaches ~-14%. **`y` buys
dispersion, not drama.** The 25% impact target is not reachable in KP14 with cross-sectional content.

Cost: `Qy` lives in `parameters_kp14.py`, which is in both `G_SOURCES` and `I_SOURCES`
(`build_vy_tables.py:100-101`), so editing it **re-keys all six KP14 solve ids**. At disaster rate 0
the tables come back byte-identical (the burn-in precedent), so this is a rebuild-and-verify, not a
re-derivation: ~2 h per tag of G + integ, three tags, then **30 KP14 seed jobs re-run** (48G, longest
7.6 h) to restamp provenance. Plus the new economy: one solve and 10 seeds.

### GS21 -- the only design that delivers a rare, permanent, dramatic, heterogeneous disaster

Two candidate channels, and the second is much stronger.

**Rejected: appending a disaster node to the Tauchen `x` grid.** The grid plumbing tolerates it
(nothing interpolates on `x`; `pr_x` rows are used only as probability vectors), but the kernel at
`gs_solve_reg.py:137-138` multiplies `gamma` by the NORMALISED innovation `(x' - rho_x*x)/sigma_x`,
and `sigma_x = 0.007046`. Computed on the committed `sol_gsbase` kernel: a node at `x_d = -0.10` is
-14.2 innovation sd and takes **69%** of the kernel mass after renormalisation; `x_d = -0.20` takes
**99.96%**. The disaster's price would be set mechanically by the grid, not by a parameter. Off the
table without re-specifying the SDF.

**Rejected as insufficient: a regime-indexed output level.** Cheap (~20 lines: make `pi_R`, `recov`,
`pi_R_e` regime-indexed at `:152-153`, `:172`; +1% wall, +17 MB) and the regime block is otherwise
the perfect host -- `p01` already IS a once-per-4-years arrival. But the value elasticities above say
a short disaster in `x` moves equity 0.5-2.7%, and the solved refinancing targets are interior
(b' in {0.632, 0.684, 0.737, 0.789} of a [0,1] grid) with default probability averaging 0.002% at the
chosen leverage. It would not bite and it would not wake the default channel.

**Recommended: capital destruction.** The value function is `(znum, xnum, bnum)` with **no capital
dimension** -- the model is homogeneous of degree 1 in `k`, and `bgrid = linspace(0,1,bnum)` is debt
**per unit of capital** (`pi_R` at `:152` nets `(1-tau)*bgrid` off per-unit profit). So destroying a
fraction `kappa_D` of the capital stock while nominal debt is untouched is exactly `b -> b/(1 -
kappa_D)`. **The code already contains that operation, in the other direction and with the
interpolation weights precomputed**: `_jg, _wg = _b_interp_weights(bgrid / g)` at `gs_solve_reg.py:179`,
used at `gs_sim_bx.py:205` (`b[t]/g` when the firm invests) and `:207-208`, `:238`
(`scale[t]`, `k[t+1] = k[t]*scale[t]`).

That single design satisfies everything the other candidates miss:

- **output drops dramatically and permanently** -- output is proportional to `k`, and `k` does not
  mean-revert, so transmission to value is one-for-one, not 0.017;
- **exposure is heterogeneous and violently non-affine** -- every firm's `b` rises by the same
  proportion, but the consequence runs through `alive = (Pn + mshock) > 0` (`gs_sim_bx.py:234`), so
  high-`b` low-`z` firms default and low-`b` firms barely notice. `lev` is already in GS21's
  characteristic set;
- **it populates the default boundary**, which is the explicit escape clause in GS21's own structural
  bound ("smooth and monotone AWAY FROM AN UNPOPULATED DEFAULT BOUNDARY"). At `kappa_D = 0.25`, b'
  around 0.70 maps to 0.93, where measured one-period default probability is percent-level rather
  than 0.002%. This is proposal **G3 given a mechanism strong enough to reach the boundary**;
- **it re-keys nothing.** A new module `gs_solve_dis.py` is the sanctioned pattern -- `gs_solve_gam.py`
  exists for exactly this reason, `tests/test_gs_solver_parity.py` holds the copy honest, and its
  `REG`/`GAM` pair does not cover a third file. All 12 committed GS21 solve ids survive untouched.

Open items: the `b` grid must extend above 1.0 (at `kappa_D = 0.4`, b = 0.70 maps to 1.17), which is
internal to the new module; and the disaster arrival is **unpriced** as a bare regime switch (the
switch carries no `cov(M, 1{s'})` term in either solver), so it needs pricing -- see the pricing fork
below. Cost: one solve at 3.5-5.9 h and ~2.7 GiB, ~100 MB artifact, then 10 seeds in the campaign's
**cheapest** class (32G, ~5 GiB, GS21 panels are the lightest).

### BGN -- reopened: `beta_zr` is a rotation knob, and `r` is the state a disaster can live in

**This reverses the verdict in the section below, which was written evaluating a hazard at the
CURRENT `beta_zr` and is correct only there.** Two moves reopen BGN, and the first is the cheapest
untried lever in the repository.

**(b) `beta_zr` is not another rung of the price ladder -- it is a ROTATION at constant total price.**
The two exported true factors are priced at `sigma_z*sqrt(1 - corr_zr**2)` and `sigma_z*corr_zr`
(`panel_functions.py:222-223`) with `corr_zr = beta_zr/(sigma_z*sigma_r)`, so the two prices are the
legs of a right triangle with hypotenuse `sigma_z = 0.4`. Moving `beta_zr` moves the price of risk
BETWEEN the two channels without changing the total:

| `beta_zr` | `corr_zr` | price, cash-flow channel | price, rate channel |
|---|---|---|---|
| -0.00014 (current) | -0.175 | +0.394 | **-0.070** |
| -0.00040 | -0.500 | +0.346 | -0.200 |
| -0.00048 | -0.600 | +0.320 | -0.240 |
| -0.00056 | -0.700 | +0.286 | -0.280 |
| -0.00080 | -1.000 | **0.000** | -0.400 |

That matters because **the two channels have different cross-sectional loading maps.** `A_mu` is a
project cash-flow-beta map (`pi*sqrt(1-corr_zr**2)*summ/B`, `summ` being the sum over live projects of
`chi*sigmaj*corr_zj`), while `A_xi` is a DURATION and option-share map, built from `dD/dr` weighted by
the assets-in-place share plus `dJstar/dr` (`loadings_compute.py:36-55`). So `beta_zr` is a rotation
between two differently-shaped premium channels -- exactly what KP14's bound forbids ("with premia
affine in one firm variable and the two priced channels nearly parallel across firms, there is nothing
to rotate between") and what BGN's permissive structure allows. **This is why it escapes the argument
that closed the `gmult` family**: `gmult` scales both channels together, which is direction-invariant
and predicted null; `beta_zr` does not.

**Direction and headroom.** `log M = -r - 0.5*sigma_z**2 - sigma_z*nu`, so
`cov(log M, r') = -beta_zr`: at the current NEGATIVE value, rate spikes are already
high-marginal-utility states, which is the sign a rate disaster needs. The experiment therefore wants
`beta_zr` MORE negative, and the wall is `corr_zr >= -1` at `beta_zr = -0.0008` -- **5.7x the current
magnitude**. The `+0.00027` cap recorded in `docs/RESULTS.md` is on the far side and does not bind
this direction (first-moment convergence of the perpetuity sums allows up to +5.8e-4, so that cap is
presumably the second-moment condition, the same shape as `2*max(gmult)*scale < 1`; confirm before
using a positive value). Caveat at the wall: `A_mu -> 0` as `corr_zr -> -1`, so the cash-flow channel's
price vanishes and BGN degenerates to one factor. **The live region is `corr_zr` about -0.5 to -0.7,
`beta_zr` about -0.0004 to -0.00056**, where the rate channel's price triples to quadruples and the
cash-flow channel loses only 19-27%.

**Cost of (b): a pure parameter override.** No code, no re-key -- `beta_zr` is already in the hashed
namespace, so a new tag gets a new J\* id and the six existing tables keep theirs. One J\* rebuild
(measured 185-201 s, `_scratch/protocol/rebuild_all_jstar.log`), then ten seeds in `bgnbase`'s 40G
class, not the 87 GiB highmem class. **Run this alone and first, as `bgnzr` off `bgnbase`.** It is
also the precondition for everything else here: if rotating the full available price onto the rate
channel produces no room, a disaster in `r` has nothing to amplify.

**(a) A disaster in `r` is better motivated than a productivity disaster would be, because BGN's
investment channel has a saturating nonlinearity that only a disaster-sized move reaches.** Project
acceptance is 10% at `r = 0` and 5% at `r = rbar = 0.006236`
(`prob_in_money_targets`, `parameters.py:20`) -- it halves over one `rbar`. The stationary sd of `r` is
0.00641, so at `rbar + 3 sd` acceptance is numerically nil: investment stops, and the project stock
then decays at `1 - pi = 1%` per month. **The output drop is endogenous and dramatic**: 0.786 of trend
after 24 months, 0.696 after 36, 0.669 after 40. Its incidence is heterogeneous by the firm's
assets-in-place versus growth-option mix, which is `A_xi`'s map, and BGN's own bound says the
affine coefficients are RATE-dependent -- so a disaster in `r` makes them jump between two very
different values, a state-dependent affine map in a RATIO characteristic.

Two honest caveats on what it is. BGN's `r` is exogenous and its only channel is the cost of capital,
so this is a **Volcker-type discount-rate disaster with a lagged investment collapse, not a
productivity disaster** -- the output drop arrives with a lag and as a byproduct, while most of the
value drop is the discount rate. And the sign is rates UP: rates DOWN is an investment boom in BGN.

**(a)'s cost is concentrated in exactly one place.** A compensated Poisson jump in a Vasicek keeps the
model affine, so `B(k, r)` survives with one extra jump-transform term in the `phi1` / `sigma12` /
`sigma1_sq` recursions (`vasicek.py:19-35`). What breaks is the OPTION value: `norm.cdf(d1)`,
`norm.cdf(d2)` at `vasicek.py:82-94` need `r` at the exercise date Gaussian, and Gaussian-plus-decayed
-jumps is not a finite Gaussian mixture, because each jump's decay depends on WHEN it arrived. The
`exp(y**2/2)` MGF identity at `sdf_compute.py:99-103` fails for the same reason. Replacing them
numerically changes BGN's solve discretisation, and the protocol makes solve precision uniform within
a model, so all six BGN tables would move to the new solver.

**One escape route, worth deriving before (a) is costed.** If the disaster's `r` is deep enough that
acceptance is numerically zero, then there IS no option exercise in the disaster state -- so no
`norm.cdf` is needed there, and the disaster-state option value is just the discounted normal-regime
`J*` evaluated at a geometrically distributed exit date. That is the finite-summation machinery
already in the file (`_ak[s, k] = P**(k-1)(s, 0)`, `vasicek.py:217-221`) plus the existing
Gauss-Hermite `r'` integration, with a bounded and reportable truncation error. If it works, (a) is a
parameter-scale change instead of a new BGN solver. Thirty minutes of algebra decides which.

**And (b) reopens the state-dependent hazard in BGN too**, since a hazard `lambda(r)` is priced
through the rate channel once that channel carries a real price. The "structurally impossible" verdict
below was conditional on `beta_zr` staying where it is.

### BGN -- why it was declined, which now applies only to a PRODUCTIVITY disaster

Three reasons, all of which are about putting the disaster in BGN's CASH FLOWS. None of them touches
the `r` route above.

1. **Its heterogeneous-disaster version has already been run and is closed.** BGN's only exposure
   heterogeneity is the project-beta continuum, and the premium enters as `exp(-beta*gmult[s])`
   (`vasicek.py:237`). A disaster whose intensity is proportional to a project's systematic exposure
   is therefore indistinguishable from a `gmult` regime -- which is `g0235r`: `p01 = 0.05/12` (once
   per 20 years, twice this proposal's rate), 27-month spells, **fair gap -0.0062**, DKKM within a
   hundredth of the equal-weighted market. `docs/NEXTUP.md` already rules out further work in that
   family.
2. **The output-channel version is the most invasive change of the three and needs new closed forms.**
   A jump or mixture in `nu` breaks three separate analytic results simultaneously -- the Vasicek
   affine bond prices, which bake in the log-normal SDF/cash-flow covariance (`vasicek.py:17-39`);
   the Gaussian MGF tilt `exp(y**2/2)` (`sdf_compute.py:99-103`); and the log-normal cash-flow
   cross-moment (`:187-189`). The feasible form is a state-dependent cash-flow level `Chat_s`, which
   is expressible in the existing finite-summation regime machinery -- but `Chat` is used as a scalar
   constant in roughly twenty closed forms, and the mixture is hardcoded in the two-branch "own state
   or the other one" form at ~35 sites.
3. **Its seeds are the campaign's most expensive.** `g0235r` and `g0235s` are the only two economies
   that cannot sit on a Phoenix node (~87 and ~83 GiB projected), highmem-only, longest seed in the
   24.5 h class.

State this in `docs/RESULTS.md` as a declined proposal with its reason. It is not an omission: BGN is
where the rare-priced-state experiment has already returned a negative -- **in the cash-flow channel.
The rate channel is a different matter and is reopened above.**

## Three questions settled: the `beta_zr` move alone, the `r` disaster's counterparts, the hazard

### 1. What moving `beta_zr` alone does -- a registered prediction, and it is a negative

`beta_zr` conserves the SDF's total volatility (`sigma_z = 0.4`), so this is not a price ladder. But it
is NOT premium-neutral, because the two channels' LOADINGS differ: assets carry a duration loading of
about 17 on the rate factor and a much smaller loading per unit of price on the cash-flow factor. The
rotation therefore raises the level of premia as well as changing their shape:

| `beta_zr` | net perpetuity discount | AIP duration |
|---|---|---|
| -0.00014 (published) | 0.00723/mo = **8.67%/yr** | 58 mo |
| -0.00040 | 0.00983/mo = 11.79%/yr | 50 mo |
| -0.00048 | 0.01063/mo = **12.75%/yr** | 48 mo |
| -0.00056 | 0.01143/mo = 13.71%/yr | 47 mo |

**Predicted, against `bgnbase` (SR_max 0.2830, market 51.2%, room +0.0267, fair gap +0.0019):**

- **SR_max RISES toward the Hansen-Jagannathan bound** of `sigma_z = 0.4`, which `bgnbase` reaches
  71% of. The rate loading is large, common-signed and low-noise, so rotating price onto it is spanned
  better than the cash-flow channel is.
- **The market's share rises above 51.2%**, for the same reason: every asset is long-duration, so the
  rate factor is closer to a single common direction. That is condition 4 moving the WRONG way, and it
  is the same trap the disaster itself has.
- **Expected returns rise about 4 pp/yr.** `bgnbase` sits inside the 3.3-12.6%/yr band of every
  non-KP14 economy; at `beta_zr = -0.00048` BGN leaves it. **So the run should be paired with a
  premium-neutral control**: rotate `beta_zr` and scale `sigma_z` to hold the oracle's mean expected
  return at `bgnbase`'s, isolating shape from level. Both are parameter overrides. Note `sigma_z`
  enters `corr_zj` and the `sigmaj` band, and `prob_in_money_targets` refits `beta_star`/`scale`
  (`vasicek.py:61-70`), so the compensating move has side effects and must be read back.
- **The linear methods improve and the fair gap does NOT.** This is the load-bearing prediction, and
  the reason is BGN's own bound: `E[R]` is exactly affine in book-to-price and 1/price with
  rate-dependent coefficients. The rate channel's dispersion comes from the assets-in-place versus
  growth-option mix, which IS book-to-price and 1/price -- so rotating onto it enlarges a premium
  spread that a linear sort on those characteristics captures by construction. Predicted fair gap
  inside the established -0.008 to +0.004 band. **Falsification: a fair gap above +0.01 means the
  rotation argument in `docs/RESULTS.md`'s negative half is wrong.**
- **One asymmetry worth noting, and it is why this is still worth running.** The cash-flow channel's
  cross-sectional dispersion is sampling noise over live projects drawn from one common distribution
  (`docs/RESULTS.md`, capability matrix); the rate channel's is a persistent, characteristic-linked
  firm attribute. So the rotation moves premium from a noise-dispersed channel to a
  characteristic-dispersed one. That should raise `room` somewhat through the curvature of `D(r)` and
  `Jstar(r)`, and it is the first BGN economy whose dispersion is not sampling noise.

**So `bgnzr`'s value is not as a gap candidate. It is (i) the cheap gate on whether the rate channel
carries enough premium to be worth putting a disaster in, and (ii) the first direct test of the
rotation argument, which is currently untested reasoning.** The gate should be stated on the rate
channel's share of SR_max, not on the gap.

### 2. Counterparts of the `r` disaster in KP14 and GS21

Universalised, the `r` disaster is **a rare spike in the cost of capital that stops investment**. It
needs three ingredients, and each model has exactly two:

| ingredient | BGN | KP14 | GS21 |
|---|---|---|---|
| a priced discount-rate STATE | **yes** -- `r` Vasicek, priced once `beta_zr` moves | no, `r = 0.05` constant; the analogue is `gmult_y` on the price of x-risk, an inert logistic hook (`parameters_kp14.py:96-98`) | no, `r = 0.1/12` constant; the analogue is `gmreg` / `gam_x(x)` |
| a threshold investment stops at | **yes** -- acceptance 10% at `r=0`, 5% at `r=rbar`, nil at `rbar+3sd` | **no** -- growth options are exercised on arrival, there is no exercise threshold. The analogue is the arrival RATE: a rare `lambda`-collapse regime | **yes** -- `icut_up/dn` against `icost ~ U[0,2000]` (`gs_solve_reg.py:247-252`) |
| decay if investment stops | **yes**, `1 - pi` = 1.0%/mo | **yes**, `delta` = 0.833%/mo (`panel_functions_kp14.py:148-152`) | **no** -- `scale = 1.0` without investment, `k` is constant and `delta` is a FLOW cost in `pi_R`, not depreciation |
| output after 24 / 40 months of no investment | 0.786 / 0.669 | 0.818 / 0.716 | **1.000 / 1.000** |

So: **BGN is the only model with all three, and that is structural, not accidental** -- it is the pure
real-options model whose state IS the discount rate. The counterparts:

- **KP14: a rare innovation-drought regime.** Not a discount-rate spike but an arrival-rate collapse,
  and it reaches the same place: no new projects, the stock decays at 0.833%/mo, growth options go
  worthless, output falls 18% over 24 months. **It is a pure parameter override of machinery already
  carried through the solve** -- the two-state `high`/`low` chain is `NS = 2*NY` in `kp14_fd_vy.py:22-34`
  with its own `G_up`/`G_down` columns. Retune `mu_L` to 0.025/yr (1-in-40) and `mu_H` to 0.5/yr
  (2-year drought). **Check first whether `lambda_L` is overridable**: it is DERIVED at
  `parameters_kp14.py:40` from `E[lambda] = 1`, so a direct override is exactly the silently-discarded
  case `variants/common/readback.py` exists to catch. Setting `lambda_H` to 1.05 with `prob_H = 0.952`
  gives a drought `lambda` near 0.01 while preserving `E[lambda] = 1`. Predicted null in the CROSS
  SECTION (the exposure map is the growth-option share, affine, per KP14's bound) but it does deliver
  the output drop, and run on top of `vyx`'s type ladder it is a rotation: the drought hits the
  option share while the ladder loads on `y`.
- **GS21: a growth halt, not an output drop.** The threshold exists but the decay does not, so
  stopping investment freezes output in levels rather than shrinking it. This is why GS21's disaster
  has to be capital destruction applied directly -- and note that in BGN and KP14 the capital decay
  arrives free with the investment shutdown, while in GS21 it must be imposed.

### 3. How `beta_zr` reopens the hazard in BGN, and the three-way comparison

`beta_zr` reopens it by supplying the missing ingredient: the hazard channel needs a state that is
both PERSISTENT and PRICED. BGN's `nu` is i.i.d. so it is not a state; `r` is a state priced at
`sigma_z*corr_zr = -0.070`, and at `beta_zr = -0.00048` that becomes **-0.240, a 3.4x increase**. So
`lambda(r)` goes from earning almost nothing to earning something real.

But the channel's size is relative, and that is the finding. Writing the hazard term in log value as
`D_eff * lambda_bar * J`, with `D_eff = min(1/discount, 1/mean-reversion)` in months, its exposure to
the state is that divided by `sd(state)` and multiplied by `phi`:

| model | state | `D_eff` | `sd(state)` | hazard exposure (`phi=1`) | DIRECT exposure | ratio |
|---|---|---|---|---|---|---|
| **gs_bx** | `x`, transitory productivity | 59 mo | 0.0384 | **0.80** | 0.17-0.22 (measured) | **4.0x -- dominates** |
| **kp_vy** | `y`, priced OU | 34 mo | 1.00 | **0.018** | 0.006 / 0.020 / 0.040 by type | **0.4-3x -- rotates** |
| **bgn** | `r`, Vasicek discount rate | 20 mo | 0.0064 | **1.63** | ~17 (duration) | **0.10x -- drowned out** |

**The principle: the state-dependent hazard is a strong lever exactly where the priced state's DIRECT
effect is weak.** A discount-rate state moves value enormously (duration), so news about disaster risk
is a rounding error beside it. A transitory productivity state barely moves value at all, so
disaster-risk news dominates. This inverts the obvious intuition and it assigns each model its design:

- **GS21: put the disaster in the HAZARD.** `lambda(x)` is four times the direct channel, and GS21's
  near-total absence of room (+0.0006 to +0.0082) with a market at 94-97% is *because* its priced
  state has almost no direct effect. The hazard channel is the only lever that gives GS21 a large,
  heterogeneous, non-affine exposure to its own priced state.
- **BGN: put the disaster in the STATE.** A jump in `r` rides the large direct exposure; a hazard
  `lambda(r)` would be swamped by it at a tenth its size. This is exactly route (a).
- **KP14: either, and the hazard is the cheaper rotation.** At `phi = 1` the hazard channel is the
  same order as the middle type's direct exposure, so `lambda(y)` tilts the exposure map without
  swamping the ladder -- which is the rotation KP14's bound says is otherwise unavailable.

## Recommendation

**Universality is achievable in FORM and not in CONTENT, and the honest version of this experiment
says so.** One recipe -- hazard `1/480`, a 25% impact value loss, a stated disaster premium -- can be
written for all three models, and in two of them it is null for reasons provable before any solve
(KP14 by exact homogeneity in `x` and `z`; GS21 by duration, if the disaster goes in `x`). The
informative version attaches the disaster to each model's existing heterogeneity channel, and then it
is model-specific.

Sequence, and **step 0 was added 2026-09-17 after checking the footprint of the KP14 error**:

0. **Settle the KP14 `y` risk adjustment BEFORE any disaster work.** This is no longer a gate on
   step 4; it is a gate on the repository's headline. The error's footprint is *exactly* the set of
   economies that have a positive result, and nothing else:

   | economy | room | fair gap | affected by the `y` error? |
   |---|---|---|---|
   | kp_vy/vyg25 | +0.4108 | **+0.1636** | **yes** -- `gamma_v` 2.5, `beta_f` up to 0.14 |
   | kp_vy/vyx | +0.3606 | **+0.1251** | **yes** -- `gamma_v` 1.8, `beta_f` up to 0.14 |
   | the other eleven | +0.0006 to +0.0303 | +0.0034 down to -0.0080 | **no** -- `kpbase` has `beta_f = 0` and `gamma_v = 0`, and no BGN or GS21 economy has a `y` state |

   Those two rows are first and second by fair gap across all thirteen economies, and the third
   place is **37x smaller** (`g0235f`, +0.0034). So the `vy` route is not merely the most promising
   KP14 route -- it is the only positive result in the project, and it is the only place the error
   lands. The distortion is also larger in the bigger-gap economy: at `y = 0`, `A` is understated
   6.8/17.7/25.2% at `gamma_v` 1.8 and 9.1/21.7/28.8% at 2.5, with the implied price of `y`-risk
   running to 3.17 against a declared 1.8 in `vyx` and 4.13 against 2.5 in `vyg25`, and the
   cross-type spread in the implied price slightly wider in `vyg25`. That ordering is consistent
   with either story -- a real mechanism whose coefficients are distorted, or a distortion doing
   some of the work -- so it does not discriminate, which is why the regression below is needed
   rather than more arithmetic.

   **DONE 2026-09-17, and it bites.** The sizing run is complete: harness
   `variants/kp_vy/check_y_common_slope.py`, job `variants/kp_vy/run_ystep0_slurm.sh` (Sol 63535705,
   48 GB, 15 min), outputs `variants/results/kp_vy_yslope_{kpbase,vyx,vyg25}_s000.json`, full write-up
   in `docs/kp14_y_risk_adjustment.md`. It ran the registered common-slope regression (rejected at
   chi2(2) = 30,681 for `vyx`) and, alongside it, an exact test needing no elasticity approximation:
   whether the model's own prices satisfy `E^Q[P' + CF']/P = exp(r*dt)` under its own three stated
   prices of risk. `kpbase`, where no firm loads on `y`, satisfies it to 0.016 %/yr; `vyx` and
   `vyg25` miss it by up to 13.3 and 17.0 %/yr, ordered monotonically in `beta_f`, with no `gamma_v`
   out to 20 that closes the spread. About half of each exposed type's premium, and 52-58% of the
   cross-type premium SPREAD, is compensation for nothing. On `vyg25`'s saved moments (same solve ids
   the economy table reports) the unpriced part is 59% of `SR_max` and 61% of `SR_orth`.

   **So the branch below is the live one**, and it is the expensive arm: the corrected generator, a
   `y_max` of about 7 inside the -8.85 well-posedness wall, `NY` from 21 to about 41, re-solve all
   three KP14 tags and re-run their 30 seed jobs. That is a decision, not a next step -- it re-keys
   six solve ids and moves the project's only positive result. Nothing further in KP14 should be run
   until it is taken.

1. **A no-solve falsification probe (hours, zero cluster time).** Off existing panels: confirm
   the KP14 jump return is constant across firms under an `x` and a `z` rescaling; confirm the GS21
   value elasticities; compute BGN's transmission factor. Whichever of these fails, the design
   changes before anything is paid for. Write it to `_scratch/` and record the numbers in the spec
   notes. This is the `docs/NEXTUP.md` G3 pattern -- settle it cheaply.

   **The KP14 `A(y)` check is DONE and it failed** -- see "The KP14 Feynman-Kac point" below. What
   remains from it is the one-panel common-slope regression, which decides whether `vyx`'s room is
   partly a cross-type pricing inconsistency. That is now step 0, above, and it gates all KP14 work.
2. **`bgn_gam/bgnzr` -- rotate BGN's price of risk onto the rate channel.** Not a disaster at all: a
   pure parameter override, `beta_zr` from -0.00014 to about -0.00048, one 3-minute J\* rebuild, ten
   seeds at 40G. Zero code, zero re-key, and it is the precondition for a disaster in `r`. Promoted
   to first because it is the cheapest untried lever in the repository and `docs/RESULTS.md` already
   records `beta_zr` as an unexhausted bound.

3. **`gs_bx/gsdis` -- GS21 capital destruction.** New module `gs_solve_dis.py`, a parity test beside
   `tests/test_gs_solver_parity.py` pinning "differs from `gs_solve_reg.py` only in the disaster
   block", precommitted solve id, one solve, ten seeds at 32G. The strongest mechanism and the
   cheapest entry, and it re-keys nothing.
4. **`kp_vy/vydis` -- a jump in `y` on the `vyx` route.** The only model with a live gap to add to.
   **Rides on step 0 and must not precede it.** If step 0's correction goes in, KP14 is re-solved at
   the corrected generator and a wider `y` grid anyway, so `vydis` costs one more tag rather than a
   re-key of its own -- and the disaster's own y-dependent term would otherwise be added on top of a
   distortion that the term multiplies. If step 0 finds the effect does not bite, `vydis` still needs
   the re-key it was costed for: rebuild all six KP14 solve ids, verify byte-identity at disaster
   rate 0, re-run the 30 KP14 seed jobs to restamp, then the new economy.
5. **`bgn_gam/bgndis` -- a disaster in `r`, conditional on two gates**: `bgnzr` showing the rate
   channel is alive, and the no-exercise-corner derivation holding so the closed forms survive. If
   either fails this is a new BGN solver and should not be paid for.
6. **A productivity disaster in BGN: declined**, recorded as above.

Each new economy is a spec with a registered prediction, a precommitted solve id, and a falsification
line, per `docs/RESULTS.md` "Adding an experiment".

## How the disaster is priced: the hazard depends on the already-priced state

**No model derives a disaster premium, because none has a preference primitive.** KP14's SDF is three
constant prices of risk entering as drift adjustments; BGN's is `exp(-beta*gmult[s])`; GS21's is a
log-normal kernel in the `x` innovation.

**A state-dependent hazard does NOT price the disaster jump, and an earlier draft of this plan said
it did.** Conditional on date-t information `lambda(x_t)` is known, so the disaster indicator is
conditionally independent of the SDF innovation, the conditional covariance is zero, and the jump
earns nothing. Left there, a disaster adds physical variance with no compensation and **lowers**
every measured Sharpe. A jump premium still requires `lambda_Q != lambda_P` (or an explicit kernel
value on the disaster transition). The two are not substitutes; they are separate channels:

| channel | what prices it | what it delivers |
|---|---|---|
| the jump itself | `lambda_Q != lambda_P`, a new parameter | a level premium proportional to `lambda*J`, with exposure proportional to each firm's disaster loss |
| **the hazard's state dependence** | **nothing new** -- the existing price of the state's risk | a premium for disaster-risk NEWS, present in every month, with exposure proportional to each firm's disaster loss TIMES `lambda'(x)` |

**The second channel is the one worth buying, and it is free.** Because `lambda` is a function of an
already-priced state, the capitalised expected disaster loss makes firm value depend on that state
BEYOND its production exposure, and the existing kernel prices that extra sensitivity. This is
Wachter's second channel, obtained without a utility function.

**It is strong enough to matter, and the risk is that it is too strong.** In GS21 the disaster-risk
term in log value is about `A * lambda_bar * J` = 100 months x (1/480) x 0.25 = **0.052**, and its
x-sensitivity is that times `lambda'/lambda = phi / sd(x)` = `26*phi`. At `phi = 0.5` that is
`d log V/dx ~ 0.68` against a measured DIRECT exposure of **0.17-0.22** -- the hazard channel is
three times the entire productivity channel, and six times it at `phi = 1`. The premium is
`gamma_x * sigma_x * 0.68` = 2.9%/yr, so the +2%/yr target lands near `phi = 0.35`. Calibrate `phi`
to the premium target and report the implied hazard range across the grid; at `phi = 1` the hazard
would run from 0.02x to 55x its mean over +/-4 sd, which is an economy about disaster-risk news
rather than about production.

**Three reasons this is the right primary design.**

1. **It rotates the exposure map rather than scaling it.** Each firm's total exposure to the priced
   state becomes (production loading) + (disaster-loss sensitivity x `lambda'(x)`), and those are
   differently-shaped functionals of the firm state: production loading is `gs_bx` or the PVGO share,
   while disaster-loss sensitivity runs through leverage and default proximity (GS21) or the type's
   exposure ladder (KP14). A rotation is precisely what KP14's bound says is unavailable ("the two
   priced channels are nearly parallel across firms") and what GS21's bound excludes only away from
   the default boundary.
2. **It bends the map with the state.** A convex `lambda(x)` makes the rotation angle vary with `x`,
   which is condition 2 -- the design rule that produced `vyx` -- rather than condition 1 alone.
3. **It makes the rarity problem irrelevant to the primary outcome.** The premium sits in `mu_t`
   every month whether or not a disaster ever arrives. A 1-in-40-year event that is never realized in
   eight of ten seeds still moves all 1,250 evaluated cross-sections. This converts an experiment
   whose power was 2.3 seeds of ten into one measured at full power, and it is the pure peso design.

**Feasibility, which differs sharply because this needs a priced state that is SOLVED ON A GRID.**

- **GS21: cheap and clean.** `lambda(x)` is a 161-vector over the existing `x` grid, and the disaster
  enters as a third Bernoulli branch in exactly the pattern already there for the refinancing shock
  -- `Q0_new = xi * (...) + (1 - xi) * (...)` at `gs_solve_reg.py:215-216`, `:221-222` -- with the
  scalar `xi = 0.01` replaced by a broadcast over `x`. The kernel at `:137-138` is untouched: no giant
  innovation, no mass relocation. Endogenous leverage is the substantive uncertainty, and it is what
  the solve would settle: firms facing a high hazard de-lever, which is the state bending the
  exposure map, but de-levering far enough would collapse the dispersion the experiment needs.
- **KP14: the hook already exists and is inert.** `lambda(y)` is a 21-vector, and a y-dependent
  discount term is already plumbed into both `const_base_y` (`parameters_kp14.py:120`) and `rho_ty`
  (`:151`) as `gmult_y`, a logistic in `y` currently off at `g_lo = g_hi = 1.0`. The disaster-risk
  term enters at those same two places. The difference that matters: `gmult_y` is a COMMON multiplier
  and is therefore predicted null by the direction-invariance argument, whereas `lambda(y) * J_f` is
  scaled by each type's disaster loss.
- **BGN: structurally impossible, which independently confirms declining it.** Its priced shock `nu`
  is i.i.d. -- there is no state to condition a hazard on. `r` is a state but is priced only through
  `beta_zr = -0.00014` (bounded at +0.00027), so a hazard in `r` earns essentially nothing. The
  `gmult` regime is a state but its switch is unpriced.

**Recommended: state-dependent hazard with the jump left unpriced, as the first economy.** It needs
no new price of risk -- `phi` is a parameter of the driving force, like `gs_gamma_slope = 0.28`, not a
fourth `gamma` -- it carries all the cross-sectional content, and its power does not depend on
realizations. `lambda_Q != lambda_P` is then a second economy on the same solve structure, and the
contrast between the two is itself the informative comparison: a disaster premium versus a
disaster-risk-news premium, same hazard, same depth.

## The KP14 Feynman-Kac point: SETTLED, and it is not the term I first named

The KP14 route values a claim on `e^{beta_f y}` flows as `ebv * A(y)`, with `ebv = exp(bvf * yreg)`
(`panel_functions_kp14.py:124`) and `A = [diag(const_ty(f)) - Qy]^{-1} 1`
(`parameters_kp14.py:128-133`), `Qy` the PHYSICAL OU generator (`:101-117`). Substituting
`V = e^{by}A` into Feynman-Kac under Q gives

```
[rho + kappa_y*y*b + gamma_v*sigma_y*b - 0.5*sigma_y^2*b^2] A  -  L^Q A  -  sigma_y^2*b*A'  =  1
```

with `L^Q` drift `-kappa_y*y - gamma_v*sigma_y`. The code's discount bracket is exactly right. Its
OPERATOR is `L^P`, so what is missing is `(gamma_v*sigma_y - sigma_y^2*b) * A'(y)` -- and the dominant
piece is the **Girsanov drift, `gamma_v*sigma_y` = 1.507**, not the Ito cross term
`sigma_y^2*b` = 0.098 that an earlier draft of this section named. I had the small term and missed the
large one.

**What the code computes instead.** It applies the risk premium as a CONSTANT addition to the
discount rate, `b*gamma_v*sigma_y` per year forever. The correct Girsanov adjustment for a
MEAN-REVERTING state saturates: cumulatively `b*gamma_v*sigma_y*(1 - e^{-kappa_y*t})/kappa_y`,
bounded by `b*gamma_v*sigma_y/kappa_y`, not growing as `b*gamma_v*sigma_y*t`. So long-dated flows are
over-discounted, increasingly so in `b`.

**Established numerically, three ways** (vyx parameters: `rho` = 0.2297, `kappa_y` = 0.35,
`sigma_y` = 0.83666, `gamma_v` = 1.8):

| `b` | `A_code` / closed form | `A_code` / "constant-rate" closed form |
|---|---|---|
| 0.00 | **1.000** | -- |
| 0.02 | 0.931 | 1.0006 |
| 0.07 | 0.823 | 1.005 |
| 0.14 | **0.748** | 1.015 |

- The left column is the error: the three `vyx` types' claim values are understated by **6.9%, 17.7%
  and 25.2%**, and by **exactly zero at `b = 0`**, so `kpbase` is untouched and the distortion is
  monotone in the very exposure `vyx` varies.
- The right column is the diagnosis: the code reproduces the constant-rate formula to within 0.06-1.5%,
  the residual being the small Ito term.
- Confirmed independently by Monte Carlo under Q: **3.1028 (se 0.0005)** against the closed form
  **3.0982** at `b = 0.14`, `y = 0`; the 0.15% gap is Euler bias at `dt = 0.01`.
- **The 21-node grid is not the issue.** 21 nodes and 2001 nodes agree to four significant figures.

**Why this would go unnoticed, and it is the strongest part of the diagnosis.** The identical
construction is used for the `x` and `z` channels (`const_base_y:120`, the `mu_z`/`gamma_z` block in
`rho_ty:154`) -- and for those it is EXACT, because `x` and `z` are GBMs, where the Girsanov
adjustment really does accumulate linearly in `t`. The pattern was right in KP14's own two shocks and
was carried over to `y`, the one mean-reverting state, where it is not. `kpbase` has
`type_bv = [0.0]` and `gamma_v = 0`, so it is exact either way.

**Backing out the price of y-risk the code's own `A` implies**, against the documented `gamma_v = 1.8`:

| `b` | implied lambda at y = -2 / 0 / +2 |
|---|---|
| 0.02 | 4.18 / 4.20 / 4.11 |
| 0.07 | 3.68 / 3.63 / 3.52 |
| 0.14 | 3.26 / 3.17 / 3.04 |

Two readings. The level is 1.8x to 2.3x the documented price. **And it is not the same across types,
so no single SDF prices all three** -- the cross-type spread in expected returns contains a component
that is not compensation for a common factor, and the oracle maximises Sharpe over exactly that
cross-section.

**What is NOT established, and must not be asserted.** That this moves `vyx`'s +0.1251. The
compensation `type_theta = (A0_ty[0,i0]/A0_ty[:,i0])**bv_comp` (`:137`) is calibrated off `A0` at
`y = 0`, so it absorbs much of the LEVEL error: at `bv_comp = 1.2` the post-compensation level errors
are about 0.931 / 0.954 / 0.973, a ~4% cross-type spread rather than 25%. What survives compensation
is the y-SHAPE and the type-varying implied lambda. Nor does a single-claim test cover `A1`-`A3` or
the growth-option discount `rho_ty`, which share the construction.

**The decisive check, and it is a run rather than an argument.** On one saved `vyx` panel, regress
each type's conditional expected excess return on its y-exposure and test for a COMMON slope. A slope
that differs by type means part of the +0.36 room is not a risk premium. Until that is done, the
headline is not withdrawn and not confirmed.

**The fix, cheap to write and expensive to adopt.** Build the generator with drift
`-kappa_y*y - gamma_v*sigma_y + sigma_y^2*b` -- type-dependent, so `Qy` becomes one generator per type
-- keeping the discount vector as it is. Four `NY x NY` solves per type is free, and `kp14_fd_vy.py`
already solves `G` per type. **But the risk-neutral long-run mean of `y` is
`-gamma_v*sigma_y/kappa_y = -4.303`, outside the grid `[-3.5, 3.5]`, whose boundaries are
REFLECTING.** Under P, +/-3.5 is 3.5 stationary sd and reflecting is harmless; under Q the process
sits on the lower boundary. So a correct solve needs a wider or asymmetric `y` grid, which changes
`NY` or `dy`, which is solve precision -- uniform within a model by protocol. **All three KP14
economies re-solve together.** Avoiding ever solving under Q is very likely why the constant-discount
route was taken.

**Bearing on this experiment.** `vydis` adds another y-dependent term to the same discount, and the
omitted term multiplies `A'`, so the error grows with exactly the curvature the disaster introduces.
Step 4 should not proceed until this is resolved, and resolving it is a prerequisite for KP14 work of
any kind, not just for the disaster.

## Verification

- `python -m pytest tests/ -q` green throughout (264 now).
- `python variants/solve_impact.py --worktree` **must report nothing** after the GS21 work. If it
  names a GS solve, the new-module pattern was not followed. It WILL name the six KP14 solves after
  step 3; that is the expected, priced-in cost.
- Byte-identity of the rebuilt KP14 tables at disaster rate 0, on one machine (`LC_ALL=C sort`; the
  `Jstar_g0235d` 5e-15 cross-machine precedent).
- `python variants/penalty_gate.py` on each new economy before any number is read.
- Each new economy's per-seed disaster counts recorded, and the realized gap reported split by them.
- Each spec's registered prediction graded, including the two negatives predicted here: an `x`/`z`
  disaster in KP14 shrinking every gap, and a short `x` regime in GS21 not moving valuations.
- The `A(y)` cross-term check above, before step 3 and before another y-dependent term is added.
- `phi` calibrated to the premium target, with the implied hazard range across the grid reported in
  the spec: a hazard that reaches tens of times its mean is a different economy from the one asked
  for.

## What this does not do

It does not change the protocol -- no economy gets a different N, T, window, grid or `rf_cols`; a
disaster is a parameter and a driving force, which is what a new economy is allowed to be. It does
not restore the legacy crash modules. It does not reopen the ridge-grid question, though it does
predict the gate will have something to say about the disaster rows.
