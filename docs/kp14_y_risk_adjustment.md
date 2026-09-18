# The priced state `y` and KP14's discount-rate construction

**Status, 2026-09-17, updated after the panel run.** A specification error in the `vy` route,
established numerically three ways and now **traced through to the reported ceiling**. It does not
touch `kp_vy/kpbase`, which prices to 0.016%/yr. It does touch `kp_vy/vyx` and `kp_vy/vyg25`, the two
economies carrying the only complexity gap in `docs/RESULTS.md`, and it is large there: at each
economy's own declared SDF the three exposure types earn +2.6 / +8.1 / +13.9 %/yr (`vyx`) and
+3.6 / +10.8 / +18.4 %/yr (`vyg25`) more than that SDF prices, and no single `gamma_v` removes the
spread. Stripping `vyg25`'s expected returns of the part no priced factor explains cuts `SR_max` on
the evaluation window from 1.4697 to 0.5996. See "The check that decides whether a result moves" at
the end, which is now run rather than proposed.

**No number in RESULTS.md is withdrawn by this file**, and that is a deliberate limit, not an
omission: what is measured is how much of the CURRENT ceiling is not a risk premium, which is not the
same as what the corrected economy would report. The corrected model has different prices, different
exposures and a different covariance matrix, and only a re-solve gives its numbers. The decision to
pay for that re-solve is recorded in `docs/plan-before-home-20260917.md` step 0.

Reproduce the algebra with `python variants/kp_vy/check_y_risk_adjustment.py` and the panel result
with `variants/kp_vy/check_y_common_slope.py` (see the last section for the exact invocation).

Paper references are to Kogan and Papanikolaou, "Growth Opportunities, Technology Shocks, and Asset
Prices", *Journal of Finance* 69(2), 2014, by its own equation numbers.

## The finding in one paragraph

KP14's assets-in-place coefficient is a **constant**, the paper's equation (11):
`A = 1/(r + gamma_x*sigma_x + delta - mu_x + theta_c)`. The risk premium for aggregate risk appears
as `gamma_x*sigma_x` **added to the discount rate**, and that is exact because the paper's priced
shocks are geometric Brownian motions. The `vy` route added a priced, *mean-reverting* state `y` that
firms load on exponentially as `exp(beta_f*y)`, and generalised (11) by keeping the paper's shape --
a premium added to the discount -- while replacing the scalar with a resolvent over a grid in `y`.
For a mean-reverting state that shape is not exact. The premium charged is
`beta_f*gamma_v*sigma_y` per year at every horizon; the correct Girsanov adjustment
**saturates** at `beta_f*gamma_v*sigma_y/kappa_y`. Equivalently, in the language of the paper's
Proposition 3: the code charges the premium on a `y`-exposure of `beta_f`, when the actual exposure
of firm value is `beta_f + A'(y)/A(y)`, about **half** of `beta_f`. The result is that assets-in-place
values are understated by 6.8%, 17.7% and 25.2% for `vyx`'s three types, by exactly zero for a type
with `beta_f = 0`, and that the price of `y`-risk the economy actually embodies is 1.7x to 2.3x the
`gamma_v` it declares -- and **is not the same number for the three types**.

## What the paper does, and why its construction is exact

The paper has three productive shocks and two priced ones.

| paper | process | eq. | priced? | how it enters output |
|---|---|---|---|---|
| `x` | GBM, `dx = mu_x*x dt + sigma_x*x dB_x` | (4) | **yes**, `gamma_x = 0.69` | multiplicatively, exponent 1 |
| `z` | GBM, `dz = mu_z*z dt + sigma_z*z dB_z` | (8) | **yes**, `gamma_z` | through `K*` and `PVGO` |
| `eps` | CIR, mean-reverting to 1 at rate `theta_eps` | (2) | no (`dB_f . dB_x = 0`) | **linearly** |
| `u` | CIR, mean-reverting to 1 at rate `theta_u` | (3) | no (`dB_j . dB_x = 0`) | **linearly** |

The kernel is equation (9), `dpi/pi = -r dt - gamma_x dB_x - gamma_z dB_z`, with constant prices of
risk. Output is equation (1), `y_fjt = eps_ft * u_jt * x_t * K_j^alpha` -- linear in each of the
three. The project value, equation (10), is `p = A(eps, u) * x * K^alpha` with, equation (11),

```
A(eps,u) =        1/(r + gamma_x*sigma_x + delta - mu_x)
         + (eps-1)/(r + gamma_x*sigma_x + delta - mu_x + theta_eps)
         +   (u-1)/(r + gamma_x*sigma_x + delta - mu_x + theta_u)
  + (eps-1)(u-1)/(r + gamma_x*sigma_x + delta - mu_x + theta_eps + theta_u)
```

Every denominator is a constant, and every term in it is exact for a specific reason:

- **`gamma_x*sigma_x` is exact because `x` is a GBM entering with exponent 1.** Value is linear in
  `x` and `A` does not depend on `x`, so the elasticity of `VAP` to `x` is exactly 1 at every date and
  in every state. A constant elasticity times a constant price of risk is a constant premium, and for
  a GBM the Girsanov drift adjustment accumulates linearly in the horizon -- so "add it to the
  discount rate" and "change measure" are the same operation.
- **`theta_eps` and `theta_u` are exact because `eps` and `u` enter LINEARLY.**
  `E[eps_s - 1 | eps_t] = (eps_t - 1)*exp(-theta_eps*(s-t))`, so adding `theta_eps` to the discount
  reproduces the decay exactly. No risk adjustment appears at all, because these shocks are
  orthogonal to `dB_x` and unpriced.

So the paper contains two cases -- a **priced GBM**, and an **unpriced mean-reverting state entering
linearly** -- and the add-to-the-discount construction is exact in both. It never has to price a
mean-reverting state, and it never has a state entering the cash flow non-linearly.

Proposition 3's proof, equation (A13), states the mechanism in one line:

```
E[R^vap] - r = -cov(dVAP/VAP, dpi/pi) = gamma_x*sigma_x
```

The premium is the price of risk times the exposure. It is a constant only because the exposure is.

## What the `vy` route added

`vy` introduced a fourth state, absent from the paper: a stationary Ornstein-Uhlenbeck `y` with
`kappa_y = 0.35` and unit stationary sd, priced at `gamma_v` (1.8 in `vyx`, 2.5 in `vyg25`), which
firm types load on **exponentially**:

```python
ebv = np.exp(bvf[None, :] * yreg[:, None])     # e^{beta_f y_t}   panel_functions_kp14.py:124
VAP  = (x*ebv)[...] * K**alpha * _Aval(...)    #                  panel_functions_kp14.py:182
PVGO = z**(alpha/(1-alpha)) * x*ebv * lambda_f * (...)   #        panel_functions_kp14.py:196
```

`beta_f` is 0.02 / 0.07 / 0.14 in `vyx`, 0 in `kpbase`. To value a claim whose flow carries
`e^{beta_f*y}`, the module generalises (11) from a scalar to a resolvent over a 21-node `y` grid:

```python
Qy = _build_Qy()                                        # parameters_kp14.py:101-117, PHYSICAL OU
const_base_y = r + gamma_x*gm_grid*sigma_x + delta - mu_x          # :120  -- (11)'s denominator
def const_ty(f):                                                   # :122-126
    b = type_bv[f]
    return const_base_y + b*gamma_v*sigma_y + b*kappa_y*y_grid - 0.5*b**2*sigma_y**2
def _solve_coeff_ty(f, theta_c):                                   # :128-129
    return np.linalg.solve(np.diag(const_ty(f) + theta_c) - Qy, np.ones(NY))
```

The reduction is right: at `b = 0` and `gamma_v = 0`, `const_ty` is flat in `y`, a generator
annihilates constants, and `A0 = 1/const_base_y = 1/0.2297 = 4.3535` -- **the paper's (11) exactly**.
`kpbase` is therefore unaffected by anything below, consistent with its 21 integral tables agreeing
across `y` to 4e-12.

The three added terms map onto (11) as follows. `b*kappa_y*y_grid` is the Ito drift of `e^{b*y}`, the
analogue of `-mu_x`, and being state-dependent it is what forces `A` to become a function of `y` and
the scalar to become a resolvent. `-0.5*b**2*sigma_y**2` is the Ito variance term, which has no
analogue because `x` enters with exponent 1. And `b*gamma_v*sigma_y` is the analogue of
`gamma_x*sigma_x` -- **the paper's pattern, applied to a state that is not a GBM.**

## Where it breaks

Substituting `V = e^{b*y} A(y)` into Feynman-Kac under Q gives

```
[rho + kappa_y*y*b + gamma_v*sigma_y*b - 0.5*sigma_y^2*b^2] A  -  L^Q A  -  sigma_y^2*b*A'  =  1
```

where `rho = const_base_y` and `L^Q` has drift `-kappa_y*y - gamma_v*sigma_y`. The residual of the
closed-form solution in this equation is **1.00000000** at every `y` tested, so the equation is the
right one.

The module's **discount bracket is exactly correct**. Its **operator is `L^P`**, the physical
generator. What is missing is therefore

```
(gamma_v*sigma_y - sigma_y^2*b) * A'(y)
```

whose dominant piece is the **Girsanov drift, `gamma_v*sigma_y = 1.507`** -- not the Ito cross term
`sigma_y^2*b = 0.098`. Two equivalent readings of the consequence:

1. **As a horizon effect.** The module charges the premium as a constant rate, a cumulative
   `b*gamma_v*sigma_y*t`. The correct cumulative adjustment is
   `b*gamma_v*sigma_y*(1 - exp(-kappa_y*t))/kappa_y`, which **saturates** at
   `b*gamma_v*sigma_y/kappa_y`. A mean-reverting state's risk does not accumulate with horizon the
   way a random walk's does, so long-dated flows are over-discounted.
2. **As an exposure error, which is Proposition 3's language.** Value is `e^{b*y}*A(y)`, so its
   `y`-exposure is `sigma_y*(b + A'(y)/A(y))`, not `sigma_y*b`. Because `A'(y) < 0` -- the effective
   discount rises with `y` -- the second term offsets part of the first. Measured:

   | `beta_f` | actual `d log V / dy` | as a share of `beta_f` |
   |---|---|---|
   | 0.02 | 0.0085 | 43% |
   | 0.07 | 0.0341 | 49% |
   | 0.14 | 0.0773 | 55% |

   **The premium is charged on roughly twice the exposure the firm actually has.** And the premium
   depends on `A` while `A` depends on the premium, so the two must be solved jointly -- which is
   precisely what putting the Girsanov drift into the operator does.

## The measurement

`vyx` parameters, `rho = 0.2297`, `kappa_y = 0.35`, `sigma_y = 0.83666`, `gamma_v = 1.8`:

| `beta_f` | `A_code` | closed form | ratio | `A_code` / constant-rate closed form |
|---|---|---|---|---|
| 0.00 | 4.3535 | 4.3535 | **1.000** | -- |
| 0.02 | 3.8519 | 4.1351 | 0.932 | 1.0006 |
| 0.07 | 3.0053 | 3.6511 | 0.823 | 1.005 |
| 0.14 | 2.3170 | 3.0982 | **0.748** | 1.015 |

at `y = 0`; the ratio runs 0.726 to 0.769 across `y` in `[-2, 2]` for `beta_f = 0.14`.

- **The error** is the third column: 6.8% / 17.7% / 25.2%, and exactly zero at `beta_f = 0`. It is
  monotone in the one exposure `vyx` varies.
- **The diagnosis** is the fourth: the module reproduces the constant-rate closed form to within
  0.06-1.5%, the residual being the small Ito term.
- **Independent confirmation.** Monte Carlo under Q at `beta_f = 0.14`, `y = 0`:
  **3.1028 (se 0.0005)** against the closed form **3.0982**; the +0.15% gap is Euler bias at
  `dt = 0.01`.
- **Not a discretisation artifact.** The 21-node grid and a 2001-node grid agree to four significant
  figures. `NY = 21` is fine for what the module is solving; what it is solving is the issue.

### The implied price of `y`-risk, and why it is not one number

Requiring the Feynman-Kac equation to hold and solving for the `lambda` that would make the module's
own `A` correct gives `lambda_f = gamma_v * beta_f / (beta_f + A_f'/A_f)`:

| `beta_f` | implied `lambda` at y = -2 / 0 / +2 | `lambda / gamma_v` |
|---|---|---|
| 0.02 | 4.18 / 4.20 / 4.11 | 2.32 / 2.33 / 2.29 |
| 0.07 | 3.68 / 3.63 / 3.52 | 2.04 / 2.02 / 1.95 |
| 0.14 | 3.26 / 3.17 / 3.04 | 1.81 / 1.76 / 1.69 |

Two readings, and the second is the serious one.

- **The level.** The economy prices `y`-risk at 1.7x to 2.3x the `gamma_v` its spec declares. `vyx`
  is documented as pricing `y` at 1.8; it prices it at about 3.2 to 4.2.
- **The dispersion.** All three types load on the *same* Brownian `dB_y`, so a single stochastic
  discount factor requires a single `lambda`. It is not single: after allowing for a common rescaling
  the types differ by about 20%. To that extent the cross-type spread in expected returns is not
  compensation for a common factor, and `sdf_compute_kp14.py` builds `max_sr` as the
  minimum-variance-efficient portfolio of exactly that cross-section.

### What this does to Proposition 3

The paper's (23) is affine in the growth-opportunity share, `E[R] - r = gamma_x*sigma_x +
(alpha/(1-alpha))*gamma_z*sigma_z*PVGO/V`, and (A15) makes the firm premium a value-weighted average
of its two components' premia. Adding `y` extends this correctly in principle: the firm gains a third
term, `gamma_v*sigma_y` times a value-weighted average of the `y`-exposures of `VAP` and `PVGO`, each
of which carries its type's `beta_f` plus a correction from the curvature of `A` and `G` in `y`. The
premium remains affine in the `PVGO` share *within* a type, with a type-dependent slope and
intercept, and that two-dimensional structure is the mechanism `vyx` was built to exhibit.

The error does not remove that structure. It distorts the coefficients, by a factor that is large
(about 2x) and type-dependent in a way no single price of risk reproduces -- i.e. along exactly the
cross-sectional dimension the experiment measures.

## What is established, and what is not

**Established.** Everything in the two tables above, reproducible by the named script, with the
closed form independently confirmed by Monte Carlo. The reduction to the paper's (11) at
`beta_f = 0`. That `kpbase` is unaffected.

**Not established, and not to be asserted.**

- ~~**That any reported number moves.**~~ **Settled by the panel run below, in the direction the
  compensation argument said it might not.** The compensation
  `type_theta = (A0_ty[0,i0]/A0_ty[:,i0])**bv_comp` (`parameters_kp14.py:137`) is calibrated off `A0`
  at `y = 0`, so it absorbs much of the *level* error -- at `bv_comp = 1.2` the post-compensation
  level errors are about 0.932 / 0.956 / 0.974 rather than 25%. It does **not** absorb the error in
  RETURNS, and it cannot: `type_theta` multiplies each type's cash flow and its price by the same
  factor, so it leaves every return unchanged by construction. What survives is the `y`-shape and the
  type-varying implied `lambda`, and those are what the mispricing below measures.

  What is still NOT established is what the CORRECTED economy reports. That needs the re-solve.
- **The growth-option side.** `rho_ty` (`:151-156`) shares the construction and was not tested here;
  its `z` block, `-alpha/(1-alpha)*(mu_z - gamma_z*sigma_z - 0.5*sigma_z**2)`, is a GBM term and is
  exact.
- **`A1`-`A3`.** Only the `theta_c = 0` coefficient was tested. The `(eps-1)`, `(u-1)` terms add a
  constant to the discount, which is the paper's exact treatment, but they inherit whatever is wrong
  with the `y` operator.

## The fix, and the constraint it runs into

Put the Girsanov drift in the generator: build it with drift
`-kappa_y*y - gamma_v*sigma_y + sigma_y**2*b`, type-dependent, so `Qy` becomes one generator per
type, keeping `const_ty` as it is. Four `NY x NY` solves per type is free, and `kp14_fd_vy.py`
already solves `G` per type.

It converges to the closed form -- **but only on a wider `y` grid**, because under Q the process no
longer sits near zero. At `gamma_v = 1.8` the risk-neutral long-run mean of `y` is
`-gamma_v*sigma_y/kappa_y = -4.30`, outside the current grid `[-3.5, 3.5]`, whose boundaries are
reflecting:

| `y_max` | `A` at y = -2 | y = 0 | y = +2 |
|---|---|---|---|
| 3.5 (current) | 3.3013 | 2.9267 | 2.5645 |
| 5.0 | 3.5585 | 3.0625 | 2.6455 |
| 6.0 | 3.6143 | 3.0920 | 2.6630 |
| 7.0 | 3.6253 | 3.0978 | 2.6665 |
| 8.0 | 3.6262 | 3.0983 | 2.6668 |
| closed form | 3.6262 | 3.0982 | 2.6667 |

So `y_max` must reach about 7. **And it cannot go much further**: `const_ty(f)` hits zero at
`y = -(rho + gamma_v*sigma_y*b - 0.5*sigma_y**2*b**2)/(kappa_y*b)`, which is **-8.85** at
`beta_f = 0.14`, `gamma_v = 1.8`. Below that the discount is negative and the resolvent is ill posed
-- the same failure mode that withdrew proposal K5, where `rho_ty` went negative for a loading of
-0.06 and "a negative discount removes the operator's dissipation" (`docs/RESULTS.md`, "Open
proposals"). At `gamma_v = 2.5` the Q-mean moves to -5.98 and the wall to -10.52, so `vyg25` needs a
wider grid still and the window between "wide enough" and "ill posed" narrows.

**Cost.** Widening `y_max` from 3.5 to 7 at the current node spacing takes `NY` from 21 to about 41.
That is solve precision, which `tests/test_protocol_is_uniform.py` holds uniform within a model, so
**all three KP14 economies re-solve together**: three tags of `G` plus `ntypes*NY` integral tables,
where the integral stage roughly doubles (`3*21 = 63` files becomes `3*41 = 123`). Then all 30 KP14
seed jobs re-run. Avoiding a solve under Q is very likely why the constant-discount route was taken.

## The check that decides whether a result moves: RUN 2026-09-17, and it moves

The check proposed here was a common-slope regression on one saved `vyx` panel. It was run, together
with a stronger test that needs no elasticity approximation, on the **protocol panel** (N 500, T 500,
burn-in 400, seed 0) for both affected economies and the control.

Harness `variants/kp_vy/check_y_common_slope.py`, cluster job
`variants/kp_vy/run_ystep0_slurm.sh` (Sol job 63535705, htc, 48 GB, 15 min for three tags).
Outputs `variants/results/kp_vy_yslope_{kpbase,vyx,vyg25}_s000.json`. Percentages a year below are
twelve times the monthly figure, not compounded.

### The test, and why it is exact rather than a decomposition

KP14 as implemented states three constant prices of risk on three Brownians, so any pricing function
consistent with that SDF must satisfy, firm by firm and month by month,

```
E^Q_t[ P_{i,t+1} + CF_{i,t+1} ] / P_{i,t}  =  exp(r*dt)
```

The script rebuilds the module's own four expected-payoff terms (`panel_functions_kp14.py:225-240`)
and re-takes their expectation under Q: `mu_x -> mu_x - gamma_x*sigma_x`,
`mu_z -> mu_z - gamma_z*sigma_z`, and the Gauss-Hermite `y` nodes displaced by
`-gamma_v*sigma_y*(1 - e^{-kappa_y*dt})/kappa_y`, the exact one-month OU Girsanov shift -- which is
precisely the term the code replaces with a constant addition to the discount. The month's cash flow
carries `x_t` and `e^{beta_f y_t}`, so it is predetermined and takes no risk adjustment; getting that
wrong is what makes an elasticity-based decomposition unreliable here, because it puts a firm-varying
`x`-exposure into what looks like a common term.

Two identities gate the run before anything is read off it, both at machine precision: the four terms
reproduce the module's own `erets` to **1.1e-15**, and `price` rebuilds from its assets-in-place and
growth-option parts to **2.5e-16**. What is differentiated is what KP14 prices.

### The zero point: `kpbase`

`kpbase` has `type_bv = [0.0]`, so no firm loads on `y` at all and its residual is the model's
ordinary discretisation error -- the 21-node resolvent, the `G` grid, the CIR steps -- with the `y`
channel absent.

| | mean E[R] - Rf |
|---|---|
| physical | +0.003911/mo = 4.69%/yr |
| after the `x` and `z` adjustments | **+0.000013/mo = 0.016%/yr** |

That is the accuracy this machinery reaches at `beta_f = 0`, and it is the only reason the rows below
can be read as a statement about `y` rather than about discretisation.

### The result

Mispricing at each economy's **own declared** SDF. 485 months, 219,295 firm-months, per type:

| type | `vyx`, gamma_v 1.8 | `vyg25`, gamma_v 2.5 |
|---|---|---|
| `beta_f` = 0.02 | +0.002156/mo = **+2.59%/yr** | +0.002939/mo = **+3.53%/yr** |
| `beta_f` = 0.07 | +0.006540/mo = **+7.85%/yr** | +0.008615/mo = **+10.34%/yr** |
| `beta_f` = 0.14 | +0.011112/mo = **+13.33%/yr** | +0.014143/mo = **+16.97%/yr** |
| spread, top minus bottom | **+10.75%/yr** | **+13.44%/yr** |
| control (`kpbase`) | +0.016%/yr | +0.016%/yr |

Three readings, and the third is the one that matters for `docs/RESULTS.md`.

**No single price of `y`-risk prices the three types.** Solving for the `gamma_v` that would zero each
type's mean mispricing gives **13.55 / 7.93 / 5.52** in `vyx` against a declared 1.8, and
**16.67 / 9.22 / 6.39** in `vyg25` against 2.5. Monotone in `beta_f`, and the gap does not close: the
trace was taken out to `gamma_v = 20`. The low-exposure type cannot be priced by any `gamma_v`,
because it has too little `y`-exposure for the price to have leverage on -- which is the signature of
a SHAPE error, not a level error, and is exactly what a constant-rate premium standing in for a
saturating one produces.

**Roughly half of each type's premium is compensation for nothing.** Against physical premia of
+0.00758 / +0.01455 / +0.02312 per month, the unpriced residuals are 28% / 45% / 48% of the premium
in `vyx`, and 33% / 47% / 47% in `vyg25`. Of the **cross-type premium spread** -- which is what the
`vy` route's exposure ladder exists to create -- **58% in `vyx` and 52% in `vyg25` is not a risk
premium**.

**The ordering matches the reported gaps.** `vyg25` has both the larger unpriced spread and the larger
fair gap (+0.1636 against `vyx`'s +0.1251). The previous version of this file noted that the
distortion being larger in the bigger-gap economy was consistent with either story and so did not
discriminate. With the mispricing measured rather than inferred, it now points one way.

### The registered regression

The common-slope regression named in `docs/plan-before-home-20260917.md` step 0 was run beside the
exact test, with `beta^y` differentiated through the module's own interpolants and a month fixed
effect absorbing the common terms. It agrees and rejects hard:

| | `vyx` implied gamma_v by type | chi2(2) |
|---|---|---|
| all 485 months | 5.43 / 1.08 / 2.50 | 30,681 |
| evaluation window, 125 months | 6.09 / 1.02 / 2.33 | 20,794 |

`vyg25`, all months: 6.50 / 1.49 / 3.55, chi2(2) = 42,721. Both are `p = 0` to machine precision, and
both are insensitive to the differencing step (`h = dy/2` and `h = dy` agree to three decimals).

The regression is the noisier instrument -- its per-type ordering is not monotone, because after
within-month demeaning the three slopes are identified partly off between-type level differences that
the exact test handles directly. **Lead with the Q-test; the regression is reported because it is what
was registered.**

### What it does to the ceiling

`variants/results/kp_vy_moments_vyg25_s000.npz` holds `mu` and `Sigma` per month for `vyg25` seed 0 at
the solve ids the economy table currently reports (`f41d052f1f960c4c, 8d1308e8f21723f8`), so the
ceiling can be recomputed without a solve. Stripping `mu` of the unpriced component and re-taking
`sqrt(mu' Sigma^-1 mu)`:

| | as reported | stripped | fall |
|---|---|---|---|
| `SR_max`, all 485 months | 1.4013 | 0.6061 | -56.7% |
| `SR_max`, evaluation window | 1.4697 | 0.5996 | **-59.2%** |
| `SR_orth` (what the market does not span), evaluation window | 1.4026 | 0.5527 | **-60.6%** |
| EW market Sharpe, evaluation window | 0.4383 | 0.2321 | -47.0% |

The reproduction is exact: recomputing `SR_max` from the saved moments returns 1.4013 and 1.4697
against the reported 1.4013 and 1.4697, and the two files' `keep` vectors were checked identical in
all 485 months before differencing.

`SR_orth` is the screen `docs/RESULTS.md` reads every proposed economy against -- a DKKM fit collapses
onto the market unless it is large. **Three fifths of `vyg25`'s is expected return that no priced
factor in the model explains.**

### What this does NOT establish

It does not say what the corrected economies report. `room` is the best nonlinear-basis const-theta
Sharpe minus the linear one, and recomputing it needs the full oracle scoring over the ridge grid,
not just `SR_max`. More fundamentally, the corrected model has a different generator, a wider `y`
grid, different exposures and a different covariance matrix; stripping mispricing from the current
panel measures how much of the CURRENT ceiling is spurious, which is a different question from what
the fixed model would report. Only the re-solve answers that.

It is also one seed. The mispricing is a structural property of the pricing functions rather than a
sampling quantity, and 219,295 firm-months leave no room for sampling doubt about its sign or
magnitude, but the `SR_max` counterfactual is seed 0 alone because it is the only saved moments file.

Sequenced with everything else this bears on in
[plan-before-home-20260917.md](plan-before-home-20260917.md), where it gates the proposed
`vydis` economy -- a disaster in `y` adds another `y`-dependent term to the same discount, and the
omitted term multiplies `A'`, which is exactly the curvature a disaster introduces.
