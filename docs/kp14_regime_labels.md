# KP14 lambda-regime labelling

**Status:** fixed 2026-09-04. Requires re-simulation of every KP14 panel; does **not**
require re-solving (no solfile depends on the changed constant — verified, the stamp
guard passes unchanged).

## The defect

`KP14_MU_H` and `KP14_MU_L` are the two switching intensities of the growth-opportunity
arrival-rate regime. The codebase pinned the implied stationary P(high) in four places,
and they did not agree.

| site | expression | implied P(high) |
|---|---|---|
| `config.py:257` — `KP14_LAMBDA_L` | solves `w·λ_H + (1−w)·λ_L = 1` with `w = MU_H/(MU_H+MU_L)` | **0.3191** |
| `utils_kp14/kp14_fd.py:118` — `G_up` | `Gbar + (MU_L/(MU_L+MU_H))·D`, i.e. `Gbar + (1−P_H)·D` | **0.3191** |
| `utils_kp14/kp14_fd.py:119` — `G_down` | `Gbar − (MU_H/(MU_L+MU_H))·D`, i.e. `Gbar − P_H·D` | **0.3191** |
| `config.py:269` — `KP14_PROB_H` | `MU_L/(MU_H+MU_L)` | 0.6809 |
| `panel_functions_kp14.py:135` | `where(high, MU_H·dt, MU_L·dt)` — reads `MU_H` as exit-from-high | 0.6809 |

Numerically, with `MU_H = 0.075`, `MU_L = 0.16`, `LAMBDA_H = 2.35`:

```
lambda_L                        =  0.367187      (from config.py:257)
E[lambda] at P_H = 0.3191       =  1.000000      <- the intended normalisation
E[lambda] at P_H = 0.6809       =  1.717188      <- what was actually simulated
```

So the economy that was simulated had a mean growth-opportunity arrival rate **72% above**
the value the calibration was constructed to produce.

## Why 0.3191 is the correct reading

Two independent arguments, both decisive.

**1. Feasibility.** `LAMBDA_L` exists to enforce `E[λ] = 1`. Solving that at the competing
probability gives

```
lambda_L = (1 − 0.6809 × 2.35) / (1 − 0.6809) = −1.8800
```

a negative arrival rate. The normalisation is simply infeasible at P_H = 0.6809, which
means `LAMBDA_H = 2.35` itself presupposes P_H = 0.3191.

**2. Majority, and it is the non-arbitrary majority.** Two of the four sites are *derived
quantities whose algebra forces the answer* — the λ normalisation and the (mean,
difference) recombination in the finite-difference solver. Neither can be written down
without committing to a stationary distribution. The two dissenting sites are a
free-standing constant and a `where()` clause, both of which can be wrong without
anything else noticing.

## The convention, stated

> **`KP14_MU_H` and `KP14_MU_L` are ENTRY rates, named for the state they lead TO.**
> `MU_H` is the low→high rate; `MU_L` is the high→low rate.
> Hence `P(high) = MU_H/(MU_H + MU_L)`, and the rate of *leaving* the high state is `MU_L`.

For a two-state chain this is the ordinary continuous-time result — stationary probability
of a state is proportional to the entry rate into it — so the constant now reads the way
the names do.

## The fix

Deliberately made at the import boundary rather than in the twelve transition expressions
that use these rates, because every one of those already uses its local `mu_H` to mean
*exit-from-high* and is correct under that reading. Editing them individually would have
been twelve chances to introduce a typo; remapping once is provable by inspection.

1. `config.py` — `KP14_PROB_H` corrected to `MU_H/(MU_H+MU_L)`, and two explicitly named
   derived constants added:

   ```python
   KP14_EXIT_H = KP14_MU_L      # rate of leaving the HIGH state
   KP14_EXIT_L = KP14_MU_H      # rate of leaving the LOW state
   ```

2. `utils_kp14/panel_functions_kp14.py`, `sdf_compute_kp14.py`, `loadings_compute_kp14.py`
   — the shared import line changes from

   ```python
   KP14_MU_H as mu_H, KP14_MU_L as mu_L,
   ```
   to
   ```python
   KP14_EXIT_H as mu_H, KP14_EXIT_L as mu_L,
   ```

3. `utils_kp14/kp14_fd.py` is **unchanged**. It uses the raw entry rates and its
   recombination coefficients were already correct.

4. `config.py:257` is **unchanged**. It was right all along.

## Guard

`tests/test_kp14_regime_labels.py` — seven tests, all of which fail under the old
convention:

- `KP14_PROB_H` equals the entry-rate ratio
- `E[λ] = 1` to 1e-12 *(this is the test that would have caught the original bug — it
  returns 1.7172 under the old value)*
- `LAMBDA_L > 0`
- exit rates are the entry rates crossed
- the FD recombination coefficients agree with `KP14_PROB_H`
- a simulated chain using the module's own transition rule reproduces `KP14_PROB_H` to 0.01
- that same chain gives a realised `E[λ]` within 0.02 of 1.0

## Consequences

- **Every published KP14 number from the main pipeline** was produced under
  `E[λ] = 1.717`. Per decision §10.1, these are to be restated with the delta visible.
- `variants/kp_vy/parameters_kp14.py:10,33-34` carries the identical inconsistency and is
  **not yet fixed** — it is a separate tree, and per the standing decision the duplicate
  BGN/KP variant code is left in place until the later phases. It must be fixed before
  any `vyx` run, since `vyx` is the largest-gap economy in the grid.
- Both lines date to the initial commit (`fc0cb6f`, Kerry Back, 23 Jan 2026) and survived
  the deliberate `6c65bf4` "KP14 Table 2" parameter audit, which changed `GAMMA_X`,
  `SIGMA_EPS` and `SIGMA_U` but did not touch the regime block.

## The one thing that would overturn this

If Kogan–Papanikolaou (2014) Table 2 states `μ_H` as the *exit* rate from the high state
**and** gives `λ_H = 2.35`, then the two are jointly inconsistent with `E[λ] = 1` and the
error is in `LAMBDA_H` or in the normalisation itself, not in the labelling. Nothing in
the repo can distinguish these; it needs the paper. The fix above is the one that keeps
every derived quantity in the codebase mutually consistent and every arrival rate
positive, which is the strongest statement the code alone supports.
