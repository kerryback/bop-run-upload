# gs_bx re-synced to the corrected GS21 calibration, and kappa_e turned on

ASU session, 2026-09-06. Task: `_scratch/handoff-gs21-table1-resync.md`. Evidence for the
calibration itself: `docs/refactor/FINDINGS-gs21-table1.md` (not restated here).

**Nothing committed. The array is written and not submitted.**

## 1. What changed

`variants/gs_bx/gs_solve_reg.py`, parameters only:

| | was | now | source |
|---|---|---|---|
| `delta` | 0.02 | `0.02 / 3` | Table I 0.02 is a per-quarter maintenance flow |
| `rho_x` | `0.96 ** (1/3)` | `0.95 ** (1/3)` | Table I and the body text both say 0.95 |
| `sigma_x` | conversion based on 0.96 | conversion rebased on 0.95 | downstream of `rho_x` |
| `kappa_e` | absent (= 0) | `0.025` | Table I benchmark; Seth chose it over the limit case |
| `tau`, `sigma_m` | 0.2, 5 | unchanged | Table I confirms both |

Verified against `config.py` directly, not through the test: **19 of 19 shared economic
parameters agree by exact float equality** — `g delta rho_x sigma_x rho_z sigma_z r gamma_x
x_bar tau phi kappa_e kappa_b xi bnum znum imin imax sigma_m`. The only remaining difference
is `xnum` 161 (CLI) vs `GS21_XNUM = 20`, which is the whitelisted cost/accuracy choice.

`python tests/test_config_parity.py` → **4/4 pass**.

## 2. kappa_e is not a one-line change: it makes the debt policy debt-dependent

The main tree charges the issuance cost on the **whole** current cash flow, per b'-choice,
*before* the max over b' (`utils_gs21/gs21_solve.py:309-311`, `GS21.m:253-260`):

```
P0_up_R[state, b'] = (1 + (prof0_up[state, b'] <= 0) * kappa_e) * prof0_up[state, b']
prof0_up[state, b'] = pi_R[b] + (1 - kappa_b)*Q0[b'] - Q0[b]
```

The variant did not have that shape. It factored the b-dependent level term out of the max:

```python
obj0 = (1 - kappa_b) * Q0 + EP0                 # 3-d, indexed by b' only
P0_up = pi_R - Q0 + obj0.max(axis=2)            # argmax independent of current b
```

That factorisation is **valid only at kappa_e = 0**. Once the cost multiplies the sum, the
level term `pi_R[b] - Q0[b]` is inside a nonlinearity and cannot be pulled out, so the
optimal refinancing target genuinely depends on current debt. So:

- `b0idx` / `bIidx` widen from `(z, x)` to `(z, x, b)`.
- the saved `b_refin_0` / `b_refin_I` widen from `(2, z, x)` to `(2, z, x, b)` — which is
  the shape `utils_gs21` has always used (`sdf_compute_gs21.py:56-57` reshapes to
  `(zpts, xpts, bpts)`).
- `gs_sim_bx.py` was doing a two-index lookup `b_refin_I_ts[f][s][zi, xt]`. That is now an
  interpolation in `b`, via the existing `_by_type`/`_interp_b_at`, linear in `b` to match
  `sdf_compute_gs21.py:104-105`, which deliberately keeps `method='linear'` for exactly
  these two step-valued tables.

**Answer to the handoff's question — exact or approximate: exact.** The variant now
evaluates the same maximand the main tree does, and the restructure is a re-association of
the same arithmetic, not an approximation. Measured below.

Cost is contained: the 4-d `(z, x, b, b')` array is built only on the 1-in-25 sweeps that
re-optimise. On the 24 frozen sweeps the stored `b0idx` gathers `Q0` and `EP0` directly with
a 3-d `take_along_axis`, so no large array is materialised.

## 3. Verification

Old code (`5e5ce1a`) vs new code forced back to the old calibration with `kappa_e = 0`,
same grid, same 400-sweep cycle-average exit — so the two are solving one economy and any
difference is the restructure alone:

| array | scale | max abs diff | relative |
|---|---|---|---|
| `P_up` | 198.9 | 1.14e-13 | 5.7e-16 |
| `Q0` | 81.4 | 8.5e-14 | 1.0e-15 |
| `icut_up` | 27.3 | 1.14e-13 | 4.2e-15 |

A few float64 eps — floating-point re-association, nothing else. Worst over all eleven saved
value/price arrays: 1.85e-13 absolute.

Policy indices match **exactly**: at `kappa_e = 0` the new `(z, x, b)` `b_refin_0` and
`b_refin_I` are constant along the b axis and equal the old `(z, x)` tables broadcast.

At `kappa_e = 0.025` the b-dependence is real: **42% of (z, x) states have a refinancing
target that varies with current debt**. Anything that assumed otherwise is now wrong —
which is why `gs_sim_bx.py` had to change rather than merely still running.

`gs_sim_bx.py` smoke-tested end to end against a new-format solution: reads `kappa_e = 0.025`
from the npz, loads the `(2, 8, 9, 20)` tables, `create_arrays(N=200, T=60)` returns a
finite panel.

## 4. Does turning kappa_e on materially change the gap results?

The handoff asked for this before the solves rather than after. Measured on a coarse probe
grid (znum 8, xnum 9, bnum 20, 400 sweeps), solving each exposure type twice:

| | gs_bx = 1.0 | gs_bx = 7.0 |
|---|---|---|
| states where the cost bites (refi branch) | 23.7% | 25.1% |
| mean `P_up` | −0.005% | −0.027% |
| mean `Q0` | −0.29% | −0.24% |
| mean `icut_up` | −0.054% | −0.059% |

Cross-type spread (bx7 − bx1), which is what the gap experiment actually measures:

| | kappa_e = 0 | kappa_e = 0.025 | change |
|---|---|---|---|
| mean `P_up` | 38.545 | 38.521 | −0.06% |
| mean `Q0` | 1.645 | 1.657 | **+0.77%** |
| mean `icut_up` | 5.442 | 5.438 | −0.07% |

**Judgment: it does not materially change the gap results.** The cost bites broadly — a
quarter of states, not a rare tail — but 2.5% of a negative flow is small, and the
differential across exposure types is under 1% on debt prices and under 0.1% on equity
values. That is consistent with the paper's own "only marginally improve our quantitative
results". The mechanism worth watching is `Q0`: the cost is convex at zero and high-`gs_bx`
types visit negative cash flow more often, so it widens the cross-type *debt-price* spread
while leaving equity values alone. If a gap result turns on leverage or credit spreads
rather than on equity returns, that is where it would show.

Caveat, stated plainly: this is a coarse grid. It truncates the productivity tails at 4 sd
with 8 z-points, so it understates tail-driven effects. It is a sanity check on the sign and
order of magnitude, not a substitute for the real solves. What it does establish is that the
mechanism is not rare — the bite frequency is already 25% at this resolution, so the small
effect is genuinely a small cost applied broadly, not a large cost applied rarely.

## 5. The five new solve_ids

All five distinct; none has an existing manifest, so every array task solves.

| outdir | overrides | **solve_id** |
|---|---|---|
| `sol_reg` | `{"gmreg":[0.6,3.0]}` | `1f83dbbdd54d40e1` |
| `sol_b25c` | `+ gs_bx 2.5, gs_ashift 0.225` | `cb68c3e9b84c6bf3` |
| `sol_b40c` | `+ gs_bx 4.0, gs_ashift 0.450` | `94c7042cd804cd10` |
| `sol_b55c` | `+ gs_bx 5.5, gs_ashift 0.675` | `89902d8cb7896fd9` |
| `sol_b70c` | `+ gs_bx 7.0, gs_ashift 0.900` | `7805f58251c416be` |

All encode `delta = 0.006666666666666667`, `rho_x = 0.9830475724915585`,
`sigma_x = 0.00704630524128845`, `kappa_e = 0.025`.

Cross-checked by running the real script and reading its own `[solstamp]` line, not only by
reconstructing the snapshot. That run also confirmed the superseded-artifact path works: it
printed that `sol_reg/solution.npz` belongs to `a8ef7a2522eda19d` and listed the parameter
differences, rather than silently reusing it.

Void, in order: `77eab49f46bfaa95` / `9ec0489872230a8f` / `af7c81605f1febb6` /
`c43c99f70ade9661` / `f68103d6d90be37d` (the 0.96 set, never solved) and
`a8ef7a2522eda19d` (the one artifact on disk, now two calibrations out of date).

## 6. The SLURM envelope still holds

Re-measured on the full grid (xnum 161, znum 200, bnum 20), same box, 4 threads, old code
and new code back to back:

| | old | new | change |
|---|---|---|---|
| policy sweep (1 in 25) | 1.86 s | 2.13 s | +14.5% |
| frozen sweep (24 in 25) | 1.82 s | 1.91 s | +5.0% |
| steady-state mean | — | — | **+5.4%** |
| peak RSS (`/usr/bin/time -l`) | 2.61 GiB | 2.74 GiB | +135 MB |

`-t 0-08:00`, `--mem=8G`, `--cpus-per-task=4` all stand. Projected walltime 3 h 24 m →
**~3 h 35 m**, still 2.2× inside the request; peak memory 2.74 GiB is 2.9× inside 8 G. The
+135 MB is the `(200,161,20,20)` = 103 MB b-by-b' array plus one temporary, as predicted.

One correction to the earlier findings: `FINDINGS-gs21.md` records "measured RSS 0.9–2.4 GB",
which was sampled with `ps` during the run. The true peak for that same code is 2.61 GiB.
Still comfortable at 8 G, but 2.74 GiB is the number to size against, and the SLURM script's
comment now says so.

Note the calibration change itself may move the sweep count in either direction — `delta` is
3× smaller and `x` less persistent. The 5600-sweep cap bounds it regardless, so the walltime
request is safe either way; the *convergence* outcome is not predictable from here.

## 7. Two things found on the way, neither mine to fix

**(a) `utils_gs21/panel_functions_gs21.py` imports `kappa_e` and never uses it.** Line 11
imports `GS21_KAPPA_E as kappa_e`; the four `prof_*` expressions at lines 111-116 compute the
cash flow with no `(1 + (prof <= 0)*kappa_e)` factor. At `GS21_KAPPA_E = 0` that was
invisible. At 0.025 the main tree's solver charges the cost and its panel does not, so
`Ecashflow` — and therefore `P_ex = P - Ecashflow`, the ex-dividend price the returns are
built on — is inconsistent with the value functions it is differencing. `sdf_compute_gs21.py`
and `loadings_compute_gs21.py` import it and do not use it either; whether that is correct
depends on what those two are differencing and I have not traced it.

This is exactly the failure class this whole effort has been chasing: a parameter that was
harmless at its old value and becomes load-bearing at its new one, with nothing checking. I
mirrored the fix in the variant (`gs_sim_bx.py`'s `div`), which is in scope. `utils_gs21/`
is not.

**(b) `tests/test_config_parity.py` does not check `sigma_x` or `kappa_e`.** The handoff said
it now checks `sigma_x`; it does not — `_parse_assignments` evals with `__builtins__`
stripped and no `np`, so `np.sqrt(...)` fails to parse and `sigma_x` is silently absent from
the parsed namespace. It is not in the checked name list either. `kappa_e` was never added.
Both agree today (verified above, by hand), so this is a gap in coverage rather than a live
divergence — but `sigma_x` is precisely the parameter that moves silently whenever `rho_x`
moves, which is the drift this test exists to catch. `tests/**` is out of my boundary.

## 8. Files touched

- `variants/gs_bx/gs_solve_reg.py` — parameters, `issue()`, the 4-d b'-optimisation, the
  frozen-policy gather, the cycle-average final pass, `kappa_e` saved to the npz
  (as its own key: `params` is unpacked positionally as a 15-tuple and cannot grow)
- `variants/gs_bx/gs_sim_bx.py` — reads `kappa_e` (defaulting to 0 for older solutions),
  interpolates `b_refin_*` in `b`, applies the issuance cost to `div`
- `variants/gs_bx/run_gs_bx7_slurm.sh` — re-measured resource justification, calibration note
- this file

Probes and comparison artifacts are in `_scratch/kappa_e_check/` (gitignored); the probe
manifests went to `_scratch/kappa_e_check/solfiles_probe/` via `BOP_SOLFILES`, so nothing
touched `experiments/solfiles/`.
