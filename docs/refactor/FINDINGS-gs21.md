# Findings — GS21 (`gs_bx`) solve stage

Task brief: `docs/refactor/TASK-gs21-solves.md`. Run the `gs_bx` solves, measure them, and
prove the solve registry works on the one model whose artifacts are too big to commit.

Status: **complete** for the agreed scope (one solve, §3.2). Nothing committed.

---

## 0. Headline

1. **`gs_solve_reg.py` did not compile.** A duplicate `gmreg=` keyword made the file a hard
   `SyntaxError`, so GS could not be solved by anyone since 2026-09-04. Fixed. §1.
2. **`sol_reg` took 3 h 23 m**, against a README claim of "about a minute each" — wrong by
   ~180×. §3. (Already corrected in `variants/README.md` by the primary session.)
3. **The KP "converged-that-verified-nothing" shape is present, and it fired on this run.**
   `sol_reg` hit the 5600-sweep cap, cycle-averaged, and printed `converged`. Measured, the
   saved solution sits **1.7× above** the tolerance it reports meeting. §4.
4. **All five registry checks pass**, including `committable=False` on a 78.6 MB artifact —
   the path this task existed to test. §5.

---

## 1. The blocker: `gs_solve_reg.py` was not valid Python

```
File "variants/gs_bx/gs_solve_reg.py", line 266
    gs_bx=gs_bx, gs_ashift=gs_ashift, gmreg=gmreg,
                                      ^^^^^^^^^^^
SyntaxError: keyword argument repeated: gmreg
```

`gmreg=gmreg` is passed at **line 246** (`Mx=np.stack(Mx_s), gmreg=gmreg, psw=psw`) and again
at **line 266**, inside the identity block the 2026-09-04 solstamp rewrite added. Python
rejects duplicate keyword arguments at *parse* time, so the module could not be imported at
all — not a latent bug reachable only on the save path.

This is why `variants/gs_bx/sol_reg/` did not exist, and why `probe_out/` is empty and dated
09:45 today.

**Fix applied** (`variants/gs_bx/**` is in scope per the brief): dropped the line-266
occurrence. `gmreg` still appears exactly once in the `.npz`, same value. Verified with
`py_compile`, and `gs_sim_bx.py` never reads `gmreg` back out (`np.load` at line 23; `gmreg`
appears only in its docstring), so no consumer is affected.

**Provenance consequence.** `gs_solve_reg.py`'s source digest feeds the `solve_id`. Every id
below is tied to the *fixed* source; repairing this the other way (deleting line 246) would
yield five different ids for numerically identical solves.

The primary session has since recorded the deeper lesson in `10da142`: the probe used to
"verify" this file exec'd only a prefix above the broken line. **A probe that runs a slice of
a file has not shown the file runs.**

---

## 2. Five distinct solve_ids ✅

Registry item 1, confirmed at solve startup — needs no artifacts:

| outdir | `gs_bx` | `gs_ashift` | solve_id |
|---|---|---|---|
| `sol_reg`  | 1.0 (default) | 0.0   | `a8ef7a2522eda19d` |
| `sol_b25c` | 2.5 | 0.225 | `3a152ef104016746` |
| `sol_b40c` | 4.0 | 0.450 | `3787a726fb79f5ef` |
| `sol_b55c` | 5.5 | 0.675 | `b4c74ad9fe6ba9a9` |
| `sol_b70c` | 7.0 | 0.900 | `2890e0d6ea375f4f` |

All five logged `gamma per regime: [0.3 1.5]`, confirming the `gmreg=[0.6,3.0]` override
reached the kernel (`gamma_x = 0.5`, so `0.5 × [0.6,3.0] = [0.3,1.5]`).

---

## 3. Timing

Measured on a 10-core M-series laptop shared with the primary session.

| field | `sol_reg` |
|---|---|
| wall | **12,216 s = 3 h 23 m 36 s** |
| sweeps | **5600 (the cap)** |
| exit path | **cycle-average fallback**, not the convergence test |
| rate | 2.18 s/sweep mean; 2.76 s/sweep over the first uncontended 200 |
| artifact | 82,431,388 B (78.6 MB), `committable=False` |
| solve_id | `a8ef7a2522eda19d` |
| sha256 | `b595a3db760b36e70fecb3fe444cb8000fd0312503a651dcdea68bfcc891f77d` |

`sol_b25c` / `sol_b40c` / `sol_b55c` / `sol_b70c` were **not solved** (§3.2).

**Against the documented figure.** `variants/README.md` said "~a minute each". Measured
3 h 23 m — wrong by roughly **180×**.

**A caution about my own intermediate estimates.** I revised the projection four times
(70 min → 2–3.5 h → "converges near sweep 3400–4400" → capped at 5600). The per-200-sweep
`qerr` factor swung non-monotonically between 0.9964 and 0.9992, so no partial trace
predicted the exit. For an oscillating iterate, extrapolating a residual is not a forecast.
**The only reliable cost model here is sweep-cap × per-sweep-rate** — budget every GS solve
as 5600 sweeps unless it is proven to converge earlier. The per-sweep rate was never the
uncertain term; the sweep count was.

### 3.1 Concurrency: what the box actually tolerates

The brief's "2–3 at a time is reasonable" is right; **five at once is not.** Launching all
five drove load average to 16–24 on 10 cores and cut `sol_reg` from 470% to 79% CPU —
aggregate throughput *fell*. The cause is memory, not cores: `smooth()` builds
`Pn = P[..., None] + mn`, a `(znum, xnum, bnum, 161)` = `(200,161,20,161)` array
≈ **830 MB per call**, 4× per sweep. RSS ran 0.9–2.4 GB per solve; five concurrent peaked
near 7.6 GB with swap in use. **GS is memory-bound, not core-bound.**

All four aborted solves died on SIGTERM leaving **empty** outdirs and **no** manifest —
`solstamp.record()` runs only after a successful save. GS fails clean on interrupt, which is
*not* what `build_vy_tables.py` did (WORKING.md §17). Empty dirs removed.

### 3.2 Scope decision (Seth, 2026-09-05)

On the revised cost, the call was to **finish `sol_reg` only**:

- Registry item 1 needs **no** artifacts and was already done (§2).
- Items 2–5 each need only **one** >32 MB artifact, not five.
- The marginal value of the other four is the runnable bx7 economy — which PLAN §0.0
  **demotes** (22nd of 24 on `gap`, +0.0013) and Phase 7 blocks from scale-up pending the
  `gs_ashift` inconsistency.

**`gs_bx` therefore remains unrunnable** — `gs_sim_bx.py` loads all five `solution.npz` —
and that is a deliberate deferral, not an oversight. Completing it costs ~17 h on this
laptop (§6.6).

### 3.3 Incidental: the shared registry survived concurrent writers

While `sol_reg` ran, the primary session's `23bb2b492adf0e16 bgn_gam Jstar_g0235` landed in
the same `experiments/registry/`. Two sessions, two models, no collision or lost update. The
brief argued this was "safe by construction"; it is now observed.

---

## 4. Trap #2 audit — does `gs_solve_reg.py` repeat the KP G-solver failure?

**The fatal half is absent. The misleading half is present, and it fired on this run.**

**(a) Is the iteration count bounded?** Yes, but not where it looks. `for it in range(60000)`
is **dead code**: `if it == 5600: … break` is unconditional, so 5601 is the real ceiling.
Anyone budgeting from 60000 is off by 10×.

**(b) Is convergence verified before writing?** **No.** Two exits — the convergence test
(`qerr/vscale < tol*20` and `perr/vscale < tol*20` and `it > 100` and `it % 25 == 24` and
`max(prob_delta)/max(prob_up) < 1e-4`), and the `it == 5600` cycle-average fallback. Both
fall through to the same unconditional `print("converged: …")` and the same `savez`.

**(c) Absolute or relative tolerance?** **Relative** — divided by
`vscale = max(1.0, max_s |P_up_s|.max())`. This is the crucial difference from KP, which
demanded an *absolute* 1e-8 while `‖G‖ ~ 1e8`, i.e. a relative 1e-16, unsatisfiable below
float64 eps. Nothing of that kind is here; this criterion is reachable.

`if not np.isfinite(perr): raise RuntimeError` also fails loudly on NaN rather than saving
garbage.

### 4.1 Which path `sol_reg` took: the fallback

```
sweep 5200: qerr 5.231e-03 perr 5.918e-03  (11469s)
sweep 5400: qerr 6.572e-03 perr 5.919e-03  (11838s)
cycle-averaged; stopping
converged: sweep 5600, qerr 6.15e-03, perr 6.69e-03 in 12211s
```

`qerr` **rises** from sweep 5200 to 5400. The iterate is oscillating, not stalling — the
period-2 behaviour the cycle-average machinery exists to absorb, and the correct remedy. The
defect is not the averaging; it is that averaging and convergence emit the **same** message.

### 4.2 How far off was it, measured from the saved solution

| quantity | value |
|---|---|
| `tol` on the CLI and in the manifest | `1e-06` |
| effective threshold (`tol * 20`) | `2e-05` |
| `vscale = max(1, max\|P_up\|)` | 180.963 |
| achieved relative `qerr` | **3.398e-05** |
| achieved relative `perr` | **3.697e-05** |
| overshoot | **1.7× / 1.85×** |

This artifact is probably numerically fine — it missed an already-loosened threshold by under
2×. **That is what makes the defect dangerous rather than harmless:** the same path prints the
same `converged` and records the same manifest if the overshoot were 1000×, and nothing
downstream could tell.

### 4.3 The undocumented `tol * 20`

The test compares against `tol * 20`, so CLI `1e-6` requests `2e-5`. The factor appears
nowhere in the usage string, and `tol` is hashed into the `solve_id` — so the manifest records
a threshold 20× tighter than the one enforced.

**Nothing in §4 was changed.** All of it moves `solve_id`s or recorded output; the brief says
report first. Recommendations in §6.

---

## 5. Registry verification — all pass ✅

| # | check | result |
|---|---|---|
| 1 | five distinct solve_ids | ✅ §2 |
| 2 | `committable: false`, path/size/sha256 still recorded | ✅ 82,431,388 B > 33,554,432 B threshold |
| 3 | re-run exits early on a cache hit | ✅ `cached: … nothing to do`, **0 s** |
| 4 | `GS_SOLVE_FORCE=1` overrides the cache | ✅ bypassed, began re-solving; killed at sweep 0 |
| 5 | a deleted artifact reports MISSING | ✅ `missing (recorded 82,431,388 B …)`, **exit 1** |
| 5b | *(added)* a **corrupted** artifact is detected | ✅ `content changed since it was recorded`, exit 1 |

`check` returned exit 0 again after restore, so failure states are not sticky. For item 5 I
**moved** the file aside rather than deleting it — it cost 3 h 23 m.

**Item 2 is the one that mattered.** Before this run the registry held only `committable: yes`
solves (two `kp_vy`, one `bgn_gam`). `a8ef7a2522eda19d` is the first `NO`, recording a 78.6 MB
artifact that can never be committed — the whole design claim is that the record outlives the
artifact.

**Not executable: `solfiles.py diff`.** The brief asked for
`diff <sol_reg id> <sol_b70c id>`. Only one `gs_bx` manifest exists, since only one solve ran
and `record()` writes solely on success. The substance is already established by §2 — five
parameter sets, five distinct ids — but the `diff` code path remains **unexercised for
`gs_bx`**.

### 5.1 One real bug: a false diagnostic under `GS_SOLVE_FORCE`

```
[solstamp] sol_reg/solution.npz exists but its provenance is unrecorded; re-solving
```

The provenance **was** recorded, by a manifest whose `solve_id` matches exactly.
`GS_SOLVE_FORCE` falsifies the first condition at `gs_solve_reg.py:66-83`, control falls into
`elif os.path.exists(_solution)`, `_find_by_artifacts` finds the *matching* manifest,
`_prior["solve_id"] != _snap.solve_id` is False, and the `else` branch — written for the
unrecorded case — runs anyway. Harmless to the artifact, but a false statement about
provenance emitted by the provenance system.

---

## 6. Recommendations

None applied. Items 1–4 change `solve_id`s or recorded output.

1. **Record what was *achieved*, not only what was *requested*.** This is the gap the primary
   session named in `10da142`, and this run supplies the number: the manifest says
   `tol: 1e-06` for a solve whose achieved relative residual was `3.4e-05`. A capped solve and
   a converged one are indistinguishable in the registry.
   **`solstamp` already has the right home: the `extra` dict, recorded but deliberately not
   hashed.** Achieved quality should not change identity — two runs of one spec are the same
   solve — but it must be recorded. Concretely
   `extra={"outdir": …, "exit": "converged"|"cycle_capped", "sweeps": it, "qerr_rel": …,
   "perr_rel": …, "vscale": …}`. No schema change; machinery that already exists.
2. **Stop printing `converged` on the capped path.** One line. The exits should say different
   things. The cycle-averaging is correct and should stay; only the message is wrong.
3. **Resolve `tol * 20`** — compare against `tol`, or document the factor in the usage string.
4. **Delete the dead `range(60000)`** — the real cap is 5600, and that is the number a cost
   model needs.
5. **Fix the false "provenance is unrecorded" message** under `GS_SOLVE_FORCE` (§5.1).
6. **Budget GS as 5600 sweeps.** At ~2.2 s/sweep that is ~3.4 h per solve and **~17 h for a
   five-type bx7 economy** on this laptop — a live input to PLAN §0.0's recommendation to
   demote bx7, and to any decision to run it on the cluster instead.

---

## Appendix — scope notes

- **`bgn_gam` second task not started, deliberately.** The brief offers it as a follow-on, but
  the primary session claimed BGN plus the Phase 1 oracle wrapper. Staying off it avoids two
  writers on one artifact — the exact failure in WORKING.md §17.
- **Oracle/estimator stages not run.** `run_gs_bx7.sh` continues into `run_oracle.py` and
  `run_estimators.py`, which write shared `variants/results/` under `--tag bx7` — the
  coordination risk the primary session flagged. Scope here is the solve stage; the solves
  were driven individually, which also gives per-solve timing the shell script would not.
- **Files touched:** `variants/gs_bx/gs_solve_reg.py` (one line, §1),
  `variants/gs_bx/sol_reg/solution.npz` (new, untracked, 78.6 MB),
  `experiments/registry/a8ef7a2522eda19d.json` (new manifest, should be committed),
  `variants/results/logs/log_gs_*`, and this file. Nothing committed.
