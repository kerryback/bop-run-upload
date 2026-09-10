# Steps 2 and 3 — what is broken, and what to do about it

Written 2026-09-09, after step 1 (the parameters → spec closure) landed. This is a
**working plan, not a record**: once these steps are done it gets folded into
`WORKING.md` and deleted, the same way the eleven documents in §43 were.

Nothing here is approved. Each section ends with the decision it needs from you.

---

# Step 2 — fix the errors before running anything

Four problems. The first three are in `gs_bx`, which is the economy we are about to run
and the only one never wired into the checked pipeline. The fourth is the hole step 1
left open on purpose.

## 2a. The bx7 spec contradicts itself

`experiments/specs/var-gs_bx-bx7-v3.json` says two incompatible things:

| field | says |
|---|---|
| `method.gs_ashift_ladder` | `"none (gs_ashift = 0 for every type)"` |
| `types.exposure.gs_ashift` | `[0.0, 0.225, 0.45, 0.675, 0.9]` |

**The `method` field is right and the `types` field is stale.** I checked all five solves
the spec pins, and every one of them records `gs_ashift = 0.0`. The ladder was set to zero
on 2026-09-07 and the `types` block was never updated. `gs_sim_bx.py` refuses to run at
all if any solution has a non-zero `gs_ashift`, so no wrong number can escape today — but
anyone reading the spec to learn what the economy is gets the wrong answer.

### Why this is not a one-line edit

`types` is inside the spec's **hashed view**, so correcting it changes `spec_hash`, and
`tests/test_specs_match_shell.py::test_every_spec_hash_is_reproducible` will fail until
the hash is recomputed. That test is doing its job: it is telling you that editing a
spec's parameters changes the spec's identity.

So there are two honest routes:

- **Edit v3 in place** and recompute `spec_hash`, recording the correction in `lineage`
  (which is excluded from the hash, so it is free to say what happened). Defensible
  because **bx7 has produced no results at all** — there is nothing in the repo stamped
  `var-gs_bx-bx7-v3`, so no published number changes meaning.
- **Create v4**, mark v3 `superseded_by`. Cleaner in principle: a spec that has been
  committed is never edited. Costs one more file, and `verify_against_spec` already
  refuses superseded specs, so nothing can accidentally use v3 afterwards.

**Recommendation: edit v3 in place.** The `types` block is a transcription of what the
five pinned solve ids already say unambiguously; it is a typo in the description, not a
change to the experiment. A v4 would imply the economy changed, and it did not.

> **Decision needed:** edit v3 in place, or cut a v4?

## 2b. `run_gs_bx7.sh` would burn ~25 hours producing an economy nothing can use

`variants/gs_bx/run_gs_bx7.sh` is the original entry point for bx7. It is now actively
dangerous:

1. **Its five solve lines still pass the old ashift ladder** (`gs_ashift: 0.225`, `0.450`,
   `0.675`, `0.900`). Those parameters do not match any solve we have, so the
   content-addressed cache **misses**, and the script re-solves all five from scratch —
   roughly 3.4 h each, ~17 h on this laptop and ~25 h on a Sol node.
2. **Then the run fails anyway.** `gs_sim_bx.py` raises `NotImplementedError` the moment
   it loads a solution with a non-zero `gs_ashift`. So the full cost is paid and nothing
   is produced.
3. It passes **no `--spec` and no `--seed`**, so even a corrected version would write
   unstamped, unseeded results — the exact shape of the stale bx7 files sitting in
   `results/` today.

### The complication: three specs point at this file

`var-gs_bx-bx7-v1`, `-v2` and `-v3` all name `run_gs_bx7.sh` as their
`provenance.source_script`, and `tests/test_specs_match_shell.py` re-parses it to check
five separate properties of v1. **Deleting the script breaks those tests**, and they are
the tests that exist specifically because this script and its spec drifted apart once
before.

There is a second, sharper problem with those tests: they check **v1**, which is
superseded. v1 and the script are stale *together*, which is why 194 tests pass while the
current spec is unguarded. The suite is green for the wrong reason.

### Three options

| | what happens | cost |
|---|---|---|
| **A. Fix the script** | zero the ashift ladder, add `--spec`/`--seed` | keeps a second, divergent way to run bx7 alongside the seed array — the duplication that caused this |
| **B. Delete it, repoint the tests at v3** | one way to run bx7 (the seed array); tests re-target `run_seeds_slurm.sh` | v1/v2 lose their `source_script`; set it to `null` as two specs already do |
| **C. Keep it solve-only** | strip the oracle/estimator lines, fix the ashift ladder, leave it as "how the five solves were made" | honest historical record, no duplicate run path |

**Recommendation: C, then B's test change.** The solves are the part worth keeping a
script for — they are the expensive, once-only step, and the file is the only record of
the five `GS_PARAM_OVERRIDES` strings that produced them. The oracle and estimator lines
are what duplicate `run_seeds_slurm.sh`, so those go. Then retarget the shell tests at
v3 so they guard the spec we actually use.

> **Decision needed:** A, B, or C?

## 2c. The shell-vs-spec tests guard the superseded pair

Covered above, but stated separately because it is a real finding rather than a
consequence: **`test_specs_match_shell.py` checks `var-gs_bx-bx7-v1` against
`run_gs_bx7.sh`.** Both are stale, they are stale in the same way, so they agree and the
test passes. Whatever is decided in 2b, these tests must end up pointed at **v3**.

## 2d. Overriding a derived parameter is still silently ignored (bgn_gam, gs_bx)

This is the hole step 1 deliberately left open, written up at the end of `WORKING.md` §42.

**The problem.** Every parameter module applies overrides with `globals().update(ov)`.
Any name that is *recomputed after that line* silently reverts to its derived value, and
any *misspelled* name is created and read by nothing. So a spec can declare a parameter
the economy never used — and the new step-1 check will happily confirm that the
environment matches that spec, because the environment does match. The check verifies
*spec == environment*; it cannot verify *environment == what the model actually used*.

`parameters_kp14.py` already closes this: it raises `"KP_PARAM_OVERRIDES entries that had
NO EFFECT"`. Verified still working. `bgn_gam/parameters.py` and the gs modules do not.
Names at risk in bgn: `prob_calm`, `Preg`, `nchars`, plus any misspelling.

### Why it was not fixed in step 1

`variants/bgn_gam/parameters.py` is **digest-bearing**: it is one of the two sources
hashed into `be222462dd017b2c`, the J\* solve that `var-bgn_gam-g0235-v2` pins and that
all ten committed g0235 seeds were built from. Adding four lines to it moves that
`solve_id`.

### What that actually costs — and one trap

The rebuild itself is trivial (minutes). The cost is the invalidation, and there is a
subtlety worth knowing **before** starting:

> The rebuilt J\* table will be **byte-identical**, because the guard changes no
> arithmetic. So after the rebuild, **two live manifests describe the same bytes**.
> `runstamp._match` returns whichever has the lexicographically smaller `solve_id`, so it
> is a coin flip whether a run resolves to the old id (spec check then **refuses**) or
> the new one (works).

This is not hypothetical: it already happened on 2026-09-07, and the fix is recorded in
the registry. `b80c6e516c132e13` is retired with the reason *"the table was rebuilt from
the same parameters and came out BYTE-IDENTICAL … it is now addressed as
be222462dd017b2c."*

**So the sequence has five steps, and skipping the fourth causes an intermittent failure:**

1. add the readback guard to `variants/bgn_gam/parameters.py`
2. run `python variants/solve_impact.py` first, to see the invalidation stated
3. `cd variants/bgn_gam && BGN_PARAM_OVERRIDES='…' python rebuild_jstar_gam.py`
4. **`python variants/solfiles.py retire be222462dd017b2c --reason "…" --superseded-by <new>`**
5. update `var-bgn_gam-g0235-v2`'s `expected_solves.jstar` and its `spec_hash`, and the
   live-id comment block in `run_seeds_slurm.sh`

The ten committed g0235 seed results then read as STALE against the new id. They are not
*wrong* — the economy is unchanged and the table is byte-identical — but
`runstamp.run_is_current` will say STALE, and re-running them is ~15 h of Sol time.

### The alternative

Add the guard to a file that is **not** digest-bearing. It cannot go in
`parameters.py`, but it could go in `run_oracle.py`: re-read the override JSON after the
model modules are imported and compare each requested key against the module's live
value. That catches exactly the same defect for **every** model at once, costs no solve
ids, and needs no re-solve. It is slightly less local than kp's version, and it cannot
help a producer run outside `run_oracle.py`.

**Recommendation: put it in `run_oracle.py` now, and defer the in-module guard** until
something else forces a bgn re-solve. That is the same "batch it" logic
REVIEW-override-shadowing used for kp's 88-minute integral rebuild, and it gets the
protection immediately at zero cost.

> **Decision needed:** guard in `run_oracle.py` now (cheap, general), or bite off the
> bgn re-solve and put it in `parameters.py` (local, but stales ten seeds)?

---

# Step 3 — wire `gs_bx` into the seeded runner

`variants/run_seeds_slurm.sh` is the one script that runs an economy the checked way:
solve verified before compute, per-seed checkpoint keyed on the solve, `--spec` passed so
the run cannot stamp itself with an economy it did not build. It accepts only
`SEED_SPEC=vyx|g0235`. We need `bx7` and `g28`.

**The good news: all six GS solves are on this laptop and intact**, verified by
`solfiles.py check --model gs_bx`, and I ran both economies end to end through the
spec-verified path at small N. The expensive part is already paid. What is missing is
plumbing.

## 3a. The blocker: the checkpoint is structurally broken for multi-solve economies

This one must be fixed or the array will misbehave every run, so it comes first.

`runstamp.run_is_current()` decides whether a seed is already done:

```python
got  = sorted(solve ids recorded in the run's _run.json)   # bx7: FIVE ids
want = live_solves(model, tag)                             # takes ONE tag
```

`bgn_gam` and `kp_vy` work because one tag covers all their solves. **`gs_bx` bx7 has five
tags** — `sol_reg`, `sol_b25c`, `sol_b40c`, `sol_b55c`, `sol_b70c` — one per exposure type.
Ask for any one of them and you get one id back, which never equals the five in the run
record. Verified:

```
live_solves(gs_bx, sol_reg)  = ['63fa7ebbc2db49ea']       # one
a bx7 run record carries     = five ids
=> run_is_current is ALWAYS False
```

**Consequence:** every seed reports STALE forever. The array re-runs all ten seeds on
every resubmission, and the checkpoint that exists to make a walltime kill recoverable
does the opposite.

**Fix:** let `live_solves` and `run_is_current` take a *list* of tags and union the
result; same for the `runstamp.py current` / `is-current` CLI. Small change, but it is a
correctness fix in shared code, so it wants its own test.

## 3b. Per-economy settings the script currently hardcodes

| | vyx / g0235 | bx7 | g28 |
|---|---|---|---|
| solve tags | 1 | **5** | 1 |
| `--kappas` | `0.001,0.01,0.1,1` | **eight values** | **eight values** |
| parameters reach the run via | `*_PARAM_OVERRIDES` | `GS_BX_SOLDIRS` / `_BETAS` / `_SHARES` | same |
| `GS_SIM_OVERRIDES` | n/a | must stay **unset** | must stay **unset** |

The kappa grid is in each spec's `estimation.kappas` and the script hardcodes the
four-value version. The gs specs ask for `0.001,0.01,0.03,0.1,0.3,1,3,10`. Left alone, gs
runs would be scored on the wrong ridge grid.

## 3c. A small correctness fix while we are in there

The script sets `WINDOW=${SEED_WINDOW:-360}` and passes it to the estimators, but calls
the oracle **without** `--eval_window`, which defaults to 360. Override `SEED_WINDOW` and
the oracle's window-restricted ceilings silently stop matching the months the estimators
score. Step 1 added a warning for exactly this, but the script should just pass
`--eval_window "$WINDOW"` so it cannot happen.

## 3d. The abort hint should say "fetch", not "re-solve"

When no live solve is found, the script prints a `SOLVE_HINT` telling you how to build it.
For gs that hint must **not** be "run the solver" — that is 17–25 h for a set of solves
that already exist and are published, content-addressed, to the shared Dropbox folder.
The right hint is:

```
python variants/fetch_solves.py --spec var-gs_bx-bx7-v3 --from "<ASU Dropbox>/.../solves"
```

## 3e. Keep the second copy pinned

`tests/test_specs_match_shell.py::test_seed_array_overrides_match_the_specs` exists
because the script restates each economy's parameters and a second copy drifts. It
currently covers bgn and kp only. Adding gs cases means extending it — and for gs the
comparison is against the spec's `env` block (three variables), not a JSON override blob,
so it needs a slightly different check.

## 3f. Cost is unmeasured for `gs_bx`, and the existing numbers do not transfer

Everything in the script's resource header was measured on `kp_vy` and `bgn_gam`. `gs_bx`
differs in ways that plausibly matter: bx7 loads **five ~100 MB `solution.npz` files** and
materialises their tables, it carries **six characteristics** instead of five (`lev`), and
its estimator stage runs **eight kappas** instead of four.

The repo's own history is the argument for measuring rather than projecting: the last
memory projection for `bgn_gam` was wrong by 3.5× and in the wrong direction.

**So: one seed at flagship before the array.** Submit `SEED_SPEC=g28` seed 0 at
N=500/T=500/window=360, read `sacct` for `Elapsed` and `MaxRSS`, and size the array from
that. g28 first because it is one solve and one type — the cheapest way to find out
whether the plumbing works — and because it asks a genuinely different question from bx7:
*can a gap open with zero nonlinear room?*

## Suggested order

1. **3a** — fix the multi-tag checkpoint (shared code, has to be right first)
2. **2a, 2b, 2c** — spec and script corrections, so the thing being wired is correct
3. **2d** — the derived-parameter guard, in whichever place you choose
4. **3b–3e** — add the two `SEED_SPEC` cases and extend the pinning test
5. **3f** — g28 seed 0 at flagship; measure
6. size and submit the arrays: g28 ×10, then bx7 ×10

Steps 1–4 are laptop work with no cluster time. Step 5 is the first real spend.

---

# What is deliberately NOT in this plan

- **The aggregate producer.** Real and still needed (`WORKING.md` §41: the ten-seed means
  exist only in `_scratch` and in prose). It belongs *after* the first gs results land, so
  it can be written against four economies rather than retrofitted from two.
- **`grid_summary.csv` and `oracle_summary.csv`.** Orphaned legacy. The aggregate producer
  replaces them; nothing here repairs them.
- **The precommitment layer.** `WORKING.md` §44 defers it on a counting trigger, currently
  24 `verified` / 0 `refused`. This campaign is the evidence that decides it. Leave it.
