# Task brief — GS21 (`gs_bx`) solve stage

For a second Claude working **concurrently on this same laptop** while the primary
session finishes the KP (`kp_vy`) integral tables. Scope is deliberately disjoint.

## Why this task

`gs_bx` is the one flagship economy with **no solve artifacts at all** —
`variants/gs_bx/sol_reg/` does not exist, so GS cannot run today. It is also the only
model whose artifacts exceed the 32 MB commit threshold (five solutions at ~85 MB), which
makes it the **only real test of the registry's `committable=False` path**: the manifest
has to outlive an artifact that can never be committed.

And `gs_solve_reg.py` was rewritten on 2026-09-04 to use `variants/common/solstamp.py`
but **has never once been executed since**. It is unproven code.

## What to do

```bash
cd variants/gs_bx
zsh run_gs_bx7.sh          # five solves, then the oracle + estimators
```

Or drive the five solves individually (each writes `<outdir>/solution.npz`):

| outdir | overrides |
|---|---|
| `sol_reg`  | `{"gmreg":[0.6,3.0]}` |
| `sol_b25c` | `{"gmreg":[0.6,3.0],"gs_bx":2.5,"gs_ashift":0.225}` |
| `sol_b40c` | `{"gmreg":[0.6,3.0],"gs_bx":4.0,"gs_ashift":0.450}` |
| `sol_b55c` | `{"gmreg":[0.6,3.0],"gs_bx":5.5,"gs_ashift":0.675}` |
| `sol_b70c` | `{"gmreg":[0.6,3.0],"gs_bx":7.0,"gs_ashift":0.900}` |

all at `xnum=161`, `tol=1e-6`.

**Measure and report the per-solve wall time.** `variants/README.md` claims "~a minute
each" — that figure is an unverified guess of the primary session's, and the equivalent
KP guess was wrong by two orders of magnitude. Replace it with a measurement.

## Then verify the registry actually works

```bash
python variants/solfiles.py list --model gs_bx
python variants/solfiles.py check --model gs_bx
python variants/solfiles.py show <solve_id>
python variants/solfiles.py diff <sol_reg id> <sol_b70c id>   # should differ in gs_bx, gs_ashift
```

Specifically confirm:
1. Five **distinct** solve_ids (they differ in `gs_bx`/`gs_ashift`).
2. `committable: false` on every one (>32 MB), and the manifest still records path, size
   and sha256 for each artifact.
3. Re-running a solve **exits early on the cache hit** instead of re-solving.
4. `GS_SOLVE_FORCE=1` overrides that.
5. Deleting a `solution.npz` makes `solfiles.py check` report it as MISSING rather than
   silently passing.

Already verified by the primary session, so don't redo: `xnum` and `tol` **do** reach the
hash (161/1e-6, 81/1e-6, 161/1e-4 give three distinct solve_ids), and `outdir` correctly
does **not**.

## Boundaries — please respect exactly

**Yours to edit:** `variants/gs_bx/**`, `variants/results/logs/log_gs_*`, and a findings
file at `docs/refactor/FINDINGS-gs21.md` (create it).

**Do not touch:** `variants/kp_vy/**`, `variants/common/solstamp.py`, `variants/solfiles.py`,
`config.py`, `main.py`, `utils*/`, `tests/**`, and above all
**`docs/refactor/WORKING.md`** — the primary session is appending to it live, and two
appenders will collide. Write to `FINDINGS-gs21.md`; it gets merged in afterwards.

`experiments/registry/` is shared but safe: one file per solve_id, and ids never collide
across models.

**Do not commit.** Seth commits himself. Leave the tree dirty.

**CPU — read this before starting.** The primary session's KP integral stage **saturates
the machine**: one integ job alone measures 684% CPU on a 10-core box, and six run
concurrently, for ~100 more minutes. Measured throughput is the same nested or serial, so
the box is genuinely full. Running GS solves now will slow both sides and will make your
own timing measurements meaningless.

**Prefer `TASK-repo-merge-assessment.md` first** (CPU-free), and start these solves once
the primary session reports the integral stage done. If you do start now, keep to one
solve at a time and treat the wall times as upper bounds, not measurements.

The five GS solves are independent, so once the box is free, 2–3 at a time is reasonable
— but see the trap below before running any two at once.

## Traps this repo has already sprung today

1. **Concurrent writers to one artifact.** Two builders were started against the same KP
   prefix twice; they halved each other's cores and could have handed a downstream stage a
   half-written table. `build_vy_tables.py` now takes a lock — **`gs_solve_reg.py` does
   not.** Never run two solves into the same `outdir`. Different outdirs are fine.
2. **A "converged" message that verified nothing.** The KP G solver iterated to an
   *absolute* tolerance of `1e-8` while `||G|| ~ 1e8` — demanding relative precision below
   float64 eps, so it was unsatisfiable by construction. On cap exhaustion it wrote the
   table anyway and printed `converged`. **Check whether `gs_solve_reg.py` has the same
   shape**: does its Gauss-Seidel loop have a bounded iteration count, does it verify
   convergence before writing, and is `tol=1e-6` compared against an absolute or a
   relative quantity? Report what you find; do not fix it without saying so first.
3. **mtimes and commit timestamps are not evidence** about what code a running process is
   executing — a long solve reads its sources once, at launch. The reliable check is
   numerical: recompute and compare.
4. **zsh does not word-split unquoted parameters.** `set -- $var` needs `${=var}`.
5. `timeout` does not exist on macOS; `setsid` does not either. To detach a process so it
   survives the terminal: `( nohup cmd > log 2>&1 < /dev/null & )`.

## Second task if the above finishes early

`variants/bgn_gam/` has three `Jstar_*.csv` on disk, **none manifested**. Running
`zsh variants/bgn_gam/run_g0235.sh` (or just its `rebuild_jstar_gam.py` step) records the
first BGN manifest. Same boundaries; report timing, since "~minutes" is also unverified.
