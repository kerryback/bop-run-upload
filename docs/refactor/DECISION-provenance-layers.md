# Decision: what provenance machinery to keep, and what to revisit

**Status:** Phases 1–2 done 2026-09-08. Phase 3 **deferred, on a usage trigger, not a
date.** Raised by ASU; measured and scoped by Personal; Seth's call on the open question.

---

## The goal, in Seth's words

> (a) I see some results are good, and think "let's pursue this further"
> (b) I know what code and parameters produced those results. That's the whole goal here.

Everything below is judged against that and nothing else.

## What was actually wrong

Five subsystems, ~1,700 lines, and **none of them recorded (b)**. `solstamp`
content-addresses solves; `runstamp` links a run to the solves it consumed. Both are
real. Neither records the **code version**, and `grid_summary.csv` and the
`*_summary.csv` files — the ones you actually read to adjudicate — carried **no link to
anything at all**.

So the complexity was real, but the diagnosis "too much provenance" was wrong. It was
*the wrong provenance*: a lot of machinery for caching and staleness, and nothing for
traceability, which is the only part Seth asked for.

## Done (Phases 1–2)

`variants/common/provenance.py`, ~150 lines including its reasons. Every summary gets
`<summary>.prov.json`: git sha, dirty flag with the **diff inline**, untracked-code
warning, `argv`, `run_from`, the `*_OVERRIDES` env, consumed solve ids, host, library
versions, and a runnable `how_to_reproduce` line. Every summary CSV gains a `prov`
column — redundant per row on purpose, because the workflow is *reading one good row*
and a row copied into a notebook must not lose its pointer.

Answering (b) is now: **read one file, `git checkout` the sha, run the argv.**

## The reframe that matters

With the sidecar in place, **`solve_id` stops being the provenance story and becomes a
cache key nobody looks at.** The mental model goes from five interacting systems to
*one file I read, plus a cache that quietly stops me re-running six-hour solves.*

That is the simplification. It required deleting nothing.

## What must NOT be deleted, and why (measured)

ASU proposed replacing staleness checking with a git-sha comparison read out of the
`.npz` — "~20 lines". **That cannot work**, and the number is decisive:

> **207 commits in this repo. SIX touched `gs_solve_reg.py`.**

A sha-based staleness check refuses on all 207. Content-addressing refuses on 6 — a
**34x false-invalidation rate** against a solve costing ~5 h per type, 5 types. You
would hit it on a docs commit, re-solve for nothing, and within a week disable the
check — at which point the thing that caught both real bugs (the bare
`[ -f solution.npz ]` reusing a different economy's solve; the KP "converged" that
verified nothing) is gone.

A repo-wide sha structurally cannot distinguish *"the solver changed"* from *"the repo
changed"*. That distinction was used on 2026-09-07 to establish that `227f5e9` landing
mid-run left task 0's solve valid.

Two further corrections to the original proposal: the oracle JSON **does** already carry
`overrides` and `solves`, so parameters were recoverable and only the code version was
missing; and `utils/solfile_stamp.py` is **not** legacy — `utils_bgn/regen_solfiles.py`
imports it, so deleting it breaks BGN's Jstar regeneration.

## Phase 3, deferred: retire the PRECOMMITMENT layer

Candidates, ~400–500 lines — **not** the 1,400 originally proposed:

| candidate | why it is a candidate |
|---|---|
| `spec_hash` + `test_every_spec_hash_is_reproducible` | reimplements `git diff` for files already in git |
| `solves_pending` | status inside a hashed view; caused a spurious "hash drifted" on 2026-09-07 |
| `expected_solves` + the refusal path | precommitment, not traceability |
| the 7 `method` keys | **read by zero Python files**; not hashed; two specs differing only here are the same run |

**Keep specs themselves.** "Different specs" is the unit of experimentation — parameters
plus the question in words. It is the *hashing ceremony* around them that is the
candidate, not the file.

### The trigger is a count, not a date

The question is: **has precommitment ever earned its keep?** As of 2026-09-08 it has
**confirmed** something (5/5 bx7 ids on Sol) but never **caught** anything. Those are
different, and only the second justifies the machinery.

`run_oracle.py` now records `extra.spec_check` in every sidecar — one of `verified`,
`refused`, `unverifiable`, `not requested`. So the deferred decision is answered by
counting, not by remembering:

```bash
grep -ho '"spec_check": "[a-z ]*"' variants/results/*.prov.json | sort | uniq -c
```

**Revisit when there are ~20 real experiment runs**, and decide on what that prints:

- **any `refused` that turned out to be a genuine mismatch** → precommitment earned its
  keep. Keep it, and this document is the record of why.
- **all `verified`, no genuine catch** → delete the four rows above. The integrity value
  is real but it is not what Seth asked for, and it costs the churn documented in
  WORKING.md §33.
- **many `not requested`** → the layer is already being routed around in practice, which
  is its own answer.

### What would reverse the Phase 1–2 decision

If a sidecar is ever found to be *wrong* rather than merely absent — pointing at a sha
that does not reproduce the result — the fault is in `provenance.py`, not in the idea,
and `tests/test_provenance.py` (8 tests) should grow a case. One such bug already
shipped and was caught: `how_to_reproduce` recorded `getcwd()`, which follows
`run_oracle.py`'s chdir into `variants/<model>` and produced an instruction that could
not run.

## Sequencing

Nothing was ripped out. Phases 1–2 are purely additive and break nothing. Phase 3 should
happen **with evidence from real use**, which is now collected automatically.
