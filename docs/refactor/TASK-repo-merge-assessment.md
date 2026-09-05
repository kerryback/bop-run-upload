# Task brief — merge assessment: `bop-analyze-remote` into `bop-run-upload`

For a second Claude working **concurrently on this same laptop**. This task is chosen to
be **CPU-free**: the primary session's KP integral stage saturates the machine (one job
alone runs at 684% CPU, six run at once, ~100 min remaining). Anything compute-heavy will
contend. This is reading, comparison and writing only.

## The question Seth asked

> "Assess the cost of merging `~/GitHub/bop-analyze-remote` into `bop-run-upload`,
> considering the two repos' differing Python environments. We make the `bop-run-upload`
> env as small as possible to facilitate its compute and minimize module bloat as much as
> possible — but if you say that's not a big deal, then perhaps a unified env for a joint
> repo will work fine."

So the deliverable is a **recommendation with evidence**, not a merge.

## What to produce

`docs/refactor/FINDINGS-repo-merge.md` covering:

1. **Environment diff.** Compare `bop-run-upload/requirements.txt` against
   `bop-analyze-remote/requirements.txt` and `environment.yml`. Which packages does the
   analysis side add? Which are heavy (lightgbm, matplotlib, jupyter, TeX tooling)? Is
   any of it needed at *run* time on the cluster, or only for post-processing on a
   laptop? Give the actual added install size / dependency count, not an impression.
2. **Does the bloat concern survive contact with the numbers?** Seth's worry is module
   bloat slowing cluster compute. Check whether that is real: does anything in the
   analysis set get imported by the run path, or would it just sit unused on disk?
   A unified env is only a problem if it is actually loaded or if it forces a version
   conflict with the run-side pins. Look for genuine conflicts.
3. **Merge mechanics.** `git subtree add --prefix=analyze <path> <branch>` preserves
   history. Confirm that works here, name the prefix you would use, and flag any path
   collisions — **note both repos have a `config.py` at top level**, which is the obvious
   one, and check for others (`__pycache__`, `figures/`, `tables/`, results dirs).
4. **The interface that matters.** `analyze.py` and `build_sdfwts_chars.py` produce the
   summarizations Seth cares about. Document what they consume (paths, file formats,
   naming conventions) and what they emit. This is the contract a merged repo has to
   preserve, and it feeds the spec/summary-linking work (Phase 2/4).
5. **Recommendation:** unified env, two envs in one repo, or stay split. Say which, why,
   and what it costs.

## Context you need

- A 10-panel simulation run produces ~80 GB. It lives on cluster scratch and is **not
  permanently retained**. So the durable artifact must be the *summary*, linked to the
  spec that produced it — that linkage is the whole point of the merge.
- `variants/common/solstamp.py` + `experiments/solfiles/` already do this for **solves**
  (content-addressed `solve_id`, manifest outlives the artifact). `experiments/specs/`
  holds content-hashed experiment specs. Read those two before proposing anything; the
  answer should extend that design rather than invent a parallel one.
- `docs/refactor/PLAN.md` §Phase 4 is the existing sketch. `docs/refactor/WORKING.md` §5
  and §6 have earlier notes on the merge and the analysis interface.

## Boundaries — please respect exactly

**Yours to create:** `docs/refactor/FINDINGS-repo-merge.md`.

**Read anything. Modify nothing else.** In particular do not touch
`docs/refactor/WORKING.md` (the primary session appends to it live and two appenders will
collide), `variants/**`, `config.py`, `main.py`, `tests/**`, or anything in
`bop-analyze-remote`.

**Do not run the merge.** This is an assessment. Seth decides.

**Do not commit.** Leave the tree dirty; Seth commits himself.

**Do not install anything** into either environment, and do not run any solve, oracle or
estimator. If you want to test an import, say so in the findings rather than doing it —
the box is saturated and the run env is deliberately minimal.
