# Findings — config.py parameters that get overridden downstream

ASU session, 2026-09-06. Question from Seth: which `config.py` parameters are later overridden
in solving files, and is that a bug? Read-only analysis; **nothing changed, nothing committed.**

## 0. Headline

There are two entirely different mechanisms, and only one of them is a problem.

1. **Main tree (`utils_*`) — the guard works, and it is firing right now.**
   `GS21_DELTA` and `GS21_TAU` were changed on 2026-09-01 without regenerating solfiles.
   `import utils_gs21` currently raises `SolfileStaleError: 54 problem(s)`.
   **GS21 cannot run in the main pipeline today.** That is the machinery working, not a bug —
   but it is a live blocker nobody has recorded. KP14 and BGN verify **CLEAN**.
2. **Variants — no override, no guard, and this is the real finding.**
   **No file under `variants/` imports `config.py` at all.** They carry private parameter
   copies. Those copies have drifted, and nothing anywhere compares the two trees.

## 1. Method

- AST scan for config names reassigned at module level after import, and for `config.X = ...`.
- Numeric evaluation of every variant parameter against its `config.py` counterpart.
- Both compared against `utils_gs21/GS21.m` (archival MATLAB) and against
  `utils_gs21/GS21_solfiles/stamp.json` (what actually built the committed solfiles).

## 2. Main tree: does anything override an imported config value?

Essentially no. One hit, and it is benign-but-fragile:

**`utils_bgn/sdf_compute_bgn.py:32`** does `from config import ... CHAT as Chat`, then
immediately `Chat = np.exp(Cbar)` — recomputing `config.py:207` verbatim. Value-identical
today. It is a latent divergence: change the *definition* of `CHAT` in config and this file
silently keeps the old formula. This is precisely the derived-parameter hazard PLAN §4
describes. One-line fix: delete line 32 and use the import.

Also: `voc_diagnosis/dead_end_K/*.py` writes `config.BGN_N_EXTRA_FACTORS`,
`BGN_SIGMAJ_BAND`, `BGN_EXTRA_OMEGA`, `BGN_EXTRA_DECAY` — **none of which exist in
`config.py`**. Dead-end diagnostics, harmless, but they are dead code referencing a config
surface that is gone.

Intentional, correct overrides (not bugs): `BOP_CHARS` → `_apply_chars_env()`, `N_JOBS` env in
`get_n_jobs_for_step`, `BOP_SCRATCH_DIR`/`BOP_TEMP_DIR` → `init_from_env()`. All cross the
subprocess boundary through the environment, which `config.py:416-425` explains is required
because `main.py` runs steps as separate processes.

## 3. Variants vs config — measured

**KP14** (`variants/kp_vy/parameters_kp14.py`): all 19 economic parameters **identical**.
Only `burnin` differs — 400 vs `KP14_BURNIN` 200.

**BGN** (`variants/bgn_gam/parameters.py`): all 8 economic parameters **identical**.
Only `burnin` differs — 300 vs `BGN_BURNIN` 200.

**GS21** (`variants/gs_bx/gs_solve_reg.py`): **six divergences, and they are economic.**

| param | variant | config.py | GS21.m | solfile stamp | who matches |
|---|---|---|---|---|---|
| `delta` | 0.02/3 = 0.006667 | **0.02** | 0.02/3 | 0.02/3 | variant + .m + stamp |
| `tau` | 0.2/3 = 0.066667 | **0.2** | 0.2/3 | 0.2/3 | variant + .m + stamp |
| `rho_x` | 0.95^⅓ = 0.983048 | **0.96^⅓ = 0.986485** | 0.95^⅓ | 0.96^⅓ | config + stamp |
| `sigma_x` | 0.00704631 | **0.00702226** | (0.95 form) | 0.00702226 | config + stamp (downstream of `rho_x`) |
| `sigma_m` | 2.5 | **5** | 5 (`% 2.5` commented) | 5 | config + .m + stamp |
| `xnum` | 161 (CLI) | **20** | 20 | 20 | config + .m + stamp |

`xi`/`GS21_ZETA` agree at 0.01; `GS21.m:35`'s `0.03/3*3 = 0.03` is the erroneous one and both
Python trees already correct it. Everything else matches.

## 4. What this means, per parameter

**`delta` and `tau`: config.py is right and `GS21.m` is wrong.** Check the payoff
(`gs_solve_reg.py:130`):
`pi_R = (1-tau)*(exp(bx·x + z + ashift) - delta) - (1-tau)*b`.
`delta` is subtracted from output *in output units*, and output is not rescaled monthly, so
dividing it by 3 is a units error. `tau` is a rate multiplying the whole flow, and rates do
not rescale with period length. Both match config.py's 2026-09-01 comments, which I read
after reaching the same conclusion from the code. **So the variant carries two superseded
unit errors.**

**`sigma_m`: the variant took a commented-out value.** `GS21.m:53` is `sigma_m = 5; % 2.5` —
5 is active, 2.5 is the comment. This is the *identical* error class already caught and
documented for `GS21_R` (config.py:369: "GS21_R was 0.074830/12, which is the COMMENTED-OUT
alternative on GS21.m:27"). It happened twice; the second instance is still live. **This
resolves PLAN §10 open decision #3 ("is sigma_m 2.5 or 5?") on the `.m` evidence: 5.**

**`rho_x`: genuinely open, and it is PLAN §10 decision #2.** config.py's comment asserts
"GS21 Table 1: rho_x = 0.96 quarterly"; `GS21.m:22` uses 0.95. The committed solfiles were
built with 0.96, so someone already acted on the 0.96 reading. **I cannot settle this from the
repo — it needs the paper.** Note the asymmetry: on `delta`/`tau` we trust our reasoning over
the `.m`; on `rho_x` we trust a comment about the paper over the `.m`. Both may be right, but
they are different kinds of claim and only one has been verified in-repo.

**`xnum` 161 vs 20: a cost decision, not an error.** `config.py:344-350` documents that
`GS21_XNUM = 20` is load-bearing for *stability* (a coarse x grid lets `i_cut` saturate and the
solve diverges; `xnum=5` overflows). 161 is finer, so it is safe in that direction — but it is
8× the grid and is the direct cause of the 3 h 23 m solve measured in `FINDINGS-gs21.md`.

## 5. The main-tree blocker (act on this)

```
SolfileStaleError: GS21 solution files failed provenance verification (54 problem(s))
  - zgrid.csv: built with GS21_DELTA=0.006666666666666667, config.py now has GS21_DELTA=0.02
  - zgrid.csv: built with GS21_TAU=0.06666666666666667, config.py now has GS21_TAU=0.2
  - zgrid.csv: gs21_solve.py has changed since this file was generated
```

Raised at **import time** of `utils_gs21` (via `sdf_compute_gs21.py:31`). KP14 and BGN are
CLEAN — GS21 is alone. Fix is the documented one, ~2.5 min:
`python utils_gs21/regen_solfiles.py`. I did **not** run it: it rewrites 18 tracked solution
files and that is a commit-shaped decision, not mine.

Worth noting the stamp is a *mixture* — `rho_x`/`sigma_x` at the new values, `delta`/`tau` at
the old. So the solfiles were regenerated once after the `rho_x` change and then config moved
again. The guard caught it, which is the system doing its job.

## 6. The structural gap

`variants/` has `solstamp` provenance, and it is good — but it hashes each variant's **own**
namespace. `config.py` is not in that namespace, so **no mechanism anywhere compares the two
parameter trees.** The main tree has a guard against config drift; the variants have a guard
against *their own* drift and nothing else. That is why GS21's `delta`/`tau` correction
reached `config.py` on 2026-09-01 and never reached `variants/gs_bx/`.

**Consequence for live numbers:** `gs21` (main) and `gs_bx` (variant) are **not the same
economy at different settings** — they are different economies. Any comparison across them is
confounded by `delta`, `tau`, `sigma_m` and `rho_x`. REPORT §19's bx7 numbers were produced
under the variant set.

**Consequence for my own GS work yesterday:** `sol_reg`
(`solve_id a8ef7a2522eda19d`, 3 h 23 m) was solved with `delta=0.02/3`, `tau=0.2/3`,
`sigma_m=2.5`, `rho_x=0.95^⅓`. If config.py is authoritative on `delta`/`tau`/`sigma_m` — and
§4 argues it is on all three — **that artifact is a solve of the superseded economy.** The
manifest is still valid provenance; it just records an economy we may not want. Do not build
bx7 on it without settling §4 first.

## 7. Recommendations

1. **Settle `rho_x` against the actual GS21 paper Table 1.** It is the one open factual
   question and it blocks naming a `gs21-base` spec (PLAN §10 #2).
2. **Port `delta`, `tau`, `sigma_m` from config.py into `variants/gs_bx/gs_solve_reg.py`** —
   or, better, have the variant import them. All three are corrections the main tree already
   made and the variant missed.
3. **Regenerate GS21 solfiles** to clear the import-time blocker (§5).
4. **Add a cross-tree parameter test.** A ~20-line test asserting that each variant's
   parameter namespace agrees with `config.py` on every shared name (whitelisting deliberate
   divergences like `burnin` and the variant's own knobs) would have caught all six of these
   at commit time. This is the cheapest durable fix and belongs with the Phase 2 spec work.
5. **Delete `utils_bgn/sdf_compute_bgn.py:32`** and use the imported `Chat` (§2).
6. **Record `burnin` as a deliberate divergence** (400/300 vs 200) or reconcile it — right now
   it is indistinguishable from drift.

## 8. What I did not do

No files changed. Did not run `regen_solfiles.py` (rewrites 18 tracked files). Did not touch
`variants/`, `config.py`, `WORKING.md`. Did not verify `rho_x` against the paper — I don't
have it. `GS21_BETA/PSI/GAMMA/CHI/KAPPA_E` and the BGN/KP14 `burnin` values are the only
remaining unexamined divergences, all of them low-stakes.

---

## Primary session's evaluation (2026-09-06)

Claims checked independently rather than accepted. **All structural findings hold.**

- `import utils_gs21` → `SolfileStaleError: 54 problem(s)`. Reproduced. `utils_kp14` and
  `utils_bgn` import clean. **GS21 is down in the main pipeline right now.**
- No file under `variants/` imports `config.py`. Reproduced (grep returns nothing).
- KP14 variant vs config spot-checked on 8 economic parameters — all identical.
- `GS21.m:53` is `sigma_m = 5; % 2.5`. Reproduced. `config.py:369` documents the same
  error class already caught for `GS21_R`. Reproduced.

### Addition: the commented-out-value class is now closed, not just twice-seen

`GS21.m` contains **exactly three** commented-out numeric alternatives. Checking each
against both trees settles the class rather than leaving it open:

| GS21.m | active | commented | config.py | variant |
|---|---|---|---|---|
| :27 `r` | `0.1/12` | `0.074830/12` | ✅ `0.1/12` (documented at config.py:369) | ✅ `0.1/12` |
| :33 `kappa_e` | `0` | `0.025` | ✅ `0` | n/a (not present) |
| :53 `sigma_m` | `5` | `2.5` | ✅ `5` | ❌ **`2.5`** |

So `config.py` transcribed all three correctly and the variant got one wrong. There is no
fourth instance to hunt.

### On `delta` / `tau`: corroborated, but label it a judgment

`config.py:311-312` and `:328` state the reasoning explicitly — delta "enters only as
maintenance relative to per-period output, which is not rescaled monthly"; tau is "a rate,
not a flow". The ASU session reached the same conclusion from `gs_solve_reg.py:130` *before*
reading those comments, which is genuine independent corroboration and worth more than
either alone.

It remains a **modeling judgment, not a fact**: `GS21.m` and the solfile stamp both divide
by 3, so `config.py` (commit `6c65bf4`, "GS21 monthly conversion and tax") is a deliberate
departure from the archival source. The finding that does not depend on resolving it: **the
variant never followed that deliberate, documented correction.**

The ASU session's own methodological criticism is fair and should be kept: on `delta`/`tau`
we trust our reading of the code over `GS21.m`, while on `rho_x` we would trust a *comment
about* the paper over `GS21.m`. Those cannot both be the default. `rho_x` needs the paper.

### Operational consequence — this changes the GS plan

Yesterday's 3 h 23 m `sol_reg` (`a8ef7a2522eda19d`) encodes `delta`, `tau` and `sigma_m`
from the superseded set. The manifest is *valid provenance of the wrong economy* — which is
the registry working exactly as designed, and the first time it has paid off that way.

So **bx7 must not be built on it**, and the GS cluster run needs its parameters settled
first. That is a second, independent reason to freeze before solving (§23 gave the
source-code reason); this one is about parameters, and it is the more expensive of the two
to get wrong — five solves at ~3.4 h.

`gs21` (main) and `gs_bx` (variant) are **different economies**, not one economy at two
settings, so any cross-tree comparison of GS results is confounded until this is fixed.

### On the proposed cross-tree test

Agreed, and its value is not limited to GS: KP14 and BGN agree *today* with nothing
preventing tomorrow's drift. The whitelist should carry a reason per entry (`burnin` differs
deliberately), so that an unexplained divergence cannot be silently added to it.
