# Working doc — variants → main pipeline refactor

Started 2026-09-04. Purpose: absorb the `variants/` experiments into the main
simulation/estimation workflow (`run_bop_job.sh` → `main.py`), reconcile solution
methods, and build durable experiment-spec + summary storage.

**This file is the resume point.** If a session dies (usage limit, crash), read this
file first, then continue at the first unchecked box in §8.

---

## 0. Session log

| date | what happened | next |
|---|---|---|
| 2026-09-04 | Scoping. Ran workflow `wf_837ba7c0-2c3` (10 readers + 12 adversarial verifications, 22 agents, 0 errors). Findings distilled into §1–§6 below. Launched design workflow `wf_5247e636-007` (judge panel on integration architecture). | Plan synthesis → present to Seth |

Artifacts:
- Per-reader structured findings: `_scratch/wf/*.json` (10 files, ~550KB total)
- Workflow journals: `~/.claude/projects/-Users-sjpruitt-GitHub-bop-run-upload/8917aaac-.../subagents/workflows/wf_837ba7c0-2c3/journal.jsonl`
- Resume a workflow: `Workflow({scriptPath, resumeFromRunId})` — cached agents replay instantly.

---

## 1. Job 1 — What the variants actually do

### 1.0 The research logic (from `variants/REPORT.md`)

The whole campaign reduces to one equation, **realized gap = room × capture**.

- **room** = (population SR of best nonlinear basis) − (population SR of best
  linear-in-ranks basis). What only a nonlinear method could add with *zero*
  estimation noise. Measured by `run_oracle.py`.
- **capture** = fraction of that room a real estimator harvests. Pinned at
  **~75–80%** by DKKM's own P/T ridge shrinkage at any realistic window. Longer
  windows help the *linear* benchmark first.

Baseline diagnosis: in the published models, room is ~5% of SR versus ~100–300% in
DKKM's data, because expected returns and betas are affine in one or two firm state
variables (BGN: book/P and 1/P; KP: PVGO/V) with **constant** prices of risk. True SDF
weights Σ⁻¹μ *are* nonlinear in characteristics (cross-sectional R² on rank-chars 0.46,
on bins 0.90) but that nonlinearity sits in directions with almost no expected return
per unit risk. The DKKM edge in the paper is **the rank transformation, not complexity**.

~35 economies across five lever families were tried. Dead ends worth not repeating:
size-dependent idiosyncratic vol (entire effect was a fat-tail artifact); KP14 parameter
search (affine in one variable — nothing helps, room ≤ +0.008); "anomaly lives in small
stocks" micro-cap spikes (too little aggregate alpha).

**The design law (§17e, §18, §19d):** a state-dependent price of risk creates learnable
cross-sectional nonlinearity **iff the state bends a heterogeneous firm-level exposure
map non-monotonically**, and the gap is harvestable only if that bending stays in
rank/interaction space — if the exposure also moves raw characteristic *levels*, cheap
rolling FMR contests it. Final form: **premium-side extremity raises both room and
capture; volatility-side extremity does not.** The empirically relevant frontier is
premium-side.

That is the single most important sentence for choosing the next experiment.

### 1.1 BGN `bgn_gam` / tag `g0235`

Fork of the **Dropbox** BGN code, not of `utils_bgn/`. Adds a 2-state Markov regime
multiplying the price of the aggregate SDF shock: σ_z → σ_z·gmult[s], s ∈ {calm,stress},
monthly switch probs p01=0.25/12, p10=0.50/12. Risk adjustment is exp(−β·gmult[s]), so
every value object splits onto two bases:
`V_s(r,β) = Ĉ·(e^{−β g0}·DA_s(r) + e^{−β g1}·DB_s(r))`, with `DA_s + DB_s = D(r)` identically.

- **gmult=[1,1] collapses exactly to baseline** — verified J0 vs J1 to 2.7e-15, and the
  full reduction to `utils_bgn` agrees to ~2e-5 relative (limited only by the different
  J* grids). This is the cleanest of the three variants.
- Shipped economy `g0235` = gmult [0.2, 3.5], parked just below the closed-form frontier
  `max(gmult) < 1/(2·scale) = 3.6548` (asserted at `sdf_compute.py:46`).
- The regime is an **observed conditioning variable** handed to estimators as panel
  column `gam_stand`.
- BGN calibration constants are **identical** to `config.py` to the digit (verified by
  execution: β*=0.7721328263, scale=0.1368073070).

### 1.2 KP14 `kp_vy` / tag `vyx`

Fork of the Dropbox KP14 simulator. Adds a priced stationary OU factor y with
heterogeneous firm exposures: type-f cash flows carry e^{β_f y_t}; the SDF prices dW_y at
constant γ_v. Every closed-form object becomes a function of (type, y-node): scalar
A_0..A_3 = 1/(CONST+θ) become (ntypes × NY) solutions of
`(diag(const_f(y)+θ_c) − Q_y) a = 1` on a 21-node OU grid. At `type_bv=[0]`, reproduces
`config.KP14_A_0..A_3` and `KP14_RHO` to machine precision.

G is re-solved as a coupled (2 λ-regimes × 21 y-nodes) = 42-block system by one-shot
sparse LU, **in the direct basis**.

### 1.3 GS21 `gs_bx` / tag `bx7`

Self-contained re-implementation sharing **no code and no parameters** with `utils_gs21/`
or `config.py` (nothing under `variants/` imports config). Same Bellman system, but:
(a) exogenous 2-state Markov price-of-risk regime γ_s = γ_x·gmreg[s], two coupled Bellman
systems Gauss-Seidel'd per sweep; (b) solved once per **exposure type** — production loads
as exp(gs_bx·x + z + gs_ashift) instead of exp(x+z) — five (gs_bx, gs_ashift) pairs into
five `sol_*/solution.npz`; (c) a much finer aggregate grid (xnum=161 vs `GS21_XNUM=20`).
`gs_sim_bx.py` glues the five solutions into one economy by assigning each firm a type
(seed 910) and dispatching every table lookup through `_by_type`; all types share one x
path, one regime path, one kernel, and grids from `_sols[0]`.

**Two README claims are misleading and should be corrected:**
- "exact kernel" = exact row renormalization E[M_s|x] = e^{−r}. **Both sides already do
  this** and it is algebraically identical (verified 1.1e-16).
- It is **not** a replacement for tauchen. The variant re-implements tauchen inline and it
  is **bit-identical** to `utils_gs21/tauchen.py` (grid and transition diff exactly 0.0).

### 1.4 Estimator + oracle machinery (`variants/common/`)

The population **oracle has no counterpart in the main pipeline at all** — this is the
single most valuable thing in `variants/`. `oracle.py` builds 8–12 feature bases per month
(`fmr_raw`, `lin_rank`, `lin_rank_rf`, `poly2`, `bins`, `rff{36,360,3600}`, plus
`lin_rank_lev`/`rffL{P}` under `--levels`) and scores each with a two-pass population
evaluation: pass 1 accumulates E[f]=mean_t Φ_t'μ_t and E[ff']=mean_t(Φ_t'Σ_tΦ_t + a_t a_t');
pass 2 evaluates constant θ*(z)=(E[ff']+z·s·I)⁻¹E[f] and, for small-P bases, the per-month
conditional oracle sqrt(a_t'B_t⁻¹a_t).

`common/dkkm_functions.py` and `fama_functions.py` are DataFrame cousins of
`utils_factors/*`: same rank standardization ((rank−0.5)/N − 0.5, pandas average ranks),
same sin/cos RFF layout, same eigendecomposition ridge with penalty WINDOW·z.

---

## 2. Job 3 — Solution-method reconciliation

99 divergences catalogued; 12 adversarially verified, **11 confirmed, 1 refuted**.
87 remain unverified — see the triage rule in the plan, do NOT verify them all up front.

### 2.1 Confirmed, material — these change numbers or are outright bugs

| # | divergence | which side is right | note |
|---|---|---|---|
| D1 | **KP14 λ regime inconsistency — see §2.1a below. Independently verified by hand.** | `config.py:269` is the correct stationary probability; `config.py:257` and the λ_H calibration presume the opposite convention. | **Highest-priority correctness item.** Blocks trusting either tree's KP numbers. |
| D2 | **KP CIR quadrature.** variants uses fixed `eps_max=10` with a **print-only** mass check; main uses a parameter-adaptive quantile interval + a real guard. | **main** | variants has regressed toward the exact pattern that caused the prior all-zeros integration bug. Currently benign (mass integrates to 1.0 at σ_eps=0.2) but would silently return ~0 if σ_eps were cut to 0.02. |
| D3 | **KP `max_sr`** — variants keeps an *estimated* max Sharpe; main replaced it with the analytic Hansen–Jagannathan bound. Measured, the variant estimate **exceeds its own HJ bound** (0.67–0.70 vs 0.614) at N=60. | **main** | An estimate above its own bound is an impossibility, i.e. a diagnostic that something is wrong. |
| D4 | **Zero-book / zero-capital firms in the SDF solve.** main filters `book > 0` before the linear solve; variants does not (BGN and KP). | **main** | Panel construction already drops these firms, so the true conditional MVE portfolio should be solved on the investable set. Impact is model-specific and narrower than "catastrophic". |
| D5 | **GS21 parameters are a hybrid matching no commit of config.py** — frozen at pre-`6c65bf4` (ρ_x=0.95^(1/3), τ=0.2/3, δ=0.02/3) but carrying post-commit σ_z. Six parameters diverge. | **main** (config.py is the papers-faithful side after `6c65bf4`) | |
| D6 | **GS21 σ_m** = 2.5 (variants) vs 5 (main). Live in both. Provenance unsettled. | **unknown** | Settle by the same residual-identification that fixed r/ξ/γ_x. |
| D7 | **GS21 period-2 orbit.** main kills it with closed-form quadrature; variants merely **tolerates** it (0.5 damping + 25-sweep policy freeze + hard cycle-average at sweep 5600), because it reintroduces a discrete 161-node smoothing measure whose max node weight (0.0199) is >2× the GH weight (0.00886) the orbit was blamed on. | **main** | |
| D8 | **GS21 off-grid extrapolation.** Structurally *eliminated* in variants (integer z/x chains, clamped linear b); still **live** in `utils_gs21` (cubic `RegularGridInterpolator`, `fill_value=None`). | **variants** | One of the few places variants is cleaner. |
| D9 | **GS21 smoothing shock in simulation.** main solves with σ_m=5 and then **simulates with no smoothing shock at all**; variants draws it from the same 161 nodes it solved on. | **variants** (internally consistent) | Internal inconsistency in main worth fixing. |
| D10 | **`sdf_loop` signature/return arity** differ across all six modules (`(t, iter)` → 4-tuple vs `(t, iter=0)` → 5-tuple with `w_true`). | adopt **variants** signature | Named "the single hardest blocker to unifying the two trees" — but the fix is backward-compatible and touches no numerics. |
| D11 | **No RNG seeding anywhere in main.** `variants/run_oracle.py` seeds both streams (`np.random.seed` and `default_rng`). | **variants** | Main runs are reproducible only *in distribution*. |
| D12 | **LATENT BUG** `variants/common/dkkm_functions.py:170` hardcodes `360` where `WINDOW` belongs, in the market-return slice. | — | Did not bite (all shipped runs used `--window 360`) but is a landmine. One-line fix. |
| D13 | burnin 300 (variants) vs 200 (main). | either, must match | Changes numbers via both panel truncation and rf standardization. |

**Refuted:** the claim that BGN calibration parameters differ. They are identical to the digit.

### 2.1a D1 in full — the KP14 λ regime inconsistency

Verified by hand, not just by agent. For a 2-state chain with H→L rate μ_H and L→H rate
μ_L, the stationary P(high) = μ_L/(μ_H+μ_L). With `KP14_MU_H = 0.075`, `KP14_MU_L = 0.16`:

```
config.py:269   KP14_PROB_H = MU_L/(MU_H+MU_L) = 0.6809      <- used by the simulator to draw the regime
config.py:257   solves lambda_L using weight MU_H/(MU_H+MU_L) = 0.3191 on lambda_H
                -> lambda_L = 0.367187, and E[lambda] = 1.000000 under THAT weight
                -> but E[lambda] = 1.717188 under the weight the simulator actually uses
```

So the intended normalization E[λ] = 1 does not hold in the economy that is actually
simulated; the realized stationary mean of λ is **1.717**. This is exactly the "E_stat[λ]
= 1.717 factor" the KP reader flagged. The two conventions cannot be reconciled by
re-deriving λ_L, because at P_H = 0.6809 the normalization gives λ_L = −1.881, which is
infeasible — λ_H = 2.35 itself presupposes P_H = 0.3191.

**Therefore the switching intensities are labelled backwards somewhere.** The coherent fix
(recommended by the verifier and consistent with the algebra above): treat μ_H as the
**L→H entry** rate everywhere, set `KP14_PROB_H = MU_H/(MU_H+MU_L) = 0.3191`, and flip
`utils_kp14/panel_functions_kp14.py:135`. This changes the KP14 economy — it is not a
cosmetic relabelling — so it must be settled against Kogan–Papanikolaou (2014) Table 2
before any KP experiment is run. **Note this affects the production pipeline, not just
`variants/`.**

### 2.2 Also true, not yet verified but load-bearing

- variants lacks main's **2026-08-31 cubic-interpolation fix** (linear everywhere).
- variants has **no solfile stamp / provenance guard** on any model. `kp_vy/meta_vyx.json`
  is a hand-rolled 5-key cache stamp omitting δ, θ_eps, θ_u, y_max and the ε grid — and
  has no producer-source hash, so editing `kp14_fd_vy.py` silently reuses stale tables.
  This is precisely the failure `solfile_stamp.py` was built to prevent.
- `variants/kp_vy/kp14_fd_gamy.py` and `build_gamy_tables.py` are **dead and broken**
  (import a nonexistent `rho_y`).
- Main pipeline runs **N=1000, T=720**; every variants run is **N=500, T=500**. Published
  variants numbers do not transfer.

---

## 3. Job 2 — The main pipeline as it stands

`run_bop_job.sh` is a SLURM array script: one task = one panel index →
`main.py MODEL $ID $((ID+1))`, after exporting `BOP_SCRATCH_DIR` / `BOP_TEMP_DIR` /
`BOP_LOG_FILE`. `main.py` is a thin orchestrator running **seven steps as separate python
subprocesses**, each independently re-importing `config` and calling `config.init_from_env()`,
reading/writing pickles keyed `{model}_{index}_*.pkl`.

**The structural consequence:** parameters cannot be passed in memory between steps. Any
spec layer must travel via environment or files. This is why `variants/` uses env-var JSON.

- `config.py` is flat module-level globals + `get_model_config(model_name)` with hardcoded
  dict lookups keyed `'bgn'|'kp14'|'gs21'`. **No named-configuration or experiment concept
  exists anywhere.** `get_model_config` is the natural seam.
- Parallelism: joblib `backend='multiprocessing'` (forced fork), worker counts from
  `config.MODEL_N_JOBS[model][step]`; BLAS pinned to 1 thread.
- 44 distinct knobs define a run (full inventory in `_scratch/wf/mainpipe.json`).
- **The main pipeline computes no summary metric at all.** `evaluate_sdfs.py` stops at
  per-month `mean` / `stdev` / `xret`. No Sharpe, no HJD, no aggregation. All
  summarization lives in the *other repo*.
- Disk: `{id}_moments.pkl` ≈ **2.9 GB** (360 months × 1000×1000 float64 cond covariance),
  and `KEEP_MOMENTS=True` **duplicates it into both TEMP_DIR and DATA_DIR**. That is the 80 GB.

**Existing provenance — the one good bone to build on.** `utils/solfile_stamp.py` hashes
each model-solution file, the config parameters its producer imports (discovered by
**AST-parsing the producer's own import statements**), the producer's source, and any
upstream solfile; it hard-fails at import time of every consumer and reports four distinct
staleness modes. `write_stamp()` already accepts an unused `extra` dict.

---

## 4. Job 4 — Experiment specs and summaries

**What is missing is one level up from solfiles: nothing records what a *run* was.**

- `main.py` names outputs `{model}_{index}` only. `evaluate_sdfs.py` stores model/chars/
  alpha lists in `_results.pkl` but not N, T, burnin, git commit, solfile digests, or a seed.
- The S3 path exists but keys on a timestamp `WORKFLOW_ID` under `koyeb-results/`, which is
  **not derivable from a spec**.
- `variants/` is a working prototype (JSON overrides in env, `--tag` in every filename,
  `{model}_oracle_{tag}.json` carrying the override string alongside the numbers, all
  outputs 8–64 KB and committed) with three honest weaknesses:
  1. **The `gs_bx` flagship spec exists only inside `run_gs_bx7.sh`.** Its own oracle JSON
     records `overrides = {}`, because it reads `GS_SIM_OVERRIDES` while the real experiment
     lives in five `GS_PARAM_OVERRIDES` solve invocations plus three `GS_BX_*` env vars.
  2. `meta_vyx.json` staleness hole (see §2.2).
  3. `collect_results.py` rebuilds `oracle_summary.csv` by globbing disk, so running it now
     would shrink a 57-row file to 3. The README warns not to run it. Destructive-by-design.

Sizing facts that make this easy: the full uppercase `config.py` snapshot serializes to
**3,210 bytes** (97 globals; only `GAMMA_GRID` needs `.tolist()`). Specs ~4 KB, summaries
~100 KB → hundreds of experiments cost a few MB. **Git is the right store for both.**

---

## 5. Repo merge

**Recommendation: merge, via `git subtree add --prefix=analyze`.** Tested for real in a
throwaway clone: grafts all 21 analyze commits, zero path conflicts, `.git` grows
**68.1 → 71.6 MB (+3.5 MB, +5.2%)**, `git blame` reaches through to the original
2026-03-25 commits.

  Two corrections to earlier numbers here (both remeasured 2026-09-05 at KB resolution
  from a *pristine* clone):
  - The cost is **all in `git fetch`**, not the graft: pristine 68.1 MB → after fetch
    71.5 MB → after `subtree add` 71.6 MB. The subtree add itself costs **16 KB**. Any
    repro that fetches before its "before" measurement will report ~0% growth and any
    that uses `du -sh` will round 71.6 to either 71 or 72 — which is where the earlier
    +4.4% and a proposed +5.9% both came from.
  - **`git blame` reaches through, but path-limited `git log` does not.**
    `git log -- analyze/` returns only the graft commit, and
    `git log --follow -- analyze/analyze.py` returns nothing: the old commits have no
    `analyze/` prefix. Recovering that would need history rewriting. Blame is the only
    view that reaches back, which is enough for attribution but should not be oversold.

- **These were one repo to begin with.** `analyze.py` was in bop-run-upload's initial commit
  (fc0cb6f, 2026-01-23), deleted in d6ec6b2 (2026-01-29); bop-analyze-remote was created
  from that split on 2026-03-25. This is re-unification.
- Only 4 tracked-path collisions (`.gitignore`, `config.py`, `environment.yml`,
  `requirements.txt`), all vanishing under the prefix. The one semantic collision is the
  module name `config`. Measured 2026-09-05: run has **41 files / 46 import lines / 156
  `config.` attribute uses**; analyze has **11 files / 11 import lines**. (The earlier
  "42 import sites" matched none of these counts; state which metric is meant.)
  **Fix: rename analyze's to `analyze/paths.py`** — 11 one-line edits, which is the
  import-line count and so still correct. Script-directory resolution masks the problem
  today, but `python -m`, pytest, a bare REPL at root, or importing `main.py` (which does
  `sys.path.insert(0, repo_root)`) all silently grab the wrong `config`.
- **The lean-env concern is measurably false.** Bare interpreter startup: 0.01 s in an
  84-package/494 MB env and 0.01 s in a 271-package/713 MB env. Installed-but-unimported
  packages cost nothing at SLURM startup. The run pipeline's actual import set
  (numpy+scipy+pandas+sklearn+joblib) is 0.95 s either way. Keep two `environment.yml`
  files anyway — for **solve time and cluster disk quota**, not runtime — chiefly to keep
  `torch` (imported by `MLP Analyze/`, declared in neither env file) off the cluster.
- **The argument for merging is not convenience, it is drift.** The interface is an
  undocumented filename+dict-key contract, and it has *already* silently broken: run commit
  `dc48c98` (2026-09-01) added a per-draw `mat` column to `dkkm_results` and changed the
  DKKM weights key from `(nfeatures, alpha)` to `(nfeatures, alpha, mat)`. `analyze.py` has
  never been updated.

**⚠ Governance blocker — this is Seth's call, not a technical one.** `bop-run-upload` is
owned by **kerryback**, with Seth at WRITE permission; 85 of 162 commits are Kerry Back's.
`bop-analyze-remote` is Seth's own repo at ADMIN. Merging pushes Seth's analysis code into
a co-author's repo. Ask before doing it.

---

## 6. Analysis repo — the interface contract

Two live scripts:
- **`analyze.py`** — reads only `{model}_*_results.pkl`; produces the numbers behind
  `all_results.pdf` (written as `tables/*.tex` + `figures/*.pdf`, **neither tracked in
  git** — only the compiled PDF is).
- **`build_sdfwts_chars.py`** — reads the ~2.9 GB `{model}_{run}_moments.pkl` plus
  `{model}_{run}_panel.pkl`, emits `{model}_sdfwts_chars.pkl`. This is the bridge that lets
  true SDF weights Σ⁻¹μ be regressed on observable characteristics. **It is the only thing
  that genuinely needs the cluster**, because of the moments file. A second tier
  (`fit_models.py`, `interpret_lightgbm.py`, `MLP Analyze/`) consumes that bridge file.

Six Mar–Apr 2026 slide scripts are legacy. `config.py` there resolves RESULTS_DIR/WTS_DIR
via env var → repo-root symlink → `/scratch/sjpruitt/bop` fallback.

---

## 7. Usage-limit resilience

- **Progress lives in this file.** Every workflow result is distilled here before moving on.
- **Workflow resume**: `Workflow({scriptPath, resumeFromRunId})` replays cached agent
  results instantly; only edited/new agents re-run. Run IDs are in §0.
- **Auto-restart after a 5-hour reset**: `CronCreate` can enqueue a prompt at a wall-clock
  time, but jobs are **session-only** (in-memory, gone when Claude Code exits) and fire only
  while the REPL is idle. It works if the terminal stays open; it cannot resurrect a closed
  session. Pattern: leave the session open, schedule a one-shot cron just after the reset
  with the prompt *"read docs/refactor/WORKING.md and continue at the first unchecked box in §8"*.

---

---

## 9. Design phase results (workflow `wf_5247e636-007`, 13 agents)

Judge scores (0-10) across three candidate architectures:

| lens | unify | adapter | spec_first |
|---|---|---|---|
| scientific validity | 7.5 | 5 | **8** |
| time to first result | 2 | 6 | **8** |
| durability | **9** | 6 | 8 |
| maintenance risk | **8** | 3 | 7 |

Outcome: **spec_first spine + unify's config discipline**; `adapter` (six permanent
simulators) rejected. Full plan in PLAN.md (absorbed into §43, 2026-09-09).

### New verified facts from this phase

- **`--chars` is silently broken.** It parses, validates, prints a reassuring `[CONFIG]`
  line, and has **zero effect** — the override mutates an in-memory dict in `main.py`'s
  interpreter while all 8 steps run as fresh subprocesses re-importing `config.py` from
  disk. A `--chars` run is byte-identical to a full-set run, with no warning. The
  documented `CHARS_FLAG` in `run_bop_job.sh:25-26` hits the same failure. `main.py:89`
  also has an unguarded `sys.argv[chars_idx+1]`. `MODEL_FACTOR_NAMES` has no consumers at
  all, and the char→factor map is re-declared in three private copies while
  `config.CHAR_TO_FACTOR` sits dead.
- **The config transport constraint that kills the naive design.** `config.py` has ~14
  *derived* module-level parameters (`KP14_LAMBDA_L:257`, `KP14_CONST:270`,
  `KP14_A_0..A_3:271-274`, `KP14_RHO:276`, `CHAT:207`) and 20 modules doing top-level
  `from config import (...)`. Any scheme that `setattr`s spec values onto an
  already-executed `config` leaves every derived parameter stale — set `KP14_MU_H` and
  get a `KP14_LAMBDA_L` from the old value. **Must generate a frozen config module and
  inject it, not splat env vars.**
- **Storage is an 87x win, and it is easy.** `{id}_moments.pkl` is 2,882,927,638 B =
  **89%** of everything a run leaves behind (3.235 GB/panel measured). Neither analysis
  script needs `cond_var` after the run: `build_sdfwts_chars.py`'s entire use of it is one
  `np.linalg.solve(cond_var, rp)` per month, collapsing (N,N)→(N,). Doing that solve inside
  `utils/calculate_moments.py:79` — where both operands are already in RAM — is
  byte-identical, costs ~6 s/panel, and is **1000x smaller**. Per-panel survivors:
  2.943 GB → **33.8 MB**. Blocker to deleting moments earlier: `utils/evaluate_sdfs.py:171`
  genuinely needs full `cond_var` in-run.
- **`max_sr` is dead code** in the main path — computed every month at
  `calculate_moments.py:85`, never read by anything downstream.
- **BGN/KP14 variants are parameterizable; GS21 is not.** git copy-detection: BGN **68.5%**
  verbatim shared with main (C055, 7 hunks), KP14 **62.2%** (C050, 9 hunks) — and the
  variants already contain the collapse-to-baseline defaults (`gmult=[1,1]`,
  `type_share=[1.0]`, `type_bv=[0.0]`). GS21: **10.7% / 6.8%**, no git ancestor,
  genuinely different numerical method. **Treat the three models differently.**
- **GS bx7 solve-stage spec is unrecoverable** outside `run_gs_bx7.sh`. The five
  `solution.npz` embed a 15-element params array that **excludes `gs_bx` and `gs_ashift`**,
  so no solution file identifies its own beta. The β set survives in README/REPORT; what is
  uniquely lost is `gmreg=[0.6,3.0]`, the ashift ladder, the shares, and the soldir↔beta pairing.
- **`gs_ashift` inconsistency:** `gs_solve_reg.py:80-81` includes the shift in the payoff
  tables; `gs_sim_bx.py:212` omits it from `op_cash_flow`, which feeds `roe` — an exported
  characteristic — for four of bx7's five types.

### The empirical finding that reorders the experiment plan

See PLAN.md §0.0 (absorbed into §43). Short version, verified directly from
`variants/results/grid_summary.csv`:

- `corr(room, gap) = +0.675` over all 24 economies, but **−0.234 once the two KP
  priced-vol rows are dropped**. Room-as-a-design-rule rests on one channel.
- The **#2 and #3 gap economies (GS gamma(x), +0.041 and +0.038) have zero room** — their
  gap is ridge-beating-OLS estimation efficiency, not complexity. Both were deleted.
- **bx7 ranks 22nd of 24 on gap** (+0.0013). Demote it from the first production run.

---

---

## 11. Decisions (answered by Seth, 2026-09-04)

| # | question | answer |
|---|---|---|
| 1 | Restate the KP14 main-pipeline numbers? | **yes** |
| 2 | Which GS21 calibration is base? | **the one currently in `config.py`** — the variant hybrid becomes an `_asrun` spec only |
| 3 | GS21 `sigma_m`? | **5** (already `config.py:354`; the variant's 2.5 is the deviation) |
| 4 | Keep the bug-compatible `_asrun` twins runnable forever? | **yes** |
| 5 | `draw_pooling`? | follow recommendation — keep mean-of-ratios, state it explicitly in the spec |
| 6 | Is `room` defined on nested bases? | **require nesting** |
| 7 | First production economy = g0235? | **yes** |
| 8 | Ask Kerry now or at Phase 4? | **"kerry is fine with edits"** — governance resolved, no gate |

Standing calls:
- **GS21 method: use main's AR(1) + cubic + Gauss-Hermite.** The variant's Tauchen-exact
  discretisation is *not* adopted. This settles Phase 7's open method question in advance —
  the cross-check becomes informational, not a gate.
- `capture` will not be raised; accepted.
- Increased compute is acceptable.
- **Duplicate BGN/KP variant code may stay** — no rerun until all phases are done, so
  interim numerical equivalence between the trees is not required. This relaxes Phase 6
  substantially and removes the need for interim collapse gates.
- Interim viability of intermediate phases is not a constraint.

### Sequencing consequence to resolve

`PLAN.md` §0.0 (my analysis) recommends **vyx first**, because it is the only economy in
the grid with a gap of economically interesting size (+0.101 vs g0235's +0.030). `PLAN.md`
§10.7 (the synthesis) recommends **g0235 first**, because vyx carried the KP bug. Decision
7 endorses g0235. Since decision 1 also authorises the KP fix — now landed for the main
tree — the two are no longer in tension: **run g0235 first (cheaper, one solve rebuild, no
KP dependency), then vyx immediately after the variants-side KP fix lands.** Nothing is
lost either way; both run in Phase 1.

---

## 12. Phase 0 — progress

**Done 2026-09-04:**

- [x] **KP14 lambda-regime fix, main tree.** `config.py` `KP14_PROB_H` corrected to
  `MU_H/(MU_H+MU_L)` = 0.3191; `KP14_EXIT_H`/`KP14_EXIT_L` added; the three consumers
  (`panel_functions_kp14`, `sdf_compute_kp14`, `loadings_compute_kp14`) repointed at the
  exit rates at the import line. `kp14_fd.py` and `config.py:257` correctly left alone.
  Derivation in [../kp14_regime_labels.md](../kp14_regime_labels.md).
  **Verified: the solfile stamp still passes, so this needs re-simulation but NOT
  re-solving.** Guard: `tests/test_kp14_regime_labels.py`, 7/7 passing, all 7 fail under
  the old convention.
- [x] **`variants/common/dkkm_functions.py` WINDOW fix.** Both the live `mve_data` and the
  block-commented earlier copy. (Correction to an earlier note in this file: the first
  `mve_data` is inside a `'''...'''` block, i.e. commented out — not shadowed.)
- [x] **Spec transcription.** `experiments/specs/{var-bgn_gam-g0235-v1, var-kp_vy-vyx-v1,
  var-gs_bx-bx7-v1}.json`, each content-hashed over a canonical view that excludes prose
  fields. Guard: `tests/test_specs_match_shell.py`, 14/14 passing; mutation-tested — a
  0.005 perturbation of one `gs_ashift` in one of five solve stages trips two independent
  tests. **bx7's solve-stage spec is now recoverable from git.**

- [x] **KP14 lambda-regime fix, variants tree** (`variants/kp_vy/`). Could not be fixed at
  the import line like the main tree — that tree uses `from parameters_kp14 import *` —
  so the 14 exit-rate expressions were swapped individually with a token-safe scripted
  edit, reviewed line by line. Changed: `parameters_kp14.py` (`prob_H`, plus
  `exit_H`/`exit_L`), `panel_functions_kp14.py` (3 lines), `loadings_compute_kp14.py`
  (3), `sdf_compute_kp14.py` (8), `kp14_fd_vy.py` (2 — the OU generator `Qs`).
  Deliberately **left alone**: `kp14_fd.py` and `rebuild_kp_tables.py` G recombination
  (uses the entry rates, already correct) and the `(mu_H+mu_L)` sums (convention-
  invariant). Verified: `prob_H` = 0.319149, `lambda_L` = 0.367187 > 0, E[λ] = 1.000000000,
  and the OU generator's stationary P(high) = 0.319149. Guards extended to 13 tests.

  *Note:* `loadings_compute_kp14.py` is CRLF; the first edit silently normalised it to LF
  and showed as a 110-line diff. Redone preserving line endings — the diff is now 3 lines.
  Watch for this on any scripted edit in `variants/`.

**Remaining in Phase 0:**

- [ ] **Rebuild the vyx G/integ tables.** `kp14_fd_vy.py` is a solve-stage producer, so
  the fix invalidates them. **68 tracked table files** (`G_vyx*.csv`, `integ_vyx*.npz`)
  were built under the old generator. `meta_vyx.json` has been **deleted and untracked** —
  its cache key omitted `mu_H`/`mu_L` and any producer-source hash, so it would have
  silently served the stale tables. This is the staleness hole predicted in §4; it fired.
  Rebuild is running (long: a 1e6-iteration fixed point per type, then 63 integ jobs).
  **Until this completes and the tables are re-committed, no vyx run is valid.**
- [x] **Phase 0 item 5 — KP quadrature regression retired.** The plan said "delete
  `variants/kp_vy/integ_kp14.py` and import main's (78.9% identical)". **That is
  not feasible**: the two have different interfaces — the variant reads
  `KP_VY_TYPE`/`KP_GAMY_YIDX` and emits one `.npz` per (type, y-node), which is the
  entire point of `kp_vy`, while main's reads `config` and emits a single
  `integ_results.npz` with no notion of a y-grid or firm types. What transferred is
  the *technique*, ~5 lines, into both `integ_kp14.py` and `rebuild_kp_tables.py`:
  the density's own quantiles as the interval, and a mass check that **raises**
  instead of printing.

  Demonstrated rather than asserted — the old fixed `[0,10]` interval reproduced
  live:

  | sigma_eps | mass, fixed [0,10] | mass, adaptive |
  |---|---|---|
  | 0.20 *(shipped)* | 1.000000 | 1.000000000 |
  | 0.10 | 1.000000 | 1.000000000 |
  | 0.05 | 1.000000 | 1.000000000 |
  | **0.02** | **0.000000** | 1.000000000 |

  At the shipped `sigma_eps = 0.2` the two agree to **1e-13**, so this removes a
  landmine without moving any published number — and the in-flight rebuild's
  integral tables stay numerically valid. Guard: `tests/test_kp_quadrature.py`, 8/8.

- [ ] Measure and record the before/after delta of the KP fix on one panel.
- [x] **Pre-refactor fixtures — the plan step was based on a false premise, and is
  unnecessary.** `variants/` **did not exist before `bba735f`**; that commit *added* 130
  files. Its message describes dropping 18 economies, but that happened in the source
  tree (Alexander Ober's copy / the Dropbox `gap_experiments` folder), never in this
  repo's history. `git diff --diff-filter=D bba735f~1 bba735f` returns **0 files**, so
  `git show bba735f~1` cannot recover anything, and neither `gap_experiments/` nor
  `Code_light/` exists on this machine (both gitignored).

  **But no fixture is needed.** The collapse test wants a baseline to compare against,
  and the real baseline is in this repo: `utils_bgn/` and `utils_kp14/`. The variants
  ship the collapse defaults already — `variants/bgn_gam/parameters.py:34` has
  `gmult = [1.0, 1.0]` ("reproduces baseline exactly") and
  `variants/kp_vy/parameters_kp14.py:26-27` has `type_share=[1.0]`, `type_bv=[0.0]`.
  So Phase 5's collapse identities run **variant-at-defaults vs `utils_*`**, which is
  also the comparison that actually matters for Phase 6 unification. Plan corrected.
- [x] **Decision 6 — `room` nesting. It did not hold, and that is the negative-room bug.**
  Measured directly (5 chars, N=500; worst unexplained fraction of a linear rank column):

  | ceiling basis | cols | unexplained | nests? |
  |---|---|---|---|
  | `poly2` | 21 | 3.4e-28 | yes — it already contains `X_rank` |
  | `bins` | 301 | 6.1e-03 | **no** — decile dummies cannot reproduce a line |
  | `rff36` | 37 | 1.7e-02 | **no** |
  | `rff360` | 361 | 4.1e-04 | **no** |

  So `bins` and `rff*` could score *below* `lin_rank`, making `room = nl_ceil - lin_ceil`
  negative — exactly the two GS gamma(x) rows (room −0.0004/−0.0005 with gap +0.038/+0.041).
  Those numbers were never headroom; they were a basis-comparison artifact.

  **Fix:** `build_feature_sets(..., nest_linear=True)` now also emits `bins_n` and
  `rff*_n`, which append the linear columns (ranks, plus levels under `--levels`) so
  nesting holds by construction and `room >= 0`. The originals are kept, so every
  published number stays reproducible. Verified: all `*_n` bases nest to <2e-29.
  Guard: `tests/test_oracle_nesting.py`, 6/6 — including a test that fires if `bins`/`rff*`
  ever start nesting on their own, so the `_n` duplicates get removed rather than
  silently doubling compute.

  **Bonus finding:** `rff*_n − rff*` is a direct measure of *how much of the linear
  signal pure RFF fails to span*. That is the GS gamma(x) estimation-efficiency channel
  from PLAN.md §0.0, now measurable rather than inferred. Worth reporting alongside room.
- [x] **`--chars` fixed.** It now travels as `BOP_CHARS`, the same way `BOP_SCRATCH_DIR`
  already crosses the process boundary. `config.py` gained `_apply_chars_env()`, applied
  at module level so every access path sees it (not just `get_model_config`); `main.py`
  exports the env var *and* keeps the in-process mutation so its own view agrees with the
  children's. Also fixed: the unguarded `sys.argv[chars_idx + 1]` (raised `IndexError` on
  a bare `--chars`) and empty values.

  **Plus a new guard.** Output filenames carry no chars token, and `main.py` short-circuits
  the whole workflow when `{id}_results.pkl` exists — so fixing propagation alone would
  have created a *fresh* silent-wrong-number path (a `--chars` run in a directory holding
  a full-set run would return the full-set results as its own). The resume check now reads
  the `chars` field already recorded at `evaluate_sdfs.py:237` and **exits with an error**
  on mismatch instead of skipping.

  Guard: `tests/test_chars_override.py`, 10/10, including one that spawns a real
  subprocess exactly as `main.py:180` does — that is the test that would have caught the
  original bug. Invalid factor names now fail loudly at config import.

  Still outstanding (deliberately not done): the char→factor map is declared in **three**
  private copies (`main.py:92`, `fama_functions.py:66` and `:182`) while
  `config.CHAR_TO_FACTOR` sits unused. A test pins the canonical contents; collapsing the
  copies touches numerical code and belongs with Phase 2.

---

## 14. KP table rebuilds — cost and the staleness hole (2026-09-04)

**Does a new parametrization need a table rebuild? Yes, for every knob worth varying —
and, worse, *silently not* for seven that also change the tables.**

`build_vy_tables.py:9-13` caches on a hand-written 5-key stamp
`{bv, gv, ky, comp, rho}`. Tested empirically by perturbing each parameter and
recomputing the key:

| parameter | key notices? | but does it change the tables? |
|---|---|---|
| `type_bv` | detected | yes — enters `rho_ty`, `const_ty`, `util` |
| `gamma_v` | detected | yes — enters `rho_ty`, `const_ty` |
| `kappa_y` | detected | yes — sets `sigma_y` and the OU drift |
| `bv_comp` | detected | yes — scales the cash-flow level |
| `delta` | **MISSED** | yes — `const_ty` → A-coefficients → `util` |
| `theta_eps` | **MISSED** | yes — eps drift in the FD operator *and* A_1/A_3 |
| `theta_u` | **MISSED** | yes — A_2/A_3 |
| `sigma_eps` | **MISSED** | yes — eps diffusion in the FD operator |
| `lambda_H` | **MISSED** | yes — `lam_rate` scales `util` directly |
| `mu_H`, `mu_L` | **MISSED** | yes — the `Qs` regime generator ← *this is what just bit us* |

The producer source is not hashed either, so editing `kp14_fd_vy.py` also reuses silently.

**Consequences for the experiment sequence:**

1. The four *detected* knobs are precisely the gap-driving ones REPORT identifies, so
   **every new KP economy pays a full rebuild.** Cost = `ntypes` G solves + `ntypes × NY`
   integral jobs, linear in the number of exposure types. Each G solve is one
   42,000 × 42,000 sparse LU factorisation followed by an iterative loop to 1e-8;
   currently **~35+ min per type**, so a 3-type economy is a multi-hour job before the
   oracle can even start. Budget this per KP spec.
2. The seven *missed* parameters are a live wrong-numbers path, not a theoretical one.
3. `build_vy_tables.py:18` runs the solve with `stdout=subprocess.DEVNULL`, so
   `kp14_fd_vy.py`'s own `flush=True` progress prints are discarded — **a multi-hour
   rebuild shows nothing at all**, no iteration count, no ETA. Fix this before running
   the experiment loop; a silent multi-hour job is indistinguishable from a hung one.

**Fix, and it should move up the plan.** `utils/solfile_stamp.py` already does exactly the
right thing for the main tree: it AST-parses the producer's own import statements to
derive the parameter list, hashes those values plus the producer's source, and hard-fails
at import. Pointing `build_vy_tables.py` at that mechanism instead of the hand-rolled key
would have caught `mu_H`/`mu_L` automatically. This was Phase 7 work; it belongs in
Phase 0/1, because every KP experiment depends on it and KP is the highest-gap model.

Artifact sizes are small — ~2.4 MB of G CSVs plus ~5.7 MB of integral NPZs per economy,
and tables are namespaced by prefix so economies accumulate rather than overwrite. Git is
a fine home for them; the problem is staleness detection, not size.

---

## 15. Solve provenance generalized to all three models (2026-09-04)

Moved up from Phase 7 per instruction. `variants/common/solstamp.py` now gives BGN,
KP and GS one content-addressed identity + registry. 23/23 tests in
`tests/test_solstamp.py`.

### Why it could not just reuse `utils/solfile_stamp.py`

That module derives a producer's parameter list by **AST-parsing its import
statements**. The variants use `from parameters import *`, so the import statement
enumerates nothing. `solstamp` instead snapshots the producer's *entire effective
parameter namespace* after overrides, plus the sha256 of every source file that can
change the output. Over-capture only ever triggers an unnecessary rebuild;
under-capture produces wrong numbers — that is the right side to err on.

### What each model had wrong, and what it has now

| model | before | after |
|---|---|---|
| KP | hand-written 5-key stamp; **missed 7 params** + producer source | full namespace + 4 source digests |
| BGN | **no stamp at all** — the output filename was the only identity | full namespace + 3 source digests + `JSTAR_TOL` |
| GS | 15-element `params` array omitting `gmreg`/`gs_bx`/`gs_ashift`/`p01`/`p10`/grids; shell guard was a bare `[ -f solution.npz ]` | full namespace, `solve_id` + identifying params written into the npz, guard removed |

Verified empirically, not by inspection:
- KP: all 11 tested parameters now move the id, **including all 7 the old key
  missed**; a whitespace-only edit to `kp14_fd_vy.py` moves it, and reverting
  restores it (a true content hash, not a counter).
- BGN: `gmult`, `sigma_z`, `kappa` and `JSTAR_TOL` all move the id.
- GS: `gs_bx`, `gs_ashift`, `gmreg`, `xnum`, `tol`, `sigma_m` all move the id, while
  **the same parameters written to a different `outdir` keep the same id** — so an
  identical solve stays reusable.

### Two bugs I introduced and caught in testing

1. `extra` is descriptive and deliberately unhashed, but I first put BGN's
   `JSTAR_TOL` there — it changes the table, so it was silently ignored. Added a
   separate hashed `env_params`.
2. GS snapshots `globals()`, which contains `outdir` and `HERE`, so the id was
   machine- and directory-dependent. Added `skip=`.

Both are now pinned by tests (`test_env_params_are_hashed`,
`test_skip_excludes_path_plumbing`).

### The registry — durable record of old experiments' solfiles

`experiments/registry/<solve_id>.json`, committed. A few KB each; hundreds of
experiments cost a few MB. Records parameters, source digests, and every artifact
with size and sha256. **The manifest outlives the artifact** — after scratch is
purged you can still say which parameters produced a number, and re-running the
producer reproduces the same `solve_id`.

    python variants/solfiles.py list | show <id> | diff <a> <b> | check | gc --dry-run

Size policy: artifacts <= 32 MB are committed (BGN ~KB, KP ~8 MB); a `gs_bx` economy
is five ~85 MB solutions and stays out of git, manifest-only. Each manifest's
`committable` field says which case applies.

### Also fixed

- **The silent multi-hour solve.** `build_vy_tables.py` ran the G solve with
  `stdout=subprocess.DEVNULL`, discarding `kp14_fd_vy.py`'s own `flush=True`
  iteration prints — a running job was indistinguishable from a hung one. Now
  streamed and prefixed. Pinned by `test_kp_streams_solver_progress`.
- `KP_VY_ADOPT=1` records tables already on disk without re-solving, so a build
  predating the registry does not cost another multi-hour rebuild to get a manifest.

### Consequence for the plan

**Every new parametrization needs a new solve** — the knobs that move the gap are
exactly the ones that enter the value functions. Documented in `variants/README.md`
with a per-model cost table. Budget the solve stage separately from the oracle
stage: KP is ~45 min *per type* (multi-hour for a 3-type economy), GS ~a minute per
type but ~85 MB each, BGN minutes and small.

---

## 16. Staged solve ids (2026-09-04)

A design flaw in §15, found while sequencing Phase 0 item 5: `solstamp` bundled every
source file into one `solve_id`, so touching the **cheap** producer (`integ_kp14.py`,
63 short jobs) invalidated the **expensive** one (`kp14_fd_vy.py`, ~45 min per type).
Fixing the quadrature would have forced a full G re-solve — hours, for a change that
cannot affect the G tables at all.

`utils/solfile_stamp.py` already models the right structure with its `inputs` field.
`solstamp` now does too: `snapshot(..., stage=..., inputs=artifact_digests([...]))`.

KP is now two independently cached stages, with the driver deliberately **excluded**
from both source lists — it only orchestrates, and including it would recreate the
coupling the split removes:

| stage | sources | artifacts | cost |
|---|---|---|---|
| `G` | `kp14_fd_vy.py`, `parameters_kp14.py` | `G_<prefix><f>.csv` | ~45 min per type |
| `integ` | `integ_kp14.py`, `parameters_kp14.py`, **+ the G tables as `inputs`** | `integ_<prefix><f>_<iy>.npz` | 63 short jobs |

Verified end to end on the real files:

- edit `integ_kp14.py` → **G id unchanged**, integ id moves ← the point of the split
- edit `kp14_fd_vy.py` → G id moves
- perturb `G_vyx0.csv` → integ id moves (mode 3: upstream moved, downstream follows)
- restore everything → both ids return to baseline (a true content hash, not a counter)

Also: the integ driver now keeps subprocess **stderr** and re-raises. `integ_kp14.py`
raises on mass loss, and the previous `stderr=DEVNULL` would have swallowed exactly
the signal the new guard exists to produce.

Tests: `tests/test_solstamp.py` 27/27, `tests/test_kp_quadrature.py` 8/8.

---

## 13. Task checklist

- [x] Job 1 understood and written up (§1)
- [x] Job 3 divergences catalogued, 12 verified (§2)
- [x] Main pipeline mapped (§3)
- [x] Spec/summary gap characterized (§4)
- [x] Repo-merge recommendation (§5)
- [x] Integration architecture chosen — spec_first + unify config discipline
- [x] Plan written to docs/refactor/PLAN.md
- [x] Plan presented to Seth + artifact published
- [x] **8 decisions answered — see §11**
- [x] **Phase 0 complete (2026-09-05)** — vyx tables rebuilt and manifested
      (`f2be637b9fec1f80` G, `d8d686da014f429a` integrals, 66 artifacts intact); G solver
      replaced by a direct solve; KP lambda delta measured at the primitive level
      (E[lambda] 1.7172 -> 1.0000); DKKM draw-averaging fixed in `utils/evaluate_sdfs.py`
      before any panel exists. Commits dd8c07e, 9e1d47d, e7df1a3.
- [ ] Phase 1 — seeded oracle array, first cluster run (g0235, then vyx)
      - [ ] BGN `Jstar_g0235.csv` has no manifest; `rebuild_jstar_gam.py` has never run
            under solstamp. Blocks g0235.
      - [ ] `gs_bx` has no solve artifacts at all (`sol_reg/` absent). Blocks bx7.
      - deferred, not blocking: the integral-stage quadrature asks `epsrel=1e-10` and
        emits ~985 IntegrationWarnings per job that roundoff prevents reaching it — the
        same unachievable-tolerance shape as the G solver, costing unknown wasted work.
        Changing it moves `d8d686da014f429a` and costs an 88 min rebuild, so measure the
        accuracy/time trade before touching it.
      - not a problem: `variants/run_estimators.py:167-169` already builds the RFF
        ensemble by averaging *weights* across draws (`rff_ens`, `mat=-1`) and evaluating
        that portfolio against Sigma. The Phase 1 path was never affected by the
        `dkkm_avg_results` bug; `utils/evaluate_sdfs.py` has now been brought in line
        with it.

## §17 — the duplicate-builder incident, and the single-builder lock (2026-09-04)

**What happened.** Resuming after a context compaction I checked pid **27880** for the
in-flight vyx rebuild, got "not running", and concluded the rebuild had died. 27880 was
never the driver — the driver is **27878**; 27879/27880 are siblings of the shell
pipeline. The rebuild had been alive the whole time: G type 0 finished at 14:04 after
~98 min, and type 1 was 50 min in.

On that false premise I launched a second builder against the same prefix. For ~42
minutes the two competed for the same 10 cores and were aimed at the same
`G_vyx0.csv`. The starvation is measurable: the survivor's CPU went **453% → 915%**
the instant the duplicate was killed. The real hazard is not the waste — it is that a
concurrent writer can hand the integ stage a **half-written G table**, which is silent
numerical corruption rather than a crash.

Two further corrections to what §16 and the session log said:

* "~45 min per G type" is wrong. It is **~90–100 min per type**; three types plus
  integ is a ~5-hour job. A transient read of the first seconds of convergence
  (err 1.1e7 → 1.7e2 in 7s) was mistaken for near-completion; the solve actually runs
  ~75,000 iterations to reach its tolerance floor.
* The dead-looking 0-byte log was the *old* driver's `stdout=DEVNULL` for the G stage,
  not evidence of a dead process. The rewrite streams it precisely so that a silent
  multi-hour solve is distinguishable from a hung one.

**The fix — one builder per prefix.** `build_vy_tables.py` now takes an exclusive
`O_CREAT|O_EXCL` lock (`.build_vy_tables.<prefix>.lock`, gitignored) recording pid and
start time, released via `atexit`. A second builder refuses with the holder's pid; a
lock held by a dead pid is taken over, so a killed build never wedges the prefix.

**The lock's own first version was broken, in the same shape as the bug it guards.**
`os.kill(1, 0)` as a normal user raises `PermissionError`, which is an `OSError`, so
`except (OSError, ProcessLookupError): return False` declared init *dead*, took the
lock, and launched a third solve — the exact duplicate it existed to prevent. EPERM
means the process exists and is not ours to signal. Only `ProcessLookupError` means
dead. Pinned by `tests/test_build_lock.py` (7 tests, incl. the pid-1 case), which
drives the real driver through `KP_VY_ADOPT_G` so the test can never start a solve.

**Also added (§17a): per-type G checkpoints.** `G_<prefix><f>.solveid` records the
stage `solve_id` *and* the table's digest, so a restart skips only types that are both
parameter-current and byte-intact — existence alone never satisfies it, unlike the bare
guard removed from `run_gs_bx7.sh`. `KP_VY_ADOPT_G=<types>` stamps tables whose
provenance the operator asserts. Type 0's table was **not** adopted: its provenance
rested on mtimes and commit times, and asserted provenance in the flagship KP economy
is the failure this whole subsystem exists to abolish.

**Standing rule for this repo.** Before starting any long solve, check for a live
builder by *name* (`ps -eo pid,args | grep build_vy_tables`), never by a remembered pid.
The lock now enforces this, but the habit is what generalises to BGN and GS.

**Cluster implication for Phase 1.** Local concurrency across G types is not worth
building: on SLURM the three types are naturally three array tasks, and the per-type
`solve_id` checkpoint is exactly what makes such an array safe to restart after a
walltime kill. That belongs in the Phase 1 wrapper.

### §17b — G_vyx0.csv was solved by pre-fix code (2026-09-04, decisive)

The driver 27878 started at **12:25:03**. The mu_H/mu_L swap in `kp14_fd_vy.py` — the
core of the regime-label fix — was committed in `83f28d3` at **12:49:54**, ~25 minutes
*after* type 0's subprocess had already launched and read the file:

```
-    Qs[2*iy, 2*iy+1] += mu_H          +    Qs[2*iy, 2*iy+1] += mu_L
-    Qs[2*iy+1, 2*iy] += mu_L          +    Qs[2*iy+1, 2*iy] += mu_H
```

So the `G_vyx0.csv` written at 14:04 is a **pre-fix** table. Refusing to adopt it was
correct, for a stronger reason than the one given at the time (mtime-based doubt): this
is positive evidence of staleness, not absence of evidence of currency.

The same reasoning clears the other two. Type 1 launched 14:05 and type 2 at 16:05,
both after the last change to either G-stage source, and `find -newermt "14:05"` over
`variants/kp_vy/*.py` returns only `build_vy_tables.py` — the driver, which is in
neither stage's source list by design. Type 1 (landed 16:05:23) is checkpointed under
`solve_id c1421d0d0e6126cb`.

**Generalisation.** A long solve reads its sources *once, at subprocess launch*. Editing
those sources mid-run neither invalidates nor updates the run — it just silently
decouples the artifact from the tree. `solstamp` catches this only at *record* time, and
the old driver recorded nothing. Two habits follow: never edit a producer while it is
solving, and prefer the per-type checkpoint written immediately after each type, which
binds the artifact to the sources as they were when it landed.

### §17c — detached-run state (2026-09-04 16:10)

Everything now runs at `ppid 1`, immune to the terminal, VSCode, and the Claude session:

| pid | work | started | ETA |
|---|---|---|---|
| 41743 | G type 0 re-solve -> `G_vyx0.new.csv` | 16:09 | ~17:40 |
| 41884 | G type 2 re-solve -> `G_vyx2.new.csv` | 16:14 | ~17:45 |
| 41925 | `_scratch/finish_vyx.sh` | 16:10 | after both |

The finisher waits on both pids, **validates** each new table (1000x44, no NaN) before
promoting it over the live name, stamps types 0 and 2, then runs the driver for the G
manifest, the 63 integ jobs, and the integ manifest. It aborts rather than promote a
truncated table. Solves write to `*.new.csv` so nothing can read a torn file.

Machine sleep suspends and resumes these cleanly; only wall-clock is lost. Session death
does not touch them.

### §17d — DEFERRED FIX: kp14_fd_vy.py saves an unconverged G and calls it converged

```python
for it in range(1_000_000):
    ...
    if err < 1e-8:
        break
# <-- no check here; falls through on exhaustion
out.to_csv(gout)
print(f"saved {gout} (converged iter {it}, err {err:.2e}, ...)")
```

If the loop exhausts its 1,000,000 iterations without reaching tolerance it writes the
table anyway and prints **"converged"**. That is the quadrature landmine's exact shape:
a silently wrong artifact plus a message asserting success. It is not hypothetical here
— `err` is not monotone near the floor (observed oscillating between 4e-8 and 6e-7
around iteration 75,000), so a slightly harder parametrization could plausibly sit above
1e-8 until the cap.

Fix (one guard, after the loop):

```python
if err >= 1e-8:
    raise RuntimeError(f"G solve did not converge: err {err:.3e} after {it+1} iters")
```

**Deliberately NOT applied yet.** Editing `kp14_fd_vy.py` changes the G-stage
`solve_id`, which would void the type-1 checkpoint already written under
`c1421d0d0e6126cb` and make the finisher's `KP_VY_ADOPT_G=0,2` stamp a *different* id —
so the driver would then re-solve type 1 from scratch. Apply it only once all three
tables and both manifests are recorded, and accept that it moves the G solve_id (a
producer change *should* invalidate the solve; that is the design working).

**Rule this instances:** a producer is frozen while any solve using it is in flight.

### §17e — measured: E[lambda] delta (2026-09-04)

`_scratch/lambda_delta.py`, probing both trees' `parameters_kp14`:

| quantity | before (f6ba305) | after | change |
|---|---|---|---|
| prob_H | 0.6809 | 0.3191 | -0.3617 |
| prob_L | 0.3191 | 0.6809 | +0.3617 |
| lambda_H | 2.3500 | 2.3500 | same |
| lambda_L | 0.3672 | 0.3672 | same |
| **E[lambda]** | **1.7172** | **1.0000** | **-0.7172** |

mu_H, mu_L, lambda_H, lambda_L are all unchanged; only the stationary regime weights
move. The fixed economy hits the KP14 normalisation **exactly** (1.0000), which is
independent confirmation of the convention chosen in `docs/kp14_regime_labels.md` —
the alternative labelling both misses the normalisation by 72% and requires the
infeasible lambda_L = -1.881.

Panel arm (`_scratch/panel_delta.py`) runs the same seeds through both trees and
compares SR_max, E[mu], cross-sectional sd(mu), and the per-basis conditional-oracle
and constant-theta ceilings. It waits on the rebuilt tables.

### §17f — the oracle's cost is the RFF sweep, not the simulation

Calibrating the panel arm on a 60x60 kp_vy panel (before arm, seed 0):

| basis config | wall | SR_max | E[mu] |
|---|---|---|---|
| `--rff 36,360,3600 --nmat 2` (default) | **1763 s** | 0.6451 | 0.0156 |
| `--rff 36 --nmat 1` | **39 s** | 0.6451 | 0.0156 |

45x cheaper, and the reported moments are **identical** — as they must be: `sr_max_mean`,
`mean_mu`, `sd_mu` and `mean_idio_sd` all come from the monthly time series, which the
feature basis never touches. The basis only enters the per-basis ceiling rows
(`cond_oracle`, `const_z0`).

Consequence for every future delta/diagnostic run: **choose the basis by what is being
compared.** Moments-only comparisons (this delta, sanity checks, regressions against a
prior parametrization) should run `--rff 36 --nmat 1`. Only ceiling/room/gap comparisons
need the full sweep, and those should be budgeted as ~45x the moments run.

This matters at the P/T ratios the gap work uses: the default sweep spends essentially
all its time building and inverting 3600-column designs twice per month for numbers a
moments comparison discards.

### §17g — oracle panel cost model (measured, `--rff 36 --nmat 1`)

Four points on the before arm, under contention from the two G solves:

| N | T | wall |
|---|---|---|
| 60 | 60 | 39 s |
| 120 | 60 | 65 s |
| 240 | 60 | 114 s |
| 60 | 120 | 74 s |

Linear in both, `r2 = 0.9998` in N at fixed T:

```
t ~= 4 + T * (0.175 + 0.0069 * N)   seconds
```

So the flagship `--N 500 --T 500` oracle is ~30 min per seed with a small basis, and the
`--N 300 --T 360` delta run ~14 min per seed per arm — about 55 min for a two-seed,
two-arm comparison, less once the G solves release their cores. With the *default*
`--rff 36,360,3600 --nmat 2` the same delta would be ~40 h.

`sr_max_mean` rises materially with N (0.645 / 0.813 / 0.961 at N = 60 / 120 / 240) —
it is the max Sharpe ratio attainable from N assets, so **it is not comparable across
panel sizes.** Any before/after or economy/economy comparison has to hold N fixed, and
any headline SR must carry its N.

### §17h — the G iteration cannot reach its own tolerance (2026-09-04 19:04)

The two re-solves ran 2h54m and **made no progress after the first 20 minutes**. Both
hit a residual floor by iteration ~30,000 and random-walked for 228,000 iterations more:

| iter | type 0 err | type 2 err |
|---|---|---|
| 30,000 | 1.48e-07 | 5.01e-06 |
| 90,000 | 6.39e-07 | 5.29e-06 |
| 150,000 | 1.37e-07 | 4.09e-06 |
| 210,000 | 4.66e-07 | 5.28e-06 |
| 240,000 | 9.16e-08 | 4.38e-06 |
| **best ever seen** | **1.96e-08** | **2.39e-06** |

Tolerance is `err < 1e-8`. Type 0 never got within 2x of it in 258,000 iterations; type 2
never within 240x. There is no trend — this is a floor, not slow convergence.

**§17d was not hypothetical; it was about to fire.** Left alone, both would have run out
the 1,000,000-iteration cap (~7 more hours), fallen through the loop, written their
tables, and printed `converged`. Two of the three flagship KP tables would have been
silently wrong-but-labelled-right. Killed at 19:04. The finisher's validation guard did
its job: it aborted rather than install, and the live tables were left untouched with no
`.new.csv` debris.

**Type 1 is genuinely fine.** It ran 14:05->16:05 = 121 min at ~25 iters/s ~= 181,000
iterations, well under the 1e6 cap, so it exited on the tolerance test rather than by
exhaustion. It stays valid and checkpointed under `c1421d0d0e6126cb`. That types 0
(`type_bv=0.02`) and 2 (`0.14`) floor out while type 1 (`0.07`) converges says the floor
is parametrization-dependent.

**Diagnosis.** `err = np.linalg.norm(Gn - G)` is an *absolute* Frobenius norm over a
1000x42 array. Its floor is set by the accuracy of the sparse LU solve on a 42,000x42,000
system, which scales with `||G||` — so a tolerance that is comfortable for one `type_bv`
is unreachable for another. A fixed absolute threshold cannot be right across types.

**Proposed fix (not yet applied).**
1. Make the criterion **relative**: `||Gn - G|| / max(||G||, 1) < tol`.
2. Add **stall detection**: if the best residual has not improved by, say, 2x over the
   last 20,000 iterations, stop — and `raise`, reporting the floor reached.
3. Keep the §17d guard: never write a table after loop exhaustion.

This also rewrites the cost story. The useful work is **~20 minutes per type**, not two
hours; the rest was spin. Three types should be ~1 hour total, not ~6.

**Consequence for the tolerance choice:** whether a ~1e-7 (type 0) or ~2e-6 (type 2)
residual is *good enough* is a numerical-accuracy question about the resulting expected
returns, not a matter of taste. Before picking a new tolerance, measure how much the G
table — and the premia derived from it — move between iteration 30,000 and 250,000. If
they are stable to many digits, the floor is harmless and the tolerance was simply set
below what the linear solve can deliver.

### §17i — the G "solve" was a linear system all along (2026-09-05)

The iteration
```
Mat @ G_new = G/dt + util,    Mat = I/dt - (F + Q)
```
has fixed point `[I/dt - (F+Q)] G = G/dt + util`, i.e.

```
(F + Q) G = -util
```

which is **linear**. One `spsolve` replaces the entire loop:

| | before | after |
|---|---|---|
| per G type | ~2 h, then plateau, never converging | **0.10 s** |
| three types | projected 6+ h | **2 s** |
| tolerance question | unanswerable | none — a residual check |
| relative residual | n/a (absolute norm) | 7e-13 to 9e-13 |

**Why the old tolerance was unsatisfiable, not merely strict.** `err` was an *absolute*
Frobenius norm compared against `1e-8`, but `||G||` is 3.05e7 (type 0), 7.09e7 (type 1),
1.96e8 (type 2). The test therefore demanded relative precision of **3.3e-16, 1.4e-16 and
5.1e-17** — at or below float64 eps (2.2e-16). Types 1 and 2 asked for better than the
machine can represent. No tolerance tuning could have fixed this; the norm had to become
relative, and once it does, the direct solve is obviously better than iterating to it.

**The plateau was the answer.** Taking one step of the *original* iteration from the
direct solution moves `1.5e-13` to `2.1e-13` relative — the same order as the observed
plateau. The iteration had reached machine precision by ~30,000 iterations; the other
228,000 were noise around the correct point. Type 1's earlier "convergence" was a lucky
random-walk dip below an unreachable threshold, not a meaningful stopping event.

**Validation.** The direct solution reproduces the two tables that were current to
`2.9e-13` (type 0) and `2.2e-13` (type 1) relative. Type 2 differs by **6.2% relative,
39.8% elementwise max** — that table was the stale pre-fix one from git, exactly as
expected, and is now the only KP number in this economy that actually moves.

### §17b CORRECTION — G_vyx0.csv was not stale after all

§17b argued that `G_vyx0.csv` (written 14:04 on 2026-09-04) was produced by pre-fix code,
because the driver launched at 12:25:03 and the `mu_H`/`mu_L` swap was committed at
12:49:54. **That inference was invalid: commit time is not edit time.** The swap was
already in the working tree when the subprocess read it; the commit merely recorded an
earlier edit.

The direct-solve probe settles it empirically — the post-fix direct solution matches that
table to `2.9e-13`, which a regime-label difference could not survive (type 2's genuine
staleness shows up as 6.2%). The table was current.

What stands from §17b is the general lesson, which is if anything strengthened: a long
solve reads its sources once at launch, and **file mtimes and commit timestamps are not
evidence about what code a running process is executing.** The reliable check is
numerical — recompute and compare — which is now cheap enough to be routine.

## §18 — Phase 0 table rebuild COMPLETE, and the DKKM `mat` bug (2026-09-05)

### The vyx tables are done

```
solve_id           model     size  git  files  tags
f2be637b9fec1f80   kp_vy    2.3 MB  yes     3  vyx     <- G stage
d8d686da014f429a   kp_vy    5.5 MB  yes    63  vyx     <- integral stage
```

`solfiles.py check` reports all 66 artifacts present and intact; both are small enough to
commit. Integral stage took 5263 s (~88 min) for 57 jobs after adopting the 6 built before
the restart. This is the **first content-addressed solve in the registry**, and it closes
the "rebuild the vyx tables" item that has been open since 2026-09-04.

### ASU session's Job 5 handoff: verified, with corrections

`_scratch/handoff-job5-repo-merge.md` +
`FINDINGS-repo-merge.md` (absorbed into §43). Its three flagged
corrections to §5 were each checked rather than accepted; see §5 for what landed. Net:
the path-limited-log point was **right**, the `config` count was **right on the file
metric** (41 files, though 46 import lines and 156 attribute uses also exist — the
original "42" matched none), and the `.git` figure was **right that +4.4% was wrong but
wrong in the replacement**: it is +3.5 MB / +5.2%, and the graft itself is free.

### The finding that matters: DKKM Sharpe is biased downward, right now

Verified independently in the source:

1. `utils/evaluate_sdfs.py:9` — since `dc48c98`, `dkkm_results` carries **one row per
   random-feature draw** (`mat`), with the docstring instruction *"average across mat
   downstream."* `fama_results` (line 8) has **no** `mat`.
2. `analyze.py` **never references `mat`** — its DKKM docstring (line 205) still lists the
   pre-`dc48c98` columns.
3. `analyze.py:215` computes `sharpe = mean / stdev` **per row**, i.e. per (month, draw).
4. `analyze.py:233` groups by `['alpha','nfeatures','iter']`. `mat` is neither a key nor
   in `agg_dict`, so pandas silently drops it and the **per-draw Sharpes are averaged**.

So the reported DKKM Sharpe is the mean of per-draw conditional Sharpes, not the Sharpe of
the draw-averaged portfolio. Since `mean = w'rp` is linear in `w` while
`stdev = sqrt(w' Σ w)` is convex, averaging draws leaves the mean unchanged and strictly
lowers the risk. **The error lands only on DKKM** (Fama has no draw dimension), the
treatment arm of the entire gap question.

**Precision about the direction.** Two facts are theorems: `mean` is exactly linear in
`w`, and `||w_bar||_Σ <= mean_i ||w_i||_Σ` (triangle inequality for the Σ-norm), which
together give `Sharpe(w_bar) >= (mean_i mu_i)/(mean_i sigma_i)`. But what analyze.py
reports is the **mean of ratios**, and Sharpe is scale-invariant in `w`, so each per-draw
Sharpe is the Sharpe of that draw's *direction*. Mean-of-ratios versus ratio-of-means is
not ordered in general, so "biased downward" is **not** a theorem — an earlier statement
of this in conversation overstated it.

It is, however, extremely robust. `_scratch/mat_bias_probe.py` (synthetic: draws are the
true MVE weights plus i.i.d. noise, 400 trials per noise level, 8 draws, N=60) finds
`Sharpe(w_bar) >= mean-of-ratios` in **100% of 2000 trials**, with both provable
inequalities asserted every trial and never violated:

| draw noise | correct | reported | understatement | risk drop |
|---|---|---|---|---|
| 0.25 | 0.1533 | 0.1457 | 1.05x | 0.95 |
| 0.50 | 0.1495 | 0.1260 | 1.19x | 0.84 |
| 1.00 | 0.1396 | 0.0903 | **1.55x** | 0.65 |
| 2.00 | 0.1102 | 0.0519 | **2.12x** | 0.47 |
| 4.00 | 0.0682 | 0.0266 | **2.57x** | 0.39 |

This is illustrative, not a measurement of the real bias: it is synthetic, uses 8 draws
where `config.NMAT = 5`, and the true RFF draw dispersion is unknown. But the scale of the
effect — plausibly tens of percent to more than 2x — is large enough that it could decide
whether DKKM appears to beat FM at all. The real magnitude stays unmeasured; no
`results.pkl` exists on this laptop.

### Why this must be fixed BEFORE Phase 1, not after

`results.pkl` stores only the three DataFrames plus metadata (`evaluate_sdfs.py:232-244`).
**`cond_var` / Σ is never saved** — it is read at line 172 and discarded. Therefore:

| quantity | in `w` | recoverable downstream from saved rows? |
|---|---|---|
| `mean`, `xret` | linear | **yes** — average across `mat` |
| `stdev` | convex quadratic | **no** — needs Σ, which is gone |

So this cannot be repaired in `analyze.py`. `utils/evaluate_sdfs.py` has to emit the
draw-averaged portfolio's own `stdev`. And a 10-panel run is ~80 GB on scratch that is not
retained — **every panel generated before the fix is permanently missing the number**, and
regenerating means re-running the whole simulation. Phase 1 is the next thing queued.

Recommendation: fix `evaluate_sdfs.py` in Phase 0, now, ahead of any Phase 1 run.

## §19 — measurement protocol, learned the hard way (2026-09-05)

Three cost figures given to Seth today were wrong, all by extrapolation rather than
measurement, and a fourth attempt was corrupted by a shared machine. The pattern is worth
writing down because every phase from here involves sizing cluster jobs.

**What went wrong**

| claim | basis | truth |
|---|---|---|
| kp_vy G solve "~45 min per type" | guess | it never converged at all |
| revised to "~2 h per type" | measured the *spin*, not the work | 0.10 s once solved directly |
| RFF basis costs "45x" | measured at N=60 on kp_vy | 6.5x at N=100 on bgn_gam |

The RFF multiplier is the instructive one: it is a *ratio* between a simulation cost that
grows with N and a basis cost that grows faster, so quoting it without its N is
meaningless. Extrapolating the N=60 ratio to the N=500 flagship would have overstated the
job by ~7x.

**Neither wall nor CPU time is valid on a contended box.** Sharing the laptop with the
ASU session, the same job measured:

| T | cpu | wall |
|---|---|---|
| 30 | 2872 s | 550 s |
| 60 | 1802 s | 331 s |

Halving the months took 1.6x *more* CPU and 1.7x *more* wall. Both metrics inverted.
Wall time fails for the obvious reason; **CPU time fails because threaded BLAS
spin-waits**, burning user time without doing work, at a rate that varies with load. The
assumption that user+sys is contention-robust is wrong — it survives time-slicing, not
spinning.

**Protocol from here**

1. **Pin BLAS threads** (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
   `VECLIB_MAXIMUM_THREADS`) for any run whose cost is being measured, and in every SLURM
   script — an unpinned job on a shared node wastes its allocation spinning.
2. **Only compare runs from the same window.** The one trustworthy number today was
   N=200 vs N=100 back-to-back under identical load: 676/324 = 2.09x, so the oracle is
   **linear in N**. That rules out the O(N^2) dual-solve scaling, which was the only
   structural reason the flagship could have been expensive.
3. **Never quote a ratio without the point it was measured at.**
4. **Concurrency protocol:** two sessions can share this box for independent *work* —
   file boundaries have held all day, no collisions — but neither can take a valid
   *timing* while the other runs. Cost measurements must be serialized or deferred to the
   cluster. Add this to the TASK briefs.

**Standing observation:** `variants/README.md` says gs_bx solves take "~a minute each".
The ASU session's first one passed **1 h 01 m** without producing `solution.npz`. That
estimate was mine and unverified, in the same family as the two above. Whether this is
slow-but-fine or the same unachievable-tolerance failure the KP G solver had
(`gs_solve_reg.py 161 1e-6`, Gauss-Seidel with damping and policy-freeze) is for the ASU
session to report — it is question 2 in their brief.

## §20 — GS21 was unrunnable, and I broke it (2026-09-05)

`variants/gs_bx/gs_solve_reg.py` has been a **hard `SyntaxError`** since commit `3e0b6ae`
— mine. The solstamp rewrite added `gmreg=gmreg` to `np.savez_compressed` at line 266
when line 246 already passed it; duplicate kwargs are rejected at *parse* time, so the
module could not be imported at all. That, not scheduling, is why `sol_reg/` never
existed. The ASU session found it and fixed it (dropping the 266 occurrence — the line I
added, so the minimal change).

**My verification of this file was structurally incapable of catching it.** The 09:45
probe that "confirmed `xnum` and `tol` reach the hash" did
`src.split('print(f"[solstamp] gs_bx')[0]` and `exec`'d only that prefix. Line 266 is far
below the split point, so the probe compiled a fragment that parses fine. I then wrote in
the task brief that the file was "unproven code — never executed since the rewrite"
without checking whether it *could* be executed. `python -c "import py_compile;
py_compile.compile(f, doraise=True)"` would have caught it in a second. **Any probe that
executes a slice of a file has not established that the file runs.** The solve_ids that
probe reported are also obsolete — they hashed the broken source.

**Freeze the fix.** Deleting 246 instead of 266 gives byte-identical `.npz` output but a
different source digest, hence a different set of five `solve_id`s. The choice is
arbitrary; churning it later orphans every recorded GS manifest. Dropping 266 stands.

### Tolerance audit — the KP failure's milder cousin, verified

| | KP `kp14_fd_vy.py` | GS `gs_solve_reg.py` |
|---|---|---|
| tolerance | **absolute** vs `‖G‖~1e8` — unsatisfiable | **relative** (`qerr/vscale`) — satisfiable |
| cap | `range(1_000_000)`, falls through and writes | `it == 5600` breaks, falls through and writes |
| message on cap exit | prints `converged` | prints `converged` |
| auditable? | no | yes — `cycle-averaged; stopping` prints first |

The fatal half is absent, but the cap still writes an artifact, records a manifest and
reports success. Two further defects: `for it in range(60000)` at line 148 is **dead**
(the 5600 break makes 5601 the ceiling — anyone budgeting from 60000 is off 10x), and
line 211 tests against `tol * 20`, so a CLI `1e-6` actually requests **2e-5**.

### The provenance smell this exposes — §15/§16 gap

`tol` is hashed into the `solve_id` and recorded in the manifest. If GS habitually exits
on the sweep cap rather than the tolerance test — and `sol_reg`'s residual was decaying at
0.99924/sweep at sweep 1400, which cannot reach tolerance by 5600 — then **`tol` is
decorative for this economy: different `tol`, different `solve_id`, identical numerics**,
and the manifest claims a precision the solve never targeted (compounded 20x by line 211).

The general defect: **solstamp hashes *requested* parameters and records nothing about
what was *achieved*.** A manifest should carry the achieved residual and the exit path
(tolerance vs cap) as recorded-but-unhashed fields, so `solfiles.py show` can say whether
a solve met its own contract. Hashing them would be wrong — they are outcomes, not inputs
— but omitting them lets a capped solve masquerade as a converged one. This applies to
every producer, not just GS.

### Concurrency, corrected

My "2-3 at a time" advice was right for the wrong reason. The ASU session measured the
binding constraint as **memory, not cores**: `smooth()` builds a `(200,161,20,161)`
temporary ≈ **830 MB per call, 4x per sweep**, with RSS 0.9-2.4 GB per solve. Five
concurrent solves hit swap and throughput *fell*. README corrected.

Two things went right: `experiments/registry/` handled genuine concurrent writers (a
`bgn_gam` manifest landed mid-GS-solve, no collision, no lost update — "safe by
construction" is now observed), and GS **fails clean on interrupt**: four SIGTERM'd solves
left empty outdirs and no manifests, because `solstamp.record()` only runs after a
successful save. That is better than `build_vy_tables.py` managed (§17).

## §21 — oracle cost: N is the expensive dimension, and it is superlinear (2026-09-05)

Measured on `bgn_gam`/g0235, full default basis (`--rff 36,360,3600 --nmat 2`), BLAS
threads pinned to 4, load 5-8. Pinning is what made these reproducible — the same runs
were unmeasurable an hour earlier (§19).

| N | T | wall | per-month, net of the 159 s fixed cost |
|---|---|---|---|
| 100 | 30 | 180 s | 0.70 s |
| 100 | 60 | 201 s | 0.70 s |
| 200 | 60 | 251 s | 1.53 s |
| 500 | 120 | **2321 s** | **18.02 s** |

**A fitted model failed a validation test by 4x, which is why the test existed.** The
first three points fit `t = 159 + 0.0070*N*T` to within 3% and projected the flagship
`--N 500 --T 500` at **32 minutes**. The N=500/T=120 check predicted 579 s and measured
**2321 s**. Had that projection been acted on instead of tested, Phase 1 would have been
sized ~5x short.

**What the fit missed.** Per-month cost is 0.70 / 1.53 / 18.02 s at N = 100 / 200 / 500 —
2.19x for the first doubling of N, then 11.8x for a 2.5x rise. Superlinear and
accelerating, roughly `N^2`, consistent with O(N^2)-O(N^3) work on the N x N conditional
covariance. The `N*T` form assumed linearity in N; the three fitted points were all at
N <= 200, where the quadratic term is still small enough to hide inside a 159 s intercept.

**Projection, extrapolating only in the dimension that is verified linear.** T-linearity
holds exactly at N=100 (0.70 s/month at both T=30 and T=60). Applying the *measured*
N=500 rate:

```
N=500, T=500:  159 + 500 * 18.02  =  9159 s  ~=  2.5 hours per seed
```

This is a 4x extrapolation in T at fixed N, not a 20x extrapolation in N*T. T-linearity is
unverified at N=500; treat 2.5 h as a floor and request walltime accordingly.

**Consequences for Phase 1**

1. **g0235 goes to the cluster.** Hours per seed, several seeds. Same conclusion as
   before, now for a measured reason rather than a guessed one.
2. **N is the knob, not T.** Cost is ~`T * N^2`, so halving N buys ~4x while halving T
   buys 2x. Any budget pressure should come off N first — but note `sr_max_mean` is
   N-dependent (§17g), so N must be held fixed across anything being compared.
3. **Pin BLAS threads in the SLURM script.** Non-negotiable: it is the difference between
   reproducible-to-3% and unmeasurable.
4. The 159 s fixed cost is basis setup (P=3600, twice at `--nmat 2`) and is independent of
   panel size — so it is amortized to nothing at flagship scale, but it dominates every
   small calibration run and will mislead anyone fitting a model without an intercept.

## §22 — manifests now record what a solve achieved (2026-09-05)

`sol_reg` landed: **3 h 23 m 36 s, 5600 sweeps, exited by cycle-averaging at the cap.**
`qerr` *rose* between sweeps 5200 and 5400, so the iterate oscillates rather than stalls —
the period-2 orbit the cycle-average machinery exists to absorb, and the same behaviour
GS21.m showed (see [[gs21-solfile-simulation-mismatch]]). The averaging is the right
remedy; the defect is only that averaging and convergence print the same word.

The gap §20 predicted, now with numbers (arithmetic re-checked: 6.15e-3/180.963 = 3.398e-5):

| | |
|---|---|
| `tol` recorded in the manifest | 1e-06 |
| threshold the loop actually enforces (`tol*20`) | 2e-05 |
| achieved relative residual | **3.398e-05** |
| overshoot | **1.70x** |

Under 2x over, so the artifact is probably fine — **which is what makes it dangerous.**
The identical code path prints the identical `converged` and writes the identical manifest
at 1000x over.

### The fix, and why it is asymmetric

`solstamp.record(achieved=...)` writes an `achieved` block that is **recorded but never
hashed**. Hashing an outcome would be wrong: two runs of the same code on the same
parameters would land in different registry slots. But a manifest that records only what
was *requested* cannot tell a converged solve from a capped one. So: inputs decide
identity, outcomes decide trust, and they are stored separately.

Wired into all three producers — GS (`exit`, `sweeps`, `qerr_rel`, `perr_rel`, `vscale`,
`threshold_enforced`, `tol_requested`), BGN (`exit`, `grid_points`, `tol_requested`), and
KP's G stage (`exit: direct_solve`, per-type relative residuals parsed from the solver's
own output). `solfiles.py show` prints the block and flags a non-tolerance exit;
`solfiles.py check` now reports `[ok, but CAPPED]` rather than a bare `[ok]`.

`a8ef7a2522eda19d`'s block is **backfilled** and says so in its own `source` field —
re-solving to record it natively costs 3 h 23 m. Everything else records natively.

### Second bug, also mine, also in two producers

Under `GS_SOLVE_FORCE=1` / `BGN_JSTAR_FORCE=1` the cache branch is falsified by the env
var rather than by a missing manifest, so control reached an `else` written for the
*unrecorded* case and announced "provenance is unrecorded" about a solve that was recorded
and matching — **a false provenance claim from the provenance system.** Both producers had
it; `build_vy_tables.py` did not, because its `_report_prior` only prints on a genuine
mismatch. Fixed with an explicit `elif prior:` branch in both.

### Registry: the point of the exercise

`a8ef7a2522eda19d` is the first **`committable: NO`** entry — 78.6 MB that can never go in
git, with the manifest outliving it. All five brief checks passed, plus a corruption case
the ASU session added. Still unexercised: `solfiles.py diff` for `gs_bx`, since only one
manifest exists and `record()` writes only on success.

### Costs, corrected again

`gs_bx` solves are **~3 h 24 m each**, so a full five-type bx7 economy is **~17 h** on this
laptop and `gs_bx` stays unrunnable until then (`gs_sim_bx.py` loads all five). The
README's "~a minute each" — mine — was wrong by **~180x**. That is a live input to PLAN
§0.0's demote-bx7 recommendation, which already ranks bx7 22nd of 24 on realized gap.

## §23 — hash scope vs source freezing: two different axes (2026-09-06)

Seth's point: if spec B changes a parameter that cannot affect a solve, B should reuse A's
solfiles instead of re-solving. **The waste is real and larger than expected — but copying
A's files to B is the wrong remedy.**

### The waste, measured

The G solve is 0.10 s, so the true dependency set can be *measured* rather than guessed:
perturb one parameter, re-solve, compare bytes (`_scratch/hash_scope_probe.py`).

| | count |
|---|---|
| hashed into every `kp_vy` solve_id | 61 |
| tested scalars that **change** the G solve | 20 |
| tested scalars with **no effect** | **26** |
| non-scalar, untested | 15 |

Among the irrelevant: `bx_seed`, `gam_seed`, `burnin`, `dt`, `sigma_u`, `theta_u`,
`mu_lambda`, `sigma_lambda`, `rho`. **Changing a simulation seed currently forces a
re-solve of a deterministic PDE.**

### Why copying is the wrong fix

Copying A's artifacts to B is *asserted* provenance — a human deciding two parametrizations
share a solve. That is precisely the failure this registry exists to end, and the same move
that produced every incident this week: the parameter-blind `[ -f sol_reg/solution.npz ]`
guard, the hand-written 5-key KP stamp that missed seven parameters, `Jstar_g0235.csv`
identified only by its filename.

The correct fix makes the copy **unnecessary**: if B genuinely cannot affect the solve, B
should compute **the same solve_id**, and the existing cache-hit path fires automatically —
verified by digest, not asserted. Copying is only needed because the hash is too broad. So
narrow the hash; do not work around it.

### Two classes in the "no effect" list, needing different arguments

1. **Genuinely unread** by the producer (`bx_seed`, `burnin`, `dt`, `theta_u`, …). Safe to
   drop from this stage's hash.
2. **Derived and recomputed after the override site.** `lambda_L` is assigned at
   `parameters_kp14.py:10`, the override lands at :32, and :33 **recomputes it** from the
   E[lambda]=1 normalisation. So `lambda_L=9.0` is silently discarded — which is why the
   probe scored it "no effect" even though it plainly enters `util`. Dropping it is safe
   **only because** its determinants (`mu_H`, `mu_L`, `lambda_H`) are all in the "affects"
   list. Same for `prob_H`, `exit_H/L`, `C`, `rho_ty`, `A_*`.

   That is a **transitive** condition and must be verified, not assumed: `rho_ty` depends
   on `gm_grid`, which is non-scalar and untested. Under-inclusion silently reuses a stale
   solve; over-inclusion only wastes compute. The asymmetry says: drop only what is proven.

### New hazard for the spec layer (Phase 2)

**`KP_PARAM_OVERRIDES` silently ignores derived parameters.** A spec setting
`"lambda_L": 0.5` is accepted, hashed into the spec id, and then discarded by line 33. The
spec would claim a value the economy never used, and nothing would say so. Any spec layer
needs a post-import readback: assert every requested override equals the module's final
value, and fail loudly otherwise.

### Freezing, restated

"Freezing" is about producer **source**, not parameters, and matters only because a source
edit moves every id under it. Reduced where possible: `rebuild_jstar_gam.py` no longer
hashes itself (numerics live in `vasicek.py`; its only knob `JSTAR_TOL` is in `env_params`),
matching `build_vy_tables.py`. `gs_solve_reg.py` cannot be fixed this way — numerics and
plumbing share one file, so any edit moves all five ids, at 3 h 24 m each to rebuild. That
is the real argument for settling GS's source before the cluster run, not parameters.

## §24 — the batched kp_vy source change (2026-09-06)

Three edits to hashed sources, paid for with **one** rebuild.

### 1. `epsrel` 1e-10 -> 1e-6 in the integral stage

§21 called the 985 roundoff warnings per job "the same unachievable-tolerance shape as
the G solver". **That was wrong**, and the refutation was a comment three lines above the
call — mine, from the quadrature transplant: `epsabs` deliberately carries the
`(eps-1)*f(eps)` integrals, whose true value crosses zero where no relative tolerance is
attainable. The warnings were the expected consequence of a documented choice, not a
landmine. Pattern-matching, not analysis.

Measured anyway, and loosening is still right — for a different reason:

| epsrel | wall | warnings | worst rel. diff vs 1e-10 |
|---|---|---|---|
| 1e-10 | 110 s | 985 | — |
| 1e-8 | 89 s | 863 | 8.3e-08 |
| **1e-6** | **12 s** | **0** | **3.8e-07** |

4e-7 is four orders of magnitude below the panel's own sampling noise (~2e-3 at N=500,
T=500), and at 1e-6 QUADPACK stops straining, so its error estimates mean something again.
`quad` stops at `max(epsabs, epsrel*|I|)`, so the zero-crossing integrands are still
bounded by `epsabs = 1e-8` exactly as before. **Integral stage: 5263 s -> 774 s (6.8x)**,
which is the dominant recurring cost of every new KP parametrization.

### 2. The dead `lambda_L` pre-assignment, removed

Per §23 / REVIEW-override-shadowing.md.

### 3. An override readback that raises

`globals().update()` was silent about overrides it discards. Now every requested key must
verify against the module's final value, with `type_share` compared proportionally
(it is renormalised to sum 1).

**A case the first version missed, found by testing it:** a *misspelled* key passed
silently, because `globals().update()` **creates** whatever key it is given — so
`gamma_vv` existed with the requested value by the time the check ran and compared equal.
Fixed by snapshotting the name set *before* the override. Four cases now behave: real vyx
overrides load; `lambda_L` raises; `gamma_vv` raises as a probable typo; `sigma_eps`
passes.

### What the rebuild showed about the registry itself

New ids `b0260fa9ca745db8` (G) and `8e4b5e820ad371da` (integrals). The **G artifacts are
byte-identical to the superseded `f2be637b9fec1f80`** — the only G-side change was
deleting a dead line, so the solve_id moved while the numbers did not. That is correct
behaviour (source is part of identity) and a clean illustration of why id churn is not
the same thing as a numerical change.

**Superseded manifests are now kept and labelled, not pruned.** `solfiles.py check`
reported the old integral manifest as `[STALE/MISSING] ... content changed since it was
recorded`, which reads as damage when it is actually the registry working. It now
distinguishes them:

```
[superseded] d8d686da014f429a  kp_vy  vyx  -> its artifacts now belong to 8e4b5e820ad371da;
             manifest kept as the record of what produced earlier results
```

This directly serves the stated requirement that old experiments stay identifiable after
their artifacts are gone. It also means **deleting the orphaned BGN manifest earlier was
the wrong call** — the right move was to label it. Nothing had used it, so no record was
lost, but the policy is now retention.

## §25 — Phase 1 plan for the primary session (2026-09-07)

Phase 0 is closed. All three economies are verified against their papers, `kp_vy` and
`bgn_gam` are manifested, `gs_bx` has five computed-but-unmanifested `solve_id`s awaiting
the cluster. This is the primary session's queue; the ASU session has
`TASK-cluster-deploy-gs-solves.md`.

### The gate everything sits behind

**22 commits are unpushed, and `origin` is `kerryback/bop-run-upload`.** Sol's checkout is
~30 commits behind and has none of this week's work. Seth decides whether that gap closes
by rsync (keeps parameter corrections out of a co-author's repo for now) or by pushing to
Kerry's `main`. **Do not push without his say-so.** Nothing below can run on the cluster
until this is settled.

### 1. Seeded oracle + estimator SLURM arrays for g0235 and vyx

Both are ready now — `b80c6e516c132e13` (BGN J*), `b0260fa9ca745db8` / `8e4b5e820ad371da`
(KP G / integrals). What the wrapper must carry, all of it measured rather than guessed:

- **Seeds are array indices.** `run_oracle.py --seed` already exists and is respected.
- **Per-seed checkpoint keyed on the consumed `solve_id`s**, same contract as the G stage,
  so a walltime kill re-runs only the missing seeds.
- **Pin BLAS threads.** `OMP/MKL/OPENBLAS/VECLIB_NUM_THREADS` to the allocated core count.
  This is not hygiene: it is the difference between unmeasurable and reproducible-to-3%
  (§19), and an unpinned job on a shared node spends its allocation spinning.
- **Walltime from the measured cost model** (§21): cost is `~T * N^2`, per-month 0.70 /
  1.53 / 18.02 s at N = 100 / 200 / 500. The flagship `--N 500 --T 500` is **~2.5 h per
  seed**, extrapolating only in T at the measured N=500 rate. Treat it as a floor:
  T-linearity is verified at N=100, not at N=500.
- **N is the expensive knob, not T.** Halving N buys ~4x; halving T buys 2x. But
  `sr_max_mean` is N-dependent (§17g), so N must be held fixed across anything compared.

### 2. Wire results to the solves that produced them

This is Job 4 and it is the point of the whole registry. Each run's output must record the
`solve_id`s it consumed, so a summary can be traced to its economy after the ~80 GB of
panel data is purged from scratch. `solstamp` already has `spec_ids` on the manifest side;
the missing half is the run side.

### 3. The flagship selection evidence is stale — flag before it is used again

PLAN §0.0's table (bx7 22nd of 24 on realized gap; `corr(room, gap) = +0.675` overall,
−0.234 excluding two KP priced-vol rows) was computed **before every parameter fix this
week**. KP ran at `E[lambda] = 1.7172` instead of 1.0, and GS ran with the wrong `delta`,
`rho_x` and `kappa_e = 0`. `G_vyx2.csv` alone moved 6.2%.

So the ranking that demoted bx7, and the room-vs-gap correlation that motivated the nested
feature bases, **cannot currently be cited**. Recomputing the grid needs the cluster and
should follow the first clean flagship runs, not precede them. Do not let §0.0's numbers
back into a recommendation until they have been regenerated.

### Not blocking, but queued

- `utils_gs21/{sdf_compute,loadings_compute}_gs21.py` import `kappa_e` and never apply it.
  Harmless — they never use it — but it is the same shape as the `panel_functions_gs21.py`
  bug that was live for a day (§ commit `ee7344a`).
- `analyze.py` in the analysis repo still needs to consume `dkkm_avg_results`; tangled with
  the repo-merge decision and Kerry's sign-off.

---

## §26. The solve_id was not reproducible (2026-09-07)

Found while starting Phase 1's seeded arrays, before writing any array code.
Two independent defects, either one fatal to the registry. Both are fixed;
`tests/test_solve_id_reproducible.py` (9 tests) pins them, and its negative
control was run — removing the fix makes exactly the two intended tests fail.

**Defect 1 — hash order.** My own commit `b85629f` (override readback) left two
`set` objects in `parameters_kp14.py`'s namespace. `solstamp._canon` had no set
branch, so a set fell through to `{'__repr__': repr(value)}`, and Python
randomises str hashing per interpreter. Identical parameters, identical code:

| PYTHONHASHSEED | kp_vy G solve_id |
|---|---|
| 0 | `fa54addb83e5bff2` |
| 1 | `a8b4cdc0fb2cab7f` |
| 2 | `da8081b77e6d5045` |
| 12345 | `6c7e520d50348425` |

`build_vy_tables.py` would have re-run the multi-hour integ stage on every
invocation, and the manifests it wrote were unreachable the moment the process
exited. Fix: sets canonicalise to a sorted form.

**Defect 2 — float drift.** Derived parameters are computed by LAPACK and libm
at import. Two interpreters *on this laptop*, same parameters, same source:

| | `A_0` | G solve_id |
|---|---|---|
| numpy 2.4.2 | 3.851851173257416 | `b5d95b8eb1afc6d1` |
| numpy 2.4.6 | 3.8518511732574208 | `3500d25aeeb64809` |

10 of 66 parameters differed, all derived, up to **45 ULPs** (`pm_tau` through a
fractional `**`; `A0_ty..A3_ty` through `np.linalg.solve`). So the laptop and
Sol could never agree on an id — which breaks the cache *and* Job 4's whole
premise, since a cluster result could not be matched to a laptop-committed
manifest.

Fix: the hash sees parameters quantised to `HASH_SIG_DIGITS = 8`; the manifest
still records exact values. **A first attempt cleared 10 mantissa bits (~2e-13
quantum) and did not work** — against 45 ULPs over ~200 array elements a
straddle is near-certain, not the ~0.2% I estimated. The quantum has to sit
several orders above the drift, not just above it. Both interpreters now agree:
G `f7be27e39d2b530f`, integ `84e195172f091cd2`.

The cost is stated in the tests: differences above ~1e-8 relative move the id,
below it deliberately do not. `test_solstamp.py`'s "0.1 vs 0.1+1e-16 are
different economies" assertion was rewritten rather than deleted, because that
precision is what was traded away.

**Registry.** Every id moved once — unavoidable. Six manifests are now
`retired` via a new `variants/solfiles.py retire`, which annotates rather than
deletes (old experiments stay identifiable) and is reported ahead of
`superseded`, being the stronger statement: a superseded id is still reachable,
a retired one is not.

| live | model | note |
|---|---|---|
| `be222462dd017b2c` | bgn_gam | rebuilt; **byte-identical** to the retired table, which independently confirms the rebuild is deterministic |
| `f7be27e39d2b530f` | kp_vy G | adopted; `achieved` carried forward from `b0260fa9ca745db8` |
| `84e195172f091cd2` | kp_vy integ | adopted |

`a8ef7a2522eda19d` (gs_bx `sol_reg`) is retired unrebuilt. I predicted it
carried the pre-correction calibration; **that was wrong** — its `delta` and
`rho_x` are the corrected values. It is the κe = 0 economy, which the five-task
array supersedes anyway.

**What this means for the deployment gate.** Nothing that ran before today can
be cited by solve_id, and Sol must be at this commit or later before it records
anything, or it writes ids under the old canonicaliser that this laptop cannot
reach.

---

## §27. Phase 1: seeded arrays, and the run→solve link (2026-09-07)

### The seed array had to fix a defect before it could mean anything

`run_oracle.py --seed` steers the global numpy stream — `norm.rvs`, `expon.rvs`,
`np.random.*` — which is every **firm-level** shock. It did not steer the
**aggregate** state path. Each economy draws that from its own generator seeded
by a module constant:

| economy | generator | drives |
|---|---|---|
| bgn_gam | `gam_seed = 555` | `sreg`, the price-of-risk regime chain |
| kp_vy | `gam_seed = 555` | `yreg`, the OU path of the priced factor |
| gs_bx | `reg_seed = 909` | `sreg`, and firm types via `rng_bx` |

So a ten-seed array would have run ten replications sharing **one** 700-month
aggregate path. Cross-seed spread would then measure firm-level noise only, with
the aggregate contribution — the channel the whole project is about — invisible,
and any t-stat built from it overstated.

`run_oracle.py` now offsets those generators by the seed. Two properties are
kept deliberately:

- **Seed 0 reproduces the historical single-run behaviour exactly** (555 + 0).
- **Common random numbers survive across specs**: seed *s* draws aggregate path
  base + *s* in every economy, so a paired comparison between two specs at one
  seed still differs only in the parameters. That is what makes the 24-spec grid
  comparison valid while still giving honest replication spread.

`gs_sim_bx.py` seeded its two aggregate streams from `reg_seed` and the literal
`909 + 1`. Tying the second to `reg_seed + 1` would have made seed *s*'s
firm-type stream bit-identical to seed *s+1*'s regime stream — overlapping
streams across replications, which is precisely what a seed array used for
standard errors must not have. Both are now spawned from one `SeedSequence`.

### The run→solve link (Job 4's missing half)

New `variants/common/runstamp.py`. After the panel is built it hashes the
artifacts the run **actually read** and asks the registry which manifest
describes those exact bytes — by content, not by trust, so a run that picked up
a stale or foreign table records what it really read (or `null`, which is itself
the finding). The ids land in the oracle summary and are carried into the
estimator's new `..._run.json`. Verified end to end:

```
[runstamp] G        <- solve_id f7be27e39d2b530f  (3/3 files, 2,386,269 B)
[runstamp] integ    <- solve_id 84e195172f091cd2  (63/63 files, 5,721,660 B)
```

Result filenames now carry the seed unconditionally (`..._vyx_s003.parquet`),
including seed 0 — suffixing only non-zero seeds would leave `..._vyx.parquet`
ambiguous between "seed 0" and "written before seeds existed", and everything in
the second category came from the pre-correction parameters (§25). The estimator
refuses to adopt an unseeded panel and says why.

### `variants/run_seeds_slurm.sh`

One script for both economies, selected by `SEED_SPEC`, submitted twice. Not one
script per economy: a copied 140-line SLURM script is how `gs_solve_reg.py`
acquired the duplicate keyword argument that made it a hard SyntaxError for two
days. The duplicated override strings are pinned to the specs by
`test_specs_match_shell.py`, and a second test asserts the array never invokes a
solver.

The array **consumes** a solve and never produces one — kp's builder would
serialise ten tasks behind one lock, bgn's takes no lock at all and ten tasks
would race on one CSV. Each task verifies a live solve exists *before* spending
compute, and checkpoints per seed on the consumed `solve_id`, so re-running after
a partial failure redoes only what is not current and re-solving the economy
invalidates every seed at once.

Walltime 12 h is ~2.5× a 2.5 h oracle + 1 h estimator expectation, and **the
2.5 h is a floor, not an estimate**: cost is `~T·N²`, and T-linearity is verified
only at N = 100, so the T = 500 extrapolation is unmeasured. Re-measure from the
first completed task before widening the array.

### Two live bugs found on the way, neither mine to expect

**`variants/common/dkkm_functions.py` was a hard SyntaxError.** A note *inside*
a triple-quoted block-comment quoted the fence characters literally, closing the
fence three lines after it opened. Broken since `6cd949e` (2026-09-04).
`run_estimators.py` imports it, so **the entire estimator side could not start
for three days** and nothing reported it. Every estimator CSV in
`variants/results/` predates that commit.

**`run_estimators.py:103` raised on the `--levels` path** — `.to_numpy()` handed
back a read-only view and the next line mutated it. `--levels` is used by every
shipped run script.

New `tests/test_sources_parse.py` compiles all 123 tracked Python files and
`bash -n`s all 8 shell scripts. That is the second SyntaxError to sit undetected
in this tree (`gs_solve_reg.py` was the first), and in both cases the rest of the
suite was blind because nothing imported the file. Negative control run.

---

## §28. Measuring the aggregate-seed claim, and closing spec → run → solve (2026-09-07)

### I overstated the aggregate contribution

In §27 I argued that holding the aggregate state path fixed across seeds would
leave the aggregate contribution "invisible" and every t-stat "overstated", and
told Seth it was "likely the dominant" source of variation. **Measured, it is
neither invisible nor dominant.**

kp_vy `vyx`, N = 100, T = 100 (86 evaluated months), 6 seeds, run twice —
aggregate path varying with the seed, then held common:

| statistic | mean | sd (varying) | sd (common) | ratio |
|---|---|---|---|---|
| SR_max | 0.6780 | 0.0378 | 0.0325 | 1.16× |
| lin_rank ceiling | 0.4991 | 0.0232 | 0.0221 | 1.05× |
| nonlinear ceiling | 0.6037 | 0.0322 | 0.0318 | 1.01× |
| **room** | **0.1046** | **0.0173** | **0.0139** | **1.24×** |

On the headline statistic the naive 6-seed t on `room` is 14.8 with the
aggregate path varying and 17.1 with it fixed — an overstatement of **1.15×**,
not the order-of-magnitude my phrasing implied.

Reproduce with:

```
for s in 0 1 2 3 4 5; do for c in "" --common_aggregate; do
  python run_oracle.py --model kp_vy --N 100 --T 100 --seed $s \
      --tag vd --levels --rff 36,360 --nmat 2 $c
done; done
```

Two things the run does confirm:

- **Seed 0 is identical in both columns** (room 0.0926 either way), which is the
  designed property: the offset is `base + 0`, so seed 0 reproduces every
  pre-2026-09-07 result exactly.
- With 6 seeds the sd of an sd is itself ~30%, so 1.24× is not precisely
  estimated; the honest range is roughly 1.0–1.6×.

**Reasoning, not measurement:** the aggregate share should be *larger* at
flagship size. Firm-level noise in a cross-sectional average falls like 1/(N·T);
the aggregate path's contribution falls like 1/T only. Going from N = 100 to
N = 500 cuts the firm term ~5× and leaves the aggregate term alone, so 1.24×
reads as a floor for the flagship rather than a typical value. Worth re-measuring
on the first real array rather than trusting the extrapolation — the last cost
extrapolation from small N was 4× wrong.

The fix stands regardless: it costs nothing, seed 0 is unchanged, common random
numbers across specs are preserved, and `--common_aggregate` now makes the
decomposition an explicit option rather than an accident.

### spec → run → solve is now closed

`solstamp` said which parameters produced an artifact; §27's `runstamp` said
which artifact a run read. The missing link was whether that is the economy the
**spec** claims. `run_oracle.py --spec <spec_id>` now verifies consumed
solve_ids against the spec's new `expected_solves` and **aborts before recording
anything** on a mismatch. All three paths exercised:

```
[spec] G        f7be27e39d2b530f  matches var-kp_vy-vyx-v2
[spec] jstar    MISMATCH: spec var-bgn_gam-g0235-v2 expects be222462dd017b2c, run consumed <nothing>
[spec] var-kp_vy-vyx-v1 declares no expected_solves -- nothing to verify against
```

A spec that pins nothing says so rather than passing silently.

### v2 specs, and what is still citable

Same discipline as `solfiles retire`: v1 is annotated, never edited (`notes` and
`lineage` are outside `spec_hash`, so v1 identities are untouched).

| spec | economy | v1's published figures |
|---|---|---|
| `var-kp_vy-vyx-v2` | **changed** — λ regime labels, E[λ] 1.7172 → 1.0; quadrature epsrel | void |
| `var-gs_bx-bx7-v2` | **changed** — delta, rho_x, sigma_x, κe 0 → 0.025 | void |
| `var-bgn_gam-g0235-v2` | **unchanged** — 11/11 vs the paper, J* rebuilt byte-identical | **levels stand, rank does not** |

That last row is the distinction worth keeping: g0235's room +0.0233 and gap
+0.0297 still describe this economy, but "rank 4 of 24" is a claim about the
other 23 specs, and two of them changed economies. Quote the levels, not the
ranking, until the grid is regenerated.

`var-gs_bx-bx7-v2` carries `solves_pending: true` and **precommits** to the five
ids ASU's array must produce. A solve_id is a hash of inputs, so it is knowable
before solving; pinning it is what turns the cluster run into a test of
cross-machine agreement. The test distinguishes a declared precommitment from an
id that is simply wrong or retired.

### gs_bx cannot be run as a panel yet

`var-gs_bx-bx7-v1` flagged this in September and it is still open, and worse than
the note said. `gs_solve_reg.py:152` applies `gs_ashift` to the payoff tables.
`gs_sim_bx.py` omits it in **two** places — line 151's `prod`, the simulated cash
flow itself, and line 229's `opcf`, which becomes `op_cash_flow` and then the
exported `roe` characteristic. There is **no per-firm ashift array in the
simulator at all**, and no `GS_BX_ASHIFTS` env var beside `GS_BX_BETAS`.

For four of the five types the solve assumes productivity exp(`gs_ashift`) times
what the simulation generates:

| type | gs_bx | gs_ashift | solve/sim ratio |
|---|---|---|---|
| 0 | 1.0 | 0.000 | 1.000 |
| 1 | 2.5 | 0.225 | 1.252 |
| 2 | 4.0 | 0.450 | 1.568 |
| 3 | 5.5 | 0.675 | 1.964 |
| 4 | 7.0 | 0.900 | 2.460 |

The five solves ASU is running are internally correct — the discrepancy is on
the simulation side. But any panel built on them prices four of five types
against value functions solved for a different productivity level. **Not fixed
here**: deciding what `gs_ashift` means in the simulation changes the economy and
the env contract, and that is Seth's call, not a silent patch.

---

## §29. The spool-directory bug was in my file too (2026-09-07)

The ASU session's first `gs_bx` submission (job 62740438) lost all five tasks in
2–7 s at ~35 MB MaxRSS — a shell failure, not a python one:

```
mkdir: cannot create directory '../results': Permission denied
```

`sbatch` stages a **copy** of the batch script into the compute node's spool
directory, so `${BASH_SOURCE[0]}` is `/var/spool/slurmd/job.../slurm_script`.
Deriving the repo root from it lands in the spool dir.

**`variants/run_seeds_slurm.sh:128` carried the identical idiom** and would have
lost a ten-task array the same way, on both economies. Fixed with the same form:

```bash
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
```

plus a guard that exits 2 with a usable message. All three invocation modes
tested: staged copy with `SLURM_SUBMIT_DIR` (the sbatch case), direct execution
by relative and absolute path (the fallback), and submission from the wrong
directory (fails loudly).

The generalisable rule is narrower than "always set the cwd", and an over-broad
first version of the guard got it wrong by flagging `run_bop_job.sh`. **SLURM
already starts a job in the directory `sbatch` was invoked from**, so a script
that never `cd`s is correct by doing nothing — which is exactly what
`run_bop_job.sh` does. The unsafe thing is the *construct*: deriving a path from
`BASH_SOURCE`. `tests/test_sources_parse.py` now pins that — any `#SBATCH`
script mentioning `BASH_SOURCE` must prefer `SLURM_SUBMIT_DIR` and fail loudly.
Negative control run.

Neither edit moves a solve_id: `gs_solve_reg.py` hashes only itself, and
`run_seeds_slurm.sh` is not a solve source.

---

## §30. gs_ashift set to 0, and what that says about mechanism variants (2026-09-07)

**Decision (Seth, 2026-09-07): `gs_ashift = 0` for every exposure type.**

The five bx7 types were meant to differ in *exposure*. Measured against GS21's
corrected x process (ρ = 0.98305, unconditional sd 0.03843), the level
compensation dominated the exposure it accompanied:

| type | gs_bx | gs_ashift | 1-sd exposure swing | level shift | ratio |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.000 | ±3.9% | 0% | — |
| 1 | 2.5 | 0.225 | ±10.1% | +25.2% | 2.3× |
| 2 | 4.0 | 0.450 | ±16.6% | +56.8% | 2.9× |
| 3 | 5.5 | 0.675 | ±23.5% | +96.4% | 3.2× |
| 4 | 7.0 | 0.900 | ±30.9% | +146.0% | 3.3× |

The ladder was a rule, not an accident — `gs_ashift = 0.15 × (gs_bx − 1)` holds
exactly for all five — but at 3.3× the exposure swing the cross-section was
mostly a size sort wearing a beta label.

**Setting it to 0 also resolves the solve/simulate inconsistency open since
2026-09-04, with no code change.** `gs_solve_reg.py` computes
`exp(b·x + z + gs_ashift)`; `gs_sim_bx.py` computes `exp(b·x + z)`. At
`gs_ashift = 0` those are identical. The trap is now guarded rather than left
latent: solutions record their own `gs_ashift`, so `gs_sim_bx.py` raises
`NotImplementedError` if handed a nonzero one.

`sol_reg`'s solve_id is **unchanged** — type 0 always ran at 0. Only four types
re-solve.

| outdir | v2 (ashift ladder) | v3 (ashift 0) |
|---|---|---|
| sol_reg | `63fa7ebbc2db49ea` | `63fa7ebbc2db49ea` — unchanged |
| sol_b25c | `13d265e388106790` | `649bb384300faadf` |
| sol_b40c | `2b2327a64c9c3583` | `a3bb50b66287307c` |
| sol_b55c | `6918c06791c0332c` | `645262a8e72d944c` |
| sol_b70c | `a0973863681df5b8` | `0818d7153d5708cc` |

`var-gs_bx-bx7-v3` precommits to these. v2 is annotated, never edited: it is the
record of a precommitment superseded before it ever ran, and the array running on
Sol at the moment of the decision was solving it.

**bx7's demotion is now an open question again.** Its rank-22-of-24 came from a
different calibration *and* a different structure. Nothing about it survives.

### The general rule this settled

Three tiers, in the order to try them:

1. **Inconsistency → fix, never switch.** Two files disagreeing about one model
   is not a design choice. A switch here preserves a bug as an option. The λ
   regime labels were this; so was `gs_ashift` across solve and sim.
2. **Constant → parameter.** Expose the number; current behaviour is one value.
   `gs_ashift` dissolved into this — no fork, no new machinery, and the existing
   override → `solve_id` → spec chain handled it end to end.
3. **Genuine alternative implementation → enumerated `method` switch.** Only for
   things with no numeric interpolation: linear vs cubic interpolation, tauchen
   vs exact quadrature, zero-book firms in or out of the SDF solve.

**Most "small mechanism tweaks" are tier 2.** The test to apply first: is current
behaviour one value of a number nobody exposed?

Forks are for a divergent research direction not intended to merge — not for a
term in an equation. Concrete cost: four cross-cutting fixes landed on
2026-09-07 alone (canonicaliser, `dkkm` SyntaxError, read-only array, SLURM
spool). Under N forks each is an N-way backport, and a fork that misses one
produces wrong numbers under a legitimate-looking solve_id. Grids cannot span
branches, and one registry cannot compare ids across them.

**Live gap in tier 3:** `experiments/specs/*.json` already carries a `method`
block with exactly these switches — `kp_interpolation`, `kp_cir_quadrature`,
`zero_book_in_sdf_solve`, `gs21_discretization`, `kp_regime_labels`. All five are
referenced in **zero** Python files. They are documentation; nothing verifies
them. That is the same failure as `kappa_e` imported and never applied. Making
`method` executable — enumerated values, read from env, landing in the hashed
namespace, verified at run time the way `expected_solves` now is — is the work
this tier needs, and the discipline it needs more: add a switch only when both
branches will actually be run and compared, and retire it once the comparison
settles.

---

## §31. Recording the environment the id deliberately ignores (2026-09-07)

§26 made the solve_id independent of the library stack on purpose, so Sol and
this laptop can agree on an id. That trade left a gap nobody had closed: **nothing
recorded which stack produced the bytes.** Two runs that legitimately share an id
still differ in the last digits of every table — that is precisely what
quantisation permits — and an investigator holding a 78 MB `solution.npz` had no
way to tell which machine made it.

`solstamp.record()` now captures python/platform/machine/hostname, the numpy,
scipy and pandas versions, and the SLURM job and thread variables when present.
It sits next to `achieved`, on the **unhashed** side of the same line: the id says
what was asked for, these say what happened and where. Hashing it would put every
machine's run of one spec in a different registry slot, which is exactly what §26
was spent removing. Verified: `kp_vy` G stays `f7be27e39d2b530f`.

Measured stacks, which is the point:

| | laptop | Sol |
|---|---|---|
| python | 3.11.14 | 3.14.3 |
| numpy | 2.4.2 | 2.4.3 |
| pandas | 3.0.0 | 3.0.1 |
| scipy | 1.17.0 | 1.17.1 |

Both are pandas 3.x, which is why the copy-on-write read-only-array crash in
`run_estimators.py` reproduces on either.

**The five GS solves running now will NOT carry this block** — they imported
`solstamp` before the change. Their environment has to be captured separately or
it is lost when the jobs exit.

### The Sol array, independently verified

All five running tasks report the `var-gs_bx-bx7-v3` precommitted ids, checked
from the logs rather than taken on report:

```
gs_bx7.0.log  63fa7ebbc2db49ea      gs_bx7.3.log  645262a8e72d944c
gs_bx7.1.log  649bb384300faadf      gs_bx7.4.log  0818d7153d5708cc
gs_bx7.2.log  a3bb50b66287307c
```

That is the cross-machine agreement test passing end to end: python 3.14.3 +
numpy 2.4.3 on Sol reproducing ids computed under 3.11.14 + numpy 2.4.2 here,
a wider gap than §26's own test covered.

Both jobs carry `TimeLimit=1-00:00:00`, so the 9.4 h projection has ~2.5 h of
margin. Task 0 runs as `62740640_0` (submitted 09:45) and tasks 1–4 as
`62740930_1..4` (10:08) — task 0 was correctly kept on the unchanged `sol_reg`
id rather than resolved.

---

## §32. First post-correction oracle numbers (2026-09-07)

Six runs, N=200, T=200, `--rff 36,360,3600 --levels`, seeds 0–2, each verified
against its spec (`--spec`, so the consumed solve_ids were checked before
anything was recorded). These are the **first numbers for either economy since
the parameter corrections** — everything in PLAN §0.0 predates them.

### kp_vy / vyx — solves `f7be27e39d2b530f`, `84e195172f091cd2`

| seed | SR_max | lin_rank | nonlinear | **room** | best basis |
|---|---|---|---|---|---|
| 0 | 0.8499 | 0.5840 | 0.7751 | 0.1911 | rff3600 |
| 1 | 0.8764 | 0.5575 | 0.7817 | 0.2241 | rff3600 |
| 2 | 0.9573 | 0.5877 | 0.8569 | 0.2692 | rff3600 |
| **mean** | 0.8945 | 0.5764 | 0.8046 | **0.2281** | |
| sd | 0.0559 | | | 0.0392 | |

### bgn_gam / g0235 — solve `be222462dd017b2c`

| seed | SR_max | lin_rank | nonlinear | **room** | best basis |
|---|---|---|---|---|---|
| 0 | 0.1371 | 0.1137 | 0.1220 | 0.0083 | rffL3600 |
| 1 | 0.1100 | 0.0918 | 0.0968 | 0.0050 | rffL3600 |
| 2 | 0.1762 | 0.1426 | 0.1566 | 0.0140 | rffL3600 |
| **mean** | 0.1411 | 0.1161 | 0.1252 | **0.0091** | |
| sd | 0.0333 | | | 0.0045 | |

### Room is strongly size-dependent, and that is a clean measurement

`g0235`'s **economy is unchanged** — 11/11 against the paper, J* table rebuilt
byte-identical — so its published figure and this one differ *only* in N and T,
with the same flags and the same seed 0:

| | N | T | room (seed 0) |
|---|---|---|---|
| published, PLAN §0.0 | 500 | 500 | 0.0233 |
| here | 200 | 200 | 0.0083 |

**2.8× on an economy that did not move.** Room is not size-invariant, so **no
ranking computed at reduced size can be cited for the flagship** — including any
temptation to re-rank the 24-spec grid cheaply at N=200. The grid has to be
regenerated at flagship size or not at all.

That also means the vyx/g0235 comparison below is only valid *at matched size*.

### What this says about the two open questions

**vyx survives the λ fix.** Its room is 0.2281 at N=200 against g0235's 0.0091 —
**25×**, far outside the cross-seed spread of either. The λ regime-label fix
(E[λ] 1.7172 → 1.0) did not remove vyx's advantage, so its rank-1 standing is
not in obvious danger. Its published 0.3496 is not comparable (different economy
*and* different size) and remains uncitable.

**g0235's room is small relative to its own seed noise.** Mean 0.0091, cross-seed
sd 0.0045 over three seeds. That is a different quantity from the published
"t 29.1", which is a time-series statistic over months, so the two must not be
compared — but it does mean a three-seed array cannot separate g0235 from
neighbouring specs in a 24-way ranking, and its "rank 4 of 24" was probably never
well identified in the seed dimension at all.

### Caveats, stated rather than buried

- N=200 with a 3600-feature basis is a regime where the basis can span a large
  share of a 200-asset weight space within a month; the binding constraint is the
  constant-θ requirement across months. Whether the N=200 → N=500 factor measured
  on g0235 transfers to vyx is **not** established.
- Three seeds. The sd of an sd at n=3 is enormous; treat the spreads as
  indicative only.
- These are oracle ceilings, not estimator performance. The realised gap
  (`room × capture`) still needs the estimator side, which needs panels.

## §33. The cost model, measured — and a spec that could stamp a run it did not earn (2026-09-07)

Two things came out of the gap while the gs_bx array ran on Sol.

### The scaling ladder, and a claim of mine that did not survive it

`run_seeds_slurm.sh` had never been run. Its `--mem=24G` rested on one `ps` snapshot
and its walltime on a `~T*N^2` cost model. I measured six points on `kp_vy` with the
flags the script actually uses (`--rff 36,360,3600 --levels`, default `--nmat`):

| N | T | wall | max RSS |
|---|---|---|---|
| 100 | 200 | 281 s | 4772 MiB |
| 200 | 200 | 413 s | 5268 MiB |
| 300 | 200 | 552 s | 7559 MiB |
| 500 | 200 | 824 s | 11209 MiB |
| 100 | 500 | 678 s | 6323 MiB |
| 200 | 500 | 1036 s | 10969 MiB |

    wall   ~ 30 + 0.58*T + 0.0070*N*T  s
    maxRSS ~ 2421 + 0.0853*N*T         MiB

Wall is **linear in N**, not quadratic: the affine fit at T=200 predicted N=500 at
822 s against 824 s (0.24%), and the bilinear form predicted N=200/T=500 at 1038 s
against 1036 s (0.2%).

**Memory tracks the PRODUCT N·T, and that is structural rather than fitted.**
N=500/T=200 and N=200/T=500 share N·T = 100000 and measured **11209 vs 10969 MiB —
2.1% apart with the large dimension swapped.** That is why the flagship extrapolation
is trustworthy: it is 2.5× a measured point, not a curve run off its end.

**A wrong claim of mine, named.** On the first three points I wrote that the ladder
refuted the `~T*N^2` comment, and later that the memory N-slope grew 3.33× for a 2.5× T
so `--mem=24G` would be *exceeded* by 1.3%. Both were artifacts of fitting T=200 memory
over N=100..300 only, where the N=100→200 increment (+496 MiB) is anomalously small and
drags the slope from ~19.8 down to 13.94 MiB/N. With N=500 included the T-ratio is
2.35×, slightly *sub*-linear, and the flagship projects to 23754 MiB — **under** the
24576 MiB cap, not over.

`--mem` was still raised to 64G, for reasons the corrected model supports better than
the wrong one did: the +822 MiB margin is only 1.3× the fit's own 644 MiB max residual,
the projection excludes `--save_panel` (which the script passes) and the estimator stage
entirely, and the oracle has no mid-run checkpoint, so an OOM writes nothing.

**The `~T*N^2` claim is not refuted for `bgn_gam`.** It came from bgn_gam at `--nmat 2`
(§ WORKING.md:1284), per-month 0.70/1.53/18.02 s at N=100/200/500 — a 12× jump over the
last 2.5× of N. `run_seeds_slurm.sh` runs both economies via `SEED_SPEC` and only
`kp_vy` was re-measured. g0235 remains unmeasured; assume the quadratic there.

### A spec could stamp a run it did not earn

`run_oracle.py --spec X` writes X into the summary. Two paths let it write a spec_id the
run had no right to claim, and they compounded:

1. `verify_against_spec` returns `None` for a spec declaring no `expected_solves`, and
   the caller guarded with `if _ok is False`. **`None is False` is False**, so an
   *unverifiable* spec passed exactly as though it had been verified.
2. The superseded check sat *after* the no-`expected_solves` early return, making it
   unreachable for every v1 — precisely the specs most likely to be superseded, since
   v1s predate the field. Same ordering bug as retired-before-superseded in
   `solfiles.cmd_check` (§26).

Concretely: `--spec var-kp_vy-vyx-v1` builds the **v2** economy today, because the
lambda regime-label fix was made by editing `parameters_kp14.py` in place rather than
behind a `method` switch. It would have stamped a v2 result `var-kp_vy-vyx-v1`.

Fixed in `9497089`: supersession is checked first and **refuses**, naming the successor;
the guard is now `is not True`. Five tests in `tests/test_spec_refusal.py`.

**The `method` block is inert.** Seven keys are declared across the specs
(`kp_interpolation`, `kp_cir_quadrature`, `kp_regime_labels`, `zero_book_in_sdf_solve`,
`gs21_calibration`, `gs21_discretization`, `gs_ashift_ladder`) and **zero** are read by
any python file. `method` is also not hashed into `solve_id`, so two specs differing
only in `method` produce identical ids and identical results while claiming to be
different experiments — and `expected_solves` would pass for both. This commit stops the
silent mislabelling; making the switches executable is a separate decision (§30) and is
not started.

### Cross-machine reproducibility, confirmed on a live solve

Two of the five gs_bx solves landed while this was written, and **both match the ids
precommitted in `var-gs_bx-bx7-v3.json`**: `sol_b55c 645262a8e72d944c` and
`sol_b70c 0818d7153d5708cc`. Sol (py 3.14.3 / numpy 2.4.3) reproduced ids computed here
(py 3.11.14 / numpy 2.4.2) *before the solve ran*. §26's 8-significant-digit
quantization holds for the gs_bx solver, not only the kp_vy tables. The remaining three
are watched by `_scratch/watch_gs_bx.sh`, which is read-only on Sol.

## §34. The estimator stage had never run, and it is the larger half (2026-09-07)

`run_estimators.py` had **never produced output**. No `*_run.json` existed anywhere in
`variants/results/`, across the whole history of the repo. It had a hard `SyntaxError`
for three days (§ fixed in the dkkm docstring), then a read-only-array crash on the
`--levels` path, and both were fixed without ever running the thing end to end.

This matters because `run_seeds_slurm.sh` runs **oracle then estimators in the same
task, under one walltime**. A broken second stage means every array task burns its full
oracle cost and then dies having recorded nothing.

**First end-to-end run, 2026-09-07** — kp_vy, N=60, T=80, window=36, 29 eval months:

| stage | wall |
|---|---|
| `run_oracle.py --save_panel` | 216 s |
| `run_estimators.py` | **741 s** |

Both exit 0. The oracle wrote a 421 KB parquet panel; the estimator wrote
`kp_vy_estimators_smoke_s000_w36_summary.csv` and the first `_run.json` in the repo's
history. That `_run.json` carries `G=f7be27e39d2b530f` and `integ=84e195172f091cd2`
forward from the oracle summary, so the spec → run → solve chain now closes on the
estimator side too, not just the oracle's.

**The estimator is 3.4x the oracle**, at a size where the oracle is already 216 s. The
walltime comment budgeted it as "roughly its own hour at window 360" — a guess, and the
wrong shape: estimator cost scales with `eval_months x N x P`, and the flagship has 138
eval months (T=500, window=360) against this run's 29.

I am deliberately **not** extrapolating that across both dimensions at once. Two claims
of mine failed that way earlier today (§33): the "refutation" of `~T*N^2`, and
`--mem=24G` "exceeds by 1.3%". Both came from fits pushed outside their fitted range.
`_scratch/est_ladder.sh` measures the two slopes separately — `eval_months` at fixed N
by re-running on the existing panel, then N at fixed window on one new panel.

Until that lands, **the total per-task cost of the seed array is unknown**, and the
oracle-only projection in the `-t` block understates it by an unknown factor. That is a
second, independent reason not to tighten `-t 2-00:00`.

## §35. Phoenix's env, and a third stack agreeing on solve_id (2026-09-08)

**The gamma(x) negative control passed.** `gs_solve_gam.py` at `gs_gamma_slope = 0`
reproduces `gs_solve_reg.py` element for element: all 31 solved arrays — every value
function, price function, policy, grid and transition matrix — are byte-identical at
xnum=31 / tol=1e-4 with gmreg=[0.6,3.0]. Only `params_json` and `solve_id` differ, and
both must: the namespace legitimately gained `gs_gamma_slope`/`lo`/`hi`, and the source
file is a different file. (My first comparison script reported "CONTROL FAILS" because
it lumped those two metadata keys in with the numerics.) The reconstruction is verified.

**Phoenix's `bop` was 8/10 against environment.yml**, missing exactly `pyarrow` and
`openpyxl` — the same two pandas I/O backends whose absence broke `--save_panel` on Sol
the day before. Everything else satisfied and python 3.12.13 was already in range, so
this was an update, not a replace: rebuilding would have moved eight working packages
for no reason. Nothing was running (empty queue, no processes in the env), so it was
safe to mutate in place rather than clone.

Now 10/10, and **functionally** verified rather than merely importable — `to_parquet`
and `read_parquet` round-trip, `to_excel`/`read_excel` round-trip, and a writable
`to_numpy(copy=True)` under pandas 3.0.3. Import success is not the test that matters
for a backend pandas resolves at runtime.

**A third stack now agrees on solve_id.** All three independently compute
`fa9032525bff9489` for `var-gs_bx-g28-v1`, from parameters alone, without solving:

| | python | numpy | pandas |
|---|---|---|---|
| laptop | 3.11.14 | 2.4.2 | — |
| Sol | 3.14.3 | 2.4.3 | — |
| Phoenix | 3.12.13 | 2.4.6 | 3.0.3 |

With Sol's five bx7 ids reproduced exactly as precommitted (§34), the portability claim
in §26 is no longer an argument from construction — it is measured on three stacks
spanning three python minor versions. The full suite passes on Phoenix, 150/150,
matching the laptop test for test.

**`pytest` was missing from `environment.yml`** and from Phoenix's env, so the suite
could only be run through the per-file `__main__` blocks. The repo ships `tests/` and
both clusters need to run them; added to `environment.yml` and `requirements.txt`, and
installed on Phoenix.

## §36. The first post-correction flagship number: vyx survives, and is stronger (2026-09-08)

`run_seeds_slurm.sh` had never been run. The first task — kp_vy/vyx, seed 0,
N=500/T=500, window 360, on Sol — completed, and it is the **first measurement of the
realized gap since the lambda regime-label fix**. Every gap figure in the repo before
this one predates that correction.

| | published (pre-fix) | **now** |
|---|---|---|
| room (const-θ, nonlinear − linear) | +0.3496 | **+0.3746** |
| realized gap (RFF − best linear) | +0.1009 | **+0.1138** |
| t vs FMR | 21.6 | **32.6** |

**The lambda fix did not cost vyx its standing; it improved it.** Room is *higher* at
flagship than the published figure, and the estimated gap is larger at a higher t. The
rank-1 position that §0.0 rested on — and that I flagged as uncitable — is now supported
by a number from the corrected economy rather than the old one.

Best per method (window 360, 125 eval months, best κ):

| method | P | sharpe | t vs FM |
|---|---|---|---|
| `rff_ens` | 360 | **0.7856** | 32.6 |
| `rff` | 3600 | 0.7841 | 30.8 |
| `linrank` | 6 | 0.6718 | 5.0 |
| `fm` | 6 | 0.6628 | — |
| `ff` | 6 | 0.5932 | −23.0 |

**One nuance, so the ceilings are not misread.** `linrank` at 0.6718 *exceeds* the
linear-in-ranks ceiling of 0.6508. That is not an error: the ceilings are best
**constant-θ**, while the estimators re-fit on a rolling 360-month window and adapt. So
"capture = gap / room" is not a clean ratio here, and I am not quoting one. Caveat: one
seed; seeds 1–9 are running.

### Two cost predictions of mine, both wrong, in opposite directions

**Wall: projected ~6 h, actual 2 h 58 m** (oracle 4450 s, estimators 6197 s). I scaled
the laptop by 4x, taking that factor from the gs_bx solve's CPU utilisation (114% on Sol
against 470% here). It did not transfer: this task had 8 CPUs and used them. The real
laptop→Sol factor was **2.05x**, not 4x.

**Memory: projected 23754 MiB, actual 30654 MiB — 29% high.** The projection was
explicitly a floor (measured without `--save_panel`, estimator stage unmeasured) and the
floor held. But note what this means: **had `--mem` stayed at the original 24G, this task
would have OOMed at hour 2 and written nothing.** Raising it was right; the model was
optimistic.

### That mistake nearly repeated on g0235, and the ladder caught it

The local bgn_gam ladder landed the same afternoon. Against kp_vy at T=200, its RSS
ratio runs 0.96x / 1.30x / 1.38x / **1.96x** at N = 100/200/300/500 — superlinear, the
`~T*N^2` behaviour this repo had assumed but never measured for g0235.

    bgn_gam maxRSS ~ -1241 + 0.2218*(N*T) MiB   ->  52.9 GiB oracle-only at flagship

Applying kp_vy's measured +29% oracle-to-task factor gives **~68 GiB, over the 64G** the
g0235 task had just been submitted with. It was cancelled while still PENDING and
resubmitted at 128G. The array default is now 128G.

Sequence worth noting: the ladder was run because I had written "assume the quadratic for
g0235" into the script rather than leaving the gap silent. That note is what made the
measurement happen before the OOM rather than after it.

## §37. Ten seeds: vyx is UNCHANGED by the lambda fix, and I over-read seed 0 (2026-09-08)

All ten vyx seeds completed at N=500/T=500/w=360. Wall 2.4–3.2 h each, MaxRSS 29.9–30.6
GiB, every one COMPLETED.

| | published (pre-fix, 1 seed) | **10 seeds now** |
|---|---|---|
| room | +0.3496 | **+0.3491**  (sd 0.0365, se 0.0115) |
| gap (DKKM − best linear) | +0.1009 | **+0.1046**  (sd 0.0155, se 0.0049) |
| t(DKKM vs FMR), time-series | 21.6 | mean 29.4, sd 12.0, range 17.6–58.2 |

**A correction I owe.** On seed 0 alone I reported room +0.3746 and gap +0.1138 and wrote
that vyx "survives the lambda fix and is STRONGER". That was over-reading one draw. Seed
0 sits +0.7 sd on room and +0.6 sd on gap; across ten seeds the means land on the
published values — room differs by 0.0005 against a standard error of 0.0115, gap by
0.0037 against 0.0049. **The right conclusion is UNCHANGED, not stronger.**

That is still a substantive result. The lambda regime-label fix moved E[lambda] from
1.7172 to 1.0 — a genuinely different economy (§WORKING.md 32) — and at flagship size it
did not detectably move either the room or the realized gap. The published vyx figures
were not reproducible in principle; they turn out to be reproduced in fact.

### The seed noise is large enough to matter for PLAN §0.0

Cross-seed sd is ~10% of the mean on room, ~15% on gap, and **41% on the t statistic**
(17.6 to 58.2 across ten draws of the same economy). Every figure in
`grid_summary.csv` is single-seed.

So the grid's ordering is identified only where economies differ by much more than one
seed-sd. The top gap is +0.1009 and the runners-up are +0.0406 and +0.0383 — those
separate cleanly from vyx. But they differ from *each other* by 0.0023, against a gap
sd of 0.0155: **ranks 2 and 3 are not distinguishable, and neither are most of the
lower rankings.** §0.0's headline (vyx first, by a wide margin) survives; its ordering
below the top does not, independently of the staleness audit in e1a1b66.

### g28 solved, and the precommitment caught its first real thing

The gamma(x) solve finished on Phoenix in 5.9 h — converged on the TOLERANCE test
(qerr_rel 8.58e-6, perr_rel 4.25e-6 against an enforced 2e-5) at sweep 2974, not on the
cycle cap. Parameters exactly as specified.

But its `solve_id` came out `8b584c38614695ac`, not the precommitted `fa9032525bff9489`.
Cause: the `experiments/solfiles` -> `experiments/registry` rename (30bef7a) ran sed over
every file naming the old path, which rewrote **one string inside a print()** in
`gs_solve_gam.py`. `solve_id` digests the whole source file, so the id moved while the
numerics did not.

**This is the first time precommitment CAUGHT something** rather than confirming it —
the exact distinction the Phase 3 trigger in DECISION-provenance-layers.md turns on.
What it caught is real and was worth catching, even though the economy is fine.

**And it exposes a design hazard that will recur.** `solve_id` cannot tell "the code
changed" from "the comments changed". That is the mirror of why content-addressing beats
a repo-wide git sha (207 commits, 6 touched the solver): the file-level digest
distinguishes solver from repo, but not code from prose. Fixing it — digesting the AST —
would move every existing solve_id, 504 MB and ~25 h of cluster time, so it is
deliberately **not** fixed. The operational rule instead: **do not run bulk sed over
solver sources.** Recorded in var-gs_bx-g28-v2's notes where the next person will hit it.

## §38. An AST-based solve_id, measured and rejected; the advisory that replaced it (2026-09-08)

After the g28 id moved on a `print()` string (§37), Seth asked for an AST-based digest
and authorised the recompute. Two measurements said no.

**Zero benefit in this repo's history.** Of the 15 commits that ever changed a solver
source, **0** were comment/docstring/format-only — every one touched real code. Even the
aggressive variant that also strips `print`/logging calls absorbs only 2, and both are
the one sed in `30bef7a`.

**It would break cross-machine portability.** `ast.dump` of the same file:

| | python | digest |
|---|---|---|
| laptop | 3.11.14 | `aa8141c3…` |
| Phoenix | 3.12.13 | `311a7a6f…` |
| Sol | 3.14.3 | `5247e4ef…` |

Three ids for one solver — undoing the 5/5 agreement verified the day before. Fixable
only with a hand-maintained canonical serializer that must be revisited on every Python
release. And it would not even have caught g28: a string in `print()` is in the AST,
because `jstar_gam_file: "Jstar_g0235.csv"` has to be.

### What was built instead

`variants/solve_impact.py`, wired as `hooks/pre-commit` (advisory, never blocks,
silent unless a cached solve is at stake). It maps a changed file to every manifest
listing it in `sources`, reports what those solves cost, and classifies the change —
functional if the AST differs, non-functional if only comments, docstrings, whitespace
or diagnostics moved. The AST comparison is done locally on one interpreter, where
version skew is irrelevant. Verified on three real commits: `30bef7a` non-functional
(six solves, 589 MB, 16,644 sweeps), `370294a` (kappa_e) functional, `413b3c1` (docs)
silent. `tests/test_solve_impact.py`, 5 tests.

### What it found on its first real run

`30bef7a` had also touched **`gs_solve_reg.py`** — the same one-string sed — so all five
bx7 solves were stale against the current source. Nothing had tried to reuse them since,
so it had not tripped; the next bx7 run would have missed the cache and re-solved 25 h.
Confirmed by probing the producer: `sol_reg` recorded `63fa7ebbc2db49ea`, current source
computed `c72f428fa0f6ea6f`.

**Fixed by reverting the one string.** Doing so restores `gs_solve_reg.py`'s digest to
`b6106234…` byte-for-byte — identical to `30bef7a^` — so all five recorded ids are valid
again with no manifest surgery, no re-stamp, no republish. The advisory now reports that
change as *"RESTORES the recorded digest for 5 solve(s)"* rather than as invalidation.

So `gs_solve_reg.py:365` deliberately still says `experiments/solfiles/` in a message.
**It is load-bearing for 25 h of solves and must not be "fixed"** — the hook will say so
to anyone who tries. `gs_solve_gam.py` keeps `registry`: its solve was recorded after the
rename and its id is consistent.

### Open

A manifest's recorded `params`, fed back through `Snapshot` with re-quantisation, do
**not** reproduce its own `solve_id` — 0 of 9 live manifests (2 impossible because a
large array's stored hash is of the raw, not quantised, bytes). The producer probe does
reproduce it, so the ids are sound; but the manifest alone is not a complete account of
how its id was computed. Not chased today; recorded so it is not rediscovered.

## §39. g0235 at flagship reproduces its published row to the digit; two of my projections did not (2026-09-09)

`bgn_gam/g0235` seed 0 at N=500/T=500/w=360 on Sol: oracle 5420 s, estimators 6233 s.

| | published (1 seed) | **flagship now, seed 0** |
|---|---|---|
| room | +0.0233 | +0.0285 |
| gap (DKKM − best linear) | **+0.0297** | **+0.0297** |
| t vs FMR | **29.1** | **29.1** |

**The gap and its t reproduce the published `grid_summary.csv` row at displayed
precision** — on a different machine, under Python 3.14.3 / numpy 2.4.3, from the
current code. This economy is the one of the 24 that is unchanged (§0.0 audit), so it is
the one row that *could* be reproduced, and it is. That validates the whole
oracle → panel → estimator pipeline against the pre-refactor results, which nothing
else in the repo had done.

Room differs (+0.0285 vs +0.0233) for a specific reason, not noise: the oracle now runs
with `--levels`, and the nonlinear ceiling here is set by `rffL3600`, a basis the
published run did not have. The estimator configuration is identical, which is why the
gap matches exactly and the room does not.

### The job was marked FAILED, and the science was fine

Both stages completed and wrote every file. The last line of the log, after the END
marker, was `STALE: built from be222462dd017b2c; registry now has nothing` — the
**post-run verification** looking up `g0235` when the solve is registered as
`Jstar_g0235`. §36's fix (e42a3a7) reached the precondition's lookup and nothing else;
the per-seed checkpoint and the post-run check made the same lookup with the output tag.
So seed 0 ran for 3 h 14 m and was failed by bookkeeping, and a resubmission would have
re-run it rather than skipped it.

A fix that reaches one call site and a test that checks one call site are the same
mistake. All three now key on `SOLVE_TAG`; the test asserts *every* `runstamp` lookup in
the script does, so a fourth cannot be added with the old key.

### Two projections against actuals

**Oracle wall: 5420 s.** Projected ~8900 s from the ladder's 2.07× bgn/kp_vy ratio at
N=500/T=200. The ratio at T=500 is **1.22×**. Wrong by 1.6×, safe direction.

**Memory: 15.8 GiB** (sacct, whole task). Projected 53–68 GiB from the bgn_gam laptop
ladder, which had bgn at **1.96×** kp_vy's RSS at N=500/T=200. On Sol at flagship it is
**0.52×** — vyx used 30.6. The laptop and Sol measurements disagree in *direction*, and I
have not established why: macOS max-RSS vs cgroup accounting, `--save_panel`, or
bgn_gam's memory scaling in T differently from N are all candidates. Both numbers are
recorded in the SLURM script; `--mem` is back to 64G, 2.1× the larger measured task.
The 128G request cost nothing on an empty queue, but the claim I wrote to justify it —
"g0235 IS THE BINDING CASE, ~2× kp_vy" — was wrong and is replaced, not appended to.

The estimator stage, by contrast, landed where the ladder said: 50.7 s per eval-month
against vyx's 49.6. N-independence holds across economies.

## §40. g0235 ×10: gap +0.0227 (se 0.0027), and the realized gap exceeds the const-θ "room" (2026-09-09)

Sol array `62901300`, seeds 1–9 of `bgn_gam/g0235` at N=500/T=500/w=360, all COMPLETED,
all nine passed the post-run `is-current` check on `be222462dd017b2c` — the three-lookup
`SOLVE_TAG` fix from §39 verified in production. With seed 0 (§39) that is ten seeds.

| seed | SR_max | room | DKKM | best lin | gap | t (max rff, `t_vs_fm`) | wall | MaxRSS GiB |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1860 | +0.0285 | 0.1752 | 0.1455 | +0.0297 | 29.2 | 3:14:20 | 15.8 |
| 1 | 0.1688 | +0.0281 | 0.1577 | 0.1373 | +0.0204 | 21.0 | 3:04:26 | 16.2 |
| 2 | 0.1410 | +0.0189 | 0.0871 | 0.0644 | +0.0228 | 44.0 | 3:31:19 | 38.9 |
| 3 | 0.1176 | +0.0163 | 0.0801 | 0.0729 | +0.0072 | 10.6 | 4:37:25 | 32.2 |
| 4 | 0.1269 | +0.0210 | 0.0458 | 0.0281 | +0.0176 | 106.5 | 3:53:05 | 28.7 |
| 5 | 0.1114 | +0.0118 | 0.0872 | 0.0584 | +0.0288 | 26.8 | 3:51:22 | 26.6 |
| 6 | 0.1100 | +0.0136 | 0.1284 | 0.1037 | +0.0248 | 19.0 | 3:16:12 | 15.7 |
| 7 | 0.1287 | +0.0153 | 0.1217 | 0.0882 | +0.0335 | 17.4 | 3:15:07 | 15.9 |
| 8 | 0.1744 | +0.0241 | 0.0962 | 0.0666 | +0.0296 | 46.7 | 3:03:11 | 15.8 |
| 9 | 0.1193 | +0.0104 | 0.1017 | 0.0895 | +0.0122 | 12.6 | 3:52:36 | 15.8 |

**gap: mean +0.0227, sd 0.0084, se 0.0027 (n=10). room: mean +0.0188, sd 0.0065, se 0.0021.**
SR_max mean 0.1384, sd 0.0280. Published single-seed row: room +0.0233, gap +0.0297 — the
published gap sits 0.8 sd above the ten-seed mean, i.e. seed 0 is an ordinary draw, not a
lucky one. Regenerate with `_scratch/g0235_aggregate.py` → `_scratch/G0235-X10.md`.

**Both flagship economies now have error bars, and they don't overlap.**

| economy | room (sd) | gap (sd) | gap/room |
|---|---|---|---|
| kp_vy/vyx | +0.3491 (0.0365) | +0.1046 (0.0155) | 0.30 |
| bgn_gam/g0235 | +0.0188 (0.0065) | +0.0227 (0.0084) | 1.21 |

The gap/room column is the ratio of the two means. The MEAN of the ten per-seed ratios is
a different statistic, 1.31, and the per-seed ratios run 0.44 to 2.44. I first wrote "1.3
(per-seed 0.44-2.44)" in this column, which silently mixed the two. Ratio of means is the
one to quote: the per-seed spread is wide enough that the mean of the ratios is dominated
by the seeds with the smallest room in the denominator.

**The realized gap exceeds the const-θ room in seven of ten g0235 seeds** (and DKKM's
estimated SR exceeds the oracle's mean SR_max in seed 6: 0.1284 vs 0.1100). `room` is the
difference between two *constant-θ* ceilings; the estimators use 360-month rolling windows
and can beat a constant-θ rule when the conditional tangency moves. So `realized gap = room ×
capture` with capture ≤ 1 is not the right decomposition for bgn_gam — capture is 1.21 here
and 0.30 for vyx. For the project goal (economies with large room between DKKM and FMR)
this matters: **ranking candidate economies by const-θ room would have ranked g0235 at
roughly 1/19 of vyx, but its realized gap is 1/4.6 of vyx.** Room is a screen, not a ceiling,
and the screen under-predicts bgn_gam. Not resolved here: whether a time-varying-θ ceiling
(e.g. the oracle's conditional max SR net of a rolling linear rule) restores gap ≤ room.

**Memory is bimodal across seeds on identical hardware.** Six seeds at 15.7–16.2 GiB, four
(2–5) at 26.6–38.9. All ten tasks ran on 128-core zen3 nodes with 515 GB (`scontrol show
node`); seeds 5 and 6 shared `sc109` and used 26.6 vs 15.7. So it is seed-dependent, not
node-dependent — and vyx ×10 was uniform at 30.6. Cause not established. `--mem=64G` held with
1.6× headroom over the worst seed; do not lower it on the strength of the 15.8 GiB seed-0
number in §39. Wall 3:03–4:37 (seed 3 the outlier; its oracle stage is not separately timed).

**The watcher fetched 54 of 63 files.** `_scratch/watch_g0235_x10.sh` made 63 back-to-back
`scp` connections with three tries each; nine failed all three tries, scattered across seeds
4, 5, 8, 9 (all nine files were on Sol). One `rsync --files-from` afterwards brought all
nine in one connection. Next watcher: one rsync per seed, not one scp per file. The
consistency tests (3/3) and the full suite (176) pass over the ten seeds' 70 files.

## §41. Inventory before describing the economies: GS has no valid results, and no tracked thing computes an aggregate (2026-09-09)

Asked what economies we have and what is stored. Four findings, none of which had a home
before this entry.

**Four economies are defined; two are current.** `run_oracle.py --model` accepts exactly
`bgn_gam`, `kp_vy`, `gs_bx`.

| economy | solves | oracle | estimators | seeds | spec |
|---|---|---|---|---|---|
| kp_vy/vyx | both stages verified | current | current | 10 | v2 |
| bgn_gam/g0235 | jstar verified | current | current | 10 | v2 |
| gs_bx/bx7 | 5 local, v3 ids | STALE | STALE | 1, unseeded | none |
| gs_bx/g28 | verified, published | none | none | 0 | v2 |

`gs_bx` has no valid results at either tag. The bx7 files date from `bba735f`, the original
import, and carry `spec_id`, `solves` and `prov` all null: they predate the provenance
system, the kappa_e correction, and the v3 solve ids. `variants/gs_bx/run_gs_bx7.sh` passes
neither `--spec` nor `--seed`, and `run_seeds_slurm.sh` accepts only `vyx|g0235`, so gs_bx
was never wired into the seeded, spec-verified protocol at all.

**Both benchmark tables are legacy and nothing regenerates them.** `variants/make_excel.py`
only reads `grid_summary.csv`; no code writes either file.
- `results/grid_summary.csv`, 24 economies, is what REPORT.md and README are built on. Three
  have code here. `bx9` — which README cites as the reason GS stops at bx7 — is prose only.
  The table is internally mixed: its g0235 row reproduces to the digit (§39), its vyx row is
  the pre-λ economy that `var-kp_vy-vyx-v2` says must not be quoted. Both stale unseeded
  files are still in `results/` beside the corrected ones, distinguished only by the missing
  `_s000` suffix (`kp_vy_oracle_vyx.json` SR_max 1.2607 vs `_s000` 1.1945).
- `results/oracle_summary.csv`, 57 rows over `bgn, bgn_dis, bgn_types, gs_dis, gs_fixed, kp,
  kp_dis`. **None of those seven model families is runnable here.** Fully orphaned.

**Nothing tracked computes an aggregate.** Per-seed storage is good: 211 tracked files,
55 MB, every post-fix file carrying `prov` and `spec_id`. But the ten-seed means and sds for
both flagships exist only in `_scratch` and in §37/§40 prose, and the producer is
`_scratch/g0235_aggregate.py`, which is gitignored and hardcodes one economy. The project's
headline numbers have no reproducible tracked producer. Of the three gaps this is the one to
close first, because a described-economies document needs that table as its input.

**A confound in the §40 room-vs-gap comparison that I have not ruled out.** The oracle
averages its const-θ conditional SR over all 485 months; the estimators average over the 125
months after the 360-month window. Different month samples, so `room` and `gap` are not
strictly commensurable even before the fixed-vs-rolling θ issue. `evaluate_bases` keeps the
per-basis SR time series as `cond_sr_ts` but `run_oracle.py` writes only its mean across
months (`sr_by_z`), so restricting the ceiling to the eval window needs an oracle re-run, not
a re-read of committed artifacts. **Check this before treating conditioning as the
established explanation for gap > room.** The cheap version: have the oracle also write
`cond_sr_ts` mean over the last `T - window - 15` months.

Also stale and worth fixing when the description is written: `variants/README.md` quotes the
superseded vyx room 0.350, states `realized gap = room x capture`, and predates g28.

## §42. The parameters → spec link, closed (2026-09-09)

Asked to confirm that `param -> solfile -> economy result` was ensured before starting a
campaign of new economies. Three of the four links were closed **by content**; the fourth
was not closed at all.

| link | mechanism | was |
|---|---|---|
| params → solve_id | `solstamp` hashes the producer's namespace + source digests | closed |
| solve_id → artifact | manifest records sha256 per file; `solfiles.py check` | closed |
| artifact → run | `runstamp.consumed_solves` re-hashes what the run READ | closed |
| **run params → spec** | recorded in the sidecar, never compared | **OPEN** |

`verify_against_spec` compared solve ids and nothing else, so the simulator's own
parameters were free to describe a different economy from the one the spec named.

**Reproduced, not theorised.** Simulating BGN at `gmult [0.2, 3.0]` while reading the J*
table built at `[0.2, 3.5]`, under `--spec var-bgn_gam-g0235-v2`:

```
[runstamp] jstar    <- solve_id be222462dd017b2c
[spec] jstar    be222462dd017b2c  matches var-bgn_gam-g0235-v2      <- PASSED
=== overrides={"gmult":[0.2,3.0],...}                               <- different economy
```

It wrote a summary carrying `spec_id: var-bgn_gam-g0235-v2` and `spec_check: "verified"`.
Solve-id checking structurally cannot see this: **the artifact is innocent; the parameters
layered on top of it are not.** The wrong overrides did reach the sidecar, so it was
discoverable after the fact — but nothing refused it, and a wrong number that has been
written down is already the expensive kind.

### What was added

`runstamp.verify_env_against_spec(spec_id, ov_env, env)`, called from `run_oracle.py`
**before the panel is built** — it needs only the spec and the environment, so it refuses
in a tenth of a second instead of after a 35-minute panel build. Three rules:

- `spec["param_env"]` names a variable carrying `spec["params"]` as JSON → compared by
  VALUE, so whitespace and key order cannot refuse a correct run.
- `spec["env"]` is a flat map of literal variable → value. A declared value that looks
  like a JSON object is compared as one (unset reads as `{}`); everything else is an
  exact string, where unset IS the mismatch.
- The run-side override variable must be **accounted for**, not merely absent from the
  spec. `GS_SIM_OVERRIDES='{"gamma_x":0.9}'` would overwrite a value the simulator read
  out of `solution.npz` — undeclared, and invisible to both rules above. Refused.

**gs_bx needed the third rule and must not get the first.** Its `params`
(`{"gmreg": [0.6, 3.0]}`) is `GS_PARAM_OVERRIDES`, read by the SOLVER and already covered
by the solve_id; its simulator takes structural parameters out of the npz. Comparing
gs_bx's `params` against `GS_SIM_OVERRIDES` would refuse every correct run. That is why
gs_bx has no `param_env` — an absence with a reason, not an omission.

### Two content guards in `gs_sim_bx.py`

Independent of any spec, so they hold for a run that passes no `--spec` at all:

1. **`GS_BX_BETAS` vs the solutions' own `gs_bx`.** Each `solution.npz` has carried its
   beta since 2026-09-04 and nothing had ever compared them. A reordered
   `GS_BX_SOLDIRS`, or a ladder edited in one place and not the other, prices firms off
   value functions solved for a different exposure — every solve id resolves, the spec
   check passes, the panel is wrong.
2. **Cross-type agreement.** `_s = _sols[0]` takes the grids, `pr_x/pr_z`, the kernel
   `Mx`, `psw` and the 15-element `params` from the FIRST soldir and applies them to all
   types. That is correct only if the types differ in `gs_bx` and nothing else. Five
   separate multi-hour jobs is exactly the setting where one drifts — `sol_reg` has
   already been re-solved once, for `kappa_e` — and the discrepancy is invisible in every
   id, because each solution is individually registered and individually correct.

Both refuse at import, which is *earlier* than the env check, so for gs_bx the more
specific message wins. Verified: a wrong ladder and a reversed soldir list are both
refused.

### The eval-window restriction (the §40 confound)

`room` came from this file averaged over all 485 months of a flagship panel; `gap` came
from `run_estimators.py` averaged over the 125 months that survive a 360-month rolling
window. §41 flagged that they were never commensurable. `run_oracle.py` now takes
`--eval_window` (default 360) and reports every ceiling **twice** — over all months, which
is what every number published before today means, and restricted to the estimator's
evaluation months.

The mask reproduces `run_estimators.py:80` exactly rather than §41's approximate "last
`T - window - 15` months"; `tests/test_eval_window_matches_estimators.py` fails if either
rule is edited without the other. `run_estimators.py` warns when the oracle's
`--eval_window` is not its own `--window` — a warning and not an abort, because re-scoring
an existing panel at a new window is legitimate and has been done three times.

It matters. On a 25×80 probe at `--eval_window 40`, `lin_rank` const-θ went 0.0770
all-month against 0.1346 window-restricted. **Do not difference a pre-2026-09-09 room
against any gap.**

### Also

`spec_id` now travels from the oracle summary into the estimator sidecar and `_run.json`.
`*_summary.csv` — the file actually read to adjudicate an economy — named its solves but
not the experiment they belonged to, so the last hop had to be reconstructed by looking
ids up in the registry by hand.

`env_check` is recorded in the sidecar as a **separate key** from `spec_check`:
DECISION-provenance-layers.md decides the fate of the precommitment layer by counting
`spec_check`, and folding a second question into that tally would corrupt it.

Tests: `tests/test_spec_env_refusal.py` (13), `tests/test_eval_window_matches_estimators.py`
(5). Suite 176 → 194, all passing.

### The one hole left open, deliberately

**Overriding a DERIVED parameter is still silently discarded in `bgn_gam` and `gs_bx`.**
`globals().update(ov)` accepts any key; a name recomputed after that line reverts, and a
misspelled name is created and read by nothing. So a spec can declare a parameter the
economy never used, and the new env check will happily confirm the environment matches
that spec. `parameters_kp14.py` already guards this — it raises `"entries that had NO
EFFECT"` — but `bgn_gam/parameters.py` and the gs modules do not.

Not fixed here because **`variants/bgn_gam/parameters.py` is digest-bearing**: it backs
`be222462dd017b2c`, so adding the guard moves that solve_id, un-pins
`var-bgn_gam-g0235-v2`, and marks all ten committed g0235 seeds STALE. The J* rebuild
itself is minutes; the invalidation is the cost. Batch it with the next functional change
to that file, exactly as REVIEW-override-shadowing recommended for kp_vy's 88-minute
integral rebuild. Residual names at risk in bgn: `prob_calm`, `Preg`, `nchars`, and any
misspelling.

---

## §43. docs/refactor consolidated into this file (2026-09-09)

Eleven documents were folded here and deleted. Everything below is what survived the
question *"would a future session act on this, or does it explain why code looks the way
it does?"* Task briefs, timing logs, completed checklists and superseded audits did not.

**Two files were kept, and one of the reasons is load-bearing.**

- **`FINDINGS-gs21-table1.md` cannot be renamed or deleted.** It is cited from
  `gs_solve_reg.py:48` and `gs_solve_gam.py:26`, and those two files are **digest-bearing**
  — they back six live solves worth roughly 30 h of cluster time. Editing a comment in
  either moves their `solve_id`s. The citation pins the filename.
- **This file** is cited from seven places including `WORKING.md:1284` by LINE number, so
  it is only ever appended to, never inserted into.

Digest-bearing sources, for reference (live solves backed): `gs_solve_reg.py` 5,
`gs_solve_gam.py` 1, `parameters_kp14.py` 2, `integ_kp14.py` 1, `kp14_fd_vy.py` 1,
`bgn_gam/parameters.py` 1, `bgn_gam/vasicek.py` 1. Regenerate with a scan of
`experiments/registry/*.json` `sources`.

### From FINDINGS-gs21 — four solver defects still open, all unfixable cheaply

Confirmed still present 2026-09-09. Each is cosmetic or diagnostic; each costs five solves
(~17–25 h) to fix, because `gs_solve_reg.py` is digest-bearing:

1. `converged` is printed on the **cycle-capped** exit as well as the tolerance exit.
   `run_g28_slurm.sh` already works around this by grepping for `cycle-averaged; stopping`
   rather than trusting the word. The manifest's `achieved.exit` is the reliable answer.
2. `tol * 20` — the enforced threshold is 20× the requested `tol`, undocumented in the
   usage string. `solfiles.py show` prints the overshoot ratio, which is the mitigation.
3. A dead `range(60000)` when the real cap is 5600. A cost model reading the source gets
   the wrong number.
4. Under `GS_SOLVE_FORCE`, the solver prints *"provenance is unrecorded; re-solving"* about
   a solve whose manifest matches exactly — a false statement emitted by the provenance
   system itself.

**Batch all four with the next functional change to that file.** The `achieved`
recommendation from the same document WAS implemented and is why manifests now carry
`exit`/`sweeps`/`qerr_rel`.

Also from that document, and still the right budget: **GS is 5600 sweeps at ~2.2 s**, so
~3.4 h per solve and ~17 h for a five-type bx7 economy on this laptop. The README's
original "about a minute each" was wrong by ~180×.

### From FINDINGS-gs21-kappa-e

`kappa_e` is not a one-line change: it makes the debt policy debt-dependent. All five bx7
solves and g28 run at `kappa_e = 0.025`, recorded in each manifest and stored as its own
key in `solution.npz` (it cannot go in the 15-element `params` array, which is unpacked
positionally and so cannot grow). Pre-2026-09-06 solutions have no such key and are read
as `kappa_e = 0`, which is what they were solved at. `a8ef7a2522eda19d` (the retired
`sol_reg`) is the one such artifact.

### From FINDINGS-config-divergence

The variants tree is deliberately self-contained: nothing imports `config.py` or `utils_*`,
and nothing in the main pipeline imports from `variants/`. The 2026-09-06 reconciliation
corrected `delta`, `rho_x`, `sigma_x` and `kappa_e` against GS21 Table I; `rho_x` was
settled then and should not be re-litigated. `tests/test_config_parity.py` is the durable
part and it cites this section — the test, not the prose, is what enforces the parity.

### From FINDINGS-repo-merge — a decision, still open

The question was whether to merge `bop-analyze-remote` into this repo under one Python
environment. **Recommendation was two environments, not unified.** The bloat concern as
originally phrased — "module bloat slowing cluster compute" — did not survive the numbers;
the real obstacle is a semantic collision on the module name `config`, which both trees
define differently. Nothing has been merged. This is a decision waiting on Seth, not a
refactor task.

### From REVIEW-override-shadowing — resolved for kp_vy, open elsewhere

Three categories of post-override re-assignment were measured across the parameter
modules: **coercion** (re-assignment reads the name it assigns → override survives, fine),
**genuine derivation** (recomputed from other parameters → required for coherence), and
**dead pre-assignment** (assigned identically on both sides of the override site → the one
real defect, `lambda_L`).

Both were fixed in `parameters_kp14.py`: the dead line is gone and a readback now raises on
any override that had no effect. `bgn_gam` and `gs_bx` still lack it — see the open hole at
the end of §42, which is the same finding reached from the other direction.

### From PLAN.md — the staleness audit is the only durable part

`variants/results/grid_summary.csv`, 24 economies, was PLAN §0.0's evidence base. **23 of
24 rows are no longer reproducible**: 21 have no code (`bba735f` kept three economies and
dropped the rest), `vyx` changed with the λ regime-label fix, `bx7` changed twice. `g0235`
is the sole survivor and reproduces to the digit (§39).

What survives of its conclusions: the correlation claim (`corr(room, gap) = +0.675` over 24
rows, `−0.234` over 22) is **arithmetically correct and permanently unverifiable**, because
the two rows carrying the entire positive association are both pre-fix. The direction
survives independently — post-correction vyx room 0.2281 against g0235's 0.0091 at matched
N=200/T=200 — so *"room does not predict gap outside the KP channel"* remains the right
prior but is no longer an established result.

The rest of PLAN.md was a phased proposal that has been overtaken: phases 1–2 shipped (see
DECISION-provenance-layers.md), phase 3 is on a counting trigger, and the spec schema and
directory layout it proposed are now what exists.

### Deleted with nothing retained

`TASK-gs21-solves.md`, `TASK-cluster-deploy-gs-solves.md`, `TASK-repo-merge-assessment.md`,
`TASK-asu-2026-09-07-gs-bx-verification.md` — task briefs whose work is done and whose git
states, queue states and machine states are long stale. Each produced a FINDINGS document,
which is where its content went.

## §44. Provenance layers: the decision record (absorbed 2026-09-09)

Was `DECISION-provenance-layers.md`. Phases 1–2 done 2026-09-08; **Phase 3 deferred on a
usage trigger, not a date.** Folded here so the refactor record is one file; the citations
in `run_oracle.py` and `tests/test_precommitment_is_real.py` now point at this section.

### The goal, in Seth's words

> (a) I see some results are good, and think "let's pursue this further"
> (b) I know what code and parameters produced those results. That's the whole goal here.

### What was actually wrong

Five subsystems, ~1,700 lines, and none recorded (b). `solstamp` content-addresses solves;
`runstamp` links a run to the solves it consumed. Both real. Neither recorded the **code
version**, and `grid_summary.csv` and the `*_summary.csv` files — the ones actually read to
adjudicate — carried **no link to anything at all**. So the complexity was real but the
diagnosis "too much provenance" was wrong. It was *the wrong provenance*: a lot of
machinery for caching and staleness, and nothing for traceability.

### Done (Phases 1–2)

`variants/common/provenance.py`, ~150 lines. Every summary gets `<summary>.prov.json`: git
sha, dirty flag with the diff inline, untracked-code warning, `argv`, `run_from`, the
`*_OVERRIDES` env, consumed solve ids, host, library versions, and a runnable
`how_to_reproduce`. Every summary CSV gains a `prov` column — redundant per row on purpose,
because a row copied into a notebook must not lose its pointer.

Answering (b) is now: **read one file, `git checkout` the sha, run the argv.**

With the sidecar in place, **`solve_id` stops being the provenance story and becomes a
cache key nobody looks at.** That is the simplification, and it required deleting nothing.

### What must NOT be deleted, and why (measured)

Replacing staleness checking with a git-sha comparison read out of the `.npz` cannot work:

> **207 commits in this repo. SIX touched `gs_solve_reg.py`.**

A sha-based check refuses on all 207; content-addressing refuses on 6 — a **34× false-
invalidation rate** against a solve costing ~5 h. You would hit it on a docs commit,
re-solve for nothing, and within a week disable the check — at which point the thing that
caught both real bugs (the bare `[ -f solution.npz ]` reusing another economy's solve; the
KP "converged" that verified nothing) is gone. A repo-wide sha structurally cannot
distinguish *"the solver changed"* from *"the repo changed"*.

`utils/solfile_stamp.py` is **not** legacy — `utils_bgn/regen_solfiles.py` imports it.

### Phase 3, deferred: retire the PRECOMMITMENT layer

Candidates, ~400–500 lines: `spec_hash` + `test_every_spec_hash_is_reproducible`
(reimplements `git diff` for files already in git); `solves_pending` (status inside a
hashed view; caused a spurious "hash drifted"); `expected_solves` + the refusal path
(precommitment, not traceability); the 7 `method` keys (**read by zero Python files**, not
hashed — two specs differing only there are the same run).

**Keep specs themselves.** "Different specs" is the unit of experimentation. It is the
*hashing ceremony* that is the candidate, not the file.

**The trigger is a count, not a date.** Has precommitment ever earned its keep? As of
2026-09-08 it had **confirmed** something (5/5 bx7 ids on Sol) but never **caught**
anything. Only the second justifies the machinery.

```bash
grep -ho '"spec_check": "[a-z ]*"' variants/results/*.prov.json | sort | uniq -c
```

**Tally as of 2026-09-09: 24 `verified`, 0 `refused`, 0 `not requested`.** Note that
`env_check` is deliberately a separate key (§42) so it does not contaminate this count.

**`verified` only counts if the spec predated the solve.** A wall of `verified` is evidence
only if the spec was written **before** the solve; if the id was read off an existing
manifest and pasted in, `verified` merely confirms that the id you copied is the id that is
there — vacuous. Same failure shape as `test_registry_ids_are_still_reachable`, which
passed vacuously because its filter excluded exactly what it was looking for. Three pinned
ids are retrofitted, and they are the two economies actually in use:

| spec | stages | spec pinned | manifest tracked | |
|---|---|---|---|---|
| `var-kp_vy-vyx-v2` | G, integ | 09:27:14 | **08:38:19** | retrofitted |
| `var-bgn_gam-g0235-v2` | jstar | 09:27:14 | **08:38:19** | retrofitted |
| `var-gs_bx-bx7-v3` | all five | 10:03:22 | 18:09:05 | genuine |
| `var-gs_bx-g28-v1` | sol_g28 | 18:16:41 | not yet solved | genuine |

`tests/test_precommitment_is_real.py` classifies every pinned id from git dates and fails
if a spec that claims precommitment in words is not one in fact. Read the tally **only over
genuinely-precommitted specs**, and treat `not requested` and `unverifiable` as the
*interesting* rows: they say the layer is being routed around in practice.

**Revisit at ~20 real experiment runs.** Any `refused` that was a genuine mismatch →
precommitment earned its keep. All `verified` with no genuine catch → delete the four rows
above. Many `not requested` → the layer is already being routed around, which is its own
answer.

**One data point since:** `var-gs_bx-g28-v1` made a real precommitment (`fa9032525bff9489`)
which was invalidated by a documentation-only `sed` that rewrote one `print()` string in
`gs_solve_gam.py`, forcing `g28-v2` with `8b584c38614695ac`. The economy never changed.
That is the layer costing churn without catching an error — evidence toward deletion, not
toward keeping.

### What would reverse Phases 1–2

A sidecar found to be *wrong* rather than merely absent — pointing at a sha that does not
reproduce the result. One such bug already shipped and was caught: `how_to_reproduce`
recorded `getcwd()`, which follows `run_oracle.py`'s chdir and produced an instruction that
could not run. `tests/test_provenance.py` should grow a case for any recurrence.

### Considered and rejected: an AST-based `solve_id` (2026-09-08)

Proposed after a rename's `sed` moved six solve_ids. Measured first: **0 of 15** historical
solver commits were comment/docstring/format-only, so an AST digest would have absorbed
none of them; `ast.dump` **is not stable across Python versions** (laptop 3.11 / Phoenix
3.12 / Sol 3.14 give three digests of one file), which would have broken the cross-machine
agreement verified 5/5 the day before; and the recompute would cost ~31 h and republish
504 MB for a benefit that has never occurred.

Instead: `variants/solve_impact.py`, wired as an advisory pre-commit hook
(`hooks/pre-commit`, enable with `git config core.hooksPath hooks`). Byte-exact id
unchanged; the tool says before a commit which solves a change invalidates and whether the
change was functional. The six stale ids were repaired by **reverting the one string** —
restoring the recorded digest byte-for-byte — not by re-stamping or re-solving.
**Do not run bulk `sed` over solver sources.**

### Open

A manifest's recorded `params` do not reproduce its own `solve_id` when fed back through
`Snapshot` (0 of 9 live manifests). The producer probe does reproduce it, so the id is
sound; but the manifest alone is not a complete record of how it was computed. Not
blocking; noted so it is not rediscovered.

## §45. NEXT.md steps 2 and 3: gs_bx wired into the checked runner (2026-09-09)

Personal session, on Seth's "let's pursue that goal". Decisions taken where NEXT.md asked for
one, each the cheapest reversible option:

**2a — bx7-v3 corrected in place.** `types.exposure.gs_ashift` said the v2 ladder while
`method` and all five pinned solves say 0.0. Fixed, `spec_hash` recomputed (the old value is
recorded in `lineage.changes`), no v4: nothing stamped `var-gs_bx-bx7-v3` exists, so no
published number changes meaning. `test_gs_bx7_ashift_ladder_is_consistent_across_the_records`
pins the two records to each other.

**2b — option B, not the recommended C.** NEXT.md's case for keeping `run_gs_bx7.sh` solve-only
was that it is "the only record of the five GS_PARAM_OVERRIDES strings". It is not:
`run_gs_bx7_slurm.sh` carries the five strings in its `OVERRIDES` array, matching v3's
`solve.stages` element for element, and it is the script that actually produced the solves on
Sol. So the laptop script was a duplicate that re-solved the wrong ladder. Deleted. v3's
`provenance` now names the SLURM script as `source_script` and the seed array as `runner`;
v1/v2's `source_script` is null with a note (all hash-excluded fields).

**2c — the shell-vs-spec tests guard CURRENT specs.** `test_specs_match_shell.py` was rewritten
around a `CURRENT` map; every GS test targets v3 and the SLURM solve script; g28 gets the same
treatment against `run_g28_slurm.sh`; and one test parses every `case` branch of the seed array
and checks it against the spec that branch's `SPEC=` names -- model, override JSON, literal env,
and the kappa grid. A branch that runs a superseded spec fails.

**2d — readback in `run_oracle.py`, not in `parameters.py`.** `variants/common/readback.py`
compares each requested override against the imported module's live value, and detects a
misspelling against the names the module's SOURCE assigns (parsed with `ast`; `hasattr` is
useless once `globals().update()` has created the key). Applied after import for every model;
outcome recorded in the sidecar as `readback`. Pinned on the real bgn module: a derived name
(`prob_calm`, `Preg`) and a misspelling (`gmulr`) are refused, the g0235 overrides verify.
The in-module guard for bgn stays deferred to the next functional change of `parameters.py`.

**3a — `live_solves` takes several tags.** Comma list or list; one tag is unchanged. Verified
the pre-fix failure (five ids vs one) and pinned it in `test_runstamp_multitag.py`.

**3b–3e.** `SEED_SPEC=g28` and `SEED_SPEC=bx7` cases; `KAPPAS` per case and
`--kappas "$KAPPAS"` (the gs specs use eight values, the array hardcoded four);
`--eval_window "$WINDOW"` passed to the oracle; GS `SOLVE_HINT`s say fetch, not re-solve.

**Smoke, g28 at N=25/T=80 with the array's exact environment, both stages.** Oracle: `[env]`
three variables verified, `[spec] sol_g28 8b584c38614695ac matches`, sidecar stamped;
`--eval_window 360` correctly reports "0 of 65 months" at that T. Estimators at the spec's
eight kappas: `_run.json` carries `spec_id var-gs_bx-g28-v2`, the solve id and all eight
kappas; the summary sidecar carries `spec_id`, `window 40`, `oracle_eval_window 360`, and the
"do NOT cover this run's evaluation months" warning fired as designed. Every file of the
chain now names the spec. The economy itself, at toy size: room +0.0014 (lin_rank 0.0735 vs
rff3600 0.0749) against an estimated rff_ens 0.0774 vs linrank 0.0289 -- the published
"gap with zero room" signature of the gamma(x) rows, visible even at N=25. Not a result;
a sign the reconstruction is the right economy.

**Cost note for 3f.** The estimator stage took 1589 s for 25 evaluation months on the
laptop: 63.6 s per month at eight kappas, 2.53x the 25.1 s/month measured at four. If the
Sol factor of 2.05x holds, a flagship gs seed's estimator stage is ~125 x 130 s = 4.5 h,
before an unmeasured oracle. `-t 2-00:00` covers it; memory is the unknown.

**Also corrected while in the file:** the seed array's memory header said g0235 was 15.8 GiB
from seed 0; ten seeds say 15.7–38.9 and bimodal (§40). gs_bx is marked UNMEASURED.

**Not done.** 3f (one g28 seed at flagship on Sol, then size the arrays) and the arrays
themselves. And a finding the same shape as 2b, left for a decision: `bgn_gam/run_g0235.sh` and
`kp_vy/run_vyx.sh` also run unspec'd and unseeded, and the seed array is the only checked path
for those economies too.

## §46. First spec-verified GS results: g28 and bx7 seed 0 (2026-09-10)

NEXT.md 3f. One probe seed per economy on Sol at N=500/T=500/w=360, `--mem=128G` so sacct
would report the truth. Both COMPLETED; every gate passed in production for the first time on
a GS economy: precondition, `[env]` check against the spec, `[readback]`, the five-solution
cross-type guard (bx7), and the post-run `is-current` on the union of five ids -- §45's 3a fix
working as designed. Consistency tests 3/3 on the fetched files.

| | g28 (62934480_0) | bx7 (62934481_0) |
|---|---|---|
| MaxRSS | 4.7 GiB | 5.2 GiB |
| oracle / estimators / total | 4616 / 9574 / 14190 s | 3414 / 10085 / 13499 s |
| estimators per eval month (125) | 76.6 s | 80.7 s |

**gs_bx is the lightest economy by 3-7x**, five 100 MB solutions and all: vyx used 30.6 GiB,
g0235 15.7-38.9. The arrays for seeds 1-9 went in with `--mem=24G`, 4.6x the measured peak.

**A projection of mine was wrong.** In §45 I sized the gs estimator stage at ~130 s per
evaluation month on Sol (63.6 laptop x the 2.05 Sol factor) and ~4.5 h; it ran at 77-81 s and
2.7 h. The 2.05x factor was measured on the kp_vy ORACLE stage and does not transfer to the
estimator stage, where Sol is only ~1.2-1.3x the laptop.

### The science, one seed each

| economy | SR_max | room all-month | room eval-window | DKKM | best lin | gap | t |
|---|---|---|---|---|---|---|---|
| g28 | 0.2960 | +0.0003 | +0.0003 | 0.2961 | 0.2842 | **+0.0119** | 36.4 |
| bx7 (v3) | 0.2745 | +0.0063 | +0.0094 | 0.3140 | 0.3037 | **+0.0104** | 11.2 |
| published GS gamma(x) | | -0.0004 | | | | +0.038 | 17.9 |
| published bx7 (v1 economy) | 0.2989 | +0.0182 | | 0.2369 | 0.2356 | +0.0013 | 10.4 |

**g28 reproduces the zero-room gap on a runnable, spec-verified economy.** Room is +0.0003
under both definitions, the gap is +0.0119 at t 36. The decomposition is the one PLAN §0.0
conjectured from the deleted rows: against the eval-window ceilings, the best linear method
falls 0.0201 short of its own ceiling (0.3043) while DKKM falls 0.0084 short of its (0.3045).
The gap is linear-estimator inefficiency, not nonlinear structure. The published +0.038 was a
different implementation of the same idea (the code is gone; this is a reconstruction from
REPORT.md:553), so the size is not expected to match; the signature does.

**bx7's demotion does not survive the corrections.** The v1 economy's gap was +0.0013 (rank 22
of 24). The v3 economy -- Table I calibration, kappa_e = 0.025, gs_ashift = 0 -- has gap
+0.0104 at t 11.2, eight times larger, with room +0.0063 (a third of v1's +0.0182). One seed;
g0235's per-seed gap sd was 0.0084, so this is a direction, not a number. Ten seeds are running.

**The eval-window ceilings are materially higher than the all-month ones**, in both economies:
lin_rank 0.2664 -> 0.3185 for bx7, 0.2943 -> 0.3043 for g28. The last 125 months of these
panels are richer than the full 485. So the §41 confound is real and not small: bx7's
commensurable room is +0.0094, not +0.0063, and gap/room is 1.1 against the right months vs
1.65 against the wrong ones. It does not eliminate gap > room (1.1 is still above 1, and DKKM
0.3140 exceeds the all-month SR_max 0.2745, as g0235 seed 6 did). **The ten vyx and g0235
oracle JSONs predate `--eval_window` and carry no eval ceilings; §40's gap/room of 1.21 for
g0235 is against all-month room.** Re-running those twenty oracles (seeded, so the panels are
identical; ~1-1.5 h each, no estimators needed) would settle whether g0235's 1.21 is also mostly
month-sample. Not started.

Arrays submitted 2026-09-10 02:2x: g28 seeds 1-9, bx7 seeds 1-9, from the repo root on Sol
at 4e9a378 with `--mem=24G`. Results, aggregates and the watcher: `_scratch/watch_gs_x10.sh`
-> `_scratch/GS-X10.md`.

## §47. The tracked aggregate producer, and a correction to §41 (2026-09-10)

`variants/aggregate_seeds.py` closes the gap §41 called the one to close first. It reads every
SEEDED oracle JSON with its estimator summary, writes `results/seed_table.csv` (one row per
run: spec_id, consumed solve ids, both prov tags, room all-month and eval-window, DKKM, best
linear, gap, t) and `results/economy_table.csv` (one row per model/tag/N/T/window with
mean, sd, se, n across seeds, and gap/room as the RATIO OF MEANS per §40). Unseeded legacy
files are never read. `tests/test_aggregate_seeds.py` pins it to §37's vyx and §40's g0235
ten-seed numbers, so the tracked table cannot drift from what was written down; the CSVs
are the complete set whatever flag was passed, and `--flagship` narrows only the printout.

**Correction to §41.** I wrote there that "nothing regenerates" either legacy table.
`variants/collect_results.py` does regenerate `results/oracle_summary.csv`: it is the
oracle-only, one-row-per-run producer from the original repo, blind to seeds, estimators,
specs and the eval window, and running it today would overwrite the 57-row legacy table with
the current 43 oracle JSONs, seeded and unseeded alike. The claim stands for
`grid_summary.csv`, which only `make_excel.py` reads. README marks `collect_results.py` legacy.

The flagship table as of this entry (g28 and bx7 one seed each; their arrays are running):

| economy | n | room all | room eval | gap | t | gap/room all | gap/room eval |
|---|---|---|---|---|---|---|---|
| kp_vy/vyx | 10 | +0.3491 (0.0365) | - | +0.1046 (0.0155) | 30.2 (11.8) | 0.30 | - |
| bgn_gam/g0235 | 10 | +0.0188 (0.0065) | - | +0.0227 (0.0084) | 33.4 (28.4) | 1.20 | - |
| gs_bx/bx7 | 1 | +0.0063 | +0.0094 | +0.0104 | 11.2 | 1.63 | 1.10 |
| gs_bx/g28 | 1 | +0.0003 | +0.0003 | +0.0119 | 36.4 | 38.9 | 45.9 |

Regenerate: `python variants/aggregate_seeds.py --flagship`.

## §48. Both GS economies at ten seeds, and an inequality that could not hold (2026-09-10)

Arrays `62942411` (g28) and `62942412` (bx7), seeds 1-9, all eighteen COMPLETED at
`--mem=24G`. Every post-run check reported CURRENT, including nine over the union of bx7's
FIVE solve ids -- §45's 3a fix in production. Wall 3:30-4:57, MaxRSS 4.4-4.9 GiB, so 24G was
5x the peak. Consistency 3/3 over all 140 files.

**All four flagship economies now have ten seeds.**

| economy | SR_max all | SR_max eval | room all | room eval | DKKM | best lin | gap | t | gap/room all |
|---|---|---|---|---|---|---|---|---|---|
| kp_vy/vyx | 1.1531 | 1.1778 | +0.3491 (0.0365) | - | 0.7475 | 0.6428 | **+0.1046** (0.0155) | 30.2 | 0.30 |
| bgn_gam/g0235 | 0.1384 | 0.1461 | +0.0188 (0.0065) | - | 0.1081 | 0.0855 | **+0.0227** (0.0084) | 33.4 | 1.20 |
| gs_bx/g28 | 0.2621 | 0.3087 | +0.0001 (0.0003) | +0.0004 (0.0003) | 0.2975 | 0.2692 | **+0.0283** (0.0136) | 24.6 | 393 |
| gs_bx/bx7 | 0.2993 | 0.3121 | +0.0086 (0.0035) | +0.0066 (0.0041) | 0.2964 | 0.2887 | **+0.0078** (0.0049) | 16.0 | 0.90 |

**g28 has the second-largest gap in the repo and essentially no room.** +0.0283 (se 0.0043)
against room +0.0001, larger than g0235's +0.0227 on an economy where the const-theta
nonlinear ceiling exceeds the linear one by 0.0001. The whole advantage is the linear
estimator failing to reach its own ceiling, not nonlinear structure. That is the mechanism
PLAN §0.0 read off the deleted gamma(x) rows, now on a runnable, spec-verified economy with
error bars. **Ranking candidate economies by const-theta room would put g28 last of four and
it is second by realized gap.** Room is not a screen for this channel at all.

**Both GS seed-0 figures were unrepresentative**, which is the §37 lesson repeating: g28 seed 0
gave +0.0119 against a ten-seed +0.0283 (2.4x), and bx7 seed 0 gave +0.0104 against +0.0078.
Worse, seed 0 had bx7's room_eval ABOVE its room_all (+0.0094 vs +0.0063) while the ten-seed
means run the other way (+0.0066 vs +0.0086). Nothing in §46's one-seed reading of the
eval-window direction survives; the ten-seed numbers replace it.

**bx7's gap is +0.0078 (se 0.0016) against the v1 economy's published +0.0013** -- six times,
still, on ten seeds. Its rank-22-of-24 demotion does not survive the calibration corrections.

### The inequality that could not hold

The ten-seed table showed **DKKM 0.2975 for g28 against SR_max 0.2621**. That is impossible
within a month: the estimators are scored against the TRUE moments, and Cauchy-Schwarz gives
`w'mu / sqrt(w' Sigma w) <= sqrt(mu' Sigma^-1 mu) = sr_max` for every w. So either something
was badly wrong, or the two sides were averaged over different months.

It is the second, and the size is the point: g28's sr_max is **0.2621 over all 485 months and
0.3087 over the 125 the estimators are scored on**, a gap of 18%. DKKM sits under the second.
Checked on all four economies and all forty runs: the bound holds every time once the window
matches, and never for g28 against the all-month figure.

This is the §41 confound stated as a violated inequality rather than a caveat, and it is
retroactively checkable: `sr_max` was always saved per month in `*_ts.csv`, so
`aggregate_seeds.py` now computes `sr_max_eval` for EVERY run including the twenty whose
oracles predate `--eval_window`. `tests/test_aggregate_seeds.py` asserts no method beats it.
`room_eval` still needs the re-runs, because the per-basis conditional series (`cond_sr_ts`)
is not saved -- only its mean across months.

**Do not difference a pre-2026-09-09 room against a gap, and do not compare a gap to an
all-month SR_max.** The vyx and g0235 oracle re-runs (arrays `62949932`, `62949933`) are
running to close the first of those.

## §49. The eval-window re-runs: bit-identical, and they do not rescue the ceiling (2026-09-10)

Twenty oracle-only re-runs (arrays `62949932` vyx, `62949933` g0235; `SEED_STAGE=oracle`,
output staged outside `variants/results`) to supply the evaluation-window room that §41 and
§48 said was missing for the two older flagships. All COMPLETED. vyx 1:00-1:36 at 30.6 GiB,
g0235 1:21-3:41 at 16.4-40.7 GiB, matching the full runs' oracle stages.

**They reproduce the committed economies EXACTLY.** 51 fields per file -- `sr_max_mean`,
`mean_mu`, `sd_mu`, `mean_idio_sd` and `const_best`/`const_z0`/`unc_best`/`cond_oracle` for
every basis -- compared against the committed JSONs: **max absolute difference 0.000e+00 on
all twenty files.** Not "within tolerance": bit-identical. The panel is a deterministic
function of (spec, seed) and that is now demonstrated rather than assumed, which is also the
strongest check yet that the seeded-replication machinery does what it claims. The staged
files are strict supersets (same numbers, same `spec_id` and `solves`, plus the `_eval`
fields, and a clean `prov` tag replacing seed 0's old `71a4b67+dirty`), so they were adopted.

**The confound is real and is not the explanation.** Room, both definitions:

| economy | room all | room eval | gap | gap/room all | gap/room eval |
|---|---|---|---|---|---|
| kp_vy/vyx | +0.3491 (0.0365) | +0.3606 (0.0394) | +0.1046 | 0.30 | 0.29 |
| bgn_gam/g0235 | +0.0188 (0.0065) | +0.0176 (0.0119) | +0.0227 | 1.20 | **1.28** |
| gs_bx/bx7 | +0.0086 (0.0035) | +0.0066 (0.0041) | +0.0078 | 0.90 | **1.17** |
| gs_bx/g28 | +0.0001 (0.0003) | +0.0004 (0.0003) | +0.0283 | 393 | 70 |

§40 flagged the month-sample mismatch as an unruled-out confound under "the realized gap
exceeds the const-θ room" and said to check it before treating the rolling window as the
explanation. **Checked, and it is not the confound: correcting the months moves g0235 from
1.20 to 1.28 and bx7 from 0.90 to 1.17, both further above one.** vyx is unmoved (0.30 to
0.29). The finding stands on the commensurable numbers, and the explanation is what §40
proposed -- a rolling-window estimator beats a fixed-coefficient rule when the conditional
tangency moves. What the confound DID explain is the apparent Cauchy-Schwarz violation in
§48, which was entirely a month-sample artifact.

`tests/test_aggregate_seeds.py`'s eval-room test was inverted: it used to assert vyx's
room_eval was ABSENT, which was pinning a temporary state. It now asserts the correspondence
-- room_eval present exactly where the oracle JSON carries `eval_window` -- plus that all four
flagships now have it on every seed.

**Also this session:** `aggregate_seeds.py` gained `room_*_pct_lin` and `gap_pct_lin`, room and
gap as a percentage of the linear Sharpe actually attained, per seed and aggregated both as
ratio-of-means (`*_over_lin`, what the tables quote) and mean-of-ratios (`*_pct_lin_mean/_sd`).
They differ for g0235, whose linear Sharpe has sd 0.0359 on a mean of 0.0855: gap reads 26.5%
as a ratio of means and 31.2% as a mean of ratios. **The proportional gap reorders the
economies**: g0235 26.5%, vyx 16.3%, g28 10.5%, bx7 2.7% -- vyx's fourfold absolute lead is
partly that everything in vyx is large, its linear methods already reaching 0.64 against
g0235's 0.086. `docs/RESULTS.md` carries both, and its "How to read this" is now a per-column
glossary.

## §50. The pre-refactor results deleted (2026-09-10)

Seth: "there is no reason I can see to keep around legacy results on code that was broken and
now has been fixed. What case is there to retain it?" Checked rather than agreed, because
"broken" does not describe all of it, and found no case that survives.

**Removed:** `variants/results/grid_summary.csv` (24 economies), `oracle_summary.csv` (57 rows
over `bgn, bgn_dis, bgn_types, gs_dis, gs_fixed, kp, kp_dis`), `summary_grid.xlsx`, the 32
unseeded run files that produced the three surviving grid rows plus the N=100/T=30-scale
calibration and timing probes, and the two scripts whose only purpose was those tables
(`make_excel.py`, which read `grid_summary.csv`, and `collect_results.py`, which wrote
`oracle_summary.csv`). All recoverable at `23f9380`.

**Kept:** `variants/REPORT.md`. It is the narrative of that study, explicitly historical,
carries the mechanism findings and rounded versions of the figures, and does not look like
live data. `docs/RESULTS.md`'s LEGACY entries now cite it.

### The distinction Seth's framing does not make, and why it does not matter

"Broken code" is accurate for the KP rows (the growth-option arrival rate ran at 1.72 instead
of 1) and the GS rows (three Table I parameters wrong). It is NOT accurate for the seven BGN
rows: BGN's calibration passed the audit 11 of 11, and the `mat` draw-averaging defect never
touched the variants path (§18). Those results were correct; their code was deleted by
`bba735f` rather than found wrong.

The distinction does not create a retention case, because the operative property is not
"broken" but **unreproducible and superseded**:

1. **All 24 rows are unreproducible.** 21 have no code at all. Of the three that do, KP's and
   GS's describe economies the current code no longer builds.
2. **The one row whose economy still exists has been strictly improved on.**
   `bgn_gam_oracle_g0235.json` gave SR_max 0.1860; the seeded run gives 0.1860 -- the same
   economy, reproduced to the digit (§39) -- and there are now ten seeds of it. One seed of an
   identical economy carries no information the ten do not.
3. **Deleting from the working tree is not deleting.** Git retains every byte, and the
   recovery is one `git show`. The choice was never keep-or-lose; it was working tree or
   history, and history is where an unreproducible single-seed measurement belongs.

### The costs of keeping them, which are real

- **A standing caveat tax.** Which of the three models' rows still describe the economy the
  code builds is a three-part rule, and it had been written out three times (README,
  RESULTS.md, here) because every reader needs it.
- **`collect_results.py` was a live trap.** It rebuilds `oracle_summary.csv` by globbing every
  oracle JSON on disk, seeded and unseeded alike, so running it would have overwritten a
  57-row historical table with a mixture that is neither history nor current results.
- **The unseeded files sat next to the seeded ones**, distinguished only by a missing `_s000`.
  `kp_vy_oracle_vyx.json` (SR_max 1.2607, the pre-lambda economy) beside
  `kp_vy_oracle_vyx_s000.json` (1.1945) is exactly the shape of a mistake, and
  `aggregate_seeds.py` carries a guard and a test *because of it*. Removing the files removes
  the hazard rather than defending against it.

### What was checked before deleting

The calibration and timing probes (`pin_*`, `cput_*`, `calib*`, `cal_*`) are cited by no file
in the repository; the cost models they fed are recorded in `run_seeds_slurm.sh`'s header and
in §19/§33/§46. Only `tests/test_oracle_nesting.py` referenced the deleted tables, in its
docstring, to explain why two grid rows reported a negative room -- repointed, since the
nesting defect it pins outlives the table that exposed it. `variants/results` now holds 316
seeded result files and the two generated tables, and nothing else.

## §51. RESULTS.md ordered by proportional gap, with the next parameterizations proposed (2026-09-10)

Seth: order RESULTS.md by percentage gap, and brainstorm new parameterizations per economy with
the reasoning. Done in `docs/RESULTS.md`: sections now run BGN (g0235, 26.5% of the linear
Sharpe), KP14 (vyx, 16.3%), GS21 (g28 10.5%, bx7 2.7%); each model section ends with a
"Proposed next parameterizations" subsection, and "Proposed next, ranked" collects them by
information per node-hour. Each proposal states what differs from the current economy in model
terms, why the number is worth having, a falsifiable prediction, and the solve and cluster cost.

Facts checked before proposing: BGN's regime switch probabilities enter the J* solve through
`Preg` (rebuild is minutes), and its multiplier frontier is 2 x gmult x scale < 1 with `scale`
the fitted beta-tail from the acceptance-probability targets; GS's simulator exports a
per-firm-month `default` flag, so the dormant default channel can be probed from one panel;
KP's type count is free (`ntypes = len(type_share)`); all forty seeded panels and moment files
are on Sol (15 GB), so a window ladder needs no new solve.

The ranking's top three: the BGN regime-persistence ladder (cheapest lever on the best
proportional economy and a direct test of the rolling-window mechanism behind §40/§49);
gamma(x) times exposure types in GS (the design rule that produced vyx, with GS21's own state;
its negative would retire the bx path); a continuum of exposures in KP (whether vyx's linear
methods live off three type points). Also flagged: vyx's premia are already 6-28% a year, and
every dial-up should report the annualised premia beside the gap.
