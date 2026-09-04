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
throwaway clone: grafts all 21 analyze commits, zero path conflicts, `.git` grows 68M→71M
(+4.4%), `git blame` reaches through to the original 2026-03-25 commits.

- **These were one repo to begin with.** `analyze.py` was in bop-run-upload's initial commit
  (fc0cb6f, 2026-01-23), deleted in d6ec6b2 (2026-01-29); bop-analyze-remote was created
  from that split on 2026-03-25. This is re-unification.
- Only 4 tracked-path collisions (`.gitignore`, `config.py`, `environment.yml`,
  `requirements.txt`), all vanishing under the prefix. The one semantic collision is the
  module name `config` (42 import sites in run, 11 in analyze). **Fix: rename analyze's to
  `analyze/paths.py`** — 11 one-line edits. Script-directory resolution masks the problem
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
simulators) rejected. Full plan in [PLAN.md](PLAN.md).

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

See [PLAN.md §0.0](PLAN.md). Short version, verified directly from
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

`experiments/solfiles/<solve_id>.json`, committed. A few KB each; hundreds of
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
- [ ] Phase 0 (see §12) — 6 of 8 done; remaining: vyx table rebuild (running), KP delta measurement
- [ ] Phase 1 — seeded oracle array, first cluster run (g0235, then vyx)
