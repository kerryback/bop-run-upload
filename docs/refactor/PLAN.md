# Recommended Plan: Ledger-First, with Unify's Config Discipline

> **Read §0.0 first.** It is an empirical finding about which economy to run next that
> was produced while checking this plan, and it changes the experiment priority.

## 0.0 The evidence on which economy to run next

`variants/results/grid_summary.csv`, all 24 economies, ranked by realized gap
(`gap_RFF-lin`, the thing we are trying to make large):

| rank | model | variant | room | **gap** | t vs FMR | lin_ceil | nl_ceil | lin | RFF |
|---|---|---|---|---|---|---|---|---|---|
| 1 | KP | priced-vol extreme **(vyx)** | +0.350 | **+0.101** | 21.6 | 0.743 | 1.092 | 0.748 | 0.848 |
| 2 | GS | gamma(x) nonlin *(deleted)* | −0.001 | **+0.041** | 14.1 | 0.395 | 0.395 | 0.351 | 0.392 |
| 3 | GS | gamma(x) *(deleted)* | −0.000 | **+0.038** | 17.9 | 0.359 | 0.358 | 0.320 | 0.358 |
| 4 | BGN | regime-g extreme **(g0235)** | +0.023 | **+0.030** | 29.1 | 0.135 | 0.158 | 0.145 | 0.175 |
| … | | | | | | | | | |
| 22 | GS | exposure-types wide **(bx7)** | +0.018 | **+0.001** | 10.4 | 0.274 | 0.292 | 0.236 | 0.237 |
| 23 | GS | exposure-types extreme (bx9) | +0.054 | −0.001 | 5.1 | 0.257 | 0.311 | 0.210 | 0.209 |

Three things follow, and they are load-bearing:

**1. `room` does not predict `gap` outside the KP priced-vol channel.**
Across all 24 economies `corr(room, gap) = +0.675`. Drop the two KP priced-vol rows and
it becomes **−0.234** (n=22). The entire positive association is carried by two
observations. "Realized gap = room × capture" is well supported *as an accounting
identity*, but as a **design rule** — pick the economy with the most room — it rests on
one channel. Selecting the next experiment by maximizing `room` is not supported by the
other 22 rows.

**2. Two of the top three gap economies have zero room, and both were deleted.**
GS gamma(x) shows `lin_ceil ≈ nl_ceil` (0.359/0.358) — genuinely no nonlinear headroom —
yet gap = +0.038 at t=17.9. Read the last four columns: RFF *attains* its ceiling
(0.358 vs 0.358) while linear-in-ranks *underperforms its own* (0.320 vs 0.359). The gap
there is **estimation efficiency — ridge shrinkage beating OLS-on-ranks at finite T — not
complexity.** That is a second, independent lever for the paper's headline claim, and the
current framing obscures it. The code for these economies is gone (kept only in
`REPORT.md` and this CSV) and would have to be reconstructed.

**3. bx7's flagship status is not supported by this table.** It ranks **22nd of 24 on
gap** (+0.0013). It was selected on room (+0.018), and `README.md` says GS stayed at bx7
rather than bx9 because "the estimators no longer capture it" — but bx7 does not capture
it either. It is also the most expensive economy to run (five 85 MB solves), the one whose
spec exists only in a shell script, and the one with an unresolved `gs_ashift`
inconsistency. **Recommendation: demote bx7 out of the first production run.**

**Revised experiment order: vyx → g0235 → (reconstruct GS gamma(x)) → bx7 last.**
This differs from the plan body below, which recommends g0235 first on the grounds that
vyx carries the KP regime bug (§8). Both are right about their own criterion. The
resolution: **fix the KP bug in Phase 0 — it is ~10 lines — and then run vyx first**,
because it is the only economy in the grid with a gap of economically interesting size.

---

**Spine:** `spec_first` — an experiment record that is a tracked artifact, with the variants oracle path treated as a *peer engine* rather than something to be merged away. That is what puts a correct, provenanced, gap-driving cluster run in week one instead of month three.

**Grafts:** `unify`'s configuration transport (import-time sourcing, AST-partitioned SETTABLE/DERIVED, content-addressed solve directories, the `_asrun`/delta protocol) — these are non-negotiable, because both `spec_first`'s and `adapter`'s transports silently corrupt derived parameters. `adapter`'s `check_protocol` / `normalize_sdf_loop`, its 1e-12 same-seed equality gate, and its `divergences.yaml` `blocks:`/`provisional:` stamp — as migration scaffolding and as the gate between a run and a table.

**Rejected outright:** `adapter`'s architecture (six permanent simulators) and its Phase 3 (scaling bx7 to N=1000/T=720 before the `gs_ashift` inconsistency is resolved). `unify`'s Phase 0-4 sequencing (three months before a new number) and its expiry of the archaeological twins.

---

## 0. The four jobs, answered up front

**Job 1 — a real configuration/experiment layer.** A named experiment becomes a tracked JSON spec. It reaches the seven subprocesses by generating a **frozen config module** per run and injecting it through the `--config CONFIG_MODULE` / `sys.modules['config']` path that all eight step scripts already accept. Not an env-var splat onto an already-executed module. This is the single most important technical decision in the plan, and the reason is §4.

**Job 2 — variant economies runnable at production scale.** Not through `main.py`, and not soon. The oracle/estimator engine already computes `room`, `gap_RFF-lin` and `t_vs_FMR`, is already seeded, and is already the thing REPORT quotes. It becomes a first-class engine behind the same spec schema, array-ified, and run at 10 seeds in week one. Porting the variant economies into `main.py`'s seven steps is Phase 6/7 work, gated on collapse tests that have never been executed.

**Job 3 — reconciling the two trees.** By demand-driven triage (§7), not by adjudicating 87 divergences up front. Three things get fixed in week one because they contaminate live results. Everything else is verified only when a spec you are about to run touches it.

**Job 4 — storage, specs and summaries linked forever.** Three tiers: spec + summary + manifest in git forever (~16 KB/run); `results.pkl` + `sdfwts.pkl` + `chars.pkl` durable (33.8 MB/run); everything else purgeable scratch. The 87× reduction lands in Phase 3, *after* byte-reproducibility, so purged evidence is recomputable rather than lost.

**Job 5 — the repo merge.** Merge, via `git subtree add --prefix=analyze`, at Phase 4. Ask Kerry in week one so the review clock runs in parallel with everything else. §9.

---

## 1. Phased plan

### Phase 0 — Rescue the record and fix what is contaminating live results
**~3 days. Blocks everything.**

1. **Transcribe the three run scripts to specs** — `var-gs_bx-bx7-v1.json`, `var-bgn_gam-g0235-v1.json`, `var-kp_vy-vyx-v1.json`. For bx7 this is the only committed record of `gmreg=[0.6,3.0]`, the `gs_ashift` ladder `{0, 0.225, 0.450, 0.675, 0.900}`, the equal 0.2 shares, and the soldir↔beta pairing; the file has one commit and no siblings, and its own oracle JSON records `overrides: "{}"`. Ship `tests/test_specs_match_shell.py`, which greps the `*_PARAM_OVERRIDES` JSON and the `GS_BX_*` lines out of each script and asserts equality with the spec, so the two cannot diverge while both exist.
2. **Retrieve the deleted baselines.** `git show bba735f~1` the pre-refactor BGN/KP/GS simulators into `tests/fixtures/prerefactor/` with recorded sha256s. Every collapse-to-baseline claim in the variant docstrings is currently unfalsifiable because the named baselines were deleted. Without these fixtures, Phase 6's premise cannot be tested.
3. **Fix the KP regime labelling — in `main`, not in `variants`.** This is the item every proposal misfiled. `config.py:257` pins `KP14_LAMBDA_L` using `mu_H/(mu_H+mu_L)` as the high-state stationary weight while `config.py:269` defines `KP14_PROB_H` as `mu_L/(mu_H+mu_L)`; the two are inconsistent, and `kp14_fd.py`'s G-recombination is byte-identical in both trees. Every published KP14 number from the *main* pipeline carries a mis-calibrated `lambda_L` and/or a mis-specified initial state. Write the derivation in `docs/kp14_regime_labels.md` from the FD block structure (`mat2 = mat + (mu_H+mu_L)I`, so column 0 is the stationary average and column 1 the difference), fix ~10 lines, and record the before/after delta on one panel. Gated on no refactor, no spec layer, no merge.
4. **Fix `variants/common/dkkm_functions.py:170`** — hardcoded `360` where `WINDOW` belongs. One character class.
5. **Retire the KP quadrature regression by deletion, not by porting.** Main's parameter-adaptive quantile interval is correct; the variants' fixed `eps_max=10` with a print-only mass check is a regression toward the pattern that produced the all-zeros integration bug. `variants/kp_vy/integ_kp14.py` is 78.9% identical to `utils_kp14/integ_kp14.py` — delete it and import. Same for the three 96–98%-identical modules (`variants/kp_vy/loadings_compute_kp14.py`, `kp14_fd.py`, `variants/bgn_gam/loadings_compute.py`). Zero numeric risk, retires four divergence rows.
6. **Validate `room` before measuring it.** `variants/results/grid_summary.csv` has GS gamma(x) rows at `room = -0.0004` with `gap_RFF-lin = +0.038`, `t = 17.9` — the DKKM advantage there comes from realized rank-linear underperforming its own ceiling, not from nonlinear room. Confirm the linear and nonlinear oracle ceilings are computed on **nested** bases; if they are not, write down what `room` means when they are not. The entire program is aimed at raising this quantity.

**Exit:** `test_specs_match_shell.py` green on all three; fixtures committed with hashes; a written derivation and a measured delta for the KP fix; a one-line stated convention for `room` on nested bases; the four duplicate-module deletions produce a byte-identical oracle JSON for g0235.

---

### Phase 1 — Seeded oracle array; the first gap-driving cluster run
**~4 days. Does not block Phase 2. This is the earliest safe result.**

`run_oracle.py` is already fully seeded (`--seed` sets both `np.random.seed` and `default_rng`), already takes `--N --T --seed --tag`, and `run_estimators.py` already computes `sharpe/hjd/real_sr/t_vs_fm`. What is missing is a spec, an array wrapper, a manifest, and a ledger.

- `bop_exp/drivers/oracle_engine.py` replaces the three `run_*.sh`, which become one-liners.
- `run_experiment.sh` with `--array`, `--seed $SLURM_ARRAY_TASK_ID`, per-spec `BOP_SCRATCH_DIR`, and `export PATH="$CONDA_PREFIX/bin:$PATH"` (2026-08-31; do not remove).
- `finalize_experiment.sh` as a `--dependency=afterany` job that writes `summary.json`, the manifest, and the ledger row. `afterany`, not `afterok`, so a partial array still gets a summary before scratch is purged. Nothing today guarantees a summary exists at all.
- Storage: the oracle path writes ~690 MB of μ/Σ per seed at N=500 (2.75 GB at N=1000). Set `evidence.purgeable` accordingly from day one; ten seeds is otherwise 7 GB of unlabelled scratch.

**Run `g0235` first, not bx7.** `grid_summary.csv`: g0235's `gap_RFF-lin` is **0.0297 at t=29.1**; bx7's is 0.0013 at t=10.4; vyx's is 0.1009 but carries the KP bug. g0235 also needs one `Jstar_g0235.csv` rebuild rather than five 85 MB solves, so it dodges the stale-solve reuse hazard and the `gs_ashift` inconsistency entirely. Order: **g0235 → vyx (after Phase 0's KP fixes) → bx7 (Phase 7 only)**.

**Exit:** 10 seeded g0235 runs in `experiments/ledger.csv`, each with a spec_hash, a manifest, and `room`/`capture`/`gap`; re-running seed 3 from the ledger row reproduces its `summary.json` byte-for-byte.

---

### Phase 2 — Spec layer for the main pipeline, done with import-time sourcing
**~2 weeks. Blocks Phases 3, 5, 6, 7.**

- `spec/{schema,resolve,freeze,ids}.py`. `SETTABLE`/`DERIVED` name sets **generated by AST-walking `config.py`**, with a test asserting every module-level assignment falls in exactly one set.
- **Freeze produces a generated config module**, not an env splat. `bop_exp` writes `$RUNDIR/frozen_config.py` — the current `config.py` with SETTABLE assignments replaced by spec values and the derived block re-executed — and each step is invoked with `--config frozen_config`, injected via the existing `sys.modules['config']` path. See §4 for why the alternative is a silent-wrong-number generator.
- **Seeding by `np.random.seed`, not a Generator sweep.** Every draw in all three simulators and in `generate_dkkm_factors.py` goes through global `np.random` or `scipy.stats.*.rvs`. `seed_for(spec_hash, panel_index, step, stream) = blake2b(...)`, called once at the top of `generate_panel.py`, `calculate_moments.py`, `generate_dkkm_factors.py`. **Zero numerical lines touched**, and the exit criterion becomes sha256 identity rather than a distributional hand-wave. Ship `assert_no_worker_draws()` — joblib's fork backend hands every worker the parent's RNG state; `generate_dkkm_factors.py` is safe today only by accident (W is drawn in the parent).
- Delete `main.py:131-135`'s mutation, `MODEL_FACTOR_NAMES` and `'factor_names'` (no consumers), the bounds bug at `main.py:89`, and the three private char→factor maps in favor of the currently-dead `config.CHAR_TO_FACTOR`. `--chars` becomes sugar that patches the spec before freezing.
- Per-spec `BOP_SCRATCH_DIR`, plus the assertion in `evaluate_sdfs.py` that each upstream pkl's already-written `'chars'` field matches the current config.

**Exit, four checks:** (i) the same spec run twice on the same commit gives sha256-identical `results.pkl`; (ii) a subprocess spawned exactly as `main.py:180` does reports the spec's `chars`, not the default set; (iii) **a spec that sets `KP14_MU_H` produces a recomputed `KP14_LAMBDA_L`, `KP14_A_0..A_3`, `KP14_RHO`** — the derived-consistency test; (iv) `base` reproduces a pre-refactor `bgn` run within 3 MC standard errors per method.

---

### Phase 3 — Storage tiering, evidence manifest, one metric definition
**~1 week. Blocks Phase 4's analyze rewiring.**

- `np.linalg.solve(cond_var, rp)` inside `calculate_moments.py:79`, where both operands are already in RAM → `{id}_sdfwts.pkl` (2.88 MB vs 2.88 GB, ~6 s/panel). Copy `_solve_weights` verbatim so singular months give an all-NaN row; preserve `sorted()` month order and positional 0..999 columns (`fit_models.py:100-105` reads the header as `firmid`). Emit `{id}_chars.pkl`. **Validate additively** — keep writing `moments.pkl` for one run and diff against `build_sdfwts()` for byte identity — then flip `KEEP_MOMENTS`/`KEEP_PANEL` to spec-driven.
- `manifest.json` per run with path/bytes/sha256/purgeable; `bop_exp evidence check` reconciles state ∈ `{live, purged, corrupt, archived}` after a scratch wipe. Purged evidence becomes recorded-and-recomputable rather than mysterious — which only works because Phase 2 landed first.
- **`contract_version: 3` written into `results.pkl`'s existing 12-key dict**, hard-asserted by one `bop_exp/summary.py` that is the sole definition of `sharpe`, `hjd`, `real_sr`, `t_vs_fm`, `room`, `capture`. Ship a `--legacy-contract` forward-mapping reader from the first bump (v2→v3 is the missing `mat` column) so the guard never becomes the reason nobody bumps it. Promote `max_sr` (computed at `calculate_moments.py:85`, currently discarded) with `sr_max_comparable: false` for kp14, where it is the hardcoded `MAX_SHARPE`.
- `draw_pooling` as an explicit spec field. `analyze.py:215` currently forms the Sharpe per row *before* the groupby, i.e. mean-of-ratios, silently.
- `comparable_key = sha256(engine, N, T, window, chars, draw_pooling, contract_version, **seeded**)`. `ledger table` refuses to put mismatched rows in one comparison without `--force`.

**Exit:** durable footprint < 40 MB/panel measured; `summarize()` on a version-less legacy pkl fails loudly naming what it needs; `analyze.py` and `run_estimators.py` emit byte-identical summary rows for the same input.

---

### Phase 4 — Repo merge
**~3 days of work; review latency is the real cost. Not on the critical path. Ask in week 1.**

`git subtree add --prefix=analyze`, rename `analyze/config.py` → `analyze/paths.py` across its 11 sites, fix `analyze.py`'s `mat` handling and its stale docstring, and **replace `analyze.py:65`'s `glob('{model}_*_results.pkl')` with a ledger read filtered by `spec_id`** — that glob matches `bgn_gam_*` under `bgn_*` and would silently pool two economies the moment both land in one directory. §9 for the governance handling.

---

### Phase 5 — Protocol scaffolding and divergence triage machinery
**~1 week. Blocks Phase 6.**

- `SdfMoments = namedtuple('SdfMoments','sdf_ret max_sr rp cond_var w_true')` + `normalize_sdf_loop` (three lines; the variants' 5-tuple is a positional superset with identical semantics for the first four) + `check_protocol`, a signature conformance assert that fails at import rather than six hours into a moments run. The free fifth element is `w_true`, the population MVE weight the main pipeline has no access to today.
- **Execute the collapse identities for the first time**: `bgn_gam` at `gmult=[1,1]` vs `bgn`; `kp_vy` at `type_bv=[0]` vs `kp14`; same derived seed, against the Phase-0 fixtures. Record the result whether or not it passes. Note `variants/bgn_gam/sdf_compute.py` descends (C015) from a *pre-refactor* BGN snapshot, so a non-trivial delta is a live possibility.
- The `adapter` 1e-12 gate as the standing inertness test: same seed, both paths, per-month `max_sr` on the intersected range. Both call the identical `sdf_loop` on the identical `arr_tuple`, so anything above 1e-12 is a harness bug.
- `divergences.yaml` with `blocks:` / `guard:` wired to stamp `provisional: true` into the ledger row and every `summary.json`, with `analyze` filtering on it.

**Exit:** both collapse identities executed with a recorded number; `provisional` stamping demonstrated end-to-end.

---

### Phase 6 — Unify BGN and KP14
**~3 weeks. Blocks nothing downstream except the "one-file diff" property.**

Only if Phase 5's collapse deltas are acceptable. Add the `state` block (`none|markov|ou`) and the `types` block to `utils_bgn`/`utils_kp14`, with one `expect_next(tables, s, transition)` in `utils_factors/state_ops.py` replacing the four hand-rolled mixture operators (`mixed`, `_mix_series`, the GH loop, `wts = (1-p, p)`), and **explicit, CI-enforced short-circuits at `state.n == 1` and `ntypes == 1`**. Those branches are silent-wrong-number paths defended only by tests; the tests are load-bearing infrastructure and must be un-skippable. Cherry-pick mechanism hunks only — drop the `parameters.py` star-import rewiring, the `kind='cubic'` revert, the `book > 0` characteristic divergence, and the removed `solfile_spec.verify` guards. Delete `variants/{bgn_gam,kp_vy}/*.py`.

**Exit:** collapse to fixtures at 1e-12; `g0235_asrun` reproduces `bgn_gam_oracle_g0235.json` to 3 decimals; both tests in CI; a re-run of g0235 under its unchanged spec diffs empty against Phase 1's summary.

---

### Phase 7 — GS21: correctness, provenance, and a measured method question
**~3 weeks. Does not commit to a port.**

1. **Content-addressed solve directories.** `solve_hash` over the parameters `solfile_spec._config_attrs()` already discovers by AST, plus producer source digest, plus that type's exposure; directory named by the hash; `write_stamp(..., extra={spec_id, spec_hash, solve_hash, type_index, gs_bx, gs_ashift, gmreg})` using the `extra` dict that has always been accepted and never used. Also write `overrides` and `stage_name` into the `.npz` itself — `gs_solve_reg.py:217` saves a 15-element `params` array that omits both `gs_bx` and `gs_ashift`, so nothing inside a solution file identifies its own beta. This makes `run_gs_bx7.sh:5`'s silent-reuse hazard structurally impossible rather than merely checked.
2. **Resolve `gs_ashift`/`opcf` before scaling anything.** `gs_solve_reg.py:80-81` includes the shift in the payoff tables; `gs_sim_bx.py:212` omits it from `op_cash_flow`, which feeds `roe`, an exported characteristic — for four of bx7's five types. Run bx7 with and without at N=500 and report the delta in `room` and `gap`. If material, REPORT §19b/§19c is corrected with the delta visible under `bx7-v2`, with `v1` retained in the ledger. **Do not run bx7 at N=1000/T=720 until this is closed.**
3. **`xcheck-gs21-method-v1`:** one economy (gmreg=[1,1], single type, β=1, one calibration, same seed) under both discretizations. That number is the gate on any future port. No port is scheduled.
4. **Parameter hybrid:** two specs, `gs21-base` (config.py, paper-sourced) and `bx7-asrun` (the hybrid verbatim, which must reproduce the published JSON to 3 decimals). See §6 for the one thing to check first.

**Exit:** every solve directory self-identifying; the `ashift` delta measured and REPORT's status stated explicitly; the method cross-check number in the ledger; `bx7-asrun` reproducing the published JSON.

---

### Phase 8 — The experiment sequence
**Ongoing.** `bop_exp new --from var-bgn_gam-g0235-v1 --set state.risk_price_mult=[0.15,3.6] --id ...-v2` writes the child with `lineage.parent` set. Push premium-side extremity across all three economies, one spec per hypothesis, ranked in the ledger by `capture` and `gap`. Standing gate: a new economy whose diff touches any `.py` file is a failed parameterization.

---

## 2. Spec schema

`spec_hash = sha256(canonical_json(spec minus title, question, notes, lineage))`, with the hashing view itself versioned by `schema_version`, and a test asserting every numerical field lies inside the hashed view.

```jsonc
{
  "schema_version": 1,                     // int
  "spec_id": "var-bgn_gam-g0235-v1",       // str, ^[a-z0-9_]+-[a-z0-9_]+-v[0-9]+$
  "title": "…", "question": "…", "notes": "…",          // str, NOT hashed
  "lineage": {                                           // NOT hashed
    "parent_spec": "…|null", "changes": ["…"], "supersedes": "…|null"
  },

  "engine": "main" | "oracle",             // which driver
  "model":  "bgn"|"kp14"|"gs21"|"bgn_gam"|"kp_vy"|"gs_bx",

  "method": {                              // named solution-method choices, all required
    "gs21_discretization": "tauchen_exact" | "ar1_cubic_gh10",
    "kp_interpolation":    "cubic" | "linear",
    "kp_cir_quadrature":   "adaptive_quantile" | "fixed_eps10",
    "kp_regime_labels":    "corrected" | "legacy_swapped",
    "zero_book_in_sdf_solve": false        // bool
  },

  "panel": { "N": 1000, "T": 720, "burnin": 300,         // int, int, int
             "n_panels": 10, "index_range": [0, 10] },   // int, [int,int]
  "seeds": { "base": 20260904, "policy": "derive" },     // int, str

  "params": { "GS21_SIGMA_M": 2.5 },       // {str: number|bool|[number]} — SETTABLE names only;
                                           // intersection with DERIVED is a hard error

  "state": {                               // omit or kind:"none" for baseline
    "kind": "none" | "markov" | "ou",
    "n": 2,                                 // int
    "switch_probs":     [0.0208, 0.0417],   // [float], markov only, len == n
    "risk_price_mult":  [0.6, 3.0],         // [float], len == n; [1.0] is baseline
    "ou": { "kappa": 0.0, "n_nodes": 21, "y_max": 0.0,   // ou only
            "price": 0.0, "g_lo": 1.0, "g_hi": 1.0, "g_steep": 0.0 },
    "seed": 909                             // int
  },

  "types": {                               // omit for single-type baseline
    "share":    [0.2,0.2,0.2,0.2,0.2],     // [float], sums to 1 within 1e-12
    "exposure": { "gs_bx":     [1.0,2.5,4.0,5.5,7.0],   // {str: [float]}, each len == len(share)
                  "gs_ashift": [0.0,0.225,0.450,0.675,0.900] },
    "seed": 910
  },

  "solve": { "stages": [                   // [] for models with no solve stage
    { "name": "sol_reg",                                    // str
      "producer": "variants/gs_bx/gs_solve_reg.py",         // str, repo-relative
      "param_env": "GS_PARAM_OVERRIDES",                    // str, legacy transport
      "params": { "gmreg": [0.6,3.0], "gs_bx": 1.0, "gs_ashift": 0.0 },
      "args":   { "xnum": 161, "tol": 1e-6 },
      "type_index": 0 }                                     // int|null
  ]},

  "estimation": {
    "chars":      ["size","bm","agr","roe","mom","lev"],    // [str], chars[0]=="size"
    "rf_cols":    ["rf_stand","gam_stand"],                 // [str], may be []
    "window": 360, "levels": true, "include_mkt": true,     // int, bool, bool
    "nfeatures":  [6,36,360,3600],                          // [int]
    "nmat": 5,                                              // int
    "alpha_lst":  [0,0.001,0.01,0.05,0.1,1],                // [float]
    "alpha_lst_fama": [0],                                  // [float]
    "kappas": [0.001,0.01,0.03,0.1,0.3,1,3,10],             // [float], oracle engine
    "draw_pooling": "sharpe_per_draw_then_mean"             // enum, explicit
  },

  "evidence": {
    "keep":      ["results","sdfwts","chars","summary","oracle_json","estimators_csv"],
    "purgeable": ["moments","panel","weights","dkkm","arr","solution_npz","oracle_moments"]
  }
}
```

**Validators that run at resolve time, not 40 minutes into a solve:** `2*max(risk_price_mult)*scale < 1` for BGN (currently buried at `variants/bgn_gam/sdf_compute.py:52`); all A-coefficients finite and `rho_ty > 0` for KP; `i_cut << imax` at the coarsest grid corner for GS.

## 3. Directory layout

```
bop-run-upload/
  config.py                          # unchanged public surface; values sourced from the frozen spec
  spec/{schema,resolve,freeze,ids,solfiles}.py
  bop_exp/{seed,summary,stamp,ledger,cli}.py
  bop_exp/drivers/{main_engine,oracle_engine}.py
  divergences.yaml                   # TRACKED. the 99 rows, triaged
  experiments/                       # TRACKED FOREVER
    specs/{main,var,xcheck}-*.json
    runs/<spec_id>/<run_id>/         # run_id = <UTCstamp>-<spec_hash[:8]>-<git_sha[:7]>
      run.json  stamp.json  summary.json  summary.csv  manifest.json  logs/
    ledger.csv
  tests/fixtures/prerefactor/        # the bba735f~1 baselines
  analyze/                           # subtree graft, Phase 4
$BOP_SOLFILES/<model>/<solve_hash>/  # off-repo, content-addressed, stamped
$BOP_DURABLE/<spec_id>/              # results + sdfwts + chars, 33.8 MB/panel
$BOP_SCRATCH/<spec_id>/              # purgeable
```

Pre-commit size check on the `experiments/` prefix. No big artifact ever lands there; one accidentally committed 390 MB npz undoes the tracking discipline permanently.

## 4. Why import-time sourcing, not an env splat

`config.py` has ~14 derived module-level parameters (`KP14_LAMBDA_L:257`, `KP14_CONST:270`, `KP14_A_0..A_3:271-274`, `KP14_RHO:276`, `CHAT:207`) and 20 modules doing top-level `from config import (...)` — `utils_factors/sdf_utils.py:12` is `from config import TEMP_DIR`.

Any design that `setattr`s spec values onto an already-executed `config` module — `spec_first`'s `apply_spec_from_env()` inside `init_from_env()`, `adapter`'s `activate()` — leaves every derived parameter stale when its parent is overridden. Set `KP14_MU_H` and you get a `KP14_LAMBDA_L` computed from the old value: a silent wrong number, in the mechanism whose purpose is preventing silent wrong numbers, invisible to every exit criterion either proposal states. Both also inherit an import-ordering hazard that `sdf_utils.py:12` demonstrates is live.

The frozen generated config module has neither defect: derived values are computed once, in order, from the spec, before any consumer imports anything. And it uses transport machinery the repo already has and already trusts.

## 5. RNG seeding

`np.random.seed(blake2b(spec_hash | panel_index | step | stream))` at the top of each step. Seeding off `spec_hash` means two economies never share a draw sequence, so "A beats B" is never a shared-shock artifact; off `panel_index` gives independent panels; off `step` isolates panel generation from DKKM's `W` draws, so changing `nmat` does not perturb the panel — which is what makes A/B specs paired. `seeded` goes into `comparable_key`: no seeded number is ever tabled next to an unseeded published one without a `--force`.

Hard cap: if Phase 2's byte-identity exit is not met by day ten, ship seeding anyway, record `seeded: partial`, and move on. Do not let reproducibility perfectionism eat the compute schedule.

## 6. GS21 parameter hybrid — check this before adjudicating

The variant tree diverges on `rho_x = 0.95^(1/3)` vs `0.96^(1/3)`, `delta = 0.02/3` vs `0.02`, `tau = 0.2/3` vs `0.2`, `sigma_m = 2.5` vs `5`, `xnum = 161` vs `20`. Three of those are exactly a quarterly→monthly conversion applied in one tree and not the other, and main's commit `6c65bf4` is titled "GS21 monthly conversion and tax." **One of the two trees is double-converted or unconverted.** That is a 30-minute check against `GS21.m` and it must happen before either set is called "base." Everything else — the two-spec `base`/`asrun` split, the per-field decision table with a `run_id` citation each — follows from the answer.

`GS21_XNUM` is a genuine judgment call: `config.py:323-330` documents 20 as load-bearing for *main's* solver stability, and 161 works under the exact-kernel solve. Take 161 only if the exact-kernel solver is adopted; otherwise amend the note, don't delete it.

## 7. Triage rule for the 87 unverified divergences

Do not verify them. **Verification is demand-driven by the run schedule, and unification retires rows wholesale.**

1. **Classify mechanically, not by reading.** Every row gets `class ∈ {cosmetic, parameter, formula, unknown}` from `git diff` hunk shape — import style / whitespace / timestamps → `cosmetic`; a changed literal on an assignment → `parameter`; anything touching an expression → `formula`; else `unknown`. Cheap, scriptable, no judgment.
2. **Retire `cosmetic` by deletion, not verification.** The four modules at 78–98% identity go away in Phase 0. Rows die with the files.
3. **Verify only on the run path.** `bop_exp verify <spec>` computes the set of source files that spec's execution path actually imports, intersects with open rows of class ≠ cosmetic, and either blocks the run or stamps `provisional: true` into the ledger and every summary. `analyze` filters on it. A divergence in a module you are not running is not a problem you have.
4. **Verify on quote, always.** Any row that intersects a spec whose number is about to appear in a draft is promoted to `must-verify` regardless of class, and closes only with a named test in `equiv/` that demonstrably failed on the parent commit.
5. **Close by unification.** When Phase 6 deletes `variants/{bgn_gam,kp_vy}/*.py`, every row scoped to those files closes as `retired-by-merge` in one commit. Expect this to be most of the 87.
6. **Standing budget:** if a single spec is blocked by more than five open material rows, stop and either fix them or accept `provisional` explicitly in writing. Never silently.

`unknown` rows that are never on any run path and never quoted are closed as `wontverify` when their file is deleted. That is a legitimate terminal state.

## 8. The four named problems

| | Decision |
|---|---|
| **KP regime arithmetic** | Fix in **main** in Phase 0, ~10 lines (`config.py:257`/`:269` + the FD G-recombination). It is byte-identical in both trees, so fixing main fixes both once the duplicates are deleted. Write the derivation; record the delta; the corrected KP14 main-pipeline numbers supersede the published ones. Do not defer this behind any refactor. |
| **KP CIR quadrature** | Main is correct. Do not port the variant version. Delete `variants/kp_vy/integ_kp14.py` and import main's (Phase 0). Gate `vyx` runs on it — vyx is the largest-gap economy and currently the least trustworthy. |
| **GS21 parameter hybrid** | Check the /3 conversion against `GS21.m` first (§6). Then two specs: `gs21-base` (paper-sourced) and `bx7-asrun` (hybrid verbatim, must reproduce the published JSON to 3 decimals). Per-field decisions in `docs/gs21_reconciliation.md`, each citing a `run_id`. No method port is scheduled; `method.gs21_discretization` is a labelled spec field and both stay runnable. |
| **RNG seeding** | `np.random.seed` per step from `blake2b(spec_hash|idx|step|stream)`. Zero numerical lines touched. Exit criterion is sha256-identical `results.pkl`. Ship `assert_no_worker_draws()` for the joblib fork hazard. |

## 9. Repo merge

**Merge.** `git subtree add --prefix=analyze` grafts all 21 commits, grows `.git` 4.4%, preserves blame; four tracked-path collisions all fixed by the prefix; one semantic collision (`config`, 42 sites in run vs 11 in analyze) fixed by renaming analyze's to `paths.py`. The two repos were one until 2026-03-25 — this is un-splitting.

The case is empirical, not aesthetic: `dc48c98` added the `mat` column and changed the DKKM weights key, `analyze.py` was never updated, its docstring is stale, and it survives only because `mat` gets absorbed into a `groupby` mean. A `contract_version` guard that spans a repo boundary is a guard nobody bumps. `specs/`, `summaries/` and `ledger.csv` are the join key between production and analysis; split across two histories, `git log experiments/specs/` cannot explain why a number moved. The "lean compute env" objection is measurably false — 0.01 s interpreter startup either way, and no step script imports `analyze/`.

**Governance.** `bop-run-upload` is Kerry's, with Seth at WRITE (85 of 162 commits are Kerry's). The exposure is not the graft — it is that a large refactor invalidates a co-author's working model of the tree. Handle it as a conversation, not a PR description:

- **Week 1:** ask, with `dc48c98` as the concrete argument, and with the merge framed as one commit (subtree + rename + 11 import fixes) separable from everything else in this plan.
- **Phase 4, on a branch in Seth's fork,** with the mechanical evidence attached: 21 commits preserved, 4.4% growth, blame intact, four prefix-fixed collisions, one rename.
- Keep `bop-analyze-remote` as a read-only mirror fed by `git subtree push` for a transition period, for as long as Kerry wants one.
- **Separately, and before Phase 6:** walk Kerry through the simulator unification specifically. That diff is the one that changes his mental model, and it should not arrive as a surprise inside a merge PR.
- **If declined:** publish `bop_exp/` (spec + summary + ids) as a package pinned by commit in both repos, with a CI check in each that fails when the pins diverge. Strictly worse — it detects drift instead of preventing it — but it preserves the one property that matters, a single definition of every metric. Do **not** copy `summary.py` into both repos. Start the fallback in parallel at day five rather than blocking Phase 2 on a review.

## 10. Open decisions that need Seth

1. **Do the KP14 main-pipeline numbers get restated?** The regime fix changes them. *Recommendation: fix, measure the delta on one panel, restate with the delta visible. It is a live contamination, not a variants issue.*
2. **Which GS21 calibration is the conversion error?** See §6. *Recommendation: check against `GS21.m` in week one; this blocks naming a `base`.*
3. **Is `sigma_m` 2.5 or 5?** Provenance is unsettled and it is live in both trees. *Recommendation: whichever the paper says; if ambiguous, make it a named difference between two specs and measure it, rather than picking.*
4. **Do the `_asrun` twins stay runnable forever?** *Recommendation: yes. Keep bug-compat quarantined but permanent. `unify` schedules its deletion; that converts every published number from reproducible to git-tag archaeology, in the plan whose thesis is provenance. The cost is one branch in three simulators.*
5. **`draw_pooling`.** Current behavior is mean-of-ratios, undocumented. *Recommendation: keep it, state it in the spec, and note in the schema that `max_sr` is not comparable across models (kp14's is a constant).*
6. **Is `room` defined on nested bases?** *Recommendation: require nesting, or write down the convention. The GS gamma(x) rows at `room = -0.0004` with `gap = +0.038` mean at least some published `room` numbers are measuring linear underperformance, not nonlinear headroom.*
7. **First production economy: g0235.** *Recommendation: yes — 0.0297 at t=29.1 vs bx7's 0.0013, one solve rebuild instead of five, and no `gs_ashift` exposure. bx7's flagship status is not supported by `grid_summary.csv`.*
8. **Ask Kerry now or at Phase 4?** *Recommendation: ask now, merge at Phase 4.*

## 11. What this plan does not solve

- **It does not decide whether GS21's Tauchen-exact method or main's AR(1)+cubic+GH method is right.** It makes the question decidable and produces a number. Which side is correct is a numerical-analysis project this plan does not schedule and cannot shortcut.
- **It does not validate the regime machinery.** The collapse tests prove `gmult=[1,1]` reduces to baseline. Nothing tests that the mixture operator computes the correct expectation at `gmult ≠ 1`. Writing that test requires the reconciliation this plan defers.
- **It does not raise `capture`.** REPORT pins it at 75–80% by DKKM's own P/T ridge shrinkage. Every phase here is aimed at `room` and at measuring it honestly. If `room` cannot be moved much, no infrastructure fixes that.
- **It does not reduce compute, and Phase 6/7 increase it.** Variants at N=1000/T=720 are dramatically more expensive than at N=500/T=500; KP's 21-node y-tables and 7-node quadrature, and its 10.1 GB transient `arr/` (a dense (921,921,1000) `EtA` plus a mislabelled-sparse upper-triangular `uj`), are untouched here and deserve their own tickets.
- **The 87× storage win does not cover the oracle engine.** It targets `moments.pkl`. The oracle path has its own ~690 MB/seed μ/Σ problem, mitigated by `evidence.purgeable` in Phase 1 but not solved.
- **It leaves duplicate BGN/KP implementations live until Phase 6** — roughly six weeks in which shared-region fixes must be applied twice, and the documented linear-interpolation regression (RMS 7.8e-3, HJ-bound breach in 770/3,960 panel-months) persists as live variant code.
- **Phases 2–5 produce no new economics.** Phase 1 produces the first result and Phase 8 the second; everything between is apparatus. A program under time pressure can honorably stop after Phase 3 — spec, seeding, storage and the metric contract all stand alone — but the forks stay and the unification is unbought.
- **The `n==1`/`ntypes==1` short-circuits from Phase 6 are silent-wrong-number branches defended only by tests.** If those tests are ever skipped, the baseline economy changes and every prior number becomes incomparable without any error.
- **It creates a one-time re-run bill.** After Phase 2, no pre-seeding number can sit next to a post-seeding number without a caveat, and clearing the caveat means re-running.