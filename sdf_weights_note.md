*KP14 crash — does the SDF-weight calc use `second_moment_inv`?*

Short answer: *No.* Every place the pipeline inverts a second moment to get SDF/portfolio weights solves the system directly (with regularization) — none of them reads the `second_moment_inv` that crashes. Here's the full map.

*1. The model's true SDF portfolio — this is the one that inverts the full second moment*

`utils_kp14/sdf_compute_kp14.py:265-277`:
```
port = scipy.linalg.solve(ER, np.ones((N + 1, 1)), assume_a="pos").reshape(-1)
...
port /= port.sum()
sdf_ret = -(port[1:] * (1 + ret[t, :] - np.exp(r*dt))).sum()
```
This solves `ER @ port = 1`, i.e. `port = ER⁻¹·1` — the HJ/MVE portfolio over the full (N+1)×(N+1) second moment (incl. risk-free). So the SDF weights *are* computed by inverting the second moment, but via a `solve`, *not* the stored `second_moment_inv`, and on the *same* degenerate `ER`.

Robust solver? Only weakly. The `try/except` adds `1e-6·I` and retries *only if solve raises*. In the crash logs the matrix is ill-conditioned in a way where Cholesky *succeeds and just warns* (`rcond≈1e-109`), so the `except` never fires and `port` is computed from the raw near-singular `ER`. The only stored outputs are the scalars `sdf_ret` and `max_sr`; `port` itself isn't even returned.

*2. The empirical SDF weights — invert the factor second moment (small, regularized)*

`utils/estimate_sdf_fama.py:117` → `utils_factors/fama_functions.py:248` `mve_data`:
```
ridge = Ridge(fit_intercept=False, alpha=360*alpha)
pi = ridge.fit(X=factor_rets, y=ones).coef_     # π = (F'F + λI)⁻¹ F'1
```
This is the MVE portfolio of the 3–5 *factor* returns — `(F'F + λI)⁻¹F'1` via ridge. Robust (ridge; α=0 falls back to stable lstsq), and on a tiny K×K matrix, not the 1000×1000 stock matrix. Fama-MacBeth also uses `linalg.pinvh` (`fama_functions.py:161`); DKKM uses `np.linalg.eigh` (`dkkm_functions.py:153`). All robust.

*3. `second_moment_inv` — bare `inv`, fed to nothing*

`utils/calculate_moments.py:84` is the *only* `np.linalg.inv` in the repo, the *only* unguarded solve, and the *only* code touching the full 1000×1000 inverse. Nothing reads its output. That's exactly why it's the only thing that crashes. The sole consumer of the moments pkl (`utils/evaluate_sdfs.py:170-172`) uses `rp`, `cond_var` (only as a quadratic form `w @ cov @ w`, never inverted), and the scalars `sdf_ret`/`max_sr` — all subset to the firms present that month.

*The bigger issue this surfaces*

The model SDF `port` inverts the *same* degenerate `ER`, and its ridge fallback does *not* trigger on the ill-conditioned-but-Cholesky-succeeds case we're hitting. So `sdf_ret`/`max_sr` for months with `bm=0` (zero-capital) firms are computed from a near-singular `solve` and are likely numerically unreliable — and `sdf_ret` *is* used downstream.

Ordering matters: in `compute_month_moments`, `sdf_loop` runs first (computes `port`/`sdf_ret`, just warns), *then* line 84 crashes. So the crash has been *masking* the fact that those `sdf_ret` values were being produced from a singular system. Fixing only line 84 lets questionable `sdf_ret`/`max_sr` flow through silently.

*Confirmed root cause (from a single-month diagnostic on kp14_0, month 560)*

• 26 firms have zero capital (`book=0` → `bm=0`); `second_moment` is finite (no NaN/inf, all prices > 0).
• Numerical rank 976/1000 (deficiency 24); smallest eigenvalue ≈ 0 → PSD but singular.
• `inv(full)` fails; `inv` after dropping the 26 zero-capital firms succeeds. → degenerate firms are the cause.

*Fix options (ordered by how much of the root issue they address)*

1. Exclude zero-capital firms from `ER` before the `solve` in `sdf_compute` (and drop the unused `second_moment_inv`). Fixes both the crash *and* the unreliable `sdf_ret`/`max_sr`; matches the `book>0` universe the panel already uses.
2. Robustify the `solve` (min-norm / `lstsq` / always-ridge) and drop or `pinv` the unused `second_moment_inv`. Keeps all firms; degenerate directions become well-defined but economically arbitrary.
3. Only fix the crash (drop/`pinv` `second_moment_inv`), leave the `port` solve as-is. Smallest change, but leaves the `sdf_ret` reliability question open.

Open question worth settling first: do `sdf_ret`/`max_sr` actually move when those 26 firms are excluded? Can extend the diagnostic to compute both ways for one month — tells us whether this is a real numerical problem or benign.
