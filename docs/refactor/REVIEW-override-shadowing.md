# Review: parameters assigned, overridden, then re-assigned

Requested by Seth 2026-09-06. A second session is reviewing this independently; this is
the primary session's finding, deliberately stated so it can be disagreed with.

## Method

`_scratch/override_audit.py` and `override_audit2.py` walk each parameter module's AST,
locate the `globals().update(<overrides>)` site, and classify every module-level
assignment after it by whether its right-hand side **reads the name it assigns**. That
distinction is what decides whether an override survives:

```python
type_share = np.array(type_share, float)          # reads it  -> override SURVIVES
lambda_L   = (1 - mu_H/(mu_H+mu_L)*lambda_H)/...  # does not  -> override DISCARDED
```

## Result: three categories, only one of them a bug

| category | count (kp_vy / bgn / gs) | verdict |
|---|---|---|
| **A. Coercion** — re-assignment reads the name | 6 / 1 / 2 | **fine** |
| **B. Genuine derivation** — recomputed from other params | 20 / 2 / 16 | **fine, and required** |
| **C. Dead pre-assignment** — assigned identically on both sides | **1** (`lambda_L`) | **a mistake** |

**A** is `type_share@42`, `type_bv@43`, `gmult@39`, `gmreg@38` — normalisation and dtype
coercion of a value the override set. Working as intended.

**B** is `prob_H`, `exit_H/L`, `Qy`, `rho_ty`, `C`, `xgrid`, `psw`, `Mx_s`, … These are
functions of other parameters and **must** be recomputed after the overrides, or an
override of `mu_H` would leave `prob_H` inconsistent with it. Recomputation is not a
mistake; it is the thing that keeps the economy coherent. Some entries here are not
parameters at all (`t0` is a timer, `S`/`acc` are solver state) and only appear because
the module is a script.

**C is the real defect**, and it is exactly the case Seth spotted:

```
parameters_kp14.py:10   lambda_L = (1 - mu_H/(mu_H+mu_L)*lambda_H)/(1 - mu_H/(mu_H+mu_L))
parameters_kp14.py:32   globals().update(json.loads(KP_PARAM_OVERRIDES))
parameters_kp14.py:33   lambda_L = (1 - mu_H/(mu_H+mu_L)*lambda_H)/(1 - mu_H/(mu_H+mu_L))
```

The two expressions are byte-identical and **nothing reads `lambda_L` between them**, so
line 10 is dead code. It is not merely redundant — it is actively misleading, because
being assigned *before* the override site is the only reason `lambda_L` looks like a free
parameter. It is the sole name in any of the three modules with this shape.

**Fix:** delete line 10. Nothing else changes; line 33 already computes the same value
from post-override inputs.

## The systemic defect, which matters more than line 10

**Overriding any category-B name is accepted and silently discarded.** `KP_PARAM_OVERRIDES
= {"lambda_L": 9.0}` runs clean, and `lambda_L` is 0.367 — verified directly. For a spec
layer that hashes the requested overrides into a spec id, that means **a spec can record a
parameter value the economy never used**, with nothing anywhere saying so. It is the same
family as every other incident this week: a claim recorded without verification.

**Proposed fix:** a readback at the end of each parameter module —

```python
_req = json.loads(os.environ.get('KP_PARAM_OVERRIDES', '{}'))
_bad = {k: (v, globals().get(k)) for k, v in _req.items()
        if k not in globals() or not _eq(globals()[k], v)}
if _bad:
    raise ValueError(f"override(s) silently discarded (derived quantities?): {_bad}")
```

Raising is right rather than warning: every name in category B is derived, so overriding
one is always a spec error, not a preference. Note category A must still pass — the
comparison has to tolerate `list -> np.array` coercion and `type_share`'s renormalisation,
so `_eq` needs to compare post-coercion (and `type_share` may legitimately differ after
being normalised to sum 1; that case should compare proportionally or be exempted
explicitly).

## Cost of fixing, and why it should be batched

`parameters_kp14.py` is in the `kp_vy` G **and** integral source lists. Deleting one dead
line moves both solve_ids and invalidates both manifests: the G rebuild is 0.3 s, but the
**integral rebuild is 88 minutes**.

So this is a live instance of the freezing question. Recommendation: **batch it** with the
other pending source change — the `epsrel=1e-10` quadrature question (§21), which emits
~985 roundoff warnings per job — and pay the 88 minutes once, before the cluster solve run
rather than after.

## What a reviewer should push back on

- Whether `_eq` can be made to pass category A without becoming so loose it stops catching
  category B. `type_share` renormalisation is the awkward case.
- Whether raising is too strict for a name a user might legitimately want to pin (e.g.
  `y_grid` or `gm_grid` directly, bypassing `y_max`/`dy`). The alternative is an explicit
  allowlist of "derived but pinnable" names, which is more machinery.
- Whether line 10 is really dead: the claim rests on nothing reading `lambda_L` in lines
  11-31, which is checked, and on the two expressions being identical, which is checked.
