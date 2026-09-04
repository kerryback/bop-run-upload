"""`room` is only a headroom measure if the nonlinear basis nests the linear one.

room = nl_ceil - lin_ceil is read as "population Sharpe available to a nonlinear
method beyond any linear method" (variants/make_excel.py:44). That reading needs
span(lin_rank) subset of span(nonlinear basis). It did not hold for `bins` or any
`rff*` basis, which is why two rows of variants/results/grid_summary.csv report a
NEGATIVE room (GS gamma(x): room = -0.0004 with gap = +0.038).

Per decision 6 (2026-09-04, "require nesting"), build_feature_sets now also emits
`*_n` bases that append the linear columns. These tests pin that.

Run: python tests/test_oracle_nesting.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))

from common.oracle import build_feature_sets, draw_W, rank_standardize  # noqa: E402

GAMMA_GRID = np.arange(0.5, 1.1, 0.1)
TOL = 1e-10


def _fixture(N=400, d=5, seed=0, with_lev=False):
    rng = np.random.default_rng(seed)
    X_raw = rng.standard_normal((N, d))
    X_rank = rank_standardize(X_raw)
    Wd = {f"rff{P}": draw_W(P, d, GAMMA_GRID, rng) for P in (36, 360)}
    X_lev = rng.standard_normal((N, 2)) if with_lev else None
    Wd_lev = (
        {f"rffL{P}": draw_W(P, d + 2, GAMMA_GRID, rng) for P in (36, 360)}
        if with_lev
        else None
    )
    feats = build_feature_sets(
        X_raw, X_rank, None, Wd, X_lev=X_lev, Wdict_lev=Wd_lev
    )
    return feats, X_rank, X_lev


def _unexplained(basis, target):
    coef, *_ = np.linalg.lstsq(basis, target, rcond=None)
    resid = target - basis @ coef
    return (resid ** 2).sum() / (target ** 2).sum()


def _worst_over_columns(basis, cols):
    return max(_unexplained(basis, cols[:, [j]]) for j in range(cols.shape[1]))


def test_poly2_already_nests():
    """poly2 contains X_rank outright, so it nested before this change too."""
    feats, X_rank, _ = _fixture()
    assert _worst_over_columns(feats["poly2"], X_rank) < TOL


def test_every_nested_basis_spans_the_linear_one():
    feats, X_rank, _ = _fixture()
    nested = [k for k in feats if k.endswith("_n")]
    assert nested, "no *_n bases were built"
    for k in nested:
        w = _worst_over_columns(feats[k], X_rank)
        assert w < TOL, f"{k} fails to span lin_rank: {w:.3e}"


def test_the_unnested_bases_genuinely_do_not_nest():
    """Guards against the *_n bases becoming redundant.

    If someone later changes `bins`/`rff*` to include the linear columns, this
    test fires and the *_n duplicates should be removed rather than left to
    double the compute silently.
    """
    feats, X_rank, _ = _fixture()
    for k in ("bins", "rff36", "rff360"):
        w = _worst_over_columns(feats[k], X_rank)
        assert w > 1e-6, (
            f"{k} now nests ({w:.3e}); the {k}_n duplicate is redundant, drop it"
        )


def test_nested_bases_exist_for_every_rff_and_bins():
    feats, _, _ = _fixture()
    for k in ("bins", "rff36", "rff360"):
        assert k + "_n" in feats, f"missing {k}_n"


def test_levels_variant_nests_ranks_and_levels():
    """With --levels the linear ceiling is lin_rank_lev, so nesting must cover both."""
    feats, X_rank, X_lev = _fixture(with_lev=True)
    both = np.column_stack([X_rank, X_lev])
    for k in [k for k in feats if k.endswith("_n")]:
        w = _worst_over_columns(feats[k], both)
        assert w < TOL, f"{k} fails to span [X_rank, X_lev]: {w:.3e}"
    assert _worst_over_columns(feats["lin_rank_lev"], both) < TOL


def test_nesting_implies_nonnegative_room_on_random_moments():
    """The point of the change: with nesting, room cannot come out negative.

    Uses the same constant-theta population objective the oracle uses --
    theta* = E[ff']^-1 E[f], value = sqrt(E[f]' theta*) -- on synthetic moments.
    A basis that spans another cannot achieve a strictly lower optimum.
    """
    feats, X_rank, _ = _fixture()
    rng = np.random.default_rng(7)
    N = feats["lin_rank"].shape[0]
    mu = rng.standard_normal(N) * 0.01
    A = rng.standard_normal((N, N)) / np.sqrt(N)
    Sigma = A @ A.T + np.eye(N) * 0.05

    def ceiling(Phi):
        a = Phi.T @ mu
        B = Phi.T @ Sigma @ Phi
        B += np.eye(B.shape[0]) * 1e-10 * np.trace(B) / B.shape[0]
        return float(a @ np.linalg.solve(B, a))

    lin = ceiling(feats["lin_rank"])
    for k in [k for k in feats if k.endswith("_n")]:
        assert ceiling(feats[k]) >= lin - 1e-8, f"{k} scored below lin_rank"


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
