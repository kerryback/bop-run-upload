"""The DKKM estimator is the draw-averaged portfolio, and its risk must be computed
where Sigma still exists.

History (2026-09-05). Run commit dc48c98 changed dkkm_results to one row per random-
feature draw (`mat`), with the docstring instruction "average across mat downstream".
analyze.py was never updated: it computes sharpe = mean/stdev per row and then groups by
['alpha','nfeatures','iter'], so `mat` is silently dropped and the PER-DRAW Sharpes are
averaged. fama_results has no draw dimension, so the error lands only on DKKM -- the
treatment arm of the whole FM-vs-DKKM gap question.

It cannot be repaired downstream. `mean = w @ rp` and `xret` are linear in w and so are
recoverable by averaging the per-draw rows, but `stdev = sqrt(w' Sigma w)` is convex and
Sigma is discarded when evaluate_sdfs returns -- results.pkl never stores it. A panel
written without the draw-averaged stdev cannot be fixed without re-running the simulation,
and a 10-panel run is ~80 GB of scratch that is not retained.

Run: python tests/test_dkkm_draw_average.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "utils", "evaluate_sdfs.py")


def _live(path):
    return "\n".join(l for l in open(path).read().splitlines()
                     if not l.lstrip().startswith("#"))


# ------------------------------------------------------- the mathematics ----
# These are the two facts the implementation relies on. They are asserted here so that
# anyone "simplifying" the code by averaging the per-draw statistics has to break a test.

def _stats(W, S, mu):
    sig = np.sqrt(np.einsum("ij,jk,ik->i", W, S, W))
    wb = W.mean(0)
    return wb @ mu, np.sqrt(wb @ S @ wb), (W @ mu), sig


def test_mean_is_linear_so_averaging_draws_reproduces_it_exactly():
    rng = np.random.default_rng(7)
    for _ in range(50):
        n, k = 40, 6
        A = rng.standard_normal((n, n)) / np.sqrt(n)
        S = A @ A.T + 0.5 * np.eye(n)
        mu = rng.standard_normal(n) * 0.02
        W = rng.standard_normal((k, n))
        m_bar, _, m_draws, _ = _stats(W, S, mu)
        assert abs(m_bar - m_draws.mean()) < 1e-12 * max(1.0, abs(m_bar))


def test_risk_of_the_averaged_portfolio_never_exceeds_the_average_risk():
    """Triangle inequality for the Sigma-norm. This is why the error has a sign."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        n, k = 40, 6
        A = rng.standard_normal((n, n)) / np.sqrt(n)
        S = A @ A.T + 0.5 * np.eye(n)
        mu = rng.standard_normal(n) * 0.02
        W = rng.standard_normal((k, n))
        _, s_bar, _, s_draws = _stats(W, S, mu)
        assert s_bar <= s_draws.mean() + 1e-12


def test_averaging_stdevs_is_not_the_same_thing():
    """The cheap wrong fix -- aggregating the per-draw stdev column -- must differ."""
    rng = np.random.default_rng(13)
    n, k = 40, 6
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    S = A @ A.T + 0.5 * np.eye(n)
    mu = rng.standard_normal(n) * 0.02
    W = rng.standard_normal((k, n))
    _, s_bar, _, s_draws = _stats(W, S, mu)
    assert s_draws.mean() > 1.05 * s_bar, \
        "these should differ materially for independent draws"


# ------------------------------------------------------------- the source ----

def test_evaluate_sdfs_emits_the_draw_averaged_portfolio():
    live = _live(SRC)
    assert "all_dkkm_avg_results" in live, "the draw-averaged portfolio is not computed"
    assert "'dkkm_avg_results': dkkm_avg_results" in live, \
        "it is computed but never saved into results.pkl"


def test_it_averages_weights_not_statistics():
    live = _live(SRC)
    assert "w_bar = np.mean(draws, axis=0)" in live, \
        "weights must be averaged before the quadratic form, not after"
    assert "w_bar @ stock_cov @ w_bar" in live, \
        "the averaged portfolio's risk must use Sigma directly"


def test_per_draw_rows_are_still_emitted():
    """The new column is additive; nothing downstream should break."""
    live = _live(SRC)
    assert "'mat': mat," in live, "per-draw rows were removed"
    assert "all_dkkm_results.append" in live


def test_docstring_tells_the_reader_which_object_is_the_estimator():
    doc = open(SRC).read().split('"""')[1]
    assert "dkkm_avg_results" in doc, "the output contract does not mention the new frame"
    assert "draw" in doc.lower()


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
