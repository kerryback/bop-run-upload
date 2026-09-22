"""The ridge-grid gate must separate a TRUNCATED tail from a FLAT one.

Until 2026-09-22 the gate asked only where the argmax landed, and an argmax at the ceiling
was read as censoring. That is wrong whenever the Sharpe curve has flattened: as kappa -> inf
the ridge coefficient (X'X + kI)^-1 X'y -> X'y / k, so the portfolio DIRECTION stops depending
on kappa, and a Sharpe ratio is scale-invariant. sharpe(kappa) therefore has a horizontal
asymptote, the argmax lands on whichever ceiling node wins in the sixth decimal, and every
further decade reproduces it exactly. The protocol-v3 campaign put 16 of 130 seeds at the
ceiling and not one of them gained more than 2.7e-05 over its best interior penalty; reading
those as censored would have bought another 130-job campaign for nothing.

The gate now asks what the edge win is WORTH. These two cases are the whole of that claim:
a tail that is still climbing must still fail, and a flat one must pass.

Run with: python -m pytest tests/ -k penalty_gate
"""
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))
sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
import penalty_gate  # noqa: E402
import protocol  # noqa: E402

KAPPAS = list(protocol.KAPPAS)


def _write(dirpath, tag, curve):
    """Ten seeds of one economy, every seed carrying the same Sharpe-vs-kappa curve."""
    for seed in range(protocol.SEEDS):
        rows = [{"method": "rff", "P": 360, "kappa": k, "sharpe": s}
                for k, s in zip(KAPPAS, curve)]
        # a non-DKKM row, which the gate must ignore however high it scores
        rows.append({"method": "ew", "P": 1, "kappa": 0.0, "sharpe": 9.0})
        pd.DataFrame(rows).to_csv(
            os.path.join(dirpath, f"m_estimators_{tag}_s{seed:03d}_w360_summary.csv"),
            index=False)


def _verdicts(curve):
    with tempfile.TemporaryDirectory() as td:
        _write(td, "t", curve)
        w = penalty_gate.winners(td)
        assert len(w) == protocol.SEEDS, w
        return penalty_gate.report(w, 8, KAPPAS)


def test_a_tail_still_climbing_is_censored():
    """Every decade above 1 buys real Sharpe, so another decade would buy more."""
    n = len(KAPPAS)
    curve = [0.30] * (n - 3) + [0.34, 0.38, 0.42]
    text, failed = _verdicts(curve)
    assert failed, f"a climbing tail passed the gate:\n{text}"
    assert "CENSORED" in text
    assert "0 of 10" in failed[0], failed


def test_a_flat_tail_passes_on_materiality():
    """The argmax is at the ceiling in all ten seeds, and it is worth 1e-06."""
    n = len(KAPPAS)
    curve = [0.30] * (n - 3) + [0.419998, 0.419999, 0.420000]
    text, failed = _verdicts(curve)
    assert not failed, f"a flat tail was reported as censored:\n{text}"
    assert "PASS (flat tail)" in text, text


def test_the_flat_tail_clause_is_not_a_blanket_pardon():
    """Just over the tolerance must still fail: the clause is a measurement, not an excuse."""
    n = len(KAPPAS)
    curve = [0.30] * (n - 1) + [0.30 + 10 * penalty_gate.TOL]
    text, failed = _verdicts(curve)
    assert failed, f"an edge win worth {10 * penalty_gate.TOL:g} passed:\n{text}"


def test_an_interior_argmax_passes_on_position_alone():
    n = len(KAPPAS)
    curve = [0.30] * n
    curve[n // 2] = 0.50
    text, failed = _verdicts(curve)
    assert not failed, text
    assert "flat tail" not in text, "an interior win must pass on position, not the clause"


def test_the_tolerance_is_below_what_the_document_prints():
    """RESULTS.md prints Sharpe to four decimals; TOL must not let the fourth one move."""
    assert penalty_gate.TOL <= 5e-5


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
