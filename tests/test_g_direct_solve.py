"""The G function is a linear solve, and its accuracy check must be relative.

History (2026-09-04/05). kp14_fd_vy.py ran an implicit-Euler iteration toward the fixed
point of a LINEAR system, stopping when an *absolute* Frobenius norm fell below 1e-8.
||G|| here is 3.05e7 to 1.96e8, so that test demanded relative precision of 3.3e-16,
1.4e-16 and 5.1e-17 for the three types -- at or below float64 eps (2.2e-16). It was
unsatisfiable by construction. A run sat at its numerical noise floor for 228,000
iterations and, left alone, would have exhausted the 1,000,000-iteration cap, fallen out
of the loop, written the table and printed "converged".

The fixed point of  Mat @ G_new = G/dt + util  with  Mat = I/dt - (F+Q)  satisfies
(F + Q) G = -util, so one spsolve replaces the loop: 0.1s instead of hours, with no
tolerance to choose. It reproduces the two tables that were current to ~2e-13 relative.

Run: python tests/test_g_direct_solve.py
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KP = os.path.join(ROOT, "variants", "kp_vy")
SOLVER = os.path.join(KP, "kp14_fd_vy.py")
OV = ('{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],'
      '"gamma_v":1.8,"bv_comp":1.2}')


def _solve(ftype):
    out = os.path.join(tempfile.mkdtemp(), f"G{ftype}.csv")
    r = subprocess.run([sys.executable, "-W", "ignore", SOLVER], cwd=KP,
                       capture_output=True, text=True,
                       env=dict(os.environ, KP_PARAM_OVERRIDES=OV, KP_VY_PREFIX="vyx",
                                KP_VY_TYPE=str(ftype), KP_VY_GOUT=out))
    assert r.returncode == 0, r.stdout + r.stderr
    return out, r.stdout


# ------------------------------------------------------------- behaviour ----

def test_solves_all_three_types_to_a_small_relative_residual():
    for ftype in (0, 1, 2):
        _, log = _solve(ftype)
        m = re.search(r"relative residual ([0-9.eE+-]+)", log)
        assert m, f"type {ftype}: no residual reported\n{log}"
        rel = float(m.group(1))
        assert rel < 1e-9, f"type {ftype}: relative residual {rel:.3e}"


def test_output_shape_is_what_the_integral_stage_expects():
    out, _ = _solve(0)
    d = pd.read_csv(out)
    assert d.shape == (1000, 44), d.shape
    assert int(d.isna().sum().sum()) == 0
    for iy in range((d.shape[1] - 2) // 2):
        assert f"G_up_y{iy}" in d.columns and f"G_down_y{iy}" in d.columns


def test_it_is_fast_enough_that_a_reparametrisation_is_cheap():
    """The whole point: a new parametrization must not cost hours."""
    import time
    t0 = time.time()
    _solve(1)
    assert time.time() - t0 < 60, "a single G type should be seconds, not minutes"


def test_the_solution_reproduces_the_committed_table():
    """Guards the numerics, not just the plumbing."""
    out, _ = _solve(1)
    a = pd.read_csv(out)
    b = pd.read_csv(os.path.join(KP, "G_vyx1.csv"))
    cols = [c for c in a.columns if c.startswith("G_")]
    A, B = a[cols].values, b[cols].values
    rel = np.linalg.norm(A - B) / np.linalg.norm(B)
    assert rel < 1e-10, f"G_vyx1.csv disagrees with a fresh solve at {rel:.3e}"


# ---------------------------------------------------------------- source ----

def test_no_iteration_loop_survives():
    src = open(SOLVER).read()
    live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "for it in range(1_000_000)" not in live, "the unbounded iteration is back"
    assert "dt_fd" not in live, "the implicit-Euler step is back"


def test_accuracy_check_is_relative_and_raises():
    src = open(SOLVER).read()
    live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "raise RuntimeError" in live, "a bad solve would pass silently"
    assert "resid / scale" in live or "resid/scale" in live, \
        "the residual check must be scaled, not an absolute norm"
    assert "err < 1e-8" not in live, "the unsatisfiable absolute tolerance is back"


def test_it_cannot_claim_convergence_it_did_not_check():
    src = open(SOLVER).read()
    live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "converged iter" not in live, \
        "the save line is again asserting a convergence it never verified"


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
