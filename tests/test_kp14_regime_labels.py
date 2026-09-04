"""The KP14 lambda-regime convention, pinned.

Background: KP14_MU_H / KP14_MU_L are ENTRY rates, named for the state they lead
TO. Three places in the codebase pin the stationary P(high) independently, and
before 2026-09-04 two of them disagreed with the other two, so the simulated
economy had E[lambda] = 1.7172 instead of the intended 1.0.

These tests make that class of drift loud. Run with: python -m pytest tests/ -k kp14
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as c


P_H_EXPECTED = 0.3191489361702128  # MU_H / (MU_H + MU_L)


def test_prob_h_is_the_entry_rate_ratio():
    """P(high) = MU_H/(MU_H+MU_L), because MU_H is the low->high ENTRY rate."""
    assert c.KP14_PROB_H == c.KP14_MU_H / (c.KP14_MU_H + c.KP14_MU_L)
    assert abs(c.KP14_PROB_H - P_H_EXPECTED) < 1e-15


def test_lambda_is_normalised_to_one_under_that_probability():
    """LAMBDA_L exists to make E[lambda] = 1. It only does so at P_H = 0.3191.

    This is the test that would have caught the original bug: the old
    KP14_PROB_H = 0.6809 gave E[lambda] = 1.7172.
    """
    e_lambda = c.KP14_PROB_H * c.KP14_LAMBDA_H + (1 - c.KP14_PROB_H) * c.KP14_LAMBDA_L
    assert abs(e_lambda - 1.0) < 1e-12, f"E[lambda] = {e_lambda}, expected 1.0"


def test_lambda_l_is_a_feasible_arrival_rate():
    """The alternative convention implies LAMBDA_L = -1.88, which is nonsense."""
    assert c.KP14_LAMBDA_L > 0
    assert c.KP14_LAMBDA_L < c.KP14_LAMBDA_H


def test_exit_rates_are_the_entry_rates_crossed():
    """You leave HIGH at the rate that enters LOW, and vice versa."""
    assert c.KP14_EXIT_H == c.KP14_MU_L
    assert c.KP14_EXIT_L == c.KP14_MU_H


def test_fd_recombination_coefficients_agree_with_prob_h():
    """utils_kp14/kp14_fd.py:118-119 pins P(high) a third way.

    It writes  G_up  = Gbar + (1 - P_H) * D   with coefficient MU_L/(MU_L+MU_H)
    and        G_down = Gbar -      P_H  * D   with coefficient MU_H/(MU_L+MU_H).
    Both must match KP14_PROB_H or the G functions are recombined on a different
    stationary distribution than the one the panel is simulated from.
    """
    up_coef = c.KP14_MU_L / (c.KP14_MU_L + c.KP14_MU_H)
    down_coef = c.KP14_MU_H / (c.KP14_MU_L + c.KP14_MU_H)
    assert abs(up_coef - (1 - c.KP14_PROB_H)) < 1e-15
    assert abs(down_coef - c.KP14_PROB_H) < 1e-15


def test_simulated_chain_reproduces_prob_h():
    """End-to-end: the transition rule in panel_functions_kp14.py:135 must give
    a long-run fraction of high months equal to KP14_PROB_H.

    Replicates that rule exactly, using the same exit-rate mapping the module
    now imports. Guards against someone 'fixing' the import back.
    """
    from config import KP14_EXIT_H as mu_H, KP14_EXIT_L as mu_L, KP14_DT as dt

    rng = np.random.default_rng(0)
    n, t = 4000, 6000
    state = rng.binomial(1, c.KP14_PROB_H, size=n).astype(float)
    total = 0.0
    for _ in range(t):
        switch = np.where(state == 1, mu_H * dt, mu_L * dt)
        state = np.where(rng.random(n) < switch, 1 - state, state)
        total += state.mean()
    frac_high = total / t
    assert abs(frac_high - c.KP14_PROB_H) < 0.01, (
        f"simulated P(high) = {frac_high:.4f}, config says {c.KP14_PROB_H:.4f}"
    )


def test_simulated_mean_lambda_is_one():
    """The economically meaningful consequence: lambda averages 1 over time."""
    from config import KP14_EXIT_H as mu_H, KP14_EXIT_L as mu_L, KP14_DT as dt

    rng = np.random.default_rng(1)
    n, t = 4000, 6000
    state = rng.binomial(1, c.KP14_PROB_H, size=n).astype(float)
    total = 0.0
    for _ in range(t):
        switch = np.where(state == 1, mu_H * dt, mu_L * dt)
        state = np.where(rng.random(n) < switch, 1 - state, state)
        rate = c.KP14_LAMBDA_L + state * (c.KP14_LAMBDA_H - c.KP14_LAMBDA_L)
        total += rate.mean()
    assert abs(total / t - 1.0) < 0.02, f"E[lambda] = {total / t:.4f}, expected 1.0"


# --------------------------------------------------------------------------
# The variants/kp_vy tree carries an independent copy of the same convention.
# It uses `from parameters_kp14 import *`, so it could not be fixed at the
# import line the way the main tree was -- the exit-rate expressions were
# swapped individually instead. These tests pin the result.
# --------------------------------------------------------------------------

VARIANTS_KP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "variants", "kp_vy"
)


def _load_variant_params():
    """Import variants/kp_vy/parameters_kp14.py under the vyx overrides."""
    import importlib.util

    os.environ.setdefault(
        "KP_PARAM_OVERRIDES",
        '{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],'
        '"gamma_v":1.8,"bv_comp":1.2}',
    )
    path = os.path.join(VARIANTS_KP, "parameters_kp14.py")
    sys.path.insert(0, VARIANTS_KP)
    try:
        spec = importlib.util.spec_from_file_location("_vparams", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(VARIANTS_KP)


def test_variants_prob_h_matches_main():
    v = _load_variant_params()
    assert abs(v.prob_H - c.KP14_PROB_H) < 1e-15
    assert abs(v.prob_H - P_H_EXPECTED) < 1e-15


def test_variants_lambda_normalised_to_one():
    v = _load_variant_params()
    e_lambda = v.prob_H * v.lambda_H + (1 - v.prob_H) * v.lambda_L
    assert abs(e_lambda - 1.0) < 1e-12, f"E[lambda] = {e_lambda}"


def test_variants_exit_rates_defined_and_crossed():
    v = _load_variant_params()
    assert (v.exit_H, v.exit_L) == (v.mu_L, v.mu_H)


def test_variants_ou_generator_has_the_right_stationary_split():
    """kp14_fd_vy.py builds Qs with H->L = mu_L and L->H = mu_H.

    Reproduces that 2x2 block and checks its stationary distribution. If someone
    swaps those two lines back, P(high) flips to 0.6809 and the G tables are
    solved on a different economy than the panel is simulated from -- silently.
    """
    v = _load_variant_params()
    src = open(os.path.join(VARIANTS_KP, "kp14_fd_vy.py")).read()
    assert "Qs[2 * iy, 2 * iy + 1] += mu_L" in src, "H->L rate is not mu_L"
    assert "Qs[2 * iy + 1, 2 * iy] += mu_H" in src, "L->H rate is not mu_H"

    Q = np.array([[-v.mu_L, v.mu_L], [v.mu_H, -v.mu_H]])
    w, vec = np.linalg.eig(Q.T)
    pi = np.real(vec[:, np.argmin(np.abs(w))])
    pi /= pi.sum()
    assert abs(pi[0] - v.prob_H) < 1e-12, f"generator P(high) = {pi[0]}"


def test_variants_panel_transition_uses_exit_rates():
    """panel_functions_kp14.py:96 must leave the HIGH state at rate mu_L."""
    src = open(os.path.join(VARIANTS_KP, "panel_functions_kp14.py")).read()
    assert "np.where(curr == 1, mu_L * dt, mu_H * dt)" in src, (
        "the regime transition is not using exit rates"
    )


def test_variants_g_recombination_left_alone():
    """kp14_fd.py:100-101 uses the ENTRY rates and was already correct.

    Guards against an over-eager future swap: these two coefficients must stay
    mu_L/(mu_L+mu_H) for G_up and mu_H/(mu_L+mu_H) for G_down.
    """
    for fn in ("kp14_fd.py", "rebuild_kp_tables.py"):
        src = open(os.path.join(VARIANTS_KP, fn)).read()
        assert "mu_L/(mu_L + mu_H) *(lambda_H - lambda_L)" in src, fn
        assert "mu_H/(mu_L + mu_H) *(lambda_H - lambda_L)" in src, fn


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
