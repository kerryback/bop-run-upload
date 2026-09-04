"""The CIR transition-density quadrature must adapt to the density, and must shout.

History: utils_kp14/integ_kp14.py once integrated over a FIXED interval [0, 10].
At small sigma_eps the non-central chi-squared transition density is a narrow
spike; QUADPACK samples only flat regions, returns 0 with a ~0 error estimate, and
never subdivides. That zeroed the integrals over the eps band holding ~98% of
firm-months and voided every KP14 expected return for two months. The three checks
guarding it tested the integrand at eps_max = 10 -- the far tail, the one end that
cannot fail -- and only printed.

Main was fixed. variants/kp_vy/integ_kp14.py and rebuild_kp_tables.py kept the
regressed pattern verbatim until 2026-09-04, when the technique (not the file --
the variant's version is adapted for the (type, y-node) structure that is the whole
point of kp_vy) was transplanted across.

Run: python tests/test_kp_quadrature.py
"""
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.stats import ncx2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THETA_EPS, DT = 0.35, 1.0 / 12.0
SHIPPED_SIGMA_EPS = 0.2          # what vyx actually runs


def _cir(sigma_eps, x0):
    c = (sigma_eps ** 2 * (1 - np.exp(-THETA_EPS * DT))) / (4 * THETA_EPS)
    d = 4 * THETA_EPS / sigma_eps ** 2
    lam = (4 * THETA_EPS * np.exp(-THETA_EPS * DT) * x0
           / (sigma_eps ** 2 * (1 - np.exp(-THETA_EPS * DT))))
    return c, d, lam


def _pdf(c, d, lam):
    return lambda ep: ncx2.pdf(ep / c, d, lam) / c


def _mass_fixed(sigma_eps, x0, eps_max=10.0):
    c, d, lam = _cir(sigma_eps, x0)
    return quad(_pdf(c, d, lam), 0, eps_max, epsabs=1e-10, epsrel=1e-10, limit=500)[0]


def _mass_adaptive(sigma_eps, x0):
    c, d, lam = _cir(sigma_eps, x0)
    lo, hi = c * ncx2.ppf(1e-13, d, lam), c * ncx2.ppf(1 - 1e-13, d, lam)
    return quad(_pdf(c, d, lam), lo, hi, epsabs=1e-12, epsrel=1e-10, limit=500)[0]


def test_the_old_fixed_interval_really_does_lose_all_mass():
    """Not a hypothetical. At sigma_eps = 0.02 it returns exactly 0."""
    m = _mass_fixed(0.02, 1.0)
    assert m < 1e-12, f"expected total mass loss, got {m}"


def test_the_adaptive_interval_holds_across_four_decades_of_sigma():
    for sigma_eps in (0.2, 0.1, 0.05, 0.02, 0.005):
        for x0 in (0.05, 1.0, 3.0):
            m = _mass_adaptive(sigma_eps, x0)
            assert abs(m - 1) < 1e-8, f"mass {m} at sigma_eps={sigma_eps}, x0={x0}"


def test_change_is_a_no_op_at_the_shipped_parameters():
    """vyx runs sigma_eps = 0.2, where both intervals agree. The transplant removes
    a landmine; it does not move the published numbers."""
    for x0 in (0.05, 1.0, 3.0):
        c, d, lam = _cir(SHIPPED_SIGMA_EPS, x0)
        f = lambda ep: ep ** 2 * _pdf(c, d, lam)(ep)  # noqa: E731
        lo, hi = c * ncx2.ppf(1e-13, d, lam), c * ncx2.ppf(1 - 1e-13, d, lam)
        old = quad(f, 0, 10, epsabs=1e-10, epsrel=1e-10, limit=500)[0]
        new = quad(f, lo, hi, epsabs=1e-8, epsrel=1e-10, limit=500)[0]
        assert abs(new - old) / abs(old) < 1e-9, f"x0={x0}: {old} vs {new}"


# ------------------------------------------------------- the source guards ----

VARIANT_FILES = ("variants/kp_vy/integ_kp14.py",
                 "variants/kp_vy/rebuild_kp_tables.py")


def test_variant_files_use_the_density_quantiles():
    for rel in VARIANT_FILES:
        src = open(os.path.join(ROOT, rel)).read()
        assert "ncx2.ppf(1e-13" in src and "ncx2.ppf(1 - 1e-13" in src, rel


def test_variant_files_no_longer_use_a_fixed_interval():
    for rel in VARIANT_FILES:
        src = open(os.path.join(ROOT, rel)).read()
        live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert "eps_max = 10" not in live, f"{rel}: fixed interval is back"


def test_variant_files_raise_rather_than_print_on_mass_loss():
    """A silent zero is what caused the original damage."""
    for rel in VARIANT_FILES:
        src = open(os.path.join(ROOT, rel)).read()
        assert "raise RuntimeError" in src, f"{rel}: mass check does not raise"
        assert "integrates to" in src, rel
        live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert "print(f'error: eps" not in live, f"{rel}: print-only check is back"


def test_main_still_has_the_guard_it_was_ported_from():
    src = open(os.path.join(ROOT, "utils_kp14/integ_kp14.py")).read()
    assert "ncx2.ppf(1e-13" in src and "raise RuntimeError" in src


def test_driver_does_not_swallow_the_raise():
    """build_vy_tables.py captures integ stderr and re-raises; discarding it would
    restore exactly the silence this guard exists to remove."""
    src = open(os.path.join(ROOT, "variants/kp_vy/build_vy_tables.py")).read()
    integ = src.split("def one(")[1].split("jobs =")[0]
    assert "stderr=subprocess.PIPE" in integ, "integ stderr is being discarded"
    assert "raise RuntimeError" in integ, "a failed integ job would pass silently"


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
