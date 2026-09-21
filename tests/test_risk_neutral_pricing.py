"""Pricing must be consistent with the SDF each model declares.

Two defects were found on 2026-09-18, both invisible to every existing check because those
compare realised with expected returns (a statement about the simulation) and never ask
whether prices satisfy E[M R] = 1 (a statement about the solve):

  * kp_vy carried the price of the mean-reverting state y the way KP14 carries its GBM shocks,
    as a constant added to the discount rate. For an OU state the Girsanov adjustment saturates,
    so that over-discounts; claim values were understated by 7% / 18% / 25% across the three
    exposure types. The corrected specification is variants/kp_vy/parameters_kp14.py, which
    substitutes W = e^{by} A and solves under the Q-generator.
  * BGN's bond recursion added the log-kernel/short-rate covariance to the cumulative variance
    once instead of twice, halving the rate-risk premium: a limiting yield spread of 1.18% a
    year where BGN (1999, p.21) report 2.4% for the same beta_zr.

Each test below states the identity that failed and would fail again if either came back.

2026-09-21: gs_bx is added as the third model. Nothing was wrong with it -- but it was the one
model with no E[M R] = 1 check anywhere, which is the blind spot both defects lived in, and the
quantity was already being computed and thrown away. An identity that is measured but never
asserted is not a check.
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code, cwd, env=None):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=cwd, env=e,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


_KP = r"""
import json, numpy as np
from scipy.integrate import quad
import parameters_kp14 as P
def exact(b, y, th):
    k, s, g, rho = P.kappa_y, P.sigma_y, P.gamma_v, P.const_base_y[0] + th
    f = lambda t: np.exp(-rho*t + b*(y*np.exp(-k*t) - g*s/k*(1 - np.exp(-k*t)) - y)
                         + 0.5*b*b*s*s*(1 - np.exp(-2*k*t))/(2*k))
    return quad(f, 0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
worst = 0.0
for f, b in enumerate(P.type_bv):
    for i in (0, P.NY//2, P.NY - 1):
        for tab, th in ((P.A0_ty, 0.0), (P.A3_ty, P.theta_eps + P.theta_u)):
            worst = max(worst, abs(tab[f, i]/P.type_theta[f]/exact(b, P.y_grid[i], th) - 1))
print(json.dumps({"worst": worst, "rn": int(P.y_risk_neutral)}))
"""


def test_kp_vy_A_matches_the_risk_neutral_closed_form():
    """A(y) = E^Q int e^{-rho t} e^{b (y_t - y)} dt has a closed form (OU moments under Q)."""
    ov = {"type_share": [0.34, 0.33, 0.33], "type_bv": [0.02, 0.07, 0.14], "gamma_v": 1.8, "bv_comp": 1.2}
    out = _run(_KP, os.path.join(ROOT, "variants", "kp_vy"), {"KP_PARAM_OVERRIDES": json.dumps(ov)})
    assert out["rn"] == 1, "risk-neutral pricing must be the default"
    assert out["worst"] < 1e-4, f"A departs from the closed form by {out['worst']:.2e}"


def test_kp_vy_legacy_switch_reproduces_the_constant_rate_error():
    """y_risk_neutral=0 exists to rebuild the pre-fix tables; it must still BE the pre-fix solve."""
    ov = {"type_share": [0.34, 0.33, 0.33], "type_bv": [0.02, 0.07, 0.14], "gamma_v": 1.8,
          "bv_comp": 0.0, "y_risk_neutral": 0}
    out = _run(_KP, os.path.join(ROOT, "variants", "kp_vy"), {"KP_PARAM_OVERRIDES": json.dumps(ov)})
    assert 0.2 < out["worst"] < 0.35, f"legacy path error is {out['worst']:.3f}, expected 0.25 at y=0 to 0.30 at the grid edge"


def test_kp_vy_without_exposure_is_kp14():
    """At type_bv=[0] nothing depends on y and A is KP14's eq. (11) constant, under either switch."""
    code = ("import json, numpy as np, parameters_kp14 as P\n"
            "print(json.dumps({'d': float(np.abs(P.A0_ty*P.const_base_y[0] - 1).max())}))")
    for rn in (0, 1):
        out = _run(code, os.path.join(ROOT, "variants", "kp_vy"),
                   {"KP_PARAM_OVERRIDES": json.dumps({"y_risk_neutral": rn})})
        assert out["d"] < 1e-9


_BGN = r"""
import json, numpy as np
from numpy.polynomial.hermite import hermgauss
import vasicek as v
from parameters import *
x, w = hermgauss(60); x = x*np.sqrt(2); w = w/np.sqrt(np.pi)
g = np.asarray(gmult, float); worst = 0.0
for r0 in (rbar - 0.004, rbar, rbar + 0.004):
    for beta in (-0.3, -0.05):
        for s in (0, 1):
            rp = kappa*r0 + (1 - kappa)*rbar - beta_zr + sigma_r*x
            cont = sum(Preg[s, s2]*(w*v.V_gam(rp, beta, s2)).sum() for s2 in (0, 1))
            rhs = np.exp(-r0)*pi*(v.Chat_v*np.exp(-beta*g[s]) + cont)
            worst = max(worst, abs(rhs/float(v.V_gam(np.array(r0), beta, s)) - 1))
k = 1400; spread = 12*(-np.log(v.B(k, rbar))/k - rbar)
print(json.dumps({"worst": worst, "spread": float(spread)}))
"""


def test_bgn_project_value_satisfies_its_euler_equation():
    """V_s(r) = E_t[m' pi (C' + V_s'(r'))], the short rate's Q-drift shifted by exactly beta_zr.
    The regime scales the price of the cash-flow shock only; rate risk stays at the baseline
    price, which is what regime-independent bond prices B(k, r) mean."""
    for gm in ([1.0, 1.0], [0.2, 3.5]):
        out = _run(_BGN, os.path.join(ROOT, "variants", "bgn_gam"),
                   {"BGN_PARAM_OVERRIDES": json.dumps({"gmult": gm})})
        assert out["worst"] < 1e-6, f"gmult={gm}: Euler error {out['worst']:.2e}"


def test_bgn_term_spread_is_the_papers():
    """BGN (1999, p.21): beta_zr = -0.00014 'gives a spread between the average short rate and the
    limiting yield of 2.4%'. The halved-covariance recursion gave 1.18%."""
    out = _run(_BGN, os.path.join(ROOT, "variants", "bgn_gam"))
    assert 0.021 < out["spread"] < 0.026, f"limiting yield spread is {out['spread']:.4f} a year"


_GS = r"""
import json, numpy as np
np.seterr(all="ignore")
import gs_sim_bx as gs
np.random.seed(11)
N, T = 40, gs.burnin + 25
arr = gs.create_arrays(N, T)
worst = max(np.abs(gs.conditional_moments(arr, t)["euler"] - 1).max()
            for t in range(gs.burnin + 5, T - 2))
print(json.dumps({"worst": float(worst)}))
"""


def test_gs_bx_prices_satisfy_its_euler_equation():
    """The third model, and the reason this file exists rather than three separate ones.

    2026-09-21: kp_vy and bgn_gam each shipped a pricing defect that every existing validator
    missed, because they compare realised with expected returns and never ask whether prices
    satisfy E[M R] = 1. gs_bx was the one model with no such check anywhere. It turns out the
    quantity was already being COMPUTED -- gs_sim_bx.conditional_moments returns an `euler`
    field and variants/gs_bx/validate_gs_bx.py prints max|euler - 1| -- and simply never
    asserted, so nothing would have failed if it drifted. This asserts it.

    Both economies are checked because they differ in the one way that could break it: gsbase
    prices the shock at a constant gamma_x = 0.5, while g28's gamma(x) is countercyclical, and
    a state-dependent price is exactly what a renormalisation can get wrong.
    gs_solve_reg.py:133 claims the per-regime kernels are "exactly renormalized
    (E[M_s|x] = e^{-r} in every state)"; this is that claim, measured. Both come back at
    machine precision, about 1e-15.

    Cost is the 88-100 MB solution load, not the panel: about 20 s of the 25 s per economy.
    """
    for soldir in ("sol_gsbase", "sol_g28"):
        out = _run(_GS, os.path.join(ROOT, "variants", "gs_bx"),
                   {"GS_BX_SOLDIRS": soldir, "GS_BX_BETAS": "1.0", "GS_BX_SHARES": "1.0"})
        assert out["worst"] < 1e-10, f"{soldir}: max |E[MR] - 1| = {out['worst']:.2e}"
