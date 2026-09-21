"""Is A(y) the value of the claim it is supposed to be?

WHY THIS EXISTS. KP14's assets-in-place coefficient is a CONSTANT, the paper's equation
(11): A = 1/(r + gamma_x*sigma_x + delta - mu_x + theta_c). The `vy` route generalised it to
a FUNCTION of a new priced, mean-reverting state y, by keeping the paper's shape -- a risk
premium added to the discount rate -- and replacing the scalar with the resolvent
[diag(const_ty(f)) - Qy]^{-1} 1 (parameters_kp14.py:122-133). The paper's shape is exact for
a GBM. It is not exact for a mean-reverting state entering the cash flow exponentially, and
this script measures by how much.

WHAT IT CHECKS. For a claim to the flow e^{b*y_s} discounted at the constant base rate rho,
under an OU y with a price of risk gamma_v, it compares three numbers at each y:

  A_code   e^{-b*y} x (the value the module computes), i.e. the resolvent above
  A_exact  the closed form, E^Q[int e^{-rho t} e^{b y_t} dt], Q-drift -kappa_y*y - gamma_v*sigma_y
  A_const  the closed form of what A_code is ACTUALLY computing: the risk premium charged as
           a constant rate b*gamma_v*sigma_y for every horizon, rather than the Girsanov
           adjustment b*gamma_v*sigma_y*(1-e^{-kappa_y t})/kappa_y, which SATURATES

A_exact is verified independently by Monte Carlo under Q, so the comparison does not rest on
the algebra. A_code/A_exact is 1.000 at b = 0 (so kpbase is unaffected) and falls monotonically
in b.

STATUS 2026-09-21: this script HARDCODES the pre-fix resolvent rather than importing the
module, so it is unaffected by merge 91095fb and its "A_code" column now describes
y_risk_neutral = 0, not the default. It is kept as the standalone derivation of the defect --
closed form, Monte Carlo under Q, and the constant-rate comparison that diagnoses it. The
DEFAULT path is verified instead by tests/test_risk_neutral_pricing.py, which asserts A
against the same closed form.

Run with: python variants/kp_vy/check_y_risk_adjustment.py
"""
import numpy as np
from scipy.integrate import quad

# ---- vyx's economy, as parameters_kp14.py builds it -----------------------------------
R, GAMMA_X, SIGMA_X, DELTA, MU_X = 0.05, 0.69, 0.13, 0.1, 0.01
RHO = R + GAMMA_X * SIGMA_X + DELTA - MU_X      # const_base_y with gmult_y inert
KAPPA_Y = 0.35
SIGMA_Y = np.sqrt(2.0 * KAPPA_Y)                # stationary sd of y is 1
GAMMA_V = 1.8                                   # vyx; 2.5 in vyg25, 0 in kpbase
TYPE_BV = (0.0, 0.02, 0.07, 0.14)               # kpbase's single type, then vyx's three
NY, Y_MAX = 21, 3.5


def build_Qy(ny=NY, ymax=Y_MAX, drift_shift=0.0):
    """parameters_kp14.py:101-117 verbatim, plus an optional constant drift shift.

    drift_shift=0 reproduces the module (the PHYSICAL generator). The Feynman-Kac equation
    for this claim wants -GAMMA_V*SIGMA_Y + SIGMA_Y**2*b, which is what the module omits.
    """
    yg = np.linspace(-ymax, ymax, ny)
    dy = yg[1] - yg[0]
    Q = np.zeros((ny, ny))
    drift = -KAPPA_Y * yg + drift_shift
    diff = 0.5 * SIGMA_Y ** 2 / dy ** 2
    for i in range(ny):
        if drift[i] > 0 and i < ny - 1:
            Q[i, i + 1] += drift[i] / dy; Q[i, i] -= drift[i] / dy
        elif drift[i] < 0 and i > 0:
            Q[i, i - 1] += -drift[i] / dy; Q[i, i] -= -drift[i] / dy
        if 0 < i < ny - 1:
            Q[i, i - 1] += diff; Q[i, i + 1] += diff; Q[i, i] -= 2 * diff
        elif i == 0:
            Q[i, i + 1] += diff; Q[i, i] -= diff
        else:
            Q[i, i - 1] += diff; Q[i, i] -= diff
    return yg, Q


def A_code(b, ny=NY, theta_c=0.0, drift_shift=0.0, ymax=Y_MAX):
    """The module's A0 for one type (parameters_kp14.py:122-133)."""
    yg, Q = build_Qy(ny, ymax, drift_shift=drift_shift)
    c = RHO + b * GAMMA_V * SIGMA_Y + b * KAPPA_Y * yg - 0.5 * b ** 2 * SIGMA_Y ** 2
    return yg, np.linalg.solve(np.diag(c + theta_c) - Q, np.ones(ny))


def _var_y(t):
    return (SIGMA_Y ** 2 / (2 * KAPPA_Y)) * (1.0 - np.exp(-2 * KAPPA_Y * t))


def A_exact(b, y0):
    """E^Q[int_0^inf e^{-rho t} e^{b(y_t - y0)} dt]; Q-drift -kappa*y - gamma_v*sigma_y.

    The risk adjustment enters as b*gamma_v*sigma_y*(1-e^{-kappa t})/kappa and SATURATES.
    """
    def f(t):
        decay = 1.0 - np.exp(-KAPPA_Y * t)
        mean_shift = y0 * np.exp(-KAPPA_Y * t) - y0 - (GAMMA_V * SIGMA_Y / KAPPA_Y) * decay
        return np.exp(-RHO * t + b * mean_shift + 0.5 * b * b * _var_y(t))
    return quad(f, 0.0, 400.0, limit=400)[0]


def A_const_rate(b, y0):
    """What A_code computes: the premium as a constant rate b*gamma_v*sigma_y*t, physical drift."""
    def f(t):
        mean_shift = y0 * np.exp(-KAPPA_Y * t) - y0
        return np.exp(-(RHO + b * GAMMA_V * SIGMA_Y) * t + b * mean_shift + 0.5 * b * b * _var_y(t))
    return quad(f, 0.0, 400.0, limit=400)[0]


def A_exact_mc(b, y0, n=200_000, horizon=120.0, dt=0.01, seed=0):
    """Monte Carlo under Q. Euler, so expect O(dt) bias -- about 0.15% at dt=0.01."""
    rng = np.random.default_rng(seed)
    y = np.full(n, float(y0))
    acc = np.zeros(n)
    sq = np.sqrt(dt)
    for i in range(int(horizon / dt)):
        acc += np.exp(-RHO * (i * dt) + b * (y - y0)) * dt
        y += (-KAPPA_Y * y - GAMMA_V * SIGMA_Y) * dt + SIGMA_Y * sq * rng.standard_normal(n)
    return acc.mean(), acc.std() / np.sqrt(n)


def implied_lambda(b, y, ny=2001):
    """The price of y-risk the module's own A implies, if the Feynman-Kac equation must hold.

    Solves [rho + b*lam*sy + b*ky*y - .5 b^2 sy^2] A - L^P A + (lam*sy - sy^2 b) A' = 1 for lam.
    A single SDF requires ONE lam for every type. Returns (d log V/dy, lam).
    """
    yg, a = A_code(b, ny=ny)
    i = int(np.argmin(abs(yg - y)))
    h = yg[i + 1] - yg[i]
    Ap = (a[i + 1] - a[i - 1]) / (2 * h)
    App = (a[i + 1] - 2 * a[i] + a[i - 1]) / h ** 2
    A = a[i]
    LP = -KAPPA_Y * y * Ap + 0.5 * SIGMA_Y ** 2 * App
    num = 1.0 - ((RHO + b * KAPPA_Y * y - 0.5 * b * b * SIGMA_Y ** 2) * A - LP - SIGMA_Y ** 2 * b * Ap)
    den = b * SIGMA_Y * A + SIGMA_Y * Ap
    return b + Ap / A, num / den


def y_critical(b, gamma_v=GAMMA_V):
    """Where the type-f effective discount const_ty(f) hits zero.

    Below it the resolvent is ill posed -- the same failure mode that withdrew proposal K5,
    where rho_ty went negative for a loading of -0.06 and a negative discount removed the
    operator's dissipation (docs/RESULTS.md, "Open proposals"). It bounds how far the y grid
    may widen, and the CORRECTED solve needs a wide grid, so the two constraints meet.
    """
    return -(RHO + gamma_v * SIGMA_Y * b - 0.5 * SIGMA_Y ** 2 * b ** 2) / (KAPPA_Y * b)


def residual_of_exact(b, y, h=1e-3):
    """Plug the closed form into the Feynman-Kac equation. Returns 1.0 if the algebra is right."""
    A = A_exact(b, y)
    Ap = (A_exact(b, y + h) - A_exact(b, y - h)) / (2 * h)
    App = (A_exact(b, y + h) - 2 * A + A_exact(b, y - h)) / h ** 2
    return ((RHO + KAPPA_Y * y * b + GAMMA_V * SIGMA_Y * b - 0.5 * SIGMA_Y ** 2 * b ** 2) * A
            - (-KAPPA_Y * y - GAMMA_V * SIGMA_Y + SIGMA_Y ** 2 * b) * Ap
            - 0.5 * SIGMA_Y ** 2 * App)


def main():
    print(f"vyx: rho={RHO:.4f}  kappa_y={KAPPA_Y}  sigma_y={SIGMA_Y:.5f}  gamma_v={GAMMA_V}")
    print(f"risk-neutral long-run mean of y = -gamma_v*sigma_y/kappa_y = {-GAMMA_V*SIGMA_Y/KAPPA_Y:.3f}"
          f"   -- the grid is [{-Y_MAX}, {Y_MAX}] with REFLECTING boundaries\n")

    print("A(y): the module against the closed forms")
    print(f"{'b':>5} {'y':>5} {'A_code(21)':>11} {'A_code(2001)':>13} {'A_exact':>9} "
          f"{'code/exact':>11} {'code/const':>11}")
    for b in TYPE_BV:
        yg21, a21 = A_code(b)
        ygf, af = A_code(b, ny=2001)
        for y in (-2.0, 0.0, 2.0):
            c21, cf = np.interp(y, yg21, a21), np.interp(y, ygf, af)
            ex, cn = A_exact(b, y), A_const_rate(b, y)
            print(f"{b:5.2f} {y:5.1f} {c21:11.4f} {cf:13.4f} {ex:9.4f} "
                  f"{c21/ex:10.3f}x {cf/cn:10.5f}x")

    print("\nMonte Carlo check on A_exact (independent of the algebra), b=0.14, y=0")
    mc, se = A_exact_mc(0.14, 0.0)
    print(f"  MC = {mc:.4f} (se {se:.4f})   closed form = {A_exact(0.14, 0.0):.4f}   "
          f"gap {100*(mc/A_exact(0.14,0.0)-1):+.2f}% is Euler bias")

    print("\nImplied price of y-risk per type (a single SDF requires one value, gamma_v = 1.8)")
    print(f"{'b':>5} {'y':>5} {'d logV/dy':>10} {'implied lambda':>15} {'lambda/gamma_v':>15}")
    for b in TYPE_BV[1:]:
        for y in (-2.0, 0.0, 2.0):
            expo, lam = implied_lambda(b, y)
            print(f"{b:5.2f} {y:5.1f} {expo:10.4f} {lam:15.4f} {lam/GAMMA_V:15.2f}")

    print("\nIs the Feynman-Kac equation below the right one? Residual of the closed form:")
    for y in (-2.0, 0.0, 2.0):
        print(f"  y={y:+.1f}  residual = {residual_of_exact(0.14, y):.8f}   (1.0 means the algebra holds)")

    print("\nThe fix -- Girsanov drift in the generator, -gamma_v*sigma_y + sigma_y**2*b --")
    print("converges to the closed form, but only on a WIDER y grid (dy held at the current value):")
    shift = -GAMMA_V * SIGMA_Y + SIGMA_Y ** 2 * 0.14
    print(f"{'ymax':>6} {'ny':>6} " + "".join(f"{'A(y=' + f'{y:+.0f}' + ')':>12}" for y in (-2.0, 0.0, 2.0)))
    for ymax in (3.5, 5.0, 6.0, 7.0, 8.0):
        ny = int(2000 * ymax / Y_MAX) // 2 * 2 + 1
        yg, a = A_code(0.14, ny=ny, drift_shift=shift, ymax=ymax)
        print(f"{ymax:6.1f} {ny:6d} " + "".join(f"{np.interp(y, yg, a):12.4f}" for y in (-2.0, 0.0, 2.0)))
    print(f"{'exact':>6} {'':>6} " + "".join(f"{A_exact(0.14, y):12.4f}" for y in (-2.0, 0.0, 2.0)))
    print(f"\n  ymax must exceed about 7 for accuracy and stay below "
          f"{abs(y_critical(0.14)):.2f} for well-posedness (b=0.14, gamma_v={GAMMA_V});"
          f"\n  at gamma_v=2.5 the Q-mean is {-2.5*SIGMA_Y/KAPPA_Y:.2f} and the wall is "
          f"{abs(y_critical(0.14, 2.5)):.2f}, so the window narrows.")


if __name__ == "__main__":
    main()
