"""The variants carry private parameter copies. They must not drift from config.py.

History (2026-09-06). No file under variants/ imports config.py -- each variant hardcodes
its own parameters. Nothing compared the two trees, and they had drifted:

  gs_bx  delta     0.02/3   vs config 0.02      (superseded quarterly->monthly unit error)
  gs_bx  tau       0.2/3    vs config 0.2       (same)
  gs_bx  sigma_m   2.5      vs config 5         (took GS21.m:53's COMMENTED-OUT value --
                                                 the same class already caught for GS21_R
                                                 and documented at config.py:369)
  gs_bx  rho_x     0.95^1/3 vs config 0.96^1/3  (genuinely open; needs the paper)

The first three are settled: config.py is right. So gs21 and gs_bx were not one economy at
two settings, they were two different economies, and any cross-tree comparison of GS
results was confounded.

KP14 and BGN agree today. That is exactly why this test exists -- nothing was preventing
tomorrow's drift either.

Every whitelisted divergence carries a reason. An entry without one is not allowed, so a
future divergence cannot be quietly parked here.

Run: python tests/test_config_parity.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# variant attribute -> config attribute
KP14 = {"theta_eps": "KP14_THETA_EPS", "sigma_eps": "KP14_SIGMA_EPS",
        "alpha": "KP14_ALPHA", "delta": "KP14_DELTA", "mu_H": "KP14_MU_H",
        "mu_L": "KP14_MU_L", "lambda_H": "KP14_LAMBDA_H", "r": "KP14_R",
        "gamma_x": "KP14_GAMMA_X", "gamma_z": "KP14_GAMMA_Z",
        "theta_u": "KP14_THETA_U", "sigma_u": "KP14_SIGMA_U",
        "mu_x": "KP14_MU_X", "sigma_x": "KP14_SIGMA_X",
        "mu_z": "KP14_MU_Z", "sigma_z": "KP14_SIGMA_Z"}

# Deliberate divergences. Key -> reason. A bare name is rejected by the test itself.
WHITELIST = {
    ("kp_vy", "burnin"): "variant simulates its own panels and wants a longer burn-in "
                         "(400) than the main pipeline's KP14_BURNIN (200); affects no solve",
    ("bgn_gam", "burnin"): "same reason as kp_vy: the variant burns in 300 months "
                           "against the main pipeline's BGN_BURNIN of 200; simulation "
                           "only, enters no solve",
    ("gs_bx", "rho_x"): "OPEN, not deliberate: config says Table 1 is 0.96 quarterly, "
                        "GS21.m:22 uses 0.95. Needs the paper. Remove this entry once "
                        "settled -- it is a placeholder, not an exemption.",
    ("gs_bx", "sigma_x"): "downstream of rho_x; resolves with it",
    ("gs_bx", "xnum"): "grid size is a cost decision, not an economic parameter: the "
                       "variant runs 161 for accuracy, config 20 for speed",
}


def _cfg():
    import config
    return config


def _kp_variant():
    os.environ.setdefault("KP_PARAM_OVERRIDES", "{}")
    sys.path.insert(0, os.path.join(ROOT, "variants", "kp_vy"))
    sys.path.insert(0, os.path.join(ROOT, "variants"))
    import parameters_kp14
    return parameters_kp14


def _parse_assignments(src, names):
    """Module-level `name = <literal expr>`, including after a semicolon."""
    out = {}
    for line in src.splitlines():
        for stmt in line.split("#")[0].split(";"):
            stmt = stmt.strip()
            if "=" not in stmt or "==" in stmt:
                continue
            name, _, expr = stmt.partition("=")
            name = name.strip()
            if name not in names:
                continue
            try:
                out[name] = eval(expr.strip(), {"__builtins__": {}}, {})
            except Exception:
                pass
    return out


def _close(a, b, tol=1e-12):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


# ------------------------------------------------------------------ tests ----

def test_every_whitelist_entry_states_a_reason():
    for key, why in WHITELIST.items():
        assert isinstance(why, str) and len(why) > 30, \
            f"{key} is whitelisted without a real reason; that is how drift gets parked"


def test_kp14_variant_matches_config():
    C, V = _cfg(), _kp_variant()
    bad = []
    for vname, cname in KP14.items():
        if ("kp_vy", vname) in WHITELIST:
            continue
        v, c = getattr(V, vname, None), getattr(C, cname, None)
        if v is None or c is None:
            continue                      # name absent one side; not a divergence
        if not _close(v, c):
            bad.append(f"{vname}={v!r} vs config.{cname}={c!r}")
    assert not bad, "kp_vy has drifted from config.py:\n  " + "\n  ".join(bad)


def test_gs_variant_matches_config_on_the_settled_parameters():
    """delta, tau and sigma_m are settled: config.py is right. rho_x is not, and is
    whitelisted as an explicit placeholder rather than silently skipped."""
    C = _cfg()
    src = open(os.path.join(ROOT, "variants", "gs_bx", "gs_solve_reg.py")).read()
    ns = _parse_assignments(src, ("delta", "tau", "sigma_m"))
    checks = [("delta", "GS21_DELTA"), ("tau", "GS21_TAU"), ("sigma_m", "GS21_SIGMA_M")]

    # A parse that finds nothing must FAIL, not pass vacuously. The first version of this
    # test only matched names at the start of a line, so it silently checked nothing --
    # `g = 1.14; delta = 0.02` puts delta after a semicolon.
    missing = [v for v, _ in checks if v not in ns]
    assert not missing, (f"could not parse {missing} out of gs_solve_reg.py; the test "
                         f"would otherwise pass without checking anything")

    bad = []
    for vname, cname in checks:
        if not hasattr(C, cname):
            continue
        if not _close(ns[vname], getattr(C, cname)):
            bad.append(f"{vname}={ns[vname]!r} vs config.{cname}={getattr(C, cname)!r}")
    assert not bad, ("gs_bx still carries superseded GS21 parameters:\n  "
                     + "\n  ".join(bad)
                     + "\n(config.py is right on all three -- see "
                       "docs/refactor/FINDINGS-config-divergence.md)")


def test_rho_x_stays_flagged_until_the_paper_settles_it():
    """Guards against the placeholder quietly becoming a permanent exemption."""
    why = WHITELIST.get(("gs_bx", "rho_x"), "")
    assert "OPEN" in why and "paper" in why, \
        "rho_x's whitelist entry no longer says it is unresolved"


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
