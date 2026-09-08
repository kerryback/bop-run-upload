"""gs_solve_gam.py is a deliberate copy of gs_solve_reg.py. Keep the copy honest.

WHY THE COPY EXISTS: solve_id digests the solver's own source, so any edit to
gs_solve_reg.py changes all five bx7 solve_ids -- five solves that cost ~5 h each on
Sol and were verified against their precommitment on 2026-09-07. Adding gamma(x) as a
switch inside that file would have invalidated them.

WHY THIS TEST EXISTS: duplication is the failure mode this repo keeps hitting -- the
config.py/variants drift that tests/test_config_parity.py guards, and the GS21.m vs
config.py disagreement that took a paper check to settle. A fork nobody diffs becomes
two economies that silently disagree. This asserts the two solvers differ ONLY in the
gamma block, its parameters, the docstring, the digested filename, and the npz keys
that record the new parameters.

Run with: python -m pytest tests/ -k parity   (or: python tests/test_gs_solver_parity.py)
"""
import difflib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSDIR = os.path.join(ROOT, "variants", "gs_bx")
REG = os.path.join(GSDIR, "gs_solve_reg.py")
GAM = os.path.join(GSDIR, "gs_solve_gam.py")


def body(path):
    """Source with the module docstring stripped -- the docstrings are meant to differ."""
    src = open(path).read()
    m = re.match(r'\s*"""', src)
    if m:
        end = src.index('"""', m.end())
        src = src[end + 3:]
    return src.splitlines()


def changed_blocks():
    a, b = body(REG), body(GAM)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag != "equal":
            out.append((tag, a[i1:i2], b[j1:j2]))
    return out


# Every line the copy is ALLOWED to add or change must match one of these.
ALLOWED = (
    "gs_gamma_slope", "gs_gamma_lo", "gs_gamma_hi", "gam_x", "sd_x", "gamma_x_grid",
    "gs_solve_gam.py", "gs_solve_reg.py", "g_cur", "gam_s", "gamma per regime",
    "gamma(x)", "state-dependent price of risk", "# ", "#", "",
)


def test_the_two_solvers_differ_only_in_the_gamma_block():
    offenders = []
    for tag, old, new in changed_blocks():
        for line in old + new:
            t = line.strip()
            if not t:
                continue
            if not any(k in line for k in ALLOWED):
                offenders.append(f"{tag}: {t[:88]}")
    assert not offenders, (
        "gs_solve_gam.py has drifted from gs_solve_reg.py outside the gamma block.\n"
        "Either port the change to both, or widen ALLOWED here deliberately:\n  "
        + "\n  ".join(offenders))


def test_the_calibration_constants_are_identical():
    """The Table I calibration must never differ between the two solvers."""
    names = ("g", "delta", "rho_x", "sigma_x", "rho_z", "sigma_z", "r", "gamma_x",
             "x_bar", "tau", "phi", "kappa_e", "kappa_b", "xi", "sigma_m")
    def consts(path):
        # the DOCSTRING is stripped first: gs_solve_gam.py's prose contains the phrase
        # "constant gamma_x = 0.5.", which the fallback regex below happily matched,
        # capturing the sentence period and reporting drift that did not exist.
        src = "\n".join(body(path))
        out = {}
        for n in names:
            m = re.search(rf"^{n} = ([^;\n]+)", src, re.M) or \
                re.search(rf"[;\s]{n} = ([^;\n]+)", src)
            if m:
                out[n] = m.group(1).strip()
        return out
    cr, cg = consts(REG), consts(GAM)
    assert cr, "parsed no constants from gs_solve_reg.py -- the regex is stale"
    diff = {k: (cr.get(k), cg.get(k)) for k in names if cr.get(k) != cg.get(k)}
    assert not diff, f"calibration drift between the two solvers: {diff}"


def test_slope_zero_is_the_documented_identity():
    """The copy must state, and structurally support, that slope 0 is a no-op."""
    src = open(GAM).read()
    assert "gs_gamma_slope = 0.0" in src, "the default slope must be 0"
    assert re.search(r"np\.clip\(gamma_x - gs_gamma_slope \* xgrid / sd_x,", src), \
        "the gamma(x) formula is not in the expected form"
    assert "sqrt(sigma_x ** 2 / (1 - rho_x ** 2))" in src, (
        "sd_x must be the STATIONARY sd; xgrid.std() is 2.32x larger and would give "
        "an economy with far less gamma variation than intended")


def test_the_new_params_are_in_the_hashed_namespace():
    """They must be module-level and above the override line, or solve_id ignores them."""
    src = open(GAM).read()
    ov = src.index('ov = json.loads(os.environ.get("GS_PARAM_OVERRIDES"')
    for n in ("gs_gamma_slope", "gs_gamma_lo", "gs_gamma_hi"):
        i = src.index(f"{n} = ")
        assert i < ov, (
            f"{n} is defined after the GS_PARAM_OVERRIDES line, so it cannot be "
            f"overridden and may not be hashed into solve_id")


def test_params_array_did_not_grow():
    """gs_sim_bx.py unpacks `params` positionally as a 15-tuple."""
    src = open(GAM).read()
    m = re.search(r"params=np\.array\(\[(.*?)\]\)", src, re.S)
    assert m, "no positional params array found"
    n = len([x for x in m.group(1).split(",") if x.strip()])
    assert n == 15, (
        f"the positional params array has {n} entries, not 15 -- gs_sim_bx.py:33 "
        f"unpacks it as a 15-tuple and every field after the insertion point would "
        f"be silently mis-assigned")


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
