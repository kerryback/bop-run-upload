"""solve_impact.py must say, before you pay, what a change costs and whether it had to.

Fixtures are three real commits, chosen because they are exactly the three cases:
  30bef7a  the registry rename's sed rewrote a print() string in TWO producers --
           non-functional, and it invalidated six solves (~30 h of cluster time)
  370294a  the kappa_e calibration correction -- functional, invalidation was correct
  413b3c1  a docs-only commit touching no producer -- must be SILENT
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants"))
import solve_impact  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "variants", "solve_impact.py"),
                        *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout


def test_the_incident_is_classified_non_functional_and_names_every_solve():
    rc, out = run("30bef7a^", "30bef7a")
    assert rc == 0
    assert "NON-FUNCTIONAL" in out
    assert "FUNCTIONAL\n" not in out.replace("NON-FUNCTIONAL", "")
    for sid in ("8b584c38614695ac", "63fa7ebbc2db49ea", "0818d7153d5708cc"):
        assert sid in out, f"{sid} should be listed as invalidated"
    assert "gs_solve_reg.py" in out and "gs_solve_gam.py" in out


def test_a_real_code_change_is_classified_functional():
    rc, out = run("370294a^", "370294a")
    assert rc == 0
    assert "change is FUNCTIONAL" in out
    assert "NON-FUNCTIONAL" not in out


def test_a_change_touching_no_producer_is_silent():
    rc, out = run("413b3c1^", "413b3c1")
    assert rc == 0 and out.strip() == "", f"expected silence, got: {out!r}"


def test_classify_distinguishes_comments_docstrings_and_prints_from_code():
    base = 'x = 1\ndef f(a):\n    """doc"""\n    return a + x\n'
    assert solve_impact.classify(base, base.replace('"""doc"""', '"""changed"""')).startswith("non-functional")
    assert solve_impact.classify(base, "# hello\n" + base).startswith("non-functional")
    assert solve_impact.classify(base, base + 'print("done")\n').startswith("non-functional")
    assert solve_impact.classify(base, base.replace("a + x", "a - x")) == "functional"
    # a string literal that is NOT a print is a parameter and must count as functional
    assert solve_impact.classify(base, base + 'name = "Jstar_g0235.csv"\n') == "functional"


def test_it_never_exits_nonzero_on_a_bad_ref():
    """Advisory only: a broken hook must never block a commit."""
    rc, _ = run("definitely-not-a-ref", "HEAD")
    assert rc == 0


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
