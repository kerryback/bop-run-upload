"""Every tracked source file must at least parse. Twice now, one did not.

2026-09-06  variants/gs_bx/gs_solve_reg.py carried a duplicate `gmreg=` keyword
            argument from commit 3e0b6ae -- a hard SyntaxError. My verification
            probe at the time exec'd only the prefix above the broken line
            (`src.split('print(f"[solstamp] gs_bx')[0]`), so it was structurally
            incapable of catching it. The ASU session found it by running the file.

2026-09-07  variants/common/dkkm_functions.py had a note INSIDE a triple-quoted
            block-comment that quoted the fence characters literally, closing the
            fence three lines after it opened. Broken since 6cd949e (2026-09-04).
            run_estimators.py imports that module, so the entire estimator side of
            the pipeline could not start for three days and nothing reported it.

Both were invisible to the rest of the suite because no test imported or compiled
those files. This one compiles all of them, which is cheap and catches the whole
class. It does not execute anything: a module with import-time side effects (the
solve drivers) must not be run by the test suite.

Shell scripts get `bash -n` for the same reason -- run_seeds_slurm.sh and
run_gs_bx7_slurm.sh are only ever executed by SLURM, hours after submission, so a
syntax error in one costs a whole allocation before it is noticed.

Run: python tests/test_sources_parse.py
"""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "_scratch", "node_modules"}


def _tracked(ext):
    """Files git knows about, so untracked scratch never fails the suite."""
    out = subprocess.run(["git", "ls-files", f"*{ext}"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:                      # not a checkout: walk instead
        found = []
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            found += [os.path.relpath(os.path.join(base, f), ROOT)
                      for f in files if f.endswith(ext)]
        return sorted(found)
    return sorted(p for p in out.stdout.split("\n") if p.strip()
                  and not any(part in SKIP_DIRS for part in p.split("/")))


def test_every_tracked_python_file_compiles():
    broken = []
    files = _tracked(".py")
    assert len(files) > 30, f"only {len(files)} python files found -- listing is wrong"
    for rel in files:
        path = os.path.join(ROOT, rel)
        try:
            with open(path, "rb") as f:
                ast.parse(f.read(), filename=rel)
        except SyntaxError as e:
            broken.append(f"{rel}:{e.lineno}: {e.msg}")
        except (OSError, ValueError) as e:
            broken.append(f"{rel}: unreadable ({e})")
    assert not broken, ("tracked python files that do not parse:\n  "
                        + "\n  ".join(broken))
    print(f"    ({len(files)} python files compile)")


def test_every_tracked_shell_script_parses():
    broken = []
    files = _tracked(".sh")
    assert len(files) >= 4, f"only {len(files)} shell scripts found -- listing is wrong"
    for rel in files:
        path = os.path.join(ROOT, rel)
        with open(path) as f:
            shebang = f.readline()
        shell = "zsh" if "zsh" in shebang else "bash"
        out = subprocess.run([shell, "-n", path], capture_output=True, text=True)
        if out.returncode != 0:
            broken.append(f"{rel} ({shell} -n): {out.stderr.strip()[:200]}")
    assert not broken, ("tracked shell scripts that do not parse:\n  "
                        + "\n  ".join(broken))
    print(f"    ({len(files)} shell scripts parse)")


def test_the_two_known_regressions_stay_fixed():
    """Named, so a future edit that reintroduces either is unambiguous."""
    dk = os.path.join(ROOT, "variants", "common", "dkkm_functions.py")
    src = open(dk).read()
    fence = "'''"
    inside = src.split(fence)
    assert len(inside) == 3, (
        f"dkkm_functions.py has {len(inside) - 1} triple-quote fences, expected 2 -- "
        "a note inside the block-comment is quoting the fence characters again")

    gs = open(os.path.join(ROOT, "variants", "gs_bx", "gs_solve_reg.py")).read()
    tree = ast.parse(gs)
    dupes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kw = [k.arg for k in node.keywords if k.arg]
            if len(kw) != len(set(kw)):
                dupes.append(f"line {node.lineno}")
    assert not dupes, f"gs_solve_reg.py has duplicate keyword arguments at {dupes}"


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
