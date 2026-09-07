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


def test_slurm_scripts_never_derive_their_path_from_BASH_SOURCE():
    """sbatch stages a COPY of the script, so ${BASH_SOURCE[0]} is not a repo path.

    2026-09-07, job 62740438: all five gs_bx tasks dead in 2-7 seconds at ~35 MB
    MaxRSS -- the signature of a shell failure, not a python one. The script derived
    its repo root from ${BASH_SOURCE[0]}, which under sbatch is
    /var/spool/slurmd/job.../slurm_script, cd'd there, and `mkdir -p ../results/logs`
    failed with a permission error. The ASU session found it; run_seeds_slurm.sh
    carried the identical idiom and would have lost a ten-task array the same way.

    The rule is about the CONSTRUCT, not about cd'ing: SLURM already starts a job in
    the directory sbatch was invoked from, so a script that never cd's is correct by
    doing nothing -- run_bop_job.sh is the example, and an earlier version of this
    test wrongly flagged it. What is unsafe is deriving a path from BASH_SOURCE. Any
    #SBATCH script that does so must prefer SLURM_SUBMIT_DIR and keep BASH_SOURCE
    only as the non-SLURM fallback.
    """
    at_risk, checked = [], 0
    for rel in _tracked(".sh"):
        src = open(os.path.join(ROOT, rel)).read()
        if "#SBATCH" not in src:
            continue
        checked += 1
        live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        if "BASH_SOURCE" not in live:
            continue                      # never derives a path: safe by construction
        if "${SLURM_SUBMIT_DIR:-" not in live:
            at_risk.append(f"{rel}: derives a path from BASH_SOURCE without preferring "
                           f"SLURM_SUBMIT_DIR -- will land in the spool directory")
        elif "exit 2" not in live:
            at_risk.append(f"{rel}: prefers SLURM_SUBMIT_DIR but does not fail loudly "
                           f"when it resolves outside the repo")
    assert checked >= 2, f"only {checked} #SBATCH scripts found -- the listing is wrong"
    assert not at_risk, ("SLURM scripts that will die in the spool directory:\n  "
                         + "\n  ".join(at_risk))
    print(f"    ({checked} SLURM scripts checked)")


def test_pandas_io_backends_are_declared():
    """pandas I/O backends are never imported, so no import scan can find them.

    2026-09-07: `pyarrow` was required by variants/run_oracle.py (to_parquet) and
    run_estimators.py (read_parquet) and declared in NEITHER environment.yml nor
    requirements.txt. It worked on this laptop only because the ambient anaconda env
    happened to carry it; on Sol's `bop` it was simply absent, which blocked the whole
    Phase 1 run side. An AST scan of imports finds nothing -- pandas resolves the engine
    at call time -- so the rule has to be written against the CALL, not the import.
    """
    want = {"to_parquet": ("pyarrow", "fastparquet"),
            "read_parquet": ("pyarrow", "fastparquet"),
            "to_excel": ("openpyxl", "xlsxwriter"),
            "read_excel": ("openpyxl", "xlrd")}
    declared = ""
    for fn in ("environment.yml", "requirements.txt"):
        declared += open(os.path.join(ROOT, fn)).read().lower()

    missing = []
    for rel in _tracked(".py"):
        if rel.startswith("tests/"):
            continue
        try:
            src = open(os.path.join(ROOT, rel)).read()
        except OSError:
            continue
        live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        for call, backends in want.items():
            if call + "(" not in live:
                continue
            if not any(b in declared for b in backends):
                missing.append(f"{rel} calls {call}() but none of "
                               f"{'/'.join(backends)} is declared in environment.yml "
                               f"or requirements.txt")
    assert not missing, ("undeclared pandas I/O backends:\n  " + "\n  ".join(missing))
    print("    (pandas I/O backends declared)")


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
