"""One builder per prefix, and 'not signalable' must never be read as 'not running'.

History (2026-09-04). Twice in one session a second build_vy_tables.py was started
against a prefix already being built. Both wrote the same G_<prefix><f>.csv and each
halved the other's cores -- the survivor's CPU went 453% -> 915% the moment the
duplicate was killed. Worse than the waste: a concurrent writer can hand the integ
stage a half-written G table, which is silent numerical corruption, not a crash.

The first version of the lock was itself broken. os.kill(1, 0) as a normal user
raises PermissionError, which is an OSError -- the original `except (OSError,
ProcessLookupError): return False` therefore declared init "dead", took the lock,
and launched exactly the duplicate solve it existed to prevent. EPERM means the
process exists and is not ours to signal. Only ProcessLookupError means dead.

Run: python tests/test_build_lock.py
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KP = os.path.join(ROOT, "variants", "kp_vy")
DRIVER = os.path.join(KP, "build_vy_tables.py")
PREFIX = "locktest"
LOCK = os.path.join(KP, f".build_vy_tables.{PREFIX}.lock")
DUMMY = os.path.join(KP, f"G_{PREFIX}0.csv")
MARKER = os.path.join(KP, f"G_{PREFIX}0.solveid")


def _run():
    """KP_VY_ADOPT_G exits before the solve loop, so this can never start a solve."""
    return subprocess.run(
        [sys.executable, "-u", "-W", "ignore", DRIVER, PREFIX],
        cwd=KP, capture_output=True, text=True,
        env=dict(os.environ, KP_PARAM_OVERRIDES="{}", KP_VY_ADOPT_G="0"))


def _setlock(pid):
    with open(LOCK, "w") as fh:
        json.dump({"pid": pid, "prefix": PREFIX, "started": "x"}, fh)


def _cleanup():
    for p in (LOCK, DUMMY, MARKER):
        if os.path.exists(p):
            os.unlink(p)


# ------------------------------------------------------------- behaviour ----

def test_refuses_when_holder_is_alive_but_unsignalable():
    """pid 1 is always running and never ours to signal. This is the exact case
    that fooled the first implementation."""
    _cleanup()
    open(DUMMY, "w").write("a,b\n1,2\n")
    _setlock(1)
    r = _run()
    _cleanup()
    assert r.returncode != 0, "second builder was allowed to run"
    assert "another build of prefix" in r.stdout + r.stderr, r.stdout + r.stderr


def test_refuses_when_holder_is_a_live_process_we_own():
    _cleanup()
    open(DUMMY, "w").write("a,b\n1,2\n")
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        _setlock(holder.pid)
        r = _run()
    finally:
        holder.kill()
        holder.wait()
    _cleanup()
    assert r.returncode != 0, "second builder was allowed to run"
    assert str(holder.pid) in r.stdout + r.stderr


def test_takes_over_a_lock_held_by_a_dead_pid():
    """A killed build must not wedge the prefix forever."""
    _cleanup()
    open(DUMMY, "w").write("a,b\n1,2\n")
    _setlock(999999)
    r = _run()
    out = r.stdout + r.stderr
    ok = os.path.exists(MARKER)
    _cleanup()
    assert r.returncode == 0, out
    assert "taking over" in out, out
    assert ok, "adopt did not run after taking the lock"


def test_lock_is_released_on_exit():
    _cleanup()
    open(DUMMY, "w").write("a,b\n1,2\n")
    r = _run()
    left = os.path.exists(LOCK)
    _cleanup()
    assert r.returncode == 0, r.stdout + r.stderr
    assert not left, "lock survived a clean exit; the next build would refuse forever"


# ---------------------------------------------------------------- source ----

def test_alive_does_not_treat_a_permission_error_as_dead():
    src = open(DRIVER).read()
    body = src.split("def _alive(")[1].split("\ndef ")[0]
    assert "except ProcessLookupError" in body, "dead is no longer detected precisely"
    assert "except (OSError, ProcessLookupError)" not in body, \
        "EPERM is being read as 'dead' again -- this launches duplicate solves"


def test_lock_is_taken_before_any_solve():
    """Anchored on the Popen/Parallel calls that actually launch work -- not on the
    string "kp14_fd_vy.py", which also appears in the module docstring above the
    lock and made an earlier version of this test fail on prose."""
    src = open(DRIVER).read()
    acquire = src.index("_acquire_lock()")
    for launcher in ("subprocess.Popen", "Parallel(n_jobs"):
        assert acquire < src.index(launcher), \
            f"the lock must be held before {launcher} can start work"


def test_lock_file_is_not_tracked():
    ignore = open(os.path.join(ROOT, ".gitignore")).read()
    assert ".build_vy_tables.*.lock" in ignore


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
    _cleanup()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
