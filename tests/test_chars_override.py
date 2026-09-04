"""--chars must cross the process boundary.

main.py runs all eight workflow steps as separate subprocesses (main.py:180),
each of which re-imports config.py from disk. Before 2026-09-04 the --chars
override was applied by mutating config.MODEL_CHARS in main.py's own
interpreter, so it reached none of them: a `--chars umd` run produced
byte-identical output to a full-set run, with no warning, on the cluster too
(run_bop_job.sh:56 passes $CHARS_FLAG).

The override now travels as BOP_CHARS, the same mechanism BOP_SCRATCH_DIR
already uses successfully.

Run: python tests/test_chars_override.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as c  # noqa: E402


def _in_subprocess(model, bop_chars=None):
    """Ask a FRESH interpreter what chars it sees, exactly as a step script does."""
    env = dict(os.environ)
    env.pop("BOP_CHARS", None)
    if bop_chars is not None:
        env["BOP_CHARS"] = bop_chars
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import config;"
        "cfg = config.get_model_config(%r);"
        "print(','.join(cfg['chars']));"
        "print(','.join(config.MODEL_FACTOR_NAMES[%r]))" % (ROOT, model, model)
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().split("\n")
    return lines[0].split(","), lines[1].split(",")


def test_subprocess_sees_no_override_by_default():
    chars, _ = _in_subprocess("bgn")
    assert chars == c.CHARS_DEFAULT


def test_subprocess_sees_the_override():
    """THE regression test. This returned the full 5-char set before the fix."""
    chars, factors = _in_subprocess("bgn", "umd")
    assert chars == ["size", "mom"], chars
    assert factors == ["smb", "umd"], factors


def test_multiple_factors():
    chars, factors = _in_subprocess("bgn", "cma,umd")
    assert chars == ["size", "agr", "mom"], chars
    assert factors == ["smb", "cma", "umd"], factors


def test_gs21_override_replaces_the_six_char_default():
    chars, _ = _in_subprocess("gs21", "mkt_lev")
    assert chars == ["size", "mkt_lev"], chars


def test_size_is_always_first_and_present():
    for spec in ("umd", "cma,umd", "hml,cma,rmw,umd"):
        chars, _ = _in_subprocess("bgn", spec)
        assert chars[0] == "size", (spec, chars)


def test_smb_is_always_produced():
    for spec in ("umd", "cma,umd"):
        _, factors = _in_subprocess("bgn", spec)
        assert factors[0] == "smb", (spec, factors)


def test_all_models_get_the_override():
    """_apply_chars_env rewrites every model, so no model silently keeps defaults."""
    for model in ("bgn", "kp14", "gs21"):
        chars, _ = _in_subprocess(model, "umd")
        assert chars == ["size", "mom"], (model, chars)


def test_unknown_factor_is_rejected_loudly():
    env = dict(os.environ)
    env["BOP_CHARS"] = "bogus"
    out = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, %r); import config" % ROOT],
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0, "invalid BOP_CHARS did not fail"
    assert "unknown factor name" in out.stderr, out.stderr


def test_whitespace_and_case_are_tolerated():
    chars, _ = _in_subprocess("bgn", " UMD , cma ")
    assert chars == ["size", "mom", "agr"], chars


def test_char_to_factor_round_trips():
    """config.FACTOR_TO_CHAR is the single mapping _apply_chars_env uses.

    main.py and fama_functions.py still carry private copies; if they drift from
    this one, a factor will silently fail to name itself. Pin the contents here.
    """
    assert c.FACTOR_TO_CHAR == {
        "hml": "bm",
        "cma": "agr",
        "rmw": "roe",
        "umd": "mom",
        "mkt_lev": "mkt_lev",
    }
    for char, factor in c.CHAR_TO_FACTOR.items():
        assert c.FACTOR_TO_CHAR[factor] == char


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
