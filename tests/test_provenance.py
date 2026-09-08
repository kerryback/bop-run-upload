"""The sidecar must answer "what code and parameters made this file".

That is goal (b): see a good result, know what produced it. Nothing recorded it before
2026-09-08 -- solstamp content-addresses SOLVES (a cache key) and runstamp links runs to
solves, but neither records the code version, and the summary CSVs carried no link at
all.

These tests guard the properties that make the sidecar trustworthy rather than
decorative. A sidecar that is merely PRESENT but points at the wrong directory, or that
silently drops the overrides, is worse than none: it invites you to trust it.

Run with: python -m pytest tests/ -k provenance   (or: python tests/test_provenance.py)
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "variants", "common"))
import provenance  # noqa: E402


def test_env_capture_is_anchored_not_substring():
    """An unanchored '_PREFIX' captured HOMEBREW_PREFIX and CONDA_PREFIX.

    Noise is not harmless here: the point of the sidecar is that a human reads it, and
    burying KP_PARAM_OVERRIDES among a dozen shell variables defeats that.
    """
    keep = dict(os.environ)
    try:
        os.environ.update({"HOMEBREW_PREFIX": "/opt/homebrew",
                           "GSETTINGS_SCHEMA_DIR": "/x",
                           "CONDA_PREFIX": "/y",
                           "KP_PARAM_OVERRIDES": '{"a":1}',
                           "GS_BX_SOLDIRS": "sol_g28",
                           "OMP_NUM_THREADS": "6"})
        env = provenance.relevant_env()
        assert "KP_PARAM_OVERRIDES" in env
        assert "GS_BX_SOLDIRS" in env
        assert "OMP_NUM_THREADS" in env
        for junk in ("HOMEBREW_PREFIX", "GSETTINGS_SCHEMA_DIR", "CONDA_PREFIX"):
            assert junk not in env, f"{junk} is noise and must not be captured"
    finally:
        os.environ.clear(); os.environ.update(keep)


def test_run_from_is_the_script_dir_not_the_cwd():
    """run_oracle.py chdirs into variants/<model> partway through.

    A cwd captured at write time says 'cd variants/kp_vy' for a run launched from
    variants/ -- and run_oracle.py is not in kp_vy, so the reproduce line fails. This is
    a real bug that shipped and was caught by reading the output.
    """
    src = (
        "import sys, os, json\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'variants', 'common')!r})\n"
        "import provenance\n"
        "os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/sub')\n"
        "print(json.dumps(provenance.run_provenance()))\n"
    )
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "sub"))
        script = os.path.join(td, "runner.py")
        open(script, "w").write(src)
        out = subprocess.run([sys.executable, script], capture_output=True, text=True,
                             cwd=td)
        assert out.returncode == 0, out.stderr
        prov = json.loads(out.stdout.strip().splitlines()[-1])
    assert prov["cwd_at_write"].endswith("sub"), prov["cwd_at_write"]
    assert not prov["run_from"].endswith("sub"), (
        "run_from followed the chdir; it must be the main script's directory so that "
        f"argv[0] resolves against it. got {prov['run_from']!r}")


def test_sidecar_lands_next_to_its_summary_and_carries_the_essentials():
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "some_summary.csv")
        open(target, "w").write("a,b\n1,2\n")
        prov, tag = provenance.write_sidecar(
            target, inputs=[{"stage": "G", "solve_id": "deadbeef"}],
            extra={"engine": "test"})
        side = target + ".prov.json"
        assert os.path.exists(side), "sidecar must sit beside the summary"
        d = json.load(open(side))
    for key in ("git", "argv", "env", "when", "run_from", "how_to_reproduce",
                "environment"):
        assert key in d, f"sidecar is missing {key}"
    assert d["inputs"][0]["solve_id"] == "deadbeef"
    assert d["extra"]["engine"] == "test"
    assert tag.startswith("some_summary@"), tag


def test_it_records_this_repos_real_sha():
    g = provenance.git_state()
    assert g["sha"] and len(g["sha"]) == 40, f"expected a full sha, got {g['sha']!r}"
    real = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    assert g["sha"] == real


def test_a_dirty_tree_stores_its_own_diff():
    """A bare sha LIES about a dirty tree. Storing the diff is what closes that."""
    g = provenance.git_state()
    if not g.get("dirty"):
        return                                  # clean tree: nothing to assert
    assert "diff" in g, "a dirty tree must carry its diff inline"
    assert "status" in g, "a dirty tree must list which paths differ"


def test_untracked_code_is_flagged_not_swallowed():
    """An untracked .py is in neither the sha nor the diff -- the sidecar must say so."""
    g = provenance.git_state()
    if g.get("untracked_code"):
        assert "untracked_code_warning" in g, (
            "untracked code was listed without warning that `git checkout` will not "
            "restore it")


def test_provenance_never_raises_without_git():
    """A six-hour run must not die because provenance had a bad day."""
    with tempfile.TemporaryDirectory() as td:
        g = provenance.git_state(cwd=td)        # not a repo
        assert g["sha"] is None
        assert "note" in g


def test_both_runners_write_a_sidecar_and_a_prov_column():
    for rel, target in (("variants/run_oracle.py", 'write_sidecar'),
                        ("variants/run_estimators.py", 'write_sidecar')):
        src = open(os.path.join(ROOT, rel)).read()
        assert "import provenance" in src, f"{rel} does not import provenance"
        assert target in src, f"{rel} never calls provenance.{target}"
        assert '"prov"' in src or "'prov'" in src, (
            f"{rel} writes no prov column; a row copied out of the CSV would lose its "
            f"pointer back to the code")


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
