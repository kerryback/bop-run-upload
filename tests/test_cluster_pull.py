"""variants/cluster_pull.sh pulls the shared cluster checkout after cluster-generated results were
committed on the laptop. It deletes files, on a checkout every job reads its code from, so it is
exercised here end to end on throwaway repositories rather than trusted by reading.

The situation it exists for: the cluster holds UNTRACKED copies of files the incoming commit brings
in as tracked, and `git pull` refuses to overwrite them even when they are byte-identical. The
contract: clear them only if every one is exactly what origin carries, then fast-forward; any
difference, or a checkout that cannot fast-forward, aborts with nothing touched.

It also updates the tree it lives in, so a pull can rewrite the running script. The whole body is
one function called on the last line, so bash has parsed the file before the pull changes it;
test_a_pull_that_rewrites_the_script_does_not_disturb_the_run checks that.

Run with: python -m pytest tests/ -k cluster_pull
"""
import os
import re
import subprocess
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "variants", "cluster_pull.sh")
ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
           GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")


def git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, capture_output=True, text=True, check=check).stdout.strip()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def read(path):
    with open(path) as fh:
        return fh.read()


@pytest.fixture
def repos():
    """origin (bare), laptop and cluster clones, all at one commit that tracks the real script."""
    with tempfile.TemporaryDirectory() as d:
        origin, laptop, cluster = (os.path.join(d, n) for n in ("origin.git", "laptop", "cluster"))
        git(d, "init", "-q", "--bare", "-b", "main", origin)
        git(d, "clone", "-q", origin, laptop)
        git(laptop, "checkout", "-q", "-b", "main")
        write(os.path.join(laptop, "variants", "cluster_pull.sh"), read(SCRIPT))
        write(os.path.join(laptop, "code.py"), "x = 1\n")
        write(os.path.join(laptop, "variants", "results", "old.csv"), "old\n")
        git(laptop, "add", "-A"); git(laptop, "commit", "-q", "-m", "init"); git(laptop, "push", "-q", "origin", "main")
        git(d, "clone", "-q", "-b", "main", origin, cluster)
        yield origin, laptop, cluster


def laptop_commits(laptop, files, msg="results"):
    for rel, text in files.items():
        write(os.path.join(laptop, rel), text)
    git(laptop, "add", "-A"); git(laptop, "commit", "-q", "-m", msg); git(laptop, "push", "-q", "origin", "main")
    return git(laptop, "rev-parse", "HEAD")


def run_pull(cluster):
    return subprocess.run(["bash", os.path.join(cluster, "variants", "cluster_pull.sh")], cwd=cluster, env=ENV,
                          capture_output=True, text=True)


def test_identical_untracked_copies_are_cleared_and_the_checkout_fast_forwards(repos):
    _, laptop, cluster = repos
    files = {"variants/results/a.csv": "1,2\n", "variants/results/b.json": "{}\n"}
    for rel, text in files.items():                       # the job wrote these on the cluster
        write(os.path.join(cluster, rel), text)
    plain = subprocess.run(["git", "pull", "-q", "--ff-only"], cwd=cluster, env=ENV, capture_output=True, text=True)
    head = laptop_commits(laptop, files)
    plain = subprocess.run(["git", "pull", "-q", "--ff-only"], cwd=cluster, env=ENV, capture_output=True, text=True)
    assert plain.returncode != 0 and "would be overwritten" in plain.stderr, "the premise: plain git pull refuses"
    r = run_pull(cluster)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cleared 0 modified + 2 untracked" in r.stdout
    assert git(cluster, "rev-parse", "HEAD") == head
    assert git(cluster, "status", "--porcelain") == ""
    for rel, text in files.items():
        assert read(os.path.join(cluster, rel)) == text


def test_a_differing_copy_aborts_with_nothing_touched(repos):
    _, laptop, cluster = repos
    before = git(cluster, "rev-parse", "HEAD")
    write(os.path.join(cluster, "variants/results/a.csv"), "1,2\n")
    write(os.path.join(cluster, "variants/results/b.json"), "{\"cluster\": true}\n")   # not what was committed
    laptop_commits(laptop, {"variants/results/a.csv": "1,2\n", "variants/results/b.json": "{}\n"})
    r = run_pull(cluster)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "DIFFERS (untracked): variants/results/b.json" in r.stderr
    assert "variants/results/a.csv" not in r.stderr
    assert git(cluster, "rev-parse", "HEAD") == before
    assert read(os.path.join(cluster, "variants/results/a.csv")) == "1,2\n"     # the identical one is not cleared either
    assert read(os.path.join(cluster, "variants/results/b.json")) == "{\"cluster\": true}\n"


def test_a_modified_tracked_file_identical_to_the_incoming_one_is_restored_and_pulled(repos):
    _, laptop, cluster = repos
    write(os.path.join(cluster, "variants/results/old.csv"), "new\n")
    head = laptop_commits(laptop, {"variants/results/old.csv": "new\n"})
    r = run_pull(cluster)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cleared 1 modified + 0 untracked" in r.stdout
    assert git(cluster, "rev-parse", "HEAD") == head and git(cluster, "status", "--porcelain") == ""


def test_a_checkout_that_cannot_fast_forward_aborts_with_nothing_touched(repos):
    _, laptop, cluster = repos
    write(os.path.join(cluster, "local.txt"), "committed on the cluster, against the convention\n")
    git(cluster, "add", "local.txt"); git(cluster, "commit", "-q", "-m", "cluster-side commit")
    before = git(cluster, "rev-parse", "HEAD")
    write(os.path.join(cluster, "variants/results/a.csv"), "1\n")
    laptop_commits(laptop, {"variants/results/a.csv": "1\n"})
    r = run_pull(cluster)
    assert r.returncode == 4, r.stdout + r.stderr
    assert git(cluster, "rev-parse", "HEAD") == before
    assert read(os.path.join(cluster, "variants/results/a.csv")) == "1\n"


def test_up_to_date_is_a_clean_no_op(repos):
    _, _, cluster = repos
    head = git(cluster, "rev-parse", "HEAD")
    r = run_pull(cluster)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cleared 0 modified + 0 untracked" in r.stdout and git(cluster, "rev-parse", "HEAD") == head


def test_a_pull_that_rewrites_the_script_does_not_disturb_the_run(repos):
    """The incoming commit replaces the script with one that is far longer and full of commands that
    must never run. Were bash still reading the old file as it executed, its read offset would now
    land inside that text. With the body in `main` and `main "$@"; exit $?` on one line, the running
    process has parsed everything before the pull and executes none of it."""
    _, laptop, cluster = repos
    poison = "\n".join(["echo POISON-FROM-NEW-VERSION; touch POISONED"] * 400) + "\n"
    new = read(SCRIPT).replace("main() {", poison + "main() {", 1)
    write(os.path.join(cluster, "variants/results/a.csv"), "1\n")
    head = laptop_commits(laptop, {"variants/cluster_pull.sh": new, "variants/results/a.csv": "1\n"})
    r = run_pull(cluster)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "POISON" not in r.stdout + r.stderr
    assert not os.path.exists(os.path.join(cluster, "POISONED"))
    assert git(cluster, "rev-parse", "HEAD") == head
    assert read(os.path.join(cluster, "variants/cluster_pull.sh")) == new


def test_the_script_keeps_its_structural_safeguards():
    """The properties the functional tests rely on, stated so an edit that drops one fails by name."""
    src = read(SCRIPT)
    code = [l for l in src.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert code[-1].strip() == 'main "$@"; exit $?', "the call and the exit must share the last line"
    top_level = [l for l in code if not l.startswith((" ", "\t", "main() {", "}"))]
    assert top_level == ['main "$@"; exit $?'], f"statements outside main would run before bash has read the file: {top_level}"
    body = "\n".join(code)                                   # comments may mention `git pull`; code may not
    assert "--ff-only" in body, "the checkout must only ever fast-forward"
    assert not re.search(r"git pull(?!.*--ff-only)", body), "a pull without --ff-only could create a merge commit"
    assert "/tmp/cp." not in body and "mktemp -d" in body, "scratch files must be private, not fixed names under /tmp"
