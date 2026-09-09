"""Which cached solves does this change invalidate, and was the change functional?

WHY: solve_id digests a producer's whole source file. That is the right trade -- it
distinguishes "the solver changed" from "the repo changed" (207 commits, 6 touched
gs_solve_reg.py), and it stays byte-exact so three Pythons agree on every id. But it
cannot tell code from comments, so a docstring edit to gs_solve_reg.py silently
invalidates five solves costing ~5 h each. That happened on 2026-09-08: a rename's sed
rewrote one string inside a print() in gs_solve_gam.py and the precommitted id moved.

An AST-based digest was measured and rejected: 0 of 15 historical solver commits were
comment-only, and ast.dump differs across py 3.11 / 3.12 / 3.14, so it would have
broken the cross-machine agreement the byte-exact digest provides.

So this is ADVISORY, not a change to the id. It answers, BEFORE you pay: which solves
does this change invalidate, what did they cost, and is the change functional (AST
differs) or only comments / docstrings / diagnostics (AST same)? The AST comparison is
done locally, on one interpreter, so version skew does not matter here.

usage:
    python variants/solve_impact.py                 # staged changes vs HEAD
    python variants/solve_impact.py --worktree      # working tree vs HEAD
    python variants/solve_impact.py REV1 REV2       # any two commits
    python variants/solve_impact.py 30bef7a^ 30bef7a   # the incident that motivated this

Always exits 0. Silent when nothing cached is at stake.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "common"))
import solstamp  # noqa: E402


def git(args, text=True):
    r = subprocess.run(["git"] + args, cwd=ROOT, capture_output=True, text=text)
    return r.stdout if r.returncode == 0 else None


def strip_docstrings(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n.body = b[1:] or [ast.Pass()]
    return tree


class _DropDiagnostics(ast.NodeTransformer):
    """Remove bare print(...) and logging calls. They cannot change a saved array."""
    def visit_Expr(self, node):
        v = node.value
        if isinstance(v, ast.Call):
            f = v.func
            if isinstance(f, ast.Name) and f.id == "print":
                return None
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id in ("logging", "log", "logger"):
                return None
        return node


def code_shape(src):
    """The parts of a source file that can change its output."""
    t = strip_docstrings(ast.parse(src))
    t = _DropDiagnostics().visit(t)
    return ast.dump(t)


def classify(old_src, new_src):
    if old_src is None or new_src is None:
        return "added/removed"
    try:
        return "functional" if code_shape(old_src) != code_shape(new_src) else \
               "non-functional (comments, docstrings, whitespace or print/log only)"
    except SyntaxError:
        return "unparseable"


def dependents():
    """{repo-relative source path: [manifest, ...]} over live manifests."""
    out = {}
    for m in solstamp.iter_manifests():
        if m.get("retired"):
            continue
        for path in m.get("sources", {}):
            out.setdefault(path, []).append(m)
    return out


def changed_files(mode, a=None, b=None):
    if mode == "staged":
        names = git(["diff", "--cached", "--name-only"])
        get_old = lambda p: git(["show", f"HEAD:{p}"])
        get_new = lambda p: git(["show", f":{p}"])
    elif mode == "worktree":
        names = git(["diff", "--name-only", "HEAD"])
        get_old = lambda p: git(["show", f"HEAD:{p}"])
        get_new = lambda p: open(os.path.join(ROOT, p)).read() \
            if os.path.exists(os.path.join(ROOT, p)) else None
    else:
        names = git(["diff", "--name-only", a, b])
        get_old = lambda p: git(["show", f"{a}:{p}"])
        get_new = lambda p: git(["show", f"{b}:{p}"])
    return [n for n in (names or "").split() if n.endswith(".py")], get_old, get_new


def main(argv):
    if not argv:
        mode, a, b = "staged", None, None
    elif argv == ["--worktree"]:
        mode, a, b = "worktree", None, None
    elif len(argv) == 2:
        mode, a, b = "range", argv[0], argv[1]
    else:
        sys.exit(__doc__)

    files, get_old, get_new = changed_files(mode, a, b)
    deps = dependents()
    hit = [f for f in files if f in deps]
    if not hit:
        return 0                                   # nothing cached is at stake: silent

    print("solve_impact: this change touches producers with cached solves\n")
    total_bytes = total_sweeps = 0
    for f in hit:
        old_src, new_src = get_old(f), get_new(f)
        kind = classify(old_src, new_src)
        ms = deps[f]
        # A change can RESTORE a recorded digest rather than break one -- that is what
        # reverting a stray sed does. Say so, instead of reporting it as invalidation.
        new_digest = hashlib.sha256(new_src.encode()).hexdigest() if new_src else None
        restored = [m for m in ms if m["sources"].get(f) == new_digest]
        broken = [m for m in ms if m["sources"].get(f) != new_digest]
        nbytes = sum(m.get("total_bytes", 0) for m in broken)
        sweeps = sum((m.get("achieved") or {}).get("sweeps", 0) for m in broken)
        total_bytes += nbytes; total_sweeps += sweeps
        print(f"  {f}")
        print(f"    change is {kind.upper()}")
        if restored:
            print(f"    RESTORES the recorded digest for {len(restored)} solve(s) -- "
                  f"they become reusable again:")
            for m in restored:
                print(f"      {m['solve_id']}  {','.join(m.get('tags') or []) or m.get('model','?')}")
        if broken:
            print(f"    invalidates {len(broken)} solve(s), {nbytes/1e6:.0f} MB of artifacts"
                  + (f", {sweeps:,} solver sweeps" if sweeps else ""))
            for m in broken:
                print(f"      {m['solve_id']}  {','.join(m.get('tags') or []) or m.get('model','?')}")
            if kind.startswith("non-functional"):
                print("    -> the existing artifacts ARE what the new source would produce.")
                print("       Either keep this edit out of the producer, or accept the recompute.")
    if total_bytes:
        print(f"\n  total invalidated: {total_bytes/1e6:.0f} MB" + (f", {total_sweeps:,} sweeps" if total_sweeps else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
