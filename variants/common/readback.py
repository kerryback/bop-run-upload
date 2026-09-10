"""Did every override TAKE EFFECT in the module that read it?

Every parameter module applies its `*_OVERRIDES` environment variable with
`globals().update(ov)`. That call is silent in two ways:

  * a name that the module RE-DERIVES after the override site is accepted and then
    overwritten -- `BGN_PARAM_OVERRIDES='{"prob_calm": 0.9}'` runs clean and leaves
    prob_calm at p10/(p01+p10);
  * a MISSPELLED name is created as a new global that nothing reads --
    `{"gmulr": [0.2, 3.5]}` runs clean at gmult [1, 1].

Either way the run records the override it was asked for, the spec check confirms the
environment matches the spec, and the economy that was actually simulated is a
different one. `parameters_kp14.py` guards this in-module and raises. `bgn_gam/parameters.py`
cannot get the same four lines without moving `be222462dd017b2c` (it is digest-bearing
for that solve, and all ten committed g0235 seeds were built from it), so the check
lives here and is applied from the run scripts AFTER the module has imported.

    from common import readback
    ok, lines = readback.overrides_took(sys.modules["parameters"],
                                        json.loads(os.environ.get("BGN_PARAM_OVERRIDES", "{}")))

Three-valued like the other verifiers: True every override took effect, False at least
one did not (or names nothing), None nothing was requested.

Misspellings are detected against the names the module's SOURCE assigns at module level,
parsed with `ast` -- not against `hasattr`, which is useless here because
`globals().update()` has already created whatever key it was handed.
"""
import ast
import json
import os

import numpy as np

# kp_vy renormalises type_share to sum 1 after the override; compare it proportionally.
NORMALISED = {"type_share"}

_MISSING = object()


class _ModuleScope(ast.NodeVisitor):
    """Collect names bound at module scope; do not descend into function/class bodies."""

    def __init__(self):
        self.names = set()

    def _target(self, t):
        if isinstance(t, ast.Name):
            self.names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._target(e)
        elif isinstance(t, ast.Starred):
            self._target(t.value)

    def visit_Assign(self, node):
        for t in node.targets:
            self._target(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._target(node.target)

    def visit_AnnAssign(self, node):
        self._target(node.target)

    def visit_NamedExpr(self, node):
        self._target(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        self._target(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars is not None:
                self._target(item.optional_vars)
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            self.names.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            if a.name != "*":
                self.names.add(a.asname or a.name)

    # scopes we must NOT enter: a name assigned inside a function is not a parameter
    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.names.add(node.name)

    def visit_Lambda(self, node):
        pass


def assigned_names_in_source(path):
    """Names bound at module scope in the Python file at `path`."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    v = _ModuleScope()
    v.visit(tree)
    return v.names


def assigned_names(module):
    return assigned_names_in_source(module.__file__)


def _took(name, requested, current):
    """Does the module's live value equal the requested one, after the coercions the
    modules themselves apply (list -> array; type_share renormalised)?"""
    if isinstance(requested, (str, bool)) or isinstance(current, (str, bool, np.bool_)):
        return bool(requested == current)
    try:
        a = np.asarray(requested, dtype=float).ravel()
        b = np.asarray(current, dtype=float).ravel()
    except (TypeError, ValueError):
        try:
            return bool(requested == current)
        except Exception:  # noqa: BLE001 -- an incomparable pair is a mismatch
            return False
    if a.shape != b.shape:
        return False
    if name in NORMALISED:
        s = a.sum()
        if s == 0:
            return False
        a = a / s
    return bool(np.allclose(a, b, rtol=1e-12, atol=0.0))


def overrides_took(module, overrides):
    """(ok, lines): did every key in `overrides` land in `module` with the requested value?"""
    if not overrides:
        return None, [f"[readback] no overrides requested for {module.__name__}; nothing to verify"]
    legit = assigned_names(module)
    lines, ok = [], True
    for k, v in overrides.items():
        if k not in legit:
            ok = False
            lines.append(f"[readback] {k}: NOT A PARAMETER of {module.__name__} -- "
                         f"globals().update() created it and nothing reads it (misspelled?)")
            continue
        cur = getattr(module, k, _MISSING)
        if cur is _MISSING:
            ok = False
            lines.append(f"[readback] {k}: assigned in {module.__name__}'s source but absent "
                         f"from the imported module")
            continue
        if _took(k, v, cur):
            lines.append(f"[readback] {k:22s} = {cur!r}  (override took effect)")
        else:
            ok = False
            lines.append(f"[readback] {k}: requested {v!r}, {module.__name__} has {cur!r} -- "
                         f"the override was DISCARDED (re-derived after the override site?)")
    return ok, lines


def overrides_took_from_env(module, env_var, env=None):
    """Convenience: read the JSON override blob from `env_var` and verify it."""
    if env is None:
        env = os.environ
    raw = env.get(env_var, "{}")
    try:
        ov = json.loads(raw or "{}")
    except ValueError:
        return False, [f"[readback] {env_var} is not valid JSON: {raw!r}"]
    return overrides_took(module, ov)
