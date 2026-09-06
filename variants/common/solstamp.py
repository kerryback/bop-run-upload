"""Provenance, content-addressing and a durable registry for variant solve artifacts.

WHY THIS EXISTS
---------------
Every variant economy has an expensive solve stage whose output is reused across
runs, and all three producers got staleness detection wrong in a different way:

  KP  `build_vy_tables.py` cached on a hand-written 5-key stamp
      {type_bv, gamma_v, kappa_y, bv_comp, rho_ty}. Measured, that key silently
      MISSES delta, theta_eps, theta_u, sigma_eps, lambda_H, mu_H and mu_L --
      all of which change the tables. The 2026-09-04 regime-label fix changed
      mu_H/mu_L, and nothing would have noticed.
  BGN `rebuild_jstar_gam.py` has no stamp at all. `jstar_gam_file` names the
      output, so the FILENAME is the only identity: reuse a name with different
      parameters, or edit vasicek.py, and nothing notices.
  GS  `gs_solve_reg.py` writes a 15-element `params` array that omits gmreg,
      gs_bx, gs_ashift, p01/p10 and every grid size, so a solution.npz cannot
      identify its own exposure type. `run_gs_bx7.sh:5` guards only with
      `[ -f sol_reg/solution.npz ]` -- parameter-blind.

`utils/solfile_stamp.py` already solves this correctly for the MAIN tree, by
AST-parsing a producer's import statements to derive its parameter list. That
approach does not transfer here: the variants use `from parameters import *`, so
the import statement enumerates nothing.

WHAT THIS DOES INSTEAD
----------------------
Snapshots the producer's ENTIRE effective parameter namespace after overrides are
applied, plus the source of every module it depends on. Capturing everything can
only ever over-trigger a rebuild, never under-trigger one -- and an unnecessary
rebuild costs compute, while a missed one costs wrong numbers. That is the right
side to err on.

From that snapshot it derives a `solve_id` (content hash). The solve_id is the
identity of a solve: same parameters + same code == same id == reusable artifacts.
Each solve gets a manifest in `experiments/solfiles/<solve_id>.json`, committed to
git, recording the parameters, the code digests, and every artifact with its size
and sha256.

Manifests are a few KB, so hundreds of experiments cost a few MB and the record of
"which solve produced this" survives forever -- even after the artifacts themselves
are purged from scratch. Artifacts under SMALL_ARTIFACT_BYTES are committed
alongside; larger ones (GS solutions are ~85 MB each) live off-repo under
$BOP_SOLFILES, with the manifest recording their hashes so staleness is still
detectable and a re-solve is still verifiable.

USAGE
-----
    from common import solstamp

    snap = solstamp.snapshot(params_module, sources=[__file__, 'parameters.py'])
    sid  = snap.solve_id

    hit = solstamp.lookup(sid)              # already solved with this exact spec?
    if hit and solstamp.artifacts_ok(hit):
        sys.exit(0)                         # reuse

    ... run the solve, writing artifacts ...

    solstamp.record(snap, artifacts=[...], model='kp_vy', tag='vyx')
"""

import hashlib
import json
import os
import sys
import types

SMALL_ARTIFACT_BYTES = 32 * 1024 * 1024   # <= this may be committed to git

_HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.dirname(_HERE)
REPO_DIR = os.path.dirname(VARIANTS_DIR)
REGISTRY_DIR = os.path.join(REPO_DIR, 'experiments', 'solfiles')


class SolveStaleError(RuntimeError):
    """Artifacts on disk do not match the parameters and code that claim to produce them."""


# ---------------------------------------------------------------- hashing ----

def file_digest(path):
    """sha256 of a file, or None if absent."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def short(d, n=12):
    return 'missing' if d is None else d[:n]


def _canon(value):
    """Canonical, exactly-round-tripping representation of a parameter value.

    Floats go through repr(), which round-trips in Python 3, so 0.1 and
    0.1000000000000001 do not collide. Large arrays are replaced by a hash of
    their bytes so the manifest stays small while remaining sensitive.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    # numpy, without importing it unless it is already loaded
    np = sys.modules.get('numpy')
    if np is not None:
        if isinstance(value, np.generic):
            return _canon(value.item())
        if isinstance(value, np.ndarray):
            if value.size <= 32:
                return {'__array__': [_canon(v) for v in value.ravel().tolist()],
                        'shape': list(value.shape)}
            h = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            return {'__array_sha256__': h, 'shape': list(value.shape),
                    'dtype': str(value.dtype)}
    return {'__repr__': repr(value)}


_SKIP_NAMES = {
    # module plumbing and things that are not parameters
    'os', 'sys', 'json', 'time', 'math', 'np', 'pd', 'sp', 'spla', 'linalg',
    '__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
    '__name__', '__package__', '__spec__', '__path__',
}


def param_namespace(module_or_dict, skip=()):
    """Every parameter-like name in a producer's namespace, canonicalised.

    Deliberately indiscriminate: functions, modules and classes are dropped, and
    everything else is kept. Over-capture triggers spurious rebuilds; under-capture
    produces wrong numbers. See the module docstring.
    """
    ns = (module_or_dict if isinstance(module_or_dict, dict)
          else vars(module_or_dict))
    skip = set(skip) | _SKIP_NAMES
    out = {}
    for name, value in ns.items():
        if name in skip or name.startswith('__'):
            continue
        if isinstance(value, (types.ModuleType, types.FunctionType,
                              types.BuiltinFunctionType, type)):
            continue
        if callable(value):
            continue
        try:
            out[name] = _canon(value)
        except Exception:
            out[name] = {'__unhashable__': type(value).__name__}
    return out


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)


# --------------------------------------------------------------- snapshot ----

class Snapshot(object):
    """The identity of one solve: its parameters, its code, and the hash of both."""

    def __init__(self, params, sources, model=None, extra=None, env_params=None,
                 inputs=None, stage=None):
        self.params = params
        self.sources = sources            # {relpath: sha256}
        self.model = model
        self.stage = stage
        self.extra = extra or {}          # descriptive ONLY -- never hashed
        self.env_params = env_params or {}  # settings from outside the module -- hashed
        self.inputs = inputs or {}        # upstream artifacts consumed -- hashed
        payload = _canonical_json({'params': params, 'sources': sources,
                                   'env_params': self.env_params,
                                   'inputs': self.inputs, 'stage': stage})
        self.solve_id = hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self):
        return {'solve_id': self.solve_id, 'model': self.model, 'stage': self.stage,
                'params': self.params, 'sources': self.sources,
                'env_params': self.env_params, 'inputs': self.inputs,
                'extra': self.extra}


def artifact_digests(paths):
    """{repo-relative path: sha256} for a set of artifacts, for use as stage inputs."""
    out = {}
    for path in paths:
        ap = os.path.abspath(path)
        rel = os.path.relpath(ap, REPO_DIR) if ap.startswith(REPO_DIR + os.sep) else ap
        out[rel] = file_digest(ap)
    return out


def snapshot(params_module, sources, model=None, skip=(), extra=None, env_params=None,
             inputs=None, stage=None):
    """Build a Snapshot from a producer's parameter namespace and its source files.

    sources: paths to every .py whose contents can change the output -- the
             producer itself, its parameter module, and any local helper it
             imports. Paths are recorded relative to the repo root.

    env_params: settings that change the OUTPUT but do not live in the parameter
             module -- environment variables and CLI arguments, e.g. BGN's
             JSTAR_TOL. These ARE part of the solve_id.

    extra:   descriptive labels that do NOT change the output -- an output
             directory or filename prefix. These are recorded but NOT hashed, so
             the same solve written to two places keeps one identity.

    skip:    names to drop from the parameter namespace. Use for path and CLI
             plumbing that would otherwise make the id machine- or
             directory-dependent (`outdir`, `HERE`, ...).

    inputs:  digests of UPSTREAM artifacts this stage consumes (see
             artifact_digests). Hashing them is what makes a multi-stage solve
             work: a cheap downstream stage can be re-run without invalidating an
             expensive upstream one, while a change upstream still invalidates
             everything downstream. This is `utils/solfile_stamp.py`'s "mode 3:
             upstream moved, downstream did not".

    stage:   name of this stage when a producer has more than one, so the two get
             distinct ids even if everything else coincides.
    """
    src = {}
    for path in sources:
        ap = os.path.abspath(path)
        rel = os.path.relpath(ap, REPO_DIR)
        src[rel] = file_digest(ap)
    return Snapshot(param_namespace(params_module, skip=skip), src,
                    model=model, extra=extra, inputs=inputs, stage=stage,
                    env_params={k: _canon(v) for k, v in sorted((env_params or {}).items())})


# --------------------------------------------------------------- registry ----

def manifest_path(solve_id):
    return os.path.join(REGISTRY_DIR, solve_id + '.json')


def lookup(solve_id):
    """Return the manifest for this solve_id, or None if never recorded."""
    p = manifest_path(solve_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def artifacts_ok(manifest, base_dir=None):
    """True when every artifact the manifest lists is present with the right hash."""
    return not artifact_problems(manifest, base_dir=base_dir)


def artifact_problems(manifest, base_dir=None):
    """Human-readable reasons this manifest's artifacts cannot be reused."""
    problems = []
    for art in manifest.get('artifacts', []):
        path = art['path']
        ap = path if os.path.isabs(path) else os.path.join(base_dir or REPO_DIR, path)
        if not os.path.exists(ap):
            problems.append(f"{path}: missing (recorded {art['bytes']:,} B, "
                            f"sha256 {short(art['sha256'])})")
            continue
        d = file_digest(ap)
        if d != art['sha256']:
            problems.append(f"{path}: content changed since it was recorded "
                            f"(manifest {short(art['sha256'])}, file {short(d)})")
    return problems


def record(snap, artifacts, tag=None, spec_id=None, base_dir=None, note=None,
           achieved=None):
    """Write (or update) the manifest for this solve.

    artifacts: list of paths just produced. Recorded relative to the repo root
               when they live inside it, absolute otherwise (off-repo stores).
    achieved:  what the solve actually DID -- exit path, iterations, residual.
               Recorded, never hashed. A solve_id must depend only on inputs, or
               two runs of the same code on the same parameters would land in
               different registry slots; but a manifest that records only what was
               *requested* cannot distinguish a converged solve from one that hit
               its iteration cap. GS21's sol_reg exposed this on 2026-09-05: the
               manifest recorded tol=1e-6, the code enforced tol*20 = 2e-5, the
               solve exited by cycle-averaging at the 5600-sweep cap having reached
               3.4e-5, and printed "converged". It was 1.7x over -- fine in that
               instance, which is what makes it dangerous, since the identical path
               writes the identical manifest at 1000x over.
    """
    base_dir = base_dir or REPO_DIR
    os.makedirs(REGISTRY_DIR, exist_ok=True)

    entries, total = [], 0
    for path in sorted(artifacts):
        ap = os.path.abspath(path)
        rel = os.path.relpath(ap, REPO_DIR) if ap.startswith(REPO_DIR + os.sep) else ap
        size = os.path.getsize(ap) if os.path.exists(ap) else 0
        total += size
        entries.append({'path': rel, 'bytes': size, 'sha256': file_digest(ap)})

    existing = lookup(snap.solve_id) or {}
    tags = sorted(set(existing.get('tags', [])) | ({tag} if tag else set()))
    specs = sorted(set(existing.get('spec_ids', [])) | ({spec_id} if spec_id else set()))

    manifest = dict(snap.as_dict())
    manifest.update({
        'artifacts': entries,
        'total_bytes': total,
        'committable': total <= SMALL_ARTIFACT_BYTES,
        'tags': tags,
        'spec_ids': specs,
        'note': note or ('Written by variants/common/solstamp.py. The solve_id is '
                         'sha256(params + source digests); identical id means the '
                         'artifacts are reusable. Do not hand-edit.'),
    })
    if achieved is not None:
        manifest['achieved'] = achieved      # recorded, deliberately NOT hashed
    with open(manifest_path(snap.solve_id), 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    return manifest


def diff_params(old, new):
    """Which parameters differ between two snapshots' param dicts."""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, '<absent>'), new.get(k, '<absent>')
        if a != b:
            out.append((k, a, b))
    return out


# ------------------------------------------------------------------ gate -----

def ensure(snap, artifacts, label, mode='error', base_dir=None):
    """Verify that `artifacts` were produced by exactly `snap`. Consumer-side gate.

    Returns the problem list (empty means everything agrees). With mode='error'
    a non-empty list raises, so a stale solve stops the run instead of quietly
    producing wrong numbers.
    """
    problems = []
    man = lookup(snap.solve_id)
    if man is None:
        problems.append(
            f'no manifest for solve_id {snap.solve_id}: these artifacts were '
            f'built by an unrecorded parameter/code combination, so nothing can '
            f'confirm they match the current settings')
        prior = _find_by_artifacts(artifacts)
        if prior:
            problems.append(
                f'the artifacts on disk match manifest {prior["solve_id"]}; '
                f'differing parameters: '
                + ', '.join(f'{k} {a!r} -> {b!r}'
                            for k, a, b in diff_params(prior['params'], snap.params)[:8]))
    else:
        problems.extend(artifact_problems(man, base_dir=base_dir))

    if problems:
        width = 74
        print('=' * width)
        print(f'[SOLVE STAMP] {label}: STALE OR UNVERIFIED  (solve_id {snap.solve_id})')
        for p in problems:
            print(f'  - {p}')
        print('=' * width, flush=True)
        if mode == 'error':
            raise SolveStaleError(
                f'{label}: solve artifacts failed verification '
                f'({len(problems)} problem(s)); see above')
    return problems


def _find_by_artifacts(artifacts):
    """Which recorded manifest, if any, describes the bytes currently on disk."""
    want = {}
    for path in artifacts:
        ap = os.path.abspath(path)
        rel = os.path.relpath(ap, REPO_DIR) if ap.startswith(REPO_DIR + os.sep) else ap
        want[rel] = file_digest(ap)
    for man in iter_manifests():
        got = {a['path']: a['sha256'] for a in man.get('artifacts', [])}
        if got and all(got.get(k) == v for k, v in want.items() if v is not None):
            return man
    return None


def iter_manifests():
    if not os.path.isdir(REGISTRY_DIR):
        return
    for fn in sorted(os.listdir(REGISTRY_DIR)):
        if fn.endswith('.json'):
            man = lookup(fn[:-5])
            if man:
                yield man
