"""
Provenance stamping and staleness detection for model solution files.

Solution files -- KP14's `G_func.csv` and `integ_results.npz`, GS21's grids and
policy cuts -- are committed binaries produced offline by a separate script from
the parameters in `config.py`. Nothing in the pipeline has ever checked that a
committed solfile was actually produced by the current code and the current
parameters. All three ways that can go wrong have happened:

  1. `config.py` moved, the solfile did not.
     (KP14 `GAMMA_X` 1.38 -> 0.69 on 2026-06-18; undetected for two months.)
  2. The producer's *code* moved, the solfile did not.
  3. An upstream solfile moved, a downstream one did not.
     (`integ_kp14.py` consumes `G_func.csv`, so this ordering is load-bearing.)

A stamp records, for each solution file: the sha256 of the file itself, the
config parameters it was built from, the sha256 of its producer's source, and
the sha256 of any upstream solfile it consumed. `verify()` recomputes all four
and reports every disagreement, so each mode above surfaces as its own message
rather than a single vague "mismatch".

Phase 1 (current state)
    No `stamp.json` exists yet, so `verify()` cannot pass -- and that is the
    point. The absence of a stamp *is* the finding: these files' provenance is
    unrecorded. `mode='warn'` puts that in every log without blocking work that
    is still useful (confirming the pipeline runs end to end after the
    2026-08-26 crash fix).

Phase 2 (lands with the solfile regeneration)
    The regeneration driver writes `stamp.json`, `verify()` can pass, and the
    call sites flip to `mode='error'`. The repo goes straight from *unstamped*
    to *stamped and valid* and never passes through *stamped and failing* --
    which is the state that would invite a permanent skip-the-check flag.

See `kp14_crash_20260826.md` for the diagnosis this exists to prevent.
"""

import hashlib
import json
import os
import textwrap

STAMP_FILENAME = 'stamp.json'

# One banner per (label, directory) per process. With joblib's fork backend the
# parent emits it once before forking; under spawn each worker emits it once.
# Either way it is bounded and greppable.
_ANNOUNCED = set()


class SolfileStaleError(RuntimeError):
    """A solution file does not match the code or parameters that claim to produce it."""


def digest(path):
    """sha256 of a file, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def short(d):
    return 'missing' if d is None else d[:12]


def read_stamp(solfiles_dir):
    """Load stamp.json. Returns (stamp_dict, error_message). Both None-safe."""
    path = os.path.join(solfiles_dir, STAMP_FILENAME)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            return json.load(f), None
    except (json.JSONDecodeError, OSError) as e:
        return None, f'{STAMP_FILENAME} exists but could not be read: {e}'


def check(solfiles_dir, spec, producer_dir=None):
    """Compare the solfiles on disk against stamp.json.

    spec: {filename: {'producer': str|None,
                      'params':   {config_name: value},
                      'inputs':   [filename, ...]}}
          Values in 'params' are the *current* config values.

    Returns (problems, digests):
        problems -- list of human-readable strings, empty if everything agrees
        digests  -- {filename: sha256 hex or None}
    """
    if producer_dir is None:
        producer_dir = solfiles_dir
    problems = []
    digests = {name: digest(os.path.join(solfiles_dir, name)) for name in spec}

    for name, d in digests.items():
        if d is None:
            problems.append(f'{name}: file is missing from {solfiles_dir}')

    stamp, err = read_stamp(solfiles_dir)
    if err:
        problems.append(err)
        return problems, digests
    if stamp is None:
        problems.append(
            f'no {STAMP_FILENAME} in this directory, so these files cannot be '
            f'checked against config.py or against their producers -- their '
            f'provenance is unrecorded')
        return problems, digests

    recorded = stamp.get('solfiles', {})
    for name, want in spec.items():
        got = recorded.get(name)
        if got is None:
            problems.append(f'{name}: not listed in {STAMP_FILENAME}')
            continue

        # mode 0 -- is the stamp even about these bytes?
        if digests[name] is not None and got.get('sha256') != digests[name]:
            problems.append(
                f'{name}: file has changed since it was stamped '
                f'(stamp {short(got.get("sha256"))}, file {short(digests[name])})')

        # mode 1 -- config.py moved, solfile did not
        stamped_params = got.get('params', {})
        for pname, pval in sorted(want['params'].items()):
            if pname not in stamped_params:
                problems.append(f'{name}: {pname} was not stamped')
            elif stamped_params[pname] != repr(pval):
                problems.append(
                    f'{name}: built with {pname}={stamped_params[pname]}, '
                    f'config.py now has {pname}={pval!r}')
        for pname in sorted(set(stamped_params) - set(want['params'])):
            problems.append(f'{name}: stamped {pname} is no longer a declared dependency')

        # mode 2 -- producer source moved, solfile did not
        producer = want.get('producer')
        if producer:
            live = digest(os.path.join(producer_dir, producer))
            if live is None:
                problems.append(f'{name}: producer {producer} not found in {producer_dir}')
            elif got.get('producer_sha256') != live:
                problems.append(
                    f'{name}: {producer} has changed since this file was generated '
                    f'(stamp {short(got.get("producer_sha256"))}, source {short(live)})')

        # mode 3 -- upstream solfile moved, downstream did not
        stamped_inputs = got.get('inputs', {})
        for inp in want.get('inputs', []):
            if inp not in stamped_inputs:
                problems.append(f'{name}: upstream {inp} was not stamped')
            elif digests.get(inp) is not None and stamped_inputs[inp] != digests[inp]:
                problems.append(
                    f'{name}: was built from an older {inp} '
                    f'(stamp {short(stamped_inputs[inp])}, file {short(digests[inp])})')

    return problems, digests


def banner(label, solfiles_dir, spec, problems, digests, known_issue=None):
    """Render the report. Deterministic: no timestamps, no per-call-site text,
    so the same text appears wherever verify() is called and is greppable."""
    width = 74
    head = 'PROVENANCE VERIFIED' if not problems else 'PROVENANCE UNVERIFIED'
    lines = ['=' * width, f'[SOLFILE STAMP] {label}: {head}']
    for name in spec:
        path = os.path.join(solfiles_dir, name)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        lines.append(f'  {name:<24s} sha256:{short(digests.get(name)):<12s} {size:>9,d} B')
    for p in problems:
        wrapped = textwrap.wrap(p, width=width - 4)
        lines.append(f'  - {wrapped[0]}')
        lines.extend(f'    {w}' for w in wrapped[1:])
    if problems and known_issue:
        lines.extend('  ' + l for l in known_issue.strip().splitlines())
    lines.append('=' * width)
    return '\n'.join(lines)


def verify(label, solfiles_dir, spec, mode='warn', known_issue=None,
           producer_dir=None, once=True):
    """Check solfile provenance and report.

    mode='warn'  -- print the banner, return the problem list (Phase 1)
    mode='error' -- print the banner, then raise SolfileStaleError (Phase 2)

    Returns the list of problems (empty means everything agrees).
    """
    key = (label, os.path.abspath(solfiles_dir))
    if once and key in _ANNOUNCED and mode != 'error':
        return []
    _ANNOUNCED.add(key)

    problems, digests = check(solfiles_dir, spec, producer_dir=producer_dir)
    print(banner(label, solfiles_dir, spec, problems, digests, known_issue), flush=True)
    if problems and mode == 'error':
        raise SolfileStaleError(
            f'{label} solution files failed provenance verification '
            f'({len(problems)} problem(s)); see the banner above')
    return problems
