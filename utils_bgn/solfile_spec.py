"""
Which config parameters and producers BGN's solution file depends on.

BGN was the last of the three models with no provenance guard at all. Its file
was not wrong -- Jstar.csv reproduces from current config.py to 1.3e-13, checked
at five grid points spanning the full range -- but nothing would have caught a
future drift, which is exactly the state KP14 was in before it shipped a
corrupted integ_results.npz for two months.

Two producers matter here and both are stamped:

  * vasicek.py holds the bond-pricing recursion, the beta-distribution fit and
    Jstar itself, and is where the `from config import ...` lives. It is also
    imported by the consumers (`from .vasicek import *`), so it is not a pure
    producer -- but a change to it invalidates Jstar.csv all the same.
  * make_jstar.py drives the adaptive grid refinement and writes the file. Its
    source digest is folded into the parameter dict, so editing the grid
    construction invalidates the solfile the same way editing a parameter does.

The parameter list is derived from the producers' own import statements via AST,
so the imports ARE the specification and cannot drift from the code.

KNOWN, ACCEPTED BIAS: vasicek.py truncates its maturity sum at k <= 400 and its
option-maturity sum at s <= 950, biasing Jstar DOWN by ~0.2% (and unevenly in r:
0.072% at r = -0.016 vs 0.124% at r = +0.029). Measured 2026-08-31; see
make_jstar.py. This is a property of the shipped numbers, not a staleness
condition, so the guard does not and should not fire on it.
"""

import ast
import os

import config
from utils.solfile_stamp import digest

_HERE = os.path.dirname(os.path.abspath(__file__))
SOLFILES_DIR = os.path.join(_HERE, 'BGN_solfiles')

PRODUCER = 'make_jstar.py'
PARAM_SOURCES = ['vasicek.py', 'make_jstar.py']
DEPENDS_ON = ['vasicek.py']          # imported by the producer


def _config_imports(sources=PARAM_SOURCES):
    """Union of the config names the producers import, with current values."""
    names = set()
    for src in sources:
        with open(os.path.join(_HERE, src)) as f:
            tree = ast.parse(f.read())
        names |= {a.name for n in ast.walk(tree)
                  if isinstance(n, ast.ImportFrom) and n.module == 'config'
                  for a in n.names}
    if not names:
        raise AssertionError(f'{sources} import no config names; '
                             f'the spec would stamp nothing')
    missing = [n for n in sorted(names) if not hasattr(config, n)]
    if missing:
        raise AttributeError(f'{sources} import names absent from config.py: {missing}')
    params = {n: getattr(config, n) for n in sorted(names)}
    for dep in DEPENDS_ON:
        params[f'__{dep}__'] = digest(os.path.join(_HERE, dep))
    return params


def spec():
    """{filename: {'producer', 'params', 'inputs'}} for utils/solfile_stamp.py."""
    return {
        'Jstar.csv': {
            'producer': PRODUCER,
            'params': _config_imports(),
            'inputs': [],
        },
    }


KNOWN_ISSUE = """\
To regenerate and re-stamp, from the repo root:
    python utils_bgn/regen_solfiles.py --n-jobs 2
This re-solves 201 grid points at ~19 s each (~32 min at n_jobs=2). If you only
need to record provenance for files you already trust, use --stamp-only.
Note the regenerated r grid differs from the committed one by ~1e-5: the
committed endpoints came from an unseeded 1e8-draw simulation, and make_jstar.py
uses the analytic quantiles instead. Background: kp14_crash_20260826.md."""


def verify(mode='error'):
    """One-line entry point for the consumers."""
    from utils.solfile_stamp import verify as _verify
    return _verify('BGN', SOLFILES_DIR, spec(), mode=mode,
                   known_issue=KNOWN_ISSUE, producer_dir=_HERE)
