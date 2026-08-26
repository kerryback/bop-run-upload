"""
Which config parameters and producers each KP14 solution file depends on.

The parameter list is *derived from the producer's own import statement* rather
than maintained by hand: `_config_imports()` parses `from config import ...` out
of the producer source. So the import statement is the specification, and the
spec cannot drift from the code the way a duplicated list would.

Note that the derived parameters (`KP14_RHO`, `KP14_C`, `KP14_A_0..3`) cover the
transitive closure of the raw ones. `KP14_GAMMA_X` is not imported by either
producer, but it feeds `KP14_CONST` -> `KP14_A_*` and `KP14_RHO`, so stamping the
derived values is what catches the 2026-06-18 `GAMMA_X` drift.

Dependency order is load-bearing: `integ_kp14.py` reads `G_func.csv`, so
`G_func.csv` must be regenerated first and `integ_results.npz` records which
version of it was consumed.
"""

import ast
import os

import config

_HERE = os.path.dirname(os.path.abspath(__file__))
SOLFILES_DIR = os.path.join(_HERE, 'KP14_solfiles')


def _config_imports(producer):
    """The config names a producer imports, mapped to their current values.

    Parsing rather than hardcoding means adding a parameter to a producer
    automatically extends what gets stamped and checked.
    """
    with open(os.path.join(_HERE, producer)) as f:
        tree = ast.parse(f.read())
    names = [a.name for n in ast.walk(tree)
             if isinstance(n, ast.ImportFrom) and n.module == 'config'
             for a in n.names]
    missing = [n for n in names if not hasattr(config, n)]
    if missing:
        raise AttributeError(f'{producer} imports names absent from config.py: {missing}')
    return {n: getattr(config, n) for n in names}


def spec():
    """{filename: {'producer', 'params', 'inputs'}} for utils/solfile_stamp.py."""
    return {
        'G_func.csv': {
            'producer': 'kp14_fd.py',
            'params': _config_imports('kp14_fd.py'),
            'inputs': [],
        },
        'integ_results.npz': {
            'producer': 'integ_kp14.py',
            'params': _config_imports('integ_kp14.py'),
            'inputs': ['G_func.csv'],
        },
    }


# Stated rather than inferred, so a reader of the log does not have to work out
# what the unverified provenance actually implies today. Remove once the files
# are regenerated and stamped.
KNOWN_ISSUE = """\
Known stale as of 2026-08-26, independently of the missing stamp:
  * integ_results.npz encodes KP14_GAMMA_X=1.38; config.py has 0.69.
  * Its quadrature returns exact 0 where ~98% of firm-months live, so
    erets, ER, cond_var, max_sr and the returned risk premia are all void.
  * G_func.csv is on the same stale KP14_GAMMA_X.
Realized returns and the FF/FM/DKKM panel are unaffected. Do not use any
KP14 moment computed from these files. See kp14_crash_20260826.md."""


def verify(mode='warn'):
    """One-line entry point for the consumers. Phase 1 default is 'warn'."""
    from utils.solfile_stamp import verify as _verify
    return _verify('KP14', SOLFILES_DIR, spec(), mode=mode,
                   known_issue=KNOWN_ISSUE, producer_dir=_HERE)
