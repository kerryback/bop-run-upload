"""
Which config parameters and producer each GS21 solution file depends on.

As of 2026-08-26 GS21's producer is Python -- utils_gs21/gs21_solve.py, a port of
GS21.m -- so config.py is the single source of truth and this module no longer
has to reconcile two of them. What used to live here was a MATLAB-source parser:
it read GS21.m's assignments and diffed them against config.py, because the .m
file could not read config.py and the two had drifted. That check is gone. It
reported five divergences, and all five are now settled:

    sigma_x, sigma_z   exponent 3/2 -> 2/3, identified from the committed grids
                       (tauchen makes them pure functions of the parameters)
    r                  0.074830/12 -> 0.1/12; config.py had transcribed the
                       COMMENTED-OUT alternative on GS21.m:27
    xi                 config.py's GS21_ZETA = 0.01 was right; GS21.m:35's
                       `0.03/3*3` carries a spurious *3
    gamma_x, sigma_m   absent from config.py entirely; now GS21_GAMMA_X = 0.5
                       and GS21_SIGMA_M = 5

r, xi and gamma_x were not identifiable from the saved files by inspection --
only by running the ported operator on the committed solfiles and seeing which
parameter set reproduces them. See the note above GS21_R in config.py for the
residual table, and gs21_solve.py's docstring for the port's validation.

GS21.m is retained as an archival reference. Nothing reads it.

The parameter list is derived from the producer's own source rather than
maintained by hand: `_config_attrs()` collects every `GS21_*` attribute the
producer touches, so the code IS the specification and the two cannot drift.
gs21_solve.py imports utils_gs21/tauchen.py, whose source digest is folded into
the same parameter dict -- a change to the discretisation must invalidate the
solfiles just as a change to a parameter does.
"""

import ast
import os

import config
from utils.solfile_stamp import digest

_HERE = os.path.dirname(os.path.abspath(__file__))
SOLFILES_DIR = os.path.join(_HERE, 'GS21_solfiles')

PRODUCER = 'gs21_solve.py'
DEPENDS_ON = ['tauchen.py']          # imported by the producer, not a solfile

# The 18 files gs21_solve.py writes: 14 policies plus the 4 grids.
SOLFILES = [
    'P_up.csv', 'P_down.csv', 'PI_up.csv', 'PI_down.csv',
    'P0_up.csv', 'P0_down.csv', 'Q_0.csv', 'Q_I.csv',
    'z_cut_up.csv', 'z_cut_down.csv', 'i_cut_up.csv', 'i_cut_down.csv',
    'b_refin_0.csv', 'b_refin_I.csv',
    'bgrid.csv', 'igrid.csv', 'xgrid.csv', 'zgrid.csv',
]


def _config_attrs(producer=PRODUCER):
    """Every GS21_* config attribute the producer reads, with current values.

    gs21_solve.py does `import config` and reads `c.GS21_BETA`, so unlike the
    KP14 producers there is no `from config import ...` line to parse. Matching
    on the attribute name is enough -- and it means adding a parameter to the
    producer automatically extends what gets stamped and checked.
    """
    with open(os.path.join(_HERE, producer)) as f:
        tree = ast.parse(f.read())
    names = sorted({n.attr for n in ast.walk(tree)
                    if isinstance(n, ast.Attribute) and n.attr.startswith('GS21_')})
    if not names:
        raise AssertionError(f'{producer} reads no GS21_* config attributes; '
                             f'the spec would stamp nothing')
    missing = [n for n in names if not hasattr(config, n)]
    if missing:
        raise AttributeError(f'{producer} reads names absent from config.py: {missing}')
    params = {n: getattr(config, n) for n in names}
    for dep in DEPENDS_ON:
        params[f'__{dep}__'] = digest(os.path.join(_HERE, dep))
    return params


def spec():
    """{filename: {'producer', 'params', 'inputs'}} for utils/solfile_stamp.py.

    One producer writes all 18 files in a single run, so they share a parameter
    set and none is an input to another.
    """
    params = _config_attrs()
    return {name: {'producer': PRODUCER, 'params': params, 'inputs': []}
            for name in SOLFILES}


# Extra context appended to the banner when verification FAILS.
KNOWN_ISSUE = """\
To regenerate and re-stamp all 18 solution files, from the repo root:
    python utils_gs21/regen_solfiles.py
One producer (gs21_solve.py) writes them in a single ~2.5 min run, so there is no
ordering to get wrong. Background: kp14_crash_20260826.md."""


def verify(mode='error'):
    """One-line entry point for the consumers."""
    from utils.solfile_stamp import verify as _verify
    return _verify('GS21', SOLFILES_DIR, spec(), mode=mode,
                   known_issue=KNOWN_ISSUE, producer_dir=_HERE)
