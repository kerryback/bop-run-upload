#!/usr/bin/env python
"""
Regenerate the GS21 solution files and stamp their provenance.

Unlike KP14 there is no ordering to get wrong: one producer (gs21_solve.py)
writes all 18 files from one solve, so they cannot be mutually inconsistent.
The producer refuses to write anything unless the solve converged.

Runtime is ~2.5 min at the shipped grid (80,000 states, ~1200 passes).

Usage (from the repo root):
    python utils_gs21/regen_solfiles.py              # write into GS21_solfiles/
    python utils_gs21/regen_solfiles.py --out DIR    # dry run into DIR
    python utils_gs21/regen_solfiles.py --stamp-only # re-stamp without regenerating
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.solfile_stamp import write_stamp, verify          # noqa: E402
# Import solfile_spec WITHOUT going through utils_gs21/__init__.py. That __init__
# imports the consumers, and the consumers verify(mode='error') at import time,
# so importing the package while the stamp is stale raises SolfileStaleError --
# i.e. the tool that repairs staleness could not start precisely when it was
# needed. A producer must not depend on the consumer package.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('_utils_gs21_solfile_spec',
                                     os.path.join(_HERE, 'solfile_spec.py'))
solfile_spec = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(solfile_spec)


def run(out_dir, tol):
    producer = solfile_spec.PRODUCER
    print(f'--- {producer} -> {out_dir} ' + '-' * (46 - len(producer)))
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(_HERE, producer),
                        '--write', out_dir, '--tol', repr(tol)],
                       cwd=os.path.dirname(_HERE))
    if r.returncode != 0:
        sys.exit(f'{producer} failed with return code {r.returncode}; '
                 f'solfiles NOT stamped')
    print(f'--- {producer} ok in {time.time() - t0:.1f}s\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=solfile_spec.SOLFILES_DIR,
                    help='destination directory (default: utils_gs21/GS21_solfiles)')
    ap.add_argument('--stamp-only', action='store_true',
                    help='stamp the files already on disk without regenerating')
    ap.add_argument('--tol', type=float, default=1e-11,
                    help='fixed-point tolerance on |dP| (default 1e-11)')
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    if not args.stamp_only:
        run(out, args.tol)

    spec = solfile_spec.spec()
    stamp = write_stamp(out, spec, producer_dir=_HERE)
    print(f'wrote {os.path.join(out, "stamp.json")}')
    n = len(stamp['solfiles'])
    any_rec = next(iter(stamp['solfiles'].values()))
    print(f'  {n} files, {len(any_rec["params"])} params each, '
          f'producer:{any_rec["producer_sha256"][:12]}')

    print()
    problems = verify('GS21', out, spec, mode='warn',
                      producer_dir=_HERE, once=False)
    if problems:
        sys.exit(f'\nregeneration finished but verification still reports '
                 f'{len(problems)} problem(s) -- do not commit these files')
    print('\nverification clean.')


if __name__ == '__main__':
    main()
