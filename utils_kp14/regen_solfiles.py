#!/usr/bin/env python
"""
Regenerate the KP14 solution files and stamp their provenance.

Ordering is load-bearing: integ_kp14.py reads G_func.csv, so kp14_fd.py must run
first. Running them by hand in the wrong order -- or from the wrong directory,
back when both used bare CWD filenames -- silently pairs fresh integrals with a
stale G. This script exists so that ordering is not a thing anyone has to
remember.

Both producers are run as subprocesses rather than imported, because each does
its work at module level.

Usage (from the repo root):
    python utils_kp14/regen_solfiles.py              # write into KP14_solfiles/
    python utils_kp14/regen_solfiles.py --out DIR    # dry run into DIR
    python utils_kp14/regen_solfiles.py --stamp-only # re-stamp without regenerating
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.solfile_stamp import write_stamp, verify          # noqa: E402
from utils_kp14 import solfile_spec                          # noqa: E402

# In dependency order. Do not reorder.
PRODUCERS = ['kp14_fd.py', 'integ_kp14.py']


def run(producer, out_dir):
    print(f'--- {producer} -> {out_dir} ' + '-' * (46 - len(producer)))
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(_HERE, producer), out_dir],
                       cwd=os.path.dirname(_HERE))
    if r.returncode != 0:
        sys.exit(f'{producer} failed with return code {r.returncode}; solfiles NOT stamped')
    print(f'--- {producer} ok in {time.time() - t0:.1f}s\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=solfile_spec.SOLFILES_DIR,
                    help='destination directory (default: utils_kp14/KP14_solfiles)')
    ap.add_argument('--stamp-only', action='store_true',
                    help='stamp the files already on disk without regenerating')
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    if not args.stamp_only:
        for producer in PRODUCERS:
            run(producer, out)

    spec = solfile_spec.spec()
    stamp = write_stamp(out, spec, producer_dir=_HERE)
    print(f'wrote {os.path.join(out, "stamp.json")}')
    for name, rec in sorted(stamp['solfiles'].items()):
        print(f'  {name:<24s} sha256:{rec["sha256"][:12]}  '
              f'{len(rec["params"])} params  producer:{rec["producer_sha256"][:12]}')

    print()
    problems = verify('KP14', out, spec, mode='warn',
                      producer_dir=_HERE, once=False)
    if problems:
        sys.exit(f'\nregeneration finished but verification still reports '
                 f'{len(problems)} problem(s) -- do not commit these files')
    print('\nverification clean.')


if __name__ == '__main__':
    main()
