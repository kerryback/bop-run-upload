#!/usr/bin/env python
"""
Regenerate BGN's solution file and stamp its provenance.

One producer, one output, so there is no ordering to get wrong. The expensive
part is the solve: ~19 s per grid point, 201 points, so ~32 min at n_jobs=2.

Jstar.csv as committed already reproduces from config.py to 1.3e-13, so if all
you need is a provenance record for files you already trust, use --stamp-only
and skip the half hour.

Usage (from the repo root):
    python utils_bgn/regen_solfiles.py --stamp-only    # record provenance only
    python utils_bgn/regen_solfiles.py --n-jobs 2      # full re-solve
    python utils_bgn/regen_solfiles.py --out DIR       # dry run elsewhere
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.solfile_stamp import write_stamp, verify          # noqa: E402

# Import solfile_spec WITHOUT going through utils_bgn/__init__.py: that imports
# the consumers, which verify(mode='error') at import time, so importing the
# package while the stamp is stale would stop the tool that repairs it.
import importlib.util as _ilu                                # noqa: E402
_spec = _ilu.spec_from_file_location('_bgn_solfile_spec',
                                     os.path.join(_HERE, 'solfile_spec.py'))
solfile_spec = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(solfile_spec)


def run(out_dir, n_jobs):
    producer = solfile_spec.PRODUCER
    print(f'--- {producer} -> {out_dir} ' + '-' * (46 - len(producer)))
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(_HERE, producer),
                        '--out', out_dir, '--n-jobs', str(n_jobs)],
                       cwd=os.path.dirname(_HERE))
    if r.returncode != 0:
        sys.exit(f'{producer} failed with return code {r.returncode}; '
                 f'solfiles NOT stamped')
    print(f'--- {producer} ok in {time.time() - t0:.1f}s\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=solfile_spec.SOLFILES_DIR)
    ap.add_argument('--stamp-only', action='store_true',
                    help='stamp the file already on disk without re-solving')
    ap.add_argument('--n-jobs', type=int, default=2)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    if not args.stamp_only:
        run(out, args.n_jobs)

    spec = solfile_spec.spec()
    stamp = write_stamp(out, spec, producer_dir=_HERE)
    print(f'wrote {os.path.join(out, "stamp.json")}')
    for name, rec in sorted(stamp['solfiles'].items()):
        print(f'  {name:<24s} sha256:{rec["sha256"][:12]}  '
              f'{len(rec["params"])} params  producer:{rec["producer_sha256"][:12]}')

    print()
    problems = verify('BGN', out, spec, mode='warn',
                      producer_dir=_HERE, once=False)
    if problems:
        sys.exit(f'\nregeneration finished but verification still reports '
                 f'{len(problems)} problem(s) -- do not commit these files')
    print('\nverification clean.')


if __name__ == '__main__':
    main()
