"""Build the per-regime J* table for the current BGN_PARAM_OVERRIDES (columns r, J0, J1).

Content-addressed and stamped via variants/common/solstamp.py. Previously this
producer had NO staleness detection of any kind: `jstar_gam_file` names the
output, so the FILENAME was the only identity. Reusing a name with different
parameters, or editing vasicek.py, silently reused a table built for a different
economy -- and the consumer (sdf_compute.py:21) just read the CSV.

Set BGN_JSTAR_FORCE=1 to rebuild even on a cache hit.
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # variants/, for `common`

import parameters as P
from parameters import *          # noqa: F401,F403  (jstar_gam_file, gmult, ...)
import vasicek
from common import solstamp

SOURCES = [os.path.join(HERE, f) for f in
           ("vasicek.py", "parameters.py", "rebuild_jstar_gam.py")]

TOL = float(os.environ.get("JSTAR_TOL", "3e-4"))
# JSTAR_TOL changes the table, so it belongs in env_params (hashed), not extra.
snap = solstamp.snapshot(P, SOURCES, model="bgn_gam",
                         env_params={"JSTAR_TOL": TOL},
                         extra={"out": jstar_gam_file})

OUT = os.path.join(HERE, jstar_gam_file)
print(f"[solstamp] bgn_gam {jstar_gam_file} solve_id={snap.solve_id}", flush=True)

hit = solstamp.lookup(snap.solve_id)
if hit and not os.environ.get("BGN_JSTAR_FORCE"):
    problems = solstamp.artifact_problems(hit)
    if not problems:
        print(f"cached: solve_id {snap.solve_id} already built ({hit['total_bytes']:,} B)")
        sys.exit(0)
    print("[solstamp] manifest exists but artifacts do not match; rebuilding:")
    for p in problems[:5]:
        print(f"  - {p}")
elif os.path.exists(OUT):
    prior = solstamp._find_by_artifacts([OUT])
    if prior and prior["solve_id"] != snap.solve_id:
        print(f"[solstamp] {jstar_gam_file} on disk belongs to solve_id "
              f"{prior['solve_id']}; this run wants {snap.solve_id}. Differences:")
        for k, a, b in solstamp.diff_params(prior["params"], snap.params)[:10]:
            print(f"    {k}: {a!r} -> {b!r}")
    else:
        print(f"[solstamp] {jstar_gam_file} exists but its provenance is unrecorded; rebuilding")

t0 = time.time()
print("building", jstar_gam_file, "gmult =", list(gmult), flush=True)
grid, J = vasicek.build_jstar_gam_table(jstar_gam_file, tol=TOL)
print(f"done {len(grid)} pts in {time.time()-t0:.0f}s; "
      f"J0 range {J[0].min():.3f}..{J[0].max():.3f}  J1 range {J[1].min():.3f}..{J[1].max():.3f}")

man = solstamp.record(snap, [OUT], tag=os.path.splitext(jstar_gam_file)[0])
print(f"[solstamp] recorded solve_id {snap.solve_id} "
      f"({man['total_bytes']:,} B, committable={man['committable']})")
