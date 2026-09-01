"""Build the per-regime J* table for the current BGN_PARAM_OVERRIDES (columns r, J0, J1)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parameters import *
import vasicek
t0 = time.time()
print("building", jstar_gam_file, "gmult =", list(gmult), flush=True)
grid, J = vasicek.build_jstar_gam_table(jstar_gam_file, tol=float(os.environ.get("JSTAR_TOL", "3e-4")))
print(f"done {len(grid)} pts in {time.time()-t0:.0f}s; J0 range {J[0].min():.3f}..{J[0].max():.3f}  J1 range {J[1].min():.3f}..{J[1].max():.3f}")
