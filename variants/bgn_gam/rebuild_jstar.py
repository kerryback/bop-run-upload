"""Rebuild the J*(r) table for the current parameter overrides.
usage: BGN_PARAM_OVERRIDES='{"sigma_r":0.004,"jstar_file":"Jstar_sr4.csv"}' python rebuild_jstar.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parameters import *
import vasicek
t0 = time.time()
print("building", jstar_file, "beta_star=%.4f scale=%.4f" % (vasicek.beta_star, vasicek.scale), flush=True)
grid, J = vasicek.build_jstar_table(jstar_file, tol=float(os.environ.get("JSTAR_TOL", "3e-4")))
print(f"done {len(grid)} pts in {time.time()-t0:.0f}s; J range {J.min():.3f}..{J.max():.3f}")
