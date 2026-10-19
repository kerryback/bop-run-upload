"""gs_solve_vy.py at NY = 1 must be gs_solve_reg.py at gmreg = [1, 1]. Runs both at low resolution
(xnum 21, znum 30, bnum 10, tol 1e-6) with the registry writer stubbed -- nothing is recorded -- and
compares every saved table. 2026-10-07 run: max |diff| 4e-13 on P_up / P_down / Q0 / icut, 0 on the
refinancing policies, identical kernels.     python variants/gs_vy/check_parity.py   (~2-4 min)"""
import os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(HERE)
RUN = """
import sys, runpy, os
sys.path.insert(0, sys.argv[1])
from common import solstamp
solstamp.record = lambda *a, **k: {"total_bytes": 0, "committable": True}
solstamp.lookup = lambda *a, **k: None
script = sys.argv[2]; sys.argv = [script] + sys.argv[3:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
"""
with tempfile.TemporaryDirectory() as tmp:
    procs = []
    for sub, script, ov in (("reg", os.path.join(V, "gs_bx", "gs_solve_reg.py"), '{"znum":30,"bnum":10,"gmreg":[1.0,1.0]}'),
                            ("vy", os.path.join(HERE, "gs_solve_vy.py"), '{"znum":30,"bnum":10,"NY":1}')):
        env = dict(os.environ, GS_PARAM_OVERRIDES=ov)
        procs.append(subprocess.Popen([sys.executable, "-W", "ignore", "-c", RUN, V, script, "21", "1e-6", os.path.join(tmp, sub)],
                                      env=env, cwd=os.path.dirname(script), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    for p in procs:
        _, err = p.communicate()
        assert p.returncode == 0, err.decode()[-2000:]
    a = np.load(os.path.join(tmp, "reg", "solution.npz")); b = np.load(os.path.join(tmp, "vy", "solution.npz"))
    for k in ("P_up", "P_down", "Q0", "icut_up", "icut_dn", "b_refin_0", "b_refin_I"):
        print(f"{k:10s} max|diff| {np.abs(a[k][0] - b[k][0]).max():.2e}  scale {np.abs(a[k]).max():.3e}")
    print("Mx equal:", np.allclose(a["Mx"][0], b["Mx"]))
