"""bgn_vy at zero exposure must be bgn_gam. Builds a zero-exposure J* table in a temp dir (nothing is
recorded), simulates the SAME seed through both models and compares prices, returns, expected returns
and two months of conditional moments. Expected: prices within ~2e-3 (the DP J* against the closed
form's 950-period truncation), expected returns within ~3e-5, SR_max to 1e-4.
    python variants/bgn_vy/check_reduction.py        (~2 min; imports both model directories in subprocesses)
2026-10-07 run: P max|rel| 1.86e-3, eret max|diff| 2.1e-5, mu max|diff| 3.7e-6, SR_max 0.22353 vs 0.22340 / 0.24046 vs 0.24046."""
import os, subprocess, sys, tempfile, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(HERE)
PY = sys.executable
BUILD = """
import sys, os, numpy as np
os.chdir(sys.argv[1]); sys.path.insert(0, ".")
import parameters as P, vasicek_vy as vv
J, OV, info = vv.solve_jstar(tol=P.jstar_tol, verbose=False)
np.savez_compressed(sys.argv[2], r_solve=vv.r_solve, y_solve=vv.y_solve, J=J, OV=OV, type_theta=vv.type_theta)
"""
SIM = """
import sys, os, numpy as np
np.seterr(all="ignore")
N, T = int(sys.argv[2]), int(sys.argv[3])
os.chdir(sys.argv[1]); sys.path.insert(0, ".")
import panel_functions as pf, sdf_compute as sc
np.random.seed(7)
arr = pf.create_arrays(N, T); loop = sc.sdf_compute(N, T, arr)
out = {"r": arr[0], "eret": arr[7], "ret": arr[8], "P": arr[9]}
for t in (T - 30, T - 10):
    o = loop(t); out[f"sr{t}"] = o[1]; out[f"mu{t}"] = o[2]; out[f"Sig{t}"] = o[3]
np.savez(sys.argv[4], **out)
"""
with tempfile.TemporaryDirectory() as tmp:
    jz = os.path.join(tmp, "Jstar_vy_zero.npz")
    env = dict(os.environ, BGN_PARAM_OVERRIDES=json.dumps({"jstar_vy_file": jz}))
    subprocess.run([PY, "-W", "ignore", "-c", BUILD, os.path.join(V, "bgn_vy"), jz], env=env, check=True, capture_output=True)
    N, T = 60, 480
    subprocess.run([PY, "-W", "ignore", "-c", SIM, os.path.join(V, "bgn_vy"), str(N), str(T), os.path.join(tmp, "vy.npz")], env=env, check=True, capture_output=True)
    env2 = dict(os.environ, BGN_PARAM_OVERRIDES=json.dumps({"jstar_gam_file": "Jstar_bgnbase.csv"}))
    subprocess.run([PY, "-W", "ignore", "-c", SIM, os.path.join(V, "bgn_gam"), str(N), str(T), os.path.join(tmp, "gam.npz")], env=env2, check=True, capture_output=True)
    a = np.load(os.path.join(tmp, "gam.npz")); b = np.load(os.path.join(tmp, "vy.npz"))
    print("r identical:", np.array_equal(a["r"], b["r"]))
    for k in ("P", "ret", "eret"):
        x, y = a[k], b[k]; m = np.isfinite(x) & np.isfinite(y)
        print(f"{k:5s} max|rel diff| {np.nanmax(np.abs(x[m]-y[m])/np.maximum(np.abs(x[m]),1e-9)):.2e}  max|diff| {np.nanmax(np.abs(x[m]-y[m])):.2e}")
    for t in (T - 30, T - 10):
        print(f"t={t}: sr_max {float(a[f'sr{t}']):.5f} vs {float(b[f'sr{t}']):.5f};  mu max|diff| {np.abs(a[f'mu{t}']-b[f'mu{t}']).max():.2e};  Sig max|rel| {np.abs(a[f'Sig{t}']-b[f'Sig{t}']).max()/np.abs(a[f'Sig{t}']).max():.2e}")
