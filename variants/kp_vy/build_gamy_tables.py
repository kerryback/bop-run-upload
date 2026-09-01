"""Build the (2 x NY) G file and the NY per-node integral tables for the current KP_PARAM_OVERRIDES.
usage: KP_PARAM_OVERRIDES='{"g_lo":0.5,"g_hi":2.5}' python build_gamy_tables.py <prefix>"""
import os, sys, subprocess, json, time
prefix = sys.argv[1] if len(sys.argv) > 1 else "gamy"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parameters_kp14 import g_lo, g_hi, g_steep, kappa_y, NY, rho_y
key = {"g": [g_lo, g_hi, g_steep, kappa_y, NY], "rho": list(map(float, rho_y))}
meta = os.path.join(HERE, f"meta_{prefix}.json")
g = os.path.join(HERE, f"G_{prefix}.csv")
if os.path.exists(meta) and json.load(open(meta)) == key and os.path.exists(g):
    print("cached"); sys.exit(0)
t0 = time.time()
env = dict(os.environ, KP_GAMY_GOUT=g)
subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "kp14_fd_gamy.py")], env=env,
               stdout=subprocess.DEVNULL, check=True, cwd=HERE)
print(f"G solved in {time.time()-t0:.0f}s", flush=True)
for iy in range(NY):
    env = dict(os.environ, KP_GAMY_YIDX=str(iy), KP_GAMY_GIN=g,
               KP_GAMY_IOUT=os.path.join(HERE, f"integ_{prefix}_{iy}.npz"))
    subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "integ_kp14.py")], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, cwd=HERE)
    print(f"integ node {iy+1}/{NY}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(key, open(meta, "w"))
print(f"built {prefix} in {time.time()-t0:.0f}s")
