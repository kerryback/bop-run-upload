"""Build per-type G and per-(type, y-node) integral tables for kp_vy.
usage: KP_PARAM_OVERRIDES='{...}' python build_vy_tables.py <prefix>"""
import os, sys, subprocess, json, time
from joblib import Parallel, delayed
prefix = sys.argv[1] if len(sys.argv) > 1 else "vy"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parameters_kp14 import ntypes, type_bv, gamma_v, kappa_y, NY, rho_ty, bv_comp
key = {"bv": list(map(float, type_bv)), "gv": gamma_v, "ky": kappa_y, "comp": bv_comp,
       "rho": [list(map(float, r)) for r in rho_ty]}
meta = os.path.join(HERE, f"meta_{prefix}.json")
if os.path.exists(meta) and json.load(open(meta)) == key:
    print("cached"); sys.exit(0)
t0 = time.time()
for f in range(ntypes):
    g = os.path.join(HERE, f"G_{prefix}{f}.csv")
    env = dict(os.environ, KP_VY_TYPE=str(f), KP_VY_GOUT=g)
    subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "kp14_fd_vy.py")], env=env,
                   stdout=subprocess.DEVNULL, check=True, cwd=HERE)
    print(f"G type {f} solved ({time.time()-t0:.0f}s)", flush=True)
def one(f, iy):
    g = os.path.join(HERE, f"G_{prefix}{f}.csv")
    env = dict(os.environ, KP_VY_TYPE=str(f), KP_GAMY_YIDX=str(iy), KP_GAMY_GIN=g,
               KP_GAMY_IOUT=os.path.join(HERE, f"integ_{prefix}{f}_{iy}.npz"))
    subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "integ_kp14.py")], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, cwd=HERE)
    return (f, iy)
jobs = [(f, iy) for f in range(ntypes) for iy in range(NY)]
done = Parallel(n_jobs=6, verbose=5)(delayed(one)(f, iy) for f, iy in jobs)
json.dump(key, open(meta, "w"))
print(f"built {prefix}: {len(done)} tables in {time.time()-t0:.0f}s")
