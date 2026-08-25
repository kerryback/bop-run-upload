"""Time the OLD (kron) vs NEW (outer + Gram) sdf_loop on identical arrays."""
import sys, os, time, importlib.util; sys.path.insert(0, os.getcwd())
import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


from utils_bgn import panel_functions_bgn as pf
from utils_bgn import sdf_compute_bgn as new
old = load_module('voc_diagnosis/refactor_checks/baseline/sdf_compute_bgn_prerefactor.py', 'sdf_old')

for N, T in [(300, 100), (500, 200)]:
    np.random.seed(20260825)
    arr = pf.create_arrays(N, T)
    lo, ln = old.sdf_compute(N, T, arr), new.sdf_compute(N, T, arr)
    ts = [T // 2, int(T * 0.75), T - 2]
    print(f"\nN={N} T={T}")
    for t in ts:
        def tm(f, n=3):
            b = np.inf
            for _ in range(n):
                t0 = time.perf_counter(); f(t, 0); b = min(b, time.perf_counter() - t0)
            return b
        a, b = tm(lo), tm(ln)
        M = int((arr[4][t, :t + 1, :] > 0).sum())
        print(f"  t={t:4d}  M={M:6d}   old {a*1e3:9.1f} ms   new {b*1e3:9.1f} ms"
              f"   speedup {a/b:6.2f}x")
