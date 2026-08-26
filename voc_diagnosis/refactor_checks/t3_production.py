"""
T3 — production-scale acceptance test for the sdf_compute_bgn refactor.

WHY IT IS A PAIRED TEST, not "run one panel and check it hits 0.4172":
the repo has NO explicit seeding anywhere, so panels are not reproducible, and per-panel
dispersion of the BGN 3600/alpha=0.01 Sharpe is sd 0.038 over the 16 archived panels
(range 0.3335 to 0.4728). A single unseeded panel could land anywhere in that range and
would say nothing about correctness. So instead: generate ONE production panel and run the
moments step through BOTH the pre-refactor and post-refactor code, on the same arrays.

WHY THE 3.5-HOUR DKKM STEP IS NOT NEEDED: only evaluate_sdfs.py consumes *_moments.pkl
(via sdf_utils.load_precomputed_moments). estimate_sdf_fama and estimate_sdf_dkkm read the
panel and the factors, never the moments. So this refactor can only affect evaluation, and
only through rp / cond_var / sdf_ret / max_sr -- which is exactly what this compares.

This also exercises the Sparse3D chi path (chi.get_row_slice), which the local N<=500
tests did not: they used dense numpy chi.

Usage (expects {panel_id}_arr/ to already exist, i.e. after utils/generate_panel.py):
    python voc_diagnosis/refactor_checks/t3_production.py bgn_t3 [n_months]
"""
import sys, os, time, pickle, resource, importlib.util
sys.path.insert(0, os.getcwd())
import numpy as np

PANEL_ID = sys.argv[1] if len(sys.argv) > 1 else 'bgn_t3'
N_MONTHS = int(sys.argv[2]) if len(sys.argv) > 2 else 360

import config
config.init_from_env()
from utils.sparse_3d import load_sparse_3d, Sparse3D


def load_arr(panel_id):
    d = os.path.join(config.TEMP_DIR, f'{panel_id}_arr')
    meta = pickle.load(open(os.path.join(d, 'metadata.pkl'), 'rb'))
    si = meta.get('sparse_info', {})
    out = []
    for i in range(meta['n_arrays']):
        info = si.get(i, {})
        if info.get('is_sparse', False):
            out.append(Sparse3D(load_sparse_3d(os.path.join(d, f'{i}_sparse'),
                                               info['n_slices']), info['shape']))
        else:
            out.append(np.load(os.path.join(d, f'{i}.npy'), mmap_mode='r'))
    return tuple(out), meta['N'], meta['T']


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == 'darwin' else r / 1e6


arr, N, T = load_arr(PANEL_ID)
burnin = config.BGN_BURNIN
start, end = burnin + 360, T + burnin - 1
months = list(range(start, min(end, start + N_MONTHS - 1) + 1))
print(f"T3: {PANEL_ID}  N={N} T={T} burnin={burnin}")
print(f"    chi type: {type(arr[4]).__name__}  (Sparse3D = the production path)")
print(f"    comparing {len(months)} months: {months[0]}..{months[-1]}")

from utils_bgn import sdf_compute_bgn as new
old = load_module('voc_diagnosis/refactor_checks/baseline/sdf_compute_bgn_prerefactor.py',
                  'sdf_pre')
loop_new = new.sdf_compute(N, T, arr)
loop_old = old.sdf_compute(N, T, arr)

worst = {'rp': 0.0, 'cond_var': 0.0, 'sdf_ret': 0.0, 'max_sr': 0.0}
t_old = t_new = 0.0
rss0 = rss_gb()
print(f"\n{'month':>7} {'t_old(s)':>9} {'t_new(s)':>9} {'x':>6} | "
      f"{'rp':>10} {'cond_var':>10} {'sdf_ret':>10} {'max_sr':>10} | {'M':>7} {'RSS(GB)':>8}")
for i, m in enumerate(months):
    t = m - 1
    a = time.perf_counter(); o = loop_old(t, 0); da = time.perf_counter() - a
    b = time.perf_counter(); n_ = loop_new(t, 0); db = time.perf_counter() - b
    t_old += da; t_new += db
    rel = {}
    for k, io in (('sdf_ret', 0), ('max_sr', 1), ('rp', 2), ('cond_var', 3)):
        A, B = np.asarray(o[io]), np.asarray(n_[io])
        sc = max(np.abs(A).max(), 1e-300)
        rel[k] = float(np.abs(A - B).max() / sc)
        worst[k] = max(worst[k], rel[k])
    if i % 30 == 0 or i == len(months) - 1:
        M = int(np.asarray(arr[4].get_row_slice(t, t + 1).toarray()
                           if hasattr(arr[4], 'get_row_slice') else arr[4][t, :t + 1, :]).astype(bool).sum())
        print(f"{m:>7} {da:9.3f} {db:9.3f} {da/max(db,1e-9):6.2f} | "
              f"{rel['rp']:10.2e} {rel['cond_var']:10.2e} {rel['sdf_ret']:10.2e} "
              f"{rel['max_sr']:10.2e} | {M:>7} {rss_gb():8.2f}")

print(f"\n{'='*78}")
print(f"WORST RELATIVE DIFFERENCE over {len(months)} months")
for k, v in worst.items():
    print(f"  {k:10s} {v:.3e}")
print(f"\nTIMING  old {t_old:8.1f}s   new {t_new:8.1f}s   speedup {t_old/max(t_new,1e-9):.2f}x")
print(f"        per month: old {t_old/len(months):.3f}s  new {t_new/len(months):.3f}s")
print(f"        extrapolated to 360 months: old {t_old/len(months)*360/60:.1f} min  "
      f"new {t_new/len(months)*360/60:.1f} min")
print(f"PEAK RSS {rss_gb():.2f} GB  (growth during comparison {rss_gb()-rss0:+.2f} GB)")
ok = worst['cond_var'] < 1e-9 and worst['rp'] < 1e-9
print(f"\n{'PASS' if ok else 'FAIL'}  (gate: rp and cond_var relative difference < 1e-9)")
print(f"{'='*78}")
sys.exit(0 if ok else 1)
