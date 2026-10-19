"""Build the per-(type, regime) J*_f(r, y, s) table for the current BGN_PARAM_OVERRIDES.

Content-addressed and stamped via variants/common/solstamp.py, as rebuild_jstar_gam.py: the
sources are vasicek_vy.py and parameters.py, the DP tolerance is hashed via env_params, the
output file name (jstar_vy_file) is a parameter and so part of the id. Set BGN_JSTAR_FORCE=1 to
rebuild on a cache hit.  usage:  BGN_PARAM_OVERRIDES='{...}' python rebuild_jstar_vy.py
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import parameters as P
from parameters import *          # noqa: F401,F403
import vasicek_vy as vv
from common import solstamp

SOURCES = [os.path.join(HERE, f) for f in ("vasicek_vy.py", "parameters.py")]
TOL = float(os.environ.get("JSTAR_TOL", str(jstar_tol)))
snap = solstamp.snapshot(P, SOURCES, model="bgn_q", env_params={"JSTAR_TOL": TOL},
                         extra={"out": jstar_vy_file})
OUT = os.path.join(HERE, jstar_vy_file)
print(f"[solstamp] bgn_q {jstar_vy_file} solve_id={snap.solve_id}", flush=True)

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
        print(f"[solstamp] {jstar_vy_file} on disk belongs to solve_id {prior['solve_id']}; this run wants {snap.solve_id}. Differences:")
        for k, a, b in solstamp.diff_params(prior["params"], snap.params)[:10]:
            print(f"    {k}: {a!r} -> {b!r}")
    elif prior:
        print(f"[solstamp] {jstar_vy_file} is recorded and matches solve_id {snap.solve_id}; rebuilding because BGN_JSTAR_FORCE is set")
    else:
        print(f"[solstamp] {jstar_vy_file} exists but its provenance is unrecorded; rebuilding")

t0 = time.time()
print(f"building {jstar_vy_file}: q_price {q_price} g_bins {np.round(g_bins, 3).tolist()} type_bv {list(type_bv)} gamma_v {gamma_v} kappa_y {kappa_y} bv_comp {bv_comp} y_kind {y_kind} NY2 {NY2} type_bv2 {list(type_bv2)} gamma_v2 {gamma_v2} "
      f"gmult {list(gmult)}; type_theta {np.round(vv.type_theta, 4).tolist()}; grid {vv.n_r} x {vv.n_y}, tol {TOL}", flush=True)
J, OV, info = vv.solve_jstar(tol=TOL)
np.savez_compressed(OUT, r_solve=vv.r_solve, y_solve=vv.y_solve, J=J, OV=OV, type_theta=vv.type_theta,
                    type_bv=type_bv, gamma_v=gamma_v, kappa_y=kappa_y, bv_comp=bv_comp, gmult=gmult,
                    type_bv2=type_bv2, NY2=NY2, gamma_v2=gamma_v2, kappa_y2=kappa_y2, q_price=q_price, g_bins=g_bins,
                    solve_id=snap.solve_id, tol=TOL, iters=info["iters"], rel_change=info["rel_change"])
i0 = vv.n_y // 2
print(f"done in {time.time()-t0:.0f}s: {info['exit']} after {info['iters']} iterations (rel change {info['rel_change']:.1e}); "
      f"J at (rbar, y=0): " + ", ".join(f"type {f} {np.interp(rbar, vv.r_solve, J[f, 0, :, vv._i_tab[i_y0 if CHAIN else NY // 2]]):.4f}" for f in range(ntypes)), flush=True)
man = solstamp.record(snap, [OUT], tag=os.path.splitext(jstar_vy_file)[0],
                      achieved={"exit": info["exit"], "iters": info["iters"], "rel_change": float(info["rel_change"]),
                                "tol_requested": float(TOL), "grid": [int(vv.n_r), int(vv.n_y)]})
print(f"[solstamp] recorded solve_id {snap.solve_id} ({man['total_bytes']:,} B, committable={man['committable']})")
