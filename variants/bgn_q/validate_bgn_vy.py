"""bgn_vy validation, in the shape verify_solves.py parses.
  1. PRICING: the project-value Euler equation with the priced state y, at the economy's own parameters:
       V_f(r, y, beta, s) = e^{-r} pi [ Chat_f e^{-beta g_s} E^Q(e^{b_f y'}) + E^Q V_f(r', y', beta, s') ],
     r' with its Q-drift shifted by -beta_zr, y' by -lam_y s_y, s' ~ Preg. Quadrature against the closed form.
  2. SOLVE: the growth-option table's fixed-point residual J_f = e^{-r} E^Q[OV_f + J_f] at the reported nodes.
  3. SIMULATION: realised vs expected excess returns by type and regime, max_sr by y-tercile, PSD.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.seterr(all="ignore")
from parameters import *
import vasicek_vy as vv

hr, wr = np.polynomial.hermite.hermgauss(40); hr = hr * np.sqrt(2); wr = wr / np.sqrt(np.pi)
worst = 0.0
for r0 in (rbar - 0.004, rbar - 0.0005, rbar, rbar + 0.0005, rbar + 0.004):
    for y0 in (-2.0, 0.0, 2.0):
        for beta in (-0.3, -0.05):
            for s in (0, 1):
                for f in range(ntypes):
                    b = float(type_bv[f])
                    rp = kappa * r0 + (1 - kappa) * rbar - beta_zr + sigma_r * hr          # (40,)
                    cont = 0.0
                    if CHAIN:
                        # on the chain the Q-transition is exact: y0 is the node nearest the value, y' runs over nodes
                        iy0 = int(np.argmin(np.abs(Y1 - y0) + np.abs(Y2))); wq = PQ_y[iy0]   # (y0, 0) on a joint chain
                        for s2 in (0, 1):
                            Vn = vv.V_vy(rp[:, None], np.arange(NY)[None, :], beta, s2, f)   # (40, NY)
                            cont += Preg[s, s2] * (wr[:, None] * wq[None, :] * Vn).sum()
                        e1 = float(wq @ np.exp(BY[f]))
                        rhs = np.exp(-r0) * pi * (vv.Chat_f[f] * np.exp(-beta * g_bins[int(bin_of_r(r0))]) * e1 + cont)
                        worst = max(worst, abs(rhs / float(vv.V_vy(r0, iy0, beta, s, f)) - 1))
                        continue
                    yp = rho_y * y0 - lam_y * s_y + s_y * hr                                  # (40,)
                    for s2 in (0, 1):
                        Vn = vv.V_vy(rp[:, None], yp[None, :], beta, s2, f)                  # (40, 40)
                        cont += Preg[s, s2] * (wr[:, None] * wr[None, :] * Vn).sum()
                    rhs = np.exp(-r0) * pi * (vv.Chat_f[f] * np.exp(-beta * g_bins[int(bin_of_r(r0))]) * Ek_y(b, 1, y0) + cont)
                    worst = max(worst, abs(rhs / float(vv.V_vy(r0, y0, beta, s, f)) - 1))
print(f"project-value Euler equation with y: worst {worst:.2e}")
k = 1400; spread = 12 * (-np.log(vv.B(k, rbar)) / k - rbar)
print(f"limiting term spread {spread:.4f}/yr")

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), jstar_vy_file))
J, OV = d["J"], d["OV"]
Wr, Wy = vv.transition_Q()
res = 0.0
for f in range(ntypes):
    for s in (0, 1):
        rhs = np.exp(-vv.r_solve)[:, None] * sum(Preg[s, sp] * (Wr @ (OV[f, sp] + J[f, sp]) @ Wy.T) for sp in (0, 1))
        res = max(res, float(np.abs(rhs[:, vv._i_tab] / J[f, s][:, vv._i_tab] - 1).max()))
print(f"J* fixed-point residual at the reported nodes: {res:.2e}; type_theta {np.round(vv.type_theta, 4).tolist()}")

import panel_functions as pf, sdf_compute as sc
np.random.seed(3)
N, T = 120, burnin + 260
arr = pf.create_arrays(N, T)
loop = sc.sdf_compute(N, T, arr)
r, eret, ret, sreg, yreg, ftype = arr[0], arr[7], arr[8], arr[-3], arr[-2], arr[-1]
rows = []
for t in range(burnin + 5, T - 2, 2):
    _, max_sr, mu, Sig, _ = loop(t - 1)
    rows.append((t, sreg[t], yreg[t], max_sr, np.nanmean(mu), np.nanstd(mu), np.linalg.eigvalsh(Sig).min()))
rows = np.array(rows)
q = np.quantile(rows[:, 2], [1 / 3, 2 / 3]); terc = np.digitize(rows[:, 2], q)
print(f"q_price={q_price} g_bins={np.round(g_bins, 3).tolist()} type_bv={list(type_bv)} gamma_v={gamma_v} kappa_y={kappa_y} bv_comp={bv_comp} gmult={list(gmult)} y_kind={y_kind} pj={pj} gj={gj} NY2={NY2} type_bv2={list(type_bv2)} gamma_v2={gamma_v2}")
for k_ in range(3):
    m = rows[terc == k_]
    print(f"y-terc {k_}: months {len(m):3.0f}  max_sr {m[:,3].mean():.4f}  E[mu] {m[:,4].mean():.5f}  cs-sd(mu) {m[:,5].mean():.5f}  min eig {m[:,6].min():.2e}")
mo = slice(burnin + 5, T - 2)
rf = (np.exp(r[mo]) - 1)[:, None]
for g in (0, 1):
    sel = sreg[mo] == g
    if sel.sum() == 0:
        continue
    re = (ret[mo] - rf).mean(1)[sel]; ee = (eret[mo] - rf).mean(1)[sel]; n = sel.sum()
    print(f"regime {g}: realized mean excess {re.mean():.5f}  expected {ee.mean():.5f}  (n={n}, se {re.std()/np.sqrt(n):.5f})")
for f in range(ntypes):
    cols = ftype == f
    re = (ret[mo][:, cols] - rf).mean(); ee = (eret[mo][:, cols] - rf).mean()
    se = (ret[mo][:, cols] - rf).mean(1).std() / np.sqrt(T - 7 - burnin)
    print(f"type {f} (bv={type_bv[f]}): realized {re:.5f}  expected {ee:.5f}  (se {se:.5f})  "
          f"premium {12*ee*100:.2f}%/yr  mean log size {np.log(arr[9][mo][:, cols]).mean():.3f}")
print("BGN Q VALIDATION DONE")
