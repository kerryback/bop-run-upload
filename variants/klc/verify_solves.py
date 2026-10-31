#!/usr/bin/env python3
"""The solve-correctness gate for the three Quest economies. Exits non-zero unless ALL pass.

    python variants/klc/verify_solves.py bgn kp      # after solve_small_klc.sh
    python variants/klc/verify_solves.py gs          # after solve_gs_klc.sh; gates the GS seeds

"The producer exited 0" is not evidence that a model is solved. Both 2026-09-18 pricing defects
lived in solves that ran clean and passed every realised-against-expected validator. So each
model is checked three ways, and the pricing identity is checked AT THE ECONOMY'S OWN PARAMETERS
rather than at the defaults tests/test_risk_neutral_pricing.py uses:

  1. PROVENANCE   every id the spec pins has a manifest, its artifacts are byte-intact, and the
                  manifest's own record of how the solve EXITED meets the solver's contract
                  (BGN: tolerance; KP14: every G type under the 1e-6 residual gate; GS21: the
                  tolerance test and not the 5600-sweep cycle cap, which prints "converged" too).
  2. PRICING      prices satisfy the model's Euler equation / risk-neutral closed form. The
                  check strings are imported from the test module, not copied.
  3. SIMULATION   realised and expected excess returns agree within sampling error on a small
                  panel (each model's validate_*.py), parsed and thresholded rather than eyeballed.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = os.path.dirname(HERE)
ROOT = os.path.dirname(VARIANTS)
sys.path.insert(0, VARIANTS)
sys.path.insert(0, os.path.join(VARIANTS, "common"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from common import solstamp                     # noqa: E402
import runstamp                                 # noqa: E402
import test_risk_neutral_pricing as rnp         # noqa: E402

SPEC = {"bgn": "var-bgn_gam-g0235df-v1", "kp": "var-kp_vy-vyg30k70-v1", "gs": "var-gs_bx-gx7w-v1",
        "bvy": "var-bgn_vy-bvy25-v1", "gvy": "var-gs_vy-gvy25-v1", "kpj": "var-kp_vj-vyj25-v1", "bq": "var-bgn_q-bq25-v1"}
# Another spec of the same model can be gated by naming it in the environment, e.g.
# BOP_VERIFY_KP_SPEC=var-kp_vy-vyg25k15-v1 (solve_kp_klc.sh does this per spec it builds).
for _m in SPEC:
    SPEC[_m] = os.environ.get(f"BOP_VERIFY_{_m.upper()}_SPEC", SPEC[_m])
Z_MAX = 3.5        # realised-vs-expected, in standard errors
fails = []


def spec(model):
    return json.load(open(os.path.join(ROOT, "experiments", "specs", SPEC[model] + ".json")))


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        fails.append(msg)
    return ok


def manifest(sid, label):
    m = solstamp.lookup(sid)
    if not check(m is not None, f"{label}: manifest {sid} exists"):
        return None
    probs = solstamp.artifact_problems(m)
    check(not probs, f"{label}: {len(m['artifacts'])} artifact(s) byte-intact" + (f" -- {probs[:2]}" if probs else ""))
    return m


def run(cmd, cwd, env):
    e = dict(os.environ); e.update(env)
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)
    print("\n".join("      | " + ln for ln in r.stdout.strip().splitlines()[-14:]), flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], flush=True)
    return r


def bgn():
    s = spec("bgn"); ov = {"BGN_PARAM_OVERRIDES": json.dumps(s["params"])}
    d = os.path.join(VARIANTS, "bgn_gam")
    print(f"== bgn_gam  {s['spec_id']}", flush=True)
    m = manifest(s["expected_solves"]["jstar"], "jstar")
    if m:
        a = m.get("achieved") or {}
        check(a.get("exit") == "tolerance", f"jstar: exit '{a.get('exit')}' at tol {a.get('tol_requested')}, {a.get('grid_points')} grid points")
    out = rnp._run(rnp._BGN, d, ov)
    check(out["worst"] < 1e-6, f"project-value Euler equation at these parameters: worst {out['worst']:.2e} (< 1e-6)")
    check(0.021 < out["spread"] < 0.026, f"limiting term spread {out['spread']:.4f}/yr (BGN 1999: 2.4%)")
    r = run([sys.executable, "-W", "ignore", "validate_bgn_gam.py"], d, ov)
    check(r.returncode == 0 and "BGN GAM VALIDATION DONE" in r.stdout, "validate_bgn_gam.py ran to completion")
    rows = re.findall(r"regime (\d): realized mean excess ([-\d.eE+]+)\s+expected ([-\d.eE+]+)\s+\(n=(\d+), se ([-\d.eE+]+)\)", r.stdout)
    check(len(rows) == 2, "both regimes visited in the validation panel")
    for g, re_, ee, n, se in rows:
        z = (float(re_) - float(ee)) / float(se)
        check(abs(z) < Z_MAX, f"regime {g}: realised {float(re_):+.5f} vs expected {float(ee):+.5f}, z {z:+.2f} (n={n})")
    eig = [float(x) for x in re.findall(r"min eig ([-\d.eE+]+)", r.stdout)]
    check(eig and min(eig) > 0, f"conditional covariance positive definite (min eig {min(eig) if eig else float('nan'):.2e})")


def kp(which="kp", model="kp_vy"):
    s = spec(which); prefix = s["env"]["KP_VY_PREFIX"]
    ov = {"KP_PARAM_OVERRIDES": json.dumps(s["params"]), "KP_VY_PREFIX": prefix}
    d = os.path.join(VARIANTS, model)
    print(f"== {model}  {s['spec_id']}", flush=True)
    gid = s["expected_solves"]["G"]
    m = manifest(gid, "G")
    if m:
        a = m.get("achieved") or {}
        res = a.get("max_rel_residual")
        check(res is not None and res < 1e-6, f"G: direct solve, max relative residual {res} over {len(a.get('per_type_rel_residual') or {})} type(s) (gate 1e-6)")
    live = runstamp.live_solves(model, prefix)
    integ = [i for i in live if i != gid and (solstamp.lookup(i) or {}).get("stage") == "integ"]
    if check(len(integ) == 1, f"exactly one live integ solve under tag {prefix}: {integ}"):
        mi = manifest(integ[0], "integ")
        if mi:
            ntab = len(s["params"]["type_bv"]) * 21
            check(len(mi["artifacts"]) == ntab, f"integ: {len(mi['artifacts'])} tables (want {ntab}); the stage raises on lost CIR density mass")
            g_in = set((mi.get("inputs") or {}).values()); g_have = {a["sha256"] for a in m["artifacts"]} if m else set()
            check(bool(g_in) and g_in == g_have, "integ was built from exactly the recorded G tables' bytes")
        want = s["expected_solves"].get("integ")
        check(want in (None, integ[0]), f"integ id {integ[0]}" + (" matches the spec" if want else " (spec does not pin it yet: chained id, fill it in now)"))
        print(f"INTEG_ID={integ[0]}", flush=True)
    pr = s["params"]
    ou = float(pr.get("pj", 0.0)) == 0.0 and float(pr.get("g_lo", 1.0)) == float(pr.get("g_hi", 1.0)) and float(pr.get("g_quad", 0.0)) == 0.0 and not any(isinstance(b, list) for b in pr.get("type_bv", [])) and not bool(pr.get("bv_on_eps", False))
    if ou:
        out = rnp._run(rnp._KP, d, ov)
        check(out["rn"] == 1 and out["worst"] < 1e-4, f"A(y) against the risk-neutral closed form at these parameters: worst {out['worst']:.2e} (< 1e-4)")
    else:
        # with a crash in the generator, a price of risk that moves with y, or VECTOR exposures (the closed form is written
        # for a scalar b; vym2t3's gate crashed on it 2026-10-14), check the A solve's own linear system instead
        # (kpgam job 9586999 failed this check at worst 0.577 for that reason): check the A solve's own linear system
        out = rnp._run(_KPJ_RES, d, ov)
        check(out["worst"] < 1e-8, f"A(y) solves its risk-neutral system (crash or y-dependent price of risk; relative residual {out['worst']:.2e} < 1e-8); jump_rate_Q {out['rq']:.4f}/yr")
    r = run([sys.executable, "-W", "ignore", "validate_vy.py"], d, ov)
    check(r.returncode == 0 and "VY VALIDATION DONE" in r.stdout, "validate_vy.py ran to completion")
    rows = re.findall(r"type (\d+) \(bv=[^)]*\): realized ([-\d.eE+]+)\s+expected ([-\d.eE+]+)\s+\(se ([-\d.eE+]+)\)", r.stdout)
    check(len(rows) == len(s["params"]["type_bv"]), f"all {len(s['params']['type_bv'])} types present in the validation panel")
    for f, re_, ee, se in rows:
        z = (float(re_) - float(ee)) / float(se)
        check(abs(z) < Z_MAX, f"type {f}: realised {float(re_):+.5f} vs expected {float(ee):+.5f}, z {z:+.2f}")
    eig = [float(x) for x in re.findall(r"min eig ([-\d.eE+]+)", r.stdout)]
    check(eig and min(eig) > 0, f"conditional covariance positive definite (min eig {min(eig) if eig else float('nan'):.2e})")


_KPJ_RES = r"""
import json, numpy as np
import parameters_kp14 as P
worst = 0.0
for f in range(P.ntypes):
    for th in (0.0, P.theta_eps + P.theta_u):
        A = P._A_solve[(f, th)]; b = 0.0 if (getattr(P, 'bv_on_eps', False) and th == 0.0) else P.b_eff[f]   # a0 unexposed under bv_on_eps
        W = A * np.exp(b * P.y_solve)
        lhs = (P._const_base_solve + th) * W - P._QyQ_solve_ty[f] @ W
        worst = max(worst, float(np.abs(lhs - np.exp(b * P.y_solve)).max() / np.abs(np.exp(b * P.y_solve)).max()))
print(json.dumps({"worst": worst, "rq": float(getattr(P, "jump_rate_Q", 0.0))}))
"""


def bq():
    """bgn_q: BGN with the quadratic price of market risk in the rate. The bin shares are a grid recursion, so the
    Euler residual is the shares' discretisation (~1e-5), not a pricing error; at q_price = 0 it is 1e-14."""
    bvy("bq", "bgn_q", euler_tol=1e-4, done="BGN Q VALIDATION DONE")


def kpj():
    kp("kpj", "kp_vj")


def exit_check(a, name):
    """A GS21 stage must exit on tolerance. The one exception is explicit: a stage named in
    BOP_ACCEPT_CAPPED (comma-separated) may exit cycle-averaged -- the solver's answer when the
    discrete investment policy sits in a limit cycle, so that no sweep count or tolerance changes
    the outcome (sol_gvy2sj: identical residuals on two runs) -- and is then accepted on its
    last-sweep residuals under the enforced threshold plus the Euler and realised-vs-expected
    checks below, with a WARN line that the report must disclose."""
    accept = {x.strip() for x in os.environ.get("BOP_ACCEPT_CAPPED", "").split(",") if x.strip()}
    if a.get("exit") != "tolerance" and name in accept:
        print(f"  WARN  {name}: exit '{a.get('exit')}' at sweep {a.get('sweeps')} ACCEPTED via BOP_ACCEPT_CAPPED "
              f"(cycle-averaged policy; residuals and Euler checks still enforced) -- DISCLOSE", flush=True)
        return
    check(a.get("exit") == "tolerance",
          f"{name}: exit '{a.get('exit')}' at sweep {a.get('sweeps')} (cap 5600 = cycle-averaged, NOT converged)")


def gs():
    s = spec("gs"); d = os.path.join(VARIANTS, "gs_bx")
    print(f"== gs_bx  {s['spec_id']}", flush=True)
    for st in s["solve"]["stages"]:
        name = st["name"]; sid = s["expected_solves"][name]
        m = manifest(sid, name)
        if not m:
            continue
        a = m.get("achieved") or {}
        exit_check(a, name)
        thr = a.get("threshold_enforced") or 0
        check(a.get("qerr_rel", 1) < thr and a.get("perr_rel", 1) < thr,
              f"{name}: qerr_rel {a.get('qerr_rel'):.2e}, perr_rel {a.get('perr_rel'):.2e} under the enforced {thr:.0e}")
        emb = solstamp.embedded_solve_id(os.path.join(d, name, "solution.npz"))
        check(emb == sid, f"{name}: solution.npz embeds solve_id {emb}")
    env = dict(s["env"])
    out = rnp._run(rnp._GS, d, env)
    check(out["worst"] < 1e-10, f"E[M R] = 1 on the five-type economy: max |E[MR] - 1| = {out['worst']:.2e} (< 1e-10)")
    r = run([sys.executable, "-W", "ignore", "validate_gs_bx.py"], d, env)
    check(r.returncode == 0 and "GSBX VALIDATION DONE" in r.stdout, "validate_gs_bx.py ran to completion")
    mm = re.search(r"max \|E\[MR\]-1\|: ([-\d.eE+]+)", r.stdout)
    check(bool(mm) and float(mm.group(1)) < 1e-10, f"validate_gs_bx.py Euler error {mm.group(1) if mm else '?'} on N=150 (< 1e-10)")


def bvy(which="bvy", model="bgn_vy", euler_tol=1e-6, done="BGN VY VALIDATION DONE"):
    """bgn_vy: the y-factor BGN. Manifest + DP exit, the project-value Euler equation WITH y, the J*
    fixed-point residual, realised vs expected by regime AND by type, PSD -- all from validate_bgn_vy.py."""
    s = spec(which); ov = {"BGN_PARAM_OVERRIDES": json.dumps(s["params"])}
    d = os.path.join(VARIANTS, model)
    print(f"== {model}  {s['spec_id']}", flush=True)
    m = manifest(s["expected_solves"]["jstar"], "jstar")
    if m:
        a = m.get("achieved") or {}
        check(a.get("exit") == "tolerance", f"jstar: exit '{a.get('exit')}' after {a.get('iters')} iterations at tol {a.get('tol_requested')}, grid {a.get('grid')}")
    r = run([sys.executable, "-W", "ignore", "validate_bgn_vy.py"], d, ov)
    check(r.returncode == 0 and done in r.stdout, "validate_bgn_vy.py ran to completion")
    mm = re.search(r"Euler equation with y: worst ([-\d.eE+]+)", r.stdout)
    check(bool(mm) and float(mm.group(1)) < euler_tol, f"project-value Euler equation with y at these parameters: worst {mm.group(1) if mm else '?'} (< {euler_tol:g})")
    mm = re.search(r"limiting term spread ([-\d.eE+]+)", r.stdout)
    check(bool(mm) and 0.021 < float(mm.group(1)) < 0.026, f"limiting term spread {mm.group(1) if mm else '?'}/yr (BGN 1999: 2.4%)")
    mm = re.search(r"fixed-point residual at the reported nodes: ([-\d.eE+]+)", r.stdout)
    check(bool(mm) and float(mm.group(1)) < 1e-5, f"J* fixed-point residual {mm.group(1) if mm else '?'} (< 1e-5)")
    rows = re.findall(r"regime (\d): realized mean excess ([-\d.eE+]+)\s+expected ([-\d.eE+]+)\s+\(n=(\d+), se ([-\d.eE+]+)\)", r.stdout)
    for g, re_, ee, n, se in rows:
        z = (float(re_) - float(ee)) / float(se)
        check(abs(z) < Z_MAX, f"regime {g}: realised {float(re_):+.5f} vs expected {float(ee):+.5f}, z {z:+.2f} (n={n})")
    rows = re.findall(r"type (\d+) \(bv=[^)]*\): realized ([-\d.eE+]+)\s+expected ([-\d.eE+]+)\s+\(se ([-\d.eE+]+)\)", r.stdout)
    check(len(rows) == len(s["params"]["type_bv"]), f"all {len(s['params']['type_bv'])} types present in the validation panel")
    for f, re_, ee, se in rows:
        z = (float(re_) - float(ee)) / float(se)
        check(abs(z) < Z_MAX, f"type {f}: realised {float(re_):+.5f} vs expected {float(ee):+.5f}, z {z:+.2f}")
    eig = [float(x) for x in re.findall(r"min eig ([-\d.eE+]+)", r.stdout)]
    check(eig and min(eig) > 0, f"conditional covariance positive definite (min eig {min(eig) if eig else float('nan'):.2e})")


def gvy():
    """gs_vy: the y-factor GS21. Per-type manifests + tolerance exits + embedded ids, E[M R] = 1 on the typed
    economy, realised vs expected by type, PSD -- from validate_gs_vy.py."""
    s = spec("gvy"); d = os.path.join(VARIANTS, "gs_vy")
    print(f"== gs_vy  {s['spec_id']}", flush=True)
    for st in s["solve"]["stages"]:
        name = st["name"]; sid = s["expected_solves"][name]
        m = manifest(sid, name)
        if not m:
            continue
        a = m.get("achieved") or {}
        exit_check(a, name)
        thr = a.get("threshold_enforced") or 0
        check(a.get("qerr_rel", 1) < thr and a.get("perr_rel", 1) < thr,
              f"{name}: qerr_rel {a.get('qerr_rel'):.2e}, perr_rel {a.get('perr_rel'):.2e} under the enforced {thr:.0e}")
        emb = solstamp.embedded_solve_id(os.path.join(d, name, "solution.npz"))
        check(emb == sid, f"{name}: solution.npz embeds solve_id {emb}")
    env = dict(s["env"])
    r = run([sys.executable, "-W", "ignore", "validate_gs_vy.py"], d, env)
    check(r.returncode == 0 and "GSVY VALIDATION DONE" in r.stdout, "validate_gs_vy.py ran to completion")
    mm = re.search(r"max \|E\[MR\]-1\|: ([-\d.eE+]+)", r.stdout)
    check(bool(mm) and float(mm.group(1)) < 1e-10, f"E[M R] = 1 on the typed economy: max |E[MR] - 1| = {mm.group(1) if mm else '?'} (< 1e-10)")
    rows = re.findall(r"type (\d+) \(bv=[^)]*\): realized ([-\d.eE+]+)\s+expected ([-\d.eE+]+)\s+\(se ([-\d.eE+]+)\)", r.stdout)
    check(len(rows) == len(s["types"]["soldirs"]), f"all {len(s['types']['soldirs'])} types present in the validation panel")
    for f, re_, ee, se in rows:
        z = (float(re_) - float(ee)) / float(se)
        check(abs(z) < Z_MAX, f"type {f}: realised {float(re_):+.5f} vs expected {float(ee):+.5f}, z {z:+.2f}")
    eig = [float(x) for x in re.findall(r"min eig ([-\d.eE+]+)", r.stdout)]
    check(eig and min(eig) > 0, f"conditional covariance positive definite (min eig {min(eig) if eig else float('nan'):.2e})")


if __name__ == "__main__":
    for which in sys.argv[1:] or ["bgn", "kp", "gs"]:
        {"bgn": bgn, "kp": kp, "gs": gs, "bvy": bvy, "gvy": gvy, "kpj": kpj, "bq": bq}[which]()
    if fails:
        print(f"\nSOLVE VERIFICATION FAILED ({len(fails)}):\n  " + "\n  ".join(fails), flush=True)
        sys.exit(1)
    print("\nSOLVE VERIFICATION PASSED", flush=True)
