"""
Oracle decomposition of the achievable Sharpe ratio for the BGN or KP14 economy (see common/oracle.py).

usage:
  python run_oracle.py --model bgn --N 300 --T 360 --tag baseline
  BGN_PARAM_OVERRIDES='{"sigma_r":0.004}' python run_oracle.py --model bgn --tag sigma_r4
  KP_PARAM_OVERRIDES='{"gamma_z":-0.7}'   python run_oracle.py --model kp  --tag gz7
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["bgn_gam", "kp_vy", "gs_bx"], required=True)
ap.add_argument("--N", type=int, default=300)
ap.add_argument("--T", type=int, default=360, help="months after burn-in")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", type=str, default="baseline")
ap.add_argument("--rff", type=str, default="36,360,3600")
ap.add_argument("--nmat", type=int, default=2, help="independent RFF draws")
ap.add_argument("--no_rf", action="store_true", help="BGN: do not feed the interest rate to the feature bases")
ap.add_argument("--save_panel", action="store_true")
ap.add_argument("--levels", action="store_true", help="add fixed-stats level features (and rffL bases)")
ap.add_argument("--spec", type=str, default=None,
                help="spec_id this run implements (experiments/specs/<id>.json). The run "
                     "VERIFIES its consumed solve_ids against the spec's expected_solves "
                     "and aborts on a mismatch, so a summary can never claim a spec it did "
                     "not actually build.")
ap.add_argument("--common_aggregate", action="store_true",
                help="hold the AGGREGATE state path fixed across seeds (do not offset gam_seed/reg_seed). "
                     "Cross-seed spread then isolates firm-level sampling noise, with the aggregate "
                     "contribution removed. This was the unintended behaviour of every seed before "
                     "2026-09-07; it is a legitimate variance decomposition, and a wrong default.")
args = ap.parse_args()

os.chdir(os.path.join(HERE, args.model))
sys.path.insert(0, os.getcwd())
np.random.seed(args.seed)
rng = np.random.default_rng(args.seed + 1000)

if args.model == "bgn_gam":
    from parameters import *              # noqa
    import panel_functions as mod
    import sdf_compute as sdf
    ov_env = "BGN_PARAM_OVERRIDES"
elif args.model == "kp_vy":
    from parameters_kp14 import *         # noqa
    import panel_functions_kp14 as mod
    import sdf_compute_kp14 as sdf
    chars = ["size", "bm", "agr", "roe", "mom"]; gamma_grid = np.arange(0.5, 1.1, 0.1)
    ov_env = "KP_PARAM_OVERRIDES"
else:  # gs_bx
    import gs_sim_bx as mod
    import gs_sim_bx as sdf
    from gs_sim_bx import burnin, chars, gamma_grid
    ov_env = "GS_SIM_OVERRIDES"
from oracle import rank_standardize, draw_W, build_feature_sets, evaluate_bases, max_sr, level_standardize
import runstamp

# ---- the AGGREGATE state path must move with the seed --------------------------------------------
# np.random.seed(args.seed) above steers every draw that goes through the global stream
# (norm.rvs, expon.rvs, np.random.*) -- the firm-level shocks. It does NOT steer the
# aggregate price-of-risk path, which each economy draws from its own generator seeded by
# a module constant: bgn_gam's `sreg` regime chain and kp_vy's `yreg` OU path from
# gam_seed = 555, gs_bx's from reg_seed = 909.
#
# Left alone, every replication in a seed array would see the SAME 700-month aggregate
# path. The cross-seed spread would then measure firm-level noise only, with the
# aggregate contribution -- the channel this whole project is about -- invisible, and
# any t-stat built from it overstated.
#
# Offsetting by the seed fixes that while KEEPING common random numbers ACROSS SPECS:
# seed s draws aggregate path (base + s) in every economy, so a paired comparison
# between two specs at one seed still differs only in the parameters. Seed 0 reproduces
# the historical single-run behaviour exactly.
_agg = {}
_offset = 0 if args.common_aggregate else args.seed
for _name in ("gam_seed", "bx_seed", "reg_seed"):
    if hasattr(mod, _name):
        _base = getattr(mod, _name)
        setattr(mod, _name, int(_base) + _offset)
        _agg[_name] = {"base": int(_base), "used": int(_base) + _offset}
_agg["common_aggregate"] = bool(args.common_aggregate)
print(f"[seed {args.seed}] aggregate-state generators: "
      + (", ".join(f"{k} {v['base']}->{v['used']}" for k, v in _agg.items() if k != "common_aggregate") or "none")
      + (" (HELD COMMON across seeds: --common_aggregate)" if args.common_aggregate else "")
      + f"; global stream seeded {args.seed}", flush=True)

t0 = time.time()
N, T = args.N, args.T
arr_tuple = mod.create_arrays(N, T + burnin)
panel = mod.create_panel(N, T + burnin, arr_tuple)
sdf_loop = sdf.sdf_compute(N, T + burnin, arr_tuple)
print(f"[{args.model}/{args.tag}] panel built in {time.time()-t0:.0f}s", flush=True)

# Which recorded solve produced the tables this panel was built from. Identified by
# CONTENT, so a stale or foreign table is reported as what it actually is.
solves = runstamp.consumed_solves(args.model, mod=sys.modules[mod.__name__])
print(runstamp.describe(solves), flush=True)
if args.spec:
    _ok, _lines = runstamp.verify_against_spec(args.spec, solves)
    print("\n".join(_lines), flush=True)
    # `is not True`, NOT `is False`: verify_against_spec returns None for a spec it
    # cannot check at all, and `None is False` is False -- so the old guard let an
    # UNVERIFIABLE spec through and stamped the summary with a spec_id the run had
    # not earned. Anything short of an explicit pass must abort.
    if _ok is not True:
        raise SystemExit(
            f"ABORT: --spec {args.spec} cannot stamp this run -- it was either not "
            f"verified or not verifiable (see the [spec] lines above). Either point "
            f"--spec at the spec these solves belong to, or re-solve. Recording the "
            f"summary anyway would put a false spec_id on a real result.")

panel["size"] = np.log(panel.mve)
panel = panel[panel.month >= 2]
panel.replace([np.inf, -np.inf], np.nan, inplace=True)
panel.set_index(["month", "firmid"], inplace=True)
nans = panel[chars + ["mve", "xret"]].isnull().any(axis=1)
panel = panel.loc[nans[~nans].index]
months = panel.index.unique("month")
months = months[(months >= burnin + 14) & (months <= T + burnin - 2)]
d = len(chars)
n_rf = 2 if args.model in ("bgn_gam", "gs_bx") else 1
use_rf = not args.no_rf   # all three economies expose conditioning variables

# ---- collect true conditional moments (N x N per month) -------------------------------------------
months_data, rows = [], []
for k, month in enumerate(months):
    sdf_ret, sr_max_code, rp, cond_var, w_true = sdf_loop(month - 1)
    data = panel.loc[month]
    keep = data.index.to_numpy()
    mu = rp[keep]; Sigma = cond_var[np.ix_(keep, keep)]
    X_raw = data[chars].to_numpy()
    md = {"month": month, "mu": mu, "Sigma": Sigma, "X_raw": X_raw, "X_rank": rank_standardize(X_raw),
          "rf": ((np.array([data.rf_stand.iloc[0], data.gam_stand.iloc[0]]) if args.model in ("bgn_gam", "gs_bx")
                  else float(data.rf_stand.iloc[0])) if use_rf else None), "w_true": w_true[keep], "keep": keep}
    months_data.append(md)
    rows.append({"month": month, "n": len(keep), "sr_max": max_sr(mu, Sigma), "sr_max_code": sr_max_code,
                 "rf": float(np.atleast_1d(md["rf"])[0]) if md["rf"] is not None else None, "mean_mu": mu.mean(), "sd_mu": mu.std(),
                 "mean_idio_sd": np.sqrt(np.diag(Sigma)).mean()})
    if k % 100 == 0:
        print(f"  moments month {month} ({k+1}/{len(months)}) n={len(keep)} SRmax={rows[-1]['sr_max']:.3f}  {time.time()-t0:.0f}s", flush=True)
ts = pd.DataFrame(rows)

# ---- feature bases ---------------------------------------------------------------------------------
Plist = [int(p) for p in args.rff.split(",") if p]
Wdict = {f"rff{P}_{m}": draw_W(P, d + (n_rf if use_rf else 0), gamma_grid, rng) for P in Plist for m in range(args.nmat)}
Wdict_lev, med, iqr = None, None, None
if args.levels:
    allX = np.vstack([m["X_raw"] for m in months_data])
    med = np.median(allX, axis=0)
    iqr = np.subtract(*np.percentile(allX, [75, 25], axis=0)); iqr[iqr == 0] = 1.0
    Wdict_lev = {f"rffL{P}_{m}": draw_W(P, 2 * d + (n_rf if use_rf else 0), gamma_grid, rng) for P in Plist for m in range(args.nmat)}
def feature_fn(md):
    X_lev = level_standardize(md["X_raw"], med, iqr) if args.levels else None
    return build_feature_sets(md["X_raw"], md["X_rank"], md["rf"], Wdict, X_lev=X_lev, Wdict_lev=Wdict_lev)
names = list(feature_fn(months_data[0]).keys())

res = evaluate_bases(months_data, feature_fn, names, verbose=True)

# ---- report ----------------------------------------------------------------------------------------
def agg_rff(res):
    """average the RFF draws with the same P"""
    out = {}
    for name, r in res.items():
        key = name.split("_")[0] if name.startswith("rff") else name
        out.setdefault(key, []).append(r)
    agg = {}
    for key, lst in out.items():
        agg[key] = {"P": lst[0]["P"], "zgrid_rel": lst[0]["zgrid_rel"],
                    "cond_sr_mean": np.mean([r["cond_sr_mean"] for r in lst], 0),
                    "unc_sr": np.mean([r["unc_sr"] for r in lst], 0),
                    "cond_oracle_mean": float(np.mean([r["cond_oracle_mean"] for r in lst]))}
    return agg
agg = agg_rff(res)

summary = {"model": args.model, "tag": args.tag, "N": N, "T": T, "seed": args.seed, "overrides": os.environ.get(ov_env, "{}"),
           "solves": solves, "aggregate_seeds": _agg, "spec_id": args.spec,
           "months": len(ts), "sr_max_mean": float(ts.sr_max.mean()), "sr_max_code": float(ts.sr_max_code.mean()),
           "mean_mu": float(ts.mean_mu.mean()), "sd_mu": float(ts.sd_mu.mean()), "mean_idio_sd": float(ts.mean_idio_sd.mean()),
           "bases": {}}
print(f"\n=== {args.model}/{args.tag}: mean SR_max = {ts.sr_max.mean():.4f}  N={N} months={len(ts)}  "
      f"E[mu]={ts.mean_mu.mean():.4f} sd_cs(mu)={ts.sd_mu.mean():.4f} idio sd={ts.mean_idio_sd.mean():.3f}  overrides={summary['overrides']}")
print(f"{'basis':>12} {'P':>5} | {'cond.oracle':>11} | {'const-theta z=0':>15} | {'best z (rel)':>18} | unc SR(best)")
for name, r in agg.items():
    zs = r["zgrid_rel"]; sr = r["cond_sr_mean"]; j = int(np.argmax(sr))
    rec = {"P": r["P"], "cond_oracle": r["cond_oracle_mean"], "const_z0": float(sr[0]), "const_best": float(sr[j]),
           "best_zrel": zs[j], "unc_best": float(r["unc_sr"][j]), "sr_by_z": [float(x) for x in sr]}
    summary["bases"][name] = rec
    print(f"{name:>12} {r['P']:>5} | {r['cond_oracle_mean']:11.4f} | {sr[0]:15.4f} | {sr[j]:8.4f} ({zs[j]:7.0e}) | {r['unc_sr'][j]:.4f}")

out = os.path.join(HERE, "results")
os.makedirs(out, exist_ok=True)
_st = lambda kind: os.path.join(out, runstamp.stem(args.model, kind, args.tag, args.seed))
ts.to_csv(_st("oracle") + "_ts.csv", index=False)
json.dump(summary, open(_st("oracle") + ".json", "w"), indent=1)
if args.save_panel:
    panel.reset_index().to_parquet(_st("panel") + ".parquet")
    np.savez_compressed(_st("moments") + ".npz",
                        months=np.array([m["month"] for m in months_data]),
                        mu=np.array([np.pad(m["mu"], (0, N - len(m["mu"]))) for m in months_data]).astype(np.float32),
                        Sigma=np.array([np.pad(m["Sigma"], ((0, N - len(m["mu"])), (0, N - len(m["mu"])))) for m in months_data]).astype(np.float32),
                        w_true=np.array([np.pad(m["w_true"], (0, N - len(m["w_true"]))) for m in months_data]),
                        keep=np.array([np.pad(m["keep"], (0, N - len(m["keep"])), constant_values=-1) for m in months_data]))
print(f"done in {time.time()-t0:.0f}s")
