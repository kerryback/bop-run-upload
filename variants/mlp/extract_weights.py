"""
Rebuild the per-firm portfolio weights behind one seed of a campaign economy, so the
MLP-analyze pipeline (external/MLP Analyze) can fit weight = f(characteristics):

  w_true   the oracle's conditional mean-variance (true SDF) weights, from *_moments_*.npz
  w_fm     Fama-MacBeth factor weights x OLS MVE theta over the trailing window
  w_ff     Fama-French 5-factor + VW market weights x OLS MVE theta
  w_dkkm   DKKM random-Fourier ridge, P=3600, unpenalised EW market column, ensemble of the
           --nmat W draws (run_estimators' method "rff_ens"), at one --kappa

Everything is recomputed exactly as variants/run_estimators.py does it (same rng stream,
same helper functions), and the result is CHECKED against the recorded per-month conditional
Sharpe ratios in *_estimators_<tag>_s<seed>_w<window>.csv before anything is written.

usage: python extract_weights.py --model kp_vy --tag vyg25 --seed 0 --kappa 0.001
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

REPO = os.environ.get("BOP_REPO", "/kellogg/proj/axa3203/bop-run-upload")
sys.path.insert(0, os.path.join(REPO, "variants", "common"))
import dkkm_functions as dkkm
import fama_functions as fama

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--kappa", type=str, required=True, help="DKKM ridge penalty, or 'auto' = the kappa whose rff_ens P=3600 mean Sharpe over the economy's seeds is highest")
ap.add_argument("--P", type=int, default=3600)
ap.add_argument("--window", type=int, default=360)
ap.add_argument("--rff", type=str, default="36,360,3600")
ap.add_argument("--nmat", type=int, default=2)
ap.add_argument("--n_jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
ap.add_argument("--results", default=os.environ.get("BOP_RESULTS_DIR", "/kellogg/proj/axa3203/bop_runs/results"))
ap.add_argument("--out", default="/kellogg/proj/axa3203/bop_runs/mlp/data")
ap.add_argument("--tol", type=float, default=1e-6, help="max |Sharpe diff| vs the recorded estimator csv")
args = ap.parse_args()
if args.kappa == "auto":
    import glob
    fs = sorted(glob.glob(os.path.join(args.results, f"{args.model}_estimators_{args.tag}_s[0-9][0-9][0-9]_w{args.window}_summary.csv")))
    sm = pd.concat([pd.read_csv(f) for f in fs])
    sm = sm[(sm.method == "rff_ens") & (sm.P == args.P)].groupby("kappa").sharpe.mean()
    args.kappa = float(sm.idxmax())
    print(f"[kappa] auto over {len(fs)} seeds: {sm.round(4).to_dict()} -> {args.kappa}", flush=True)
else:
    args.kappa = float(args.kappa)

dkkm.WINDOW = args.window; fama.WINDOW = args.window
rng = np.random.default_rng(args.seed + 7)            # run_estimators.py: default_rng(args.seed + 7)
chars = ["size", "bm", "agr", "roe", "mom"] + (["lev"] if args.model in ("gs_vy", "gs_bx") else [])   # run_estimators.py, 2026-10-26
gamma_grid = np.arange(0.5, 1.1, 0.1)
stem = f"{args.model}_%s_{args.tag}_s{args.seed:03d}"
R = args.results
t0 = time.time()

panel = pd.read_parquet(os.path.join(R, stem % "panel" + ".parquet"))
mom = np.load(os.path.join(R, stem % "moments" + ".npz"))
osum = json.load(open(os.path.join(R, stem % "oracle" + ".json")))
rec = pd.read_csv(os.path.join(R, stem % "estimators" + f"_w{args.window}.csv"))

_RF_ALL = {"bgn_gam": ["rf_stand", "gam_stand"], "gs_bx": ["rf_stand", "gam_stand"], "kp_vy": ["rf_stand"],
           "bgn_vy": ["rf_stand", "y_stand", "y2_stand"], "gs_vy": ["rf_stand", "y_stand", "y2_stand"],
           "kp_vj": ["rf_stand"], "bgn_q": ["rf_stand", "y_stand", "y2_stand"]}
dkkm.RF_COLS = [c for c in _RF_ALL[args.model] if c in panel.columns]
assert list(dkkm.RF_COLS) == list(osum["rf_cols"]), (dkkm.RF_COLS, osum["rf_cols"])
RFC = list(dkkm.RF_COLS)

months_m, MU, SIG, KEEP, WT = mom["months"], mom["mu"], mom["Sigma"], mom["keep"], mom["w_true"]
midx = {int(m): i for i, m in enumerate(months_m)}
panel = panel.set_index(["month", "firmid"]).sort_index()
start, end = int(months_m.min()), int(months_m.max())
eval_months = [m for m in range(start + args.window, end + 1) if m in midx]
print(f"[{args.model}/{args.tag}/s{args.seed}] panel {panel.shape}, months {start}-{end}, "
      f"{len(eval_months)} eval months, rf_cols={RFC}", flush=True)

# ---- factor return histories (identical call order to run_estimators.py) ----
ff_rets = fama.factors(fama.fama_french, panel, n_jobs=args.n_jobs, start=start, end=end, chars=chars)
fm_rets = fama.factors(fama.fama_macbeth, panel, n_jobs=args.n_jobs, start=start, end=end, chars=chars)
Plist = [int(p) for p in args.rff.split(",") if p]
Pmax = max(Plist); half = Pmax // 2
assert args.P in Plist
Ws, frets = [], []
for i in range(args.nmat):
    W = rng.standard_normal(size=(half, len(chars) + len(dkkm.RF_COLS)))
    W = rng.choice(gamma_grid, size=(half, 1)) * W
    Ws.append(W)
    frets.append(dkkm.factors(panel=panel, W=W, n_jobs=args.n_jobs, start=start, end=end, model="bgn", chars=chars)[0])
mkt_rets = panel.groupby("month").xret.mean()
print(f"factor histories built in {time.time()-t0:.0f}s", flush=True)

def sharpe(w, mu, Sigma):
    return float(w @ mu / np.sqrt(max(w @ Sigma @ w, 1e-300)))

num = args.P // 2
nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])
rec_idx = rec.set_index(["method", "P", "kappa", "mat", "month"]).sharpe
checks = {"fm": [], "ff": [], "dkkm": []}
parts = []
extra = [c for c in ["mve", "xret", "lev", "ftype", "default"] + RFC if c in panel.columns]
for k, month in enumerate(eval_months):
    i = midx[month]
    data = panel.loc[month]
    keep = data.index.to_numpy(); n = len(keep)
    assert np.array_equal(KEEP[i, :n], keep)
    mu, Sigma = MU[i, :n].astype(float), SIG[i, :n, :n].astype(float)
    out = pd.DataFrame({"month": month, "firmid": keep}, index=data.index)
    for c in chars + extra:
        out[c] = data[c].to_numpy()
    out["w_true"] = WT[i, :n]
    for name, fr, meth in [("ff", ff_rets, fama.fama_french), ("fm", fm_rets, fama.fama_macbeth)]:
        fw = meth(data[chars], chars, mve=data.mve)
        theta = fama.mve_data(fr, month, 0)
        w = fw @ theta.to_numpy()
        out[f"w_{name}"] = w
        checks[name].append(abs(sharpe(w, mu, Sigma) - float(rec_idx[(name, fw.shape[1], 0.0, 0, month)])))
    ens = np.zeros(n)
    for W, fr in zip(Ws, frets):
        fw = dkkm.rff(data[chars], data[RFC], W=W[:num, :], model="bgn")[0]
        fw.columns = [str(ind) for ind in nf_indx]
        fw["mkt_rf"] = 1.0 / len(fw)
        thetas = dkkm.mve_data(fr.iloc[:, nf_indx], month, args.P * np.array([0.0, args.kappa]), mkt_rets)
        thetas = thetas.iloc[:, 1:]
        ens += (fw @ thetas.iloc[:, 0]).to_numpy() / len(Ws)
    out["w_dkkm"] = ens
    checks["dkkm"].append(abs(sharpe(ens, mu, Sigma) - float(rec_idx[("rff_ens", args.P, args.kappa, -1, month)])))
    parts.append(out.reset_index(drop=True))
    if k % 25 == 0:
        print(f"  month {month} ({k+1}/{len(eval_months)}) {time.time()-t0:.0f}s  "
              f"maxdiff fm {max(checks['fm']):.2e} ff {max(checks['ff']):.2e} dkkm {max(checks['dkkm']):.2e}", flush=True)

maxd = {k: float(max(v)) for k, v in checks.items()}
print(f"[check] max |Sharpe(rebuilt) - Sharpe(recorded)| per month: {maxd}", flush=True)
if max(maxd.values()) > args.tol:
    raise SystemExit(f"REBUILT WEIGHTS DO NOT REPRODUCE THE RECORDED RESULTS (tol {args.tol}); nothing written")
df = pd.concat(parts, ignore_index=True)
os.makedirs(args.out, exist_ok=True)
fn = os.path.join(args.out, f"wts_{args.model}_{args.tag}_s{args.seed:03d}.parquet")
df.to_parquet(fn)
json.dump({"model": args.model, "tag": args.tag, "seed": args.seed, "P": args.P, "kappa": args.kappa,
           "window": args.window, "nmat": args.nmat, "rf_cols": RFC, "spec_id": osum.get("spec_id"),
           "eval_months": len(eval_months), "rows": int(len(df)), "check_max_sharpe_diff": maxd,
           "prov": str(rec.prov.iloc[0]) if "prov" in rec else None},
          open(fn.replace(".parquet", ".json"), "w"), indent=1)
print(f"wrote {fn} ({len(df):,} rows) in {time.time()-t0:.0f}s  WEIGHTS OK", flush=True)
