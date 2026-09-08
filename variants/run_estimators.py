"""
Small-scale version of the paper's estimators (FMR, FFC, DKKM random-Fourier ridge) on a saved panel,
evaluated with the TRUE conditional moments saved by run_oracle.py --save_panel.

usage: python run_estimators.py --model bgn --tag baseline --window 120 --rff 36,360,3600
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import scipy.linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
import dkkm_functions as dkkm
import fama_functions as fama
import runstamp
import provenance

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["bgn_gam", "kp_vy", "gs_bx"], required=True)
ap.add_argument("--tag", type=str, default="baseline")
ap.add_argument("--window", type=int, default=120)
ap.add_argument("--rff", type=str, default="36,360,3600")
ap.add_argument("--nmat", type=int, default=2)
ap.add_argument("--kappas", type=str, default="0.001,0.01,0.05,0.1,1")
ap.add_argument("--seed", type=int, default=0,
                help="replication seed; must match the run_oracle.py --seed whose panel this reads")
ap.add_argument("--n_jobs", type=int, default=6)
ap.add_argument("--include_mkt", action="store_true", help="append the unpenalised EW market to the RFF factor sets (as in the paper)")
ap.add_argument("--levels", action="store_true", help="add rff_lev / linlev methods using fixed-stats level features")
ap.add_argument("--winsor", type=float, default=0.0, help="if >0, winsorise xret at +/- this level BEFORE the estimators see it (evaluation still uses the true moments)")
args = ap.parse_args()
dkkm.WINDOW = args.window; fama.WINDOW = args.window
rng = np.random.default_rng(args.seed + 7)

chars = ["size", "bm", "agr", "roe", "mom"]
gamma_grid = np.arange(0.5, 1.1, 0.1)
# BOP_RESULTS_DIR relocates output without touching code. Default unchanged.
# Measured 2026-09-08: /data/sjpruitt is 1.0 TB with 989 GB free (4% used), and a full
# 10-seed x 2-economy flagship campaign adds ~5.5 GB -- 0.6% of free space. So there is
# no case for symlinking intermediates to scratch today, and scratch is PURGED, which
# would cost 2.3 h to regenerate a panel you wanted to re-run estimators against at a
# different window (a thing already done three times on 2026-09-07). This exists so
# that if /data ever does get tight, moving output is a flag rather than a refactor.
out = os.environ.get("BOP_RESULTS_DIR") or os.path.join(HERE, "results")
_st = lambda kind: os.path.join(out, runstamp.stem(args.model, kind, args.tag, args.seed))
_panel_path = _st("panel") + ".parquet"
if not os.path.exists(_panel_path):
    legacy = os.path.join(out, f"{args.model}_panel_{args.tag}.parquet")
    raise SystemExit(
        f"no panel at {_panel_path}.\n"
        f"Run:  python run_oracle.py --model {args.model} --tag {args.tag} "
        f"--seed {args.seed} --save_panel\n"
        + (f"An UNSEEDED panel exists at {legacy}. It is NOT adopted as seed "
           f"{args.seed}: everything written before 2026-09-07 came from the "
           f"pre-correction parameters (WORKING.md \u00a725) and silently treating it "
           f"as a replication would mix two economies. Re-run the oracle, or rename "
           f"the file deliberately if you know what it is.\n"
           if os.path.exists(legacy) else ""))
panel = pd.read_parquet(_panel_path)
mom = np.load(_st("moments") + ".npz")

# The oracle summary records which solve built this panel; carry that forward so the
# estimator output is traceable to the same economy without re-deriving it.
_oracle_summary = _st("oracle") + ".json"
_solves = None
if os.path.exists(_oracle_summary):
    _solves = json.load(open(_oracle_summary)).get("solves")
    if _solves:
        print(runstamp.describe(_solves), flush=True)
months_m, MU, SIG, KEEP = mom["months"], mom["mu"], mom["Sigma"], mom["keep"]
midx = {int(m): i for i, m in enumerate(months_m)}
panel = panel.set_index(["month", "firmid"]).sort_index()
panel["xret_true"] = panel.xret
if args.winsor > 0:
    panel["xret"] = panel.xret.clip(-args.winsor, args.winsor)
    print(f"winsorised xret at +/-{args.winsor}: {np.mean(np.abs(panel.xret_true) > args.winsor):.5f} of observations clipped", flush=True)
N = int(panel.index.get_level_values("firmid").max()) + 1
start, end = int(months_m.min()), int(months_m.max())
eval_months = [m for m in range(start + args.window, end + 1) if m in midx]
t0 = time.time()
model_for_rff = "bgn"   # all three economies pass conditioning variables to the RFF (see common/dkkm_functions.py)
if args.model in ("bgn_gam", "gs_bx"):
    dkkm.RF_COLS = ["rf_stand", "gam_stand"]
RFC = list(dkkm.RF_COLS)   # capture by value for closures shipped to loky workers

# ---- factor return histories ------------------------------------------------------------------------
ff_rets = fama.factors(fama.fama_french, panel, n_jobs=args.n_jobs, start=start, end=end, chars=chars)
fm_rets = fama.factors(fama.fama_macbeth, panel, n_jobs=args.n_jobs, start=start, end=end, chars=chars)
Plist = [int(p) for p in args.rff.split(",") if p]
Pmax = max(Plist); half = Pmax // 2
Ws, frets = [], []
for i in range(args.nmat):
    W = rng.standard_normal(size=(half, len(chars) + (len(dkkm.RF_COLS) if model_for_rff == "bgn" else 0)))
    W = rng.choice(gamma_grid, size=(half, 1)) * W
    Ws.append(W)
    frets.append(dkkm.factors(panel=panel, W=W, n_jobs=args.n_jobs, start=start, end=end, model=model_for_rff, chars=chars)[0])
# linear-in-rank-characteristics factors (constant + 5 rank-standardised chars), estimated by ridge like DKKM
def lin_rank_weights(data):
    X = dkkm.rank_standardize(data[chars])
    X.insert(0, "const", 1.0 / len(X))
    X.columns = [str(i) for i in range(X.shape[1])]
    return X
lr_rets = pd.concat([(lin_rank_weights(panel.loc[m]).T @ panel.loc[m].xret).rename(m) for m in range(start, end + 1)], axis=1).T
lr_rets.index.name = "month"
mkt_rets = panel.groupby("month").xret.mean() if args.include_mkt else None
lev_stats = None
if args.levels:
    first = panel.loc[start:start + args.window - 1]
    med = first[chars].median().to_numpy(copy=True)
    # copy=True is required, not defensive: pandas can hand back a read-only view here,
    # and the next line mutates it ("assignment destination is read-only"). The --levels
    # path is used by every shipped run script, so this raised on every levels run once
    # pandas started returning a view.
    iqr = (first[chars].quantile(0.75) - first[chars].quantile(0.25)).to_numpy(copy=True)
    iqr[iqr == 0] = 1.0
    lev_stats = (med, iqr)
    def lev_X(data):
        Xr = dkkm.rank_standardize(data[chars]).to_numpy()
        Xl = np.clip((data[chars].to_numpy() - med) / iqr, -3, 3)
        cols = [Xr, Xl]
        if model_for_rff == "bgn":
            for c in RFC:
                cols.append(np.full((len(Xr), 1), float(data[c].iloc[0])))
        return np.column_stack(cols)
    WsL, fretsL = [], []
    for i in range(args.nmat):
        WL = rng.standard_normal(size=(half, 2 * len(chars) + (len(dkkm.RF_COLS) if model_for_rff == "bgn" else 0)))
        WL = rng.choice(gamma_grid, size=(half, 1)) * WL
        WsL.append(WL)
        def month_lev(mth, WL=WL):
            data = panel.loc[mth]
            Z = lev_X(data) @ WL.T
            F = dkkm.rank_standardize(pd.DataFrame(np.column_stack([np.sin(Z), np.cos(Z)]), index=data.index))
            return (F.T @ data.xret).rename(mth)
        from joblib import Parallel, delayed
        fr = pd.concat(Parallel(n_jobs=args.n_jobs)(delayed(month_lev)(mth) for mth in range(start, end + 1)), axis=1).T
        fr.index.name = "month"; fr.columns = [str(c) for c in fr.columns]
        fretsL.append(fr)
    # linear-in-(ranks+levels) factors
    def lin_lev_weights(data):
        X = pd.DataFrame(np.column_stack([np.full(len(data), 1.0 / len(data)),
                                          dkkm.rank_standardize(data[chars]).to_numpy(),
                                          np.clip((data[chars].to_numpy() - med) / iqr, -3, 3)]), index=data.index)
        X.columns = [str(i) for i in range(X.shape[1])]
        return X
    ll_rets = pd.concat([(lin_lev_weights(panel.loc[mth]).T @ panel.loc[mth].xret).rename(mth) for mth in range(start, end + 1)], axis=1).T
    ll_rets.index.name = "month"
print(f"factor histories built in {time.time()-t0:.0f}s", flush=True)
kappas = [float(k) for k in args.kappas.split(",")]

def evaluate(w, mu, Sigma, xret):
    # a month where the estimator itself blew up (e.g. FMR on explosive raw levels) scores NaN
    if not np.all(np.isfinite(w)):
        return {"mn": np.nan, "stdev": np.nan, "sharpe": np.nan, "xret": np.nan, "hjd": np.nan}
    mean = w @ mu; sd = np.sqrt(max(w @ Sigma @ w, 1e-300))
    M2 = Sigma + np.outer(mu, mu)
    errs = mu - M2 @ w
    try:
        hjd = errs @ la.solve(M2, errs, assume_a="pos")
    except (ValueError, la.LinAlgError):
        hjd = np.nan
    return {"mn": mean, "stdev": sd, "sharpe": mean / sd, "xret": w @ xret, "hjd": hjd}

rows = []
for k, month in enumerate(eval_months):
    i = midx[month]
    data = panel.loc[month]
    keep = data.index.to_numpy()
    n = len(keep)
    assert np.array_equal(KEEP[i, :n], keep)
    mu, Sigma, xret = MU[i, :n].astype(float), SIG[i, :n, :n].astype(float), data.xret_true.to_numpy()
    # classical
    for name, fr, meth in [("ff", ff_rets, fama.fama_french), ("fm", fm_rets, fama.fama_macbeth)]:
        fw = meth(data[chars], chars, mve=data.mve)
        theta = fama.mve_data(fr, month, 0)
        w = fw @ theta.to_numpy()
        rows.append({"month": month, "method": name, "P": fw.shape[1], "kappa": 0.0, "mat": 0, **evaluate(w, mu, Sigma, xret)})
    # linear in rank-standardised chars + EW market, ridge (KPS-6 / BSV-linear with shrinkage)
    fw = lin_rank_weights(data)
    thetas = dkkm.mve_data(lr_rets, month, 6 * np.array([0.0] + kappas))
    for j, kap in enumerate([0.0] + kappas):
        w = (fw @ thetas.iloc[:, j]).to_numpy()
        rows.append({"month": month, "method": "linrank", "P": 6, "kappa": kap, "mat": 0, **evaluate(w, mu, Sigma, xret)})
    # DKKM
    rf = data[RFC] if model_for_rff == "bgn" else None
    ens = {}
    for mi, (W, fr) in enumerate(zip(Ws, frets)):
        for P in Plist:
            num = P // 2
            nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])
            fw = dkkm.rff(data[chars], rf, W=W[:num, :], model=model_for_rff)[0]
            fw.columns = [str(ind) for ind in nf_indx]
            if args.include_mkt:
                fw["mkt_rf"] = 1.0 / len(fw)
                thetas = dkkm.mve_data(fr.iloc[:, nf_indx], month, P * np.array([0.0] + kappas), mkt_rets)
                thetas = thetas.iloc[:, 1:]         # drop the unpenalised column; keep the kappa grid
                thetas.columns = kappas
            else:
                thetas = dkkm.mve_data(fr.iloc[:, nf_indx], month, P * np.array(kappas))
            for j, kap in enumerate(kappas):
                w = (fw @ thetas.iloc[:, j]).to_numpy()
                rows.append({"month": month, "method": "rff", "P": P, "kappa": kap, "mat": mi, **evaluate(w, mu, Sigma, xret)})
                ens[(P, kap)] = ens.get((P, kap), 0) + w / len(Ws)
    if len(Ws) > 1:   # ensemble over RFF draws (the paper averages the SDF estimates across draws)
        for (P, kap), w in ens.items():
            rows.append({"month": month, "method": "rff_ens", "P": P, "kappa": kap, "mat": -1, **evaluate(w, mu, Sigma, xret)})
    if args.levels:
        # linear in ranks + level features, ridge
        fw = lin_lev_weights(data)
        thetas = dkkm.mve_data(ll_rets, month, 11 * np.array([0.0] + kappas))
        for j, kap in enumerate([0.0] + kappas):
            w = (fw @ thetas.iloc[:, j]).to_numpy()
            rows.append({"month": month, "method": "linlev", "P": fw.shape[1], "kappa": kap, "mat": 0, **evaluate(w, mu, Sigma, xret)})
        # RFF on ranks + levels (+rf)
        ensL = {}
        for mi, (WL, fr) in enumerate(zip(WsL, fretsL)):
            for P in Plist:
                num = P // 2
                nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])
                Z = lev_X(data) @ WL[:num, :].T
                fw = dkkm.rank_standardize(pd.DataFrame(np.column_stack([np.sin(Z), np.cos(Z)]), index=data.index))
                fw.columns = [str(ind) for ind in nf_indx]
                if args.include_mkt:
                    fw["mkt_rf"] = 1.0 / len(fw)
                    thetas = dkkm.mve_data(fr.iloc[:, nf_indx], month, P * np.array([0.0] + kappas), mkt_rets)
                    thetas = thetas.iloc[:, 1:]
                    thetas.columns = kappas
                else:
                    thetas = dkkm.mve_data(fr.iloc[:, nf_indx], month, P * np.array(kappas))
                for j, kap in enumerate(kappas):
                    w = (fw @ thetas.iloc[:, j]).to_numpy()
                    rows.append({"month": month, "method": "rff_lev", "P": P, "kappa": kap, "mat": mi, **evaluate(w, mu, Sigma, xret)})
                    ensL[(P, kap)] = ensL.get((P, kap), 0) + w / len(WsL)
        if len(WsL) > 1:
            for (P, kap), w in ensL.items():
                rows.append({"month": month, "method": "rff_lev_ens", "P": P, "kappa": kap, "mat": -1, **evaluate(w, mu, Sigma, xret)})
    if k % 50 == 0:
        print(f"  month {month} ({k+1}/{len(eval_months)}) {time.time()-t0:.0f}s", flush=True)

tagw = f"_wins{args.winsor}" if args.winsor > 0 else ""
_base = _st("estimators") + f"_w{args.window}{tagw}"
res = pd.DataFrame(rows)
res.to_csv(_base + ".csv", index=False)
summ = res.groupby(["method", "P", "kappa"]).agg(sharpe=("sharpe", "mean"), hjd=("hjd", lambda x: np.sqrt(x.mean())),
                                                   real_sr=("xret", lambda x: x.mean() / x.std())).reset_index()
fm_sr = res[res.method == "fm"].groupby("month").sharpe.mean()
# paired t-stat of monthly conditional SR vs FMR
tstats = []
for (m, P, kap), g in res.groupby(["method", "P", "kappa"]):
    d = g.groupby("month").sharpe.mean() - fm_sr
    tstats.append(d.mean() / d.std() * np.sqrt(d.count()))
summ["t_vs_fm"] = tstats
pd.set_option("display.width", 200)
print(f"\n=== {args.model}/{args.tag}: window={args.window}, eval months={len(eval_months)}, N={N}")
print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
_prov, _ptag = provenance.write_sidecar(
    _base + "_summary.csv", inputs=_solves,
    extra={"engine": "estimators", "window": args.window,
           "panel": os.path.basename(_panel_path)})
summ["prov"] = _ptag           # every row carries it; see provenance.short_tag
res["prov"] = _ptag
res.to_csv(_base + ".csv", index=False)          # rewritten so the full results carry it too
summ.to_csv(_base + "_summary.csv", index=False)
print(f"[prov] {_ptag}  -> {os.path.basename(_base)}_summary.csv.prov.json", flush=True)
json.dump({"model": args.model, "tag": args.tag, "seed": args.seed, "prov": _ptag,
           "window": args.window, "winsor": args.winsor, "kappas": kappas,
           "eval_months": len(eval_months), "N": N,
           "solves": _solves,
           "panel": os.path.basename(_panel_path)},
          open(_base + "_run.json", "w"), indent=1)
print(f"done in {time.time()-t0:.0f}s")
