"""Screen one reduced-form economy: build the panel, measure the population ceilings with the
project's own oracle code, run the project's own estimators on a reduced grid, and write one JSON.

    python variants/reduced/rf_screen.py --config base --seed 0 --out /path/to/screen_dir
    python variants/reduced/rf_screen.py --params '{"J": 1}' --name notype --seed 3 ...

OFF PROTOCOL BY CONSTRUCTION. This is a screen, not an economy: it never writes under
variants/results, its grid is cut (default rff 36,360 and one feature draw), and nothing it
produces goes in docs/RESULTS.md. `--full` restores the protocol grid (rff 36,360,3600, nmat 2)
for a validation run against a real economy's numbers.

What is measured, in the record's own definitions (variants/aggregate_seeds.py):
  room        best const-theta nonlinear ceiling minus lin_rank's, over the evaluation months
  room_s      the same against lin_rank_rf -- the linear basis GIVEN the state and its
              interactions. room - room_s is the part of "room" that is linear in (chars, state).
  dkkm        best Sharpe over rff / rff_ens, any P, any kappa
  classical   dkkm - max(fm, ff)                      the DKKM-paper comparison
  fair_gap    dkkm - max(fm, ff, linrank, linrank_m, mkt_est)   finding 8's benchmark
  fair_s_gap  dkkm - max(the above, linrank_s)        linrank_s = ranks, state, ranks x state,
              unpenalised market. The linear benchmark that sees what DKKM's features see.
  theta       per-month and pooled 1 - R^2 of mu on the ranked chars (premium_shape.py), and
              their difference, the time-variation share.
The estimator loop mirrors variants/run_estimators.py method by method; the methods themselves
are imported from variants/common, not re-implemented.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import scipy.linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import dkkm_functions as dkkm
import fama_functions as fama
from oracle import rank_standardize, draw_W, build_feature_sets, evaluate_bases
import rf_model
import rf_configs

CHARS = rf_model.CHARS
GAMMA_GRID = np.arange(0.5, 1.1, 0.1)
PROTOCOL_KAPPAS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def evaluate(w, mu, Sigma, xret):
    if not np.all(np.isfinite(w)):
        return {"mn": np.nan, "stdev": np.nan, "sharpe": np.nan, "xret": np.nan}
    mean = w @ mu
    sd = np.sqrt(max(w @ Sigma @ w, 1e-300))
    return {"mn": mean, "stdev": sd, "sharpe": mean / sd, "xret": w @ xret}


def theta_stats(months_data):
    """premium_shape.py's three statistics on the true premia: mean per-month theta, variance-
    weighted per-month theta, pooled theta; time-variation share = pooled - weighted."""
    fits = []
    for md in months_data:
        A = np.column_stack([np.ones(len(md["mu"])), md["X_rank"]])
        b, *_ = np.linalg.lstsq(A, md["mu"], rcond=None)
        fits.append((A, md["mu"], b))
    per = []
    for A, y, b in fits:
        tss = float(((y - y.mean()) ** 2).sum())
        if tss > 0:
            per.append(float(((y - A @ b) ** 2).sum()) / tss)
    rss_m = sum(float(((y - A @ b) ** 2).sum()) for A, y, b in fits)
    bp, *_ = np.linalg.lstsq(np.vstack([A for A, _, _ in fits]), np.concatenate([y for _, y, _ in fits]), rcond=None)
    rss_p = sum(float(((y - A @ bp) ** 2).sum()) for A, y, _ in fits)
    tss = sum(float(((y - y.mean()) ** 2).sum()) for _, y, _ in fits)
    return {"theta_month": float(np.mean(per)), "theta_wtd": rss_m / tss, "theta_pooled": rss_p / tss,
            "time_variation": rss_p / tss - rss_m / tss}


def run(params, name, seed, out, window=360, rff=(36, 360), nmat=1, kappas=PROTOCOL_KAPPAS,
        stride=1, n_jobs=4, trim=(14, 2), save_panel=False, verbose=True, oracle_only=False):
    t0 = time.time()
    econ = rf_model.Economy(params, seed=seed)
    p = econ.p
    N, T = econ.N, econ.T
    panel = econ.panel()
    if verbose:
        print(f"[{name} s{seed}] {econ.describe()}", flush=True)
        print(f"[{name} s{seed}] panel {panel.shape} in {time.time()-t0:.1f}s", flush=True)
    panel = panel.set_index(["month", "firmid"]).sort_index()
    panel["xret_true"] = panel.xret
    start, end = 0, T - 1
    months = [m for m in range(trim[0], T - trim[1] + 1) if m <= end]
    eval_months = [m for m in months if m >= months[0] + window][::stride]   # 125 at protocol, as run_estimators.py
    if not eval_months:
        raise SystemExit(f"T={T} leaves no evaluation month at window {window}")

    # ---- true moments and population ceilings ------------------------------------------------
    rng = np.random.default_rng(seed + 1000)
    months_data, ts = [], []
    for m in months:
        mu, Sigma, B = econ.month_moments(m)
        data = panel.loc[m]
        X_raw = data[CHARS].to_numpy()
        md = {"month": m, "mu": mu, "Sigma": Sigma, "X_raw": X_raw, "X_rank": rank_standardize(X_raw),
              "rf": np.array([float(data["rf_stand"].iloc[0])])}
        months_data.append(md)
        d = 1.0 / econ.sig_eps ** 2
        BtD = B.T * d
        M = np.linalg.inv(econ.Omega) + BtD @ B
        v = BtD @ mu
        srmax = float(np.sqrt(max(mu @ (d * mu) - v @ np.linalg.solve(M, v), 0.0)))
        sr_ew = float(mu.mean() / np.sqrt(max(Sigma.mean(), 1e-300)))
        ts.append({"month": m, "sr_max": srmax, "sr_ew": sr_ew, "sr_orth": float(np.sqrt(max(srmax ** 2 - sr_ew ** 2, 0))),
                   "mean_mu": float(mu.mean()), "sd_mu": float(mu.std()), "s": float(md["rf"][0]), "g": float(econ.g[econ.burn + m])})
    ts = pd.DataFrame(ts)
    ev = ts.month.isin(eval_months).to_numpy()
    d = len(CHARS)
    Wdict = {f"rff{P}_{k}": draw_W(P, d + 1, GAMMA_GRID, rng) for P in rff for k in range(nmat)}
    feature_fn = lambda md: build_feature_sets(md["X_raw"], md["X_rank"], md["rf"], Wdict)
    names = list(feature_fn(months_data[0]).keys())
    res = evaluate_bases(months_data, feature_fn, names)
    mask = np.isin([md["month"] for md in months_data], eval_months)
    ceil = {}
    for nm, r in res.items():
        key = nm.split("_")[0] if nm.startswith("rff") else nm
        sr_all = r["cond_sr_mean"]; sr_e = r["cond_sr_ts"][mask].mean(0)
        rec = ceil.setdefault(key, {"P": r["P"], "all": [], "eval": []})
        rec["all"].append(float(sr_all.max())); rec["eval"].append(float(sr_e.max()))
    ceil = {k: {"P": v["P"], "const_best": float(np.mean(v["all"])), "const_best_eval": float(np.mean(v["eval"]))} for k, v in ceil.items()}
    nl_keys = [k for k in ceil if k.startswith("rff") or k in ("bins", "poly2")]
    nl_keys = [k for k in nl_keys if not k.endswith("_n")]
    best_nl = max(ceil[k]["const_best_eval"] for k in nl_keys)
    best_nl_all = max(ceil[k]["const_best"] for k in nl_keys)
    oracle = {
        "sr_max": float(ts.sr_max.mean()), "sr_max_eval": float(ts.sr_max[ev].mean()),
        "sr_ew_eval": float(ts.sr_ew[ev].mean()), "sr_orth_eval": float(ts.sr_orth[ev].mean()),
        "mkt_share": float(ts.sr_ew[ev].mean() / ts.sr_max[ev].mean()),
        "mean_mu": float(ts.mean_mu.mean()), "sd_mu": float(ts.sd_mu.mean()),
        "ann_excess_pct": float(100 * 12 * ts.mean_mu.mean()),
        "lin_ceiling_eval": ceil["lin_rank"]["const_best_eval"],
        "lin_rf_ceiling_eval": ceil["lin_rank_rf"]["const_best_eval"],
        "nonlin_ceiling_eval": best_nl,
        "room_eval": best_nl - ceil["lin_rank"]["const_best_eval"],
        "room_s_eval": best_nl - ceil["lin_rank_rf"]["const_best_eval"],
        "room_all": best_nl_all - ceil["lin_rank"]["const_best"],
        "bases": ceil, "months": len(months), "eval_months": len(eval_months),
        **theta_stats([md for md, keep in zip(months_data, mask) if keep]),
    }
    if verbose:
        print(f"[{name} s{seed}] oracle in {time.time()-t0:.0f}s: SR_max {oracle['sr_max_eval']:.4f} "
              f"mkt {oracle['mkt_share']*100:.0f}% room {oracle['room_eval']:+.4f} room_s {oracle['room_s_eval']:+.4f} "
              f"theta {oracle['theta_wtd']:.3f} tv {oracle['time_variation']:+.3f}", flush=True)

    if oracle_only:
        return {"name": name, "seed": seed, "params": p, "oracle": oracle}

    # ---- estimators, mirroring run_estimators.py -------------------------------------------------
    dkkm.WINDOW = window; fama.WINDOW = window
    dkkm.RF_COLS = ["rf_stand"]
    RFC = ["rf_stand"]
    rng_e = np.random.default_rng(seed + 7)
    kappas = [float(k) for k in kappas]
    fair_kappas = sorted(set(kappas) | {10 * max(kappas), 100 * max(kappas)})
    ff_rets = fama.factors(fama.fama_french, panel, n_jobs=n_jobs, start=start, end=end, chars=CHARS)
    fm_rets = fama.factors(fama.fama_macbeth, panel, n_jobs=n_jobs, start=start, end=end, chars=CHARS)
    Plist = list(rff); Pmax = max(Plist); half = Pmax // 2
    Ws, frets = [], []
    for i in range(nmat):
        W = rng_e.standard_normal(size=(half, len(CHARS) + len(RFC)))
        W = rng_e.choice(GAMMA_GRID, size=(half, 1)) * W
        Ws.append(W)
        frets.append(dkkm.factors(panel=panel, W=W, n_jobs=n_jobs, start=start, end=end, model="bgn", chars=CHARS, rf_cols=RFC)[0])

    def lin_rank_weights(data):
        X = dkkm.rank_standardize(data[CHARS])
        X.insert(0, "const", 1.0 / len(X))
        X.columns = [str(i) for i in range(X.shape[1])]
        return X

    def lin_state_weights(data):
        """ranks, the state, ranks x state: the linear basis that sees what DKKM's features see."""
        Xr = dkkm.rank_standardize(data[CHARS]).to_numpy()
        s = float(data["rf_stand"].iloc[0])
        X = pd.DataFrame(np.column_stack([Xr, np.full((len(Xr), 1), s / len(Xr)), Xr * s]), index=data.index)
        X.columns = [str(i) for i in range(X.shape[1])]
        return X

    lr_rets = pd.concat([(lin_rank_weights(panel.loc[m]).T @ panel.loc[m].xret).rename(m) for m in range(start, end + 1)], axis=1).T
    lr_rets.index.name = "month"
    ls_rets = pd.concat([(lin_state_weights(panel.loc[m]).T @ panel.loc[m].xret).rename(m) for m in range(start, end + 1)], axis=1).T
    ls_rets.index.name = "month"
    mkt_rets = panel.groupby("month").xret.mean()
    mkt_frame = mkt_rets.to_frame("mkt")
    if verbose:
        print(f"[{name} s{seed}] factor histories in {time.time()-t0:.0f}s", flush=True)

    midx = {md["month"]: i for i, md in enumerate(months_data)}
    rows = []
    for k, month in enumerate(eval_months):
        md = months_data[midx[month]]
        data = panel.loc[month]
        mu, Sigma, xret = md["mu"], md["Sigma"], data.xret_true.to_numpy()
        n_ = len(data)
        for nm, fr, meth in [("ff", ff_rets, fama.fama_french), ("fm", fm_rets, fama.fama_macbeth)]:
            fw = meth(data[CHARS], CHARS, mve=data.mve)
            theta = fama.mve_data(fr, month, 0)
            rows.append({"month": month, "method": nm, "P": fw.shape[1], "kappa": 0.0, **evaluate(fw @ theta.to_numpy(), mu, Sigma, xret)})
        fw = lin_rank_weights(data)
        thetas = dkkm.mve_data(lr_rets, month, 6 * np.array([0.0] + kappas))
        for j, kap in enumerate([0.0] + kappas):
            rows.append({"month": month, "method": "linrank", "P": 6, "kappa": kap, **evaluate((fw @ thetas.iloc[:, j]).to_numpy(), mu, Sigma, xret)})
        th_m = float(fama.mve_data(mkt_frame, month, 0).iloc[0])
        rows.append({"month": month, "method": "mkt_est", "P": 1, "kappa": 0.0, **evaluate(np.full(n_, th_m / n_), mu, Sigma, xret)})
        rows.append({"month": month, "method": "ew", "P": 1, "kappa": 0.0, **evaluate(np.full(n_, 1.0 / n_), mu, Sigma, xret)})
        fw = lin_rank_weights(data).iloc[:, 1:].copy(); fw["mkt_rf"] = 1.0 / n_
        thetas = dkkm.mve_data(lr_rets.iloc[:, 1:], month, len(CHARS) * np.array([0.0] + fair_kappas), mkt_rets)
        for j, kap in enumerate([0.0] + fair_kappas):
            rows.append({"month": month, "method": "linrank_m", "P": fw.shape[1], "kappa": kap, **evaluate((fw @ thetas.iloc[:, j]).to_numpy(), mu, Sigma, xret)})
        fw = lin_state_weights(data); fw["mkt_rf"] = 1.0 / n_
        thetas = dkkm.mve_data(ls_rets, month, 11 * np.array([0.0] + fair_kappas), mkt_rets)
        for j, kap in enumerate([0.0] + fair_kappas):
            rows.append({"month": month, "method": "linrank_s", "P": fw.shape[1], "kappa": kap, **evaluate((fw @ thetas.iloc[:, j]).to_numpy(), mu, Sigma, xret)})
        rf = data[RFC]
        ens = {}
        for mi, (W, fr) in enumerate(zip(Ws, frets)):
            for P in Plist:
                num = P // 2
                nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])
                fw = dkkm.rff(data[CHARS], rf, W=W[:num, :], model="bgn")[0]
                fw.columns = [str(ind) for ind in nf_indx]
                fw["mkt_rf"] = 1.0 / len(fw)
                thetas = dkkm.mve_data(fr.iloc[:, nf_indx], month, P * np.array([0.0] + kappas), mkt_rets)
                thetas = thetas.iloc[:, 1:]; thetas.columns = kappas
                for j, kap in enumerate(kappas):
                    w = (fw @ thetas.iloc[:, j]).to_numpy()
                    rows.append({"month": month, "method": "rff", "P": P, "kappa": kap, "mat": mi, **evaluate(w, mu, Sigma, xret)})
                    ens[(P, kap)] = ens.get((P, kap), 0) + w / len(Ws)
        if len(Ws) > 1:
            for (P, kap), w in ens.items():
                rows.append({"month": month, "method": "rff_ens", "P": P, "kappa": kap, "mat": -1, **evaluate(w, mu, Sigma, xret)})
        if verbose and k % 25 == 0:
            print(f"  month {month} ({k+1}/{len(eval_months)}) {time.time()-t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    summ = res.groupby(["method", "P", "kappa"]).agg(sharpe=("sharpe", "mean"), real_sr=("xret", lambda x: x.mean() / x.std())).reset_index()
    best = summ.loc[summ.groupby("method").sharpe.idxmax()].set_index("method")
    sr = best.sharpe
    dk_methods = [m for m in ("rff", "rff_ens") if m in sr.index]
    dkkm_sr = float(sr[dk_methods].max())
    dk_row = best.loc[sr[dk_methods].idxmax()]
    classical = max(sr["fm"], sr["ff"])
    lin = max(sr["linrank"], sr["fm"], sr["ff"])
    fair = max(lin, sr["linrank_m"], sr["mkt_est"])
    fair_s = max(fair, sr["linrank_s"])
    # the kappa gate, by position only (penalty_gate.py's materiality clause needs the full grid)
    kap_edge = float(dk_row.kappa) in (min(kappas), max(kappas))
    est = {
        "dkkm": dkkm_sr, "dkkm_P": int(dk_row.P), "dkkm_kappa": float(dk_row.kappa), "dkkm_at_edge": bool(kap_edge),
        "fm": float(sr["fm"]), "ff": float(sr["ff"]), "linrank": float(sr["linrank"]), "linrank_m": float(sr["linrank_m"]),
        "linrank_s": float(sr["linrank_s"]), "mkt_est": float(sr["mkt_est"]), "ew": float(sr["ew"]),
        "classical_gap": dkkm_sr - classical, "gap": dkkm_sr - lin, "fair_gap": dkkm_sr - fair, "fair_s_gap": dkkm_sr - fair_s,
        "fair_method": str(sr[["linrank", "fm", "ff", "linrank_m", "mkt_est"]].idxmax()),
        "dkkm_pct_ceiling": dkkm_sr / oracle["nonlin_ceiling_eval"] if oracle["nonlin_ceiling_eval"] > 0 else np.nan,
        "fair_pct_ceiling": fair / oracle["lin_ceiling_eval"] if oracle["lin_ceiling_eval"] > 0 else np.nan,
        "excess_loss": oracle["room_eval"] - (dkkm_sr - fair),
        "best": {m: {"sharpe": float(r.sharpe), "P": int(r.P), "kappa": float(r.kappa)} for m, r in best.iterrows()},
        "eval_months": len(eval_months), "window": window, "rff": list(rff), "nmat": nmat, "kappas": kappas, "stride": stride,
    }
    rec = {"name": name, "seed": seed, "params": p, "describe": econ.describe(), "oracle": oracle, "estimators": est,
           "seconds": time.time() - t0}
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"screen_{name}_s{seed:03d}")
    json.dump(rec, open(stem + ".json", "w"), indent=1, default=float)
    summ.to_csv(stem + "_summary.csv", index=False)
    ts.to_csv(stem + "_ts.csv", index=False)
    if save_panel:
        panel.reset_index().to_parquet(stem + "_panel.parquet")
    if verbose:
        print(f"[{name} s{seed}] DKKM {dkkm_sr:.4f} (P={est['dkkm_P']} k={est['dkkm_kappa']:g}) fm {sr['fm']:.4f} ff {sr['ff']:.4f} "
              f"linrank_m {sr['linrank_m']:.4f} linrank_s {sr['linrank_s']:.4f} | classical {est['classical_gap']:+.4f} "
              f"fair {est['fair_gap']:+.4f} fair_s {est['fair_s_gap']:+.4f} | room {oracle['room_eval']:+.4f} "
              f"room_s {oracle['room_s_eval']:+.4f} | {time.time()-t0:.0f}s", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="a name from rf_configs.CONFIGS")
    ap.add_argument("--params", default=None, help="JSON overrides on top of --config (or the defaults)")
    ap.add_argument("--name", default=None, help="output name; default the config name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None, help="e.g. 0-9: run several seeds in sequence")
    ap.add_argument("--out", default=os.environ.get("BOP_SCREEN_DIR", os.path.join(HERE, "screen")))
    ap.add_argument("--window", type=int, default=360)
    ap.add_argument("--rff", default="36,360")
    ap.add_argument("--nmat", type=int, default=1)
    ap.add_argument("--kappas", default=None, help="default: the protocol grid")
    ap.add_argument("--stride", type=int, default=1, help="score every k-th evaluation month")
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--full", action="store_true", help="protocol grid: rff 36,360,3600 and nmat 2")
    ap.add_argument("--save_panel", action="store_true")
    ap.add_argument("--oracle_only", action="store_true", help="population ceilings only; no estimators, no files")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    args = ap.parse_args(argv)
    params = dict(rf_configs.CONFIGS[args.config]) if args.config else {}
    if args.params:
        params.update(json.loads(args.params))
    if args.N: params["N"] = args.N
    if args.T: params["T"] = args.T
    name = args.name or args.config or "adhoc"
    # --full writes under its own name so a protocol-grid check never overwrites the reduced-grid
    # record of the same config (2026-10-07: fullcheck did exactly that to 21 files).
    if args.full and not args.name:
        name += "_full"
    rff = (36, 360, 3600) if args.full else tuple(int(x) for x in args.rff.split(",") if x)
    nmat = 2 if args.full else args.nmat
    kappas = PROTOCOL_KAPPAS if args.kappas is None else tuple(float(k) for k in args.kappas.split(","))
    if args.seeds:
        a, b = args.seeds.split("-"); seeds = range(int(a), int(b) + 1)
    else:
        seeds = [args.seed]
    for s in seeds:
        run(params, name, int(s), args.out, window=args.window, rff=rff, nmat=nmat, kappas=kappas,
            stride=args.stride, n_jobs=args.n_jobs, save_panel=args.save_panel, oracle_only=args.oracle_only)


if __name__ == "__main__":
    main()
