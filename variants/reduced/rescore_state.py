"""Re-score a SAVED structural panel with the state-aware linear benchmark.

    python variants/reduced/rescore_state.py --dir /kellogg/proj/axa3203/bop_runs/results kp_vy/vyg30k70 --seed 0

Reads {model}_panel_{tag}_s{seed}.parquet, the moments .npz and the estimator summary that already
scores DKKM, FM, FF, linrank_m and mkt_est. Adds two linear methods, each with the EW market as a
separate unpenalised column exactly as linrank_m has it (run_estimators.py, --fair_linear):
  linrank_m   re-scored here as a check that this file reproduces the recorded number
  linrank_s   ranks, the state (rf_stand), ranks x state           -- what DKKM's features see
Prints the recorded DKKM, the recorded fair benchmark, and the fair gap before and after linrank_s
joins the benchmark. Linear only: minutes, not hours. Writes nothing under the results directory.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import dkkm_functions as dkkm
import fama_functions as fama

CHARS = ["size", "bm", "agr", "roe", "mom"]
KAPPAS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
STATE_COLS = {"kp_vy": ["rf_stand"], "bgn_gam": ["rf_stand", "gam_stand"], "gs_bx": ["rf_stand", "gam_stand"],
              "bgn_vy": ["rf_stand", "y_stand"], "gs_vy": ["rf_stand", "y_stand"]}


def evaluate(w, mu, Sigma):
    if not np.all(np.isfinite(w)):
        return np.nan
    return float(w @ mu / np.sqrt(max(w @ Sigma @ w, 1e-300)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("economy")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=360)
    ap.add_argument("--state_cols", default=None, help="default: the model's conditioning columns")
    ap.add_argument("--out", default=None, help="write a JSON here")
    args = ap.parse_args(argv)
    model, tag = args.economy.split("/", 1)
    stem = os.path.join(args.dir, f"{model}_%s_{tag}_s{args.seed:03d}")
    t0 = time.time()
    panel = pd.read_parquet((stem % "panel") + ".parquet").set_index(["month", "firmid"]).sort_index()
    mom = np.load((stem % "moments") + ".npz")
    summ = pd.read_csv((stem % "estimators") + f"_w{args.window}_summary.csv")
    rec = summ.groupby("method").sharpe.max()
    scols = STATE_COLS[model] if args.state_cols is None else args.state_cols.split(",")
    months_m, MU, SIG, KEEP = mom["months"], mom["mu"], mom["Sigma"], mom["keep"]
    midx = {int(m): i for i, m in enumerate(months_m)}
    start, end = int(months_m.min()), int(months_m.max())
    eval_months = [m for m in range(start + args.window, end + 1) if m in midx]
    dkkm.WINDOW = args.window; fama.WINDOW = args.window
    kappas = list(KAPPAS); fair_kappas = sorted(set(kappas) | {10 * max(kappas), 100 * max(kappas)})

    def w_m(data):
        X = dkkm.rank_standardize(data[CHARS]); X.columns = [str(i) for i in range(5)]
        return X

    def w_s(data):
        Xr = dkkm.rank_standardize(data[CHARS]).to_numpy()
        n = len(Xr)
        cols = [Xr]
        for c in scols:
            s = float(data[c].iloc[0])
            cols += [np.full((n, 1), s / n), Xr * s]
        X = pd.DataFrame(np.column_stack(cols), index=data.index); X.columns = [str(i) for i in range(X.shape[1])]
        return X

    rets = {}
    for nm, fn in (("linrank_m", w_m), ("linrank_s", w_s)):
        r = pd.concat([(fn(panel.loc[m]).T @ panel.loc[m].xret).rename(m) for m in range(start, end + 1)], axis=1).T
        r.index.name = "month"; rets[nm] = r
    mkt = panel.groupby("month").xret.mean()
    print(f"[{args.economy} s{args.seed}] factor histories in {time.time()-t0:.0f}s; state columns {scols}", flush=True)
    rows = []
    for month in eval_months:
        i = midx[month]; data = panel.loc[month]; n = len(data)
        mu, Sigma = MU[i, :n].astype(float), SIG[i, :n, :n].astype(float)
        for nm, fn in (("linrank_m", w_m), ("linrank_s", w_s)):
            fw = fn(data).copy(); fw["mkt_rf"] = 1.0 / n
            P = fw.shape[1] - 1
            th = dkkm.mve_data(rets[nm], month, P * np.array([0.0] + fair_kappas), mkt)
            for j, kap in enumerate([0.0] + fair_kappas):
                rows.append({"month": month, "method": nm, "kappa": kap, "sharpe": evaluate((fw @ th.iloc[:, j]).to_numpy(), mu, Sigma)})
    res = pd.DataFrame(rows).groupby(["method", "kappa"]).sharpe.mean().reset_index()
    best = res.loc[res.groupby("method").sharpe.idxmax()].set_index("method")
    fair_rec = max(rec.get(k, -np.inf) for k in ("linrank", "linlev", "fm", "ff", "linrank_m", "linlev_m", "mkt_est"))
    dk = max(rec.get(k, -np.inf) for k in ("rff", "rff_ens", "rff_lev", "rff_lev_ens"))
    out = {"economy": args.economy, "seed": args.seed, "dkkm": float(dk), "fair_recorded": float(fair_rec),
           "linrank_m_recorded": float(rec.get("linrank_m", np.nan)), "linrank_m_here": float(best.loc["linrank_m", "sharpe"]),
           "linrank_s": float(best.loc["linrank_s", "sharpe"]), "linrank_s_kappa": float(best.loc["linrank_s", "kappa"]),
           "fair_gap": float(dk - fair_rec), "fair_s_gap": float(dk - max(fair_rec, best.loc["linrank_s", "sharpe"])),
           "state_cols": scols, "eval_months": len(eval_months)}
    print(f"[{args.economy} s{args.seed}] DKKM {out['dkkm']:.4f} | fair recorded {fair_rec:.4f} | linrank_m recorded {out['linrank_m_recorded']:.4f} "
          f"here {out['linrank_m_here']:.4f} | linrank_s {out['linrank_s']:.4f} (k={out['linrank_s_kappa']:g}) | "
          f"fair gap {out['fair_gap']:+.4f} -> with state {out['fair_s_gap']:+.4f} | {time.time()-t0:.0f}s", flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
