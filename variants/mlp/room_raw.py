"""Recompute the population ceilings of one saved panel with the random-Fourier basis built WITH and WITHOUT the
rank-standardisation applied after the sines and cosines (dkkm.rff / oracle.rff_features(restandardize)), next to
the linear, quadratic and bin bases that never restandardise. Same evaluation as run_oracle.py: constant theta from
the true conditional moments over all months, conditional Sharpe averaged over the evaluation window, best ridge z.
usage: python room_raw.py --model kp_vy --tag vy1 --seed 0"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
REPO = "/kellogg/proj/axa3203/bop-run-upload"
sys.path.insert(0, os.path.join(REPO, "variants", "common"))
from oracle import rank_standardize, draw_W, rff_features, poly2_features, bin_features, evaluate_bases
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--P", type=int, default=3600); ap.add_argument("--window", type=int, default=360)
ap.add_argument("--results", default="/kellogg/proj/axa3203/bop_runs/results"); ap.add_argument("--out", default="/kellogg/proj/axa3203/bop_runs/mlp/room_raw")
a = ap.parse_args(); t0 = time.time()
stem = f"{a.model}_%s_{a.tag}_s{a.seed:03d}"
panel = pd.read_parquet(os.path.join(a.results, stem % "panel" + ".parquet")).set_index(["month", "firmid"]).sort_index()
mom = np.load(os.path.join(a.results, stem % "moments" + ".npz")); osum = json.load(open(os.path.join(a.results, stem % "oracle" + ".json")))
chars = ["size", "bm", "agr", "roe", "mom"]; rf_cols = list(osum["rf_cols"])
months, MU, SIG, KEEP = mom["months"], mom["mu"], mom["Sigma"], mom["keep"]
md_list = []
for i, m in enumerate(months):
    data = panel.loc[int(m)]; keep = data.index.to_numpy(); n = len(keep); assert np.array_equal(KEEP[i, :n], keep)
    X_raw = data[chars].to_numpy()
    md_list.append({"month": int(m), "mu": MU[i, :n].astype(float), "Sigma": SIG[i, :n, :n].astype(float), "X_raw": X_raw,
                    "X_rank": rank_standardize(X_raw), "rf": (np.array([float(data[c].iloc[0]) for c in rf_cols]) if rf_cols else None)})
rng = np.random.default_rng(1000 + a.seed); d = len(chars) + len(rf_cols)
W = draw_W(a.P, d, np.arange(0.5, 1.1, 0.1), rng)
def feature_fn(md):
    N = len(md["mu"]); one = np.ones((N, 1)); Xr, rf = md["X_rank"], md["rf"]
    Fs = rff_features(Xr, W, rf, restandardize=True); Fr = rff_features(Xr, W, rf, restandardize=False)
    return {"lin_rank": np.column_stack([one, Xr]), "poly2": np.column_stack([one, poly2_features(Xr, rf)]),
            "bins": np.column_stack([one, bin_features(md["X_raw"])]),
            "rff_std": np.column_stack([one, Fs]), "rff_raw": np.column_stack([one, Fr]),
            "rff_std_n": np.column_stack([one, Xr, Fs]), "rff_raw_n": np.column_stack([one, Xr, Fr])}
names = list(feature_fn(md_list[0]).keys())
res = evaluate_bases(md_list, feature_fn, names, verbose=True)
marr = np.array([md["month"] for md in md_list]); ev = marr >= marr.min() + a.window
out = {"model": a.model, "tag": a.tag, "seed": a.seed, "P": a.P, "eval_months": int(ev.sum())}
for nm in names:
    sr = res[nm]["cond_sr_ts"][ev].mean(0); j = int(np.argmax(sr))
    out[nm] = {"const_best_eval": float(sr[j]), "best_zrel": res[nm]["zgrid_rel"][j], "sr_by_z": [float(x) for x in sr]}
rec = osum["bases"]
out["recorded"] = {k: rec[k]["const_best_eval"] for k in ("lin_rank", "poly2", "bins", "rff3600_0", "rff3600_0_n") if k in rec}
os.makedirs(a.out, exist_ok=True); fn = os.path.join(a.out, f"room_raw_{a.model}_{a.tag}_s{a.seed:03d}.json")
json.dump(out, open(fn, "w"), indent=1)
print(f"[{a.tag} s{a.seed}] " + "  ".join(f"{nm}={out[nm]['const_best_eval']:.4f}" for nm in names) + f"   recorded {out['recorded']}   {time.time()-t0:.0f}s", flush=True)
