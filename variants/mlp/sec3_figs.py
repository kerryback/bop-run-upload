"""Section-3 weight figures: how nonlinear is each method's portfolio weight in the characteristics?

For each section-3 economy (3a control, 3b interaction, 3c curvature; BGN / GS21 / KP14) an MLP
(common.py: 5x32 ReLU, 30 epochs) is fitted in sample to each method's per-firm weight as a function
of the characteristics the estimators see (size, bm, agr, roe, mom, plus lev in GS21), pooled over
seeds and evaluation months, characteristics rank-standardised within each (seed, month) cross-section
and weights z-scored within each cross-section (so the four methods' curves share one scale: the
weight in units of its own cross-sectional sd).  Partial-dependence curves are then drawn with every
other characteristic at its pooled mean.

  fig_3a.png, fig_3c.png   1 x 3 (BGN, GS21, KP14): weight vs the most important characteristic,
                            four curves (true SDF, FM, FF, DKKM)
  fig_3b.png               3 x 4 (model x method): weight vs the first characteristic of the
                            interaction at five quantiles of the second

The 'most important' characteristic is the one whose one-way partial-dependence curve of the TRUE
weight has the largest variance; the 3b pair is the pair whose two-way interaction part of the true
weight has the largest variance.  Both are printed and compared with the economy's construction.
usage: python sec3_figs.py [3a 3b 3c] [--seeds N] [--pd] [--construction | default]
  default = projection E[fitted weight | construction characteristic(s)]; --pd = partial dependence (others at mean);
  data-driven characteristic choice only when --pd is given without --construction (file suffix _datadriven)
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
sys.path.insert(0, "/kellogg/proj/axa3203/external/MLP Analyze")
from common import rank_standardize_per_date, drop_constant_cols, winsorize_per_group, get_or_train_mlp, predict_mlp

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
DATA = Path("/kellogg/proj/axa3203/bop_runs/mlp/data"); HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"; CACHE = HERE / "cache"; OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
METHODS = [("true", "true SDF", "k"), ("fm", "FM", "tab:green"), ("ff", "FF", "tab:orange"), ("dkkm", "DKKM", "tab:blue")]
FAMS = ["BGN", "GS21", "KP14"]
ECONS = {  # section -> family -> (model, tag, expected main char, expected second char for 3b)
    "3a": {"BGN": ("bgn_vy", "bvy1ch", "bm", None), "GS21": ("gs_vy", "gvylf10", "lev", None), "KP14": ("kp_vy", "vy1", "bm", None)},
    "3b": {"BGN": ("bgn_vy", "bvychs35", "bm", "size"), "GS21": ("gs_vy", "gvylfs6i", "lev", "size"), "KP14": ("kp_vy", "vyg25s", "bm", "size")},
    "3c": {"BGN": ("bgn_vy", "bvyuq2", "size", None), "GS21": ("gs_vy", "gvylqup5", "lev", None), "KP14": ("kp_vy", "vyuq2", "size", None)},
}
NICE = {"size": "size", "bm": "book-to-market", "agr": "asset growth", "roe": "profitability", "mom": "momentum", "lev": "market leverage"}
if os.environ.get("SEC3_BGN3C"):   # preview another BGN 3c economy without editing the table above
    ECONS["3c"]["BGN"] = ("bgn_vy", os.environ["SEC3_BGN3C"], "size", None)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SECS = [s for s in ["3a", "3b", "3c"] if s in args] or ["3a", "3b", "3c"]
NSEEDS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 10
BYCON = "--construction" in sys.argv or "--pd" not in sys.argv   # x axis = the characteristic the exposure is written on, not the data-driven one
PROJ = "--pd" not in sys.argv     # default: projection E[fitted weight | char]; --pd: partial dependence with the other chars at their mean
DETREND = "--detrend" in sys.argv   # subtract an OLS line in the characteristic; --detrend-own: each curve's own line (default: the TRUE curve's line from every method)
DETREND_OWN = "--detrend-own" in sys.argv
SUF = ("" if BYCON else "_datadriven") + ("" if PROJ else "_pd") + ("_detrended_own" if DETREND_OWN else "_detrended" if DETREND else "")


def chars_for(model):
    return ["size", "bm", "agr", "roe", "mom"] + (["lev"] if model.startswith("gs") else [])


def load(model, tag):
    files = sorted(DATA.glob(f"wts_{model}_{tag}_s[0-9][0-9][0-9].parquet"))[:NSEEDS]
    if not files: return None
    cc = chars_for(model); parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["month", "firmid"] + cc + ["w_true", "w_fm", "w_ff", "w_dkkm"]); d["_panel"] = int(f.stem[-3:]); parts.append(d)
    return pd.concat(parts, ignore_index=True), cc, len(files)


def fit(df, cc, wcol, cache):
    m = df[["_panel", "month", "firmid"] + cc].copy(); m["weight"] = df[wcol].to_numpy()
    rank_standardize_per_date(m, cc); m = m.dropna(subset=cc + ["weight"]); cols = drop_constant_cols(m, cc)
    winsorize_per_group(m, "weight")
    g = m.groupby(["_panel", "month"])["weight"]
    m["weight"] = (m["weight"] - g.transform("mean")) / g.transform("std").replace(0, 1.0)     # z-score per cross-section
    X = m[cols].to_numpy(np.float32); y = m["weight"].to_numpy(np.float32)
    net, sc = get_or_train_mlp(cache, X, y)
    yhat = predict_mlp(net, sc, X)
    return X, cols, net, sc, yhat


def pd1(X, cc, net, sc, a, n=80, lo_hi=(0.025, 0.975)):
    j = cc.index(a); g = np.linspace(*np.quantile(X[:, j], lo_hi), n).astype(np.float32)
    rows = np.tile(X.mean(axis=0).astype(np.float32), (n, 1)); rows[:, j] = g
    return g, predict_mlp(net, sc, rows)


def pd2(X, cc, net, sc, a, b, n=21):
    ja, jb = cc.index(a), cc.index(b)
    ga = np.linspace(*np.quantile(X[:, ja], [0.025, 0.975]), n); gb = np.linspace(*np.quantile(X[:, jb], [0.025, 0.975]), n)
    rows = np.tile(X.mean(axis=0).astype(np.float32), (n * n, 1)); A, B = np.meshgrid(ga, gb, indexing="ij")
    rows[:, ja] = A.ravel(); rows[:, jb] = B.ravel()
    return predict_mlp(net, sc, rows).reshape(n, n)


def proj1(X, cc, yhat, a, nb=25, lo_hi=(0.02, 0.98)):
    """E[fitted weight | char a] by quantile bins of a: the projection of the fitted surface onto a, the other
    characteristics averaged over as they co-move with a in the data (not held fixed)."""
    j = cc.index(a); x = X[:, j]; lo, hi = np.quantile(x, lo_hi); m = (x >= lo) & (x <= hi)
    b = pd.qcut(pd.Series(x[m]).rank(method="first"), nb, labels=False).to_numpy()
    gx = np.bincount(b, x[m]) / np.bincount(b); gy = np.bincount(b, yhat[m]) / np.bincount(b)
    return gx, gy


def proj2(X, cc, yhat, a, b, qs=(0.10, 0.30, 0.50, 0.70, 0.90), nb=15):
    """E[fitted weight | a, b in band k] for five quantile bands of b (edges at the midpoints between the qs)."""
    ja, jb = cc.index(a), cc.index(b); edges = np.concatenate([[0], (np.array(qs[:-1]) + np.array(qs[1:])) / 2, [1]])
    cuts = np.quantile(X[:, jb], edges); out = []
    for k in range(len(qs)):
        m = (X[:, jb] >= cuts[k]) & (X[:, jb] <= cuts[k + 1]); x = X[m, ja]; y = yhat[m]
        lo, hi = np.quantile(x, [0.02, 0.98]); mm = (x >= lo) & (x <= hi)
        bb = pd.qcut(pd.Series(x[mm]).rank(method="first"), nb, labels=False).to_numpy()
        out.append((np.bincount(bb, x[mm]) / np.bincount(bb), np.bincount(bb, y[mm]) / np.bincount(bb)))
    return out


def main_char(fit_true):
    X, cc, net, sc, _ = fit_true
    v = {a: float(np.var(pd1(X, cc, net, sc, a)[1])) for a in cc}
    return max(v, key=v.get), v


def main_pair(fit_true):
    X, cc, net, sc, _ = fit_true; v = {}
    for i, a in enumerate(cc):
        for b in cc[i + 1:]:
            S = pd2(X, cc, net, sc, a, b); I = S - (S.mean(1, keepdims=True) + S.mean(0, keepdims=True) - S.mean()); v[(a, b)] = float(np.var(I))
    return max(v, key=v.get), v


def one_way_fig(sec, fits, mains, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, fam in zip(axes, FAMS):
        if fam not in fits: ax.set_axis_off(); continue
        a = mains[fam]; tline = None
        for mk, lbl, col in METHODS:
            X, cc, net, sc, yh = fits[fam][mk]; g, y = proj1(X, cc, yh, a) if PROJ else pd1(X, cc, net, sc, a)
            if mk == "true": tline = np.polyfit(g, y, 1)
            if DETREND or DETREND_OWN: y = y - np.polyval(np.polyfit(g, y, 1) if DETREND_OWN else tline, g)
            ax.plot(g, y, color=col, lw=2.2 if mk == "true" else 1.5, ls="-" if mk in ("true", "dkkm") else "--", label=lbl)
        ax.axhline(0, color="0.7", lw=0.6); ax.set_title(f"{fam} ({ECONS[sec][fam][1]})", fontsize=11)
        ax.set_xlabel(f"{NICE[a]} (rank-standardised)")
    axes[0].set_ylabel(("nonlinear part of E[weight | char]  (cs sd units)" if DETREND_OWN else "E[weight | char] minus true linear trend  (cs sd units)" if DETREND else "E[weight | char]  (cross-sectional sd units)" if PROJ else "portfolio weight (cross-sectional sd units)")); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight"); plt.close(fig)


def two_way_fig(sec, fits, pairs, path):
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.6), sharey="row")
    qs = (0.10, 0.30, 0.50, 0.70, 0.90); cmap = plt.get_cmap("viridis")
    for r, fam in enumerate(FAMS):
        if fam not in fits:
            for ax in axes[r]: ax.set_axis_off()
            continue
        a, b = pairs[fam]
        for c, (mk, lbl, _) in enumerate(METHODS):
            X, cc, net, sc, yh = fits[fam][mk]; ax = axes[r, c]; ja, jb = cc.index(a), cc.index(b)
            fv = np.quantile(X[:, jb], qs); base = X.mean(axis=0).astype(np.float32)
            bands = np.concatenate([[0], (np.array(qs[:-1]) + np.array(qs[1:])) / 2, [1]])
            curves = proj2(X, cc, yh, a, b, qs) if PROJ else None
            for k, q in enumerate(qs):
                if PROJ:
                    g, y = curves[k]
                else:
                    lo, hi = np.quantile(X[:, jb], bands[k:k + 2]); mask = (X[:, jb] >= lo) & (X[:, jb] <= hi)
                    g = np.linspace(*np.quantile(X[mask, ja], [0.025, 0.975]), 80).astype(np.float32)
                    rows = np.tile(base, (80, 1)); rows[:, ja] = g; rows[:, jb] = fv[k]; y = predict_mlp(net, sc, rows)
                ax.plot(g, y, color=cmap(k / 4), lw=1.5, label=f"{NICE[b]} p{int(q*100)}")
            ax.axhline(0, color="0.7", lw=0.6)
            ax.set_title(f"{fam} ({ECONS[sec][fam][1]}): {lbl}" if c else f"{fam} ({ECONS[sec][fam][1]}): {lbl}", fontsize=10)
            if r == 2: ax.set_xlabel("")
            ax.set_xlabel(f"{NICE[a]} (rank-standardised)")
        axes[r, 0].set_ylabel("E[weight | chars]  (cs sd units)" if PROJ else "weight (cs sd units)"); axes[r, 3].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight"); plt.close(fig)


def main():
    summary = {}
    for sec in SECS:
        fits, mains, pairs = {}, {}, {}
        for fam in FAMS:
            model, tag, exp_a, exp_b = ECONS[sec][fam]
            got = load(model, tag)
            if got is None: print(f"[{sec}/{fam}] {tag}: no weights yet, skipped", flush=True); continue
            df, cc, ns = got; print(f"[{sec}/{fam}] {tag}: {ns} seeds, {len(df):,} rows", flush=True)
            fits[fam] = {mk: fit(df, cc, f"w_{mk}", CACHE / f"mlp_{tag}_{mk}_n{ns}.pt") for mk, _, _ in METHODS}
            a, va = main_char(fits[fam]["true"]); mains[fam] = exp_a if BYCON else a
            line = f"  main char (true PD var): {a} {dict(sorted(((k, round(v, 4)) for k, v in va.items()), key=lambda kv: -kv[1]))}  expected {exp_a}"
            rec = {"tag": tag, "seeds": ns, "main": a, "main_var": va, "expected_main": exp_a}
            if sec == "3b":
                (p, q), vp = main_pair(fits[fam]["true"])
                top3 = sorted(vp.items(), key=lambda kv: -kv[1])[:3]
                # order the pair so the construction's continuous state is on the x axis when it is in the pair
                pair = (p, q) if exp_a in (p, q) and p == exp_a or exp_a not in (p, q) else (q, p)
                pairs[fam] = (exp_a, exp_b) if BYCON else pair; rec.update({"pair": pair, "pair_var_top3": [(f"{x}x{y}", round(v, 5)) for (x, y), v in top3], "expected_pair": (exp_a, exp_b)})
                line += f"\n  main pair (true interaction var): {pair} top3 {rec['pair_var_top3']}  expected ({exp_a}, {exp_b})"
            print(line, flush=True); summary[f"{sec}/{fam}"] = rec
        if not fits: continue
        if sec == "3b": two_way_fig(sec, fits, pairs, OUT / f"fig_3b{SUF}.png")
        else: one_way_fig(sec, fits, mains, OUT / f"fig_{sec}{SUF}.png")
        print(f"wrote fig_{sec}{SUF}.png", flush=True)
    json.dump(summary, open(OUT / f"summary{SUF}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
