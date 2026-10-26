"""BOP CAMPAIGN RUN: the `Full run` pipeline applied to the Quest campaign economies.

Three benchmark economies and the three group-3b economies that produce a fair gap:

  bgn_base  bgn_gam / bgnbase   BGN, regime inert            bgn_3b  bgn_vy / bvych0  three types x leverage
  gs_base   gs_bx   / gsbase    GS21, one type, regime inert  gs_3b   gs_vy  / gvyp10  productivity x leverage product
  kp_base   kp_vy   / kpbase    KP14, one type, no priced y   kp_3b   kp_vy  / vyg25   three types x leverage

Inputs: wts_<model>_<tag>_s<seed>.parquet written by bop_runs/mlp/extract_weights.py: one row per
(seed, month, firm) over the 125 evaluation months, with the five characteristics the estimators
see (size, bm, agr, roe, mom) and four weight columns: w_true (oracle conditional mean-variance),
w_fm, w_ff, w_dkkm (rff_ens, P=3600, the economy's best kappa). Seeds play the role of panels.

What is added to the Full-run pipeline (same MLP, same rank-standardisation, same PD panels):
  * a linear / nonlinear split of every fitted surface inside each (seed, month) cross-section
    (OLS on the five ranks + constant; the remainder is what no linear-in-ranks portfolio can
    hold), with the cross-sectional correlation of each method's nonlinear remainder with the
    truth's nonlinear remainder;
  * an interaction statistic H for a pair of characteristics: the share of the two-way partial
    dependence surface that is not additive in the two characteristics;
  * for every family a 2x4 figure (benchmark row, 3b row; true / fm / ff / dkkm) at the pair the
    3b truth surface interacts most strongly in.
"""
import json, os, sys, glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (rank_standardize_per_date, drop_constant_cols, winsorize_per_group,
                    get_or_train_mlp, predict_mlp, make_pd_panel, cross_sectional_corr)

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
DATA = Path(os.environ.get("BOP_MLP_DATA", "/kellogg/proj/axa3203/bop_runs/mlp/data"))
RESULTS = HERE / "results"; CACHE = HERE / "cache"
CHARS = ["size", "bm", "agr", "roe", "mom"]
METHODS = ["fm", "ff", "dkkm"]
ECONS = {  # key -> (model, tag, family, role)
    "bgn_base": ("bgn_gam", "bgnbase", "BGN", "benchmark"), "bgn_3b": ("bgn_vy", "bvych0", "BGN", "3b"),
    "gs_base": ("gs_bx", "gsbase", "GS21", "benchmark"),    "gs_3b": ("gs_vy", "gvyp10", "GS21", "3b"),
    "kp_base": ("kp_vy", "kpbase", "KP14", "benchmark"),    "kp_3b": ("kp_vy", "vyg25", "KP14", "3b"),
}
ONLY = [e for e in sys.argv[1:] if e in ECONS] or list(ECONS)
# The pair each family's figures are drawn in, (vary, fixed): the two characteristics the 3b exposure is written in.
# KP14: exposure = type x growth-option share; size proxies the type, bm the option share (the truth's largest
# interaction is bm x mom, where momentum proxies the type, but DKKM recovers the size x bm one; both are reported).
# GS21: productivity (roe) x leverage (bm).  BGN: type (size) x leverage (bm).
FIG_PAIRS = {"kp": ("bm", "size"), "gs": ("roe", "bm"), "bgn": ("bm", "size")}


def load_econ(key):
    model, tag, _, _ = ECONS[key]
    files = sorted(DATA.glob(f"wts_{model}_{tag}_s[0-9][0-9][0-9].parquet"))
    if not files:
        return None
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["month", "firmid"] + CHARS + ["w_true", "w_fm", "w_ff", "w_dkkm"])
        d["_panel"] = int(f.stem[-3:]); parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    meta = json.load(open(str(files[0]).replace(".parquet", ".json")))
    return df, meta, len(files)


def prepare(df, wcol, label):
    m = df[["_panel", "month", "firmid"] + CHARS].copy()
    m["weight"] = df[wcol].to_numpy()
    rank_standardize_per_date(m, CHARS)
    before = len(m)
    m = m.dropna(subset=CHARS + ["weight"])
    cc = drop_constant_cols(m, CHARS)
    winsorize_per_group(m, "weight")
    print(f"  {label}: {before-len(m):,} NaN rows dropped, N={len(m):,}, chars={cc}", flush=True)
    X = m[cc].to_numpy(np.float32); y = m["weight"].to_numpy(np.float32)
    key = m[["_panel", "month", "firmid", "weight"]].reset_index(drop=True)
    return X, y, cc, key


def split_linear(key, X, col):
    """Within each (_panel, month) cross-section, OLS of key[col] on [1, X]; returns (linear, remainder)."""
    lin = np.empty(len(key)); grp = key.groupby(["_panel", "month"]).indices
    for _, idx in grp.items():
        A = np.column_stack([np.ones(len(idx)), X[idx]]); yv = key[col].to_numpy()[idx]
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None); lin[idx] = A @ beta
    return lin, key[col].to_numpy() - lin


def cell_mean_t(df, a, b):
    """mean over (panel, month) cells of corr(a, b); t from the across-panel sd of panel means."""
    c = df.groupby(["_panel", "month"]).apply(lambda x: x[a].corr(x[b]) if len(x) > 5 else np.nan).dropna()
    pm = c.groupby(level=0).mean()
    t = pm.mean() / (pm.std(ddof=1) / np.sqrt(len(pm))) if len(pm) > 1 else np.nan
    return float(c.mean()), float(t)


def pd2(net, sc, X, cc, a, b, n=21):
    """two-way partial dependence on a n x n grid over the [2.5, 97.5] pct range of each char."""
    ja, jb = cc.index(a), cc.index(b)
    ga = np.linspace(*np.quantile(X[:, ja], [0.025, 0.975]), n).astype(np.float32)
    gb = np.linspace(*np.quantile(X[:, jb], [0.025, 0.975]), n).astype(np.float32)
    base = X.mean(axis=0).astype(np.float32)
    rows = np.tile(base, (n * n, 1)); A, B = np.meshgrid(ga, gb, indexing="ij")
    rows[:, ja] = A.ravel(); rows[:, jb] = B.ravel()
    return predict_mlp(net, sc, rows).reshape(n, n)


def interaction_part(S):
    """S minus its additive fit (row + column means); the two-way interaction alone, centred."""
    return S - (S.mean(axis=1, keepdims=True) + S.mean(axis=0, keepdims=True) - S.mean())


def interaction_fig(fits_rows, labels_rows, a, b, path, title):
    """rows = economies, cols = true/fm/ff/dkkm: heat map of the interaction part of the fitted weight over (a, b),
    on each surface's own symmetric colour scale; the panel title carries the interaction's sd in weight units."""
    n = 41
    fig, axes = plt.subplots(len(fits_rows), 4, figsize=(17, 4.2 * len(fits_rows)), squeeze=False)
    for r, (fits, lbl) in enumerate(zip(fits_rows, labels_rows)):
        for c, mk in enumerate(["true", "fm", "ff", "dkkm"]):
            X, cc, net, sc = fits[mk]; ax = axes[r, c]
            ja, jb = cc.index(a), cc.index(b)
            ga = np.linspace(*np.quantile(X[:, ja], [0.025, 0.975]), n); gb = np.linspace(*np.quantile(X[:, jb], [0.025, 0.975]), n)
            I = interaction_part(pd2(net, sc, X, cc, a, b, n=n)); v = float(np.abs(I).max()) or 1.0
            im = ax.pcolormesh(ga, gb, I.T, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
            ax.contour(ga, gb, I.T, levels=6, colors="k", linewidths=0.4, alpha=0.5)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.set_title(f"{lbl}: {mk}   interaction sd {I.std():.3f}", fontsize=10)
            ax.set_xlabel(a); ax.set_ylabel(b)
    fig.suptitle(title, y=1.0); fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def H_stat(S, share=True):
    """share of var(PD_ab) not explained by the additive fit PD_a + PD_b (Friedman-style H^2);
    share=False returns the variance of the non-additive part itself (same units as the weights)."""
    add = S.mean(axis=1, keepdims=True) + S.mean(axis=0, keepdims=True) - S.mean()
    return float(np.var(S - add) / max(np.var(S), 1e-300)) if share else float(np.var(S - add))


def pd_panel(ax, net, sc, X, cc, vary, fixed):
    """common.make_pd_panel with the y-limits set from the curves, not from the zero line."""
    make_pd_panel(ax, net, sc, X, cc, vary=vary, fixed=fixed)
    ys = np.concatenate([l.get_ydata() for l in ax.get_lines() if len(l.get_xdata()) > 2] or [np.zeros(1)])
    lo, hi = float(ys.min()), float(ys.max()); pad = 0.08 * (hi - lo or 1.0)
    ax.set_ylim(lo - pad, hi + pad)


def main():
    RESULTS.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    rows, hrows, fits_all, pairs = [], [], {}, {}
    for key in ONLY:
        model, tag, fam, role = ECONS[key]
        got = load_econ(key)
        if got is None:
            print(f"\n========== {key} ({model}/{tag}): no extracted weights yet, skipped =========="); continue
        df, meta, nseeds = got
        print(f"\n========== {key} ({model}/{tag}, {nseeds} seeds, kappa={meta['kappa']}, {len(df):,} rows) ==========")
        fits, keys = {}, {}
        for m in ["true"] + METHODS:
            X, y, cc, k = prepare(df, f"w_{m}", f"{key}/{m}")
            net, sc = get_or_train_mlp(CACHE / f"mlp_{key}_{m}.pt", X, y)
            k["fitted"] = predict_mlp(net, sc, X)
            k["lin"], k["nl"] = split_linear(k, X, "fitted")
            k["alin"], k["anl"] = split_linear(k, X, "weight")
            fits[m] = (X, cc, net, sc); keys[m] = k
        fits_all[key] = fits
        # pair: the truth surface's strongest interaction
        Xt, cct, nett, sct = fits["true"]
        surf = {(a, b): pd2(nett, sct, Xt, cct, a, b) for i, a in enumerate(cct) for b in cct[i + 1:]}
        hs = {k: H_stat(S) for k, S in surf.items()}; iv = {k: H_stat(S, share=False) for k, S in surf.items()}
        top = max(iv, key=iv.get)                            # the pair whose interaction moves the true weight most
        fp = FIG_PAIRS[key.split("_")[0]]; pair = fp if fp in iv else fp[::-1]; pairs[key] = pair
        for (a, b), h in sorted(hs.items(), key=lambda kv: -iv[kv[0]]):
            hrows.append({"econ": key, "method": "true", "pair": f"{a}x{b}", "H": h, "int_var": iv[(a, b)]})
        for m in METHODS:
            X, cc, net, sc = fits[m]
            for (a, b) in hs:
                S = pd2(net, sc, X, cc, a, b)
                hrows.append({"econ": key, "method": m, "pair": f"{a}x{b}", "H": H_stat(S), "int_var": H_stat(S, share=False)})
        isd = {m: {k: float(np.sqrt(H_stat(pd2(fits[m][2], fits[m][3], fits[m][0], fits[m][1], *k), share=False)))
                   for k in {pair, top}} for m in ["true"] + METHODS}
        kt = keys["true"].rename(columns={"weight": "sdfwt", "fitted": "tfit", "lin": "tlin", "nl": "tnl", "alin": "talin", "anl": "tanl"})
        fc, _ = cell_mean_t(kt, "tfit", "sdfwt")
        nl_share_t = float(np.var(kt["tnl"]) / np.var(kt["tfit"]))
        rows.append({"econ": key, "family": fam, "role": role, "method": "true", "kappa": meta["kappa"], "seeds": nseeds,
                     "corr_actual_vs_truth": 1.0, "corr_fitted_vs_truth": fc, "nl_share_fitted": nl_share_t,
                     "corr_lin_vs_truth_lin": 1.0, "t_lin": np.nan, "corr_nl_vs_truth_nl": 1.0, "t_nl": np.nan,
                     "corr_actual_nl_vs_truth_nl": 1.0, "t_actual_nl": np.nan,
                     "H_pair": f"{pair[0]}x{pair[1]}", "H": hs[pair], "int_sd_pair": isd["true"][pair],
                     "top_pair": f"{top[0]}x{top[1]}", "int_sd_top": isd["true"][top], "n_obs": len(kt)})
        for m in METHODS:
            j = keys[m].merge(kt[["_panel", "month", "firmid", "sdfwt", "tlin", "tnl", "tanl"]], on=["_panel", "month", "firmid"])
            ac, _ = cell_mean_t(j, "weight", "sdfwt"); fc, _ = cell_mean_t(j, "fitted", "sdfwt")
            cl, tl = cell_mean_t(j, "lin", "tlin"); cn, tn = cell_mean_t(j, "nl", "tnl"); can, tan = cell_mean_t(j, "anl", "tanl")
            Xm, ccm, netm, scm = fits[m]
            rows.append({"econ": key, "family": fam, "role": role, "method": m, "kappa": meta["kappa"], "seeds": nseeds,
                         "corr_actual_vs_truth": ac, "corr_fitted_vs_truth": fc,
                         "nl_share_fitted": float(np.var(j["nl"]) / np.var(j["fitted"])),
                         "corr_lin_vs_truth_lin": cl, "t_lin": tl, "corr_nl_vs_truth_nl": cn, "t_nl": tn,
                         "corr_actual_nl_vs_truth_nl": can, "t_actual_nl": tan,
                         "H_pair": f"{pair[0]}x{pair[1]}", "H": H_stat(pd2(netm, scm, Xm, ccm, *pair)), "int_sd_pair": isd[m][pair],
                         "top_pair": f"{top[0]}x{top[1]}", "int_sd_top": isd[m][top], "n_obs": len(j)})
            print(f"  {m:5s} actual~truth {ac:+.4f}  fitted~truth {fc:+.4f}  lin~lin {cl:+.3f} (t {tl:.1f})  "
                  f"nl~nl {cn:+.3f} (t {tn:.1f})  H({pair[0]}x{pair[1]})={rows[-1]['H']:.3f}", flush=True)
        # PD 1x4 at the economy's own pair, both directions
        labels = [("true", "true SDF"), ("fm", "fm"), ("ff", "ff"), ("dkkm", f"dkkm k={meta['kappa']:g}")]
        for vary, fixed in [pair, pair[::-1]]:
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
            for ax, (mk, lbl) in zip(axes, labels):
                X, cc, net, sc = fits[mk]; pd_panel(ax, net, sc, X, cc, vary, fixed); ax.set_title(lbl)
            axes[0].set_ylabel("fitted weight")
            fig.suptitle(f"{key} ({model}/{tag}): fitted weight vs {vary} at five levels of {fixed}", y=1.02)
            fig.tight_layout(); fig.savefig(RESULTS / f"pd_1x4_{key}_vary_{vary}_fix_{fixed}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # family comparison figures: benchmark row vs 3b row at the 3b pair
    for fam in ["bgn", "gs", "kp"]:
        kb, k3 = f"{fam}_base", f"{fam}_3b"
        if kb not in fits_all or k3 not in fits_all: continue
        pair = pairs[k3]
        interaction_fig([fits_all[kb], fits_all[k3]], [f"{kb} ({ECONS[kb][1]})", f"{k3} ({ECONS[k3][1]})"], pair[0], pair[1],
                        RESULTS / f"interaction_{fam}_{pair[0]}x{pair[1]}.png",
                        f"{ECONS[k3][2]}: the interaction part of the fitted weight over ({pair[0]}, {pair[1]}); benchmark (top) vs 3b (bottom)")
        print(f"wrote interaction_{fam}_{pair[0]}x{pair[1]}.png")
        for vary, fixed in [pair, pair[::-1]]:
            fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
            for r, kk in enumerate([kb, k3]):
                for c, mk in enumerate(["true", "fm", "ff", "dkkm"]):
                    X, cc, net, sc = fits_all[kk][mk]; ax = axes[r, c]
                    pd_panel(ax, net, sc, X, cc, vary, fixed)
                    ax.set_title(f"{kk} ({ECONS[kk][1]}): {mk}")
                axes[r, 0].set_ylabel("fitted weight")
            fig.suptitle(f"{ECONS[k3][2]}: benchmark (top) vs 3b (bottom); weight vs {vary} at five levels of {fixed}", y=1.01)
            fig.tight_layout(); fig.savefig(RESULTS / f"compare_{fam}_vary_{vary}_fix_{fixed}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"wrote compare_{fam}_vary_{vary}_fix_{fixed}.png")
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame(hrows).to_csv(RESULTS / "H_pairs.csv", index=False)
    print("\n=== metrics ===\n" + out.drop(columns=["family", "n_obs", "t_lin", "t_actual_nl", "corr_actual_nl_vs_truth_nl"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
