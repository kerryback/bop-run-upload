"""fig_3c_premium.png: the premium each method bets on, projected on the attribute the 3c curvature is written on.

For firm i in month t the true conditional expected excess return mu_it comes from the oracle moments; a method with
portfolio weights w_m is the mean-variance portfolio of the implied premium Sigma_t w_m (up to a scalar), so
mu_hat^m_it = c_m (Sigma_t w^m_t)_i with one scalar c_m per economy and method, fitted by least squares to mu_it
(for the true weights this reproduces mu exactly up to the risk-aversion scalar). Curves: E[mu | characteristic] and
E[mu_hat^m | characteristic] by 25 quantile bins of the rank-standardised characteristic, pooled over seeds and the
evaluation months, in percent per month.  usage: python sec3_premium_fig.py [--seeds N]"""
import sys, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES = "/kellogg/proj/axa3203/bop_runs/results"; DATA = "/kellogg/proj/axa3203/bop_runs/mlp/data"
CACHE = f"{HERE}/cache" if 'HERE' in dir() else "/kellogg/proj/axa3203/bop_runs/mlp/sec3/cache"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
NS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 10
ECONS = [("BGN", "bgn_vy", "bvyhx4", "size", "size"), ("GS21", "gs_vy", "gvylqun10", "lev", "market leverage"), ("KP14", "kp_vy", "vyhx3", "size", "size")]
METHODS = [("true", "true premium", "k", "-", 2.2), ("fm", "FM", "tab:green", "--", 1.5), ("ff", "FF", "tab:orange", "--", 1.5), ("dkkm", "DKKM", "tab:blue", "-", 1.6)]


def implied(model, tag):
    """Per firm-month: true premium mu, chars, and each method's implied premium hat_m = c_m (Sigma w_m) with
    c_m the least-squares scale of Sigma w_m onto mu. Reads the cache written by fix_wtrue_cache.py."""
    d = pd.read_parquet(f"{CACHE}/prem_{model}_{tag}.parquet")
    d = d[d.seed < NS].copy()
    for c in ["size", "lev"]:
        if c in d: d[c] = d.groupby(["seed", "month"])[c].rank(pct=True) - 0.5
    d["hat_true"] = d.mu                                           # the truth IS the premium
    for mk in ["fm", "ff", "dkkm"]:                                # one scalar per method
        x = d[f"sw_{mk}"].to_numpy(); c = float(x @ d.mu.to_numpy() / (x @ x)); d[f"hat_{mk}"] = c * x
        print(f"  {tag} {mk:5s} scale {c:+.3g}  corr(hat, mu) {np.corrcoef(d[f'hat_{mk}'], d.mu)[0,1]:+.3f}", flush=True)
    return d


def proj(x, y, nb=25):
    lo, hi = np.quantile(x, [0.02, 0.98]); m = (x >= lo) & (x <= hi)
    b = pd.qcut(pd.Series(x[m]).rank(method="first"), nb, labels=False).to_numpy()
    return np.bincount(b, x[m]) / np.bincount(b), np.bincount(b, y[m]) / np.bincount(b)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, (fam, model, tag, ch, nice) in zip(axes, ECONS):
        print(f"[{fam}] {tag}", flush=True); d = implied(model, tag); x = d[ch].to_numpy()
        for mk, lbl, col, ls, lw in METHODS:
            gx, gy = proj(x, d[f"hat_{mk}"].to_numpy()); ax.plot(gx, gy, color=col, ls=ls, lw=lw, label=lbl)
        ax.axhline(0, color="0.7", lw=0.6); ax.set_title(f"{fam} ({tag})", fontsize=11); ax.set_xlabel(f"{nice} (rank-standardised)")
    axes[0].set_ylabel("E[premium | char]  (% per month)"); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_3c_premium.png", dpi=160, bbox_inches="tight"); fig.savefig(f"{OUT}/fig_3c_premium.pdf", bbox_inches="tight")
    print("wrote fig_3c_premium.png")


if __name__ == "__main__":
    main()
