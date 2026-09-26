"""How linearly spannable is an economy's TRUE conditional premium?

Two statistics, both measured on a saved panel and its true conditional moments, both defined
against the five rank-standardised characteristics the estimators actually see:

  theta_month   1 - R^2 of mu_t on [1, chars], fit FRESH each month. The share of the premium
                cross-section no linear map can reach at a point in time. Reported two ways: the
                UNWEIGHTED mean over months (finding 11's convention, and what its withdrawal
                quotes) and the VARIANCE-WEIGHTED sum-of-squares ratio, which is the one that
                compares to theta_pooled.
  theta_pooled  the same with ONE coefficient vector for every month, variance-weighted.

theta_pooled - theta_month is the share of premium variance carried by TIME VARIATION in the
linear map. A rolling estimator fits one coefficient vector per window, so that difference is
what it structurally cannot track.

This is the evidence behind finding 15 in docs/RESULTS.md, and behind the WITHDRAWAL of finding
11's implied-theta column. The lesson it encodes: measure spannability against the basis the
ESTIMATOR uses, not against whatever parameters the economy was designed in. `vym3` was designed
so that no linear map in (magnitude, alignment) could represent its premium -- true, R^2 0.885 --
and it made no difference, because the estimators never see magnitude or alignment.

Needs the panel and moments files, which are gitignored (~21 MB and ~324 MB per seed). Pull them
from the cluster results directory first:

    rsync -a sjpruitt@phx.asu.edu:/data/sjpruitt/GitHub/bop-run-upload/variants/results/kp_vy_panel_vyg25_s000.parquet .
    rsync -a sjpruitt@phx.asu.edu:/data/sjpruitt/GitHub/bop-run-upload/variants/results/kp_vy_moments_vyg25_s000.npz .
    python variants/diagnostics/premium_shape.py --dir . kp_vy/vyg25 kp_vy/vym3

Measured 2026-09-25, seed 0, 485 months:

    vyg25   mean 0.1995   weighted 0.1955   pooled 0.2928   time-variation +0.0973   fair gap +0.0148
    vym3    mean 0.1871   weighted 0.1971   pooled 0.3007   time-variation +0.1037   fair gap -0.0032
"""
import argparse
import os

import numpy as np
import pandas as pd

CHARS = ["size", "bm", "agr", "roe", "mom"]


def ranked(df, cols):
    """The estimators see rank-standardised characteristics; span them the same way."""
    out = np.empty((len(df), len(cols)))
    for j, c in enumerate(cols):
        out[:, j] = df.groupby("month")[c].rank(pct=True).to_numpy() - 0.5
    return out


def monthly_fits(panel, mom, chars=CHARS, min_firms=30):
    """Per month, the design matrix [1, ranked chars], the true premia, and the month's own OLS fit."""
    months, mu, keep = mom["months"], mom["mu"], mom["keep"]
    have = [c for c in chars if c in panel.columns]
    missing = sorted(set(chars) - set(have))
    if missing:
        raise SystemExit(f"panel is missing characteristics {missing}")
    fits = []
    for i, m in enumerate(months):
        sub = panel[panel.month == m]
        if sub.empty:
            continue
        k = keep[i]
        k = k[k >= 0]
        mus = mu[i][: len(k)]
        pos = pd.Index(sub.firmid.to_numpy()).get_indexer(k)
        ok = pos >= 0
        if ok.sum() < min_firms:
            continue
        X, y = ranked(sub, have)[pos[ok]], mus[ok]
        g = np.isfinite(y) & np.isfinite(X).all(1)
        if g.sum() < min_firms:
            continue
        A = np.column_stack([np.ones(int(g.sum())), X[g]])
        b, *_ = np.linalg.lstsq(A, y[g], rcond=None)
        fits.append((A, y[g], b))
    if not fits:
        raise SystemExit("no month had enough firms to fit")
    return fits, have


def theta(fits):
    """(mean per-month theta, variance-weighted per-month theta, pooled theta).

    Each month is demeaned by its OWN mean throughout. The mean and the weighted figure differ
    because months vary in both firm count and premium dispersion; quote the mean against
    finding 11, the weighted one against theta_pooled."""
    per_month = []
    for A, y, b in fits:
        tss_m = float(((y - y.mean()) ** 2).sum())
        if tss_m > 0:
            per_month.append(float(((y - A @ b) ** 2).sum()) / tss_m)
    rss_month = sum(float(((y - A @ b) ** 2).sum()) for A, y, b in fits)
    bp, *_ = np.linalg.lstsq(np.vstack([A for A, _, _ in fits]),
                             np.concatenate([y for _, y, _ in fits]), rcond=None)
    rss_pooled = sum(float(((y - A @ bp) ** 2).sum()) for A, y, _ in fits)
    tss = sum(float(((y - y.mean()) ** 2).sum()) for _, y, _ in fits)
    return float(np.mean(per_month)), rss_month / tss, rss_pooled / tss


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("economies", nargs="+", metavar="model/tag",
                    help="e.g. kp_vy/vyg25 kp_vy/vym3")
    ap.add_argument("--dir", default="variants/results",
                    help="where the panel and moments files are (default variants/results)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print(f"{'economy':16s} {'months':>7s} {'theta mean':>11s} {'theta wtd':>10s} "
          f"{'theta pooled':>13s} {'time-var':>10s}")
    for spec in args.economies:
        model, tag = spec.split("/", 1)
        stem = os.path.join(args.dir, f"{model}_%s_{tag}_s{args.seed:03d}")
        panel = pd.read_parquet((stem % "panel") + ".parquet")
        mom = np.load((stem % "moments") + ".npz", allow_pickle=True)
        fits, _ = monthly_fits(panel, mom)
        tmean, twtd, tp = theta(fits)
        print(f"{spec:16s} {len(fits):>7d} {tmean:>11.4f} {twtd:>10.4f} {tp:>13.4f} "
              f"{tp - twtd:>+10.4f}")


if __name__ == "__main__":
    main()
