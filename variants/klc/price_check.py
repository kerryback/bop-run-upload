#!/usr/bin/env python3
"""Does the economy's SDF, projected onto its own returns, price the SIMULATED returns?

    python variants/klc/price_check.py --results DIR --model bgn_gam --tag g0235df [--seeds 0,1,...]

WHAT IS BEING TESTED, AND WHY IT IS A TEST OF THE SOLVE. In all three models the SDF M is
exogenous and known in closed form: BGN  log M = -r - 0.5 (gmult_s sigma_z)^2 - gmult_s sigma_z nu
(plus the rate channel through beta_zr); KP14  constant prices gamma_x, gamma_z on the two GBM
shocks and gamma_v on the OU state's Brownian; GS21  M(x, x') = exp(-r - 0.5 gamma(x)^2 -
gamma(x) eps') renormalised so E[M | x] = e^{-r}. The SOLVE is what turns that kernel into firm
values -- J* for BGN's projects, the A and G tables for KP14's assets in place and growth options,
the value-function tables for GS21's equity and debt -- and sdf_compute then differentiates those
values into every firm's conditional expected excess return mu_t and covariance Sigma_t
analytically. The simulated return R^e_{t+1} is produced from the SAME values along the drawn
shock path. So if the solve is right, mu_t and Sigma_t are the true conditional moments of the
simulated R^e_{t+1}, and the SDF's projection onto excess returns,

        m_{t+1} = 1 - mu_t' Sigma_t^{-1} (R^e_{t+1} - mu_t)        (E_t[m] = 1)

prices every asset: E_t[m R^e_i] = mu_i - (Sigma Sigma^{-1} mu)_i = 0, with E_t[m] = 1. In an exact
K-factor economy the projection onto returns is the SDF up to the part orthogonal to returns,
so this is the whole of what asset returns can say about M. The projection is NOT a trivial
identity on the simulated data: it uses the ANALYTICAL mu_t and Sigma_t but the REALISED
R^e_{t+1}, so it fails when the solve gives a wrong mean (mispriced claims -- the 2026-09-18
defects were of this kind) or a wrong covariance (the loadings that Sigma_t^{-1} mu_t weights).

The sample pricing errors e_{i,t} = m_{t+1} R^e_{i,t+1} are martingale differences under the
truth, so per-firm t-statistics should look standard normal across firms, and the same for
characteristic-sorted long-short portfolios and the market. A naive SDF (m = 1, risk neutrality)
is scored beside it as the power check: it should fail wherever premia are non-trivial.

Also reported: the realised Sharpe of the mimicking portfolio against the mean conditional
maximum Sharpe sqrt(mu' Sigma^{-1} mu) (they agree only if BOTH moments are right), the slope of
realised on expected excess returns (1 under the truth), and the agreement of this file's
Sigma^{-1} mu direction with the `w_true` the simulator's own sdf_compute saved.

Reads the files run_oracle.py --save_panel writes: {model}_panel_{tag}_s{seed}.parquet and
{model}_moments_{tag}_s{seed}.npz (mu, Sigma, w_true, keep, months). Nothing is re-simulated.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.linalg as la

CHARS = ["size", "bm", "agr", "roe", "mom"]


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def pooled_slope(sq, nboot=400, seed=0):
    """Pooled OLS slope of realised on expected excess returns across all firm-months, with a
    month-block bootstrap t-statistic against 1. Per-month slopes are far too noisy to average:
    the cross-sectional dispersion of mu is a few basis points against return noise of several
    percent, so each month's slope has a standard error in the tens."""
    S, Q = np.array([s for s, _ in sq]), np.array([q for _, q in sq])
    slope = S.sum() / Q.sum()
    rng = np.random.default_rng(seed); n = len(S)
    bs = np.array([(S[i].sum() / Q[i].sum()) for i in (rng.integers(0, n, n) for _ in range(nboot))])
    return float(slope), float((slope - 1.0) / bs.std(ddof=1))


def one_seed(results, model, tag, seed):
    stem = os.path.join(results, f"{model}_%s_{tag}_s{seed:03d}")
    panel = pd.read_parquet(stem % "panel" + ".parquet").set_index(["month", "firmid"]).sort_index()
    mom = np.load(stem % "moments" + ".npz")
    months, MU, SIG, KEEP, WT = mom["months"], mom["mu"], mom["Sigma"], mom["keep"], mom["w_true"]
    err_m, err_naive, port_m, port_naive, mimic, srmax, slopes, cosines = {}, {}, [], [], [], [], [], []
    for i, month in enumerate(months):
        data = panel.loc[int(month)]
        keep = data.index.to_numpy(); n = len(keep)
        assert np.array_equal(KEEP[i, :n], keep), f"month {month}: firm set differs from the moments file"
        mu, Sig, x = MU[i, :n].astype(float), SIG[i, :n, :n].astype(float), data.xret.to_numpy(float)
        b = la.solve(Sig, mu, assume_a="pos")
        m = 1.0 - b @ (x - mu)                                   # the SDF projected onto excess returns
        srmax.append(np.sqrt(mu @ b)); mimic.append(b @ x / np.sqrt(mu @ b))   # unit-variance mimicking return
        w = WT[i, :n].astype(float)
        if np.linalg.norm(w) > 0:
            cosines.append(abs(w @ b) / (np.linalg.norm(w) * np.linalg.norm(b)))
        dm, dx = mu - mu.mean(), x - x.mean()
        slopes.append((dm @ dx, dm @ dm))               # month's covariance and variance: pooled slope later
        for f, e_m, e_n in zip(keep, m * x, x):
            err_m.setdefault(int(f), []).append(e_m); err_naive.setdefault(int(f), []).append(e_n)
        # portfolios: EW market and decile long-short on each characteristic, formed on this month's chars
        ew = np.full(n, 1.0 / n)
        ports = {"ew_mkt": ew}
        for c in CHARS:
            v = data[c].to_numpy(float); ok = np.isfinite(v)
            if ok.sum() < 20:
                continue
            lo, hi = np.nanquantile(v, 0.1), np.nanquantile(v, 0.9)
            wgt = np.zeros(n); wgt[ok & (v >= hi)] = 1.0; wgt[ok & (v <= lo)] = -1.0
            if (wgt > 0).any() and (wgt < 0).any():
                wgt[wgt > 0] /= (wgt > 0).sum(); wgt[wgt < 0] /= (wgt < 0).sum()
                ports[f"hml_{c}"] = wgt
        port_m.append({k: m * (wv @ x) for k, wv in ports.items()})
        port_naive.append({k: wv @ x for k, wv in ports.items()})
    firm_t = np.array([tstat(v) for v in err_m.values() if len(v) >= 24])
    firm_t_naive = np.array([tstat(v) for v in err_naive.values() if len(v) >= 24])
    pm, pn = pd.DataFrame(port_m), pd.DataFrame(port_naive)
    mimic = np.array(mimic); srmax = np.array(srmax)
    return {
        "months": len(months), "firms": len(firm_t),
        "firm_t_mean": firm_t.mean(), "firm_t_sd": firm_t.std(ddof=1), "firm_t_rej5": (np.abs(firm_t) > 1.96).mean(),
        "firm_t_max": np.abs(firm_t).max(),
        "naive_firm_t_sd": firm_t_naive.std(ddof=1), "naive_firm_t_rej5": (np.abs(firm_t_naive) > 1.96).mean(),
        "port_t": {k: tstat(pm[k]) for k in pm}, "naive_port_t": {k: tstat(pn[k]) for k in pn},
        "mimic_sr_realised": mimic.mean() / mimic.std(ddof=1), "srmax_cond_mean": srmax.mean(),
        "slope_realised_on_expected": pooled_slope(slopes)[0], "slope_t_vs_1": pooled_slope(slopes)[1],
        "cos_wtrue": float(np.min(cosines)) if cosines else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default=None, help="comma list; default: every seed with a saved panel")
    a = ap.parse_args()
    if a.seeds:
        seeds = [int(s) for s in a.seeds.split(",")]
    else:
        seeds = sorted(int(re.search(r"_s(\d{3})\.parquet$", f).group(1))
                       for f in glob.glob(os.path.join(a.results, f"{a.model}_panel_{a.tag}_s???.parquet")))
    if not seeds:
        sys.exit(f"no saved panels for {a.model}/{a.tag} under {a.results}")
    print(f"{a.model}/{a.tag}: {len(seeds)} seed(s) {seeds}\n")
    hdr = f"{'seed':>4} {'months':>6} {'firms':>5} | {'firm t: mean':>12} {'sd':>6} {'rej5%':>6} {'max|t|':>6} | {'naive sd':>8} {'rej5%':>6} | {'mkt t':>7} {'naive':>7} | {'SR mimic':>8} {'SRmax':>6} | {'slope':>6} {'t(=1)':>6} | {'cos w':>6}"
    print(hdr); print("-" * len(hdr))
    bad = 0
    for s in seeds:
        r = one_seed(a.results, a.model, a.tag, s)
        print(f"{s:>4} {r['months']:>6} {r['firms']:>5} | {r['firm_t_mean']:>12.3f} {r['firm_t_sd']:>6.3f} {r['firm_t_rej5']:>6.3f} {r['firm_t_max']:>6.2f} | "
              f"{r['naive_firm_t_sd']:>8.2f} {r['naive_firm_t_rej5']:>6.3f} | {r['port_t']['ew_mkt']:>7.2f} {r['naive_port_t']['ew_mkt']:>7.2f} | "
              f"{r['mimic_sr_realised']:>8.4f} {r['srmax_cond_mean']:>6.4f} | {r['slope_realised_on_expected']:>6.3f} {r['slope_t_vs_1']:>6.2f} | {r['cos_wtrue']:>6.4f}")
        ls = "    long-short deciles, t of E[m R^e]:  " + "  ".join(f"{k[4:]} {v:+.2f} (naive {r['naive_port_t'][k]:+.2f})" for k, v in r["port_t"].items() if k != "ew_mkt")
        print(ls)
        # what counts as a failure: a firm-t distribution clearly wider than N(0,1), a market or
        # long-short pricing error beyond 3 se, or the mimicking Sharpe far from the conditional max
        flags = []
        if r["firm_t_sd"] > 1.25 or r["firm_t_rej5"] > 0.12: flags.append("firm pricing errors wider than N(0,1)")
        if abs(r["port_t"]["ew_mkt"]) > 3: flags.append("market not priced")
        if any(abs(v) > 3 for k, v in r["port_t"].items() if k != "ew_mkt"): flags.append("a long-short portfolio not priced")
        if abs(r["slope_t_vs_1"]) > 3: flags.append("realised-on-expected slope differs from 1")
        if np.isfinite(r["cos_wtrue"]) and r["cos_wtrue"] < 0.999: flags.append("Sigma^-1 mu disagrees with the saved w_true")
        if flags:
            bad += 1; print("    FLAG: " + "; ".join(flags))
    print("\nReading: with the correct solve the firm-t column is ~N(0,1) (sd near 1, about 5% beyond 1.96) and every "
          "portfolio t is within a few units of zero, while the naive m = 1 columns reject wherever premia exist. "
          "SR mimic is the realised Sharpe of Sigma^-1 mu over these months against the mean conditional maximum.")
    print(f"\n{'PRICING CHECK PASSED' if not bad else f'PRICING CHECK: {bad} seed(s) flagged'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
