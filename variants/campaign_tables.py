"""The campaign report's main tables: per group of experiments, the theoretical room for nonlinearity
relative to the linear ceiling, and the Sharpe of each standard estimator.

For every economy and seed, the ORACLE file gives the ceilings over the evaluation window: the best linear
(lin_rank) and the best nonlinear (rff / bins / poly2) conditional Sharpe a population investor reaches, and
room = nonlinear - linear (the room % column is room / linear). The ESTIMATORS summary gives each method's
out-of-sample Sharpe at the protocol window: DKKM (best of the rff variants), FMR (Fama-MacBeth), FF
(Fama-French factors), the linear model (best of linrank / linlev, with and without the market), CAPM
(mkt_est, the estimated market line), and the fair gap = DKKM - best of (FMR, FF, linear, CAPM), the
project's headline statistic (aggregate_seeds.py's definition).

Rows are averages over seeds; '+' is the number of seeds with a positive fair gap. A dagger marks an
economy whose oracle predates the evaluation-window ceilings (room over all months instead).

usage: python variants/campaign_tables.py --out docs/campaign_tables   (writes .md and .tex)
"""
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aggregate_seeds import RFF, FAIR, _nonlin_keys          # noqa: E402

LEGACY = os.path.join(HERE, "results")
# the campaign's results live on Quest and are committed, same files and names, in variants/results (the panels and
# moments stay on Quest; variants/results/README_quest.md); the reduced-form screens in variants/reduced/screen
CAMPAIGN = next(d for d in ("/kellogg/proj/axa3203/bop_runs/results", LEGACY) if os.path.isdir(d))
SCREEN = next(d for d in ("/kellogg/proj/axa3203/bop_runs/screen", os.path.join(HERE, "reduced", "screen")) if os.path.isdir(d))
# GS21 estimators re-scored on the oracle's six characteristics (the five plus leverage, 2026-10-26; run_estimators.py).
# The earlier scores in CAMPAIGN / LEGACY used five, so every GS21 estimator there is read against a room it could not
# see. A GS21 economy with any re-scored seed takes its estimator columns from here and only from here (never a mix of
# feature sets); one with none keeps the five-characteristic scores and is marked with a double dagger.
LEV = next((d for d in ("/kellogg/proj/axa3203/bop_runs/results_lev", os.path.join(HERE, "results_lev")) if os.path.isdir(d)), None)
LINEAR = ("linrank", "linlev", "linrank_m", "linlev_m")

# (tag, model, results dir, what) in the order the tables print them
E = {
    "bgnbase": ("BGN", LEGACY, "baseline, no priced state beyond the rate"),
    "gsbase": ("GS21", LEGACY, "baseline"),
    "kpbase": ("KP14", LEGACY, "baseline, one type, no priced state"),
    # group 1: distributions
    "bvych0": ("BGN", CAMPAIGN, "y as a 21-node chain, three types (control)"),
    "bvyj25": ("BGN", CAMPAIGN, "the chain with a priced crash (1\\%/month to $-y_{\\max}$, priced $e^2$)"),
    "gvyp10": ("GS21", CAMPAIGN, "product exposure $\\tilde z\\tilde b$ at 0.10 (control)"),
    "gvyp10j": ("GS21", CAMPAIGN, "the product exposure with a priced crash"),
    "vyg25": ("KP14", LEGACY, "three types on y, price 2.5 (control)"),
    "vyj25": ("KP14", CAMPAIGN, "the same with a priced crash in y"),
    # group 2: prices of risk
    "g0235f": ("BGN", LEGACY, "regime price $\\times$0.2 / $\\times$3.5, spells 4$\\times$ faster"),
    "g0235df": ("BGN", CAMPAIGN, "stress 2/3 of the time, 4$\\times$ faster"),
    "g28": ("GS21", LEGACY, "countercyclical $\\gamma(x)$, clipped"),
    "bx7": ("GS21", LEGACY, "two-state regime on $\\gamma$, five types on x"),
    "kpgam": ("KP14", CAMPAIGN, "$\\gamma_x \\times$ logistic(y), 1 to 3"),
    # group 3a: single-characteristic exposures
    "gx7": ("GS21", LEGACY, "five types on productivity x"),
    "gvy25": ("GS21", CAMPAIGN, "production-level exposure to y, three types"),
    "gvys25": ("GS21", CAMPAIGN, "signed fixed-cost exposure to y"),
    "bvysg": ("BGN", CAMPAIGN, "signed exposure ladder ($-0.07$ / 0 / $+0.07$)"),
    "vyx": ("KP14", LEGACY, "three types on y, price 1.8"),
    # group 3b: interactions
    "bvy25": ("BGN", CAMPAIGN, "three types on y, level-compensated (1.2)"),
    "bvyc0": ("BGN", CAMPAIGN, "the same without compensation"),
    "gvyp05": ("GS21", CAMPAIGN, "product $\\tilde z\\tilde b$ at 0.05"),
    "gvyp075": ("GS21", CAMPAIGN, "product at 0.075"),
    "gvyp15": ("GS21", CAMPAIGN, "product at 0.15"),
    "gvyp20": ("GS21", CAMPAIGN, "product at 0.20"),
    "gvyp10x": ("GS21", CAMPAIGN, "product at 0.10 plus a curvature loading on x"),
    # group 4: larger factor models
    "vym3": ("KP14", LEGACY, "three priced states, twenty rank-3 types"),
    "vym3t3": ("KP14", LEGACY, "three priced states, three types"),
    "bvyk2": ("BGN", CAMPAIGN, "two priced chains, three loading vectors"),
    "gvypk2": ("GS21", CAMPAIGN, "two priced chains, two product types"),
    "gvyzl": ("GS21", CAMPAIGN, "productivity exposure, three fixed-coupon leverage types (diluting)"),
    "gvyzlf": ("GS21", CAMPAIGN, "productivity exposure, three leverage types, capital structure fixed for life"),
    "gvyzlw": ("GS21", CAMPAIGN, "gvyzlf with coupons 0.05 / 0.45 / 0.85"),
    "gvyzl5": ("GS21", CAMPAIGN, "gvyzlf with five leverage types"),
    "gvyzlb": ("GS21", CAMPAIGN, "gvyzlf with the exposure doubled"),
    "gvyzlr": ("GS21", CAMPAIGN, "gvyzlf with riskless debt"),
    "gvyzlp": ("GS21", CAMPAIGN, "gvyzlf with a persistent y ($\\kappa_y$ 0.10)"),
    "gvyzlrp": ("GS21", CAMPAIGN, "gvyzlf with riskless debt and a persistent y"),
    "gsbg1": ("GS21", CAMPAIGN, "benchmark at $\\gamma_x$ 0.1"),
    "gvyz10g": ("GS21", CAMPAIGN, "gvyz10 at $\\gamma_x$ 0.1"),
    "gvyzlg": ("GS21", CAMPAIGN, "gvyzlf at $\\gamma_x$ 0.1"),
    "gvyp10c": ("GS21", CAMPAIGN, "the product with a leverage-level term, exposure 0.10 $\\tilde b(\\tilde z + 2)$"),
    "gvytl": ("GS21", CAMPAIGN, "three exposure types $\\times$ three fixed leverage types (KP14's recipe)"),
    "gvytlg": ("GS21", CAMPAIGN, "the same at $\\gamma_x$ 0.1"),
    # harmonized representatives (2026-10-14): one type exposed to y, the crash-only factor, the quadratic price, two states
    "vy1": ("KP14", CAMPAIGN, "one type, cash flows $\\times e^{0.07y}$"),
    "bvy1ch": ("BGN", CAMPAIGN, "one type, project cash flows $\\times e^{0.035y}$"),
    "gvy1": ("GS21", CAMPAIGN, "one type, output $\\times e^{0.05y}$"),
    "vy1j": ("KP14", CAMPAIGN, "crash-only factor: $e^{0.07y}$, y unpriced except a 1\\%/month jump to $-3.5$ (weight $e^2$)"),
    "bvy1j": ("BGN", CAMPAIGN, "crash-only factor: $e^{0.035y}$, the same crash"),
    "gvy1j": ("GS21", CAMPAIGN, "crash-only factor: $e^{0.05y}$, the same crash"),
    "kpq": ("KP14", CAMPAIGN, "$\\gamma_x \\times \\min(1 + 0.25\\,y^2, 3)$, y unpriced"),
    "gq25": ("GS21", CAMPAIGN, "$\\gamma_x \\times \\min(1 + 0.25\\,\\tilde x^2, 3)$"),
    "vym2t3": ("KP14", CAMPAIGN, "two priced states, three loading vectors"),
    "vy9t2": ("KP14", CAMPAIGN, "two priced states, nine types (independent ladders), total price 2.5"),
    "vy9t2f": ("KP14", CAMPAIGN, "two priced states, nine types, each state priced 2.5"),
    "bvy9t2": ("BGN", CAMPAIGN, "two priced chains, nine types (independent ladders), total price 2.5"),
    # the sign of the exposure and the level compensation (2026-10-26)
    "vyg25s": ("KP14", CAMPAIGN, "vyg25 with sign-switching types: six types, loadings $\\pm 0.02, \\pm 0.07, \\pm 0.14$"),
    "vyg25u": ("KP14", CAMPAIGN, "vyg25 without the level compensation: the type shows in value at $y = 0$"),
    "bvychs": ("BGN", CAMPAIGN, "bvych0 with sign-switching types: six types, loadings $\\pm 0.01, \\pm 0.035, \\pm 0.07$"),
    # the three routes to a larger gap (2026-10-27): a hidden sign in GS21, sign-switching prices of risk, K past the characteristic count
    "gvy25s": ("GS21", CAMPAIGN, "gvy25 with sign-switching types: six level-exposure types, $\\pm 0.015, \\pm 0.05, \\pm 0.1$"),
    "gvy2s": ("GS21", CAMPAIGN, "gvy25s's signed ladder over two priced chains, cash-flow exposure only: twelve mean-zero types"),
    "gvytls": ("GS21", CAMPAIGN, "gvytl with the exposure ladder mirrored: six signed exposure types $\\times$ three fixed coupons, eighteen mean-zero types"),
    "bvychs4": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 0.4"),
    "bvychs6": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 0.6"),
    "bvychs8": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 0.8"),
    "vyg25s6": ("KP14", CAMPAIGN, "vyg25s at bv\\_comp 0.6"),
    "vyg25s18": ("KP14", CAMPAIGN, "vyg25s at bv\\_comp 1.8"),
    "gvyib": ("GS21", CAMPAIGN, "investment cost $\\times e^{-b\\tilde{} y}$: leverage times the option's exposure, one type"),
    "gvyib5": ("GS21", CAMPAIGN, "gvyib at half the magnitude (cost $\\times e^{-0.5\\,b\\tilde{} y}$; the $-1.0$ solve diverged)"),
    "gvyib2": ("GS21", CAMPAIGN, "gvyib at gs\\_bib $-0.2$ (the $-1.0$ and $-0.5$ solves diverged: the firm picks the coupon that makes investment near-free)"),
    "gxq35": ("GS21", CAMPAIGN, "gsbase with the x loading hump-shaped in productivity: $1 -0.35\\,(\\tilde z^2 - 1)/\\sqrt2$"),
    "gxq70": ("GS21", CAMPAIGN, "gsbase with the x loading hump-shaped in productivity: $1 -0.70\\,(\\tilde z^2 - 1)/\\sqrt2$"),
    "gvy25sq": ("GS21", CAMPAIGN, "gvy25s at $\\gamma_v$ 0.25, bv\\_comp 12.0: the level shifts kept, the y price cut"),
    "gvy25sh": ("GS21", CAMPAIGN, "gvy25s at $\\gamma_v$ 0.5, bv\\_comp 9.0: the level shifts kept, the y price cut"),
    "bvychs18": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 1.8"),
    "bvychs25": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 2.5"),
    "bvychs30": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 3.0"),
    "bvychs35": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 3.5"),
    "bvychs40": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 4.0"),
    "bvychs45": ("BGN", CAMPAIGN, "bvychs at bv\\_comp 4.5"),
    "bvychsn5": ("BGN", CAMPAIGN, "bvychs at bv\\_comp -0.5"),
    "bvyvee": ("BGN", CAMPAIGN, "V-shaped loading across a visible level ladder: level 0.6..1.4, $b$ = +0.07, +0.00, -0.07, +0.00, +0.07"),
    "bvyhump": ("BGN", CAMPAIGN, "hump-shaped loading across a visible level ladder: level 0.6..1.4, $b$ = -0.07, +0.00, +0.07, +0.00, -0.07"),
    "gvyql10": ("GS21", CAMPAIGN, "U-shaped exposure in the firm's own leverage: $+0.1\\,(\\tilde b^2 - 1)$, one type"),
    "gvyql20": ("GS21", CAMPAIGN, "U-shaped exposure in the firm's own leverage: $+0.2\\,(\\tilde b^2 - 1)$, one type"),
    "gvyqh10": ("GS21", CAMPAIGN, "hump-shaped exposure in the firm's own leverage: $-0.1\\,(\\tilde b^2 - 1)$, one type"),
    "vyg25sx2": ("KP14", CAMPAIGN, "vyg25s with the loadings doubled: $\\pm$0.04, $\\pm$0.14, $\\pm$0.28"),
    "vyg25sx3": ("KP14", CAMPAIGN, "vyg25s with the loadings tripled: $\\pm$0.06, $\\pm$0.21, $\\pm$0.42"),
    "gvyzz10": ("GS21", CAMPAIGN, "U-shaped exposure in the firm's own productivity: $+0.1\\,(\\tilde z^2 - 1)$, one type"),
    "gvyzz20": ("GS21", CAMPAIGN, "U-shaped exposure in the firm's own productivity: $+0.2\\,(\\tilde z^2 - 1)$, one type"),
    "gvyzzh10": ("GS21", CAMPAIGN, "hump-shaped exposure in the firm's own productivity: $-0.1\\,(\\tilde z^2 - 1)$, one type"),
    "gvy25sc25": ("GS21", CAMPAIGN, "gvy25s at bv\\_comp 2.5: strong level compensation"),
    "gvy25sc32": ("GS21", CAMPAIGN, "gvy25s at bv\\_comp 3.2: strong level compensation"),
    "bvyhx2": ("BGN", CAMPAIGN, "hump-shaped loading across a mild level ladder 0.85..1.15: $b$ = -0.14, +0.00, +0.14, +0.00, -0.14"),
    "bvyhx3": ("BGN", CAMPAIGN, "hump-shaped loading across a mild level ladder 0.8..1.2: $b$ = -0.21, +0.00, +0.21, +0.00, -0.21"),
    "bvyhx4": ("BGN", CAMPAIGN, "hump-shaped loading across a mild level ladder 0.8..1.2: $b$ = -0.28, +0.00, +0.28, +0.00, -0.28"),
    "bvyq2x": ("BGN", CAMPAIGN, "two chains, the second loading quadratic in the first: $b_2 = -7\\,(b_1^2 - \\bar{b_1^2})$, $b_1 = \\pm$0.02, $\\pm$0.07, $\\pm$0.14"),
    "bvyq2k": ("BGN", CAMPAIGN, "two chains, the second loading quadratic in the first: $b_2 = -28\\,(b_1^2 - \\bar{b_1^2})$, $b_1 = \\pm$0.01, $\\pm$0.035, $\\pm$0.07"),
    "bvyq2": ("BGN", CAMPAIGN, "two chains, the second loading quadratic in the first: $b_2 = -14\\,(b_1^2 - \\bar{b_1^2})$, $b_1 = \\pm$0.01, $\\pm$0.035, $\\pm$0.07"),
    "vyg25su": ("KP14", CAMPAIGN, "vyg25s without level compensation (bv\\_comp 0): $\\pm$0.02, $\\pm$0.07, $\\pm$0.14"),
    "vyg25sux2": ("KP14", CAMPAIGN, "vyg25sx2 without level compensation (bv\\_comp 0): $\\pm$0.04, $\\pm$0.14, $\\pm$0.28"),
    "vyg25x3": ("KP14", CAMPAIGN, "vyg25 (positive ladder) with the loadings x3: 0.06, 0.21, 0.42"),
    "vyg25x2c": ("KP14", CAMPAIGN, "vyg25x2 (positive ladder x2) with the level compensation strengthened to 2.5"),
    "vyg35x2": ("KP14", CAMPAIGN, "vyg25x2 (positive ladder x2) with the $y$ price raised to 3.5"),
    "vyg25x2": ("KP14", CAMPAIGN, "vyg25 (positive ladder) with the loadings x2: 0.04, 0.14, 0.28"),
    "bvychsx2": ("BGN", CAMPAIGN, "bvychs with the loadings x 2: $\\pm$0.02, $\\pm$0.07, $\\pm$0.14"),
    "bvychsx3": ("BGN", CAMPAIGN, "bvychs with the loadings x 3: $\\pm$0.03, $\\pm$0.105, $\\pm$0.21"),
    "gvy25sx2": ("GS21", CAMPAIGN, "gvy25s with the loadings x 2: $\\pm$0.03, $\\pm$0.1, $\\pm$0.2"),
    "gvy25sx3": ("GS21", CAMPAIGN, "gvy25s with the loadings x 3: $\\pm$0.045, $\\pm$0.15, $\\pm$0.3"),
    "kpsgn": ("KP14", CAMPAIGN, "the price of x switches sign with y: $\\gamma_x (2\\,\\mathrm{logistic}(-2y) - 1)$ (conditioning room)"),
    "bgsgn": ("BGN", CAMPAIGN, "the regime price of the market shock at $-\\sigma_z$ / $+\\sigma_z$, 12-month spells (conditioning room)"),
    "vym6": ("KP14", CAMPAIGN, "six priced states, 24 rank-6 types (vym3's construction at K = 6)"),
    # the signed ladder over K states (2026-10-27): vyg25s's hidden mean-zero loading spread over K priced states
    "vym3s": ("KP14", CAMPAIGN, "vyg25s's signed ladder over three priced states: 24 mean-zero types, mirrored alignments"),
    "vym6s": ("KP14", CAMPAIGN, "vyg25s's signed ladder over six priced states: 24 mean-zero types, mirrored alignments, orthogonal off-$\\gamma$ directions"),
    "gvypk3": ("GS21", CAMPAIGN, "two priced chains, three product types"),
    # the 2026-10-17 fixes
    "gvyz10": ("GS21", CAMPAIGN, "one type, output $\\times e^{0.10\\,\\tilde z\\,y}$: the product economy without the coupon"),
    "gvyz10j": ("GS21", CAMPAIGN, "crash-only factor on $0.10\\,\\tilde z\\,y$: y unpriced except the 1\\%/month jump to $-3.5$ (weight $e^2$)"),
    "bq25": ("BGN", CAMPAIGN, "$\\sigma_z \\times \\min(1 + 0.25\\,s^2, 3)$, s the standardized rate, 7 bins"),
    "bvych41": ("BGN", CAMPAIGN, "bvych0 on a 41-node chain"),
}

GROUPS = [
    # ---- the main tables: one representative per model per group, the same construction across models ----
    ("g0", "Baselines", "The three models as published, one type each: the floor every group is read against.",
     ["bgnbase", "gsbase", "kpbase"]),
    ("g1", "Group 1. A different distribution of shocks: a crash-only factor",
     "Every firm's cash flows load on a factor y whose only priced shock is a rare crash (1\\% a month to $-3.5$ sd, kernel weight $e^2$).",
     ["bvy1j", "gvyz10j", "vy1j"]),
    ("g2", "Group 2. A quadratic price of the model's own risk",
     "The price of the baseline's aggregate risk becomes $1 + 0.25 s^2$ times its value, capped at 3, $s$ a standardized aggregate state.",
     ["bq25", "gq25", "kpq"]),
    ("g3a", "Group 3a. Exposure captured by a single characteristic",
     "A priced state y with one type: a firm's exposure varies only through one firm state (project mix, operating leverage, growth-option share).",
     ["bvy1ch", "gvyz10", "vy1"]),
    ("g3b", "Group 3b. Exposure as an interaction of characteristics",
     "The same y with three types (BGN, KP14) or a product written into production (GS21): exposure = type or productivity $\\times$ how levered the claim is.",
     ["bvych0", "gvyp10", "vyg25"]),
    ("g4", "Group 4. Larger factor models: two priced states",
     "Two independent priced states at total price 2.5 ($2.5/\\sqrt2$ each). BGN and GS21: three types with loading vectors (1,0), (0,1), (1,1)/$\\sqrt2$. KP14: nine types, an independent draw from the 3b ladder for each state (the three-direction version and the full-price version are in the appendix).",
     ["bvyk2", "gvypk3", "vy9t2"]),
    # ---- the appendix: everything else that was run, grouped by what it tested ----
    ("xa", "Additional economies", "Everything else that was run, by model, each described in the appendix against its parent; the RF rows are the reduced-form ladders with bounded (tanh), curvature (square) and size-orthogonal curvature (kfo) loadings, and the sign ladder (lsh: the K=1 latent product with the factors' means shifted; lrt: a one-signed ratio).",
     ["g0235f", "g0235df", "bvy25", "bvych41", "bvyc0", "bvyj25", "bvysg", "g28", "bx7", "gx7", "gvy1", "gvy1j", "gvy25", "gvys25", "gvyp05", "gvyp075", "gvyp15", "gvyp20",
      "gvyp10x", "gvyp10j", "gvypk2", "gvyzl", "gvyzlf", "gvyzlw", "gvyzl5", "gvyzlb", "gvyzlr", "gvyzlp", "gvyzlrp", "gsbg1", "gvyz10g", "gvyzlg", "gvyp10c", "gvytl", "gvytlg", "bvy9t2", "bvychs", "vyg25s", "vyg25u", "gvy25s", "kpsgn", "bgsgn", "vym6", "vym3s", "vym6s", "gvy2s", "gvytls", "bvychs4", "bvychs6", "bvychs8", "vyg25s6", "vyg25s18", "gvyib", "gvyib5", "gvyib2", "gxq35", "gxq70", "gvy25sq", "gvy25sh", "bvychs18", "bvychs25", "bvychs30", "bvychs35", "bvychs40", "bvychs45", "bvychsn5", "bvyvee", "bvyhump", "gvyql10", "gvyql20", "gvyqh10", "vyg25sx2", "vyg25sx3", "gvyzz10", "gvyzz20", "gvyzzh10", "gvy25sc25", "gvy25sc32", "bvyhx2", "bvyhx3", "bvyhx4", "bvyq2", "bvyq2x", "bvyq2k", "vyg25su", "vyg25sux2", "vyg25x2", "vyg25x3", "vyg35x2", "vyg25x2c", "bvychsx2", "bvychsx3", "gvy25sx2", "gvy25sx3", "kpgam", "vyx", "vyj25", "vym2t3", "vy9t2f", "vym3t3", "vym3"]),
]


def estimator_dir(tag):
    """where the tag's estimator summaries are read from, and whether those are the five-characteristic GS21 scores"""
    model, d, _ = E[tag]
    if model == "GS21" and LEV and glob.glob(os.path.join(LEV, f"*_estimators_{tag}_s[0-9][0-9][0-9]_w360_summary.csv")):
        return LEV, False
    if model == "GS21":
        # GS21 economies run after the leverage characteristic was added (2026-10-26) are scored on six characteristics in
        # the campaign directory itself: their summaries carry linlev rows. Only summaries without them are the Sol-era five.
        for f in glob.glob(os.path.join(d, f"*_estimators_{tag}_s[0-9][0-9][0-9]_w360_summary.csv")):
            with open(f) as fh:
                return d, ("linlev" not in fh.read())
        return d, True
    return d, False


def seed_rows(tag):
    model, d, _ = E[tag]
    est_dir, five_char = estimator_dir(tag)
    rows = []
    for f in sorted(glob.glob(os.path.join(d, f"*_oracle_{tag}_s[0-9][0-9][0-9].json"))):
        o = json.load(open(f)); b = o["bases"]
        seed = int(re.search(r"_s(\d{3})\.json$", f).group(1))
        have_eval = "const_best_eval" in b["lin_rank"] and "sr_max_eval" in o
        key = "const_best_eval" if have_eval else "const_best"
        lin = b["lin_rank"][key]; nl = max(b[k][key] for k in _nonlin_keys(b))
        srm = o["sr_max_eval"] if have_eval else o["sr_max_mean"]
        mkt = o["sr_ew_eval"] / o["sr_max_eval"] if "sr_ew_eval" in o else np.nan
        r = dict(tag=tag, seed=seed, eval=have_eval, sr_max=srm, lin_ceiling=lin, nonlin_ceiling=nl, room=nl - lin,
                 room_pct=100 * (nl - lin) / lin if lin > 1e-6 else np.nan, mkt_share=mkt,
                 # the interaction's Sharpe ORTHOGONAL to the best linear portfolio: Sharpes add in quadrature, so the
                 # difference nl - lin understates a nonlinearity that sits on a large linear base (GS21's market)
                 orth=float(np.sqrt(max(nl ** 2 - lin ** 2, 0.0))), five_char=five_char)
        est = glob.glob(os.path.join(est_dir, f"*_estimators_{tag}_s{seed:03d}_w360_summary.csv"))
        if est:
            s = pd.read_csv(est[0]); best = s.loc[s.groupby("method").sharpe.idxmax()].set_index("method").sharpe
            r.update(dkkm=best[[k for k in RFF if k in best.index]].max(), fmr=best.get("fm", np.nan), ff=best.get("ff", np.nan),
                     linear=best[[k for k in LINEAR if k in best.index]].max(), capm=best.get("mkt_est", np.nan))
            r["fair_gap"] = r["dkkm"] - best[[k for k in FAIR if k in best.index]].max()
            _lb = best[[k for k in FAIR if k in best.index]].max()
            r["dkkm_orth"] = float(np.sign(r["dkkm"] - _lb) * np.sqrt(abs(r["dkkm"] ** 2 - _lb ** 2)))   # DKKM's Sharpe orthogonal to the best classical
        rows.append(r)
    return pd.DataFrame(rows)


def economy(tag):
    s = seed_rows(tag)
    if s.empty:
        return None
    e = s[s.get("dkkm", pd.Series(dtype=float)).notna()] if "dkkm" in s else s.iloc[0:0]
    out = dict(tag=tag, model=E[tag][0], what=E[tag][2], n_oracle=len(s), n=len(e), eval=bool(s["eval"].all()), five_char=bool(s["five_char"].any()),
               sr_max=s.sr_max.mean(), lin_ceiling=s.lin_ceiling.mean(), room=s.room.mean(), room_pct=s.room_pct.mean(), mkt_share=s.mkt_share.mean(),
               orth=s.orth.mean())
    for c in ("dkkm", "fmr", "ff", "linear", "capm", "fair_gap", "dkkm_orth"):
        out[c] = e[c].mean() if len(e) else np.nan
    out["fair_gap_sd"] = e.fair_gap.std(ddof=1) if len(e) > 1 else np.nan
    out["fair_gap_t"] = (e.fair_gap.mean() / (e.fair_gap.std(ddof=1) / np.sqrt(len(e)))) if len(e) > 1 and e.fair_gap.std(ddof=1) > 0 else np.nan
    out["pos"] = int((e.fair_gap > 0).sum()) if len(e) else 0
    return out


def fmt(v, p=3, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:+.{p}f}" if sign else f"{v:.{p}f}"


HEAD = ["model", "economy", "SR$_{\\max}$", "market share", "linear ceiling", "room", "room \\% of linear", "interaction SR",
        "DKKM", "FMR", "FF", "linear", "CAPM", "fair gap", "DKKM $\\perp$", "+"]
HEAD_MD = ["model", "economy", "SR_max", "market %", "linear ceiling", "room", "room % lin", "interaction SR",
           "DKKM", "FMR", "FF", "linear", "CAPM", "fair gap", "DKKM ⊥", "+"]


def row_cells(r, md=False):
    dag = "" if r["eval"] else ("†" if md else "$^\\dagger$")
    dag += "" if not r.get("five_char") else ("‡" if md else "$^\\ddagger$")
    what = r["what"] if not md else re.sub(r"\$|\\%|\\times|\\tilde |\\max|\\gamma|\\|_\{|\}|\{", lambda m: {"$": "", "\\%": "%", "\\times": "x", "\\tilde ": "", "\\max": "max", "\\gamma": "gamma", "\\,": " ", "\\": "", "_{": "_", "}": "", "{": ""}[m.group(0)], r["what"].replace("\\,", " "))
    name = f"`{r['tag']}`" if md else "\\texttt{" + r['tag'].replace("_", "\\_") + "}"
    return [r["model"], f"{name}{dag} -- {what}", fmt(r["sr_max"]), fmt(100 * r["mkt_share"], 0) + ("%" if md else "\\%"),
            fmt(r["lin_ceiling"], 3),
            fmt(r["room"], 3, True), fmt(r["room_pct"], 0), fmt(r["orth"], 2), fmt(r["dkkm"]), fmt(r["fmr"]), fmt(r["ff"]), fmt(r["linear"]),
            fmt(r["capm"]), fmt(r["fair_gap"], 4, True) + (f" (t {r['fair_gap_t']:.1f})" if not np.isnan(r["fair_gap_t"]) else ""),
            fmt(r["dkkm_orth"], 2, True), f"{r['pos']}/{r['n']}" if r["n"] else "--"]


def reduced_rows():
    sys.path.insert(0, os.path.join(HERE, "reduced"))
    import rf_table                                                   # noqa: E402
    df = rf_table.load(SCREEN)
    # the FIXED-TOTAL-PRICE ladders (kfh_ / lfh_: lam ~ 1/sqrt(K), SR_max held); K = 1 is the same economy under both normalisations
    pick = [("kfh_lin_1", "K=1 factor on observed characteristics, linear loadings"), ("kfh_lin_4", "K=4"), ("kfh_lin_16", "K=16"),
            ("lfh_L5_K1_n0", "K=1 product of latent states, no observation noise"), ("lfh_L5_K2_n0", "K=2"), ("lfh_L5_K3_n0", "K=3"), ("lfh_L5_K5_n0", "K=5"), ("lfh_L5_K10_n0", "K=10"),
            ("lfh_L5_K1_n1", "K=1 product of latent states, observation noise 1.0"), ("lfh_L5_K2_n1", "K=2"), ("lfh_L5_K3_n1", "K=3"), ("lfh_L5_K5_n1", "K=5"), ("lfh_L5_K10_n1", "K=10")]
    # the appendix: bounded and curvature loadings on observed characteristics, and the size-orthogonal curvature ladder
    pick_x = [("kfh_tanh_1", "K=1, bounded (tanh) loadings on observed characteristics"), ("kfh_tanh_4", "K=4"), ("kfh_tanh_16", "K=16"),
              ("kfh_sq_1", "K=1, pure curvature (squares) of observed characteristics"), ("kfh_sq_4", "K=4"), ("kfh_sq_16", "K=16"),
              ("kfo_sq_1", "K=1, squares, directions orthogonal to size"), ("kfo_sq_4", "K=4"), ("kfo_sq_16", "K=16"),
              # the sign ladder (2026-10-26): the K=1 latent product with the factors' means shifted, and a one-signed ratio
              ("lsh_a0_b0_n0", "K=1 product of two mean-zero latent states (sign ladder origin), no noise"), ("lsh_a0_b1_n0", "one factor's mean shifted 1 sd"),
              ("lsh_a0_b2_n0", "one factor shifted 2 sd"), ("lsh_a1_b1_n0", "both factors shifted 1 sd"), ("lsh_a2_b2_n0", "both 2 sd"), ("lsh_a3_b3_n0", "both 3 sd"),
              ("lsh_a0_b0_n1", "mean-zero product, observation noise 1.0"), ("lsh_a2_b2_n1", "both shifted 2 sd, noise 1.0"),
              ("lrt_n0", "one-signed coverage ratio of the two latents, no noise")]
    return [_rf_rows(df, pick), _rf_rows(df, pick_x)]


def _rf_rows(df, pick):
    out = []
    for name, what in pick:
        sub = df[df.name == name]
        if sub.empty:
            continue
        m = sub.mean(numeric_only=True)
        out.append(dict(name=name, what=what, n=len(sub), sr_max=m["sr_max_eval"], mkt=m["mkt_share"], lin=m["lin_ceiling_eval"],
                        room=m["room_eval"], room_pct=100 * m["room_eval"] / m["lin_ceiling_eval"], dkkm=m["dkkm"], fmr=m["fm"], ff=m["ff"],
                        linear=max(m["linrank"], m["linrank_m"]), capm=m["mkt_est"], gap=m["fair_gap"], gap_sd=sub["fair_gap"].std(ddof=1),
                        pos=int((sub["fair_gap"] > 0).sum())))
    return out


def inject(path, key, block):
    """replace the text between <!-- TABLE:key --> and <!-- /TABLE:key --> in a markdown file (left alone if absent)"""
    if not os.path.exists(path):
        return False
    t = open(path).read(); m1, m2 = f"<!-- TABLE:{key} -->", f"<!-- /TABLE:{key} -->"
    if m1 not in t or m2 not in t:
        return False
    i, j = t.index(m1) + len(m1), t.index(m2)
    t = t[:i] + "\n" + block.strip("\n") + "\n" + t[j:]
    tmp = path + ".tmp"; open(tmp, "w").write(t); os.replace(tmp, path); return True


NOTE_MD = ("Columns: SR_max, the linear ceiling and the room are the population "
           "investor's conditional Sharpe over the evaluation window (room = best nonlinear ceiling minus the linear ceiling, "
           "and room % lin = room as a percentage of the linear ceiling); DKKM, FMR, FF, linear (the best of rank and level "
           "regressions, with and without the market) and CAPM (the estimated market line) are out-of-sample Sharpe ratios at "
           "the 360-month window; fair gap = DKKM minus the best of (FMR, FF, linear, CAPM), mean across seeds with its t-statistic (mean over the standard error across seeds); "
           "interaction SR = the nonlinear ceiling's Sharpe orthogonal to the linear ceiling, sqrt(nl^2 - lin^2), the size of the interaction itself (Sharpes add in quadrature, so the room understates an interaction that sits on a large linear base); "
           "DKKM perp = DKKM's Sharpe orthogonal to the best classical method, with the sign of the gap; "
           "+ = seeds with a positive fair gap. A dagger marks an oracle that predates the evaluation-window ceilings. "
           "GS21 estimators are scored on the oracle's six characteristics (size, book-to-market, asset growth, profitability, momentum, leverage); "
           "a double dagger marks a GS21 economy still scored on the first five, without leverage (the Sol-era scoring), whose DKKM, FMR, FF, linear and fair gap are not comparable.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "docs", "campaign_tables"), help="directory for the per-group .md/.tex files")
    ap.add_argument("--report", default=os.path.join(os.path.dirname(HERE), "docs", "CAMPAIGN_REPORT_2026-10-11.md"), help="markdown report whose TABLE markers are filled")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    econ = {t: economy(t) for t in E}
    allmd = [NOTE_MD, ""]
    rr_main, rr_x = reduced_rows()
    def rf_as_econ(r):
        """a reduced-form row in the economy-row format, so the moved ladders sit in the appendix table"""
        nl = r["lin"] + r["room"]; best = max(r["fmr"], r["ff"], r["linear"], r["capm"])
        return dict(tag=r["name"], model="RF", what=r["what"], eval=True, n=r["n"], sr_max=r["sr_max"], mkt_share=r["mkt"], lin_ceiling=r["lin"],
                    room=r["room"], room_pct=r["room_pct"], orth=float(np.sqrt(max(nl ** 2 - r["lin"] ** 2, 0.0))), dkkm=r["dkkm"], fmr=r["fmr"], ff=r["ff"],
                    linear=r["linear"], capm=r["capm"], fair_gap=r["gap"], fair_gap_t=r["gap"] / (r["gap_sd"] / np.sqrt(r["n"])) if r["gap_sd"] > 0 else np.nan,
                    dkkm_orth=float(np.sign(r["dkkm"] - best) * np.sqrt(abs(r["dkkm"] ** 2 - best ** 2))), pos=r["pos"])
    for key, title, blurb, tags in GROUPS:
        rows = [econ[t] for t in tags if econ.get(t)]
        if key == "xa":
            rows += [rf_as_econ(r) for r in rr_x]        # the bounded, curvature and size-orthogonal reduced-form ladders
        md = ["| " + " | ".join(HEAD_MD) + " |", "|" + "---|" * len(HEAD_MD)]
        md += ["| " + " | ".join(row_cells(r, md=True)) + " |" for r in rows]
        tex = ["\\begin{table}[htbp]\\centering\\footnotesize", f"\\caption{{{title}. {blurb}}}\\label{{tab:{key}}}",
               "\\resizebox{\\textwidth}{!}{\\begin{tabular}{ll" + "r" * (len(HEAD) - 2) + "}\\hline",
               " & ".join(HEAD) + " \\\\\\hline"]
        tex += [" & ".join(row_cells(r)) + " \\\\" for r in rows]
        tex += ["\\hline\\end{tabular}}\\end{table}"]
        open(os.path.join(a.out, key + ".md"), "w").write("\n".join(md) + "\n")
        open(os.path.join(a.out, key + ".tex"), "w").write("\n".join(tex) + "\n")
        allmd += [f"### {title}", "", blurb, ""] + md + [""]
        print(key, "injected" if inject(a.report, key, "\n".join(md)) else "no marker in the report")
    # the reduced form's main table (linear loadings and latent products); the other ladders are rows of the appendix table
    for key, rr, cap, title in [
        ("rf", rr_main, "Reduced form: priced factors loading linearly on observed characteristics against products of latent states. Five seeds per cell, the total price of risk fixed across K (per-factor price $\\propto 1/\\sqrt{K}$); K = number of priced factors; the lfh rows load products of pairs of the five latent processes, observed with noise of sd 0 (n0) or 1 (n1).", "Reduced form")]:
        h = ["configuration", "SR_max", "market %", "linear ceiling", "room", "room % lin", "interaction SR", "DKKM", "FMR", "FF", "linear", "CAPM", "fair gap", "+"]
        md = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
        tex = ["\\begin{table}[htbp]\\centering\\footnotesize", f"\\caption{{{cap}}}\\label{{tab:{key}}}",
               "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lr" + "r" * (len(h) - 2) + "}\\hline",
               " & ".join(["configuration", "SR$_{\\max}$", "market share", "linear ceiling", "room", "room \\% of linear", "interaction SR", "DKKM", "FMR", "FF", "linear", "CAPM", "fair gap", "+"]) + " \\\\\\hline"]
        for r in rr:
            cells = [f"{r['name']} -- {r['what']}", fmt(r["sr_max"]), fmt(100 * r["mkt"], 0), fmt(r["lin"]), fmt(r["room"], 3, True),
                     fmt(r["room_pct"], 0), fmt(float(np.sqrt(max((r["lin"] + r["room"]) ** 2 - r["lin"] ** 2, 0.0))), 2), fmt(r["dkkm"]), fmt(r["fmr"]), fmt(r["ff"]), fmt(r["linear"]), fmt(r["capm"]),
                     fmt(r["gap"], 4, True) + f" (t {r['gap'] / (r['gap_sd'] / np.sqrt(r['n'])):.1f})", f"{r['pos']}/{r['n']}"]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(c.replace("_", "\\_") if i == 0 else c for i, c in enumerate(cells)) + " \\\\")
        tex += ["\\hline\\end{tabular}}\\end{table}"]
        open(os.path.join(a.out, key + ".md"), "w").write("\n".join(md) + "\n"); open(os.path.join(a.out, key + ".tex"), "w").write("\n".join(tex) + "\n")
        allmd += [f"### {title}", ""] + md + [""]
        print(key, "injected" if inject(a.report, key, "\n".join(md)) else "no marker in the report")
    open(os.path.join(a.out, "ALL.md"), "w").write("\n".join(allmd) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
