"""Turn `BOP campaign/results/{metrics,H_pairs}.csv` into docs/campaign_tables/mlp.{md,tex}, copy the comparison
figures into docs/mlp_figures/, and fill the <!-- TABLE:mlp --> block of the markdown report."""
import os, shutil, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/kellogg/proj/axa3203/bop-run-upload/variants")
from campaign_tables import inject

RES = "/kellogg/proj/axa3203/external/MLP Analyze/BOP campaign/results"
DOCS = "/kellogg/proj/axa3203/bop-run-upload/docs"
os.makedirs(f"{DOCS}/campaign_tables", exist_ok=True); os.makedirs(f"{DOCS}/mlp_figures", exist_ok=True)
m = pd.read_csv(f"{RES}/metrics.csv")
ORDER = ["bgn_base", "bgn_3b", "gs_base", "gs_3b", "kp_base", "kp_3b"]
TAG = {"bgn_base": "bgnbase", "bgn_3b": "bvych0", "gs_base": "gsbase", "gs_3b": "gvyp10", "kp_base": "kpbase", "kp_3b": "vyg25"}
LBL = {"true": "true SDF", "fm": "FMR", "ff": "FF", "dkkm": "DKKM"}

def f(v, d=3, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"

md = ["| model | economy | weights | corr with truth | corr, MLP fit | nonlinear share | corr of nonlinear parts (t) | interaction sd at pair |",
      "|---|---|---|---|---|---|---|---|"]
tex = [r"\begin{table}[htbp]\centering\small", r"\begin{tabular}{llrrrrrr}", r"\toprule",
       r"model & economy & weights & \shortstack{corr with\\truth} & \shortstack{corr,\\MLP fit} & \shortstack{nonlinear\\share} & \shortstack{corr of non-\\linear parts ($t$)} & \shortstack{interaction\\sd at pair} \\", r"\midrule"]
for e in ORDER:
    s = m[m.econ == e]
    if s.empty: continue
    fam = s.family.iloc[0]; pair = s.H_pair.iloc[0].replace("x", " $\\times$ "); pairmd = s.H_pair.iloc[0].replace("x", " x ")
    for i, meth in enumerate(["true", "fm", "ff", "dkkm"]):
        r = s[s.method == meth].iloc[0]
        econ_md = f"`{TAG[e]}` ({s.role.iloc[0]}; pair {pairmd})" if i == 0 else ""
        econ_tx = f"\\texttt{{{TAG[e]}}} ({s.role.iloc[0]}; {pair})" if i == 0 else ""
        nl = "" if meth == "true" else f"{f(r.corr_nl_vs_truth_nl, 2, True)} ({r.t_nl:.1f})"
        md.append(f"| {fam if i == 0 else ''} | {econ_md} | {LBL[meth]} | {f(r.corr_actual_vs_truth, 2)} | {f(r.corr_fitted_vs_truth, 2)} | {f(r.nl_share_fitted, 2)} | {nl} | {f(r.int_sd_pair, 3)} |")
        tex.append(f"{fam if i == 0 else ''} & {econ_tx} & {LBL[meth]} & {f(r.corr_actual_vs_truth, 2)} & {f(r.corr_fitted_vs_truth, 2)} & {f(r.nl_share_fitted, 2)} & {nl} & {f(r.int_sd_pair, 3)} \\\\")
    tex.append(r"\midrule")
tex[-1] = r"\bottomrule"
note = ("Columns: corr with truth = mean over (seed, month) cross-sections of the correlation between the method's weights and the "
        "true SDF weights; corr, MLP fit = the same for the MLP surface fitted to the method's weights; nonlinear share = the part of "
        "the fitted surface's variance that an OLS fit on the five ranks within each cross-section leaves (one minus the within-cross-section R-squared); "
        "corr of nonlinear parts = cross-sectional correlation of that nonlinear remainder with the truth surface's nonlinear remainder, "
        "with its t-statistic across seeds; interaction sd at pair = standard deviation, in weight units, of the two-way interaction part of the "
        "fitted surface over the named pair (the surface minus its additive fit in the two characteristics), so the truth's row says how large "
        "the interaction is and a method's row how much of it the method's weights carry. The pair is the one the economy's exposure is written in "
        "(KP14 and BGN: type, proxied by size, times leverage, bm; GS21: productivity, roe, times leverage, bm) and is the pair the figures use. Seeds: three per benchmark, "
        "ten per 3b economy; evaluation months only; DKKM at P=3600 and the economy's best kappa; weights winsorized at 1/99 within cross-sections.")
md.append(""); md.append(note)
# figures: the family comparison (benchmark row over 3b row) once both exist, else the 3b economy's own 1x4 panel
FIGS = {"kp": ("bm", "size"), "gs": ("roe", "bm"), "bgn": ("bm", "size")}   # (vary, fixed) shown in the report
PAIRS = {"kp": "sizexbm", "gs": "bmxroe", "bgn": "sizexbm"}
NAME = {"kp": "KP14", "gs": "GS21", "bgn": "BGN"}
LONG = {"bm": "book-to-market", "size": "size", "roe": "profitability", "mom": "momentum"}
figs_md, figs_tex = [], []
for fam, (v, x) in FIGS.items():
    for fn, cap in [(f"interaction_{fam}_{PAIRS[fam]}.png",
                     f"{NAME[fam]}: the interaction part of the fitted weight over {LONG[v]} and {LONG[x]} (benchmark above, 3b below); "
                     "panels: true SDF, Fama-MacBeth, Fama-French, DKKM, each on its own colour scale, the interaction's sd in weight units in the title."),
                    (f"compare_{fam}_vary_{v}_fix_{x}.png",
                     f"{NAME[fam]}: fitted weight against {LONG[v]} at five levels of {LONG[x]} (benchmark above, 3b below); "
                     "panels: true SDF, Fama-MacBeth, Fama-French, DKKM; other characteristics at their mean.")]:
        if not os.path.exists(f"{RES}/{fn}"): continue
        shutil.copy(f"{RES}/{fn}", f"{DOCS}/mlp_figures/{fn}")
        figs_md.append(f"![{cap}](mlp_figures/{fn})\n\n*{cap}*")
        figs_tex.append("\\begin{figure}[htbp]\\centering\\includegraphics[width=\\textwidth]{mlp_figures/" + fn + "}\\caption{" + cap.replace("Fama-", "Fama--") + "}\\end{figure}")
md += [""] + [f + "\n" for f in figs_md]
tex += [r"\end{tabular}", r"\caption{Neural-network fits of the portfolio weights: benchmark and group-3b economies.}\label{tab:mlp}",
        r"\begin{minipage}{\textwidth}\footnotesize " + note.replace("R-squared", "$R^2$").replace("P=3600", "$P=3600$").replace("kappa", "$\\kappa$") + r"\end{minipage}", r"\end{table}"] + figs_tex
open(f"{DOCS}/campaign_tables/mlp.md", "w").write("\n".join(md) + "\n"); open(f"{DOCS}/campaign_tables/mlp.tex", "w").write("\n".join(tex) + "\n")
for fn in sorted(os.listdir(RES)):
    if fn.startswith("compare_") and fn.endswith(".png"): shutil.copy(f"{RES}/{fn}", f"{DOCS}/mlp_figures/{fn}")
print("md injected:", inject(f"{DOCS}/CAMPAIGN_REPORT_2026-10-11.md", "mlp", "\n".join(md)))
print("\n".join(md))
