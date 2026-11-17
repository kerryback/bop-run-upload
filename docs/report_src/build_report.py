#!/usr/bin/env python
"""Build CAMPAIGN_REPORT_2026-10-11.{md,tex} from report_src/main.md.

main.md is the single prose source (inline math in $...$).  The appendix item lists (one bullet per economy) are
taken from report_before_cleanup.md (markdown) and appendix_items.tex (LaTeX), with the same plain-language
substitutions applied to both.  Tables are filled afterwards by variants/campaign_tables.py (markdown) and
\\input (LaTeX).  Run from anywhere; then run campaign_tables.py and pdflatex twice.
"""
import re, os
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(D)
MAIN = open(os.path.join(D, "main.md")).read()
OLD = open(os.path.join(D, "report_before_cleanup.md")).read()
ITEMS_TEX = open(os.path.join(D, "appendix_items.tex")).read()
PRE = open(os.path.join(D, "preamble.tex")).read()

# ---------------------------------------------------------------- plain-language substitutions for the item lists
SUB_MD = [
    (" bgn_vy's SDF step had its K4 pair sum rewritten as a blocked sum (commit 78961c4) after these ran out of memory at 36 GB; the arithmetic is unchanged to 3e-15.", ""),
    ("Solved (81-point J*) but not simulated", "Solved but not simulated"),
    ("(type 0, the small exposed type, is the compensation's reference; J* = 0)", "(the small exposed type is the compensation's reference and its value solves to zero)"),
    ("the oracle runs out of memory (section 4)", "the true-moments step runs out of memory"),
    ("stopped at the oracle stage (section 4)", "stopped at the true-moments stage"),
    ("stopped at the oracle stage", "stopped at the true-moments stage"),
    ("stopped at the oracle", "stopped at the true-moments stage"),
    ("oracle seeds", "true-moments seeds"), ("estimator seeds", "method seeds"),
    ("Stage j's solve is cycle-averaged (section 4).", "The solve is averaged over its cycle."),
    ("two solves cycle-capped", "two solves did not converge"), ("oscillated to the sweep cap", "did not converge"),
    ("cycles at the sweep cap", "does not converge"),
    ("a Markov chain", "a two-state process"), ("Markov regime", "two-state regime"),
    ("-node chain", "-point grid"), ("node chain", "point grid"), ("both chains", "both grids"), ("SECOND priced chain", "second priced grid"),
    ("second independent chain", "second independent grid"), ("its 9-node chain", "its 9-point grid"), ("the Tauchen transition", "the ordinary grid transition"),
    ("the chain,", "the grid,"), ("chain, price", "grid, price"), ("(9-node chain", "(9-point grid"), ("9-node chain)", "9-point grid)"),
    ("The kernel weights", "The pricing rule weights"), ("the kernel prices", "the pricing rule prices"),
    ("risk-neutral probability", "probability under the pricing rule"), ("under P and", "under the real probabilities and"), ("under Q", "under the pricing probabilities"),
    ("three independent OU states", "three independent mean-reverting states"), ("two independent OU states", "two independent mean-reverting states"), ("K independent OU states", "K independent mean-reverting states"),
    ("bv_comp", "compensation exponent"), ("level-compensated", "compensated"),
    ("(NY 9, y_mult 2.5, price 2.5)", "(9-point grid over ±2.5 sd, price 2.5)"), ("(NY 9, y_mult 2.5)", "(9-point grid over ±2.5 sd)"),
    ("bv_comp 1.0, NY 9, y_mult 2.5, price 2.5", "compensation exponent 1.0, 9-point grid, price 2.5"),
    ("(NY2 21, both grids priced", "(21 points, both grids priced"), ("xi 0, y_mult 3.5", "grid over ±3.5 sd"),
    ("linear ceiling", "best linear"), ("linear ceilings", "best linear"), ("fair gap", "gap"), ("SR_max", "best Sharpe"), ("FMR", "FM"),
    ("`gs_qcomp` 1: production divided by the priced value of the firm's own exposure, so value does not reveal leverage", "production divided by the priced value of the firm's own exposure, so value does not reveal leverage"),
    ("(gs_qcomp 1)", "(the same per-firm compensation)"), ("(`gs_bzq`)", ""), ("(`gs_bzz`, new; gs_bzl 0, no level term)", "(no level term)"),
    ("(gs_bzl 0, gs_bzi 1)", ""), ("(`gs_bv_cap`)", ""), ("(`gs_fixcs` 2)", ""), ("production-level exposure gs_bv (0.3, 0, 0.3)", "loadings 0.3, 0, 0.3"),
    ("twelve mean-zero types (gs_bv, gs_bv2)", "twelve mean-zero types"), ("DKKM sees both standardized states", "DKKM sees both states"),
    ("through a new `type_scale` parameter in bgn_vy (a per-type multiplier on the firm's investment unit I that scales book, cash flows and value together, so only size moves and book-to-market, profitability and the return moments are unchanged to 1e-12; 1.0 leaves every earlier economy byte-identical)", "through a per-type firm scale (a multiplier on the project cost that scales book, cash flows and value together, so only size moves)"),
    ("through the new `gs_level` (an explicit per-type production-level multiplier, log gs_level in the production exponent; 1.0 leaves every earlier economy unchanged)", "through a per-type production level"),
    ("(`type_level`)", ""), ("through `type_level`", "through the cash-flow level"), ("(new in kp_vy)", ""), ("`type_level` 0.6 / 1.0 / 1.6", "cash-flow levels 0.6 / 1.0 / 1.6"),
    ("production levels (gs_level)", "production levels"), ("two industries (gs_level 0.8 / 1.2)", "two industries (production levels 0.8 / 1.2)"),
    ("(`bv_on_eps`: cash flow x K^α θ u [1 + (ε − 1) e^{b_f y}])", "(cash flow x K^α θ u [1 + (ε − 1) e^{b_f y}])"),
    ("(`bv_eps_cap` c: h_b(y) = c / (1 + (c − 1) e^{−c b y/(c−1)}) in place of e^{b_f y})", "(h_b(y) = c / (1 + (c − 1) e^{−c b y/(c−1)}) in place of e^{b_f y}, c the bound)"),
    ("(gs_vy at one y node and no exposure, which is gsbase's solver)", ""), ("the solver prices type f at b_eff = m, γ_eff = c × 2.5/sqrt(K).", "each type is priced at its size m and at c times the state price."),
    ("The solver prices type f at b_eff = m, γ_eff = c × 2.5/sqrt(K).", "Each type is priced at its size m and at c times the state price."),
    ("(group 3b)", "(the 3b control)"), ("`bgn_gam` with the regime multiplier", "`g0235f`'s regime design with the multiplier"),
    ("Its room is conditioning room: DKKM sees y, the constant-weight benchmarks do not.", "Its room comes from conditioning: DKKM sees y and the constant-weight methods do not."),
    ("in the table; the other seven were stopped at the true-moments stage (section 4)", "in the table; the other seven were stopped at the true-moments stage"),
    ("Returns identical to vyg25s; only the characteristics change.", "Returns identical to `vyg25s`; only the characteristics change."),
    ("the two (magnitude, cosine) grids are nearly the same", "the two grids of sizes and cosines are nearly the same"),
    ("36 GB tasks", "tasks"), ("once the room was read", "once the room was known"),
    ("NY 9, y_mult 2.5, price 2.5", "9-point grid, price 2.5"), ("gs_bv_f z̃ ,", "b_f z̃,"),
]
SUB_TEX = [
    (" bgn\\_vy's SDF step had its K4 pair sum rewritten as a blocked sum (commit 78961c4) after these ran out of memory at 36 GB; the arithmetic is unchanged to $3\\times10^{-15}$.", ""),
    ("Solved (81-point $J^*$) but not simulated", "Solved but not simulated"),
    ("(type 0, the small exposed type, is the compensation's reference; $J^*=0$)", "(the small exposed type is the compensation's reference and its value solves to zero)"),
    ("the oracle runs out of memory (Section~4)", "the true-moments step runs out of memory"),
    ("stopped at the oracle stage (Section~4)", "stopped at the true-moments stage"),
    ("stopped at the oracle stage", "stopped at the true-moments stage"), ("stopped at the oracle", "stopped at the true-moments stage"),
    ("oracle seeds", "true-moments seeds"), ("estimator seeds", "method seeds"),
    ("Stage $j$'s solve is cycle-averaged (Section~4).", "The solve is averaged over its cycle."),
    ("two solves cycle-capped", "two solves did not converge"), ("oscillated to the sweep cap", "did not converge"), ("cycles at the sweep cap", "does not converge"),
    ("a Markov chain", "a two-state process"), ("Markov regime", "two-state regime"),
    ("-node chain", "-point grid"), ("node chain", "point grid"), ("both chains", "both grids"), ("SECOND priced chain", "second priced grid"),
    ("second independent chain", "second independent grid"), ("its 9-node chain", "its 9-point grid"), ("the Tauchen transition", "the ordinary grid transition"),
    ("the chain,", "the grid,"), ("chain, price", "grid, price"), ("(9-node chain", "(9-point grid"), ("9-node chain)", "9-point grid)"),
    ("The kernel weights", "The pricing rule weights"), ("the kernel prices", "the pricing rule prices"),
    ("risk-neutral probability", "probability under the pricing rule"), ("under $P$ and", "under the real probabilities and"), ("under $Q$", "under the pricing probabilities"),
    ("three independent OU states", "three independent mean-reverting states"), ("two independent OU states", "two independent mean-reverting states"), ("$K$ independent OU states", "$K$ independent mean-reverting states"),
    ("bv\\_comp", "compensation exponent"), ("level-compensated", "compensated"),
    ("(NY 9, y\\_mult 2.5, price 2.5)", "(9-point grid over $\\pm2.5$ sd, price 2.5)"), ("(NY 9, y\\_mult 2.5)", "(9-point grid over $\\pm2.5$ sd)"),
    ("compensation exponent 1.0, NY 9, y\\_mult 2.5, price 2.5", "compensation exponent 1.0, 9-point grid, price 2.5"),
    ("(NY2 21, both grids priced", "(21 points, both grids priced"), ("xi 0, y\\_mult 3.5", "grid over $\\pm3.5$ sd"),
    ("linear ceilings", "best linear"), ("linear ceiling", "best linear"), ("fair gap", "gap"), ("$\\mathrm{SR}_{\\max}$", "best Sharpe"), ("FMR", "FM"),
    ("gs\\_qcomp 1: production divided by the priced value of the firm's own exposure, so value does not reveal leverage", "production divided by the priced value of the firm's own exposure, so value does not reveal leverage"),
    ("(gs\\_qcomp 1)", "(the same per-firm compensation)"), ("(gs\\_bzq)", ""), ("(gs\\_bzz, new; gs\\_bzl 0, no level term)", "(no level term)"),
    ("(gs\\_bzl 0, gs\\_bzi 1)", ""), ("(gs\\_bv\\_cap)", ""), ("(gs\\_fixcs 2)", ""), ("production-level exposure gs\\_bv (0.3, 0, 0.3)", "loadings 0.3, 0, 0.3"),
    ("twelve mean-zero types (gs\\_bv, gs\\_bv2)", "twelve mean-zero types"), ("DKKM sees both standardized states", "DKKM sees both states"),
    ("(group 3b)", "(the 3b control)"),
    ("Its room is conditioning room: DKKM sees $y$, the constant-weight benchmarks do not.", "Its room comes from conditioning: DKKM sees $y$ and the constant-weight methods do not."),
    (", $J^* = 0$)", " and its value solves to zero)"), (" (\\texttt{gs\\_bzz}, new)", ""),
    ("through the new gs\\_level (a per-type production-level multiplier; 1.0 leaves every earlier economy unchanged)", "through a per-type production level"),
    ("production exposure gs\\_bv (0.3, 0, 0.3)", "loadings 0.3, 0, 0.3"), ("production levels (gs\\_level)", "production levels"), ("(gs\\_level 0.8 / 1.2)", "(production levels 0.8 / 1.2)"),
    ("times the elasticity of its value, which rises", "times the sensitivity of its value to the cash flow, which rises"),
    ("(\\texttt{bv\\_on\\_eps}, cash flow", "(cash flow"), ("with \\texttt{type\\_level} 0.6", "with cash-flow levels 0.6"),
    ("(\\texttt{bv\\_eps\\_cap} $c$: ", "("), ("in place of $e^{b_f y}$), six mirrored", "in place of $e^{b_f y}$, $c$ the bound), six mirrored"),
    ("through type\\_level:", "through the cash-flow level:"), ("through type\\_level,", "through the cash-flow level,"),
    ("standardized idiosyncratic productivity ,", "standardized idiosyncratic productivity,"),
    ("$b_f\\in\\{-0.07,-0.035,-0.01,0.01,0.035,0.07\\}$", "$b_f$ = $-0.07$, $-0.035$, $-0.01$, 0.01, 0.035, 0.07"),
    ("$b_f\\in\\{-0.10,-0.05,-0.015,0.015,0.05,0.10\\}$", "$b_f$ = $-0.10$, $-0.05$, $-0.015$, 0.015, 0.05, 0.10"),
    ("$b_f\\in\\{-0.14,-0.07,-0.02,0.02,0.07,0.14\\}$", "$b_f$ = $-0.14$, $-0.07$, $-0.02$, 0.02, 0.07, 0.14"),
    (" (\\texttt{gs\\_vy} at one $y$ node and no exposure, which is \\texttt{gsbase}'s solver)", ""),
    ("six mirrored types $\\pm0.05/0.1/0.15$ with production levels 0.5--2.0, exposure", "six mirrored types $\\pm0.05$ / 0.1 / 0.15 with production levels 0.5--2.0, exposure"),
    ("36 GB tasks", "tasks"), ("once the room was read", "once the room was known"),
]

def apply(s, subs):
    for a, b in subs:
        s = s.replace(a, b)
    return s

# ---------------------------------------------------------------- the markdown item lists (from the pre-cleanup report)
def md_items():
    lines = OLD.split("\n")
    def block(head):
        i = next(k for k, l in enumerate(lines) if l == head and k + 1 < len(lines) and lines[k + 1].startswith("- "))
        j = i + 1
        while j < len(lines) and lines[j].startswith("- "):
            j += 1
        return lines[i + 1:j]
    bgn, gs, kp = block("**BGN**"), block("**GS21**"), block("**KP14**")
    # KP14 economy items that had been appended to the "what they add" list
    i = lines.index("What they add to the main results:")
    extra = [l for l in lines[i + 1:] if re.match(r"- `(vyg25sx2|vyg25su|vyge|vygelb|vyhx3|vyuq|vyg25x2)`", l)]
    assert len(extra) == 7, len(extra)
    out = ["**BGN**", ""] + bgn + ["", "**GS21**", ""] + gs + ["", "**KP14**", ""] + kp + extra
    return apply("\n".join(out), SUB_MD)

# ---------------------------------------------------------------- markdown -> LaTeX for the main body
def esc(s):
    """escape LaTeX specials in text outside math and code"""
    return s.replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")

UNI = {"−": "$-$", "±": "$\\pm$", "×": "$\\times$", "≤": "$\\le$", "≥": "$\\ge$", "≈": "$\\approx$", "→": "$\\to$", "√": "$\\surd$"}

def inline(s, figlabel):
    """convert one line of markdown prose to LaTeX"""
    parts = re.split(r"(\$[^$]*\$|`[^`]*`)", s)
    out = []
    for p in parts:
        if p.startswith("$"):
            out.append(p)
        elif p.startswith("`"):
            out.append("\\texttt{" + p[1:-1].replace("_", "\\_") + "}")
        else:
            t = esc(p)
            for a, b in UNI.items():
                t = t.replace(a, b)
            t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
            t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", t)
            t = re.sub(r"\bsection 5\b", "Section~\\\\ref{sec:appendix}", t)
            t = re.sub(r"\bsections (\d)", r"Sections~\1", t)
            t = re.sub(r"\bsection (\d[abc]?)\b", r"Section~\1", t)
            t = t.replace("(figure)", "(Figure~\\ref{%s})" % figlabel) if figlabel else t
            out.append(t)
    return "".join(out)

def convert(md):
    lines = md.split("\n")
    tex = []
    i = 0
    figlabel = None
    para = []
    stack = []           # open itemize depth
    def flush():
        nonlocal para
        if para:
            tex.append(inline(" ".join(para), figlabel))
            tex.append("")
            para = []
    def close_lists(depth=0):
        while len(stack) > depth:
            tex.append("\\end{itemize}")
            stack.pop()
    while i < len(lines):
        l = lines[i]
        if i < 3:                                   # title and byline live in the preamble
            i += 1; continue
        if l.startswith("<!-- TABLE:"):
            flush(); close_lists()
            key = l[len("<!-- TABLE:"):-len(" -->")]
            tex.append("\\input{campaign_tables/%s.tex}" % key); tex.append("")
            while not lines[i].startswith("<!-- /TABLE:"):
                i += 1
            i += 1; continue
        if l.startswith("<!-- ITEMS -->"):
            flush(); close_lists()
            tex.append(apply(ITEMS_TEX, SUB_TEX).rstrip()); tex.append("")
            while not lines[i].startswith("<!-- /ITEMS -->"):
                i += 1
            i += 1; continue
        m = re.match(r"^(#{2,3}) (.*)$", l)
        if m:
            flush(); close_lists()
            title = re.sub(r"^\d[abc]?\. ", "", m.group(2))
            if m.group(1) == "##":
                lab = "\\label{sec:appendix}" if title.startswith("Additional") else ""
                tex.append("\\section{%s}%s" % (inline(title, None), lab))
            else:
                star = "" if re.match(r"^3[abc]\. ", m.group(2)) else "*"
                tex.append("\\subsection%s{%s}" % (star, inline(title, None)))
                figlabel = {"3a": "fig:w3a", "3b": "fig:w3b", "3c": "fig:w3c"}.get(m.group(2)[:2], figlabel)
            tex.append("")
            i += 1; continue
        m = re.match(r"^!\[(.*)\]\((.*)\)$", l)
        if m:
            flush(); close_lists()
            cap, path = m.group(1), m.group(2).replace(".png", ".pdf")
            lab = "fig:w" + re.search(r"fig_(\w+)\.", path).group(1)
            tex += ["\\begin{figure}[htbp]\\centering", "\\includegraphics[width=\\textwidth]{%s}" % path,
                    "\\caption{%s}\\label{%s}" % (inline(cap, None), lab), "\\end{figure}", ""]
            i += 1; continue
        m = re.match(r"^(\s*)- (.*)$", l)
        if m:
            flush()
            depth = len(m.group(1)) // 2 + 1
            while len(stack) < depth:
                tex.append("\\begin{itemize}"); stack.append(1)
            close_lists(depth)
            tex.append("\\item " + inline(m.group(2), figlabel))
            i += 1; continue
        if l.strip() == "":
            flush()
            # a blank line ends a list only if the next non-blank line is not a bullet
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if not (j < len(lines) and re.match(r"^\s*- ", lines[j])):
                close_lists()
            i += 1; continue
        if stack and l.startswith("  ") and not re.match(r"^\s*- ", l):   # continuation of an item
            tex[-1] += " " + inline(l.strip(), figlabel)
            i += 1; continue
        if l.startswith("*Figure: "):          # the markdown-only caption line
            i += 1; continue
        para.append(l)
        i += 1
    flush(); close_lists()
    return "\n".join(tex)

if __name__ == "__main__":
    items = md_items()
    md = MAIN.replace("<!-- ITEMS -->\n<!-- /ITEMS -->", "<!-- ITEMS -->\n" + items + "\n<!-- /ITEMS -->")
    # GitHub does not show alt text, so the markdown gets a visible caption line under each figure
    md_out = re.sub(r"^(!\[(.*)\]\(.*\))$", lambda m: m.group(1) + "\n\n*Figure: " + m.group(2) + "*", md, flags=re.M)
    open(os.path.join(OUT, "CAMPAIGN_REPORT_2026-10-11.md"), "w").write(md_out)
    body = convert(md)
    pre = PRE.replace("\\title{What produces a capturable complexity gap:\\\\one representative economy per model and group}",
                      "\\title{What produces a complexity gap:\\\\one economy per model and per change}")
    pre = re.sub(r"\\date\{.*\}", "\\\\date{October 11, 2026; final numbers November 12; plain-language edit November 16, 2026}", pre)
    open(os.path.join(OUT, "CAMPAIGN_REPORT_2026-10-11.tex"), "w").write(pre + "\n" + body + "\n\\end{document}\n")
    print("wrote md and tex;", md.count("\n- "), "md bullets;", body.count("\\item"), "tex items")
