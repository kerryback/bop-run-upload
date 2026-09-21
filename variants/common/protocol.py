"""The measurement protocol: the numerical choices every economy shares.

WHY THIS FILE EXISTS. Results in this repository differ along two axes, and only one of
them is interesting. An economy differs from another economy in its PARAMETERS and its
DRIVING FORCES -- a price of risk, a regime, a priced factor, a set of exposure types.
It must not differ in how many firms were simulated, how many months were burned in, how
many months were evaluated, or how finely anything was discretised. When those drift, a
table's rows are no longer comparable and no amount of prose fixes it: before 2026-09-15
one KP14 row carried a quarter of its headline gap from an extra ridge-penalty decade that
its siblings never saw (the vyxT860 row, since retired).

So: the numbers below are the protocol. Every current economy is run at exactly these
values, `tests/test_protocol_is_uniform.py` refuses any spec or runner case that departs
from them, and `docs/RESULTS.md` states them once instead of per row.

WHAT IS NOT HERE. Solve-side precision is per model and cannot be shared -- the three
papers have different solvers. It is uniform ACROSS each model's economies, which is the
property that matters, and it is recorded in every manifest under
`experiments/registry/<solve_id>.json`: KP14 NY=21 y-nodes with a byte-identical Qy and
CIR quadrature at epsrel 1e-6; GS21 xnum=161, znum=200, bnum=20, tol=1e-6; BGN 100-node
Gauss-Laguerre and Gauss-Hermite with J* at tol 3e-4. `docs/RESULTS.md` tabulates them.

HOW THE MODEL MODULES RELATE TO THIS FILE. `burnin` stays a plain literal in
`bgn_gam/parameters.py`, `kp_vy/parameters_kp14.py` and `gs_bx/gs_sim_bx.py` rather than
an import from here, for two reasons. Those files' RAW BYTES are digested into solve ids
(`common/solstamp.py`), so an import line costs a re-key of every cached solve for no
functional change; and `gs_sim_bx.py` applies GS_SIM_OVERRIDES by `globals().update()`
over its own module scope, which wants module-level assignments it can see. The literals
are pinned to BURNIN by `tests/test_protocol_is_uniform.py` instead -- the loop is closed
by a test rather than by an import, which is what was missing when the three modules sat
at 300 / 400 / 300 and twelve of nineteen specs declared a fourth value.
"""

# ---- the panel ---------------------------------------------------------------------
N = 500          # firms
T = 500          # months RETAINED, i.e. after burn-in
BURNIN = 400     # months simulated and discarded before month 0, all three models

# ---- the evaluation window ----------------------------------------------------------
WINDOW = 360     # months of the rolling estimation window, and the oracle's --eval_window
EVAL_TRIM = 15   # months run_oracle.py drops at the ends: 14 at the front, 1 at the back
                 # (`months >= burnin + 14` and `months <= T + burnin - 2`)

# The count nothing sets directly and every row must share. 500 - 15 - 360 = 125.
EVAL_MONTHS = T - EVAL_TRIM - WINDOW

SEEDS = 10       # seeds 0..SEEDS-1, every economy, no exceptions: a run with fewer is a
                 # smoke test and may not be reported (cross-seed sd of a gap runs 15% to
                 # 63% of its mean, so a short run ranks economies by luck)

# ---- the estimators -----------------------------------------------------------------
# The ridge penalty grid, shared by DKKM and by the linear ridge methods. It is wide
# enough on BOTH sides that the argmax is INTERIOR: a grid whose edge wins reports the
# grid, not the economy. `tests/test_protocol_is_uniform.py` pins the value; the campaign
# checks the argmax is interior per economy, which is a separate and necessary gate.
#
# 2026-09-21: widened from (1e-5 ... 10), seven values, to eleven. That gate FAILED in five
# of thirteen economies in the 2026-09-15 campaign, and it failed in both directions, which
# is why both ends move: vyx 3/10 and vyg25 4/10 won at the floor 1e-5, while gx7 2/10,
# bx7 7/10 and g0235s 5/10 won at the ceiling 10. A censored DKKM Sharpe is a lower bound
# of unknown size, and KP14's censoring sits under the project's only positive result. Two
# decades each side rather than one, because gx7's ceiling won in 8 of 10 seeds and one
# decade would plausibly not have uncensored it.
#
# This is a PROTOCOL AMENDMENT, not an experiment: it is estimator-side, so it moves every
# row of docs/RESULTS.md, and every live spec's estimation.kappas moves with it.
KAPPAS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

RFF = (36, 360, 3600)   # random-feature counts
NMAT = 2                # independent RFF draws per count
WINSOR = 0.0            # no winsorisation of excess returns

# Conditioning columns the feature bases see. None means THE MODEL'S FULL SET, for every
# economy including the baselines. Narrowing a baseline to the state its paper has (which
# is what --rf_cols was for until 2026-09-15) makes baseline and parameterization differ
# in their feature bases as well as their economics, and then "what the route added" is
# not a difference in the economy alone.
RF_COLS = None

# The RFF bandwidth grid. Every model module and both run_*.py scripts spell this as
# `np.arange(0.5, 1.1, 0.1)`, which is SEVEN values ending at 1.0999999999999999 -- an
# arange artifact, faithfully recorded in every solve manifest as shape [7]. It is left
# exactly as it is: correcting it would change the feature bases of every economy ever
# run, which is a change to the measurement, not to its bookkeeping.
BANDWIDTH_EXPR = "np.arange(0.5, 1.1, 0.1)"
N_BANDWIDTHS = 7


def kappas_csv():
    """The grid as run_estimators.py --kappas wants it. Used by run_seeds_slurm.sh so the
    shell holds no second copy: `KAPPAS=$(python -c 'import protocol; print(...)')`."""
    return ",".join(repr(k) for k in KAPPAS)


def describe():
    """One line per protocol quantity, for a log header or a doc check."""
    return (f"N={N} T={T} burnin={BURNIN} window={WINDOW} eval_months={EVAL_MONTHS} "
            f"seeds={SEEDS} kappas={kappas_csv()} rff={','.join(map(str, RFF))} "
            f"nmat={NMAT} winsor={WINSOR} rf_cols={'full' if RF_COLS is None else RF_COLS}")


if __name__ == "__main__":
    import sys
    print(describe() if len(sys.argv) == 1 else getattr(sys.modules[__name__], sys.argv[1]))
