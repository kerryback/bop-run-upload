"""`room` and `gap` must be averaged over the SAME months.

run_oracle.py reports every population ceiling twice: over all months (what every
number published before 2026-09-09 means) and restricted to the evaluation window
(what can be differenced against an estimator result). The restriction is only
meaningful if it selects exactly the months run_estimators.py scores.

Those two rules live in two files and can drift. WORKING.md 41 flagged the confound
this creates: the oracle averaged over all 485 months of a flagship panel while the
estimators averaged over the 125 that survive a 360-month window, so the finding
"realized gap exceeds const-theta room" had an unruled-out month-sample confound
underneath it. This test is what stops that recurring silently.

Run with: python -m pytest tests/ -k eval_window
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EST = os.path.join(ROOT, "variants", "run_estimators.py")
ORC = os.path.join(ROOT, "variants", "run_oracle.py")

# The two rules, as they read on 2026-09-09. If either assertion fails, the rule was
# edited: re-derive the equivalence below and update BOTH constants together.
EST_RULE = "eval_months = [m for m in range(start + args.window, end + 1) if m in midx]"
ORC_RULE = "_eval_mask = _months_arr >= (_months_arr.min() + args.eval_window)"


def test_estimator_month_rule_is_unchanged():
    assert EST_RULE in open(EST).read(), (
        "run_estimators.py's evaluation-month rule changed. run_oracle.py's "
        "--eval_window mask must be changed to match, or the *_eval ceilings stop "
        "being differenceable against a gap.")


def test_oracle_month_rule_is_unchanged():
    assert ORC_RULE in open(ORC).read(), (
        "run_oracle.py's --eval_window mask changed; re-check it against EST_RULE.")


def test_the_two_rules_select_the_same_months():
    """Numeric equivalence of the two rules, at the same window.

    `start`/`end` in run_estimators.py are min/max of the months the ORACLE saved into
    the moments file, and `midx` is keyed on that same array -- so the `if m in midx`
    filter is a no-op over that set and the two reduce to `month >= min + window`.
    Checked on a gappy month array, since the oracle drops months with NaN chars.
    """
    months = [314] + list(range(320, 800, 1)) + [810, 900]
    for window in (0, 1, 36, 120, 360, 500):
        start, end, midx = min(months), max(months), {m: i for i, m in enumerate(months)}
        est = [m for m in range(start + window, end + 1) if m in midx]
        orc = [m for m in months if m >= min(months) + window]
        assert est == orc, f"window={window}: {est[:5]}... != {orc[:5]}..."


def test_oracle_defaults_to_the_flagship_window():
    """--eval_window's default must be the window the campaign actually scores at,
    so a run that forgets the flag still produces comparable numbers."""
    m = re.search(r'"--eval_window", type=int, default=(\d+)', open(ORC).read())
    assert m and int(m.group(1)) == 360, "flagship runs use --window 360"


def test_estimators_warn_when_the_windows_disagree():
    src = open(EST).read()
    assert "_oracle_eval_window" in src and "do NOT cover" in src, (
        "run_estimators.py must say so when the oracle's --eval_window is not this "
        "run's --window; re-running estimators at a new window against an existing "
        "panel is legitimate and has been done, so this is a warning, not an abort.")
