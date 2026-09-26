"""A live document may only cite economies whose code still exists.

WHY: on 2026-09-17 `docs/RESULTS.md` carried eight citations into the pre-protocol narrative, and the
worst of them rested on five economies with no code in this repository -- one crash recipe built three
times, quoted as the controlled portability experiment. Nothing in `tests/` could see it: the only
test that reads RESULTS.md pins the unified-column TABLES cell by cell and never looks at prose, so a
number or an economy name in a sentence was unpoliced.

What makes those citations wrong is not that they are old. It is that they cannot be checked:

  * the legacy code was NEVER IN THIS REPOSITORY. `bba735f` adds 130 files under `variants/` and
    deletes none; `variants/` did not exist before it; and across every commit reachable from `--all`
    the count of paths matching `kp_gam`, `kp_gamy`, `kp_bx`, `kp_dis`, `bgn_types`, `bgn_dis`,
    `bgn_gamr`, `gs_fixed` or `gs_dis` is zero. So `git show <commit>:<path>` cannot recover them, and
    `docs/refactor/WORKING.md` section 50's "git keeps every byte" is true of the legacy TABLES
    (deleted at `9607f87`, recoverable at `23f9380`) and false of the legacy CODE.
  * every figure in that study is a single seed, and cross-seed sd of a gap runs 15% to over 300% of
    its mean on the economies later measured properly;
  * its KP14 levels ran at growth-option arrival rate 1.72 and its GS21 levels on three wrong Table I
    parameters.

So the rule this file enforces: the BODY of RESULTS.md cites only the economies in
`economy_table.csv`. Dead routes may be NAMED, once, in the "Routes with no code" appendix, which says
of itself that nothing in it is citable.

A denylist rather than a heuristic. The dead set is finite and known, so naming it has no false
positives, fails loudly if a route is quietly reintroduced, and doubles as the graveyard's index.
`tests/test_results_md_matches_table.py::_is_near_miss` argues the opposite way about TABLE SHAPES --
there the space of legitimate shapes is open, so an allowlist needed editing forever. Here the space
of dead economies is closed.

Run with: python -m pytest tests/ -k live_docs
"""
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "docs", "RESULTS.md")
CSV = os.path.join(ROOT, "variants", "results", "economy_table.csv")
APPENDIX = "## Routes with no code"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_specs_match_shell import CURRENT  # noqa: E402

# Routes with no code in this repository, and never any. Verified as a property of history, not of
# the working tree: `git log --all --pretty=format: --name-only` matches none of the module
# directories. Tags and module names both, since the prose names them either way.
GONE = {
    "kp_bx", "kp_gam", "kp_gamy", "kp_dis", "kpd_al", "kpd_off", "bxs", "gux", "g0525",
    "bgn_types", "bgn_dis", "bgn_gamr", "types3", "types3eq", "micro10", "sect9", "dis_p10",
    "dis_long", "g0530",
    "gs_fixed", "gs_dis", "gsd", "dis_rebase", "sol_gnl", "gs_sim_dis", "gs_sim_reg",
    # expressible from live code but never run at protocol, so equally uncitable for numbers
    "grot", "bx9", "g0520", "g0330", "vys",
}


def body_and_appendix():
    """RESULTS.md split at the appendix heading: (body, appendix)."""
    txt = open(MD).read()
    i = txt.find(APPENDIX)
    assert i > 0, f"{MD} has no {APPENDIX!r} section to fence the dead routes into"
    return txt[:i], txt[i:]


def live_tags():
    e = pd.read_csv(CSV)
    return {(r["model"], r["tag"]) for _, r in e.iterrows()}


def test_the_live_universe_is_one_set_agreed_by_three_sources():
    """economy_table.csv, the CURRENT spec map, and the model directories must agree.

    If they ever disagree, every other test here is policing the wrong set.
    """
    csv = live_tags()
    assert {t for _, t in csv} == set(CURRENT), (
        f"economy_table.csv tags {sorted({t for _, t in csv})} != CURRENT {sorted(CURRENT)}")
    models = {m for m, _ in csv}
    for m in models:
        assert os.path.isdir(os.path.join(ROOT, "variants", m)), f"no live model dir for {m}"
    assert len(csv) == 15, f"expected 15 live economies, found {len(csv)}"


def test_no_dead_route_is_named_outside_the_appendix():
    body, _ = body_and_appendix()
    hits = sorted({g for g in GONE if re.search(rf"(?<![\w-]){re.escape(g)}(?![\w-])", body)})
    assert not hits, (
        "the body of docs/RESULTS.md names routes with no code: " + ", ".join(hits)
        + f"\nA dead route may be named only in the {APPENDIX!r} appendix, which states that nothing "
          "in it is citable. Either move the mention there, or re-ground the claim on one of the 13 "
          "live economies, on live code, or on a stated argument.")


def test_the_body_does_not_cite_the_legacy_narrative():
    """archive/REPORT.md is history. Its numbers are single-seed and pre-correction."""
    body, _ = body_and_appendix()
    hits = [i + 1 for i, l in enumerate(body.splitlines()) if "REPORT.md" in l]
    assert not hits, (
        f"the body of docs/RESULTS.md cites REPORT.md at line(s) {hits}. It is a historical "
        "narrative: 21 of its 24 economies have no code, every figure is one seed, and its KP14 and "
        "GS21 levels ran on since-corrected calibrations. Cite live data, live code, or an argument.")


def test_every_model_slash_tag_in_the_body_is_live():
    """A `model/tag` token in the body must name one of the 13."""
    body, _ = body_and_appendix()
    live = live_tags()
    models = {m for m, _ in live}
    bad = sorted({f"{m}/{t}" for m, t in re.findall(r"\b([a-z]+_[a-z]+)/([A-Za-z0-9]+)\b", body)
                  if m in models and (m, t) not in live})
    assert not bad, (
        f"the body of docs/RESULTS.md names economies that are not live: {bad}\n"
        f"live: {sorted(f'{m}/{t}' for m, t in live)}")


def test_the_appendix_says_it_is_not_citable():
    """The fence is only a fence if it says so."""
    _, app = body_and_appendix()
    low = app.lower()
    assert "no claim in this section is citable" in low, (
        f"the {APPENDIX!r} appendix must state that no claim in it is citable")
    assert "never in this repository" in low or "never in this repository's git history" in low \
        or "was never in this repository" in low or "never was" in low, (
        "the appendix must say WHY these routes cannot be cited -- that the code was never in this "
        "repository's history, so no `git show` recovers it")


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
