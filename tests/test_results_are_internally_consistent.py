"""An oracle summary and the time series beside it must describe the same run.

WHY: on 2026-09-08, commit e2cd44e replaced kp_vy_oracle_vyx_s000/1/2.json with the
flagship N=500/T=500 summaries fetched from Sol, but the *_ts.csv beside them were never
fetched -- they stayed as the 185-row N=200/T=200 series from the 2026-09-07 laptop runs.
Nothing noticed until the cluster refused a pull because its (correct, 485-row) copies
differed from what the repo carried. A JSON saying months=485 next to a CSV with 185
rows is the kind of quiet inconsistency that becomes a wrong figure a month later.

The oracle writes `months = len(ts)` into its own summary, so the pairing is checkable
with no assumptions about sizes.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "variants", "results")


def pairs():
    for js in sorted(glob.glob(os.path.join(RESULTS, "*_oracle_*.json"))):
        if js.endswith(".prov.json"):
            continue
        ts = js[:-5] + "_ts.csv"
        if os.path.exists(ts):
            yield js, ts


def test_every_oracle_json_matches_the_row_count_of_its_time_series():
    bad = []
    n = 0
    for js, ts in pairs():
        d = json.load(open(js))
        if "months" not in d:
            continue
        rows = sum(1 for _ in open(ts)) - 1
        n += 1
        if rows != d["months"]:
            bad.append(f"{os.path.basename(js)}: months={d['months']} but "
                       f"{os.path.basename(ts)} has {rows} rows")
    assert n > 0, "found no oracle JSON / _ts.csv pairs to check"
    assert not bad, ("oracle summaries whose time series is from a DIFFERENT run:\n  "
                     + "\n  ".join(bad))


def test_every_estimator_run_json_names_the_solves_its_oracle_consumed():
    """The estimator stage carries the oracle's solve_ids forward; they must agree."""
    bad = []
    n = 0
    for rj in sorted(glob.glob(os.path.join(RESULTS, "*_estimators_*_run.json"))):
        m = re.match(r"(.*)_estimators_(.*)_s(\d+)_w\d+.*_run\.json", os.path.basename(rj))
        if not m:
            continue
        oj = os.path.join(RESULTS, f"{m.group(1)}_oracle_{m.group(2)}_s{m.group(3)}.json")
        if not os.path.exists(oj):
            continue
        n += 1
        r = json.load(open(rj)); o = json.load(open(oj))
        rs = sorted(x["solve_id"] for x in (r.get("solves") or []))
        os_ = sorted(x["solve_id"] for x in (o.get("solves") or []))
        if rs != os_:
            bad.append(f"{os.path.basename(rj)}: {rs} vs oracle {os_}")
    assert n > 0, "found no estimator/oracle pairs to check"
    assert not bad, "estimator runs disagreeing with their oracle about solves:\n  " + "\n  ".join(bad)



def test_every_estimator_summary_is_the_aggregate_of_its_full_csv():
    """The summary is COMPUTED from the full per-month csv, by run_estimators.py:

        summ = res.groupby(["method","P","kappa"]).agg(
            sharpe=("sharpe","mean"),
            hjd=("hjd", lambda x: np.sqrt(x.mean())),
            real_sr=("xret", lambda x: x.mean()/x.std()))

    So regrouping the full file must reproduce the summary. If it does not, the two
    files are from different runs -- the same failure as the oracle/_ts.csv mismatch,
    one level down. The full csv is the evidence behind every t-statistic in the
    summary (a plain iid t over 125 monthly differences), which is why it is committed.
    """
    import numpy as np
    import pandas as pd
    bad = []
    n = 0
    for full in sorted(glob.glob(os.path.join(RESULTS, "*_estimators_*_w*.csv"))):
        if full.endswith("_summary.csv"):
            continue
        summ_path = full[:-4] + "_summary.csv"
        if not os.path.exists(summ_path):
            continue
        n += 1
        res = pd.read_csv(full)
        want = pd.read_csv(summ_path)
        got = (res.groupby(["method", "P", "kappa"])
                  .agg(sharpe=("sharpe", "mean"),
                       hjd=("hjd", lambda x: np.sqrt(x.mean())),
                       real_sr=("xret", lambda x: x.mean() / x.std()))
                  .reset_index())
        m = want.merge(got, on=["method", "P", "kappa"], suffixes=("", "_regrouped"))
        if len(m) != len(want):
            bad.append(f"{os.path.basename(full)}: {len(want)} summary rows, "
                       f"{len(m)} matched in the full csv")
            continue
        for col in ("sharpe", "hjd", "real_sr"):
            if not np.allclose(m[col], m[col + "_regrouped"], rtol=1e-7, atol=1e-10,
                               equal_nan=True):
                worst = (m[col] - m[col + "_regrouped"]).abs().max()
                bad.append(f"{os.path.basename(full)}: {col} differs from its "
                           f"summary, max |diff| {worst:.3e}")
    assert n > 0, "found no full/summary estimator pairs to check"
    assert not bad, ("estimator summaries that are NOT the aggregate of the full csv "
                     "beside them:\n  " + "\n  ".join(bad))

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
