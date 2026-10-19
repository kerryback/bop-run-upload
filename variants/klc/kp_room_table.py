#!/usr/bin/env python3
"""Tabulate the KP14 oracle-only room screen (variants/klc/kp_room_klc.sh).

    python variants/klc/kp_room_table.py [--dir /kellogg/proj/axa3203/bop_runs/kpscreen] [--ref /kellogg/proj/axa3203/bop_runs/results]

For every kp_vy oracle JSON in --dir: SR_max (eval), mean expected excess return (%/yr), room_eval
(best nonlinear ceiling minus lin_rank, aggregate_seeds.py's definition) and room_s_eval (the same
against lin_rank_rf, the linear basis GIVEN the state), averaged over seeds, with the vyg25 and
vyg30k70 reference rows from --ref. Only room_s is the screening statistic: KP14 converts room_s to
a fair gap at roughly 0.16 to 0.18 (vyg25 +0.0148 / +0.0820, vyg30k70 +0.0159 / +0.1017), so a
point needs room_s near +0.15 to promise a gap of +0.025. Nothing here is a gap.
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd

PAT = re.compile(r"kp_vy_oracle_(?P<tag>.+)_s(?P<seed>\d{3})\.json$")


def rows(d, tags=None):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "kp_vy_oracle_*_s???.json"))):
        m = PAT.search(os.path.basename(f))
        if not m or (tags and m["tag"] not in tags):
            continue
        r = json.load(open(f)); b = r["bases"]
        nl = [k for k in b if k.startswith("rff") or k in ("bins", "poly2")]
        best = max(b[k]["const_best_eval"] for k in nl)
        out.append(dict(tag=m["tag"], seed=int(m["seed"]), sr_max=r["sr_max_eval"], mu_pct=1200 * r["mean_mu"],
                        mkt_share=r["sr_ew_eval"] / r["sr_max_eval"],
                        room=best - b["lin_rank"]["const_best_eval"], room_s=best - b["lin_rank_rf"]["const_best_eval"],
                        overrides=r.get("overrides", "")))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/kellogg/proj/axa3203/bop_runs/kpscreen")
    ap.add_argument("--ref", default="/kellogg/proj/axa3203/bop_runs/results")
    args = ap.parse_args()
    df = rows(args.dir)
    REF_TAGS = {"vyg25", "vyx", "kpbase", "vyg30k70", "vyg25k15", "vyg25k70"}
    here = os.path.dirname(os.path.abspath(__file__))
    ref_dirs = [d for d in (args.ref, os.path.join(here, "..", "results")) if d and os.path.isdir(d)]
    ref = pd.concat([rows(d, tags=REF_TAGS) for d in ref_dirs], ignore_index=True) if ref_dirs else pd.DataFrame()
    both = pd.concat([ref.assign(kind="ref"), df.assign(kind="screen")], ignore_index=True)
    if both.empty:
        raise SystemExit("no kp_vy oracle JSONs found")
    g = both.groupby(["kind", "tag"]).agg(n=("seed", "size"), sr_max=("sr_max", "mean"), mu_pct=("mu_pct", "mean"),
                                           mkt=("mkt_share", "mean"), room=("room", "mean"), room_sd=("room", "std"),
                                           room_s=("room_s", "mean"), room_s_sd=("room_s", "std")).reset_index()
    g["state_share"] = 1 - g.room_s / g.room
    g = g.sort_values(["kind", "room_s"], ascending=[False, False])
    print(f"{'tag':10s} {'kind':6s} {'n':>2s} {'SRmax':>6s} {'mu%/yr':>6s} {'mkt%':>5s} {'room':>8s} {'sd':>7s} {'room_s':>8s} {'sd':>7s} {'state%':>6s}")
    for _, r in g.iterrows():
        print(f"{r.tag:10s} {r.kind:6s} {r.n:2d} {r.sr_max:6.3f} {r.mu_pct:6.1f} {100*r.mkt:5.0f} {r.room:+8.4f} {r.room_sd:7.4f} "
              f"{r.room_s:+8.4f} {r.room_s_sd:7.4f} {100*r.state_share:6.0f}")
    print("\nroom_s is the screening statistic (KP14 converts it to a fair gap at ~0.16-0.18). Oracle only: no gap is measured here.")


if __name__ == "__main__":
    main()
