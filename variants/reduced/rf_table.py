"""Aggregate screen_*.json files into one table per configuration: mean, sd and t across seeds
of the four gaps, the two rooms, and the premium statistics.

    python variants/reduced/rf_table.py --dir /path/to/screen_dir [--ladder speed]
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_configs

COLS = ["sr_max_eval", "mkt_share", "room_eval", "room_s_eval", "theta_wtd", "time_variation",
        "dkkm", "fm", "ff", "linrank_m", "linrank_s", "classical_gap", "fair_gap", "fair_s_gap", "excess_loss", "dkkm_pct_ceiling"]


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "screen_*_s???.json"))):
        r = json.load(open(f))
        row = {"name": r["name"], "seed": r["seed"], "seconds": r["seconds"]}
        row.update({k: r["oracle"][k] for k in r["oracle"] if not isinstance(r["oracle"][k], dict)})
        row.update({k: r["estimators"][k] for k in r["estimators"] if not isinstance(r["estimators"][k], (dict, list))})
        rows.append(row)
    return pd.DataFrame(rows)


def table(df, names=None):
    g = df.groupby("name")
    out = []
    for name, sub in g:
        if names and name not in names:
            continue
        rec = {"name": name, "n": len(sub)}
        for c in COLS:
            if c in sub:
                v = sub[c].astype(float)
                rec[c] = v.mean()
                rec[c + "_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
        for c in ("fair_gap", "fair_s_gap", "classical_gap"):
            v = sub[c].astype(float)
            rec[c + "_t"] = v.mean() / v.std(ddof=1) * np.sqrt(len(v)) if len(v) > 1 and v.std(ddof=1) > 0 else np.nan
            rec[c + "_pos"] = int((v > 0).sum())
        rec["at_edge"] = int(sub["dkkm_at_edge"].sum())
        out.append(rec)
    t = pd.DataFrame(out)
    if names:
        t["order"] = t.name.map({n: i for i, n in enumerate(names)}); t = t.sort_values("order").drop(columns="order")
    return t


def print_table(t):
    print(f"{'config':10s} {'n':>2s} {'SRmax':>6s} {'mkt%':>5s} {'room':>8s} {'room_s':>8s} {'theta':>6s} {'tvar':>6s} | "
          f"{'DKKM':>6s} {'FM':>6s} {'FF':>6s} {'lin_m':>6s} {'lin_s':>6s} | {'classical':>16s} {'fair gap':>16s} {'fair_s gap':>16s} {'edge':>4s}")
    for _, r in t.iterrows():
        def g(c):
            return f"{r[c]:+.4f} ({r[c+'_t']:4.1f},{r[c+'_pos']:2d})"
        print(f"{r['name']:10s} {r['n']:2d} {r['sr_max_eval']:6.3f} {100*r['mkt_share']:5.0f} {r['room_eval']:+8.4f} {r['room_s_eval']:+8.4f} "
              f"{r['theta_wtd']:6.3f} {r['time_variation']:+6.3f} | {r['dkkm']:6.4f} {r['fm']:6.4f} {r['ff']:6.4f} {r['linrank_m']:6.4f} {r['linrank_s']:6.4f} | "
              f"{g('classical_gap'):>16s} {g('fair_gap'):>16s} {g('fair_s_gap'):>16s} {r['at_edge']:4d}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ladder", default=None, help="one of rf_configs.LADDERS, to order the rows")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    df = load(args.dir)
    if df.empty:
        raise SystemExit("no screen_*_s???.json files")
    names = rf_configs.LADDERS[args.ladder] if args.ladder else None
    t = table(df, names)
    print_table(t)
    print("\n(t, positive seeds) in parentheses. 'edge': seeds whose winning DKKM penalty sat at a grid end.")
    if len(df) > 2:
        x, y = df.room_eval.astype(float), df.excess_loss.astype(float)
        b = np.polyfit(x, y, 1)
        print(f"excess_loss = {b[1]:+.4f} + {b[0]:.3f} x room  (corr {np.corrcoef(x, y)[0,1]:.2f}, {len(df)} seed-rows; the record's cross-economy slope is 0.887)")
    if args.out:
        t.to_csv(args.out, index=False); df.to_csv(args.out.replace(".csv", "_seeds.csv"), index=False)


if __name__ == "__main__":
    main()
