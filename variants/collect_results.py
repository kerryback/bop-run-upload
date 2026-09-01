"""Collect all oracle-decomposition JSONs into one table."""
import glob, json, os, pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "*_oracle_*.json"))):
    s = json.load(open(f))
    b = s["bases"]
    lin = b["lin_rank"]["const_best"]
    nl = max(b[k]["const_best"] for k in b if k.startswith("rff") or k in ("bins", "poly2"))
    rows.append({"model": s["model"], "tag": s["tag"], "N": s["N"], "months": s["months"], "overrides": s["overrides"],
                 "SR_max": s["sr_max_mean"], "lin_rank": lin, "lin_rank_rf": b.get("lin_rank_rf", {}).get("const_best", float("nan")),
                 "fmr_raw": b["fmr_raw"]["const_best"], "poly2": b["poly2"]["const_best"], "bins": b["bins"]["const_best"],
                 "rff36": b.get("rff36", {}).get("const_best"), "rff360": b.get("rff360", {}).get("const_best"),
                 "rff3600": b.get("rff3600", {}).get("const_best"),
                 "nonlin_gain": nl - lin, "unlearnable": s["sr_max_mean"] - nl,
                 "cs_sd_mu": s["sd_mu"], "idio_sd": s["mean_idio_sd"]})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "oracle_summary.csv"), index=False)
