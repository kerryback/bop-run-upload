#!/usr/bin/env python3
"""Write the solve ids the gs_vy solve array printed into its spec, recomputing spec_hash.

    python variants/klc/fill_gs_vy_ids.py experiments/specs/<spec>.json <log dir>

Each task's detail log (<log dir>/solve_gvy.detail.<soldir>.log) carries "[solstamp] gs_vy <soldir> solve_id=...".
Refuses to overwrite a different id already pinned."""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from precommit_spec import _spec_hash

spec_path, logdir = sys.argv[1], sys.argv[2]
s = json.load(open(spec_path))
ids = {}
for st in s["solve"]["stages"]:
    name = st["name"]
    log = os.path.join(logdir, f"solve_gvy.detail.{name}.log")
    m = re.findall(rf"\[solstamp\] gs_vy {re.escape(name)} solve_id=([0-9a-f]{{16}})", open(log).read())
    if len(set(m)) != 1:
        sys.exit(f"{s['spec_id']}: {name}: found ids {sorted(set(m))} in {log}; need exactly one")
    old = (s.get("expected_solves") or {}).get(name)
    if old and old != m[0]:
        sys.exit(f"{s['spec_id']}: {name} already pinned to {old}, build printed {m[0]}; refusing to overwrite")
    ids[name] = m[0]
s["expected_solves"] = ids; s["solves_pending"] = None
s.setdefault("provenance", {})["solve_note"] = f"built on Quest by variants/klc/solve_gs_vy_klc.sh; ids read from the detail logs in {logdir} (see PRECOMMITMENT STATUS in notes)"
s["spec_hash"] = _spec_hash(s)
json.dump(s, open(spec_path, "w"), indent=1)
print(f"{s['spec_id']}: expected_solves {ids} spec_hash {s['spec_hash'][:16]}")
