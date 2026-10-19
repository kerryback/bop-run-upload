#!/usr/bin/env python3
"""Write the solve ids a kp_vy build printed into its spec, recomputing spec_hash.

    python variants/klc/fill_spec_ids.py experiments/specs/<spec>.json <build log>

build_vy_tables.py prints "[solstamp] kp_vy prefix=<p> stage=G solve_id=..." and the same for integ.
The integ id is CHAINED (it hashes the G tables' bytes) and so can only be known after the build; this
is the step that records it. It refuses to overwrite a different id already pinned."""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from precommit_spec import _spec_hash   # the repo's one hashing rule

spec_path, log_path = sys.argv[1], sys.argv[2]
s = json.load(open(spec_path))
if s["model"] in ("bgn_vy", "bgn_q"):
    # one line: [solstamp] bgn_vy <jstar_vy_file> solve_id=<id>
    m = re.findall(rf"\[solstamp\] {s['model']} {re.escape(s['params']['jstar_vy_file'])} solve_id=([0-9a-f]{{16}})", open(log_path).read())
    if len(set(m)) != 1:
        sys.exit(f"{s['spec_id']}: found jstar ids {sorted(set(m))} in {log_path}; need exactly one")
    ids = {"jstar": m[0]}
else:
    prefix = s["env"]["KP_VY_PREFIX"]; model = s["model"]
    ids = dict(re.findall(rf"\[solstamp\] {model} prefix={re.escape(prefix)} stage=(\w+) solve_id=([0-9a-f]{{16}})", open(log_path).read()))
    if set(ids) != {"G", "integ"}:
        sys.exit(f"{s['spec_id']}: found ids {ids} in {log_path}; need both G and integ")
for k, v in ids.items():
    old = (s.get("expected_solves") or {}).get(k)
    if old and old != v:
        sys.exit(f"{s['spec_id']}: {k} already pinned to {old}, build printed {v}; refusing to overwrite")
s["expected_solves"] = dict(ids); s["solves_pending"] = None
s.setdefault("provenance", {})["solve_note"] = f"built on Quest by variants/klc/solve_{'bgn_vy' if s['model'] in ('bgn_vy', 'bgn_q') else 'kp'}_klc.sh; ids read from the build log {os.path.basename(log_path)} (see PRECOMMITMENT STATUS in notes)"
s["spec_hash"] = _spec_hash(s)
json.dump(s, open(spec_path, "w"), indent=1)
print(f"{s['spec_id']}: expected_solves {s['expected_solves']} spec_hash {s['spec_hash'][:16]}")
