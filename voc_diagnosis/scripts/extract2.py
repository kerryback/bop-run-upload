import json, os
J='/Users/sjpruitt/.claude/projects/-Users-sjpruitt-GitHub-bop-run-upload/c19e08d3-1736-4cbf-9117-7e2b1e92760e/subagents/workflows/wf_b0f7103e-c85/journal.jsonl'
OUT='/private/tmp/claude-503/-Users-sjpruitt-GitHub-bop-run-upload/c19e08d3-1736-4cbf-9117-7e2b1e92760e/scratchpad/wf'
os.makedirs(OUT, exist_ok=True)
for l in open(J):
    d = json.loads(l)
    if d.get('type') != 'result':
        continue
    r = d['result']
    if isinstance(r, str):
        try:
            r = json.loads(r)
        except Exception:
            pass
    if isinstance(r, dict):
        if 'scope' in r:
            name = 'diag_' + r['scope'][:24].replace(' ', '_').replace('/', '_')
        elif 'target' in r and 'proposals' in r:
            name = 'prop_' + str(r['target'])[:20].replace(' ', '_')
        elif 'lens' in r:
            name = 'crit_' + str(r.get('target')) + '_' + str(r.get('lens'))
        else:
            name = 'other_' + d['agentId'][:8]
    else:
        name = 'raw_' + d['agentId'][:8]
    p = os.path.join(OUT, name + '.json')
    json.dump(r, open(p, 'w'), indent=1)
    print("%-52s %8.1f KB" % (name, os.path.getsize(p) / 1024))
