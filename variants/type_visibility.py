"""Type-visibility diagnostic: how much of a hidden exposure type b_f the ranked characteristics reveal,
and whether the realised premium is monotone in b_f.
usage: python type_visibility.py <model kp|gs|bgn> <econ> <seed> [--chars size,bm,...]
Reconstructs the type draw (KP14: default_rng(777+seed); GS21: SeedSequence(909+seed).spawn(2)[1]; BGN: panel ftype),
checks it with the per-type y-beta (must be monotone in b_f), then reports per-type mean return, y-beta, mean chars,
and the R2 / correlations of b_f on within-month ranks of the characteristics.
"""
import sys, json, argparse, numpy as np, pandas as pd
RES = '/kellogg/proj/axa3203/bop_runs/results'; REPO = '/kellogg/proj/axa3203/bop-run-upload'
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('econ'); ap.add_argument('seed', type=int)
ap.add_argument('--chars', default=None); ap.add_argument('--N', type=int, default=None); a = ap.parse_args()
fam = {'kp': 'kp_vy', 'gs': 'gs_vy', 'bgn': 'bgn_vy'}[a.model]
spec = json.load(open(f'{REPO}/experiments/specs/var-{fam}-{a.econ}-v1.json'))
if a.model == 'gs':
    bv = np.array(spec['types']['exposure']['gs_bv']); share = np.array(spec['types']['share'])
else:
    bv = np.array(spec['params']['type_bv']); share = np.array(spec['params']['type_share'])
chars = (a.chars or ','.join(spec['estimation']['chars'])).split(',')
p = pd.read_parquet(f'{RES}/{fam}_panel_{a.econ}_s{a.seed:03d}.parquet')
state = 'y_stand' if 'y_stand' in p else 'rf_stand'
N = a.N or int(p.firmid.max()) + 1
if a.model == 'bgn':
    ft = p.groupby('firmid').ftype.first().astype(int)
else:
    if a.model == 'kp':
        rng = np.random.default_rng(777 + a.seed)
    else:
        rng = np.random.default_rng(np.random.SeedSequence(909 + a.seed).spawn(2)[1])
    ft = pd.Series(rng.choice(len(bv), size=N, p=share / share.sum()), index=np.arange(N))
p['ftype'] = p.firmid.map(ft).astype(int); p['b'] = bv[p.ftype.values]
chars = [c for c in chars if c in p]
# per-type table
rows = []
for k in range(len(bv)):
    s = p[p.ftype == k]
    dy = s.groupby('firmid')[state].diff(); ok = dy.notna().values
    X = np.c_[np.ones(ok.sum()), dy.values[ok]]; beta = np.linalg.lstsq(X, s.xret.values[ok], rcond=None)[0][1]
    r = dict(type=k, b=bv[k], n_fm=len(s), mean_ret_pct=100 * s.xret.mean(), dybeta=beta, mean_mve=s.mve.mean())
    for c in chars: r['mean_' + c] = s[c].mean()
    rows.append(r)
tab = pd.DataFrame(rows); pd.set_option('display.width', 200); print(tab.round(4).to_string(index=False))
mv = tab.mean_mve.values; mono = np.all(np.diff(mv) > 0) or np.all(np.diff(mv) < 0)
print(f'mean mve monotone in b: {mono}  (type draw {"CONFIRMED" if mono else "NOT confirmed -- check N / seed rule"}; a wrong draw gives flat means)')
mr = tab.mean_ret_pct.values; print(f'mean return monotone in b: {np.all(np.diff(mr) > 0) or np.all(np.diff(mr) < 0)}')
# visibility: b_f on within-month ranks
for c in chars: p['rk_' + c] = p.groupby('month')[c].rank(pct=True) - 0.5
R = p[['rk_' + c for c in chars]].values; X = np.c_[np.ones(len(p)), R]
coef, *_ = np.linalg.lstsq(X, p.b.values, rcond=None); fit = X @ coef
r2 = 1 - ((p.b.values - fit) ** 2).sum() / ((p.b.values - p.b.mean()) ** 2).sum()
print(f'R2 of b_f on ranks of {chars}: {r2:.3f}')
print('corr(b_f, rank):', {c: round(np.corrcoef(p.b.values, p['rk_' + c].values)[0, 1], 3) for c in chars})
# how much of the premium a linear function of ranks spans: regress type mean return on fitted b
tm = p.groupby('ftype').xret.mean() * 100
print('premium by type (%/mo):', tm.round(3).to_dict())
