"""Per-type premium decomposition: type EW portfolio return regressed on EW market and d(state).
usage: hump_decomp.py <kp|gs|bgn> <econ> <seed>"""
import sys, json, numpy as np, pandas as pd
m, econ, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
RES='/kellogg/proj/axa3203/bop_runs/results'; REPO='/kellogg/proj/axa3203/bop-run-upload'
fam={'kp':'kp_vy','gs':'gs_vy','bgn':'bgn_vy'}[m]
spec=json.load(open(f'{REPO}/experiments/specs/var-{fam}-{econ}-v1.json'))
if m=='gs': bv=np.array(spec['types']['exposure']['gs_bv']); sh=np.array(spec['types']['share'])
else: bv=np.array(spec['params']['type_bv']); sh=np.array(spec['params']['type_share'])
p=pd.read_parquet(f'{RES}/{fam}_panel_{econ}_s{seed:03d}.parquet')
st='y_stand' if 'y_stand' in p else 'rf_stand'
if m=='bgn': ft=p.groupby('firmid').ftype.first().astype(int)
else:
    rng=np.random.default_rng(777+seed) if m=='kp' else np.random.default_rng(np.random.SeedSequence(909+seed).spawn(2)[1])
    N=500; ft=pd.Series(rng.choice(len(bv),size=N,p=sh/sh.sum()))
p['ft']=p.firmid.map(ft).astype(int)
mk=p.groupby('month').xret.mean(); y=p.groupby('month')[st].first(); dy=y.diff()
R=p.groupby(['month','ft']).xret.mean().unstack()
X=pd.concat([mk,dy],axis=1).dropna(); X.columns=['mkt','dy']
lam=X.mean()   # sample 'premia' of the two regressors
print(f'{econ} s{seed}: E[mkt] {100*lam.mkt:.3f}%/mo')
for k in R.columns:
    d=pd.concat([R[k],X],axis=1).dropna(); A=np.c_[np.ones(len(d)),d.mkt,d.dy]
    c=np.linalg.lstsq(A,d.iloc[:,0].values,rcond=None)[0]; res=d.iloc[:,0].values-A@c
    print(f' b {bv[k]:+.3f}  mean {100*d.iloc[:,0].mean():+.3f}  beta_mkt {c[1]:.3f}  beta_dy {c[2]:+.4f}  mkt part {100*c[1]*lam.mkt:+.3f}  alpha {100*c[0]:+.3f}  resid sd {100*res.std():.2f}')
