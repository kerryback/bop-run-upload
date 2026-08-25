"""
Section D -- the full complexity x shrinkage surface from the saved results pickles.
This is the table that shows the complexity curve is flat.
"""
import pickle, glob, numpy as np, pandas as pd
pd.set_option('display.width', 200)
D = "/Users/sjpruitt/ASU Dropbox/Seth Pruitt/BGN and Kelly Malamud/Code/aws_results"

for model in ["bgn", "kp14", "gs21"]:
    fs = sorted(glob.glob(f"{D}/{model}_*_results.pkl"))
    S, H, fam_s, fam_h = [], [], [], []
    for f in fs:
        try:
            r = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        ret, fa, dk = r.get('returns'), r.get('fama_results'), r.get('dkkm_results')
        if ret is None or dk is None:
            continue
        dk = dk.copy()
        dk['sharpe'] = dk['mean'] / dk['stdev']
        S.append(dk.pivot_table(index='alpha', columns='nfeatures', values='sharpe', aggfunc='mean'))
        m = dk.merge(ret[['month', 'sdf_ret']], on='month', how='left')
        m['se'] = (m.xret - m.sdf_ret) ** 2
        H.append(np.sqrt(m.pivot_table(index='alpha', columns='nfeatures', values='se', aggfunc='mean')))
        fa = fa.copy()
        fa['sharpe'] = fa['mean'] / fa['stdev']
        fam_s.append(fa.groupby('method')['sharpe'].mean())
        g = fa.merge(ret[['month', 'sdf_ret']], on='month', how='left')
        g['se'] = (g.xret - g.sdf_ret) ** 2
        fam_h.append(g.groupby('method')['se'].mean().apply(np.sqrt))
    if not S:
        continue
    print(f"\n{'=' * 92}\n{model.upper()}   ({len(S)} panels)\n{'=' * 92}")
    print("CONDITIONAL SHARPE  (rows = ridge alpha, cols = # RFF features)")
    print(pd.concat(S).groupby(level=0).mean().round(4).to_string())
    print("  benchmarks:", pd.concat(fam_s, axis=1).mean(axis=1).round(4).to_dict())
    print("\nHJD  (lower better)")
    print(pd.concat(H).groupby(level=0).mean().round(4).to_string())
    print("  benchmarks:", pd.concat(fam_h, axis=1).mean(axis=1).round(4).to_dict())
