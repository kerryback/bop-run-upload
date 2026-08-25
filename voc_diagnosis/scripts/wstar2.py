"""
Section E -- the decisive diagnostic, run on the actual saved TRUE SDF weights.
How much of w* is spanned by
  (a) the LINEAR span of the characteristics  -> exactly what Fama-MacBeth can produce
  (b) linear + quadratic                       -> what DKKM AS CODED can produce (eff.rank ~17)
  (c) a wide-bandwidth RFF basis               -> a genuinely nonlinear basis
The (c)-(a) gap bounds how much a nonlinear characteristic-based estimator can add
IN WEIGHT SPACE. Note it does NOT bound the Sharpe gain -- see findings.md section E.
"""
import pickle, numpy as np, warnings
warnings.filterwarnings('ignore')
BASE = '/Users/sjpruitt/ASU Dropbox/Seth Pruitt/BGN and Kelly Malamud/wts_prediction'
DROP = {'month', 'firmid', 'xret', 'default', 'default_lag', 'rf_stand'}


def rankstd(v):
    n = v.shape[0]
    o = np.argsort(v, axis=0)
    r = np.empty(v.shape)
    r[o, np.arange(v.shape[1])] = np.arange(1, n + 1).reshape(-1, 1)
    return (r - 0.5) / n - 0.5


def r2(y, B):
    yh = B @ np.linalg.lstsq(B, y, rcond=None)[0]
    return 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)


print(f"{'model':6s} {'#cs':>4s} {'chars':>28s} | {'R2 LINEAR':>10s} {'R2 +QUAD':>9s} {'R2 wideRFF':>11s} | {'headroom':>9s}")
for model in ['bgn', 'kp14', 'gs21']:
    d = pickle.load(open(f'{BASE}/{model}_sdfwts_chars.pkl', 'rb'))
    sw, ch = d['sdfwts'], d['chars']
    rng = np.random.default_rng(0)
    L_, Q_, R_, cols = [], [], [], None
    for p in sorted(set(sw) & set(ch))[:3]:
        W, C = sw[p], ch[p]
        cols = [c for c in C.columns if c not in DROP]
        for m in list(W.index)[::12]:
            X = C[C.month == m]
            if len(X) < 200:
                continue
            ids = X.firmid.values
            valid = [i for i, f in enumerate(ids) if f in W.columns]
            if len(valid) < 200:
                continue
            Xv = X.iloc[valid][cols].replace([np.inf, -np.inf], np.nan)
            ok = Xv.notna().all(axis=1).values
            Xv, fid = Xv[ok], ids[valid][ok]
            y = W.loc[m, fid].values.astype(float)
            good = np.isfinite(y)
            y, Xv = y[good], Xv[good]
            if len(y) < 200:
                continue
            lo, hi = np.percentile(y, [1, 99])
            y = np.clip(y, lo, hi)                      # winsorize, as the MLP code does
            Z = rankstd(Xv.values.astype(float))
            n, L = Z.shape
            lin = np.column_stack([np.ones(n), Z])
            quad = np.column_stack([lin] + [(Z[:, i] * Z[:, j])[:, None]
                                            for i in range(L) for j in range(i, L)])
            Wm = rng.choice(np.arange(.5, 1.1, .1) * 5.1, size=(150, 1)) * rng.standard_normal((150, L))
            z = Z @ Wm.T
            rf = np.column_stack([np.ones(n), np.sin(z), np.cos(z)])
            L_.append(r2(y, lin)); Q_.append(r2(y, quad)); R_.append(r2(y, rf))
    if L_:
        print(f"{model.upper():6s} {len(L_):4d} {','.join(cols):>28s} | {np.mean(L_):10.3f} "
              f"{np.mean(Q_):9.3f} {np.mean(R_):11.3f} | {np.mean(R_) - np.mean(L_):+9.3f}")
