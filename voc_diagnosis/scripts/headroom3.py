"""
Section B3 -- THE key experiment.
Sweep K = number of PRICED aggregate shocks, holding total max Sharpe FIXED.
Loadings are nonlinear+interactive in the 5 characteristics (theta=0.7).
Question: at what K does a low-dimensional empirical factor model stop spanning the SDF?
"""
import numpy as np
N, L = 1000, 5


def sh(S, Sig, rp):
    M = S.T @ Sig @ S
    v = S.T @ rp
    return float(np.sqrt(max(v @ np.linalg.solve(M + 1e-10 * np.trace(M) / len(M) * np.eye(len(M)), v), 0.0)))


def build(K, seed, theta=0.7):
    r = np.random.default_rng(seed)
    x = r.standard_normal((N, L))
    Xl = np.column_stack([np.ones(N), x])
    raw = np.column_stack([x[:, 0] * x[:, 1], x[:, 1] * x[:, 2], np.abs(x[:, 3]),
                           (x[:, 0] > .6) * (x[:, 1] < -.4),
                           np.tanh(2 * x[:, 2]) * x[:, 4], np.exp(-x[:, 0] ** 2), x[:, 1] ** 3,
                           np.sign(x[:, 4]) * np.sqrt(np.abs(x[:, 0])), x[:, 2] * x[:, 3] * x[:, 4]])
    raw = raw - Xl @ np.linalg.lstsq(Xl, raw, rcond=None)[0]
    raw /= raw.std(0)
    lin = x @ r.standard_normal((L, K))
    nl = raw @ r.standard_normal((raw.shape[1], K))
    lin /= lin.std(0)
    nl /= nl.std(0)
    B = 1.0 + 0.5 * (np.sqrt(1 - theta) * lin + np.sqrt(theta) * nl)
    sig_f = 0.035
    lam = r.standard_normal(K)
    lam = lam / np.linalg.norm(lam) * (3.2 * sig_f)   # total factor SR fixed at 3.2
    Om = np.eye(K) * sig_f ** 2
    d = (0.08 ** 2) * np.exp(0.8 * (x @ np.array([1, 0, 0, 0, 0.])) - 0.32)
    return x, B @ Om @ B.T + np.diag(d), B @ lam


def bases(x, seed):
    one = np.ones((N, 1))
    lin = np.column_stack([one, x])
    size = x[:, 0]
    big = size > np.median(size)
    cols = [np.ones(N) / N, (big / big.sum() - (~big) / (~big).sum())]
    for j in range(1, L):
        c = x[:, j]
        H = c > np.quantile(c, .7)
        Lo = c <= np.quantile(c, .3)
        cols.append(0.5 * ((H & big) / max((H & big).sum(), 1) + (H & ~big) / max((H & ~big).sum(), 1)
                           - (Lo & big) / max((Lo & big).sum(), 1) - (Lo & ~big) / max((Lo & ~big).sum(), 1)))
    ff = np.column_stack(cols)
    rr = np.random.default_rng(seed + 7)
    xs = x / x.std(0) / np.sqrt(12)      # put x on the rank-standardized scale the code uses

    def rff(gm, P):
        W = rr.choice(np.arange(.5, 1.1, .1) * gm, size=(P, 1)) * rr.standard_normal((P, L))
        z = xs @ W.T
        return np.column_stack([one, np.sin(z), np.cos(z)])

    return {'FM': lin, 'FF': ff, 'DKKMcoded': rff(1.0, 200), 'DKKMwide': rff(5.1, 200)}


print(f"{'K (priced shocks)':>18} | {'FM':>6} {'FF':>6} {'DKKM coded':>11} {'DKKM wide':>10} | {'ORACLE':>7} | {'wide/FM':>8}")
print("-" * 80)
for K in [1, 2, 3, 5, 10, 20, 40]:
    acc, orc = {}, []
    for s in range(3):
        x, Sig, rp = build(K, 300 + s)
        for k, S in bases(x, 300 + s).items():
            acc.setdefault(k, []).append(sh(S, Sig, rp))
        orc.append(float(np.sqrt(rp @ np.linalg.solve(Sig, rp))))
    m = {k: np.mean(v) for k, v in acc.items()}
    print(f"{K:18d} | {m['FM']:6.3f} {m['FF']:6.3f} {m['DKKMcoded']:11.3f} {m['DKKMwide']:10.3f} | "
          f"{np.mean(orc):7.3f} | {m['DKKMwide'] / m['FM']:7.2f}x")
