"""
Section A -- effective dimension of the 3600-feature DKKM basis, exactly as coded
(including the second rank_standardize applied to the sin/cos features).
Run time is a few minutes; reduce N/D if impatient.
"""
import numpy as np, sys
rng = np.random.default_rng(1)
N, D = 1500, 900          # D half-features -> 2*D total
G = np.arange(0.5, 1.1, 0.1)


def rankstd(v):
    n = v.shape[0]
    o = np.argsort(v, axis=0)
    r = np.empty(v.shape)
    r[o, np.arange(v.shape[1])] = np.arange(1, n + 1).reshape(-1, 1)
    return (r - 0.5) / n - 0.5


def spec(L, g, label):
    X = rankstd(rng.standard_normal((N, L)))
    W = rng.choice(g, size=(D, 1)) * rng.standard_normal((D, L))
    Z = X @ W.T
    F = rankstd(np.hstack([np.sin(Z), np.cos(Z)]))   # matches dkkm_functions.rff (double rank-std)
    F = F - F.mean(0)
    s = np.linalg.svd(F, compute_uv=False) ** 2
    s = s / s.sum()
    cum = np.cumsum(s)
    k = lambda p: int(np.searchsorted(cum, p) + 1)
    part = (s.sum() ** 2) / (s ** 2).sum()            # participation ratio = effective rank
    print(f"  {label:20s} eff.rank(PR)={part:7.1f}   #PCs for 90%={k(.90):5d}  95%={k(.95):5d}  "
          f"99%={k(.99):5d}   top-21 share={cum[20]:.3f}")
    sys.stdout.flush()


print("Effective dimension of the 3600-feature DKKM basis, exactly as coded\n")
spec(5,   G,       "L=5  gam .5-1.0")
spec(6,   G,       "L=6  gam .5-1.0")
spec(5,   G * 3,   "L=5  gam x3")
spec(5,   G * 5.1, "L=5  gam x5.1")
spec(130, G,       "L=130 gam .5-1.0")
