"""How much default risk does GS21 PRICE, at the states it actually simulates?

The value functions integrate over a price-adjustment shock m ~ N(0, sigma_m^2)
truncated to +-4*sigma_m: shock_survive(P) = P(P + m > 0) is the no-default
probability, and it enters the debt payoff directly. The panel draws x, z, eta
and i_cost -- but NOT m. So default in the panel is P(state) <= 0, while default
in the pricing is P(state) + m <= 0. This measures the gap.
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
REPO = '/Users/sjpruitt/GitHub/bop-run-upload'
sys.path.insert(0, REPO)
import config
import importlib.util as ilu
sp = ilu.spec_from_file_location('_gs', os.path.join(REPO,'utils_gs21','gs21_solve.py'))
gs = ilu.module_from_spec(sp); sp.loader.exec_module(gs)

su = gs.Setup()                      # shipped params, quad='exact'
sigma_m = su.sigma_m
print(f'sigma_m = {sigma_m}, truncation +-{4*sigma_m}\n')

# --- 1. priced default probability across the whole solved grid ---
import pandas as pd
D = os.path.join(REPO,'utils_gs21','GS21_solfiles')
rd = lambda n: pd.read_csv(f'{D}/{n}.csv', header=None).values.ravel()
for nm in ('P_up','P_down'):
    P = rd(nm)
    pd_def = 1.0 - su.shock_survive(P)
    print(f'{nm}: P in [{P.min():.2f}, {P.max():.1f}]')
    print(f'   priced monthly default prob: max {pd_def.max():.4%}  mean {pd_def.mean():.4%}')
    print(f'   annualised at the worst grid point: {1-(1-pd_def.max())**12:.2%}')
    print(f'   grid points with priced monthly prob > 0.1%: '
          f'{(pd_def>0.001).sum():,} of {len(P):,}')

# --- 2. now at the states the panel actually visits ---
print('\n--- simulating a panel and evaluating there ---')
from utils_gs21.panel_functions_gs21 import create_arrays
np.random.seed(11)
N, T = 500, 300
arrs = create_arrays(N, T)
# locate P in the returned tuple by shape/scale
Pmat = None
for a in arrs:
    if isinstance(a, np.ndarray) and a.shape == (T+1, N) and a.dtype == float:
        if a.min() > 1 and a.max() > 50:      # P is the only large positive (T+1,N) float
            Pmat = a; break
if Pmat is None:
    print('  could not locate P in arr_tuple; shapes:',
          [getattr(a,'shape',type(a).__name__) for a in arrs])
else:
    burn = config.GS21_BURNIN
    Ps = Pmat[burn:].ravel()
    pdf = 1.0 - su.shock_survive(Ps)
    print(f'  simulated P over {len(Ps):,} firm-months: '
          f'min {Ps.min():.2f}  p1 {np.percentile(Ps,1):.2f}  median {np.median(Ps):.1f}')
    print(f'  PRICED monthly default prob: mean {pdf.mean():.4%}  max {pdf.max():.4%}')
    print(f'  implied annual default rate: {1-(1-pdf.mean())**12:.3%}')
    print(f'  REALISED defaults in the panel: {int((Ps<=0).sum())}  ({(Ps<=0).mean():.4%})')
    print(f'\n  if m WERE drawn, expected defaults over this panel: '
          f'{pdf.sum():,.0f} firm-months of {len(Ps):,}')
