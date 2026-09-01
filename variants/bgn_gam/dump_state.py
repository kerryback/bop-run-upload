"""Run create_arrays/create_panel/sdf_compute in THIS directory's model and dump comparable state.
usage: python dump_state.py <out.npz> <N> <T> <t1> <t2>"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.seterr(all="ignore")
out, N, T, t1, t2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
import panel_functions as pf, sdf_compute as sc
np.random.seed(7)
arr = pf.create_arrays(N, T)
pan = pf.create_panel(N, T, arr)
loop = sc.sdf_compute(N, T, arr)
o1, o2 = loop(t1), loop(t2)
r, eret, P = arr[0], arr[7], arr[9]
np.savez_compressed(out, r=r, eret=eret, P=P,
                    mve=pan.mve.values, xret=pan.xret.values, bm=pan.bm.fillna(-999).values,
                    mu1=o1[2], Sig1=o1[3], sr1=o1[1], mu2=o2[2], Sig2=o2[3], sr2=o2[1])
print("dumped", out)
