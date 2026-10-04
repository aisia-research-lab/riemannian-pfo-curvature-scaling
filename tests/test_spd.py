import numpy as np
from spd_spectral import *
for n in [2,3,5]:
 X,Y,_=construct_pair(n,2.,1.3,seed=42+n); M=numerical_dlog_matrix(X,Y); sv=np.linalg.svd(M,compute_uv=False); ex=exact_root_singulars(X,Y)
 assert np.max(np.abs(np.sort(sv)[::-1]-ex)/ex)<2e-5
 assert abs(dist_airm(X,X))<1e-10
print('SPD SELF-TEST: PASS')
