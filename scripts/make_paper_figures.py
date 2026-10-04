#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced_figures'; OUT.mkdir(exist_ok=True)

# Hyperbolic representative trajectories.
fig,ax=plt.subplots(figsize=(6.6,4.4))
for p,idx in [(4,4),(5,22),(6,40),(8,58)]:
    path=ROOT/'reference_results'/'hyperbolic'/f'primary_{idx:03d}_p{p}_rho0.04_k1_ym1_trajectory.csv'
    df=pd.read_csv(path); z=df[(df.t>=1)&(df.h>0)]
    slope=-p/(p-1)
    ax.loglog(z.t,z.h,label=fr'$p={p}$ (pred. {slope:.3f})')
ax.set_xlabel('iteration $t$'); ax.set_ylabel('objective gap')
ax.legend(); fig.tight_layout(); fig.savefig(OUT/'hyperbolic_same_policy_rates.pdf'); plt.close(fig)

# Jacobi: directional curves and exact transverse factor.
df=pd.read_csv(ROOT/'reference_results'/'jacobi'/'jacobi_all_cases.csv')
fig,ax=plt.subplots(figsize=(6.6,4.4))
for s in [0.5,1.,2.,4.]:
    z=df[(df.kappa==1.0)&np.isclose(df.kr,s)]
    ax.plot(z.theta_deg,z.directional_num,'o-',label=fr'$\kappa r={s:g}$')
ax.set_xlabel('angle from radial direction (degrees)'); ax.set_ylabel(r'$\|D\mathrm{Log}^*w\|/\|w\|$')
ax.legend(); fig.tight_layout(); fig.savefig(OUT/'jacobi_distortion_certificate.pdf'); plt.close(fig)

# SPD: collapse versus relative log-spectral spread.
df=pd.read_csv(ROOT/'reference_results'/'spd'/'spd_all_cases.csv')
z=df[df['case'].str.startswith('primary')]
fig,ax=plt.subplots(figsize=(6.6,4.4))
for n in sorted(z.n.unique()):
    q=z[z.n==n]
    ax.plot(q.log_spread,q.sigma_min_num,'o',label=fr'$n={int(n)}$')
s=np.linspace(0,6,400); q=np.exp(s); exact=np.ones_like(s); m=s>0; exact[m]=np.sqrt(q[m])*np.log(q[m])/(q[m]-1)
ax.plot(s,exact,'--',label='exact certificate')
ax.set_xlabel('relative log-spectral spread'); ax.set_ylabel(r'$\sigma_{\min}(D\mathrm{Log})$')
ax.legend(); fig.tight_layout(); fig.savefig(OUT/'spd_spectral_certificate.pdf'); plt.close(fig)
print('Wrote figures to',OUT)
