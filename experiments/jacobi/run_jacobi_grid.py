import argparse, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jacobi_distortion import evaluate_case

KR=[0.05,0.10,0.25,0.50,1.0,2.0,3.0,4.0]
KAPPAS=[0.5,1.0,2.0]
ANGLES=[0,15,30,45,60,75,90]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir',default='results_jacobi')
    ap.add_argument('--tol-rel',type=float,default=2e-5)
    args=ap.parse_args(); os.makedirs(args.outdir,exist_ok=True)
    rows=[]
    for k in KAPPAS:
        for s in KR:
            r=s/k
            for a in ANGLES:
                q=evaluate_case(k,r,a); rows.append(q)
                print(f"k={k:g} kr={s:g} theta={a:2d} smin={q['sigma_min_num']:.10g} pred={q['rauch_exact']:.10g} direrr={q['err_dir_rel']:.2e}")
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir,'jacobi_all_cases.csv'),index=False)
    # one row per (k,kr) for singular-value audit
    sv=df[df.theta_deg==0].copy()
    sv.to_csv(os.path.join(args.outdir,'jacobi_singular_summary.csv'),index=False)
    summary={
      'n_cases':len(df),
      'max_smin_rel_error':df.err_smin_rel.max(),
      'max_directional_rel_error':df.err_dir_rel.max(),
      'min_certificate_slack':df.certificate_slack.min(),
      'tolerance_rel':args.tol_rel,
      'pass_smin':bool(df.err_smin_rel.max()<=args.tol_rel),
      'pass_directional':bool(df.err_dir_rel.max()<=args.tol_rel),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir,'jacobi_audit_summary.csv'),index=False)
    print('\nSUMMARY',summary)

    # Figure 1: exact space-form Rauch factor; curves for k collapse versus kr.
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for k in KAPPAS:
        z=sv[sv.kappa==k]
        ax.plot(z.kr,z.sigma_min_num,'o-',label=f'numerical k={k:g}')
    ss=np.linspace(0.001,4,400)
    ax.plot(ss,ss/np.sinh(ss),'--',label='exact/Rauch  s/sinh(s)')
    ax.set_xlabel('dimensionless radius  kappa r'); ax.set_ylabel('sigma_min(D Log)')
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(args.outdir,'jacobi_rauch_certificate.pdf')); plt.close(fig)

    # Figure 2: directional distortion at k=1 for several radii
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for s in [0.5,1.,2.,4.]:
        z=df[(df.kappa==1.0)&(np.isclose(df.kr,s))]
        ax.plot(z.theta_deg,z.directional_num,'o-',label=f'kappa r={s:g}')
    ax.set_xlabel('angle of w from radial direction (degrees)'); ax.set_ylabel('||DLog* w|| / ||w||')
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(args.outdir,'jacobi_directional_distortion.pdf')); plt.close(fig)

if __name__=='__main__': main()
