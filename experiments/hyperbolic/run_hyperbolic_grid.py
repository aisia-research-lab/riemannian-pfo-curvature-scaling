#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import pandas as pd
from hyperbolic_same_policy import RunCfg, run, save_run

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',default='results_hyperbolic'); ap.add_argument('--max-iter',type=int,default=20000); ap.add_argument('--lmo-grid',type=int,default=1024); a=ap.parse_args()
    # Frozen primary grid. p=3 is diagnostic only and must not be pooled with theorem-validation runs.
    primary=[]
    for p in [4,5,6,8]:
      for rho in [0.04,0.08]:
       for kappa in [0.5,1.0,2.0]:
        for ym in [0.9,1.0,1.1]:
         primary.append(RunCfg(p=p,rho=rho,kappa=kappa,u0=0.08,y0_mult=ym,max_iter=a.max_iter,lmo_grid=a.lmo_grid))
    diagnostic=[RunCfg(p=3,rho=.04,kappa=1,u0=.08,y0_mult=ym,max_iter=a.max_iter,lmo_grid=a.lmo_grid) for ym in [.9,1,1.1]]
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); summaries=[]
    for kind,cfgs in [('primary',primary),('diagnostic_p3',diagnostic)]:
      for i,cfg in enumerate(cfgs):
        tag=f'{kind}_{i:03d}_p{cfg.p:g}_rho{cfg.rho:g}_k{cfg.kappa:g}_ym{cfg.y0_mult:g}'
        try:
          df,s=run(cfg); s['case_class']=kind; s['tag']=tag; save_run(df,s,out,tag); summaries.append(s)
          print(tag,'fit_h=',s['fit_h_slope'],'pred=',s['pred_h_slope'])
        except Exception as e:
          summaries.append({'case_class':kind,'tag':tag,'p':cfg.p,'rho':cfg.rho,'kappa':cfg.kappa,'y0_mult':cfg.y0_mult,'error':repr(e)})
          print(tag,'ERROR',repr(e))
    pd.DataFrame(summaries).to_csv(out/'hyperbolic_summary.csv',index=False)
    with open(out/'hyperbolic_manifest.json','w') as f: json.dump(summaries,f,indent=2)
if __name__=='__main__': main()
