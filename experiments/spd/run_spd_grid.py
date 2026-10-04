import argparse, os, csv, numpy as np
from spd_spectral import *

def main():
 p=argparse.ArgumentParser(); p.add_argument('--outdir',default='results_spd'); p.add_argument('--eps',type=float,default=2e-6); a=p.parse_args(); os.makedirs(a.outdir,exist_ok=True)
 dims=[2,3,5,10]; spreads=[0.,0.25,0.5,1.,2.,4.,6.]; shifts=[0.,2.5]; rows=[]; cid=0
 for n in dims:
  for s in spreads:
   for shift in shifts:
    X,Y,_=construct_pair(n,s,shift,seed=1000+cid,general_X=True); M=numerical_dlog_matrix(X,Y,a.eps); sv=np.linalg.svd(M,compute_uv=False); ex=exact_root_singulars(X,Y); lam=relative_spectrum(X,Y); q=lam[-1]/lam[0]; pred=q_factor(q); d=dist_airm(X,Y)
    rows.append(dict(case=f'primary_{cid:03d}',n=n,log_spread=s,common_shift=shift,q=q,distance=d,sigma_min_num=sv[-1],sigma_min_formula=pred,relerr_min=abs(sv[-1]-pred)/pred,max_spectrum_relerr=np.max(np.abs(np.sort(sv)[::-1]-ex)/np.maximum(ex,1e-300)),sigma_max_num=sv[0])); cid+=1
 # isotropic diagnostic: q=1, arbitrarily increasing distance
 for n in dims:
  for shift in [-5.,-2.,2.,5.]:
   X,Y,_=construct_pair(n,0.,shift,seed=5000+cid,general_X=True); M=numerical_dlog_matrix(X,Y,a.eps); sv=np.linalg.svd(M,compute_uv=False); q=np.linalg.cond(invsqrtm_spd(X)@Y@invsqrtm_spd(X)); rows.append(dict(case=f'isotropic_{cid:03d}',n=n,log_spread=0.,common_shift=shift,q=q,distance=dist_airm(X,Y),sigma_min_num=sv[-1],sigma_min_formula=1.,relerr_min=abs(sv[-1]-1),max_spectrum_relerr=np.max(abs(sv-1)),sigma_max_num=sv[0])); cid+=1
 path=os.path.join(a.outdir,'spd_all_cases.csv');
 with open(path,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 prim=[r for r in rows if r['case'].startswith('primary')]; iso=[r for r in rows if r['case'].startswith('isotropic')]
 summary={'n_primary':len(prim),'n_isotropic':len(iso),'max_relerr_sigma_min':max(r['relerr_min'] for r in prim),'max_full_spectrum_relerr':max(r['max_spectrum_relerr'] for r in prim),'max_isotropic_sigma_error':max(r['relerr_min'] for r in iso),'max_isotropic_distance':max(r['distance'] for r in iso)}
 with open(os.path.join(a.outdir,'spd_audit_summary.csv'),'w',newline='') as f: w=csv.writer(f); w.writerow(['metric','value']); w.writerows(summary.items())
 for k,v in summary.items(): print(k,'=',v)
 print('PASS prereg tolerance?', summary['max_relerr_sigma_min']<=5e-5 and summary['max_full_spectrum_relerr']<=1e-4 and summary['max_isotropic_sigma_error']<=1e-4)
if __name__=='__main__': main()
