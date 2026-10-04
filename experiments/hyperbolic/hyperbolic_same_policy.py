#!/usr/bin/env python3
"""Reference implementation for the hyperbolic same-policy hard family.

Implements Riemannian Frank--Wolfe on the Klein disk model of H^2_{-kappa^2}
for C_{p,rho}=(-rho,0)+rho*B_p and f(x)=1/2 d_kappa(x,0)^2.
The LMO is solved numerically on the full superellipse boundary, and the update
uses the exact hyperbolic geodesic.  No asymptotic formulas from the proof are
used in the iteration itself.
"""
from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

EPS = 1e-15

def ldot(a,b):
    return -a[0]*b[0] + float(np.dot(a[1:], b[1:]))

def klein_to_hyper(x,kappa):
    x=np.asarray(x,dtype=float); q=(kappa*kappa)*float(np.dot(x,x))
    if q >= 1.0: raise ValueError(f"Point outside Klein disk: kappa^2||x||^2={q}")
    gam=1.0/math.sqrt(max(1.0-q, EPS))
    return np.r_[gam, gam*kappa*x]

def hyper_to_klein(X,kappa):
    return np.asarray(X[1:],dtype=float)/(kappa*float(X[0]))

def hdist_X(X,Y,kappa):
    c=max(1.0, -ldot(X,Y))
    return math.acosh(c)/kappa

def hlog_X(X,Y):
    c=max(1.0, -ldot(X,Y)); th=math.acosh(c)
    if th < 1e-12:
        # first-order tangent projection
        Z=Y-c*X
        return Z
    return (th/math.sinh(th))*(Y-c*X)

def hinner(U,V,kappa):
    return ldot(U,V)/(kappa*kappa)

def geodesic_X(X,Y,eta):
    c=max(1.0,-ldot(X,Y)); th=math.acosh(c)
    if th < 1e-12: return X.copy()
    den=math.sinh(th)
    Z=(math.sinh((1-eta)*th)/den)*X + (math.sinh(eta*th)/den)*Y
    # numerical renormalization to Lorentz hyperboloid
    n=-ldot(Z,Z)
    return Z/math.sqrt(n)

def boundary_phi(phi,p,rho):
    c=math.cos(phi); s=math.sin(phi)
    z1=math.copysign(abs(c)**(2.0/p),c) if abs(c)>0 else 0.0
    z2=math.copysign(abs(s)**(2.0/p),s) if abs(s)>0 else 0.0
    return rho*np.array([z1-1.0,z2])

def Cp_flat(p):
    return (p/(p+1.0))**((p-1.0)/p)

def x_from_uy(u,y,sigma,rho,p):
    alpha=(p-1.0)/p
    return rho*np.array([-u, sigma*y*u**(1.0+alpha)])

def uy_from_x(x,rho,p):
    u=max(-float(x[0])/rho, 0.0)
    if u <= 0: return u, np.nan, 0
    alpha=(p-1.0)/p
    sig=1 if x[1]>=0 else -1
    y=abs(float(x[1]))/(rho*u**(1.0+alpha))
    return u,y,sig

def objective(x,kappa):
    X=klein_to_hyper(x,kappa); O=np.array([1.0,0.0,0.0])
    d=hdist_X(X,O,kappa); return 0.5*d*d

def grad_and_X(x,kappa):
    X=klein_to_hyper(x,kappa); O=np.array([1.0,0.0,0.0])
    grad=-hlog_X(X,O)  # representation valid under constant metric rescaling
    return grad,X

def lmo(x,p,rho,kappa,grid=1024):
    grad,X=grad_and_X(x,kappa)
    def val(phi):
        v=boundary_phi(phi,p,rho); V=klein_to_hyper(v,kappa)
        return hinner(grad,hlog_X(X,V),kappa)
    phis=np.linspace(-math.pi, math.pi, grid, endpoint=False)
    vals=np.array([val(q) for q in phis])
    # refine several best grid cells to reduce risk of selecting a local branch
    idx=np.argsort(vals)[:min(8,grid)]
    best=(float(vals[idx[0]]),float(phis[idx[0]]))
    h=2*math.pi/grid
    for j in idx:
        center=float(phis[j])
        # unwrap periodic objective for local bounded optimization
        f=lambda q: val(((q+math.pi)%(2*math.pi))-math.pi)
        res=minimize_scalar(f,bounds=(center-h,center+h),method='bounded',options={'xatol':1e-13})
        if res.fun < best[0]: best=(float(res.fun),float(((res.x+math.pi)%(2*math.pi))-math.pi))
    v=boundary_phi(best[1],p,rho); V=klein_to_hyper(v,kappa)
    return v,V,best[0],best[1],grad,X

def max_radius_C(p,rho,kappa):
    # maximize Euclidean Klein radius on the full boundary; hyperbolic radius is monotone in it.
    f=lambda phi: -float(np.dot(boundary_phi(phi,p,rho),boundary_phi(phi,p,rho)))
    grid=4096; phis=np.linspace(-math.pi,math.pi,grid,endpoint=False)
    vals=np.array([f(q) for q in phis]); j=int(np.argmin(vals)); h=2*math.pi/grid
    res=minimize_scalar(lambda q:f(((q+math.pi)%(2*math.pi))-math.pi),bounds=(phis[j]-h,phis[j]+h),method='bounded',options={'xatol':1e-14})
    rE=math.sqrt(max(0.0,-float(res.fun)))
    if kappa*rE >= 1: raise ValueError("Feasible set is not compactly contained in Klein disk")
    return math.atanh(kappa*rE)/kappa

def smoothness_L(p,rho,kappa):
    rmax=max_radius_C(p,rho,kappa); z=kappa*rmax
    if z < 1e-8: return 1.0 + z*z/3.0
    return max(1.0,z/math.tanh(z))

@dataclass
class RunCfg:
    p: float=4.0; rho: float=0.08; kappa: float=1.0; u0: float=0.08
    y0_mult: float=1.0; sigma0: int=1; max_iter: int=20000; lmo_grid: int=1024
    tail_fraction: float=0.35; min_tail: int=500

def run(cfg:RunCfg):
    if cfg.p < 2: raise ValueError('p must be >=2')
    if cfg.kappa<=0 or cfg.rho<=0: raise ValueError('kappa,rho must be positive')
    cp=Cp_flat(cfg.p); y0=cfg.y0_mult*cp
    x=x_from_uy(cfg.u0,y0,cfg.sigma0,cfg.rho,cfg.p)
    # membership and disk sanity
    z=np.array([x[0]/cfg.rho+1.0,x[1]/cfg.rho])
    if np.sum(np.abs(z)**cfg.p)>1+1e-10: raise ValueError('Initialization is outside C_{p,rho}')
    L=smoothness_L(cfg.p,cfg.rho,cfg.kappa)
    rows=[]
    for t in range(cfg.max_iter+1):
        h=objective(x,cfg.kappa); u,y,sig=uy_from_x(x,cfg.rho,cfg.p)
        if t==cfg.max_iter or h < 1e-30:
            rows.append(dict(t=t,x1=x[0],x2=x[1],u=u,y=y,sigma=sig,h=h,g=np.nan,r=np.nan,eta=np.nan,lmo_phi=np.nan,cap_active=False))
            break
        v,V,phi_val,phi,grad,X=lmo(x,cfg.p,cfg.rho,cfg.kappa,cfg.lmo_grid)
        g=-phi_val; r=hdist_X(X,V,cfg.kappa)
        eta=min(max(g/(L*r*r),0.0),1.0) if r>0 else 0.0
        rows.append(dict(t=t,x1=x[0],x2=x[1],u=u,y=y,sigma=sig,h=h,g=g,r=r,eta=eta,lmo_phi=phi,cap_active=eta>=1-1e-14))
        Xn=geodesic_X(X,V,eta); x=hyper_to_klein(Xn,cfg.kappa)
    df=pd.DataFrame(rows)
    # preregistered tail: final max(min_tail, tail_fraction*n) positive finite points; no visual selection.
    good=df[(df.t>=1)&np.isfinite(df.h)&(df.h>0)&np.isfinite(df.u)&(df.u>0)].copy()
    n=len(good); nt=min(n,max(cfg.min_tail,int(math.ceil(cfg.tail_fraction*n))))
    tail=good.iloc[-nt:].copy() if nt>=2 else good
    def slope(col):
        if len(tail)<2:return np.nan
        return float(np.polyfit(np.log(tail.t.to_numpy()),np.log(tail[col].to_numpy()),1)[0])
    hs=slope('h'); us=slope('u')
    pred_h=-cfg.p/(cfg.p-1.0); pred_u=-cfg.p/(2*(cfg.p-1.0))
    norm_h=(tail.t.to_numpy()**(-pred_h))*tail.h.to_numpy() if len(tail) else np.array([])
    norm_u=(tail.t.to_numpy()**(-pred_u))*tail.u.to_numpy() if len(tail) else np.array([])
    def spread(a):
        if len(a)<2:return np.nan
        med=float(np.median(a)); return float((np.quantile(a,.9)-np.quantile(a,.1))/abs(med)) if med!=0 else np.inf
    summ=dict(**asdict(cfg),Cp=cp,L=L,n_steps=int(df.t.iloc[-1]),tail_n=int(len(tail)),pred_h_slope=pred_h,fit_h_slope=hs,h_slope_error=hs-pred_h,pred_u_slope=pred_u,fit_u_slope=us,u_slope_error=us-pred_u,norm_h_spread_10_90=spread(norm_h),norm_u_spread_10_90=spread(norm_u),tail_y_median=float(np.nanmedian(tail.y)) if len(tail) else np.nan,tail_y_rel_to_Cp=float(np.nanmedian(tail.y)/cp-1) if len(tail) else np.nan,cap_fraction=float(df.cap_active.mean()))
    return df,summ

def save_run(df,summ,outdir,tag):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/f'{tag}_trajectory.csv',index=False)
    with open(out/f'{tag}_summary.json','w') as f: json.dump(summ,f,indent=2)
    p=summ['p']; pred=summ['pred_h_slope']
    good=df[(df.t>=1)&(df.h>0)]
    fig,ax=plt.subplots(figsize=(6.4,4.5)); ax.loglog(good.t,good.h,label='objective gap')
    if len(good):
        j=max(0,len(good)//2); tt=good.t.to_numpy(); hh=good.h.to_numpy(); ref=hh[j]*(tt/tt[j])**pred; ax.loglog(tt,ref,'--',label=f'reference slope {pred:.4f}')
    ax.set_xlabel('iteration t'); ax.set_ylabel('h_t'); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{tag}_gap.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4.5)); ax.plot(good.t,good.y,label='y_t'); ax.axhline(summ['Cp'],ls='--',label='flat C_p'); ax.set_xlabel('iteration t'); ax.set_ylabel('normalized transverse state y_t'); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{tag}_y.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4.5)); norm=(good.t.to_numpy()**(p/(p-1.0)))*good.h.to_numpy(); ax.semilogx(good.t,norm); ax.set_xlabel('iteration t'); ax.set_ylabel(r'$t^{p/(p-1)} h_t$'); fig.tight_layout(); fig.savefig(out/f'{tag}_normalized_gap.pdf'); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--p',type=float,default=4); ap.add_argument('--rho',type=float,default=.08); ap.add_argument('--kappa',type=float,default=1); ap.add_argument('--u0',type=float,default=.08); ap.add_argument('--y0-mult',type=float,default=1); ap.add_argument('--sigma0',type=int,default=1); ap.add_argument('--max-iter',type=int,default=20000); ap.add_argument('--lmo-grid',type=int,default=1024); ap.add_argument('--outdir',default='results'); ap.add_argument('--tag',default=None)
    a=ap.parse_args(); cfg=RunCfg(p=a.p,rho=a.rho,kappa=a.kappa,u0=a.u0,y0_mult=a.y0_mult,sigma0=a.sigma0,max_iter=a.max_iter,lmo_grid=a.lmo_grid)
    df,s=run(cfg); tag=a.tag or f'p{a.p:g}_rho{a.rho:g}_k{a.kappa:g}_u{a.u0:g}_ym{a.y0_mult:g}'
    save_run(df,s,a.outdir,tag); print(json.dumps(s,indent=2))
if __name__=='__main__': main()
