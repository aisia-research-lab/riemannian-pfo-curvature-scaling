"""Numerical DLog distortion on H^2_{-kappa^2}.
Independent finite-difference implementation of D Log_x at v in the hyperboloid model.
The closed-form Rauch/space-form factor is used only for comparison, never to compute DLog_num.
"""
from __future__ import annotations
import math
import numpy as np


def mdot(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return -a[0]*b[0] + float(np.dot(a[1:],b[1:]))


def origin(kappa):
    return np.array([1.0/kappa,0.0,0.0])


def norm_tan(u):
    val=mdot(u,u)
    return math.sqrt(max(val,0.0))


def project_tangent(x,u,kappa):
    # <x,x>_L=-1/k^2; remove normal component
    return np.asarray(u,float) + (kappa*kappa)*mdot(x,u)*np.asarray(x,float)


def exp_map(x,u,kappa):
    u=project_tangent(x,u,kappa)
    n=norm_tan(u)
    if n < 1e-14:
        return np.asarray(x,float).copy()
    s=kappa*n
    return math.cosh(s)*np.asarray(x,float) + (math.sinh(s)/(kappa*n))*u


def distance(x,y,kappa):
    z=-kappa*kappa*mdot(x,y)
    z=max(1.0,z)
    return math.acosh(z)/kappa


def log_map(x,y,kappa):
    x=np.asarray(x,float); y=np.asarray(y,float)
    z=-kappa*kappa*mdot(x,y)
    z=max(1.0,z)
    s=math.acosh(z)
    if s < 1e-10:
        # first-order stable branch
        return project_tangent(x,y-x,kappa)
    sh=math.sinh(s)
    return (s/sh)*(y-z*x)


def tangent_basis(x,kappa, preferred=None):
    """Minkowski-orthonormal basis of T_x H2."""
    candidates=[]
    if preferred is not None:
        candidates.append(np.asarray(preferred,float))
    candidates += [np.array([0.,1.,0.]), np.array([0.,0.,1.]), np.array([1.,0.,0.])]
    basis=[]
    for c in candidates:
        z=project_tangent(x,c,kappa)
        for q in basis:
            z=z-mdot(z,q)*q
        n=norm_tan(z)
        if n>1e-10:
            basis.append(z/n)
        if len(basis)==2: break
    if len(basis)!=2: raise RuntimeError('failed tangent basis')
    return np.stack(basis,axis=1) # ambient 3 x 2


def coords(B,u):
    return np.array([mdot(B[:,j],u) for j in range(B.shape[1])])


def point_at_distance(kappa,r):
    x=origin(kappa)
    e=np.array([0.,1.,0.])
    v=exp_map(x,r*e,kappa)
    return x,v,e


def radial_basis_at_v(x,v,kappa):
    # radial direction at v pointing away from x = - normalized Log_v(x)
    rv=-log_map(v,x,kappa)
    rv=rv/norm_tan(rv)
    Bv=tangent_basis(v,kappa,preferred=rv)
    # force first basis orientation radial-away
    if mdot(Bv[:,0],rv)<0: Bv[:,0]*=-1
    return Bv


def radial_basis_at_x(x,v,kappa):
    rx=log_map(x,v,kappa); rx=rx/norm_tan(rx)
    Bx=tangent_basis(x,kappa,preferred=rx)
    if mdot(Bx[:,0],rx)<0: Bx[:,0]*=-1
    return Bx


def numerical_dlog_matrix(x,v,kappa,eps=None):
    """Matrix of (D Log_x)_v between orthonormal tangent bases.
    Central differences perturb v intrinsically using Exp_v(±eps z).
    """
    r=distance(x,v,kappa)
    if eps is None:
        # intrinsic length; enough above roundoff and tiny relative to geometric scale
        eps=max(2e-7, 2e-6*max(1.0,r))
    Bx=radial_basis_at_x(x,v,kappa)
    Bv=radial_basis_at_v(x,v,kappa)
    M=np.zeros((2,2))
    for j in range(2):
        z=Bv[:,j]
        vp=exp_map(v, eps*z, kappa)
        vm=exp_map(v,-eps*z, kappa)
        lp=log_map(x,vp,kappa)
        lm=log_map(x,vm,kappa)
        d=(lp-lm)/(2*eps)
        M[:,j]=coords(Bx,d)
    return M,Bx,Bv,eps


def phi(kappa,r):
    s=kappa*r
    if abs(s)<1e-6:
        return 1-s*s/6+7*s**4/360
    return s/math.sinh(s)


def analytic_directional(kappa,r,theta):
    ph=phi(kappa,r)
    return math.sqrt(math.cos(theta)**2 + (ph*math.sin(theta))**2)


def evaluate_case(kappa,r,theta_deg,eps=None):
    x,v,_=point_at_distance(kappa,r)
    M,_,_,used_eps=numerical_dlog_matrix(x,v,kappa,eps)
    sv=np.linalg.svd(M,compute_uv=False)
    smin=float(np.min(sv)); smax=float(np.max(sv))
    th=math.radians(theta_deg)
    w=np.array([math.cos(th),math.sin(th)]) # x orthonormal basis
    directional=float(np.linalg.norm(M.T@w))
    pred_dir=analytic_directional(kappa,r,th)
    cert=phi(kappa,r)
    return {
        'kappa':kappa,'r':r,'kr':kappa*r,'theta_deg':theta_deg,'eps':used_eps,
        'sigma_min_num':smin,'sigma_max_num':smax,'rauch_exact':cert,
        'directional_num':directional,'directional_exact':pred_dir,
        'err_smin_abs':abs(smin-cert),
        'err_smin_rel':abs(smin-cert)/max(abs(cert),1e-15),
        'err_dir_abs':abs(directional-pred_dir),
        'err_dir_rel':abs(directional-pred_dir)/max(abs(pred_dir),1e-15),
        'certificate_slack':smin-cert,
        'M00':M[0,0],'M01':M[0,1],'M10':M[1,0],'M11':M[1,1],
    }


def self_test():
    for k in [0.5,1.,2.]:
        x,v,_=point_at_distance(k,0.7/k)
        assert abs(distance(x,v,k)-0.7/k)<1e-10
        M,_,_,_=numerical_dlog_matrix(x,v,k)
        target=np.array([1.0,phi(k,0.7/k)])
        got=np.sort(np.linalg.svd(M,compute_uv=False))[::-1]
        # expected singulars [1,phi]
        assert np.max(np.abs(got-target))<2e-6, (k,M,got,target)
    return True
