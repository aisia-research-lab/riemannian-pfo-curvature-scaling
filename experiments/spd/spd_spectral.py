import numpy as np
from scipy.linalg import eigh


def sym(A): return 0.5*(A+A.T)

def spd_fun(A, f):
    w,Q=eigh(sym(A)); return (Q*f(w))@Q.T

def sqrtm_spd(A): return spd_fun(A,np.sqrt)
def invsqrtm_spd(A): return spd_fun(A,lambda x:1/np.sqrt(x))
def logm_spd(A): return spd_fun(A,np.log)
def expm_sym(A):
    w,Q=eigh(sym(A)); return (Q*np.exp(w))@Q.T

def airm_inner(X,U,V):
    Xi=invsqrtm_spd(X); return float(np.sum((Xi@U@Xi)*(Xi@V@Xi)))
def airm_norm(X,U): return np.sqrt(max(0.0,airm_inner(X,U,U)))

def exp_airm(X,U):
    S=sqrtm_spd(X); Si=invsqrtm_spd(X)
    return sym(S@expm_sym(Si@U@Si)@S)

def log_airm(X,Y):
    S=sqrtm_spd(X); Si=invsqrtm_spd(X)
    return sym(S@logm_spd(Si@Y@Si)@S)

def dist_airm(X,Y):
    Si=invsqrtm_spd(X); A=logm_spd(Si@Y@Si); return float(np.linalg.norm(A,'fro'))

def sym_frob_basis(n):
    B=[]
    for i in range(n):
        E=np.zeros((n,n)); E[i,i]=1.; B.append(E)
    for i in range(n):
        for j in range(i+1,n):
            E=np.zeros((n,n)); E[i,j]=E[j,i]=1/np.sqrt(2); B.append(E)
    return B

def airm_orthonormal_basis(X):
    S=sqrtm_spd(X)
    return [sym(S@E@S) for E in sym_frob_basis(X.shape[0])]

def numerical_dlog_matrix(X,Y,eps=2e-6):
    # Columns: D Log_X at Y applied to an AIRM-ON basis at Y;
    # rows: AIRM-ON coordinates at X. Intrinsic central difference via Exp_Y.
    BX=airm_orthonormal_basis(X); BY=airm_orthonormal_basis(Y)
    m=len(BX); M=np.empty((m,m))
    for j,H in enumerate(BY):
        Yp=exp_airm(Y, eps*H); Ym=exp_airm(Y,-eps*H)
        Z=(log_airm(X,Yp)-log_airm(X,Ym))/(2*eps)
        M[:,j]=[airm_inner(X,E,Z) for E in BX]
    return M

def relative_spectrum(X,Y):
    Si=invsqrtm_spd(X); R=Si@Y@Si
    lam=np.linalg.eigvalsh(sym(R)); return lam

def exact_root_singulars(X,Y):
    a=np.log(relative_spectrum(X,Y)); vals=[1.0]*len(a)
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            s=abs(a[i]-a[j])/2
            vals.append(1.0 if s<1e-14 else s/np.sinh(s))
    return np.sort(np.array(vals))[::-1]

def q_factor(q):
    if abs(q-1)<1e-10: return 1.0
    return np.sqrt(q)*np.log(q)/(q-1)

def random_orthogonal(n,rng):
    Q,R=np.linalg.qr(rng.normal(size=(n,n))); Q=Q@np.diag(np.sign(np.diag(R))); return Q

def construct_pair(n,log_spread,common_shift=0.,seed=0,general_X=True):
    rng=np.random.default_rng(seed)
    # prescribed log-relative spectrum with exact max-min spread
    if n==1: a=np.array([common_shift])
    else: a=common_shift+np.linspace(-log_spread/2,log_spread/2,n)
    Q=random_orthogonal(n,rng); R=Q@np.diag(np.exp(a))@Q.T
    if general_X:
        P=random_orthogonal(n,rng); lx=np.linspace(-0.7,0.6,n)+0.08*rng.normal(size=n)
        X=P@np.diag(np.exp(lx))@P.T
    else: X=np.eye(n)
    S=sqrtm_spd(X); Y=sym(S@R@S)
    return X,Y,a
