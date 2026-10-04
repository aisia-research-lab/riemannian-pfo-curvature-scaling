import numpy as np
from hyperbolic_same_policy import *

def main():
    k=1.; p=4.; rho=.04
    x=x_from_uy(.08,Cp_flat(p),1,rho,p)
    X=klein_to_hyper(x,k); x2=hyper_to_klein(X,k)
    assert np.linalg.norm(x-x2)<1e-12
    O=np.array([1.,0.,0.]); assert abs(objective(x,k)-.5*hdist_X(X,O,k)**2)<1e-13
    v1,*rest1=lmo(x,p,rho,k,256); v2,*rest2=lmo(x,p,rho,k,1024)
    # compare LMO objective values (index 1 after v/V in unpacked rest is phi_val)
    _,_,phi1,_,_,_=lmo(x,p,rho,k,256); _,_,phi2,_,_,_=lmo(x,p,rho,k,1024)
    assert abs(phi1-phi2) < 1e-9
    Y=klein_to_hyper(v2,k)
    for eta in [0,.2,.7,1]:
        Z=geodesic_X(X,Y,eta)
        assert abs(-ldot(Z,Z)-1)<1e-10
        assert abs(hdist_X(X,Z,k)-eta*hdist_X(X,Y,k))<1e-9
    print('Hyperbolic geometry/oracle self-test: PASS')
if __name__=='__main__': main()
