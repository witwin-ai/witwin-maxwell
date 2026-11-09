"""Independent algebra checks for the CPU operators and reflection elimination."""
import json

import numpy as np
from scipy import linalg

from model import Block,Cell
from reflection import transfer


def main():
    rows=[]
    worst=0.
    for pol in ("TE","TM"):
        b=Block(Cell(polarization=pol),material="vacuum")
        P=b.field_metric()
        generator=b.bare_generator()
        skew=float(np.linalg.norm(P@generator+generator.T@P)/np.linalg.norm(generator))
        worst=max(worst,skew)
        # Disable stretching on this analysis object to check the independently
        # derived staggered storage against the actual H-then-E step.
        b.ke[:]=b.kh[:]=1.
        b.ce[:]=b.ch[:]=0.
        field_size=len(P)
        A=b.matrix()[:field_size,:field_size]
        Y=b.yee_metric()
        yee=float(np.linalg.norm(A.T@Y@A-Y)/np.linalg.norm(Y))
        worst=max(worst,yee)
        rows.append(dict(polarization=pol,skew_adjoint_relative_error=skew,yee_storage_relative_error=yee))
        for preset in ("real","cfs"):
            b=Block(Cell(polarization=pol,preset=preset,theta=35.),material="vacuum")
            for method in ("current","c1","c2","c3_old","c3_average"):
                A=b.matrix(method)
                values,vectors=linalg.eig(A,check_finite=False)
                osc=(abs(np.angle(values))>2*np.pi*b.cell.dt*1e6)&(np.linalg.norm(vectors[:b.n-1],axis=0)>.01)
                i=np.argmax(np.where(osc,abs(values),0))
                lam=values[i]
                u=vectors[:b.n-1,i]
                ge=transfer(b.profile[0],lam,b.cell.dt,method)[1:-1]
                gh=transfer(b.profile[1],lam,b.cell.dt,method)
                K2=-(lam-1)**2/(b.t*b.t*lam)-b.k*b.k
                L=ge[:,None]*(b.B@(gh[:,None]*b.F))+K2*np.eye(b.n-1)
                error=float(np.linalg.norm(L@u)/(np.linalg.norm(L)*np.linalg.norm(u)))
                worst=max(worst,error)
                rows.append(dict(polarization=pol,preset=preset,method=method,wave_elimination_relative_error=error))
    print(json.dumps(dict(max_relative_error=worst,rows=rows),indent=2))
    raise SystemExit(0 if worst<1e-11 else 1)


if __name__=="__main__":
    main()
