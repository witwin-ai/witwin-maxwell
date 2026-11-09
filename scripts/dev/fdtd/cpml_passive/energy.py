"""Exact leapfrog work balance and storage counterexamples, CPU float64."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from model import Block, Cell


def work_balance(block, x):
    """Complex state represents two orthogonal real harmonic phases."""
    u,v,w,pe,ph = block.split(x)
    y=block.step(x)
    un,vn,wn,pen,phn=block.split(y)
    m=sum(block.lengths[:3])
    P=block.yee_metric()
    W0=.5*np.vdot(x[:m],P@x[:m]).real
    W1=.5*np.vdot(y[:m],P@y[:m]).real
    if block.cell.polarization=="TM":
        qh=block.t*((1/block.kh-1)*(block.F@u)+phn/block.h)
        qe=block.t*block.a*((1/block.ke-1)*(block.B@wn)+pen/block.h)
        magnetic=.5*np.real((wn+w).conj()*qh)
        electric=.5*np.real((un+u).conj()*qe/block.a)
    else:
        qh=-block.t*((1/block.kh-1)*(block.F@u)+phn/block.h)
        qe=-block.t*block.a*((1/block.ke-1)*(block.B@vn)+pen/block.h)
        magnetic=.5*np.real((vn+v).conj()*qh)
        electric=.5*np.real((un+u).conj()*qe/block.a)
    work=float(sum(electric)+sum(magnetic))
    return dict(storage_before=float(W0),relative_increment=float((W1-W0)/W0),
        electric_relative_work=float(sum(electric)/W0),
        magnetic_relative_work=float(sum(magnetic)/W0),
        relative_residual=float((W1-W0-work)/W0),
        peak_positive_magnetic_z_m=float(((block.cell.nodes[:-1]+block.cell.nodes[1:])/2)[np.argmax(magnetic)]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    b=Block(Cell())
    vals,vecs=linalg.eig(b.matrix(),check_finite=False)
    i=np.argmax(abs(vals))
    mode=work_balance(b,vecs[:,i])
    rng=np.random.default_rng(431)
    active=b.active_indices("current")
    residuals=[]
    for _ in range(32):
        x=np.zeros(b.size)
        x[active]=rng.normal(size=len(active))
        residuals.append(abs(work_balance(b,x)["relative_residual"]))
    candidates={}
    for c in ("current","c1","c2","c3_old","c3_average","passive_control"):
        A=b.matrix(c)
        active=b.active_indices(c)
        A=A[np.ix_(active,active)]
        values,vectors=linalg.eig(A,check_finite=False)
        j=np.argmax(abs(values))
        v=vectors[:,j]
        P=np.eye(len(A))
        P[:len(b.field_metric()),:len(b.field_metric())]=b.field_metric()
        if c=="passive_control":
            P=b.field_metric()
        Q=A.T@P@A-P
        random=rng.normal(size=(len(A),256))
        maxrandom=float(max(np.sum(random*(Q@random),axis=0)/np.sum(random*(P@random),axis=0)))
        candidates[c]=dict(rho=float(abs(values[j])),
            eigenmode_relative_increment=float(np.vdot(v,Q@v).real/np.vdot(v,P@v).real),
            max_random_relative_increment=maxrandom,
            max_increment_eigenvalue=float(linalg.eigvalsh(Q,subset_by_index=(len(A)-1,len(A)-1))[0]),
            storage="physical field norm plus unit-weight auxiliary norm" if c!="passive_control" else "physical field norm",
            certified=bool(c=="passive_control" and linalg.eigvalsh(Q)[-1]<=1e-12),
            no_coercive_storage=bool(abs(values[j])>1+1e-12))
    report=dict(grazing_TM=mode,max_random_work_balance_residual=max(residuals),candidates=candidates)
    print(json.dumps(report,indent=2),flush=True)
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    if max(residuals)>1e-12:
        raise SystemExit("Work identity failed")


if __name__=="__main__":
    main()
