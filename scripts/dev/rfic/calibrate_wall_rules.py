"""CPU entry-error calibration of P1 rules for sheet-wall and wall-wall pairs.

Uses triangle-relative coordinates, aspect ratios 1/.2/.05, twelve separation
directions, both source orientations and the fixed 8x8 reference triangle rule.
Reports max absolute error in the nine vector moments normalized by their
maximum reference magnitude, and the relative scalar entry error separately.
The independent static kernel is 1/R; homogeneous vector coupling is isotropic.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from witwin.maxwell.mom.layered.assembly import _symmetric_rule


def reference_rule():
    u,w=leggauss(8)
    u,w=(u+1)/2,w/2
    return np.array([(1-a,a*(1-b),a*b,wa*wb*a)
                     for a,wa in zip(u,w) for b,wb in zip(u,w)])


def moments(t,s,rule):
    # Barycentric offsets preserve tiny triangles at extreme separation.
    a=rule[:,1,None]*(t[1]-t[0])+rule[:,2,None]*(t[2]-t[0])
    b=rule[:,1,None]*(s[1]-s[0])+rule[:,2,None]*(s[2]-s[0])
    delta=(t[0]-s[0])+a[:,None]-b[None]
    kernel=1/np.linalg.norm(delta,axis=-1)
    weight=rule[:,3,None]*rule[None,:,3]*kernel
    v=np.einsum("ij,iad,jbd->ab",weight,a[:,None]-(t-t[0])[None],
                b[:,None]-(s-s[0])[None])
    return np.r_[v.ravel(),weight.sum()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=[]
    reference=reference_rule()
    for pair in ("sheet_wall","wall_wall","wall_wall_parallel"):
        for nodes,eta in ((25,2.),(13,3.),(12,4.),(7,8.),(6,24.),(3,1024.),(1,1e8)):
            rule=_symmetric_rule(nodes,"cpu").numpy()
            errors=np.zeros(2)
            for aspect in (1.,.2,.05):
                t=np.array(((0.,0.,0.),(1.,0.,0.),(0.,aspect,0.)))
                if pair.startswith("wall_wall"):
                    t=t[:,(2,0,1)]
                for angle in np.linspace(0,2*math.pi,13)[:-1]:
                    for flip in (-1,1):
                        axes=(2,0,1) if pair=="wall_wall_parallel" else (0,2,1)
                        s=np.array(((0.,0.,0.),(flip,0.,0.),(0.,aspect,0.)))[:,axes]
                        size=max(np.linalg.norm(t-t[[1,2,0]],axis=-1).max(),
                                 np.linalg.norm(s-s[[1,2,0]],axis=-1).max())
                        s-=s.mean(0)
                        s+=t.mean(0)+eta*size*np.array((math.cos(angle),0.,math.sin(angle)))
                        ref,new=moments(t,s,reference),moments(t,s,rule)
                        errors=np.maximum(errors,(np.max(abs(new[:9]-ref[:9]))/np.max(abs(ref[:9])),
                                                  abs(new[9]-ref[9])/abs(ref[9])))
            row=dict(pair=pair,points=nodes,minimum_eta=eta,vector_error=errors[0],scalar_error=errors[1])
            out.append(row)
            print(row,flush=True)
    args.output.write_text(json.dumps(out,indent=2))
    assert max(max(row["vector_error"],row["scalar_error"]) for row in out)<1e-6


if __name__=="__main__":
    main()
