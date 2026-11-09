"""Supplementary fixed-k harmonic CPU check at the plan's six carrier angles."""
import argparse
from dataclasses import asdict,replace
from itertools import product
import json
from pathlib import Path

import numpy as np

from model import Block,Cell,C0,DT,oscillatory_spectrum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    methods=("current","c1","c2","c3_old","c3_average")
    rows=[]
    for fmin,angle,pol,q,factor in product((.3e9,2e9),(0.,10.,30.,45.,60.,75.),("TE","TM"),(0,1,2,3),(1.,.5,.25)):
        cell=Cell(polarization=pol,order=q,dt=DT*factor,preset="cfs",theta=angle,f_min=fmin)
        ky=2*np.pi*q/cell.period+2*np.pi*fmin/C0*np.sin(np.deg2rad(angle))
        hy=cell.period/cell.ny
        b=Block(cell,wavenumber=2/hy*np.sin(ky*hy/2))
        results={}
        for method in methods:
            A=b.matrix(method)
            idx=b.active_indices(method)
            results[method]=oscillatory_spectrum(A[np.ix_(idx,idx)],cell.dt)
        rows.append(dict(cell=asdict(cell),physical_ky=ky,results=results))
    summary={m:dict(failed=sum(r["results"][m]["rho"]>1+1e-12 for r in rows),
        worst_rho=max(r["results"][m]["rho"] for r in rows)) for m in methods}
    report=dict(cases=len(rows),summary=summary,rows=rows,
        scope="12 layers, base grid, fixed ky=Bloch carrier plus Floquet q; projected transverse-mean slab")
    print(json.dumps(summary,indent=2),flush=True)
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
