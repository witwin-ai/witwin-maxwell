"""Bounded one-memory ordering search; not a nonexistence proof."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from model import Block, Cell, DT, oscillatory_spectrum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    grid=(-1.,0.,.5,1.,1.5,2.,3.,4.,8.)
    rows=[]
    blocks=[Block(Cell(polarization=pol,dt=DT*f)) for pol,f in product(("TE","TM"),(1.,.5,.25))]
    for fe,fh in product(grid,repeat=2):
        worst=0.
        cases=[]
        for b in blocks:
            A=b.step(np.eye(b.size),(fe,fh))
            idx=b.active_indices("current")
            result=oscillatory_spectrum(A[np.ix_(idx,idx)],b.cell.dt)
            cases.append(dict(pol=b.cell.polarization,dt=b.cell.dt,**result))
            worst=max(worst,result["rho"])
        rows.append(dict(electric_fraction=fe,magnetic_fraction=fh,worst_rho=worst,
                         passed=bool(worst<=1+1e-12),cases=cases))
    report=dict(scope="fixed native b/c/kappa, output=(1-f)*psi_old+f*psi_new; periodic q1, TE/TM, dt/dt2/dt4",
                searched=len(rows),passing=sum(r["passed"] for r in rows),
                best=min(rows,key=lambda r:r["worst_rho"]),rows=rows)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2),flush=True)
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
