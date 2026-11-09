"""Matrix-free 4x-period Fourier-stack power check, all resolved y harmonics.

The slab coefficients are the repository's projected transverse means. This is
a larger harmonic-model check, not a native full-y closure qualification.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from model import Block, Cell


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",type=int,default=100000)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    base=Cell(period=4*1.28)
    results=[]
    start=time.monotonic()
    for pol in ("TE","TM"):
        b=Block(replace(base,polarization=pol))
        orders=np.arange(base.ny//2+1)
        # Each column is an independent Fourier harmonic of the same larger cell.
        b.k=2*np.sin(np.pi*orders/base.ny)/(base.period/base.ny)
        for method,ordering in (("current","fresh"),("c1","trapezoid")):
            x=np.random.default_rng(431).normal(size=(b.size,len(orders)))
            x[b.offsets[3]:]=0
            x/=np.linalg.norm(x)
            total=0.
            checkpoints=[]
            for step in range(1,args.steps+1):
                x=b.step(x,ordering)
                if step%100==0:
                    norm=np.linalg.norm(x)
                    total+=np.log(norm)
                    x/=norm
                    if step>=args.steps//2:
                        checkpoints.append((step,total))
                if step%20000==0:
                    print(json.dumps(dict(pol=pol,method=method,step=step,seconds=time.monotonic()-start)),flush=True)
            fit=np.polyfit(np.array(checkpoints)[:,0],np.array(checkpoints)[:,1],1)
            dominant=int(orders[np.argmax(np.linalg.norm(x,axis=0))])
            results.append(dict(polarization=pol,method=method,steps=args.steps,period_m=base.period,
                ny=base.ny,harmonics=len(orders),dominant_order=dominant,
                fitted_rho=float(np.exp(fit[0])),passes=bool(fit[0]<=1e-12)))
    report=dict(results=results,seconds=time.monotonic()-start,
        limitation="Fourier-stack model uses transverse-mean material coefficients; no claim of full-y native invariance")
    print(json.dumps(report,indent=2),flush=True)
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
