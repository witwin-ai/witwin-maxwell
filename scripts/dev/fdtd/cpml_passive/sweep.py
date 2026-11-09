"""Cartesian float64 CPU spectral sweep. Failed gates are retained, never clipped.

This command exits 1 when no candidate meets G1; that is a numerical result,
not an execution error. JSONL keeps every case available for independent audit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from model import Block, Cell, DT, oscillatory_spectrum

METHODS=("current","c1","c2","c3_old","c3_average","passive_control")


def batch(geometry):
    wall,layers,refinement=geometry
    dy=.0125 if refinement=="y_half" else .025
    dz=.0125 if refinement=="z_half" else .025
    base=Cell(wall=wall,layers=layers,dy=dy,dz=dz,theta=35.)
    rows=[]
    for pol,q,factor in product(("TE","TM"),(1,2,3),(1.,.5,.25)):
        previous=None
        for preset in ("real","cfs","alpha0"):
            cell=replace(base,polarization=pol,order=q,dt=DT*factor,preset=preset)
            b=Block(cell)
            if preset=="alpha0":
                results=previous
            else:
                results={}
                for method in METHODS:
                    A=b.matrix(method)
                    idx=b.active_indices(method)
                    A=A[np.ix_(idx,idx)]
                    result=oscillatory_spectrum(A,cell.dt)
                    result["passes_oscillatory_gate"]=bool(result["rho"]<=1+1e-12)
                    results[method]=result
                if preset=="real":
                    previous=results
            rows.append(dict(cell=asdict(cell),refinement=refinement,results=results))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"results")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--quick",action="store_true",help="only baseline geometry; not a full sweep")
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    geometry=list(product(("periodic","pec","pmc"),(6,12,24),("base","y_half","z_half")))
    if args.quick:
        geometry=[("periodic",12,"base")]
    summary={m:dict(failed=0,worst_rho=0.,worst_case=None) for m in METHODS}
    count=0
    start=time.monotonic()
    with (args.output_dir/"sweep.jsonl").open("w") as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for group,rows in zip(geometry,pool.map(batch,geometry)):
            for row in rows:
                out.write(json.dumps(row)+"\n")
                for m,r in row["results"].items():
                    summary[m]["failed"]+=not r["passes_oscillatory_gate"]
                    if r["rho"]>summary[m]["worst_rho"]:
                        summary[m].update(worst_rho=r["rho"],worst_case=row["cell"],frequency_GHz=r["frequency_GHz"])
                count+=1
            out.flush()
            print(json.dumps(dict(geometry=group,complete_cases=count,seconds=time.monotonic()-start)),flush=True)
    report=dict(complete=not args.quick,cases=count,methods=summary,threshold=1+1e-12,
        cfs_profile_theta_degrees=35.,seconds=time.monotonic()-start,
        selected=[m for m in METHODS if m not in ("current","passive_control") and summary[m]["failed"]==0])
    (args.output_dir/"sweep_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if report["selected"] else 1)


if __name__=="__main__":
    main()
