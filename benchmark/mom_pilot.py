"""32-operator / 8-RHS CUDA resource pilot, independent of model training.

This records algebra and cost on a requested mesh. It does not certify the
mesh as a physical reference or silently equate low residual with low error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig, build_rhs
from witwin.maxwell.mom.linalg import solve_reference
from witwin.maxwell.mom.fields import far_field


def run_pilot(*, subdivisions=1, order=4, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")
    report={"operators_requested":32,"rhs_per_operator":8,"physical_reference_validated":False,
            "scope":"setup and multi-RHS algebra/resource pilot; mesh accuracy is separate",
            "subdivisions":subdivisions,"order":order,"torch":torch.__version__,
            "python":platform.python_version(),"device":torch.cuda.get_device_name(),
            "revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("witwin/maxwell/mom").glob("*.py")},
            "cases":[]}
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    for index in range(32):
        radius=.065+.005*(index%4)
        frequency=2.5e8+2.5e7*(index//8)
        eps=1.5+.25*(index%8)
        sigma=.002*(index%3)
        geometry=(mw.Sphere(radius=radius) if index%2==0 else mw.Ellipsoid(radii=(radius,.85*radius,.7*radius)))
        scene=mw.Scene(structures=[mw.Structure(geometry,mw.Material(eps_r=eps,sigma_e=sigma))],sources=[mw.PlaneWave()])
        config=mw.MoMConfig(subdivisions=subdivisions,integration=IntegrationConfig(order=order,singular_order=order))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        prepared=mw.Simulation.mom(scene,frequency=frequency,config=config).prepare()
        handle=prepared.solver.handle
        b=[]
        for source_index in range(8):
            angle=2*math.pi*source_index/8
            direction=(math.sin(angle),0.,math.cos(angle))
            polarization=(0.,1.,0.)
            source=mw.PlaneWave(direction=direction,polarization=polarization,
                                source_time=mw.CW(frequency=frequency,amplitude=1.,phase=source_index*.17))
            b.append(build_rhs(handle.basis,handle.exterior,[source],order=order))
        rhs=torch.stack(b)
        solution,solve_log=solve_reference(handle,rhs)
        check=rhs-solution@handle.matrix.T
        residuals=torch.linalg.vector_norm(check,dim=-1)/torch.linalg.vector_norm(rhs,dim=-1)
        torch.cuda.synchronize(); matvec_start=time.perf_counter()
        for _ in range(10):
            handle.apply(solution)
            handle.adjoint(solution)
        torch.cuda.synchronize(); action_seconds=(time.perf_counter()-matvec_start)/20
        directions=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],device="cuda",dtype=torch.float64)
        decode_start=time.perf_counter()
        far_field(handle,solution[0],directions)
        torch.cuda.synchronize()
        case={"operator_index":index,"geometry":geometry.kind,"radius_m":radius,"frequency_hz":frequency,
              "epsilon_r":eps,"sigma_e_S_per_m":sigma,"edges":handle.basis.n_edges,
              "bare_dense_bytes":64*handle.basis.n_edges**2,
              "assembly":handle.metadata,"solve":solve_log,"maximum_true_relative_residual":float(residuals.max()),
              "batched_action_seconds":action_seconds,"decode_three_directions_seconds":time.perf_counter()-decode_start,
              "peak_cuda_bytes":torch.cuda.max_memory_allocated(),"total_seconds":time.perf_counter()-start}
        report["cases"].append(case)
        output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"operator":index,"edges":handle.basis.n_edges,"converged":solve_log["converged"],"seconds":case["total_seconds"]}),flush=True)
        del prepared,handle,rhs,solution,b,check
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--subdivisions",type=int,default=1)
    parser.add_argument("--order",type=int,default=4)
    parser.add_argument("--output",type=Path,default=Path("benchmark/cache/mom/pilot.json"))
    args=parser.parse_args(argv)
    run_pilot(subdivisions=args.subdivisions,order=args.order,output=args.output)


if __name__=="__main__":
    main()
