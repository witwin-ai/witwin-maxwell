"""Long source-free native Yee modified-energy control for true PEC surfaces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.cut_surface import apply_cut_surface_constraints
from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, update_magnetic_fields


def run(kind, steps):
    source_hashes = {name:hashlib.sha256(Path(mw.__file__).parent.joinpath(name).read_bytes()).hexdigest()
        for name in ("compiler/cut_surface.py","fdtd/cut_surface.py","geometry/surface_mesh.py","fdtd/runtime/stepping.py")}
    if kind == "bent":
        geometry = mw.SurfaceMesh.bent_rectangle(width=.02, length=.016, radius=.03,
                                                 segments=12, device="cuda")
    else:
        vertices = torch.tensor(((-.008,-.008,0.),(.008,-.008,0.),(.008,.008,0.),(-.008,.008,0.)),
                                device="cuda", dtype=torch.float64)
        vertices[:,2] = .3*vertices[:,0]+.001
        geometry = mw.SurfaceMesh(vertices, ((0,1,2),(0,2,3)))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.04,.04),)*3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(), device="cuda",
        structures=(mw.Structure(geometry=geometry, material=mw.Material.pec()),))
    solver = mw.Simulation.fdtd(scene, frequency=1.e9, run_time=mw.TimeConfig(time_steps=steps, dt=5.e-12)).prepare().solver
    generator = torch.Generator(device="cuda").manual_seed(33)
    electric_weight, magnetic_weight = {}, {}
    for axis, name in enumerate(("Ex","Ey","Ez")):
        value = getattr(solver,name)
        value[3:-3,3:-3,3:-3] = torch.randn(value[3:-3,3:-3,3:-3].shape,device="cuda",generator=generator)
        widths = [torch.as_tensor(getattr(solver.scene, coordinate+("_primal64" if i == axis else "_dual64")),
                                  device="cuda") for i,coordinate in enumerate(("dx","dy","dz"))]
        electric_weight[name] = getattr(solver,"eps_"+name).double()*widths[0][:,None,None]*widths[1][None,:,None]*widths[2][None,None,:]
    for axis,name in enumerate(("Hx","Hy","Hz")):
        widths = [torch.as_tensor(getattr(solver.scene, coordinate+("_dual64" if i == axis else "_primal64")),
                                  device="cuda") for i,coordinate in enumerate(("dx","dy","dz"))]
        magnetic_weight[name] = getattr(solver,"mu_"+name).double()*widths[0][:,None,None]*widths[1][None,:,None]*widths[2][None,None,:]
    apply_cut_surface_constraints(solver)

    def energy():
        previous = {name:getattr(solver,name).clone() for name in magnetic_weight}
        update_magnetic_fields(solver,solver.Hx,solver.Hy,solver.Hz,solver.Ex,solver.Ey,solver.Ez)
        value = sum((weight*getattr(solver,name).double().square()).sum() for name,weight in electric_weight.items())
        value += sum((weight*previous[name].double()*getattr(solver,name).double()).sum() for name,weight in magnetic_weight.items())
        for name,old in previous.items():
            getattr(solver,name).copy_(old)
        return float(value/2)

    def packet():
        for _ in range(64):
            _field_update_block(solver,0.)
            apply_cut_surface_constraints(solver)

    initial_fields = {name:getattr(solver,name).clone() for name in (*electric_weight,*magnetic_weight)}
    replay = CudaGraphRunner(device=solver.device,enabled=True,warmup_steps=1).capture(packet)
    for name,value in initial_fields.items():
        getattr(solver,name).copy_(value)
    initial = energy()
    points, completed = [], 0
    start = time.perf_counter()
    while completed+64 <= steps:
        replay()
        completed += 64
        if completed % 64000 == 0:
            current = energy()
            points.append(dict(step=completed,energy_j=current,relative_drift=abs(current-initial)/initial))
            print(json.dumps(dict(kind=kind,**points[-1])),flush=True)
    while completed < steps:
        _field_update_block(solver,0.)
        apply_cut_surface_constraints(solver)
        completed += 1
    current = energy()
    points.append(dict(step=completed,energy_j=current,relative_drift=abs(current-initial)/initial))
    return dict(kind=kind,steps=steps,dt_s=solver.dt,initial_energy_j=initial,
        source_hashes=source_hashes,
        maximum_relative_drift=max(point["relative_drift"] for point in points),
        elapsed_seconds=time.perf_counter()-start,checkpoints=points,
        all_finite=all(bool(torch.isfinite(getattr(solver,name)).all()) for name in initial_fields))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=1000000)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=[]
    for kind in ("oblique","bent"):
        report.append(run(kind,args.steps))
        args.output.write_text(json.dumps(report,indent=2)+"\n")
