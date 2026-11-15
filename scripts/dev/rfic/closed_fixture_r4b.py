"""Acquire closed-conductor physics and timing evidence outside pytest."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
import witwin.maxwell as mw
from tests.support.rfic_fixtures import fixture_fa,SWEEP
from tests.support.rfic_closed_references import strip_fixture,network_metrics


def simulation(args,frequencies):
    if args.case=="strip":
        scene,config=strip_fixture(model=args.model,maximum_um=args.maximum,edge_um=args.edge,
            thickness_um=args.thickness,lossy=not args.pec)
    else:
        fixture=fixture_fa(substrate=args.substrate,lossy=not args.pec,mesh="test",
                           top_thickness_um=args.thickness)
        scene=fixture.scene
        if args.thin_all:
            rebuilt=mw.Scene(background=scene.background)
            for structure in scene.structures:
                geometry=structure.geometry
                if getattr(geometry,"kind",None)=="planar_pec":
                    geometry=mw.geometry.PlanarPEC(geometry.vertices,holes=geometry.holes,normal_axis="z",
                        coordinate=geometry.coordinate,physical_thickness=args.thickness*1e-6)
                rebuilt = rebuilt.updated(structures=(*rebuilt.structures, mw.Structure(geometry,structure.material)))
            rebuilt = rebuilt.updated(ports=(*rebuilt.ports, *scene.ports))
            scene=rebuilt
        config=replace(fixture.config,layered=replace(fixture.config.layered,
            conductor_model=args.model,max_edge_length_m=args.maximum*1e-6,edge_cell_m=args.edge*1e-6))
    return mw.Simulation.mom(scene,frequencies=frequencies,config=config)


def acquire(args):
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frequencies=SWEEP if args.sweep else tuple(args.frequencies)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    sim=simulation(args,frequencies)
    prepared=sim.prepare()
    torch.cuda.synchronize()
    prepare_seconds=time.perf_counter()-start
    print("prepared",prepared.solver.prepared.metadata["unknowns"],flush=True)
    first=None
    for repeat in range(args.repeat):
        if repeat:
            # PreparedSimulation is single-use. Repeat complete public scene
            # construction and preparation, rather than bypassing that contract.
            del result,prepared
            gc.collect();torch.cuda.empty_cache()
            sim=simulation(args,frequencies)
            prepared=sim.prepare()
        start=time.perf_counter()
        result=prepared.run()
        torch.cuda.synchronize()
        solve_seconds=time.perf_counter()-start
        if first is None:
            first=(result.network.s.clone(),mw.mom.surface_currents(result).currents.clone())
        else:
            assert torch.equal(first[0],result.network.s)
            assert torch.equal(first[1],mw.mom.surface_currents(result).currents)
    y=result.network.to_y().detach().cpu()
    reciprocity=((y-y.transpose(-1,-2)).abs().max()/y.abs().max()).item()
    singular=torch.linalg.matrix_norm(result.network.s,ord=2).max().item()
    data=network_metrics(result)
    data.update(case=args.case,model=args.model,substrate=args.substrate,pec=args.pec,
        maximum_um=args.maximum,edge_um=args.edge,thickness_um=args.thickness,thin_all=args.thin_all,
        y_real=y.real.tolist(),y_imag=y.imag.tolist(),metadata=dict(result.network.metadata),
        prepare_seconds=prepare_seconds,solve_seconds=solve_seconds,exact_repeats=args.repeat,
        reciprocity=reciprocity,max_singular_S=singular,peak_bytes=torch.cuda.max_memory_allocated(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("witwin/maxwell/mom/layered").rglob("*")
            if p.suffix in (".py",".cu",".cuh",".cpp")})
    if args.case=="strip" and prepared.solver.prepared.loss is not None:
        # Analysis oracle only: split port resistance into surface Joule loss
        # and the radiating vacuum circuit remainder, using actual RWG currents.
        native=prepared.solver.prepared
        basis=native.layout.basis
        coefficients=mw.mom.surface_currents(result).currents[:,:,0]
        port_current=coefficients @ native.excitation[:,0]
        local=coefficients[:,basis.face_edges]/port_current[:,None,None]
        gram=basis.mass_blocks().to(torch.complex128)
        norm=torch.einsum("fti,tij,ftj->ft",local.conj(),gram,local).real
        conductor=(norm*native.loss.impedance.real).sum(-1).cpu()
        data.update(conductor_R_ohm=conductor.tolist(),
                    radiation_R_ohm=(torch.tensor(data["R_ohm"],dtype=torch.float64)-conductor).tolist())
    # SRF is Im(Zin)'s first negative-to-positive zero, linearly bracketed.
    zin=1/y[:,0,0]
    srf=None
    for i in range(len(frequencies)-1):
        if zin.imag[i]<0<=zin.imag[i+1]:
            fraction=(-zin.imag[i]/(zin.imag[i+1]-zin.imag[i])).item()
            srf=frequencies[i]+fraction*(frequencies[i+1]-frequencies[i])
            break
    peak=max(range(len(data["Q"])),key=lambda i:data["Q"][i])
    data.update(SRF_Hz=srf,Q_peak=data["Q"][peak],Q_peak_frequency_Hz=frequencies[peak])
    args.output.write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k not in ("metadata","source_sha256","y_real","y_imag")}),flush=True)


def timing(args):
    from witwin.maxwell.mom.layered.lowfreq import assemble_chunk,transform_chunk
    from witwin.maxwell.mom.layered.network import (
        _wall_nodes,solve_scaled_chunk,recover_scaled_chunk,frequency_chunk_capacity,available_work_memory)
    frequencies=tuple(args.frequencies)
    prepared=simulation(args,frequencies).prepare().solver.prepared
    capacity=frequency_chunk_capacity(prepared.layout.basis.n_edges,len(prepared.layout.mesh.faces),
        prepared.excitation.shape[1],len(frequencies),available_work_memory(prepared.excitation.device),
        wall_nodes=_wall_nodes(prepared.table))
    values=[]
    for _ in range(3):
        totals=dict(assembly_seconds=0.,transform_seconds=0.,factor_solve_seconds=0.,recovery_seconds=0.)
        for first in range(0,len(frequencies),capacity):
            stop=min(first+capacity,len(frequencies))
            torch.cuda.synchronize();start=time.perf_counter()
            system=assemble_chunk(prepared,frequency_start=first,frequency_stop=stop)
            torch.cuda.synchronize();assembled=time.perf_counter()
            transform_chunk(prepared,system,frequency_start=first)
            torch.cuda.synchronize();transformed=time.perf_counter()
            solution,_=solve_scaled_chunk(system)
            torch.cuda.synchronize();solved=time.perf_counter()
            y,current=recover_scaled_chunk(prepared,solution,system.rhs,system.recovery,frequency_start=first)
            torch.cuda.synchronize();end=time.perf_counter()
            for key,value in zip(totals,(assembled-start,transformed-assembled,solved-transformed,end-solved)):
                totals[key]+=value/len(frequencies)
            del system,solution,y,current
        totals["total_seconds"]=sum(totals.values())
        values.append(totals)
        gc.collect();torch.cuda.empty_cache()
    args.output.write_text(json.dumps(dict(unknowns=prepared.metadata["unknowns"],case=args.case,
        model=args.model,substrate=args.substrate,maximum_um=args.maximum,edge_um=args.edge,frequencies_Hz=frequencies,
        frequency_chunk_size=capacity,units="seconds_per_frequency",trials=values),indent=2))
    print(args.output,values,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--case",choices=("strip","fa"),default="fa")
    p.add_argument("--model",choices=("sheet","closed"),default="closed")
    p.add_argument("--maximum",type=float,default=20.)
    p.add_argument("--edge",type=float,default=5.)
    p.add_argument("--thickness",type=float,default=2.)
    p.add_argument("--thin-all",action="store_true")
    p.add_argument("--pec",action="store_true")
    p.add_argument("--substrate",action="store_true")
    p.add_argument("--sweep",action="store_true")
    p.add_argument("--frequencies",type=float,nargs="+",default=(1e8,1e9))
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--timing",action="store_true")
    a=p.parse_args()
    timing(a) if a.timing else acquire(a)
