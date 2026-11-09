"""Acquire F-A vacuum controls and isolated shunt mesh convergence on the public path."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
import witwin.maxwell as mw
from tests.support.rfic_fixtures import fixture_fa
from tests.support.rfic_r6b_references import plate_fixture,FREQUENCIES

def record(directory,name,simulation,repeat=1):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    result=simulation.run()
    y=result.network.to_y().detach().cpu()
    s=result.network.s.detach().clone()
    currents=result.surface_currents.currents.detach().clone()
    for _ in range(repeat-1):
        other=simulation.run()
        assert torch.equal(s,other.network.s)
        assert torch.equal(currents,other.surface_currents.currents)
        del other
    torch.cuda.synchronize()
    series=-1/y[:,1,0] if y.shape[-1]==2 else 1/y[:,0,0]
    data=dict(frequency_Hz=list(simulation.frequencies),y_real=y.real.tolist(),y_imag=y.imag.tolist(),
        L_H=(-series.imag/(2*torch.pi*torch.tensor(simulation.frequencies,dtype=torch.float64))).tolist(),
        series_R_ohm=series.real.tolist(),summed_shunt_G_S=y.sum((-1,-2)).real.tolist(),
        unknowns=result.network.metadata["unknowns"],faces=result.network.metadata["faces"],
        seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),
        exact_repeats=repeat,metadata=dict(result.network.metadata),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path("witwin/maxwell/mom/layered").rglob("*")
                       if p.suffix in (".py",".cu",".cuh",".cpp")})
    if name.startswith("vacuum"):
        from witwin.maxwell.mom.layered.runtime import compile_layered_scene
        _,rectangles,_,materials,_,_,_=compile_layered_scene(simulation.scene)
        layout=result.surface_currents.layout
        centres=layout.mesh.centroids
        thickness=torch.zeros(len(centres),device=centres.device,dtype=torch.float64)
        for (x0,x1,y0,y1,z),(_,t,_) in zip(rectangles,materials):
            owned=((centres[:,0]>x0)&(centres[:,0]<x1)&(centres[:,1]>y0)
                   &(centres[:,1]<y1)&(centres[:,2]==z))
            thickness[owned]=t
        edges=layout.basis.face_edges
        local=result.surface_currents.currents[:,edges.clamp_min(0),:]
        local=local*(edges>=0)[None,:,:,None]
        blocks=(layout.basis.mass_blocks()*thickness[:,None,None]).to(torch.complex128)
        bilinear=torch.einsum("ftia,tij,ftjb->fab",local,blocks,local)
        omega=2*torch.pi*result.network.frequencies
        delta_y=-(1j*omega[:,None,None]*(4e-7*torch.pi)/4)*bilinear
        actual_y=result.network.to_y()
        delta_z=delta_y[:,1,0]/actual_y[:,1,0]**2
        data["leading_two_face_magnetic_thickness_dL_H"]=(-delta_z.imag/omega).cpu().tolist()
        data["leading_two_face_magnetic_thickness_dR_ohm"]=delta_z.real.cpu().tolist()
        data["thickness_diagnostic"]="First-order equal-face magnetic kernel expansion, not a runtime correction"
    (directory/(name+".json")).write_text(json.dumps(data,indent=2))
    print(name,json.dumps({k:v for k,v in data.items() if k not in
        ("y_real","y_imag","metadata","source_sha256")}),flush=True)
    del result,y,s,currents
    gc.collect()
    torch.cuda.empty_cache()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--group",choices=("all","vacuum","substrate","plate"),default="all")
    args=p.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    for lossy in ((False,True) if args.group in ("all","vacuum") else ()):
        f=fixture_fa(substrate=False,lossy=lossy,mesh="full")
        record(args.directory,f"vacuum-lossy{int(lossy)}-full",f.simulation((.1e9,1e9)))
        del f
    if args.group in ("all","vacuum"):
        f=fixture_fa(substrate=False,lossy=True,mesh="full")
        scene=mw.Scene(background=f.scene.background,device="cuda")
        for structure in f.scene.structures:
            geometry=structure.geometry
            if getattr(geometry,"kind",None)=="planar_pec":
                geometry=mw.PlanarPEC(geometry.vertices,holes=geometry.holes,normal_axis="z",
                    coordinate=geometry.coordinate,physical_thickness=.1e-6)
            scene.add_structure(mw.Structure(geometry,structure.material))
        for port in f.scene.ports:
            scene.add_port(port)
        record(args.directory,"vacuum-thin-lossy1-full",
               mw.Simulation.mom(scene,frequencies=(.1e9,1e9),config=f.config))
        del f,scene
    for mesh in (("shield","test","full") if args.group in ("all","substrate") else ()):
        f=fixture_fa(lossy=False,mesh=mesh)
        record(args.directory,f"substrate-pec-{mesh}",f.simulation((1e9,3.4e9)))
        del f
    for maximum,edge in (((40.,10.),(20.,5.),(10.,2.5),(5.,1.25)) if args.group in ("all","plate") else ()):
        scene,config=plate_fixture(maximum,edge)
        record(args.directory,f"plate-{maximum:g}-{edge:g}",
               mw.Simulation.mom(scene,frequencies=FREQUENCIES,config=config),repeat=3 if maximum==40 else 1)
        del scene,config
