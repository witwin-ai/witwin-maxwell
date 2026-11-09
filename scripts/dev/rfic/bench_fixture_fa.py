"""F-A stage timing and complete public-path evidence on the assigned GPU 1.

Invoke through codex/runners/gpu1_timing.py; each lock hold is bounded by
that runner. Green build includes admission, with vertical-span time measured
inside the parent build. Per-frequency timings exclude one-time preparation.
Logical traffic counts repeated table/rule loads, not physical DRAM traffic.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
os.environ["OMP_NUM_THREADS"]="8"
import numpy as np
import torch
from tests.support.rfic_fixtures import fixture_fa,SWEEP
from tests.support.rfic_fa_references import extract_fa,resonance_and_peak,greenhouse_fa,dc_resistance_fa,mohan_spiral,stranded_greenhouse_fa
from scripts.dev.rfic.bench_layered_network import timed,device_peaks,profile_kernels
from witwin.maxwell.mom.layered.green import LayeredGreenTable
from witwin.maxwell.mom.layered.vertical import VerticalGreenTable
from witwin.maxwell.mom.layered import lowfreq,network


def kernel_model(prepared, chunk_size):
    """P1 counts for every ordered face pair, including vertical contractions."""
    mesh,table,lt=prepared.layout.mesh,prepared.table,prepared.loop_tree
    centres=mesh.centroids.cpu().numpy()
    triangles=mesh.triangles.cpu().numpy()
    planes=prepared.layout.triangle_planes.cpu().numpy()
    size=((triangles-np.roll(triangles,1,axis=1))**2).sum(-1).max(-1)
    images=table.image_coefficients[:chunk_size].ne(0).any(0).cpu().numpy()
    distances=table.image_distances.cpu().numpy()
    offsets=table.image_offsets.cpu().numpy()
    vi=table.vertical.image_coefficients[0].cpu().numpy()
    vg=table.vertical.image_geometry.cpu().numpy()
    vk=table.vertical.image_kinds.cpu().numpy()
    thresholds=np.array([0,0,3,4,8,24,1024,1e8],dtype=float)**2
    counts=np.array([64,25,13,12,7,6,3,1])
    samples=dict(horizontal_remainder=0,horizontal_static=0,horizontal_analytic_nodes=0,
                 wall_remainder=0,wall_mixed_remainder=0,wall_ray=0,wall_analytic_nodes=0,
                 horizontal_remainder_fp64=0,horizontal_static_fp64=0,
                 horizontal_regular_images=0,horizontal_near_images=0,
                 horizontal_vector_static=0,horizontal_vector_analytic_nodes=0,
                 horizontal_vector_regular_images=0,horizontal_vector_near_images=0,
                 horizontal_scalar_regular_images=0,horizontal_scalar_near_images=0)
    faces=len(centres)
    for begin in range(0,faces,64):
        end=min(begin+64,faces)
        delta=centres[:,None,:]-centres[None,begin:end,:]
        lateral=(delta[...,:2]**2).sum(-1)
        scale=np.maximum(size[:,None],size[None,begin:end])
        wall=(planes[:,None]<0)|(planes[None,begin:end]<0)
        q2=counts[np.searchsorted(thresholds,(lateral+delta[...,2]**2)/scale,side="right")-1]**2
        samples["wall_remainder"]+=int(q2[wall].sum()) if not table.exact_mode else 0
        mixed=(planes[:,None]<0)^(planes[None,begin:end]<0)
        samples["wall_mixed_remainder"]+=int(q2[mixed].sum()) if not table.exact_mode else 0
        samples["horizontal_remainder"]+=int(q2[~wall].sum())
        near=(lateral+delta[...,2]**2)<4*scale
        samples["horizontal_remainder_fp64"]+=int(q2[~wall&near].sum())
        for a in range(len(table.planes)):
            for b in range(a,len(table.planes)):
                selected=(~wall)&(((planes[:,None]==a)&(planes[None,begin:end]==b))|
                                 ((planes[:,None]==b)&(planes[None,begin:end]==a)))
                pair=a*len(table.planes)-a*(a-1)//2+b-a
                for i in range(offsets[pair],offsets[pair+1]):
                    if not np.any(images[:,i]): continue
                    r2=(lateral+distances[i]**2)/scale
                    analytic=selected&(r2<4)
                    outer=np.select([r2<.5,r2<1,r2<2],[1024,576,256],default=64)
                    samples["horizontal_analytic_nodes"]+=int(outer[analytic].sum())
                    samples["horizontal_near_images"]+=int(analytic.sum())
                    regular=selected&~analytic
                    samples["horizontal_regular_images"]+=int(regular.sum())
                    q2=counts[np.searchsorted(thresholds,r2,side="right")-1]**2
                    samples["horizontal_static"]+=int(q2[regular].sum())
                    samples["horizontal_static_fp64"]+=int(q2[regular&near].sum())
                    if images[0,i]:
                        samples["horizontal_vector_static"]+=int(q2[regular].sum())
                        samples["horizontal_vector_analytic_nodes"]+=int(outer[analytic].sum())
                        samples["horizontal_vector_regular_images"]+=int(regular.sum())
                        samples["horizontal_vector_near_images"]+=int(analytic.sum())
                    if images[1,i]:
                        samples["horizontal_scalar_regular_images"]+=int(regular.sum())
                        samples["horizontal_scalar_near_images"]+=int(analytic.sum())
        # F-A has one vertical segment. Its reflected distances are affine.
        zs=np.minimum(centres[:,None,2],centres[None,begin:end,2])
        zo=np.maximum(centres[:,None,2],centres[None,begin:end,2])
        for i,(alpha,beta,offset) in enumerate(vg):
            if not np.any(vi[:,i]): continue
            distance=zo-zs if vk[i]==1 else np.maximum(0,alpha*zs+beta*zo+offset)
            r2=(lateral+distance**2)/scale
            q2=counts[np.searchsorted(thresholds,r2,side="right")-1]**2
            samples["wall_ray"]+=int(q2[wall].sum())
            analytic=wall&(r2<4)
            outer=np.select([r2<.5,r2<1,r2<2],[1024,576,256],default=64)
            samples["wall_analytic_nodes"]+=int(outer[analytic].sum())
    n,t=prepared.layout.basis.n_edges,lt.trees
    nnz,ncharge=lt.sparse_q[1].numel(),lt.sparse_charge[1].numel()
    # Loss adds a second V-sized congruence. Each sparse term is a real
    # coefficient times a complex value: 4 real FLOP, 16 B complex input.
    sparse_terms=4*n*nnz+(faces+t)*ncharge
    rem,static=samples["horizontal_remainder"],samples["horizontal_static"]
    wr,ws=samples["wall_remainder"],samples["wall_ray"]
    analytic=samples["horizontal_analytic_nodes"]+samples["wall_analytic_nodes"]
    k=table.vertical.z_nodes
    wall_components=5*wr-2*samples["wall_mixed_remainder"]
    return dict(counts=samples,definitions={
        "regular_face_pairs":{"logical_bytes":128*rem+128*(samples["horizontal_remainder_fp64"]*((chunk_size+1)//2)/chunk_size+(rem-samples["horizontal_remainder_fp64"])*((chunk_size+3)//4)/chunk_size)+320*(faces-int((planes<0).sum()))**2,
                              "real_FLOP":(270+40*((chunk_size+1)//2)/chunk_size)*samples["horizontal_remainder_fp64"]+(300+40*((chunk_size+3)//4)/chunk_size)*(rem-samples["horizontal_remainder_fp64"]),
                              "note":"Fp64 scalar and selected-precision vectors; rule/grid/coordinate setup shared per actual batch. Static-cache read and tile write included."},
        "wall_regular_face_pairs":{"logical_bytes":64*wr+64*k*k*wall_components+64*ws+160*(faces**2-(faces-int((planes<0).sum()))**2),
                                   "real_FLOP":600*wr+16*k*k*wall_components+600*ws,
                                   "note":"Three Hermite components for horizontal/wall pairs, five for wall/wall; SFU excluded."},
        "horizontal_static_regular_geometry":{
            "logical_bytes":(96*static+160*faces**2)/chunk_size+288*samples["horizontal_vector_regular_images"]+32*samples["horizontal_scalar_regular_images"],
            "real_FLOP":(50*static+90*samples["horizontal_vector_static"])/chunk_size+36*samples["horizontal_vector_regular_images"]+4*samples["horizontal_scalar_regular_images"],
            "note":"Real moments once/chunk; exact-zero vector images omit their nine moments/cache updates. Scalar-only samples and mixed precision are counted separately."},
        "horizontal_static_near_geometry":{
            "logical_bytes":32*samples["horizontal_analytic_nodes"]/chunk_size+288*samples["horizontal_vector_near_images"]+32*samples["horizontal_scalar_near_images"],
            "real_FLOP":(420*samples["horizontal_analytic_nodes"]+80*samples["horizontal_vector_analytic_nodes"])/chunk_size+36*samples["horizontal_vector_near_images"]+4*samples["horizontal_scalar_near_images"]},
        "near_image_pairs":{"logical_bytes":32*samples["wall_analytic_nodes"]+320*(faces**2-(faces-int((planes<0).sum()))**2),
                            "real_FLOP":590*samples["wall_analytic_nodes"],
                            "note":"Only wall analytic rays remain frequency-local; radial cross SFU is additional."},
        "scalar_points_kernel":{"logical_bytes":(72*faces+48*faces*8*65)/chunk_size,
                                "real_FLOP":8*faces*8*65/chunk_size,
                                "note":"Fp64 relative-coordinate cache once per chunk; 16 B output per rule point."},
        "radial_metadata_kernel":{"logical_bytes":40/chunk_size,"real_FLOP":2/chunk_size,
                                  "note":"Grid constants once per chunk; log excluded."},
        "wall_weights_kernel":{"logical_bytes":8*(2*k+2)*faces*8*65/chunk_size,
                               "real_FLOP":12*k*k*faces*8*65/chunk_size,
                               "note":"Geometry setup executes once per chunk."},
        "gather_potentials":{"logical_bytes":256*n*n+32*faces*faces,"real_FLOP":32*n*n},
        "sparse_product_kernel":{"logical_bytes":16*sparse_terms+32*(2*n*n+t*t),"real_FLOP":4*sparse_terms},
        "scaled_kernel":{"logical_bytes":128*n*n,"real_FLOP":70*n*n},
        "loss_kernel":{"logical_bytes":6*40*n+16*n*n,"real_FLOP":6*6*n},
        "port_work_kernel":{"logical_bytes":(8*n*n+16*n*2)/chunk_size,"real_FLOP":2*n*n*2/chunk_size,
                            "note":"Frequency-independent port setup executes once per chunk."},
        "unscale_kernel":{"logical_bytes":40*n*2,"real_FLOP":4*n*2},
        "admittance_kernel":{"logical_bytes":32*4,"real_FLOP":8*4},
        "refinement_check_kernel":{"logical_bytes":32*2,"real_FLOP":20*2},
        "refinement_update_kernel":{"logical_bytes":48*n*2,"real_FLOP":2*n*2},
        "LU_complex64":{"logical_bytes":8*n*n,"real_FLOP":8*n**3/3,
                        "note":"Factorization only; refinement GEMMs, norms and solves counted separately by profiler."}},
        nnz_transform=nnz,nnz_charge=ncharge,
        note="Analytical logical model, not measured DRAM or hardware instruction counts; wall cross SFU is unmodeled.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mesh",choices=("test","full"),default="full")
    parser.add_argument("--vacuum",action="store_true")
    parser.add_argument("--one-port",action="store_true")
    parser.add_argument("--lossless",action="store_true")
    parser.add_argument("--sigma",type=float,default=10.)
    parser.add_argument("--thickness",type=float,default=2.)
    parser.add_argument("--frequencies",type=float,nargs="+")
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--profile-kernels",action="store_true")
    args=parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1","Use GPU-1 timing runner"
    os.nice(10)
    torch.set_num_threads(8)
    peaks=device_peaks()
    data=dict(gpu=torch.cuda.get_device_name(),physical_gpu=1,device_peaks=peaks,
              options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},runs=[])
    previous=None
    for repeat in range(args.repeat):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        stages={name:[] for name in ("table_total","table_vertical","assembly","transform","LU_refinement","recovery")}
        original_green,original_vertical=LayeredGreenTable.build,VerticalGreenTable.build
        def wrap(name,function):
            def measured(*a,**kw):
                elapsed,result=timed(lambda:function(*a,**kw))
                stages[name].append(elapsed)
                print(name,elapsed,flush=True)
                return result
            return measured
        fixture=fixture_fa(lossy=not args.lossless,substrate=not args.vacuum,
                           substrate_sigma=args.sigma,ring=not args.one_port,
                           mesh=args.mesh,top_thickness_um=args.thickness)
        frequencies=tuple(args.frequencies) if args.frequencies else SWEEP
        simulation=fixture.simulation(frequencies)
        with patch.object(LayeredGreenTable,"build",new=wrap("table_total",original_green)), patch.object(
                VerticalGreenTable,"build",new=wrap("table_vertical",original_vertical)):
            prepare_time,ready=timed(simulation.prepare)
        prepared=ready.solver.prepared
        print("Prepared",args.mesh,prepared.layout.basis.n_edges,prepare_time,flush=True)
        with patch.object(lowfreq,"assemble_potentials",new=wrap("assembly",lowfreq.assemble_potentials)), patch.object(
                lowfreq,"scale_potentials",new=wrap("transform",lowfreq.scale_potentials)), patch.object(
                network,"solve_scaled_chunk",new=wrap("LU_refinement",network.solve_scaled_chunk)), patch.object(
                network,"recover_scaled_chunk",new=wrap("recovery",network.recover_scaled_chunk)):
            run_time,result=timed(ready.run)
        extracted=extract_fa(result.network)
        if len(frequencies)>3: extracted.update(resonance_and_peak(extracted))
        y=result.network.to_y()
        reciprocity=(torch.linalg.matrix_norm(y-y.transpose(-1,-2))/torch.linalg.matrix_norm(y)).max().item()
        singular=torch.linalg.matrix_norm(result.network.s,ord=2).max().item()
        equal=True if previous is None else torch.equal(previous[0],result.network.s) and torch.equal(previous[1],result.surface_currents.currents)
        previous=(result.network.s.clone(),result.surface_currents.currents.clone())
        count=len(frequencies)
        row=dict(unknowns=prepared.layout.basis.n_edges,faces=len(prepared.layout.mesh.faces),
                 prepare=prepare_time,run=run_time,stages=stages,
                 frequency_chunk_size=result.network.metadata["frequency_chunk_size"],
                 table_horizontal_seconds=stages["table_total"][0]["gpu_seconds"]-stages["table_vertical"][0]["gpu_seconds"],
                 seconds_per_frequency={name:sum(x["gpu_seconds"] for x in stages[name])/count
                                        for name in ("assembly","transform","LU_refinement","recovery")},
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 extraction=extracted,reciprocity=reciprocity,max_singular_S=singular,torch_equal=equal,
                 factor_dtype=result.network.metadata["factor_dtype"],refinement_steps=result.network.metadata["refinement_steps"],
                 backward_error=result.network.metadata["backward_error"],
                 table_admission=dict(horizontal=prepared.table.interpolation_error.item(),
                                      vertical_z=prepared.table.vertical.z_interpolation_error.item(),
                                      vertical_rho=prepared.table.vertical.rho_interpolation_error.item()))
        if args.one_port:
            row.update(greenhouse=greenhouse_fa(fixture),greenhouse_stranded=stranded_greenhouse_fa(fixture),
                       dc_reference=dc_resistance_fa(fixture),mohan=mohan_spiral())
        if repeat==0:
            row["kernel_model"]=kernel_model(prepared,row["frequency_chunk_size"])
            if args.profile_kernels:
                row["profile_frequency_count"]=row["frequency_chunk_size"]
                row["kernel_seconds"]=profile_kernels(prepared,args.output,row["frequency_chunk_size"])
        data["runs"].append(row)
        source_paths=[Path("tests/support/rfic_fixtures.py"),Path("tests/support/rfic_fa_references.py"),
                      Path(__file__).relative_to(Path.cwd())]
        source_paths.extend(p for p in Path("witwin/maxwell/mom/layered").rglob("*")
                            if p.suffix in (".py",".cu",".cpp",".cuh"))
        data["source_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        args.output.write_text(json.dumps(data,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ("kernel_model","kernel_seconds","stages")}),flush=True)
        del result,prepared,ready,simulation,fixture
    print("completed",str(args.output),flush=True)

if __name__=="__main__":
    with torch.no_grad():
        main()
