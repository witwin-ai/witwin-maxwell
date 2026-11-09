"""Compare physical Huygens quadrature independently of a fixed antenna field mesh."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import witwin.maxwell as mw
from benchmark.scenes.antenna.printed_patch import printed_patch_scene
from scripts.dev.fdtd.pcb_acceptance import package_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spacing',type=float,default=.002)
    parser.add_argument('--edge-dl',type=float,default=None)
    parser.add_argument('--lateral-dl',type=float,default=.001)
    parser.add_argument('--substrate-cells',type=int,default=3)
    parser.add_argument('--pml-layers',type=int,default=8)
    parser.add_argument('--air-padding',type=float,default=0.)
    parser.add_argument('--duration',type=float,default=16e-9)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    source=package_identity(mw,'witwin-maxwell')
    frequencies=tuple(float(x) for x in np.linspace(2.9e9,3.5e9,31))
    scene=printed_patch_scene(lateral_dl=args.lateral_dl,substrate_cells=args.substrate_cells,
        edge_dl=args.edge_dl,pml_layers=args.pml_layers,air_padding=args.air_padding,frequencies=frequencies)
    old=scene.monitors[0]
    scene=scene.clone(monitors=[mw.ClosedSurfaceMonitor.box('radiation',position=tuple((a+b)/2 for a,b in old.bounds),
        size=tuple(b-a for a,b in old.bounds),frequencies=frequencies,
        sample_spacing=args.spacing if args.spacing else None)])
    config=mw.Simulation.fdtd(scene,frequencies=frequencies,full_field_dft=False,
        excitations=mw.PortExcitation('feed',source_time=mw.GaussianPulse(frequency=3.2e9,fwidth=2.2e9)),
        run_time=mw.TimeConfig(time_steps=1),spectral_sampler=mw.SpectralSampler(window='none'))
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();prepared=config.prepare();torch.cuda.synchronize();prepare_seconds=time.perf_counter()-start
    solver=prepared.solver;config.config.run_time=mw.TimeConfig(time_steps=math.ceil(args.duration/solver.dt))
    start=time.perf_counter();result=prepared.run();torch.cuda.synchronize();run_seconds=time.perf_counter()-start
    start=time.perf_counter();antenna=result.antenna(surface='radiation',driven_port='feed',theta_points=91,phi_points=73,radius=10.);torch.cuda.synchronize();post_seconds=time.perf_counter()-start
    reflection=result.port('feed').reflection_coefficient.cpu().numpy()
    record={'schema':'physical-surface-quadrature/1','source':source,
        'source_unchanged':source['source_sha256']==package_identity(mw,'witwin-maxwell')['source_sha256'],
        'field_grid_configuration':scene.metadata,'gpu':torch.cuda.get_device_name(),'frequency_hz':frequencies,
        'sample_spacing_m':args.spacing,'grid_shape':[solver.scene.Nx,solver.scene.Ny,solver.scene.Nz],
        'steps':config.config.run_time.time_steps,'dt_s':solver.dt,
        'duration_s':config.config.run_time.time_steps*solver.dt,
        'prepare_seconds':prepare_seconds,'run_seconds':run_seconds,'postprocess_seconds':post_seconds,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'observer_spectral_bytes':sum(g[k].numel()*g[k].element_size() for g in solver._plane_observer_groups.values() for k in ('real','imag')),
        'reflection_real':reflection.real.tolist(),'reflection_imag':reflection.imag.tolist(),
        'antenna':{'p_rad':antenna.p_rad.cpu().tolist(),'p_accepted':antenna.p_accepted.cpu().tolist(),
                   'broadside_directivity_dbi':antenna.directivity_db[:,0,:].amax(-1).cpu().tolist(),
                   'max_directivity_dbi':antenna.directivity_db.amax((-2,-1)).cpu().tolist()}}
    if args.reference:
        document=json.loads(args.reference.read_text());reference=document.get('run',document)
        expected=np.array(reference['reflection_real'])+1j*np.array(reference['reflection_imag'])
        record['comparison']={'reference':str(args.reference),'max_complex_reflection_difference':float(abs(reflection-expected).max()),
            'maximum_radiated_power_relative_difference':float(abs(np.array(record['antenna']['p_rad'])/np.array(reference['antenna']['p_rad'])-1).max()),
            'maximum_directivity_difference_db':float(abs(np.array(record['antenna']['max_directivity_dbi'])-np.array(reference['antenna']['max_directivity_dbi'])).max())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:record[key] for key in ('run_seconds','postprocess_seconds','observer_spectral_bytes','comparison') if key in record}),flush=True)
    if not record['source_unchanged']:raise RuntimeError('Numerical source changed during quadrature run.')


if __name__=='__main__':main()
