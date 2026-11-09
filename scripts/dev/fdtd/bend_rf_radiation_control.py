"""Native mapped finite-feed radiation, physical closed flux and loss controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.bend_huygens_reference import dipole_far
from scripts.dev.fdtd.pcb_acceptance import package_identity


def build_simulation(*,spacing=.01,curvature=3.,pcb=False,transient=64,steady=32,graph=True):
    frequency=1e9
    chart=mw.BendCoordinates(curvature=curvature)
    source_s=.01
    port=mw.LumpedPort('feed',positive=(source_s,.02,0.),negative=(source_s,0.,0.),voltage_path=mw.AxisPath('y'),
        current_surface=mw.Box(position=torch.tensor((source_s,spacing/2,0.),dtype=torch.float64),
            size=torch.tensor((.02+spacing,0.,.02+spacing),dtype=torch.float64)),
        voltage_aperture=mw.Box(position=torch.tensor((source_s,.01,0.),dtype=torch.float64),
            size=torch.tensor((.02,.02,.02),dtype=torch.float64)))
    structures=[];monitors=[]
    if pcb:
        for name,u,size in (('ground',0.,.14),('patch',.02,.08)):
            structures.append(mw.Structure(name=name,geometry=mw.Box(
                position=torch.tensor((0.,u,0.),dtype=torch.float64),size=torch.tensor((size,0.,size),dtype=torch.float64)),
                material=mw.CopperFoil(thickness=35e-6,conductivity=5.8e7,frequency_range=(.5e9,2e9))))
        structures.append(mw.Structure(name='substrate',geometry=mw.Box(
            position=torch.tensor((0.,.01,0.),dtype=torch.float64),size=torch.tensor((.14,.02,.14),dtype=torch.float64)),
            material=mw.Material(eps_r=2.2,sigma_e=.001)))
        monitors.append(mw.PowerLossMonitor('dielectric',position=(0.,.01,0.),size=(.14,.02,.14)))
    monitors.append(mw.ClosedSurfaceMonitor.box('huygens',position=(0.,.01,0.),size=(.18,.1,.18),sample_spacing=.005))
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.12,.12),(-.1,.1),(-.12,.12))),grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=round(.08/spacing)),coordinates=chart,structures=structures,ports=(port,),monitors=monitors,device='cuda')
    return mw.Simulation.fdtd(scene,frequency=frequency,excitations=mw.PortExcitation('feed'),full_field_dft=pcb,
        cuda_graph=graph,run_time=mw.TimeConfig(transient_cycles=transient,steady_cycles=steady),
        spectral_sampler=mw.SpectralSampler(window='hanning'))


def run(output,*,load_result=None,**kwargs):
    start=time.perf_counter();simulation=build_simulation(**kwargs)
    if load_result is None:
        result=simulation.run();snapshot=output.with_suffix('.pt');result.save(snapshot)
    else:
        snapshot=load_result;result=mw.Result.load(snapshot,map_location='cuda',scene=simulation.scene)
    antenna=result.antenna(surface='huygens',driven_port='feed',theta_points=61,phi_points=121,batch_size=128)
    raw=result._native_raw_monitor('huygens')
    flux=sum(float(face['flux'].reshape(-1)[0]) for face in raw['faces'].values())
    radiated=float(antenna.p_rad[0]);accepted=float(antenna.p_accepted[0])
    copper=sum(float(record['surface_power'][0]) for record in result.conductor_losses.values())
    dielectric=float(result.power_loss('dielectric').total[0]) if kwargs['pcb'] else 0.
    report=dict(curvature_m_inverse=kwargs['curvature'],spacing_m=kwargs['spacing'],pcb=kwargs['pcb'],
        frequency_hz=1e9,elapsed_seconds=time.perf_counter()-start,steps=result.stats().get('steps_run'),dt=result.stats().get('dt'),
        graph_active=None if result.solver is None else bool(result.solver._cuda_graph_active),snapshot=str(snapshot),
        loaded_snapshot=load_result is not None,
        accepted_w=accepted,closed_flux_w=flux,radiated_w=radiated,copper_w=copper,dielectric_w=dielectric,
        flux_nf2ff_relative_error=abs(radiated-flux)/abs(flux),
        spectral_power_balance_relative_error=abs(flux+copper+dielectric-accepted)/abs(accepted),
        acceptance=dict(flux_nf2ff_relative_error=.03,spectral_power_balance_relative_error=.03))
    report['analysis_source_sha256']=package_identity(mw,'witwin-maxwell')['source_sha256']
    report['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['snapshot_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if not kwargs['pcb']:
        from witwin.maxwell.postprocess.nfft import NearFieldFarFieldTransformer
        from witwin.maxwell.postprocess.stratton_chu import equivalent_surface_currents_from_monitor
        directions=torch.tensor(((1.,0.,0.),(0.,0.,1.),(1.,2.,3.),(-2.,1.,3.),(2.,-3.,1.),(-3.,-2.,1.)),dtype=torch.float64,device='cuda')
        directions=directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True)
        current=result.port('feed').current.reshape(-1)[0]
        chart=result.scene.coordinates
        q=torch.tensor(((.01,.01,0.),),dtype=torch.float64,device='cuda')
        position=chart.position(q)[0]
        direction=chart.jacobian(q)[0,:,1]
        actual=NearFieldFarFieldTransformer(equivalent_surface_currents_from_monitor(result,'huygens'),solver=result.solver,
            c=mw.constants.C_0,eps0=mw.constants.EPSILON_0,mu0=mw.constants.MU_0,device='cuda').transform_directions(directions,radius=12.)
        exact=dipole_far(directions,frequency=1e9,position=position,direction=direction,radius=12.)
        # Native Ampere subtracts I_branch*w/volume. PortData.current=-I_branch
        # is the field-work convention, so the impressed Maxwell J moment is
        # -I_port times the declared negative-to-positive displacement.
        report['hertzian_complex_field_relative_error']=max(float(torch.linalg.vector_norm(actual[key]+value*current*.02)/torch.linalg.vector_norm(value*current*.02))
            for key,value in zip(('E','H'),exact))
        report['hertzian_reference_scope']='Leading electric dipole of the declared finite .02m radial current segment and .02m transverse aperture; actual measured branch current, no fitted amplitude/phase.'
        report['impressed_current_contract']='Ampere subtracts I_branch*w/volume; PortData.current=-I_branch; physical J moment=-I_port*terminal displacement.'
        report['acceptance']['hertzian_complex_field_relative_error']=.03
    report['passed']=(report['flux_nf2ff_relative_error']<.03 and report['spectral_power_balance_relative_error']<.03)
    if not kwargs['pcb']:
        report['passed']=report['passed'] and report['hertzian_complex_field_relative_error']<.03
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spacing',type=float,default=.01)
    parser.add_argument('--curvature',type=float,default=3.);parser.add_argument('--pcb',action='store_true')
    parser.add_argument('--transient',type=int,default=64);parser.add_argument('--steady',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--load-result',type=Path);args=parser.parse_args()
    report=run(args.output,load_result=args.load_result,spacing=args.spacing,curvature=args.curvature,pcb=args.pcb,transient=args.transient,steady=args.steady)
    if not report['passed']:raise SystemExit('Physical mapped radiation/power acceptance failed; raw snapshot retained.')
