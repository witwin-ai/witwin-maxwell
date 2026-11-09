"""Native high-R intrinsic response with explicitly measured source current."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import witwin.maxwell as mw
import witwin.maxwell.postprocess.fdtd_loss
from scripts.dev.fdtd.bend_rf_radiation_control import build_simulation
from scripts.dev.fdtd.pcb_acceptance import package_identity

DRIVER_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_binding():
    root=Path(__file__).resolve().parents[3]
    base=root/'SOURCE_BASE_COMMIT'
    if not base.exists():
        raise RuntimeError('Native physical qualification controls require an immutable source archive with SOURCE_BASE_COMMIT and frozen controller overlays.')
    package=package_identity(mw,'witwin-maxwell')
    if not Path(mw.__file__).resolve().is_relative_to(root):raise RuntimeError('The native package was not imported from the frozen source archive.')
    return dict(base_commit=base.read_text().strip(),module_file=package['module_file'],source_sha256=package['source_sha256'],
        helpers={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__).resolve(),Path(build_simulation.__code__.co_filename).resolve(),root/'scripts/dev/fdtd/bend_huygens_reference.py')})

def serial(value):
    value=torch.as_tensor(value).detach().cpu().numpy()
    return dict(real=value.real.tolist(),imag=value.imag.tolist())


def run(output,*,spacing=.01,duration=96e-9,resistance=1e12,vacuum=False,normal_samples=1,material_samples=None,averaging='polarized',metal='copper'):
    source_before=source_binding()
    start=time.perf_counter();base=build_simulation(spacing=spacing,pcb=not vacuum)
    old=base.scene.ports[0]
    port=mw.LumpedPort(old.name,positive=old.positive,negative=old.negative,voltage_path=old.voltage_path,
        voltage_aperture=old.voltage_aperture,current_surface=old.current_surface,reference_impedance=resistance)
    structures=base.scene.structures
    if metal=='pec':
        replacements=[]
        for structure in structures:
            if isinstance(structure.material,mw.CopperFoil):
                box=structure.geometry;size=box.size.detach().cpu().tolist();position=box.position.detach().cpu().tolist()
                vertices=[(position[0]+a*size[0]/2,position[2]+b*size[2]/2) for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))]
                structure=mw.Structure(name=structure.name,geometry=mw.PlanarPEC(vertices,normal_axis='y',coordinate=position[1]),material=mw.Material.pec())
            replacements.append(structure)
        structures=replacements
    scene=base.scene.clone(ports=(port,),structures=structures)
    samples=(1,normal_samples,1) if material_samples is None else tuple(material_samples)
    if any(value>1 for value in samples):
        scene=scene.clone(subpixel_samples=mw.SubpixelSpec(samples=samples,averaging=averaging))
    configuration=mw.Simulation.fdtd(scene,frequency=1e9,cuda_graph=True,full_field_dft=True,
        excitations=mw.PortExcitation('feed',source_time=mw.GaussianPulse(frequency=1e9,fwidth=.7e9,phase=math.pi/2)),
        run_time=mw.TimeConfig(time_steps=1),spectral_sampler=mw.SpectralSampler(window='none'))
    prepared=configuration.prepare();configuration.config.run_time=mw.TimeConfig(time_steps=math.ceil(duration/prepared.solver.dt))
    result=prepared.run();result.save(output.with_suffix('.pt'))
    runtime=result.solver._port_runtimes[0]
    drive,_=runtime.drive_accumulator.phasors(normalization='peak',integration_dt=result.solver._spectral_integration_dt,
        reference_frequency=result.solver._spectral_reference_frequency)
    current=drive/resistance;observed=result.port('feed');voltage=observed.voltage
    impedance=voltage/current
    antenna=result.antenna(surface='huygens',driven_port='feed',theta_points=25,phi_points=49,radius=12.,batch_size=128)
    report=dict(schema='native-bend-known-current-response/1',spacing_m=spacing,curvature_m_inverse=3.,
        frequency_hz=1e9,duration_s=duration,steps=result.stats().get('steps_run'),dt_s=result.stats().get('dt'),
        graph_active=bool(result.solver._cuda_graph_active),elapsed_s=time.perf_counter()-start,
        source_resistance_ohm=resistance,source_voltage=serial(drive),known_current=serial(current),
        actual_current=serial(observed.current),voltage=serial(voltage),intrinsic_impedance=serial(impedance),
        source_feedback_bound=float((torch.abs(impedance)/resistance).max()),
        observed_current_known_relative_error=float((torch.abs(observed.current-current)/torch.abs(current)).max()),
        analytic_50ohm_gamma=serial((impedance-50)/(impedance+50)),
        theta=antenna.theta.cpu().tolist(),phi=antenna.phi.cpu().tolist(),
        E_theta_per_amp=serial(antenna.e_theta/current[:,None,None]),E_phi_per_amp=serial(antenna.e_phi/current[:,None,None]),
        radiated_w_per_amp2=float((antenna.p_rad/current.abs().square())[0]),
        accepted_w_per_amp2=float((.5*(voltage*current.conj()).real/current.abs().square())[0]),
        copper_w_per_amp2=sum(float(record['surface_power'][0]/current.abs().square()[0]) for record in result.conductor_losses.values()),
        dielectric_w_per_amp2=0. if vacuum else float(result.power_loss('dielectric').total[0]/current.abs().square()[0]),
        vacuum=vacuum,metal=metal,
        substrate_normal_samples=normal_samples,
        material_samples=samples,material_averaging=averaging if any(value>1 for value in samples) else 'arithmetic point',
        copper_capacity_sha256=hashlib.sha256(b''.join(entry['capacity'].detach().cpu().numpy().tobytes() for entry in result.solver._copper_foils)).hexdigest(),
        copper_conductance_sha256=hashlib.sha256(b''.join(entry['conductance'].detach().cpu().numpy().tobytes() for entry in result.solver._copper_foils)).hexdigest(),
        source_contract='I_known=measured drive-voltage phasor/R at midpoint current clock; fixed clipped voltage-aperture functional; finite-R feedback bounded explicitly',
        snapshot=str(output.with_suffix('.pt')),snapshot_sha256=hashlib.sha256(output.with_suffix('.pt').read_bytes()).hexdigest(),
        execution_driver_sha256=DRIVER_SHA256,analysis_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification='intrinsic source-control response; external comparison, mesh and material-model limits reported separately')
    if any(value>1 for value in samples) and not vacuum and metal=='copper':
        point=base.prepare().solver
        comparison={}
        for key in ('capacity','conductance'):
            differences=[]
            for actual,baseline in zip(result.solver._copper_foils,point._copper_foils,strict=True):
                torch.testing.assert_close(actual['indices'],baseline['indices'],rtol=0,atol=0)
                scale=baseline[key].abs().max().clamp_min(torch.finfo(baseline[key].dtype).tiny)
                differences.append(float(((actual[key]-baseline[key]).abs()/scale).max()))
            comparison[key+'_maximum_relative_change']=max(differences)
            comparison[key+'_point_sha256']=hashlib.sha256(b''.join(entry[key].detach().cpu().numpy().tobytes() for entry in point._copper_foils)).hexdigest()
        report['actual_copper_coefficient_comparison_to_point_sampling']=comparison
    source_after=source_binding()
    if source_before!=source_after:raise RuntimeError('Native source or controller changed during the physical control; saved fields remain diagnostic only.')
    report['source_binding_before']=source_before;report['source_binding_after']=source_after
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:report[name] for name in ('spacing_m','elapsed_s','intrinsic_impedance','source_feedback_bound','observed_current_known_relative_error','radiated_w_per_amp2','accepted_w_per_amp2','copper_w_per_amp2','dielectric_w_per_amp2')}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spacing',type=float,default=.01);parser.add_argument('--duration',type=float,default=96e-9)
    parser.add_argument('--resistance',type=float,default=1e12);parser.add_argument('--vacuum',action='store_true')
    parser.add_argument('--metal',choices=('copper','pec'),default='copper')
    parser.add_argument('--normal-samples',type=int,default=1);parser.add_argument('--material-samples',type=int,nargs=3)
    parser.add_argument('--averaging',choices=('arithmetic','polarized'),default='polarized');args=parser.parse_args()
    run(args.output,spacing=args.spacing,duration=args.duration,resistance=args.resistance,vacuum=args.vacuum,
        normal_samples=args.normal_samples,material_samples=args.material_samples,averaging=args.averaging,metal=args.metal)
