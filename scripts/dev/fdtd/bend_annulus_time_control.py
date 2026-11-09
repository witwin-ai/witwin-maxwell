"""Private TE1 raw native time capture with independent GPU complex128 spectra.

The mapped FieldTimeMonitor lowering is diagnostic-only here. No production
capability guard or solver operation is modified. Fields remain native GPU
float32; only the final Fourier reduction of retained samples uses complex128.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.bend_cylinder_native_control import _private_ring_drive,source_binding
from scripts.dev.fdtd.bend_annulus_em_audit import em_hankel_basis,project_em,modal_power_unit,encoded


def definitions(radius,spacing):
    result=[]
    for side,u in (('left',-radius/6),('right',radius/6)):
        for label,offset,component in (('e_m',-spacing,'Ex'),('e_0',0.,'Ex'),('e_p',spacing,'Ex'),
                ('h_m',-spacing/2,'Hz'),('h_p',spacing/2,'Hz')):
            result.append((side+'_'+label,(spacing/2,u+offset,0.),component))
    return result


def gpu_phasor(buffer,steps,start,end,dt,frequency,component):
    n=torch.as_tensor(steps,device=buffer.device,dtype=torch.float64)
    valid=(n>=start)&(n<end)
    fraction=(n[valid]-start)/(end-start)
    weights=.5*(1-torch.cos(2*torch.pi*fraction))
    angle=2*torch.pi*frequency*dt*(n[valid]-(.5 if component.startswith('H') else 0.))
    phase=torch.complex(torch.cos(angle),torch.sin(angle))
    return 2*torch.sum(buffer[valid].to(torch.complex128)*weights*phase)/weights.sum()


def run(output,*,spacing=.0025,direction=1,vacuum=False):
    before=source_binding();own_before=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    radius=.3;frequency=1e9;order=1;sigma=5.8e7
    k=2*math.pi*frequency/mw.constants.C_0;angle=math.asin(order/(radius*k))
    carrier=(math.sin(angle),direction*math.cos(angle),0.)
    electric=(direction*math.cos(angle),-math.sin(angle),0.)
    delta=math.sqrt(2/(2*math.pi*frequency*mw.constants.MU_0*sigma))
    specs=definitions(radius,spacing)
    monitors=[]
    for name,position,component in specs:
        monitors.append(mw.PointMonitor('point_'+name,position,fields=(component,)))
        monitors.append(mw.FieldTimeMonitor('time_'+name,position=position,components=(component,)))
    scene=mw.Scene(domain=mw.Domain(bounds=((-2*spacing,2*spacing),(-radius/2,radius/2),(-2*spacing,2*spacing))),
        grid=mw.GridSpec.uniform(spacing),boundary=mw.BoundarySpec.faces(default='none',num_layers=12,
            bloch_wavevector='auto',x='bloch',y='pml',z='bloch'),device='cuda',coordinates=mw.BendCoordinates(curvature=-1/radius),
        structures=() if vacuum else (mw.Structure(name='film',geometry=mw.Box(position=(0.,0.,0.),size=(16*spacing,0.,16*spacing)),
            material=mw.CopperFoil(thickness=delta,conductivity=sigma,frequency_range=(frequency/2,2*frequency))),),
        monitors=monitors,sources=(mw.PlaneWave(name='ring_mode_drive',direction=carrier,polarization=electric,injection_axis='y',
            source_time=mw.CW(frequency,amplitude=80.)),))
    from witwin.maxwell.compiler import coordinates
    with _private_ring_drive((order/radius,0.,0.),public_metric=True):
        original=coordinates.validate_coordinate_scene
        def diagnostic_time_validation(scene,**kwargs):
            # Every original material/source/metric guard remains in force;
            # only raw-time observation registration is private to this test.
            return original(scene.clone(monitors=[monitor for monitor in scene.monitors if not isinstance(monitor,mw.FieldTimeMonitor)]),**kwargs)
        coordinates.validate_coordinate_scene=diagnostic_time_validation
        try:
            prepared=mw.Simulation.fdtd(scene,frequency=frequency,cuda_graph=True,full_field_dft=True,
                run_time=mw.TimeConfig(transient_cycles=24,steady_cycles=20),spectral_sampler=mw.SpectralSampler(window='hanning')).prepare()
            result=prepared.run()
        finally:
            coordinates.validate_coordinate_scene=original
    solver=result.solver;entry=solver._observer_spectral_entries[0]
    start,end=int(entry['start_step']),int(entry['end_step'])
    records={record['name']:record for record in solver.time_observers}
    spectra={};comparison={};raw={}
    for name,position,component in specs:
        record=records['time_'+name];buffer=record['buffers'][component]
        value=gpu_phasor(buffer,record['sample_steps'],start,end,solver.dt,frequency,component)
        point=result._native_raw_monitor('point_'+name)['components'][component]
        index=record['component_meta'][component]['field_index']
        row=dict(native_index=list(index),position=position,complex128=encoded(value.detach().cpu().numpy()),
            native_point=encoded(point.detach().cpu().numpy()),point_relative_error=float(abs(point-value)/abs(value)))
        if component=='Ex':
            full=result.chart_fields.tensor('E_s')[index]
            row.update(native_full=encoded(full.detach().cpu().numpy()),full_relative_error=float(abs(full-value)/abs(value)))
        spectra[name]=value
        comparison[name]=row
        raw[name]=dict(component=component,steps=torch.as_tensor(record['sample_steps']),values=buffer.detach().cpu(),native_index=index)
    heat=0. if vacuum else float(result.conductor_losses['film']['surface_power'][0])
    area=(4*spacing)**2;retard=math.cos(math.pi*frequency*solver.dt)
    flux={};ports=[]
    for side,u in (('left',-radius/6),('right',radius/6)):
        get=lambda label:spectra[side+'_'+label]
        e=get('e_0');h=(get('h_m')+get('h_p'))/2
        flux[side]=dict(node_e=float(-.5*(e*h.conj()).real)*area*retard,
            lower_half_h=float(-.5*((get('e_m')+e)/2*get('h_m').conj()).real)*area*retard,
            upper_half_h=float(-.5*((e+get('e_p'))/2*get('h_p').conj()).real)*area*retard)
        a=project_em(complex(e.detach().cpu()),complex(h.detach().cpu()),radius+u,radius,frequency,order)
        ports.append(a)
    incoming=np.asarray((ports[0][0],ports[1][1]));outgoing=np.asarray((ports[0][1],ports[1][0]))
    norm=modal_power_unit(radius,frequency,area)*retard
    report=dict(schema='private-annulus-native-time-precision/1',spacing_m=spacing,direction=direction,vacuum=vacuum,
        radius_m=radius,frequency_hz=frequency,dt_s=solver.dt,steps=result.stats()['steps_run'],spectral_start=start,spectral_end=end,
        raw_field_dtype='native float32 real/imaginary fields stored as complex64 samples',fourier_dtype='GPU complex128 torch.sum',
        private_capability='FieldTimeMonitor mapped registration bypassed for this diagnostic only; no public mapped-time capability claim',
        source_binding_before=before,controller_sha256=own_before,comparison=comparison,flux=flux,copper_heat_w=heat,
        incoming=encoded(incoming),outgoing=encoded(outgoing),incoming_midpoint_power_w=float(norm*np.vdot(incoming,incoming).real),
        direct_heat_absorption=float(heat/(norm*np.vdot(incoming,incoming).real)),
        flux_minus_heat_w={key:flux['left'][key]-flux['right'][key]-heat for key in flux['left']})
    after=source_binding()
    if before!=after or own_before!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError('Private time diagnostic source changed during execution.')
    report['source_binding_after']=after
    raw_path=output.with_suffix('.samples.pt');torch.save(raw,raw_path)
    report['raw_samples']=str(raw_path);report['raw_samples_sha256']=hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spacing',type=float,default=.0025);parser.add_argument('--direction',type=int,choices=(-1,1),default=1)
    parser.add_argument('--vacuum',action='store_true');args=parser.parse_args()
    run(args.output,spacing=args.spacing,direction=args.direction,vacuum=args.vacuum)
