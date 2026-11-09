"""Independent broadband native-VI physics, separate from modal-port S tests."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import witwin.maxwell as mw
from benchmark.scenes.rf.printed_lines import printed_line_scene,WIDTH,HEIGHT,GAP,PAIR_WIDTH,PAIR_OFFSET
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.printed_line_physical_modes import coupled_physical_state,coupled_transfer
from scripts.dev.fdtd.printed_line_regional_power import regional_monitors,regional_power,regional_plane_power
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.sources import evaluate_source_time

ROOT=Path(__file__).resolve().parents[3]


def encode(value):
    if isinstance(value,complex):return {'real':value.real,'imag':value.imag}
    if torch.is_tensor(value):return encode(value.detach().cpu().numpy())
    if isinstance(value,np.ndarray):return encode(value.tolist())
    if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [encode(v) for v in value]
    if isinstance(value,Path):return str(value)
    return value


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box(position,size):
    return mw.Box(position=torch.tensor(position,dtype=torch.float64),size=torch.tensor(size,dtype=torch.float64))


def physical_sources(scene,kind,normal_dl):
    """Fixed finite apertures; explicit current loops follow actual half grids.

    The loop is a diagnostic row. Norton deposition uses only the fixed physical
    aperture voltage row. Balanced two-sided feeds avoid introducing an extra
    ground-difference TEM mode in the three-conductor CPW and stripline.
    """
    axes=[np.asarray(getattr(scene.grid,a+'_coords'),dtype=np.float64) for a in 'xyz']
    halves=[.5*(a[1:]+a[:-1]) for a in axes]
    separation=scene.metadata['port_separation_m'];ports=[];groups=[]
    def add(name,x,y,z_ground,width,axis='z'):
        if axis=='z':
            positive=(x,y,0.);negative=(x,y,z_ground)
            center=[x,y,z_ground/2];extent=[2*normal_dl,width,abs(z_ground)]
        else:
            positive=(x,y,0.);negative=(x,z_ground,0.)
            center=[x,(y+z_ground)/2,0.];extent=[2*normal_dl,abs(z_ground-y),0.]
        normal='xyz'.index(axis);loop_center=list(center);loop_extent=[0.,0.,0.]
        candidates=halves[normal]
        low,high=sorted((positive[normal],negative[normal]))
        candidates=candidates[(candidates>low)&(candidates<high)]
        if len(candidates)==0:raise ValueError('Source gap needs an interior native magnetic plane.')
        loop_center[normal]=float(candidates[np.argmin(abs(candidates-center[normal]))])
        for dim in range(3):
            if dim==normal:continue
            low=center[dim]-extent[dim]/2;high=center[dim]+extent[dim]/2
            lower=halves[dim][halves[dim]<low-1e-14];upper=halves[dim][halves[dim]>high+1e-14]
            if len(lower)==0 or len(upper)==0:raise ValueError('Finite source aperture needs native loop clearance.')
            loop_center[dim]=float((lower[-1]+upper[0])/2);loop_extent[dim]=float(upper[0]-lower[-1])
        port=mw.LumpedPort(name,positive=positive,negative=negative,voltage_path=mw.AxisPath(axis),
            current_surface=box(loop_center,loop_extent),voltage_aperture=box(center,extent),reference_impedance=50.)
        ports.append(port);return name
    for side,x in (('left',-separation/2),('right',separation/2)):
        if kind=='stripline':
            groups.append((add(side+'_bottom',x,0.,-HEIGHT,WIDTH),add(side+'_top',x,0.,HEIGHT,WIDTH)))
        elif kind=='cpw':
            groups.append((add(side+'_right',x,WIDTH/2,WIDTH/2+GAP,0.,'y'),
                           add(side+'_left',x,-WIDTH/2,-WIDTH/2-GAP,0.,'y')))
        elif kind=='differential_pair':
            for label,y in (('positive',PAIR_OFFSET),('negative',-PAIR_OFFSET)):
                groups.append((add(side+'_'+label,x,y,-HEIGHT,PAIR_WIDTH),))
        else:groups.append((add(side,x,0.,-HEIGHT,WIDTH),))
    return tuple(ports),tuple(groups)


def port_copy(port,termination):
    return mw.LumpedPort(port.name,positive=port.positive,negative=port.negative,
        voltage_path=port.voltage_path,current_surface=port.current_surface,
        voltage_aperture=port.voltage_aperture,reference_impedance=port.reference_impedance,
        reference_plane=port.reference_plane,termination=termination)


def load_helper(kind):
    filename='coupled_contours.py' if kind=='differential_pair' else 'physical_contours.py'
    path=ROOT/'docs/assessments/printed-lines-2026-10-22'/filename
    spec=importlib.util.spec_from_file_location('line_native_helper',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper,path


def configuration(args,device):
    return printed_line_scene(args.kind,dx=args.dx,longitudinal_dl=args.longitudinal_dl,
        transverse_padding=args.padding,port_separation=args.separation,axial_half_length=args.axial_half_length,
        transverse_refinement_radius=args.transverse_refinement_radius,sigma_e=args.sigma,metal=args.metal,device=device,
        copper_frequency_range=(.1e9,4e9) if args.metal=="copper" else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('microstrip','stripline','cpw','differential_pair'),required=True)
    parser.add_argument('--dx',type=float,default=.00125);parser.add_argument('--longitudinal-dl',type=float,default=.0025)
    parser.add_argument('--padding',type=float,default=.02);parser.add_argument('--separation',type=float,default=.4)
    parser.add_argument('--axial-half-length',type=float,default=.3);parser.add_argument('--transverse-refinement-radius',type=float)
    parser.add_argument('--sigma',type=float,default=0.);parser.add_argument('--metal',choices=('pec','copper'),default='pec')
    parser.add_argument('--positions',type=float,nargs='+',default=[-.045,-.03,-.015,0.,.015,.03,.045])
    parser.add_argument('--duration',type=float,default=32e-9);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--power-roi',type=float,nargs=6,metavar=('XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX'))
    parser.add_argument('--power-observation',choices=('points','planes'),default='points')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--save-results',action='store_true')
    args=parser.parse_args()
    if args.power_observation=='planes' and args.power_roi is None:
        raise ValueError('Native-plane power observations require explicit --power-roi bounds.')
    if args.output.exists():raise ValueError('Preserve prior evidence; choose a different output path.')
    if max(map(abs,args.positions))>=args.separation/2:raise ValueError('Physical stations must lie strictly between sources.')
    started=time.perf_counter();identity=package_identity(mw,'witwin-maxwell');driver_sha=digest(__file__)
    fixture=Path(inspect.getfile(printed_line_scene));fixture_sha=digest(fixture)
    helper,helper_path=load_helper(args.kind);helper_sha=digest(helper_path)
    state_helper=Path(inspect.getfile(coupled_transfer));state_sha=digest(state_helper)
    regional_helper=Path(inspect.getfile(regional_power));regional_sha=digest(regional_helper)
    base=configuration(args,'cpu' if args.prepare_only else 'cuda')
    ports,groups=physical_sources(base,args.kind,args.longitudinal_dl)
    base=base.clone(ports=ports)
    prepared=prepare_scene(base)
    compiled=[compile_port_geometry(prepared,p,device='cpu') for p in prepared.ports]
    source_geometry=[{'name':p.name,'positive':p.positive,'negative':p.negative,
        'voltage_aperture':{'position':p.voltage_aperture.position,'size':p.voltage_aperture.size},
        'current_surface':{'position':p.current_surface.position,'size':p.current_surface.size},
        'voltage_dof_count':len(c.voltage_weights),'voltage_weight_sum':float(c.voltage_weights.sum())}
        for p,c in zip(prepared.ports,compiled)]
    common={'schema':'printed-line-broadband-native-physics/1','configuration':vars(args),
        'source_geometry':source_geometry,'coherent_drive_groups':groups,'source':identity,
        'scope':'Independent native line transfer; these S data are finite physical feed S, not WavePort modal-port S. Fixed apertures; paired symmetric stripline/CPW physical drives, four independent pair drives.',
        'fixture_sha256':fixture_sha,'native_contour_helper_sha256':helper_sha,'state_helper_sha256':state_sha,'driver_sha256':driver_sha}
    frequencies=tuple(float(f) for f in np.linspace(.8e9,1.2e9,5))
    roi_monitors=[];roi_schema=None;roi_cpu_scene=None;roi_bounds=None
    if args.power_roi is not None:
        roi_bounds=tuple(tuple(args.power_roi[i:i+2]) for i in (0,2,4))
        roi_cpu_base=configuration(args,'cpu')
        roi_monitors,roi_schema=regional_monitors(roi_cpu_base,roi_bounds,frequencies,observation=args.power_observation)
        roi_cpu_scene=roi_cpu_base.clone(monitors=roi_monitors)
        common.update(regional_power_helper_sha256=regional_sha,power_roi_bounds_m=roi_bounds,
            regional_observation_schema=roi_schema)
    if args.prepare_only:
        common['status']='CPU geometry compiled only; no field solve or physical acceptance'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(encode(common),indent=2)+'\n');return
    if args.kind=='differential_pair':
        monitors,integrals,expected=helper.probes(base,args.dx,args.longitudinal_dl,args.padding,args.positions,args.metal)
    else:
        monitors,single,expected=helper.native_probe_monitors(base,args.kind,args.dx,args.longitudinal_dl,args.padding,args.positions,args.metal)
        integrals=[[pair] for pair in single]
    monitor_by_name={m.name:m for m in monitors};tails=[]
    for station,modes in enumerate(integrals):
        for mode,(voltage,_) in enumerate(modes):
            name,field,_=voltage[len(voltage)//2];tails.append(mw.FieldTimeMonitor(f'tail_{station}_{mode}',
                components=(field,),position=monitor_by_name[name].position,interval=32))
    common['integration_rows']=integrals
    common['integration_arithmetic']='complex128 SI sums and transfer algebra; native recorded samples retain their original field precision'
    pulse=mw.GaussianPulse(1e9,fwidth=.8e9)
    values=[];runs=[];spectra={name:[] for name in expected}
    for column,active in enumerate(groups):
        scene=base.clone(ports=[port_copy(p,None if p.name in active else mw.SeriesRLC(r=50.)) for p in ports],monitors=[*monitors,*tails,*roi_monitors])
        simulation=mw.Simulation.fdtd(scene,frequencies=frequencies,
            excitations=[mw.PortExcitation(name,source_time=pulse) for name in active],
            run_time=mw.TimeConfig(time_steps=1),spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False),full_field_dft=False)
        execution=simulation.prepare();dt=execution.solver.dt;steps=math.ceil(args.duration/dt)
        simulation.config.run_time=mw.TimeConfig(time_steps=steps);result=execution.run()
        native={name:result.monitor(name)['components'][field].detach().cpu().numpy().astype(np.complex128).reshape(5) for name,(field,_) in expected.items()}
        for name,(field,index) in expected.items():
            if tuple(result.monitor(name)['field_indices'][field])!=index:raise ValueError('Native physical probe index changed.')
            spectra[name].append(native[name])
        def integral(rows):return sum(native[name]*weight for name,_,weight in rows)
        stations=[]
        for modes in integrals:
            pairs=[(integral(v),integral(i)) for v,i in modes]
            if args.kind=='differential_pair':
                stations.append(coupled_physical_state(pairs[0][0],pairs[1][0],pairs[0][1],pairs[1][1]))
            else:stations.append(np.stack(pairs[0]))
        values.append(stations)
        trace_records={}
        for monitor in tails:
            payload=result.monitor(monitor.name)
            field=np.asarray(encode(payload['field']));time_axis=np.asarray(encode(payload['t']))
            peak=float(abs(field).max());last=time_axis>=args.duration-2e-9
            trace_records[monitor.name]={'time_s':time_axis,'native_field':field,
                'last2ns_peak_relative':float(abs(field[last]).max()/peak) if peak else None}
        time_axis=np.arange(steps,dtype=float)*dt;waveform=np.array([evaluate_source_time(pulse,float(t)) for t in time_axis])
        pulse_spectrum=2/steps*np.exp(2j*np.pi*np.asarray(frequencies)[:,None]*time_axis[None,:])@waveform
        run={'column':column,'active_ports':active,'dt_s':dt,'steps':steps,'actual_duration_s':steps*dt,
            'source_spectrum':pulse_spectrum,'native_time_traces':trace_records,'solver_stats':result.stats()}
        if roi_schema is not None:
            reducer=regional_plane_power if args.power_observation=='planes' else regional_power
            run['regional_power']=reducer(result,roi_cpu_scene,roi_schema,roi_bounds)
        if args.save_results:
            path=args.output.with_name(args.output.stem+f'-drive{column}.result.pt');result.save(path);run['saved_result']=str(path)
        runs.append(run)
        del result,execution,simulation
        torch.cuda.empty_cache()
    values=np.asarray(values) # [drive,station,state,frequency]
    frequency_records=[]
    for index,frequency in enumerate(frequencies):
        state=values[:,:,:,index].transpose(1,2,0)
        if args.kind=='differential_pair':transfers=coupled_transfer(state[:,:2],state[:,2:],args.positions)
        else:transfers=helper.state_transfer(state[:,0],state[:,1],args.positions)
        frequency_records.append({'frequency_hz':frequency,'actual_physical_state':state,'state_transfer':transfers})
    common.update(frequencies_hz=frequencies,frequency_records=frequency_records,runs=runs,
        native_probe_spectra={m.name:{'position_m':m.position,'component':expected[m.name][0],'index':expected[m.name][1],
            'all_drive_frequency_phasors':spectra[m.name]} for m in monitors},seconds=time.perf_counter()-started,
        source_unchanged=identity['source_sha256']==package_identity(mw,'witwin-maxwell')['source_sha256'],
        fixture_unchanged=fixture_sha==digest(fixture),driver_unchanged=driver_sha==digest(__file__),
        helper_unchanged=helper_sha==digest(helper_path),state_helper_unchanged=state_sha==digest(state_helper),
        regional_helper_unchanged=regional_sha==digest(regional_helper))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(encode(common),indent=2,allow_nan=False)+'\n')
    if not all(common[k] for k in ('source_unchanged','fixture_unchanged','driver_unchanged','helper_unchanged','state_helper_unchanged','regional_helper_unchanged')):raise RuntimeError('Physical broadband evidence identity changed; reject qualification.')
    print(json.dumps({'kind':args.kind,'dx':args.dx,'frequencies':frequencies,'drive_count':len(groups),'seconds':common['seconds']}),flush=True)


if __name__=='__main__':main()
