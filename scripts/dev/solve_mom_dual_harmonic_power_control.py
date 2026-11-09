"""Public dual-conductive HB power refinement; no physical qualification claim."""
from dataclasses import replace,fields,asdict
from io import BytesIO
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig


def portable(value):
    if isinstance(value,torch.Tensor): return portable(value.detach().cpu().tolist())
    if isinstance(value,complex): return {'real':portable(value.real),'imag':portable(value.imag)}
    if isinstance(value,float) and not math.isfinite(value): return {'nonfinite_float':repr(value)}
    if isinstance(value,dict): return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [portable(v) for v in value]
    return value


def validate_frequency_axis(raw, harmonics):
    frequencies = raw['state']['frequencies']
    expected = torch.arange(harmonics + 1, dtype=torch.float64, device=frequencies.device) * 3e8
    if frequencies.dtype != expected.dtype or not torch.equal(frequencies, expected):
        raise ValueError('Snapshot frequency axis differs from the 300 MHz control')


def setup():
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    one=mw.TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    vertices=torch.cat((vertices,vertices.mean(0,keepdim=True)))
    cells=torch.cat((one.boundary.faces[:,[0,2,1]],torch.full((4,1),4,device='cuda',dtype=torch.int64)),dim=1)
    volume=mw.TetrahedralVolumeMesh(vertices,cells,region_id=1)
    material=mw.Material(eps_r=2.,sigma_e=.001,sigma_m=.002,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.)))
    scene=mw.Scene(structures=[mw.Structure(geometry=mw.Mesh(one.boundary.vertices,one.boundary.faces,
        recenter=False,fill_mode='solid'),material=material)],sources=[mw.PlaneWave()])
    hb=mw.HarmonicBalanceConfig(harmonics=2,volumes=(volume,),branch_id='linear_start',pump_scales=(.5,1.))
    config=mw.MoMConfig(harmonic_balance=hb,riesz_storage='compact_rwg_v1',
        integration=IntegrationConfig(order=3,singular_order=3),
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
    return scene,config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--surface-orders',type=int,nargs='+',default=[2,4,8])
    parser.add_argument('--volume-orders',type=int,nargs='+',default=[2,3,4])
    parser.add_argument('--max-query-order',type=int,default=192)
    parser.add_argument('--query-batch-size',type=int,default=32)
    args=parser.parse_args();root=args.destination;root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(mw.__file__).parent.rglob('*.py'))}
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record={'version':'dual-harmonic-power-control-v1','sources':sources,'physical_qualification':False,
            'status':'running','phase':'setup','surface_orders':args.surface_orders,'volume_orders':args.volume_orders,
            'max_query_order':args.max_query_order,'query_batch_size':args.query_batch_size,'integration_rtol':1e-4,'integration_atol_W':1e-20}
    def write(name,value): (root/name).write_text(json.dumps(portable(value),sort_keys=True,indent=2,allow_nan=False))
    write('start.json',record)
    try:
        scene,config=setup()
        record['phase']='restore' if args.snapshot else 'solve'
        print(record['phase'],flush=True)
        if args.snapshot:
            snapshot=args.snapshot.read_bytes()
            record['input_snapshot_sha256']=hashlib.sha256(snapshot).hexdigest()
            raw=torch.load(BytesIO(snapshot),weights_only=True)['harmonic_data']
            validate_frequency_axis(raw,config.harmonic_balance.harmonics)
            numerical={name:getattr(config,name) for name in ('subdivisions','riesz_storage')}
            numerical.update(dtype=str(config.dtype),integration=asdict(config.integration),
                stopping=asdict(config.stopping),query_config=asdict(config.query_config))
            settings={f.name:getattr(config.harmonic_balance,f.name) for f in fields(config.harmonic_balance) if f.name!='volumes'}
            stored_settings=dict(raw['harmonic_config'])
            # Pre-budget snapshots used these fixed native preparation limits.
            stored_settings.setdefault('source_max_order',32)
            stored_settings.setdefault('source_max_inner_order',64)
            if raw['numerical']!=numerical or stored_settings!=settings:
                raise ValueError('Snapshot numerical or continuation configuration differs from the control')
            expected=config.harmonic_balance.volumes
            if len(raw['volumes'])!=len(expected) or any(v['region_id']!=w.region_id or
                    any(v[key].dtype!=getattr(w,key).dtype or not torch.equal(v[key].to(w.vertices.device),getattr(w,key))
                        for key in ('vertices','cells')) for v,w in zip(raw['volumes'],expected)):
                raise ValueError('Snapshot volume discretization differs from the control')
            result=mw.Result.load(BytesIO(snapshot),scene=scene)
        else:
            result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
        result.save(root/'result.pt');write('solver-stats.json',result.stats())
        record['result_sha256']=hashlib.sha256((root/'result.pt').read_bytes()).hexdigest()
        record['phase']='power_refinement';print(record['phase'],flush=True)
        observations=[]
        def progress(step):
            name=f'power-step-{len(observations):03d}.json'
            write(name,step);observations.append(name)
            record['completed_power_steps']=list(observations)
            print(name,step['axis'],flush=True)
        report=result.refine_harmonic_power(sphere_mesh(radius=.2,subdivisions=0,device='cuda'),
            surface_orders=tuple(args.surface_orders),volume_orders=tuple(args.volume_orders),
            relative_tolerance=1e-4,absolute_tolerance=1e-20,
            query_config=replace(config.query_config,max_order=args.max_query_order),
            query_batch_size=args.query_batch_size,progress_callback=progress)
        write('power.json',report)
        record.update(status='completed_unqualified',relative_power_residual=report['relative_balance_residual'],
            integration_refinement_passed=report['integration_refinement_passed'])
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in sources.items()):
            raise RuntimeError('Source bytes changed during the control')
        print(portable(record['relative_power_residual']),flush=True)
    except BaseException as error:
        record.update(status='failed',error={'type':type(error).__name__,'message':str(error),
                                           'receipt':getattr(error,'receipt',None)})
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-start;write('terminal.json',record)


if __name__=='__main__': main()
