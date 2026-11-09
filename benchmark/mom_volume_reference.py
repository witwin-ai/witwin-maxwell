"""Independent CUDA MoM volume loss against a separately integrated Mie field."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import witwin.maxwell as mw
from benchmark.mom_fdtd import (FREQUENCY, INTEGRATION, PROCESS_PROVENANCE,
                                control_sphere, tetrahedral_volume_rule)
from benchmark.sie_reference import sphere_fields, sphere_far_field, sphere_cross_sections
from witwin.maxwell.mom.observables import volume_absorption, plane_wave_cross_sections
from witwin.maxwell.mom.quadrature import gauss_rule


def radial_face_volume_rule(mesh, radial_order):
    """Positive center-fan rule: radial Gauss times degree-two face cubature."""
    center=mesh.vertices.mean(0);triangles=mesh.triangles
    vectors=triangles-center
    volumes=(vectors[:,0]*torch.linalg.cross(vectors[:,1],vectors[:,2])).sum(-1)/6
    if not bool((volumes>0).all()):
        raise ValueError('The center fan requires an outward convex surface')
    bary=torch.tensor([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]],device=mesh.device,dtype=mesh.dtype)
    face=torch.einsum('qi,tic->tqc',bary,vectors)
    r,w=gauss_rule(radial_order,device=mesh.device,dtype=mesh.dtype)
    points=center+r[None,:,None,None]*face[:,None]
    weights=volumes[:,None,None]*(r*r*w)[None,:,None].expand(-1,-1,3)
    return points.reshape(-1,3),weights.reshape(-1)


def mie_volume(radius,k,epsilon,eta,sigma,radial_order,angular_order):
    """Independent exact-sphere SI Joule integral from analytic interior E."""
    r,wr=np.polynomial.legendre.leggauss(radial_order);r=radius*(r+1)/2;wr=wr*radius/2
    z,wz=np.polynomial.legendre.leggauss(angular_order)
    phi=np.arange(2*angular_order)*np.pi/angular_order
    rr,zz,pp=np.meshgrid(r,z,phi,indexing='ij')
    points=np.stack((rr*np.sqrt(1-zz*zz)*np.cos(pp),rr*np.sqrt(1-zz*zz)*np.sin(pp),rr*zz),-1).reshape(-1,3)
    weights=((r*r*wr)[:,None,None]*wz[None,:,None]*np.ones((1,1,len(phi)))*np.pi/angular_order).reshape(-1)
    e,_=sphere_fields(radius,k,epsilon,points,eta=eta,cutoff=24)
    density=.5*sigma*np.sum(abs(e)**2,-1)
    return {'radial_order':radial_order,'angular_order':angular_order,'points':len(points),
            'volume_m3':float(weights.sum()),'independent_absorption_W':float(np.sum(weights*density))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--subdivision',type=int,default=4)
    args=parser.parse_args(argv)
    radius,sigma,phase=.1,.004,.23
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=radius),mw.Material(eps_r=2.25,sigma_e=sigma))],
                   sources=[mw.PlaneWave(direction=(0,0,1),polarization=(1,0,0),source_time=mw.CW(frequency=FREQUENCY,amplitude=1.,phase=phase))])
    integration=replace(INTEGRATION,far_pair_batch=4096)
    config=mw.MoMConfig(subdivisions=args.subdivision,integration=integration,
                        query_config=mw.FieldQueryConfig(relative_tolerance=1e-5,max_order=128))
    report={'scope':'Independent physical volume loss on the solved polyhedron and analytic exact sphere; balance loss retained only as a separate diagnostic.',
            'complete_G0':False,'provenance_segments':[PROCESS_PROVENANCE],
            'source_sha256':{str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'benchmark/sie_reference.py':hashlib.sha256(Path('benchmark/sie_reference.py').read_bytes()).hexdigest()},
            'frequency_Hz':FREQUENCY,'radius_m':radius,'epsilon_real':2.25,'conductivity_S_per_m':sigma,
            'incident_phase_radians':phase,'configuration':{'integration':asdict(integration),'query':asdict(config.query_config)},
            'physical_relative_budget':.005,'reference_quadrature_relative_budget':.0005,'volume_rules':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint():args.output.write_text(json.dumps(report,indent=2))
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    result=mw.Simulation.mom(scene,frequency=FREQUENCY,config=config).run()
    data=result.surface_currents;handle=data.handle
    report.update(edges=handle.basis.n_edges,solve=result.stats(),assembly=handle.metadata,
                  epsilon_r=[handle.interior.epsilon_r.real,handle.interior.epsilon_r.imag])
    native=args.output.with_suffix('.surface.pt')
    result.save(native)
    report['native_surface_snapshot']={'path':str(native),'sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
                                       'scope':'Exact mesh and physical currents; no dense matrix; reload with the declared Scene.'}
    report['reference_volume']=[mie_volume(radius,handle.exterior.k.real,handle.interior.epsilon_r,
        handle.exterior.eta.real,sigma,r,a) for r,a in ((12,12),(18,18))]
    oracle=report['reference_volume'][-1]['independent_absorption_W']
    report['reference_quadrature_relative_change']=abs(report['reference_volume'][0]['independent_absorption_W']/oracle-1)
    checkpoint();print('assembled and independent Mie volume integrated',flush=True)
    rules=[('tetrahedral_degree1',lambda:tetrahedral_volume_rule(data.mesh,degree=1)),
           ('tetrahedral_degree2',lambda:tetrahedral_volume_rule(data.mesh,degree=2)),
           ('radial3_face_degree2',lambda:radial_face_volume_rule(data.mesh,3)),
           ('radial4_face_degree2',lambda:radial_face_volume_rule(data.mesh,4))]
    for name,rule in rules:
        begin=time.perf_counter();points,weights=rule()
        power=data.absorbed_power(points,weights,query_config=config.query_config)
        torch.cuda.synchronize()
        row={'rule':name,'points':len(points),'volume_m3':float(weights.sum()),'independent_absorption_W':float(power),
             'relative_error_to_exact_sphere_Mie_volume':abs(float(power)/oracle-1),'seconds':time.perf_counter()-begin,
             'query_log':dict(handle.last_field_query_log)}
        report['volume_rules'].append(row);checkpoint();print(json.dumps(row),flush=True)
    _,directions,_,weights=control_sphere(order=12)
    cross=data.plane_wave_cross_sections(directions,weights)
    reference=sphere_cross_sections(radius,handle.exterior.k.real,handle.interior.epsilon_r,cutoff=24)
    far,_=result.far_field(directions)
    exact=torch.as_tensor(sphere_far_field(radius,handle.exterior.k.real,handle.interior.epsilon_r,directions.cpu().numpy(),cutoff=24)*np.exp(-1j*phase),device=far.device)
    far_error=float(((weights*(far-exact).abs().square().sum(-1)).sum()/(weights*exact.abs().square().sum(-1)).sum()).sqrt())
    report['cross_sections']={key:float(value) for key,value in cross.items()}
    report['far_samples']={'directions':directions.cpu().tolist(),'solid_angle_weights':weights.cpu().tolist(),
        'numeric_E':{'real':far.real.cpu().tolist(),'imag':far.imag.cpu().tolist()},
        'Mie_E':{'real':exact.real.cpu().tolist(),'imag':exact.imag.cpu().tolist()}}
    report['reference_cross_sections_m2']=reference
    report['complex_far_relative_error']=far_error
    report['scattering_relative_error']=abs(float(cross['scattering_m2'])/reference['scattering']-1)
    report['extinction_relative_error']=abs(float(cross['extinction_m2'])/reference['extinction']-1)
    report['balance_absorption_relative_error']=abs(float(cross['absorption_by_balance_m2'])/reference['absorption']-1)
    report['Mie_volume_vs_Mie_balance_relative_difference']=abs(oracle/(reference['absorption']*float(cross['incident_intensity_W_per_m2']))-1)
    report['final_volume_quadrature_relative_change']=abs(report['volume_rules'][-2]['independent_absorption_W']/report['volume_rules'][-1]['independent_absorption_W']-1)
    report['total_seconds']=time.perf_counter()-start;report['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated()
    checkpoint();print(json.dumps({key:value for key,value in report.items() if key.endswith('error') or key.endswith('change') or key=='total_seconds'}),flush=True)


if __name__=='__main__':main()
