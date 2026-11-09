"""Complete regular-mode sphere response and spatial-convergence evidence.

All PMCHWT assembly, modal RHS, factorization and field decoding use CUDA.
The independent reference uses radial Bessel functions and spherical harmonics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import scipy
from scipy.special import sph_harm_y, spherical_jn, spherical_yn
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fields import far_field
from witwin.maxwell.mom.modes import RegularModes, modal_response
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.modes import angular_grid
from witwin.maxwell.mom.physics import compile_medium


def radial_scattering_coefficient(l, radius, exterior, interior, kind):
    """Solve the two analytic tangential transmission equations for one mode."""
    z0, z1 = exterior.k*radius, interior.k*radius
    j0, j1 = spherical_jn(l,z0), spherical_jn(l,z1)
    h0 = j0+1j*spherical_yn(l,z0)
    d0 = j0+z0*spherical_jn(l,z0,derivative=True)
    d1 = j1+z1*spherical_jn(l,z1,derivative=True)
    dh = h0+z0*(spherical_jn(l,z0,derivative=True)+1j*spherical_yn(l,z0,derivative=True))
    ratio = exterior.mu/interior.mu if kind == 'TE' else exterior.epsilon/interior.epsilon
    return (ratio*j0*d1-d0*j1)/(dh*j1-ratio*h0*d1)


def reference_modal_far(ordering, directions, radius, exterior, interior, scales):
    """Cartesian complex far E using independent radial and harmonic functions."""
    theta=np.arccos(directions[:,2]); phi=np.arctan2(directions[:,1],directions[:,0])
    tangent=np.stack((np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)),-1)
    azimuth=np.stack((-np.sin(phi),np.cos(phi),np.zeros_like(phi)),-1)
    columns=[]
    for (l,m,kind),scale in zip(ordering,scales):
        _,derivatives=sph_harm_y(l,m,theta,phi,diff_n=1)
        grad=(derivatives[:,0,None]*tangent+derivatives[:,1,None]/np.sin(theta)[:,None]*azimuth)/math.sqrt(l*(l+1))
        te=np.cross(directions,grad)
        density=te if kind=='TE' else np.cross(directions,te)
        outgoing=(-1j)**(l+1)/exterior.k
        scattering=radial_scattering_coefficient(l,radius,exterior,interior,kind)
        columns.append(scale*scattering*outgoing*density)
    return np.stack(columns,1)


def _encode(array):
    return {'real':array.real.tolist(),'imag':array.imag.tolist()}


def _norm(response,handle):
    # SurfaceOperator stores diag(M/eta^2,M). Its existing Cholesky blocks
    # recover diag(eta*chol(M)^H,chol(M)^H) without another dense matrix.
    n=handle.basis.n_edges; eta=abs(handle.exterior.eta)
    f=handle._riesz_factor
    return torch.cat((eta**2*(f[:n,:n].mH@response[:n]), f[n:,n:].mH@response[n:]))


def _spectral(matrix):
    return torch.linalg.matrix_norm(matrix,ord=2)


def plane_wave_projection(cutoffs,control_radius=.3):
    """Project an analytic plane wave; validate on a separate angular grid."""
    medium=compile_medium(mw.Material(),3e8)
    rows=[]
    for cutoff in cutoffs:
        modes=RegularModes(cutoff,medium,angular_order=24)
        modes.normalize_control_surface(control_radius,order=24)
        coefficients=[]
        for order in [20,24]:
            directions,weights,*_=angular_grid(order,device='cuda')
            points=control_radius*directions
            e,h=modes.evaluate(points)
            phase=torch.exp(1j*medium.k*points[:,2])
            incident_e=torch.zeros_like(points,dtype=torch.complex128);incident_e[:,0]=phase
            incident_h=torch.zeros_like(incident_e);incident_h[:,1]=phase/medium.eta
            c=control_radius**2/(4*medium.eta.real)*(torch.einsum('qmc,qc,q->m',e.conj(),incident_e,weights.to(e.dtype))+abs(medium.eta)**2*torch.einsum('qmc,qc,q->m',h.conj(),incident_h,weights.to(e.dtype)))
            coefficients.append(c)
        directions,weights,*_=angular_grid(32,device='cuda');points=control_radius*directions
        e,h=modes.evaluate(points,coefficients[-1]);phase=torch.exp(1j*medium.k*points[:,2])
        exact_e=torch.zeros_like(e);exact_e[:,0]=phase
        exact_h=torch.zeros_like(h);exact_h[:,1]=phase/medium.eta
        norm_e=torch.sum(weights*exact_e.abs().square().sum(-1));norm_h=torch.sum(weights*exact_h.abs().square().sum(-1))
        err_e=torch.sum(weights*(e-exact_e).abs().square().sum(-1));err_h=torch.sum(weights*(h-exact_h).abs().square().sum(-1))
        rows.append({'cutoff':cutoff,'columns':modes.count,'relative_control_E_L2_error':float(torch.sqrt(err_e/norm_e)),
                     'relative_control_H_L2_error':float(torch.sqrt(err_h/norm_h)),
                     'relative_control_energy_error':float(torch.sqrt((err_e+abs(medium.eta)**2*err_h)/(norm_e+abs(medium.eta)**2*norm_h))),
                     'coefficient_quadrature_relative_difference':float(torch.linalg.vector_norm(coefficients[0]-coefficients[1])/torch.linalg.vector_norm(coefficients[1]))})
    return {'source':'unit x-polarized plane wave along +z','frequency_Hz':3e8,'control_radius_m':control_radius,
            'projection_angular_orders':[20,24],'independent_validation_angular_order':32,'rows':rows,
            'scope':'this analytic plane wave on the control sphere; no universal source-family or unsampled-tail bound'}


def _provenance(output):
    root=Path(__file__).resolve().parents[1]
    paths=sorted((root/'witwin/maxwell').rglob('*.py'))+[Path(__file__).resolve()]
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identifier=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot=output.parent/'modal-exact-source'/identifier
    for path in paths:
        target=snapshot/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=path.read_bytes():raise RuntimeError('Immutable source snapshot mismatch')
        else:shutil.copy2(path,target)
    return {'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),
            'source_sha256':hashes,'source_snapshot_directory':str(snapshot),
            'source_scope':'all local witwin/maxwell Python sources and this driver; installed shared core and binary dependencies described by environment only',
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'device':torch.cuda.get_device_name(),'timing_scope':'shared GPU, not isolated throughput'}


def run_case(level,cutoffs,angular_order=12,rhs_order=6):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    radius=.1;control_radius=.3; frequency=3e8
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=radius),mw.Material(eps_r=2.5,sigma_e=.003))],sources=[mw.PlaneWave()])
    prepared=mw.Simulation.mom(scene,frequency=frequency,config=mw.MoMConfig(subdivisions=level,integration=IntegrationConfig(order=4,singular_order=4))).prepare()
    handle=prepared.solver.handle
    rows=[];previous=None;previous_count=0
    for cutoff in cutoffs:
        responses=[];logs=[];scales=None;modes=None
        for angular in [angular_order,angular_order+4]:
            modes=RegularModes(cutoff,handle.exterior,angular_order=angular)
            gram=modes.normalize_control_surface(control_radius,order=angular_order+4)
            scales=gram.diagonal().real.rsqrt().cpu().numpy()
            response,log=modal_response(handle,modes,order=rhs_order)
            responses.append(response);logs.append(log)
        response=responses[-1];weighted=_norm(response,handle);norm=_spectral(weighted)
        row={'cutoff':cutoff,'mode_count':modes.count,'added_column_count':modes.count-previous_count,
             'angular_orders':[angular_order,angular_order+4], 'angular_relative_operator_error':float(_spectral(_norm(responses[0]-response,handle))/norm),
             'prefix_relative_operator_error':None if previous is None else float(_spectral(_norm(response[:,:previous_count]-previous,handle))/norm),
             'added_response_relative_operator_norm':None if previous is None else float(_spectral(weighted[:,previous_count:])/norm),
             'solver_logs':logs}
        rows.append(row);previous=response;previous_count=modes.count
        print(json.dumps({'level':level,'cutoff':cutoff,'columns':modes.count,'angular_error':row['angular_relative_operator_error'],'added_response_norm':row['added_response_relative_operator_norm']}),flush=True)
    mu,w=np.polynomial.legendre.leggauss(12);phi=np.arange(24)*2*np.pi/24
    z,p=np.meshgrid(mu,phi,indexing='ij');directions=np.stack((np.sqrt(1-z*z)*np.cos(p),np.sqrt(1-z*z)*np.sin(p),z),-1).reshape(-1,3)
    weights=np.repeat(w,24)*2*np.pi/24
    reference=reference_modal_far(modes.ordering,directions,radius,handle.exterior,handle.interior,scales)
    query=torch.as_tensor(directions,device=handle.device,dtype=torch.float64)
    numeric=np.stack([far_field(handle,response[:,i].contiguous(),query)[0].detach().cpu().numpy() for i in range(modes.count)],1)
    err=np.sqrt(np.sum(weights[:,None]*np.sum(abs(numeric-reference)**2,-1),axis=0));refnorm=np.sqrt(np.sum(weights[:,None]*np.sum(abs(reference)**2,-1),axis=0))
    per_column=[{'l':l,'m':m,'kind':kind,'reference_weighted_L2_V_per_sqrt_W':float(refnorm[i]),'absolute_weighted_L2_error_V_per_sqrt_W':float(err[i]),'relative_weighted_L2_error':float(err[i]/refnorm[i]) if refnorm[i]>0 else None,'reference_numerically_small':bool(refnorm[i]<1e-8)} for i,(l,m,kind) in enumerate(modes.ordering)]
    # The joint operator error is supplemental: every column remains visible.
    weighted_error=torch.as_tensor((weights[:,None,None]**.5*(numeric-reference)).transpose(0,2,1).reshape(-1,modes.count),device=handle.device)
    weighted_reference=torch.as_tensor((weights[:,None,None]**.5*reference).transpose(0,2,1).reshape(-1,modes.count),device=handle.device)
    total_error=float(_spectral(weighted_error)/_spectral(weighted_reference))
    torch.cuda.synchronize()
    return {'level':level,'edges':handle.basis.n_edges,'radius_m':radius,'frequency_Hz':frequency,'epsilon_r':[handle.interior.epsilon_r.real,handle.interior.epsilon_r.imag],
            'control_radius_m':control_radius,'source_normalization':'unit positive E/H control-surface energy W','response_metric':'impedance_balanced_surface_L2_proxy',
            'assembly_order':4,'singular_order':4,'rhs_order':rhs_order,'cutoff_calibration':rows,'mode_ordering':modes.ordering,
            'column_errors':per_column,'far_response_relative_operator_error':total_error,'directions':directions.tolist(),'solid_angle_weights':weights.tolist(),
            'numeric_far_E_V_per_sqrt_W':_encode(numeric),'reference_far_E_V_per_sqrt_W':_encode(reference),
            'peak_cuda_allocator_bytes':torch.cuda.max_memory_allocated(),'total_seconds':time.perf_counter()-start,
            'uncomputed_tail_bound_certified':False,'all_columns_below_0_5_percent':all(c['relative_weighted_L2_error'] is not None and c['relative_weighted_L2_error']<=.005 for c in per_column)}


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--levels',nargs='+',type=int,default=[2,3,4])
    parser.add_argument('--cutoffs',nargs='+',type=int,default=[1,3,5,7,9])
    parser.add_argument('--projection-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('benchmark/cache/mom/modal-convergence.json'))
    args=parser.parse_args(argv)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if any(b-a!=2 for a,b in zip(args.cutoffs,args.cutoffs[1:])):raise ValueError('Cutoffs must advance by two')
    report={'run_provenance':_provenance(args.output),'complete_G0':False,'scope':'complete normalized finite regular-mode sphere scattering; no unsampled tail certificate',
            'cutoff_and_projection_relative_budget':.0005,'field_relative_budget':.005,
            'plane_wave_control_projection':plane_wave_projection(args.cutoffs),'cases':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for level in ([] if args.projection_only else args.levels):
        try:case=run_case(level,args.cutoffs)
        except Exception as exc:
            report['failure']={'level':level,'exception':type(exc).__name__,'message':str(exc)}
            args.output.write_text(json.dumps(report,indent=2)+'\n');raise
        report['cases'].append(case);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'edges':case['edges'],'maximum_column_error':max(c['relative_weighted_L2_error'] for c in case['column_errors']),'operator_error':case['far_response_relative_operator_error'],'seconds':case['total_seconds']}),flush=True)


if __name__=='__main__':main()
