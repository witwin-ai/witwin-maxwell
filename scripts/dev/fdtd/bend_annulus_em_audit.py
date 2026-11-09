"""Independent E/H Hankel port reconstruction from retained native spectra.

This diagnostic preserves the original H-only result and never fits vacuum or
adjusts a source amplitude. The declared reference planes are u=+/-R/6.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import hankel1,hankel2,h1vp,h2vp
import torch
from scripts.dev.fdtd.copper_cylinder_reference import EPS_0,MU_0,shell_scattering


C_0=1/math.sqrt(EPS_0*MU_0)


def em_hankel_basis(radius,reference_radius,frequency,order):
    """Rows are (covariant E_s, H_z); columns outward/inward Hankel waves.

    For kappa=-1/R, e_s=-e_phi and e_u=e_r. With exp(-i*omega*t),
    E_s=+i*h_s/(omega*epsilon)*dH_z/dr. Hence positive radial power
    is -Re(E_s*conj(H_z))/2 times chart area.
    """
    omega=2*math.pi*frequency;k=omega/C_0;argument=k*radius
    magnetic=np.asarray((hankel1(order,argument),hankel2(order,argument)))
    electric=1j*(radius/reference_radius)*k/(omega*EPS_0)*np.asarray((h1vp(order,argument),h2vp(order,argument)))
    return np.stack((electric,magnetic))


def project_em(electric,magnetic,radius,reference_radius,frequency,order):
    return np.linalg.solve(em_hankel_basis(radius,reference_radius,frequency,order),np.asarray((electric,magnetic)))


def modal_power_unit(reference_radius,frequency,chart_area):
    return chart_area/(math.pi*(2*math.pi*frequency)*EPS_0*reference_radius)


def encoded(value):
    value=np.asarray(value)
    return dict(real=value.real.tolist(),imag=value.imag.tolist())


def decoded(value):
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])


def snapshot_ports(path,*,reference_radius,frequency,spacing,order):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['solver_stats']['normalize_source']:
        raise ValueError('The raw E/H diagnostic requires physical unnormalized field spectra.')
    field=np.asarray(payload['fields']['EX'])
    if field.ndim==4:field=field[0]
    count=math.ceil(reference_radius/spacing)
    physical=np.linspace(-reference_radius/2,reference_radius/2,count+1,dtype=np.float64)
    nodes=np.r_[physical[0]-(physical[1]-physical[0])*np.arange(12,0,-1),physical,
        physical[-1]+(physical[-1]-physical[-2])*np.arange(1,13)]
    halves=(nodes[1:]+nodes[:-1])/2
    samples={}
    positions=set()
    for name,record in payload['monitors'].items():
        if not name.startswith('magnetic_profile_'):continue
        i,j,k=record['field_indices']['Hz']
        positions.add((i,k))
        samples[j]=complex(record['components']['Hz'])
    if len(positions)!=1:raise ValueError('H profiles must retain one declared native s/z sample.')
    if not samples:raise ValueError('The retained result has no actual H-profile samples.')
    i,k=next(iter(positions));magnetic_indices=np.asarray(sorted(samples))
    magnetic=np.asarray([samples[j]for j in magnetic_indices]);coordinates=halves[magnetic_indices]
    electric=field[i,:,k]
    rows=[]
    for u in (-reference_radius/6,reference_radius/6):
        if not nodes[0]<=u<=nodes[-1] or not coordinates[0]<=u<=coordinates[-1]:
            raise ValueError('Declared E/H reference plane lacks retained native neighbors.')
        # Profiles avoid the foil and source supports. Interpolation uses each
        # component's actual normal stagger, never a shared plotting grid.
        low=np.searchsorted(coordinates,u,side='right')-1
        if low<0 or low+1>=len(coordinates) or coordinates[low]*coordinates[low+1]<0:
            raise ValueError('H interpolation cannot bridge the foil or leave the retained profile.')
        e=np.interp(u,nodes,electric.real)+1j*np.interp(u,nodes,electric.imag)
        h=np.interp(u,coordinates,magnetic.real)+1j*np.interp(u,coordinates,magnetic.imag)
        amplitudes=project_em(e,h,reference_radius+u,reference_radius,frequency,order)
        rows.append(dict(u_m=u,electric_s=encoded(e),magnetic_z=encoded(h),amplitudes=encoded(amplitudes),
            physical_flux_per_chart_area=float(-.5*np.real(e*np.conj(h)))))
    norm=modal_power_unit(reference_radius,frequency,(4*spacing)**2)
    inner=decoded(rows[0]['amplitudes']);outer=decoded(rows[1]['amplitudes'])
    incoming=np.asarray((inner[0],outer[1]));outgoing=np.asarray((inner[1],outer[0]))
    heat=sum(float(record['surface_power'][0])for record in payload['metadata'].get('conductor_losses',{}).values())
    midpoint=math.cos(math.pi*frequency*payload['solver_stats']['dt'])
    incident=norm*np.vdot(incoming,incoming).real
    net_flux=(rows[0]['physical_flux_per_chart_area']-rows[1]['physical_flux_per_chart_area'])*(4*spacing)**2
    return incoming,outgoing,dict(snapshot=str(path),snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ports=rows,incoming=encoded(incoming),outgoing=encoded(outgoing),incoming_physical_power_w=float(incident),
        incoming_midpoint_power_w=float(incident*midpoint),copper_heat_w=heat,
        physical_flux_difference_w=float(net_flux),midpoint_flux_difference_w=float(net_flux*midpoint),
        direct_heat_absorption=float(heat/(incident*midpoint)),
        midpoint_flux_heat_relative_error=None if heat==0 else float(abs(net_flux*midpoint-heat)/heat),
        dt_s=payload['solver_stats']['dt'],normalize_source=False)


def audit(report_path,output):
    paths=(Path(__file__).resolve(),Path(shell_scattering.__code__.co_filename).resolve(),report_path)
    def binding():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths}
    before=binding()
    report=json.loads(report_path.read_text());rows=[]
    if report.get('source_binding_before') is None or report['source_binding_before']!=report.get('source_binding_after'):
        raise ValueError('E/H port analysis requires complete matching native source identities.')
    for original in report['rows']:
        if original['polarization']!='te':raise ValueError('This independent EM reconstruction currently derives TE only.')
        row={'order':original['order'],'polarization':'te','original_hz_only':{key:original[key]for key in (
            'vacuum_S_absolute_error','reflection_absolute_error','transmission_relative_error','absorption_relative_error')}}
        measured={}
        for label in ('vacuum','film'):
            runs=[snapshot_ports(Path(run['snapshot']),reference_radius=report['radius_m'],frequency=report['frequency_hz'],
                spacing=report['spacing_m'],order=original['order'])for run in original[label]['runs']]
            inputs=np.stack([run[0]for run in runs],-1);outputs=np.stack([run[1]for run in runs],-1)
            matrix=np.linalg.solve(inputs.T,outputs.T).T;measured[label]=matrix
            row[label]=dict(S=encoded(matrix),runs=[run[2]for run in runs],incoming_condition_number=float(np.linalg.cond(inputs)))
        sigma=5.8e7;omega=2*math.pi*report['frequency_hz'];delta=math.sqrt(2/(omega*MU_0*sigma))
        exact=shell_scattering(report['radius_m'],delta,report['frequency_hz'],sigma,order=original['order'],polarization='te')
        actual=measured['film'];reference_absorption=1-(abs(exact)**2).sum(0);absorption=1-(abs(actual)**2).sum(0)
        row.update(reference_S=encoded(exact),reference_absorption=reference_absorption.tolist(),actual_absorption=absorption.tolist(),
            vacuum_S_absolute_error=float(abs(measured['vacuum']-np.asarray(((0,1),(1,0)))).max()),
            reflection_absolute_error=float(abs(np.diag(actual-exact)).max()),
            transmission_relative_error=float(max(abs(actual[0,1]-exact[0,1])/abs(exact[0,1]),abs(actual[1,0]-exact[1,0])/abs(exact[1,0]))),
            absorption_relative_error=float(np.max(abs(absorption-reference_absorption)/reference_absorption)))
        absorption_matrix=np.eye(2)-exact.conj().T@exact
        for run in row['film']['runs']:
            a=decoded(run['incoming']);expected=float(np.vdot(a,absorption_matrix@a).real/np.vdot(a,a).real)
            run['coherent_reference_absorption']=expected
            run['direct_heat_relative_error']=abs(run['direct_heat_absorption']-expected)/expected
        row['direct_heat_relative_error']=max(run['direct_heat_relative_error']for run in row['film']['runs'])
        row['passed_original_physical_S_gates']=(row['vacuum_S_absolute_error']<.003 and row['reflection_absolute_error']<.003
            and row['transmission_relative_error']<.03 and row['absorption_relative_error']<.03)
        rows.append(row)
    after=binding()
    for row in rows:
        for label in ('vacuum','film'):
            for run in row[label]['runs']:
                if hashlib.sha256(Path(run['snapshot']).read_bytes()).hexdigest()!=run['snapshot_sha256']:
                    raise RuntimeError('A retained native snapshot changed during E/H projection.')
    if before!=after:raise RuntimeError('Independent E/H projection controller changed during analysis.')
    result=dict(schema='native-annulus-independent-em-port-audit/1',source_report=str(report_path),
        source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),analysis_binding_before=before,analysis_binding_after=after,
        plane_contract='u=+/-R/6; actual native Es nodes and Hz halves interpolated independently away from foil/source; no amplitude fit or vacuum subtraction',
        physical_reference='Analytic cylindrical TE E/H modes, physical frequency/order, full Wronskian power; native midpoint cos(pi*f*dt) recorded separately',rows=rows)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('report',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();audit(args.report,args.output)
