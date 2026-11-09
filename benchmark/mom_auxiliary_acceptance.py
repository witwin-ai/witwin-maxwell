"""Qualify the independent auxiliary-source reference and compare frozen MoM data.

All fitting/field arithmetic uses the independent CUDA point-dipole module.
MoM report files are read as immutable data; no MoM numerical functions are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from . import mom_auxiliary_reference as auxiliary
from .sie_reference import sphere_fields, sphere_far_field

BUDGET = 5e-4
CHANNELS = ('tangential_E', 'tangential_H', 'normal_D', 'normal_B')

# An explicit declaration or a hash-bound reviewed legacy contract is mandatory.
COMPARISON_CONTRACT = dict(
    frequency_hz=3e8, semiaxes_m=[.1, .08, .065], center_m=[0, 0, 0],
    orientation='axis_aligned', epsilon_r=[2.25, 0], mu_r=1, sigma_S_per_m=0,
    exterior='vacuum', phasor_convention='exp(-i omega t)',
    source=dict(kind='plane_wave', phase_radians=.23, amplitude_V_per_m=1,
                propagation=[0, 0, 1], polarization=[1, 0, 0]))


def validate_mom_input(path, reviewed_manifest=None):
    """Read immutable MoM data; reject undeclared or unreviewed physical inputs.

    Sidecars are explicit reviewer assertions, not automatic physical inference.
    Their report and source bytes are verified before accepting that assertion.
    """
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    report = json.loads(data)
    identity = dict(path=str(path), sha256=digest, bytes=len(data))
    contract = report.get('physical_contract')
    if contract is not None:
        identity['contract_evidence'] = dict(kind='explicit_report_metadata')
    else:
        if reviewed_manifest is None:
            raise ValueError('MoM report lacks a complete physical_contract or reviewed sidecar')
        manifest_path = Path(reviewed_manifest)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        entry = manifest.get('reports', {}).get(digest)
        if not isinstance(entry, dict):
            raise ValueError('MoM report SHA256 is absent from reviewed sidecar')
        if manifest.get('schema') != 'mom-reviewed-physical-contract-v1':
            raise ValueError('Unsupported reviewed physical contract schema')
        review = entry.get('review', {})
        if not review.get('reviewed_by') or not review.get('evidence_notes'):
            raise ValueError('Sidecar must identify its reviewer and physical source evidence')
        hashes = entry.get('source_sha256', {})
        required = {'benchmark/mom_ellipsoid_convergence.py', 'benchmark/mom_fdtd.py'}
        if not required.issubset(hashes):
            raise ValueError('Reviewed legacy contract must pin the driver and scene definition')
        declared = dict(report.get('source_sha256', {}))
        for segment in report.get('provenance_segments', []):
            for name, value in segment.get('source_sha256', {}).items():
                if name in declared and declared[name] != value:
                    raise ValueError('MoM report has conflicting source provenance')
                declared[name] = value
        snapshot = Path(entry.get('source_snapshot_directory', ''))
        if not snapshot.is_absolute():
            snapshot = manifest_path.parent / snapshot
        for name, expected in hashes.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or declared.get(name) != expected:
                raise ValueError('Reviewed source is not pinned by the MoM report')
            source = snapshot / relative
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError('Reviewed source snapshot is missing or has changed')
        contract = entry.get('physical_contract')
        identity['contract_evidence'] = dict(kind='reviewed_legacy_sidecar',
            manifest_path=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            review=review, source_sha256=hashes, source_snapshot_directory=str(snapshot))
    if contract != COMPARISON_CONTRACT:
        raise ValueError('MoM physical contract does not match the qualified ellipsoid reference')
    if report.get('frequency_hz') != contract['frequency_hz'] or report.get('semiaxes_m') != contract['semiaxes_m']:
        raise ValueError('MoM report header conflicts with its physical contract')
    identity['physical_contract'] = contract
    return report, identity


def power_changes(first, second):
    """Separate power convergence gate; a field bound is not a power bound."""
    return {key:abs(first['far_reference_powers'][key]/value-1)
            for key,value in second['far_reference_powers'].items()}


def encode(value):
    array = value.detach().cpu().numpy()
    return {'real':array.real.tolist(), 'imag':array.imag.tolist()}


def decode(value, device='cuda'):
    return torch.tensor(np.asarray(value['real'])+1j*np.asarray(value['imag']),device=device)


def source_snapshot(output):
    root = Path(__file__).resolve().parents[1]
    names = ('benchmark/mom_auxiliary_reference.py','benchmark/mom_auxiliary_acceptance.py','benchmark/sie_reference.py')
    data = {name:(root/name).read_bytes() for name in names}
    hashes = {name:hashlib.sha256(value).hexdigest() for name,value in data.items()}
    identity = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    destination = output.parent/'source_snapshots'/identity
    for name,value in data.items():
        target = destination/name
        if target.exists() and target.read_bytes()!=value:
            raise RuntimeError('Immutable auxiliary source snapshot changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(value)
    return dict(source_sha256=hashes,source_snapshot_directory=str(destination),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()))


def norm_error(actual, reference, weights):
    return auxiliary.weighted_relative(actual,reference,weights)


def measure(config, epsilon, axes, *, phase=.23, query_spec=None):
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    reference = auxiliary.solve_reference(axes=axes,epsilon=epsilon,phase=phase,config=config)
    values, metrics, sample_weights, sample_points = {}, {}, {}, {}
    directions, weights = auxiliary.angular_rule(6,offset=.413)
    for factor in (.3,.7,1.15,1.5,3.):
        points = directions*reference.axes*factor
        for kind in (('total',) if factor<1 else ('total','scattered')):
            fields = reference.fields(points,region='interior' if factor<1 else 'exterior',kind=kind)
            exact = sphere_fields(axes[0],reference.k0,epsilon,points.cpu().numpy(),eta=auxiliary.ETA0,
                                  field_kind=kind,cutoff=24) if len(set(axes))==1 else None
            for index,field in enumerate(('E','H')):
                key = f'shell_{factor}_{kind}_{field}'
                values[key] = fields[index]
                sample_weights[key] = weights
                sample_points[key] = points.cpu().tolist()
                if exact is not None:
                    oracle = torch.tensor(exact[index]*np.exp(-1j*phase),device=reference.axes.device)
                    metrics[key] = norm_error(fields[index],oracle,weights)
    far_directions, far_weights = auxiliary.angular_rule(12,offset=.139)
    values['far'] = reference.far_field(far_directions)
    sample_weights['far'] = far_weights
    sample_points['far'] = far_directions.cpu().tolist()
    scattering_cross_section = float(torch.sum(far_weights*torch.sum(abs(values['far'])**2,dim=-1)))
    forward = reference.far_field([[0.,0.,1.]])[0,0]
    extinction_cross_section = float(4*math.pi/reference.k0*(complex(math.cos(phase),math.sin(phase))*forward).imag)
    far_powers = dict(scattering_W=scattering_cross_section/(2*auxiliary.ETA0),
        extinction_W=extinction_cross_section/(2*auxiliary.ETA0))
    if len(set(axes))==1:
        exact = sphere_far_field(axes[0],reference.k0,epsilon,far_directions.cpu().numpy(),cutoff=24)*np.exp(-1j*phase)
        metrics['far'] = norm_error(values['far'],torch.tensor(exact,device=reference.axes.device),far_weights)
    matched = {}
    matched_powers = {}
    if query_spec is not None:
        for region in ('exterior','interior'):
            points = torch.tensor(query_spec[region+'_points_m'],device=reference.axes.device,dtype=torch.float64)
            for kind in (('total','scattered') if region=='exterior' else ('total',)):
                for name,field in zip(('E','H'),reference.fields(points,region=region,kind=kind)):
                    key = region+'_'+kind+'_'+name
                    matched[key] = field
                    if region=='exterior':
                        values[key+'_fixed'] = field[:3]
                        sample_weights[key+'_fixed'] = torch.ones(3,device=field.device,dtype=torch.float64)
                        values[key+'_control'] = field[3:]
                        sample_weights[key+'_control'] = torch.tensor(query_spec['solid_angle_weights'],device=field.device,dtype=torch.float64)
                    else:
                        values[key] = field
                        sample_weights[key] = torch.ones(len(field),device=field.device,dtype=torch.float64)
        matched['far_E'] = reference.far_field(query_spec['far_directions'])
        values['matched_far'] = matched['far_E']
        sample_weights['matched_far'] = torch.tensor(query_spec['solid_angle_weights'],device=reference.axes.device,dtype=torch.float64)
        control = torch.tensor(query_spec['exterior_points_m'][3:],device=reference.axes.device,dtype=torch.float64)
        radius = torch.linalg.vector_norm(control,dim=-1)
        normal = control/radius[:,None]
        areas = sample_weights['matched_far']*radius.square()
        es,hs = matched['exterior_scattered_E'][3:],matched['exterior_scattered_H'][3:]
        et,ht = matched['exterior_total_E'][3:],matched['exterior_total_H'][3:]
        flux = lambda e,h: float(.5*torch.sum(areas*torch.sum(torch.linalg.cross(e,h.conj(),dim=-1).real*normal,dim=-1)))
        matched_powers = dict(scattering_W=flux(es,hs),
            extinction_W=-flux(et-es,hs)-flux(es,ht-hs),absorption_W=-flux(et,ht))
    holdout = reference.holdout()
    row = dict(config=asdict(config),epsilon_r=[epsilon.real,epsilon.imag],axes_m=list(axes),
        frequency_Hz=reference.frequency,phase=phase,solve=reference.solve_log,holdout=holdout,
        mie_errors=metrics,holdout_within_budget=max(holdout[name] for name in CHANNELS)<=BUDGET,
        mie_within_budget=all(value<=BUDGET for value in metrics.values()) if metrics else None,
        values={key:encode(value) for key,value in values.items()},
        matched_values={key:encode(value) for key,value in matched.items()},
        matched_powers=matched_powers,
        far_reference_powers=far_powers,
        cross_sections_m2=dict(scattering=scattering_cross_section,extinction=extinction_cross_section,
            absorption_balance=extinction_cross_section-scattering_cross_section),
        independent_power_closure_relative={key:abs(matched_powers[key]/far_powers[key]-1)
            for key in ('scattering_W','extinction_W')} if matched_powers else {},
        query_points_m={key:value for key,value in sample_points.items() if key!='far'},
        far_directions=far_directions.cpu().tolist(),
        matched_query_spec={key:query_spec[key] for key in ('exterior_points_m','interior_points_m','far_directions','solid_angle_weights')} if query_spec else None,
        weights={key:value.cpu().tolist() for key,value in sample_weights.items()},
        source_positions_m={'exterior':reference.exterior_sources.cpu().tolist(),'interior':reference.interior_sources.cpu().tolist()},
        fitted_moments_V={'exterior':encode(reference.exterior_moments),'interior':encode(reference.interior_moments)},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-start)
    return row, reference


def changes(first, second):
    return {key:norm_error(decode(first['values'][key]),decode(value),
                torch.tensor(second['weights'][key],device='cuda',dtype=torch.float64)) for key,value in second['values'].items()}


def compare_mom(row, cases):
    result = []
    for case in cases:
        errors = {}
        for key,value in row['matched_values'].items():
            actual, reference = decode(case['values'][key]),decode(value)
            if key.startswith('exterior'):
                errors[key+'_fixed'] = norm_error(actual[:3],reference[:3],torch.ones(3,device='cuda'))
                errors[key+'_control'] = norm_error(actual[3:],reference[3:],torch.tensor(case['solid_angle_weights'],device='cuda',dtype=torch.float64))
            else:
                weights = torch.tensor(case['solid_angle_weights'],device='cuda',dtype=torch.float64) if key=='far_E' else torch.ones(len(actual),device='cuda')
                errors[key] = norm_error(actual,reference,weights)
        af,bf = decode(case['values']['far_E']),decode(row['matched_values']['far_E'])
        rcs_a,rcs_b = 4*math.pi*torch.sum(abs(af)**2,dim=-1,keepdim=True),4*math.pi*torch.sum(abs(bf)**2,dim=-1,keepdim=True)
        weights = torch.tensor(case['solid_angle_weights'],device='cuda',dtype=torch.float64)
        power_errors = {key:abs(case['powers'][key]/row['matched_powers'][key]-1)
                        for key in ('scattering_W','extinction_W')}
        far_power_errors = {key:abs(case['powers'][key]/row['far_reference_powers'][key]-1)
                           for key in ('scattering_W','extinction_W')}
        incident_projected_power = math.pi*.1*.08/(2*auxiliary.ETA0)
        result.append(dict(subdivision=case['subdivision'],relative_errors=errors,
            weighted_rcs_relative_L2=norm_error(rcs_a,rcs_b,weights),power_relative_errors=power_errors,
            far_reference_power_relative_errors=far_power_errors,
            lossless_absorption_incident_normalized=abs(case['powers']['absorption_W'])/incident_projected_power))
    return result



def independent_queries():
    """Control sphere remains measured when no MoM data are supplied."""
    directions, weights = auxiliary.angular_rule(6, offset=.271)
    return dict(exterior_points_m=[[.2, 0., 0.], [0., .2, 0.], [0., 0., .2]]
                + (directions * .2).cpu().tolist(),
                interior_points_m=[[0., 0., 0.]],
                far_directions=directions.cpu().tolist(),
                solid_angle_weights=weights.cpu().tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mom-report',type=Path,action='append',default=[])
    parser.add_argument('--mom-contract',type=Path,help='Reviewed SHA-bound physical contracts for legacy MoM reports')
    args = parser.parse_args(argv)
    provenance = source_snapshot(args.output)
    cases = []
    inputs = []
    for path in args.mom_report:
        report, identity = validate_mom_input(path,args.mom_contract)
        inputs.append(identity)
        cases.extend(report['cases'])
    query_spec = cases[-1] if cases else independent_queries()
    if cases:
        for case in cases:
            for key in ('exterior_points_m','interior_points_m','far_directions','solid_angle_weights'):
                if case[key]!=query_spec[key]:raise ValueError('MoM cases do not share identical physical queries')
    report = dict(**provenance,budget=BUDGET,scope='Independent auxiliary electric-dipole collocation reference; empirical sampled qualification, no rigorous error bound',
        convention='SI; exp(-i omega t); vacuum exterior; passive scalar nonmagnetic homogeneous ellipsoid',
        MU0_H_per_m=auxiliary.MU0,ETA0_ohm=auxiliary.ETA0,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,
            gpu=torch.cuda.get_device_name(),precision='complex128',execution='CUDA kernels and SVD; CPU analytic oracle only'),
        shared_mu0_relative_difference=abs(auxiliary.MU0/1.25663706212e-6-1),
        comparison_scene_contract=COMPARISON_CONTRACT,
        control_query_origin='frozen_mom_reports' if cases else 'independent_control_sphere',
        input_mom_reports=inputs,sphere_cases=[],ellipsoid_cases=[],controls=[],
        physical_gate_certified=False,strict_error_bound=False)
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report))
    for epsilon in (2.25+0j,2.25+.1j):
        for order,inner in ((4,.4),(6,.4),(8,.4),(10,.4),(14,.6),(16,.6)):
            row,_ = measure(auxiliary.AuxiliaryConfig(order=order,inner_factor=inner),epsilon,(.1,.1,.1))
            report['sphere_cases'].append(row);save()
            print(json.dumps(dict(stage='sphere',order=order,epsilon=[epsilon.real,epsilon.imag],mie_max=max(row['mie_errors'].values()),holdout=row['holdout'])),flush=True)
            torch.cuda.empty_cache()
    for order in (8,10,12,14,16):
        row,reference = measure(auxiliary.AuxiliaryConfig(order=order),2.25+0j,(.1,.08,.065),query_spec=query_spec)
        if report['ellipsoid_cases']:
            row['changes_from_previous'] = changes(report['ellipsoid_cases'][-1],row)
            row['power_changes_from_previous'] = power_changes(report['ellipsoid_cases'][-1],row)
        report['ellipsoid_cases'].append(row);save()
        print(json.dumps(dict(stage='ellipsoid',order=order,holdout=row['holdout'],change=max(row.get('changes_from_previous',{'none':1}).values()))),flush=True)
        if order==16:
            row['holdout_refined'] = reference.holdout(order=43,offset=.613)
        del reference
        torch.cuda.empty_cache()
    baseline = report['ellipsoid_cases'][-1]
    for config in (auxiliary.AuxiliaryConfig(inner_factor=.55,outer_factor=2),
                   auxiliary.AuxiliaryConfig(inner_factor=.65,outer_factor=2.5),
                   auxiliary.AuxiliaryConfig(rcond=1e-10),auxiliary.AuxiliaryConfig(rcond=1e-13)):
        row,_ = measure(config,2.25+0j,(.1,.08,.065),query_spec=query_spec)
        row['changes_from_baseline'] = changes(row,baseline)
        row['power_changes_from_baseline'] = power_changes(row,baseline)
        report['controls'].append(row);save()
        print(json.dumps(dict(stage='control',config=asdict(config),holdout=row['holdout'],change=max(row['changes_from_baseline'].values()))),flush=True)
        torch.cuda.empty_cache()
    qualified = all(row['mie_within_budget'] and row['holdout_within_budget'] for row in report['sphere_cases'] if row['config']['order']>=8)
    qualified = qualified and all(row['holdout_within_budget'] for row in report['ellipsoid_cases'][-2:])
    qualified = qualified and max(baseline['changes_from_previous'].values())<=BUDGET
    qualified = qualified and max(baseline['power_changes_from_previous'].values())<=BUDGET
    qualified = qualified and max(baseline['holdout_refined'][name] for name in CHANNELS)<=BUDGET
    qualified = qualified and all(row['holdout_within_budget'] and max(row['changes_from_baseline'].values())<=BUDGET and max(row['power_changes_from_baseline'].values())<=BUDGET for row in report['controls'])
    qualified = qualified and all(value<=BUDGET for value in baseline['independent_power_closure_relative'].values())
    report['empirical_sampled_reference_qualified'] = qualified
    report['mom_comparisons'] = compare_mom(baseline,cases) if qualified and cases else []
    save()
    print(json.dumps(dict(empirical_sampled_reference_qualified=qualified,mom_comparisons=report['mom_comparisons'])),flush=True)
    return 0 if qualified else 1


if __name__=='__main__':
    raise SystemExit(main())
