"""Evaluate measured rows against the prospectively committed local budget."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from benchmark.mom_fdtd import control_sphere
from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
from witwin.maxwell.postprocess.nfft import NearFieldFarFieldTransformer
from witwin.maxwell.postprocess.stratton_chu import equivalent_surface_currents_from_surface_samples

def tensor(value):
    return torch.tensor(value,device='cuda',dtype=torch.float64)

def complex_value(value):
    return tensor(value['real'])+1j*tensor(value['imag'])

def norm(value,weights=None):
    if weights is None:
        return float(torch.linalg.vector_norm(value))
    squared = value.abs().square()
    if squared.ndim > 1:
        squared = squared.sum(-1)
    return float((weights*squared).sum().sqrt())

def metric(name,value,limit,units='dimensionless'):
    return {'quantity':name,'value':float(value),'limit':limit,'units':units,'passed':float(value)<=limit}

def field_metrics(row):
    checks = [metric('empty_'+component,row['empty_background_'+component+'_shape_error'],.005)
              for component in ('E','H')]
    for component in ('E','H'):
        observed = complex_value(row['fields']['total_'+component])
        reference = complex_value(row['mom_fields']['total_'+component])
        for region,indices in [('exterior',slice(None,-1)),('interior',slice(-1,None))]:
            checks.append(metric(region+'_total_'+component,
                norm(observed[indices]-reference[indices])/norm(reference[indices]),.02))
        if row['family'] == 'no_contrast':
            checks.append(metric('zero_scattered_'+component,
                row['incident_normalized_scattered_errors'][component],.005))
        else:
            checks.append(metric('scattered_'+component,row['relative_errors']['scattered_'+component],.05))
    return checks

def observable_metrics(row):
    checks = [metric('empty_'+component,row['empty_background_'+component+'_shape_error'],.005)
              for component in ('E','H')]
    zero = row['family'] == 'no_contrast'
    fdtd,mom = row['powers']['fdtd'],row['powers']['mom']
    projected_area = math.pi*.1*(.08 if row['family']=='ellipsoid' else .1)
    incident_power = projected_area/(2*math.sqrt(MU_0/EPSILON_0))
    if zero:
        checks.append(metric('zero_far',row['complex_far_absolute_l2_error_m']/(math.sqrt(4*math.pi)*.1),.005))
        weights = tensor(row['solid_angle_weights'])
        rcs = tensor(row['fdtd_rcs_m2'])-tensor(row['native_mom_rcs_m2'])
        checks.append(metric('zero_rcs',norm(rcs,weights)/(math.sqrt(4*math.pi)*math.pi*.1**2),.005))
    else:
        checks += [metric('complex_far',row['complex_far_relative_error'],.05),
                   metric('rcs',row['rcs_relative_error'],.10)]
    for quantity in ('scattered_flux_W','extinction_interference_W'):
        value = abs(fdtd[quantity]-mom[quantity])/(incident_power if zero else abs(mom[quantity]))
        checks.append(metric(quantity,value,.005 if zero else .05))
    if row['family'] != 'lossy_sphere':
        for backend in ('fdtd','mom'):
            checks.append(metric(backend+'_zero_absorption',abs(row['powers'][backend]['absorption_inward_flux_W'])/incident_power,.005))
    return checks

def common_control_far(row,directions):
    normals = tensor(row['far_directions'])
    radius = row['control_surface']['radius_m']
    surface = equivalent_surface_currents_from_surface_samples(frequency=3e8,
        points=radius*normals,normals=normals,areas=radius**2*tensor(row['solid_angle_weights']),
        E=complex_value(row['fields']['scattered_E'])[3:],H=complex_value(row['fields']['scattered_H'])[3:])
    value = NearFieldFarFieldTransformer(surface,device='cuda',c=C_0,eps0=EPSILON_0,mu0=MU_0).transform_directions(directions)['E']
    return value*torch.exp(tensor(-2*math.pi*3e8/C_0)*1j)

def assess(path):
    data = json.loads(path.read_text())
    output = {'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':[]}
    for row in data['cases']:
        identity = {key:row[key] for key in ('family','spacing_m','mom_subdivisions','parameter','value') if key in row}
        if row['status'] != 'measured':
            output['rows'].append({**identity,'status':'failed','error':row.get('error')})
            continue
        if 'mom_fields' in row:
            checks = field_metrics(row)
        elif 'powers' in row:
            checks = observable_metrics(row)
        elif 'fdtd_volume_absorption_W' in row:
            checks = [metric('independent_volume_loss',row['relative_volume_power_difference'],.05),
                      metric('volume_quadrature_change',row['mom_volume_quadrature_relative_change'],.01)]
        else:
            checks = []
        output['rows'].append({**identity,'status':'measured','checks':checks,
            'assessment_status':'assessed' if checks else 'unassessed',
            'all_recorded_limits_passed':all(check['passed'] for check in checks) if checks else None})
    if data.get('observables') == 'closed_surface_angular_and_radius_sensitivity':
        _,directions,_,weights = control_sphere(order=10)
        controls = []
        rows = [row for row in data['cases'] if row['status']=='measured']
        for parameter,reference_value in [('angular_order',10),('radius',.24)]:
            selected = [row for row in rows if row['parameter']==parameter]
            reference = next((row for row in selected if row['value']==reference_value),None)
            if reference is None:
                continue
            reference_far = common_control_far(reference,directions)
            for row in selected:
                if row is reference:
                    continue
                far = common_control_far(row,directions)
                checks = [metric('common_direction_complex_far',norm(far-reference_far,weights)/norm(reference_far,weights),.01)]
                for backend in ('fdtd','mom'):
                    for quantity in ('scattered_flux_W','extinction_interference_W','scattered_far_integral_W'):
                        value,expected = row['powers'][backend][quantity],reference['powers'][backend][quantity]
                        checks.append(metric(backend+'_'+quantity,abs(value-expected)/abs(expected),.01))
                controls.append({'parameter':parameter,'value':row['value'],'reference_value':reference_value,'checks':checks})
        output['independent_surface_controls'] = controls
    sweeps = data.get('sensitivity_sweeps',[])
    controls = []
    for parameter,reference_value in [('pml',data.get('reference_pml_m',.25)),('dt_fraction',.4),('cycles',28)]:
        selected = [row for row in sweeps if row['parameter']==parameter and row['status']=='measured']
        reference = next((row for row in selected if row['value']==reference_value),None)
        if reference is None:
            continue
        for row in selected:
            if row is reference:
                continue
            checks = []
            for quantity in ('total_E','total_H','scattered_E','scattered_H'):
                value,expected = complex_value(row['fields'][quantity]),complex_value(reference['fields'][quantity])
                checks.append(metric(quantity,norm(value-expected)/norm(expected),.01))
            controls.append({'parameter':parameter,'value':row['value'],'reference_value':reference_value,'checks':checks})
    if controls:
        output['independent_fdtd_controls'] = controls
    return output

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    result = {'budget_commit':'6fe3897','budget_document':'docs/reference/mom_cross_acceptance_budget.md',
        'scope':'local subset measurements; independent MoM reference qualification remains separate',
        'H2_certified':False,'G0_certified':False,'assessment_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reports':[assess(path) for path in args.inputs]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(str(args.output),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
