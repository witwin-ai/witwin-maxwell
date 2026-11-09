"""Compare independently executed cylindrical PCB refinements without fitted phase."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def complex_value(value):
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])


def validate_report(report):
    if report['schema']!='independent-cylindrical-bent-pcb-reference/1':
        raise ValueError('Expected a completed independent cylindrical reference report.')
    if report['source_binding_before']!=report['source_binding_after']:
        raise ValueError('The independent execution source changed during a run.')
    if report['execution_driver_sha256']!=report['analysis_driver_sha256']:
        raise ValueError('Forward and retained-field analysis controllers differ.')
    current=complex_value(report['current'])
    if current.shape!=()or not np.isfinite(current)or abs(current)==0:
        raise ValueError('Declared current phasor must be finite and nonzero.')
    for component in ('E_theta','E_phi'):
        field=complex_value(report[component])
        if field.shape!=(len(report['theta']),len(report['phi']))or not np.isfinite(field).all():
            raise ValueError('Retained complex far fields do not match the declared angular grid.')
    if max(abs(value+1)for value in report['source_normalized_radial_flux'])>1e-12:
        raise ValueError('Impressed radial current does not integrate to the declared signed ampere.')


def compare_pair(coarse,fine):
    """Known-current transfer fields retain their absolute complex phase and scale."""
    for report in (coarse,fine):validate_report(report)
    for key in ('curvature_m_inverse','frequency_hz','duration_s','metal_model',
                'known_current_peak_a','phase_center_world_m','reference_phase_center_cylindrical',
                'theta','phi','source_identity'):
        if coarse[key]!=fine[key]:raise ValueError('Unmatched physical/reference input: '+key)
    for key in ('source','metal','vacuum','current_amplitude','duration','field_dft_oversampling',
                'grid_construction','canonical_grid_faces','source_helper_sha256'):
        if coarse['raw_input_manifest'][key]!=fine['raw_input_manifest'][key]:
            raise ValueError('Unmatched declared source/material/grid construction: '+key)
    if not coarse['spacing_m']>fine['spacing_m']>0:
        raise ValueError('Reports must be ordered from coarse to fine positive spacing.')
    currents=[complex_value(report['current'])for report in (coarse,fine)]
    fields=[np.stack([complex_value(report[key])/current for key in ('E_theta','E_phi')])
            for report,current in zip((coarse,fine),currents)]
    theta=np.asarray(fine['theta']);phi=np.asarray(fine['phi'])
    def integrate(value):return np.trapezoid(np.trapezoid(value,phi,axis=-1)*np.sin(theta),theta,axis=-1)
    denominator=integrate((abs(fields[1])**2).sum(axis=0))
    if denominator<=0:raise ValueError('Fine far-field transfer has zero angular norm.')
    complex_change=float(np.sqrt(integrate((abs(fields[0]-fields[1])**2).sum(axis=0))/denominator))
    impedances=[complex_value(report['impedance'])for report in (coarse,fine)]
    radiation=[report['radiated_spectral_w_s2']/abs(current)**2 for report,current in zip((coarse,fine),currents)]
    return dict(coarse_spacing_m=coarse['spacing_m'],fine_spacing_m=fine['spacing_m'],
        impedance_relative_change=float(abs(impedances[0]-impedances[1])/abs(impedances[1])),
        complex_radiation_transfer_relative_l2_change=complex_change,
        radiated_power_per_amp_squared_relative_change=float(abs(radiation[0]-radiation[1])/abs(radiation[1])),
        directivity_relative_change=abs(coarse['directivity']-fine['directivity'])/abs(fine['directivity']),
        field_normalization='Declared known impressed current, no fitted amplitude, phase or pattern rotation.')


def run(paths,output):
    identity=lambda:{str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in (*paths,Path(__file__))}
    before=identity();reports=[json.loads(path.read_text())for path in paths]
    for report in reports:validate_report(report)
    comparisons=[compare_pair(coarse,fine)for coarse,fine in zip(reports,reports[1:])]
    after=identity()
    if before!=after:raise RuntimeError('Comparison inputs/controller changed during analysis.')
    rows=[]
    for report in reports:
        rows.append({key:report[key]for key in ('spacing_m','dt_s','steps','impedance','current',
            'directivity','accepted_spectral_w_s2','radiated_spectral_w_s2','source_edge_count')})
    result=dict(schema='independent-bent-pcb-spatial-trend/1',python=sys.version,numpy_version=np.__version__,source_before=before,source_after=after,
        rows=rows,adjacent_comparisons=comparisons,
        qualification='Independent PEC spatial trends only. Paired native 2.5 mm, finite-Cu heat, duration/PML and general cut-surface gates remain open.')
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args();print(json.dumps(run(arguments.reports,arguments.output),indent=2))
