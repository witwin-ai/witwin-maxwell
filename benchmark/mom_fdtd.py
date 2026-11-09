"""Same-Scene CUDA MoM/FDTD convergence and empty-background phasor audit.

Run ``python -m benchmark.mom_fdtd --output /tmp/mom_fdtd.json`` for the full
three-resolution diagnostic. Independent modes measure common far-field/RCS,
surface power, volume dissipation, and sensitivity to numerical controls.
This driver does not certify H2; acceptance requires an independently frozen
joint error budget and demonstrated convergence.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import itertools
import tempfile
import hashlib
import inspect
import marshal
import subprocess
import sys
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.physics import compile_medium, incident_fields

FREQUENCY = 3e8
POINTS = ((.21, .08, .03), (-.18, .04, -.05), (.04, -.19, .06), (0., 0., 0.))
FAMILIES = ('no_contrast', 'lossless_sphere', 'lossy_sphere', 'ellipsoid')
INTEGRATION = IntegrationConfig(order=4,singular_order=4)


def add_query_stencils(scene, points=POINTS):
    """Declare PointMonitors at every component's true trilinear stencil node."""
    prepared = prepare_scene(scene)
    seen = set()
    for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        axes = component_coords(prepared, component)
        for point in points:
            lower = [int(np.searchsorted(axis, value, side='right').clip(1, len(axis)-1)-1)
                     for axis, value in zip(axes, point)]
            for corner in itertools.product((0, 1), repeat=3):
                index = tuple(a+b for a, b in zip(lower, corner))
                key = (component, *index)
                if key not in seen:
                    seen.add(key)
                    position = tuple(float(axis[i]) for axis, i in zip(axes, index))
                    scene.add_monitor(mw.PointMonitor('query_'+'_'.join(map(str, key)), position, fields=(component,)))
    return scene


def make_scene(family, spacing, *, pml=.2, empty=False, query_points=POINTS):
    if family not in FAMILIES:
        raise ValueError(f'Unknown family {family}')
    extent = .6
    boundary = mw.BoundarySpec.pml(num_layers=math.ceil(pml/spacing))
    source = mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0),
                          source_time=mw.CW(frequency=FREQUENCY, amplitude=1., phase=.23),
                          injection=mw.TFSF(bounds=((-0.3, .3), (-.3, .3), (-.3, .3))))
    structures = []
    if not empty:
        geometry = (mw.Ellipsoid(radii=(.1, .08, .065)) if family == 'ellipsoid'
                    else mw.Sphere(radius=.1))
        epsilon = 1. if family == 'no_contrast' else 2.25
        sigma = .004 if family == 'lossy_sphere' else 0.
        structures = [mw.Structure(geometry=geometry, material=mw.Material(eps_r=epsilon, sigma_e=sigma))]
    scene = mw.Scene(domain=mw.Domain(bounds=((-extent, extent),)*3),
                     grid=mw.GridSpec.uniform(spacing), boundary=boundary,
                     structures=structures, sources=[source], device='cuda')
    return add_query_stencils(scene, query_points)


def physical_scene_without_stencils(scene):
    """Remove only this driver's auxiliary Yee sampling from a shallow Scene copy."""
    physical = copy.copy(scene)
    physical.monitors = [monitor for monitor in scene.monitors if not monitor.name.startswith('query_')]
    return physical


def _relative(actual, expected):
    return float(torch.linalg.vector_norm(actual-expected) / torch.linalg.vector_norm(expected).clamp_min(1e-14))


def _weighted_norm(value, weights):
    squared = value.abs().square()
    if squared.ndim > 1:
        squared = squared.sum(-1)
    return float((weights*squared).sum().sqrt())


def _complex_data(tensor):
    return {'real': tensor.real.detach().cpu().tolist(), 'imag': tensor.imag.detach().cpu().tolist()}


def _resume_cases(previous, expected, cycles):
    rows = []
    for original in previous['cases']:
        if original['status'] != 'measured':
            continue
        key = (original['family'],original['spacing_m'],original['mom_subdivisions'])
        if (key not in expected or original.get('transient_cycles') != cycles or original.get('steady_cycles') != cycles
                or original.get('pml_m') != .2 or original.get('dt_fraction_of_uniform_cfl') != .8
                or original.get('mom_integration') != asdict(INTEGRATION)):
            raise ValueError('Resume cases do not match the requested family, grid, MoM integration, mesh, or time window')
        row = dict(original)
        if 'complex_far_relative_error' in row:
            row.setdefault('error_norm_version','legacy_unweighted_sample_L2')
        row.setdefault('zero_reference_policy','legacy_clamped_relative_denominator')
        rows.append(row)
    return rows


def _resume_fdtd_sweeps(previous, spacing, cycles):
    allowed = {'pml':(.15,.2,.25),'dt_fraction':(.8,.6,.4),'cycles':(12,20,28)}
    rows = []
    for row in previous.get('sensitivity_sweeps',[]):
        if row['status'] != 'measured':
            continue
        parameter,value = row['parameter'],row['value']
        settings = {'pml':.2,'dt_fraction':.8,'cycles':cycles}
        if parameter not in allowed or value not in allowed[parameter]:
            raise ValueError('Resume sensitivity sweep parameter or value differs')
        settings[parameter] = value
        expected = {'family':'lossless_sphere','spacing_m':spacing,'pml_m':settings['pml'],
            'dt_fraction_of_uniform_cfl':settings['dt_fraction'],'transient_cycles':settings['cycles'],
            'steady_cycles':settings['cycles'],'frequency_hz':FREQUENCY}
        if any(row.get(key) != value for key,value in expected.items()):
            raise ValueError('Resume sensitivity sweep uses different fixed settings')
        rows.append(dict(row))
    return rows


def fdtd_pair(family, spacing, *, pml=.2, dt_fraction=.8, cycles=16, query_points=POINTS, full_field_dft=False, runtime_results=None):
    settings = {'family': family, 'spacing_m': spacing, 'pml_m': pml,
                'dt_fraction_of_uniform_cfl': dt_fraction, 'transient_cycles': cycles,
                'steady_cycles': cycles, 'frequency_hz': FREQUENCY, 'spectral_window': 'hanning',
                'pml_layers': math.ceil(pml/spacing), 'dft_exponent': 'exp(+i omega t)',
                'physical_phasor': 'exp(-i omega t)', 'conjugation_applied': False,
                'source': {'direction': [0,0,1], 'polarization': [1,0,0], 'amplitude_V_per_m': 1., 'CW_phase_radians': .23},
                'physical_domain_bounds_m': [[-.6,.6]]*3, 'TFSF_bounds_m': [[-.3,.3]]*3}
    scene = make_scene(family, spacing, pml=pml, query_points=query_points)
    empty = make_scene(family, spacing, pml=pml, empty=True, query_points=query_points)
    dt = dt_fraction * spacing/(C_0*math.sqrt(3))
    run_time = mw.TimeConfig(transient_cycles=cycles, steady_cycles=cycles, dt=dt)
    sampler = mw.SpectralSampler(window='hanning', normalize_source=False)
    start = time.perf_counter()
    reference = mw.Simulation.fdtd(empty, frequency=FREQUENCY, run_time=run_time,
                                  spectral_sampler=sampler, cuda_graph=False, full_field_dft=full_field_dft).run()
    total = mw.Simulation.fdtd(scene, frequency=FREQUENCY, run_time=run_time,
                              spectral_sampler=sampler, cuda_graph=False, full_field_dft=full_field_dft).run()
    points = torch.tensor(query_points, dtype=torch.float64, device='cuda')
    reference_e, reference_h = reference.query_fields(points, region_id='exterior', field_kind='total')
    medium = compile_medium(mw.Material(), FREQUENCY, background=True)
    analytic_e, analytic_h = incident_fields(scene.sources[0], points, medium)
    # One common complex E-derived gain calibrates source and DFT origin;
    # applying that same gain to H preserves the independent E/H phase test.
    gain = (analytic_e.conj()*reference_e).sum() / analytic_e.abs().square().sum()
    if abs(complex(gain)) < 1e-12:
        raise RuntimeError('Empty-background E amplitude is too small for phasor calibration')
    exterior_e, exterior_h = total.query_fields(points[:-1], region_id='exterior', field_kind='total')
    interior_e, interior_h = total.query_fields(points[-1:], region_id='interior', field_kind='total')
    scattered_e, scattered_h = total.query_fields(points[:-1], region_id='exterior', field_kind='scattered', reference=reference)
    fields = {'total_E': torch.cat((exterior_e, interior_e))/gain,
              'total_H': torch.cat((exterior_h, interior_h))/gain,
              'scattered_E': scattered_e/gain, 'scattered_H': scattered_h/gain}
    report = {**settings, 'elapsed_seconds': time.perf_counter()-start,
              'dt_seconds': total.stats()['dt'], 'time_steps': total.stats()['time_steps'],
              'actual_primal_spacing_range_m': [float(total.prepared_scene.dx_primal64.min()), float(total.prepared_scene.dx_primal64.max())],
              'common_phasor_gain': [float(gain.real), float(gain.imag)],
              'empty_background_E_shape_error': _relative(reference_e/gain, analytic_e),
              'empty_background_H_shape_error': _relative(reference_h/gain, analytic_h),
              'sampling': 'trilinear_at_each_component_Yee_coordinates',
              'H_time_stagger': 'native_observer_half_step_retard; no second correction',
              'scattering_definition': 'total_minus_matched_empty_background',
              'fields': {name: _complex_data(value) for name, value in fields.items()}}
    if runtime_results is not None:
        runtime_results.update(total=total, reference=reference, common_phasor_gain=gain)
    report['auxiliary_sampling'] = 'FDTD grouped Yee stencil monitors; MoM queries physical points directly'
    return physical_scene_without_stencils(scene), fields, report


def compare_case(family, spacing, subdivisions, *, pml=.2, dt_fraction=.8, cycles=16, order=4):
    scene, numerical, report = fdtd_pair(family, spacing, pml=pml, dt_fraction=dt_fraction, cycles=cycles)
    try:
        # Both solvers share physical Scene objects; Yee stencil monitors belong
        # only to FDTD interpolation, not to the requested physical queries.
        config = mw.MoMConfig(subdivisions=subdivisions,
                             integration=IntegrationConfig(order=order, singular_order=order),
                             query_config=mw.FieldQueryConfig(max_order=128))
        mom = mw.Simulation.mom(scene, frequency=FREQUENCY, config=config).run()
        points = torch.tensor(POINTS, dtype=torch.float64, device='cuda')
        e, h = mom.query_fields(points[:-1], region_id='exterior', field_kind='total')
        ei, hi = mom.query_fields(points[-1:], region_id='interior', field_kind='total')
        es, hs = mom.query_fields(points[:-1], region_id='exterior', field_kind='scattered')
        surface_fields = {'total_E': torch.cat((e, ei)), 'total_H': torch.cat((h, hi)),
                          'scattered_E': es, 'scattered_H': hs}
        exterior = compile_medium(mw.Material(), FREQUENCY, background=True)
        inc_e, inc_h = incident_fields(scene.sources[0], points[:-1], exterior)
        report['incident_normalized_scattered_errors'] = {
            'E': float(torch.linalg.vector_norm(numerical['scattered_E']-es) / torch.linalg.vector_norm(inc_e)),
            'H': float(torch.linalg.vector_norm(numerical['scattered_H']-hs) / torch.linalg.vector_norm(inc_h))}
        report.update({'mom_subdivisions': subdivisions, 'mom_quadrature_order': order, 'mom_query_budget': asdict(config.query_config),
                       'mom_integration':asdict(config.integration),
                       'zero_reference_policy':'analytic_zero_relative_null_use_absolute_or_incident_norm',
                       'mom_solver': mom.stats(),
                       'relative_errors': {name: (None if family == 'no_contrast' and name.startswith('scattered')
                                                 else _relative(numerical[name], value)) for name, value in surface_fields.items()},
                       'absolute_sample_l2_errors': {name:float(torch.linalg.vector_norm(numerical[name]-value))
                                                     for name,value in surface_fields.items()},
                       'mom_fields': {name: _complex_data(value) for name, value in surface_fields.items()}})
    except Exception as exc:
        report.update({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                       'mom_subdivisions': subdivisions, 'mom_quadrature_order': order})
    return report


def run_convergence(*, families=FAMILIES, spacings=(.06, .05, .04), subdivisions=(1, 2, 3),
                    cycles=16, include_sweeps=True, on_update=None, resume_report=None):
    if len(spacings) != len(subdivisions):
        raise ValueError('spacings and subdivisions must have equal lengths')
    report = {'schema_version': 1, 'device': torch.cuda.get_device_name(), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'frequency_hz': FREQUENCY, 'query_points_m': POINTS,
              'H2_certified': False, 'cases': [], 'sensitivity_sweeps': [],
              'provenance_segments': [PROCESS_PROVENANCE],
              'remaining_acceptance': ['independent physical reference agreement',
                  'complex far field/RCS and power comparison', 'frozen joint discretization error budget',
                  'demonstrated convergence rather than merely three executed levels']}
    if resume_report is not None:
        if resume_report.get('schema_version') != report['schema_version'] or resume_report.get('frequency_hz') != FREQUENCY:
            raise ValueError('Cannot resume an incompatible cross-backend report')
        report['provenance_segments'] = [*resume_report.get('provenance_segments', []), PROCESS_PROVENANCE]
        report['resumed_rows_without_source_snapshot'] = not bool(resume_report.get('provenance_segments'))
        report['cases'] = _resume_cases(resume_report,
            {(family,spacing,subdivision) for family in families for spacing,subdivision in zip(spacings,subdivisions)},cycles)
        expected_orders = {subdivision:4 for subdivision in subdivisions}
        if any(row.get('mom_quadrature_order') != expected_orders[row['mom_subdivisions']] for row in report['cases']):
            raise ValueError('Resume cases use a different MoM quadrature order')
        report['sensitivity_sweeps'] = _resume_fdtd_sweeps(resume_report,spacings[-1],cycles)
    completed = {(row['family'], row['spacing_m'], row['mom_subdivisions']) for row in report['cases']}
    for family in families:
        for level, (spacing, subdivision) in enumerate(zip(spacings, subdivisions)):
            if (family, spacing, subdivision) in completed:
                continue
            try:
                row = compare_case(family, spacing, subdivision, cycles=cycles, order=4)
                row.setdefault('status', 'measured')
            except Exception as exc:
                row = {'family': family, 'spacing_m': spacing, 'mom_subdivisions': subdivision,
                       'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            row['provenance_segment'] = len(report['provenance_segments'])-1
            report['cases'].append(row)
            if on_update is not None:
                on_update(report)
    if include_sweeps:
        for name, values in [('pml', (.15, .2, .25)), ('dt_fraction', (.8, .6, .4)), ('cycles', (12, 20, 28))]:
            for value in values:
                if any(row['parameter'] == name and row['value'] == value for row in report['sensitivity_sweeps']):
                    continue
                settings = {'pml': .2, 'dt_fraction': .8, 'cycles': cycles}
                settings[name] = value
                try:
                    _, _, row = fdtd_pair('lossless_sphere', spacings[-1], **settings)
                    row.setdefault('status', 'measured')
                except Exception as exc:
                    row = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                report['sensitivity_sweeps'].append({'parameter': name, 'value': value,
                    'provenance_segment':len(report['provenance_segments'])-1, **row})
                if on_update is not None:
                    on_update(report)
    return report



def control_sphere(order=6, radius=.2):
    """CUDA Gauss polar / periodic azimuth rule, with explicit SI surface weights."""
    from witwin.maxwell.mom.quadrature import gauss_rule
    u, weights = gauss_rule(order, device='cuda', dtype=torch.float64)
    z = 2*u-1
    phi = torch.arange(2*order, device='cuda', dtype=torch.float64)*math.pi/order
    ring = (1-z*z).sqrt()
    directions = torch.stack((ring[:,None]*phi.cos(), ring[:,None]*phi.sin(),
                              z[:,None].expand(-1,len(phi))), -1).reshape(-1,3)
    solid_angles = (2*weights[:,None].expand(-1,len(phi))*math.pi/order).reshape(-1)
    return radius*directions, directions, radius*radius*solid_angles, solid_angles


def compare_observables(family, spacing, subdivision, *, angular_order=6, radius=.2, cycles=16):
    """Independent common closed-surface flux, complex far field and RCS audit."""
    from witwin.maxwell.postprocess.stratton_chu import equivalent_surface_currents_from_surface_samples
    from witwin.maxwell.postprocess.nfft import NearFieldFarFieldTransformer
    from witwin.maxwell.mom.fields import poynting_flux, radar_cross_section
    points, normals, weights, solid_angles = control_sphere(angular_order,radius=radius)
    point_config = (*POINTS[:-1], *(tuple(row) for row in points.cpu().tolist()), POINTS[-1])
    scene, fdtd, report = fdtd_pair(family, spacing, cycles=cycles, query_points=point_config)
    config = mw.MoMConfig(subdivisions=subdivision,
        integration=INTEGRATION,
        query_config=mw.FieldQueryConfig(max_order=128))
    result = mw.Simulation.mom(scene, frequency=FREQUENCY, config=config).run()
    mom_e, mom_h = result.query_fields(points, region_id='exterior', field_kind='scattered')
    total_e, total_h = result.query_fields(points, region_id='exterior', field_kind='total')
    fields = {'fdtd': (fdtd['scattered_E'][3:], fdtd['scattered_H'][3:],
                        fdtd['total_E'][3:-1], fdtd['total_H'][3:-1]),
              'mom': (mom_e, mom_h, total_e, total_h)}
    medium = result.surface_currents.handle.exterior
    far_fields = {}
    powers = {}
    for backend, (es,hs,et,ht) in fields.items():
        surface = equivalent_surface_currents_from_surface_samples(
            frequency=FREQUENCY, points=points, normals=normals, areas=weights, E=es, H=hs)
        transformed = NearFieldFarFieldTransformer(surface, device='cuda', c=C_0, eps0=EPSILON_0, mu0=MU_0).transform_directions(normals, radius=1.)
        propagation = torch.exp(torch.tensor(-1j*medium.k, device='cuda', dtype=torch.complex128))
        far = transformed['E']*propagation
        far_fields[backend] = far
        ei,hi = et-es,ht-hs
        scattered = poynting_flux(es,hs,normals,weights)
        extinction = -poynting_flux(ei,hs,normals,weights)-poynting_flux(es,hi,normals,weights)
        absorption = -poynting_flux(et,ht,normals,weights)
        far_power = (solid_angles*far.abs().square().sum(-1)).sum()/(2*medium.eta.real)
        powers[backend] = {'scattered_flux_W': float(scattered), 'extinction_interference_W': float(extinction),
                           'absorption_inward_flux_W': float(absorption), 'scattered_far_integral_W': float(far_power),
                           'flux_far_absolute_difference_W':float(abs(scattered-far_power)),
                           'flux_far_relative_difference': (None if family == 'no_contrast' else
                               float(abs(scattered-far_power)/abs(far_power).clamp_min(1e-14)))}
    native_far,_ = result.far_field(normals)
    native_rcs = radar_cross_section(native_far, 1.)
    fdtd_rcs = radar_cross_section(far_fields['fdtd'], 1.)
    report.update({'control_surface': {'kind':'sphere','radius_m':radius,'angular_order':angular_order,
                    'quadrature_points':len(points), 'area_m2':float(weights.sum()),
                    'sampling':'same_physical_points; FDTD_trilinear_Yee; MoM_adaptive_potentials'},
                   'mom_subdivisions': subdivision, 'mom_integration':asdict(config.integration), 'powers': powers,
                   'complex_far_relative_error': (None if family == 'no_contrast' else
                       _weighted_norm(far_fields['fdtd']-native_far,solid_angles)/max(_weighted_norm(native_far,solid_angles),1e-14)),
                   'rcs_relative_error': (None if family == 'no_contrast' else
                       _weighted_norm(fdtd_rcs-native_rcs,solid_angles)/max(_weighted_norm(native_rcs,solid_angles),1e-14)),
                   'mom_surface_to_native_far_relative_error': (None if family == 'no_contrast' else
                       _weighted_norm(far_fields['mom']-native_far,solid_angles)/max(_weighted_norm(native_far,solid_angles),1e-14)),
                   'complex_far_absolute_l2_error_m':_weighted_norm(far_fields['fdtd']-native_far,solid_angles),
                   'mom_surface_to_native_far_absolute_l2_error_m':_weighted_norm(far_fields['mom']-native_far,solid_angles),
                   'angular_error_norm':'sqrt(integral |value|^2 dOmega) with Gauss-polar and periodic-azimuth weights',
                   'error_norm_version':'solid_angle_weighted_L2_v1',
                   'zero_reference_policy':'analytic_zero_relative_null_use_absolute_or_incident_norm',
                   'solid_angle_weights':solid_angles.cpu().tolist(),
                   'far_directions':normals.cpu().tolist(),
                   'fdtd_far_amplitude': _complex_data(far_fields['fdtd']),
                   'native_mom_far_amplitude': _complex_data(native_far),
                   'fdtd_rcs_m2':fdtd_rcs.cpu().tolist(), 'native_mom_rcs_m2':native_rcs.cpu().tolist(),
                   'status':'measured', 'H2_certified':False,
                   'remaining_acceptance':['control_surface_quadrature_and_radius_refinement',
                       'independent_internal_volume_dissipation', 'frozen_joint_error_budget']})
    return report



def tetrahedral_volume_rule(mesh, *, degree=2):
    """Positive-volume centroid/four-point rules on a convex surface's center fan."""
    center = mesh.vertices.mean(0)
    triangles = mesh.triangles
    relative = triangles-center
    signed = (relative[:,0]*torch.linalg.cross(relative[:,1],relative[:,2])).sum(-1)/6
    if not bool((signed > 0).all()):
        raise ValueError('Volume center fan requires an outward convex surface enclosing its center')
    # Every face plane must bound all vertices; reject nonconvex center fans.
    normals = mesh.normals
    for batch in mesh.vertices.split(128):
        if bool((((batch[:,None]-triangles[None,:,0])*normals[None]).sum(-1) > 1e-10).any()):
            raise ValueError('Tetrahedral volume rule currently requires convex geometry')
    tetrahedra = torch.cat((center.expand(len(triangles),1,3),triangles),1)
    if degree == 1:
        return tetrahedra.mean(1), signed
    if degree != 2:
        raise ValueError('Tetrahedral quadrature degree must be 1 or 2')
    a = (5+3*math.sqrt(5))/20
    b = (5-math.sqrt(5))/20
    barycentric = torch.full((4,4),b,device=mesh.device,dtype=mesh.dtype)
    barycentric.diagonal().fill_(a)
    points = torch.einsum('qi,tic->tqc',barycentric,tetrahedra).reshape(-1,3)
    return points,(signed[:,None].expand(-1,4)/4).reshape(-1)


def fdtd_volume_absorption(result, common_phasor_gain):
    """Independent physical Joule integral on the actual conductive Yee edges."""
    from witwin.maxwell.compiler.power_loss import _selected_control_volumes
    total = torch.zeros((),device=result.scene.device,dtype=torch.float64)
    component_counts = {}
    for component in ('Ex','Ey','Ez'):
        sigma = getattr(result.solver,'sigma_e_'+component)
        mask = sigma > 0
        component_counts[component] = int(mask.sum())
        if not bool(mask.any()):
            continue
        field = result.tensor(component)
        if not field.is_complex():
            raise ValueError('Independent volume loss requires full_field_dft=True')
        volume = _selected_control_volumes(result.prepared_scene,component,mask).to(torch.float64)
        e = field[mask]/common_phasor_gain
        total += (.5*sigma[mask].to(torch.float64)*e.abs().square()*volume).sum()
    return total,component_counts


def compare_volume_absorption(spacing, subdivision, *, cycles=16):
    """Independent lossy-sphere volume powers, without extinction-minus-scattering."""
    from witwin.maxwell.mom.observables import volume_absorption
    runtime = {}
    scene,_,report = fdtd_pair('lossy_sphere',spacing,cycles=cycles,
                               full_field_dft=True,runtime_results=runtime)
    fdtd_power,counts = fdtd_volume_absorption(runtime['total'],runtime['common_phasor_gain'])
    config = mw.MoMConfig(subdivisions=subdivision,
        integration=INTEGRATION,query_config=mw.FieldQueryConfig(max_order=128))
    result = mw.Simulation.mom(scene,frequency=FREQUENCY,config=config).run()
    data = result.surface_currents
    powers = []
    for degree in (1,2):
        points,weights = tetrahedral_volume_rule(data.mesh,degree=degree)
        power = data.absorbed_power(points,weights,query_config=config.query_config)
        powers.append(float(power))
    report.update({'mom_subdivisions':subdivision,'status':'measured',
        'mom_integration':asdict(config.integration),
        'fdtd_volume_absorption_W':float(fdtd_power),'fdtd_conductive_edge_counts':counts,
        'fdtd_measure':'0.5 sum sigma_e_component |E_component/common_gain|^2 dV_Yee',
        'mom_volume_absorption_W':powers[-1],'mom_tetrahedral_degree1_W':powers[0],
        'mom_volume_quadrature_relative_change':abs(powers[1]-powers[0])/max(abs(powers[1]),1e-14),
        'mom_measure':'0.5 omega integral Im(epsilon) |E_interior|^2 dV_actual_polyhedron',
        'relative_volume_power_difference':abs(float(fdtd_power)-powers[-1])/max(abs(powers[-1]),1e-14),
        'independent_of_surface_balance':True,'H2_certified':False})
    return report


def run_volume_convergence(*, on_update=None, spacings=(.06,.05,.04), subdivisions=(1,2,3), cycles=16, resume_report=None):
    report = {'schema_version':1,'H2_certified':False,'cases':[], 'frequency_hz':FREQUENCY,
        'device':torch.cuda.get_device_name(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'observables':'independent_polyhedral_and_Yee_volume_absorption','provenance_segments':[PROCESS_PROVENANCE]}
    if resume_report is not None:
        if resume_report.get('observables') != report['observables']:
            raise ValueError('Cannot resume incompatible volume report')
        report['provenance_segments'] = [*resume_report.get('provenance_segments',[]),PROCESS_PROVENANCE]
        report['resumed_rows_without_source_snapshot'] = not bool(resume_report.get('provenance_segments'))
        report['cases'] = _resume_cases(resume_report,
            {('lossy_sphere',spacing,subdivision) for spacing,subdivision in zip(spacings,subdivisions)},cycles)
    completed = {(row['spacing_m'],row['mom_subdivisions']) for row in report['cases']}
    for spacing,subdivision in zip(spacings,subdivisions):
        if (spacing,subdivision) in completed:
            continue
        try:
            row = compare_volume_absorption(spacing,subdivision,cycles=cycles)
        except Exception as exc:
            row = {'family':'lossy_sphere','spacing_m':spacing,'mom_subdivisions':subdivision,
                   'status':'failed','error':f'{type(exc).__name__}: {exc}'}
        row['provenance_segment'] = len(report['provenance_segments'])-1
        report['cases'].append(row)
        if on_update is not None:
            on_update(report)
    return report


def run_surface_sensitivity(*, on_update=None, resume_report=None, spacing=.04,
                            subdivision=2, cycles=16):
    report = {'schema_version':1,'H2_certified':False,'cases':[], 'frequency_hz':FREQUENCY,
        'device':torch.cuda.get_device_name(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'observables':'closed_surface_angular_and_radius_sensitivity','provenance_segments':[PROCESS_PROVENANCE]}
    if resume_report is not None:
        if resume_report.get('observables') != report['observables']:
            raise ValueError('Cannot resume incompatible surface sensitivity report')
        report['provenance_segments'] = [*resume_report.get('provenance_segments',[]),PROCESS_PROVENANCE]
        report['resumed_rows_without_source_snapshot'] = not bool(resume_report.get('provenance_segments'))
        report['cases'] = _resume_cases(resume_report,{('lossless_sphere',spacing,subdivision)},cycles)
        for row in report['cases']:
            parameter,value = row['parameter'],row['value']
            allowed = {'angular_order':(6,8,10),'radius':(.18,.20,.24)}
            if parameter not in allowed or value not in allowed[parameter]:
                raise ValueError('Resume surface sweep parameter or value differs')
            expected = {'angular_order':10,'radius_m':.2}
            expected['radius_m' if parameter == 'radius' else parameter] = value
            if any(row['control_surface'].get(key) != setting for key,setting in expected.items()):
                raise ValueError('Resume surface sweep uses different fixed settings')
    for parameter,values in [('angular_order',(6,8,10)),('radius',(.18,.20,.24))]:
        for value in values:
            if any(row['parameter'] == parameter and row['value'] == value for row in report['cases']):
                continue
            options = dict(angular_order=10,radius=.2)
            options[parameter] = value
            try:
                row = compare_observables('lossless_sphere',spacing,subdivision,cycles=cycles,**options)
            except Exception as exc:
                row = {'status':'failed','error':f'{type(exc).__name__}: {exc}'}
            row.update(parameter=parameter,value=value,provenance_segment=len(report['provenance_segments'])-1)
            report['cases'].append(row)
            if on_update is not None:
                on_update(report)
    return report

def run_observable_convergence(*, on_update=None, families=FAMILIES, spacings=(.06,.05,.04), subdivisions=(1,2,3), cycles=16, resume_report=None):
    report = {'schema_version':1, 'H2_certified':False, 'cases':[], 'frequency_hz':FREQUENCY,
              'device':torch.cuda.get_device_name(), 'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'), 'observables':'complex_far_RCS_closed_surface_power','provenance_segments':[PROCESS_PROVENANCE]}
    if resume_report is not None:
        if resume_report.get('observables') != report['observables']:
            raise ValueError('Cannot resume incompatible observable report')
        report['provenance_segments'] = [*resume_report.get('provenance_segments',[]), PROCESS_PROVENANCE]
        report['resumed_rows_without_source_snapshot'] = not bool(resume_report.get('provenance_segments'))
        report['cases'] = _resume_cases(resume_report,
            {(family,spacing,subdivision) for family in families for spacing,subdivision in zip(spacings,subdivisions)},cycles)
        if any(row['control_surface']['angular_order'] != 6 or row['control_surface']['radius_m'] != .2 for row in report['cases']):
            raise ValueError('Resume cases use different closed-surface sampling')
    completed = {(row['family'],row['spacing_m'],row['mom_subdivisions']) for row in report['cases']}
    for family in families:
        for spacing, subdivision in zip(spacings,subdivisions):
            if (family,spacing,subdivision) in completed:
                continue
            try:
                row = compare_observables(family, spacing, subdivision, cycles=cycles)
            except Exception as exc:
                row = {'family':family,'spacing_m':spacing,'mom_subdivisions':subdivision,
                       'status':'failed','error':f'{type(exc).__name__}: {exc}'}
            row['provenance_segment'] = len(report['provenance_segments'])-1
            report['cases'].append(row)
            if on_update is not None:
                on_update(report)
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/mom_fdtd.json'))
    parser.add_argument('--quick', action='store_true', help='One coarse no-contrast case; does not constitute convergence acceptance')
    parser.add_argument('--volume', action='store_true', help='Run independent lossy-volume dissipation convergence')
    parser.add_argument('--surface-sweep', action='store_true', help='Refine closed-surface angular quadrature and radius independently')
    parser.add_argument('--observables', action='store_true', help='Run common closed-surface far-field/RCS/power convergence')
    parser.add_argument('--resume', action='store_true', help='Resume measured rows from the output checkpoint')
    args = parser.parse_args(argv)
    if sum((args.volume,args.observables,args.surface_sweep)) > 1:
        parser.error('Select only one of --volume, --observables, and --surface-sweep')
    options = {'families': ('no_contrast',), 'spacings': (.06,), 'subdivisions': (0,), 'cycles': 8,
               'include_sweeps': False} if args.quick else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(report):
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({'cases': len(report['cases']), 'sweeps': len(report.get('sensitivity_sweeps', []))}), flush=True)
    previous = json.loads(args.output.read_text()) if args.resume and args.output.exists() else None
    if args.surface_sweep:
        report = run_surface_sensitivity(on_update=checkpoint,resume_report=previous)
    elif args.volume:
        volume_options = dict(spacings=(.06,),subdivisions=(1,),cycles=8) if args.quick else {}
        report = run_volume_convergence(on_update=checkpoint,resume_report=previous, **volume_options)
    elif args.observables:
        observable_options = dict(families=('lossless_sphere',),spacings=(.06,),subdivisions=(1,),cycles=8) if args.quick else {}
        report = run_observable_convergence(on_update=checkpoint, resume_report=previous, **observable_options)
    else:
        report = run_convergence(**options, on_update=checkpoint, resume_report=previous)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(args.output), 'cases': len(report['cases']),
                      'failed': sum(row['status'] == 'failed' for row in report['cases']),
                      'H2_certified': report['H2_certified']}))
    return 1 if any(row['status'] == 'failed' for row in report['cases'] + report.get('sensitivity_sweeps', [])) else 0



def _freeze_process_provenance():
    root = Path(__file__).resolve().parents[1]
    # Freeze lazy-import and native-kernel inputs as well as loaded modules.
    contents = {str(path.relative_to(root)): path.read_bytes()
               for path in (root/'witwin'/'maxwell').rglob('*')
               if path.is_file() and path.suffix in ('.py','.cu','.cuh','.cpp','.h')}
    sources = {key:hashlib.sha256(value).hexdigest() for key,value in contents.items()}
    bytecode = {}
    prefixes = ('witwin.maxwell.mom', 'witwin.maxwell.fdtd', 'witwin.maxwell.compiler',
                'witwin.maxwell.postprocess', 'witwin.maxwell.scene', 'witwin.maxwell.simulation',
                'witwin.maxwell.result', 'witwin.maxwell.sources', 'witwin.maxwell.constants')
    for name,module in tuple(sys.modules.items()):
        path = getattr(module,'__file__',None)
        if path is None or not name.startswith(prefixes):
            continue
        path = Path(path).resolve()
        if path.suffix != '.py' or not path.is_relative_to(root):
            continue
        key = str(path.relative_to(root))
        contents[key] = path.read_bytes()
        sources[key] = hashlib.sha256(contents[key]).hexdigest()
        definitions = []
        for attr,value in vars(module).items():
            if getattr(value,'__module__',None) != name:
                continue
            if inspect.isfunction(value):
                definitions.append((attr,value.__code__))
            elif inspect.isclass(value):
                for method_name,method in vars(value).items():
                    method = method.__func__ if isinstance(method,(staticmethod,classmethod)) else method
                    if inspect.isfunction(method):
                        definitions.append((attr+'.'+method_name,method.__code__))
        digest = hashlib.sha256()
        for attr,code in sorted(definitions):
            digest.update(attr.encode());digest.update(marshal.dumps(code))
        bytecode[key] = digest.hexdigest()
    benchmark_key = str(Path(__file__).resolve().relative_to(root))
    contents[benchmark_key] = Path(__file__).read_bytes()
    sources[benchmark_key] = hashlib.sha256(contents[benchmark_key]).hexdigest()
    snapshot_id = hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    snapshot = Path(tempfile.gettempdir())/'witwin-mom-cross-source'/snapshot_id
    for key,value in contents.items():
        target = snapshot/key
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(value)
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    return {'captured_utc':datetime.now(timezone.utc).isoformat(), 'git_revision_at_import':revision,
            'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'source_sha256':sources,'imported_runtime_bytecode_sha256':bytecode,
            'source_snapshot_directory':str(snapshot),'source_snapshot_id':snapshot_id,
            'scope':'frozen at benchmark module import; source files plus loaded Python function bytecode',
            'unrecorded_prior_runs_are_not_source_verified':True}


PROCESS_PROVENANCE = _freeze_process_provenance()

if __name__ == '__main__':
    raise SystemExit(main())
