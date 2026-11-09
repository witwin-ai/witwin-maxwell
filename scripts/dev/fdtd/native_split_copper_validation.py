"""Genuine prepared-native cut-plane job preparation and guarded execution.

CPU preflight compiles real scene/grid/source metadata only. GPU physics is
explicitly unqualified until the separately scheduled commands actually run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import PreparedScene
from witwin.maxwell.copper_foil import EPSILON_0, MU_0
from scripts.dev.fdtd.native_split_copper_plane import NativeSplitPlane, compile_native_plane_layout, native_source_identity
from scripts.dev.fdtd.split_copper_plane import independent_film_reference
from scripts.dev.fdtd.split_copper_plane_validation import imported_package_identity


def build_native_simulation(*, device, spacing=.005, polarization='Ez', complex_fields=True,
                            frequency=1e9, ramp_cycles=8., settling_cycles=32., measurement_cycles=16.):
    half = round(.2/spacing)
    x = np.arange(-half, half+1, dtype=np.float64)*spacing
    yz = np.arange(-2, 3, dtype=np.float64)*.005
    direction = (0., 1., 0.) if polarization == 'Ey' else (0., 0., 1.)
    paired = 'bloch' if complex_fields else 'periodic'
    probes = list(range(math.ceil(-.12/spacing), math.floor(-.04/spacing)+1))
    probes += list(range(math.ceil(.04/spacing), math.floor(.12/spacing)+1))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, .2), (-.01, .01), (-.01, .01))),
        grid=mw.GridSpec.custom(x, yz, yz), boundary=mw.BoundarySpec.faces(default='pml', num_layers=12,
            y=paired, z=paired, bloch_wavevector=(0., 0., 0.)), device=device,
        sources=(mw.PlaneWave(direction=(1., 0., 0.), polarization=direction,
            source_time=mw.CW(frequency, phase=.37), injection=mw.TFSF.slab(axis='x', bounds=(-.15, .15))),),
        monitors=tuple(mw.PointMonitor(f'probe-{index:+d}', (index*spacing, 0., 0.), fields=(polarization,)) for index in probes))
    return mw.Simulation.fdtd(scene, frequency=frequency, full_field_dft=False, cuda_graph=False,
        run_time=mw.TimeConfig(transient_cycles=ramp_cycles+settling_cycles, steady_cycles=measurement_cycles),
        spectral_sampler=mw.SpectralSampler(window='hanning'))


@contextmanager
def smooth_actual_native_cw(ramp_seconds):
    from unittest.mock import patch
    from witwin.maxwell.fdtd.excitation import temporal
    original = temporal._resolve_cw_signal
    def signal(omega, source_time, time_value):
        q = min(1., max(0., float(time_value)/ramp_seconds))
        envelope = q**3*(10-15*q+6*q*q)
        return tuple(envelope*value for value in original(omega, source_time, time_value))
    with patch.object(temporal, '_resolve_cw_signal', signal):
        yield


def encode(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.is_complex():
            return dict(real=value.real.tolist(), imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value, np.ndarray):
        return encode(torch.as_tensor(value))
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def compile_preflight(*, spacing, offset, polarization, complex_fields):
    simulation = build_native_simulation(device='cpu', spacing=spacing, polarization=polarization, complex_fields=complex_fields)
    scene = PreparedScene(simulation.scene, lazy_meshgrid=True)
    layout = compile_native_plane_layout(scene, cut=offset*spacing, polarization=polarization,
                                       implicit_bounds=(-spacing, 2*spacing))
    return dict(scope='Real Scene -> PreparedScene grid/source compilation only; no native CUDA preparation or field execution',
        device='cpu', native_gpu_execution_completed=False,
        physical_cut_m=offset*spacing, actual_split_intervals_m=layout.geometry.magnetic_lengths[layout.support['magnetic']],
        electric_component=layout.electric_name, native_magnetic_component=layout.magnetic_name,
        native_magnetic_to_positive_line_sign=layout.magnetic_sign,
        physical_line_area_m2=layout.physical_line_area, summed_physical_area_m2=float(layout.physical_line_area.sum()),
        actual_native_original_x_nodes64_m=scene.x_nodes64,
        actual_local_electric_positions_m=layout.geometry.electric_positions[layout.support['electric']],
        actual_local_magnetic_positions_m=layout.geometry.magnetic_positions[layout.support['magnetic']],
        native_electric_write_indices=layout.native_electric, side_aware_electric_read_indices=layout.local_electric,
        native_magnetic_write_indices=layout.native_magnetic, side_aware_magnetic_read_indices=layout.local_magnetic,
        invalid_discarded_native_magnetic_indices=layout.discarded_native_magnetic,
        outside_magnetic_indices=layout.outside_magnetic, dependency_audit=layout.dependency_audit,
        compiled_real_source=layout.source_metadata,
        planned_native_method='Scene -> Simulation.fdtd(...,cuda_graph=False,full_field_dft=False).prepare() -> NativeSplitPlane',
        actual_native_dt_s=None, public_surface_mesh_copper_enabled=False)


def execute_native(args):
    import os
    if os.environ.get('WITWIN_RUN_NATIVE_SPLIT_CUDA') != '1':
        raise RuntimeError('Actual prepared-native execution requires a separately scheduled WITWIN_RUN_NATIVE_SPLIT_CUDA=1 window.')
    frequency, sigma = 1e9, 5.8e7
    thickness = math.sqrt(2/(2*math.pi*frequency*MU_0*sigma))
    material = mw.CopperFoil(thickness=thickness, conductivity=sigma, frequency_range=(.5e9, 2e9))
    simulation = build_native_simulation(device='cuda', spacing=args.spacing, polarization=args.polarization,
        complex_fields=not args.real_fields, ramp_cycles=args.ramp_cycles,
        settling_cycles=args.settling_cycles, measurement_cycles=args.measurement_cycles)
    with smooth_actual_native_cw(args.ramp_cycles/frequency):
        prepared = simulation.prepare()
        adapter = NativeSplitPlane(prepared, cut=args.offset*args.spacing, polarization=args.polarization,
            material=material, implicit_bounds=(-args.spacing, 2*args.spacing))
        records = adapter.run()
    audit = records['actual_native_weight_audit']
    eps = audit[adapter.layout.electric_name]['effective_mass_coefficient_from_actual_gain']
    mu = audit[adapter.layout.magnetic_name]['effective_mass_coefficient_from_actual_gain']
    eta, c = math.sqrt(mu/eps), 1/math.sqrt(mu*eps)
    dt = records['actual_native_dt_s']
    k = 2/args.spacing*math.asin(math.sin(math.pi*frequency*dt)*args.spacing/(c*dt))
    positions, values = [], []
    for monitor in simulation.scene.monitors:
        record = records['exterior_monitors'][monitor.name]
        index = int(record['field_index'][0])
        positions.append(float(prepared.solver.scene.x_nodes64[index])-args.offset*args.spacing)
        values.append(complex(torch.as_tensor(record['data']).reshape(-1)[0].cpu()))
    x, e = np.asarray(positions), np.asarray(values)
    basis = np.stack((np.exp(1j*k*x), np.exp(-1j*k*x)), -1)
    left = np.linalg.lstsq(basis[x < 0], e[x < 0], rcond=None)[0]
    right = np.linalg.lstsq(basis[x > 0], e[x > 0], rcond=None)[0]
    r, t = left[1]/left[0], right[0]/left[0]
    reference_r, reference_t, reference_a = independent_film_reference(thickness, sigma, frequency)
    area = float(adapter.area.sum().cpu())
    heat_absorption = records['actual_shared_film_spectral_heat_W']/(abs(left[0])**2/(2*eta)*area)
    measured_absorption = 1-abs(r)**2-abs(t)**2
    errors = dict(reflection_absolute_error=abs(r-reference_r), transmission_relative_error=abs(t-reference_t)/abs(reference_t),
        scattering_absorption_relative_error=abs(measured_absorption-reference_a)/reference_a,
        shared_heat_absorption_relative_error=abs(heat_absorption-reference_a)/reference_a)
    return dict(scope='Actual prepared native CUDA background with a private uniform-plane split/CN patch',
        native_gpu_execution_completed=True, native_curved_surface_qualified=False, raw=records,
        actual_native_probe_x_relative_cut_m=x, actual_native_probe_E=e,
        known_native_carrier_wavenumber=k, incident_amplitude_at_cut=left[0], reflected_amplitude_at_cut=left[1],
        transmitted_amplitude_at_cut=right[0], right_incoming_amplitude=right[1], reflection=r, transmission=t,
        independent_reference_reflection=reference_r, independent_reference_transmission=reference_t,
        independent_reference_absorption=reference_a, measured_absorption=measured_absorption,
        shared_heat_absorption=heat_absorption, errors=errors,
        original_plane_gates=dict(reflection_absolute_error_max=.002, transmission_relative_error_max=.03,
            absorption_relative_error_max=.03),
        original_plane_gates_pass=errors['reflection_absolute_error'] < .002 and all(value < .03 for key, value in errors.items() if key != 'reflection_absolute_error'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--spacing', type=float, default=.005)
    parser.add_argument('--offset', type=float, default=.3)
    parser.add_argument('--polarization', choices=('Ey', 'Ez'), default='Ez')
    parser.add_argument('--real-fields', action='store_true')
    parser.add_argument('--ramp-cycles', type=float, default=8.)
    parser.add_argument('--settling-cycles', type=float, default=32.)
    parser.add_argument('--measurement-cycles', type=float, default=16.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to replace an existing native cut-plane record.')
    if args.spacing not in (.005, .0025) or not 0 <= args.offset <= 1:
        raise ValueError('This fixed validation family uses 5/2.5mm exterior grids and offsets in [0,1].')
    before, packages = native_source_identity(), imported_package_identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = dict(status='running', parameters=vars(args), source_sha256_before=before,
        imported_packages_before=packages, native_gpu_execution_completed=False)
    args.output.write_text(json.dumps(encode(started), indent=2)+'\n')
    try:
        report = (compile_preflight(spacing=args.spacing, offset=args.offset, polarization=args.polarization,
                  complex_fields=not args.real_fields) if args.cpu_preflight else execute_native(args))
    except BaseException as exc:
        started.update(status='failed', error_type=type(exc).__name__, error=str(exc),
            source_sha256_after=native_source_identity(), imported_packages_after=imported_package_identity())
        started['source_unchanged'] = started['source_sha256_before'] == started['source_sha256_after']
        args.output.write_text(json.dumps(encode(started), indent=2)+'\n')
        raise
    after = native_source_identity()
    report.update(status='completed', parameters=vars(args), source_sha256_before=before, source_sha256_after=after,
        source_unchanged=before == after, imported_packages_before=packages,
        imported_packages_after=imported_package_identity())
    args.output.write_text(json.dumps(encode(report), indent=2)+'\n')
    if before != after:
        raise RuntimeError('The executed native diagnostic source changed during validation.')
    print(json.dumps(dict(cpu_preflight=args.cpu_preflight, output=str(args.output), source_unchanged=before == after,
        native_gpu_execution_completed=report['native_gpu_execution_completed'])))
    if not args.cpu_preflight and not report['original_plane_gates_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
