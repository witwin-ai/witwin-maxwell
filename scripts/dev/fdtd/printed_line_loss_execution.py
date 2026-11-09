"""Bounded real CUDA two-drive stripline loss pilot and coherent diagnostics.

The default run is one combined material at 120 mm source separation. Small
execution tests explicitly override the planned step count; they are not a
64 ns physical qualification. No loop expands to other materials or families.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import witwin.core
import witwin.maxwell as mw

from benchmark.scenes.rf.printed_lines import printed_line_scene, analytic_homogeneous_line
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.printed_line_broadband import encode, load_helper, physical_sources
from scripts.dev.fdtd.printed_line_loss_basis import (
    SpectralContract, capture_loss_basis, coherent_conductor_records, coherent_fields,
)
from scripts.dev.fdtd.printed_line_regional_power import regional_monitors, regional_plane_power
from witwin.maxwell.fdtd.quadrature import plane_normal_poynting
from witwin.maxwell.scene import prepare_scene


DRIVE_BASIS = ((1., .25), (.25, 1.))


@dataclass(frozen=True)
class PilotCase:
    dx: float = .0025
    longitudinal_dl: float | None = None
    metal: str = "copper"
    probe_reference_dx: float | None = None
    record_legacy_native_vi: bool = False
    axial_half_length: float = .14
    padding: float = .08
    separation: float = .12
    sigma: float = .01
    frequencies: tuple[float, ...] = (.8e9, .9e9, 1e9, 1.1e9, 1.2e9)
    positions: tuple[float, ...] = (-.045, -.03, -.015, 0., .015, .03, .045)
    duration: float = 64e-9
    checkpoint_time: float = 32e-9


def canonical_case_payload(payload):
    """Read only the three explicitly known immutable pilot case schemas."""
    current = set(PilotCase.__dataclass_fields__)
    prior_material = current - {'record_legacy_native_vi'}
    original_copper = prior_material - {'metal', 'longitudinal_dl', 'probe_reference_dx'}
    if not isinstance(payload, dict) or set(payload) not in (current, prior_material, original_copper):
        raise ValueError('Pilot case must retain every physical field of a known saved schema.')
    values = dict(payload)
    if 'record_legacy_native_vi' in values and type(values['record_legacy_native_vi']) is not bool:
        raise ValueError('Saved dual V/I recording flag must be an explicit boolean.')
    return encode(asdict(PilotCase(**values)))


def tensor_signature(value):
    """Exact dtype/shape/value signature for actual native geometry/operators."""
    def canonical(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            return {'dtype': str(item.dtype), 'shape': list(item.shape),
                'sha256': hashlib.sha256(item.tobytes()).hexdigest()}
        if isinstance(item, dict):
            return {key: canonical(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [canonical(value) for value in item]
        return encode(item)
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def execution_identity():
    core_root = Path(next(iter(witwin.core.__path__))).resolve()
    core = {str(path.relative_to(core_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(core_root.rglob('*'))
        if path.is_file() and path.suffix in {'.py', '.cpp', '.cu', '.h', '.cuh'}}
    helpers = {name: Path(inspect.getfile(function)) for name, function in (
        ('fixture', printed_line_scene), ('sources', physical_sources),
        ('regional', regional_plane_power), ('basis', capture_loss_basis))}
    _, contours = load_helper('stripline')
    from scripts.dev.fdtd.printed_line_fixed_vi import fixed_physical_probe_monitors
    helpers.update(execution=Path(__file__), contours=contours,
        fixed_VI=Path(inspect.getfile(fixed_physical_probe_monitors)))
    return {'maxwell': package_identity(mw, 'witwin-maxwell'), 'actual_core_root': str(core_root),
        'actual_core_sources': core, 'actual_core_sha256': tensor_signature(core),
        'helpers': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in helpers.items()}}


def build_case(case):
    """CPU-only real scene/observer compilation, reused unchanged across drives."""
    if (len(case.frequencies) < 2 or any(not math.isfinite(frequency) or frequency <= 0 for frequency in case.frequencies)
        or any(b <= a for a, b in zip(case.frequencies, case.frequencies[1:]))):
        raise ValueError('Pilot frequencies must be a strictly increasing complete broadband axis.')
    if not 0 < case.checkpoint_time < case.duration:
        raise ValueError('Pilot checkpoint time must lie strictly inside its fixed planned horizon.')
    if any(abs(position) >= case.separation / 2 for position in case.positions):
        raise ValueError('Pilot stations must be strictly between its fixed source planes.')
    if type(case.record_legacy_native_vi) is not bool or case.record_legacy_native_vi and case.probe_reference_dx is None:
        raise ValueError('Paired legacy V/I recording requires an explicit boolean and a fixed primary reference.')
    if case.metal not in ('copper', 'pec'):
        raise ValueError('Pilot metal must explicitly be copper or pec.')
    normal = case.dx if case.longitudinal_dl is None else case.longitudinal_dl
    scene = printed_line_scene('stripline', dx=case.dx, longitudinal_dl=normal,
        axial_half_length=case.axial_half_length, port_separation=case.separation,
        transverse_padding=case.padding, transverse_refinement_radius=.02,
        sigma_e=case.sigma, metal=case.metal, copper_frequency_range=(.1e9, 4e9), device='cpu')
    ports, groups = physical_sources(scene, 'stripline', normal)
    scene = scene.clone(ports=ports)
    bounds = ((-.015, .015), *tuple(scene.domain.bounds[axis] for axis in (1, 2)))
    regional, schema = regional_monitors(scene, bounds, case.frequencies, observation='planes')
    helper, _ = load_helper('stripline')
    if case.probe_reference_dx is None:
        probes, rows, expected = helper.native_probe_monitors(scene, 'stripline', case.dx,
            normal, case.padding, case.positions, case.metal)
    else:
        from scripts.dev.fdtd.printed_line_fixed_vi import fixed_physical_probe_monitors
        reference = printed_line_scene('stripline', dx=case.probe_reference_dx, longitudinal_dl=normal,
            axial_half_length=case.axial_half_length, port_separation=case.separation,
            transverse_padding=case.padding, transverse_refinement_radius=.02,
            sigma_e=case.sigma, metal=case.metal, copper_frequency_range=(.1e9, 4e9), device='cpu')
        original, original_rows, _ = helper.native_probe_monitors(reference, 'stripline', case.probe_reference_dx,
            normal, case.padding, case.positions, case.metal)
        probes, rows, expected, observation_contract = fixed_physical_probe_monitors(scene, reference, original, original_rows)
        scene = scene.clone(metadata={**scene.metadata, 'fixed_VI_observation_contract': observation_contract})
    if case.record_legacy_native_vi:
        legacy, legacy_rows, legacy_expected = helper.native_probe_monitors(scene, 'stripline', case.dx,
            normal, case.padding, case.positions, case.metal)
        renamed = {monitor.name: 'legacy_' + monitor.name for monitor in legacy}
        legacy_rows = [tuple([(renamed[name], component, weight) for name, component, weight in row]
            for row in pair) for pair in legacy_rows]
        legacy_expected = {renamed[name]: value for name, value in legacy_expected.items()}
        if set(expected).intersection(legacy_expected):
            raise ValueError('Primary and secondary native V/I monitor names must be disjoint.')
        probes = [*probes, *(mw.PointMonitor(renamed[monitor.name], monitor.position, fields=monitor.fields) for monitor in legacy)]
        expected = {**expected, **legacy_expected}
        scene = scene.clone(metadata={**scene.metadata, 'legacy_native_VI_observation_contract': {
            'role': 'secondary_mesh_native', 'state_key': 'legacy_native_physical_state',
            'grid_spacing_m': case.dx, 'primary_reference_spacing_m': case.probe_reference_dx,
            'station_positions_m': case.positions,
            'native_positions_m': {renamed[monitor.name]: monitor.position for monitor in legacy},
            'integration_rows': legacy_rows, 'expected_native_probes': legacy_expected,
            'scope': 'Additional native fine-grid H loop; physical curve varies with dx. Not the fixed primary functional or an unmatched external-reference parity claim.'}})
    scene = scene.clone(metadata={**scene.metadata, 'VI_observation_roles': {
        'primary': {'state_key': 'physical_state', 'functional': 'fixed_reference' if case.probe_reference_dx is not None else 'mesh_native',
            'reference_spacing_m': case.probe_reference_dx, 'actual_grid_spacing_m': case.dx},
        'secondary': None if not case.record_legacy_native_vi else {'state_key': 'legacy_native_physical_state', 'functional': 'mesh_native', 'actual_grid_spacing_m': case.dx}}})
    by_name = {monitor.name: monitor for monitor in probes}
    tails = []
    for index, (voltage, _) in enumerate(rows):
        name, component, _ = voltage[len(voltage) // 2]
        tails.append(mw.FieldTimeMonitor(f'loss_tail_{index}', components=(component,),
            position=by_name[name].position, interval=32))
    scene = scene.clone(monitors=[*regional, *probes, *tails])
    return scene, groups, bounds, schema, rows, expected, tails


def make_simulation(scene, groups, amplitudes, case, *, graph, steps, device='cuda'):
    if len(amplitudes) != len(groups) or not np.isfinite(np.asarray(amplitudes, dtype=complex)).all():
        raise ValueError('Each independent drive group needs one finite physical source amplitude.')
    if any(amplitude == 0 for amplitude in amplitudes):
        raise ValueError('Active Norton pilot ports require nonzero spectra; use the declared nonzero independent drive basis.')
    group_by_port = {name: column for column, group in enumerate(groups) for name in group}
    # All ports retain identical Norton 50-ohm updates. Both independent source
    # columns are nonzero because the active-port spectral API rejects a zero
    # available-power spectrum. They are not pure left/right source columns.
    pulse = mw.GaussianPulse(1e9, fwidth=.8e9)
    excitations = [mw.PortExcitation(port.name, amplitude=amplitudes[group_by_port[port.name]],
        source_impedance=50., source_time=pulse) for port in scene.ports]
    return mw.Simulation.fdtd(scene.clone(device=device), frequencies=case.frequencies,
        excitations=excitations, cuda_graph=graph,
        spectral_sampler=mw.SpectralSampler(window='none', normalize_source=False),
        run_time=mw.TimeConfig(time_steps=steps))


def runtime_operator_signature(solver):
    names = ('linear_indices', 'voltage_weights', 'injection', 'dt', 'resistance',
        'inductance', 'capacitance', 'coupling_impedance', 'discrete_port_impedance',
        'inverse_denominator', 'parallel_total_admittance', 'series_inductive_impedance',
        'series_capacitive_impedance', 'topology')
    return tensor_signature({runtime.port.name: {name: getattr(runtime.lumped, name) for name in names}
        for runtime in solver._port_runtimes})


def _material_signature(solver):
    names = [prefix + component for prefix in ('eps_', 'sigma_e_') for component in ('Ex', 'Ey', 'Ez')]
    names += [prefix + component for prefix in ('mu_', 'sigma_m_') for component in ('Hx', 'Hy', 'Hz')]
    names += [f'c{component.lower()}_{kind}' for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        for kind in ('curl', 'decay')]
    return tensor_signature({name: getattr(solver, name) for name in names})


def _log(event, *, log_path=None, **values):
    line = json.dumps(encode({'event': event, **values}), allow_nan=False)
    print(line, flush=True)
    if log_path is not None:
        with Path(log_path).open('a') as stream:
            stream.write(line + '\n')


def run_drive(case, amplitudes, output, *, graph=True, resume=True, planned_steps=None, stop_step=None):
    """Run one actual physical column and capture its real runtime sidecar.

    ``planned_steps`` is a small execution-test override, explicitly recorded.
    No previous partial output or completed Result is extended or overwritten.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if planned_steps is not None and (type(planned_steps) is not int or planned_steps <= 0
        or resume and (type(stop_step) is not int or not 0 < stop_step < planned_steps)):
        raise ValueError('Execution-test step override requires a complete positive horizon and interior checkpoint.')
    output.mkdir(parents=True)
    before = execution_identity()
    scene, groups, bounds, schema, rows, expected, tails = build_case(case)
    (output / 'input.json').write_text(json.dumps(encode({'case': asdict(case),
        'amplitudes': amplitudes, 'sources': {'groups': groups, 'resistance_ohm': 50.,
            'waveform': asdict(mw.GaussianPulse(1e9, fwidth=.8e9)),
            'ports': [{'name': port.name, 'positive': port.positive, 'negative': port.negative,
                'voltage_aperture': {'position': port.voltage_aperture.position, 'size': port.voltage_aperture.size},
                'current_surface': {'position': port.current_surface.position, 'size': port.current_surface.size}}
                for port in scene.ports]},
        'observation_schema': schema, 'bounds_m': bounds, 'integration_rows': rows,
        'VI_observation_roles': scene.metadata['VI_observation_roles'],
        'legacy_native_VI_observation_contract': scene.metadata.get('legacy_native_VI_observation_contract'),
        'expected_native_probes': expected, 'source': before, 'test_step_override': planned_steps}),
        indent=2, allow_nan=False) + '\n')
    simulation = make_simulation(scene, groups, amplitudes, case, graph=graph, steps=1)
    prepared = simulation.prepare()
    dt = float(prepared.solver.dt)
    steps = math.ceil(case.duration / dt) if planned_steps is None else planned_steps
    split = math.floor(case.checkpoint_time / dt) if stop_step is None else stop_step
    if type(steps) is not int or steps <= 0 or resume and not 0 < split < steps:
        raise ValueError('Checkpoint must lie inside the same positive full planned horizon.')
    simulation.config.run_time = mw.TimeConfig(time_steps=steps)
    _log('prepared', log_path=output / 'run.log', amplitudes=amplitudes, dt_s=dt, planned_steps=steps, split_step=split,
        requested_duration_s=case.duration, test_override=planned_steps is not None)
    started = time.perf_counter()
    checkpoint_info = None
    if resume:
        checkpoint = prepared.run_until(split)
        checkpoint.save(output / 'checkpoint.pt')
        checkpoint_info = {'step': checkpoint.step, 'total_steps': checkpoint.total_steps,
            'dt': checkpoint.dt, 'signature': checkpoint.signature,
            'sha256': hashlib.sha256((output / 'checkpoint.pt').read_bytes()).hexdigest()}
        _log('checkpoint_saved', log_path=output / 'run.log', step=checkpoint.step, total_steps=checkpoint.total_steps)
        del checkpoint, prepared
        torch.cuda.empty_cache()
        restored = mw.FDTDResumeCheckpoint.load(output / 'checkpoint.pt', map_location='cpu')
        prepared = simulation.prepare()
        if float(prepared.solver.dt) != dt or restored.total_steps != steps:
            raise RuntimeError('Fresh preparation changed actual dt or planned checkpoint horizon.')
        result = prepared.run(resume_from=restored)
    else:
        result = prepared.run()
    elapsed = time.perf_counter() - started
    solver = result.solver
    prepared_scene = solver.scene
    grid = {axis: getattr(prepared_scene, axis + '_nodes64') for axis in 'xyz'}
    operator = runtime_operator_signature(solver)
    contract = SpectralContract(tensor_signature({'maxwell': before['maxwell']['source_sha256'],
        'actual_core': before['actual_core_sha256']}), tensor_signature(grid),
        tensor_signature({'material': _material_signature(solver), 'port_operator': operator}),
        tensor_signature({'schema': schema, 'rows': rows, 'frequencies': case.frequencies,
            'bounds': bounds, 'groups': groups}), steps, 'none')
    basis = capture_loss_basis(result, contract)
    _require_physical_metal_basis(case, basis)
    source_spectra = {}
    for runtime in solver._port_runtimes:
        voltage, _ = runtime.drive_accumulator.phasors(
            normalization='peak', integration_dt=solver._spectral_integration_dt,
            reference_frequency=solver._spectral_reference_frequency)
        if tuple(float(frequency) for frequency in runtime.frequencies) != case.frequencies:
            raise RuntimeError('Actual source-spectrum frequency axis changed.')
        source_spectra[runtime.port.name] = {'thevenin_peak_voltage': voltage.detach().cpu(),
            'resistance_ohm': float(runtime.lumped.resistance),
            'available_power_w': result.port(runtime.port.name).available_power.detach().cpu()}
    torch.save(basis, output / 'basis.pt')
    torch.save({name: getattr(solver, name).detach().cpu().clone()
        for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        + tuple(name + '_imag' for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz') if hasattr(solver, name + '_imag'))},
        output / 'final-fields.pt')
    result.save(output / 'result.pt')
    native = {}
    for name, (component, index) in expected.items():
        payload = result.monitor(name)
        if tuple(payload['field_indices'][component]) != index:
            raise RuntimeError('Actual physical V/I monitor changed native index.')
        native[name] = payload['components'][component].detach().cpu().numpy().astype(np.complex128).reshape(len(case.frequencies))
    state = np.asarray([[sum(native[name] * weight for name, _, weight in row) for row in pair] for pair in rows])
    legacy_rows = scene.metadata.get('legacy_native_VI_observation_contract', {}).get('integration_rows')
    legacy_state = None if legacy_rows is None else np.asarray([[sum(native[name] * weight for name, _, weight in row) for row in pair] for pair in legacy_rows])
    tail = {}
    for monitor in tails:
        payload = result.monitor(monitor.name)
        field = torch.as_tensor(payload['field']).detach().cpu().numpy()
        times = torch.as_tensor(payload['t']).detach().cpu().numpy()
        peak = float(abs(field).max())
        selection = times >= steps * dt - 2e-9
        tail[monitor.name] = {'t_s': times, 'field': field,
            'last2ns_peak_relative': float(abs(field[selection]).max() / peak) if peak and bool(selection.any()) else None}
    regional = regional_plane_power(result, scene, schema, bounds)
    after = execution_identity()
    if before != after:
        raise RuntimeError('Executed source/core/helper identity changed; preserve and reject this run.')
    report = {'case': asdict(case), 'amplitudes': amplitudes, 'actual_dt_s': dt,
        'planned_steps': steps, 'actual_duration_s': steps * dt, 'test_step_override': planned_steps,
        'seconds_including_checkpoint_prepare': elapsed, 'physical_state': state, 'native_phasors': native,
        'legacy_native_physical_state': legacy_state,
        'VI_observation_roles': scene.metadata['VI_observation_roles'],
        'legacy_native_VI_observation_contract': scene.metadata.get('legacy_native_VI_observation_contract'),
        'actual_source_spectra': source_spectra,
        'runtime_environment': {'python': sys.executable, 'torch': torch.__version__,
            'torch_cuda': torch.version.cuda, 'cuda_home': os.environ.get('CUDA_HOME'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'device_name': torch.cuda.get_device_name(solver.device),
            'device_capability': torch.cuda.get_device_capability(solver.device)},
        'tails': tail, 'regional_power': regional, 'source_before': before, 'source_after': after,
        'all_source_identity_unchanged': True, 'port_operator_sha256': operator,
        'contract': asdict(contract), 'checkpoint': checkpoint_info,
        'actual_field_graph': bool(solver._cuda_graph_active),
        'actual_observer_graph': bool(solver._observer_cuda_graph_active)}
    report['saved_files_sha256'] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ('basis.pt', 'result.pt', 'final-fields.pt', 'input.json')
        + (('checkpoint.pt',) if resume else ())}
    (output / 'execution.json').write_text(json.dumps(encode(report), indent=2, allow_nan=False) + '\n')
    _log('drive_complete', log_path=output / 'run.log', seconds=elapsed, path=str(output), operator_sha256=operator)
    return report


def _require_physical_metal_basis(case, basis):
    if case.metal == 'copper' and not basis.realizations:
        raise RuntimeError('A copper pilot must capture its actual nonempty copper realizations.')
    if case.metal == 'pec' and (basis.realizations or basis.conductor_records):
        raise RuntimeError('A PEC pilot must not contain fabricated copper realizations or records.')


def _geometry_equal(left, right):
    return tensor_signature(left) == tensor_signature(right)


class CoherentObservations:
    """Detached coherent view consumed by the unchanged shared loss reducer."""
    def __init__(self, results, bases, coefficients):
        self.results = results
        self.frequencies = tuple(results[0].frequencies)
        if any(tuple(result.frequencies) != self.frequencies or result.stats().get('normalize_source', False) for result in results):
            raise ValueError('Actual coherent Results disagree on frequency order or normalization.')
        if torch.is_tensor(coefficients):
            # Own the values while retaining the caller's differentiable graph.
            self.coefficients = coefficients.to(dtype=torch.complex128, device='cpu').clone()
        else:
            # Broadcast NumPy views can be read-only; never alias caller storage.
            self.coefficients = torch.as_tensor(np.array(coefficients, copy=True),
                dtype=torch.complex128, device='cpu')
        self.conductor_losses = coherent_conductor_records(bases, self.coefficients)

    def stats(self):
        return {'normalize_source': False}

    @staticmethod
    def _components(payloads, output, weights, component_key='components'):
        keys = output[component_key].keys()
        if any(payload.get(component_key, {}).keys() != keys for payload in payloads):
            raise ValueError('Actual coherent native-plane component supports differ.')
        for component, entry in output[component_key].items():
            originals = [payload[component_key][component] for payload in payloads]
            geometry = {key: value for key, value in entry.items() if key != 'data'}
            if any(not _geometry_equal(geometry, {key: item for key, item in candidate.items() if key != 'data'})
                   for candidate in originals):
                raise ValueError('Actual coherent native-plane geometry differs.')
            entry['data'] = coherent_fields([{'E': torch.as_tensor(candidate['data'])[None]}
                for candidate in originals], weights)['E'][0]

    @staticmethod
    def _native_products(payloads, output, weights):
        if any(('native_components' in value) != ('native_components' in output) for value in payloads):
            raise ValueError('Coherent native-product field supports differ.')
        if 'native_components' in output:
            CoherentObservations._components(payloads, output, weights, 'native_components')
        heads=output.get('native_flux_heads',())
        if any(len(value.get('native_flux_heads',())) != len(heads) for value in payloads):
            raise ValueError('Coherent copper flux side-head counts differ.')
        for index, head in enumerate(heads):
            originals=[value['native_flux_heads'][index] for value in payloads]
            geometry={key:value for key,value in head.items() if key not in ('negative','positive')}
            if any(not _geometry_equal(geometry,{key:value for key,value in candidate.items() if key not in ('negative','positive')}) for candidate in originals):
                raise ValueError('Coherent copper flux side-head geometry differs.')
            for side in ('negative','positive'):
                head[side]=coherent_fields([{'Et':torch.as_tensor(candidate[side])[None]} for candidate in originals],weights)['Et'][0]

    def monitor(self, name, *, frequency):
        index = self.frequencies.index(frequency)
        payloads = [result.monitor(name, frequency=frequency) for result in self.results]
        if any(tuple(payload['frequencies']) != (frequency,) for payload in payloads):
            raise ValueError('Actual coherent monitor changed requested frequency.')
        output = copy.deepcopy(payloads[0])
        weights = self.coefficients[index:index + 1]
        if 'faces' in output:
            if any(payload['faces'].keys() != output['faces'].keys() for payload in payloads):
                raise ValueError('Actual coherent closed surfaces have different face sets.')
            for name, face in output['faces'].items():
                originals = [payload['faces'][name] for payload in payloads]
                keys = tuple(component for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz') if component in face)
                geometry_names = ('axis', 'normal_direction', 'position', 'x', 'y', 'z', 'cell_widths', 'collocated')
                geometry = {key: face[key] for key in geometry_names if key in face}
                if any(not _geometry_equal(geometry, {key: value[key] for key in geometry_names if key in value})
                       or tuple(component for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz') if component in value) != keys
                       for value in originals):
                    raise ValueError('Actual coherent face geometry or component supports differ.')
                combined = coherent_fields([{key: torch.as_tensor(value[key])[None] for key in keys} for value in originals], weights)
                for key in keys:
                    face[key] = combined[key][0]
                if 'components' in face:
                    self._components(originals, face, weights)
                self._native_products(originals, face, weights)
                density, area = plane_normal_poynting(face)
                face['flux'] = face['power'] = (density * area).sum()
            output['flux'] = output['power'] = sum(face['flux'] for face in output['faces'].values())
        else:
            self._components(payloads, output, weights)
            self._native_products(payloads, output, weights)
            # This view exposes exact native component data only. A collocated
            # top-level presentation retained from a single drive would be
            # stale and is not used by the regional native-plane reducer.
            for key in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'flux', 'power'):
                output.pop(key, None)
        return output


def reduce_pair(case, folders):
    if len(folders) != 2 or len({Path(folder).resolve() for folder in folders}) != 2:
        raise ValueError('First scalar pilot requires exactly two completed drive folders.')
    scene, groups, bounds, schema, rows, expected, tails = build_case(case)
    reports = [json.loads((Path(folder) / 'execution.json').read_text()) for folder in folders]
    for index, (folder, report) in enumerate(zip(folders, reports)):
        if (canonical_case_payload(report['case']) != encode(asdict(case)) or report['amplitudes'] != list(DRIVE_BASIS[index])
            or report['source_before'] != report['source_after'] or not report['all_source_identity_unchanged']
            or report['source_before'] != reports[0]['source_before']):
            raise ValueError('Actual pair must retain its complete declared case, independent source columns and executed source identity.')
        if not case.record_legacy_native_vi and report.get('legacy_native_physical_state') is not None:
            raise ValueError('A primary-only saved case cannot carry undeclared secondary V/I data.')
        if (report['test_step_override'] is not None or report['actual_dt_s'] != reports[0]['actual_dt_s']
            or report['planned_steps'] != reports[0]['planned_steps']
            or report['planned_steps'] != math.ceil(case.duration / report['actual_dt_s'])):
            raise ValueError('Physical pilot reduction requires two completed full-duration fixed-step records, not short execution controls.')
        required = {'basis.pt', 'result.pt', 'final-fields.pt', 'input.json'}
        if report['checkpoint'] is not None:
            required.add('checkpoint.pt')
        if set(report['saved_files_sha256']) != required:
            raise ValueError('Actual pair must retain the complete raw-file signature set.')
        for name, expected_sha in report['saved_files_sha256'].items():
            if hashlib.sha256((Path(folder) / name).read_bytes()).hexdigest() != expected_sha:
                raise ValueError('Actual pair raw Result, sidecar, checkpoint or source input changed after execution.')
    bases = [torch.load(Path(folder) / 'basis.pt', weights_only=False, map_location='cpu') for folder in folders]
    results = [mw.Result.load(Path(folder) / 'result.pt', scene=scene, map_location='cpu') for folder in folders]
    def complex_array(value):
        if isinstance(value, dict) and set(value) == {'real', 'imag'}:
            return complex(value['real'], value['imag'])
        return [complex_array(item) for item in value] if isinstance(value, list) else value
    states = np.asarray([complex_array(report['physical_state']) for report in reports], dtype=np.complex128)
    if states.shape != (2, len(case.positions), 2, len(case.frequencies)) or not np.isfinite(states).all():
        raise ValueError('Completed physical state must cover two drives, every station, V/I and frequency.')
    helper, _ = load_helper('stripline')
    records = []
    coefficient = []
    for index, frequency in enumerate(case.frequencies):
        state = states[:, :, :, index].transpose(1, 2, 0)
        transfers = helper.state_transfer(state[:, 0], state[:, 1], case.positions)
        left = case.positions.index(-.015)
        right = case.positions.index(.015)
        x0 = state[left].copy(); x0[1] *= 50
        x1 = state[right].copy(); x1[1] *= 50
        transfer = np.linalg.solve(x0.T, x1.T).T
        values, vectors = np.linalg.eig(transfer)
        forward = [j for j, value in enumerate(values) if np.angle(value) > 0]
        if len(forward) != 1:
            raise ValueError('Physical transfer has no unique unfitted forward branch.')
        selected = forward[0]
        vector = vectors[:, selected]
        power = .5 * np.real(vector[0] * np.conj(vector[1] / 50))
        if power <= 0:
            raise ValueError('Physical forward branch has nonpositive passive V/I power.')
        vector /= np.sqrt(power)
        c = np.linalg.solve(x0, vector)
        coefficient.append(c)
        distance = case.positions[right] - case.positions[left]
        records.append({'frequency_hz': frequency, 'all_station_transfers': transfers,
            'roi_positions_m': [case.positions[left], case.positions[right]],
            'roi_transfer_scaled': transfer, 'roi_eigenvalues': values,
            'forward_eigenvector_scaled': vector, 'coherent_coefficients': c,
            'state_condition_number_50ohm': [float(np.linalg.cond(x0)), float(np.linalg.cond(x1))],
            'solve_residual': float(np.linalg.norm(x0 @ c - vector) / np.linalg.norm(vector)),
            'characteristic_impedance_ohm': complex(50 * vector[0] / vector[1]),
            'alpha_np_per_m': float(-np.log(abs(values[selected])) / distance),
            'beta_rad_per_m': float(np.angle(values[selected]) / distance)})
    observation = CoherentObservations(results, bases, np.asarray(coefficient))
    power = regional_plane_power(observation, scene, schema, bounds)
    reference = analytic_homogeneous_line('stripline', case.frequencies, sigma_e=case.sigma)
    tail_values = [trace['last2ns_peak_relative'] for report in reports for trace in report['tails'].values()]
    closure_values = power['balance_relative_to_entering']
    source_ratios = []
    for report in reports:
        for source in report['actual_source_spectra'].values():
            magnitudes = np.abs(np.asarray(complex_array(source['thevenin_peak_voltage'])))
            source_ratios.append(float(magnitudes.min() / magnitudes.max()) if magnitudes.max() > 0 else 0.)
    gate_results = {'source_identity_unchanged': all(report['all_source_identity_unchanged'] for report in reports),
        'same_actual_port_operator': len({report['port_operator_sha256'] for report in reports}) == 1,
        'state_condition': all(max(record['state_condition_number_50ohm']) <= 1e4 for record in records),
        'coherent_solve': all(record['solve_residual'] <= 1e-10 for record in records),
        'source_band_coverage': all(ratio >= 1e-3 for ratio in source_ratios),
        'tail_decay': bool(tail_values) and all(value is not None and value <= 1e-3 for value in tail_values),
        'coherent_regional_power_balance': all(value is not None and value <= .03 for value in closure_values)}
    # With real Cu the homogeneous PEC reference is a diagnostic decomposition,
    # not an independent truth for total attenuation or complex impedance.
    return {'schema': 'combined-stripline-two-drive-physical-loss/1', 'case': asdict(case),
        'frequency_records': records, 'coherent_regional_power': power,
        'homogeneous_PEC_reference_diagnostic': reference,
        'actual_source_band_min_relative': source_ratios, 'pilot_functional_gates': gate_results,
        'pilot_functional_passed': all(gate_results.values()), 'physical_release_qualified': False,
        'release_gates': {'beta_and_Z_relative': .02, 'resolved_alpha_relative': .05,
            'complex_S_absolute': .02, 'line_power_balance_relative': .03,
            'state_condition_number_max': 1e4, 'solve_residual_max': 1e-10, 'tail_peak_relative_max': 1e-3},
        'qualification_status': 'Diagnostic pilot only: Cu/PEC and sigma controls, source/time/padding and mesh refinement, resolved two-length attenuation and independent reference remain required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Explicit bounded two-drive CUDA pilot; otherwise CPU prepare only.')
    parser.add_argument('--public-line', choices=('cpw','differential_pair'), default=None,
        help='Explicit one-column public Cu query workflow; no automatic drive queue.')
    parser.add_argument('--length-index', type=int, choices=(0,1), default=0)
    parser.add_argument('--drive-index', type=int, default=0)
    # Convergence controls for --public-line only; defaults keep the fixed case.
    parser.add_argument('--dx', type=float, default=None, help='Transverse step inside the 20 mm refinement radius.')
    parser.add_argument('--longitudinal-dl', type=float, default=None)
    parser.add_argument('--padding', type=float, default=None, help='Transverse padding beyond the 60/50 mm core.')
    parser.add_argument('--sigma', type=float, default=None, help='Bulk dielectric conductivity (0 isolates Cu loss).')
    args = parser.parse_args()
    overrides = {k: getattr(args, k) for k in ('dx', 'longitudinal_dl', 'padding', 'sigma') if getattr(args, k) is not None}
    if args.public_line is not None:
        from scripts.dev.fdtd.printed_line_public_execution import command
        return command(args.output,args.public_line,args.length_index,args.drive_index,execute=args.execute,overrides=overrides)
    if args.length_index != 0 or args.drive_index != 0 or overrides:
        parser.error('--length-index, --drive-index and convergence overrides require --public-line')
    if args.output.exists():
        raise FileExistsError(args.output)
    case = PilotCase()
    if not args.execute:
        scene, groups, bounds, schema, rows, expected, tails = build_case(case)
        args.output.mkdir(parents=True)
        record = {'case': asdict(case), 'groups': groups, 'bounds': bounds, 'regional_schema': schema,
            'source_drive_basis': DRIVE_BASIS, 'source_basis_condition_number': float(np.linalg.cond(DRIVE_BASIS)),
            'physical_integration_rows': rows, 'expected_native_probes': expected,
            'source': execution_identity(), 'cuda_initialized': torch.cuda.is_initialized(),
            'status': 'CPU prepared only; no CUDA execution, checkpoint or physical acceptance.'}
        (args.output / 'preparation.json').write_text(json.dumps(encode(record), indent=2, allow_nan=False) + '\n')
        return
    args.output.mkdir(parents=True)
    folders = [args.output / f'drive{column}' for column in range(2)]
    for column, amplitudes in enumerate(DRIVE_BASIS):
        run_drive(case, amplitudes, folders[column])
    report = reduce_pair(case, folders)
    (args.output / 'physical-loss-report.json').write_text(json.dumps(encode(report), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
