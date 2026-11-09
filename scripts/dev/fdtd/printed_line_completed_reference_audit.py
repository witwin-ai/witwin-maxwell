"""Audit completed independent native-field references without advancing fields.

Run from the executed immutable geometry package, with its PYTHONPATH. Raw
timestamps/native addresses and physical V/I are re-read; process success is
not treated as impedance, propagation, continuum or modal qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import witwin.core
import witwin.maxwell as mw

from benchmark.scenes.rf.printed_lines import printed_line_scene, analytic_homogeneous_line
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.printed_line_external_reference import (
    contour_driver, native_addresses, read_native_spectra, reference_identity,
    reference_line_scene, validate_completed_execution, completed_drive_provenance,
)
from scripts.dev.fdtd.printed_line_physical_modes import coupled_physical_state, coupled_transfer
from witwin.maxwell.scene import prepare_scene


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(value):
    if isinstance(value, dict) and set(value) == {'real', 'imag'}:
        return complex(value['real'], value['imag'])
    return [decode(item) for item in value] if isinstance(value, list) else value


def encode(value):
    if isinstance(value, complex): return {'real': value.real, 'imag': value.imag}
    if isinstance(value, np.ndarray): return encode(value.tolist())
    if isinstance(value, dict): return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [encode(item) for item in value]
    return value


def mode_rows(data):
    if 'state_transfer' in data:
        return data['state_transfer']
    if [row['frequency_hz'] for row in data['frequency_records']] != data['frequencies_hz']:
        raise ValueError('Native frequency records are incomplete or reordered.')
    return [row['state_transfer'] for row in data['frequency_records']]


def compare_modes(coarse, fine, *, scope, relative_to_fine=False):
    frequencies = fine['frequencies_hz']
    left_frequency_rows, right_frequency_rows = mode_rows(coarse), mode_rows(fine)
    if (not frequencies or len(set(frequencies)) != len(frequencies)
        or coarse['frequencies_hz'] != frequencies
        or len(left_frequency_rows) != len(frequencies) or len(right_frequency_rows) != len(frequencies)):
        raise ValueError('Physical comparison requires complete matching frequency order.')
    records = []
    for frequency, left_rows, right_rows in zip(frequencies, left_frequency_rows, right_frequency_rows):
        lookup = {tuple(row['positions_m']): row for row in left_rows}
        if (not right_rows or len(lookup) != len(left_rows)
            or len({tuple(row['positions_m']) for row in right_rows}) != len(right_rows)):
            raise ValueError('Duplicated or empty physical station pair.')
        pairs = []
        for right in right_rows:
            left = lookup[tuple(right['positions_m'])]
            modes = right.get('modes') or {'single': right}
            values = {}
            for mode, b in modes.items():
                a = left['modes'][mode] if mode != 'single' else left
                key = 'characteristic_impedance_ohm' if mode != 'single' else 'forward_characteristic_impedance_ohm'
                za, zb = complex(decode(a[key])), complex(decode(b[key]))
                if not np.isfinite([za, zb, a['beta_rad_per_m'], b['beta_rad_per_m'],
                    a['alpha_np_per_m'], b['alpha_np_per_m']]).all():
                    raise ValueError('Nonfinite retained physical mode.')
                denominator = zb if relative_to_fine else za
                beta_denominator = b['beta_rad_per_m'] if relative_to_fine else a['beta_rad_per_m']
                values[mode] = {'coarse_or_native_Z_ohm': za, 'fine_reference_Z_ohm': zb,
                    'complex_Z_relative_difference': abs(zb - za) / abs(denominator),
                    'coarse_or_native_beta_rad_per_m': a['beta_rad_per_m'], 'fine_reference_beta_rad_per_m': b['beta_rad_per_m'],
                    'beta_relative_difference': abs(b['beta_rad_per_m'] - a['beta_rad_per_m']) / abs(beta_denominator),
                    'coarse_or_native_signed_alpha_np_per_m': a['alpha_np_per_m'],
                    'fine_reference_signed_alpha_np_per_m': b['alpha_np_per_m']}
            pairs.append({'positions_m': right['positions_m'], 'modes': values})
        records.append({'frequency_hz': frequency, 'station_pairs': pairs})
    metrics = [mode for record in records for pair in record['station_pairs'] for mode in pair['modes'].values()]
    return {'scope': scope, 'denominator': 'fine reference' if relative_to_fine else 'coarse level',
        'frequency_records': records,
        'max_complex_Z_relative_difference': max(row['complex_Z_relative_difference'] for row in metrics),
        'max_beta_relative_difference': max(row['beta_relative_difference'] for row in metrics)}


def symmetry_diagnostic(records, spectra, kind, frequencies):
    """Unfitted magnetic ground-symmetry diagnostic within each actual drive."""
    if kind not in ('stripline', 'cpw'): return None
    component, mirror = ('Hy', 2) if kind == 'stripline' else ('Hz', 1)
    selected = {tuple(np.round(record['physical_component_position_m'], 12)): name
        for name, record in records.items() if record['component'] == component}
    pairs = []
    for point, name in selected.items():
        if point[mirror] <= 0: continue
        reflected = list(point); reflected[mirror] *= -1
        if tuple(reflected) in selected:
            pairs.append((name, selected[tuple(reflected)]))
    if not pairs: raise ValueError('No native mirrored magnetic support for ground-symmetry audit.')
    a = np.asarray([spectra[left] for left, right in pairs])
    b = np.asarray([spectra[right] for left, right in pairs])
    # Target balanced signal mode is odd in this magnetic component. No fitted
    # global amplitude or phase is removed from either actual field sample.
    denominator = np.linalg.norm(a - b, axis=0)
    ratio = np.linalg.norm(a + b, axis=0) / np.maximum(denominator, np.finfo(float).tiny)
    return {'component': component, 'mirror_axis': 'xyz'[mirror], 'matched_pair_count': len(pairs),
        'coordinate_pairing_rounding_m': 1e-12, 'frequencies_hz': frequencies,
        'even_to_target_odd_magnetic_l2': ratio,
        'scope': 'Direct native magnetic parity, per drive. A nonzero ratio indicates broken balanced-ground symmetry; this alone does not identify a complete modal scattering basis.'}


def audit(reference_directory, job_directory):
    directory, job = Path(reference_directory), Path(job_directory)
    report_path = directory / 'report.json'; report = json.loads(report_path.read_text())
    all_files = {str(path.relative_to(directory)): sha(path) for path in sorted(directory.rglob('*')) if path.is_file()}
    guards = ('source_unchanged', 'fixture_unchanged', 'helper_unchanged', 'state_helper_unchanged', 'driver_unchanged', 'reference_unchanged')
    if any(report[key] is not True for key in guards): raise ValueError('Completed reference identity guard failed.')
    config = json.loads((job / 'config.json').read_text()); completion = json.loads((job / 'completion.json').read_text())
    if (completion['status'] != 'process_completed' or completion['returncode'] != 0
        or completion['config_sha256'] != sha(job / 'config.json')):
        raise ValueError('Root durable job did not retain a successful unchanged configuration.')
    mismatches = [path for path, expected in config['required_sha256'].items() if not Path(path).is_file() or sha(path) != expected]
    if mismatches: raise ValueError('Required executed source/input changed: ' + str(mismatches[:3]))
    identity = package_identity(mw, 'witwin-maxwell')
    if identity['source_sha256'] != report['maxwell_source_sha256']:
        raise ValueError('Reconstruction must import the executed immutable numerical geometry source.')
    fixture = Path(inspect.getfile(printed_line_scene))
    contour = Path(config['argv'][config['argv'].index('--contour-driver') + 1])
    if sha(fixture) != report['fixture_sha256'] or sha(contour) != report['contour_driver_sha256']:
        raise ValueError('Executed geometry or physical integration helper changed.')
    source_root = min(map(Path, report['reference']['pins']), key=lambda path: len(path.parts))
    actual_reference = reference_identity(source_root, SimpleNamespace(__file__=report['reference']['python_package']))
    if actual_reference != report['reference']: raise ValueError('Primary reference source/build/binding changed.')
    helper = contour_driver(contour)
    args = SimpleNamespace(kind=report['kind'], **report['configuration'])
    scene = reference_line_scene(args); prepared = prepare_scene(scene)
    positions = report['positions_m']; frequencies = report['frequencies_hz']
    if frequencies != [.8e9, .9e9, 1e9, 1.1e9, 1.2e9]: raise ValueError('Missing or reordered declared five-frequency axis.')
    if report['kind'] == 'differential_pair':
        _, rows, expected = helper.probes(scene, args.dx, args.longitudinal_dl, args.padding, positions, 'pec')
    else:
        _, rows, expected = helper.native_probe_monitors(scene, report['kind'], args.dx, args.longitudinal_dl, args.padding, positions, 'pec')
    addresses = native_addresses(prepared, expected)
    if addresses != report['probe_contract']: raise ValueError('Native probe addresses/positions changed during reconstruction.')
    expected_drives = 4 if report['kind'] == 'differential_pair' else 2
    if len(report['runs']) != expected_drives: raise ValueError('Incomplete independent drive basis.')
    values, run_audits = [], []
    for index, run in enumerate(report['runs']):
        folder = directory / f'drive-{index}'
        execution = json.loads((folder / 'execution.json').read_text())
        validate_completed_execution(execution, folder / 'simulation.xml', report['dt_s'], report['steps'])
        if run['active_drive'] != index or run['seconds'] != execution['seconds']:
            raise ValueError('Drive order or retained true execution timing changed.')
        field_driver = execution.get('executed_driver_sha256')
        if run['reused_completed_drive']:
            provenance = completed_drive_provenance(folder, {'maxwell_source_sha256': report['maxwell_source_sha256'],
                'fixture_sha256': report['fixture_sha256'], 'contour_driver_sha256': report['contour_driver_sha256'],
                'state_helper_sha256': report['state_helper_sha256'], 'reference': report['reference']})
            if provenance != run['reuse_provenance']: raise ValueError('Reused-drive original provenance changed.')
            field_driver = provenance['executed_driver_sha256']
            original_directory = Path(provenance['original_drive_directory'])
            for relative, expected_sha in provenance['original_field_input_sha256'].items():
                if sha(original_directory / relative) != expected_sha or sha(folder / relative) != expected_sha:
                    raise ValueError('Reused completed field file differs from its pinned original: ' + relative)
            if sha(provenance['original_preparation_path']) != provenance['original_preparation_sha256']:
                raise ValueError('Original reference preparation changed.')
        else:
            provenance = None
        if (field_driver != run['field_execution_driver_sha256']
            or run['orchestration_driver_sha256'] != report['driver_sha256']):
            raise ValueError('Executed field/orchestration driver provenance is inconsistent.')
        coverage = []
        for name in addresses:
            raw = np.loadtxt(folder / 'raw' / name, comments='%')
            if raw.ndim != 2 or raw.shape[1] != 4 or len(raw) < 3 or not np.isfinite(raw).all() or not np.all(np.diff(raw[:, 0]) > 0):
                raise ValueError('Raw native trace is truncated, nonfinite or not strictly increasing: ' + name)
            interval = float(np.max(np.diff(raw[:, 0])))
            end_gap = report['actual_duration_s'] - raw[-1, 0]
            if end_gap > interval + 2 * report['dt_s'] or raw[0, 0] > interval:
                raise ValueError('Raw native trace does not span its declared full horizon: ' + name)
            coverage.append((len(raw), float(end_gap), interval))
        spectra, tails = read_native_spectra(folder / 'raw', addresses, frequencies)
        def integral(row): return sum(spectra[name] * weight for name, _, weight in row)
        if report['kind'] == 'differential_pair':
            values.append(np.asarray([coupled_physical_state(integral(common[0]), integral(odd[0]),
                integral(common[1]), integral(odd[1])) for common, odd in rows]))
        else:
            values.append(np.asarray([(integral(voltage), integral(current)) for voltage, current in rows]))
        run_audits.append({'drive': index, 'raw_probe_count': len(spectra),
            'sample_count_range': [min(row[0] for row in coverage), max(row[0] for row in coverage)],
            'max_end_gap_s': max(row[1] for row in coverage), 'max_sample_interval_s': max(row[2] for row in coverage),
            'seconds': execution['seconds'], 'reused_completed_drive': run['reused_completed_drive'],
            'field_execution_driver_sha256': field_driver, 'reuse_provenance_verified': provenance is not None,
            'raw_max_late_trace_relative': max(tails.values()),
            'magnetic_ground_symmetry': symmetry_diagnostic(addresses, spectra, report['kind'], frequencies)})
    values = np.asarray(values).transpose(3, 1, 2, 0)
    voltage = values[:, :, :2] if report['kind'] == 'differential_pair' else values[:, :, 0]
    current = values[:, :, 2:] if report['kind'] == 'differential_pair' else values[:, :, 1]
    retained = [np.asarray(decode(report[key]), dtype=np.complex128) for key in ('voltage_V', 'current_A')]
    if any(a.shape != b.shape or not np.isfinite(b).all() for a, b in zip((voltage, current), retained)):
        raise ValueError('Retained voltage/current station, mode, drive or frequency axes are incomplete.')
    errors = [float(np.max(abs(a - b)) / max(np.max(abs(b)), np.finfo(float).tiny)) for a, b in zip((voltage, current), retained)]
    if max(errors) > 2e-13: raise ValueError('Recomputed raw V/I differs from the completed report.')
    transfers = [coupled_transfer(row[:, :2], row[:, 2:], positions) if report['kind'] == 'differential_pair'
        else helper.state_transfer(row[:, 0], row[:, 1], positions) for row in values]
    recomputed = dict(report, state_transfer=json.loads(json.dumps(encode(transfers))))
    transfer_error = compare_modes(report, recomputed, scope='Raw re-reduction identity')
    if max(transfer_error['max_complex_Z_relative_difference'], transfer_error['max_beta_relative_difference']) > 2e-12:
        raise ValueError('Raw re-reduction changed physical eigenvectors or propagation.')
    analytic = None
    if report['kind'] in ('stripline', 'cpw'):
        reference = analytic_homogeneous_line(report['kind'], frequencies, sigma_e=args.sigma)
        records = []
        for i, frequency in enumerate(frequencies):
            records.append({'frequency_hz': frequency, 'analytic_Z_ohm': reference['characteristic_impedance_ohm'][i],
                'analytic_beta_rad_per_m': reference['beta_rad_per_m'][i],
                'station_pairs': [{'positions_m': row['positions_m'],
                    'complex_Z_relative_difference': abs(complex(decode(row['forward_characteristic_impedance_ohm'])) / reference['characteristic_impedance_ohm'][i] - 1),
                    'beta_relative_difference': abs(row['beta_rad_per_m'] / reference['beta_rad_per_m'][i] - 1)} for row in report['state_transfer'][i]]})
        analytic = {'scope': 'Infinite-ground homogeneous target TEM reference; finite boundaries and unbalanced-ground mode contamination are separate controls.',
            'frequency_records': records,
            'max_complex_Z_relative_difference': max(pair['complex_Z_relative_difference'] for record in records for pair in record['station_pairs']),
            'max_beta_relative_difference': max(pair['beta_relative_difference'] for record in records for pair in record['station_pairs'])}
    if any(sha(directory / relative) != expected for relative, expected in all_files.items()):
        raise RuntimeError('Completed raw/reference files changed during audit.')
    core_root = Path(next(iter(witwin.core.__path__))).resolve()
    core_files = {str(path.relative_to(core_root)): sha(path) for path in sorted(core_root.rglob('*'))
        if path.is_file() and path.suffix in {'.py', '.cpp', '.cu', '.h', '.cuh'}}
    if torch.cuda.is_initialized(): raise RuntimeError('Raw CPU inference audit unexpectedly initialized CUDA.')
    return {'schema': 'completed-independent-reference-raw-audit/1', 'reference_directory': str(directory.resolve()),
        'source_sha256': identity['source_sha256'], 'actual_module': mw.__file__, 'fixture_sha256': sha(fixture),
        'actual_core_root': str(core_root), 'actual_core_source_sha256': hashlib.sha256(json.dumps(core_files, sort_keys=True).encode()).hexdigest(),
        'root_job': str(job), 'root_required_sha256_count': len(config['required_sha256']),
        'all_root_required_files_verified': True, 'reference_build_verified': True,
        'reference_input_sha256': all_files, 'input_file_count': len(all_files),
        'input_bytes': sum((directory / path).stat().st_size for path in all_files), 'inputs_unchanged': True,
        'frequencies_hz': frequencies, 'positions_m': positions, 'runs': run_audits,
        'raw_voltage_relative_reproduction_error': errors[0], 'raw_current_relative_reproduction_error': errors[1],
        'raw_transfer_reproduction': transfer_error, 'physical_frequency_records': transfers,
        'analytic_target_TEM_diagnostic': analytic, 'same_grid_native_comparison': 'pending: no matching native full-band field run exists',
        'cuda_initialized': False, 'physical_release_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--coarse-reference', type=Path)
    parser.add_argument('--native-coarse', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    started_sha = sha(__file__)
    record = audit(args.reference, args.job)
    fine = json.loads((args.reference / 'report.json').read_text())
    if args.coarse_reference:
        coarse = json.loads(args.coarse_reference.read_text())
        keys = ('kind', 'geometry', 'reference', 'fixture_sha256', 'contour_driver_sha256')
        if fine['kind'] == 'differential_pair': keys += ('state_helper_sha256',)
        for key in keys:
            if coarse[key] != fine[key]: raise ValueError('Continuum control changed fixed physical/reference identity: ' + key)
        if any(coarse['configuration'][key] != value for key, value in fine['configuration'].items() if key not in ('dx', 'balanced_feeds')):
            raise ValueError('Continuum control changed more than transverse grid and automatic CFL dt.')
        if ([run['source_contract'] for run in coarse['runs']] != [run['source_contract'] for run in fine['runs']]):
            raise ValueError('Continuum control changed actual source/load geometry or drive basis.')
        record['continuum_reference_change'] = compare_modes(coarse, fine, scope='Same physical geometry/reference build; transverse h halves and the actual CFL dt changes. Reference continuum control only; native solver qualification remains pending.')
        record['continuum_reference_change'].update(input_sha256=sha(args.coarse_reference),
            Z_gate_passed=record['continuum_reference_change']['max_complex_Z_relative_difference'] <= .02,
            beta_gate_passed=record['continuum_reference_change']['max_beta_relative_difference'] <= .02)
    if args.native_coarse:
        native = json.loads(args.native_coarse.read_text())
        record['different_grid_native_diagnostic'] = compare_modes(native, fine, relative_to_fine=True,
            scope='Different transverse grids and potentially different feed bases. Combined discretization/source diagnostic only; this is not a same-grid solver-error or independent native acceptance result.')
        record['different_grid_native_diagnostic']['input_sha256'] = sha(args.native_coarse)
    record.update(driver_sha256=started_sha, driver_unchanged=started_sha == sha(__file__))
    if not record['driver_unchanged']: raise RuntimeError('Audit driver changed during reduction.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(encode(record), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'reference': str(args.reference), 'raw_files': record['input_file_count'],
        'raw_V_relative_error': record['raw_voltage_relative_reproduction_error'],
        'raw_I_relative_error': record['raw_current_relative_reproduction_error']}), flush=True)


if __name__ == '__main__': main()
