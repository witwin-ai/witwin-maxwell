"""Read all original failed CUDA controls; never relabel their original verdict."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase, build_case, CoherentObservations
from scripts.dev.fdtd.printed_line_loss_assertions import superposition_metrics, flux_metrics, physical_electric_vectors
from scripts.dev.fdtd.printed_line_regional_power import regional_plane_power


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-output', type=Path, required=True)
    parser.add_argument('--junit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    case = replace(PilotCase(), axial_half_length=.06, padding=.02, separation=.06,
        positions=(-.015, 0., .015), frequencies=(.8e9, 1e9, 1.2e9))
    scene, _, bounds, schema, *_ = build_case(case)
    input_hashes = {str(path): sha(path) for path in args.original_output.rglob('*') if path.is_file()}
    results = []
    for index, (phase, graph) in enumerate(((-1., False), (-1., True), (1j, False), (1j, True))):
        directory = args.original_output / 'pytest' / f'test_actual_two_drive_spectra_{index}'
        folders = [directory / name for name in ('basis0_resume', 'basis1', 'direct', 'basis0_uninterrupted')]
        data = [mw.Result.load(path / 'result.pt', scene=scene, map_location='cpu') for path in folders]
        bases = [torch.load(path / 'basis.pt', weights_only=False, map_location='cpu') for path in folders]
        combined = CoherentObservations(data[:2], bases[:2], torch.tensor([[1., phase]] * 3, dtype=torch.complex128))
        physical_vectors = []
        for frequency in case.frequencies:
            vectors, volumes, geometry = physical_electric_vectors([combined, data[2], *data[:2]], scene, schema, frequency)
            metric = superposition_metrics(vectors[0], vectors[1], vectors[2:], (1., phase), physical_weights=volumes)
            physical_vectors.append(dict(frequency_hz=frequency, physical_norm_geometry=geometry, metric=metric))
        fields = []
        for frequency in case.frequencies:
            for component, selection in schema.items():
                for plane in selection['planes']:
                    samples = [result.monitor(plane['name'], frequency=frequency)['components'][component]['data'] for result in data]
                    actual = combined.monitor(plane['name'], frequency=frequency)['components'][component]['data']
                    metric = superposition_metrics(actual, samples[2], samples[:2], (1., phase))
                    metric.update(component=component, plane=plane['name'], frequency_hz=frequency)
                    fields.append(metric)
        for name, record in combined.conductor_losses.items():
            lookup = [{entry['component']: entry for entry in result.conductor_losses[name]['bulk_electric_traces']} for result in data]
            for trace in record['bulk_electric_traces']:
                component = trace['component']
                for frequency_index, frequency in enumerate(case.frequencies):
                    metric = superposition_metrics(trace['electric_fields'][frequency_index], lookup[2][component]['electric_fields'][frequency_index],
                        [lookup[column][component]['electric_fields'][frequency_index] for column in (0, 1)], (1., phase))
                    metric.update(component=component, copper_face=name, frequency_hz=frequency)
                    fields.append(metric)
        actual_power = regional_plane_power(combined, scene, schema, bounds)
        direct_power = regional_plane_power(data[2], scene, schema, bounds)
        heat = {name: (abs(torch.as_tensor(actual_power[name], dtype=torch.float64)
            - torch.as_tensor(direct_power[name], dtype=torch.float64))
            / torch.as_tensor(direct_power[name], dtype=torch.float64).abs()).tolist()
            for name in ('bulk_conduction_w', 'copper_film_w')}
        first, last = [torch.load(folders[column] / 'final-fields.pt', weights_only=False, map_location='cpu') for column in (0, 3)]
        resume = {component: bool(torch.equal(first[component], last[component])) for component in first}
        flux = flux_metrics(actual_power, direct_power)
        results.append({'case_index': index, 'phase': str(phase), 'graph': graph,
            'original_verdict': 'FAIL: original criteria retained; this is reduction-only diagnosis, not a replacement GPU pass',
            'physical_full_E_vector_metrics': physical_vectors, 'field_metrics': fields, 'conditioned_field_checks_would_pass': all(metric['passed'] for metric in fields),
            'legacy_pointwise_failure_count': sum(metric['original_pointwise_failure_count'] for metric in fields),
            'heat_relative_errors': heat, 'signed_flux_metrics': flux, 'bitwise_resume_final_fields': resume})
    witnesses = []
    for path in sorted((args.original_output / 'witness').glob('*/witness.json')):
        witness = json.loads(path.read_text())
        pairs = list(zip(witness['events'][::2], witness['events'][1::2]))
        coefficient_checks = []
        for before, after in pairs:
            a, b = [torch.load(event['path'], weights_only=False, map_location='cpu') for event in (before, after)]
            lookup = {(entry['face'], entry['component']): entry for entry in b['copper']}
            unchanged = all(all(torch.equal(entry[name], lookup[(entry['face'], entry['component'])][name])
                for name in ('B', 'rates', 'D', 'indices', 'area')) for entry in a['copper'])
            coefficient_checks.append({'operation': before['operation'], 'actual_static_coefficients_unchanged': unchanged})
        witnesses.append({'path': str(path), 'sha256': sha(path), 'nodeid': witness['nodeid'],
            'actual_counter_values': witness['actual_counter_values'],
            'actual_fields_nonzero_after_advancement': witness['actual_fields_nonzero_after_advancement'],
            'actual_copper_coefficients_captured': witness['actual_copper_coefficients_captured'],
            'before_after_coefficient_checks': coefficient_checks})
    junit = ET.parse(args.junit).getroot()
    suites = list(junit) if junit.tag == 'testsuites' else [junit]
    original = {key: sum(int(suite.get(key, '0')) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    if any(sha(path) != expected for path, expected in input_hashes.items()): raise RuntimeError('Original failed control evidence changed during diagnosis.')
    assert not torch.cuda.is_initialized()
    record = {'schema': 'original-failed-loss-controls-diagnosis/1', 'original_junit': original,
        'original_junit_sha256': sha(args.junit), 'original_input_files_sha256': input_hashes,
        'all_original_inputs_unchanged': True, 'GPU_executed_by_this_audit': False,
        'verdict': 'Original failures preserved. Conditioned criteria are CPU reanalysis only; new actual4GPU execution remains required.',
        'cases': results, 'actual_runtime_witnesses': witnesses, 'cuda_initialized': False,
        'driver_sha256': sha(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'original_junit': original, 'cases': len(results), 'witnesses': len(witnesses)}), flush=True)


if __name__ == '__main__': main()
