"""Real CUDA combined Cu/sigma source superposition and disk resume parity.

These bounded 1024-step controls exercise actual forward operators and spectra.
They are not full-duration attenuation or physical release acceptance.
"""
from dataclasses import replace
import json

import pytest
import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.printed_line_loss_execution import (
    DRIVE_BASIS, CoherentObservations, PilotCase, build_case, run_drive,
)
from scripts.dev.fdtd.printed_line_regional_power import regional_plane_power
from scripts.dev.fdtd.printed_line_loss_assertions import superposition_metrics, flux_metrics, physical_electric_vectors, actual_basis_bilinear_flux_metrics


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual two-drive loss execution requires CUDA.')


@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('phase', [-1., .5])
def test_actual_two_drive_spectra_and_shared_heat_match_direct_source_and_disk_resume(graph, phase, tmp_path):
    case = replace(PilotCase(), axial_half_length=.06, padding=.02, separation=.06,
        positions=(-.015, 0., .015), frequencies=(.8e9, 1e9, 1.2e9))
    scene, groups, bounds, schema, rows, expected, tails = build_case(case)
    folders = [tmp_path / name for name in ('basis0_resume', 'basis1', 'direct', 'basis0_uninterrupted')]
    direct_amplitudes = tuple(a + phase * b for a, b in zip(*DRIVE_BASIS))
    reports = [run_drive(case, amplitudes, folder, graph=graph, resume=resume,
        planned_steps=1024, stop_step=512) for amplitudes, folder, resume in zip(
        (*DRIVE_BASIS, direct_amplitudes, DRIVE_BASIS[0]), folders, (True, False, False, False))]
    assert all(report['actual_field_graph'] == graph and report['actual_observer_graph'] == graph for report in reports)
    assert len({report['port_operator_sha256'] for report in reports}) == 1
    bases = [torch.load(folder / 'basis.pt', weights_only=False) for folder in folders]
    assert all(basis.realizations for basis in bases)
    assert all(realization.B.numel() and realization.rates.numel() and realization.D.shape == (2, 2)
        for basis in bases for realization in basis.realizations)
    assert all(realization.banks == ('real',) for basis in bases for realization in basis.realizations)
    results = [mw.Result.load(folder / 'result.pt', scene=scene, map_location='cpu') for folder in folders]
    coefficients = torch.tensor([[1., phase]] * len(case.frequencies), dtype=torch.complex128)
    combined = CoherentObservations(results[:2], bases[:2], coefficients)
    field_diagnostics = []
    physical_vector_diagnostics = []
    for frequency in case.frequencies:
        vectors, volumes, geometry = physical_electric_vectors([combined, results[2], *results[:2]], scene, schema, frequency)
        metric = superposition_metrics(vectors[0], vectors[1], vectors[2:], (1., phase), physical_weights=volumes)
        metric.update(frequency_hz=frequency, physical_norm_geometry=geometry)
        physical_vector_diagnostics.append(metric)
        for component, selection in schema.items():
            for plane in selection['planes']:
                actual = combined.monitor(plane['name'], frequency=frequency)['components'][component]['data']
                direct = results[2].monitor(plane['name'], frequency=frequency)['components'][component]['data']
                basis = [result.monitor(plane['name'], frequency=frequency)['components'][component]['data'] for result in results[:2]]
                metric = superposition_metrics(actual, direct, basis, (1., phase))
                metric.update(component=component, frequency_hz=frequency, plane=plane['name'])
                field_diagnostics.append(metric)
    coherent_power = regional_plane_power(combined, scene, schema, bounds)
    direct_power = regional_plane_power(results[2], scene, schema, bounds)
    for key in ('bulk_conduction_w', 'copper_film_w'):
        torch.testing.assert_close(torch.as_tensor(coherent_power[key], dtype=torch.float64), torch.as_tensor(direct_power[key], dtype=torch.float64),
            rtol=3e-5, atol=1e-17)
    signed_flux = flux_metrics(coherent_power, direct_power)
    input_bilinear_flux = actual_basis_bilinear_flux_metrics(combined, results[2], results[:2], coefficients)
    assert input_bilinear_flux['original_entering_power_metrics'] == signed_flux
    assert bool(torch.as_tensor(direct_power['copper_film_w']).gt(0).all())
    # Actual split-trace phasors, not only integrated powers, must superpose.
    for name, record in combined.conductor_losses.items():
        actual = {entry['component']: entry for entry in record['bulk_electric_traces']}
        direct = {entry['component']: entry for entry in results[2].conductor_losses[name]['bulk_electric_traces']}
        for component in actual:
            basis_traces = [next(entry for entry in result.conductor_losses[name]['bulk_electric_traces'] if entry['component'] == component)
                for result in results[:2]]
            for index, frequency in enumerate(case.frequencies):
                metric = superposition_metrics(actual[component]['electric_fields'][index], direct[component]['electric_fields'][index],
                    [entry['electric_fields'][index] for entry in basis_traces], (1., phase))
                metric.update(component=component, frequency_hz=frequency, copper_face=name)
                field_diagnostics.append(metric)
    diagnostics = {'field_metrics': field_diagnostics, 'signed_flux_metrics': signed_flux,
        'physical_full_E_vector_metrics': physical_vector_diagnostics,
        'actual_basis_bilinear_flux_metrics': input_bilinear_flux,
        'original_pointwise_criteria_passed': all(record['original_pointwise_passed'] for record in field_diagnostics),
        'component_relative_criteria_passed': all(record['passed'] for record in field_diagnostics),
        'conditioned_field_criteria_passed': all(record['passed'] for record in physical_vector_diagnostics),
        'scope': 'Actual finite-time real-source GPU superposition; unchanged fullphysicalEvector rtol1e-4 and power rtol3e-5, explicit native dual-volume and actual-input-basis bilinear W scales; old signed-entering verdicts remain diagnostics, without relative weak-flux precision claim. Weak-component relative/absolute and original pointwise verdicts retained without independentcomponent precision claim.'}
    (tmp_path / 'superposition-diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + '\n')
    assert diagnostics['conditioned_field_criteria_passed']
    assert input_bilinear_flux['passed']
    resumed = torch.load(folders[0] / 'final-fields.pt', weights_only=False)
    uninterrupted = torch.load(folders[3] / 'final-fields.pt', weights_only=False)
    assert resumed.keys() == uninterrupted.keys()
    for name in resumed:
        torch.testing.assert_close(resumed[name], uninterrupted[name], rtol=0, atol=0)
    assert reports[0]['checkpoint']['step'] == 512 and reports[0]['checkpoint']['total_steps'] == 1024
    for name, record in results[0].conductor_losses.items():
        torch.testing.assert_close(record['surface_power'], results[3].conductor_losses[name]['surface_power'], rtol=0, atol=0)
    for frequency in case.frequencies:
        for component, selection in schema.items():
            for plane in selection['planes']:
                resumed_data = results[0].monitor(plane['name'], frequency=frequency)['components'][component]['data']
                original_data = results[3].monitor(plane['name'], frequency=frequency)['components'][component]['data']
                torch.testing.assert_close(resumed_data, original_data, rtol=0, atol=0)
