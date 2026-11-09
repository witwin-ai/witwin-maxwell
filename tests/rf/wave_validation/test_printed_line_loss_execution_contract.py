"""CPU scene/source contracts for the bounded actual CUDA runner."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.printed_line_loss_execution import (
    CoherentObservations, DRIVE_BASIS, PilotCase, build_case, execution_identity, make_simulation, run_drive, tensor_signature,
)
from scripts.dev.fdtd.printed_line_loss_basis import LossBasis, SpectralContract


@pytest.fixture(scope='module')
def contour_helpers():
    # The printed-line builders load their contour helpers from this evidence directory.
    if not (Path(__file__).resolve().parents[3]/'docs/assessments/printed-lines-2026-10-22').exists():
        pytest.skip('repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-22')


@pytest.mark.usefixtures('contour_helpers')
def test_actual_pilot_cpu_scene_and_independent_nonzero_drives_keep_all_norton_resistors():
    case = PilotCase()
    scene, groups, bounds, schema, rows, expected, tails = build_case(case)
    simulations = [make_simulation(scene, groups, amplitudes, case, graph=True, steps=20000, device='cpu')
        for amplitudes in DRIVE_BASIS]
    assert len(groups) == 2 and all(len(group) == 2 for group in groups)
    assert all(port.termination is None for port in scene.ports)
    for column, simulation in enumerate(simulations):
        assert len(simulation.excitations) == 4
        for excitation in simulation.excitations:
            assert excitation.source_impedance == 50.
            group_index = next(index for index, group in enumerate(groups) if excitation.port_name in group)
            assert float(excitation.amplitude) == DRIVE_BASIS[column][group_index]
            assert excitation.source_time.frequency == 1e9 and excitation.source_time.fwidth == .8e9
    assert bounds[0] == (-.015, .015)
    assert set(schema) == {'Ex', 'Ey', 'Ez'} and len(rows) == len(case.positions)
    assert expected and len(tails) == len(case.positions)
    assert case.duration == 64e-9 and case.checkpoint_time == 32e-9


@pytest.mark.usefixtures('contour_helpers')
def test_zero_active_norton_spectrum_is_rejected_before_cuda_prepare():
    case = PilotCase()
    scene, groups, *_ = build_case(case)
    with pytest.raises(ValueError, match='nonzero spectra'):
        make_simulation(scene, groups, (1., 0.), case, graph=False, steps=1024, device='cpu')


@pytest.mark.parametrize('change', [{'frequencies': (1e9, .9e9)}, {'positions': (-.06, 0., .015)}])
def test_real_case_rejects_missing_frequency_order_or_source_clearance(change):
    with pytest.raises(ValueError): build_case(replace(PilotCase(), **change))


def test_preexisting_output_is_rejected_before_any_gpu_operation(tmp_path):
    with pytest.raises(FileExistsError): run_drive(PilotCase(), (1., 0.), tmp_path)


def test_signature_retains_native_float64_intervals_and_tensor_dtype():
    base = torch.tensor([1e6, 1e6 + 1e-6], dtype=torch.float64)
    shifted = base.clone(); shifted[1] += 1e-8
    assert tensor_signature(base) != tensor_signature(shifted)
    assert tensor_signature(base) != tensor_signature(base.float())


@pytest.mark.usefixtures('contour_helpers')
def test_identity_retains_actual_core_and_execution_dependencies_without_cuda():
    identity = execution_identity()
    assert len(identity['maxwell']['source_sha256']) == 64
    assert len(identity['actual_core_sha256']) == 64 and identity['actual_core_sources']
    assert set(identity['helpers']) == {'fixture', 'sources', 'regional', 'basis', 'execution', 'contours', 'fixed_VI'}


def prescribed_observation():
    frequencies = (1e9,)
    contract = SpectralContract('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64, 1024, 'none')
    basis = LossBasis(frequencies, 1e-12, contract, (), {})
    coords = (torch.tensor([-.005, .005], dtype=torch.float64),) * 2
    electric = torch.ones((2, 2), dtype=torch.complex128)
    magnetic = 3 * electric
    face = {'axis': 'z', 'normal_direction': '+', 'position': .01, 'frequency': 1e9,
        'frequencies': frequencies, 'x': coords[0], 'y': coords[1], 'collocated': True,
        'cell_widths': {'x': torch.tensor([.01, .01], dtype=torch.float64), 'y': torch.tensor([.01, .01], dtype=torch.float64)},
        'Ex': electric, 'Hy': magnetic, 'flux': torch.tensor(999.), 'power': torch.tensor(999.)}
    entry = dict(data=electric, coords=coords, plane_index=2, plane_indices=(2,), plane_weights=(1.,),
        plane_positions=(.01,), collocated=False)
    monitor = dict(kind='plane', frequency=1e9, frequencies=frequencies, axis='z', components={'Ex': entry}, Ex=electric)
    scene = mw.Scene(domain=mw.Domain(bounds=((- .01, .01),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(), device='cpu')
    result = mw.Result(method='fdtd', scene=scene, frequencies=frequencies, solver_stats={'normalize_source': False},
        monitors={'plane': monitor, 'surface': {'kind': 'closed_surface', 'frequency': 1e9,
            'frequencies': frequencies, 'faces': {'z+': face}}})
    return result, basis


@pytest.mark.parametrize('phase,expected_factor', [(1., 4.), (-1., 0.), (1j, 2.)])
def test_coherent_actual_result_view_recomputes_flux_and_native_data_without_stale_single_drive_power(phase, expected_factor):
    result, basis = prescribed_observation()
    observation = CoherentObservations([result, result], [basis, basis], [[1., phase]])
    plane = observation.monitor('plane', frequency=1e9)
    torch.testing.assert_close(plane['components']['Ex']['data'], torch.full((2, 2), 1 + phase, dtype=torch.complex128))
    assert 'Ex' not in plane
    surface = observation.monitor('surface', frequency=1e9)
    assert float(surface['flux']) == pytest.approx(.5 * 3 * .02 ** 2 * expected_factor, rel=2e-14, abs=1e-25)
    assert float(surface['faces']['z+']['flux']) != 999
    assert float(result.monitor('surface')['faces']['z+']['flux']) == 999


def test_coherent_result_view_rejects_shifted_native_geometry():
    result, basis = prescribed_observation()
    other, _ = prescribed_observation()
    other._monitors['plane']['components']['Ex']['plane_positions'] = (.01000000001,)
    observation = CoherentObservations([result, other], [basis, basis], [[1., 1.]])
    with pytest.raises(ValueError, match='native-plane geometry'):
        observation.monitor('plane', frequency=1e9)


@pytest.fixture(autouse=True)
def _cpu_preparation_keeps_CUDA_initialization_state():
    before = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == before
