"""Independent coherent cross terms, with no field propagation or CUDA context."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.printed_line_loss_basis import (
    SpectralContract, capture_loss_basis, coherent_conductor_records, coherent_fields,
)
from witwin.maxwell.postprocess.fdtd_loss import conductor_region_power


FREQUENCIES = (8e8, 1e9, 1.2e9)
DT = 1e-12


def basis(amplitude=1., complex_bank=False):
    indices = torch.tensor([[1, 2, 3], [2, 2, 3]])
    area = torch.tensor([2., 3.], dtype=torch.float64)
    field = torch.full((3, 2, 2), complex(amplitude), dtype=torch.complex128)
    face = SimpleNamespace(name='signal', axis=2, normal_index=3)
    entry = dict(face=face, electric_name='Ex', imaginary=False,
        indices=indices, area=area, B=torch.empty((0, 2), dtype=torch.float64),
        rates=torch.empty(0, dtype=torch.float64), D=torch.diag(torch.tensor([2., 4.], dtype=torch.float64)),
        spectral_frequencies=torch.tensor(FREQUENCIES, dtype=torch.float64), spectral_norm=torch.ones(3),
        spectral_end=64000, spectral_step=torch.tensor(64000.), spectral_window='none')
    bounds = torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    surface = dict(component='Ex', indices=indices.clone(), area=area.clone(),
        power=torch.zeros((3, 2)), lower_bounds=bounds,
        upper_bounds=bounds + torch.tensor([1., 1., 0.]), normal_axis='z', normal_coordinate=0.)
    bulk = dict(component='Ex', indices=indices.clone(), electric_fields=field,
        normal_axis='z', normal_coordinate=0., normal_half_widths=torch.ones(2, 2),
        conduction_coefficient=torch.ones(3, 2, 2), dispersion_coefficient=torch.zeros(3, 2, 2))
    record = dict(frequencies=torch.tensor(FREQUENCIES, dtype=torch.float64),
        spectral_valid=torch.ones(3, dtype=torch.bool), coordinate_frame='cartesian', periodic_axes=(),
        surface_power_traces=[surface], bulk_electric_traces=[bulk],
        surface_power=torch.zeros(3), dissipated_energy=7., stored_energy=9.)
    entries = [entry]
    if complex_bank:
        entries.append(dict(entry, imaginary=True, electric_name='Ex_imag'))
    result = SimpleNamespace(solver=SimpleNamespace(dt=DT, _copper_foils=entries),
        frequencies=FREQUENCIES, conductor_losses={'signal': record}, stats=lambda: {'normalize_source': False})
    return capture_loss_basis(result, SpectralContract('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64,
        planned_steps=64000, window='none')), result


@pytest.mark.parametrize('phase,expected_factor', [(0., 4.), (torch.pi, 0.), (torch.pi / 2, 2.)])
def test_coherent_film_power_has_independent_constructive_destructive_cross_terms(phase, expected_factor):
    left, _ = basis()
    right, _ = basis()
    weights = torch.tensor([[1., complex(torch.exp(torch.tensor(1j * phase, dtype=torch.complex128)))]] * 3, dtype=torch.complex128)
    records = coherent_conductor_records([left, right], weights)
    # Independent memoryless two-face resistor oracle: .5*(2+4)*sum(area).
    # Native midpoint endpoint samples introduce the documented cos² factor.
    expected = 15 * expected_factor * torch.cos(torch.pi * torch.tensor(FREQUENCIES, dtype=torch.float64) * DT).square()
    torch.testing.assert_close(records['signal']['surface_power'], expected, rtol=2e-14, atol=1e-25)
    expected_edges = expected[:, None] * torch.tensor([.4, .6], dtype=torch.float64)
    torch.testing.assert_close(records['signal']['surface_power_traces'][0]['power'], expected_edges, rtol=2e-14, atol=1e-25)
    if expected_factor != 2:
        incoherent = 30 * torch.cos(torch.pi * torch.tensor(FREQUENCIES, dtype=torch.float64) * DT).square()
        assert not torch.allclose(records['signal']['surface_power'], incoherent)
    assert 'dissipated_energy' not in records['signal'] and 'stored_energy' not in records['signal']
    # Existing ROI operation consumes the recomputed per-edge work directly.
    actual_roi = conductor_region_power(records['signal'], ((0., 1.), (0., 1.), (-1., 1.)))
    torch.testing.assert_close(actual_roi, expected * .4, rtol=2e-14, atol=1e-25)


def test_actual_complex_phasors_and_frequency_dependent_weights_are_combined_before_work():
    left, _ = basis(1 + 2j, complex_bank=True)
    right, _ = basis(2 - 1j, complex_bank=True)
    weights = torch.tensor([[1., 1j], [1., -1j], [.5j, -.25]], dtype=torch.complex128)
    record = coherent_conductor_records([left, right], weights)['signal']
    amplitude = weights[:, 0] * (1 + 2j) + weights[:, 1] * (2 - 1j)
    expected = 15 * amplitude.abs().square() * torch.cos(torch.pi * torch.tensor(FREQUENCIES, dtype=torch.float64) * DT).square()
    torch.testing.assert_close(record['surface_power'], expected, rtol=2e-14, atol=0)
    assert left.realizations[0].banks == ('real', 'imaginary')


@pytest.mark.parametrize('phase,amplitude_squared', [(1., 4.), (-1., 0.), (1j, 2.)])
def test_bulk_sigma_shared_api_retains_same_cross_terms(phase, amplitude_squared):
    scene = mw.Scene(domain=mw.Domain(bounds=((- .01, .01),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(),
        structures=[mw.Structure(name='bulk', geometry=mw.Box(position=(0., 0., 0.), size=(1., 1., 1.)),
            material=mw.Material(sigma_e_tensor=mw.DiagonalTensor3(2., 3., 4.)))],
        monitors=[mw.PowerLossMonitor('loss', position=(0., 0., 0.), size=(.01, .01, .01),
            quadrature='physical_intersection')], device='cpu')
    from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
    from witwin.maxwell.scene import prepare_scene
    prepared = prepare_scene(scene)
    compiled = compile_power_loss_monitor(prepared, scene.monitors[0])
    fields = {component: torch.ones((3, *shape), dtype=torch.complex128)
        for component, shape in compiled.full_component_shapes.items()}
    combined = coherent_fields([fields, fields], torch.tensor([[1., phase]] * 3))
    diagnostic = mw.Result(method='fdtd', scene=scene, frequencies=FREQUENCIES)
    actual = diagnostic.power_loss('loss', electric_fields=combined).conduction
    # Three independent constant native electric components in a 1cm³ ROI.
    expected = torch.full((3,), .5 * 9 * 1e-6 * amplitude_squared, dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-25)


@pytest.mark.parametrize('change', ['dt', 'frequency', 'contract', 'banks', 'area', 'indices', 'D', 'material', 'tile', 'missing'])
def test_mismatched_actual_basis_cannot_be_coherently_reduced(change):
    left, _ = basis()
    right = copy.deepcopy(left)
    if change == 'dt': right = replace(right, dt=DT * 2)
    elif change == 'frequency': right = replace(right, frequencies=FREQUENCIES[::-1])
    elif change == 'contract': right = replace(right, execution_contract=replace(right.execution_contract, window='hanning'))
    elif change == 'banks': right = replace(right, realizations=(replace(right.realizations[0], banks=('real', 'imaginary')),))
    elif change in ('area', 'indices', 'D'): getattr(right.realizations[0], change).flatten()[0] += 1
    elif change == 'material': right.conductor_records['signal']['bulk_electric_traces'][0]['conduction_coefficient'][0, 0, 0] += 1
    elif change == 'tile': right.conductor_records['signal']['surface_power_traces'][0]['upper_bounds'][0, 0] += .001
    elif change == 'missing': right.conductor_records.clear()
    with pytest.raises(ValueError): coherent_conductor_records([left, right], torch.ones(3, 2))


@pytest.mark.parametrize('change', ['normalization', 'bank', 'frequency', 'orphan', 'partial', 'horizon'])
def test_runtime_sidecar_rejects_inconsistent_spectral_bank(change):
    _, result = basis(complex_bank=True)
    real, imaginary = result.solver._copper_foils
    if change == 'normalization': result.stats = lambda: {'normalize_source': True}
    elif change == 'bank': imaginary['D'] = imaginary['D'] + 1
    elif change == 'frequency': real['spectral_frequencies'] = real['spectral_frequencies'].flip(0)
    elif change == 'orphan': result.solver._copper_foils = [imaginary]
    elif change == 'partial': real['spectral_step'] = torch.tensor(32000.)
    elif change == 'horizon': real['spectral_end'] = 32000
    with pytest.raises(ValueError): capture_loss_basis(result, SpectralContract('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64, 64000, 'none'))


def test_sidecar_is_detached_cloned_and_serializable(tmp_path):
    actual, result = basis()
    result.solver._copper_foils[0]['D'].zero_()
    assert actual.realizations[0].D[0, 0] == 2
    path = tmp_path / 'basis.pt'
    torch.save(actual, path)
    restored = torch.load(path, weights_only=False)
    records = coherent_conductor_records([restored], torch.ones(3, 1))
    assert bool((records['signal']['surface_power'] > 0).all())


@pytest.mark.parametrize('weights', [torch.ones(2, 2), torch.ones(3, 1), torch.full((3, 2), float('nan'))])
def test_coefficients_require_complete_finite_frequency_drive_axes(weights):
    first, _ = basis()
    with pytest.raises(ValueError): coherent_conductor_records([first, first], weights)


@pytest.mark.parametrize('change', ['digest', 'steps', 'window', 'normalized', 'phasor'])
def test_typed_execution_contract_rejects_ambiguous_sampling_or_source_identity(change):
    contract = dict(source_sha256='a' * 64, grid_sha256='b' * 64, material_sha256='c' * 64,
        observation_sha256='d' * 64, planned_steps=64000, window='none')
    if change == 'digest': contract['grid_sha256'] = 'unknown'
    elif change == 'steps': contract['planned_steps'] = 0
    elif change == 'window': contract['window'] = 'unknown'
    elif change == 'normalized': contract['normalize_source'] = True
    elif change == 'phasor': contract['phasor_convention'] = 'exp(+iwt)'
    with pytest.raises(ValueError): SpectralContract(**contract)


@pytest.mark.parametrize('fields', [[], [{}], [{'E': torch.tensor(1j)}],
    [{'E': torch.ones(3, dtype=torch.complex128), 'H': torch.ones(2, dtype=torch.complex128)}]])
def test_coherent_field_contract_rejects_missing_or_inconsistent_frequency_axis(fields):
    with pytest.raises(ValueError): coherent_fields(fields, torch.ones(3, 1))


@pytest.fixture(autouse=True)
def _cpu_preparation_keeps_CUDA_initialization_state():
    before = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == before
