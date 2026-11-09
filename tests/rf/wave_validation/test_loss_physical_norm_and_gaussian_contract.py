"""CPU analytic quadrature and actual real-port Gaussian source semantics."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.printed_line_loss_assertions import physical_electric_vectors, superposition_metrics
from scripts.dev.fdtd.printed_line_regional_power import regional_monitors
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.compiler.power_loss import _component_cell_bounds
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.fdtd.ports import _drive_value
from witwin.maxwell.scene import prepare_scene


def prescribed_field(copper=False):
    nodes = np.linspace(-.01, .01, 5)
    grid = mw.GridSpec.custom(nodes, nodes, [-.01, -.003, 0., .004, .01])
    structures = [mw.Structure(mw.Box(position=(0., 0., 0.), size=(1., 1., 1.)), mw.Material(sigma_e=1.), name='bulk')]
    if copper:
        structures.append(mw.Structure(mw.Box(position=torch.zeros(3, dtype=torch.float64),
            size=torch.tensor([.01, .01, 0.], dtype=torch.float64)),
            mw.CopperFoil(thickness=35e-6, frequency_range=(.1e9, 4e9)), name='foil'))
    scene = mw.Scene(domain=mw.Domain(bounds=((- .01, .01),) * 3), grid=grid,
        structures=structures, boundary=mw.BoundarySpec.none(), device='cpu')
    monitors, schema = regional_monitors(scene, scene.domain.bounds, (1e9,), observation='planes', include_flux=False)
    scene = scene.clone(monitors=monitors); prepared = prepare_scene(scene)
    fields = {name: torch.ones(selection['full_shape'], dtype=torch.complex128) for name, selection in schema.items()}
    records = {}; extra_norm = 0.
    if copper:
        for face in compile_copper_faces(prepared, device='cpu'):
            comp = 'E' + 'xyz'[face.component]; indices = face.indices
            fields[comp][tuple(indices.T)] = 2.
            bounds = _component_cell_bounds(prepared, comp)
            area = torch.ones(len(indices), dtype=torch.float64)
            for axis in (0, 1): area *= bounds[axis][1][indices[:, axis]] - bounds[axis][0][indices[:, axis]]
            widths = torch.tensor([.0015, .002], dtype=torch.float64)[:, None]
            values = torch.stack((torch.full((len(indices),), 2.), torch.full((len(indices),), 7.))).to(torch.complex128)[None]
            record = records.setdefault(face.name, dict(frequencies=torch.tensor([1e9], dtype=torch.float64),
                coordinate_frame='cartesian', periodic_axes=(), surface_power_traces=[], bulk_electric_traces=[]))
            record['surface_power_traces'].append(dict(component=comp, area=area))
            record['bulk_electric_traces'].append(dict(component=comp, indices=indices,
                normal_axis='z', normal_coordinate=0., normal_half_widths=widths, electric_fields=values))
            extra_norm += float((area * (.0015 * (2 ** 2 - 1) + .002 * (7 ** 2 - 1))).sum())
    payloads = {}
    for comp, selection in schema.items():
        axes = component_coords(prepared, comp)
        for plane in selection['planes']:
            payloads[plane['name']] = dict(kind='plane', monitor_type='plane', axis='z',
                frequency=1e9, frequencies=(1e9,), components={comp: dict(data=fields[comp][:, :, plane['normal_index']],
                    coords=tuple(np.asarray(axis) for axis in axes[:2]), plane_index=plane['normal_index'],
                    plane_indices=(plane['normal_index'],), plane_weights=(1.,),
                    plane_positions=(plane['normal_position_m'],), collocated=False)})
    result = mw.Result(method='fdtd', scene=scene, frequencies=(1e9,), monitors=payloads,
        metadata={'conductor_losses': records}, solver_stats={'normalize_source': False})
    return result, scene, schema, 3 * .02 ** 3 + extra_norm


@pytest.mark.parametrize('copper', [False, True])
def test_full_E_vector_uses_actual_graded_dual_volumes_and_unequal_foil_half_volumes(copper):
    result, scene, schema, expected = prescribed_field(copper)
    vectors, weights, geometry = physical_electric_vectors([result], scene, schema, 1e9)
    assert float((abs(vectors[0]) ** 2 * weights).sum()) == pytest.approx(expected, rel=2e-14)
    assert float(weights.sum()) == pytest.approx(3 * .02 ** 3, rel=2e-14)
    assert geometry['squared_L2_units'] == 'V^2 m'


@pytest.mark.parametrize('fault', ['amplitude', 'phase'])
def test_physical_full_vector_metric_rejects_point1percent_source_fault(fault):
    result, scene, schema, _ = prescribed_field(True)
    vectors, weights, _ = physical_electric_vectors([result], scene, schema, 1e9)
    field = vectors[0]
    expected = field - field
    changed = field - (1.001 if fault == 'amplitude' else np.exp(.001j)) * field
    assert not superposition_metrics(expected, changed, [field, field], [1., -1.], physical_weights=weights)['passed']


def test_production_port_complex_Gaussian_is_real_carrier_phase_and_has_frequency_dependent_quadrature():
    pulse = mw.GaussianPulse(1e9, fwidth=.8e9)
    descriptor = dict(source_kind='gaussian_pulse', source_delay=pulse.delay,
        source_frequency=pulse.frequency, source_fwidth=pulse.fwidth, source_phase=pulse.phase,
        drive_buffer=torch.zeros((), dtype=torch.float64))
    times = torch.arange(64000, dtype=torch.float64) * 1e-12
    cosine = _drive_value(SimpleNamespace(**descriptor, source_amplitude=torch.tensor(1 + 0j, dtype=torch.complex128)), times)
    sine = _drive_value(SimpleNamespace(**descriptor, source_amplitude=torch.tensor(1j, dtype=torch.complex128)), times)
    assert not sine.is_complex()
    assert not torch.allclose(sine.to(torch.complex128), 1j * cosine)
    frequencies = np.linspace(.8e9, 1.2e9, 5)
    phase = np.exp(2j * np.pi * frequencies[:, None] * times.numpy()[None])
    ratio = np.trapezoid(phase * sine.numpy(), times.numpy(), axis=-1) / np.trapezoid(phase * cosine.numpy(), times.numpy(), axis=-1)
    low = np.exp(-.5 * ((frequencies - 1e9) / .8e9) ** 2)
    high = np.exp(-.5 * ((frequencies + 1e9) / .8e9) ** 2)
    independent = 1j * (low - high) / (low + high)
    np.testing.assert_allclose(ratio, independent, rtol=2e-8, atol=2e-9)
    assert min(abs(ratio - 1j)) > .04
    assert max(abs(ratio - 1j)) > .15


@pytest.mark.parametrize('coefficient', [-1., .5])
def test_real_source_coefficients_superpose_actual_production_Gaussian_waveforms(coefficient):
    pulse = mw.GaussianPulse(1e9, fwidth=.8e9)
    descriptor = dict(source_kind='gaussian_pulse', source_delay=pulse.delay,
        source_frequency=pulse.frequency, source_fwidth=pulse.fwidth, source_phase=pulse.phase,
        drive_buffer=torch.zeros((), dtype=torch.float64))
    times = torch.arange(4096, dtype=torch.float64) * 1e-12
    def actual(amplitude):
        return _drive_value(SimpleNamespace(**descriptor, source_amplitude=torch.tensor(amplitude, dtype=torch.complex128)), times)
    for left, right in ((1., .25), (.25, 1.)):
        expected = actual(left) + coefficient * actual(right)
        direct = actual(left + coefficient * right)
        assert not direct.is_complex()
        torch.testing.assert_close(expected, direct, rtol=2e-15, atol=2e-16)


@pytest.fixture(autouse=True)
def _cpu_preparation_keeps_CUDA_initialization_state():
    before = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == before
