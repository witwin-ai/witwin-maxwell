"""Finite-thickness copper constitutive checks against independent layer propagation."""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.copper_foil import compile_copper_response


MU = 1.25663706212e-6
EPS = 8.8541878128e-12
SIGMA = 5.8e7
FREQUENCY = 1e9
DELTA = math.sqrt(2 / (2 * math.pi * FREQUENCY * MU * SIGMA))


def _film_transfer(thickness, *, incident_eps=1., outgoing_eps=1., angle=0., polarization="te"):
    """Full Maxwell conductive-layer ABCD reference, including displacement current."""
    s = -2j * math.pi * FREQUENCY
    tangential_k = 2 * math.pi * FREQUENCY * math.sqrt(MU * EPS * incident_eps) * math.sin(angle)
    propagation = np.sqrt(tangential_k**2 + s * MU * (SIGMA + s * EPS))
    conductor_z = s * MU / propagation if polarization == "te" else propagation / (SIGMA + s * EPS)
    incident_cos = math.cos(angle)
    outgoing_cos = math.sqrt(1 - incident_eps / outgoing_eps * math.sin(angle)**2)
    eta_i, eta_t = math.sqrt(MU / (EPS * incident_eps)), math.sqrt(MU / (EPS * outgoing_eps))
    zi = eta_i / incident_cos if polarization == "te" else eta_i * incident_cos
    zt = eta_t / outgoing_cos if polarization == "te" else eta_t * outgoing_cos
    a = np.cosh(propagation * thickness)
    b = conductor_z * np.sinh(propagation * thickness)
    c = np.sinh(propagation * thickness) / conductor_z
    denominator = a + b / zt + zi * c + zi * a / zt
    reflection = (a + b / zt - zi * c - zi * a / zt) / denominator
    transmission = 2 / denominator
    return reflection, transmission, zi, zt


def _response_from_z(z, zi, zt):
    system = np.array([[1 + z[0, 0] / zi, z[0, 1] / zt],
                       [z[1, 0] / zi, 1 + z[1, 1] / zt]])
    target = np.array([z[0, 0] / zi - 1, z[1, 0] / zi])
    return np.linalg.solve(system, target)


@pytest.mark.parametrize("ratio", [.01, .1, 1., 3., 10., 30.])
@pytest.mark.parametrize("incident_eps,outgoing_eps", [(1., 1.), (1., 4.), (4., 4.)])
@pytest.mark.parametrize("angle,polarization", [(0., "te"), (math.radians(35), "te"), (math.radians(35), "tm")])
def test_two_sided_diffusion_response_matches_independent_full_maxwell_film(ratio, incident_eps, outgoing_eps, angle, polarization):
    foil = mw.CopperFoil(thickness=ratio * DELTA, frequency_range=(.5e9, 2e9))
    reflection, transmission, zi, zt = _film_transfer(foil.thickness, incident_eps=incident_eps,
        outgoing_eps=outgoing_eps, angle=angle, polarization=polarization)
    z = foil.surface_impedance([FREQUENCY])[0].numpy()
    actual_r, actual_t = _response_from_z(z, zi, zt)
    np.testing.assert_allclose(actual_r, reflection, rtol=2e-6, atol=2e-10)
    np.testing.assert_allclose(actual_t, transmission, rtol=2e-6, atol=1e-15)
    expected_absorption = 1 - abs(reflection)**2 - zi / zt * abs(transmission)**2
    actual_absorption = 1 - abs(actual_r)**2 - zi / zt * abs(actual_t)**2
    assert abs(actual_absorption - expected_absorption) / expected_absorption < 2e-5
    currents = np.array([(1 - actual_r) / zi, -actual_t / zt])
    constitutive_absorption = zi * np.real(currents.conj() @ z @ currents)
    assert constitutive_absorption > 0
    assert abs(constitutive_absorption - expected_absorption) / expected_absorption < 2e-5
    # Reversing the normal exchanges both physical faces and their inward currents.
    np.testing.assert_allclose(z[::-1, ::-1], z, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [.01, .1, 1., 3., 10., 30.])
def test_analytic_rational_realization_is_globally_passive_and_band_accurate(ratio):
    foil = mw.CopperFoil(thickness=ratio * DELTA, frequency_range=(.5e9, 2e9))
    response = compile_copper_response(foil)
    f = torch.linspace(.5e9, 2e9, 401, dtype=torch.float64)
    actual = response.rational.evaluate(f)
    expected = foil.surface_admittance(f)
    relative = torch.linalg.matrix_norm(actual - expected) / torch.linalg.matrix_norm(expected)
    assert float(relative.max()) < 2e-4
    assert torch.count_nonzero(response.rates == 0) == 1
    assert torch.all(response.rates >= 0)
    torch.testing.assert_close(response.state_space.B, response.state_space.C.T, rtol=0, atol=0)
    assert float(torch.linalg.eigvalsh(response.state_space.D).min()) >= 0
    for index in range(response.rational.order):
        assert float(torch.linalg.eigvalsh(response.rational.residues[:, :, index].real).min()) >= -1e-5
    # An off-grid frequency set, not the order-selection samples, constrains the
    # balanced shared state-space realization independently of RationalModel.
    state_values = response.state_space.evaluate(f[::37])
    torch.testing.assert_close(state_values, actual[::37], rtol=1e-10, atol=1e-8)


def test_unresolved_modal_order_and_out_of_band_use_are_rejected():
    foil = mw.CopperFoil(thickness=30 * DELTA, frequency_range=(.5e9, 2e9))
    with pytest.raises(ValueError, match="exceeds.*modes"):
        compile_copper_response(foil, max_modes=2)
    with pytest.raises(ValueError, match="frequency_range"):
        foil.surface_impedance([3e9])
    with pytest.raises(ValueError, match="diffusion requires"):
        mw.CopperFoil(thickness=1e-6, conductivity=1., frequency_range=(.5e9, 2e9))


@pytest.mark.parametrize("thickness,conductivity", [(0., SIGMA), (-1., SIGMA), (float("nan"), SIGMA), (DELTA, 0.), (DELTA, float("inf"))])
def test_nonphysical_parameters_are_rejected(thickness, conductivity):
    with pytest.raises(ValueError, match="finite and positive"):
        mw.CopperFoil(thickness=thickness, conductivity=conductivity, frequency_range=(.5e9, 2e9))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CopperFoil forward fields require CUDA.")
@pytest.mark.parametrize("ratio", [.01, .1, 1., 3., 10., 30.])
def test_gpu_two_sided_scattering_matches_finite_layer_including_small_absorption(ratio):
    from tests.materials.sheet.sheet_measurement import measure_sheet_scattering
    foil = mw.CopperFoil(thickness=ratio * DELTA, frequency_range=(.5e9, 2e9))
    actual_t, actual_r = measure_sheet_scattering(foil, FREQUENCY, absorber="cpml", dl=.005,
        half_length=.30, half_width=.01, fit_gap=.025, fit_extent=.15,
        steady_cycles=20, transient_cycles=24)
    expected_r, expected_t, zi, zt = _film_transfer(foil.thickness)
    expected_absorption = 1 - abs(expected_r)**2 - zi / zt * abs(expected_t)**2
    actual_absorption = 1 - abs(actual_r)**2 - zi / zt * abs(actual_t)**2
    print(dict(ratio=ratio, reflection=actual_r, transmission=actual_t,
               expected_r=expected_r, expected_t=expected_t,
               absorption=actual_absorption, expected_absorption=expected_absorption))
    assert abs(actual_r - expected_r) < 2e-3
    if abs(expected_t) >= 1e-6:
        assert abs(actual_t - expected_t) / abs(expected_t) < .03
    else:
        # The plan specifies an absolute floor for extremely small transmission;
        # 1e-8 is much stricter than its 0.005 complex-amplitude acceptance budget.
        assert abs(actual_t - expected_t) < 1e-8
    assert abs(actual_absorption - expected_absorption) / expected_absorption < .03


def test_fixed_modal_topology_preserves_native_parameter_gradients():
    thickness = torch.tensor(DELTA, dtype=torch.float64, requires_grad=True)
    conductivity = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
    foil = mw.CopperFoil(thickness=thickness, conductivity=conductivity,
                         frequency_range=(.5e9, 2e9), mode_count=16)
    assert foil.thickness is thickness
    assert foil.conductivity is conductivity
    response = compile_copper_response(foil)
    value = response.state_space.evaluate([FREQUENCY])[0, 0, 0].real
    actual = torch.autograd.grad(value, (thickness, conductivity))
    for index, (parameter, step) in enumerate(((DELTA, DELTA * 1e-4), (SIGMA, SIGMA * 1e-4))):
        evaluated = []
        for sign in (-1, 1):
            candidate = mw.CopperFoil(thickness=DELTA + sign * step if index == 0 else DELTA,
                conductivity=SIGMA + sign * step if index == 1 else SIGMA,
                frequency_range=(.5e9, 2e9), mode_count=16)
            evaluated.append(compile_copper_response(candidate).state_space.evaluate([FREQUENCY])[0, 0, 0].real)
        expected = (evaluated[1] - evaluated[0]) / (2 * step)
        torch.testing.assert_close(actual[index], expected, rtol=1e-6, atol=1e-12)
    with pytest.raises(ValueError, match="explicit mode_count"):
        mw.CopperFoil(thickness=thickness, frequency_range=(.5e9, 2e9))


def test_thin_resistive_film_is_admitted_by_extended_band_floor():
    from copy import copy

    from witwin.maxwell.compiler.copper_foil import _truncated_copper_response

    eta0 = math.sqrt(MU / EPS)
    foil = mw.CopperFoil(thickness=(0.5 / eta0) / 1e5, conductivity=1e5,
                         frequency_range=(.3e9, 4e9))
    response = compile_copper_response(foil)
    vectors = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64) / math.sqrt(2)
    state = response.state_space
    weights = (state.B @ vectors.T).square()
    direct = (vectors @ state.D @ vectors.T).diagonal()

    def modal(samples):
        # Evaluate each parity separately to avoid cancellation in this thin two-port.
        s = -2j * math.pi * samples
        return direct + (weights / (s[:, None, None] + response.rates[None, :, None])).sum(1)

    frequencies = torch.from_numpy(np.geomspace(*foil.frequency_range, 8193))
    exact = foil.modal_admittance(frequencies)
    in_band_error = float(((modal(frequencies) - exact).abs() / exact.abs()).max())
    assert in_band_error <= response.relative_error_bound <= 5e-5

    extended_band = (foil.frequency_range[0] / 10, foil.frequency_range[1] * 10)
    extended_f = torch.from_numpy(np.geomspace(*extended_band, 4097))
    reference = copy(foil)
    object.__setattr__(reference, "frequency_range", extended_band)
    extended_exact = reference.modal_admittance(extended_f)
    previous = _truncated_copper_response(foil, 2e-4, 256, torch.device("cpu"))
    previous_y = previous.rational.evaluate(extended_f)
    previous_modal = torch.stack((previous_y[:, 0, 0] + previous_y[:, 0, 1],
                                  previous_y[:, 0, 0] - previous_y[:, 0, 1]), -1)
    old_extended_limit = 2 * float(((previous_modal - extended_exact).abs() / extended_exact.abs()).max())
    extended_limit = max(old_extended_limit, 1e-6)
    residual = (modal(extended_f) - extended_exact).abs()
    moments = torch.tensor([foil.conductivity * foil.thickness / 2,
                            foil.conductivity * foil.thickness / 6], dtype=torch.float64)
    curvature = torch.from_numpy(np.diff(np.log(extended_f.numpy()))).square()[:, None] * moments / 8
    extended_bound = float(((torch.maximum(residual[:-1], residual[1:]) + curvature)
                            / extended_exact[1:].abs()).max())
    assert extended_limit == 1e-6
    assert old_extended_limit < extended_bound <= extended_limit
    print(dict(in_band_error=in_band_error, extended_bound=extended_bound,
               extended_limit=extended_limit, old_extended_limit=old_extended_limit))


@pytest.mark.parametrize("thickness,old_counts", [(35e-6, (256, 128, 128, 128, 128, 128)),
    (18e-6, (128, 64, 64, 64, 64, 64)), (2e-6, (8, 8, 8, 8, 8, 8))])
@pytest.mark.parametrize("band_index,band", list(enumerate(((.1, 10), (1, 6), (.1, 4),
                                                          (2, 4), (1, 7), (2.4, 2.5)))))
def test_passive_compression_case_table_against_exact_diffusion(thickness, old_counts, band_index, band):
    band = tuple(value * 1e9 for value in band)
    foil = mw.CopperFoil(thickness=thickness, frequency_range=band, edge_model="none")
    new = compile_copper_response(foil)
    old = compile_copper_response(mw.CopperFoil(thickness=thickness, frequency_range=band,
        edge_model="none", mode_count=old_counts[band_index]))
    # Twice the certificate grid density, independently offset between its nodes.
    frequencies = torch.from_numpy(np.geomspace(*band, 8193))
    exact = foil.modal_admittance(frequencies)

    def modal(response, samples=frequencies):
        y = response.rational.evaluate(samples)
        return torch.stack((y[:, 0, 0] + y[:, 0, 1], y[:, 0, 0] - y[:, 0, 1]), -1)

    actual, previous = modal(new), modal(old)
    error = float(((actual - exact).abs() / exact.abs()).max())
    old_error = float(((previous - exact).abs() / exact.abs()).max())
    difference = float(((actual - previous).abs() / previous.abs()).max())
    assert error <= new.relative_error_bound <= 5e-5
    assert error <= 2e-4
    assert old_error <= 2e-4
    assert len(new.rates) < len(old.rates)
    assert torch.all(new.rates[:-1] > 0) and new.rates[-1] == 0
    state = new.state_space
    torch.testing.assert_close(state.B, state.C.T, rtol=0, atol=0)
    torch.testing.assert_close(state.D, state.D.T, rtol=0, atol=0)
    assert torch.linalg.eigvalsh(state.D).min() >= 0
    assert torch.all(new.rational.residues.diagonal(dim1=0, dim2=1).real >= 0)
    vectors = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64) / math.sqrt(2)
    weights = (state.B @ vectors.T).square()
    moments = (vectors @ state.D @ vectors.T).diagonal() + (weights[:-1] / new.rates[:-1, None]).sum(0)
    torch.testing.assert_close(moments, torch.tensor([SIGMA * thickness / 2, SIGMA * thickness / 6],
                                                    dtype=torch.float64), rtol=2e-14, atol=0)
    torch.testing.assert_close(weights[-1, 1], torch.tensor(2 / (MU * thickness), dtype=torch.float64),
                               rtol=2e-14, atol=0)
    extended_band = (band[0] / 10, band[1] * 10)
    extended_f = torch.from_numpy(np.geomspace(*extended_band, 8193))
    extended_exact = mw.CopperFoil(thickness=thickness, frequency_range=extended_band,
                                  edge_model="none").modal_admittance(extended_f)
    old_extended_error = float(((modal(old, extended_f) - extended_exact).abs() / extended_exact.abs()).max())
    new_extended_error = float(((modal(new, extended_f) - extended_exact).abs() / extended_exact.abs()).max())
    assert new_extended_error <= 2 * old_extended_error
    even_count = int((state.B[:-1, 0] * state.B[:-1, 1] > 0).sum())
    odd_count = len(new.rates) - even_count - 1
    assert max(even_count, odd_count) <= 128
    print(dict(thickness_um=thickness * 1e6, band_GHz=tuple(value / 1e9 for value in band),
        old_states=len(old.rates), new_states=len(new.rates), old_error=old_error,
        new_error=error, new_vs_old=difference, certified_bound=new.relative_error_bound,
        even_states=even_count, odd_states=odd_count, old_extended_error=old_extended_error,
        new_extended_error=new_extended_error,
        min_positive_rate=float(new.rates[:-1].min()), min_D_eigenvalue=float(torch.linalg.eigvalsh(state.D).min())))
