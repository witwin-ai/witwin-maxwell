import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0
from witwin.maxwell.network import PortData

pp = mw.postprocess

FREQUENCY = 1.0e9
WAVELENGTH = C_0 / FREQUENCY


def _sphere(theta_count=181, phi_count=361):
    theta = torch.linspace(0.0, math.pi, theta_count, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, phi_count, dtype=torch.float64)
    return theta, phi


def test_uniform_linear_array_matches_closed_form():
    count, spacing = 7, 0.6 * WAVELENGTH
    positions = pp.rectangular_lattice(count, 1, spacing)
    theta = torch.linspace(0.0, math.pi, 301, dtype=torch.float64)[:, None]
    phi = torch.linspace(0.0, 2.0 * math.pi, 7, dtype=torch.float64)[None, :]
    af = pp.array_factor(positions, None, theta, phi, frequency=FREQUENCY)
    psi = 2.0 * math.pi / WAVELENGTH * spacing * torch.sin(theta) * torch.cos(phi)
    # Centred positions make the geometric sum real.
    safe = torch.where(torch.abs(torch.sin(0.5 * psi)) > 1e-12, torch.sin(0.5 * psi), torch.ones_like(psi))
    expected = torch.where(
        torch.abs(torch.sin(0.5 * psi)) > 1e-12, torch.sin(0.5 * count * psi) / safe, torch.full_like(psi, count)
    )
    assert af.shape == (301, 7)
    torch.testing.assert_close(af, expected.to(torch.complex128), rtol=0.0, atol=1e-10)


def test_arbitrary_positions_and_phase_convention():
    positions = torch.tensor([[0.1, -0.2, 0.05], [0.0, 0.3, -0.1], [-0.25, 0.1, 0.2]], dtype=torch.float64)
    weights = torch.tensor([1.0 + 0.5j, -0.3 + 0.2j, 0.7 - 0.1j], dtype=torch.complex128)
    theta = torch.tensor([0.3, 1.1, 2.4], dtype=torch.float64)
    phi = torch.tensor([0.2, 3.5, 5.9], dtype=torch.float64)
    af = pp.array_factor(positions, weights, theta, phi, frequency=FREQUENCY)
    k = 2.0 * math.pi / WAVELENGTH
    direction = torch.stack(
        (torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)), dim=-1
    )
    expected = torch.stack(
        [torch.sum(weights * torch.exp(-1j * k * (positions @ d))) for d in direction]
    )
    torch.testing.assert_close(af, expected)


def test_steering_weights_match_codebook_and_co_phase_target():
    positions = pp.rectangular_lattice(4, 3, 0.5 * WAVELENGTH, 0.45 * WAVELENGTH)
    theta0, phi0 = 0.4, 1.2
    weights = pp.steering_weights(positions, theta0, phi0, frequency=FREQUENCY)
    codebook = mw.BeamCodebook.from_scan_angles(
        element_positions=positions,
        frequencies=torch.tensor([FREQUENCY], dtype=torch.float64),
        theta=torch.tensor([theta0], dtype=torch.float64),
        phi=torch.tensor([phi0], dtype=torch.float64),
    )
    torch.testing.assert_close(weights, codebook.weights[0, 0])
    af = pp.array_factor(
        positions, None, torch.tensor(theta0), torch.tensor(phi0), frequency=FREQUENCY, steering=(theta0, phi0)
    )
    assert float(af.real) == pytest.approx(12.0) and abs(float(af.imag)) < 1e-12


@pytest.mark.parametrize("count", [2, 4, 8])
def test_half_wave_isotropic_broadside_directivity_equals_element_count(count):
    theta, phi = _sphere()
    data = pp.compute_array_pattern(
        pp.rectangular_lattice(count, 1, 0.5 * WAVELENGTH), theta=theta, phi=phi, frequencies=FREQUENCY
    )
    assert data.element_kind == "isotropic"
    assert data.directivity.shape == (1, 181, 361)
    assert float(data.directivity_max[0]) == pytest.approx(count, rel=1e-3)
    # Isotropic elements: P_rad = 4*pi * sum |w|^2 when cross terms vanish at lambda/2.
    assert float(data.p_rad[0]) == pytest.approx(4.0 * math.pi * count, rel=1e-3)
    assert float(data.array_factor_db.max()) == pytest.approx(0.0)


def test_multi_frequency_weights_keep_explicit_frequency_axis():
    theta, phi = _sphere(37, 73)
    frequencies = torch.tensor([0.8e9, 1.0e9], dtype=torch.float64)
    positions = pp.rectangular_lattice(3, 1, 0.5 * WAVELENGTH)
    weights = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]], dtype=torch.complex128)
    data = pp.compute_array_pattern(positions, weights, theta=theta, phi=phi, frequencies=frequencies)
    assert data.array_factor.shape == (2, 37, 73) and data.p_rad.shape == (2,)
    for index in range(2):
        single = pp.compute_array_pattern(
            positions, weights[index], theta=theta, phi=phi, frequencies=frequencies[index]
        )
        torch.testing.assert_close(data.directivity[index], single.directivity[0])


def test_directivity_gradients_through_weights_and_steering():
    theta, phi = _sphere(19, 37)
    positions = pp.rectangular_lattice(3, 1, 0.5 * WAVELENGTH)

    def peak_directivity(weights, steer_theta):
        data = pp.compute_array_pattern(
            positions, weights, theta=theta, phi=phi, frequencies=FREQUENCY, steering=(steer_theta, 0.0)
        )
        return data.directivity[0, 6, 0]

    weights = torch.tensor([0.8 + 0.1j, 1.0 + 0.0j, 0.7 - 0.2j], dtype=torch.complex128, requires_grad=True)
    steer_theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(peak_directivity, (weights, steer_theta))


def _port(frequencies, *, incident, accepted):
    a = torch.sqrt(torch.as_tensor(incident, dtype=torch.float64)).to(torch.complex128)
    b = torch.sqrt(torch.as_tensor(incident - accepted, dtype=torch.float64)).to(torch.complex128)
    return PortData.from_power_waves(port_name="feed", frequencies=frequencies, a=a, b=b, z0=50.0)


def _dipole_antenna(theta, phi, *, radius=2.0, with_fields=True):
    frequencies = torch.tensor([FREQUENCY], dtype=torch.float64)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    e_theta = (3.0 * torch.sin(theta_grid) * torch.exp(1j * 0.2 * phi_grid))[None].to(torch.complex128)
    e_phi = (0.5j * torch.sin(theta_grid) * torch.cos(phi_grid))[None].to(torch.complex128)
    port = _port(frequencies, incident=torch.tensor([2.0]), accepted=torch.tensor([1.5]))
    full = mw.postprocess.compute_antenna_data(
        frequencies=frequencies, theta=theta, phi=phi, driven_port=port,
        e_theta=e_theta, e_phi=e_phi, radius=radius,
    )
    if with_fields:
        return full
    return mw.postprocess.compute_antenna_data(
        frequencies=frequencies, theta=theta, phi=phi, driven_port=port,
        radiation_intensity=full.radiation_intensity,
    )


def test_antenna_data_element_single_element_reproduces_antenna_metrics():
    theta, phi = _sphere(91, 181)
    antenna = _dipole_antenna(theta, phi)
    data = pp.compute_array_pattern(torch.zeros(1, 3, dtype=torch.float64), element=antenna)
    assert data.element_kind == "vector"
    torch.testing.assert_close(data.radiation_intensity, antenna.radiation_intensity)
    torch.testing.assert_close(data.directivity, antenna.directivity)
    torch.testing.assert_close(data.gain, antenna.gain)
    torch.testing.assert_close(data.realized_gain, antenna.realized_gain)
    torch.testing.assert_close(data.e_theta, antenna.e_theta)


def test_antenna_data_element_pattern_multiplication_and_isolated_gain():
    theta, phi = _sphere(91, 181)
    antenna = _dipole_antenna(theta, phi)
    positions = pp.rectangular_lattice(4, 1, 0.5 * WAVELENGTH)
    weights = pp.taper_weights(4, "hamming").to(torch.complex128)
    steering = (0.3, 0.0)
    data = pp.compute_array_pattern(positions, weights, element=antenna, steering=steering)
    af = pp.array_factor(positions, weights, antenna.theta, antenna.phi, frequency=FREQUENCY, steering=steering)
    torch.testing.assert_close(data.e_theta[0], af * antenna.e_theta[0])
    torch.testing.assert_close(data.e_phi[0], af * antenna.e_phi[0])
    drive = torch.sum(weights.abs().square())
    efficiency = data.p_rad / (drive * antenna.p_accepted)
    torch.testing.assert_close(data.gain, efficiency[:, None, None] * data.directivity)
    torch.testing.assert_close(data.realized_gain * antenna.p_incident, data.gain * antenna.p_accepted)

    intensity_only = pp.compute_array_pattern(
        positions, weights, element=_dipole_antenna(theta, phi, with_fields=False), steering=steering
    )
    assert intensity_only.element_kind == "scalar"
    torch.testing.assert_close(intensity_only.directivity, data.directivity)


def test_far_field_record_element_matches_antenna_data_path():
    theta, phi = _sphere(61, 121)
    antenna = _dipole_antenna(theta, phi, radius=2.0)
    record = {
        "theta": antenna.theta,
        "phi": antenna.phi,
        "E_theta": antenna.e_theta[0],
        "E_phi": antenna.e_phi[0],
        "radius": torch.tensor(2.0, dtype=torch.float64),
        "frequency": FREQUENCY,
    }
    positions = pp.rectangular_lattice(2, 2, 0.5 * WAVELENGTH)
    from_record = pp.compute_array_pattern(positions, element=record)
    from_antenna = pp.compute_array_pattern(positions, element=antenna)
    torch.testing.assert_close(from_record.radiation_intensity, from_antenna.radiation_intensity)
    torch.testing.assert_close(from_record.directivity, from_antenna.directivity)
    assert from_record.gain is None


def test_invalid_inputs_raise():
    theta, phi = _sphere(19, 37)
    positions = pp.rectangular_lattice(3, 1, 0.1)
    with pytest.raises(ValueError):
        pp.compute_array_pattern(positions, theta=theta, phi=phi)
    with pytest.raises(ValueError):
        pp.compute_array_pattern(positions, torch.zeros(3, dtype=torch.complex128), theta=theta, phi=phi, frequencies=FREQUENCY)
    with pytest.raises(ValueError):
        pp.compute_array_pattern(positions, torch.ones(4), theta=theta, phi=phi, frequencies=FREQUENCY)
    with pytest.raises(ValueError):
        pp.compute_array_pattern(positions, theta=theta[:10], phi=phi, frequencies=FREQUENCY)
    with pytest.raises(ValueError):
        pp.array_factor(torch.zeros(3, 2, dtype=torch.float64), None, theta, phi, frequency=FREQUENCY)
