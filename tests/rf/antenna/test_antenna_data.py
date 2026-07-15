import math

import pytest
import torch

from witwin.maxwell.antenna import AntennaData, Ludwig3
from witwin.maxwell.network import PortData
from witwin.maxwell.postprocess.antenna import compute_antenna_data


def _grid(*, device="cpu", dtype=torch.float64, theta_count=91, phi_count=73):
    theta = torch.linspace(0.0, math.pi, theta_count, device=device, dtype=dtype)
    phi = torch.linspace(0.0, 2.0 * math.pi, phi_count, device=device, dtype=dtype)
    return theta, phi


def _port(frequencies, *, incident, accepted):
    real_dtype = frequencies.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    incident = torch.as_tensor(incident, device=frequencies.device, dtype=real_dtype)
    accepted = torch.as_tensor(accepted, device=frequencies.device, dtype=real_dtype)
    a = torch.sqrt(incident).to(complex_dtype)
    b = torch.sqrt(incident - accepted).to(complex_dtype)
    return PortData.from_power_waves(
        port_name="feed",
        frequencies=frequencies,
        a=a,
        b=b,
        z0=50.0,
    )


def _intensity_data(*, device="cpu", frequency_count=2):
    frequencies = torch.linspace(
        1.0e9,
        1.1e9,
        frequency_count,
        device=device,
        dtype=torch.float64,
    )
    theta, phi = _grid(device=device)
    levels = torch.linspace(
        0.5, 1.0, frequency_count, device=device, dtype=torch.float64
    )
    intensity = levels[:, None, None].expand(-1, theta.numel(), phi.numel())
    port = _port(
        frequencies,
        incident=torch.linspace(10.0, 20.0, frequency_count, device=device),
        accepted=torch.linspace(8.0, 16.0, frequency_count, device=device),
    )
    data = compute_antenna_data(
        frequencies=frequencies,
        theta=theta,
        phi=phi,
        radiation_intensity=intensity,
        driven_port=port,
    )

    return data, intensity


def test_power_gain_and_efficiency_identities_keep_explicit_frequency_axis():
    data, intensity = _intensity_data(frequency_count=2)

    assert isinstance(data, AntennaData)
    assert data.radiation_intensity.shape == (2, 91, 73)
    assert data.p_rad.shape == (2,)
    assert data.eirp.shape == (2,)
    assert data.co_polarized is None
    assert data.cross_polarized is None
    assert data.axial_ratio is None
    assert data.axial_ratio_db is None
    assert data.observation_radius is None
    assert data.wave_impedance is None

    expected_radiated = 4.0 * math.pi * intensity[:, 0, 0]
    torch.testing.assert_close(data.p_rad, expected_radiated, rtol=1.1e-4, atol=1e-10)
    torch.testing.assert_close(
        data.gain,
        data.radiation_efficiency[:, None, None] * data.directivity,
    )
    torch.testing.assert_close(
        data.realized_gain,
        data.system_efficiency[:, None, None] * data.directivity,
    )
    torch.testing.assert_close(
        data.realized_gain,
        data.mismatch_efficiency[:, None, None] * data.gain,
    )
    torch.testing.assert_close(
        data.eirp,
        torch.amax(data.p_incident[:, None, None] * data.realized_gain, dim=(-2, -1)),
    )
    torch.testing.assert_close(
        data.system_efficiency,
        data.radiation_efficiency * data.mismatch_efficiency,
    )
    assert data.phase_center.shape == (3,)
    torch.testing.assert_close(data.frame, torch.eye(3, dtype=torch.float64))
    assert data.field_basis == "spherical_theta_phi"
    assert data.driven_port_name == "feed"
    assert "incident power" in data.power_normalization


def test_single_frequency_is_not_squeezed():
    data, _ = _intensity_data(frequency_count=1)

    assert data.frequencies.shape == (1,)
    assert data.radiation_intensity.shape == (1, 91, 73)
    assert data.directivity.shape == (1, 91, 73)
    assert data.directivity_max.shape == (1,)
    assert data.gain_max.shape == (1,)
    assert data.realized_gain_max.shape == (1,)


def test_ludwig3_resolves_linear_x_polarization_and_linear_axial_ratio():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    theta, phi = _grid(theta_count=9, phi_count=17)
    _, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    e_theta = torch.cos(phi_grid)[None].to(torch.complex128)
    e_phi = -torch.sin(phi_grid)[None].to(torch.complex128)
    port = _port(frequencies, incident=[2.0], accepted=[1.0])

    data = compute_antenna_data(
        frequencies=frequencies,
        theta=theta,
        phi=phi,
        e_theta=e_theta,
        e_phi=e_phi,
        driven_port=port,
        polarization=Ludwig3(),
    )

    torch.testing.assert_close(
        data.co_polarized, torch.ones_like(e_theta), atol=1e-14, rtol=0.0
    )
    torch.testing.assert_close(
        data.cross_polarized, torch.zeros_like(e_theta), atol=1e-14, rtol=0.0
    )
    assert data.observation_radius.shape == e_theta.shape
    assert data.wave_impedance.shape == e_theta.shape
    assert torch.all(torch.isinf(data.axial_ratio))


def test_circular_polarization_boresight_axial_ratio_is_below_half_db():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    theta, phi = _grid(theta_count=9, phi_count=17)
    shape = (1, theta.numel(), phi.numel())
    e_theta = torch.ones(shape, dtype=torch.complex128)
    e_phi = 1j * torch.ones(shape, dtype=torch.complex128)
    port = _port(frequencies, incident=[2.0], accepted=[1.0])

    data = compute_antenna_data(
        frequencies=frequencies,
        theta=theta,
        phi=phi,
        e_theta=e_theta,
        e_phi=e_phi,
        driven_port=port,
    )

    assert data.axial_ratio_db[0, 0, 0] < 0.5
    torch.testing.assert_close(
        data.axial_ratio_db,
        torch.zeros_like(data.axial_ratio_db),
        atol=1e-12,
        rtol=0.0,
    )


def test_half_wave_dipole_directivity_exit_gate_is_within_quarter_db():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    theta, phi = _grid(theta_count=721, phi_count=361)
    sine = torch.sin(theta)
    pattern = torch.where(
        torch.abs(sine) > 1e-12,
        (torch.cos(0.5 * math.pi * torch.cos(theta)) / sine).square(),
        torch.zeros_like(theta),
    )
    intensity = pattern[None, :, None].expand(1, -1, phi.numel())
    port = _port(frequencies, incident=[2.0], accepted=[1.5])

    data = compute_antenna_data(
        frequencies=frequencies,
        theta=theta,
        phi=phi,
        radiation_intensity=intensity,
        driven_port=port,
    )

    expected_peak_db = 10.0 * math.log10(1.640922376984585)
    measured_peak_db = 10.0 * torch.log10(data.directivity_max[0])
    assert abs(float(measured_peak_db) - expected_peak_db) < 0.25


def test_antenna_metrics_preserve_far_field_and_port_autograd_graphs():
    frequencies = torch.tensor([1.0e9, 1.2e9], dtype=torch.float64)
    theta, phi = _grid(theta_count=13, phi_count=17)
    shape = (2, theta.numel(), phi.numel())
    e_theta = torch.full(shape, 1.0 + 0.2j, dtype=torch.complex128, requires_grad=True)
    e_phi = torch.full(shape, 0.3 - 0.4j, dtype=torch.complex128, requires_grad=True)
    a = torch.tensor(
        [2.0 + 0.1j, 2.2 - 0.1j], dtype=torch.complex128, requires_grad=True
    )
    b = torch.tensor(
        [0.4 - 0.1j, 0.5 + 0.2j], dtype=torch.complex128, requires_grad=True
    )
    port = PortData.from_power_waves(
        port_name="feed",
        frequencies=frequencies,
        a=a,
        b=b,
        z0=50.0,
    )

    data = compute_antenna_data(
        frequencies=frequencies,
        theta=theta,
        phi=phi,
        e_theta=e_theta,
        e_phi=e_phi,
        driven_port=port,
    )
    loss = (
        data.p_rad.sum()
        + data.gain.square().mean()
        + data.realized_gain.square().mean()
        + data.eirp.sum()
    )
    loss.backward()

    for tensor in (e_theta, e_phi, a, b):
        assert tensor.grad is not None
        assert torch.all(torch.isfinite(tensor.grad))


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_all_outputs_remain_on_the_input_device(device):
    resolved_device = torch.empty((), device=device).device
    frequencies = torch.tensor([1.0e9], device=device, dtype=torch.float32)
    theta, phi = _grid(device=device, dtype=torch.float32, theta_count=7, phi_count=9)
    shape = (1, theta.numel(), phi.numel())
    e_theta = torch.ones(shape, device=device, dtype=torch.complex64)
    e_phi = torch.zeros_like(e_theta)
    port = _port(frequencies, incident=[2.0], accepted=[1.0])

    data = compute_antenna_data(
        frequencies=frequencies,
        theta=theta,
        phi=phi,
        e_theta=e_theta,
        e_phi=e_phi,
        driven_port=port,
        phase_center=torch.zeros(3, device=device),
        frame=torch.eye(3, device=device),
    )

    for value in data.__dict__.values():
        if isinstance(value, torch.Tensor):
            assert value.device == resolved_device


def test_rejects_ambiguous_shapes_normalizations_and_nonpositive_powers():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    theta, phi = _grid(theta_count=7, phi_count=9)
    shape = (1, theta.numel(), phi.numel())
    e_theta = torch.ones(shape, dtype=torch.complex128)
    e_phi = torch.zeros_like(e_theta)
    port = _port(frequencies, incident=[2.0], accepted=[1.0])

    with pytest.raises(ValueError, match="supplied together"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            e_theta=e_theta,
            driven_port=port,
        )
    with pytest.raises(ValueError, match="either"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            e_theta=e_theta,
            e_phi=e_phi,
            radiation_intensity=torch.ones(shape, dtype=torch.float64),
            driven_port=port,
        )
    with pytest.raises(ValueError, match=r"\[F, T, P\]"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            e_theta=e_theta[0],
            e_phi=e_phi[0],
            driven_port=port,
        )
    with pytest.raises(TypeError, match="complex"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            e_theta=e_theta.real,
            e_phi=e_phi,
            driven_port=port,
        )
    with pytest.raises(ValueError, match="span exactly"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi[:-1],
            radiation_intensity=torch.ones(
                (1, theta.numel(), phi.numel() - 1), dtype=torch.float64
            ),
            driven_port=port,
        )

    reflecting_port = _port(frequencies, incident=[1.0], accepted=[0.0])
    with pytest.raises(ValueError, match="accepted power"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            radiation_intensity=torch.ones(shape, dtype=torch.float64),
            driven_port=reflecting_port,
        )


def test_rejects_port_frequency_excitation_and_frame_contract_violations():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    theta, phi = _grid(theta_count=7, phi_count=9)
    intensity = torch.ones((1, theta.numel(), phi.numel()), dtype=torch.float64)
    mismatched_port = _port(
        torch.tensor([1.1e9], dtype=torch.float64),
        incident=[2.0],
        accepted=[1.0],
    )
    with pytest.raises(ValueError, match="match frequencies"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            radiation_intensity=intensity,
            driven_port=mismatched_port,
        )

    voltage = torch.ones((2, 1), dtype=torch.complex128)
    batched_port = PortData(
        port_name="feed",
        frequencies=frequencies,
        voltage=voltage,
        current=voltage / 50.0,
        z0=50.0,
    )
    with pytest.raises(ValueError, match="one excitation"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            radiation_intensity=intensity,
            driven_port=batched_port,
        )

    port = _port(frequencies, incident=[2.0], accepted=[1.0])
    invalid_frame = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="right-handed"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            radiation_intensity=intensity,
            driven_port=port,
            frame=invalid_frame,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rejects_cross_device_far_field_and_port_inputs():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    theta, phi = _grid(theta_count=7, phi_count=9)
    shape = (1, theta.numel(), phi.numel())
    cuda_port = _port(
        frequencies.cuda(),
        incident=[2.0],
        accepted=[1.0],
    )

    with pytest.raises(ValueError, match="same device"):
        compute_antenna_data(
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            radiation_intensity=torch.ones(shape, dtype=torch.float64),
            driven_port=cuda_port,
        )
