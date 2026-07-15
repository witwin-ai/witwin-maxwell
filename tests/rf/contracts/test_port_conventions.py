import pytest
import torch

from witwin.maxwell.network import (
    PHASOR_CONVENTION,
    POWER_WAVE_CONVENTION,
    PortData,
    power_waves_to_voltage_current,
    voltage_current_to_power_waves,
)


def _power_waves():
    incident = torch.tensor(
        [1.0 + 0.2j, 0.8 - 0.1j, 1.2 + 0.4j],
        dtype=torch.complex128,
    )
    reflected = torch.tensor(
        [0.1 - 0.05j, -0.2 + 0.1j, 0.3 + 0.2j],
        dtype=torch.complex128,
    )
    z0 = torch.tensor(
        [50.0 + 8.0j, 55.0 - 4.0j, 72.0 + 3.0j],
        dtype=torch.complex128,
    )
    return incident, reflected, z0


def test_kurokawa_power_waves_round_trip_complex_reference_impedance():
    incident, reflected, z0 = _power_waves()

    voltage, current = power_waves_to_voltage_current(incident, reflected, z0)
    actual_incident, actual_reflected = voltage_current_to_power_waves(voltage, current, z0)

    torch.testing.assert_close(actual_incident, incident, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(actual_reflected, reflected, rtol=1.0e-12, atol=1.0e-12)


def test_peak_phasor_power_identity_and_orientation_reversal():
    incident, reflected, z0 = _power_waves()
    voltage, current = power_waves_to_voltage_current(incident, reflected, z0)

    accepted_from_waves = incident.abs().square() - reflected.abs().square()
    accepted_from_vi = 0.5 * torch.real(voltage * torch.conj(current))
    torch.testing.assert_close(accepted_from_waves, accepted_from_vi, rtol=1.0e-12, atol=1.0e-12)

    reversed_incident, reversed_reflected = voltage_current_to_power_waves(-voltage, -current, z0)
    torch.testing.assert_close(reversed_incident, -incident, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(reversed_reflected, -reflected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        reversed_incident.abs().square() - reversed_reflected.abs().square(),
        accepted_from_waves,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("z0", [0.0 + 1.0j, -50.0 + 2.0j])
def test_power_wave_conversion_rejects_nonpositive_reference_resistance(z0):
    voltage = torch.ones(2, dtype=torch.complex128)
    current = torch.ones(2, dtype=torch.complex128)

    with pytest.raises(ValueError, match=r"Re\(z0\) must be strictly positive"):
        voltage_current_to_power_waves(voltage, current, z0)


def test_port_data_exposes_power_impedance_return_loss_and_vswr():
    frequencies = torch.tensor([1.0e9, 2.0e9, 3.0e9], dtype=torch.float64)
    incident, reflected, z0 = _power_waves()
    data = PortData.from_power_waves(
        port_name="p1",
        frequencies=frequencies,
        a=incident,
        b=reflected,
        z0=z0,
        direction="+",
        reference_plane=0.25,
    )

    assert data.port_name == "p1"
    assert data.direction == "+"
    assert data.reference_plane == pytest.approx(0.25)
    assert data.phasor_convention == PHASOR_CONVENTION
    assert data.power_wave_convention == POWER_WAVE_CONVENTION
    assert data.voltage.shape == frequencies.shape
    assert data.current.shape == frequencies.shape
    torch.testing.assert_close(data.a, incident, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(data.b, reflected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(data.incident_power, incident.abs().square())
    torch.testing.assert_close(data.reflected_power, reflected.abs().square())
    torch.testing.assert_close(
        data.accepted_power,
        0.5 * torch.real(data.voltage * torch.conj(data.current)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(data.z_in, data.voltage / data.current)

    reflection = reflected / incident
    torch.testing.assert_close(data.reflection_coefficient, reflection, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(data.return_loss_db, -20.0 * torch.log10(reflection.abs()))
    torch.testing.assert_close(data.vswr, (1.0 + reflection.abs()) / (1.0 - reflection.abs()))


def test_port_data_preserves_autograd_graph():
    frequencies = torch.tensor([1.0e9, 2.0e9], dtype=torch.float64)
    incident = torch.tensor([1.0 + 0.2j, 0.7 - 0.1j], dtype=torch.complex128, requires_grad=True)
    reflected = torch.tensor([0.2 - 0.1j, 0.1 + 0.05j], dtype=torch.complex128)
    z0 = torch.tensor([50.0 + 5.0j, 65.0 - 3.0j], dtype=torch.complex128)

    data = PortData.from_power_waves(
        port_name="p1",
        frequencies=frequencies,
        a=incident,
        b=reflected,
        z0=z0,
    )
    loss = data.accepted_power.sum() + data.voltage.abs().sum()
    loss.backward()

    assert incident.grad is not None
    assert torch.all(torch.isfinite(incident.grad))


def test_port_data_validates_frequency_and_signal_shapes():
    frequencies = torch.tensor([1.0e9, 2.0e9], dtype=torch.float64)

    with pytest.raises(ValueError, match="last dimension"):
        PortData(
            port_name="p1",
            frequencies=frequencies,
            voltage=torch.ones(3, dtype=torch.complex128),
            current=torch.ones(3, dtype=torch.complex128),
            z0=50.0,
        )

    with pytest.raises(TypeError, match="complex"):
        PortData(
            port_name="p1",
            frequencies=frequencies,
            voltage=torch.ones(2, dtype=torch.float64),
            current=torch.ones(2, dtype=torch.float64),
            z0=50.0,
        )
