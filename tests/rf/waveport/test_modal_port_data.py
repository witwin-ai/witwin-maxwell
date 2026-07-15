import pytest
import torch

from witwin.maxwell.network import PortData


def _modal_port_data(*, requires_grad=False):
    frequencies = torch.tensor((1.0e9, 2.0e9, 3.0e9), dtype=torch.float64)
    a = torch.ones((2, 3), dtype=torch.complex128, requires_grad=requires_grad)
    b = torch.tensor(
        ((0.1 + 0.2j, 0.2 + 0.1j, 0.3 - 0.1j), (0.0, 0.1j, -0.2j)),
        dtype=torch.complex128,
    )
    impedance = torch.tensor(
        ((50.0, 51.0, 52.0), (75.0, 76.0, 77.0)),
        dtype=torch.complex128,
    )
    return PortData.from_power_waves(
        port_name="wave",
        frequencies=frequencies,
        a=a,
        b=b,
        z0=impedance,
        mode_names=("TE0", "TE1"),
        beta=torch.tensor(
            ((10.0, 20.0, 30.0), (8.0, 18.0, 28.0)),
            dtype=torch.float64,
        ),
        characteristic_impedance=impedance,
        tracking_confidence=torch.ones((2, 3), dtype=torch.float64),
    )


def test_modal_port_data_preserves_typed_mode_contract_and_autograd():
    data = _modal_port_data(requires_grad=True)

    assert data.mode_names == ("TE0", "TE1")
    assert data.voltage.shape == (2, 3)
    assert data.beta.shape == (2, 3)
    assert data.characteristic_impedance.shape == (2, 3)
    assert data.tracking_confidence.shape == (2, 3)

    data.reflected_power.sum().backward()
    assert data.voltage.grad_fn is not None


def test_lumped_and_wave_modal_views_share_the_same_power_wave_s_parameter():
    frequencies = torch.tensor((1.0e9, 1.5e9, 2.0e9), dtype=torch.float64)
    voltage = torch.tensor(
        (1.0 + 0.2j, 0.8 - 0.1j, 1.1 + 0.05j),
        dtype=torch.complex128,
    )
    current = torch.tensor(
        (0.015 - 0.002j, 0.012 + 0.001j, 0.017 - 0.001j),
        dtype=torch.complex128,
    )
    lumped = PortData(
        port_name="lumped",
        frequencies=frequencies,
        voltage=voltage,
        current=current,
        z0=50.0,
    )
    wave = PortData(
        port_name="wave",
        frequencies=frequencies,
        voltage=voltage.unsqueeze(0),
        current=current.unsqueeze(0),
        z0=torch.full((1, 3), 50.0, dtype=torch.complex128),
        mode_names=("TEM0",),
        beta=torch.ones((1, 3), dtype=torch.float64),
        characteristic_impedance=torch.full(
            (1, 3),
            50.0,
            dtype=torch.complex128,
        ),
        tracking_confidence=torch.ones((1, 3), dtype=torch.float64),
    )

    difference = torch.max(
        torch.abs(lumped.reflection_coefficient - wave.reflection_coefficient[0])
    )
    assert difference < 0.03
    torch.testing.assert_close(difference, torch.zeros_like(difference))


def test_modal_port_data_persistence_round_trip_is_detached(tmp_path):
    data = _modal_port_data(requires_grad=True)
    path = tmp_path / "wave-port.pt"

    data.save(path)
    loaded = PortData.load(path)

    assert loaded.mode_names == data.mode_names
    torch.testing.assert_close(loaded.beta, data.beta.detach())
    torch.testing.assert_close(
        loaded.characteristic_impedance,
        data.characteristic_impedance.detach(),
    )
    torch.testing.assert_close(loaded.tracking_confidence, data.tracking_confidence)
    assert not loaded.voltage.requires_grad


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"mode_names": ("TE0",)}, "mode dimension"),
        (
            {
                "mode_names": ("TE0", "TE1"),
                "beta": torch.ones((2, 2)),
            },
            "shape",
        ),
        (
            {
                "mode_names": ("TE0", "TE1"),
                "tracking_confidence": torch.full((2, 3), 1.1),
            },
            r"\[0, 1\]",
        ),
    ),
)
def test_modal_port_data_rejects_invalid_mode_contract(kwargs, match):
    frequencies = torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64)
    voltage = torch.ones((2, 3), dtype=torch.complex128)

    with pytest.raises(ValueError, match=match):
        PortData(
            port_name="wave",
            frequencies=frequencies,
            voltage=voltage,
            current=voltage,
            z0=50.0,
            **kwargs,
        )
