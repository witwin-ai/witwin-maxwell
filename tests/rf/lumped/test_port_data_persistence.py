from collections.abc import Mapping

import pytest
import torch

from witwin.maxwell.network import PERSISTENCE_SCHEMA_VERSION, PortData


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def test_port_data_save_load_round_trip_is_a_detached_cpu_snapshot(tmp_path):
    frequencies = torch.tensor([1.0e9, 2.0e9], dtype=torch.float64)
    incident = torch.tensor(
        [1.0 + 0.2j, 0.8 - 0.1j],
        dtype=torch.complex128,
        requires_grad=True,
    )
    reflected = torch.tensor([0.2 - 0.05j, 0.1 + 0.02j], dtype=torch.complex128)
    available_power = torch.tensor([1.25, 1.1], dtype=torch.float64, requires_grad=True)
    calibration = torch.tensor([0.98, 1.01], dtype=torch.float64, requires_grad=True)
    data = PortData.from_power_waves(
        port_name="feed",
        frequencies=frequencies,
        a=incident,
        b=reflected,
        z0=torch.tensor([50.0 + 4.0j, 55.0 - 2.0j], dtype=torch.complex128),
        direction="-",
        reference_plane=0.125,
        available_power=available_power,
        metadata={
            "units": {"voltage": "V", "current": "A", "power": "W"},
            "calibration": calibration,
            "labels": ("drive", "sense"),
        },
        phasor_convention="custom peak convention",
        power_wave_convention="custom power-wave convention",
    )
    path = tmp_path / "nested" / "feed.port.pt"

    data.save(path)
    loaded = PortData.load(path, map_location=torch.device("cpu"))

    assert loaded.schema_version == PERSISTENCE_SCHEMA_VERSION
    assert loaded.port_name == data.port_name
    assert loaded.direction == data.direction
    assert loaded.reference_plane == pytest.approx(data.reference_plane)
    assert loaded.phasor_convention == data.phasor_convention
    assert loaded.power_wave_convention == data.power_wave_convention
    assert loaded.metadata["units"] == data.metadata["units"]
    assert loaded.metadata["labels"] == data.metadata["labels"]
    torch.testing.assert_close(loaded.frequencies, data.frequencies)
    torch.testing.assert_close(loaded.voltage, data.voltage)
    torch.testing.assert_close(loaded.current, data.current)
    torch.testing.assert_close(loaded.z0, data.z0)
    torch.testing.assert_close(loaded.available_power, data.available_power)
    torch.testing.assert_close(loaded.metadata["calibration"], calibration)
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(loaded.__dict__))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(loaded.__dict__))

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == PERSISTENCE_SCHEMA_VERSION
    assert payload["data_type"] == "PortData"
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(payload))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(payload))
    assert incident.requires_grad
    assert available_power.requires_grad
    assert calibration.requires_grad


def test_port_data_load_rejects_a_different_persisted_type(tmp_path):
    path = tmp_path / "wrong-type.pt"
    torch.save(
        {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "data_type": "NetworkData",
        },
        path,
    )

    with pytest.raises(ValueError, match="contains NetworkData, not PortData"):
        PortData.load(path)


def test_port_data_save_rejects_unsafe_metadata_before_writing(tmp_path):
    data = PortData.from_power_waves(
        port_name="feed",
        frequencies=torch.tensor([1.0]),
        a=torch.tensor([1.0 + 0.0j]),
        b=torch.tensor([0.0 + 0.0j]),
        z0=50.0,
        metadata={"unsafe": object()},
    )
    path = tmp_path / "unsafe.pt"

    with pytest.raises(TypeError, match="unsupported persistence type object"):
        data.save(path)
    assert not path.exists()
