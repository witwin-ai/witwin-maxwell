from collections.abc import Mapping

import pytest
import torch

from witwin.maxwell.network import NetworkData, PERSISTENCE_SCHEMA_VERSION


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def test_network_data_save_load_preserves_port_order_metadata_and_conventions(tmp_path):
    frequencies = torch.tensor([1.0e9, 2.0e9], dtype=torch.float64)
    s = torch.tensor(
        [
            [[0.10 + 0.02j, 0.80 - 0.10j], [0.75 + 0.05j, 0.12 - 0.03j]],
            [[0.15 - 0.01j, 0.70 + 0.20j], [0.68 - 0.15j, 0.08 + 0.04j]],
        ],
        dtype=torch.complex128,
        requires_grad=True,
    )
    convergence = torch.tensor([1.0e-6, 2.0e-6], dtype=torch.float64, requires_grad=True)
    network = NetworkData(
        frequencies=frequencies,
        s=s,
        z0=torch.tensor(
            [[50.0 + 2.0j, 75.0 - 3.0j], [52.0 + 1.0j, 73.0 + 4.0j]],
            dtype=torch.complex128,
        ),
        port_names=("out", "in"),
        valid_columns=torch.tensor([True, False]),
        metadata={
            "solver": "single-device-fdtd",
            "reference_planes": {"out": 0.1, "in": -0.2},
            "convergence": convergence,
            "mode_ids": (0, 0),
        },
        phasor_convention="network peak convention",
        power_wave_convention="network power-wave convention",
    )
    path = tmp_path / "network.sdata.pt"

    network.save(path)
    loaded = NetworkData.load(path, map_location="cpu")

    assert loaded.schema_version == PERSISTENCE_SCHEMA_VERSION
    assert loaded.port_names == ("out", "in")
    assert loaded.metadata["solver"] == network.metadata["solver"]
    assert loaded.metadata["reference_planes"] == network.metadata["reference_planes"]
    assert loaded.metadata["mode_ids"] == network.metadata["mode_ids"]
    assert loaded.phasor_convention == network.phasor_convention
    assert loaded.power_wave_convention == network.power_wave_convention
    torch.testing.assert_close(loaded.frequencies, network.frequencies)
    torch.testing.assert_close(loaded.s, network.s)
    torch.testing.assert_close(loaded.z0, network.z0)
    torch.testing.assert_close(loaded.valid_columns, network.valid_columns)
    torch.testing.assert_close(loaded.metadata["convergence"], convergence)
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(loaded.__dict__))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(loaded.__dict__))
    assert s.requires_grad
    assert convergence.requires_grad


def test_network_data_load_rejects_unknown_schema_version(tmp_path):
    network = NetworkData(
        frequencies=torch.tensor([1.0], dtype=torch.float64),
        s=torch.zeros((1, 1, 1), dtype=torch.complex128),
        z0=50.0,
        port_names=("p1",),
    )
    path = tmp_path / "network.pt"
    network.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = PERSISTENCE_SCHEMA_VERSION + 1
    torch.save(payload, path)

    with pytest.raises(ValueError, match="Unsupported NetworkData schema_version"):
        NetworkData.load(path)
