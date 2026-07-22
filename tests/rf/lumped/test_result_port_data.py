from collections.abc import Mapping

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.network import PERSISTENCE_SCHEMA_VERSION, PortData
from witwin.maxwell.result import RESULT_SNAPSHOT_SCHEMA_VERSION, Result


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )


def _port_data(*, requires_grad=False):
    return PortData.from_power_waves(
        port_name="feed",
        frequencies=torch.tensor([1.0e9, 2.0e9], dtype=torch.float64),
        a=torch.tensor(
            [1.0 + 0.2j, 0.8 - 0.1j],
            dtype=torch.complex128,
            requires_grad=requires_grad,
        ),
        b=torch.tensor([0.2 - 0.05j, 0.1 + 0.02j], dtype=torch.complex128),
        z0=torch.tensor([50.0 + 4.0j, 55.0 - 2.0j], dtype=torch.complex128),
        direction="-",
        reference_plane=0.125,
        available_power=torch.tensor([1.25, 1.1], dtype=torch.float64),
        metadata={
            "units": {"voltage": "V", "current": "A", "power": "W"},
            "calibration": torch.tensor(
                [0.98, 1.01],
                dtype=torch.float64,
                requires_grad=requires_grad,
            ),
        },
    )


def test_result_exposes_named_port_data_without_detaching_live_tensors():
    port = _port_data(requires_grad=True)
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequencies=(1.0e9, 2.0e9),
        ports={"feed": port},
    )

    assert result.port("feed") is port
    assert result.ports == {"feed": port}
    assert result.port("feed").voltage.requires_grad
    assert result.stats()["num_ports"] == 1

    snapshot = result.ports
    snapshot.clear()
    assert result.port("feed") is port

    with pytest.raises(KeyError, match="Port 'missing' is not available"):
        result.port("missing")


def test_result_rejects_invalid_port_mapping_entries():
    port = _port_data()

    with pytest.raises(ValueError, match="does not match PortData.port_name"):
        Result(
            method="fdtd",
            scene=_scene(),
            frequencies=(1.0e9, 2.0e9),
            ports={"other": port},
        )

    with pytest.raises(TypeError, match="PortData"):
        Result(
            method="fdtd",
            scene=_scene(),
            frequencies=(1.0e9, 2.0e9),
            ports={"feed": object()},
        )


def test_result_save_embeds_detached_cpu_port_snapshots(tmp_path):
    port = _port_data(requires_grad=True)
    field = torch.ones((2, 2), dtype=torch.complex128, requires_grad=True)
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequencies=(1.0e9, 2.0e9),
        fields={"EX": field},
        ports={"feed": port},
        metadata={"run": "phase1"},
    )
    path = tmp_path / "nested" / "result.pt"

    result.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["schema_version"] == RESULT_SNAPSHOT_SCHEMA_VERSION
    assert payload["data_type"] == "ResultSnapshot"
    assert "scene" not in payload
    assert tuple(payload["ports"]) == ("feed",)
    assert payload["circuits"] == {}
    saved = payload["ports"]["feed"]
    assert saved["schema_version"] == PERSISTENCE_SCHEMA_VERSION
    assert saved["data_type"] == "PortData"
    assert saved["port_name"] == port.port_name
    assert saved["direction"] == port.direction
    assert saved["reference_plane"] == pytest.approx(port.reference_plane)
    assert saved["metadata"]["units"] == port.metadata["units"]
    assert saved["phasor_convention"] == port.phasor_convention
    assert saved["power_wave_convention"] == port.power_wave_convention
    torch.testing.assert_close(saved["frequencies"], port.frequencies)
    torch.testing.assert_close(saved["voltage"], port.voltage)
    torch.testing.assert_close(saved["current"], port.current)
    torch.testing.assert_close(saved["z0"], port.z0)
    torch.testing.assert_close(saved["available_power"], port.available_power)
    torch.testing.assert_close(saved["metadata"]["calibration"], port.metadata["calibration"])
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(payload))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(payload))

    loaded = Result.load(path, scene=_scene())
    restored = loaded.port("feed")
    assert restored.port_name == port.port_name
    assert restored.direction == port.direction
    assert restored.reference_plane == pytest.approx(port.reference_plane)
    torch.testing.assert_close(restored.voltage, port.voltage.detach())
    torch.testing.assert_close(restored.current, port.current.detach())
    torch.testing.assert_close(restored.z0, port.z0.detach())
    assert all(not tensor.requires_grad for tensor in _iter_tensors(restored.metadata))
    assert field.requires_grad
    assert port.voltage.requires_grad
