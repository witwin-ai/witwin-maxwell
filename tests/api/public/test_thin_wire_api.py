from __future__ import annotations

import sys
from types import ModuleType
from types import MappingProxyType

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def _wire(**overrides):
    parameters = {
        "name": "feed",
        "points": ((0.0, 0.0, -0.25), (0.0, 0.0, 0.0), (0.0, 0.0, 0.25)),
        "radius": 5.0e-4,
        "conductor": mw.WireConductor.pec(),
    }
    parameters.update(overrides)
    return mw.ThinWire(**parameters)


def _wire_data(*, monitor_name="state", metadata=None):
    frequencies = torch.tensor([1.0e9, 2.0e9], dtype=torch.float64)
    current = torch.tensor(
        [[1.0 + 0.5j, 0.75 - 0.25j], [0.5 + 0.25j, 0.25 - 0.125j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    charge = torch.tensor(
        [[0.1j, 0.0j, -0.1j], [0.05j, 0.0j, -0.05j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    return mw.WireData(
        monitor_name=monitor_name,
        wire_name="feed",
        frequencies=frequencies,
        current=current,
        charge=charge,
        ohmic_loss=torch.zeros_like(current.real),
        metadata={} if metadata is None else metadata,
    )


def test_thin_wire_objects_are_top_level_public_api():
    assert mw.ThinWire is not None
    assert mw.WireConductor is not None
    assert mw.WireEnd is not None
    assert mw.WireMonitor is not None
    assert mw.WireData is not None


def test_conductor_and_endpoint_contracts_are_explicit():
    assert mw.WireConductor.pec().kind == "pec"
    with pytest.raises(ValueError, match="supports the 'pec'"):
        mw.WireConductor("finite")

    assert mw.WireEnd.open() == mw.WireEnd("open")
    grounded = mw.WireEnd.grounded(structure="ground")
    assert grounded.kind == "grounded"
    assert grounded.structure == "ground"
    with pytest.raises(ValueError, match="must not be empty"):
        mw.WireEnd.grounded(structure=" ")
    with pytest.raises(ValueError, match="cannot reference"):
        mw.WireEnd("open", structure="ground")
    junction = mw.WireEnd.node("junction")
    assert junction.kind == "node"
    assert junction.node_name == "junction"
    with pytest.raises(ValueError, match="must not be empty"):
        mw.WireEnd.node(" ")
    with pytest.raises(ValueError, match="reserved"):
        mw.WireEnd.node("__closed__:ghost")
    with pytest.raises(ValueError, match="cannot reference a structure"):
        mw.WireEnd("node", structure="ground", node_name="junction")


def test_thin_wire_preserves_trainable_tensors_and_validates_geometry():
    points = torch.tensor(
        [[0.0, 0.0, -0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.25]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radius = torch.tensor([4.0e-4, 5.0e-4], dtype=torch.float64, requires_grad=True)
    # Trainable points are only accepted under the continuous-snap coordinate
    # gradient contract; the tensors must survive construction unchanged.
    wire = _wire(points=points, radius=radius, snap="continuous")

    assert wire.points is points
    assert wire.radius is radius
    assert wire.segment_count == 2
    assert wire.endpoints == (mw.WireEnd.open(), mw.WireEnd.open())
    assert wire.snap == "continuous"

    # Trainable points with a non-continuous snap are rejected fail-closed so the
    # fixed-stencil gradient contract stays explicit.
    with pytest.raises(ValueError, match="trainable points require snap='continuous'"):
        _wire(points=points, radius=radius)

    with pytest.raises(ValueError, match="P >= 2"):
        _wire(points=torch.zeros((1, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="zero-length"):
        _wire(points=((0, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError, match="duplicate nodes"):
        _wire(points=((0, 0, 0), (1, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError, match="contain 2 values"):
        _wire(radius=(1.0e-3,))
    with pytest.raises(ValueError, match="positive"):
        _wire(radius=0.0)
    with pytest.raises(ValueError, match="nearest.*strict"):
        _wire(snap="project")
    with pytest.raises(ValueError, match="exactly two"):
        _wire(endpoints=(mw.WireEnd.open(),))

    loop = _wire(
        points=((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0))
    )
    assert loop.is_closed
    assert loop.endpoints == ()
    with pytest.raises(ValueError, match="must not specify endpoints"):
        _wire(
            points=((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)),
            endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
        )


def test_scene_owns_wires_separately_and_propagates_through_clone_and_prepare():
    wire = _wire()
    scene = mw.Scene(thin_wires=(wire,), device="cpu")

    assert scene.thin_wires == (wire,)
    assert scene.structures == []
    with pytest.raises(AttributeError):
        scene.thin_wires = ()
    with pytest.raises(ValueError, match="already present"):
        scene.add_thin_wire(wire)
    with pytest.raises(TypeError, match="ThinWire"):
        mw.Scene(device="cpu").add_thin_wire(object())

    assert scene.clone().thin_wires == (wire,)
    prepared = prepare_scene(scene)
    assert prepared.thin_wires == (wire,)
    assert prepared._public_scene is scene


def test_non_fdtd_simulation_rejects_thin_wire_scene_at_construction():
    scene = mw.Scene(thin_wires=(_wire(),), device="cpu")
    with pytest.raises(ValueError, match="FDTD-only"):
        mw.Simulation.fdfd(scene, frequency=1.0e9)


def test_trainable_wire_radius_is_discovered_by_fdtd_simulation():
    radius = torch.tensor(5.0e-4, dtype=torch.float64, requires_grad=True)
    scene = mw.Scene(thin_wires=(_wire(radius=radius),), device="cpu")
    simulation = mw.Simulation.fdtd(scene, frequency=1.0e9)
    assert simulation.has_trainable_parameters is True


def test_scene_thin_wire_compile_seams_forward_to_compiler(monkeypatch):
    calls = []
    compiler = ModuleType("witwin.maxwell.compiler.thin_wire")

    def compile_thin_wires(scene, *, device):
        calls.append(("wires", scene, device))
        return "compiled-wires"

    def compile_wire_monitors(scene, network):
        calls.append(("monitors", scene, network))
        return "compiled-monitors"

    compiler.compile_thin_wires = compile_thin_wires
    compiler.compile_wire_monitors = compile_wire_monitors
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)

    scene = mw.Scene(thin_wires=(_wire(),), device="cpu")
    prepared = prepare_scene(scene)
    assert prepared.compile_thin_wires() == "compiled-wires"
    assert prepared.compile_wire_monitors(device="cpu") == "compiled-monitors"
    assert calls == [
        ("wires", prepared, "cpu"),
        ("wires", prepared, "cpu"),
        ("monitors", prepared, "compiled-wires"),
    ]


def test_wire_monitor_validates_frequency_and_quantity_contracts():
    monitor = mw.WireMonitor(
        name="state",
        wire="feed",
        frequencies=(1.0e9, 2.0e9),
        quantities=("charge", "current"),
    )
    assert monitor.frequencies == (1.0e9, 2.0e9)
    assert monitor.quantities == ("charge", "current")
    assert monitor.kind == "wire"

    with pytest.raises(ValueError, match="strictly positive"):
        mw.WireMonitor("state", "feed", frequencies=(0.0,))
    with pytest.raises(ValueError, match="Unsupported"):
        mw.WireMonitor("state", "feed", frequencies=(1.0,), quantities=("voltage",))
    with pytest.raises(ValueError, match="unique"):
        mw.WireMonitor(
            "state", "feed", frequencies=(1.0,), quantities=("current", "current")
        )


def test_wire_data_is_frequency_first_device_resident_and_typed_in_result():
    data = _wire_data()
    monitor = mw.WireMonitor("state", "feed", frequencies=(1.0e9, 2.0e9))
    scene = mw.Scene(thin_wires=(_wire(),), monitors=(monitor,), device="cpu")
    result = mw.Result(
        method="fdtd",
        scene=scene,
        frequencies=(1.0e9, 2.0e9),
        monitors={"state": data},
    )

    assert data.current.shape == (2, 2)
    assert data.charge.shape == (2, 3)
    assert data.ohmic_loss.shape == (2, 2)
    assert data.ohmic_loss.device == data.current.device
    assert torch.count_nonzero(data.ohmic_loss) == 0
    assert result.monitor("state") is data
    assert result.raw_monitor("state") is data
    with pytest.raises(ValueError, match="explicit frequency axis"):
        result.monitor("state", freq_index=0)
    with pytest.raises(ValueError, match="explicit frequency axis"):
        result.at(freq_index=0).monitor("state")

    with pytest.raises(ValueError, match="does not match"):
        mw.Result(
            method="fdtd",
            scene=scene,
            frequencies=(1.0e9, 2.0e9),
            monitors={"wrong": data},
        )


def test_wire_data_allows_only_requested_quantities():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    current = torch.ones((1, 2), dtype=torch.complex128)
    data = mw.WireData(
        monitor_name="current_only",
        wire_name="feed",
        frequencies=frequencies,
        current=current,
        charge=None,
        ohmic_loss=None,
        metadata={"quantities": ("current",)},
    )
    assert data.current is current
    assert data.charge is None
    assert data.ohmic_loss is None


def test_result_snapshot_restores_detached_wire_data(tmp_path):
    data = _wire_data(metadata={"validity": {"a_over_delta": torch.tensor(0.01)}})
    monitor = mw.WireMonitor("state", "feed", frequencies=(1.0e9, 2.0e9))
    scene = mw.Scene(thin_wires=(_wire(),), monitors=(monitor,), device="cpu")
    result = mw.Result(
        method="fdtd",
        scene=scene,
        frequencies=(1.0e9, 2.0e9),
        monitors={"state": data},
    )
    path = tmp_path / "result.pt"

    result.save(path)
    restored = mw.Result.load(path, scene=scene)
    restored_data = restored.monitor("state")

    assert isinstance(restored_data, mw.WireData)
    assert restored_data.monitor_name == "state"
    assert restored_data.wire_name == "feed"
    assert restored_data.current.device.type == "cpu"
    assert not restored_data.current.requires_grad
    torch.testing.assert_close(restored_data.current, data.current.detach())
    torch.testing.assert_close(restored_data.charge, data.charge.detach())
    torch.testing.assert_close(restored_data.ohmic_loss, data.ohmic_loss)
    torch.testing.assert_close(
        restored_data.metadata["validity"]["a_over_delta"],
        data.metadata["validity"]["a_over_delta"],
    )


def test_result_snapshot_recurses_through_nested_mapping_metadata(tmp_path):
    nested = MappingProxyType(
        {"validity": MappingProxyType({"a_over_delta": torch.tensor(0.01)})}
    )
    data = _wire_data(metadata=nested)
    scene = mw.Scene(device="cpu")
    result = mw.Result(
        method="fdtd",
        scene=scene,
        frequencies=(1.0e9, 2.0e9),
        monitors={"state": data},
    )

    path = tmp_path / "mapping.pt"
    result.save(path)
    restored = mw.Result.load(path, scene=scene).monitor("state")
    torch.testing.assert_close(
        restored.metadata["validity"]["a_over_delta"],
        torch.tensor(0.01),
    )


def test_result_snapshot_rejects_unsafe_wire_metadata(tmp_path):
    data = _wire_data(metadata={"unsafe": object()})
    scene = mw.Scene(device="cpu")
    result = mw.Result(
        method="fdtd",
        scene=scene,
        frequencies=(1.0e9, 2.0e9),
        monitors={"state": data},
    )

    with pytest.raises(TypeError, match="unsupported persistence type"):
        result.save(tmp_path / "unsafe.pt")
