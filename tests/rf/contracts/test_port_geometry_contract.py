import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.compiler.ports import CompiledPortGeometry, compile_port_geometry
from witwin.maxwell.ports import AxisPath, LumpedPort
from witwin.maxwell.scene import prepare_scene


def _prepared_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    return prepare_scene(scene)


def _port(**overrides):
    values = {
        "name": "feed",
        "negative": (0.5, 0.5, 0.2),
        "positive": (0.5, 0.5, 0.8),
        "voltage_path": AxisPath("z"),
        "current_surface": Box(position=(0.5, 0.5, 0.45), size=(0.5, 0.5, 0.0)),
        "reference_impedance": 50.0,
    }
    values.update(overrides)
    return LumpedPort(**values)


def test_axis_path_and_lumped_port_freeze_the_phase_zero_convention():
    port = _port()

    assert port.kind == "lumped_port"
    assert port.voltage_path.axis == "z"
    assert port.negative == (0.5, 0.5, 0.2)
    assert port.positive == (0.5, 0.5, 0.8)
    assert port.reference_impedance == 50.0
    assert port.phasor_convention == "peak"
    assert port.power_convention == "0.5*Re(V*conj(I))"


@pytest.mark.parametrize("axis", ["xy", "", 2])
def test_axis_path_rejects_unknown_axes(axis):
    with pytest.raises((TypeError, ValueError), match="axis"):
        AxisPath(axis)


def test_lumped_port_requires_an_axis_aligned_nonzero_voltage_path():
    with pytest.raises(ValueError, match="axis-aligned"):
        _port(positive=(0.6, 0.5, 0.8))

    with pytest.raises(ValueError, match="distinct"):
        _port(positive=(0.5, 0.5, 0.2))


def test_lumped_port_requires_positive_real_reference_impedance():
    with pytest.raises(ValueError, match="real part"):
        _port(reference_impedance=-1.0)


def test_compiled_geometry_is_sparse_and_resident_on_the_requested_device():
    scene = _prepared_scene()
    compiled = compile_port_geometry(scene, _port(), device="cpu")

    assert isinstance(compiled, CompiledPortGeometry)
    assert compiled.port_name == "feed"
    assert compiled.axis == "z"
    assert compiled.direction == 1
    assert compiled.voltage_component == "Ez"
    assert compiled.voltage_indices.shape == (6, 3)
    assert compiled.voltage_indices.dtype == torch.int64
    assert compiled.voltage_weights.device.type == "cpu"
    assert compiled.current_components == ("Hx", "Hy", "Hx", "Hy")
    assert all(indices.ndim == 2 and indices.shape[1] == 3 for indices in compiled.current_indices)
    assert all(weights.device.type == "cpu" for weights in compiled.current_weights)


def test_compiler_rejects_current_surfaces_that_do_not_follow_the_yee_contract():
    scene = _prepared_scene()

    with pytest.raises(ValueError, match="planar"):
        compile_port_geometry(
            scene,
            _port(current_surface=Box(position=(0.5, 0.5, 0.45), size=(0.5, 0.5, 0.1))),
        )

    with pytest.raises(ValueError, match="normal"):
        compile_port_geometry(
            scene,
            _port(current_surface=Box(position=(0.5, 0.5, 0.45), size=(0.0, 0.5, 0.5))),
        )

    with pytest.raises(ValueError, match="half-grid"):
        compile_port_geometry(
            scene,
            _port(current_surface=Box(position=(0.5, 0.5, 0.45), size=(0.4, 0.4, 0.0))),
        )
