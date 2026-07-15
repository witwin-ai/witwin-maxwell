from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.scene import prepare_scene


def _pec_box(name, *, position, size=(0.5, 0.5, 0.2), rotation=None):
    return mw.Structure(
        geometry=mw.Box(position=position, size=size, rotation=rotation),
        material=mw.Material.pec(),
        name=name,
    )


def _structures():
    return (
        _pec_box("signal", position=(0.5, 0.5, 0.8)),
        _pec_box("ground", position=(0.5, 0.5, 0.2)),
    )


def _port(**overrides):
    values = {
        "name": "feed",
        "positive_terminal": mw.TerminalRef("signal"),
        "negative_terminal": mw.TerminalRef("ground"),
        "integration_path": mw.AxisPath("z"),
        "reference_plane": 0.45,
        "reference_impedance": 50.0,
    }
    values.update(overrides)
    return mw.TerminalPort(**values)


def _scene(*, structures=None, port=None):
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        structures=_structures() if structures is None else structures,
        ports=(_port() if port is None else port,),
        device="cpu",
    )


def test_terminal_public_contract_and_scene_resolution():
    termination = mw.SeriesRLC(r=50.0)
    scene = _scene(port=_port(termination=termination))
    port = scene.ports[0]

    assert isinstance(port, mw.TerminalPort)
    assert port.kind == "terminal_port"
    assert port.positive_terminal == mw.TerminalRef("signal")
    assert port.negative_terminal == mw.TerminalRef("ground")
    assert port.integration_path == mw.AxisPath("z")
    assert port.voltage_path is port.integration_path
    assert port.positive == pytest.approx((0.5, 0.5, 0.7))
    assert port.negative == pytest.approx((0.5, 0.5, 0.3))
    assert tuple(port.current_surface.position) == pytest.approx((0.5, 0.5, 0.45))
    assert tuple(port.current_surface.size) == pytest.approx((0.5, 0.5, 0.0))
    assert port.reference_impedance == 50.0
    assert port.reference_plane == 0.45
    assert port.termination is termination
    assert port.phasor_convention == "peak"
    assert port.power_convention == "0.5*Re(V*conj(I))"


def test_terminal_port_accepts_the_frozen_positional_contract():
    port = mw.TerminalPort(
        "feed",
        mw.TerminalRef("signal"),
        mw.TerminalRef("ground"),
        mw.AxisPath("z"),
        0.45,
    )

    assert port.reference_impedance == 50.0
    assert port.reference_plane == 0.45


def test_scene_clone_and_prepare_preserve_resolved_terminal_contract():
    scene = _scene()

    for resolved_scene in (scene.clone(), prepare_scene(scene)):
        port = resolved_scene.ports[0]
        assert isinstance(port, mw.TerminalPort)
        assert port.positive == pytest.approx((0.5, 0.5, 0.7))
        assert port.negative == pytest.approx((0.5, 0.5, 0.3))
        assert tuple(port.current_surface.size) == pytest.approx((0.5, 0.5, 0.0))


def test_terminal_port_lowers_to_the_same_compiled_geometry_as_lumped_port():
    scene = _scene()
    terminal = scene.ports[0]
    explicit = mw.LumpedPort(
        name="feed",
        positive=terminal.positive,
        negative=terminal.negative,
        voltage_path=terminal.voltage_path,
        current_surface=terminal.current_surface,
        reference_plane=terminal.reference_plane,
        reference_impedance=terminal.reference_impedance,
    )

    compiled_terminal = compile_port_geometry(scene, terminal, device="cpu")
    compiled_explicit = compile_port_geometry(scene, explicit, device="cpu")

    assert compiled_terminal.port_name == compiled_explicit.port_name
    assert compiled_terminal.axis == compiled_explicit.axis
    assert compiled_terminal.direction == compiled_explicit.direction
    assert compiled_terminal.voltage_component == compiled_explicit.voltage_component
    assert torch.equal(compiled_terminal.voltage_indices, compiled_explicit.voltage_indices)
    assert torch.equal(compiled_terminal.voltage_weights, compiled_explicit.voltage_weights)
    assert compiled_terminal.current_components == compiled_explicit.current_components
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            compiled_terminal.current_indices,
            compiled_explicit.current_indices,
        )
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            compiled_terminal.current_weights,
            compiled_explicit.current_weights,
        )
    )


def test_terminal_port_uses_the_transverse_footprint_intersection():
    structures = (
        _pec_box("signal", position=(0.55, 0.5, 0.8), size=(0.6, 0.5, 0.2)),
        _pec_box("ground", position=(0.45, 0.5, 0.2), size=(0.6, 0.5, 0.2)),
    )
    scene = _scene(structures=structures)
    port = scene.ports[0]

    assert port.positive == pytest.approx((0.5, 0.5, 0.7))
    assert port.negative == pytest.approx((0.5, 0.5, 0.3))
    assert tuple(port.current_surface.position) == pytest.approx((0.5, 0.5, 0.45))
    assert tuple(port.current_surface.size) == pytest.approx((0.5, 0.5, 0.0))


def test_terminal_port_rejects_the_same_terminal_with_port_name():
    with pytest.raises(ValueError, match="TerminalPort 'feed'.*distinct"):
        _port(negative_terminal=mw.TerminalRef("signal"))


def test_terminal_port_rejects_missing_or_nonunique_structure_names():
    with pytest.raises(ValueError, match="TerminalPort 'feed'.*does not exist"):
        _scene(port=_port(positive_terminal=mw.TerminalRef("missing")))

    duplicate = _structures() + (
        _pec_box("signal", position=(0.5, 0.5, 0.9), size=(0.5, 0.5, 0.1)),
    )
    with pytest.raises(ValueError, match="TerminalPort 'feed'.*not unique"):
        _scene(structures=duplicate)


@pytest.mark.parametrize(
    ("structures", "message"),
    [
        (
            (
                mw.Structure(
                    geometry=mw.Box(position=(0.5, 0.5, 0.8), size=(0.5, 0.5, 0.2)),
                    material=mw.Material(eps_r=2.0),
                    name="signal",
                ),
                _pec_box("ground", position=(0.5, 0.5, 0.2)),
            ),
            "PEC",
        ),
        (
            (
                mw.Structure(
                    geometry=mw.Sphere(position=(0.5, 0.5, 0.8), radius=0.1),
                    material=mw.Material.pec(),
                    name="signal",
                ),
                _pec_box("ground", position=(0.5, 0.5, 0.2)),
            ),
            "Box",
        ),
        (
            (
                _pec_box(
                    "signal",
                    position=(0.5, 0.5, 0.8),
                    rotation=(0.9238795, 0.0, 0.0, 0.3826834),
                ),
                _pec_box("ground", position=(0.5, 0.5, 0.2)),
            ),
            "unrotated",
        ),
    ],
)
def test_terminal_port_rejects_unsupported_terminal_structures(structures, message):
    with pytest.raises(ValueError, match=rf"TerminalPort 'feed'.*{message}"):
        _scene(structures=structures)


def test_terminal_port_rejects_nonoverlapping_or_nonfacing_boxes():
    no_overlap = (
        _pec_box("signal", position=(0.9, 0.5, 0.8), size=(0.2, 0.5, 0.2)),
        _pec_box("ground", position=(0.5, 0.5, 0.2)),
    )
    with pytest.raises(ValueError, match="TerminalPort 'feed'.*do not overlap"):
        _scene(structures=no_overlap)

    not_facing = (
        _pec_box("signal", position=(0.5, 0.5, 0.25)),
        _pec_box("ground", position=(0.5, 0.5, 0.2)),
    )
    with pytest.raises(ValueError, match="TerminalPort 'feed'.*not distinct facing"):
        _scene(structures=not_facing)


def test_terminal_port_rejects_out_of_bounds_geometry_and_reference_plane():
    outside = (
        _pec_box("signal", position=(0.95, 0.5, 0.8), size=(0.2, 0.5, 0.2)),
        _pec_box("ground", position=(0.5, 0.5, 0.2)),
    )
    with pytest.raises(ValueError, match="TerminalPort 'feed'.*outside the Scene domain"):
        _scene(structures=outside)

    with pytest.raises(ValueError, match="TerminalPort 'feed'.*reference_plane"):
        _scene(port=_port(reference_plane=0.8))


@pytest.mark.parametrize(
    ("structures", "port", "message"),
    [
        (
            (
                _pec_box("signal", position=(0.5, 0.5, 0.82)),
                _pec_box("ground", position=(0.5, 0.5, 0.2)),
            ),
            _port(),
            "Yee z node grid",
        ),
        (_structures(), _port(reference_plane=0.44), "z half-grid plane"),
    ],
)
def test_terminal_port_rejects_off_grid_path_or_reference_plane(structures, port, message):
    scene = _scene(structures=structures, port=port)
    with pytest.raises(ValueError, match=rf"TerminalPort 'feed'.*{message}"):
        scene.compile_ports(device="cpu")
