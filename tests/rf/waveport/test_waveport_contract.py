from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.waveports import (
    CompiledWaveModeSpec,
    CompiledWavePortCrossSection,
    compile_waveport_cross_section,
)
from witwin.maxwell.scene import prepare_scene


def _voltage_path(*, plane=0.4):
    return ((0.3, 0.5, plane), (0.7, 0.5, plane))


def _current_contour(*, plane=0.45, size=(0.5, 0.5, 0.0), rotation=None):
    return mw.Box(
        position=(0.5, 0.5, plane),
        size=size,
        rotation=rotation,
    )


def _tem_mode(**overrides):
    values = {
        "family": "tem",
        "voltage_path": _voltage_path(),
        "current_contour": _current_contour(),
    }
    values.update(overrides)
    return mw.WaveModeSpec(**values)


def _port(**overrides):
    values = {
        "name": "wp1",
        "position": (0.5, 0.5, 0.4),
        "size": (0.6, 0.6, 0.0),
        "direction": "+",
        "reference_plane": 0.4,
        "modes": (_tem_mode(), mw.WaveModeSpec("te", mode_index=1)),
    }
    values.update(overrides)
    return mw.WavePort(**values)


def _scene(*, ports=None):
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        ports=(_port(),) if ports is None else ports,
        device="cpu",
    )


def test_wave_mode_impedance_contracts_and_stable_default_names():
    tem = _tem_mode()
    te = mw.WaveModeSpec("TE", mode_index=2)
    tm = mw.WaveModeSpec("tm", mode_index=3)
    hybrid_voltage = mw.WaveModeSpec(
        "hybrid",
        name="odd",
        impedance_definition="power_voltage",
        voltage_path=_voltage_path(),
    )
    hybrid_current = mw.WaveModeSpec(
        "hybrid",
        impedance_definition="power_current",
        current_contour=_current_contour(),
    )

    assert (tem.name, tem.impedance_definition, tem.impedance_formula) == (
        "TEM0",
        "voltage_current",
        "V/I",
    )
    assert (te.name, te.impedance_definition, te.impedance_formula) == (
        "TE2",
        "te_wave",
        "omega*mu/beta",
    )
    assert te.polarization == "auto"
    assert (tm.name, tm.impedance_definition, tm.impedance_formula) == (
        "TM3",
        "tm_wave",
        "beta/(omega*epsilon)",
    )
    assert hybrid_voltage.name == "odd"
    assert hybrid_voltage.impedance_formula == "abs(V)**2/(2*P)"
    assert hybrid_current.name == "HYBRID0"
    assert hybrid_current.impedance_formula == "2*P/abs(I)**2"


def test_wave_port_freezes_explicit_direction_reference_plane_and_mode_names():
    port = _port()

    assert port.kind == "wave_port"
    assert port.normal_axis == "z"
    assert port.direction == "+"
    assert port.direction_sign == 1
    assert port.reference_plane == 0.4
    assert port.mode_name(port.modes[0]) == "wp1::TEM0"
    assert port.mode_name(port.modes[1]) == "wp1::TE1"
    assert port.phasor_convention == "peak"
    assert port.power_convention == "0.5*Re(V*conj(I))"


def test_scene_accepts_clones_and_prepares_waveports_without_expanding_modeport():
    mode_port = mw.ModePort("optical", source_time=None)
    scene = _scene(ports=(mode_port, _port()))

    assert scene.resolved_sources() == []
    assert [monitor.name for monitor in scene.resolved_monitors()] == ["optical"]
    for resolved_scene in (scene.clone(), prepare_scene(scene)):
        assert isinstance(resolved_scene.ports[0], mw.ModePort)
        assert isinstance(resolved_scene.ports[1], mw.WavePort)
        assert resolved_scene.ports[1].modes[0].name == "TEM0"


def test_compile_waveport_cross_section_and_tem_geometry_on_cpu():
    scene = _scene()
    compiled = scene.compile_waveports(device="cpu")[0]

    assert isinstance(compiled, CompiledWavePortCrossSection)
    assert compiled.port_name == "wp1"
    assert compiled.normal_axis == "z"
    assert compiled.tangential_axes == ("x", "y")
    assert compiled.direction_sign == 1
    assert compiled.reference_plane == 0.4
    assert compiled.electric_plane_index == 4
    assert compiled.aperture_lower_indices == (2, 2, 4)
    assert compiled.aperture_upper_indices == (8, 8, 4)
    assert compiled.transverse_shape == (7, 7)
    assert compiled.aperture_indices.shape == (7, 7, 3)
    assert compiled.device.type == "cpu"

    tem, te = compiled.modes
    assert isinstance(tem, CompiledWaveModeSpec)
    assert tem.tracking_id == "wp1::TEM0"
    assert tem.voltage_component == "Ex"
    assert tem.voltage_direction == 1
    assert tem.voltage_indices.shape == (4, 3)
    assert torch.allclose(tem.voltage_weights, torch.full((4,), 0.1, dtype=torch.float64))
    assert tem.current_components == ("Hx", "Hy", "Hx", "Hy")
    assert tem.current_plane_index == 4
    assert tem.current_plane_coordinate == pytest.approx(0.45)
    assert all(indices.device.type == "cpu" for indices in tem.current_indices)
    assert te.tracking_id == "wp1::TE1"
    assert te.voltage_indices is None
    assert te.current_indices == ()


@pytest.mark.parametrize(
    ("position", "size", "path", "contour", "normal", "component", "current_components"),
    [
        (
            (0.4, 0.5, 0.5),
            (0.0, 0.6, 0.6),
            ((0.4, 0.3, 0.5), (0.4, 0.7, 0.5)),
            mw.Box(position=(0.45, 0.5, 0.5), size=(0.0, 0.5, 0.5)),
            "x",
            "Ey",
            ("Hy", "Hz", "Hy", "Hz"),
        ),
        (
            (0.5, 0.4, 0.5),
            (0.6, 0.0, 0.6),
            ((0.5, 0.4, 0.3), (0.5, 0.4, 0.7)),
            mw.Box(position=(0.5, 0.45, 0.5), size=(0.5, 0.0, 0.5)),
            "y",
            "Ez",
            ("Hz", "Hx", "Hz", "Hx"),
        ),
    ],
)
def test_cross_section_compiler_is_axis_generic(
    position,
    size,
    path,
    contour,
    normal,
    component,
    current_components,
):
    mode = mw.WaveModeSpec(
        "tem",
        voltage_path=path,
        current_contour=contour,
    )
    axis_index = "xyz".index(normal)
    port = mw.WavePort(
        "axis-port",
        position,
        size,
        "+",
        position[axis_index],
        (mode,),
    )

    compiled = compile_waveport_cross_section(_scene(ports=()), port, device="cpu")

    assert compiled.normal_axis == normal
    assert compiled.modes[0].voltage_component == component
    assert compiled.modes[0].current_components == current_components


def test_compiler_preserves_all_impedance_definitions_without_solving_modes():
    modes = (
        mw.WaveModeSpec("te"),
        mw.WaveModeSpec("tm"),
        mw.WaveModeSpec(
            "hybrid",
            name="hv",
            impedance_definition="power_voltage",
            voltage_path=_voltage_path(),
        ),
        mw.WaveModeSpec(
            "hybrid",
            name="hi",
            impedance_definition="power_current",
            current_contour=_current_contour(),
        ),
        mw.WaveModeSpec(
            "hybrid",
            name="hvi",
            impedance_definition="voltage_current",
            voltage_path=_voltage_path(),
            current_contour=_current_contour(),
        ),
    )
    compiled = compile_waveport_cross_section(
        _scene(ports=()),
        _port(modes=modes),
        device="cpu",
    )

    assert tuple(mode.impedance_definition for mode in compiled.modes) == (
        "te_wave",
        "tm_wave",
        "power_voltage",
        "power_current",
        "voltage_current",
    )
    assert tuple(mode.tracking_id for mode in compiled.modes) == (
        "wp1::TE0",
        "wp1::TM0",
        "wp1::hv",
        "wp1::hi",
        "wp1::hvi",
    )


def test_wave_mode_polarization_uses_the_existing_tangential_axis_contract():
    auto_port = mw.WavePort(
        "auto-pol",
        (0.4, 0.5, 0.5),
        (0.0, 0.8, 0.6),
        "+",
        0.4,
        (mw.WaveModeSpec("te"),),
    )
    explicit_port = mw.WavePort(
        "explicit-pol",
        (0.4, 0.5, 0.5),
        (0.0, 0.8, 0.6),
        "+",
        0.4,
        (mw.WaveModeSpec("te", polarization="Ez"),),
    )

    auto = compile_waveport_cross_section(_scene(ports=()), auto_port, device="cpu").modes[0]
    explicit = compile_waveport_cross_section(
        _scene(ports=()),
        explicit_port,
        device="cpu",
    ).modes[0]

    assert auto.polarization_axis == "y"
    assert auto.polarization == (0.0, 1.0, 0.0)
    assert explicit.polarization_axis == "z"
    assert explicit.polarization == (0.0, 0.0, 1.0)


def test_wave_mode_rejects_invalid_or_normal_polarization():
    with pytest.raises(ValueError, match="polarization"):
        mw.WaveModeSpec("te", polarization="diagonal")
    with pytest.raises(ValueError, match="axis-aligned"):
        mw.WaveModeSpec("te", polarization=(1.0, 1.0, 0.0))

    with pytest.raises(ValueError, match="WavePort 'normal-pol'.*tangential"):
        mw.WavePort(
            "normal-pol",
            (0.4, 0.5, 0.5),
            (0.0, 0.8, 0.6),
            "+",
            0.4,
            (mw.WaveModeSpec("te", polarization="Ex"),),
        )


def test_negative_direction_uses_the_preceding_magnetic_half_plane_and_orientation():
    negative_tem = _tem_mode(current_contour=_current_contour(plane=0.35))
    positive = compile_waveport_cross_section(_scene(), _port(), device="cpu")
    negative = compile_waveport_cross_section(
        _scene(ports=()),
        _port(direction="-", modes=(negative_tem,)),
        device="cpu",
    )

    assert negative.direction_sign == -1
    assert negative.modes[0].current_plane_index == 3
    assert negative.modes[0].current_plane_coordinate == pytest.approx(0.35)
    for positive_weight, negative_weight in zip(
        positive.modes[0].current_weights,
        negative.modes[0].current_weights,
    ):
        assert torch.equal(negative_weight, -positive_weight)


@pytest.mark.parametrize("family", ["", "quasi-tem", 3])
def test_wave_mode_rejects_unknown_families(family):
    with pytest.raises(ValueError, match="family"):
        mw.WaveModeSpec(family)


def test_wave_mode_rejects_invalid_index_or_name():
    with pytest.raises(ValueError, match="mode_index"):
        mw.WaveModeSpec("te", -1)
    with pytest.raises(ValueError, match="name"):
        mw.WaveModeSpec("te", name="")
    with pytest.raises(ValueError, match="reserved"):
        mw.WaveModeSpec("te", name="wp::te0")


def test_tem_and_hybrid_require_the_geometry_used_by_their_impedance_definition():
    with pytest.raises(ValueError, match="TEM requires both"):
        mw.WaveModeSpec("tem")
    with pytest.raises(ValueError, match="voltage_current"):
        _tem_mode(impedance_definition="power_voltage")
    with pytest.raises(ValueError, match="explicitly"):
        mw.WaveModeSpec("hybrid")
    with pytest.raises(ValueError, match="requires voltage_path"):
        mw.WaveModeSpec("hybrid", impedance_definition="power_voltage")
    with pytest.raises(ValueError, match="requires current_contour"):
        mw.WaveModeSpec("hybrid", impedance_definition="power_current")


@pytest.mark.parametrize(
    ("family", "impedance_definition"),
    [("te", "tm_wave"), ("tm", "te_wave")],
)
def test_te_tm_reject_wrong_impedance_contracts(family, impedance_definition):
    with pytest.raises(ValueError, match="impedance_definition"):
        mw.WaveModeSpec(family, impedance_definition=impedance_definition)
    with pytest.raises(ValueError, match="does not accept"):
        mw.WaveModeSpec(family, voltage_path=_voltage_path())


def test_wave_mode_rejects_invalid_voltage_path_or_current_contour():
    with pytest.raises(ValueError, match="axis-aligned"):
        mw.WaveModeSpec(
            "tem",
            voltage_path=((0.3, 0.3, 0.4), (0.7, 0.7, 0.4)),
            current_contour=_current_contour(),
        )
    with pytest.raises(TypeError, match="current_contour"):
        mw.WaveModeSpec(
            "tem",
            voltage_path=_voltage_path(),
            current_contour="loop",
        )


@pytest.mark.parametrize(
    "size",
    [
        (0.6, 0.6, 0.6),
        (0.6, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (-0.1, 0.6, 0.0),
        (float("nan"), 0.6, 0.0),
    ],
)
def test_wave_port_requires_one_zero_and_two_positive_finite_size_components(size):
    with pytest.raises(ValueError, match="WavePort 'wp1'.*size"):
        _port(size=size)


def test_wave_port_rejects_direction_reference_plane_modes_and_duplicate_names():
    with pytest.raises(ValueError, match="WavePort 'wp1'.*direction"):
        _port(direction="forward")
    with pytest.raises(ValueError, match="WavePort 'wp1'.*reference_plane"):
        _port(reference_plane=0.5)
    with pytest.raises(ValueError, match="WavePort 'wp1'.*modes"):
        _port(modes=())
    with pytest.raises(TypeError, match="WavePort 'wp1'.*WaveModeSpec"):
        _port(modes=(object(),))
    with pytest.raises(TypeError, match="WavePort 'wp1'.*iterable"):
        _port(modes=None)
    duplicate = mw.WaveModeSpec("te")
    with pytest.raises(ValueError, match="WavePort 'wp1'.*unique"):
        _port(modes=(duplicate, duplicate))


@pytest.mark.parametrize(
    ("port", "message"),
    [
        (_port(position=(0.8, 0.5, 0.4)), "outside the Scene domain"),
        (_port(position=(0.5, 0.5, 0.44), reference_plane=0.44), "aperture plane"),
        (_port(size=(0.5, 0.6, 0.0)), "aperture lower bound"),
    ],
)
def test_compiler_rejects_out_of_domain_or_off_grid_apertures(port, message):
    with pytest.raises(ValueError, match=rf"WavePort 'wp1'.*{message}"):
        compile_waveport_cross_section(_scene(ports=()), port, device="cpu")


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (((0.3, 0.5, 0.5), (0.7, 0.5, 0.5)), "aperture plane"),
        (((0.1, 0.5, 0.4), (0.7, 0.5, 0.4)), "outside the aperture"),
        (((0.33, 0.5, 0.4), (0.7, 0.5, 0.4)), "Yee x node grid"),
        (((0.5, 0.5, 0.3), (0.5, 0.5, 0.5)), "aperture plane"),
    ],
)
def test_compiler_rejects_invalid_tem_voltage_path_geometry(path, message):
    mode = _tem_mode(voltage_path=path)
    port = _port(modes=(mode,))
    with pytest.raises(ValueError, match=rf"WavePort 'wp1'.*{message}"):
        compile_waveport_cross_section(_scene(ports=()), port, device="cpu")


@pytest.mark.parametrize(
    ("contour", "message"),
    [
        (
            _current_contour(rotation=(0.9238795, 0.0, 0.0, 0.3826834)),
            "unrotated",
        ),
        (_current_contour(size=(0.5, 0.5, 0.1)), "planar"),
        (_current_contour(size=(0.0, 0.5, 0.5)), "normal"),
        (_current_contour(size=(0.7, 0.5, 0.0)), "outside the aperture"),
        (_current_contour(size=(0.4, 0.4, 0.0)), "half-grid boundary"),
        (_current_contour(plane=0.55), "direction-adjacent"),
        (_current_contour(plane=0.44), "current contour plane"),
    ],
)
def test_compiler_rejects_invalid_tem_current_contour_geometry(contour, message):
    mode = _tem_mode(current_contour=contour)
    port = _port(modes=(mode,))
    with pytest.raises(ValueError, match=rf"WavePort 'wp1'.*{message}"):
        compile_waveport_cross_section(_scene(ports=()), port, device="cpu")


def test_direction_at_grid_edge_requires_an_adjacent_magnetic_plane():
    mode = _tem_mode(
        voltage_path=_voltage_path(plane=1.0),
        current_contour=_current_contour(plane=0.95),
    )
    port = _port(
        position=(0.5, 0.5, 1.0),
        reference_plane=1.0,
        modes=(mode,),
    )
    with pytest.raises(ValueError, match="no adjacent magnetic half-grid"):
        compile_waveport_cross_section(_scene(ports=()), port, device="cpu")
