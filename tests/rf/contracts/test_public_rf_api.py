import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def _lumped_port(*, name="feed", reference_impedance=50.0):
    return mw.LumpedPort(
        name=name,
        negative=(0.5, 0.5, 0.2),
        positive=(0.5, 0.5, 0.8),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.5, 0.5, 0.45),
            size=(0.5, 0.5, 0.0),
        ),
        reference_impedance=reference_impedance,
        reference_plane=0.45,
    )


def test_phase_zero_rf_types_are_top_level_public_api():
    assert mw.AxisPath.__module__ == "witwin.maxwell.ports"
    assert mw.LumpedPort.__module__ == "witwin.maxwell.ports"
    assert mw.PortData.__module__ == "witwin.maxwell.network"
    assert mw.NetworkData.__module__ == "witwin.maxwell.network"


def test_scene_keeps_lumped_ports_declarative_and_compiles_through_scene_helper():
    port = _lumped_port()
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        ports=(port,),
        device="cpu",
    )

    assert scene.ports == [port]
    assert scene.resolved_sources() == []
    assert scene.resolved_monitors() == []
    compiled = scene.compile_ports()
    assert len(compiled) == 1
    assert compiled[0].port_name == "feed"
    assert compiled[0].reference_plane == pytest.approx(0.45)


def test_scene_rejects_duplicate_and_unknown_port_objects():
    scene = mw.Scene(device="cpu")
    scene.add_port(_lumped_port())

    with pytest.raises(ValueError, match="already present"):
        scene.add_port(_lumped_port())
    with pytest.raises(TypeError, match="ModePort or LumpedPort"):
        scene.add_port(object())


def test_scene_rejects_lumped_port_terminals_outside_domain_early():
    port = mw.LumpedPort(
        name="outside",
        negative=(0.5, 0.5, 0.2),
        positive=(0.5, 0.5, 1.2),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.5, 0.5, 0.45),
            size=(0.5, 0.5, 0.0),
        ),
    )

    with pytest.raises(ValueError, match="LumpedPort 'outside'.*positive terminal.*outside"):
        mw.Scene(
            domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            grid=mw.GridSpec.uniform(0.1),
            ports=(port,),
            device="cpu",
        )


def test_scene_rejects_nonfinite_lumped_port_terminals_early():
    port = mw.LumpedPort(
        name="nonfinite",
        negative=(0.5, 0.5, 0.2),
        positive=(0.5, 0.5, float("nan")),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.5, 0.5, 0.45),
            size=(0.5, 0.5, 0.0),
        ),
    )

    with pytest.raises(ValueError, match="LumpedPort 'nonfinite'.*positive terminal"):
        mw.Scene(
            domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            grid=mw.GridSpec.uniform(0.1),
            ports=(port,),
            device="cpu",
        )


def test_lumped_port_preserves_trainable_reference_impedance_tensor():
    z0 = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    port = _lumped_port(reference_impedance=z0)

    assert port.reference_impedance is z0
    compiled = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            grid=mw.GridSpec.uniform(0.1),
            ports=(port,),
            device="cpu",
        )
    ).compile_ports()
    assert compiled[0].reference_impedance is z0


def test_public_scene_has_no_second_solver_entrypoint():
    scene = mw.Scene(device="cpu")
    assert not hasattr(mw, "FDTD")
    assert not hasattr(mw, "FDFD")
    assert hasattr(scene, "compile_ports")
