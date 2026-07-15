import pytest
import torch

import witwin.maxwell as mw


def _scene(*, ports=()):
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        ports=ports,
        device="cpu",
    )


def _lumped_port(name="p1"):
    return mw.LumpedPort(
        name=name,
        positive=(0.5, 0.5, 0.75),
        negative=(0.5, 0.5, 0.25),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.5, 0.5, 0.5),
            size=(0.5, 0.5, 0.0),
        ),
    )


def test_fdtd_accepts_one_explicit_port_excitation():
    excitation = mw.PortExcitation("p1")
    simulation = mw.Simulation.fdtd(
        _scene(ports=(_lumped_port(),)),
        frequency=1.0e9,
        excitations=excitation,
    )

    assert simulation.excitations == (excitation,)


def test_fdtd_rejects_duplicate_or_multiple_active_ports_in_phase_one():
    with pytest.raises(ValueError, match="at most once"):
        mw.Simulation.fdtd(
            _scene(ports=(_lumped_port(),)),
            frequency=1.0e9,
            excitations=(mw.PortExcitation("p1"), mw.PortExcitation("p1")),
        )

    with pytest.raises(NotImplementedError, match="one active RF port"):
        mw.Simulation.fdtd(
            _scene(ports=(_lumped_port("p1"), _lumped_port("p2"))),
            frequency=1.0e9,
            excitations=(mw.PortExcitation("p1"), mw.PortExcitation("p2")),
        )


def test_prepare_reports_missing_excitation_port_before_backend_initialization():
    simulation = mw.Simulation.fdtd(
        _scene(),
        frequency=1.0e9,
        excitations=mw.PortExcitation("missing"),
    )

    with pytest.raises(ValueError, match="missing port 'missing'"):
        simulation.prepare()


def test_fdfd_rejects_port_excitation():
    with pytest.raises(ValueError, match="Simulation.fdtd"):
        mw.Simulation(
            scene=_scene(ports=(_lumped_port(),)),
            method=mw.SimulationMethod.FDFD,
            frequencies=(1.0e9,),
            excitations=mw.PortExcitation("p1"),
        )


def test_trainable_scene_rejects_lumped_port_until_adjoint_replay_is_supported():
    class TrainablePortScene(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.design = torch.nn.Parameter(torch.tensor(1.0))

        def to_scene(self):
            return _scene(ports=(_lumped_port(),))

    with pytest.raises(NotImplementedError, match="does not replay"):
        mw.Simulation.fdtd(
            TrainablePortScene(),
            frequency=1.0e9,
            excitations=mw.PortExcitation("p1"),
        )
