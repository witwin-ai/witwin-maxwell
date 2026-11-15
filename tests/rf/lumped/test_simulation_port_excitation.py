import pytest
import torch

import witwin.maxwell as mw


def _scene(*, ports=()):
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        ports=ports,
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
    excitation = mw.rf.PortExcitation("p1")
    simulation = mw.Simulation.fdtd(
        _scene(ports=(_lumped_port(),)),
        frequencies=1.0e9,
        excitations=excitation,
    )

    assert simulation.excitations == (excitation,)


def test_fdtd_rejects_duplicate_but_accepts_distinct_lumped_port_excitations():
    with pytest.raises(ValueError, match="at most once"):
        mw.Simulation.fdtd(
            _scene(ports=(_lumped_port(),)),
            frequencies=1.0e9,
            excitations=(mw.rf.PortExcitation("p1"), mw.rf.PortExcitation("p1")),
        )

    simulation = mw.Simulation.fdtd(
        _scene(ports=(_lumped_port("p1"), _lumped_port("p2"))),
        frequencies=1.0e9,
        excitations=(mw.rf.PortExcitation("p1"), mw.rf.PortExcitation("p2")),
    )

    assert tuple(excitation.port_name for excitation in simulation.excitations) == (
        "p1",
        "p2",
    )


def test_prepare_reports_missing_excitation_port_before_backend_initialization():
    simulation = mw.Simulation.fdtd(
        _scene(),
        frequencies=1.0e9,
        excitations=mw.rf.PortExcitation("missing"),
    )

    with pytest.raises(ValueError, match="missing port 'missing'"):
        simulation.prepare()


def test_fdfd_rejects_port_excitation():
    with pytest.raises(ValueError, match="Simulation.fdtd"):
        mw.Simulation(
            scene=_scene(ports=(_lumped_port(),)),
            method=mw.SimulationMethod.FDFD,
            frequencies=(1.0e9,),
            excitations=mw.rf.PortExcitation("p1"),
        )


def test_trainable_scene_accepts_lumped_port_for_adjoint_runtime():
    class TrainablePortScene(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.design = torch.nn.Parameter(torch.tensor(1.0))

        def to_scene(self):
            return _scene(ports=(_lumped_port(),))

    simulation = mw.Simulation.fdtd(
        TrainablePortScene(),
        frequencies=1.0e9,
        excitations=mw.rf.PortExcitation("p1"),
    )

    assert simulation.has_trainable_parameters
    assert simulation.excitations[0].port_name == "p1"
