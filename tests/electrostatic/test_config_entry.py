"""``Simulation.electrostatic/capacitance(scene, config=...)`` return a ``Simulation``.

CPU scenes; the frozen configs, the single entry, ``check()``/``prepare()`` and a
``SceneModule`` scene.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.electrostatic import CompiledElectrostatics

pytestmark = pytest.mark.host_validation


def _scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(0.1), electrostatic_terminals=[
            mw.electrostatic.ElectrostaticTerminal("a", geometry=mw.Box(position=(-0.25, 0, 0), size=(0.2, 0.2, 0.2)), potential=1.0),
            mw.electrostatic.ElectrostaticTerminal("b", geometry=mw.Box(position=(0.25, 0, 0), size=(0.2, 0.2, 0.2)), grounded=True),
        ])


@pytest.mark.parametrize("method, config_type", [("electrostatic", mw.electrostatic.ElectrostaticConfig),
                                                 ("capacitance", mw.electrostatic.CapacitanceConfig)])
def test_static_factories_return_a_simulation(method, config_type):
    simulation = getattr(mw.Simulation, method)(_scene(), config=config_type(device="cpu"))
    assert isinstance(simulation, mw.Simulation)
    assert simulation.method is mw.SimulationMethod(method)
    assert simulation.frequencies == () and simulation.frequency is None
    assert simulation.config == config_type(device="cpu")
    report = simulation.check()
    assert report.method == method and report.unsupported == ()
    prepared = simulation.prepare()
    assert isinstance(prepared.solver, CompiledElectrostatics)
    assert [terminal.name for terminal in prepared.solver.terminals] == ["a", "b"]
    assert prepared.run().method == method
    with pytest.raises(TypeError, match="config"):
        getattr(mw.Simulation, method)(_scene(), config=mw.FDTDConfig())
    with pytest.raises(TypeError, match="unexpected keyword argument 'solver'"):
        getattr(mw.Simulation, method)(_scene(), solver=mw.electrostatic.ElectrostaticSolverConfig())
    with pytest.raises(ValueError, match="take no frequencies"):
        mw.Simulation(scene=_scene(), method=method, frequencies=(1e9,))


def test_static_configs_are_frozen_and_validated():
    config = mw.electrostatic.CapacitanceConfig(terminals=["a", "b"], reference="b")
    assert config.terminals == ("a", "b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.reference = "a"
    with pytest.raises(TypeError, match="ElectrostaticBoundarySpec"):
        mw.electrostatic.ElectrostaticConfig(boundary=None)
    with pytest.raises(TypeError, match="ElectrostaticSolverConfig"):
        dataclasses.replace(config, solver=None)
    # A truncation estimate needs a Dirichlet enclosure; refused when the config is built.
    with pytest.raises(mw.SetupError, match="Dirichlet enclosure"):
        dataclasses.replace(config, boundary=mw.electrostatic.ElectrostaticBoundarySpec.neumann(),
                            truncation_estimate=mw.electrostatic.TruncationEstimate(padding_cells=2))


@pytest.mark.parametrize("build, message", [
    (lambda es: es.ElectrostaticSolverConfig(tolerance=0.0), "tolerance must be > 0"),
    (lambda es: es.ElectrostaticSolverConfig(max_iterations=0), "max_iterations must be > 0"),
    (lambda es: es.ElectrostaticSolverConfig(dtype=torch.complex64), "dtype must be torch.float32 or torch.float64"),
    (lambda es: es.TruncationEstimate(padding_cells=0), "padding_cells must be > 0"),
    (lambda es: es.ElectrostaticBoundarySpec(default="robin"), "boundary kind must be one of"),
    (lambda es: es.ElectrostaticBoundarySpec(x_low=("dirichlet", 1.0, 2.0)), "a kind or a \\(kind, value\\) pair"),
    (lambda es: es.ElectrostaticBoundarySpec(x_low=("neumann", 1.0)), "only Dirichlet boundaries carry a non-zero"),
], ids=["tolerance", "max_iterations", "dtype", "padding_cells", "kind", "pair", "neumann_value"])
def test_static_config_values_are_refused_with_setup_error(build, message):
    with pytest.raises(mw.SetupError, match=message):
        build(mw.electrostatic)


def test_static_simulation_accepts_a_scene_module():
    class Plates(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.eps = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))

        def to_scene(self):
            scene = _scene()
            scene = scene.updated(structures=(
                *scene.structures,
                mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(0.2, 0.6, 0.6)),
                             material=mw.Material(eps_r=self.eps)),
            ))
            return scene

    module = Plates()
    capacitance = mw.electrostatic.capacitance(mw.Simulation.capacitance(module, config=mw.electrostatic.CapacitanceConfig(device="cpu")).run())
    (gradient,) = torch.autograd.grad(capacitance.matrix[0, 0], module.eps)
    assert torch.isfinite(gradient) and gradient > 0
