"""Public API surface and fail-closed guards for the electrostatic solver."""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box, Sphere

from .oracles import OutsideSphere

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="electrostatics is GPU-first")


def test_public_exports_present():
    for name in (
        "ElectrostaticTerminal",
        "ChargeDensity",
        "ElectrostaticBoundarySpec",
        "ElectrostaticSolverConfig",
        "ElectrostaticResultData",
    ):
        assert hasattr(mw, name)
        assert name in mw.__all__


def test_terminal_potential_charge_mutually_exclusive():
    with pytest.raises(ValueError):
        mw.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(1, 1, 1)), potential=1.0, charge=1e-9)


def test_terminal_requires_a_constraint():
    with pytest.raises(ValueError):
        mw.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(1, 1, 1)))


def test_terminal_grounded_is_zero_volts():
    t = mw.ElectrostaticTerminal(name="g", geometry=Box(position=(0, 0, 0), size=(1, 1, 1)), grounded=True)
    assert t.potential == 0.0
    assert not t.is_floating


def test_terminal_geometry_or_structure_exclusive():
    box = Box(position=(0, 0, 0), size=(1, 1, 1))
    with pytest.raises(ValueError):
        mw.ElectrostaticTerminal(name="t", geometry=box, structure=mw.Structure(geometry=box, material=mw.Material()), potential=1.0)


def test_boundary_spec_faces_and_helpers():
    spec = mw.ElectrostaticBoundarySpec(default="neumann", z_low=("dirichlet", 2.0))
    assert spec.face("z", "low") == ("dirichlet", 2.0)
    assert spec.face("z", "high") == ("neumann", 0.0)
    assert spec.has_dirichlet
    assert mw.ElectrostaticBoundarySpec.grounded_box().face("x", "low") == ("dirichlet", 0.0)
    assert not mw.ElectrostaticBoundarySpec.neumann().has_dirichlet


def test_solver_config_validation():
    with pytest.raises(ValueError):
        mw.ElectrostaticSolverConfig(tolerance=0.0)
    with pytest.raises(ValueError):
        mw.ElectrostaticSolverConfig(max_iterations=0)


def test_scene_terminal_collection_is_separate_from_ports():
    scene = mw.Scene(device="cpu")
    t = mw.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), potential=1.0)
    scene.add_electrostatic_terminal(t)
    assert scene.electrostatic_terminals == [t]
    assert scene.ports == []
    # Duplicate names rejected.
    with pytest.raises(ValueError):
        scene.add_electrostatic_terminal(
            mw.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), potential=0.0)
        )


def test_scene_clone_preserves_electrostatic_collections():
    scene = mw.Scene(device="cpu")
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), potential=1.0)
    )
    scene.add_charge_density(mw.ChargeDensity(geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), density=1.0))
    clone = scene.clone()
    assert len(clone.electrostatic_terminals) == 1
    assert len(clone.charge_densities) == 1


def _grounded_shell_scene(n=24):
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=0.3), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="outer", geometry=OutsideSphere(0.8), grounded=True)
    )
    return scene


@gpu
def test_result_accessor_and_method():
    scene = _grounded_shell_scene()
    result = mw.Simulation.electrostatic(scene, boundary=mw.ElectrostaticBoundarySpec.neumann()).run()
    assert result.method == "electrostatic"
    es = result.electrostatic
    assert es.potential.shape == es.epsilon_r.shape
    assert es.E.shape[-1] == 3
    assert set(es.terminal_names) == {"inner", "outer"}
    with pytest.raises(KeyError):
        es.terminal_charge("missing")
    assert result.solver_stats["iterations"] == es.iterations


@gpu
def test_pml_boundary_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.pml(num_layers=4))
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=0.3), potential=1.0)
    )
    with pytest.raises(NotImplementedError):
        mw.Simulation.electrostatic(scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box()).run()


@gpu
def test_floating_charge_terminal_rejected_in_this_stage():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="float", geometry=Sphere(position=(0, 0, 0), radius=0.3), charge=1e-12)
    )
    with pytest.raises(NotImplementedError):
        mw.Simulation.electrostatic(scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box()).run()


@gpu
def test_pure_neumann_without_fixed_potential_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_charge_density(mw.ChargeDensity(geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), density=1.0))
    with pytest.raises(NotImplementedError):
        mw.Simulation.electrostatic(scene, boundary=mw.ElectrostaticBoundarySpec.neumann()).run()


@gpu
def test_thin_conductor_swallowed_by_grid_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 8)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    # A zero-thickness plate lands between cell centres and covers no cell.
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="plate", geometry=Box(position=(0.05, 0, 0), size=(0.0, 1.0, 1.0)), potential=1.0)
    )
    with pytest.raises(ValueError):
        mw.Simulation.electrostatic(scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box()).run()


@gpu
def test_overlapping_terminals_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="a", geometry=Sphere(position=(0, 0, 0), radius=0.4), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="b", geometry=Sphere(position=(0.1, 0, 0), radius=0.4), grounded=True)
    )
    with pytest.raises(ValueError):
        mw.Simulation.electrostatic(scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box()).run()


@gpu
def test_dispersive_material_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    material = mw.Material(
        eps_r=2.0,
        lorentz_poles=[mw.LorentzPole(delta_eps=1.0, resonance_frequency=1e14, gamma=1e12)],
    )
    scene.add_structure(mw.Structure(geometry=Box(position=(0, 0, 0), size=(1, 1, 1)), material=material))
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=0.2), potential=1.0)
    )
    with pytest.raises(NotImplementedError):
        mw.Simulation.electrostatic(scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box()).run()
