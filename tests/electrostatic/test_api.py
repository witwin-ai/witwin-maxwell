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
        "CapacitanceData",
    ):
        assert hasattr(mw.electrostatic, name)
        assert name in mw.electrostatic.__all__


def test_terminal_potential_charge_mutually_exclusive():
    with pytest.raises(ValueError):
        mw.electrostatic.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(1, 1, 1)), potential=1.0, charge=1e-9)


def test_terminal_requires_a_constraint():
    with pytest.raises(ValueError):
        mw.electrostatic.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(1, 1, 1)))


def test_terminal_grounded_is_zero_volts():
    t = mw.electrostatic.ElectrostaticTerminal(name="g", geometry=Box(position=(0, 0, 0), size=(1, 1, 1)), grounded=True)
    assert t.potential == 0.0
    assert not t.is_floating


def test_terminal_geometry_or_structure_exclusive():
    box = Box(position=(0, 0, 0), size=(1, 1, 1))
    with pytest.raises(ValueError):
        mw.electrostatic.ElectrostaticTerminal(name="t", geometry=box, structure=mw.Structure(geometry=box, material=mw.Material()), potential=1.0)


def test_boundary_spec_faces_and_helpers():
    spec = mw.electrostatic.ElectrostaticBoundarySpec(default="neumann", z_low=("dirichlet", 2.0))
    assert spec.face("z", "low") == ("dirichlet", 2.0)
    assert spec.face("z", "high") == ("neumann", 0.0)
    assert spec.has_dirichlet
    assert mw.electrostatic.ElectrostaticBoundarySpec.grounded_box().face("x", "low") == ("dirichlet", 0.0)
    assert not mw.electrostatic.ElectrostaticBoundarySpec.neumann().has_dirichlet


def test_solver_config_validation():
    with pytest.raises(ValueError):
        mw.electrostatic.ElectrostaticSolverConfig(tolerance=0.0)
    with pytest.raises(ValueError):
        mw.electrostatic.ElectrostaticSolverConfig(max_iterations=0)


def test_scene_terminal_collection_is_separate_from_ports():
    t = mw.electrostatic.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), potential=1.0)
    scene = mw.Scene(electrostatic_terminals=(t,))
    assert scene.electrostatic_terminals == (t,)
    assert scene.ports == ()
    # Duplicate names rejected.
    with pytest.raises(ValueError):
        scene = scene.updated(electrostatic_terminals=(
            *scene.electrostatic_terminals,
            mw.electrostatic.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), potential=0.0),
        ))


def test_scene_clone_preserves_electrostatic_collections():
    scene = mw.Scene(electrostatic_terminals=(
                mw.electrostatic.ElectrostaticTerminal(name="t", geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), potential=1.0),
            ), charge_densities=(
                mw.electrostatic.ChargeDensity(geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), density=1.0),
            ))
    clone = scene
    assert len(clone.electrostatic_terminals) == 1
    assert len(clone.charge_densities) == 1


def _grounded_shell_scene(n=24):
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none(), electrostatic_terminals=(
                mw.electrostatic.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=0.3), potential=1.0),
                mw.electrostatic.ElectrostaticTerminal(name="outer", geometry=OutsideSphere(0.8), grounded=True),
            ))
    return scene


@gpu
def test_result_accessor_and_method():
    scene = _grounded_shell_scene()
    result = mw.Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(boundary=mw.electrostatic.ElectrostaticBoundarySpec.neumann(), device="cpu")).run()
    assert result.method == "electrostatic"
    es = mw.electrostatic.electrostatic(result)
    assert es.potential.shape == es.epsilon_r.shape
    assert es.E.shape[-1] == 3
    assert set(es.terminal_names) == {"inner", "outer"}
    with pytest.raises(KeyError):
        es.terminal_charge("missing")
    assert result.stats()["iterations"] == es.iterations


@gpu
def test_pml_boundary_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.pml(num_layers=4), electrostatic_terminals=(
                mw.electrostatic.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=0.3), potential=1.0),
            ))
    with pytest.raises(NotImplementedError):
        mw.Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(boundary=mw.electrostatic.ElectrostaticBoundarySpec.grounded_box(), device="cpu")).run()


@gpu
def test_material_region_rejected():
    """A MaterialRegion (RF design-region object) must fail closed rather than
    silently solving with eps_r=1 inside the declared region."""
    n = 16
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none(), electrostatic_terminals=(
                mw.electrostatic.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=0.3), potential=1.0),
            ), material_regions=(
                mw.MaterialRegion(
            name="design",
            geometry=Box(position=(0, 0, 0), size=(1.0, 1.0, 1.0)),
            density=torch.full((n, n, n), 0.5),
            eps_bounds=(1.0, 4.0),
            ),
            ))
    with pytest.raises(NotImplementedError):
        mw.Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(boundary=mw.electrostatic.ElectrostaticBoundarySpec.grounded_box(), device="cpu")).run()


@gpu
def test_pure_neumann_without_fixed_potential_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none(), charge_densities=(
                mw.electrostatic.ChargeDensity(geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), density=1.0),
            ))
    with pytest.raises(NotImplementedError):
        mw.Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(boundary=mw.electrostatic.ElectrostaticBoundarySpec.neumann(), device="cpu")).run()


@gpu
def test_thin_conductor_swallowed_by_grid_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 8)
    # A zero-thickness plate lands between cell centres and covers no cell.
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none(), electrostatic_terminals=(
                mw.electrostatic.ElectrostaticTerminal(name="plate", geometry=Box(position=(0.05, 0, 0), size=(0.0, 1.0, 1.0)), potential=1.0),
            ))
    with pytest.raises(ValueError):
        mw.Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(boundary=mw.electrostatic.ElectrostaticBoundarySpec.grounded_box(), device="cpu")).run()


@gpu
def test_overlapping_terminals_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none(), electrostatic_terminals=(
                mw.electrostatic.ElectrostaticTerminal(name="a", geometry=Sphere(position=(0, 0, 0), radius=0.4), potential=1.0),
                mw.electrostatic.ElectrostaticTerminal(name="b", geometry=Sphere(position=(0.1, 0, 0), radius=0.4), grounded=True),
            ))
    with pytest.raises(ValueError):
        mw.Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(boundary=mw.electrostatic.ElectrostaticBoundarySpec.grounded_box(), device="cpu")).run()


@gpu
def test_dispersive_material_rejected():
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    material = mw.Material(
        eps_r=2.0,
        lorentz_poles=[mw.LorentzPole(delta_eps=1.0, resonance_frequency=1e14, gamma=1e12)],
    )
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none(), structures=(
                mw.Structure(geometry=Box(position=(0, 0, 0), size=(1, 1, 1)), material=material),
            ), electrostatic_terminals=(
                mw.electrostatic.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=0.2), potential=1.0),
            ))
    with pytest.raises(NotImplementedError):
        mw.Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(boundary=mw.electrostatic.ElectrostaticBoundarySpec.grounded_box(), device="cpu")).run()


def test_anisotropic_tensor_permittivity_supported():
    """Anisotropic (tensor) permittivity now compiles and solves (SPD tensor FVM)."""
    domain = mw.Domain(bounds=((-1, 1),) * 3)
    grid = mw.GridSpec.uniform(2.0 / 16)
    material = mw.Material(
        epsilon_tensor=mw.media.Tensor3x3(((3.0, 0.6, 0.4), (0.6, 2.5, 0.3), (0.4, 0.3, 2.0)))
    )
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none(), structures=(
                mw.Structure(geometry=Box(position=(0, 0, 0), size=(1, 1, 1)), material=material),
            ), electrostatic_terminals=(
                mw.electrostatic.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=0.2), potential=1.0),
            ))
    result = mw.Simulation.electrostatic(
        scene, config=mw.electrostatic.ElectrostaticConfig(boundary=mw.electrostatic.ElectrostaticBoundarySpec.grounded_box(), device="cpu")
    ).run()
    es = mw.electrostatic.electrostatic(result)
    assert es.epsilon_tensor is not None
    assert es.residual <= 1e-9
    # Positive stored energy for a nonzero driven potential.
    assert float(es.energy) > 0.0
