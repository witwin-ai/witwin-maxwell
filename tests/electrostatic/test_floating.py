"""Floating-conductor constraints via linear superposition (Phase 2)."""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box, Sphere

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="electrostatics is GPU-first")


def _tol(tol=1e-11):
    return mw.ElectrostaticSolverConfig(tolerance=tol)


def test_floating_conductor_prescribed_charge_conservation():
    """A floating sphere in a grounded box carries exactly its prescribed charge."""
    half = 1.0
    q_target = 3.0e-12
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / 40)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="float", geometry=Sphere(position=(0, 0, 0), radius=0.3), charge=q_target)
    )
    es = mw.Simulation.electrostatic(
        scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box(), solver=_tol()
    ).run().electrostatic

    q_float = float(es.terminal_charge("float"))
    assert abs(q_float - q_target) / q_target < 1e-8

    # Positive charge floats to a positive potential relative to the grounded box.
    assert float(es.potential.max()) > 0.0

    # Gauss closure on free cells still sits at the solver floor.
    assert es.gauss_error / abs(q_target) < 1e-6


def test_floating_conductor_is_equipotential():
    """Every cell of a floating conductor shares one potential (uniform level)."""
    half = 1.0
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / 40)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="drive", geometry=Sphere(position=(-0.55, 0, 0), radius=0.2), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="float", geometry=Sphere(position=(0.4, 0, 0), radius=0.25), charge=0.0)
    )
    compiled = mw.Simulation.capacitance(scene).prepare()  # reuse compiler for the mask
    float_mask = next(t.mask for t in compiled.terminals if t.name == "float")

    es = mw.Simulation.electrostatic(
        scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box(), solver=_tol()
    ).run().electrostatic
    phi_on_float = es.potential[float_mask]
    spread = float(phi_on_float.max() - phi_on_float.min())
    level = float(phi_on_float.mean())
    assert spread / (abs(level) + 1e-30) < 1e-9
    # A charge-neutral conductor near a 1 V electrode floats to an intermediate level.
    assert 0.0 < level < 1.0


def test_floating_midplane_slab_floats_to_average_potential():
    """A charge-neutral slab midway between 1 V and 0 V plates floats to 0.5 V."""
    gap = 1.0e-3
    lateral = 2.0e-3
    domain = mw.Domain(bounds=((0.0, lateral), (0.0, lateral), (0.0, gap)))
    grid = mw.GridSpec.uniform(gap / 40.0)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    boundary = mw.ElectrostaticBoundarySpec(
        default="neumann",
        z_low=("dirichlet", 1.0),
        z_high=("dirichlet", 0.0),
    )
    # Thickness 6 cells (6*gap/40) with faces landing between cell centres, so the
    # mask is symmetric about the midplane and free of cell-boundary ties.
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(
            name="slab",
            geometry=Box(position=(lateral / 2, lateral / 2, gap / 2), size=(lateral, lateral, 6.0 * gap / 40.0)),
            charge=0.0,
        )
    )
    es = mw.Simulation.electrostatic(scene, boundary=boundary, solver=_tol()).run().electrostatic

    q_slab = float(es.terminal_charge("slab"))
    assert abs(q_slab) < 1e-18  # essentially zero net charge

    compiled = mw.Simulation.capacitance(scene, boundary=boundary).prepare()
    slab_mask = next(t.mask for t in compiled.terminals if t.name == "slab")
    level = float(es.potential[slab_mask].mean())
    assert abs(level - 0.5) < 5e-3


def test_two_floating_conductors_charge_split():
    """Two floating conductors each hold their own prescribed charge."""
    half = 1.0
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / 40)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="a", geometry=Sphere(position=(-0.45, 0, 0), radius=0.22), charge=2.0e-12)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="b", geometry=Sphere(position=(0.45, 0, 0), radius=0.22), charge=-1.0e-12)
    )
    es = mw.Simulation.electrostatic(
        scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box(), solver=_tol()
    ).run().electrostatic
    assert abs(float(es.terminal_charge("a")) - 2.0e-12) / 2.0e-12 < 1e-7
    assert abs(float(es.terminal_charge("b")) - (-1.0e-12)) / 1.0e-12 < 1e-7


def test_isolated_floating_conductor_nonzero_charge_incompatible():
    """A charged floating conductor in a fully insulated box is gauge/charge-incompatible."""
    half = 1.0
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / 24)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="iso", geometry=Sphere(position=(0, 0, 0), radius=0.3), charge=1.0e-12)
    )
    with pytest.raises(ValueError):
        mw.Simulation.electrostatic(
            scene, boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol()
        ).run()
