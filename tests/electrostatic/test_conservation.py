"""Discrete conservation laws: Gauss closure and the energy identity."""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box, Sphere

from .oracles import OutsideSphere

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="electrostatics is GPU-first")


def _sphere_scene(n=60, a=0.2, b=0.8):
    half = 1.0
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=a), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="outer", geometry=OutsideSphere(b), grounded=True)
    )
    return scene


def test_free_cell_gauss_residual_at_solver_tolerance():
    """The per-cell discrete Gauss residual on free cells sits at the solver floor."""
    scene = _sphere_scene()
    es = mw.Simulation.electrostatic(
        scene,
        boundary=mw.ElectrostaticBoundarySpec.neumann(),
        solver=mw.ElectrostaticSolverConfig(tolerance=1e-11),
    ).run().electrostatic
    # gauss_error is an absolute per-cell divergence; compare against the scale
    # of the terminal charge (the largest surface flux in the problem).
    scale = abs(float(es.terminal_charge("inner")))
    assert es.gauss_error / scale < 1e-8


def test_terminal_charge_conservation():
    """Sum of all conductor charges vanishes (no free volume charge present)."""
    scene = _sphere_scene()
    es = mw.Simulation.electrostatic(
        scene,
        boundary=mw.ElectrostaticBoundarySpec.neumann(),
        solver=mw.ElectrostaticSolverConfig(tolerance=1e-11),
    ).run().electrostatic
    q_in = float(es.terminal_charge("inner"))
    q_out = float(es.terminal_charge("outer"))
    assert abs(q_in + q_out) / abs(q_in) < 1e-9


def test_poisson_volume_charge_gauss_closure():
    """Grounded box enclosing a uniform charge: enclosed flux equals free charge.

    The boundary electrode charge integrates ``D.n`` over the grounded enclosure,
    so ``Q_boundary = -Q_free`` is exactly the discrete divergence theorem.
    """
    half = 1.0
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / 50)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_charge_density(
        mw.ChargeDensity(geometry=Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), density=1.0e-6)
    )
    es = mw.Simulation.electrostatic(
        scene,
        boundary=mw.ElectrostaticBoundarySpec.grounded_box(),
        solver=mw.ElectrostaticSolverConfig(tolerance=1e-11),
    ).run().electrostatic
    q_free = float(es.free_charge.sum())
    q_boundary = float(es.boundary_charge)
    assert q_free > 0.0
    assert abs(q_free + q_boundary) / q_free < 1e-9


def test_energy_identity_field_vs_terminal_work():
    """0.5 integral(E.D) equals 0.5 sum(V_i Q_i) to solver tolerance."""
    scene = _sphere_scene()
    es = mw.Simulation.electrostatic(
        scene,
        boundary=mw.ElectrostaticBoundarySpec.neumann(),
        solver=mw.ElectrostaticSolverConfig(tolerance=1e-11),
    ).run().electrostatic
    field_energy = float(es.energy)
    # Terminal potentials: inner = 1 V, outer = 0 V.
    work = 0.5 * (1.0 * float(es.terminal_charge("inner")) + 0.0 * float(es.terminal_charge("outer")))
    assert abs(field_energy - work) / abs(work) < 1e-9


def test_energy_matches_cell_integral_of_E_dot_D():
    """The operator energy agrees with the volume integral of the reported fields."""
    scene = _sphere_scene(n=60)
    es = mw.Simulation.electrostatic(
        scene,
        boundary=mw.ElectrostaticBoundarySpec.neumann(),
        solver=mw.ElectrostaticSolverConfig(tolerance=1e-11),
    ).run().electrostatic
    ed = es.Ex * es.Dx + es.Ey * es.Dy + es.Ez * es.Dz
    integral = 0.5 * float((ed * es.cell_volume).sum())
    # Cell-centred midpoint integration is second order; a few percent from the
    # face-based operator energy on a staircased sphere is expected.
    assert abs(integral - float(es.energy)) / float(es.energy) < 0.05
