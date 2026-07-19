"""N-terminal Maxwell capacitance matrix extraction (Phase 3)."""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Sphere

from .oracles import OutsideSphere, concentric_sphere_capacitance

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="electrostatics is GPU-first")


def _tol(tol=1e-11):
    return mw.ElectrostaticSolverConfig(tolerance=tol)


def _shell_scene(n, a=0.2, b=0.8):
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


def test_sphere_in_grounded_shell_analytic_capacitance():
    a, b = 0.2, 0.8
    scene = _shell_scene(80, a=a, b=b)
    cap = mw.Simulation.capacitance(
        scene, terminals=("inner", "outer"), reference="outer",
        boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol(),
    ).run().capacitance

    assert cap.terminal_order == ("inner",)
    assert cap.reference == "outer"
    c_numeric = float(cap.capacitance("inner", "inner"))
    c_analytic = concentric_sphere_capacitance(a, b)
    assert c_numeric > 0.0
    assert abs(c_numeric - c_analytic) / c_analytic < 0.06


def test_sphere_in_grounded_shell_capacitance_convergence():
    a, b = 0.2, 0.8
    c_analytic = concentric_sphere_capacitance(a, b)
    errors = []
    for n in (40, 60, 80):
        cap = mw.Simulation.capacitance(
            _shell_scene(n, a=a, b=b), terminals=("inner", "outer"), reference="outer",
            boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol(),
        ).run().capacitance
        c = float(cap.capacitance("inner", "inner"))
        errors.append(abs(c - c_analytic) / c_analytic)
    assert errors[0] > errors[1] > errors[2]


def _three_terminal_scene(n=40):
    half = 1.0
    r = 0.18
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="a", geometry=Sphere(position=(-0.5, 0, 0), radius=r), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="b", geometry=Sphere(position=(0.5, 0, 0), radius=r), potential=2.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="c", geometry=Sphere(position=(0, 0.5, 0), radius=r), potential=-1.0)
    )
    return scene


def test_three_terminal_matrix_symmetry_and_signs():
    scene = _three_terminal_scene()
    cap = mw.Simulation.capacitance(
        scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box(), solver=_tol(),
    ).run().capacitance

    C = cap.matrix
    assert C.shape == (3, 3)
    assert cap.terminal_order == ("a", "b", "c")
    # Reciprocity: raw matrix is symmetric (no silent symmetrization applied).
    assert cap.reciprocity_error < 1e-6
    # Diagonal self terms positive, off-diagonal mutual terms non-positive.
    diag = torch.diagonal(C)
    assert bool((diag > 0).all())
    off = C - torch.diag(diag)
    assert float(off.max()) <= 1e-18


def test_three_terminal_energy_equivalence():
    """0.5 Vᵀ C V equals the field energy of the same driven configuration."""
    scene = _three_terminal_scene()
    boundary = mw.ElectrostaticBoundarySpec.grounded_box()
    cap = mw.Simulation.capacitance(scene, boundary=boundary, solver=_tol()).run().capacitance
    es = mw.Simulation.electrostatic(scene, boundary=boundary, solver=_tol()).run().electrostatic

    v = torch.tensor([1.0, 2.0, -1.0], dtype=cap.matrix.dtype, device=cap.matrix.device)
    w_matrix = 0.5 * float(v @ cap.matrix @ v)
    w_field = float(es.energy)
    assert abs(w_matrix - w_field) / abs(w_field) < 1e-6


def test_three_terminal_reordering_invariance():
    scene = _three_terminal_scene()
    boundary = mw.ElectrostaticBoundarySpec.grounded_box()
    cap1 = mw.Simulation.capacitance(
        scene, terminals=("a", "b", "c"), boundary=boundary, solver=_tol()
    ).run().capacitance
    cap2 = mw.Simulation.capacitance(
        scene, terminals=("c", "a", "b"), boundary=boundary, solver=_tol()
    ).run().capacitance
    for x in ("a", "b", "c"):
        for y in ("a", "b", "c"):
            c1 = float(cap1.capacitance(x, y))
            c2 = float(cap2.capacitance(x, y))
            assert abs(c1 - c2) / (abs(c1) + 1e-30) < 1e-9


def test_row_sum_charge_conservation_under_insulating_boundary():
    """With a Neumann boundary and an explicit reference, the conductor set is
    isolated, so driving every conductor to 1 V induces no charge."""
    half = 1.0
    r = 0.2
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / 40)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="a", geometry=Sphere(position=(-0.45, 0, 0), radius=r), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="b", geometry=Sphere(position=(0.45, 0, 0), radius=r), potential=0.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="ref", geometry=OutsideSphere(0.9), grounded=True)
    )
    cap = mw.Simulation.capacitance(
        scene, terminals=("a", "b", "ref"), reference="ref",
        boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol(),
    ).run().capacitance
    assert cap.terminal_order == ("a", "b")
    assert cap.row_sum_error < 1e-6


def test_two_terminal_equivalent_capacitance():
    a, b = 0.2, 0.8
    scene = _shell_scene(60, a=a, b=b)
    cap = mw.Simulation.capacitance(
        scene, terminals=("inner", "outer"), reference="outer",
        boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol(),
    ).run().capacitance
    # Single active terminal: capacitance-to-reference equals the self term.
    c_self = float(cap.capacitance_to_reference("inner"))
    assert abs(c_self - float(cap.capacitance("inner", "inner"))) < 1e-18


def test_capacitance_needs_return_path():
    scene = _three_terminal_scene()
    with pytest.raises(ValueError):
        mw.Simulation.capacitance(
            scene, boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol()
        ).run()


def test_capacitance_result_accessor_guard():
    scene = _three_terminal_scene()
    es = mw.Simulation.electrostatic(
        scene, boundary=mw.ElectrostaticBoundarySpec.grounded_box(), solver=_tol()
    ).run()
    with pytest.raises(AttributeError):
        _ = es.capacitance
