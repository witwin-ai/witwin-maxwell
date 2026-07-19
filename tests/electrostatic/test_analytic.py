"""Analytic validation of the scalar electrostatic Laplace/Poisson solver."""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Cylinder, Sphere
from witwin.core.material import VACUUM_PERMITTIVITY as EPS0

from .oracles import (
    OutsideCylinder,
    OutsideSphere,
    coaxial_capacitance_per_length,
    concentric_sphere_capacitance,
    parallel_plate_capacitance,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="electrostatics is GPU-first")


def _tol_config(tol=1e-11):
    return mw.ElectrostaticSolverConfig(tolerance=tol)


def test_parallel_plate_linear_potential_and_capacitance():
    """Dirichlet top/bottom with insulating side walls: an ideal 1D capacitor.

    Neumann lateral walls suppress fringing, so the discrete solution is the
    exact infinite-plate field: a linear potential and C = eps0 * A / d with no
    edge-effect correction.
    """
    gap = 1.0e-3
    lateral = 2.0e-3
    domain = mw.Domain(bounds=((0.0, lateral), (0.0, lateral), (0.0, gap)))
    grid = mw.GridSpec.uniform(gap / 20.0)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    boundary = mw.ElectrostaticBoundarySpec(
        default="neumann",
        z_low=("dirichlet", 1.0),
        z_high=("dirichlet", 0.0),
    )
    result = mw.Simulation.electrostatic(scene, boundary=boundary, solver=_tol_config(1e-12)).run()
    es = result.electrostatic

    # Linear potential profile phi(z) = 1 - z/gap.
    zc = es.zc
    profile = es.potential.mean(dim=(0, 1))
    expected = 1.0 - zc / gap
    assert float((profile - expected).abs().max()) < 1e-9

    # Capacitance from the field energy: W = 0.5 C V^2, V = 1.
    energy = float(es.energy)
    c_numeric = 2.0 * energy
    c_analytic = parallel_plate_capacitance(lateral * lateral, gap)
    assert abs(c_numeric - c_analytic) / c_analytic < 1e-6


def _run_concentric_sphere(n, a=0.2, b=0.8, eps_r=1.0):
    half = 1.0
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    if eps_r != 1.0:
        scene.add_structure(
            mw.Structure(
                geometry=mw.Box(position=(0, 0, 0), size=(2 * half, 2 * half, 2 * half)),
                material=mw.Material(eps_r=eps_r),
            )
        )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=a), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="outer", geometry=OutsideSphere(b), grounded=True)
    )
    result = mw.Simulation.electrostatic(
        scene, boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol_config()
    ).run()
    return result.electrostatic


def test_concentric_spheres_potential_and_capacitance():
    a, b = 0.2, 0.8
    es = _run_concentric_sphere(80, a=a, b=b)
    q_inner = float(es.terminal_charge("inner"))
    c_numeric = q_inner / 1.0
    c_analytic = concentric_sphere_capacitance(a, b)
    assert abs(c_numeric - c_analytic) / c_analytic < 0.06

    # Radial 1/r potential in the vacuum gap sampled along the +x axis.
    xc = es.xc
    ny = es.potential.shape[1] // 2
    nz = es.potential.shape[2] // 2
    line = es.potential[:, ny, nz]
    r = xc.abs()
    interior = (r > (a + 0.08)) & (r < (b - 0.08)) & (xc > 0)
    phi_analytic = (1.0 / r - 1.0 / b) / (1.0 / a - 1.0 / b)
    err = (line[interior] - phi_analytic[interior]).abs().max()
    assert float(err) < 0.05


def test_concentric_sphere_grid_convergence():
    a, b = 0.2, 0.8
    c_analytic = concentric_sphere_capacitance(a, b)
    errors = []
    for n in (40, 60, 80):
        es = _run_concentric_sphere(n, a=a, b=b)
        c = float(es.terminal_charge("inner"))
        errors.append(abs(c - c_analytic) / c_analytic)
    # Monotone refinement: staircased spherical conductors converge from below.
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.06


def test_coaxial_cylinder_capacitance_and_convergence():
    a, b, lz = 0.2, 0.8, 1.0
    c_analytic = coaxial_capacitance_per_length(a, b)
    errors = []
    for n in (48, 72, 96):
        half = 1.0
        nz = max(4, int(n * lz / (2 * half)))
        domain = mw.Domain(bounds=((-half, half), (-half, half), (0.0, lz)))
        grid = mw.GridSpec.anisotropic(2 * half / n, 2 * half / n, lz / nz)
        scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
        inner = Cylinder(position=(0, 0, lz / 2), radius=a, height=4 * lz, axis="z")
        scene.add_electrostatic_terminal(mw.ElectrostaticTerminal(name="inner", geometry=inner, potential=1.0))
        scene.add_electrostatic_terminal(mw.ElectrostaticTerminal(name="outer", geometry=OutsideCylinder(b), grounded=True))
        result = mw.Simulation.electrostatic(
            scene, boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol_config()
        ).run()
        es = result.electrostatic
        c_per_len = float(es.terminal_charge("inner")) / lz
        errors.append(abs(c_per_len - c_analytic) / c_analytic)
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.06


def test_dielectric_fill_scales_capacitance():
    a, b = 0.2, 0.8
    q_vac = float(_run_concentric_sphere(60, a=a, b=b, eps_r=1.0).terminal_charge("inner"))
    q_diel = float(_run_concentric_sphere(60, a=a, b=b, eps_r=4.0).terminal_charge("inner"))
    assert abs(q_diel / q_vac - 4.0) < 1e-3
