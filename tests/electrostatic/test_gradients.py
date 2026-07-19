"""Implicit-differentiation gates for the electrostatic solver.

The reduced electrostatic solve routes through a ``torch.autograd.Function`` whose
forward runs Jacobi-PCG under ``no_grad`` and whose backward solves the adjoint
system ``A^T lambda = dL/dphi`` (A is SPD, so the same reduced operator serves).
These tests are central-difference gates on the resulting parameter gradients:

- ``d(energy)/d(eps_region)`` for a scalar controlling a dielectric slab,
- ``dC_ij/d(eps)`` for a capacitance-matrix entry,
- ``d(energy)/d(charge)`` for a volumetric free-charge magnitude,

each to a relative error below ``1e-4`` on a small float64 problem. A floating
conductor combined with a differentiable permittivity fails closed (the
superposition gradient is not implemented).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.compiler.electrostatic import compile_electrostatics
from witwin.maxwell.electrostatic.capacitance import extract_capacitance
from witwin.maxwell.electrostatic.runtime import differentiable_solve, solve_electrostatics

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="electrostatics is GPU-first")

_L = 1.0e-3


def _config(tol=1e-12):
    return mw.ElectrostaticSolverConfig(tolerance=tol)


def _plate_compiled(n=12):
    """Dirichlet top/bottom (1 V / 0 V), insulating sides: a 1D capacitor cell."""
    domain = mw.Domain(bounds=((0.0, _L), (0.0, _L), (0.0, _L)))
    grid = mw.GridSpec.uniform(_L / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    boundary = mw.ElectrostaticBoundarySpec(
        default="neumann",
        z_low=("dirichlet", 1.0),
        z_high=("dirichlet", 0.0),
    )
    return compile_electrostatics(scene, boundary, dtype=torch.float64)


def _slab_mask(compiled, lo=0.35, hi=0.65):
    """A dielectric slab spanning the middle band of the z axis (full cells)."""
    zc = compiled.zc
    band = (zc > lo * _L) & (zc < hi * _L)
    mask = torch.zeros(compiled.shape, dtype=compiled.dtype, device=compiled.device)
    mask[:, :, band] = 1.0
    assert float(mask.sum()) > 0.0
    return mask


def _central_difference(fn, theta0, h):
    with torch.no_grad():
        plus = float(fn(torch.tensor(theta0 + h, dtype=torch.float64, device="cuda")))
        minus = float(fn(torch.tensor(theta0 - h, dtype=torch.float64, device="cuda")))
    return (plus - minus) / (2.0 * h)


def test_energy_gradient_wrt_region_permittivity():
    compiled = _plate_compiled()
    mask = _slab_mask(compiled)
    base = compiled.epsilon_r.detach()
    config = _config()

    def energy_of(theta):
        eps = base + mask * (theta - 1.0)
        comp = dataclasses.replace(compiled, epsilon_r=eps)
        return solve_electrostatics(comp, config).energy

    theta0 = 3.0
    theta = torch.tensor(theta0, dtype=torch.float64, device="cuda", requires_grad=True)
    energy_of(theta).backward()
    g_ad = float(theta.grad)

    g_fd = _central_difference(energy_of, theta0, 1.0e-4)
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-4


def _center_terminal_compiled(n=14):
    """A small cube conductor inside a grounded (Dirichlet) enclosure."""
    domain = mw.Domain(bounds=((-_L, _L), (-_L, _L), (-_L, _L)))
    grid = mw.GridSpec.uniform(2.0 * _L / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(
            name="t",
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.6 * _L, 0.6 * _L, 0.6 * _L)),
            potential=1.0,
        )
    )
    return compile_electrostatics(
        scene, mw.ElectrostaticBoundarySpec.grounded_box(), dtype=torch.float64
    )


def test_potential_probe_gradient_wrt_permittivity():
    """Non-variational probe functional: directly gates the eps -> phi implicit path.

    The field energy at fixed Dirichlet voltages is variationally stationary, so
    ``d(energy)/d(eps)`` is dominated by the explicit ``dA/d(eps)`` term. A probe of
    the potential at an interior cell is *not* stationary and its gradient flows
    entirely through the implicit-diff backward (``dphi/d(eps)``).
    """
    compiled = _plate_compiled()
    mask = _slab_mask(compiled)
    base = compiled.epsilon_r.detach()
    config = _config()
    nx, ny, nz = compiled.shape
    probe = (nx // 2, ny // 2, nz // 2)

    def probe_of(theta):
        eps = base + mask * (theta - 1.0)
        comp = dataclasses.replace(compiled, epsilon_r=eps)
        return solve_electrostatics(comp, config).potential[probe]

    theta0 = 3.0
    theta = torch.tensor(theta0, dtype=torch.float64, device="cuda", requires_grad=True)
    probe_of(theta).backward()
    g_ad = float(theta.grad)
    # Sanity: the probe genuinely depends on the dielectric (non-trivial gradient).
    assert abs(g_ad) > 1e-3

    g_fd = _central_difference(probe_of, theta0, 1.0e-4)
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-4


def test_capacitance_gradient_wrt_permittivity():
    compiled = _center_terminal_compiled()
    # Dielectric shell region: cells between the conductor and the enclosure.
    zc = compiled.zc
    band = zc.abs() > 0.55 * _L
    mask = torch.zeros(compiled.shape, dtype=compiled.dtype, device=compiled.device)
    mask[:, :, band] = 1.0
    assert float(mask.sum()) > 0.0
    base = compiled.epsilon_r.detach()
    config = _config()

    def c00_of(theta):
        eps = base + mask * (theta - 1.0)
        comp = dataclasses.replace(compiled, epsilon_r=eps)
        data = extract_capacitance(comp, None, None, config)
        return data.matrix[0, 0]

    theta0 = 2.5
    theta = torch.tensor(theta0, dtype=torch.float64, device="cuda", requires_grad=True)
    c00_of(theta).backward()
    g_ad = float(theta.grad)

    g_fd = _central_difference(c00_of, theta0, 1.0e-4)
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-4


def test_energy_gradient_wrt_free_charge():
    """Poisson case: gradient of the field energy in a volumetric charge magnitude."""
    compiled = _plate_compiled()
    mask = _slab_mask(compiled, lo=0.4, hi=0.6)
    base_eps = compiled.epsilon_r.detach()
    cell_volume = compiled.cell_volume.detach()
    config = _config()

    def energy_of(rho):
        # rho (C/m^3) integrated over the slab cells -> per-cell Coulombs.
        free_charge = mask * rho * cell_volume
        comp = dataclasses.replace(compiled, epsilon_r=base_eps, free_charge=free_charge)
        return solve_electrostatics(comp, config).energy

    rho0 = 5.0e-3
    rho = torch.tensor(rho0, dtype=torch.float64, device="cuda", requires_grad=True)
    energy_of(rho).backward()
    g_ad = float(rho.grad)

    g_fd = _central_difference(energy_of, rho0, 1.0e-5)
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-4


def test_potential_probe_gradient_wrt_terminal_voltage():
    """Gates the pinned-cell adjoint multiplier: gradient w.r.t. a terminal voltage.

    The terminal potential enters through the Dirichlet-lift ``fixed_value`` on the
    pinned cells, so its gradient exercises the pinned-cell block of the implicit
    adjoint (``mu_p``), which the eps / free-charge gates do not touch.
    """
    compiled = _center_terminal_compiled()
    mask = compiled.terminals[0].mask
    free_mask = ~mask
    config = _config()
    nx, ny, nz = compiled.shape
    column = free_mask[nx // 2, ny // 2, :]
    zi = int(torch.nonzero(column)[0])
    probe = (nx // 2, ny // 2, zi)
    mask_f = mask.to(torch.float64)

    def probe_of(voltage):
        fixed_value = mask_f * voltage
        phi, _ = differentiable_solve(
            compiled, fixed_value, free_mask, config, use_free_charge=False
        )
        return phi[probe]

    v0 = 2.0
    voltage = torch.tensor(v0, dtype=torch.float64, device="cuda", requires_grad=True)
    probe_of(voltage).backward()
    g_ad = float(voltage.grad)
    assert abs(g_ad) > 1e-3

    g_fd = _central_difference(probe_of, v0, 1.0e-4)
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-4


def test_floating_conductor_gradient_fails_closed():
    """Gradients through the floating-superposition solve are not implemented."""
    domain = mw.Domain(bounds=((-_L, _L), (-_L, _L), (-_L, _L)))
    grid = mw.GridSpec.uniform(2.0 * _L / 12)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(
            name="float",
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.6 * _L, 0.6 * _L, 0.6 * _L)),
            charge=1.0e-15,
        )
    )
    compiled = compile_electrostatics(
        scene, mw.ElectrostaticBoundarySpec.grounded_box(), dtype=torch.float64
    )
    eps = compiled.epsilon_r.detach().clone().requires_grad_(True)
    comp = dataclasses.replace(compiled, epsilon_r=eps)
    with pytest.raises(NotImplementedError, match="floating-conductor superposition"):
        solve_electrostatics(comp, _config())
