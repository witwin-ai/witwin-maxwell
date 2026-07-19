"""Reproduce the recorded electrostatic acceptance metrics.

Run (GPU required):

    export CUDA_HOME=.../nvidia/cu13 && export PATH="$CUDA_HOME/bin:$PATH"
    export PYTHONPATH=<worktree>
    conda run -n maxwell --no-capture-output python \
        docs/assessments/a12_electrostatics_metrics.py

Prints the parallel-plate, concentric-sphere, coaxial-cylinder, and dielectric
fill numbers quoted in ``a12-electrostatics-acceptance-2026-07-19.md`` so every
recorded figure has a committed command that regenerates it. This is a validation
helper, not part of the shipped package.
"""

from __future__ import annotations

import math

import torch

import witwin.maxwell as mw
from witwin.core import Cylinder, Sphere
from witwin.core.material import VACUUM_PERMITTIVITY as EPS0


class OutsideSphere:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def signed_distance(self, x, y, z):
        return self.radius - torch.sqrt(x * x + y * y + z * z)


class OutsideCylinder:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def signed_distance(self, x, y, z):
        return self.radius - torch.sqrt(x * x + y * y)


def _tol(tol=1e-11):
    return mw.ElectrostaticSolverConfig(tolerance=tol)


def parallel_plate():
    gap, lateral = 1.0e-3, 2.0e-3
    domain = mw.Domain(bounds=((0.0, lateral), (0.0, lateral), (0.0, gap)))
    grid = mw.GridSpec.uniform(gap / 20.0)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    boundary = mw.ElectrostaticBoundarySpec(
        default="neumann", z_low=("dirichlet", 1.0), z_high=("dirichlet", 0.0)
    )
    es = mw.Simulation.electrostatic(scene, boundary=boundary, solver=_tol(1e-12)).run().electrostatic
    profile = es.potential.mean(dim=(0, 1))
    expected = 1.0 - es.zc / gap
    max_err = float((profile - expected).abs().max())
    c_numeric = 2.0 * float(es.energy)
    c_analytic = EPS0 * lateral * lateral / gap
    return max_err, abs(c_numeric - c_analytic) / c_analytic


def _sphere(n, a=0.2, b=0.8, eps_r=1.0):
    half = 1.0
    domain = mw.Domain(bounds=((-half, half),) * 3)
    grid = mw.GridSpec.uniform(2.0 * half / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    if eps_r != 1.0:
        scene.add_structure(
            mw.Structure(
                geometry=mw.Box(position=(0, 0, 0), size=(2 * half,) * 3),
                material=mw.Material(eps_r=eps_r),
            )
        )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="inner", geometry=Sphere(position=(0, 0, 0), radius=a), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="outer", geometry=OutsideSphere(b), grounded=True)
    )
    es = mw.Simulation.electrostatic(
        scene, boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol()
    ).run().electrostatic
    return float(es.terminal_charge("inner"))


def concentric_spheres(a=0.2, b=0.8):
    c_analytic = 4.0 * math.pi * EPS0 / (1.0 / a - 1.0 / b)
    errs = [abs(_sphere(n, a, b) - c_analytic) / c_analytic for n in (40, 60, 80)]
    return c_analytic, errs


def coaxial(a=0.2, b=0.8, lz=1.0):
    c_analytic = 2.0 * math.pi * EPS0 / math.log(b / a)
    errs = []
    for n in (48, 72, 96):
        half = 1.0
        nz = max(4, int(n * lz / (2 * half)))
        domain = mw.Domain(bounds=((-half, half), (-half, half), (0.0, lz)))
        grid = mw.GridSpec.anisotropic(2 * half / n, 2 * half / n, lz / nz)
        scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
        inner = Cylinder(position=(0, 0, lz / 2), radius=a, height=4 * lz, axis="z")
        scene.add_electrostatic_terminal(mw.ElectrostaticTerminal(name="inner", geometry=inner, potential=1.0))
        scene.add_electrostatic_terminal(mw.ElectrostaticTerminal(name="outer", geometry=OutsideCylinder(b), grounded=True))
        es = mw.Simulation.electrostatic(
            scene, boundary=mw.ElectrostaticBoundarySpec.neumann(), solver=_tol()
        ).run().electrostatic
        errs.append(abs(float(es.terminal_charge("inner")) / lz - c_analytic) / c_analytic)
    return c_analytic, errs


def dielectric_ratio(a=0.2, b=0.8):
    return _sphere(60, a, b, eps_r=4.0) / _sphere(60, a, b, eps_r=1.0)


def stationarity():
    """Implicit contribution to d(energy)/d(eps), relative to the total gradient.

    For a fixed-Dirichlet (fixed-voltage) parallel-plate cell the field energy is
    variationally stationary in the potential, so d(energy)/d(eps) is dominated by
    the explicit dA/d(eps) term and the implicit dphi/d(eps) path contributes only
    at the residual/round-off floor. This computes the total gradient through the
    implicit-diff solve, subtracts the explicit-only gradient (energy re-evaluated
    at the frozen solved potential), and returns the implicit fraction of the
    total -- the number the A3 acceptance stationarity note quotes.
    """
    import dataclasses

    from witwin.maxwell.compiler.electrostatic import compile_electrostatics
    from witwin.maxwell.electrostatic.runtime import ElectrostaticOperator, solve_electrostatics

    gap, n = 1.0e-3, 12
    domain = mw.Domain(bounds=((0.0, gap), (0.0, gap), (0.0, gap)))
    grid = mw.GridSpec.uniform(gap / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    boundary = mw.ElectrostaticBoundarySpec(
        default="neumann", z_low=("dirichlet", 1.0), z_high=("dirichlet", 0.0)
    )
    compiled = compile_electrostatics(scene, boundary, dtype=torch.float64)
    config = _tol(1e-12)

    zc = compiled.zc
    band = (zc > 0.35 * gap) & (zc < 0.65 * gap)
    mask = torch.zeros(compiled.shape, dtype=compiled.dtype, device=compiled.device)
    mask[:, :, band] = 1.0
    base = compiled.epsilon_r.detach()
    theta0 = 3.0

    # Total gradient d(energy)/d(theta) through the implicit-diff backward.
    theta = torch.tensor(theta0, dtype=torch.float64, device=compiled.device, requires_grad=True)
    eps = base + mask * (theta - 1.0)
    solve_electrostatics(dataclasses.replace(compiled, epsilon_r=eps), config).energy.backward()
    g_total = float(theta.grad)

    # Explicit-only gradient: energy re-evaluated at the frozen solved potential,
    # so the implicit dphi/d(eps) path is severed. The difference is the implicit part.
    with torch.no_grad():
        eps0 = base + mask * (theta0 - 1.0)
        phi = solve_electrostatics(dataclasses.replace(compiled, epsilon_r=eps0), config).potential
    theta_e = torch.tensor(theta0, dtype=torch.float64, device=compiled.device, requires_grad=True)
    eps_e = base + mask * (theta_e - 1.0)
    ElectrostaticOperator(dataclasses.replace(compiled, epsilon_r=eps_e)).field_energy(phi).backward()
    g_explicit = float(theta_e.grad)

    return (g_total - g_explicit) / g_total


def main():
    plate_err, plate_c_err = parallel_plate()
    print(f"parallel_plate: max_potential_err={plate_err:.2e}  rel_C_err={plate_c_err:.2e}")
    c_s, s_errs = concentric_spheres()
    print(f"concentric_spheres: C_analytic={c_s:.4e}  errs(40/60/80)="
          + "/".join(f"{100*e:.1f}%" for e in s_errs))
    c_c, c_errs = coaxial()
    print(f"coaxial: Cp_analytic={c_c:.4e}  errs(48/72/96)="
          + "/".join(f"{100*e:.1f}%" for e in c_errs))
    print(f"dielectric_fill ratio (eps_r=4): {dielectric_ratio():.5f}")
    print(f"stationarity: implicit/total d(energy)/d(eps) = {stationarity():.2e}")


if __name__ == "__main__":
    main()
