"""Shared builders for the deterministic dielectric-breakdown validation suite.

Two test styles share these helpers:

* *Direct-drive* tests prepare a solver (so ``solver.breakdown_runtime`` exists),
  then prescribe the electric field by hand and call ``advance_breakdown_state``
  step by step. The field evolution is therefore exactly known, which makes the
  trigger step, contiguous-reset behaviour and dissipated energy analytically
  predictable and independent of the Maxwell update.
* *Integration* tests run a real FDTD solve and inspect ``Result.breakdown``.

A node whose colocated ``|E|`` is set from a single Yee component ``Ez = v``
(with ``Ex = Ey = 0``) has cell-center magnitude exactly ``v`` because the
energy-consistent colocation reduces to ``sqrt(0.5*(v^2+v^2)) = v``. All
direct-drive helpers exploit that identity.
"""

from __future__ import annotations

import torch

import witwin.maxwell as mw


def build_breakdown_scene(
    *,
    critical_field: float,
    minimum_duration: float,
    post_sigma: float = 5.0,
    ramp_time: float | None = None,
    default_ramp_steps: int = 10,
    sigma_e: float = 0.0,
    eps_r: float = 3.0,
    h: float = 0.02,
    half: float = 0.3,
    box_size: tuple[float, float, float] = (0.12, 0.12, 0.12),
    box_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    with_source: bool = True,
    source_amplitude: float = 1.0,
    source_position: tuple[float, float, float] = (-0.2, 0.0, 0.0),
):
    """A dielectric box carrying a breakdown descriptor inside a PML box."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(h),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1e6),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="dielectric",
            geometry=mw.Box(position=box_position, size=box_size),
            material=mw.Material(
                eps_r=eps_r,
                sigma_e=sigma_e,
                breakdown=mw.DielectricBreakdown(
                    critical_field=critical_field,
                    minimum_duration=minimum_duration,
                    post_breakdown_conductivity=post_sigma,
                    ramp_time=ramp_time,
                    default_ramp_steps=default_ramp_steps,
                ),
            ),
        )
    )
    if with_source:
        scene.add_source(
            mw.PointDipole(
                position=source_position,
                width=0.04,
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=1e9, amplitude=source_amplitude),
            )
        )
    return scene


def build_plain_scene(*, eps_r: float = 3.0, sigma_e: float = 0.0, **kwargs):
    """The same geometry/source with a plain (breakdown-free) dielectric box."""
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-kwargs.get("half", 0.3), kwargs.get("half", 0.3)),
            )
            * 3
        ),
        grid=mw.GridSpec.uniform(kwargs.get("h", 0.02)),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1e6),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="dielectric",
            geometry=mw.Box(
                position=kwargs.get("box_position", (0.0, 0.0, 0.0)),
                size=kwargs.get("box_size", (0.12, 0.12, 0.12)),
            ),
            material=mw.Material(eps_r=eps_r, sigma_e=sigma_e),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=kwargs.get("source_position", (-0.2, 0.0, 0.0)),
            width=0.04,
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1e9, amplitude=kwargs.get("source_amplitude", 1.0)),
        )
    )
    return scene


def prepare_solver(scene, *, time_steps: int = 10):
    """Prepare an FDTD solver (breakdown runtime compiled) without running it."""
    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
    ).prepare()
    return prepared.solver


def set_uniform_ez(solver, value: float) -> None:
    """Prescribe a spatially uniform Ez field (Ex = Ey = 0).

    Every breakdown-capable node then has colocated ``|E| = value``.
    """
    solver.Ez.fill_(float(value))
    solver.Ex.zero_()
    solver.Ey.zero_()


def run_breakdown(scene, *, time_steps: int = 80):
    """Run a real FDTD solve with full-field DFT output and return the Result."""
    return mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    ).run()
