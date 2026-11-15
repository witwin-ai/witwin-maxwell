"""Dispersive and lossy spheres against the Mie series (analytic truth for media rows).

Spheres close to the benchmark media rows whose external-reference residual is
open: the ``lorentz_resonator`` material (eps_inf 2, delta_eps 3, f0 = 2 GHz,
gamma = 0.05 GHz, driven exactly at resonance: eps = 2 + 120i) on a sphere of the
resonator's radius, the ``sellmeier_sphere`` and ``debye_sphere`` materials and
radius, and a conductive sphere. Each runs at the benchmark grid (25 mm) and
successive refinements under the TFSF harness of ``sphere_ladder``.

    python -m benchmark.analytic.dispersive_spheres --out <dir> [--cases ...] [--dx ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import witwin.maxwell as mw
from benchmark.analytic import mie
from benchmark.analytic.sphere_ladder import SphereCase, _update_csv, run_case

FREQUENCY = 2.0e9
BENCHMARK_DX = 0.025

CASES = {
    "lorentz_sphere": (
        0.10,
        mw.Material.lorentz(eps_inf=2.0, delta_eps=3.0, resonance_frequency=2.0e9, gamma=0.05e9),
        mie.lorentz_permittivity(FREQUENCY, eps_inf=2.0, delta_eps=3.0,
                                 resonance_frequency=2.0e9, gamma=0.05e9),
    ),
    "sellmeier_sphere": (
        0.15,
        mw.Material.sellmeier(eps_inf=1.5, b_coefficients=(1.5,), c_coefficients=(2.5e-3,)),
        mie.sellmeier_permittivity(FREQUENCY, eps_inf=1.5, b_coefficients=(1.5,),
                                   c_coefficients=(2.5e-3,)),
    ),
    "debye_sphere": (
        0.15,
        mw.Material.debye(eps_inf=2.0, delta_eps=2.0, tau=5.0e-11),
        mie.debye_permittivity(FREQUENCY, eps_inf=2.0, delta_eps=2.0, tau=5.0e-11),
    ),
    "conductive_sphere": (
        0.15,
        mw.Material(eps_r=3.0, sigma_e=0.05),
        mie.conductive_permittivity(FREQUENCY, eps_r=3.0, sigma_e=0.05),
    ),
}


def case(name: str, dx: float) -> SphereCase:
    radius, material, eps = CASES[name]
    return SphereCase(name=f"{name}_dx{dx * 1e3:g}mm", radius=radius, dx=dx, material=complex(eps),
                      scene_material=material, frequency=FREQUENCY,
                      transient_cycles=150, steady_cycles=40)


# Lorentz-family poles (Lorentz and Sellmeier) as (eps_inf, ((delta_eps, f0, gamma), ...)).
LORENTZ_POLES = {
    "lorentz_sphere": (2.0, ((3.0, 2.0e9, 0.05e9),)),
    "sellmeier_sphere": (1.5, ((1.5, mie.C0 / np.sqrt(2.5e-3), 0.0),)),
}


def _discrete_material_reference(name: str, record: dict) -> dict:
    """Mie for the permittivity the time-stepped Lorentz ADE realizes at this dt.

    Separates the ADE's own discretization (a frequency-warped resonance) from the
    spatial error: if FDTD tracks Mie(eps_discrete) but not Mie(eps(f)), the
    residual is the material update, not the geometry or the grid.
    """
    if name not in LORENTZ_POLES:
        return {}
    eps_inf, poles = LORENTZ_POLES[name]
    dt = float(record["dt"])
    eps = eps_inf + sum(
        mie.lorentz_permittivity_discrete(FREQUENCY, dt, eps_inf=0.0, delta_eps=d, resonance_frequency=f0, gamma=g)
        for d, f0, g in poles)
    radius = CASES[name][0]
    k = 2 * np.pi * FREQUENCY / mie.C0
    reference = mie.cross_sections(radius, k, eps)
    return {
        "eps_discrete": str(complex(eps)),
        "sigma_ext_mie_discrete": reference["extinction"],
        "sigma_abs_mie_discrete": reference["absorption"],
        "ext_rel_err_vs_discrete": abs(record["sigma_ext"] / reference["extinction"] - 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", default=sorted(CASES))
    parser.add_argument("--dx", type=float, nargs="+",
                        default=[BENCHMARK_DX, BENCHMARK_DX / 2, BENCHMARK_DX / 4])
    parser.add_argument("--budget-gib", type=float, default=7.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        radius, _, eps = CASES[name]
        k = 2 * np.pi * FREQUENCY / mie.C0
        print(f"{name}: eps(f)={eps:.6g} |m|={abs(np.sqrt(eps)):.4g} ka={k * radius:.4g} "
              f"Mie {mie.cross_sections(radius, k, eps)}", flush=True)
        for dx in args.dx:
            run = case(name, dx)
            output = run_case(run, budget_gib=args.budget_gib)
            if output is None:
                row = {"name": run.name, "skipped": "memory", "cells": run.cells()}
            else:
                row = {**output["record"], "eps_f": str(eps), **_discrete_material_reference(name, output["record"])}
                np.savez(args.out / f"{run.name}.npz", **output["curves"])
            _update_csv(args.out / "dispersive_spheres.csv", row)


if __name__ == "__main__":
    main()
