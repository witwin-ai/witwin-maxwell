"""Synthetic-golden 1 g averaging phantom (hand-computable mass window).

A homogeneous cube whose grid spacing is chosen so that a symmetric 3x3x3 cell
window weighs EXACTLY 1 gram: with mass density ``rho = 1000`` kg/m^3 and cell
volume ``dx^3``, ``27 * rho * dx^3 = 1e-3`` gives ``dx = (1e-3 / 27000)^(1/3)``.
The smallest averaging cube reaching 1 g is therefore the 3x3x3 window (half
width 1 cell), its enclosed mass is exactly 1e-3 kg, and under a uniform field
the 1 g averaged SAR equals the point SAR ``0.5*sigma*|E|^2/rho`` exactly. This
gives the mass-averaging kernel a closed-form golden target.

Redistributable canonical geometry only (a homogeneous cube).
"""

from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.sar._tissue import BENCHMARK_FREQUENCY

FREQUENCIES = (BENCHMARK_FREQUENCY,)

TARGET_MASS = 1e-3  # kg (1 gram)
MASS_DENSITY = 1000.0  # kg / m^3
CELLS_PER_EDGE = 3  # the 1 g window is a 3x3x3 cube
# dx so that CELLS_PER_EDGE^3 cells weigh exactly TARGET_MASS.
ONE_GRAM_DX = (TARGET_MASS / (CELLS_PER_EDGE**3 * MASS_DENSITY)) ** (1.0 / 3.0)

# A domain wide enough that a strict-interior 3x3x3 window exists (11 cells/edge).
_CELLS = 11
SIGMA_E = 0.99


def one_gram_cell_mass() -> float:
    """Exact per-cell mass (kg) of the phantom on the one-gram grid."""

    return MASS_DENSITY * ONE_GRAM_DX**3


def build_scene(*, device: str = "cuda") -> mw.Scene:
    extent = _CELLS * ONE_GRAM_DX
    material = mw.Material(
        eps_r=42.0, sigma_e=SIGMA_E, mass_density=MASS_DENSITY, name="phantom"
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, extent),) * 3),
        grid=mw.GridSpec.uniform(ONE_GRAM_DX),
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                geometry=mw.Box(position=(0.5 * extent,) * 3, size=(10.0,) * 3),
                material=material,
                name="phantom",
            ),
        ),
        device=device,
    )
    scene.add_monitor(
        mw.PowerLossMonitor(
            "loss",
            position=(0.5 * extent,) * 3,
            size=(extent,) * 3,
            frequencies=FREQUENCIES,
            channels=("conduction",),
        )
    )
    return scene


SCENARIO = ScenarioDefinition(
    name="sar_one_gram_cube",
    description="Synthetic hand-computable 1 g mass-averaging phantom (golden window)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="loss",
    display_component="Ex",
)
