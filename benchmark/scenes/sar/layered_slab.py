"""Three-layer skin / fat / muscle slab under a normally incident plane wave.

A canonical layered exposure phantom: three homogeneous tissue layers stacked
along z (skin, then fat, then muscle), illuminated by a +z plane wave. The
layers use the published-class dielectric and mass values documented in
``_tissue.py``. This is the scene the H3 track drives the peak 1 g SAR
grid-convergence and power-conservation closure on.

Redistributable canonical geometry only (stacked homogeneous boxes); no licensed
anatomical model.
"""

from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.sar._tissue import BENCHMARK_FREQUENCY, tissue_material

FREQUENCIES = (BENCHMARK_FREQUENCY,)

# Layer thicknesses along z (m). Chosen thick enough to resolve each tissue on a
# modest grid while keeping the skin/fat/muscle ordering of a canonical phantom.
SKIN_THICKNESS = 0.008
FAT_THICKNESS = 0.012
MUSCLE_THICKNESS = 0.040


def layer_bounds() -> dict[str, tuple[float, float]]:
    """z bounds of each layer; the stack starts at z=0 (illuminated face)."""

    z0 = 0.0
    z1 = z0 + SKIN_THICKNESS
    z2 = z1 + FAT_THICKNESS
    z3 = z2 + MUSCLE_THICKNESS
    return {"skin_dry": (z0, z1), "fat": (z1, z2), "muscle": (z2, z3)}


def build_scene(*, dx: float = 0.004, device: str = "cuda") -> mw.Scene:
    half = 0.06
    bounds = layer_bounds()
    stack_end = bounds["muscle"][1]
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-0.05, stack_end + 0.05))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=10),
        device=device,
    )
    for tissue, (z_lo, z_hi) in bounds.items():
        thickness = z_hi - z_lo
        scene.add_structure(
            mw.Structure(
                geometry=mw.Box(
                    position=(0.0, 0.0, 0.5 * (z_lo + z_hi)),
                    size=(10.0, 10.0, thickness),
                ),
                material=tissue_material(tissue),
                name=tissue,
            )
        )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=BENCHMARK_FREQUENCY, fwidth=0.5 * BENCHMARK_FREQUENCY),
            name="pw",
        )
    )
    scene.add_monitor(
        mw.PowerLossMonitor(
            "loss",
            position=(0.0, 0.0, 0.5 * stack_end),
            size=(2.0 * half, 2.0 * half, stack_end),
            frequencies=FREQUENCIES,
            channels=("conduction",),
        )
    )
    scene.add_monitor(
        mw.IncidentPowerDensityMonitor(
            "incident", axis="z", position=-0.03, frequencies=FREQUENCIES, spatial_average=4e-4
        )
    )
    return scene


SCENARIO = ScenarioDefinition(
    name="sar_layered_slab",
    description="Skin/fat/muscle three-layer slab under a plane wave (layered SAR phantom)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="loss",
    display_component="Ex",
)
