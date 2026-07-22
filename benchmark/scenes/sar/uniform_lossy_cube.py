"""Uniform lossy phantom illuminated by a normally incident plane wave.

A single homogeneous lossy tissue block fills the transverse extent of the
domain (an effectively 1D slab) and is illuminated by a +z plane wave. This is
the base SAR benchmark: the volume integral of the absorbed-power density must
close exactly against the shared PowerLossData electric-channel total, and the
point SAR is a positive field that attenuates with depth into the tissue.

Redistributable canonical geometry only (a homogeneous box); tissue dielectric
and mass values are the published-class numbers documented in ``_tissue.py``.
"""

from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.sar._tissue import BENCHMARK_FREQUENCY, tissue_material

FREQUENCIES = (BENCHMARK_FREQUENCY,)


def build_scene(*, dx: float = 0.004, device: str = "cuda") -> mw.Scene:
    half = 0.06
    # Tissue block spans z in [0, thickness]; the plane wave enters at z=0.
    thickness = 0.04
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-0.05, thickness + 0.05))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=10),
        device=device,
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.5 * thickness), size=(10.0, 10.0, thickness)),
            material=tissue_material("phantom"),
            name="tissue",
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
            position=(0.0, 0.0, 0.5 * thickness),
            size=(2.0 * half, 2.0 * half, thickness),
            frequencies=FREQUENCIES,
            channels=("conduction",),
        )
    )
    scene.add_monitor(
        mw.IncidentPowerDensityMonitor(
            "incident", axis="z", position=-0.03, frequencies=FREQUENCIES
        )
    )
    return scene


SCENARIO = ScenarioDefinition(
    name="sar_uniform_lossy_cube",
    description="Uniform lossy phantom under a normally incident plane wave (SAR base case)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="loss",
    display_component="Ex",
)
