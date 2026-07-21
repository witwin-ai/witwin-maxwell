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


def build_conservation_scene(*, dx: float = 0.004, device: str = "cuda") -> mw.Scene:
    """Periodic-transverse variant for a clean 1-D power-conservation balance.

    The shipped :func:`build_scene` uses PML on every face: it illuminates a
    finite transverse patch, so a closed surface around the slab leaks laterally
    (edge diffraction into the transverse PML) and the surface Poynting balance
    does not close against the absorbed-power volume integral. For an infinite
    planar slab under normal incidence the physical boundary is periodic in the
    transverse plane; the field is then transverse-uniform and the net power
    balance reduces to two z-planes:

        P_absorbed = flux(z_in, +z) - flux(z_out, +z)

    where ``z_in`` sits in the vacuum ahead of the slab and ``z_out`` in the
    vacuum behind it. The absorbed power measured this way (surface E x H) is
    independent of the volume ``sigma |E|^2`` integral that SAR is built from, so
    their agreement is a wave-level conservation check, not a self-consistency
    identity. A small transverse extent is sufficient because the field is
    uniform there; the loss monitor still spans the full cross-section.
    """

    lat = 0.02
    bounds = layer_bounds()
    stack_end = bounds["muscle"][1]
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-0.5 * lat, 0.5 * lat), (-0.5 * lat, 0.5 * lat), (-0.05, stack_end + 0.05))
        ),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec(kind="pml", num_layers=10, x="periodic", y="periodic"),
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
            size=(lat, lat, stack_end),
            frequencies=FREQUENCIES,
            channels=("conduction",),
        )
    )
    scene.add_monitor(
        mw.FluxMonitor("flux_in", axis="z", position=-0.02, frequencies=FREQUENCIES, normal_direction="+")
    )
    scene.add_monitor(
        mw.FluxMonitor(
            "flux_out", axis="z", position=stack_end + 0.02, frequencies=FREQUENCIES, normal_direction="+"
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
