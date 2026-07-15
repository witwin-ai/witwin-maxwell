from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import with_plot_monitors


FREQUENCY = 1.0e12
FREQUENCIES = (FREQUENCY,)
HALF_WIDTH = 150.0e-6
HALF_LENGTH = 450.0e-6
DL = 15.0e-6


def build_scene() -> mw.Scene:
    # Graphene's 0.1 ps scattering time is a THz-scale material clock. A matched
    # sub-millimetre scene resolves both that relaxation and the 1 THz wave with
    # a few thousand steps; the old metre/GHz scene forced more than four million
    # unnecessarily small steps and probed a nearly static sheet.
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-HALF_WIDTH, HALF_WIDTH),
                (-HALF_WIDTH, HALF_WIDTH),
                (-HALF_LENGTH, HALF_LENGTH),
            )
        ),
        grid=mw.GridSpec.uniform(DL),
        boundary=mw.BoundarySpec.pml(num_layers=12).with_faces(
            x_low="periodic",
            x_high="periodic",
            y_low="periodic",
            y_high="periodic",
        ),
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=(2.0 * HALF_WIDTH, 2.0 * HALF_WIDTH, 0.0),
            ),
            material=mw.Graphene(
                chemical_potential=0.4,
                scattering_time=1.0e-13,
                name="graphene",
            ),
            name="graphene_sheet",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=mw.GaussianPulse(
                frequency=FREQUENCY,
                fwidth=0.2e12,
            ),
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            name="reflected",
            axis="z",
            position=-200.0e-6,
            frequencies=FREQUENCIES,
            normal_direction="-",
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            name="transmitted",
            axis="z",
            position=200.0e-6,
            frequencies=FREQUENCIES,
        )
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="graphene_sheet",
    description="Zero-thickness Graphene conductive sheet (Tidy3D Medium2D export)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="plot_field_y",
    display_component="Ex",
    run_time_factor=20.0,
)
