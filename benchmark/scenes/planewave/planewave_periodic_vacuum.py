from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


FREQUENCIES = (2.0e9,)


def build_scene() -> mw.Scene:
    reference = base_scene()
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=mw.BoundarySpec.pml(num_layers=12).with_faces(
            x_low="periodic",
            x_high="periodic",
            y_low="periodic",
            y_high="periodic",
        ),
        subpixel_samples=reference.subpixel,
        device="cpu",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(
                frequency=FREQUENCIES[0],
                fwidth=0.5e9,
                amplitude=1.0,
            ),
            name="plane_wave",
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            name="field_xz",
            axis="y",
            position=0.0,
            fields=("Ex", "Ey", "Ez"),
            frequencies=FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            name="flux_pos_z",
            axis="z",
            position=0.30,
            frequencies=FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            name="flux_neg_z",
            axis="z",
            position=-0.30,
            frequencies=FREQUENCIES,
            normal_direction="-",
        )
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="planewave_periodic_vacuum",
    description="Vacuum scene with a +z plane wave and periodic transverse boundaries.",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ex",
)
