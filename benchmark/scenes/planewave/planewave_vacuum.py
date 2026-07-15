from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


FREQUENCIES = (2.0e9,)


def build_scene() -> mw.Scene:
    scene = (
        base_scene()
        .add_source(
            mw.PlaneWave(
                direction=(0.0, 0.0, 1.0),
                polarization=(1.0, 0.0, 0.0),
                source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9),
                name="plane_wave",
            )
        )
        .add_monitor(mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="flux_pos_z", axis="z", position=0.30, frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="flux_neg_z", axis="z", position=-0.30, frequencies=FREQUENCIES, normal_direction="-"))
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="planewave_vacuum",
    description="Vacuum scene with a +z plane wave source.",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ex",
)
