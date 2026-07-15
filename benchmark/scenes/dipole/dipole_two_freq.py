from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


FREQUENCIES = (1.0e9, 2.0e9)


def build_scene() -> mw.Scene:
    scene = (
        base_scene()
        .add_source(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                profile="ideal",
                source_time=mw.GaussianPulse(frequency=1.5e9, fwidth=1.0e9),
            )
        )
        .add_monitor(mw.PlaneMonitor(name="field_xy", axis="z", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="flux_z", axis="z", position=0.3, frequencies=FREQUENCIES))
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="dipole_two_freq",
    description="Vacuum dipole with two-frequency extraction",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xy",
    display_component="Ez",
    maxwell_alignment={"source": {"profile": "ideal"}},
)
