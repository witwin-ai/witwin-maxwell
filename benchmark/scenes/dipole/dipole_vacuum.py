from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


FREQUENCIES = (1.5e9,)


def build_scene() -> mw.Scene:
    scene = (
        base_scene()
        .add_source(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                profile="ideal",
                source_time=mw.GaussianPulse(frequency=1.5e9, fwidth=0.5e9),
            )
        )
        .add_monitor(
            mw.PlaneMonitor(
                name="field_xy",
                axis="z",
                position=0.0,
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                frequencies=FREQUENCIES,
            )
        )
        .add_monitor(mw.FluxMonitor(name="flux_z_pos", axis="z", position=0.3, frequencies=FREQUENCIES))
        .add_monitor(
            mw.FluxMonitor(name="flux_z_neg", axis="z", position=-0.3, frequencies=FREQUENCIES, normal_direction="-")
        )
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="dipole_vacuum",
    description="Ez point-dipole in vacuum (radiation pattern + flux)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xy",
    display_component="Ez",
    maxwell_alignment={"source": {"profile": "ideal"}, "monitor_fields": {"field_xy": ("Ez", "Hx", "Hy")}},
)
