from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


FREQUENCIES = (2.0e9,)


def build_scene() -> mw.Scene:
    half = 0.64
    scene = (
        base_scene()
        .add_structure(
            mw.Structure(
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(2.0 * half, 2.0 * half, 0.1)),
                material=mw.Material(eps_r=4.0),
                name="slab",
            )
        )
        .add_source(
            mw.PlaneWave(
                direction=(0.0, 0.0, 1.0),
                polarization=(1.0, 0.0, 0.0),
                source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9),
            )
        )
        .add_monitor(mw.FluxMonitor(name="reflected", axis="z", position=-0.3, frequencies=FREQUENCIES, normal_direction="-"))
        .add_monitor(mw.FluxMonitor(name="transmitted", axis="z", position=0.3, frequencies=FREQUENCIES))
        .add_monitor(mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES))
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="dielectric_slab",
    description="Plane wave through dielectric slab (reflection/transmission)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ex",
)
