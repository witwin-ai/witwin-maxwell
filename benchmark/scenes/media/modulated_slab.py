from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.media._common import FREQUENCY, slab_scene


def build_scene() -> mw.Scene:
    return slab_scene(
        mw.Material(
            eps_r=4.0,
            modulation=mw.ModulationSpec(frequency=0.2e9, amplitude=0.1, phase=0.0),
            name="modulated_dielectric",
        ),
        name="modulated_slab",
    )


SCENARIO = ScenarioDefinition(
    name="modulated_slab",
    description="Time-modulated permittivity slab (Tidy3D Medium + ModulationSpec export)",
    builder=build_scene,
    frequencies=(FREQUENCY,),
    display_monitor="plot_field_y",
    display_component="Ex",
    run_time_factor=20.0,
)
