from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.media._common import FREQUENCY, slab_scene


def build_scene() -> mw.Scene:
    return slab_scene(
        mw.Material.debye(eps_inf=2.0, delta_eps=3.0, tau=8.0e-11),
        name="debye_slab",
    )


SCENARIO = ScenarioDefinition(
    name="debye_slab",
    description="Plane wave through a Debye-dispersive slab (Tidy3D td.Debye export)",
    builder=build_scene,
    frequencies=(FREQUENCY,),
    display_monitor="plot_field_y",
    display_component="Ex",
    run_time_factor=20.0,
)
