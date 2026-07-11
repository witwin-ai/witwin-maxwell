from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.media._common import FREQUENCY, slab_scene


def build_scene() -> mw.Scene:
    return slab_scene(
        mw.Material(
            eps_r=2.0,
            sigma_e=0.05,
            drude_poles=(mw.DrudePole(plasma_frequency=3.0e9, gamma=0.2e9),),
            name="sigma_e_drude",
        ),
        name="sigma_e_drude_slab",
    )


SCENARIO = ScenarioDefinition(
    name="sigma_e_drude_slab",
    description="Drude slab with static electric conductivity (folds to one Tidy3D PoleResidue)",
    builder=build_scene,
    frequencies=(FREQUENCY,),
    display_monitor="plot_field_y",
    display_component="Ex",
    run_time_factor=20.0,
)
