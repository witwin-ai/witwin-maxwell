from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.media._common import FREQUENCY, slab_scene


def build_scene() -> mw.Scene:
    return slab_scene(
        mw.Material(
            epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0),
            name="uniaxial_crystal",
        ),
        name="anisotropic_slab",
    )


SCENARIO = ScenarioDefinition(
    name="anisotropic_slab",
    description="Diagonal-anisotropic dielectric slab (Tidy3D AnisotropicMedium export)",
    builder=build_scene,
    frequencies=(FREQUENCY,),
    display_monitor="plot_field_y",
    display_component="Ex",
    run_time_factor=20.0,
)
