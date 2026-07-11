from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.media._common import FREQUENCY, slab_scene


def build_scene() -> mw.Scene:
    # A zero-thickness box carries the Graphene sheet; the intraband Kubo channel
    # exports as a Tidy3D Medium2D (Drude sheet term).
    return slab_scene(
        mw.Graphene(chemical_potential=0.4, scattering_time=1.0e-13, name="graphene"),
        name="graphene_sheet",
        thickness=0.0,
    )


SCENARIO = ScenarioDefinition(
    name="graphene_sheet",
    description="Zero-thickness Graphene conductive sheet (Tidy3D Medium2D export)",
    builder=build_scene,
    frequencies=(FREQUENCY,),
    display_monitor="plot_field_y",
    display_component="Ex",
    run_time_factor=20.0,
)
