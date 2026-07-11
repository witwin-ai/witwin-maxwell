from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.media._common import FREQUENCY, slab_scene


def build_scene() -> mw.Scene:
    return slab_scene(
        mw.Material(eps_r=4.0, kerr_chi3=1.0e-18, name="kerr_dielectric"),
        name="kerr_slab",
    )


SCENARIO = ScenarioDefinition(
    name="kerr_slab",
    description="Kerr (chi3) nonlinear slab (Tidy3D NonlinearSusceptibility export)",
    builder=build_scene,
    frequencies=(FREQUENCY,),
    display_monitor="plot_field_y",
    display_component="Ex",
    run_time_factor=20.0,
)
