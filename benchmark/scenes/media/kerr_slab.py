from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.media._common import FREQUENCY, slab_scene


def build_scene() -> mw.Scene:
    return slab_scene(
        mw.Material(eps_r=4.0, kerr_chi3=1.0e-18, name="kerr_dielectric"),
        name="kerr_slab",
        # A weak-but-observable drive: the nonlinear field differs from the
        # matched linear slab by a few percent, avoiding a solver-specific
        # strongly nonlinear trajectory.
        source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9, amplitude=3.0e7),
        boundary=mw.BoundarySpec.pml(num_layers=12).with_faces(
            x_low="periodic", x_high="periodic", y_low="periodic", y_high="periodic"
        ),
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
