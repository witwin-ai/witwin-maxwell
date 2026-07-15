from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import SAFE_FLUX_POSITION, base_scene, with_plot_monitors


FREQUENCIES = (2.0e9,)


def build_scene() -> mw.Scene:
    scene = (
        base_scene()
        .add_structure(mw.Structure(geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.12), material=mw.Material(eps_r=4.0), name="sphere"))
        .add_source(
            mw.PlaneWave(
                direction=(0.0, 0.0, 1.0),
                polarization=(1.0, 0.0, 0.0),
                source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9),
            )
        )
        .add_monitor(mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="forward_flux", axis="z", position=SAFE_FLUX_POSITION, frequencies=FREQUENCIES))
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="dielectric_sphere",
    description="Plane wave scattering off dielectric sphere (eps_r=4)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ex",
)
