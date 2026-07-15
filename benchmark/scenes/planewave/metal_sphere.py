from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import SAFE_FLUX_POSITION, base_scene, with_plot_monitors


FREQUENCIES = (1.5e9,)


def build_scene() -> mw.Scene:
    scene = (
        base_scene()
        .add_structure(
            mw.Structure(
                geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.15),
                material=mw.Material.drude(eps_inf=1.0, plasma_frequency=5.0e9, gamma=0.1e9),
                name="metal_sphere",
            )
        )
        .add_source(
            mw.PlaneWave(
                direction=(0.0, 0.0, 1.0),
                polarization=(1.0, 0.0, 0.0),
                source_time=mw.GaussianPulse(frequency=1.5e9, fwidth=0.5e9),
            )
        )
        .add_monitor(mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="forward_flux", axis="z", position=SAFE_FLUX_POSITION, frequencies=FREQUENCIES))
        .add_monitor(
            mw.FluxMonitor(
                name="backward_flux",
                axis="z",
                position=-SAFE_FLUX_POSITION,
                frequencies=FREQUENCIES,
                normal_direction="-",
            )
        )
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="metal_sphere",
    description="Plane wave scattering off Drude metal sphere",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ex",
    run_time_factor=20.0,
)
