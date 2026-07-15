from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


FREQUENCIES = (1.5e9,)


def build_scene() -> mw.Scene:
    scene = (
        base_scene()
        .add_structure(mw.Structure(geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.12), material=mw.Material(eps_r=4.0), name="sphere"))
        .add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.22),
                polarization="Ez",
                profile="ideal",
                source_time=mw.GaussianPulse(frequency=1.5e9, fwidth=0.5e9),
            )
        )
        .add_monitor(mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="flux_z_pos", axis="z", position=0.35, frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="flux_z_neg", axis="z", position=-0.35, frequencies=FREQUENCIES, normal_direction="-"))
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="dipole_dielectric_sphere",
    description="Ez point-dipole coupled to a dielectric sphere with a finite source gap",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ez",
    run_time_factor=18.0,
    maxwell_alignment={"source": {"profile": "ideal"}},
)
