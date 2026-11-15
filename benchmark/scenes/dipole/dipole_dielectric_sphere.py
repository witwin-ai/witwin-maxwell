from __future__ import annotations

import math

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


FREQUENCIES = (1.5e9,)


def build_scene() -> mw.Scene:
    scene = base_scene()
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.12), material=mw.Material(eps_r=4.0), name="sphere"),
    ), sources=(
        *scene.sources,
        mw.PointDipole(
            position=(0.0, 0.0, -0.22),
            polarization="Ez",
            profile="ideal",
            # Sine carrier (phase pi/2): zero DC, so no static charge field keeps
            # the observation window from settling (benchmark-references-2026-11-11).
            source_time=mw.GaussianPulse(frequency=1.5e9, fwidth=0.5e9, phase=math.pi / 2),
        ),
    ), monitors=(
        *scene.monitors,
        mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES),
        mw.FluxMonitor(name="flux_z_pos", axis="z", position=0.35, frequencies=FREQUENCIES),
        mw.FluxMonitor(name="flux_z_neg", axis="z", position=-0.35, frequencies=FREQUENCIES, normal_direction="-"),
    ))
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
