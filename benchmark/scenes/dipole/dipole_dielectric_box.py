from __future__ import annotations

import math

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


# 0.5 GHz (bench-respec 2026-11-12): 12.2 cells per wavelength in the eps 4 box on the 24.6 mm
# grid. At 1.5 GHz it had 4.1 (sin(k dx/2) = 0.77) and the Maxwell -z flux moved 45% under
# refinement (diag17). Pulse scaled by 1/3 with fwidth/f unchanged.
FREQUENCIES = (0.5e9,)


def build_scene() -> mw.Scene:
    scene = base_scene()
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.24, 0.24)), material=mw.Material(eps_r=4.0), name="dielectric_box"),
    ), sources=(
        *scene.sources,
        mw.PointDipole(
            position=(0.0, 0.0, -0.20),
            polarization="Ez",
            profile="ideal",
            # Sine carrier (phase pi/2): zero DC, so no static charge field keeps
            # the observation window from settling (benchmark-references-2026-11-11).
            source_time=mw.GaussianPulse(frequency=0.5e9, fwidth=0.5e9 / 3, phase=math.pi / 2),
            name="dipole",
        ),
    ), monitors=(
        *scene.monitors,
        mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES),
        mw.FluxMonitor(name="flux_pos_z", axis="z", position=0.30, frequencies=FREQUENCIES),
        mw.FluxMonitor(name="flux_neg_z", axis="z", position=-0.30, frequencies=FREQUENCIES, normal_direction="-"),
    ))
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="dipole_dielectric_box",
    # Default 8x window: at the re-specified band the 2x-runtime change is below 4e-4
    # (field and flux; bench-respec 2026-11-12).
    run_time_factor=8.0,
    description="Ez dipole coupled to a dielectric box.",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ez",
)
