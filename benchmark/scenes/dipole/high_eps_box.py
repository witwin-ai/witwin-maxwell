from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import base_scene, with_plot_monitors


FREQUENCIES = (1.5e9,)


def build_scene() -> mw.Scene:
    scene = (
        base_scene()
        .add_structure(mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)), material=mw.Material(eps_r=12.0), name="si_box"))
        .add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.2),
                polarization="Ez",
                profile="ideal",
                source_time=mw.GaussianPulse(frequency=1.5e9, fwidth=0.5e9),
            )
        )
        .add_monitor(mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="flux_z", axis="z", position=0.3, frequencies=FREQUENCIES))
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="high_eps_box",
    description="Dipole near high-eps box (eps_r=12, silicon-like)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ez",
    maxwell_alignment={"source": {"profile": "ideal"}},
)
