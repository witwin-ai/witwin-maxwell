from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import SAFE_FLUX_POSITION, base_scene, with_plot_monitors


FREQUENCIES = (2.0e9,)


def build_scene() -> mw.Scene:
    scene = (
        base_scene()
        .add_structure(
            mw.Structure(
                geometry=mw.Cylinder(position=(0.0, 0.0, 0.0), radius=0.1, height=0.2, axis="z"),
                material=mw.Material.lorentz(eps_inf=2.0, delta_eps=3.0, resonance_frequency=2.0e9, gamma=0.05e9),
                name="resonator",
            )
        )
        .add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.2),
                polarization="Ez",
                profile="ideal",
                source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9),
            )
        )
        .add_monitor(mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES))
        .add_monitor(mw.FluxMonitor(name="radiated_flux", axis="z", position=SAFE_FLUX_POSITION, frequencies=FREQUENCIES))
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="lorentz_resonator",
    description="Lorentz-dispersive cylinder excited by dipole",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ez",
    run_time_factor=20.0,
    maxwell_alignment={"source": {"profile": "ideal"}},
    compare_flux=False,
)
