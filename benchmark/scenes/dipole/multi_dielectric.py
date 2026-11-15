from __future__ import annotations

import math

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import DX, PML_LAYERS, SAFE_FLUX_POSITION, base_scene, with_plot_monitors


# Octave pair at half the original band on a dx/3 grid (bench-respec 2026-11-12): 12.2 cells per
# wavelength in the eps 9 box at 1 GHz (8.2 mm grid). At 2 GHz on the 24.6 mm grid it had 2.0
# (sin(k dx/2) = 1.55, beyond the Yee band edge) and the Maxwell 2 GHz flux moved by 3.6x under
# refinement (diag17). Lowering the band alone cannot work here: 10 cells per wavelength in eps 9
# on the 24.6 mm grid needs f <= 0.41 GHz, where the boxes sit within lambda0/2 of the absorber
# (box_a is 0.34 m from it). At 0.5 GHz lambda0/2 = 0.30 m. The absorber keeps its 0.2 m depth.
FREQUENCIES = (0.5e9, 1.0e9)


def _scene() -> mw.Scene:
    return base_scene().updated(grid=mw.GridSpec.uniform(DX / 3),
                                boundary=mw.BoundarySpec.pml(num_layers=3 * PML_LAYERS))


def build_scene() -> mw.Scene:
    scene = _scene()
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(geometry=mw.Box(position=(-0.2, 0.0, 0.0), size=(0.2, 0.2, 0.2)), material=mw.Material(eps_r=2.25), name="box_a"),
        mw.Structure(geometry=mw.Box(position=(0.2, 0.0, 0.0), size=(0.15, 0.15, 0.15)), material=mw.Material(eps_r=9.0), name="box_b"),
    ), sources=(
        *scene.sources,
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            # Sine carrier (phase pi/2): zero DC, so no static charge field keeps
            # the observation window from settling (benchmark-references-2026-11-11).
            source_time=mw.GaussianPulse(frequency=0.75e9, fwidth=0.5e9, phase=math.pi / 2),
        ),
    ), monitors=(
        *scene.monitors,
        mw.PlaneMonitor(name="field_xy", axis="z", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=FREQUENCIES),
        mw.FluxMonitor(name="flux_x_pos", axis="x", position=SAFE_FLUX_POSITION, frequencies=FREQUENCIES),
    ))
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="multi_dielectric",
    # 2x the default window (bench-respec 2026-11-12): 2x-runtime change 6.7e-4 field / 2.4e-3 flux
    # at 8x, 8.7e-5 / 4.2e-4 at 16x.
    run_time_factor=16.0,
    description="Two dielectric boxes, broadband dipole, multi-frequency",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xy",
    display_component="Ez",
    maxwell_alignment={"source": {"profile": "ideal"}},
)
