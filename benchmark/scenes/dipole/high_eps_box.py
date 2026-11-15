from __future__ import annotations

import math

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import DX, PML_LAYERS, base_scene, with_plot_monitors


# 0.6 GHz on a dx/2 grid (bench-respec 2026-11-12): 11.7 cells per wavelength in the eps 12 box
# (12.3 mm grid). At 1.5 GHz on the 24.6 mm grid it had 2.3 (sin(k dx/2) = 1.34, beyond the Yee
# band edge), the box was 3.5 medium wavelengths across and needed a 512x window; resolving it
# there would cost about 0.8 credit per reference (diag17). 0.3 GHz on the 24.6 mm grid would put
# the dipole 0.44 m from the absorber, inside lambda0/2 = 0.5 m; at 0.6 GHz lambda0/2 = 0.25 m.
# Pulse scaled by 0.4 with fwidth/f unchanged; the absorber keeps its 0.2 m depth.
FREQUENCIES = (0.6e9,)


def _scene() -> mw.Scene:
    return base_scene().updated(grid=mw.GridSpec.uniform(DX / 2),
                                boundary=mw.BoundarySpec.pml(num_layers=2 * PML_LAYERS))


def build_scene() -> mw.Scene:
    scene = _scene()
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)), material=mw.Material(eps_r=12.0), name="si_box"),
    ), sources=(
        *scene.sources,
        mw.PointDipole(
            position=(0.0, 0.0, -0.2),
            polarization="Ez",
            profile="ideal",
            # Sine carrier (phase pi/2): zero DC, so no static charge field keeps
            # the observation window from settling (benchmark-references-2026-11-11).
            source_time=mw.GaussianPulse(frequency=0.6e9, fwidth=0.2e9, phase=math.pi / 2),
        ),
    ), monitors=(
        *scene.monitors,
        mw.PlaneMonitor(name="field_xz", axis="y", position=0.0, fields=("Ex", "Ez", "Hy"), frequencies=FREQUENCIES),
        mw.FluxMonitor(name="flux_z", axis="z", position=0.3, frequencies=FREQUENCIES),
    ))
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


SCENARIO = ScenarioDefinition(
    name="high_eps_box",
    # 8x the default window (bench-respec 2026-11-12): the eps 12 box rings down; 2x-runtime change
    # 1.6e-2 field / 0.51 flux at 8x, 1.5e-4 / 4.1e-3 at 32x, 1.5e-5 / 9.4e-5 at 64x.
    run_time_factor=64.0,
    description="Dipole near high-eps box (eps_r=12, silicon-like)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="field_xz",
    display_component="Ez",
    maxwell_alignment={"source": {"profile": "ideal"}},
)
