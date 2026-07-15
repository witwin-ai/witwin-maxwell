from __future__ import annotations

import witwin.maxwell as mw

from benchmark.scenes._common import HALF_SPAN, base_scene, with_plot_monitors


# P3-media benchmark scenarios share the meter-scale ``base_scene`` layout so
# their Tidy3D export lands on the same domain/PML footprint as the existing
# dipole/planewave families. A single low-GHz operating frequency keeps every
# dispersive/modulated coefficient in a regime the coarse benchmark grid can
# still resolve.
FREQUENCY = 2.0e9


def slab_scene(
    material: mw.Material,
    *,
    name: str,
    thickness: float = 0.1,
    frequencies: tuple[float, ...] = (FREQUENCY,),
    source_time=None,
    boundary: mw.BoundarySpec | None = None,
) -> mw.Scene:
    """A full-transverse slab of ``material`` under a +z plane wave.

    Reused by every P3-media benchmark scenario so the only difference between
    them is the material under test. A ``thickness`` of ``0.0`` produces the
    zero-size box a ``Medium2D`` sheet requires.
    """
    scene = base_scene()
    if boundary is not None:
        scene = mw.Scene(
            domain=scene.domain,
            grid=scene.grid,
            boundary=boundary,
            subpixel_samples=scene.subpixel,
            device="cpu",
        )
    scene = (
        scene
        .add_structure(
            mw.Structure(
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(2.0 * HALF_SPAN, 2.0 * HALF_SPAN, thickness)),
                material=material,
                name=name,
            )
        )
        .add_source(
            mw.PlaneWave(
                direction=(0.0, 0.0, 1.0),
                polarization=(1.0, 0.0, 0.0),
                source_time=source_time or mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9),
            )
        )
        .add_monitor(
            mw.FluxMonitor(name="reflected", axis="z", position=-0.3, frequencies=frequencies, normal_direction="-")
        )
        .add_monitor(mw.FluxMonitor(name="transmitted", axis="z", position=0.3, frequencies=frequencies))
    )
    return with_plot_monitors(scene, frequencies=frequencies)
