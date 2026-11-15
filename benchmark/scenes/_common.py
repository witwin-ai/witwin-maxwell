from __future__ import annotations

import witwin.maxwell as mw


DX = 0.025
HALF_SPAN = 0.64
PML_LAYERS = 8
PHYSICAL_HALF_SPAN = HALF_SPAN
FLUX_MONITOR_MARGIN = 4 * DX
SAFE_FLUX_POSITION = max(DX, PHYSICAL_HALF_SPAN - FLUX_MONITOR_MARGIN)


def base_scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-HALF_SPAN, HALF_SPAN),
                (-HALF_SPAN, HALF_SPAN),
                (-HALF_SPAN, HALF_SPAN),
            )
        ),
        grid=mw.GridSpec.uniform(DX),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        # Staircase PEC edge handling is the benchmark default. Conformal PEC
        # implements the partial fill as a per-step open-fraction factor on the E
        # update, i.e. an effective conductivity eps*fill/dt on every cut edge, so a
        # lossless PEC scatterer picks up spurious absorption (measured: a closed PEC
        # cavity holding a PEC sphere retains 0.45 of its energy after 5200
        # source-free steps under conformal versus 1.00 under staircase). Scenes that
        # want conformal must opt in per scene with an explicit SubpixelSpec.
        # Dielectric scenes are unaffected either way (no PEC material).
        subpixel=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"),
    )


def kept_half(scene: mw.Scene) -> mw.Scene:
    """The kept part of ``scene`` folded at the center of every symmetry axis.

    ``scene.symmetry`` names the planes; each one moves to the center of its axis
    (a low-face plane keeps the upper half), so the explicit mirrored domain of the
    returned scene is ``scene`` itself and its reference is that of ``scene``.
    """
    bounds = []
    for (lower, upper), entry in zip(scene.domain.bounds, scene.symmetry):
        middle = 0.5 * (float(lower) + float(upper))
        if entry is None:
            bounds.append((float(lower), float(upper)))
        else:
            bounds.append((middle, float(upper)) if entry[1] == "low" else (float(lower), middle))
    return scene.updated(domain=mw.Domain(bounds=tuple(bounds)))


def with_standard_plane_monitor(
    scene: mw.Scene,
    *,
    name: str,
    axis: str,
    position: float,
    frequencies: tuple[float, ...],
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez"),
) -> mw.Scene:
    return scene.updated(monitors=(
        *scene.monitors,
        mw.PlaneMonitor(
            name=name,
            axis=axis,
            position=position,
            fields=fields,
            frequencies=frequencies,
        ),
    ))


def with_plot_monitors(scene: mw.Scene, *, frequencies: tuple[float, ...]) -> mw.Scene:
    scene = scene.updated(monitors=(
        *scene.monitors,
        *(mw.PlaneMonitor(
              name=f"plot_field_{axis}",
              axis=axis,
              position=0.0,
              fields=("Ex", "Ey", "Ez"),
              frequencies=frequencies,
          ) for axis in ("x", "y", "z")),
    ))
    return scene


def with_standard_flux_monitors(scene: mw.Scene, *, frequencies: tuple[float, ...]) -> mw.Scene:
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.FluxMonitor(name="flux_pos_z", axis="z", position=0.30, frequencies=frequencies),
        mw.FluxMonitor(
            name="flux_neg_z",
            axis="z",
            position=-0.30,
            frequencies=frequencies,
            normal_direction="-",
        ),
    ))
    return scene
