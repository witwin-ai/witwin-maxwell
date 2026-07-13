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
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
        device="cpu",
    )


def with_standard_plane_monitor(
    scene: mw.Scene,
    *,
    name: str,
    axis: str,
    position: float,
    frequencies: tuple[float, ...],
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez"),
) -> mw.Scene:
    return scene.add_monitor(
        mw.PlaneMonitor(
            name=name,
            axis=axis,
            position=position,
            fields=fields,
            frequencies=frequencies,
        )
    )


def with_plot_monitors(scene: mw.Scene, *, frequencies: tuple[float, ...]) -> mw.Scene:
    for axis in ("x", "y", "z"):
        scene.add_monitor(
            mw.PlaneMonitor(
                name=f"plot_field_{axis}",
                axis=axis,
                position=0.0,
                fields=("Ex", "Ey", "Ez"),
                frequencies=frequencies,
            )
        )
    return scene


def with_standard_flux_monitors(scene: mw.Scene, *, frequencies: tuple[float, ...]) -> mw.Scene:
    scene.add_monitor(
        mw.FluxMonitor(name="flux_pos_z", axis="z", position=0.30, frequencies=frequencies)
    )
    scene.add_monitor(
        mw.FluxMonitor(
            name="flux_neg_z",
            axis="z",
            position=-0.30,
            frequencies=frequencies,
            normal_direction="-",
        )
    )
    return scene
