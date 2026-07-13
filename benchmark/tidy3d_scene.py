from __future__ import annotations


def benchmark_physical_bounds(scene) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    bounds = tuple(tuple(float(value) for value in axis_bounds) for axis_bounds in scene.domain.bounds)
    boundary = scene.boundary
    if boundary is None or boundary.kind != "pml" or int(boundary.num_layers) <= 0:
        return bounds

    spacings = scene.grid.min_spacing if scene.grid.is_custom else (scene.dx, scene.dy, scene.dz)
    trims = tuple(float(boundary.num_layers) * float(spacing) for spacing in spacings)
    physical_bounds = []
    for axis_bounds, trim in zip(bounds, trims):
        lo = axis_bounds[0] + trim
        hi = axis_bounds[1] - trim
        if not lo < hi:
            raise ValueError(
                "PML trim leaves no physical interior for benchmark comparison. "
                f"axis_bounds={axis_bounds}, trim={trim}"
            )
        physical_bounds.append((lo, hi))
    return tuple(physical_bounds)


def prepare_tidy3d_benchmark_scene(scene):
    # Keep the full simulation domain so Tidy3D sources and monitors stay
    # in the same physical/PML layout as the Maxwell run. Benchmark
    # comparisons crop to the physical interior separately.
    return scene.clone()
