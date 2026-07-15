from __future__ import annotations


def benchmark_physical_bounds(scene) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return tuple(
        tuple(float(value) for value in axis_bounds)
        for axis_bounds in scene.domain.bounds
    )


def prepare_tidy3d_benchmark_scene(scene):
    # Domain.bounds is physical in both solvers; each runtime appends PML cells.
    return scene.clone()
