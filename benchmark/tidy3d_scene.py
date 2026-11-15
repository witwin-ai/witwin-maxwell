from __future__ import annotations


def benchmark_physical_bounds(scene) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return tuple(
        tuple(float(value) for value in axis_bounds)
        for axis_bounds in scene.domain.bounds
    )


def benchmark_reference_bounds(scene) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Physical bounds of the reference domain: the mirrored domain of a face-symmetric scene."""
    from witwin.maxwell.adapters.tidy3d import mirrored_domain_bounds

    return mirrored_domain_bounds(scene) if scene.has_symmetry else benchmark_physical_bounds(scene)


def benchmark_reference_scene(scene):
    """The scene the reference export describes: the explicit mirrored twin of a face-symmetric scene."""
    from witwin.maxwell.adapters.tidy3d import mirrored_twin_scene

    return mirrored_twin_scene(scene) if scene.has_symmetry else scene


def benchmark_reference_domain(scene):
    """The Domain the reference export describes (the scene's own Domain without symmetry)."""
    import witwin.maxwell as mw

    return mw.Domain(bounds=benchmark_reference_bounds(scene)) if scene.has_symmetry else scene.domain


def prepare_tidy3d_benchmark_scene(scene):
    # Domain.bounds is physical in both solvers; each runtime appends PML cells.
    return scene
