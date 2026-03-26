from .plots import (
    extract_orthogonal_slice,
    plot_cross_section_panels,
    plot_orthogonal_views,
    plot_slice_image,
    visualize_material_slice,
    visualize_slice,
)

__all__ = [
    "build_fdtd_pyvista_grid",
    "extract_orthogonal_slice",
    "plot_cross_section_panels",
    "plot_orthogonal_views",
    "plot_slice_image",
    "show_pyvista_solution",
    "visualize_material_slice",
    "visualize_slice",
]


def __getattr__(name):
    if name in ("build_fdtd_pyvista_grid", "show_pyvista_solution"):
        from .interactive import build_fdtd_pyvista_grid, show_pyvista_solution
        globals()["build_fdtd_pyvista_grid"] = build_fdtd_pyvista_grid
        globals()["show_pyvista_solution"] = show_pyvista_solution
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
