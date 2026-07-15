"""Grid and geometry coverage scenarios (primitive export paths + grid specs)."""

from __future__ import annotations

import numpy as np

import witwin.maxwell as mw
from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import HALF_SPAN
from benchmark.scenes.planned import FREQUENCIES, _make, _plane_scene, _scatter_scene

_C0 = 299_792_458.0


def _pentagon(radius: float = 0.18) -> tuple[tuple[float, float], ...]:
    """Regular 5-gon cross-section for the PolySlab analytic export path."""
    angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False) + 0.5 * np.pi
    return tuple(
        (float(radius * np.cos(angle)), float(radius * np.sin(angle))) for angle in angles
    )


def _graded_nodes() -> np.ndarray:
    """Symmetric non-uniformly spaced node array spanning [-HALF_SPAN, HALF_SPAN].

    Three segments per half: dl = 0.025 in the core, then 0.030 and 0.032 toward the
    absorber. Neighbouring-cell ratios stay <= 1.2 and the minimum step matches the
    uniform benchmark grid, so this stresses the custom-grid path without making the
    scene finer or larger than the baseline.
    """
    half = np.concatenate([
        np.linspace(0.0, 0.30, 13)[:-1],     # 12 cells, dl = 0.025
        np.linspace(0.30, 0.48, 7)[:-1],     # 6 cells,  dl = 0.030
        np.linspace(0.48, HALF_SPAN, 6),     # 5 cells,  dl = 0.032
    ])
    nodes = np.concatenate([-half[:0:-1], half])
    nodes[0] = -HALF_SPAN
    nodes[-1] = HALF_SPAN
    return nodes


def _nonuniform_grid() -> mw.GridSpec:
    nodes = _graded_nodes()
    return mw.GridSpec.custom(nodes, nodes.copy(), nodes.copy())


def _autogrid_slab_scene() -> mw.Scene:
    # Resolve the mesh against the vacuum wavelength; a low-index slab keeps the
    # refined transverse steps close to the 0.025 m baseline.
    return _plane_scene(
        material=mw.Material(eps_r=2.0),
        grid=mw.GridSpec.auto(
            min_steps_per_wavelength=5.0,
            wavelength=_C0 / FREQUENCIES[0],
            max_ratio=1.4,
        ),
    )


def _explicit_mesh_scene() -> mw.Scene:
    """User-supplied watertight triangle mesh, not an implicit primitive fallback."""
    vertices, faces = mw.Pyramid(
        position=(0.0, 0.0, -0.13),
        base_size=0.28,
        height=0.26,
        axis="z",
    ).to_mesh()
    mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")
    return _scatter_scene(mesh)


def _autogrid_override_refinement_scene() -> mw.Scene:
    """Auto grid with both an explicit local override and thin-layer refinement."""
    override_geometry = mw.Box(
        position=(0.0, 0.0, 0.0),
        size=(0.36, 0.36, 0.20),
    )
    grid = mw.GridSpec.auto(
        min_steps_per_wavelength=5.0,
        wavelength=_C0 / FREQUENCIES[0],
        max_ratio=1.4,
        override_structures=(
            mw.MeshOverrideStructure(geometry=override_geometry, dl=0.02),
        ),
        layer_refinement=mw.LayerRefinementSpec(min_cells=4, axes=("z",)),
    )
    return _plane_scene(material=mw.Material(eps_r=3.0), grid=grid)


GRID_GEOMETRY_COVERAGE_SCENARIOS = (
    _make(
        "cylinder_scatter",
        "grid_geometry",
        "analytic cylinder scattering",
        lambda: _scatter_scene(mw.Cylinder(position=(0, 0, 0), radius=0.14, height=0.24, axis="z")),
    ),
    _make(
        "cone_scatter",
        "grid_geometry",
        "sidewall-angle cone scattering",
        lambda: _scatter_scene(mw.Cone(position=(0, 0, -0.12), radius=0.15, height=0.24, axis="z")),
    ),
    _make(
        "ellipsoid_scatter",
        "grid_geometry",
        "anisotropic ellipsoid scattering",
        lambda: _scatter_scene(mw.Ellipsoid(position=(0, 0, 0), radii=(0.18, 0.12, 0.09))),
    ),
    _make(
        "pyramid_scatter",
        "grid_geometry",
        "pyramid mesh scattering",
        lambda: _scatter_scene(mw.Pyramid(position=(0, 0, -0.13), base_size=0.28, height=0.26, axis="z")),
    ),
    _make(
        "prism_scatter",
        "grid_geometry",
        "hexagonal prism scattering",
        lambda: _scatter_scene(mw.Prism(position=(0, 0, 0), radius=0.16, height=0.22, num_sides=6, axis="z")),
    ),
    _make(
        "hollow_box_scatter",
        "grid_geometry",
        "hollow-box shell scattering",
        lambda: _scatter_scene(mw.HollowBox(position=(0, 0, 0), outer_size=(0.32, 0.32, 0.32), inner_size=(0.22, 0.22, 0.22))),
    ),
    _make(
        "polyslab_pentagon",
        "grid_geometry",
        "pentagonal PolySlab scattering",
        lambda: _scatter_scene(mw.PolySlab(vertices=_pentagon(0.18), bounds=(-0.1, 0.1))),
    ),
    _make(
        "autogrid_slab",
        "grid_geometry",
        "auto-mesher dielectric slab",
        _autogrid_slab_scene,
    ),
    _make(
        "nonuniform_custom_grid",
        "grid_geometry",
        "graded custom-grid slab",
        lambda: _plane_scene(material=mw.Material(eps_r=3.0), grid=_nonuniform_grid()),
    ),
    _make(
        "anisotropic_uniform_grid",
        "grid_geometry",
        "per-axis uniform grid slab",
        lambda: _plane_scene(material=mw.Material(eps_r=3.0), grid=mw.GridSpec.anisotropic(0.032, 0.025, 0.020)),
    ),
    _make(
        "explicit_mesh_scatter",
        "grid_geometry",
        "user-supplied watertight triangle-mesh scattering",
        _explicit_mesh_scene,
    ),
    _make(
        "autogrid_override_refinement",
        "grid_geometry",
        "auto grid with local override and thin-layer refinement",
        _autogrid_override_refinement_scene,
    ),
)
