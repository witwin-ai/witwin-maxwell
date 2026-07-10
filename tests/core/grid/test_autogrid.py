from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.meshing import mesh_axis
from witwin.maxwell.fdtd.meshing.autogrid import (
    _collect_boundaries,
    _interval_targets,
    _uniformize_boundary_band,
)
from witwin.maxwell.scene import prepare_scene

_C0 = 299_792_458.0
_WAVELENGTH = 0.3
_MSW = 10
_RATIO = 1.4
_BASE_DL = _WAVELENGTH / _MSW
_TOL = 1.0 + 1e-9


def _check_invariants(nodes, lo, hi, index_regions=(), override_regions=(), max_ratio=_RATIO):
    """Assert the three meshing invariants: exact coverage + face snapping,
    per-interval target honored, global adjacent-cell ratio bounded."""
    span = hi - lo
    assert nodes[0] == lo and nodes[-1] == hi
    sizes = np.diff(nodes)
    assert np.all(sizes > 0.0)
    # Absolute float64 node coordinates carry ULP-scale rounding, and cumsum
    # node materialization accumulates up to ~N ULP into an interval's last
    # cell, so the realized ratio/target slack scales with N * ULP / min cell.
    tol = _TOL + 4.0 * sizes.size * np.spacing(max(abs(lo), abs(hi))) / sizes.min()

    for r_lo, r_hi, _ in tuple(index_regions) + tuple(override_regions):
        for face in (r_lo, r_hi):
            clamped = min(max(face, lo), hi)
            assert np.abs(nodes - clamped).min() <= 5e-9 * span, face

    ratios = sizes[1:] / sizes[:-1]
    assert ratios.max() <= max_ratio * tol
    assert ratios.min() >= 1.0 / (max_ratio * tol)

    midpoints = 0.5 * (nodes[:-1] + nodes[1:])
    targets = np.full(sizes.size, _BASE_DL)
    constraints = [
        (r_lo, r_hi, _BASE_DL / max(index, 1.0)) for r_lo, r_hi, index in index_regions
    ]
    constraints += [(r_lo, r_hi, dl) for r_lo, r_hi, dl in override_regions]
    for r_lo, r_hi, dl in constraints:
        inside = (midpoints > r_lo + 2e-9 * span) & (midpoints < r_hi - 2e-9 * span)
        targets[inside] = np.minimum(targets[inside], dl)
    assert np.all(sizes <= targets * tol)


def _cell_count_bound(lo, hi, index_regions, override_regions, layer_min_cells, max_ratio):
    """Generous upper bound on the axis cell count: per interval, the target
    fill plus two geometric ramps plus a small grading constant, doubled."""
    faces = [f for region in tuple(index_regions) + tuple(override_regions) for f in region[:2]]
    boundaries = _collect_boundaries(lo, hi, faces)
    constraints = [
        (r_lo, r_hi, _BASE_DL / max(index, 1.0)) for r_lo, r_hi, index in index_regions
    ]
    constraints += [(r_lo, r_hi, dl) for r_lo, r_hi, dl in override_regions]
    targets = _interval_targets(boundaries, _BASE_DL, constraints, layer_min_cells)
    lengths = np.diff(boundaries)
    log_r = np.log(max_ratio)
    ramps = 2.0 * (np.log(targets / targets.min()) + np.log(2.0)) / log_r
    const = (2.0 * max_ratio + 1.0) / (max_ratio - 1.0) + 2.0 * max_ratio + 4.0
    return 2.0 * float(np.sum(lengths / targets + ramps + const))


def _axis(**kwargs):
    return mesh_axis(
        -0.5,
        0.5,
        wavelength=_WAVELENGTH,
        min_steps_per_wavelength=_MSW,
        max_ratio=_RATIO,
        **kwargs,
    )


def test_structure_faces_snap_and_domain_coverage():
    regions = [(-0.13, 0.071, 2.0), (0.2, 0.9, 1.5)]
    nodes = _axis(index_regions=regions)
    _check_invariants(nodes, -0.5, 0.5, index_regions=regions)


def test_high_index_region_honors_target_dl():
    regions = [(-0.05, 0.05, 3.5)]
    nodes = _axis(index_regions=regions)
    sizes = np.diff(nodes)
    inside = (nodes[:-1] >= -0.05 - 1e-12) & (nodes[1:] <= 0.05 + 1e-12)
    fine_dl = _WAVELENGTH / (3.5 * _MSW)
    assert sizes[inside].max() <= fine_dl * _TOL
    # Background grades back up to substantially coarser cells.
    assert sizes[~inside].max() > 2.0 * fine_dl
    _check_invariants(nodes, -0.5, 0.5, index_regions=regions)


@pytest.mark.parametrize("seed", list(range(8)))
def test_global_ratio_invariant_randomized_multi_structure_axis(seed):
    rng = np.random.default_rng(seed)
    lo = float(rng.uniform(-10.0, 10.0))
    hi = lo + 1.0
    max_ratio = float(rng.uniform(1.05, 1.6))
    regions = []
    for _ in range(rng.integers(3, 9)):
        r_lo = lo + rng.uniform(-0.1, 1.05)
        regions.append((r_lo, r_lo + rng.uniform(1e-5, 0.4), rng.uniform(1.0, 4.0)))
    overrides = []
    for _ in range(rng.integers(0, 3)):
        r_lo = lo + rng.uniform(0.0, 0.9)
        overrides.append((r_lo, r_lo + rng.uniform(0.001, 0.2), rng.uniform(1e-4, 0.02)))
    layer_min_cells = int(rng.integers(1, 7)) if rng.random() < 0.5 else None
    nodes = mesh_axis(
        lo, hi, wavelength=_WAVELENGTH, min_steps_per_wavelength=_MSW,
        max_ratio=max_ratio, index_regions=regions, override_regions=overrides,
        layer_min_cells=layer_min_cells,
    )
    _check_invariants(
        nodes, lo, hi, index_regions=regions, override_regions=overrides, max_ratio=max_ratio
    )
    # Cell-count sanity: a graded mesh must stay near the per-interval optimum.
    assert nodes.size - 1 <= _cell_count_bound(
        lo, hi, regions, overrides, layer_min_cells, max_ratio
    )


def test_thin_layer_neighbor_grades_up_without_cell_explosion():
    # Regression: a thin refined layer must not pin its short neighbor to a
    # uniformly fine fill when the full ramp to the target does not fit; the
    # graded profile lowers its plateau instead (thin film, gentle grading).
    regions = [(0.9, 0.901, 3.5)]
    nodes = mesh_axis(
        0.0, 1.2, wavelength=_WAVELENGTH, min_steps_per_wavelength=_MSW,
        max_ratio=1.2, index_regions=regions, layer_min_cells=2,
    )
    _check_invariants(nodes, 0.0, 1.2, index_regions=regions, max_ratio=1.2)
    sizes = np.diff(nodes)
    right = sizes[nodes[:-1] >= 0.901]
    assert nodes.size < 150  # pre-fix: ~550 nodes, right interval ~500 cells
    assert right.max() > 0.01  # pre-fix uniform fallback stays at ~6e-4


def test_absurdly_small_override_dl_raises_clear_error():
    with pytest.raises(ValueError, match="cells"):
        _axis(override_regions=[(-0.1, 0.1, 1e-30)])


def test_override_region_forces_finer_local_mesh():
    overrides = [(0.1, 0.2, 0.004)]
    nodes = _axis(override_regions=overrides)
    sizes = np.diff(nodes)
    inside = (nodes[:-1] >= 0.1 - 1e-12) & (nodes[1:] <= 0.2 + 1e-12)
    assert sizes[inside].max() <= 0.004 * _TOL
    assert sizes[~inside].max() > 0.01
    _check_invariants(nodes, -0.5, 0.5, override_regions=overrides)


def test_layer_refinement_forces_min_cells_across_thin_layer():
    thin = [(0.5, 0.504, 1.0)]
    coarse = mesh_axis(
        0.0, 1.0, wavelength=_WAVELENGTH, min_steps_per_wavelength=_MSW,
        max_ratio=_RATIO, index_regions=thin,
    )
    refined = mesh_axis(
        0.0, 1.0, wavelength=_WAVELENGTH, min_steps_per_wavelength=_MSW,
        max_ratio=_RATIO, index_regions=thin, layer_min_cells=4,
    )

    def _layer_cells(nodes):
        inside = (nodes >= 0.5 - 1e-12) & (nodes <= 0.504 + 1e-12)
        return int(np.count_nonzero(inside)) - 1

    assert _layer_cells(coarse) == 1
    assert _layer_cells(refined) >= 4
    sizes = np.diff(refined)
    ratios = sizes[1:] / sizes[:-1]
    assert ratios.max() <= _RATIO * _TOL and ratios.min() >= 1.0 / (_RATIO * _TOL)


def test_gridspec_auto_validation():
    with pytest.raises(ValueError, match="max_ratio must be > 1"):
        mw.GridSpec.auto(max_ratio=1.0)
    with pytest.raises(ValueError, match="min_steps_per_wavelength"):
        mw.GridSpec.auto(min_steps_per_wavelength=0)
    with pytest.raises(ValueError, match="wavelength"):
        mw.GridSpec.auto(wavelength=-0.1)
    with pytest.raises(TypeError, match="MeshOverrideStructure"):
        mw.GridSpec.auto(override_structures=(object(),))
    with pytest.raises(TypeError, match="LayerRefinementSpec"):
        mw.GridSpec.auto(layer_refinement=object())

    grid = mw.GridSpec.auto()
    assert grid.is_auto is True and grid.is_custom is False
    with pytest.raises(ValueError, match="no scalar spacing"):
        _ = grid.spacing
    with pytest.raises(ValueError, match="no scalar spacing"):
        _ = grid.is_uniform
    with pytest.raises(ValueError, match="prepare"):
        _ = grid.min_spacing


def test_mesh_override_and_layer_refinement_spec_validation():
    box = mw.Box(position=(0.0, 0.0, 0.0), size=(0.1, 0.1, 0.1))
    override = mw.MeshOverrideStructure(geometry=box, dl=0.01)
    assert override.dl == (0.01, 0.01, 0.01)
    assert mw.MeshOverrideStructure(geometry=box, dl=(0.01, 0.02, 0.03)).dl == (0.01, 0.02, 0.03)
    with pytest.raises(ValueError, match="> 0"):
        mw.MeshOverrideStructure(geometry=box, dl=-0.01)
    with pytest.raises(ValueError, match="scalar or"):
        mw.MeshOverrideStructure(geometry=box, dl=(0.01, 0.02))

    layer = mw.LayerRefinementSpec(min_cells=3)
    assert layer.covers("x") and layer.covers("z")
    layer_xy = mw.LayerRefinementSpec(min_cells=2, axes=("x", "y"))
    assert layer_xy.covers("y") and not layer_xy.covers("z")
    with pytest.raises(ValueError, match="min_cells"):
        mw.LayerRefinementSpec(min_cells=0)
    with pytest.raises(ValueError, match="axes"):
        mw.LayerRefinementSpec(min_cells=2, axes=("w",))


def _auto_scene(grid=None, device="cpu"):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=grid or mw.GridSpec.auto(),
        device=device,
    )


def test_scene_scalar_spacing_raises_on_auto_grid():
    scene = _auto_scene()
    with pytest.raises(ValueError, match="scalar spacing undefined"):
        _ = scene.dx


def test_wavelength_derived_from_highest_source_frequency():
    scene = _auto_scene()
    scene.add_source(mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez",
                                    source_time=mw.CW(frequency=1e9)))
    scene.add_source(mw.PointDipole(position=(0.1, 0.0, 0.0), polarization="Ez",
                                    source_time=mw.CW(frequency=2e9)))
    prepared = prepare_scene(scene)
    assert prepared.grid.is_custom is True
    # Highest frequency (2 GHz) sets the meshing wavelength.
    assert prepared.dx_primal64.max() <= _C0 / 2e9 / _MSW * _TOL
    # The public scene keeps its unresolved auto grid.
    assert scene.grid.is_auto is True


def test_broadband_source_meshes_for_characteristic_frequency():
    """A broadband pulse must mesh for its characteristic (upper spectral)
    frequency, not just its center frequency, matching FDTD's auto_dt."""
    scene = _auto_scene()
    scene.add_source(mw.PointDipole(
        position=(0.0, 0.0, 0.0), polarization="Ez",
        source_time=mw.GaussianPulse(frequency=2e9, fwidth=0.5e9),
    ))
    prepared = prepare_scene(scene)
    # GaussianPulse content extends to frequency + 3 * fwidth = 3.5 GHz.
    assert prepared.dx_primal64.max() <= _C0 / 3.5e9 / _MSW * _TOL


def test_far_offset_domain_keeps_face_snap_and_refinement():
    """Regression: the face-snap tolerance must not scale linearly with the
    coordinate offset, or a domain far from the origin silently loses its
    structure faces and the high-index refinement with them."""
    lo, hi = 1e6, 1e6 + 1.0
    regions = [(lo + 0.3, lo + 0.7, 3.5)]
    nodes = mesh_axis(
        lo, hi, wavelength=_WAVELENGTH, min_steps_per_wavelength=_MSW,
        max_ratio=_RATIO, index_regions=regions,
    )
    _check_invariants(nodes, lo, hi, index_regions=regions)


def test_explicit_wavelength_overrides_source_derivation():
    scene = _auto_scene(grid=mw.GridSpec.auto(wavelength=0.3))
    scene.add_source(mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez",
                                    source_time=mw.CW(frequency=20e9)))
    prepared = prepare_scene(scene)
    assert prepared.dx_primal64.max() > 0.02  # 0.3 / 10, not C0/20e9/10


def test_auto_grid_without_frequency_source_raises():
    with pytest.raises(ValueError, match="wavelength"):
        prepare_scene(_auto_scene())


def test_fdfd_rejects_auto_grid():
    scene = _auto_scene(grid=mw.GridSpec.auto(wavelength=0.3))
    simulation = mw.Simulation.fdfd(scene, frequency=1e9)
    with pytest.raises(NotImplementedError, match="FDFD does not support nonuniform"):
        simulation.prepare()


def test_auto_scene_layer_refinement_and_override_resolution():
    scene = _auto_scene(
        grid=mw.GridSpec.auto(
            wavelength=0.3,
            override_structures=(
                mw.MeshOverrideStructure(
                    geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
                    dl=0.005,
                ),
            ),
            layer_refinement=mw.LayerRefinementSpec(min_cells=2),
        )
    )
    prepared = prepare_scene(scene)
    x = prepared.x_nodes64
    sizes = np.diff(x)
    inside = (x[:-1] >= -0.1 - 1e-6) & (x[1:] <= 0.1 + 1e-6)
    assert sizes[inside].max() <= 0.005 * _TOL
    assert sizes[~inside].max() > 0.01


def test_material_region_refines_and_snaps_auto_grid():
    scene = _auto_scene(grid=mw.GridSpec.auto(wavelength=_WAVELENGTH))
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
            density=torch.rand(8, 8, 8),
            eps_bounds=(1.0, 12.25),
        )
    )
    prepared = prepare_scene(scene)
    x = prepared.x_nodes64
    dx = np.diff(x)
    inside = (x[:-1] >= -0.1 - 1e-7) & (x[1:] <= 0.1 + 1e-7)
    # eps upper bound 12.25 -> n = 3.5 sets the in-region target step.
    assert dx[inside].max() <= _BASE_DL / 3.5 * _TOL * (1.0 + 1e-6)
    assert dx[~inside].max() > 2.0 * _BASE_DL / 3.5
    # Region faces snap to cell boundaries (up to geometry AABB rounding).
    for face in (-0.1, 0.1):
        assert np.abs(x - face).min() <= 1e-7


def test_high_index_waveguide_uses_substantially_fewer_cells():
    wavelength = _C0 / 2e9
    core_dl = wavelength / (3.5 * _MSW)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.auto(min_steps_per_wavelength=_MSW),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.2, 0.06, 0.06)),
            material=mw.Material(eps_r=3.5**2),
            name="core",
        )
    )
    scene.add_source(mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez",
                                    source_time=mw.CW(frequency=2e9)))
    prepared = prepare_scene(scene)

    # Core cells honor the high-index target along the transverse axes.
    y = prepared.y_nodes64
    dy = np.diff(y)
    core = (y[:-1] >= -0.03 - 1e-6) & (y[1:] <= 0.03 + 1e-6)
    assert dy[core].max() <= core_dl * _TOL * (1.0 + 1e-6)
    # Background grades up to coarser cells than the core target.
    assert dy[~core].max() > 2.0 * core_dl

    # Total cell count is substantially below the uniform-fine equivalent.
    uniform_fine = (
        int(np.ceil(1.2 / core_dl))
        * int(np.ceil(0.6 / core_dl))
        * int(np.ceil(0.6 / core_dl))
    )
    assert prepared.N_total < 0.35 * uniform_fine


@pytest.mark.parametrize(
    "lo, hi, inset",
    [
        (-0.03, 0.03, 1e-8),      # 0.03 rounds inward under float32 (~6.7e-10)
        (99.97, 100.03, 1e-5),    # offset domain: float32 ULP ~7.6e-6 at |x|~100
    ],
)
def test_domain_spanning_face_does_not_spawn_degenerate_cells(lo, hi, inset):
    """Regression: a structure face on (or a float-hair inside) a domain boundary
    must not spawn a sub-cell sliver that the ratio fixpoint fans into near-zero
    cells (which would crush the Courant limit). ``to_mesh`` yields float32 AABBs,
    so a nominally-on-boundary face can land ~1e-7*|coord| inside; the snap
    tolerance scales with coordinate magnitude to cover offset domains too."""
    index = 3.5
    fine_dl = _WAVELENGTH / (index * _MSW)
    span = hi - lo
    # Whole-axis region whose faces land ``inset`` inside the domain edges.
    regions = [(lo + inset, hi - inset, index)]
    nodes = mesh_axis(
        lo, hi, wavelength=_WAVELENGTH, min_steps_per_wavelength=_MSW,
        max_ratio=_RATIO, index_regions=regions,
    )
    sizes = np.diff(nodes)
    target_count = int(np.ceil(span / fine_dl))
    assert nodes[0] == lo and nodes[-1] == hi
    # No degenerate cells: the smallest cell stays within a small factor of the
    # clean uniform fill instead of collapsing to ~0.
    assert sizes.min() > 0.5 * span / target_count
    # The near-boundary faces snapped away, so the refinement is still honored
    # across the full span and the count is the single-interval fill.
    assert sizes.max() <= fine_dl * _TOL
    assert nodes.size - 1 <= target_count + 3


def test_full_span_slab_scene_has_healthy_transverse_mesh():
    """Scene-level regression through the real float32 ``to_mesh`` AABB path: a
    slab spanning the full transverse extent (its x/y faces coincide with the
    domain boundary) must mesh cleanly, not explode into near-zero cells."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.16, 0.16), (-0.03, 0.03), (-0.16, 0.16))),
        grid=mw.GridSpec.auto(min_steps_per_wavelength=_MSW, wavelength=0.15),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.32, 0.06, 0.12)),
            material=mw.Material(eps_r=12.0),
        )
    )
    prepared = prepare_scene(scene)
    fine_dl = 0.15 / (np.sqrt(12.0) * _MSW)
    # x and y span the full domain -> uniformly at the box target, no slivers.
    for nodes in (prepared.x_nodes64, prepared.y_nodes64):
        d = np.diff(nodes)
        assert d.min() > 0.5 * fine_dl
        assert d.max() <= fine_dl * _TOL
    assert prepared.y_nodes64.size - 1 <= np.ceil(0.06 / fine_dl) + 3


def test_uniformize_boundary_band_forces_uniform_edges():
    nodes = np.array([0.0, 0.001, 0.003, 0.007, 0.015, 0.03, 0.06, 0.10, 0.15, 0.20])
    edges = np.array([0.0, 0.20])
    out = _uniformize_boundary_band(nodes, edges, 3, 3)
    dl = np.diff(out)
    assert np.allclose(dl[:3], dl[0])       # low band uniform
    assert np.allclose(dl[-3:], dl[-1])     # high band uniform
    assert out[0] == nodes[0] and out[-1] == nodes[-1]     # domain edges fixed
    assert np.allclose(out[3:-3], nodes[3:-3])             # interior untouched
    # Bands that would overlap on a too-thin axis leave the mesh unchanged.
    assert np.array_equal(_uniformize_boundary_band(nodes, edges, 5, 5), nodes)
    # No PML on a face -> no forced band.
    assert np.array_equal(_uniformize_boundary_band(nodes, edges, 0, 0), nodes)


def test_uniformize_boundary_band_preserves_hard_faces():
    """A structure face inside the absorber band stays a node: the band is
    re-spaced piecewise on each side of the face, not across it."""
    nodes = np.array([0.0, 0.001, 0.003, 0.007, 0.015, 0.03, 0.06, 0.10, 0.15, 0.20])
    boundaries = np.array([0.0, 0.15, 0.20])  # hard face at 0.15, 2nd-to-last node
    out = _uniformize_boundary_band(nodes, boundaries, 0, 4)
    dl = np.diff(out)
    assert out[-2] == 0.15                   # face stayed snapped
    assert np.allclose(dl[-4:-1], dl[-4])    # uniform up to the face
    assert np.allclose(out[:-4], nodes[:-4]) # interior untouched


def test_autogrid_uniform_cells_under_pml_faces():
    """The absorber sees a constant-step band: the outermost ``num_layers`` cells
    under every PML face are uniform between hard faces, even when a nearby
    structure would let a graded ramp intrude into the absorber. Uniformization
    must not undo face snapping or the high-index target inside the band."""
    NL = 8
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.auto(min_steps_per_wavelength=_MSW, wavelength=_WAVELENGTH),
        boundary=mw.BoundarySpec.pml(num_layers=NL), device="cpu",
    )
    # High-index block hugging the high-z edge; its high-z face (0.45) lands
    # inside the absorber band, its x/y faces are interior.
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0.0, 0.0, 0.32), size=(0.2, 0.2, 0.26)),
        material=mw.Material(eps_r=12.0),
    ))
    prepared = prepare_scene(scene)
    # No face inside the x/y bands -> fully uniform absorber bands.
    for nodes in (prepared.x_nodes64, prepared.y_nodes64):
        dl = np.diff(nodes)
        assert np.allclose(dl[:NL], dl[0])      # low absorber band uniform
        assert np.allclose(dl[-NL:], dl[-1])    # high absorber band uniform
    # The z face at 0.45 stays snapped and splits the high band in two
    # uniform segments; the high-index cells keep honoring their target.
    z = prepared.z_nodes64
    dz = np.diff(z)
    assert np.allclose(dz[:NL], dz[0])
    face = int(np.argmin(np.abs(z - 0.45)))
    assert abs(z[face] - 0.45) <= 1e-7          # float32 AABB rounding only
    assert z.size - 1 - NL < face < z.size - 1  # face is inside the band
    assert np.allclose(dz[-NL:face], dz[-NL])   # uniform below the face
    assert np.allclose(dz[face:], dz[-1])       # uniform above the face
    fine_dl = _WAVELENGTH / (np.sqrt(12.0) * _MSW)
    inside = (z[:-1] >= 0.19 - 1e-7) & (z[1:] <= 0.45 + 1e-7)
    assert dz[inside].max() <= fine_dl * _TOL
    # The band is genuinely reshaped: without uniformization the mesh is not
    # uniform in the high-z absorber, so the assertions above exercise the fix.
    raw = mesh_axis(-0.5, 0.5, wavelength=_WAVELENGTH, min_steps_per_wavelength=_MSW,
                    max_ratio=_RATIO, index_regions=[(0.19, 0.45, 12.0 ** 0.5)])
    assert not np.allclose(np.diff(raw)[-NL:], np.diff(raw)[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_autogrid_fdtd_smoke():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.35, 0.35), (-0.35, 0.35), (-0.35, 0.35))),
        grid=mw.GridSpec.auto(),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.1, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9),
            name="src",
        )
    )
    solver = mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver
    solver.solve(time_steps=40, dft_frequency=None, dft_window="none", full_field_dft=False)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        field = getattr(solver, name)
        assert torch.isfinite(field).all(), name
    assert float(torch.abs(solver.Ez).max()) > 0.0
