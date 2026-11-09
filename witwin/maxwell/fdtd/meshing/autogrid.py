"""Adaptive (auto) mesh generation for nonuniform FDTD Yee grids.

Pure host-side float64 meshing. Per axis, every structure / override AABB face
becomes a hard cell boundary; each face-bounded interval receives a target step
``wavelength / (n_max * min_steps_per_wavelength)`` (further reduced by mesh
overrides and layer refinement); intervals are then filled with uniform or
geometrically graded cells so adjacent cells never differ by more than
``max_ratio`` anywhere on the axis.

Nodes are absolute float64 coordinates, so the snapping / target / ratio
invariants hold up to rounding that scales with ``ulp(|coordinate|) / cell``;
cells within a few orders of magnitude of the coordinate ULP degrade.
"""

from __future__ import annotations

import math

import numpy as np

_C0 = 299_792_458.0
# Structure AABBs come from float32 ``to_mesh`` vertices, whose relative
# precision is ~1.2e-7. A face nominally on a domain boundary (or on another
# face) can therefore land that far away from it. Snap faces closer than this
# together so a float32-rounded coordinate never spawns a sub-cell sliver
# interval, which the ratio fixpoint would otherwise resolve into a fan of
# near-zero-width cells (crushing the Courant limit).
_FACE_SNAP_REL = 1e-6
_SLACK = 1.0 + 1e-12
_MAX_AXIS_CELLS = 10_000_000


def _axis_tolerance(lo: float, hi: float) -> float:
    """Face-snapping tolerance for one axis.

    The relative term covers domains near the origin; the float32-ULP term
    covers domains offset far from the origin, where the absolute face rounding
    grows with coordinate magnitude. The ULP term is a few ULP (not scaled by
    the span/offset ratio) so a large offset never swallows real structure
    faces inside a much smaller domain span.
    """
    ulp = float(np.spacing(np.float32(max(abs(lo), abs(hi)))))
    return max(_FACE_SNAP_REL * (hi - lo), 4.0 * ulp)


def _collect_boundaries(lo: float, hi: float, faces) -> np.ndarray:
    tolerance = _axis_tolerance(lo, hi)
    interior = sorted({min(max(float(face), lo), hi) for face in faces})
    boundaries = [lo]
    for value in interior:
        if value - boundaries[-1] > tolerance and hi - value > tolerance:
            boundaries.append(value)
    boundaries.append(hi)
    return np.asarray(boundaries, dtype=np.float64)


def _interval_targets(boundaries, base_dl, constraints, layer_min_cells) -> np.ndarray:
    lengths = np.diff(boundaries)
    targets = np.full(lengths.size, float(base_dl), dtype=np.float64)
    tolerance = _axis_tolerance(float(boundaries[0]), float(boundaries[-1]))
    for c_lo, c_hi, c_dl in constraints:
        overlap = (boundaries[:-1] < c_hi - tolerance) & (boundaries[1:] > c_lo + tolerance)
        targets[overlap] = np.minimum(targets[overlap], c_dl)
    if layer_min_cells is not None:
        thin = lengths < layer_min_cells * targets
        targets[thin] = lengths[thin] / layer_min_cells
    return targets


def _uniform_fixpoint(lengths, targets, max_ratio):
    """Per-interval uniform cell sizes with cross-boundary ratio <= max_ratio.

    Fixpoint of ``u_i = L_i / ceil(L_i / min(t_i, r*u_{i-1}, r*u_{i+1}))``:
    every interval is filled exactly, its uniform step never exceeds the
    interval target, and adjacent intervals differ by at most ``max_ratio``.
    """
    counts = np.maximum(1, np.ceil(lengths / targets - 1e-12).astype(np.int64))
    sizes = lengths / counts
    changed = True
    while changed:
        changed = False
        for i in range(sizes.size):
            allowed = targets[i]
            if i > 0:
                allowed = min(allowed, max_ratio * sizes[i - 1])
            if i + 1 < sizes.size:
                allowed = min(allowed, max_ratio * sizes[i + 1])
            if sizes[i] > allowed * _SLACK:
                counts[i] = max(counts[i] + 1, int(math.ceil(lengths[i] / allowed - 1e-12)))
                sizes[i] = lengths[i] / counts[i]
                changed = True
    return sizes, counts


def _graded_sizes(length, cap, edge_lo, edge_hi, max_ratio):
    """Graded fill: exact edge cells, geometric ramps at rate ``max_ratio``,
    and a remainder-absorbing plateau in ``[cap/max_ratio, cap]``.

    ``edge_lo`` / ``edge_hi`` pin the boundary-adjacent cell size (``None`` at
    a free domain edge). When the interval is too short to ramp all the way up
    to ``cap``, the plateau cap is lowered geometrically until a valid profile
    fits; every reduced-cap profile preserves the target and ratio invariants.
    Returns ``None`` only when no plateau above the pinned edges fits, in which
    case the uniform fill is already within a few cells of optimal.
    """
    edges = [edge for edge in (edge_lo, edge_hi) if edge is not None]
    floor = max(edges) if edges else 0.0
    while cap > floor:
        ramp_lo, ramp_hi = [], []
        for edge, ramp in ((edge_lo, ramp_lo), (edge_hi, ramp_hi)):
            if edge is None:
                continue
            ramp.append(edge)
            value = edge * max_ratio
            while value < cap * (1.0 - 1e-12):
                ramp.append(value)
                value *= max_ratio
        middle = length - sum(ramp_lo) - sum(ramp_hi)
        if middle >= cap / (max_ratio - 1.0):
            plateau_count = max(1, int(math.ceil(middle / cap - 1e-12)))
            plateau = middle / plateau_count
            return np.asarray(
                ramp_lo + [plateau] * plateau_count + ramp_hi[::-1], dtype=np.float64
            )
        cap /= max_ratio
    return None


def _fill_axis(boundaries, targets, max_ratio) -> np.ndarray:
    lengths = np.diff(boundaries)
    sizes, counts = _uniform_fixpoint(lengths, targets, max_ratio)
    nodes = [np.asarray([boundaries[0]])]
    for i in range(lengths.size):
        edge_lo = min(sizes[i], sizes[i - 1]) if i > 0 else None
        edge_hi = min(sizes[i], sizes[i + 1]) if i + 1 < lengths.size else None
        cells = _graded_sizes(lengths[i], targets[i], edge_lo, edge_hi, max_ratio)
        if cells is None or cells.size >= counts[i]:
            cells = np.full(counts[i], sizes[i])
        # Distribute float64 summation drift across the profile instead of
        # letting the exact-boundary snap dump it all into the last cell.
        cells = cells * (lengths[i] / cells.sum())
        nodes.append(boundaries[i] + np.cumsum(cells[:-1]))
        nodes.append(np.asarray([boundaries[i + 1]]))
    return np.concatenate(nodes)


def mesh_axis(
    lo,
    hi,
    *,
    wavelength,
    min_steps_per_wavelength,
    max_ratio,
    index_regions=(),
    override_regions=(),
    layer_min_cells=None,
    pml_cells=(0, 0),
    snap_points=(),
    refine_regions=(),
) -> np.ndarray:
    """Mesh one axis of the domain ``[lo, hi]`` into float64 node coordinates.

    ``index_regions`` are ``(lo, hi, refractive_index)`` structure extents and
    ``override_regions`` are ``(lo, hi, max_dl)`` mesh-override extents; the
    faces of both snap to cell boundaries. ``pml_cells`` gives the CPML
    thickness in cells at the low / high domain edge; those bands are
    uniformized piecewise between hard faces (see
    ``_uniformize_boundary_band``).

    Opt-in layer refinement adds ``snap_points`` ``(coordinate, merge_tol)``,
    hard faces dropped when closer than ``merge_tol`` to another face, and
    ``refine_regions`` ``(lo, hi, max_dl)``, soft windows whose edges are moved
    outward onto nearby faces so they never leave a sliver interval (see
    ``_place_refine_regions``). With both empty the mesh is unchanged.
    """
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        raise ValueError("mesh_axis requires hi > lo.")
    base_dl = float(wavelength) / float(min_steps_per_wavelength)
    faces = []
    constraints = []
    for r_lo, r_hi, index in index_regions:
        faces += [r_lo, r_hi]
        constraints.append(
            (float(r_lo), float(r_hi), base_dl / max(float(index), 1.0))
        )
    for r_lo, r_hi, dl in override_regions:
        faces += [r_lo, r_hi]
        constraints.append((float(r_lo), float(r_hi), float(dl)))
    boundaries = _collect_boundaries(lo, hi, faces)
    if snap_points or refine_regions:
        faces = faces + _accepted_snaps(boundaries, lo, hi, snap_points)
        boundaries = _collect_boundaries(lo, hi, faces)
        regions, edges = _place_refine_regions(boundaries, lo, hi, refine_regions)
        faces = faces + edges
        constraints = constraints + regions
        boundaries = _collect_boundaries(lo, hi, faces)
    targets = _interval_targets(boundaries, base_dl, constraints, layer_min_cells)
    estimate = float(np.ceil(np.diff(boundaries) / targets).sum())
    if not math.isfinite(estimate) or estimate > _MAX_AXIS_CELLS:
        raise ValueError(
            f"GridSpec.auto meshing of [{lo:g}, {hi:g}] implies ~{estimate:.3g} cells "
            f"on one axis (limit {_MAX_AXIS_CELLS:.0e}); check the wavelength, "
            "min_steps_per_wavelength, and override dl units."
        )
    nodes = _fill_axis(boundaries, targets, float(max_ratio))
    return _uniformize_boundary_band(nodes, boundaries, *pml_cells)


def _accepted_snaps(boundaries, lo, hi, snap_points) -> list[float]:
    """Snap faces inside ``[lo, hi]`` farther than their merge tolerance from
    every existing or previously accepted face (processed in ascending order)."""
    tolerance = _axis_tolerance(lo, hi)
    current = sorted(float(b) for b in boundaries)
    accepted = []
    for coord, merge_tol in sorted((float(c), float(t)) for c, t in snap_points):
        if not lo < coord < hi:
            continue
        index = int(np.searchsorted(current, coord))
        nearest = min(
            abs(current[j] - coord) for j in (index - 1, index) if 0 <= j < len(current)
        )
        if nearest > max(merge_tol, tolerance):
            current.insert(index, coord)
            accepted.append(coord)
    return accepted


def _merge_refine_regions(lo, hi, refine_regions):
    """Clip windows to the domain and fuse same-step windows that overlap or sit
    closer than one step apart."""
    regions = sorted(
        (max(float(r_lo), lo), min(float(r_hi), hi), float(dl))
        for r_lo, r_hi, dl in refine_regions
        if float(r_hi) > lo and float(r_lo) < hi and float(dl) > 0.0
    )
    merged = []
    for r_lo, r_hi, dl in regions:
        for k, (m_lo, m_hi, m_dl) in enumerate(merged):
            if m_dl == dl and r_lo <= m_hi + dl and r_hi >= m_lo - dl:
                merged[k] = (min(m_lo, r_lo), max(m_hi, r_hi), dl)
                break
        else:
            merged.append((r_lo, r_hi, dl))
    return merged


def _place_refine_regions(boundaries, lo, hi, refine_regions):
    """Place soft refinement windows on an axis without creating sliver cells.

    Each window edge is moved outward (never inward, so the requested window is
    always covered): onto the nearest face within one window step outside it,
    or, when a face lies less than one step inside it, to one step beyond that
    face (and then onto any face within a step outside). Every accepted edge is
    therefore either an existing face or at least one window step away from all
    faces, so the ratio fill grades smoothly into the window. An edge that falls
    inside a strictly finer window is not added as a face. Returns the placed
    windows and the new face coordinates.
    """
    edges = sorted(float(b) for b in boundaries)

    def place(edge, step, outward):
        def near(a, b):
            return [face for face in edges if a <= face <= b]

        if outward < 0:
            outside = near(edge - step, edge)
            if outside:
                return max(outside)
            inside = [face for face in edges if edge < face < edge + step]
            if inside:
                edge = min(inside) - step
                outside = near(edge - step, edge)
                if outside:
                    return max(outside)
            return max(edge, lo)
        outside = near(edge, edge + step)
        if outside:
            return min(outside)
        inside = [face for face in edges if edge - step < face < edge]
        if inside:
            edge = max(inside) + step
            outside = near(edge, edge + step)
            if outside:
                return min(outside)
        return min(edge, hi)

    regions = _merge_refine_regions(lo, hi, refine_regions)
    placed, added = [], []
    for r_lo, r_hi, dl in regions:
        new_lo = place(r_lo, dl, -1)
        new_hi = place(r_hi, dl, +1)
        for edge in (new_lo, new_hi):
            covered = any(o_lo < edge < o_hi and o_dl < dl for o_lo, o_hi, o_dl in regions)
            if not covered and edge not in edges:
                edges.append(edge)
                edges.sort()
                added.append(edge)
        placed.append((new_lo, new_hi, dl))
    return placed, added


def _geometry_aabb(geometry):
    from ...geometry.planar_pec import PlanarPEC
    if isinstance(geometry, PlanarPEC):
        # Mesh vertices are float32 visualization data. Preserve the physical
        # float64 plane so auto meshing inserts precisely the declared support.
        points = geometry.vertices.detach().cpu().numpy()
        normal = "xyz".index(geometry.normal_axis)
        transverse = [d for d in range(3) if d != normal]
        lo = np.full(3, geometry.coordinate, dtype=np.float64)
        hi = lo.copy()
        lo[transverse], hi[transverse] = points.min(axis=0), points.max(axis=0)
        return lo, hi
    try:
        vertices, _ = geometry.to_mesh()
    except NotImplementedError as error:
        raise ValueError(
            "GridSpec.auto requires an axis-aligned bounding box for every geometry; "
            f"{type(geometry).__name__}.to_mesh() is not implemented."
        ) from error
    points = vertices.detach().cpu().numpy().astype(np.float64)
    return points.min(axis=0), points.max(axis=0)


def _max_real_component(value) -> float:
    if hasattr(value, "as_tuple"):
        return float(max(value.as_tuple()))
    return float(np.real(value))


def _refractive_index(material, frequency: float) -> float:
    try:
        sample = material.evaluate_at_frequency(frequency)
    except (AttributeError, NotImplementedError):
        sample = material.evaluate_static() if hasattr(material, "evaluate_static") else material
    eps = max(_max_real_component(getattr(sample, "eps_r", 1.0)), 1.0)
    mu = max(_max_real_component(getattr(sample, "mu_r", 1.0)), 1.0)
    return math.sqrt(eps * mu)


def _meshing_wavelength(scene) -> float:
    if scene.grid.wavelength is not None:
        return float(scene.grid.wavelength)
    frequencies = []
    for source in scene.resolved_sources():
        source_time = getattr(source, "source_time", None)
        # Broadband waveforms (GaussianPulse, RickerWavelet) carry significant
        # spectral content above their center frequency; mesh for the same
        # characteristic frequency that auto_dt() uses.
        frequency = getattr(source_time, "characteristic_frequency", None)
        if frequency is None:
            frequency = getattr(source_time, "frequency", None)
        if frequency is not None:
            frequencies.append(float(frequency))
    if not frequencies:
        raise ValueError(
            "GridSpec.auto could not derive a meshing wavelength: no scene source "
            "carries a source_time frequency. Pass GridSpec.auto(wavelength=...) or "
            "give the sources a frequency-bearing source_time."
        )
    return _C0 / max(frequencies)


def _uniformize_boundary_band(
    nodes: np.ndarray, boundaries: np.ndarray, n_low: int, n_high: int
) -> np.ndarray:
    """Uniformize the outermost ``n_low`` / ``n_high`` cells piecewise.

    The CPML absorber occupies the outermost ``pml_thickness`` cells of the
    domain and grades by physical depth, so a uniform absorber band gives the
    cleanest absorption. Each band is re-spaced piecewise: hard interval
    boundaries (structure / override faces) inside the band stay fixed, and the
    nodes of each sub-segment between them are spread uniformly. Face snapping
    is therefore preserved, and so are the per-interval targets: every replaced
    cell already satisfied its interval target, so the sub-segment mean does
    too. The domain edges and the first interior node past each band stay
    fixed, so the interior mesh is untouched. It is a no-op where the band is
    already uniform (the common vacuum-edge case). Bands are skipped (rather
    than overlapped) on domains too thin to hold both plus an interior cell.
    """
    cells = nodes.size - 1
    n_low = max(0, int(n_low))
    n_high = max(0, int(n_high))
    if n_low + n_high == 0 or n_low + n_high >= cells:
        return nodes
    out = nodes.copy()
    # Interval boundaries are exact node values by construction (_fill_axis
    # appends them verbatim), so exact-match lookup is safe.
    positions = np.searchsorted(nodes, boundaries)
    hard = {int(i) for i, b in zip(positions, boundaries) if nodes[i] == b}
    for start, stop in ((0, n_low), (cells - n_high, cells)):
        anchors = [start] + sorted(i for i in hard if start < i < stop) + [stop]
        for a, b in zip(anchors, anchors[1:]):
            out[a : b + 1] = np.linspace(out[a], out[b], b - a + 1)
    return out


def _layer_features(scene, specs, vacuum_dl):
    """Per-axis snap points, soft refinement windows and hard regions of the
    located ``LayerRefinementSpec`` entries (corners, gaps, layer bounds)."""
    from . import corners as cg

    snaps = ([], [], [])
    soft = ([], [], [])
    hard = ([], [], [])
    bounds = [tuple(map(float, pair)) for pair in scene.domain.bounds]
    structures = [s for s in scene.structures if getattr(s, "enabled", True)]
    for spec in specs:
        if not spec.is_layer:
            continue
        axis = "xyz".index(spec.axis)
        b_lo, b_hi = spec.bounds
        length = b_hi - b_lo
        steps_dl = math.inf
        if spec.min_steps_along_axis is not None and length > 0.0:
            steps_dl = length / spec.min_steps_along_axis
            hard[axis].append((b_lo, b_hi, steps_dl))
        planes = {
            None: (), "lower": (b_lo,), "upper": (b_hi,),
            "center": (0.5 * (b_lo + b_hi),), "bounds": (b_lo, b_hi) if length > 0.0 else (b_lo,),
        }[spec.bounds_snapping]
        snaps[axis].extend((plane, 0.0) for plane in planes)
        if spec.bounds_refinement is not None:
            step = spec.bounds_refinement.grid_size(vacuum_dl)
            if step < steps_dl:
                half = 0.5 * spec.bounds_refinement.num_cells * step
                for plane in ((b_lo, b_hi) if length > 0.0 else (b_lo,)):
                    soft[axis].append((plane - half, plane + half, step))
        if spec.corner_finder is None and not spec.dl_min_from_gap_width:
            continue
        u, v = cg.inplane_axes(axis)
        coord = 0.5 * (b_lo + b_hi)
        tol = _axis_tolerance(*bounds[axis])
        sections = []
        for structure in structures:
            loops = cg.cross_section_loops(structure.geometry, axis, coord, tol)
            if loops:
                sections.append((cg.is_metal(structure), cg.orient_loops(loops)))
        if spec.corner_finder is not None:
            _corner_features(spec, sections, bounds, (u, v), vacuum_dl, snaps, soft)
        if spec.dl_min_from_gap_width:
            metal = [loops for is_metal, loops in sections if is_metal]
            for along, index in ((0, u), (1, v)):
                for g_lo, g_hi in cg.scanline_gaps(metal, along, 2.0 * vacuum_dl, bounds[index]):
                    # Two cells across the gap, nodes on both metal edges.
                    hard[index].append((g_lo, g_hi, 0.5 * (g_hi - g_lo)))
    return snaps, soft, hard


def _corner_features(spec, sections, bounds, plane_axes, vacuum_dl, snaps, soft):
    from . import corners as cg

    finder = spec.corner_finder
    medium = finder.medium
    chosen = [
        loops for is_metal, loops in sections
        if medium == "all" or is_metal == (medium == "metal")
    ]
    refinement = spec.corner_refinement
    step = refinement.grid_size(vacuum_dl) if refinement is not None else vacuum_dl
    merge = max(0.25 * step, finder.distance_threshold or 0.0)
    u, v = plane_axes
    tolerances = (_axis_tolerance(*bounds[u]), _axis_tolerance(*bounds[v]))
    resolutions = (
        (finder.convex_resolution, lambda a, b: a & b),
        (finder.concave_resolution, lambda a, b: ~(a | b)),
        (finder.mixed_resolution, lambda a, b: a ^ b),
    )
    for owner, loops in enumerate(chosen):
        for loop in loops:
            points, convex = cg.find_corners(loop, finder.angle_threshold, finder.distance_threshold)
            for res, select in resolutions:
                if res is None or points.shape[0] < 2:
                    continue
                following = np.roll(points, -1, axis=0)
                lengths = np.linalg.norm(following - points, axis=1)
                mask = select(convex, np.roll(convex, -1))
                for start, end, length in zip(points[mask], following[mask], lengths[mask]):
                    for along, index in ((0, u), (1, v)):
                        lo, hi = sorted((start[along], end[along]))
                        if hi - lo > tolerances[along]:
                            soft[index].append((lo, hi, length / res))
            points, convex = cg.drop_covered_corners(points, convex, owner, chosen, max(tolerances))
            for point in points:
                inside = all(
                    bounds[index][0] - tolerances[along] <= point[along] <= bounds[index][1] + tolerances[along]
                    for along, index in ((0, u), (1, v))
                )
                if spec.refinement_inside_sim_only and not inside:
                    continue
                for along, index in ((0, u), (1, v)):
                    coord = float(point[along])
                    if spec.corner_snapping:
                        snaps[index].append((coord, merge))
                    if refinement is not None:
                        half = 0.5 * refinement.num_cells * step
                        soft[index].append((coord - half, coord + half, step))


def resolve_auto_grid(scene):
    """Materialize a ``GridSpec.auto`` scene grid into per-axis node arrays.

    Returns float64 ``(x_nodes, y_nodes, z_nodes)`` spanning the scene domain
    exactly, suitable for ``GridSpec.custom``. The outermost cells under each
    PML face are uniformized between structure faces so the absorber sees a
    clean, constant-step band without losing face snapping or refinement.
    """
    grid = scene.grid
    wavelength = _meshing_wavelength(scene)
    frequency = _C0 / wavelength
    index_regions = []
    for structure in scene.structures:
        if not getattr(structure, "enabled", True):
            continue
        aabb_lo, aabb_hi = _geometry_aabb(structure.geometry)
        index_regions.append(
            (aabb_lo, aabb_hi, _refractive_index(structure.material, frequency))
        )
    for region in scene.material_regions:
        aabb_lo, aabb_hi = _geometry_aabb(region.geometry)
        index = math.sqrt(
            max(max(region.eps_bounds), 1.0) * max(max(region.mu_bounds), 1.0)
        )
        index_regions.append((aabb_lo, aabb_hi, index))
    override_regions = []
    for override in grid.override_structures:
        aabb_lo, aabb_hi = _geometry_aabb(override.geometry)
        override_regions.append((aabb_lo, aabb_hi, override.dl))
    layer = grid.layer_refinement
    if isinstance(layer, tuple):
        layer_specs, layer = layer, None
    else:
        layer_specs = ()
    features = None
    if any(spec.is_layer for spec in layer_specs) or (layer is not None and layer.is_layer):
        features = _layer_features(
            scene,
            layer_specs if layer is None else (layer,),
            wavelength / float(grid.min_steps_per_wavelength),
        )
    nodes = []
    for axis_index, (axis_lo, axis_hi) in enumerate(scene.domain.bounds):
        axis = "xyz"[axis_index]
        if layer_specs:
            counts = [spec.min_cells for spec in layer_specs if spec.covers(axis)]
            layer_min_cells = max(counts) if counts else None
        else:
            layer_min_cells = (
                layer.min_cells if layer is not None and layer.covers(axis) else None
            )
        extra = {}
        if features is not None:
            snaps, soft, hard = features
            extra = {"snap_points": snaps[axis_index], "refine_regions": soft[axis_index]}
            override_axis = [
                (lo[axis_index], hi[axis_index], dl[axis_index]) for lo, hi, dl in override_regions
            ] + hard[axis_index]
        else:
            override_axis = [
                (lo[axis_index], hi[axis_index], dl[axis_index]) for lo, hi, dl in override_regions
            ]
        axis_nodes = mesh_axis(
            axis_lo,
            axis_hi,
            wavelength=wavelength,
            min_steps_per_wavelength=grid.min_steps_per_wavelength,
            max_ratio=grid.max_ratio,
            index_regions=[
                (lo[axis_index], hi[axis_index], index) for lo, hi, index in index_regions
            ],
            override_regions=override_axis,
            layer_min_cells=layer_min_cells,
            # Domain is physical. PreparedScene appends the PML cells outside
            # these nodes, so the physical auto mesh has no internal PML band.
            pml_cells=(0, 0),
            **extra,
        )
        nodes.append(axis_nodes)
    return tuple(nodes)
