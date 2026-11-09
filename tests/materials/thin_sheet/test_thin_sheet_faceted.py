"""Piecewise-planar (faceted) conformal thin sheets: folds, closed and bent sheets.

A faceted SurfaceMesh is split into planar facets joined at fold lines. Nodes
take the side of their nearest facet, a Yee face pierced by a fold is cut by
the two-piece polyline through the fold point, and dual cells near a fold are
clipped exactly by the fold wedge. Tolerances fixed before the runs:

* structure: facet/fold extraction exact; no circulation row of a spanning
  sheet reads both sides; every (sub-)face circulation integrates linear
  fields to 1e-9 relative; fold-wedge dual volumes (also with a fold crossing
  a box corner) within 1e-4 of a fine quadrature; planar trace widths are
  exactly the strip widths;
* continuity: the change of the cavity fields caused by a fold drops at least
  fivefold per decade of the fold angle from 1e-4 to 1e-7 rad, down to the
  float32 floor of the series (1e-5 of the peak);
* stability (CUDA): bent PEC sheet cavity energy drift <= 1e-3 over 2000 ->
  20000 steps, bent copper sheet passive (energy non-increasing, heat > 0);
* watertightness (CUDA): a closed faceted PEC prism spanning the cavity keeps
  its interior exactly zero; a spanning folded (V) PEC sheet keeps its far side
  exactly zero; a closed 35 um copper prism leaks at most the reduced copper
  two-port floor;
* physics (CUDA): quasi-2D scattering by a 32-facet closed PEC sheet of
  area-equivalent radius lambda/2 vs the circular-cylinder Bessel series,
  relative RMS error of the scattered phasor at 8 observers one wavelength
  from the axis: lambda/20 <= 0.10, lambda/40 <= 0.04, lambda/80 <= 0.015, and
  observed order >= 1.5 from lambda/20 to lambda/80, for TE (E along the axis)
  and TM (H along the axis); 35 um copper at lambda/40 within 0.015 (TE) /
  0.01 (TM) of the series and within 1e-3 of the PEC sheet;
* exact right angles on grid lines: an L bracket has only its own on-sheet
  edges and a closed PEC box keeps its interior exactly zero.
"""
import math
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.thin_sheet import _clip_box_volume, _halfspace_volume, compile_thin_sheet, sheet_surface
from witwin.maxwell.fdtd.thin_sheet import thin_sheet_energy
from witwin.maxwell.scene import prepare_scene

from . import harness as h
from . import test_thin_sheet_runtime as runtime_tests

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Native thin sheets require CUDA.")
FOIL = dict(thickness=2e-6, frequency_range=(0.5e9, 4e9))


def _raw_mesh(vertices, triangles):
    """Exercise compiler admission independently of public SurfaceMesh validation."""
    vertices = torch.as_tensor(vertices, dtype=torch.float64)
    triangles = torch.as_tensor(triangles, dtype=torch.int64)
    return SimpleNamespace(to_mesh=lambda: (vertices, triangles))


def test_sheet_compiler_rejects_trainable_vertices():
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                            dtype=torch.float64, requires_grad=True)
    with pytest.raises(NotImplementedError, match="trainable SurfaceMesh vertices"):
        sheet_surface(_raw_mesh(vertices, [[0, 1, 2]]))


def test_sheet_compiler_rejects_inconsistent_triangle_orientation():
    mesh = _raw_mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 3, 2]])
    with pytest.raises(NotImplementedError, match="not consistently oriented or not manifold"):
        sheet_surface(mesh)


def test_nearly_coplanar_chain_must_still_form_a_planar_facet():
    # Neighbouring strips pass the merge tolerance, but accumulated curvature does not.
    angles = np.arange(20) * 1e-6
    points = np.r_[[[0., 0.]], np.cumsum(.001 * np.c_[np.cos(angles), np.sin(angles)], axis=0)]
    with pytest.raises(NotImplementedError, match="facet is not planar"):
        sheet_surface(h.extruded_polyline_mesh(points, (-1., 1.)))


def test_merged_facet_fold_must_lie_in_both_planes():
    # A tiny warp passes facet grouping but tilts its averaged plane off the short seam.
    mesh = _raw_mesh([[-1., 0., 0.], [-1., .01, 1e-10], [0., 0., 0.], [0., .01, 0.],
                      [1., 0., .5], [1., .01, .5]], [[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]])
    with pytest.raises(NotImplementedError, match="non-straight fold"):
        sheet_surface(mesh)


def test_segment_crossing_both_sides_of_a_prism_is_rejected():
    surface = sheet_surface(h.extruded_polyline_mesh(h.polygon_points(1., 4, phase=.2), (-2., 2.), closed=True))
    with pytest.raises(NotImplementedError, match="A Yee edge crosses the faceted sheet more than once"):
        surface.pierce([-2., 0., 0.], [2., 0., 0.], 1e-10)


@pytest.mark.parametrize("kind", ["disconnected", "inflection"])
def test_dual_volume_rejects_disconnected_or_mixed_fold_kinds(kind):
    if kind == "disconnected":
        surface = sheet_surface(h.extruded_polyline_mesh(h.polygon_points(1., 4), (-2., 2.), closed=True))
        facets = [0, 2]
    else:
        surface = sheet_surface(h.extruded_polyline_mesh([[-2., 0.], [-1., .2], [0., 0.], [1., .2]], (-2., 2.)))
        facets = [0, 1, 2]
    with pytest.raises(NotImplementedError, match="not joined by folds of one kind"):
        surface.below_volume(np.array([-2., -2., -1.]), np.array([2., 2., 1.]), facets)


def _coarse_compile(mesh, *, material=None, boundary=None):
    scene = prepare_scene(mw.Scene(domain=mw.Domain(bounds=((-3., 3.),) * 3),
                                   grid=mw.GridSpec.uniform(1.), device="cpu",
                                   boundary=boundary or mw.BoundarySpec.faces(default="pec")))
    return compile_thin_sheet(scene, SimpleNamespace(geometry=mesh, material=material or mw.Material.pec()))


def test_sheet_compiler_rejects_dielectric_sheet_material():
    mesh = h.rectangle_mesh((0, 0, 0), (1, 0, 0), 1., 1.)
    with pytest.raises(NotImplementedError, match="support PEC or CopperFoil materials"):
        _coarse_compile(mesh, material=mw.Material(eps_r=4.))


def test_sheet_cut_edges_reject_open_domain_walls():
    mesh = h.rectangle_mesh((.2, .3, 0), (1, .6, 0), 10., 10.)
    with pytest.raises(NotImplementedError, match="PEC or PMC domain walls only"):
        _coarse_compile(mesh, boundary=mw.BoundarySpec.none())


def test_two_crossings_inside_one_grid_edge_are_rejected():
    mesh = h.extruded_polyline_mesh(h.polygon_points(.3, 8, center=(.5, .1)), (-2.7, 2.7), closed=True)
    with pytest.raises(NotImplementedError, match="A Yee edge crosses the faceted thin sheet more than once"):
        _coarse_compile(mesh)


def test_saddle_with_four_crossings_of_one_face_is_rejected():
    mesh = _raw_mesh([[-1., -1., .2], [2., -1., -.2], [2., 2., .2], [-1., 2., -.2], [.5, .5, 0.]],
                     [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    scene = prepare_scene(mw.Scene(
        domain=mw.Domain(bounds=((0., 1.), (0., 1.), (-1., 1.))),
        grid=mw.GridSpec.custom([0., 1.], [0., 1.], [-1., 0., 1.]),
        boundary=mw.BoundarySpec.faces(default="pec"), device="cpu"))
    with pytest.raises(NotImplementedError, match="A Yee face is cut more than once"):
        compile_thin_sheet(scene, SimpleNamespace(geometry=mesh, material=mw.Material.pec()))


def test_mass_partition_cannot_reach_across_a_nearby_prism_wall():
    # A narrow six-sided prism: the nearest mass receiver lies across another wall.
    points = [[-.977874, -.712896], [-.313646, -1.659699], [.526667, -1.204978],
              [.702751, .196545], [.038523, 1.143348], [-.801790, .688628]]
    mesh = h.extruded_polyline_mesh(points, (-2.7, 2.7), closed=True)
    with pytest.raises(NotImplementedError, match="mass partition reaches across another part"):
        _coarse_compile(mesh)


def _bent(**kwargs):
    return mw.SurfaceMesh.bent_rectangle(width=0.06, length=0.05, radius=0.04, segments=8,
                                         position=(0.001, 0.0013, -0.0207), **kwargs)


def _vee(**kwargs):
    return h.extruded_polyline_mesh([(-0.2, -0.13), (0.0013, 0.0011), (0.2, -0.13)], (-0.2, 0.2), **kwargs)


def _prism(**kwargs):
    return h.extruded_polyline_mesh(h.polygon_points(0.03, 12, center=(0.0011, 0.0007)), (-0.2, 0.2), closed=True,
                                    **kwargs)


MESHES = {"bent": _bent, "vee": _vee, "prism": _prism}


def _structure(kind, material):
    if material == "pec":
        return mw.Structure(geometry=MESHES[kind](sheet_model="conformal_two_sided"), material=h.pec_material())
    foil = mw.CopperFoil(**FOIL) if material == "copper" else material
    return mw.Structure(geometry=MESHES[kind](), material=foil)


def _complex(kind, material):
    scene = prepare_scene(runtime_tests._cavity(_structure(kind, material)))
    return compile_thin_sheet(scene, scene.structures[0])


def test_facets_and_folds_are_extracted():
    bent = sheet_surface(_bent())
    assert len(bent.facets) == 8 and len(bent.adjacency) == 7 and len(set(bent.convex.values())) == 1
    prism = sheet_surface(_prism())
    assert len(prism.facets) == 12 and len(prism.adjacency) == 12 and all(prism.convex.values())
    vee = sheet_surface(_vee())
    assert len(vee.facets) == 2 and list(vee.adjacency) == [(0, 1)]
    flat = sheet_surface(h.rectangle_mesh((0, 0, 0), runtime_tests.NORMAL, 0.03, 0.025))
    assert flat.planar
    # Fold lines lie in both facet planes.
    for (a, b), (point, direction) in prism.adjacency.items():
        for index in (a, b):
            facet = prism.facets[index]
            assert abs(point @ facet.normal - facet.offset) < 1e-12 and abs(direction @ facet.normal) < 1e-12


def test_fold_wedge_volumes_match_fine_quadrature():
    rng = np.random.default_rng(4)
    # A single plane reproduces the exact half-space clip.
    for _ in range(20):
        normal = rng.standard_normal(3)
        normal /= np.linalg.norm(normal)
        lower = rng.uniform(-1, 0, 3)
        upper = lower + rng.uniform(0.5, 1.5, 3)
        offset = float(normal @ (0.5 * (lower + upper))) + rng.uniform(-0.3, 0.3)
        assert abs(_clip_box_volume(lower, upper, [(normal, offset)])
                   - _halfspace_volume(lower, upper, normal, offset)) <= 1e-12
    # Fold wedges of the prism and the vee are z-invariant: the reference
    # integrates, column by column in x, the exact y-length below the sheet
    # (sign changes located on a 64-point y grid and bisected to round-off).
    # A 2000^2 midpoint-grid reference was tried first and is itself only
    # accurate to about 2e-4 (cells cut by the sheet), above this gate.
    for surface in (sheet_surface(_prism()), sheet_surface(_vee())):
        for (a, b), (point, _) in list(surface.adjacency.items())[:4]:
            for shift in ((0.0, 0.0), (0.0013, -0.0021), (-0.0017, 0.0009)):
                lower = np.array((point[0] + shift[0] - 0.0025, point[1] + shift[1] - 0.0025, 0.0))
                upper = lower + np.array((0.005, 0.005, 0.005))
                exact = surface.below_volume(lower, upper, (a, b))
                reference = _wedge_reference(surface, lower, upper)
                assert abs(exact - reference) <= 1e-4 * 0.005 ** 3, (exact, reference)


def test_fold_crossing_a_box_corner_is_seen():
    """A fold line through a dual-box corner region enters the wedge clip even if no sample point sees its facet."""
    surface = sheet_surface(_vee())
    (a, b), (point, _) = next(iter(surface.adjacency.items()))
    missed = 0
    for offset in ((0.0049, 0.0049), (0.0001, 0.0049), (0.0049, 0.0001), (0.00495, 0.0002)):
        lower = np.array((point[0] - offset[0], point[1] - offset[1], 0.0))
        upper = lower + 0.005
        samples = np.stack(np.meshgrid(*[np.array((lower[ax], 0.5 * (lower[ax] + upper[ax]), upper[ax]))
                                         for ax in range(3)], indexing="ij"), -1).reshape(-1, 3)
        missed += len(set(surface.classify(samples)[1].tolist())) == 1
        exact = surface.box_below_volume(lower, upper, 1e-12)
        reference = _wedge_reference(surface, lower, upper)
        assert abs(exact - reference) <= 1e-4 * 0.005 ** 3, (offset, exact, reference)
    print(dict(boxes_missed_by_samples=missed))


def _wedge_reference(surface, lower, upper, columns=4000, rows=64):
    x = lower[0] + (np.arange(columns) + 0.5) / columns * (upper[0] - lower[0])
    y = np.linspace(lower[1], upper[1], rows + 1)
    z = 0.5 * (lower[2] + upper[2])

    def level(xs, ys):
        phi, _ = surface.classify(np.stack((xs, ys, np.full(xs.shape, z)), -1).reshape(-1, 3))
        return phi.reshape(xs.shape)

    X, Y = np.meshgrid(x, y, indexing="ij")
    phi = level(X, Y)
    below = phi < 0
    # Length below the sheet per column: full intervals plus bisected crossings.
    length = np.zeros(columns)
    dy = y[1] - y[0]
    for j in range(rows):
        a_below, b_below = below[:, j], below[:, j + 1]
        same = a_below == b_below
        length += np.where(same & a_below, dy, 0.0)
        cross = np.flatnonzero(~same)
        if not len(cross):
            continue
        lo, hi = np.full(len(cross), y[j]), np.full(len(cross), y[j + 1])
        start_below = a_below[cross]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            mid_below = level(x[cross], mid) < 0
            move_lo = mid_below == start_below
            lo, hi = np.where(move_lo, mid, lo), np.where(move_lo, hi, mid)
        root = 0.5 * (lo + hi)
        length[cross] += np.where(start_below, root - y[j], y[j + 1] - root)
    return float(length.mean() * (upper[0] - lower[0]) * (upper[2] - lower[2]))


def test_planar_trace_widths_are_exact_strip_widths():
    """A straight cut's trace width is exactly dual * sqrt(1 - n_a^2) of its face axis (no length round trip)."""
    foil = mw.CopperFoil(**FOIL)
    structure = mw.Structure(geometry=h.rectangle_mesh(runtime_tests.CENTER, runtime_tests.NORMAL, 0.03, 0.025),
                             material=foil)
    scene = prepare_scene(runtime_tests._cavity(structure))
    complex_ = compile_thin_sheet(scene, scene.structures[0])
    from witwin.maxwell.compiler.thin_sheet import _dual_extent
    normal = sheet_surface(structure.geometry).facets[0].normal
    candidates = set()
    for axis, name in enumerate("xyz"):
        primal = np.asarray(getattr(scene, f"d{name}_primal64"))
        for dual in np.unique(_dual_extent(primal, np.arange(1, len(primal)))):
            candidates.add(float(dual) * math.sqrt(max(0.0, 1 - normal[axis] ** 2)))
    assert len(complex_.trace_width) > 300
    assert all(float(width) in candidates for width in complex_.trace_width)


@pytest.mark.parametrize("kind", ["vee", "prism"])
@pytest.mark.parametrize("material", ["pec", "copper"])
def test_spanning_faceted_complex_never_mixes_sides(kind, material):
    complex_ = _complex(kind, material)
    assert complex_.statistics["fold_faces"] > 0
    sides = complex_.e_side
    for row in np.unique(complex_.k_rows):
        present = set(sides[complex_.k_cols[complex_.k_rows == row]].tolist()) - {0}
        assert len(present) <= 1, f"face row {row} couples both sheet sides"
    for dofs in complex_.trace_dofs:
        assert np.array_equal(sides[dofs], [-1, 1])


@pytest.mark.parametrize("kind", ["bent", "vee", "prism"])
def test_faceted_circulations_are_exact_for_linear_fields(kind):
    scene = prepare_scene(runtime_tests._cavity(_structure(kind, "copper")))
    complex_ = compile_thin_sheet(scene, scene.structures[0])
    spacing = max(float(np.diff(getattr(scene, axis + "_nodes64")).max()) for axis in "xyz")
    runtime_tests._check_chord_structure_and_circulations(complex_, spacing)


def _chord_structure(kind, material):
    if kind == "plate":
        kwargs = dict(sheet_model="conformal_two_sided") if material == "pec" else {}
        return runtime_tests._plate(h.pec_material() if material == "pec" else mw.CopperFoil(**FOIL), **kwargs)
    return _structure(kind, material)


# SHA256 of the raw COO K, Cartesian volumes, true/effective H areas and dual
# lengths, in that order, for these fixed CPU scenes. vee/prism: e2a93e4e.
# plate/bent re-pinned by regeneration at the user-approved rim fix (partial rim
# chords rejected; 2026-11-01), which removes their excess rim capacitance.
PEC_COMPLEX_DIGESTS = {
    "plate": "98709eb97e57f97b1fbf426a3e4abb948330b7ae52996083bc598c4a1963a97a",
    "bent": "b21858aa720a0e9bb32855652de196e883e5bfaf5eb6bad00f9886e6470e653c",
    "vee": "f4aef616757b2eed4abf9d0810a80e2a0f743fdf9f1bdb97f3f418949d62e6ae",
    "prism": "75f9aad643621359aec857aca13d8b7c9d2cece0ac48cd80ee205b32cfe23db9",
}


@pytest.mark.parametrize("kind", ["plate", "bent", "vee", "prism"])
def test_chord_complex_preserves_pec_operator(kind):
    """V2/V7 structural: unchanged historical PEC and identical held-chord operator."""
    complexes = []
    for material in ("pec", "copper"):
        scene = prepare_scene(runtime_tests._cavity(_chord_structure(kind, material)))
        complexes.append(compile_thin_sheet(scene, scene.structures[0]))
    pec, copper = complexes
    arrays = ("k_rows", "k_cols", "k_vals", "e_volume", "h_area", "h_true_area", "h_dual")
    digest = hashlib.sha256(b"".join(getattr(pec, name).tobytes() for name in arrays)).hexdigest()
    assert digest == PEC_COMPLEX_DIGESTS[kind]
    assert pec.chord_face is None and pec.sheet_face_area is None and pec.k_c_vals is None
    for suffix in ("rows", "cols", "vals"):
        assert np.array_equal(getattr(copper, "k_x_" + suffix), getattr(pec, "k_" + suffix))
    for name in ("e_volume", "e_fixed", "h_area", "h_true_area", "h_dual"):
        assert np.array_equal(getattr(copper, name), getattr(pec, name))
    print(dict(kind=kind, historical_pec_digest=digest, explicit_nnz=len(copper.k_x_vals),
               changed_pec_entries=0, changed_explicit_entries=0, changed_mass_entries=0))


@pytest.mark.parametrize("kind", ["plate", "bent", "vee", "prism"])
def test_chord_complex_structure_and_linear_circulations(kind):
    scene = prepare_scene(runtime_tests._cavity(_chord_structure(kind, "copper")))
    spacing = max(float(np.diff(getattr(scene, axis + "_nodes64")).max()) for axis in "xyz")
    runtime_tests._check_chord_structure_and_circulations(compile_thin_sheet(scene, scene.structures[0]), spacing)


@pytest.mark.parametrize("normal", [(0., 0., 1.), (1., 1., 0.)], ids=["aligned", "grid_edge"])
def test_chord_complex_on_grid_coincidences(normal):
    """V6: only on-sheet pairs belonging to cut-cell loops become chords."""
    structure = mw.Structure(geometry=h.rectangle_mesh((0., 0., 0.), normal, 0.03, 0.025),
                             material=mw.CopperFoil(**FOIL))
    scene = prepare_scene(runtime_tests._cavity(structure))
    c = compile_thin_sheet(scene, scene.structures[0])
    if normal == (0., 0., 1.):
        assert c.statistics["on_sheet_traces"] > 0
        assert c.chord_face.shape == (0, 4) and c.sheet_face_area.size == 0 and c.k_c_vals.size == 0
        for suffix in ("rows", "cols", "vals"):
            assert np.array_equal(getattr(c, "k_x_" + suffix), getattr(c, "k_" + suffix))
        print(dict(coincidence="aligned", chords=0, sheet_faces=0))
    else:
        spacing = max(float(np.diff(getattr(scene, axis + "_nodes64")).max()) for axis in "xyz")
        runtime_tests._check_chord_structure_and_circulations(c, spacing)
        promoted = np.flatnonzero(c.chord_face[:, 0] < 0)
        assert len(promoted) > 0
        for chord in promoted:
            dofs = c.trace_dofs[c.chord_trace[chord] - c.statistics["active_split_faces"]]
            assert np.array_equal(c.chord_volume[chord], c.e_volume[dofs])
            assert not np.isin(dofs, c.k_x_cols).any()


def _two_facet_plate(angle, **kwargs):
    """A 0.06 x 0.05 plate tilted like the runtime tests' plate, folded by ``angle`` along its middle."""
    normal = runtime_tests.NORMAL
    seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    p = runtime_tests.CENTER
    bend = math.cos(angle) * u + math.sin(angle) * normal
    vertices = np.stack([p - 0.03 * u - 0.025 * v, p - 0.025 * v, p + 0.03 * bend - 0.025 * v,
                         p + 0.03 * bend + 0.025 * v, p + 0.025 * v, p - 0.03 * u + 0.025 * v])
    triangles = [(0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)]
    return mw.SurfaceMesh(vertices, triangles, **kwargs)


FOLD_ANGLES = (1e-4, 1e-5, 1e-6, 1e-7)
# Measured float32 floor: 8.38e-6 (PEC), 7.18e-6 (copper) of the peak at
# 1e-7 rad. With float64 fields/coefficients the 1e-6 -> 1e-7 changes fall
# 10.04x / 10.01x, to 1.34e-6 / 3.32e-6: this plateau is precision error.
FOLD_FLOAT32_FLOOR = 1e-5


@pytest.mark.parametrize("material", ["pec", "copper"])
def test_vanishing_fold_converges_to_the_planar_sheet(material):
    """The fold's effect on the cavity fields falls linearly with the fold angle.

    A fold of angle a moves the far half of the plate by about 0.03 a, so the
    monitored change must drop about tenfold per decade of a (asserted >= 5x)
    from 1e-4 to 1e-7 rad, until it reaches the float32 floor of the series.
    (This replaces an envelope that compared the 1e-4 rad fold with a rigid
    shift of the planar sheet.)
    """
    kwargs = dict(sheet_model="conformal_two_sided") if material == "pec" else {}

    def series(mesh):
        structure = mw.Structure(geometry=mesh, material=h.pec_material() if material == "pec" else mw.CopperFoil(**FOIL))
        scene = runtime_tests._cavity(structure)
        scene.monitors = (mw.FieldTimeMonitor("p", components=("Ex", "Ey", "Ez"), position=(0.03, -0.02, 0.01)),)
        result = runtime_tests._run(scene, 600)
        values = torch.cat([torch.as_tensor(result.monitors["p"]["components"][c]).double().cpu().reshape(-1)
                            for c in ("Ex", "Ey", "Ez")])
        return values, result.solver._thin_sheets[0].complex.statistics

    planar, _ = series(_two_facet_plate(0.0, **kwargs))
    scale = float(planar.abs().max())
    changes = {}
    for angle in FOLD_ANGLES:
        folded, statistics = series(_two_facet_plate(angle, **kwargs))
        assert statistics["fold_faces"] > 0
        changes[angle] = float((folded - planar).abs().max()) / scale
    print(dict(material=material, changes=changes))
    assert changes[FOLD_ANGLES[0]] > 10 * FOLD_FLOAT32_FLOOR
    for large, small in zip(FOLD_ANGLES, FOLD_ANGLES[1:]):
        if changes[small] > FOLD_FLOAT32_FLOOR:
            assert changes[large] >= 5 * changes[small]


@pytest.mark.parametrize("material", ["pec", "copper"])
def test_bent_sheet_cavity_is_stable_for_twenty_thousand_steps(material):
    scene = runtime_tests._cavity(_structure("bent", material))
    dt = runtime_tests.pec_bound_dt(scene)
    early = runtime_tests._run(scene, 2000, dt=dt)
    late = runtime_tests._run(scene, 20000, dt=dt)
    e0, e1 = thin_sheet_energy(early.solver), thin_sheet_energy(late.solver)
    sheet = late.solver._thin_sheets[0]
    m0, m1 = (thin_sheet_energy(result.solver, modified=True) for result in (early, late))
    print(dict(material=material, early=e0, late=e1, drift=(e1 - e0) / e0,
               modified_early=m0, modified_late=m1, modified_ratio=m1/m0, heat=sheet.cumulative_loss,
               dt=late.solver.dt, limit=late.solver._thin_sheet_cfl_limit, folds=sheet.complex.statistics["fold_faces"],
               added_newmark_phase_rad_per_step=(0. if material == "pec" else runtime_tests.newmark_phase_error(late.solver.dt, 2e9))))
    assert np.isfinite(e1) and e0 > 0
    if material == "pec":
        assert abs(e1 - e0) / e0 <= 1e-3
    else:
        assert e1 <= e0 * (1 + 1e-9) and sheet.cumulative_loss > 0
        assert 0 < m1 <= m0 * (1 + 1e-9)


def _inside_outside(structure, inside, *, source=(-0.04, 0.035, 0.01)):
    scene = runtime_tests._cavity(structure, source_position=source, frequency=3e9, fwidth=0.6e9, device="cuda")
    solver = runtime_tests._run(scene, 600).solver
    nodes = [torch.as_tensor(getattr(solver.scene, axis + "_nodes64"), device=solver.Ez.device) for axis in "xyz"]
    x, y, z = torch.meshgrid(nodes[0], nodes[1], 0.5 * (nodes[2][1:] + nodes[2][:-1]), indexing="ij")
    mask = inside(x, y, z)
    return float(solver.Ez[mask].abs().max()), float(solver.Ez[~mask].abs().max())


def _prism_interior(x, y, z):
    return torch.hypot(x - 0.0011, y - 0.0007) < 0.03 * math.cos(math.pi / 12) - 0.006


def _prism_exterior(x, y, z):
    return torch.hypot(x - 0.0011, y - 0.0007) > 0.03 + 0.006


@cuda
def test_closed_faceted_pec_prism_is_watertight():
    inside, outside = _inside_outside(_structure("prism", "pec"), _prism_interior)
    print(dict(inside=inside, outside=outside))
    assert outside > 1.0 and inside == 0.0


@cuda
def test_spanning_folded_pec_sheet_is_watertight():
    def far(x, y, z):   # the region above the vee, more than 6 mm from it
        return (y - 0.0011 + 0.65 * (x - 0.0013).abs()) / math.sqrt(1 + 0.65 ** 2) > 0.006
    inside, outside = _inside_outside(_structure("vee", "pec"), far, source=(0.0, -0.04, 0.01))
    print(dict(far=inside, near=outside))
    assert outside > 1e3 and inside == 0.0


@cuda
def test_closed_faceted_copper_prism_leaks_only_through_the_copper_model():
    from witwin.maxwell.compiler.copper_foil import compile_copper_response
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(0.5e9, 4e9))
    inside, outside = _inside_outside(mw.Structure(geometry=_prism(), material=foil), _prism_interior)
    model = compile_copper_response(foil).state_space.evaluate(torch.tensor([3e9], dtype=torch.float64))[0]
    floor = float(model[0, 1].abs() / model[0, 0].abs())
    print(dict(inside=inside, outside=outside, ratio=inside / outside, floor=floor))
    assert outside > 1.0 and inside / outside <= floor


_SCATTERING = {}


def _scattering(material, polarization, dl):
    key = (material, polarization, dl)
    if key not in _SCATTERING:
        foil = h.pec_material() if material == "pec" else mw.CopperFoil(thickness=35e-6, frequency_range=(0.3e9, 2e9))
        _SCATTERING[key] = h.cylinder_scattering(foil, polarization, dl, absorber="cpml")
        error, info = _SCATTERING[key]
        print(dict(material=material, pol=polarization, dl=dl, error=error, facet_cells=info["facet_width"] / dl,
                   fold_faces=info["info"]["fold_faces"], dt=info["dt"]))
    return _SCATTERING[key]


WAVELENGTH = h.C0 / 1.0e9


@cuda
@pytest.mark.parametrize("polarization", ["te", "tm"])
def test_faceted_pec_cylinder_matches_the_bessel_series_and_converges(polarization):
    gates = {20: 0.10, 40: 0.04, 80: 0.015}
    errors = {n: _scattering("pec", polarization, WAVELENGTH / n)[0] for n in gates}
    order = math.log(errors[20] / errors[80]) / math.log(4.0)
    print(dict(pol=polarization, errors=errors, order=order))
    for n, gate in gates.items():
        assert errors[n] <= gate
    assert order >= 1.5


def _copper_cylinder(polarization):
    dl = WAVELENGTH / 40
    copper_error, copper = _scattering("copper", polarization, dl)
    _, pec = _scattering("pec", polarization, dl)
    difference = float(np.sqrt(np.mean(np.abs(copper["measured"] - pec["measured"]) ** 2)
                               / np.mean(np.abs(pec["measured"]) ** 2)))
    phase = math.pi * 1e9 * copper["dt"]
    print(dict(pol=polarization, copper_error=copper_error, difference=difference,
               newmark_minus_leapfrog_rad_per_step=2 * (math.atan(phase) - math.asin(phase))))
    return copper_error, difference


@cuda
@pytest.mark.parametrize("polarization", ["te", "tm"])
def test_faceted_copper_cylinder_matches_the_pec_sheet(polarization):
    copper_error, difference = _copper_cylinder(polarization)
    assert copper_error <= (0.015 if polarization == "te" else 0.01)
    assert difference <= 1e-3


def _l_bracket(**kwargs):
    """Two 20 x 40 mm facets meeting at an exact right angle along a grid line (h = 5 mm)."""
    profile = np.array([(-0.02, 0.0), (0.0, 0.0), (0.0, -0.02)])
    vertices = [(x, y, z) for y in (-0.0201, 0.0199) for (x, z) in profile]
    return mw.SurfaceMesh(np.array(vertices), [(0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4)], **kwargs)


def _closed_box(half, **kwargs):
    """Closed, outward-oriented cube on grid lines (every fold an exact right angle, three-facet corners)."""
    corner = lambda x, y, z: x * 4 + y * 2 + z
    vertices = np.array([(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    quads = [((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
             ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
             ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))]
    triangles = []
    for quad in quads:
        a, b, c, d = (corner(*v) for v in quad)
        triangles += [(a, b, c), (a, c, d)]
    return mw.SurfaceMesh(vertices, triangles, **kwargs)


@pytest.mark.parametrize("material", ["pec", "copper"])
def test_right_angle_fold_on_grid_lines_adds_no_phantom_sheet(material):
    """A facet's plane extended beyond an exact 90-degree fold is not part of the sheet.

    Grid nodes on that extension are equidistant from both facets; before the
    tie-break the lower facet index won, the extension became on-sheet and 178
    phantom trace pairs ran out to the compilation box.
    """
    kwargs = dict(sheet_model="conformal_two_sided") if material == "pec" else {}
    foil = h.pec_material() if material == "pec" else mw.CopperFoil(**FOIL)
    scene = prepare_scene(runtime_tests._cavity(mw.Structure(geometry=_l_bracket(**kwargs), material=foil)))
    c = compile_thin_sheet(scene, scene.structures[0])
    on_sheet = c.e_split
    position = c.e_position[on_sheet]
    on_horizontal = (np.abs(position[:, 2]) <= 1e-9) & (position[:, 0] >= -0.02 - 1e-9) & (position[:, 0] <= 1e-9)
    on_vertical = (np.abs(position[:, 0]) <= 1e-9) & (position[:, 2] >= -0.02 - 1e-9) & (position[:, 2] <= 1e-9)
    print(dict(material=material, statistics=c.statistics, off_surface=int(np.sum(~(on_horizontal | on_vertical)))))
    assert np.all(on_horizontal | on_vertical)
    # 4 x 8 + 5 x 7 edges per facet (h = 5 mm) less the 7 shared fold-line edges.
    assert c.statistics["on_sheet_traces"] == 2 * (4 * 8 + 5 * 7) - 7


def test_closed_right_angle_box_on_grid_lines_is_watertight():
    """A PEC shield can on grid lines: exact 90-degree folds and three-facet corners, interior exactly zero."""
    structure = mw.Structure(geometry=_closed_box(0.015, sheet_model="conformal_two_sided"), material=h.pec_material())
    scene = runtime_tests._cavity(structure, source_position=(-0.035, 0.03, 0.01))
    solver = runtime_tests._run(scene, 600).solver
    statistics = solver._thin_sheets[0].complex.statistics
    nodes = [torch.as_tensor(getattr(solver.scene, axis + "_nodes64")) for axis in "xyz"]
    x, y, z = torch.meshgrid(nodes[0], nodes[1], 0.5 * (nodes[2][1:] + nodes[2][:-1]), indexing="ij")
    inside = (x.abs() < 0.0149) & (y.abs() < 0.0149) & (z.abs() < 0.0149)
    interior, exterior = float(solver.Ez[inside].abs().max()), float(solver.Ez[~inside].abs().max())
    print(dict(statistics=statistics, interior=interior, exterior=exterior))
    assert statistics["on_sheet_traces"] == 6 * 2 * 6 * 7 - 12 * 6
    assert exterior > 1.0 and interior == 0.0
