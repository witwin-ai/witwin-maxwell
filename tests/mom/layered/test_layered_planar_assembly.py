"""B1 assembly gates on general triangles: closed-form inner integrals and Galerkin near pairs.

Oracles are independent of the runtime formulas: signed-fan adaptive quadrature
for the static source integrals, and the open-surface single-layer assembly
(Duffy touching rules, anisotropic touching refinement and gap-aware near
rules at near_tolerance=1e-7) for whole Galerkin matrices.
"""

import math

import numpy as np
import pytest
import torch

from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered import LayeredGreenTable, assemble_layered
from witwin.maxwell.mom.layered.assembly import triangle_cross_integrals, triangle_static_integrals
from witwin.maxwell.mom.layered.planar_mesh import GATE_ASPECT, PolygonSheet, mesh_planar_layout
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.single_layer import assemble_single_layer
from tests.support.layered_assembly_oracle import static_triangle
from tests.support.layered_via_references import static_cross_triangle


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")]
EPS0, C0 = 8.8541878128e-12, 299792458.0


def rotated_triangle(aspect, angle, skew):
    """Triangle with longest-edge/altitude ratio near ``aspect``, rotated in plane by ``angle``."""
    points = np.array(((0., 0.), (1., 0.), (skew, 1. / aspect)))
    rotation = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    return np.column_stack((points @ rotation.T, np.zeros(3)))


def test_static_integrals_on_rotated_high_aspect_triangles(record_property):
    cases = []
    # The adaptive CPU oracle costs ~1-20 s per query; 30 queries span the classes.
    for aspect, angle in ((3., 1.1), (GATE_ASPECT, math.pi / 4), (20., math.pi / 8)):
        triangle = rotated_triangle(aspect, angle, .3)
        centre = triangle.mean(0)
        edge_mid = (triangle[0] + triangle[1]) / 2
        for point in (centre, edge_mid, triangle[2], 2 * triangle[1] - centre):
            for height in (0., .1):
                cases.append((triangle, point + np.array((0., 0., height)), triangle[1]))
        cases.append((triangle, edge_mid + np.array((0., 0., 1e-3 / aspect)), triangle[1]))
        cases.append((triangle, edge_mid + (centre - edge_mid) * 1e-6 + np.array((0., 0., 1e-6)), triangle[1]))
    expected = [static_triangle(*case) for case in cases]
    triangles = torch.tensor(np.stack([c[0] for c in cases]), device="cuda")
    observers = torch.tensor(np.stack([c[1] for c in cases]), device="cuda")
    free = torch.tensor(np.stack([c[2] for c in cases]), device="cuda")
    scalar, vector = (v.cpu().numpy() for v in triangle_static_integrals(triangles, observers, free))
    scalar_error = np.max(np.abs(scalar - [e[0] for e in expected]) / np.abs([e[0] for e in expected]))
    vector_ref = np.stack([e[1] for e in expected])
    vector_error = np.max(np.linalg.norm(vector - vector_ref, axis=-1) / np.linalg.norm(vector_ref, axis=-1))
    record_property("static_scalar_relative", scalar_error)
    record_property("static_vector_relative", vector_error)
    assert scalar_error <= 1e-10
    assert vector_error <= 1e-10


def test_cross_integrals_on_oblique_vertical_faces(record_property):
    """Polygonal via walls have horizontal normals at arbitrary in-plane angles."""
    cases = []
    for angle in (math.pi / 8, 2.):
        direction = np.array((math.cos(angle), math.sin(angle), 0.))
        for wall in (((0., -1.), (1., -1.), (.2, 0.)), ((0., -1.), (1., -1.), (0., 0.)), ((0., 0.), (.05, -1.), (1., 0.))):
            triangle = np.array([direction * u + np.array((0., 0., z)) for u, z in wall])
            normal = np.cross(direction, (0., 0., 1.))
            for offset in (0., 1e-3, -.1):
                # Image rays have d = beta (z_obs - z_src) >= 0: observers at or above the face.
                for u, z in ((.3, 0.), (1.2, .1)):
                    cases.append((triangle, direction * u + normal * offset + np.array((0., 0., z)), 1.))
    reference = np.stack([static_cross_triangle(*case) for case in cases])
    triangles = torch.tensor(np.stack([c[0] for c in cases]), device="cuda")
    observers = torch.tensor(np.stack([c[1] for c in cases]), device="cuda")
    beta = torch.tensor([c[2] for c in cases], dtype=torch.float64, device="cuda")
    actual = triangle_cross_integrals(triangles, observers, beta).cpu().numpy()
    relative = (np.abs(actual - reference).max(-1) / np.abs(reference).max(-1)).max()
    record_property("oblique_cross_relative", relative)
    assert relative < 1e-10


def homogeneous_reference(layout, frequency):
    basis = layout.basis
    k = 2 * math.pi * frequency / C0
    config = IntegrationConfig(order=8, singular_order=16, near_tolerance=1e-7, near_max_order=32,
                               pair_batch=4, far_pair_batch=64)
    vector, scalar, _ = assemble_single_layer(((basis, 0, 0),), k, config)
    div = basis.divergence.to(torch.complex128)
    return -1j * (2 * math.pi * frequency) / (EPS0 * C0 ** 2) * (vector - div.T @ scalar @ div / k ** 2)


def pair_classes(layout):
    """[N,N] boolean masks: far (every support pair beyond twice the larger edge) and touching."""
    basis, mesh = layout.basis, layout.mesh
    supports = torch.stack((basis.plus_faces, basis.minus_faces), -1)
    centres = mesh.centroids[supports]
    size2 = (mesh.triangles - mesh.triangles.roll(1, 1)).square().sum(-1).amax(-1)
    distance2 = (centres[:, None, :, None] - centres[None, :, None]).square().sum(-1)
    threshold2 = 4 * torch.maximum(size2[supports][:, None, :, None], size2[supports][None, :, None])
    far = (distance2 >= threshold2).all(dim=(-2, -1))
    faces = mesh.faces[supports]
    shared = (faces[:, None, :, None, :, None] == faces[None, :, None, :, None, :]).any(-1).any(-1).any(-1).any(-1)
    return far, shared


def planar_vacuum(planes, extent):
    stack = LayerStack((Layer("uniform", 1e-3, 1.0),), HalfSpace(), HalfSpace())
    return LayeredGreenTable.build(stack, planes, (10e9,), extent, device="cuda")


def test_planar_mesh_matches_independent_galerkin(record_property):
    """Octagon with hole, 45-degree traces and graded Delaunay cells against the open-surface oracle."""
    plane = .5e-3
    octagon = [(1e-4 * math.cos(math.pi / 8 + k * math.pi / 4), 1e-4 * math.sin(math.pi / 8 + k * math.pi / 4))
               for k in range(8)]
    hole = [(.6 * x, .6 * y) for x, y in octagon]
    table = planar_vacuum((plane,), 2.2e-4 * math.sqrt(2))
    layout = mesh_planar_layout([PolygonSheet((tuple(octagon), tuple(hole)), plane)], table.planes, 3e-5,
                                h_edge=1.5e-5)
    actual = assemble_layered(layout, table)
    reference = homogeneous_reference(layout, 10e9)
    whole = (torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)).item()
    far, touching = pair_classes(layout)
    scale = reference.abs().amax(1)
    # Oblique RWG pairs can cancel to a tiny entry; far entries are compared where
    # they carry at least 1e-3 of their row maximum, all entries row-relative.
    significant = far & (reference.abs() >= 1e-3 * scale[:, None])
    far_error = ((actual[significant] - reference[significant]).abs() / reference[significant].abs()).max().item()
    far_row = ((actual - reference).abs() / scale[:, None])[far].max().item()
    near_error = ((actual - reference).abs() / scale[:, None])[~far].max().item()
    record_property("far_row_relative", far_row)
    record_property("unknowns", actual.shape[0])
    record_property("whole_frobenius", whole)
    record_property("far_entry_relative", far_error)
    record_property("near_entry_row_relative", near_error)
    record_property("mesh_report", layout.mesh_report)
    assert whole <= 1e-5
    assert far_error <= 1e-9
    assert far_row <= 1e-10
    assert near_error <= 1e-5
    assert (actual - actual.T).abs().max().item() <= 1e-12 * actual.abs().max().item()
    assert torch.equal(actual, assemble_layered(layout, table))


def strip_layout(table, aspect, angle):
    """Rotated strip of 2x6 cells, each split into two triangles of aspect ratio ~``aspect``."""
    from witwin.maxwell.mom.layered.mesh import LayeredMesh
    from witwin.maxwell.mom.open_surface import OpenRWGBasis, OpenSurfaceMesh
    length, width = 1e-5 * aspect, 1e-5
    u, v = np.meshgrid(np.arange(7) * length / 2, np.arange(3) * width, indexing="ij")
    rotation = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    xy = np.stack((u.ravel(), v.ravel()), 1) @ rotation.T
    points = torch.tensor(np.column_stack((xy, np.full(len(xy), table.planes[0].item()))), device="cuda")
    faces = []
    for i in range(6):
        for j in range(2):
            a, b, c, d = i * 3 + j, (i + 1) * 3 + j, (i + 1) * 3 + j + 1, i * 3 + j + 1
            faces.extend(((a, b, c), (a, c, d)) if (i + j) % 2 == 0 else ((a, b, d), (b, c, d)))
    mesh = OpenSurfaceMesh(points, torch.tensor(faces, device="cuda"))
    return LayeredMesh(mesh, OpenRWGBasis(mesh), torch.zeros(len(faces), dtype=torch.int64, device="cuda"))


@pytest.mark.parametrize("angle", [math.pi / 6, math.pi / 4])
def test_near_pair_error_versus_aspect_ratio(angle, record_property):
    """The documented sliver defect: touching-pair error grows with aspect; the gate keeps it bounded.

    Strip parameters 2..40 give measured triangle aspects (longest edge^2 / twice
    the area) 2, 2.5, 3.3, 5.2, 10.1 and 20. Measured, row-relative: touching
    pairs 2.0e-6..7.1e-6 up to aspect 5.2, 5.3e-5 at 10.1; separated near pairs
    <= 1.6e-11. Aspects inside the gate must stay within 1e-5 / 1e-10; larger
    aspects are recorded only.
    """
    plane = .5e-3
    errors = {}
    for aspect in (2., 4., 6., 10., 20., 40.):
        # The assembly bounds the axis-aligned box diagonal, which exceeds the rotated strip's.
        table = planar_vacuum((plane,), 2 * math.hypot(3e-5 * aspect, 2e-5))
        layout = strip_layout(table, aspect, angle)
        actual = assemble_layered(layout, table)
        reference = homogeneous_reference(layout, 10e9)
        far, touching = pair_classes(layout)
        scale = reference.abs().amax(1)
        errors[aspect] = dict(
            touching=((actual - reference).abs() / scale[:, None])[touching].max().item(),
            near=((actual - reference).abs() / scale[:, None])[~far & ~touching].max().item(),
            aspect_measured=float(layout.quality["max_aspect_ratio"]))
    record_property("errors", errors)
    inside = [error for error in errors.values() if error["aspect_measured"] <= GATE_ASPECT]
    assert len(inside) == 4
    for error in inside:
        assert error["touching"] <= 1e-5 and error["near"] <= 1e-10, error
