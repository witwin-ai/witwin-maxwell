"""R2a CUDA acceptance gates; run under the assigned GPU-0 memory cap."""

import math

import numpy as np
import pytest
import torch

from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered import LayeredGreenTable, assemble_layered, mesh_rectangles
from witwin.maxwell.mom.layered.assembly import triangle_static_integrals
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.single_layer import assemble_single_layer
from tests.support.layered_assembly_oracle import static_triangle


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")]
EPS0, C0 = 8.8541878128e-12, 299792458.0


@pytest.mark.parametrize("epsilon", [1.0, 4.0])
def test_homogeneous_single_layer(epsilon):
    frequency, side, plane = 10e9, 1e-3, .5e-3
    stack = LayerStack((Layer("uniform", 1e-3, epsilon),), HalfSpace(epsilon), HalfSpace(epsilon))
    table = LayeredGreenTable.build(stack, (plane,), (frequency,), side * math.sqrt(2) * 1.05,
                                   device="cuda")
    layout = mesh_rectangles(((0, side, 0, side, plane),), table.planes, side / 8)
    actual = assemble_layered(layout, table)
    basis = layout.basis
    k = 2 * math.pi * frequency / C0 * math.sqrt(epsilon)
    config = IntegrationConfig(order=8, singular_order=16, near_tolerance=1e-7,
                               near_max_order=32, pair_batch=4, far_pair_batch=64)
    vector, scalar, _ = assemble_single_layer(((basis, 0, 0),), k, config)
    div = basis.divergence.to(torch.complex128)
    # single_layer's L=+i*w*mu*(V-div^T*S*div/k^2); R2a solves Z=-L.
    reference = -1j * (2 * math.pi * frequency) / (EPS0 * C0**2) * (
        vector - div.T @ scalar @ div / k**2)
    relative = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
    print(f"eps={epsilon}: whole-matrix relative Frobenius={relative.item():.12g}")
    assert relative.item() <= 1e-5
    support = torch.stack((basis.plus_faces, basis.minus_faces), -1)
    centres = layout.mesh.centroids[support]
    size2 = (layout.mesh.triangles - layout.mesh.triangles.roll(1, 1)).square().sum(-1).amax(-1)
    distances2 = (centres[:, None, :, None] - centres[None, :, None]).square().sum(-1)
    threshold2 = 4 * torch.maximum(size2[support][:, None, :, None], size2[support][None, :, None])
    far = (distances2 >= threshold2).all(dim=(-2, -1))
    far_error = ((actual[far] - reference[far]).abs() / reference[far].abs()).max().item()
    print(f"eps={epsilon}: maximum far-entry relative error={far_error:.12g}")
    assert far_error <= 1e-9
    # Exercise both arithmetic instantiations; the production default stays fp64.
    reduced = assemble_layered(layout, table, far_dtype=torch.float32)
    assert (torch.linalg.vector_norm(reduced - actual) / torch.linalg.vector_norm(actual)).item() < 1e-5


def test_analytic_inner_integrals():
    triangle = np.array(((0., 0., 0.), (1., 0., 0.), (.2, .8, 0.)))
    points = [(x, y, z) for x, y in ((.25, .2), (.5, 0.), (0., 0.), (1.1, -.1))
              for z in (0., 1e-3, 1e-2, .1, 1.)]
    points.extend(((.5, -1e-8, 0.), (1e-8, 1e-8, 0.), (.5, 1e-8, -1e-3)))
    expected = [static_triangle(triangle, np.array(point), triangle[0]) for point in points]
    triangles = torch.tensor(triangle, device="cuda").expand(len(points), -1, -1).contiguous()
    observers = torch.tensor(points, dtype=torch.float64, device="cuda")
    free = torch.tensor(triangle[0], device="cuda").expand(len(points), -1).contiguous()
    scalar, vector = triangle_static_integrals(triangles, observers, free)
    scalar, vector = scalar.cpu().numpy(), vector.cpu().numpy()
    scalar_ref = np.array([item[0] for item in expected])
    vector_ref = np.array([item[1] for item in expected])
    scalar_error = np.max(abs(scalar - scalar_ref) / abs(scalar_ref))
    vector_error = np.max(np.linalg.norm(vector - vector_ref, axis=-1)
                          / np.linalg.norm(vector_ref, axis=-1))
    print(f"Analytic scalar/vector max relative error={scalar_error:.12g}/{vector_error:.12g}")
    assert scalar_error <= 1e-10
    assert vector_error <= 1e-10


@pytest.fixture(scope="module")
def rfic():
    stack = LayerStack((Layer("silicon", 300e-6, 11.9, 10.), Layer("oxide1", 2e-6, 4.1),
                        Layer("oxide2", 3e-6, 4.1), Layer("nitride", .1e-6, 7.5),
                        Layer("oxide3", 4e-6, 4.1)), HalfSpace(11.9, 10.), HalfSpace())
    planes = (stack.interfaces()[1], stack.interfaces()[5])
    table = LayeredGreenTable.build(stack, planes, (1e9, 100e9), 40e-6, device="cuda")
    layout = mesh_rectangles(((0, 20e-6, 0, 20e-6, planes[0]),
                              (2e-6, 22e-6, 1e-6, 21e-6, planes[1])), table.planes, 5e-6)
    return layout, table


def test_reciprocity(rfic):
    layout, table = rfic
    for frequency in range(2):
        z = assemble_layered(layout, table, frequency_index=frequency)
        error = (torch.linalg.vector_norm(z - z.T) / torch.linalg.vector_norm(z)).item()
        print(f"frequency index {frequency}: transpose relative error={error:.12g}")
        assert error <= 1e-12


def test_three_assemblies_equal(rfic):
    layout, table = rfic
    first = assemble_layered(layout, table, source_tile=7)
    assert torch.equal(first, assemble_layered(layout, table, source_tile=7))
    assert torch.equal(first, assemble_layered(layout, table, source_tile=7))


def test_manhattan_conformity_and_rejections():
    planes = torch.tensor((0., .7), dtype=torch.float64, device="cuda")
    rectangles = ((0., 2., 0., 1., 0.), (0., 1., 0., 2., 0.), (.25, 1.5, .25, 1.5, .7))
    layout = mesh_rectangles(rectangles, planes, .3, h_edge=.04, growth_ratio=1.4,
                              cut_lines=[((.37,), (.63,)), ((.8,), ())])
    mesh, topology = layout.mesh, layout.mesh.topology
    assert bool((topology.face_count <= 2).all())
    assert bool((topology.face_count[topology.kind == 0] == 2).all())
    assert topology.orientable and topology.manifold_vertices and topology.components == 2
    assert torch.allclose(mesh.areas.sum(), mesh.areas.new_tensor(3 + 1.25**2), rtol=1e-13, atol=0)
    assert torch.allclose(mesh.normals[:, 2], torch.ones_like(mesh.normals[:, 2]), rtol=1e-15, atol=0)
    # A vertex strictly inside any unsplit edge would be a hanging node.
    a, b = mesh.vertices[topology.edges[:, 0]], mesh.vertices[topology.edges[:, 1]]
    delta = mesh.vertices[:, None] - a[None]
    collinear = torch.linalg.cross(delta, (b - a)[None]).norm(dim=-1) < 1e-13
    interior = (delta * (mesh.vertices[:, None] - b[None])).sum(-1) < -1e-13
    assert not bool((collinear & interior).any())
    for plane in range(2):
        xy = mesh.vertices[mesh.vertices[:, 2] == planes[plane]]
        for axis in range(2):
            spacing = xy[:, axis].unique(sorted=True).diff()
            assert spacing.max().item() <= .3 * (1 + 1e-12)
            ratios = torch.maximum(spacing[1:] / spacing[:-1], spacing[:-1] / spacing[1:])
            assert ratios.max().item() <= 1.4 * (1 + 1e-10)
            assert spacing[0].item() <= .04 * (1 + 1e-12)
    with pytest.raises(ValueError, match="Axis-aligned"):
        mesh_rectangles((((0., 0.), (1., .1), (1., 1.), (0., 1.)),), planes, .3)
    with pytest.raises(ValueError, match="Axis-aligned"):
        mesh_rectangles(((1., 0., 0., 1., 0.),), planes, .3)
    with pytest.raises(ValueError, match="off every metal plane"):
        mesh_rectangles(((0., 1., 0., 1., .1),), planes, .3)
