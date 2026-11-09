"""pFFT grid projection and single-medium D/K actions."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh, sphere_mesh
from witwin.maxwell.mom.nearfield import scalar_edge_pair_channels
from witwin.maxwell.mom.operator import IntegrationConfig, PreparedRWGGeometry
from witwin.maxwell.mom.pfft_grid import PFFTGridSpec, PFFTProjection
from witwin.maxwell.mom.pfft_operator import PFFTRegion
from witwin.maxwell.mom.physics import compile_medium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
FREQUENCY = 299792458.0


@pytest.fixture(scope='module')
def setup():
    mesh = sphere_mesh(radius=.5, frequency=6, device='cuda')
    basis = RWGBasis(SurfaceMesh(mesh.vertices, mesh.faces))
    medium = compile_medium(mw.Material(eps_r=1.), FREQUENCY, background=True)
    geometry = PreparedRWGGeometry(basis)
    n = basis.n_edges
    rows = torch.arange(0, n, 7, device='cuda')
    er = rows[:, None].expand(-1, n).reshape(-1)
    ec = torch.arange(n, device='cuda')[None].expand(len(rows), -1).reshape(-1)
    d, k = scalar_edge_pair_channels(basis, basis, er, ec, medium, IntegrationConfig(order=4, singular_order=4),
                                     geometry=(geometry, geometry))
    return basis, medium, rows, d.reshape(len(rows), n), k.reshape(len(rows), n)


def test_projection_reproduces_basis_moments(setup):
    basis, *_ = setup
    projection = PFFTProjection(basis, PFFTGridSpec(spacing=.1, order=3))
    # Zeroth moments: total current vector and total charge (zero for every RWG).
    weights = projection.weights.sum(1)
    assert float(weights[:, 3].abs().max()) < 1e-12
    from witwin.maxwell.mom.operator import local_geometry
    from witwin.maxwell.mom.quadrature import triangle_rule
    edges, triangles, slopes = local_geometry(basis)
    points, w = triangle_rule(triangles, 3)
    integral = torch.zeros((basis.n_edges, 3), device='cuda', dtype=torch.float64)
    for local in range(3):
        current = slopes[:, local, None, None]*(points-triangles[:, None, local])
        integral.index_add_(0, edges[:, local], (current*w[..., None]).sum(1))
    torch.testing.assert_close(weights[:, :3], integral, rtol=1e-12, atol=1e-14)


def test_grid_entries_equal_grid_action_columns(setup):
    basis, medium, *_ = setup
    region = PFFTRegion.from_basis(basis, medium, PFFTGridSpec(spacing=.1, order=2))
    columns = torch.tensor([3, 50, 400], device='cuda')
    rows = torch.arange(basis.n_edges, device='cuda')
    for column in columns.tolist():
        unit = torch.zeros(basis.n_edges, device='cuda', dtype=torch.complex128)
        unit[column] = 1
        d, k = region.grid_entries(rows, torch.full_like(rows, column))
        torch.testing.assert_close(region.apply_single_layer(unit), d, rtol=1e-10, atol=1e-14)
        torch.testing.assert_close(region.apply_double_layer(unit), k, rtol=1e-10, atol=1e-14)


def test_single_layer_is_symmetric(setup):
    basis, medium, *_ = setup
    region = PFFTRegion.from_basis(basis, medium, PFFTGridSpec(spacing=.1, order=3))
    generator = torch.Generator(device='cuda').manual_seed(3)
    x, y = (torch.randn(basis.n_edges, generator=generator, device='cuda', dtype=torch.complex128) for _ in range(2))
    left, right = (y*region.apply_single_layer(x)).sum(), (x*region.apply_single_layer(y)).sum()
    assert abs(complex(left-right)) <= 1e-10*abs(complex(left))


@pytest.mark.parametrize('order, ppw, budget', [(2, 10, 3e-2), (3, 15, 3e-3)])
def test_precorrected_action_matches_original_rows(setup, order, ppw, budget):
    basis, medium, rows, exact_d, exact_k = setup
    region = PFFTRegion.from_basis(basis, medium, PFFTGridSpec(spacing=2*math.pi/medium.k.real/ppw, order=order))
    midpoints = (basis.mesh.vertices[basis.edges[:, 0]]+basis.mesh.vertices[basis.edges[:, 1]])/2
    near = torch.cdist(midpoints[rows], midpoints) < region.config.near_radius
    pr, pc = near.nonzero(as_tuple=True)
    grid_d, grid_k = region.grid_entries(rows[pr], pc)
    x = torch.randn(basis.n_edges, generator=torch.Generator(device='cuda').manual_seed(5), device='cuda',
                    dtype=torch.complex128)
    for exact, grid, action in ((exact_d, grid_d, region.apply_single_layer),
                                (exact_k, grid_k, region.apply_double_layer)):
        reference = exact@x
        corrected = action(x)[rows]+torch.zeros_like(reference).index_add_(0, pr, (exact[pr, pc]-grid)*x[pc])
        assert float(torch.linalg.vector_norm(corrected-reference)/torch.linalg.vector_norm(reference)) < budget


def test_configuration_validation():
    for bad in (dict(spacing=0.), dict(spacing=.1, order=0), dict(spacing=.1, precision='float64')):
        with pytest.raises(ValueError):
            PFFTGridSpec(**bad)
