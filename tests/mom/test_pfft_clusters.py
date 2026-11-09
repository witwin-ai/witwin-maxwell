"""Clustered pFFT exterior: per-cluster grids with single-level plane-wave coupling (CUDA)."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
FREQUENCY = 299792458.0
INTEGRATION = IntegrationConfig(order=4, singular_order=4)
CENTRES = (((-1.2, 0., 0.), 4.), ((1.1, .3, 0.), 2.5), ((0., 1.4, .2), 3.))


def _scene():
    structures = []
    for centre, eps in CENTRES:
        mesh = sphere_mesh(radius=.25, frequency=5, center=centre, device='cuda')
        structures.append(mw.Structure(mw.Mesh(mesh.vertices, mesh.faces, recenter=False, fill_mode='solid'),
                                       mw.Material(eps_r=eps)))
    return mw.Scene(structures=structures, sources=[mw.PlaneWave()])


def _config(grids, **kwargs):
    return mw.MoMConfig(integration=INTEGRATION, operator_backend='pfft', riesz_storage='local_rwg_v1',
                        pfft=mw.PFFTConfig(exterior_grids=grids, solve_dual_tolerance=1e-4,
                                           original_sample_tolerance=3e-2),
                        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', rtol=1e-5, atol=0.,
                                              maxiter=2000, restart=300, stagnation_window=300), **kwargs)


def _body_lu(context):
    blocks = []
    for body in context.system.bodies:
        edges = torch.arange(body.edge_offset, body.edge_offset+body.basis.n_edges, device=context.device)
        blocks.append((context.coefficient_indices(edges), torch.linalg.lu_factor(context.principal_block(edges))))

    def apply(residual):
        result = torch.empty_like(residual)
        for index, (factor, pivot) in blocks:
            result[index] = torch.linalg.lu_solve(factor, pivot, residual[index, None])[:, 0]
        return result
    return apply


def _relative(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))


@pytest.fixture(scope='module')
def prepared():
    dense = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=mw.MoMConfig(
        integration=INTEGRATION, riesz_storage='compact_rwg_v1',
        stopping=StoppingRule(method='direct', rtol=1e-10, atol=0.))).prepare()
    single = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=_config('single')).prepare()
    clusters = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=_config('clusters')).prepare()
    return dense, single, clusters


def test_translation_reproduces_the_green_function():
    from witwin.maxwell.mom.pfft_coupling import sphere_quadrature, translation, truncation_order
    k = 2*math.pi
    displacement = torch.tensor([1.3, -0.4, 0.9], device='cuda', dtype=torch.float64)
    radius = 0.35
    order = truncation_order(k, 2*radius, 6)
    directions, weights = sphere_quadrature(order, 'cuda')
    t = translation(order, k, displacement, directions)
    generator = torch.Generator(device='cuda').manual_seed(0)
    d = torch.nn.functional.normalize(torch.randn((50, 3), generator=generator, device='cuda', dtype=torch.float64),
                                      dim=-1)*radius*torch.rand((50, 1), generator=generator, device='cuda',
                                                                dtype=torch.float64)*2
    distance = torch.linalg.vector_norm(displacement+d, dim=-1)
    exact = torch.exp(1j*k*distance)/(4*math.pi*distance)
    approximate = 1j*k/(16*math.pi**2)*(torch.exp(1j*k*(d@directions.T))*(weights*t)).sum(-1)
    assert float(((approximate-exact).abs()/exact.abs()).max()) < 1e-5


def test_clustered_action_matches_single_grid_and_dense(prepared):
    dense, single, clusters = prepared
    handle = clusters.solver.handle
    receipt = handle.persistence_receipt()['construction']['regions']['exterior']
    assert len(receipt['clusters']) == 3 and receipt['coupling']['minimum_separation_ratio'] >= 2
    generator = torch.Generator(device='cuda').manual_seed(5)
    n = handle.shape[0]
    x, y = (torch.randn(n, generator=generator, device='cuda', dtype=torch.complex128) for _ in range(2))
    exact = dense.solver.handle.apply(x)
    reference = single.solver.handle.apply(x)
    action = handle.apply(x)
    assert _relative(action, exact) < 3e-3
    assert _relative(action, reference) < 3e-3
    left, right = torch.vdot(y, action), torch.vdot(handle.adjoint(y), x)
    assert abs(complex(left-right)) <= 1e-6*abs(complex(left))


def test_clustered_solve_matches_dense(prepared):
    dense, _, _ = prepared
    reference = dense.run()
    config = _config('clusters', right_preconditioner=mw.FunctionRightPreconditioner(_body_lu, name='body_lu'))
    result = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=config).run()
    stats = result.stats()
    assert stats['converged'] and stats['iterations'] < 60
    directions = torch.nn.functional.normalize(torch.randn((64, 3), device='cuda', dtype=torch.float64), dim=-1)
    far, _ = result.surface_currents.far_field(directions)
    far_ref, _ = reference.surface_currents.far_field(directions)
    assert _relative(far, far_ref) < 5e-3


def test_clusters_merge_close_bodies_and_fail_closed():
    from witwin.maxwell.mom.pfft_coupling import form_clusters
    from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh
    bases = [RWGBasis(SurfaceMesh(m.vertices, m.faces)) for m in
             (sphere_mesh(radius=.25, frequency=2, center=c, device='cuda') for c in
              ((0., 0., 0.), (.7, 0., 0.), (3., 0., 0.)))]
    groups, _ = form_clusters(bases, 2.)
    assert groups == [[0, 1], [2]]
    with pytest.raises(ValueError):
        mw.PFFTConfig(exterior_grids='per_body')
    with pytest.raises(ValueError):
        mw.PFFTConfig(cluster_separation=1.)
