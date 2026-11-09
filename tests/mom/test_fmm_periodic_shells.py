"""Adaptive image cubes bind both action and original residual to selected shells."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.periodic_nearfield import PeriodicImageSystemBlocks
from witwin.maxwell.mom.fmm_periodic_operator import PeriodicMultipoleOperator
from witwin.maxwell.mom.fmm_periodic_order import PeriodicOrderError
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_selected_cube_binds_krylov_reference_and_preserves_failed_smaller_cubes(monkeypatch):
    exterior = compile_medium(mw.Material(eps_r=1.5, sigma_e=.02), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.2), 3e8)
    basis = RWGBasis(sphere_mesh(radius=.01, subdivisions=0, device='cuda'))
    system = SurfaceSystem.create([(0, 'body', basis, interior)])
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.tensor([.37, 0., 0.], device='cuda', dtype=torch.float64))
    provider = PeriodicImageSystemBlocks(system, exterior, IntegrationConfig(order=4, singular_order=3), lattice, shells=1)
    kwargs = dict(order=3, max_order=6, max_shells=12, separation_ratio=.1, leaf_size=30,
        table_storage='compact', exterior_kernel_error_budget=KernelErrorBudget(1e-6, 1e-5),
        interior_kernel_error_budget=KernelErrorBudget(1e-6, 1e-5))
    op = PeriodicMultipoleOperator(provider, **kwargs)
    plan = op.actions.order_plan
    print('selected periodic parameters', plan['selected_shells'], plan['selected_order'])
    assert 1 < plan['selected_shells'] <= 12
    assert all(not a['admitted'] for a in plan['attempts'][:-1])
    assert op.provider is op.actions.provider and op.provider is not provider
    assert provider.shells == 1
    assert op.provider.shells == op.actions.exterior.remote.shells == plan['selected_shells']
    edges = torch.arange(basis.n_edges, device='cuda')
    reference = op.provider(edges, edges)
    original = provider(edges, edges)
    assert float(torch.linalg.vector_norm(reference-original)) > 0
    torch.manual_seed(428)
    target = torch.randn((2, 2*basis.n_edges), device='cuda', dtype=torch.complex128)
    rhs = target@reference.mT
    solved, log = op.solve(rhs, StoppingRule(method='block_gmres', preconditioner='rwg_diagonal',
        rtol=1e-12, atol=0., maxiter=100, restart=60), original_edges=edges)
    assert log['converged'] and log['error_budget']['checked']
    residual = float(torch.linalg.vector_norm(solved@reference.mT-rhs)/torch.linalg.vector_norm(rhs))
    print('selected-cube residual', residual)
    assert residual < 1e-8
    import witwin.maxwell.mom.fmm_periodic_system as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Exhausted shell search must reject before moments')
    monkeypatch.setattr(module, 'NestedRWGMultipoles', forbidden)
    with pytest.raises(PeriodicOrderError) as failure:
        PeriodicMultipoleOperator(provider, **{**kwargs, 'max_shells': plan['selected_shells']-1})
    assert failure.value.receipt['reason'] == 'shell_order_range_exhausted'
    op.provider.shells += 1
    with pytest.raises(ValueError, match='changed'):
        op.apply(target)
