"""Ideal RWG multipole error admission before numerical expansion setup."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_rwg_error import system_multipole_bound
from witwin.maxwell.mom.fmm_order import MultipoleOrderError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_rwg_admission_matches_constructed_bound_and_rejects_before_moments(monkeypatch):
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.2), 3e8)
    system = SurfaceSystem.create([(i, str(i), RWGBasis(sphere_mesh(radius=.01,
        center=(x, 0., 0.), subdivisions=0, device='cuda')), interior) for i, x in enumerate((-.5, .5))])
    provider = ScalarSystemBlocks(system, exterior, IntegrationConfig(order=2, singular_order=3))
    settings = dict(order=2, max_order=10, kernel_error_budget=mw.KernelErrorBudget(1., 10.),
                    rwg_absolute_budget=1e-6, leaf_size=30)
    op = MultipoleSystemActions(provider, **settings)
    plan = op.order_plan
    actual = system_multipole_bound(op)
    print('RWG multipole admission', plan['selected_order'], actual['absolute_bound'])
    assert actual['absolute_bound'] <= 1e-6
    assert actual['absolute_bound'] == pytest.approx(plan['rwg_bound']['absolute_bound'], rel=1e-13)
    assert all(a['kernel_pass'] and not a['rwg_pass'] for a in plan['attempts'][:-1])
    assert not actual['implemented_operator_error_certified']
    assert actual['exterior']['polynomial_exact_in_exact_arithmetic']
    config = mw.FMMConfig(**{k: v for k, v in settings.items() if k != 'leaf_size'})
    assert mw.FMMConfig.restore(config.payload()) == config
    import witwin.maxwell.mom.fmm_system as construction
    def forbidden(*args, **kwargs):
        raise AssertionError('Failed RWG admission must precede moment construction')
    monkeypatch.setattr(construction, 'NestedRWGMultipoles', forbidden)
    with pytest.raises(ValueError, match='polynomial-exact'):
        MultipoleSystemActions(provider, **settings, quadrature_order=1)
    with pytest.raises(MultipoleOrderError):
        MultipoleSystemActions(provider, **{**settings, 'max_order': 2})


def test_unavailable_kernel_does_not_report_partial_rwg_bound():
    from witwin.maxwell.mom.cluster import build_support_octree
    from witwin.maxwell.mom.fmm_order import plan_multipole_order
    lower = torch.tensor([[0., 0., 0.], [1., 0., 0.]], device='cuda', dtype=torch.float64)
    tree = build_support_octree(lower, lower+.45/(3**.5), leaf_size=1)
    statistics = torch.zeros((tree.node_count, 3), device='cuda', dtype=torch.float64)
    with pytest.raises(MultipoleOrderError) as error:
        plan_multipole_order(((tree, compile_medium(mw.Material(), 3e8)),), mw.KernelErrorBudget(1., 10.),
            max_order=1, rwg_statistics=(statistics,), rwg_absolute_budget=1., balancing_impedance=377.)
    assert all(a['unavailable_bounds'] == 2 and a['rwg_bound'] is None and not a['rwg_pass']
               for a in error.value.receipt['attempts'])
