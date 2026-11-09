"""Residual admission distinguishes inner convergence from original-row checks."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.hmatrix import HMatrixConfig, HierarchicalOperator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture
def operator():
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(0, 'body', basis, interior)])
    return HierarchicalOperator(ScalarSystemBlocks(system, exterior, IntegrationConfig(order=2, singular_order=3)))


def test_versioned_configuration_keeps_all_budget_parameters():
    config = HMatrixConfig(max_rank=17, solve_dual_tolerance=2e-9, original_sample_tolerance=3e-9, pivoting='adaptive')
    assert HMatrixConfig.restore(config.payload()) == config
    for key, value in (('version', 'unknown'), ('parameters', {})):
        corrupted = copy.deepcopy(config.payload())
        corrupted[key] = value
        with pytest.raises(ValueError):
            HMatrixConfig.restore(corrupted)
    with pytest.raises(ValueError):
        HMatrixConfig(solve_dual_tolerance=True)
    with pytest.raises(ValueError):
        HMatrixConfig(pivoting='unverified')


def test_actual_solve_requires_both_residual_budgets(operator):
    g = torch.Generator(device='cuda').manual_seed(201)
    exact = torch.randn(operator.shape[0], device='cuda', dtype=operator.dtype, generator=g)
    rhs = operator.apply(exact)
    value, log = operator.solve(rhs)
    assert log['converged'] and log['inner_converged'], log
    assert log['error_budget']['compressed_dual_pass'] and log['error_budget']['original_sampled_pass']
    assert not log['original_global_residual_certified'] and not log['physical_qualification']
    torch.testing.assert_close(value, exact, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('failure', ['compressed', 'original', 'inner'])
def test_overstated_inner_success_cannot_bypass_admission(operator, monkeypatch, failure):
    import witwin.maxwell.mom.linalg as linalg
    exact = torch.ones(operator.shape[0], device='cuda', dtype=operator.dtype)
    rhs = operator.apply(exact)
    result = exact*1.1 if failure == 'compressed' else exact
    monkeypatch.setattr(linalg, 'solve_reference', lambda *args, **kwargs:
        (result, {'converged': failure != 'inner', 'reason': 'stagnation' if failure == 'inner' else 'converged'}))
    if failure == 'original':
        original = operator.provider.sampled_action
        monkeypatch.setattr(operator.provider, 'sampled_action', lambda *args, **kwargs: original(*args, **kwargs)+.1)
    _, log = operator.solve(rhs)
    assert not log['converged']
    if failure == 'inner':
        assert log['reason'] == 'stagnation' and not log['error_budget']['checked']
    else:
        expected = 'compressed_dual_residual_above_budget' if failure == 'compressed' else 'original_sampled_residual_above_budget'
        assert log['reason'] == expected and log['inner_converged']
        assert log['error_budget']['checked']
