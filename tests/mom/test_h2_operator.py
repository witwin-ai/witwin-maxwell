"""Shared H2 Krylov, nested equilibration and original-operator residuals."""
import gc
import weakref
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.hmatrix import HierarchicalOperator, HMatrixConfig
from witwin.maxwell.mom.h2 import H2Config
from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
from witwin.maxwell.mom.operator_protocol import SurfaceOperatorProtocol
from witwin.maxwell.mom.linalg import StoppingRule, residual_loss
from witwin.maxwell.mom.runtime import SystemDecodeHandle, MoMConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_actual_h2_protocol_equilibration_dual_gradient_and_physical_multi_rhs(monkeypatch):
    a = RWGBasis(sphere_mesh(radius=.01, center=(-.5, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.01, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(0, 'a', a, interior), (1, 'b', b, interior)])
    integration = IntegrationConfig(order=3, singular_order=3)
    h = HierarchicalOperator(ScalarSystemBlocks(system, exterior, integration), HMatrixConfig(pivoting='adaptive'))
    op = HierarchicalH2Operator(h)
    h_reference = weakref.ref(h)
    del h
    gc.collect()
    assert h_reference() is None
    assert isinstance(op, SurfaceOperatorProtocol) and not op.supports_direct
    assert not hasattr(op, 'matrix') and not hasattr(op.actions, 'blocks')
    identity = torch.eye(120, device=op.device, dtype=op.dtype)
    reconstructed = op.apply(identity).mT
    dense, _ = assemble_system(system, exterior, integration)
    # Equilibration must use small nested Gram/form traversals, never expanded
    # internal bases or a reconstructed far block after construction.
    def forbidden(*args, **kwargs):
        raise AssertionError('Solver actions and norms must not expand node bases')
    monkeypatch.setattr(op.actions.test_basis, 'expand', forbidden)
    monkeypatch.setattr(op.actions.trial_basis, 'expand', forbidden)
    for kind in ('row_norm', 'row_column_norm'):
        left, right, _ = op.preconditioner_scales(kind)
        expected_right = (torch.linalg.vector_norm(reconstructed, dim=0).reciprocal()
                          if kind == 'row_column_norm' else torch.ones_like(right))
        expected_left = torch.linalg.vector_norm(reconstructed*expected_right[None, :], dim=1).reciprocal()
        torch.testing.assert_close(right, expected_right, rtol=2e-12, atol=1e-14)
        torch.testing.assert_close(left, expected_left, rtol=2e-12, atol=1e-14)
    x = torch.randn((2, 120), device=op.device, dtype=op.dtype, requires_grad=True)
    y = torch.randn_like(x)
    gradient, = torch.autograd.grad(residual_loss(op, x, y), x)
    torch.testing.assert_close(gradient, op.adjoint(op.riesz_inverse(op.apply(x)-y)), rtol=1e-10, atol=1e-8)
    from witwin.maxwell.mom.lifting import build_lifted_rhs
    rhs_values = []
    for polarization in ('Ex', 'Ey'):
        decoder = SystemDecodeHandle(system, exterior, [mw.PlaneWave(polarization=polarization)],
                                     MoMConfig(integration=integration), {})
        rhs = torch.empty(120, device=op.device, dtype=op.dtype)
        for i, local in enumerate(decoder.body_handles):
            value, _ = build_lifted_rhs(local)
            n = local.basis.n_edges
            rhs[system.J_slices[i]], rhs[system.M_slices[i]] = value[:n], value[n:]
        rhs_values.append(rhs)
    rhs = torch.stack(rhs_values)
    reference = torch.linalg.solve(dense, rhs.mT).mT
    def dual(value):
        return (value.conj()*op.riesz_inverse(value)).sum(-1).real.clamp_min(0).sqrt()
    for method in ('gmres', 'block_gmres'):
        solved, log = op.solve(rhs, StoppingRule(method=method, preconditioner='row_column_norm',
            rtol=1e-12, atol=0., maxiter=240, restart=120))
        assert log['converged'] and log['error_budget']['checked'], log
        original_dual = dual(solved@dense.mT-rhs)/dual(rhs)
        current_error = torch.linalg.vector_norm((solved-reference)/op.actions.column_scale, dim=-1)/torch.linalg.vector_norm(reference/op.actions.column_scale, dim=-1)
        assert float(original_dual.max()) <= 1e-8
        assert float(current_error.max()) <= 1e-7
        print({'method': method, 'original_dual_residual': float(original_dual.max()),
               'current_error': float(current_error.max()), 'sampled_original': log['error_budget']['residual_report']['original_sampled']})
    receipt = op.persistence_receipt()
    assert receipt['configuration'] == op.config.payload() and not receipt['executable_snapshot']
    storage = op.storage_breakdown()
    print({'h2_solver_storage': storage})
    assert storage['total_referenced_tensor_bytes'] > storage['action']['total_referenced_tensor_bytes']
    next(iter(op.metric.tensors())).add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(x)


def test_h2_configuration_preserves_conversion_and_solve_budgets():
    config = H2Config(max_rank=17, tolerance=2e-11, solve_dual_tolerance=3e-9, original_sample_tolerance=4e-9)
    assert H2Config.restore(config.payload()) == config
    with pytest.raises(ValueError):
        H2Config(original_sample_tolerance=True)
    payload = config.payload()
    payload['parameters'].pop('solve_dual_tolerance')
    with pytest.raises(ValueError):
        H2Config.restore(payload)
