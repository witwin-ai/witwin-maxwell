"""Physical two-body direct-kernel solves through the shared iterative protocol."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.direct_operator import DirectHierarchicalOperator
from witwin.maxwell.mom.operator_protocol import SurfaceOperatorProtocol
from witwin.maxwell.mom.linalg import StoppingRule, residual_loss
from witwin.maxwell.mom.runtime import SystemDecodeHandle, MoMConfig
from witwin.maxwell.mom.lifting import build_lifted_rhs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize("backend", ["direct_h2", "fmm"])
def test_direct_kernel_physical_multi_rhs_krylov_and_riesz_gradient(monkeypatch, backend):
    import witwin.maxwell.mom.hmatrix as hmatrix
    def forbid(*args, **kwargs):
        raise AssertionError('Direct-kernel construction must not build reference H')
    monkeypatch.setattr(hmatrix.HierarchicalOperator, '__init__', forbid)
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2., mu_r=1.1), 3e8)
    entries = [(i, str(i), RWGBasis(sphere_mesh(radius=.01, center=(x, 0., 0.), subdivisions=0, device='cuda')), interior)
               for i, x in enumerate((-.5, .5))]
    system = SurfaceSystem.create(entries)
    integration = IntegrationConfig(order=3, singular_order=3)
    operator_type = DirectHierarchicalOperator
    if backend == 'fmm':
        from witwin.maxwell.mom.fmm_operator import MultipoleOperator
        operator_type = MultipoleOperator
    op = operator_type(ScalarSystemBlocks(system, exterior, integration), order=6, leaf_size=30)
    assert isinstance(op, SurfaceOperatorProtocol) and not op.supports_direct and not hasattr(op, 'matrix')
    assert len(op.actions.exterior.far) == 2
    left, right, _ = op.preconditioner_scales('rwg_diagonal')
    n = system.total_edges
    torch.testing.assert_close(left[:n]/left[n:], torch.full_like(left[:n], abs(exterior.eta)))
    torch.testing.assert_close(right[:n], left[n:])
    with pytest.raises(ValueError, match='supports none'):
        op.preconditioner_scales('row_norm')
    x = torch.randn((2, 120), device=op.device, dtype=op.dtype, requires_grad=True)
    y = torch.randn_like(x)
    gradient, = torch.autograd.grad(residual_loss(op, x, y), x)
    torch.testing.assert_close(gradient, op.adjoint(op.riesz_inverse(op.apply(x)-y)), rtol=1e-10, atol=1e-8)
    rhs_values = []
    for polarization in ('Ex', 'Ey'):
        decoder = SystemDecodeHandle(system, exterior, [mw.PlaneWave(polarization=polarization)],
                                     MoMConfig(integration=integration), {})
        rhs = torch.empty(120, device=op.device, dtype=op.dtype)
        for i, local in enumerate(decoder.body_handles):
            value, _ = build_lifted_rhs(local)
            count = local.basis.n_edges
            rhs[system.J_slices[i]], rhs[system.M_slices[i]] = value[:count], value[count:]
        rhs_values.append(rhs)
    rhs = torch.stack(rhs_values)
    dense, _ = assemble_system(system, exterior, integration)
    reference = torch.linalg.solve(dense, rhs.mT).mT
    def dual(value):
        return (value.conj()*op.riesz_inverse(value)).sum(-1).real.clamp_min(0).sqrt()
    scale = torch.cat((torch.ones(n, device=op.device), torch.full((n,), abs(exterior.eta), device=op.device)))
    for method in ('gmres', 'block_gmres'):
        solved, log = op.solve(rhs, StoppingRule(method=method, preconditioner='rwg_diagonal',
            rtol=1e-12, atol=0., maxiter=240, restart=120))
        assert log['converged'] and log['error_budget']['checked'], log
        original_dual = dual(solved@dense.mT-rhs)/dual(rhs)
        error = torch.linalg.vector_norm((solved-reference)/scale, dim=-1)/torch.linalg.vector_norm(reference/scale, dim=-1)
        print(method, 'original dual', float(original_dual.max()), 'current error', float(error.max()))
        assert float(original_dual.max()) <= 1e-8
        assert float(error.max()) <= 1e-7
    _, failed = op.solve(rhs, StoppingRule(preconditioner='rwg_diagonal',
        rtol=1e-12, atol=0., maxiter=1, restart=1))
    assert not failed['converged'] and not failed['inner_converged']
    assert not failed['error_budget']['checked'] and not failed['fallback_used']
    for rule in (StoppingRule(method='direct'), StoppingRule(fallback='direct')):
        with pytest.raises(ValueError, match='requires a dense operator'):
            op.solve(rhs, rule)
    assert not op.persistence_receipt()['executable_snapshot']
    assert not op.operator_stats()['physical_qualification']
    op.metric.diagonal.add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(x)
