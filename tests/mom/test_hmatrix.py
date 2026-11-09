"""Actual hierarchical scalar actions against an independently assembled oracle."""
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.hmatrix import HMatrixConfig, HierarchicalOperator
from witwin.maxwell.mom.lowrank import LowRankBlock
from witwin.maxwell.mom.operator_protocol import SurfaceOperatorProtocol
from witwin.maxwell.mom.linalg import residual_loss, solve_reference, StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_actual_hierarchical_operator_forward_adjoint_metric_and_gradient():
    a = RWGBasis(sphere_mesh(radius=.01, center=(-.5, 0., 0.), subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.01, center=(.5, 0., 0.), subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(0, 'a', a, interior), (1, 'b', b, interior)])
    config = IntegrationConfig(order=3, singular_order=3)
    torch.cuda.synchronize(system.device)
    torch.cuda.reset_peak_memory_stats(system.device)
    allocated_before = torch.cuda.memory_allocated(system.device)
    provider = ScalarSystemBlocks(system, exterior, config)
    op = HierarchicalOperator(provider, HMatrixConfig(leaf_size=32, max_rank=32, tolerance=1e-8, pivoting='adaptive'))
    torch.cuda.synchronize(system.device)
    setup_peak_increment = torch.cuda.max_memory_allocated(system.device)-allocated_before
    assert isinstance(op, SurfaceOperatorProtocol) and not hasattr(op, 'matrix')
    assert op.metadata['far_blocks'] == 2 and op.metadata['exact_blocks'] == 2
    dense, _ = assemble_system(system, exterior, config)
    identity = torch.eye(len(dense), device=op.device, dtype=op.dtype)
    reconstructed = op.apply(identity).mT
    rs = torch.cat((torch.full((60,), abs(exterior.eta), device='cuda', dtype=torch.float64),
                    torch.ones(60, device='cuda', dtype=torch.float64)))
    cs = rs.flip(0)
    error = torch.linalg.vector_norm((reconstructed-dense)*rs[:, None]*cs[None, :])
    scale = torch.linalg.vector_norm(dense*rs[:, None]*cs[None, :])
    assert float(error/scale) <= 1e-8
    for rows, cols, value in op.blocks:
        if not isinstance(value, LowRankBlock):
            torch.testing.assert_close(value, dense[rows[:, None], cols], rtol=2e-12, atol=1e-16)
    g = torch.Generator(device='cuda').manual_seed(73)
    x = torch.randn((3, len(dense)), generator=g, device=op.device, dtype=op.dtype, requires_grad=True)
    y = torch.randn((3, len(dense)), generator=g, device=op.device, dtype=op.dtype)
    left = (y.conj()*op.apply(x)).sum()
    right = (op.adjoint(y).conj()*x).sum()
    assert float(((left-right).abs()/(torch.linalg.vector_norm(y)*torch.linalg.vector_norm(x))).detach()) < 1e-11
    loss = residual_loss(op, x, y)
    gradient, = torch.autograd.grad(loss, x)
    expected = op.adjoint(op.riesz_inverse(op.apply(x)-y))
    torch.testing.assert_close(gradient, expected, rtol=1e-10, atol=1e-8)
    tensors = list(op.tree.tensors())+list(op.metric.tensors())+list(op._block_tensors())+list(provider.geometry_tensors())
    assert op.metadata['retained_bytes'] == sum(t.numel()*t.element_size() for t in tensors)
    storage = op.storage_breakdown()
    assert storage['owned_retained_bytes'] == op.metadata['retained_bytes']
    assert storage['total_referenced_tensor_bytes'] == storage['owned_retained_bytes']+storage['shared_input_bytes']
    print({'relative_block_frobenius_error': float(error/scale),
           'retained_bytes': op.metadata['retained_bytes'], 'dense_matrix_bytes': dense.numel()*dense.element_size(),
           'far_blocks': op.metadata['far_blocks'], 'exact_blocks': op.metadata['exact_blocks'],
           'storage': storage, 'torch_cuda_setup_peak_increment_bytes': setup_peak_increment,
           'peak_scope': 'shared pytest process; existing cache bytes excluded'})
    # Compare equilibration to the actual compressed object, not the original
    # matrix: this isolates scaling algebra from the approximation error.
    for kind in ('row_norm', 'row_column_norm'):
        row_scale, col_scale, _ = op.preconditioner_scales(kind)
        expected_col = (torch.linalg.vector_norm(reconstructed, dim=0).reciprocal()
                        if kind == 'row_column_norm' else torch.ones_like(col_scale))
        expected_row = torch.linalg.vector_norm(reconstructed*expected_col[None, :], dim=1).reciprocal()
        torch.testing.assert_close(col_scale, expected_col, rtol=2e-12, atol=1e-14)
        torch.testing.assert_close(row_scale, expected_row, rtol=2e-12, atol=1e-14)
    from witwin.maxwell.mom.runtime import SystemDecodeHandle, MoMConfig
    from witwin.maxwell.mom.lifting import build_lifted_rhs
    rhs_columns = []
    for polarization in ('Ex', 'Ey'):
        decode = SystemDecodeHandle(system, exterior, [mw.PlaneWave(polarization=polarization)],
                                    MoMConfig(integration=config), {})
        rhs = torch.empty(op.shape[0], device=op.device, dtype=op.dtype)
        for i, handle in enumerate(decode.body_handles):
            local, _ = build_lifted_rhs(handle)
            n = handle.basis.n_edges
            rhs[system.J_slices[i]], rhs[system.M_slices[i]] = local[:n], local[n:]
        rhs_columns.append(rhs)
    rhs = torch.stack(rhs_columns)
    reference_solution = torch.linalg.solve(dense, rhs.mT).mT
    conditions = {}
    for kind in ('row_norm', 'row_column_norm'):
        left_scale, right_scale, _ = op.preconditioner_scales(kind)
        conditions[kind] = float(torch.linalg.cond(left_scale[:, None]*reconstructed*right_scale[None, :]))
    print({'equilibrated_condition_numbers': conditions})
    for method, kind in (('gmres', 'row_norm'), ('gmres', 'row_column_norm'), ('block_gmres', 'row_column_norm')):
        solved, log = solve_reference(op, rhs, StoppingRule(method=method, preconditioner=kind,
            rtol=1e-12, atol=0., maxiter=240, restart=120))
        if not log['converged'] and kind == 'row_norm':
            # Row-only equilibration can stagnate at this strict raw tolerance.
            # Preserve its failure receipt; robust acceptance below uses both
            # sides of the same operator, at unchanged residual/current limits.
            assert log['reason'] == 'batch_failure', log
            for item in log['rhs_logs']:
                if not item['converged']:
                    assert item['reason'] == 'stagnation', item
                    assert math.isfinite(item['absolute_residual']), item
                    assert item['absolute_residual'] > item['threshold'], item
            print({'unqualified_row_norm_attempt': [
                {key: item.get(key) for key in ('converged', 'reason', 'iterations', 'absolute_residual', 'relative_residual', 'threshold')}
                for item in log['rhs_logs']]})
            continue
        assert log['converged'], log
        own_residual = op.apply(solved)-rhs
        original_residual = solved@dense.mT-rhs
        def dual_norm(value):
            return (value.conj()*op.riesz_inverse(value)).sum(-1).real.clamp_min(0).sqrt()
        own = dual_norm(own_residual)/dual_norm(rhs)
        original = dual_norm(original_residual)/dual_norm(rhs)
        print({'method': method, 'preconditioner': kind, 'own_dual_before_acceptance': float(own.max()),
               'original_dual_before_acceptance': float(original.max())})
        assert float(own.max()) < 1e-8
        assert float(original.max()) < 1e-8
        relative_currents = torch.linalg.vector_norm((solved-reference_solution)/cs, dim=-1)/torch.linalg.vector_norm(reference_solution/cs, dim=-1)
        assert float(relative_currents.max()) < 1e-7
        print({'method': method, 'preconditioner': kind, 'own_dual_residual': float(own.max()),
               'original_dual_residual': float(original.max()), 'current_difference': float(relative_currents.max())})
    op.blocks[0][0].add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(x)


def test_original_residual_report_samples_actual_rows_with_bounded_sources():
    basis = RWGBasis(sphere_mesh(radius=.05, subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.), 3e8)
    system = SurfaceSystem.create([(0, 'body', basis, interior)])
    config = IntegrationConfig(order=2, singular_order=3)
    op = HierarchicalOperator(ScalarSystemBlocks(system, exterior, config))
    g = torch.Generator(device='cuda').manual_seed(19)
    x = torch.randn((2, 60), device='cuda', dtype=op.dtype, generator=g)
    rhs = op.apply(x)
    rhs[:, 0] += .2
    edges = torch.tensor([29, 0, 7], device='cuda')
    report = op.residual_report(x, rhs, original_edges=edges, source_batch=7)
    dense, _ = assemble_system(system, exterior, config)
    indices = torch.cat((edges, edges+30))
    scale = torch.tensor([abs(exterior.eta)]*3+[1.]*3, device='cuda', dtype=torch.float64)
    expected = torch.linalg.vector_norm((x@dense.mT-rhs)[..., indices]*scale, dim=-1)
    actual = torch.tensor(report['original_sampled']['absolute'], device='cuda', dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-10)
    assert report['original_edges'] == [29, 0, 7]
    assert not report['original_global_residual_certified']
