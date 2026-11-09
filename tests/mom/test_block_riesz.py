"""Per-body compact Riesz actions without a global block-diagonal allocation."""
import pytest
import torch
from witwin.maxwell.mom.linalg import DenseOperator, CompactRWGRiesz, BlockCompactRWGRiesz, apply_test_riesz_inverse, residual_loss, solve_reference, StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_operator():
    grams = (torch.tensor([[2., .2], [.2, 1.]], device='cuda', dtype=torch.float64),
             torch.tensor([[3.]], device='cuda', dtype=torch.float64))
    metric = BlockCompactRWGRiesz(tuple(CompactRWGRiesz(g, 4.) for g in grams))
    a = torch.eye(6, device='cuda', dtype=torch.complex128)*(2+.3j)
    return DenseOperator(a, metric), grams


def test_block_inverse_order_batch_gradient_and_receipt():
    op, grams = make_operator()
    reference = torch.block_diag(*(g/4 for g in grams), *grams).to(op.dtype)
    r = torch.arange(18, device=op.device, dtype=torch.float64).reshape(3, 6).to(op.dtype)+.2j
    actual = apply_test_riesz_inverse(op, r)
    torch.testing.assert_close(actual, torch.linalg.solve(reference, r.mT).mT, rtol=1e-13, atol=1e-13)
    x = r.clone().requires_grad_()
    gradient = torch.autograd.grad(residual_loss(op, x, r), x)[0]
    expected = (op.matrix.mH@torch.linalg.solve(reference, (op.apply(x)-r).mT)).mT
    torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-13)
    receipt = op.persistence_receipt()
    assert receipt['test_riesz']['storage'] == 'block_compact_rwg_v1'
    assert len(op._riesz_factor) == 2 and sum(f.numel() for f in op._riesz_factor) == 5
    assert op.operator_stats()['storage_bytes'] > 0
    grams[0].mul_(2)
    assert op.persistence_receipt() != receipt
    reference = torch.block_diag(*(g/4 for g in grams), *grams).to(op.dtype)
    torch.testing.assert_close(apply_test_riesz_inverse(op, r), torch.linalg.solve(reference, r.mT).mT)


def test_block_compact_sealed_direct_and_autograd_guard():
    op, grams = make_operator()
    rhs = torch.ones(6, device=op.device, dtype=op.dtype)
    rule = StoppingRule(method='direct', direct_storage='sealed_host_restore_v1', rtol=1e-12)
    result, log = solve_reference(op, rhs, rule)
    assert log['converged']
    torch.testing.assert_close(result, rhs/(2+.3j))
    grams[1].requires_grad_()
    with pytest.raises(ValueError, match='non-autograd'):
        solve_reference(op, rhs, rule)
