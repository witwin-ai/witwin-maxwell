"""Finite carrier phases cannot admit unrepresentable actions or changed backward maps."""
import pytest
import torch
from witwin.maxwell.mom.fmm_directional_shift import DirectionalFixedCarrierShift

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_fixed_carrier_finite_phase_action_overflow_underflow_and_saved_backward_binding():
    u = torch.tensor([1., 0., 0.], device='cuda', dtype=torch.float64)
    op = DirectionalFixedCarrierShift(u*700., u, 1j, 0)
    assert bool(torch.isfinite(op.source_phase) & torch.isfinite(op.receiver_phase)
                & (op.source_phase.abs() > 0) & (op.receiver_phase.abs() > 0))
    for fn, amplitude in ((op.source, 1e308), (op.receiver, 1e-308)):
        with pytest.raises(ValueError, match='translated moments exceed native precision'):
            fn(torch.full((1, 4), amplitude, device='cuda', dtype=torch.complex128))
    fixed = DirectionalFixedCarrierShift(u*.1, u, 5.+.2j, 0)
    value = torch.ones((2, 1, 4), device='cuda', dtype=torch.complex128, requires_grad=True)
    result = fixed.source(value)
    fixed.source_phase.add_(1)
    with pytest.raises(ValueError, match='changed'):
        torch.autograd.grad(result, value, torch.ones_like(result))
