"""Fixed quadrature policy must remain guarded across actions and backward."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.scalar_near_channels import ScalarNearChannels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_complete_integration_configuration_bound_in_actions_receipt_and_backward():
    basis = RWGBasis(sphere_mesh(radius=.02, subdivisions=0, device='cuda', dtype=torch.float64))
    rows = torch.tensor([7, 2], device='cuda'); cols = torch.tensor([5, 1], device='cuda')
    config = IntegrationConfig(order=4, singular_order=3)
    op = ScalarNearChannels(basis, basis, rows, cols, compile_medium(mw.Material(), 3e8), integration=config)
    value = torch.ones(op.shape[1], device='cuda', dtype=torch.complex128, requires_grad=True)
    output = op.apply(value)
    # Frozen configurations can still be changed deliberately through object.__setattr__.
    changes = dict(order=5, singular_order=4, pair_batch=65, far_pair_batch=1025,
                   near_tolerance=2e-5, near_max_order=17)
    for field, changed in changes.items():
        original = getattr(config, field)
        object.__setattr__(config, field, changed)
        try:
            for action in (op.assert_current, op.receipt, lambda: op.apply(value),
                           lambda: op.adjoint(torch.ones(op.shape[0], device='cuda', dtype=torch.complex128)),
                           lambda: torch.autograd.grad(output, value, torch.ones_like(output), retain_graph=True)):
                with pytest.raises(ValueError, match='changed'):
                    action()
        finally:
            object.__setattr__(config, field, original)
        op.assert_current()
    torch.autograd.grad(output, value, torch.ones_like(output))
