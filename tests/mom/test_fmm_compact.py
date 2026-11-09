"""Compact Taylor storage gives the same electromagnetic action as dense M2L."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.fmm_expansion import multipole_to_local
from witwin.maxwell.mom.fmm_compact import CompactElectromagneticM2L
from witwin.maxwell.mom.scalar_moment_action import scalar_moment_action

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('order,batch', [(0, 1), (3, 7), (6, 16)])
def test_compact_matches_dense_adjoint_current_gradients_and_capacity(order, batch, monkeypatch):
    medium = compile_medium(mw.Material(eps_r=1.5, mu_r=1.2, sigma_e=.02), 3e8)
    r = torch.tensor([1., .3, -.2], device='cuda', dtype=torch.float64)
    op = CompactElectromagneticM2L(r, medium, order, row_batch=batch)
    green, gradient = multipole_to_local(r, medium.k, order)
    gradient = gradient.movedim(0, -1)
    x = torch.randn((2, 3, 2, op.count, 4), device='cuda', dtype=torch.complex128, requires_grad=True)
    y = torch.randn_like(x)
    reference = scalar_moment_action(green, gradient, medium, x)
    torch.testing.assert_close(op.apply(x), reference, rtol=3e-12, atol=1e-10)
    torch.testing.assert_close(op.adjoint(y), scalar_moment_action(green, gradient, medium, y, adjoint=True),
                               rtol=3e-12, atol=1e-10)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        actual = op.apply(x)
    assert not saved
    derivative, = torch.autograd.grad(actual, x, y)
    torch.testing.assert_close(derivative, op.adjoint(y), rtol=3e-12, atol=1e-10)
    receipt = op.receipt()
    print(receipt)
    if order >= 3:
        assert receipt['retained_bytes'] < receipt['dense_table_bytes']
    import witwin.maxwell.mom.fmm_compact as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Capacity failure must precede Taylor coefficients')
    monkeypatch.setattr(module, 'helmholtz_taylor', forbidden)
    with pytest.raises(MemoryError, match='before coefficient'):
        CompactElectromagneticM2L(r, medium, order, retained_budget_bytes=op.retained_bytes-1)
    op.coefficients.add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.apply(x)



def test_compact_rejects_generated_table_overflow(monkeypatch):
    medium = compile_medium(mw.Material(), 3e8)
    r = torch.tensor([1., 0., 0.], device='cuda', dtype=torch.float64)
    # Isolate generated-weight overflow from the earlier Taylor recurrence guard.
    import witwin.maxwell.mom.fmm_compact as module
    monkeypatch.setattr(module, 'helmholtz_taylor', lambda *args: torch.full(
        (560,), 1e307, device='cuda', dtype=torch.complex128))
    op = CompactElectromagneticM2L(r, medium, 6)
    assert bool(torch.isfinite(op.coefficients).all())
    with pytest.raises(ValueError, match='Generated compact M2L table'):
        op.apply(torch.ones((2, op.count, 4), device='cuda', dtype=op.dtype))
