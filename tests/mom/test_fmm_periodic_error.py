"""Absorbing shell bounds cover independent omitted-image increments."""
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec, LossyImageReference
from witwin.maxwell.mom.fmm_periodic_error import absorbing_image_tail_bound, PeriodicImageTailError
from witwin.maxwell.mom.fmm_periodic import PeriodicImageM2L
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget, KernelBoundUnavailable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_absorbing_shell_tail_covers_remote_image_increment(dimension):
    vectors = torch.tensor([[1.4, 0., 0.], [.2, 1.5, 0.], [.1, .2, 1.6]], device='cuda', dtype=torch.float64)[:dimension]
    lattice = LatticeSpec(vectors, torch.arange(1, dimension+1, device='cuda', dtype=torch.float64)@vectors*.13)
    r = vectors.new_tensor([.1, .05, .08])
    wave = 2.+.7j
    previous = None
    for shell in (1, 2, 4):
        bound = absorbing_image_tail_bound(lattice, r, .05, .05, wave, shell)
        delta = r+vectors.new_tensor([[.07, 0., 0.], [0., -.07, 0.], [0., 0., .09]])
        g, dg = LossyImageReference(lattice, shell, image_batch=32).evaluate(delta, wave)
        outer, douter = LossyImageReference(lattice, shell+5, image_batch=32).evaluate(delta, wave)
        assert float((outer-g).abs().max()) <= bound['green_absolute']
        assert float((douter-dg).abs().max()) <= bound['gradient_component_absolute']
        if previous is not None:
            assert bound['green_absolute'] < previous
        previous = bound['green_absolute']
    print(dimension, bound)
    with pytest.raises(KernelBoundUnavailable, match='separate'):
        absorbing_image_tail_bound(lattice, r, 10., 10., wave, 1)


def test_tail_budget_rejects_before_tables_and_strong_decay_stays_positive(monkeypatch):
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.zeros(3, device='cuda', dtype=torch.float64))
    r = lattice.bloch
    bound = absorbing_image_tail_bound(lattice, r, .1, .1, 2.+1000j, 4)
    assert 0 < bound['green_absolute'] < 1e-300
    import witwin.maxwell.mom.fmm_periodic as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Tail admission must precede coefficient construction')
    monkeypatch.setattr(module, 'multipole_to_local', forbidden)
    with pytest.raises(PeriodicImageTailError):
        PeriodicImageM2L(lattice, r, .1, .1, 2.+.1j, shells=1,
            image_tail_budget=KernelErrorBudget(1e-12, 1e-12))


def test_admitted_tail_receipt_is_copied_and_bound_to_operator():
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.zeros(3, device='cuda', dtype=torch.float64))
    op = PeriodicImageM2L(lattice, lattice.bloch, .1, .1, 2.+3j, order=2, shells=2,
        image_tail_budget=KernelErrorBudget(1e-3, 1e-3))
    receipt = op.receipt()
    assert receipt['image_tail_admitted']
    assert receipt['image_tail_bound']['gradient_component_absolute'] < 1e-3
    receipt['image_tail_bound']['green_absolute'] = -1
    assert op.receipt()['image_tail_bound']['green_absolute'] > 0
    op._tail_text = '{}'
    with pytest.raises(ValueError, match='changed'):
        op.assert_current()


def test_combined_budget_requires_retained_m2l_and_outer_tail(monkeypatch):
    from witwin.maxwell.mom.fmm_periodic_error import periodic_multipole_error_bound, PeriodicMultipoleError
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.zeros(3, device='cuda', dtype=torch.float64))
    r = lattice.bloch
    budget = KernelErrorBudget(1e-6, 1e-6)
    low = periodic_multipole_error_bound(lattice, r, .01, .01, 2.+3j, order=2, shells=5)
    assert low['image_tail']['gradient_component_absolute'] < budget.gradient_component_absolute
    assert low['gradient_component_absolute'] > budget.gradient_component_absolute
    import witwin.maxwell.mom.fmm_periodic as module
    original = module.multipole_to_local
    def forbidden(*args, **kwargs):
        raise AssertionError('Combined error must reject before M2L construction')
    monkeypatch.setattr(module, 'multipole_to_local', forbidden)
    with pytest.raises(PeriodicMultipoleError):
        PeriodicImageM2L(lattice, r, .01, .01, 2.+3j, order=2, shells=5, kernel_error_budget=budget)
    monkeypatch.setattr(module, 'multipole_to_local', original)
    op = PeriodicImageM2L(lattice, r, .01, .01, 2.+3j, order=6, shells=5, kernel_error_budget=budget)
    receipt = op.receipt()
    combined = receipt['combined_kernel_bound']
    assert receipt['combined_kernel_admitted']
    assert combined['gradient_component_absolute'] <= 1e-6
    assert combined['far_images'] == 10 and combined['exact_near_images'] == 1
    print(combined)
    combined['green_absolute'] = -1
    assert op.receipt()['combined_kernel_bound']['green_absolute'] > 0
