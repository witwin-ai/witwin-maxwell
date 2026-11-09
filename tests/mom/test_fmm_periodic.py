"""Periodic remote M2L plus exact near matches independent finite image sums."""
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec, LossyImageReference
from witwin.maxwell.mom.fmm_periodic import PeriodicImageM2L
from witwin.maxwell.mom.fmm_expansion import monomials
from witwin.maxwell.mom.kernels import FREE_SPACE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_periodic_remote_expansion_and_exact_near_ownership(dimension):
    vectors = torch.tensor([[1.4, 0., 0.], [.2, 1.5, 0.], [.1, .2, 1.6]], device='cuda', dtype=torch.float64)[:dimension]
    bloch = torch.arange(1, dimension+1, device='cuda', dtype=torch.float64)@vectors*.13
    lattice = LatticeSpec(vectors, bloch)
    displacement = vectors.new_tensor([.1, .05, .08])
    wave = 2.+.7j
    op = PeriodicImageM2L(lattice, displacement, .05, .05, wave, order=6, shells=1, image_batch=7)
    target = vectors.new_tensor([[.01, -.005, .007], [-.008, .011, -.009]])
    source = vectors.new_tensor([[-.012, .004, .003], [.007, -.013, .005], [.01, .004, -.003]])
    bt, bs = (monomials(p, 6).to(op.green.dtype) for p in (target, source))
    actual = bt@op.green@bs.mT
    gradient = torch.einsum('ib,abc,jc->ija', bt, op.gradient, bs)
    delta = displacement+target[:, None]-source[None]
    for index in op.near_indices:
        g, dg = FREE_SPACE.evaluate(delta-lattice.translations(index), wave)
        phase = lattice.phase(index)
        actual += phase*g
        gradient += phase*dg
    reference, reference_gradient = LossyImageReference(lattice, 1, image_batch=5).evaluate(delta, wave)
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-13)
    torch.testing.assert_close(gradient, reference_gradient, rtol=1e-10, atol=1e-13)
    # Isolate remote images so dominant exact-near terms cannot hide errors.
    remote, remote_gradient = LossyImageReference(lattice, 1, image_batch=5).evaluate(delta, wave, exclude_zero=True)
    torch.testing.assert_close(bt@op.green@bs.mT, remote, rtol=1e-9, atol=1e-13)
    torch.testing.assert_close(torch.einsum('ib,abc,jc->ija', bt, op.gradient, bs), remote_gradient, rtol=1e-9, atol=1e-13)
    assert torch.equal(op.near_indices, torch.zeros((1, dimension), device='cuda', dtype=torch.int64))
    receipt = op.receipt()
    assert receipt['far_images']+receipt['exact_near_images'] == 3**dimension
    assert not receipt['outer_image_tail_certified']
    print(dimension, receipt)
    op.near_indices.add_(1)
    with pytest.raises(ValueError, match='changed'):
        op.assert_current()


def test_periodic_m2l_rejects_unqualified_lossless_and_insufficient_table_budget(monkeypatch):
    lattice = LatticeSpec(torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float64),
                          torch.zeros(3, device='cuda', dtype=torch.float64))
    r = torch.zeros(3, device='cuda', dtype=torch.float64)
    with pytest.raises(ValueError, match='positive absorption'):
        PeriodicImageM2L(lattice, r, .1, .1, 2.)
    import witwin.maxwell.mom.fmm_periodic as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Table admission must precede coefficient construction')
    monkeypatch.setattr(module, 'multipole_to_local', forbidden)
    with pytest.raises(MemoryError, match='one image table'):
        PeriodicImageM2L(lattice, r, .1, .1, 2.+1j, table_budget_bytes=1)


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_compact_periodic_coefficients_match_dense_actions_and_budget(dimension):
    import witwin.maxwell as mw
    from witwin.maxwell.mom.physics import compile_medium
    medium = compile_medium(mw.Material(eps_r=1.5, sigma_e=.02), 3e8)
    vectors = torch.tensor([[1.4, 0., 0.], [.2, 1.5, 0.], [.1, .2, 1.6]], device='cuda', dtype=torch.float64)[:dimension]
    lattice = LatticeSpec(vectors, vectors.sum(0)*.13)
    r = vectors.new_tensor([.1, .05, .08])
    args = (lattice, r, .05, .05, medium.k)
    dense = PeriodicImageM2L(*args, order=6, shells=1, image_batch=7)
    compact = PeriodicImageM2L(*args, order=6, shells=1, image_batch=7, table_storage='compact', row_batch=7)
    x = torch.randn((2, 2, compact.count, 4), device='cuda', dtype=torch.complex128)
    for adjoint in (False, True):
        torch.testing.assert_close(compact.action(medium, x, adjoint=adjoint),
            dense.action(medium, x, adjoint=adjoint), rtol=3e-12, atol=1e-10)
    torch.testing.assert_close(compact.near_indices, dense.near_indices)
    assert compact.receipt()['table_bytes'] == 11760
    assert compact.receipt()['table_bytes'] < dense.receipt()['table_bytes']/30
    coefficient_bytes = compact.coefficients.numel()*compact.coefficients.element_size()
    with pytest.raises(MemoryError, match='one image table'):
        PeriodicImageM2L(*args, order=6, table_storage='compact', table_budget_bytes=11760+coefficient_bytes-1)
    compact.coefficients.add_(1)
    with pytest.raises(ValueError, match='changed'):
        compact.action(medium, x)
