"""Row-blocked operator/metric validation keeps large-matrix transients bounded."""
import pytest
import torch

from witwin.maxwell.mom import linalg
from witwin.maxwell.mom.linalg import CompactRWGRiesz, DenseOperator, apply_A
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_mass_matrix_in_place_accumulation_is_bitwise_identical():
    basis = RWGBasis(sphere_mesh(subdivisions=1))
    points, weights = basis.mesh.quadrature(2)
    values = basis.local_values(points)
    blocks = torch.einsum("tqid,tqjd,tq->tij", values, values, weights)
    indices = basis.face_edges[:, :, None] * basis.n_edges + basis.face_edges[:, None, :]
    reference = torch.zeros(basis.n_edges**2, device='cuda', dtype=torch.float64)
    reference = reference.scatter_add(0, indices.reshape(-1), blocks.reshape(-1)).reshape(basis.n_edges, basis.n_edges)
    assert torch.equal(basis.mass_matrix(), reference)


@pytest.mark.parametrize('rows', [1, 3, 7, 64])
def test_row_blocked_predicates_match_dense_expressions(monkeypatch, rows):
    monkeypatch.setattr(linalg, '_VALIDATION_ROWS', rows)
    torch.manual_seed(3)
    n = 10
    a = torch.randn(n, n, dtype=torch.complex128, device='cuda')
    hermitian = a + a.mH
    g = torch.randn(n, n, dtype=torch.float64, device='cuda')
    symmetric = g + g.mT
    eps = torch.finfo(torch.float64).eps
    for tensor, conjugate in ((hermitian, True), (symmetric, False)):
        other = tensor.mH if conjugate else tensor.mT
        scale = float(tensor.abs().max())
        assert linalg._absolute_maximum(tensor) == scale
        assert linalg._all_finite(tensor) is bool(torch.isfinite(tensor).all()) is True
        assert (linalg._self_adjoint(tensor, conjugate=conjugate, rtol=32 * eps, atol=32 * eps * scale)
                is torch.allclose(tensor, other, rtol=32 * eps, atol=32 * eps * scale) is True)
    # Defects in the final partial block and in an off-diagonal column block.
    broken = hermitian.clone()
    broken[n - 1, 0] += 1e-6
    scale = float(broken.abs().max())
    assert not linalg._self_adjoint(broken, conjugate=True, rtol=32 * eps, atol=32 * eps * scale)
    assert not torch.allclose(broken, broken.mH, rtol=32 * eps, atol=32 * eps * scale)
    for value in (float('nan'), float('inf')):
        bad = hermitian.clone()
        bad[n - 1, n - 1] = value
        assert not linalg._all_finite(bad)
        bad = symmetric.clone()
        bad[0, n - 1] = value
        assert not linalg._all_finite(bad)
    tiny = a.new_zeros(1, 1)
    assert linalg._absolute_maximum(tiny) == 0.0 and linalg._all_finite(tiny)


def test_operator_validation_rejects_defects_through_public_construction():
    torch.manual_seed(11)
    n = 6
    a = torch.randn(n, n, dtype=torch.complex128, device='cuda') + 4 * torch.eye(n, dtype=torch.complex128, device='cuda')
    p = torch.randn(n // 2, n // 2, dtype=torch.float64, device='cuda')
    gram = p @ p.mT + torch.eye(n // 2, dtype=torch.float64, device='cuda')
    bad = a.clone()
    bad[n - 1, n - 1] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        DenseOperator(bad, CompactRWGRiesz(gram, 1.5))
    skew = gram.clone()
    skew[0, 1] += 1e-3
    with pytest.raises(ValueError, match='symmetric'):
        DenseOperator(a, CompactRWGRiesz(skew, 1.5))
    dense = torch.block_diag(gram, gram).to(a.dtype)
    dense[1, 2] += 1e-3
    with pytest.raises(ValueError, match='Hermitian'):
        DenseOperator(a, dense)
    op = DenseOperator(a, CompactRWGRiesz(gram, 1.5))
    x = torch.ones(n, dtype=a.dtype, device='cuda')
    apply_A(op, x)
    op.matrix[n - 1, 0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        apply_A(op, x)


def test_validation_transients_stay_bounded_on_a_large_operator():
    n = 4096  # 268 MB complex128; dense finiteness/symmetry checks previously allocated about 0.9x this
    torch.manual_seed(5)
    a = torch.randn(n, n, dtype=torch.complex128, device='cuda')
    a.diagonal().add_(n)
    e = n // 2
    p = torch.randn(e, e, dtype=torch.float64, device='cuda')
    gram = p @ p.mT
    gram.diagonal().add_(e)
    del p
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    op = DenseOperator(a, CompactRWGRiesz(gram, 2.0))
    torch.cuda.synchronize()
    factor_bytes = op._riesz_factor.numel() * op._riesz_factor.element_size()
    matrix_bytes = a.numel() * a.element_size()
    transient = torch.cuda.max_memory_allocated() - before - factor_bytes
    retained = torch.cuda.memory_allocated() - before
    assert retained <= factor_bytes + 2**20
    assert transient < 8 * linalg._VALIDATION_ROWS * n * a.element_size()
    assert transient < 0.3 * matrix_bytes
