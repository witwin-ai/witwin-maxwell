"""Matrix-free Krylov contract checked against independent diagonal algebra."""
from contextlib import nullcontext
import pytest
import torch
from witwin.maxwell.mom.operator_protocol import IterativeState, SurfaceOperatorProtocol
from witwin.maxwell.mom.linalg import DenseOperator, StoppingRule, solve_reference, apply_A, apply_A_adjoint, residual_loss, condition_diagnostics

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


class DiagonalActions:
    supports_direct = False
    def __init__(self):
        self.diagonal = torch.arange(1, 7, device='cuda', dtype=torch.float64).to(torch.complex128)*(1+.2j)
        self.shape = (6, 6)
        self.device, self.dtype = self.diagonal.device, self.diagonal.dtype
        self.iterative_state = IterativeState()
    @property
    def matrix(self): raise AssertionError('Iterative protocol accessed dense storage')
    def refresh(self): pass
    def read_access(self): return nullcontext()
    def apply(self, x): return x*self.diagonal
    def adjoint(self, y): return y*self.diagonal.conj()
    def riesz_inverse(self, r): return r
    def preconditioner_scales(self, kind):
        one = torch.ones(6, device=self.device, dtype=torch.float64)
        if kind == 'none': return one, one, False
        if kind == 'row_norm': return 1/self.diagonal.abs(), one, False
        if kind == 'row_column_norm': return one, 1/self.diagonal.abs(), False
        raise ValueError(kind)
    def persistence_receipt(self): return {'version': 'test_diagonal_v1'}
    def operator_stats(self): return {'scope': 'test_oracle'}


@pytest.mark.parametrize('method', ['gmres', 'block_gmres'])
@pytest.mark.parametrize('preconditioner', ['none', 'row_norm', 'row_column_norm'])
def test_iterative_solver_never_requires_dense_storage(method, preconditioner):
    op = DiagonalActions()
    assert isinstance(op, SurfaceOperatorProtocol)
    b = torch.arange(18, device=op.device, dtype=torch.float64).reshape(3, 6).to(op.dtype)+.3j
    rule = StoppingRule(method=method, preconditioner=preconditioner, rtol=1e-11, atol=0, maxiter=30, restart=12)
    value, log = solve_reference(op, b, rule)
    assert log['converged'], log
    torch.testing.assert_close(value, b/op.diagonal, rtol=1e-10, atol=1e-11)


def test_matrix_free_adjoint_riesz_gradient_and_diagnostics():
    op = DiagonalActions()
    x = torch.arange(6, device=op.device, dtype=torch.float64).to(op.dtype)+.7j
    y = (x+.8j).flip(0)
    left = torch.vdot(y, apply_A(op, x))
    right = torch.vdot(apply_A_adjoint(op, y), x)
    assert float(abs(left-right)/abs(left)) < 1e-11
    x.requires_grad_()
    gradient = torch.autograd.grad(residual_loss(op, x, y), x)[0]
    torch.testing.assert_close(gradient, op.adjoint(op.apply(x)-y), rtol=1e-12, atol=1e-12)
    estimate = condition_diagnostics(op)
    assert not estimate['exact_svd'] and not estimate['certified_stability_bound']
    assert abs(estimate['condition_number']-6) < 1e-10


def test_direct_capability_is_rejected_before_action():
    op = DiagonalActions()
    for rule in (StoppingRule(method='direct'), StoppingRule(fallback='direct')):
        with pytest.raises(ValueError, match='Direct/LU'):
            solve_reference(op, torch.ones(6, device=op.device, dtype=op.dtype), rule)


def test_dense_protocol_receipt_and_counters():
    op = DiagonalActions()
    dense = DenseOperator(torch.diag(op.diagonal), torch.eye(6, device=op.device, dtype=op.dtype))
    assert isinstance(dense, SurfaceOperatorProtocol)
    first = dense.persistence_receipt()
    assert first == dense.persistence_receipt()
    dense.apply(op.diagonal); dense.adjoint(op.diagonal)
    assert dense.operator_stats()['apply_calls'] == 1
    assert dense.operator_stats()['adjoint_calls'] == 1
    dense.matrix[0, 0] += 1
    assert dense.persistence_receipt()['matrix']['sha256'] != first['matrix']['sha256']
