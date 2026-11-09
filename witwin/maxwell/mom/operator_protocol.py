"""Backend contract for CUDA surface operators and iterative solver state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, ContextManager
import torch


@dataclass
class IterativeState:
    """Bounded recycle vectors; a backend must clear these when its action changes."""
    recycle_u: torch.Tensor | None = None
    recycle_c: torch.Tensor | None = None


@runtime_checkable
class SurfaceOperatorProtocol(Protocol):
    shape: tuple[int, int]
    device: torch.device
    dtype: torch.dtype
    supports_direct: bool
    iterative_state: IterativeState

    def refresh(self) -> None: ...
    def read_access(self) -> ContextManager: ...
    def apply(self, x: torch.Tensor) -> torch.Tensor: ...
    def adjoint(self, y: torch.Tensor) -> torch.Tensor: ...
    def riesz_inverse(self, residual: torch.Tensor) -> torch.Tensor: ...
    def preconditioner_scales(self, kind: str) -> tuple[torch.Tensor, torch.Tensor, bool]: ...
    def persistence_receipt(self) -> dict: ...
    def operator_stats(self) -> dict: ...


class KrylovView:
    """Solver-owned cache interface around an operator with no matrix access."""
    def __init__(self, operator: SurfaceOperatorProtocol):
        if not isinstance(operator, SurfaceOperatorProtocol):
            raise TypeError('Operator must implement the complete surface action protocol')
        if (len(operator.shape) != 2 or operator.shape[0] != operator.shape[1] or operator.shape[0] < 1
                or operator.device.type != 'cuda' or operator.dtype not in (torch.complex64, torch.complex128)):
            raise ValueError('Surface action operators require square complex CUDA spaces')
        self.operator = operator

    @property
    def shape(self): return self.operator.shape
    @property
    def device(self): return self.operator.device
    @property
    def dtype(self): return self.operator.dtype
    @property
    def supports_direct(self): return False
    @property
    def _recycle_u(self): return self.operator.iterative_state.recycle_u
    @_recycle_u.setter
    def _recycle_u(self, value): self.operator.iterative_state.recycle_u = value
    @property
    def _recycle_c(self): return self.operator.iterative_state.recycle_c
    @_recycle_c.setter
    def _recycle_c(self, value): self.operator.iterative_state.recycle_c = value
    def refresh(self): self.operator.refresh()
    def read_access(self): return self.operator.read_access()
    def apply(self, value): return self.operator.apply(value)
    def adjoint(self, value): return self.operator.adjoint(value)
    def riesz_inverse(self, value): return self.operator.riesz_inverse(value)
    def preconditioner_scales(self, kind): return self.operator.preconditioner_scales(kind)


@torch.no_grad()
def estimate_condition(operator, *, steps=32):
    """Projected normal-operator Ritz estimate, never an exact SVD certificate.

    A finite Krylov subspace can miss extremal or null modes. In particular its
    minimum Ritz singular value is not a lower bound for the full operator.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError('Condition estimator steps must be a positive integer')
    operator.refresh()
    n = operator.shape[0]
    real_dtype = torch.float64 if operator.dtype == torch.complex128 else torch.float32
    grid = torch.arange(1, n+1, device=operator.device, dtype=real_dtype)
    v = torch.complex(torch.sin(grid), torch.cos(grid)).to(operator.dtype)
    v = v/torch.linalg.vector_norm(v)
    basis, images = [], []
    for _ in range(min(steps, n)):
        basis.append(v)
        av = operator.apply(v)
        images.append(av)
        candidate = operator.adjoint(av)
        for _ in range(2):
            q = torch.stack(basis)
            candidate = candidate-(q.conj()@candidate)@q
        norm = torch.linalg.vector_norm(candidate)
        if bool(norm <= 128*torch.finfo(real_dtype).eps*torch.linalg.vector_norm(av).square()):
            break
        v = candidate/norm
    projected_images = torch.stack(images, dim=1)
    values = torch.linalg.svdvals(projected_images)
    return {'sigma_max': float(values[0]), 'sigma_min': float(values[-1]),
            'condition_number': float(values[0]/values[-1]), 'certified_stability_bound': False,
            'method': 'projected_normal_krylov_ritz', 'exact_svd': False,
            'subspace_dimension': len(basis), 'apply_calls': len(images), 'adjoint_calls': len(images),
            'limitation': 'Finite subspace may miss extremal/null modes; minimum is not a stability lower bound'}
