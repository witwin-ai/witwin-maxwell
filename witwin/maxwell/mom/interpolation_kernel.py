"""Free-space scalar Green tables coupling four-channel node moments."""
import math
import torch
from .scalar_moment_action import scalar_moment_action
from .interpolation import InterpolationBox
from .physics import Medium
from .kernels import FREE_SPACE
from .interpolation_error import scalar_kernel_interpolation_bound, require_kernel_budget


class InterpolationKernel:
    def __init__(self, test, source, medium, *, admissibility=1., kernel_row_batch=32,
                 kernel_budget_bytes=512*1024**2, error_budget=None):
        if not isinstance(test, InterpolationBox) or not isinstance(source, InterpolationBox):
            raise TypeError('Kernel tables require interpolation boxes')
        test.assert_current()
        source.assert_current()
        if test.device != source.device or test.dtype != source.dtype:
            raise ValueError('Kernel boxes must share device and precision')
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Kernel interpolation requires a finite-frequency scalar medium')
        if type(admissibility) not in (int, float) or not math.isfinite(admissibility) or admissibility <= 0:
            raise ValueError('Admissibility must be positive and finite')
        if any(type(v) is not int or v < 1 for v in (kernel_row_batch, kernel_budget_bytes)):
            raise ValueError('Kernel row batch and retained table budget must be positive integers')
        gap = torch.maximum(test.lower-source.upper, source.lower-test.upper).clamp_min(0)
        distance = torch.linalg.vector_norm(gap)
        diameter = torch.maximum(torch.linalg.vector_norm(test.upper-test.lower), torch.linalg.vector_norm(source.upper-source.lower))
        if not bool((distance > 0) & (diameter <= admissibility*distance)):
            raise ValueError('Interpolation is restricted to admissible full-support far boxes')
        self.test, self.source, self.medium = test, source, medium
        self.device = test.device
        self.dtype = torch.complex128 if test.dtype == torch.float64 else torch.complex64
        self.error_budget = error_budget
        if error_budget is not None:
            require_kernel_budget(scalar_kernel_interpolation_bound(test, source, medium.k), error_budget)
        nt, ns = len(test.nodes), len(source.nodes)
        self.kernel_bytes = 4*nt*ns*(16 if self.dtype == torch.complex128 else 8)
        if self.kernel_bytes > kernel_budget_bytes:
            raise MemoryError('Far interpolation kernel table budget exceeded before allocation')
        self.green = torch.empty((nt, ns), device=self.device, dtype=self.dtype)
        self.gradient = torch.empty((nt, ns, 3), device=self.device, dtype=self.dtype)
        for start in range(0, nt, kernel_row_batch):
            values, derivatives = FREE_SPACE.evaluate(test.nodes[start:start+kernel_row_batch, None]-source.nodes[None, :], medium.k)
            self.green[start:start+kernel_row_batch] = values
            self.gradient[start:start+kernel_row_batch] = derivatives
        if not bool(torch.isfinite(self.green).all() & torch.isfinite(self.gradient).all()):
            raise ValueError('Far interpolation kernel table exceeds native precision')
        self._binding = (id(test), id(source), id(medium), self.device, self.dtype, self.error_budget)
        self._medium = (medium.frequency, medium.epsilon_r, medium.mu_r, medium.k, medium.eta)
        self._versions = tuple((id(t), t._version) for t in (self.green, self.gradient))

    def assert_current(self):
        if ((id(self.test), id(self.source), id(self.medium), self.device, self.dtype, self.error_budget) != self._binding
                or (self.medium.frequency, self.medium.epsilon_r, self.medium.mu_r, self.medium.k, self.medium.eta) != self._medium
                or tuple((id(t), t._version) for t in (self.green, self.gradient)) != self._versions):
            raise ValueError('Interpolation kernel changed')
        self.test.assert_current()
        self.source.assert_current()

    def truncation_bound(self):
        self.assert_current()
        return scalar_kernel_interpolation_bound(self.test, self.source, self.medium.k)

    def action(self, value, *, adjoint=False):
        """Map [...,2,P,4] moments to testing coefficients; channels are xyz/div."""
        self.assert_current()
        source_nodes = len(self.test.nodes) if adjoint else len(self.source.nodes)
        if (not isinstance(value, torch.Tensor) or value.ndim < 3
                or value.shape[-3:] != (2, source_nodes, 4) or value.device != self.device
                or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Kernel action requires finite matching complex node moments')
        return scalar_moment_action(self.green, self.gradient, self.medium, value, adjoint=adjoint)
