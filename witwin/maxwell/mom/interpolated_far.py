"""Direct scalar far-region actions through vector and charge moments."""
import torch
from .rwg_moments import RWGInterpolationMoments
from .interpolation_kernel import InterpolationKernel
from .kernels import FREE_SPACE


class InterpolatedRegionalBlock:
    def __init__(self, test, source, medium, *, admissibility=1., kernel_row_batch=32,
                 kernel_budget_bytes=512*1024**2):
        if not isinstance(test, RWGInterpolationMoments) or not isinstance(source, RWGInterpolationMoments):
            raise TypeError('Far interpolation requires RWG moment objects')
        test.assert_current()
        source.assert_current()
        self.test, self.source = test, source
        self.device, self.dtype = test.device, test.dtype
        self.shape = (2*len(test.edges), 2*len(source.edges))
        self.kernel = InterpolationKernel(test.box, source.box, medium, admissibility=admissibility,
            kernel_row_batch=kernel_row_batch, kernel_budget_bytes=kernel_budget_bytes)
        self.kernel_row_batch = kernel_row_batch
        self._binding = (id(test), id(source), id(self.kernel), self.shape, self.device, self.dtype)

    def assert_current(self):
        if (id(self.test), id(self.source), id(self.kernel), self.shape, self.device, self.dtype) != self._binding:
            raise ValueError('Interpolated regional block changed')
        self.test.assert_current()
        self.source.assert_current()
        self.kernel.assert_current()

    def _action(self, value, *, adjoint):
        self.assert_current()
        width = self.shape[0] if adjoint else self.shape[1]
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != width
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Far actions require matching finite complex CUDA coefficients')
        source, test = (self.test, self.source) if adjoint else (self.source, self.test)
        currents = torch.stack((value[..., :width//2], value[..., width//2:]), dim=-2)
        moments = torch.cat((torch.einsum('...e,eac->...ac', currents, source.vector),
                             (currents@source.divergence)[..., None]), dim=-1)
        coefficients = self.kernel.action(moments, adjoint=adjoint)
        result = (torch.einsum('eac,...ac->...e', test.vector.conj(), coefficients[..., :3])
                  +coefficients[..., 3]@test.divergence.mH)
        return torch.cat((result[..., 0, :], result[..., 1, :]), dim=-1)

    def apply(self, value):
        return self._action(value, adjoint=False)

    def adjoint(self, value):
        return self._action(value, adjoint=True)

    def receipt(self):
        self.assert_current()
        return {'version': 'interpolated_scalar_regional_block_v1', 'shape': list(self.shape),
                'kernel': FREE_SPACE.receipt(), 'frequency': self.kernel.medium.frequency,
                'kernel_table_bytes': self.kernel.kernel_bytes, 'maximum_kernel_row_batch': self.kernel_row_batch,
                'test_moments': self.test.receipt(), 'source_moments': self.source.receipt(),
                'interpolation_error_certified': False, 'physical_qualification': False}
