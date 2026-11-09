"""Fixed Cartesian Green/gradient M2L using native CUDA circular convolution.

Complex128 primitives support the explicit free-space FFT FMM configuration.
FFT roundoff, full resources, physics and high-frequency qualification are open.
"""
import math
import torch
from .fmm_expansion import _points, multi_indices, helmholtz_taylor
from .fmm_compact import _coefficient_index
from .direct_system import DirectSystemActions
from .fmm_storage import m2l_storage_bytes


def _rhs_chunks(value, tail_axes, batch_size):
    """Copy/validate at most one chunk, including noncontiguous leading axes."""
    leading = value.shape[:-tail_axes]
    rows = math.prod(leading)
    flat = value.view(rows, *value.shape[-tail_axes:]) if value.is_contiguous() else None
    for start in range(0, rows, batch_size):
        stop = min(start+batch_size, rows)
        if flat is not None:
            chunk = flat[start:stop]
        else:
            indices = []
            for row in range(start, stop):
                coordinates = []
                for size in reversed(leading):
                    coordinates.append(row % size)
                    row //= size
                indices.append(value[tuple(reversed(coordinates))])
            chunk = torch.stack(indices)
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError('FFT M2L requires finite current moments')
        yield start, chunk


class CartesianFFTTranslation:
    """Four scalar translation fields, without a dense expansion table.

    Cube side >2p prevents aliasing at all retained target multiindices. Fixed
    factorial/length scaling changes representation only, not truncation.
    """
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, displacement, wave_number, order, *, rhs_batch=4,
                 tensor_workspace_budget_bytes=128*1024**2, retained_budget_bytes=1024**2):
        self._prepare(displacement, order, rhs_batch, tensor_workspace_budget_bytes, retained_budget_bytes)
        self._bind_coefficients(helmholtz_taylor(displacement, wave_number, 2*order+1), wave_number)

    @classmethod
    def create_many(cls, displacements, wave_number, order, *, rhs_batch=4,
                    tensor_workspace_budget_bytes=128*1024**2, retained_budget_bytes=1024**2):
        """Generate Taylor vectors in batches of32 with independent owned storage."""
        _points(displacements)
        if displacements.ndim != 2:
            raise ValueError('FFT M2L batch requires fixed center displacements')
        expected = m2l_storage_bytes(order, displacements.element_size(), 'fft_v1')
        if any(type(x) is not int or x < 1 for x in
               (rhs_batch, tensor_workspace_budget_bytes, retained_budget_bytes)):
            raise ValueError('FFT M2L batches and tensor budgets must be positive integers')
        if len(displacements)*expected > retained_budget_bytes:
            raise MemoryError('FFT M2L batch retained budget exceeded before coefficients')
        if not len(displacements):
            return ()
        indices = torch.tensor(multi_indices(order), device=displacements.device, dtype=torch.long)
        result = []
        for start in range(0, len(displacements), 32):
            chunk = displacements[start:start+32]
            objects = []
            for displacement in chunk:
                obj = cls.__new__(cls)
                obj._prepare(displacement, order, rhs_batch, tensor_workspace_budget_bytes,
                             retained_budget_bytes, indices=indices)
                objects.append(obj)
            coefficients = helmholtz_taylor(chunk, wave_number, 2*order+1)
            for obj, vector in zip(objects, coefficients, strict=True):
                obj._bind_coefficients(vector.clone(), wave_number)
                result.append(obj)
            del vector, coefficients
        return tuple(result)

    def _prepare(self, displacement, order, rhs_batch, tensor_workspace_budget_bytes,
                 retained_budget_bytes, *, indices=None, normalization_length=None):
        _points(displacement)
        if displacement.shape != (3,) or displacement.dtype != torch.float64:
            raise ValueError('FFT M2L requires one fixed CUDA float64 center displacement')
        if type(order) is not int or not 0 <= order <= 12:
            raise ValueError('FFT M2L native order must lie between zero and twelve')
        if any(type(x) is not int or x < 1 for x in
               (rhs_batch, tensor_workspace_budget_bytes, retained_budget_bytes)):
            raise ValueError('FFT M2L batches and tensor budgets must be positive integers')
        self.order, self.count = order, len(multi_indices(order))
        self.side = 1 << (2*order).bit_length()
        self.device, self.dtype = displacement.device, torch.complex128
        self.rhs_batch = rhs_batch
        self.tensor_workspace_budget_bytes = tensor_workspace_budget_bytes
        # Conservative live Python-visible array allowance: spatial/frequency
        # fields, products/inverse transforms, geometry/index and scale temporaries.
        # Backend FFT workspace/allocator and caller tensors are explicitly excluded.
        self.planned_tensor_workspace_bytes = self.side**3*(16*(32+20*rhs_batch)+8*48)
        self.planned_retained_bytes = (len(multi_indices(2*order+1))*16+3*8+8
            +self.count*(3*8+2*8+2*8))
        if self.planned_retained_bytes > retained_budget_bytes:
            raise MemoryError('FFT M2L retained tensor budget exceeded before coefficients')
        if self.planned_tensor_workspace_bytes > tensor_workspace_budget_bytes:
            raise MemoryError('FFT M2L temporary tensor budget exceeded before coefficients')
        self.displacement = displacement.clone()
        if normalization_length is None:
            self.scale = torch.linalg.vector_norm(displacement)/(2*order+2)
        else:
            if (not isinstance(normalization_length, torch.Tensor) or normalization_length.shape != ()
                    or normalization_length.device != self.device or normalization_length.dtype != torch.float64
                    or normalization_length.requires_grad):
                raise ValueError('FFT M2L normalization must be one fixed matching CUDA float64 length')
            self.scale = normalization_length.clone()
        if not bool(torch.isfinite(self.scale) & (self.scale > 0)):
            raise ValueError('FFT M2L requires a representable positive normalization length')
        self.indices = (torch.tensor(multi_indices(order), device=self.device, dtype=torch.long)
                        if indices is None else indices.clone())
        degrees = self.indices.sum(-1)
        log_weights = -torch.lgamma(self.indices.double()+1).sum(-1)-degrees*self.scale.log()
        self.target_weights = log_weights.exp()
        self.source_weights = self.target_weights*(1-2*(degrees%2))
        if not bool(torch.isfinite(self.target_weights).all() & (self.target_weights > 0).all()):
            raise ValueError('FFT M2L normalization exceeds native precision')
        self.target_indices = self._flat(self.indices)
        self.source_indices = self._flat((-self.indices).remainder(self.side))

    def _bind_coefficients(self, coefficients, wave_number, *, require_nonzero_amplitude=True):
        self.coefficients = coefficients
        if require_nonzero_amplitude and not bool(self.coefficients[0].abs() > 0):
            raise ValueError('FFT M2L kernel amplitude underflows native precision')
        self.wave_number = None if wave_number is None else complex(wave_number)
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != self.planned_retained_bytes:
            raise RuntimeError('FFT M2L retained tensors differ from structural plan')
        self._binding = self._state()
        # Validate normalization of every used field before admitting this object.
        self._fourier_fields()

    def _flat(self, indices):
        return (indices[..., 0]*self.side+indices[..., 1])*self.side+indices[..., 2]

    def tensors(self):
        return (self.displacement, self.scale, self.indices, self.target_weights,
                self.source_weights, self.target_indices, self.source_indices, self.coefficients)

    def _state(self):
        return (self.order, self.count, self.side, self.device, self.dtype, self.rhs_batch,
                self.tensor_workspace_budget_bytes, self.planned_tensor_workspace_bytes,
                self.retained_bytes, self.wave_number,
                tuple((id(t), t._version) for t in self.tensors()))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('FFT M2L fixed translation changed')

    def _fourier_fields(self):
        grid = torch.arange(self.side, device=self.device)
        gamma = torch.stack(torch.meshgrid(grid, grid, grid, indexing='ij'), dim=-1)
        valid = gamma.sum(-1) <= 2*self.order
        selected = gamma[valid]
        degrees = selected.sum(-1)
        spatial = torch.zeros((4, self.side, self.side, self.side), device=self.device, dtype=self.dtype)
        for field in range(4):
            derivative = selected.clone()
            if field:
                derivative[:, field-1] += 1
            factors = (torch.lgamma(derivative.double()+1).sum(-1)+degrees*self.scale.log()).exp()
            original = self.coefficients[_coefficient_index(derivative)]
            scaled = original*factors
            lost = ((original.real != 0) & (scaled.real == 0)) | ((original.imag != 0) & (scaled.imag == 0))
            if not bool(torch.isfinite(factors).all() & (factors > 0).all()
                        & torch.isfinite(scaled).all() & ~lost.any()):
                raise ValueError('FFT M2L normalized field exceeds native precision')
            spatial[field, valid] = scaled
        fourier = torch.fft.fftn(spatial, dim=(-3, -2, -1))
        if not bool(torch.isfinite(fourier).all()):
            raise ValueError('FFT M2L frequency grid exceeds native precision')
        return fourier

    def _action(self, value, *, adjoint):
        self.assert_current()
        tail = (4, self.count) if adjoint else (self.count,)
        if (not isinstance(value, torch.Tensor) or value.ndim < len(tail)
                or value.shape[-len(tail):] != tail or value.device != self.device
                or value.dtype != self.dtype):
            raise ValueError('FFT M2L requires matching finite complex128 scalar moments')
        leading = value.shape[:-len(tail)]
        rows = math.prod(leading)
        if not rows:
            return value.new_empty((*leading, self.count) if adjoint else (*leading, 4, self.count))
        result = value.new_empty((rows, self.count) if adjoint else (rows, 4, self.count))
        fields = self._fourier_fields()
        for start, batch in _rhs_chunks(value, len(tail), self.rhs_batch):
            if adjoint:
                embedded = value.new_zeros((len(batch), 4, self.side**3))
                embedded[..., self.target_indices] = batch*self.target_weights
                spectrum = torch.fft.fftn(embedded.reshape(len(batch), 4, self.side, self.side, self.side), dim=(-3,-2,-1))
                translated = torch.fft.ifftn(spectrum*fields.conj(), dim=(-3,-2,-1)).reshape(len(batch),4,-1)
                block = translated[..., self.source_indices].sum(-2)*self.source_weights
            else:
                embedded = value.new_zeros((len(batch), self.side**3))
                embedded[..., self.source_indices] = batch*self.source_weights
                spectrum = torch.fft.fftn(embedded.reshape(len(batch), self.side, self.side, self.side), dim=(-3,-2,-1))
                translated = torch.fft.ifftn(spectrum[:,None]*fields, dim=(-3,-2,-1)).reshape(len(batch),4,-1)
                block = translated[..., self.target_indices]*self.target_weights
            if not bool(torch.isfinite(embedded).all() & torch.isfinite(block).all()):
                raise ValueError('FFT M2L action exceeds native precision')
            result[start:start+len(batch)] = block
        return result.reshape((*leading, self.count) if adjoint else (*leading, 4, self.count))

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_cartesian_fft_scalar_m2l_v1', order=self.order,
            scalar_fields=('green', 'gradient_x', 'gradient_y', 'gradient_z'), cube_side=self.side,
            retained_bytes=self.retained_bytes, planned_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,
            tensor_workspace_excludes=('FFT library workspace', 'allocator cache', 'caller input/output tensors'),
            rhs_batch=self.rhs_batch, floating_point_certified=False, em_composition_qualified=False,
            physical_qualification=False, complexity_qualification=False, peak_memory_qualified=False)


class FFTElectromagneticM2L:
    """Internal PMCHWT composition of scalar FFT Green/gradient translations."""
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, displacement, medium, order, **kwargs):
        from .physics import Medium
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Electromagnetic FFT M2L requires a finite-frequency scalar medium')
        self._bind_scalar(CartesianFFTTranslation(displacement, medium.k, order, **kwargs), medium)

    @classmethod
    def create_many(cls, displacements, medium, order, **kwargs):
        from .physics import Medium
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Electromagnetic FFT M2L requires a finite-frequency scalar medium')
        scalars = CartesianFFTTranslation.create_many(displacements, medium.k, order, **kwargs)
        result = []
        for scalar in scalars:
            obj = cls.__new__(cls)
            obj._bind_scalar(scalar, medium)
            result.append(obj)
        return tuple(result)

    def _bind_scalar(self, scalar, medium):
        self.medium, self.scalar = medium, scalar
        self.device, self.dtype, self.count = scalar.device, scalar.dtype, scalar.count
        # Version1 bounds physical RHS BEFORE expanding eight channels/four fields.
        self.rhs_batch = 4
        self.planned_tensor_workspace_bytes = (scalar.planned_tensor_workspace_bytes
                                               +self.rhs_batch*self.count*16*128)
        if self.planned_tensor_workspace_bytes > scalar.tensor_workspace_budget_bytes:
            raise MemoryError('Electromagnetic FFT M2L chunk workspace exceeds tensor budget')
        self._binding = self._state()

    def _state(self):
        m = self.medium
        return (id(m), m.frequency, m.epsilon_r, m.mu_r, m.k, m.eta,
                id(self.scalar), self.device, self.dtype, self.count, self.rhs_batch,
                self.planned_tensor_workspace_bytes)

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Electromagnetic FFT M2L changed')
        self.scalar.assert_current()

    def tensors(self):
        return self.scalar.tensors()

    def _action(self, value, *, adjoint):
        self.assert_current()
        return self._action_after_validation(value, adjoint=adjoint)

    def _action_after_validation(self, value, *, adjoint):
        """Translate inside a caller that validates the complete fixed graph."""
        if (not isinstance(value, torch.Tensor) or value.ndim < 3
                or value.shape[-3:] != (2, self.count, 4) or value.device != self.device
                or value.dtype != self.dtype):
            raise ValueError('Electromagnetic FFT M2L requires matching finite complex128 moment channels')
        leading = value.shape[:-3]
        rows = math.prod(leading)
        result = value.new_empty((rows, 2, self.count, 4))
        for start, block in _rhs_chunks(value, 3, self.rhs_batch):
            result[start:start+len(block)] = self._action_chunk(block, adjoint=adjoint)
        return result.reshape(value.shape)

    def _action_chunk(self, value, *, adjoint):
        medium = self.medium
        electric, magnetic = 1j*medium.omega*medium.epsilon, -1j*medium.omega*medium.mu
        if adjoint:
            # A=[C,eB;mB,C]; C^H is the negative curl of adjoint
            # derivative translations, while B^H conjugates its charge factor.
            fields = value.new_zeros((*value.shape[:-3], 2, 4, 4, self.count))
            dual = value.movedim(-2, -1)
            fields[..., 0, :, 0, :] = magnetic.conjugate()*dual[..., 1, :, :]
            fields[..., 1, :, 0, :] = electric.conjugate()*dual[..., 0, :, :]
            fields[..., :, 3, 0, :] *= -1/(complex(medium.k).conjugate()**2)
            fields[..., :, 0, 3, :] = dual[..., :, 1, :]
            fields[..., :, 0, 2, :] = -dual[..., :, 2, :]
            fields[..., :, 1, 1, :] = dual[..., :, 2, :]
            fields[..., :, 1, 3, :] = -dual[..., :, 0, :]
            fields[..., :, 2, 2, :] = dual[..., :, 0, :]
            fields[..., :, 2, 1, :] = -dual[..., :, 1, :]
            result = self.scalar.adjoint(fields).movedim(-1, -2)
        else:
            # Scalar leading axes are J/M and vector/divergence channels.
            fields = self.scalar.apply(value.movedim(-2, -1)).movedim(-1, -3)
            potential = fields[..., 0]
            curl = torch.stack((fields[..., 2, 2]-fields[..., 1, 3],
                                fields[..., 0, 3]-fields[..., 2, 1],
                                fields[..., 1, 1]-fields[..., 0, 2]), dim=-1)
            weak_electric = torch.cat((potential[..., :3],(-potential[..., 3]/(medium.k**2))[...,None]),dim=-1)
            weak_magnetic = torch.cat((curl,torch.zeros_like(curl[..., :1])),dim=-1)
            result = torch.stack((weak_magnetic[..., 0, :, :]+electric*weak_electric[..., 1, :, :],
                                  magnetic*weak_electric[..., 0, :, :]+weak_magnetic[..., 1, :, :]),dim=-3)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Electromagnetic FFT M2L action exceeds native precision')
        return result

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_fft_pmchwt_m2l_v1',scalar_translation=self.scalar.receipt(),
            ordering='J/M, total-degree moment, vector/divergence',retained_bytes=self.scalar.retained_bytes,
            physical_rhs_batch=self.rhs_batch, planned_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,
            tensor_workspace_excludes=self.scalar.receipt()['tensor_workspace_excludes'],
            physical_qualification=False,complexity_qualification=False,peak_memory_qualified=False)
