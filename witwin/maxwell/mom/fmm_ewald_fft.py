"""Internal FFT action of actual Ewald root coefficients and exact-near exclusions."""
import math
import torch
from .fmm_ewald_translation import EwaldRootM2L
from .fmm_fft_coefficients import CoefficientFFTElectromagneticM2L
from .fmm_storage import m2l_storage_bytes
from .direct_system import DirectSystemActions


class EwaldFFTTerminalM2L:
    """Configured Ewald minus primary/declared near images; not public admission.

    The original compact constructor is temporary setup workspace. Its copied
    coefficients and near map retain their complete fixed kernel/medium bindings.
    """
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, kernel, medium, test_radius, source_radius, *, order=6, row_batch=16,
                 separation_ratio=.25, retained_budget_bytes=512*1024**2,
                 coefficient_budget_bytes=512*1024**2, near_index_budget_bytes=1024**2,
                 tensor_workspace_budget_bytes=128*1024**2):
        from .periodic import EwaldHelmholtzKernel
        if not isinstance(kernel, EwaldHelmholtzKernel):
            raise TypeError('Ewald FFT requires a configured Ewald kernel')
        if type(retained_budget_bytes) is not int or retained_budget_bytes < 1:
            raise ValueError('Ewald FFT retained budget must be a positive integer')
        real = kernel.lattice.vectors.element_size()
        fft_bytes = m2l_storage_bytes(order, real, 'fft_v1')
        compact_bytes = m2l_storage_bytes(order, real, 'compact')
        if type(tensor_workspace_budget_bytes) is not int or tensor_workspace_budget_bytes < 1:
            raise ValueError('Ewald FFT workspace budget must be a positive integer')
        side = 1 << (2*order).bit_length()
        planned_workspace = side**3*(16*(32+20*4)+8*48)+4*math.comb(order+3,3)*16*128
        if planned_workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Ewald FFT workspace budget exceeded before expansions')
        allowance = retained_budget_bytes-(fft_bytes-compact_bytes)
        if allowance < 1:
            raise MemoryError('Ewald FFT capacity exceeded before expansions')
        # Tighten the old gate by the exact additional FFT layout bytes. Thus
        # FFT inventory plus all near indices fits BEFORE expensive coefficients.
        original = EwaldRootM2L(kernel, medium, test_radius, source_radius, order=order,
            row_batch=row_batch, separation_ratio=separation_ratio, retained_budget_bytes=allowance,
            coefficient_budget_bytes=coefficient_budget_bytes, near_index_budget_bytes=near_index_budget_bytes)
        scale = torch.linalg.svdvals(kernel.lattice.vectors.double())[-1]/(2*order+2)
        self.translation = CoefficientFFTElectromagneticM2L(original.coefficients,
            kernel.lattice.vectors.new_zeros(3), scale, medium, order,
            retained_budget_bytes=fft_bytes, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
        self.near_indices = original.near_indices.clone()
        self.kernel, self.medium = kernel, medium
        self.test_radius, self.source_radius = test_radius, source_radius
        self.order, self.separation_ratio, self.row_batch = order, separation_ratio, row_batch
        self.device, self.dtype, self.count = self.translation.device, self.translation.dtype, self.translation.count
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        self.planned_retained_bytes = fft_bytes+self.near_indices.numel()*self.near_indices.element_size()
        if self.retained_bytes != self.planned_retained_bytes or self.retained_bytes > retained_budget_bytes:
            raise RuntimeError('Ewald FFT owned inventory differs from preallocated capacity')
        self._binding = self._state()
        # No compact object or coefficient/index/binomial view survives setup.
        del original

    def tensors(self):
        return (*self.translation.tensors(), self.near_indices)

    def _state(self):
        parameters = tuple(getattr(self.kernel, name) for name in
            ('eta', 'real_shells', 'reciprocal_shells', 'quadrature_order', 'image_batch', 'query_batch',
             'pole_relative_tolerance', 'kernel_id', 'version'))
        m = self.medium
        return (id(self.kernel), id(self.kernel.lattice), parameters, id(m),
            m.frequency, m.epsilon_r, m.mu_r, m.k, m.eta,
            self.test_radius, self.source_radius, self.order, self.separation_ratio, self.row_batch,
            id(self.translation), self.device, self.dtype, self.count, self.retained_bytes,
            self.planned_retained_bytes, tuple((id(t), t._version) for t in self.tensors()))

    def assert_current(self):
        self.kernel.lattice.assert_current()
        if self._state() != self._binding:
            raise ValueError('Ewald FFT root M2L changed')
        self.translation.assert_current()

    def action(self, medium, value, *, adjoint=False):
        if medium is not self.medium:
            raise ValueError('Ewald FFT action requires its bound medium')
        return self.adjoint(value) if adjoint else self.apply(value)

    def _action(self, value, *, adjoint):
        self.assert_current()
        return self.translation.adjoint(value) if adjoint else self.translation.apply(value)

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_ewald_root_fft_m2l_v1', kernel=self.kernel.receipt(),
            order=self.order, exact_near_images=len(self.near_indices), retained_bytes=self.retained_bytes,
            table_bytes=self.translation.scalar.retained_bytes,
            root_displacement='zero', normalization='sigma_min_lattice_over_2p_plus2',
            scope='configured_ewald_minus_declared_free_space_near_images',
            scalar_translation=self.translation.scalar.receipt(),
            support_separation_checked=True, expansion_error_certified=False, ewald_cutoff_certified=False,
            floating_point_certified=False, physical_qualification=False, complexity_qualification=False,
            peak_memory_qualified=False, material_gradient_qualified=False,
            current_backward='fixed_operator_adjoint_no_saved_generated_tables',
            setup_workspace_excludes=('temporary compact coefficients and layout', 'coefficient recurrence/kernel',
                'FFT library workspace', 'allocator cache', 'shared kernel/lattice', 'caller input/output'))
