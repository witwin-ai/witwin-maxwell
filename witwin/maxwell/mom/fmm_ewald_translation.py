"""Root-centered Ewald M2L with explicitly excluded exact-near images."""
import math
import torch
from .periodic import EwaldHelmholtzKernel
from .physics import Medium
from .fmm_expansion import multi_indices, helmholtz_taylor
from .fmm_ewald_real import ewald_regular_taylor
from .fmm_storage import m2l_storage_bytes
from .fmm_compact import compact_tables, compact_moment_action
from .direct_system import DirectSystemActions


class EwaldRootM2L:
    """Experimental periodic root correction for coincident global root centers.

    Represents G_periodic minus all declared near free-space images, including
    the primary image. The caller owns each near image exactly once. Analytic
    support separation is checked, but truncation/cutoff/roundoff qualification
    is still required before this can become a public periodic FMM backend.
    """
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, kernel, medium, test_radius, source_radius, *, order=6, row_batch=16,
                 separation_ratio=.25, retained_budget_bytes=512*1024**2,
                 coefficient_budget_bytes=512*1024**2, near_index_budget_bytes=1024**2):
        if not isinstance(kernel, EwaldHelmholtzKernel) or not isinstance(medium, Medium) or medium.frequency <= 0:
            raise TypeError('Ewald M2L requires an Ewald kernel and finite-frequency scalar medium')
        kernel._guard(complex(medium.k))
        for radius in (test_radius, source_radius):
            if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
                raise ValueError('Ewald support radii must be finite nonnegative scalars')
        if (type(separation_ratio) not in (int, float) or not math.isfinite(separation_ratio)
                or not 0 < separation_ratio < 1):
            raise ValueError('Ewald M2L separation ratio must lie strictly between zero and one')
        for value in (row_batch, retained_budget_bytes, coefficient_budget_bytes, near_index_budget_bytes):
            if type(value) is not int or value < 1:
                raise ValueError('Ewald M2L batches and budgets must be positive integers')
        real = kernel.lattice.vectors.element_size()
        expected = m2l_storage_bytes(order, real, 'compact')
        sigma = float(torch.linalg.svdvals(kernel.lattice.vectors.double())[-1])
        radius_sum = test_radius+source_radius
        if sigma*(kernel.real_shells+1) <= radius_sum/separation_ratio:
            raise ValueError('Ewald real cutoff does not cover every possible near image')
        near, size = [], 0
        for indices in kernel._indices(kernel.real_shells, kernel.lattice.vectors.device):
            distance = torch.linalg.vector_norm(kernel.lattice.translations(indices), dim=-1)
            selected = indices[(distance == 0) | (radius_sum > separation_ratio*distance)]
            if len(selected):
                size += selected.numel()*selected.element_size()
                if size > near_index_budget_bytes or expected+size > retained_budget_bytes:
                    raise MemoryError('Ewald M2L retained near/coefficient budget exceeded before expansions')
                near.append(selected)
        self.near_indices = torch.cat(near)
        self.coefficients = ewald_regular_taylor(kernel, medium.k, 2*order+1,
            coefficient_budget_bytes=coefficient_budget_bytes)
        coefficient_bytes = self.coefficients.numel()*self.coefficients.element_size()
        batch = min(kernel.image_batch, coefficient_budget_bytes//coefficient_bytes-1)
        for selected in self.near_indices[(self.near_indices != 0).any(-1)].split(batch):
            if not len(selected):
                continue
            coefficients = helmholtz_taylor(-kernel.lattice.translations(selected), medium.k, 2*order+1)
            self.coefficients -= torch.einsum('b,bc->c', kernel.lattice.phase(selected), coefficients)
            del coefficients
        self.indices = torch.tensor(multi_indices(order), device=kernel.lattice.vectors.device, dtype=torch.int64)
        self.binomial = kernel.lattice.vectors.new_tensor([[math.comb(n, k) if k <= n else 0 for k in range(order+1)]
                                                         for n in range(2*order+2)])
        if not bool(torch.isfinite(self.coefficients).all() & torch.isfinite(self.binomial).all()):
            raise ValueError('Ewald M2L coefficients exceed native precision')
        self.kernel, self.medium = kernel, medium
        self.test_radius, self.source_radius = test_radius, source_radius
        self.separation_ratio = separation_ratio
        self.order, self.count, self.row_batch = order, len(self.indices), row_batch
        self.device, self.dtype = self.coefficients.device, self.coefficients.dtype
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != expected+size:
            raise RuntimeError('Ewald M2L retained storage differs from its plan')
        self._binding = self._state()

    def tables(self):
        return (self.coefficients, self.indices, self.binomial)

    def action(self, medium, value, *, adjoint=False):
        if medium is not self.medium:
            raise ValueError("Ewald M2L action requires its bound medium")
        return self._action(value, adjoint=adjoint)

    def tensors(self):
        return (self.coefficients, self.indices, self.binomial, self.near_indices)

    def _state(self):
        kernel_parameters = tuple(getattr(self.kernel, name) for name in
            ('eta', 'real_shells', 'reciprocal_shells', 'quadrature_order', 'image_batch', 'query_batch',
             'pole_relative_tolerance', 'kernel_id', 'version'))
        return (id(self.kernel), id(self.kernel.lattice), kernel_parameters, id(self.medium),
            self.medium.frequency, self.medium.epsilon_r, self.medium.mu_r, self.medium.k, self.medium.eta,
            self.test_radius, self.source_radius, self.separation_ratio, self.order, self.count, self.row_batch,
            self.device, self.dtype, self.retained_bytes, tuple((id(t), t._version) for t in self.tensors()))

    def assert_current(self):
        self.kernel.lattice.assert_current()
        if self._state() != self._binding:
            raise ValueError('Ewald root M2L changed')

    def _tables(self, start, stop):
        return compact_tables(self, start, stop)

    def _action(self, value, *, adjoint):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 3 or value.shape[-3:] != (2, self.count, 4)
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Ewald M2L requires finite matching complex moment channels')
        return compact_moment_action(self, self.medium, value, adjoint=adjoint)

    def receipt(self):
        self.assert_current()
        return dict(version='ewald_root_compact_m2l_v1', kernel=self.kernel.receipt(), order=self.order,
            exact_near_images=len(self.near_indices), retained_bytes=self.retained_bytes,
            table_bytes=sum(t.numel()*t.element_size() for t in self.tables()),
            root_displacement='zero', scope='configured_ewald_minus_declared_free_space_near_images',
            support_separation_checked=True, expansion_error_certified=False, ewald_cutoff_certified=False,
            floating_point_certified=False, physical_qualification=False, peak_memory_qualified=False,
            current_backward='fixed_operator_adjoint_no_saved_generated_tables',
            excludes=['coefficient recurrence and kernel workspace', 'generated action tables and workspace',
                      'shared kernel/lattice', 'allocator and host metadata'])
