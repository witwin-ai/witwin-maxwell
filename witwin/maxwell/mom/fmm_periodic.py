"""Bloch-weighted remote-image M2L with explicit exact-near image ownership."""
import math
import json
import torch
from .periodic import LatticeSpec
from .fmm_periodic_error import (absorbing_image_tail_bound, require_image_tail_budget,
    periodic_image_partition, periodic_multipole_error_bound, require_periodic_multipole_budget)
from .fmm_expansion import multipole_to_local, multi_indices, _points, helmholtz_taylor

from .fmm_storage import m2l_storage_bytes
from .fmm_compact import compact_tables, compact_moment_action
from .scalar_moment_action import scalar_moment_action


class PeriodicImageM2L:
    """Finite absorbing lattice expansion, not an infinite-periodic certificate.

    For each image n, use R-A*n and exp(+i beta.A*n). Unseparated support
    spheres are returned as exact-near images and never enter the M2L sum.
    The caller must evaluate those images exactly once. Omitted outer shells
    remain unqualified; this primitive does not enable the public FMM backend.
    """
    def __init__(self, lattice, displacement, test_radius, source_radius, wave_number, *,
                 order=6, shells=2, image_batch=16, separation_ratio=.5,
                 table_budget_bytes=512*1024**2, near_index_budget_bytes=1024**2, image_tail_budget=None, kernel_error_budget=None,
                 table_storage="dense", row_batch=16):
        if table_storage in ('fft_v1','fft_ewald_v1'):
            raise NotImplementedError('Periodic FFT M2L requires separately admitted screened coefficients')
        if not isinstance(lattice, LatticeSpec):
            raise TypeError('Periodic M2L requires LatticeSpec')
        lattice.assert_current()
        _points(displacement)
        if (displacement.shape != (3,) or displacement.device != lattice.vectors.device
                or displacement.dtype != lattice.vectors.dtype):
            raise ValueError('Periodic M2L displacement must match lattice coordinates')
        for radius in (test_radius, source_radius):
            if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
                raise ValueError('Support radii must be finite nonnegative fixed scalars')
        if type(wave_number) not in (int, float, complex):
            raise ValueError('Periodic M2L wave number must be a fixed scalar')
        wave = complex(wave_number)
        if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag <= 0:
            raise ValueError('Finite periodic-image M2L requires positive absorption')
        if type(shells) is not int or shells < 1:
            raise ValueError('Periodic M2L shells must be a positive integer')
        for value in (image_batch, table_budget_bytes, near_index_budget_bytes, row_batch):
            if type(value) is not int or value < 1:
                raise ValueError('Periodic M2L batch and budgets must be positive integers')
        if type(separation_ratio) not in (int, float) or not math.isfinite(separation_ratio) or not 0 < separation_ratio < 1:
            raise ValueError('Periodic M2L separation ratio must lie strictly between zero and one')
        tail = None
        if image_tail_budget is not None:
            tail = absorbing_image_tail_bound(lattice, displacement, test_radius, source_radius, wave, shells)
            require_image_tail_budget(tail, image_tail_budget)
        combined = None
        if kernel_error_budget is not None:
            combined = periodic_multipole_error_bound(lattice, displacement, test_radius, source_radius,
                wave, order=order, shells=shells, separation_ratio=separation_ratio, image_batch=image_batch)
            require_periodic_multipole_budget(combined, kernel_error_budget)
        self.kernel_error_budget = kernel_error_budget
        self._kernel_bound_text = json.dumps(combined, sort_keys=True, allow_nan=False)
        self.image_tail_budget = image_tail_budget
        self._tail_text = json.dumps(tail, sort_keys=True, allow_nan=False)
        count = len(multi_indices(order))
        itemsize = 16 if displacement.dtype == torch.float64 else 8
        retained = m2l_storage_bytes(order, displacement.element_size(), table_storage)
        coefficient_count = math.comb(2*order+4, 3)
        image_bytes = retained if table_storage == 'dense' else coefficient_count*itemsize
        # Count returned batch tables plus accumulated tables; recurrence and
        # reduction temporaries are outside this storage gate.
        if table_budget_bytes < retained+image_bytes:
            raise MemoryError('Periodic M2L budget must hold output and one image table')
        batch = min(image_batch, (table_budget_bytes-retained)//image_bytes)
        dtype = torch.complex128 if itemsize == 16 else torch.complex64
        self.table_storage, self.row_batch, self.count = table_storage, row_batch, count
        self.coefficients = self.indices = self.binomial = None
        green = gradient = None
        if table_storage == 'dense':
            green = torch.zeros((count, count), device=displacement.device, dtype=dtype)
            gradient = torch.zeros((3, count, count), device=displacement.device, dtype=dtype)
        else:
            self.coefficients = torch.zeros(coefficient_count, device=displacement.device, dtype=dtype)
            self.indices = torch.tensor(multi_indices(order), device=displacement.device, dtype=torch.int64)
            self.binomial = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(order+1)]
                for n in range(2*order+2)], device=displacement.device, dtype=displacement.dtype)
            if not bool(torch.isfinite(self.binomial).all()):
                raise ValueError('Periodic compact binomial weights exceed native precision')
        dimension = lattice.dimension
        near, near_count, far_count = [], 0, 0
        for indices, delta, far in periodic_image_partition(lattice, displacement, test_radius+source_radius,
                                                           shells, separation_ratio, batch):
            close = indices[~far]
            if len(close):
                near_count += len(close)
                if near_count*dimension*8 > near_index_budget_bytes:
                    raise MemoryError('Periodic exact-near image index budget exceeded')
                near.append(close)
            selected = indices[far]
            if len(selected):
                phase = lattice.phase(selected)
                if table_storage == 'dense':
                    g, dg = multipole_to_local(delta[far], wave, order,
                        table_budget_bytes=table_budget_bytes-retained)
                    green.add_(torch.einsum('b,bij->ij', phase, g))
                    gradient.add_(torch.einsum('b,bcij->cij', phase, dg))
                    del g, dg
                else:
                    coefficients = helmholtz_taylor(delta[far], wave, 2*order+1)
                    self.coefficients.add_(torch.einsum('b,bc->c', phase, coefficients))
                    del coefficients
                far_count += len(selected)
        lattice.assert_current()
        self.green, self.gradient = green, gradient
        if not all(bool(torch.isfinite(t).all()) for t in self.tables()):
            raise ValueError('Periodic M2L sum exceeds native precision')
        self.lattice, self.displacement = lattice, displacement.clone()
        self.near_indices = torch.cat(near) if near else torch.empty((0, dimension), device=displacement.device, dtype=torch.int64)
        self.order, self.shells, self.wave_number = order, shells, wave
        self.test_radius, self.source_radius = test_radius, source_radius
        self.separation_ratio, self.image_batch = separation_ratio, batch
        self.far_count = far_count
        if combined is not None and (far_count != combined['far_images'] or near_count != combined['exact_near_images']):
            raise RuntimeError('Periodic image setup differs from its error plan')
        self._binding = self._state()

    def _state(self):
        return (id(self.lattice), self.order, self.shells, self.wave_number, self.test_radius,
            self.source_radius, self.separation_ratio, self.image_batch, self.far_count,
            self.image_tail_budget, self._tail_text, self.kernel_error_budget, self._kernel_bound_text,
            self.table_storage, self.row_batch, self.count,
            tuple((id(t), t._version) for t in (self.displacement, self.near_indices, *self.tables())))

    def tables(self):
        return ((self.green, self.gradient) if self.table_storage == 'dense' else
                (self.coefficients, self.indices, self.binomial))

    def _tables(self, start, stop):
        return compact_tables(self, start, stop)

    def action(self, medium, value, *, adjoint=False):
        self.assert_current()
        if medium.k != self.wave_number:
            raise ValueError('Periodic M2L action medium differs from its wave number')
        if (not isinstance(value, torch.Tensor) or value.ndim < 3 or value.shape[-3:] != (2, self.count, 4)
                or value.device != self.displacement.device or value.dtype != self.tables()[0].dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Periodic M2L requires finite matching complex moment channels')
        if self.table_storage == 'compact':
            return compact_moment_action(self, medium, value, adjoint=adjoint)
        return scalar_moment_action(self.green, self.gradient.movedim(0, -1), medium, value, adjoint=adjoint)

    def assert_current(self):
        self.lattice.assert_current()
        if self._state() != self._binding:
            raise ValueError('Periodic M2L state changed')

    def receipt(self):
        self.assert_current()
        return dict(version='finite_bloch_image_cartesian_m2l_v2', dimension=self.lattice.dimension,
            order=self.order, shells=self.shells, images=(2*self.shells+1)**self.lattice.dimension,
            far_images=self.far_count, exact_near_images=len(self.near_indices),
            image_batch=self.image_batch, phase='exp(+i beta.A*n)', displacement='R-A*n',
            table_bytes=sum(t.numel()*t.element_size() for t in self.tables()),
            table_storage=self.table_storage, row_batch=self.row_batch,
            table_budget_scope='accumulated_and_returned_batch_tables_only',
            table_budget_excludes=['recurrence workspace', 'reduction temporaries',
                                  'image metadata', 'generated action tables and workspace', 'allocator overhead'],
            outer_image_tail_certified=False, image_tail_admitted=self.image_tail_budget is not None,
            image_tail_bound=json.loads(self._tail_text), expansion_error_certified=False,
            combined_kernel_admitted=self.kernel_error_budget is not None,
            combined_kernel_bound=json.loads(self._kernel_bound_text),
            physical_qualification=False, peak_memory_qualified=False,
            scope='finite_absorbing_image_sum_with_exact_near_excluded')
