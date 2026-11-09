"""Separated-support electromagnetic multipole-to-local translations."""
import math
import torch
from .fmm_moments import NestedRWGMultipoles
from .fmm_expansion import multipole_to_local
from .physics import Medium
from .fmm_error import (multipole_truncation_bound, multipole_truncation_bounds,
                        require_multipole_budget, validate_kernel_bound_rule)
from .interpolation_error import KernelErrorBudget
from .scalar_moment_action import scalar_moment_action
from .fmm_compact import CompactElectromagneticM2L, SignedAxisCompactElectromagneticM2L, RadialSeriesElectromagneticM2L
from .fmm_fft import FFTElectromagneticM2L
from .fmm_storage import m2l_storage_bytes, shared_displacement_catalog, signed_axis_displacement_catalog, signed_axis_catalog_rule, radial_series_pair_catalog, radial_series_pair_storage_bytes, radial_series_catalog_rule


class ElectromagneticM2L:
    """Fixed-order PMCHWT far coupling; no truncation/physical qualification."""
    def __init__(self, test, source, test_node, source_node, medium, *,
                 separation_ratio=.5, table_budget_bytes=512*1024**2, error_budget=None, table_storage='dense', row_batch=16,
                 kernel_bound_rule='bivariate_cauchy_v1'):
        if table_storage in ('compact_shared_signed_axis_v1','radial_series_shared_radius_v1') and (type(row_batch) is not int or not 1 <= row_batch <= 16):
            raise ValueError('Signed-axis M2L requires row batch1..16')
        displacement = self._prepare(test, source, test_node, source_node, medium,
            separation_ratio=separation_ratio, error_budget=error_budget, kernel_bound_rule=kernel_bound_rule)
        m2l_storage_bytes(test.order, displacement.element_size(), table_storage)
        if table_storage == 'radial_series_shared_radius_v1':
            if test.basis is not source.basis or test.tree is not source.tree:
                raise ValueError('Shared radial series M2L requires identical physical spaces and trees')
            translation=RadialSeriesElectromagneticM2L.create_many(displacement[None],medium,test.order,
                row_batch=row_batch,retained_budget_bytes=table_budget_bytes)[0]
            self._bind_compact(translation,storage=table_storage)
            return
        if table_storage == 'fft_v1':
            self._bind_compact(FFTElectromagneticM2L(displacement, medium, test.order,
                retained_budget_bytes=table_budget_bytes), storage='fft_v1')
            return
        if table_storage in ('compact','compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1') and (test.basis is not source.basis or test.tree is not source.tree):
                raise ValueError('Shared compact M2L requires identical physical spaces and trees')
            owner = CompactElectromagneticM2L(displacement, medium, test.order,
                row_batch=row_batch, retained_budget_bytes=table_budget_bytes)
            translation = (SignedAxisCompactElectromagneticM2L(owner,(0,1,2),(1,1,1))
                if table_storage == 'compact_shared_signed_axis_v1' else owner)
            self._bind_compact(translation,storage=table_storage)
            return
        green, gradient = multipole_to_local(displacement, medium.k, test.order,
                                             table_budget_bytes=table_budget_bytes)
        self._bind_tables(green, gradient.movedim(-3, -1).contiguous())

    @classmethod
    def create_many(cls, test, source, pairs, medium, *, separation_ratio=.5,
                    table_budget_bytes=512*1024**2, error_budget=None, table_storage='dense', row_batch=16,
                 kernel_bound_rule='bivariate_cauchy_v1'):
        """Validate shared spaces once and admit support pairs in CUDA batches."""
        if table_storage in ('compact_shared_signed_axis_v1','radial_series_shared_radius_v1') and (type(row_batch) is not int or not 1 <= row_batch <= 16):
            raise ValueError('Signed-axis M2L requires row batch1..16')
        pairs = tuple(pairs)
        catalog=None
        if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            cls._validate_shared(test,source,medium,separation_ratio,kernel_bound_rule)
            if test.basis is not source.basis or test.tree is not source.tree:
                raise ValueError('Shared compact M2L requires identical physical spaces and trees')
            catalog=(radial_series_pair_catalog if table_storage == 'radial_series_shared_radius_v1'
                else signed_axis_displacement_catalog if table_storage == 'compact_shared_signed_axis_v1'
                else shared_displacement_catalog)(test.centers,pairs)
        required = m2l_storage_bytes(test.order, test.centers.element_size(), table_storage)*(len(pairs) if catalog is None else len(catalog[0]))
        if table_storage == 'radial_series_shared_radius_v1':
            required += 24*len(pairs)
        if type(table_budget_bytes) is not int or table_budget_bytes < 1:
            raise ValueError('M2L batch retained budget must be a positive integer')
        if required > table_budget_bytes:
            raise MemoryError('M2L batch retained budget exceeded before coefficients')
        objects, displacements = [], []
        if pairs:
            cls._validate_shared(test, source, medium, separation_ratio, kernel_bound_rule)
        for first in range(0, len(pairs), 32):
            current = pairs[first:first+32]
            # Preserve ordered scalar failure details for malformed indices/budgets.
            if (any(type(i) is not int or i < 0 or i >= p.tree.node_count
                    for a, b in current for i, p in ((a, test), (b, source)))
                    or error_budget is not None and not isinstance(error_budget, KernelErrorBudget)):
                for a, b in current:
                    cls.__new__(cls)._prepare(test, source, a, b, medium,
                        separation_ratio=separation_ratio, error_budget=error_budget,
                        kernel_bound_rule=kernel_bound_rule)
                raise AssertionError('Invalid M2L admission did not reject')
            aa, bb = torch.tensor(current, device=test.device, dtype=torch.long).unbind(-1)
            delta = test.centers[aa]-source.centers[bb]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            separated = (distance > 0) & (test.radii[aa]+source.radii[bb] <= separation_ratio*distance)
            admitted = separated.clone()
            if error_budget is not None:
                # Invalid geometry must not preempt an earlier kernel failure.
                selected = separated.nonzero().flatten()
                if selected.numel():
                    bounds = multipole_truncation_bounds(delta[selected], test.radii[aa[selected]],
                        source.radii[bb[selected]], medium.k, test.order, kernel_bound_rule=kernel_bound_rule)
                    admitted[selected] = (bounds['available']
                        & (bounds['green_absolute'] <= error_budget.green_absolute)
                        & (bounds['gradient_component_absolute'] <= error_budget.gradient_component_absolute))
            if not bool(admitted.all()):
                index = int((~admitted).nonzero()[0, 0])
                a, b = current[index]
                cls.__new__(cls)._prepare(test, source, a, b, medium,
                    separation_ratio=separation_ratio, error_budget=error_budget,
                    kernel_bound_rule=kernel_bound_rule)
                # Native batching can differ from scalar rounding at the budget boundary.
                raise ValueError('Batched M2L truncation admission is unresolved at native precision')
            for index, (a, b) in enumerate(current):
                obj = cls.__new__(cls)
                obj._bind_geometry(test, source, a, b, medium,
                    separation_ratio=separation_ratio, error_budget=error_budget,
                    kernel_bound_rule=kernel_bound_rule)
                if catalog is None or table_storage == 'radial_series_shared_radius_v1':
                    displacements.append(delta[index])
                objects.append(obj)
        if not objects:
            return ()
        if table_storage == 'radial_series_shared_radius_v1':
            translations=RadialSeriesElectromagneticM2L.create_many(torch.stack(displacements),medium,test.order,
                row_batch=row_batch,retained_budget_bytes=table_budget_bytes)
            for obj,translation in zip(objects,translations,strict=True):
                obj._bind_compact(translation,storage=table_storage)
            return tuple(objects)
        if table_storage == 'fft_v1':
            translations = FFTElectromagneticM2L.create_many(torch.stack(displacements), medium, test.order,
                retained_budget_bytes=table_budget_bytes)
            for obj, translation in zip(objects, translations, strict=True):
                obj._bind_compact(translation, storage='fft_v1')
            return tuple(objects)
        if catalog is not None:
            representatives,bindings=catalog
            owners=[]
            for start in range(0,len(representatives),32):
                current=[pairs[n] for n in representatives[start:start+32]]
                aa,bb=torch.tensor(current,device=test.device,dtype=torch.long).unbind(-1)
                representative_delta=test.centers[aa]-source.centers[bb]
                owners.extend(CompactElectromagneticM2L.create_many(representative_delta,
                    medium,test.order,row_batch=row_batch,retained_budget_bytes=table_budget_bytes))
                del aa,bb,representative_delta
            if table_storage == 'compact_shared_signed_axis_v1':
                for obj,(index,permutation,signs) in zip(objects,bindings,strict=True):
                    obj._bind_compact(SignedAxisCompactElectromagneticM2L(owners[index],permutation,signs),storage=table_storage)
                return tuple(objects)
            reverses={}
            for obj,(index,reverse) in zip(objects,bindings,strict=True):
                if reverse and index not in reverses:
                    reverses[index]=owners[index].reversed_translation()
                obj._bind_compact(reverses[index] if reverse else owners[index],storage=table_storage)
            return tuple(objects)
        if table_storage == 'compact':
            compact = CompactElectromagneticM2L.create_many(torch.stack(displacements), medium, test.order,
                row_batch=row_batch, retained_budget_bytes=table_budget_bytes)
            for obj, coefficients in zip(objects, compact, strict=True):
                obj._bind_compact(coefficients)
            return tuple(objects)
        green, gradient = multipole_to_local(torch.stack(displacements), medium.k, test.order,
                                             table_budget_bytes=table_budget_bytes)
        gradient = gradient.movedim(-3, -1).contiguous()
        for i, obj in enumerate(objects):
            obj._bind_tables(green[i], gradient[i])
        return tuple(objects)

    @staticmethod
    def _validate_shared(test, source, medium, separation_ratio, kernel_bound_rule):
        validate_kernel_bound_rule(kernel_bound_rule)
        if not isinstance(test, NestedRWGMultipoles) or not isinstance(source, NestedRWGMultipoles):
            raise TypeError('M2L requires RWG multipole spaces')
        test.assert_current()
        if source is not test:
            source.assert_current()
        if test.device != source.device or test.dtype != source.dtype or test.order != source.order:
            raise ValueError('M2L spaces must share order, device and precision')
        if type(separation_ratio) not in (int, float) or not math.isfinite(separation_ratio) or not 0 < separation_ratio < 1:
            raise ValueError('M2L separation ratio must lie strictly between zero and one')
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Electromagnetic M2L requires a finite-frequency scalar medium')

    def _prepare(self, test, source, test_node, source_node, medium, *, separation_ratio, error_budget,
                 kernel_bound_rule):
        validate_kernel_bound_rule(kernel_bound_rule)
        if not isinstance(test, NestedRWGMultipoles) or not isinstance(source, NestedRWGMultipoles):
            raise TypeError('M2L requires RWG multipole spaces')
        test.assert_current()
        source.assert_current()
        if test.device != source.device or test.dtype != source.dtype or test.order != source.order:
            raise ValueError('M2L spaces must share order, device and precision')
        if any(type(i) is not int or i < 0 or i >= p.tree.node_count
               for i, p in ((test_node, test), (source_node, source))):
            raise ValueError('M2L node index is invalid')
        if type(separation_ratio) not in (int, float) or not math.isfinite(separation_ratio) or not 0 < separation_ratio < 1:
            raise ValueError('M2L separation ratio must lie strictly between zero and one')
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Electromagnetic M2L requires a finite-frequency scalar medium')
        displacement = test.centers[test_node]-source.centers[source_node]
        distance = torch.linalg.vector_norm(displacement)
        radii = test.radii[test_node]+source.radii[source_node]
        if not bool((distance > 0) & (radii <= separation_ratio*distance)):
            raise ValueError('M2L requires separated complete-support spheres')
        if error_budget is not None:
            bound = multipole_truncation_bound(displacement, test.radii[test_node], source.radii[source_node],
                                              medium.k, test.order, kernel_bound_rule=kernel_bound_rule)
            require_multipole_budget(bound, error_budget)
        self._bind_geometry(test, source, test_node, source_node, medium,
            separation_ratio=separation_ratio, error_budget=error_budget,
            kernel_bound_rule=kernel_bound_rule)
        return displacement

    def _bind_geometry(self, test, source, test_node, source_node, medium, *,
                       separation_ratio, error_budget, kernel_bound_rule):
        self.error_budget = error_budget
        self.kernel_bound_rule = kernel_bound_rule
        self.test, self.source, self.medium = test, source, medium
        self.test_node, self.source_node = test_node, source_node
        self.separation_ratio = separation_ratio
        self.device, self.dtype = test.device, test.dtype

    def _bind_tables(self, green, gradient):
        self.table_storage = 'dense'
        self._compact = None
        self.green, self.gradient = green, gradient
        self._binding = self._state()
        self._versions = tuple((id(t), t._version) for t in self.tensors())

    def _bind_compact(self, compact, *, storage='compact'):
        self.table_storage = storage
        self._compact = compact
        self.green = self.gradient = None
        self._binding = self._state()
        self._versions = tuple((id(t), t._version) for t in self.tensors())

    def tensors(self):
        return self._compact.tensors() if self._compact is not None else (self.green, self.gradient)

    def _state(self):
        return (id(self.test), id(self.source), id(self.medium), self.test_node, self.source_node,
                self.separation_ratio, self.device, self.dtype, self.medium.frequency,
                self.medium.epsilon_r, self.medium.mu_r, self.medium.k, self.medium.eta, self.error_budget, self.kernel_bound_rule, id(self._compact), self.table_storage)

    def _assert_tables_current(self):
        """Validate this coupling; composed callers validate shared spaces once."""
        if self._state() != self._binding or tuple((id(t), t._version) for t in self.tensors()) != self._versions:
            raise ValueError('Electromagnetic M2L changed')
        if self._compact is not None:
            self._compact.assert_current()

    def assert_current(self):
        self._assert_tables_current()
        self.test.assert_current()
        if self.source is not self.test:
            self.source.assert_current()

    def action(self, value, *, adjoint=False):
        self.assert_current()
        return self._action_with_current_spaces(value, adjoint=adjoint)

    def _action_with_current_spaces(self, value, *, adjoint=False):
        """Internal action after both shared projections have been validated."""
        self._assert_tables_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 3
                or value.shape[-3:] != (2, self.source.count, 4) or value.device != self.device
                or value.dtype != self.dtype or self.table_storage != 'radial_series_shared_radius_v1'
                and not bool(torch.isfinite(value).all())):
            raise ValueError('M2L requires finite matching complex moment channels')
        if self._compact is not None:
            return self._compact.adjoint(value) if adjoint else self._compact.apply(value)
        return scalar_moment_action(self.green, self.gradient, self.medium, value, adjoint=adjoint)

    def truncation_bound(self):
        self.assert_current()
        return multipole_truncation_bound(self.test.centers[self.test_node]-self.source.centers[self.source_node],
            self.test.radii[self.test_node], self.source.radii[self.source_node], self.medium.k, self.test.order, kernel_bound_rule=self.kernel_bound_rule)

    def receipt(self):
        self.assert_current()
        return {'version': 'scalar_pmchwt_cartesian_m2l_v1',
                **({} if self.kernel_bound_rule == 'bivariate_cauchy_v1' else {'kernel_bound_rule': self.kernel_bound_rule}), 'order': self.test.order,
                'table_bytes': sum(t.numel()*t.element_size() for t in self.tensors()),
                'table_storage': self.table_storage,
                'separation_ratio': self.separation_ratio, 'truncation_certified': False,
                'physical_qualification': False, 'peak_memory_qualified': False}
