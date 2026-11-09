"""Periodic RWG regional action with root remote and hierarchical near images."""
from dataclasses import replace
import math
import json
import torch
from .mesh import RWGBasis, SurfaceMesh
from .periodic_nearfield import translated_system_provider
from .nearfield import ScalarSystemBlocks
from .physics import Medium
from .operator import IntegrationConfig
from .fmm_moments import NestedRWGMultipoles
from .fmm_regional import MultipoleRegionalOperator
from .fmm_periodic import PeriodicImageM2L
from .fmm_periodic_error import periodic_hierarchy_error_bound, require_periodic_multipole_budget


class PeriodicMultipoleRegionalOperator:
    """Experimental local/global exterior action; interiors are never periodized.

    Globally separated images share one root M2L. Remaining images use the
    existing hierarchical free-space regional operator, including exact leaf
    quadrature. This is not a global periodic system or public solve adapter.
    """
    def __init__(self, test, source, medium, lattice, *, integration=IntegrationConfig(),
                 shells=2, separation_ratio=.5, image_batch=16,
                 table_budget_bytes=512*1024**2, near_budget_bytes=1024**3,
                 projection_budget_bytes=1024**3, remote_kernel_error_budget=None,
                 table_storage="dense", m2l_row_batch=16, near_action_batch=32, hierarchy_kernel_error_budget=None, ewald_kernel=None):
        if table_storage == 'fft_v1':
            raise NotImplementedError('Periodic FFT M2L requires separately admitted screened coefficients')
        if table_storage == 'fft_ewald_v1' and ewald_kernel is None:
            raise NotImplementedError('Ewald FFT requires actual screened Ewald coefficients')
        if type(near_action_batch) is not int or near_action_batch < 1:
            raise ValueError('Near action batch must be a positive integer')
        if not isinstance(test, NestedRWGMultipoles) or not isinstance(source, NestedRWGMultipoles):
            raise TypeError('Periodic region requires nested RWG multipoles')
        if test.tree.expansion_centers is not None or source.tree.expansion_centers is not None:
            raise NotImplementedError('Dyadic expansion centers currently require free space')
        if table_storage == 'fft_ewald_v1':
            from .fmm_storage import m2l_storage_bytes
            m2l_storage_bytes(test.order, test.centers.element_size(), 'fft_v1')
        test.assert_current()
        source.assert_current()
        global_space = isinstance(test.basis, ScalarSystemBlocks)
        if table_storage == 'fft_ewald_v1' and global_space:
            from .ewald_nearfield import EwaldSystemBlocks
            if not isinstance(test.basis, EwaldSystemBlocks) or test.basis.kernel is not ewald_kernel:
                raise ValueError('Ewald FFT region requires matching original Ewald provider and kernel')
        if global_space:
            if (source.basis is not test.basis or medium is not test.basis.exterior
                    or integration is not test.basis.config):
                raise ValueError('Periodic global exterior requires one shared provider and integration')
        elif not isinstance(test.basis, RWGBasis) or not isinstance(source.basis, RWGBasis):
            raise TypeError('Periodic region requires matching local or global RWG spaces')
        if test.device != source.device or test.dtype != source.dtype or test.order != source.order:
            raise ValueError('Periodic regional spaces must share device, precision and order')
        if not isinstance(medium, Medium) or medium.frequency <= 0:
            raise ValueError('Periodic region requires a finite-frequency scalar medium')
        if not isinstance(integration, IntegrationConfig):
            raise TypeError('Periodic integration requires IntegrationConfig')
        for budget in (table_budget_bytes, near_budget_bytes, projection_budget_bytes):
            if type(budget) is not int or budget < 1:
                raise ValueError('Periodic regional budgets must be positive integers')
        lattice.assert_current()
        margin = 512*torch.finfo(lattice.vectors.dtype).eps
        bases = ([body.basis for body in test.basis.system.bodies] if global_space else [test.basis, source.basis])
        for basis in bases:
            vertices = basis.mesh.vertices
            if vertices.device != lattice.vectors.device or vertices.dtype != lattice.vectors.dtype:
                raise ValueError('Periodic region geometry must match lattice')
            if bool(((vertices@lattice.reciprocal.T/(2*math.pi)).abs() >= .5-margin).any()):
                raise ValueError('Periodic bodies must lie strictly inside the centered cell')
        self.test, self.source, self.medium = test, source, medium
        self.shape, self.device, self.dtype = (2*test.tree.size, 2*source.tree.size), test.device, test.dtype
        if ewald_kernel is not None:
            from .periodic import EwaldHelmholtzKernel
            if not isinstance(ewald_kernel, EwaldHelmholtzKernel):
                raise TypeError('Ewald region requires EwaldHelmholtzKernel')
            if hierarchy_kernel_error_budget is not None or remote_kernel_error_budget is not None:
                raise NotImplementedError('Ewald hierarchy error admission requires its own complete bound')
        hierarchy = None
        if hierarchy_kernel_error_budget is not None:
            hierarchy = periodic_hierarchy_error_bound(test.tree, source.tree, lattice, medium.k,
                order=test.order, shells=shells, separation_ratio=separation_ratio, image_batch=image_batch)
            require_periodic_multipole_budget(hierarchy, hierarchy_kernel_error_budget)
        self.hierarchy_kernel_error_budget = hierarchy_kernel_error_budget
        self._hierarchy_text = json.dumps(hierarchy, sort_keys=True, allow_nan=False)
        if ewald_kernel is None:
            self.remote = PeriodicImageM2L(lattice, test.centers[0]-source.centers[0],
                float(test.radii[0]), float(source.radii[0]), medium.k, order=test.order,
                shells=shells, separation_ratio=separation_ratio, image_batch=image_batch,
                table_budget_bytes=table_budget_bytes, kernel_error_budget=remote_kernel_error_budget,
                table_storage=table_storage, row_batch=m2l_row_batch)
            self._remote_active = bool(self.remote.far_count)
        else:
            from .fmm_ewald_translation import EwaldRootM2L
            if test is not source or ewald_kernel.lattice is not lattice:
                raise ValueError('Ewald root composition requires identical projection and lattice')
            if table_storage == 'fft_ewald_v1':
                from .fmm_ewald_fft import EwaldFFTTerminalM2L
                root_translation = EwaldFFTTerminalM2L
            else:
                root_translation = EwaldRootM2L
            self.remote = root_translation(ewald_kernel, medium, float(test.radii[0]), float(source.radii[0]),
                order=test.order, row_batch=m2l_row_batch, separation_ratio=separation_ratio,
                retained_budget_bytes=table_budget_bytes, coefficient_budget_bytes=table_budget_bytes)
            self._remote_active = True
        table_used = self.remote.receipt()['table_bytes']
        near_used, projection_used = 0, 0
        regions, phases = [], []
        for index in self.remote.near_indices:
            translation = lattice.translations(index)
            if bool((index == 0).all()):
                shifted = source
            else:
                def translated(original):
                    return RWGBasis(SurfaceMesh(original.mesh.vertices+translation, original.mesh.faces, validate=False))
                if global_space:
                    basis = translated_system_provider(source.basis, translation)
                else:
                    basis = translated(source.basis)
                tree = replace(source.tree, lower=source.tree.lower+translation, upper=source.tree.upper+translation)
                shifted = NestedRWGMultipoles(basis, tree, order=source.order,
                    quadrature_order=source.quadrature_order,
                    retained_budget_bytes=max(1, projection_budget_bytes-projection_used),shift_storage=source.shift_storage)
                projection_used += sum(t.numel()*t.element_size() for t in shifted.tensors())
                if projection_used > projection_budget_bytes:
                    raise MemoryError('Periodic translated projection budget exceeded')
            region = MultipoleRegionalOperator(test, shifted, (medium,), integration=integration,
                separation_ratio=separation_ratio, table_budget_bytes=max(1, table_budget_bytes-table_used),
                near_budget_bytes=max(1, near_budget_bytes-near_used),
                table_storage='fft_v1' if table_storage == 'fft_ewald_v1' else table_storage,
                m2l_row_batch=m2l_row_batch, near_action_batch=near_action_batch)
            table_used += region.table_bytes
            near_used += 0 if region.near is None else region.near.receipt()['retained_bytes']
            if table_used > table_budget_bytes or near_used > near_budget_bytes:
                raise MemoryError('Cumulative periodic regional table/near budget exceeded')
            regions.append(region)
            phases.append(lattice.phase(index))
        self.regions = tuple(regions)
        self.phases = torch.stack(phases) if phases else torch.empty(0, device=self.device, dtype=self.dtype)
        self.table_bytes, self.near_bytes, self.translated_projection_bytes = table_used, near_used, projection_used
        self._binding = self._state()

    def _state(self):
        return (id(self.test), id(self.source), id(self.medium), id(self.remote), self.shape, self.device, self.dtype,
            tuple(map(id, self.regions)), id(self.phases), self.phases._version,
            self.table_bytes, self.near_bytes, self.translated_projection_bytes, self._remote_active,
            self.hierarchy_kernel_error_budget, self._hierarchy_text,
            self.medium.frequency, self.medium.epsilon_r, self.medium.mu_r, self.medium.k, self.medium.eta)

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Periodic regional state changed')
        self.test.assert_current()
        self.source.assert_current()
        self.remote.assert_current()
        for region in self.regions:
            region.assert_current()

    def _action(self, value, *, adjoint):
        self.assert_current()
        source, target = (self.test, self.source) if adjoint else (self.source, self.test)
        ns, nt = source.tree.size, target.tree.size
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != 2*ns
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Periodic action requires finite matching complex coefficients')
        result = value.new_zeros((*value.shape[:-1], 2*nt))
        if self._remote_active:
            moments = source.upward(torch.stack((value[..., :ns], value[..., ns:]), dim=-2))
            local = [value.new_zeros((*value.shape[:-1], 2, target.count, 4)) for _ in range(target.tree.node_count)]
            local[0] = self.remote.action(self.medium, moments[0], adjoint=adjoint)
            projected = target.downward(local)
            result = torch.cat((projected[..., 0, :], projected[..., 1, :]), dim=-1)
        for phase, region in zip(self.phases, self.regions):
            result = result+(phase.conj()*region.adjoint(value) if adjoint else phase*region.apply(value))
        return result

    def apply(self, value):
        return self._action(value, adjoint=False)

    def adjoint(self, value):
        return self._action(value, adjoint=True)

    def receipt(self):
        self.assert_current()
        return dict(version='periodic_regional_multipole_v3', shape=list(self.shape),
            remote=self.remote.receipt(), hierarchical_near_images=len(self.regions),
            hierarchy_kernel_admitted=self.hierarchy_kernel_error_budget is not None,
            hierarchy_kernel_bound=json.loads(self._hierarchy_text),
            table_bytes=self.table_bytes, near_bytes=self.near_bytes,
            translated_projection_bytes=self.translated_projection_bytes,
            excludes=['input projections', 'translated geometry/trees', 'action/setup/backward workspace'],
            remote_budget_scope='root_remote_images_only_excludes_near_image_hierarchy',
            physical_qualification=False, complexity_qualification=False, peak_memory_qualified=False)
