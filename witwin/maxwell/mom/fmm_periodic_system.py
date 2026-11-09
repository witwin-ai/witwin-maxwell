"""One global periodic exterior plus disjoint free-space body interiors."""
import torch
import json
from .fmm_capacity import MultipoleCapacityError
from .fmm_periodic_order import plan_periodic_order, plan_periodic_parameters
from .fmm_periodic_capacity import plan_periodic_storage, periodic_action_tensors, storage_bytes
from .nearfield import ScalarSystemBlocks
from .periodic_nearfield import PeriodicImageSystemBlocks
from .ewald_nearfield import EwaldSystemBlocks
from .cluster import build_support_octree
from .fmm_moments import NestedRWGMultipoles
from .fmm_regional import MultipoleRegionalOperator
from .fmm_periodic_regional import PeriodicMultipoleRegionalOperator
from .direct_system import DirectSystemActions


class PeriodicMultipoleSystemActions:
    """Experimental scalar periodic actions; no public solve qualification."""
    _action = DirectSystemActions._action
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, provider, lattice, *, order=6, quadrature_order=None, leaf_size=32,
                 shells=2, separation_ratio=.5, image_batch=16,
                 table_budget_bytes=512*1024**2, near_budget_bytes=1024**3,
                 projection_budget_bytes=1024**3, remote_kernel_error_budget=None,
                 table_storage="dense", m2l_row_batch=16, near_action_batch=32, exterior_kernel_error_budget=None,
                 action_retained_budget_bytes=1024**3, max_order=None, interior_kernel_error_budget=None, max_shells=None, ewald_kernel=None,shift_storage='dense'):
        if table_storage=='compact_shared_displacement_v1':
            raise NotImplementedError('Shared compact FMM currently requires free space')
        if table_storage == 'fft_v1':
            raise NotImplementedError('Periodic FFT M2L requires separately admitted screened coefficients')
        if table_storage == 'fft_ewald_v1' and ewald_kernel is None:
            raise NotImplementedError('Ewald FFT requires actual screened Ewald coefficients')
        if type(near_action_batch) is not int or near_action_batch < 1:
            raise ValueError('Near action batch must be a positive integer')
        if not isinstance(provider, ScalarSystemBlocks):
            raise TypeError('Periodic multipole system requires scalar system blocks')
        if table_storage == 'fft_ewald_v1':
            from .fmm_storage import m2l_storage_bytes
            m2l_storage_bytes(order, provider.system.dtype.itemsize, 'fft_v1')
            if interior_kernel_error_budget is not None:
                raise NotImplementedError('Ewald FFT error admission requires a complete Ewald error plan')
        provider.assert_current()
        self._ewald_mode = ewald_kernel is not None
        if self._ewald_mode:
            if not isinstance(provider, EwaldSystemBlocks) or provider.kernel is not ewald_kernel or ewald_kernel.lattice is not lattice:
                raise ValueError('Ewald actions require matching original Ewald blocks and lattice')
            if any(value is not None for value in (max_order, max_shells, exterior_kernel_error_budget, remote_kernel_error_budget)):
                raise NotImplementedError('Ewald automatic/kernel admission requires a complete Ewald error plan')
            shells = ewald_kernel.real_shells
        elif isinstance(provider, PeriodicImageSystemBlocks):
            if provider.lattice is not lattice or provider.shells != shells:
                raise ValueError('Periodic action and original blocks must share lattice and image shells')
        else:
            # Bind the original reference to the same image cube as the action.
            # Exterior-only inherited methods remain free-space primitives.
            provider = PeriodicImageSystemBlocks(provider.system, provider.exterior,
                provider.config, lattice, shells=shells)
        for budget in (table_budget_bytes, near_budget_bytes, projection_budget_bytes, action_retained_budget_bytes):
            if type(budget) is not int or budget < 1:
                raise ValueError('Periodic system budgets must be positive integers')
        self.provider = provider
        self.device = provider.system.device
        self.dtype = torch.complex128 if provider.system.dtype == torch.float64 else torch.complex64
        self.shape = (2*provider.system.total_edges,)*2
        lower, upper = provider.support_boxes()
        global_tree = build_support_octree(lower, upper, leaf_size=leaf_size)
        body_trees = tuple(build_support_octree(lower[b.edge_offset:b.edge_offset+b.basis.n_edges],
            upper[b.edge_offset:b.edge_offset+b.basis.n_edges], leaf_size=leaf_size) for b in provider.system.bodies)
        order_plan = None
        if max_shells is not None and max_order is None:
            raise ValueError('Automatic periodic shell selection requires max_order')
        if max_order is not None:
            planner = plan_periodic_order if max_shells is None else plan_periodic_parameters
            shell_options = dict(shells=shells) if max_shells is None else dict(min_shells=shells, max_shells=max_shells)
            order_plan = planner(provider, (global_tree, *body_trees), lattice,
                min_order=order, max_order=max_order, **shell_options, separation_ratio=separation_ratio,
                exterior_budget=exterior_kernel_error_budget, interior_budget=interior_kernel_error_budget,
                remote_budget=remote_kernel_error_budget, table_storage=table_storage,
                action_budget_bytes=action_retained_budget_bytes, table_budget_bytes=table_budget_bytes,
                near_budget_bytes=near_budget_bytes, projection_budget_bytes=projection_budget_bytes,shift_storage=shift_storage)
            order = order_plan['selected_order']
            selected_shells = order_plan.get('selected_shells', shells)
            if selected_shells != shells:
                shells = selected_shells
                provider = PeriodicImageSystemBlocks(provider.system, provider.exterior, provider.config,
                    lattice, shells=shells)
                self.provider = provider
            capacity = order_plan['capacity_plan']
        else:
            capacity = plan_periodic_storage(provider, (global_tree, *body_trees), lattice, order=order,
                shells=shells, separation_ratio=separation_ratio, budget_bytes=action_retained_budget_bytes,
                table_storage=table_storage, ewald=self._ewald_mode,shift_storage=shift_storage)
        self._order_plan_text = json.dumps(order_plan, sort_keys=True, allow_nan=False)
        if not capacity['admitted']:
            raise MultipoleCapacityError(capacity)
        self._capacity_text = json.dumps(capacity, sort_keys=True, allow_nan=False)
        global_projection = NestedRWGMultipoles(provider, global_tree, order=order,
            quadrature_order=quadrature_order, retained_budget_bytes=projection_budget_bytes,shift_storage=shift_storage)
        projection_used = global_projection.receipt()['owned_retained_bytes']
        self.exterior = PeriodicMultipoleRegionalOperator(global_projection, global_projection,
            provider.exterior, lattice, integration=provider.config, shells=shells,
            separation_ratio=separation_ratio, image_batch=image_batch,
            table_budget_bytes=table_budget_bytes, near_budget_bytes=near_budget_bytes,
            projection_budget_bytes=max(1, projection_budget_bytes-projection_used),
            remote_kernel_error_budget=remote_kernel_error_budget,
            table_storage=table_storage, m2l_row_batch=m2l_row_batch, near_action_batch=near_action_batch,
            hierarchy_kernel_error_budget=exterior_kernel_error_budget, ewald_kernel=ewald_kernel)
        projection_used += self.exterior.translated_projection_bytes
        if projection_used > projection_budget_bytes:
            raise MemoryError('Cumulative periodic system projection budget exceeded')
        table_used, near_used = self.exterior.table_bytes, self.exterior.near_bytes
        indices, interiors, projections = [], [], []
        for body, geometry, tree in zip(provider.system.bodies, provider.geometry, body_trees):
            start, stop = body.edge_offset, body.edge_offset+body.basis.n_edges
            remaining = projection_budget_bytes-projection_used
            if remaining <= 0:
                raise MemoryError('Cumulative periodic system projection budget exceeded')
            projection = NestedRWGMultipoles(body.basis, tree, order=order,
                quadrature_order=quadrature_order, geometry=geometry, retained_budget_bytes=remaining,shift_storage=shift_storage)
            projection_used += projection.receipt()['owned_retained_bytes']
            interior = MultipoleRegionalOperator(projection, projection, (body.interior,),
                integration=provider.config, separation_ratio=separation_ratio,
                kernel_error_budget=interior_kernel_error_budget,
                table_budget_bytes=max(1, table_budget_bytes-table_used),
                near_budget_bytes=max(1, near_budget_bytes-near_used),
                table_storage='fft_v1' if table_storage == 'fft_ewald_v1' else table_storage,
                m2l_row_batch=m2l_row_batch, near_action_batch=near_action_batch)
            table_used += interior.table_bytes
            near_used += 0 if interior.near is None else interior.near.receipt()['retained_bytes']
            if table_used > table_budget_bytes or near_used > near_budget_bytes:
                raise MemoryError('Cumulative periodic system table/near budget exceeded')
            edges = torch.arange(start, stop, device=self.device)
            indices.append(torch.cat((edges, edges+provider.system.total_edges)))
            projections.append(projection)
            interiors.append(interior)
        self.indices, self.interiors, self.projections = tuple(indices), tuple(interiors), tuple(projections)
        self.table_bytes, self.near_bytes, self.projection_bytes = table_used, near_used, projection_used
        actual = storage_bytes(periodic_action_tensors(self))
        if actual != capacity['planned_bytes']:
            raise RuntimeError(f'Periodic action storage differs from structural plan: {actual} != {capacity["planned_bytes"]}')
        self._binding = self._state()

    def _state(self):
        return (id(self.provider), id(self.exterior), self.shape, self.device, self.dtype,
            tuple(map(id, self.interiors)), tuple(map(id, self.projections)),
            tuple((id(t), t._version) for t in self.indices),
            self.table_bytes, self.near_bytes, self.projection_bytes, self._capacity_text, self._order_plan_text, self._ewald_mode)

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Periodic multipole system changed')
        self.provider.assert_current()
        self.exterior.assert_current()
        for region in self.interiors:
            region.assert_current()

    @property
    def order_plan(self):
        return json.loads(self._order_plan_text)

    @property
    def capacity_plan(self):
        return json.loads(self._capacity_text)

    def receipt(self):
        self.assert_current()
        return dict(version='global_periodic_exterior_free_space_interiors_multipole_v1',
            shape=list(self.shape), bodies=len(self.interiors), table_bytes=self.table_bytes,
            original_reference_scope=("configured_ewald_quadrature_and_cutoffs" if self._ewald_mode else "configured_finite_periodic_image_cube"),
            original_reference_shells=self.provider.kernel.real_shells if self._ewald_mode else self.provider.shells, infinite_periodic_certified=False,
            near_bytes=self.near_bytes, projection_bytes=self.projection_bytes, capacity_plan=self.capacity_plan, order_plan=self.order_plan,
            exterior=self.exterior.receipt(), interiors=[r.receipt() for r in self.interiors],
            interiors_periodic=False, current_backward='fixed_operator_adjoint_no_saved_action_intermediates',
            retained_budget_excludes=self.capacity_plan['excludes'],
            original_global_error_certified=False, physical_qualification=False,
            complexity_qualification=False, peak_memory_qualified=False)
