"""Global exterior and disjoint interior composition of true multipole actions."""
import torch
import json
from .fmm_order import plan_multipole_order
from .fmm_error import validate_kernel_bound_rule
from .fmm_storage import m2l_storage_bytes
from .fmm_capacity import plan_multipole_storage, MultipoleCapacityError, multipole_action_tensors, storage_bytes
from .rwg_interpolation_error import rwg_integral_logs, integral_statistics_from_logs
from .nearfield import ScalarSystemBlocks, _validate_scalar_channel_medium
from .cluster import build_support_octree
from .fmm_moments import NestedRWGMultipoles
from .fmm_regional import MultipoleRegionalOperator
from .direct_system import DirectSystemActions


class MultipoleSystemActions:
    """Fixed-order scalar system action; no production/complexity qualification."""
    _action = DirectSystemActions._action
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, provider, *, order=6, quadrature_order=None, leaf_size=32,
                 separation_ratio=.5, table_budget_bytes=512*1024**2,
                 near_budget_bytes=1024**3, projection_budget_bytes=1024**3,
                 max_order=None, kernel_error_budget=None, rwg_absolute_budget=None,
                 action_retained_budget_bytes=1024**3, table_storage='dense', m2l_row_batch=16, near_action_batch=32,shift_storage='dense',
                 kernel_bound_rule='bivariate_cauchy_v1',near_storage='full_matrix', far_action_batch=1,
                 expansion_center_policy='support_box_midpoint_v1',near_coefficient_workspace_budget_bytes=128*1024**2,
                 near_coefficient_cache_budget_bytes=128*1024**2):
        if expansion_center_policy not in ('support_box_midpoint_v1', 'dyadic_cell_v1', 'binary_lattice_v1'):
            raise ValueError('Unsupported Cartesian FMM expansion center policy')
        if expansion_center_policy in ('dyadic_cell_v1','binary_lattice_v1') and (
                table_storage not in ('compact', 'compact_shared_displacement_v1', 'compact_shared_signed_axis_v1','radial_series_shared_radius_v1')
                or order > 12 or max_order is not None and max_order > 12):
            raise ValueError('Dyadic cell centers require free-space compact order0..12')
        self.expansion_center_policy = expansion_center_policy
        validate_kernel_bound_rule(kernel_bound_rule)
        self.kernel_bound_rule = kernel_bound_rule
        if type(far_action_batch) is not int or not 1 <= far_action_batch <= 32:
            raise ValueError('Compact FAR action batch must lie between one and32')
        if far_action_batch > 1 and (table_storage not in ('compact','compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1')
                or order > 12 or max_order is not None and max_order > 12):
            raise ValueError('Batched compact FAR requires compact order0..12')
        self.far_action_batch = far_action_batch
        if near_storage not in ('full_matrix', 'scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1'):
            raise ValueError('Unsupported FMM near storage')
        self.near_storage = near_storage
        if (type(near_coefficient_workspace_budget_bytes) is not int
                or near_coefficient_workspace_budget_bytes < 1):
            raise ValueError('Near coefficient workspace budget must be a positive integer')
        if (near_storage not in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1')
                and near_coefficient_workspace_budget_bytes != 128*1024**2):
            raise ValueError('Explicit near coefficient workspace requires streamed reciprocal storage')
        self.near_coefficient_workspace_budget_bytes = near_coefficient_workspace_budget_bytes
        if (type(near_coefficient_cache_budget_bytes) is not int
                or near_coefficient_cache_budget_bytes < 0):
            raise ValueError('Near coefficient cache budget must be a nonnegative integer')
        if (near_storage != 'scalar_reciprocal_cached_dk_v1'
                and near_coefficient_cache_budget_bytes != 128*1024**2):
            raise ValueError('Explicit near coefficient cache budget requires cached reciprocal storage')
        self.near_coefficient_cache_budget_bytes = near_coefficient_cache_budget_bytes
        if type(near_action_batch) is not int or near_action_batch < 1:
            raise ValueError('Near action batch must be a positive integer')
        if not isinstance(provider, ScalarSystemBlocks):
            raise TypeError('Multipole system requires scalar system blocks')
        provider.assert_current()
        if expansion_center_policy in ('dyadic_cell_v1','binary_lattice_v1') and provider.system.dtype != torch.float64:
            raise ValueError('Dyadic cell centers require native float64 geometry')
        if far_action_batch > 1 and provider.system.dtype != torch.float64:
            raise ValueError('Batched compact FAR requires native float64 geometry')
        if near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1') and provider.system.dtype != torch.float64:
            raise ValueError('Scalar D/K FMM requires native float64 geometry')
        if near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1'):
            for medium in (provider.exterior, *(body.interior for body in provider.system.bodies)):
                _validate_scalar_channel_medium(medium)
        if table_storage in ('compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            if type(m2l_row_batch) is not int or not 1 <= m2l_row_batch <= 16:
                raise ValueError('Signed-axis M2L requires row batch1..16')
            if max_order is not None:
                m2l_storage_bytes(max_order,provider.system.dtype.itemsize,table_storage)
        m2l_storage_bytes(order, provider.system.dtype.itemsize, table_storage)
        if table_storage == 'fft_v1' and max_order is not None:
            m2l_storage_bytes(max_order, provider.system.dtype.itemsize, table_storage)
        if type(m2l_row_batch) is not int or m2l_row_batch < 1:
            raise ValueError('M2L row batch must be a positive integer')
        if any(type(v) is not int or v < 1 for v in (table_budget_bytes, near_budget_bytes, projection_budget_bytes, action_retained_budget_bytes)):
            raise ValueError('Multipole system retained budgets must be positive integers')
        if rwg_absolute_budget is not None and max_order is None:
            raise ValueError('RWG absolute admission requires automatic FMM order selection')
        self.provider = provider
        self.device = provider.system.device
        self.dtype = torch.complex128 if provider.system.dtype == torch.float64 else torch.complex64
        self.shape = (2*provider.system.total_edges,)*2
        lower, upper = provider.support_boxes()
        global_tree = build_support_octree(lower, upper, leaf_size=leaf_size,
            expansion_center_policy=expansion_center_policy)
        body_trees = tuple(build_support_octree(lower[b.edge_offset:b.edge_offset+b.basis.n_edges],
            upper[b.edge_offset:b.edge_offset+b.basis.n_edges], leaf_size=leaf_size,
            expansion_center_policy=expansion_center_policy) for b in provider.system.bodies)
        plan = None
        if max_order is not None:
            statistics, impedance = None, None
            if rwg_absolute_budget is not None:
                logs = tuple(rwg_integral_logs(body.basis, geometry)
                             for body, geometry in zip(provider.system.bodies, provider.geometry))
                statistics = (integral_statistics_from_logs(global_tree, torch.cat(logs)),
                    *tuple(integral_statistics_from_logs(tree, values) for tree, values in zip(body_trees, logs)))
                impedance = abs(provider.exterior.eta)
            plan = plan_multipole_order(((global_tree, provider.exterior),
                *tuple((tree, body.interior) for tree, body in zip(body_trees, provider.system.bodies))),
                kernel_error_budget, min_order=order, max_order=max_order,
                separation_ratio=separation_ratio, table_budget_bytes=table_budget_bytes,
                rwg_statistics=statistics, rwg_absolute_budget=rwg_absolute_budget, balancing_impedance=impedance, table_storage=table_storage, kernel_bound_rule=kernel_bound_rule)
            order = plan['selected_order']
            if rwg_absolute_budget is not None:
                required = (order+4)//2
                if quadrature_order is not None and (type(quadrature_order) is not int or quadrature_order < required):
                    raise ValueError('RWG absolute admission requires polynomial-exact moment quadrature')
                del statistics, logs
        self._order_plan_text = json.dumps(plan, sort_keys=True, allow_nan=False)
        capacity = plan_multipole_storage((global_tree, *body_trees), order=order,
            separation_ratio=separation_ratio, budget_bytes=action_retained_budget_bytes, table_storage=table_storage,shift_storage=shift_storage,near_storage=near_storage,
            near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes,
            near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)
        if not capacity['admitted']:
            raise MultipoleCapacityError(capacity)
        categories = capacity['categories']
        projection_planned = sum(categories[key] for key in
            ('centers_radii', 'leaf_maps', 'leaf_moments', 'shifts'))
        for component, planned, budget in (
                ('table', categories['m2l'], table_budget_bytes),
                ('near', categories['near'], near_budget_bytes),
                ('projection', projection_planned, projection_budget_bytes)):
            if planned > budget:
                raise MemoryError(f'Multipole system {component} budget exceeded before moment setup '
                                  f'({planned}>{budget} bytes)')
        self._capacity_text = json.dumps(capacity, sort_keys=True, allow_nan=False)
        table_used = near_used = projection_used = 0
        def build(basis, tree, medium, geometry=None):
            nonlocal table_used, near_used, projection_used
            remaining = projection_budget_bytes-projection_used
            if remaining <= 0:
                raise MemoryError('Cumulative multipole projection budget exceeded')
            moments = NestedRWGMultipoles(basis, tree, order=order, quadrature_order=quadrature_order,
                geometry=geometry, retained_budget_bytes=remaining,shift_storage=shift_storage)
            projection_used += moments.receipt()['owned_retained_bytes']
            region = MultipoleRegionalOperator(moments, moments, (medium,), integration=provider.config,
                separation_ratio=separation_ratio, table_budget_bytes=max(1, table_budget_bytes-table_used),
                near_budget_bytes=max(1, near_budget_bytes-near_used), kernel_error_budget=kernel_error_budget,
                table_storage=table_storage, m2l_row_batch=m2l_row_batch, near_action_batch=near_action_batch, kernel_bound_rule=kernel_bound_rule,near_storage=near_storage,far_action_batch=far_action_batch,
                near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes,
                near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)
            table_used += region.table_bytes
            near_used += 0 if region.near is None else region.near.receipt()['retained_bytes']
            if table_used > table_budget_bytes or near_used > near_budget_bytes:
                raise MemoryError('Cumulative multipole regional budget exceeded')
            return moments, region
        _, self.exterior = build(provider, global_tree, provider.exterior)
        projections, interiors, indices = [], [], []
        for body, geometry, tree in zip(provider.system.bodies, provider.geometry, body_trees):
            start, stop = body.edge_offset, body.edge_offset+body.basis.n_edges
            moments, region = build(body.basis, tree, body.interior, geometry)
            projections.append(moments)
            interiors.append(region)
            edges = torch.arange(start, stop, device=self.device)
            indices.append(torch.cat((edges, edges+provider.system.total_edges)))
        self.projections, self.interiors, self.indices = tuple(projections), tuple(interiors), tuple(indices)
        self.table_bytes, self.near_bytes, self.projection_bytes = table_used, near_used, projection_used
        if plan is not None and table_used != plan['table_bytes']:
            raise RuntimeError('FMM constructed tables differ from order plan')
        if storage_bytes(multipole_action_tensors(self)) != capacity['planned_bytes']:
            raise RuntimeError('Multipole action storage differs from structural capacity plan')
        self._binding = self._state()
        self._versions = tuple((id(t), t._version) for t in self.indices)

    def _state(self):
        return (id(self.provider), self.shape, self.device, self.dtype, id(self.exterior),
                tuple(map(id, self.projections)), tuple(map(id, self.interiors)),
                self.table_bytes, self.near_bytes, self.projection_bytes, self._order_plan_text, self._capacity_text, self.kernel_bound_rule, self.near_storage, self.far_action_batch, self.expansion_center_policy,
                self.near_coefficient_workspace_budget_bytes,self.near_coefficient_cache_budget_bytes)

    def assert_current(self):
        if self._state() != self._binding or tuple((id(t), t._version) for t in self.indices) != self._versions:
            raise ValueError('Multipole system changed')
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
        return {'version': 'global_exterior_scalar_multipole_system_v1',
                **({} if self.expansion_center_policy == 'support_box_midpoint_v1' else
                    dict(expansion_center_policy=self.expansion_center_policy)),
                **({} if self.far_action_batch == 1 else dict(far_action_batch=self.far_action_batch,
                    far_action_policy='bounded_compact_pairs_rhs4_rows16_v1')),
                **({} if self.near_storage == 'full_matrix' else {'near_storage': self.near_storage}),
                **({} if self.near_storage not in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1') else
                    {'near_coefficient_workspace_budget_bytes':self.near_coefficient_workspace_budget_bytes}),
                **({} if self.near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                    near_coefficient_cache_budget_bytes=self.near_coefficient_cache_budget_bytes,
                    near_coefficient_cache_scope='per_regional_owner_native_payload; aggregate retained admitted before numerical setup')),
                **({} if self.kernel_bound_rule == 'bivariate_cauchy_v1' else {'kernel_bound_rule': self.kernel_bound_rule}), 'shape': list(self.shape),
                'bodies': len(self.interiors), 'table_bytes': self.table_bytes, 'near_bytes': self.near_bytes,
                'projection_bytes': self.projection_bytes, 'order_plan': self.order_plan, 'capacity_plan': self.capacity_plan,
                'exterior': self.exterior.receipt(), 'interiors': [r.receipt() for r in self.interiors],
                'current_backward': 'fixed_operator_adjoint_no_saved_action_intermediates',
                'physical_qualification': False, 'complexity_qualification': False, 'peak_memory_qualified': False,
                'retained_budget_excludes': ['provider geometry and input system', 'Riesz metric',
                    'solver and setup/action/backward workspace', 'allocator and host metadata']}


class DirectionalSystemActions:
    """Prescribed complete exterior plus disjoint scalar interior PMCHWT actions."""
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    @classmethod
    def from_automatic(cls, provider, config):
        """Admit all physical regions and the global ideal bound before maps."""
        import math
        from .fmm_config import DirectionalFMMConfig
        from .fmm_order import AutomaticDirectionalRegionPlan, AutomaticDirectionalDAGPlan
        from .rwg_interpolation_error import directional_system_ideal_bound
        if type(config) is not DirectionalFMMConfig or not isinstance(provider, ScalarSystemBlocks):
            raise TypeError('Automatic directional system requires its scalar provider and configuration')
        provider.assert_current()
        if provider.system.dtype != torch.float64:
            raise ValueError('Automatic directional system requires float64 geometry')
        if 16*provider.system.total_edges > config.embedding_budget_bytes:
            raise MemoryError('Automatic directional embedding capacity exceeded before regional plans')
        count = len(provider.system.bodies)
        if count < 1:
            raise ValueError('Automatic directional system requires at least one closed body')
        eta = abs(provider.exterior.eta)
        # Reserve half for the exterior and half for the RSS of disjoint
        # interiors. Admission still checks the selected complete global bound.
        exterior_budget = config.global_ideal_rwg_error_budget/2
        interior_budget = config.global_ideal_rwg_error_budget/(2*math.sqrt(count))
        common = dict(kernel_error_budget=config.kernel_error_budget, balancing_impedance=eta,
            leaf_size=config.leaf_size, separation_ratio=config.separation_ratio,
            min_order=config.min_order, max_order=config.max_order,
            cone_subdivisions=config.cone_subdivisions, max_pairs=config.max_pairs,
            max_families=config.max_families, retained_budget_bytes=config.regional_plan_budget_bytes,
            tensor_workspace_budget_bytes=config.tensor_workspace_budget_bytes)
        roots = [AutomaticDirectionalRegionPlan(provider, provider.exterior,
            ideal_rwg_error_budget=exterior_budget, **common)]
        roots.extend(AutomaticDirectionalRegionPlan(body.basis, body.interior, geometry=geometry,
            ideal_rwg_error_budget=interior_budget, **common)
            for body, geometry in zip(provider.system.bodies, provider.geometry, strict=True))
        plans = [AutomaticDirectionalDAGPlan(root, quadrature_order=config.quadrature_order,
            face_batch=config.face_batch, rhs_batch=config.rhs_batch,
            phase_relative_budget=config.phase_relative_budget, max_candidates=config.max_candidates,
            max_contexts=config.max_contexts, retained_budget_bytes=config.regional_projection_budget_bytes,
            tensor_workspace_budget_bytes=config.tensor_workspace_budget_bytes) for root in roots]
        bounds = [plan.receipt()['ideal_bound'] for plan in plans]
        bound = directional_system_ideal_bound(bounds[0], bounds[1:],
            device=provider.system.device, balancing_impedance=eta)
        if bound['absolute_bound'] > config.global_ideal_rwg_error_budget:
            raise ValueError('Automatic directional global ideal admission failed before maps')
        regions = [plan.build_region(integration=provider.config,
            table_retained_budget_bytes=config.regional_table_budget_bytes,
            near_retained_budget_bytes=config.regional_near_budget_bytes,near_storage=config.near_storage,
            **({} if config.near_storage not in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1') else dict(
                near_coefficient_workspace_budget_bytes=config.near_coefficient_workspace_budget_bytes)),
            **({} if config.near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                near_coefficient_cache_budget_bytes=config.near_coefficient_cache_budget_bytes))) for plan in plans]
        return cls(provider, regions[0], regions[1:],
            ideal_rwg_error_budget=config.global_ideal_rwg_error_budget, balancing_impedance=eta,
            rhs_batch=config.rhs_batch, retained_budget_bytes=config.embedding_budget_bytes,
            tensor_workspace_budget_bytes=config.tensor_workspace_budget_bytes)

    def __init__(self,provider,exterior,interiors,*,ideal_rwg_error_budget,
                 balancing_impedance=None,rhs_batch=4,retained_budget_bytes=1024**2,
                 tensor_workspace_budget_bytes=1024**3):
        import math
        from .fmm_regional import DirectionalRegionalOperator
        from .rwg_interpolation_error import directional_system_ideal_bound
        if not isinstance(provider,ScalarSystemBlocks):
            raise TypeError('Directional system requires original scalar system blocks')
        provider.assert_current()
        if not isinstance(exterior,DirectionalRegionalOperator) or not isinstance(interiors,(tuple,list)):
            raise ValueError('Directional system requires explicit complete exterior and interior regions')
        exterior.assert_current()
        system = provider.system
        if (system.dtype!=torch.float64 or len(interiors)!=len(system.bodies)
                or exterior.test.basis is not provider or exterior.source.basis is not provider
                or exterior.medium is not provider.exterior or exterior.integration is not provider.config
                or exterior.shape!=(2*system.total_edges,)*2):
            raise ValueError('Directional exterior must bind the complete original provider and physical domain')
        eta = abs(provider.exterior.eta) if balancing_impedance is None else balancing_impedance
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (eta,ideal_rwg_error_budget)):
            raise ValueError('Directional global impedance and ideal absolute budget must be positive and finite')
        if any(type(v) is not int or v<1 for v in (rhs_batch,retained_budget_bytes,tensor_workspace_budget_bytes)):
            raise ValueError('Directional system batches and component budgets must be positive integers')
        offset = 0
        for body,region in zip(system.bodies,interiors,strict=True):
            if not isinstance(region,DirectionalRegionalOperator):
                raise TypeError('Directional interiors must be complete prescribed regional operators')
            region.assert_current()
            n = body.basis.n_edges
            if (type(body.edge_offset) is not int or body.edge_offset!=offset or n<1
                    or region.test.basis is not body.basis or region.source.basis is not body.basis
                    or region.medium is not body.interior or region.integration is not provider.config
                    or region.shape!=(2*n,)*2 or region.device!=exterior.device
                    or region.medium.frequency!=provider.exterior.frequency):
                raise ValueError('Directional interior must bind its disjoint actual body and medium')
            offset += n
        if offset!=system.total_edges or any(r.far.balancing_impedance!=eta for r in (exterior,*interiors)):
            raise ValueError('Directional regional bounds require complete bodies and one common global impedance')
        retained = 16*offset
        if retained>retained_budget_bytes:
            raise MemoryError('Directional system index budget exceeded before bound logs or indices')
        workspace = (4096+max(r.planned_tensor_workspace_bytes for r in (exterior,*interiors))
            +16*rhs_batch*32*offset+32*len(interiors))
        if workspace>tensor_workspace_budget_bytes:
            raise MemoryError('Directional system workspace exceeded before bound logs or indices')
        bounds = [r.ideal_rwg_truncation_bound() for r in (exterior,*interiors)]
        bound = directional_system_ideal_bound(bounds[0],bounds[1:],device=exterior.device,balancing_impedance=eta)
        bound.update(disjoint_actual_body_interiors_checked=True,
            complete_coefficient_pair_coverage=True,global_full_operator_certified=False,
            production_qualification=False)
        if bound['absolute_bound']>ideal_rwg_error_budget:
            raise ValueError('Directional exterior/interior ideal error exceeds global absolute budget')
        self.provider,self.exterior,self.interiors = provider,exterior,tuple(interiors)
        self.device,self.dtype,self.shape = exterior.device,torch.complex128,exterior.shape
        self.balancing_impedance,self.ideal_rwg_error_budget = eta,ideal_rwg_error_budget
        self.rhs_batch,self.retained_budget_bytes = rhs_batch,retained_budget_bytes
        self.tensor_workspace_budget_bytes,self.planned_tensor_workspace_bytes = tensor_workspace_budget_bytes,workspace
        self.planned_retained_bytes,self.retained_bytes = retained,retained
        self._ideal_bound = bound
        indices = []
        for body in system.bodies:
            edges = torch.arange(body.edge_offset,body.edge_offset+body.basis.n_edges,device=self.device)
            indices.append(torch.cat((edges,edges+offset)))
        self.indices = tuple(indices)
        if (sum(t.untyped_storage().nbytes() for t in self.indices)!=retained
                or any(t.untyped_storage().nbytes()!=t.numel()*t.element_size() for t in self.indices)):
            raise RuntimeError('Directional system embedding storage differs from its exact plan')
        self._binding = self._state()
        self.assert_current()

    def tensors(self):
        return self.indices

    def _state(self):
        return (id(self.provider),id(self.exterior),tuple(id(r) for r in self.interiors),
                tuple((b.edge_offset,b.basis.n_edges,id(b.basis),id(b.interior)) for b in self.provider.system.bodies),
                self.shape,self.device,self.dtype,self.rhs_batch,self.balancing_impedance,self.ideal_rwg_error_budget,
                self.retained_budget_bytes,self.tensor_workspace_budget_bytes,self.planned_tensor_workspace_bytes,
                self.planned_retained_bytes,self.retained_bytes,tuple(self._ideal_bound.items()),
                tuple((id(t),t._version) for t in self.indices))

    def assert_current(self):
        if self._state()!=self._binding:
            raise ValueError('Directional system regions, embeddings or admitted metadata changed')
        self.provider.assert_current();self.exterior.assert_current()
        for region in self.interiors:region.assert_current()

    def _action(self,value,*,adjoint):
        import math
        from .fmm_fft import _rhs_chunks
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=self.shape[0]
                or value.device!=self.device or value.dtype!=self.dtype):
            raise ValueError('Directional system requires matching physical complex128 currents')
        result = value.new_empty((math.prod(value.shape[:-1]),self.shape[0]))
        for start,chunk in _rhs_chunks(value,1,self.rhs_batch):
            batch = self.exterior.adjoint(chunk) if adjoint else self.exterior.apply(chunk)
            for indices,region in zip(self.indices,self.interiors,strict=True):
                selected = chunk[...,indices]
                local = region.adjoint(selected) if adjoint else region.apply(selected)
                batch.index_add_(-1,indices,local)
                del selected,local
            if not bool(torch.isfinite(batch).all()):
                raise ValueError('Directional system accumulation exceeds native precision')
            result[start:start+len(chunk)] = batch
            del batch
        self.assert_current()
        return result.reshape(value.shape)

    def ideal_rwg_truncation_bound(self):
        self.assert_current()
        return dict(self._ideal_bound)

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_prescribed_directional_scalar_system_v1',shape=self.shape,
            bodies=len(self.interiors),complete_coefficient_pair_coverage=True,
            ideal_rwg_error_budget=self.ideal_rwg_error_budget,balancing_impedance=self.balancing_impedance,
            ideal_rwg_truncation_admission=self.ideal_rwg_truncation_bound(),
            owned_embedding_bytes=self.retained_bytes,planned_owned_embedding_bytes=self.planned_retained_bytes,
            caller_regions_maps_couplings_near_trees_geometry_excluded=True,
            planned_action_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,rhs_batch=self.rhs_batch,
            workspace_excludes=('original near integration setup','caller input/output','allocator cache','backend FFT library'),
            current_backward='guarded_full_Hermitian_no_saved_internal_graph',
            quadrature_certified=False,floating_point_certified=False,physical_qualification=False,
            complexity_qualification=False,peak_memory_qualified=False,production_qualification=False)
