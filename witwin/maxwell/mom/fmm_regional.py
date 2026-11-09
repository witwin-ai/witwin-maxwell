"""True Cartesian M2L far interactions and original exact near integration."""
import torch
from dataclasses import asdict
from .fmm_moments import NestedRWGMultipoles
from .fmm_translation import ElectromagneticM2L
from .fmm_storage import m2l_storage_bytes, shared_displacement_catalog, signed_axis_displacement_catalog, signed_axis_catalog_rule, radial_series_pair_catalog, radial_series_pair_storage_bytes, radial_series_catalog_rule
from .fmm_error import validate_kernel_bound_rule
from .cluster import sphere_block_partition
from .nearfield import ScalarSystemBlocks, scalar_regional_block, scalar_regional_channels, scalar_block_channels, _validate_scalar_channel_medium
from .packed_near import PackedNearBlocks, PackedScalarNearChannels, PackedReciprocalScalarNearChannels, StreamedReciprocalScalarNearChannels, BoundedCachedReciprocalScalarNearChannels
from .operator import IntegrationConfig
from .physics import Medium
from .interpolation_error import KernelErrorBudget
from .direct_regional import DirectRegionalOperator


# Retained D/K near blocks are integrated in equal-shape batches. The per-block
# factory remains the reference path and is used when this flag is False.
BATCHED_NEAR_ASSEMBLY = True


class MultipoleRegionalOperator:
    """Fixed-order regional FMM action, without adaptive/complexity qualification."""
    _action = DirectRegionalOperator._action
    _exact_action = DirectRegionalOperator._exact_action
    apply = DirectRegionalOperator.apply
    adjoint = DirectRegionalOperator.adjoint

    def __init__(self, test, source, media, *, integration=IntegrationConfig(),
                 separation_ratio=.5, pair_batch=32, table_budget_bytes=512*1024**2,
                 near_budget_bytes=1024**3, kernel_error_budget=None, table_storage='dense', m2l_row_batch=16, near_action_batch=32,
                 kernel_bound_rule='bivariate_cauchy_v1', near_storage='full_matrix', far_action_batch=1,
                 near_coefficient_workspace_budget_bytes=128*1024**2,
                 near_coefficient_cache_budget_bytes=128*1024**2):
        if type(far_action_batch) is not int or not 1 <= far_action_batch <= 32:
            raise ValueError('Compact FAR action batch must lie between one and32')
        if far_action_batch > 1 and not isinstance(test,NestedRWGMultipoles):
            raise TypeError('Batched compact FAR requires native nested RWG spaces')
        if far_action_batch > 1 and (table_storage not in ('compact','compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1')
                or test.order > 12 or test.dtype != torch.complex128):
            raise ValueError('Batched compact FAR requires native complex128 compact order0..12')
        self.far_action_batch = far_action_batch
        validate_kernel_bound_rule(kernel_bound_rule)
        self.kernel_bound_rule = kernel_bound_rule
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
        if not isinstance(test, NestedRWGMultipoles) or not isinstance(source, NestedRWGMultipoles):
            raise TypeError('Regional FMM requires nested RWG multipoles')
        test.assert_current()
        source.assert_current()
        if test.device != source.device or test.dtype != source.dtype or test.order != source.order:
            raise ValueError('Regional multipoles must share order, device and precision')
        media = tuple(media)
        if far_action_batch > 1 and len(media) != 1:
            raise ValueError('Batched compact FAR requires one regional medium')
        if (not media or any(not isinstance(m, Medium) or m.frequency <= 0 or m.k == 0 for m in media)
                or any(m.frequency != media[0].frequency for m in media)):
            raise ValueError('Regional FMM requires finite-frequency scalar media at one frequency')
        if test.basis is not source.basis and len(media) != 1:
            raise ValueError('Cross-body blocks contain only the common exterior medium')
        global_space = isinstance(test.basis, ScalarSystemBlocks) or isinstance(source.basis, ScalarSystemBlocks)
        if global_space and (not isinstance(test.basis, ScalarSystemBlocks)
                             or not isinstance(source.basis, ScalarSystemBlocks) or len(media) != 1
                             or media[0] is not test.basis.exterior or media[0] is not source.basis.exterior
                             or integration is not test.basis.config or integration is not source.basis.config):
            raise ValueError('Global FMM region requires shared exterior medium and integration')
        if not isinstance(integration, IntegrationConfig):
            raise TypeError('Regional integration requires IntegrationConfig')
        if any(type(v) is not int or v < 1 for v in (table_budget_bytes, near_budget_bytes)):
            raise ValueError('Regional retained table/near budgets must be positive integers')
        if kernel_error_budget is not None and not isinstance(kernel_error_budget, KernelErrorBudget):
            raise TypeError('Regional FMM error budget must be KernelErrorBudget')
        if near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1'):
            if len(media) != 1 or test.dtype != torch.complex128:
                raise ValueError('Scalar D/K FMM requires one medium and native complex128')
            _validate_scalar_channel_medium(media[0])
            if near_storage in ('scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1') and (
                    test.basis is not source.basis or test.tree is not source.tree):
                raise ValueError('Reciprocal scalar FMM near storage requires identical physical spaces and trees')
        self.test, self.source, self.media, self.integration = test, source, media, integration
        self.device, self.dtype = test.device, test.dtype
        self.shape = (2*test.tree.size, 2*source.tree.size)
        per_pair = m2l_storage_bytes(test.order, test.centers.element_size(), table_storage)
        if type(m2l_row_batch) is not int or m2l_row_batch < 1:
            raise ValueError('M2L row batch must be a positive integer')
        if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1') and (
                test.basis is not source.basis or test.tree is not source.tree):
            raise ValueError('Shared compact region requires identical physical spaces and trees')
        if table_storage in ('compact_shared_signed_axis_v1','radial_series_shared_radius_v1') and (
                test.dtype != torch.complex128 or test.order > 12 or m2l_row_batch > 16):
            raise ValueError('Signed-axis regional M2L requires native complex128 order0..12 rows1..16')
        self.table_storage, self.m2l_row_batch = table_storage, m2l_row_batch
        shared_pairs=[]
        far, exact, table_bytes = [], [], 0
        for aa, bb, flags in sphere_block_partition(test.tree, source.tree,
                separation_ratio=separation_ratio, pair_batch=pair_batch):
            pairs = []
            for a, b, admissible in zip(aa.tolist(), bb.tolist(), flags.tolist()):
                if admissible:
                    pairs.append((a, b))
                else:
                    rows = test.tree.permutation[test.starts[a]:test.stops[a]]
                    cols = source.tree.permutation[source.starts[b]:source.stops[b]]
                    exact.append((a, b+test.tree.node_count, torch.cat((rows, rows+test.tree.size)),
                                  torch.cat((cols, cols+source.tree.size))+self.shape[0]))
            if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
                shared_pairs.extend(pairs)
                continue
            required = len(pairs)*len(media)*per_pair
            if required > table_budget_bytes-table_bytes:
                raise MemoryError('Cumulative regional M2L table budget exceeded')
            if pairs:
                batches = [ElectromagneticM2L.create_many(test, source, pairs, medium,
                    separation_ratio=separation_ratio, table_budget_bytes=table_budget_bytes-table_bytes,
                    error_budget=kernel_error_budget, table_storage=table_storage, row_batch=m2l_row_batch, kernel_bound_rule=kernel_bound_rule) for medium in media]
                for index, (a, b) in enumerate(pairs):
                    far.extend((a, b, batch[index]) for batch in batches)
                table_bytes += required
        if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            table_bytes=(radial_series_pair_storage_bytes(test.centers,shared_pairs,test.order)*len(media)
                if table_storage == 'radial_series_shared_radius_v1' else
                len((signed_axis_displacement_catalog if table_storage == 'compact_shared_signed_axis_v1'
                    else shared_displacement_catalog)(test.centers,shared_pairs)[0])*len(media)*per_pair)
            if table_bytes>table_budget_bytes:
                raise MemoryError('Shared compact regional M2L budget exceeded before coefficients')
            if table_storage == 'radial_series_shared_radius_v1' and shared_pairs:
                from .fmm_compact import radial_series_batch_plan
                radial_series_batch_plan(test.order,min(far_action_batch,len(shared_pairs)),4,m2l_row_batch)
            if table_storage == 'compact_shared_signed_axis_v1' and shared_pairs:
                import math
                from .fmm_compact import compact_batch_workspace_bytes
                pairs = min(far_action_batch,len(shared_pairs))
                jets = math.comb(2*test.order+4,3)
                required = compact_batch_workspace_bytes(test.order,pairs,4,m2l_row_batch)+16*pairs*jets+64*jets*3
                if required > 128*1024**2:
                    raise MemoryError('Signed-axis regional visible workspace exceeded before coefficients')
            batches=[ElectromagneticM2L.create_many(test,source,shared_pairs,medium,
                separation_ratio=separation_ratio,table_budget_bytes=table_budget_bytes,
                error_budget=kernel_error_budget,table_storage=table_storage,row_batch=m2l_row_batch,
                kernel_bound_rule=kernel_bound_rule) for medium in media]
            for index,(a,b) in enumerate(shared_pairs):
                far.extend((a,b,batch[index]) for batch in batches)
        self.far = tuple(far)
        def near_block(a, b, rows, cols):
            rows, cols = rows[:len(rows)//2], cols[:len(cols)//2]-self.shape[0]
            if near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1'):
                if global_space:
                    return test.basis.exterior_channels(rows, cols, source_provider=source.basis)
                d, k, _ = scalar_regional_channels(test.basis, source.basis, rows, cols,
                    media[0], integration, geometry=(test.geometry, source.geometry))
                return d, k
            if global_space:
                return (test.basis.exterior_block(rows, cols) if test.basis is source.basis else
                        test.basis.exterior_cross_block(source.basis, rows, cols))
            block, _ = scalar_regional_block(test.basis, source.basis, rows, cols,
                media, integration, geometry=(test.geometry, source.geometry))
            return block
        def near_blocks(entries):
            # Equal-shape D/K blocks in one call; triangle pairs shared between
            # neighbouring leaf blocks are classified and integrated once.
            rows = torch.stack([r[:len(r)//2] for _, _, r, _ in entries])
            cols = torch.stack([c[:len(c)//2] for _, _, _, c in entries])-self.shape[0]
            if global_space:
                d, k = test.basis.exterior_channels_many(rows, cols, source_provider=source.basis)
            else:
                d, k = scalar_block_channels(test.basis, source.basis, rows, cols, media[0], integration,
                                             geometry=(test.geometry, source.geometry))
            return d, k
        if BATCHED_NEAR_ASSEMBLY and near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1'):
            near_block.many = near_blocks
        if exact and near_storage in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            bindings = tuple((owner,owner.basis,owner.geometry,owner.tree) for owner in (test,source))
            def physical_guard():
                for owner,basis,geometry,tree in bindings:
                    if owner.basis is not basis or owner.geometry is not geometry or owner.tree is not tree:
                        raise ValueError('Streamed regional physical owner changed')
                    tree.assert_current()
                    if isinstance(basis,ScalarSystemBlocks):
                        basis.assert_current()
                    else:
                        geometry.assert_current(basis)
            physical_inputs = tuple(t for _,basis,geometry,_ in bindings for t in
                ((*basis.input_tensors(),*basis.geometry_tensors()) if isinstance(basis,ScalarSystemBlocks)
                 else (*geometry.inputs(),*geometry.tensors())))
            owner = (BoundedCachedReciprocalScalarNearChannels
                     if near_storage == 'scalar_reciprocal_cached_dk_v1'
                     else StreamedReciprocalScalarNearChannels)
            self.near = owner.from_layout(exact,*self.shape,near_block,
                source_node_offset=test.tree.node_count,medium=media[0],integration=integration,
                device=self.device,dtype=self.dtype,action_batch=near_action_batch,
                physical_guard=physical_guard,physical_inputs=physical_inputs,
                coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes,
                **({} if near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                    coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)),
                retained_budget_bytes=near_budget_bytes)
        elif exact and near_storage == 'scalar_reciprocal_dk_v1':
            self.near = PackedReciprocalScalarNearChannels.from_layout(exact,*self.shape,near_block,
                source_node_offset=test.tree.node_count,medium=media[0],integration=integration,
                device=self.device,dtype=self.dtype,action_batch=near_action_batch,
                retained_budget_bytes=near_budget_bytes)
        elif exact and near_storage == 'scalar_dk_v1':
            self.near = PackedScalarNearChannels.from_layout(exact, *self.shape, near_block,
                medium=media[0], integration=integration, device=self.device, dtype=self.dtype,
                action_batch=near_action_batch, retained_budget_bytes=near_budget_bytes)
        else:
            self.near = PackedNearBlocks.from_layout(exact, sum(self.shape), near_block,
                device=self.device, dtype=self.dtype, action_batch=near_action_batch,
                retained_budget_bytes=near_budget_bytes) if exact else None
        self.table_bytes, self.separation_ratio = table_bytes, separation_ratio
        self._binding = self._state()

    def _state(self):
        return (id(self.test), id(self.source), tuple(map(id, self.media)), self.integration,
                self.shape, self.device, self.dtype, id(self.near), self.table_bytes, self.separation_ratio,
                self.table_storage, self.m2l_row_batch, self.kernel_bound_rule, self.near_storage, self.far_action_batch,
                self.near_coefficient_workspace_budget_bytes,self.near_coefficient_cache_budget_bytes,
                tuple(asdict(self.integration).items()),
                tuple((a, b, id(k)) for a, b, k in self.far),
                tuple((m.frequency, m.epsilon_r, m.mu_r, m.k, m.eta) for m in self.media))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Regional multipole operator changed')
        self.test.assert_current()
        if self.source is not self.test:
            self.source.assert_current()
        if self.near is not None:
            self.near.assert_current()
        for _, _, coupling in self.far:
            coupling._assert_tables_current()

    def _action(self, value, *, adjoint):
        if self.far_action_batch == 1 and self.table_storage not in ('compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            return DirectRegionalOperator._action(self,value,adjoint=adjoint)
        self.assert_current()
        width = self.shape[0] if adjoint else self.shape[1]
        if (not isinstance(value,torch.Tensor) or value.ndim < 1 or value.shape[-1] != width
                or value.device != self.device or value.dtype != self.dtype):
            raise ValueError('Batched compact region requires matching CUDA current coefficients')
        from .fmm_fft import _rhs_chunks
        import math
        output_width = self.shape[1] if adjoint else self.shape[0]
        result = value.new_empty((math.prod(value.shape[:-1]),output_width))
        for start,chunk in _rhs_chunks(value,1,4):
            result[start:start+len(chunk)] = DirectRegionalOperator._action(self,chunk,adjoint=adjoint)
        self.assert_current()
        return result.reshape(*value.shape[:-1],output_width)

    def _accumulate_far(self, moments, coefficients, *, adjoint):
        # The regional action has validated both shared projections. Each
        # coupling still checks its own identity, medium and coefficient state.
        if self.far_action_batch == 1 and self.table_storage not in ('compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            for a,b,coupling in self.far:
                target,origin = (b,a) if adjoint else (a,b)
                coefficients[target] = coefficients[target]+coupling._action_with_current_spaces(
                    moments[origin],adjoint=adjoint)
            return
        from .fmm_compact import compact_moment_actions, signed_axis_compact_moment_actions, radial_series_moment_actions
        for start in range(0,len(self.far),self.far_action_batch):
            pairs = self.far[start:start+self.far_action_batch]
            for _,_,coupling in pairs:coupling._assert_tables_current()
            origins = [a if adjoint else b for a,b,_ in pairs]
            current = torch.stack([moments[node] for node in origins],dim=-4)
            action = (radial_series_moment_actions if self.table_storage == 'radial_series_shared_radius_v1'
                else signed_axis_compact_moment_actions if self.table_storage == 'compact_shared_signed_axis_v1'
                else compact_moment_actions)
            blocks = action([coupling._compact for _,_,coupling in pairs],current,adjoint=adjoint)
            # Keep every logical pair and its original ordered accumulation.
            for (a,b,_),block in zip(pairs,blocks.unbind(-4),strict=True):
                target = b if adjoint else a
                coefficients[target] = coefficients[target]+block
            del current,blocks

    def _local_coefficients(self, test, value):
        return [value.new_zeros((*value.shape[:-1], 2, test.count, 4)) for _ in range(test.tree.node_count)]

    def receipt(self):
        self.assert_current()
        return {'version': 'scalar_cartesian_multipole_regional_v1',
                **({} if self.table_storage != 'radial_series_shared_radius_v1' else dict(
                    coefficient_sharing_rule=radial_series_catalog_rule,physical_pairs_retained=True,
                    per_physical_displacement_bytes=24,full_cartesian_jets_retained=False,
                    radial_series_visible_tensor_budget_bytes=128*1024**2,production_qualification=False)),
                **({} if self.table_storage != 'compact_shared_signed_axis_v1' else dict(
                    coefficient_sharing_rule=signed_axis_catalog_rule,physical_pairs_retained=True,
                    signed_axis_visible_tensor_budget_bytes=128*1024**2,
                    temporary_actual_axis_jets_in_workspace=True,production_qualification=False)),
                **({} if self.far_action_batch == 1 else dict(far_action_batch=self.far_action_batch,
                    far_action_policy='bounded_compact_pairs_rhs4_rows16_v1',far_visible_tensor_budget_bytes=128*1024**2)),
                **({} if self.near_storage == 'full_matrix' else {'near_storage': self.near_storage}),
                **({} if self.kernel_bound_rule == 'bivariate_cauchy_v1' else {'kernel_bound_rule': self.kernel_bound_rule}), 'shape': list(self.shape),
                'order': self.test.order, 'far_regional_tables': len(self.far), 'table_bytes': self.table_bytes,
                'separation_ratio': self.separation_ratio,
                'near': None if self.near is None else self.near.receipt(),
                'near_construction': ('bounded_validated_original_first_use_cache' if self.near_storage == 'scalar_reciprocal_cached_dk_v1'
                                      else 'recompute_original_per_action' if self.near_storage == 'scalar_reciprocal_streamed_dk_v1'
                                      else 'stream_one_block_into_final_groups'),
                'table_storage': self.table_storage, 'm2l_row_batch': self.m2l_row_batch,
                'm2l_construction': (radial_series_catalog_rule if self.table_storage == 'radial_series_shared_radius_v1' else signed_axis_catalog_rule if self.table_storage == 'compact_shared_signed_axis_v1' else 'exact_native_equal_or_opposite_displacement_catalog' if self.table_storage == 'compact_shared_displacement_v1' else 'fft_taylor_coefficients_v1' if self.table_storage == 'fft_v1' else 'compact_taylor_coefficients' if self.table_storage == 'compact'
                                     else 'bounded_same_medium_displacement_batches'),
                'truncation_certified': False, 'physical_qualification': False,
                'complexity_qualification': False, 'peak_memory_qualified': False}


class DirectionalRegionalOperator:
    """Complete prescribed directional FAR plus original scalar D/K near blocks."""
    from .direct_system import DirectSystemActions as _LinearActions
    apply = _LinearActions.apply
    adjoint = _LinearActions.adjoint
    _exact_action = DirectRegionalOperator._exact_action

    def __init__(self, far, near_nodes, *, integration, near_retained_budget_bytes=64*1024**2,
                 near_action_batch=32, tensor_workspace_budget_bytes=512*1024**2,near_storage='scalar_dk_v1',
                 near_coefficient_workspace_budget_bytes=128*1024**2,
                     near_coefficient_cache_budget_bytes=128*1024**2):
        from .fmm_directional_rwg import (DirectionalNestedFarActions,
            DirectionalSparseFarActions, DirectionalDAGFarActions)
        if not isinstance(far, (DirectionalNestedFarActions, DirectionalSparseFarActions, DirectionalDAGFarActions)):
            raise TypeError('Directional region requires a prescribed disjoint FAR component')
        far.assert_current()
        if not isinstance(integration, IntegrationConfig) or not isinstance(near_nodes,(tuple,list)):
            raise ValueError('Directional region requires original integration and explicit near-node pairs')
        if any(type(v) is not int or v<1 for v in
               (near_retained_budget_bytes,near_action_batch,tensor_workspace_budget_bytes)):
            raise ValueError('Directional region near/action budgets and batch must be positive integers')
        test, source, medium = far.test, far.source, far.medium
        if near_storage not in ('scalar_dk_v1','scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            raise ValueError('Unsupported directional scalar near storage')
        if (type(near_coefficient_workspace_budget_bytes) is not int or near_coefficient_workspace_budget_bytes<1):
            raise ValueError('Directional near coefficient workspace must be a positive integer')
        streamed=near_storage in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1')
        if not streamed and near_coefficient_workspace_budget_bytes!=128*1024**2:
            raise ValueError('Explicit directional coefficient workspace requires streamed reciprocal storage')
        if (type(near_coefficient_cache_budget_bytes) is not int
                or near_coefficient_cache_budget_bytes < 0):
            raise ValueError('Directional near coefficient cache budget must be a nonnegative integer')
        if (near_storage != 'scalar_reciprocal_cached_dk_v1'
                and near_coefficient_cache_budget_bytes != 128*1024**2):
            raise ValueError('Explicit directional coefficient cache budget requires cached reciprocal storage')
        if near_storage in ('scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1') and (test.basis is not source.basis or test.tree is not source.tree):
            raise ValueError('Directional reciprocal near requires identical physical spaces and trees')
        global_space = isinstance(test.basis,ScalarSystemBlocks) or isinstance(source.basis,ScalarSystemBlocks)
        if global_space and any(not isinstance(p.basis,ScalarSystemBlocks)
                or p.basis.exterior is not medium or p.basis.config is not integration for p in (test,source)):
            raise ValueError('Directional global near blocks require common exterior and integration identities')
        _validate_scalar_channel_medium(medium)
        pairs, rectangles = [], []
        for pair in near_nodes:
            if not isinstance(pair,(tuple,list)) or len(pair)!=2:
                raise ValueError('Directional near nodes must be explicit test/source pairs')
            a,b = pair
            for projection,node in ((test,a),(source,b)):
                if type(node) is not int or not 0<=node<projection.tree.node_count:
                    raise ValueError('Directional near node is outside its complete tree')
                if bool((projection.tree.children[node]>=0).any()):
                    raise ValueError('Directional original near blocks require terminal physical leaf pairs')
            pairs.append((a,b))
            rectangles.append((int(test.tree.starts[a]),int(test.tree.stops[a]),
                               int(source.tree.starts[b]),int(source.tree.stops[b]),a,b))
        joined = (*far.rectangles,*rectangles)
        DirectionalNestedFarActions._validate_disjoint_rectangles(joined)
        nt,ns = test.tree.size,source.tree.size
        if (any(stop>nt or high>ns for _,stop,_,high,_,_ in joined)
                or sum((stop-start)*(high-low) for start,stop,low,high,_,_ in joined)!=nt*ns):
            raise ValueError('Directional region partition does not cover every physical edge pair exactly once')
        # The packed D/K planner's exact native numeric, map and CSR inventory.
        retained = (8*(2*len({a for a,_ in pairs})+1+3*len(pairs))
            +sum(32*(stop-start)*(high-low)+16*((stop-start)+(high-low))
                 for start,stop,low,high,_,_ in rectangles)) if pairs else 0
        if near_storage in ('scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            from .fmm_capacity import plan_directional_reciprocal_near_storage
            retained=plan_directional_reciprocal_near_storage(test.tree,pairs,
                budget_bytes=near_retained_budget_bytes,
                **({} if not streamed else dict(near_storage=near_storage,
                    near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes)),
                **({} if near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                    near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)))['retained_bytes']
        if retained>near_retained_budget_bytes:
            raise MemoryError('Directional near retained budget exceeded before layout or original integration')
        max_rows = max((stop-start for start,stop,_,_,_,_ in rectangles),default=0)
        max_cols = max((high-low for _,_,low,high,_,_ in rectangles),default=0)
        rhs = far.rhs_batch
        near_workspace = (4096+16*rhs*16*(nt+ns)
            +16*min(len(pairs),near_action_batch)*rhs*(16*(max_rows+max_cols)+16))
        if near_storage in ('scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            near_workspace+=16*min(sum(a<=b for a,b in pairs),near_action_batch)*(max_rows+max_cols)
        if streamed:
            near_workspace+=min(near_coefficient_workspace_budget_bytes,near_action_batch*3*2*max_rows*max_cols*16)
        workspace = far.planned_tensor_workspace_bytes+near_workspace+16*rhs*16*(nt+ns)
        if workspace>tensor_workspace_budget_bytes:
            raise MemoryError('Directional region workspace exceeded before near layout or current actions')
        self.near_storage=near_storage
        self.near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes
        self.near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes
        self.far,self.test,self.source,self.medium = far,test,source,medium
        self.integration,self.near_nodes,self.rectangles = integration,tuple(pairs),tuple(rectangles)
        self.shape,self.device,self.dtype = far.shape,far.device,far.dtype
        self.rhs_batch,self.near_action_batch = rhs,near_action_batch
        self.near_retained_budget_bytes = near_retained_budget_bytes
        self.tensor_workspace_budget_bytes = tensor_workspace_budget_bytes
        self.planned_tensor_workspace_bytes = workspace
        self.planned_retained_bytes = retained
        layout = []
        for a,b in pairs:
            rows = test.tree.permutation[int(test.tree.starts[a]):int(test.tree.stops[a])]
            cols = source.tree.permutation[int(source.tree.starts[b]):int(source.tree.stops[b])]
            layout.append((a,b+test.tree.node_count,torch.cat((rows,rows+nt)),
                torch.cat((cols,cols+ns))+self.shape[0]))
        def original_channels(a,b,rows,cols):
            rows = rows[:len(rows)//2];cols = cols[:len(cols)//2]-self.shape[0]
            if global_space:
                return test.basis.exterior_channels(rows,cols,source_provider=source.basis)
            d,k,_ = scalar_regional_channels(test.basis,source.basis,rows,cols,medium,
                integration,geometry=(test.geometry,source.geometry))
            return d,k
        if streamed:
            bindings=tuple((owner,owner.basis,owner.geometry,owner.tree) for owner in (test,source))
            def physical_guard():
                for owner,basis,geometry,tree in bindings:
                    if owner.basis is not basis or owner.geometry is not geometry or owner.tree is not tree:
                        raise ValueError('Streamed directional physical owner changed')
                    tree.assert_current()
                    if isinstance(basis,ScalarSystemBlocks):
                        basis.assert_current()
                    else:
                        geometry.assert_current(basis)
            physical_inputs=tuple(t for _,basis,geometry,_ in bindings for t in
                ((*basis.input_tensors(),*basis.geometry_tensors()) if isinstance(basis,ScalarSystemBlocks)
                 else (*geometry.inputs(),*geometry.tensors())))
            owner = (BoundedCachedReciprocalScalarNearChannels
                     if near_storage == 'scalar_reciprocal_cached_dk_v1'
                     else StreamedReciprocalScalarNearChannels)
            self.near=owner.from_layout(layout,*self.shape,original_channels,
                source_node_offset=test.tree.node_count,medium=medium,integration=integration,
                device=self.device,dtype=self.dtype,action_batch=near_action_batch,
                physical_guard=physical_guard,physical_inputs=physical_inputs,
                coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes,
                **({} if near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                    coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)),
                retained_budget_bytes=near_retained_budget_bytes) if pairs else None
        elif near_storage=='scalar_reciprocal_dk_v1':
            self.near=PackedReciprocalScalarNearChannels.from_layout(layout,*self.shape,original_channels,
                source_node_offset=test.tree.node_count,medium=medium,integration=integration,
                device=self.device,dtype=self.dtype,action_batch=near_action_batch,
                retained_budget_bytes=near_retained_budget_bytes) if pairs else None
        else:
            self.near = PackedScalarNearChannels.from_layout(layout,*self.shape,original_channels,
            medium=medium,integration=integration,device=self.device,dtype=self.dtype,
            action_batch=near_action_batch,retained_budget_bytes=near_retained_budget_bytes) if pairs else None
        tensors = self.tensors()
        inventory=tuple(t for t in tensors if t.numel() or t.untyped_storage().nbytes()) if streamed else tensors
        if (len({(t.device,t.untyped_storage().data_ptr()) for t in inventory})!=len(inventory)
                or any(t.untyped_storage().nbytes()!=t.numel()*t.element_size() for t in tensors)):
            raise RuntimeError('Directional near inventory must own distinct compact native storages')
        self.retained_bytes = sum(t.untyped_storage().nbytes() for t in tensors)
        if self.retained_bytes!=retained:
            raise RuntimeError('Directional near storage differs from its exact preallocation plan')
        self._binding = self._state()
        self.assert_current()

    def tensors(self):
        return () if self.near is None else self.near.tensors()

    def _state(self):
        return (id(self.far),id(self.test),id(self.source),id(self.medium),id(self.integration),
                tuple(asdict(self.integration).items()),self.near_nodes,self.rectangles,id(self.near),
                self.shape,self.device,self.dtype,self.rhs_batch,self.near_action_batch,
                self.near_retained_budget_bytes,self.tensor_workspace_budget_bytes,
                self.planned_tensor_workspace_bytes,self.planned_retained_bytes,self.retained_bytes,self.near_storage,
                self.near_coefficient_workspace_budget_bytes,self.near_coefficient_cache_budget_bytes)

    def assert_current(self):
        if self._state()!=self._binding:
            raise ValueError('Directional region partition, integration or budgets changed')
        self.far.assert_current()
        if self.near is not None:self.near.assert_current()

    def _action(self,value,*,adjoint):
        import math
        from .fmm_fft import _rhs_chunks
        self.assert_current()
        ni,no = (self.shape[0],self.shape[1]) if adjoint else (self.shape[1],self.shape[0])
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=ni
                or value.device!=self.device or value.dtype!=self.dtype):
            raise ValueError('Directional region requires matching physical complex128 currents')
        result = value.new_empty((math.prod(value.shape[:-1]),no))
        for start,chunk in _rhs_chunks(value,1,self.rhs_batch):
            batch = self.far.adjoint(chunk) if adjoint else self.far.apply(chunk)
            exact = self._exact_action(chunk,adjoint=adjoint)
            batch = batch+exact
            if not bool(torch.isfinite(batch).all()):
                raise ValueError('Directional regional accumulation exceeds native precision')
            result[start:start+len(chunk)] = batch
            del batch,exact
        self.assert_current()
        return result.reshape(*value.shape[:-1],no)

    def ideal_rwg_truncation_bound(self):
        self.assert_current()
        record = self.far.ideal_rwg_truncation_bound()
        record.update(version='directional_complete_region_ideal_frobenius_v1',
            far_component_only=False,near_entries_zero=False,complete_coefficient_pair_coverage=True,
            ideal_exact_integration_near_error_zero=True,finite_near_integration_certified=False,
            global_full_operator_certified=False,production_qualification=False)
        return record

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_complete_directional_region_v1',shape=self.shape,
            **({} if self.near_storage=='scalar_dk_v1' else dict(near_storage=self.near_storage)),
            **({} if self.near_storage!='scalar_reciprocal_streamed_dk_v1' else dict(
                near_coefficient_workspace_budget_bytes=self.near_coefficient_workspace_budget_bytes,
                near_construction='recompute_original_per_action',retained_near_kernel_bytes=0,
                streamed_coefficient_chunks_in_planned_action_workspace=True,
                original_quadrature_workspace_qualified=False)),
            **({} if self.near_storage!='scalar_reciprocal_cached_dk_v1' else dict(
                near_coefficient_workspace_budget_bytes=self.near_coefficient_workspace_budget_bytes,
                near_coefficient_cache_budget_bytes=self.near_coefficient_cache_budget_bytes,
                near_construction='bounded_validated_original_first_use_cache',
                retained_near_kernel_bytes=0 if self.near is None else self.near.receipt()['retained_kernel_bytes'],
                cache_scope='per_regional_owner_native_payload; host/events/quadrature excluded',
                streamed_coefficient_chunks_in_planned_action_workspace=True,
                original_quadrature_workspace_qualified=False)),
            near_node_index_space='physical_support_tree',
            near_blocks=len(self.near_nodes),near=None if self.near is None else self.near.receipt(),
            far=self.far.receipt(),complete_coefficient_pair_coverage=True,
            ideal_rwg_truncation_admission=self.ideal_rwg_truncation_bound(),
            owned_retained_bytes=self.retained_bytes,planned_retained_bytes=self.planned_retained_bytes,
            caller_far_maps_couplings_and_trees_excluded=True,rhs_batch=self.rhs_batch,
            planned_action_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,
            workspace_excludes=('near integration setup','caller input/output','allocator cache','backend FFT library'),
            current_backward='guarded_full_Hermitian_no_saved_internal_graph',
            quadrature_certified=False,floating_point_certified=False,physical_qualification=False,
            complexity_qualification=False,peak_memory_qualified=False,production_qualification=False)
