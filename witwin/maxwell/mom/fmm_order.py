"""Plan uniform Cartesian FMM order before moment or coefficient allocation."""
from dataclasses import asdict
import math
import torch
from .cluster import sphere_block_partition, support_spheres
from .fmm_storage import m2l_storage_bytes, shared_displacement_catalog, signed_axis_displacement_catalog, signed_axis_catalog_rule, radial_series_pair_catalog, radial_series_pair_storage_bytes, radial_series_catalog_rule
from .fmm_error import multipole_truncation_bounds, multipole_truncation_bounds_many, validate_kernel_bound_rule
from .fmm_rwg_error import multipole_block_error_log, multipole_error_receipt
from .interpolation_error import KernelErrorBudget


class MultipoleOrderError(ValueError):
    def __init__(self, receipt):
        super().__init__('No Cartesian FMM order satisfies the declared kernel/table budgets')
        self.receipt = receipt



def _kernel_order_chunk(regions, orders, separation_ratio, table_storage, kernel_bound_rule):
    """Reduce bounded CUDA contours in one geometric traversal per region.

    No far-pair list is retained. Contour workspace has at most32pairs and16orders;
    Accumulated maxima/counters transfer after each regional traversal. Input
    validation and partition control still synchronize with the host.
    """
    attempts = [dict(order=order, table_bytes=0, far_pairs=0, unavailable_bounds=0,
        green_absolute_max=0., gradient_component_absolute_max=0.,
        rwg_pass=True, rwg_bound=None) for order in orders]
    for tree, medium in regions:
        tree.assert_current()
        centers,radii = support_spheres(tree)
        green = torch.zeros(len(orders), device=tree.device, dtype=torch.float64)
        gradient = torch.zeros_like(green)
        unavailable = torch.zeros(len(orders), device=tree.device, dtype=torch.int64)
        far_pairs = 0
        shared_pairs=[]
        for aa, bb, flags in sphere_block_partition(tree, separation_ratio=separation_ratio, pair_batch=32):
            a, b = aa[flags], bb[flags]
            if not a.numel():
                continue
            far_pairs += a.numel()
            if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
                shared_pairs.extend(zip(a.cpu().tolist(),b.cpu().tolist()))
            bounds = multipole_truncation_bounds_many(centers[a]-centers[b], radii[a], radii[b], medium.k, orders, kernel_bound_rule=kernel_bound_rule)
            unavailable += (~bounds['available']).sum(0)
            green = torch.maximum(green, bounds['green_absolute'].amax(0))
            gradient = torch.maximum(gradient, bounds['gradient_component_absolute'].amax(0))
        tree.assert_current()
        stored_pairs=far_pairs
        if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            stored_pairs=len((radial_series_pair_catalog if table_storage == 'radial_series_shared_radius_v1'
                else signed_axis_displacement_catalog if table_storage == 'compact_shared_signed_axis_v1'
                else shared_displacement_catalog)(centers,shared_pairs)[0])
        # These transfers form the public host receipt, not numerical fallback.
        for attempt, g, h, missing in zip(attempts, green.tolist(), gradient.tolist(), unavailable.tolist(), strict=True):
            attempt['table_bytes'] += stored_pairs*m2l_storage_bytes(attempt['order'], tree.lower.element_size(), table_storage)
            if table_storage == 'radial_series_shared_radius_v1':
                attempt['table_bytes'] += 24*far_pairs
            attempt['far_pairs'] += far_pairs
            attempt['unavailable_bounds'] += missing
            attempt['green_absolute_max'] = max(attempt['green_absolute_max'], g)
            attempt['gradient_component_absolute_max'] = max(attempt['gradient_component_absolute_max'], h)
    return attempts


def plan_multipole_order(regions, budget, *, min_order=0, max_order=12,
                         separation_ratio=.5, table_budget_bytes=512*1024**2,
                         rwg_statistics=None, rwg_absolute_budget=None, balancing_impedance=None, table_storage='dense',
                         kernel_bound_rule='bivariate_cauchy_v1'):
    validate_kernel_bound_rule(kernel_bound_rule)
    policy = {} if kernel_bound_rule == 'bivariate_cauchy_v1' else {'kernel_bound_rule': kernel_bound_rule}
    if table_storage == 'compact_shared_signed_axis_v1':
        policy['coefficient_sharing_rule'] = signed_axis_catalog_rule
    if table_storage == 'radial_series_shared_radius_v1':
        policy.update(coefficient_sharing_rule=radial_series_catalog_rule,per_physical_displacement_bytes=24)
    # RWG aggregation assumes region zero is the common exterior and all
    # remaining regions have disjoint interior coefficient spaces. The system
    # constructor supplies this decomposition; arbitrary overlapping regions
    # do not satisfy this planner's RWG contract.
    if not isinstance(budget, KernelErrorBudget):
        raise TypeError('Automatic FMM order requires KernelErrorBudget')
    if any(type(p) is not int or p < 0 for p in (min_order, max_order)) or max_order < min_order:
        raise ValueError('FMM order range must be nonnegative and ordered')
    if type(table_budget_bytes) is not int or table_budget_bytes < 1:
        raise ValueError('FMM table budget must be a positive integer')
    m2l_storage_bytes(min_order, 8, table_storage)
    if table_storage in ('fft_v1','radial_series_shared_radius_v1'):
        m2l_storage_bytes(max_order, 8, table_storage)
    regions = tuple(regions)
    if not regions:
        raise ValueError('FMM planning requires regional trees and media')
    if table_storage == 'fft_v1':
        for tree, _ in regions:
            m2l_storage_bytes(min_order, tree.lower.element_size(), table_storage)
    if rwg_absolute_budget is not None:
        if type(rwg_absolute_budget) not in (int, float) or not math.isfinite(rwg_absolute_budget) or rwg_absolute_budget <= 0:
            raise ValueError('RWG absolute budget must be finite and positive')
        if type(balancing_impedance) not in (int, float) or not math.isfinite(balancing_impedance) or balancing_impedance <= 0:
            raise ValueError('RWG balancing impedance must be finite and positive')
        rwg_statistics = tuple(rwg_statistics) if rwg_statistics is not None else ()
        if len(rwg_statistics) != len(regions):
            raise ValueError('RWG statistics require exterior then disjoint interior regions')
        for statistics, (tree, _) in zip(rwg_statistics, regions):
            if (not isinstance(statistics, torch.Tensor) or statistics.shape != (tree.node_count, 3)
                    or statistics.device != tree.device or statistics.dtype != torch.float64 or statistics.requires_grad
                    or not bool(torch.isfinite(statistics).all())):
                raise ValueError('RWG statistics must be finite fixed matching node log sums')
        versions = tuple((id(t), t._version) for t in rwg_statistics)
    elif rwg_statistics is not None or balancing_impedance is not None:
        raise ValueError('RWG statistics require an absolute budget')
    attempts = []
    def fail(reason):
        raise MultipoleOrderError({'version': 'cartesian_fmm_order_plan_v1', **policy, 'table_storage': table_storage, 'reason': reason,
            'attempts': attempts, 'kernel_budget': asdict(budget), 'table_budget_bytes': table_budget_bytes,
            'rwg_absolute_budget': rwg_absolute_budget, 'balancing_impedance': balancing_impedance})
    if rwg_absolute_budget is None:
        # The optional RWG path below retains its ordered scalar accumulation.
        # Plain kernel plans share geometry and contours across bounded orders.
        for first in range(min_order, max_order+1, 16):
            orders = tuple(range(first, min(max_order+1, first+16)))
            chunk = _kernel_order_chunk(regions, orders, separation_ratio, table_storage, kernel_bound_rule)
            for attempt in chunk:
                passed = (attempt['unavailable_bounds'] == 0
                    and attempt['green_absolute_max'] <= budget.green_absolute
                    and attempt['gradient_component_absolute_max'] <= budget.gradient_component_absolute)
                attempt['kernel_pass'] = passed
                attempts.append(attempt)
                if attempt['table_bytes'] > table_budget_bytes:
                    fail('table_budget')
                if passed:
                    return {'version': 'cartesian_fmm_order_plan_v1', **policy, 'table_storage': table_storage,
                        'selected_order': attempt['order'], 'table_bytes': attempt['table_bytes'],
                        'kernel_budget': asdict(budget), 'attempts': attempts, 'rwg_absolute_budget': None,
                        'rwg_bound': None, 'scope': 'ideal_exact_arithmetic_kernel_truncation_only',
                        'implemented_operator_certified': False, 'physical_qualification': False}
        fail('maximum_order_or_unavailable_bound')
    for order in range(min_order, max_order+1):
        count = math.comb(order+3, 3)
        total, far_count, unavailable = 0, 0, 0
        green, gradient = 0., 0.
        region_logs = []
        for region_index, (tree, medium) in enumerate(regions):
            shared_pairs=[]
            region_log_squared = tree.lower.new_tensor(-torch.inf, dtype=torch.float64)
            tree.assert_current()
            centers,radii = support_spheres(tree)
            for aa, bb, flags in sphere_block_partition(tree, separation_ratio=separation_ratio, pair_batch=32):
                a, b = aa[flags], bb[flags]
                if not a.numel():
                    continue
                far_count += a.numel()
                if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
                    shared_pairs.extend(zip(a.cpu().tolist(),b.cpu().tolist()))
                else:
                    total += a.numel()*m2l_storage_bytes(order, tree.lower.element_size(), table_storage)
                bounds = multipole_truncation_bounds(centers[a]-centers[b], radii[a], radii[b], medium.k, order, kernel_bound_rule=kernel_bound_rule)
                available = bounds['available']
                unavailable += int((~available).sum())
                green = max(green, float(bounds['green_absolute'].amax()))
                gradient = max(gradient, float(bounds['gradient_component_absolute'].amax()))
                if rwg_absolute_budget is not None:
                    # Preserve the existing ordered RWG log accumulation and
                    # scalar material factors. Only kernel evaluation batches.
                    statistics = rwg_statistics[region_index]
                    for index in available.nonzero().flatten().tolist():
                        bound = {name: float(values[index]) for name, values in bounds.items() if name != 'available'}
                        block_log = multipole_block_error_log(statistics[a[index]], statistics[b[index]], medium, bound, balancing_impedance)
                        region_log_squared = torch.logaddexp(region_log_squared, 2*block_log)
            tree.assert_current()
            if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
                if table_storage == 'radial_series_shared_radius_v1':
                    total+=radial_series_pair_storage_bytes(centers,shared_pairs,order)
                else:
                    total+=len((signed_axis_displacement_catalog if table_storage == 'compact_shared_signed_axis_v1'
                        else shared_displacement_catalog)(centers,shared_pairs)[0])*m2l_storage_bytes(order,tree.lower.element_size(),table_storage)
            region_logs.append(.5*region_log_squared)
        rwg_receipt, rwg_pass = None, True
        if rwg_absolute_budget is not None:
            if tuple((id(t), t._version) for t in rwg_statistics) != versions:
                raise ValueError("RWG statistics changed during order planning")
            logs = torch.stack(region_logs)
            total_log = torch.logaddexp(logs[0], .5*torch.logsumexp(2*logs[1:], dim=0))
            rwg_pass = False
            try:
                if unavailable:
                    raise ValueError('Incomplete regional kernel bounds')
                rwg_receipt = multipole_error_receipt(total_log, balancing_impedance)
                rwg_pass = unavailable == 0 and rwg_receipt["absolute_bound"] <= rwg_absolute_budget
            except ValueError:
                rwg_pass = False
        passed = unavailable == 0 and green <= budget.green_absolute and gradient <= budget.gradient_component_absolute
        attempts.append({'order': order, 'table_bytes': total, 'far_pairs': far_count,
                         'unavailable_bounds': unavailable, 'green_absolute_max': green,
                         'gradient_component_absolute_max': gradient, 'kernel_pass': passed,
                         'rwg_pass': rwg_pass, 'rwg_bound': rwg_receipt})
        if total > table_budget_bytes:
            fail('table_budget')
        if passed and rwg_pass:
            return {'version': 'cartesian_fmm_order_plan_v1', **policy, 'table_storage': table_storage, 'selected_order': order,
                    'table_bytes': total, 'kernel_budget': asdict(budget), 'attempts': attempts,
                    'rwg_absolute_budget': rwg_absolute_budget, 'rwg_bound': rwg_receipt,
                    'scope': 'ideal_kernel_and_exactly_integrated_rwg' if rwg_absolute_budget is not None else 'ideal_exact_arithmetic_kernel_truncation_only',
                    'implemented_operator_certified': False, 'physical_qualification': False}
    fail('maximum_order_or_unavailable_bound')


from .fmm_error import (_bound_inputs, directional_bound_workspace_bytes,
    directional_multipole_truncation_bounds_many)


_DIRECTIONAL_PLAN_TENSORS = ('displacements', 'test_radii', 'source_radii', 'directions',
    'cone_keys', 'orders', 'normalization_lengths', 'green_bounds', 'gradient_bounds',
    'green_log_bounds', 'gradient_log_bounds')


class DirectionalConeOrderError(ValueError):
    """No executable plan is published when even one declared pair fails."""
    def __init__(self, receipt):
        super().__init__('No directional cone/order satisfies every declared pair')
        self.receipt = receipt


def _cube_carriers(displacements, subdivisions):
    """Quantize dominant-axis cube faces without allocating a cone catalog.

    Axis ties choose the first axis. Face coordinates round halfway upward,
    using floor(x + 0.5); keys are (subdivisions, signed face, u index, v index).
    The two remaining axes retain their increasing Cartesian order.
    """
    axis = displacements.abs().argmax(-1)
    dominant = displacements.gather(1, axis[:, None]).squeeze(1)
    sign = torch.where(dominant < 0, -torch.ones_like(dominant), torch.ones_like(dominant))
    all_axes = torch.arange(3, device=displacements.device).expand(displacements.shape[0], 3)
    other = all_axes[all_axes != axis[:, None]].reshape(-1, 2)
    ratio = displacements.gather(1, other)/dominant.abs()[:, None]
    indices = torch.floor((ratio+1)*(subdivisions/2)+.5).to(torch.int64).clamp(0, subdivisions)
    cube = torch.zeros_like(displacements)
    cube.scatter_(1, axis[:, None], sign[:, None])
    cube.scatter_(1, other, indices.to(displacements.dtype)*(2/subdivisions)-1)
    directions = cube/torch.linalg.vector_norm(cube, dim=-1, keepdim=True)
    keys = torch.cat((torch.full_like(axis[:, None], subdivisions),
        (2*axis+(dominant < 0).to(torch.int64))[:, None], indices), dim=1)
    return directions, keys


class DirectionalConeOrderPlan:
    """Bounded native kernel-only admission for at most 32 declared pairs.

    Select minimum passing order, breaking ties by the coarsest configured
    cube grid. All pairs pass before any translation coefficients are allocated.
    Declared radii do not establish physical RWG support, quadrature, phase-path,
    native error or global system admission. This is not a full-tree cone planner.
    """
    def __init__(self, displacements, test_radii, source_radii, wave_number, kernel_error_budget,
                 *, min_order=0, max_order=12, cone_subdivisions=(1, 2, 4, 8, 16, 32, 64),
                 retained_budget_bytes=1024**2, tensor_workspace_budget_bytes=128*1024**2):
        if not isinstance(kernel_error_budget, KernelErrorBudget):
            raise TypeError('Directional planning requires KernelErrorBudget')
        KernelErrorBudget(**asdict(kernel_error_budget))
        if (type(min_order) is not int or type(max_order) is not int
                or not 0 <= min_order <= max_order <= 12):
            raise ValueError('Directional order range must lie in 0..12')
        grids = tuple(cone_subdivisions)
        if (not grids or any(type(m) is not int or not 1 <= m <= 2**20 for m in grids)
                or any(a >= b for a, b in zip(grids, grids[1:]))):
            raise ValueError('Cone grids must be strictly increasing positive bounded integers')
        if any(type(b) is not int or b < 1 for b in (retained_budget_bytes, tensor_workspace_budget_bytes)):
            raise ValueError('Directional capacities must be positive integers')
        candidates = tuple(range(min_order, max_order+1))
        wave, distances = _bound_inputs(displacements, test_radii, source_radii, wave_number, candidates)
        count = displacements.shape[0]
        if displacements.dtype != torch.float64 or count > 32:
            raise ValueError('Directional planning requires CUDA float64 and at most 32 pairs')
        retained = 144*count
        # Bounds include their contour tensors. This additional conservative
        # envelope covers quantization, selection, returned fields and cloning;
        # it excludes allocator/library state and caller tensors, not peak use.
        workspace = directional_bound_workspace_bytes(count, len(candidates))+4096+count*len(candidates)*1024
        if retained > retained_budget_bytes:
            raise MemoryError('Directional plan retained capacity exceeded before contour allocation')
        if workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Directional plan workspace capacity exceeded before contour allocation')
        inputs = (displacements, test_radii, source_radii)
        input_state = tuple((id(t), t._version) for t in inputs)
        orders = torch.full((count,), max_order+1, dtype=torch.int64, device=displacements.device)
        directions = torch.zeros_like(displacements)
        keys = torch.zeros((count, 4), device=displacements.device, dtype=torch.int64)
        selected = [torch.zeros(count, device=displacements.device, dtype=torch.float64) for _ in range(4)]
        attempts = []
        if count:
            candidate_tensor = orders.new_tensor(candidates)
            for grid in grids:
                carrier, cone_key = _cube_carriers(displacements, grid)
                bounds = directional_multipole_truncation_bounds_many(displacements, carrier,
                    test_radii, source_radii, wave, candidates,
                    tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
                passed = (bounds['available']
                    & (bounds['green_absolute'] <= kernel_error_budget.green_absolute)
                    & (bounds['gradient_component_absolute'] <= kernel_error_budget.gradient_component_absolute))
                choices = torch.where(passed, candidate_tensor[None, :], max_order+1)
                best, column = choices.min(-1)
                improve = best < orders
                orders = torch.where(improve, best, orders)
                directions = torch.where(improve[:, None], carrier, directions)
                keys = torch.where(improve[:, None], cone_key, keys)
                for index, name in enumerate(('green_absolute', 'gradient_component_absolute',
                        'green_log_absolute', 'gradient_component_log_absolute')):
                    value = bounds[name].gather(1, column[:, None]).squeeze(1)
                    selected[index] = torch.where(improve, value, selected[index])
                # Only admission counts and integer policy leave CUDA here.
                attempts.append((grid, int((~bounds['available']).sum()), int(passed.any(-1).sum())))
        if tuple((id(t), t._version) for t in inputs) != input_state:
            raise ValueError('Declared geometry changed during directional planning')
        missing = orders > max_order
        if bool(missing.any()):
            raise DirectionalConeOrderError({'version': 'directional_cone_order_plan_v1',
                'reason': 'maximum_order_or_unavailable_bound', 'unadmitted_rows': missing.nonzero().flatten().tolist(),
                'attempts': attempts, 'physical_qualification': False})
        distance_scale = distances/(2*orders+2)
        support_scale = torch.maximum(test_radii, source_radii)
        normalization = torch.where(support_scale > 0,
            torch.minimum(distance_scale, support_scale), distance_scale)
        if not bool(torch.isfinite(normalization).all() & (normalization > 0).all()):
            raise ValueError('Selected distance/order normalization is not representable')
        self.displacements = displacements.clone()
        self.test_radii, self.source_radii = test_radii.clone(), source_radii.clone()
        self.directions, self.cone_keys, self.orders = directions, keys, orders
        self.normalization_lengths = normalization
        self.green_bounds, self.gradient_bounds, self.green_log_bounds, self.gradient_log_bounds = selected
        self.wave_number, self.kernel_error_budget = wave, kernel_error_budget
        self.min_order, self.max_order, self.cone_subdivisions = min_order, max_order, grids
        self.retained_budget_bytes, self.tensor_workspace_budget_bytes = retained_budget_bytes, tensor_workspace_budget_bytes
        self.retained_bytes, self.planned_workspace_bytes = retained, workspace
        self._attempts, self._inputs, self._input_state = tuple(attempts), inputs, input_state
        self._state = self._current_state()

    def _current_state(self):
        return (tuple((id(getattr(self, name)), getattr(self, name)._version) for name in _DIRECTIONAL_PLAN_TENSORS),
            tuple((id(t), t._version) for t in self._inputs), self._input_state, self.wave_number,
            (self.kernel_error_budget.green_absolute, self.kernel_error_budget.gradient_component_absolute),
            self.min_order, self.max_order, self.cone_subdivisions,
            self.retained_budget_bytes, self.tensor_workspace_budget_bytes, self.retained_bytes,
            self.planned_workspace_bytes, self._attempts)

    def assert_current(self):
        if self._current_state() != self._state:
            raise ValueError('Directional cone/order plan or caller geometry changed')

    def receipt(self):
        self.assert_current()
        return {'version': 'directional_cone_order_plan_v1', 'pair_count': self.orders.numel(),
            'selection_policy': 'minimum_order_then_coarsest_configured_cube_grid',
            'orders': self.orders.tolist(), 'cone_keys': self.cone_keys.tolist(),
            'normalization_rule': 'minimum_distance_order_and_max_active_support_radius',
            'kernel_budget': asdict(self.kernel_error_budget), 'attempts': list(self._attempts),
            'retained_bytes': self.retained_bytes, 'planned_workspace_bytes': self.planned_workspace_bytes,
            'scope': 'ideal_continuous_declared_support_kernel_only',
            'implemented_operator_certified': False, 'physical_qualification': False}

    def create_translation(self, row, *, rhs_batch=4, retained_budget_bytes=1024**2,
                           tensor_workspace_budget_bytes=128*1024**2):
        """Allocate one admitted primitive, independently owning its tensors.

        Translation capacity is additional to this plan's owned inventory.
        No aggregate full-system capacity or native accuracy is inferred.
        """
        from .fmm_directional_fft import CartesianDirectionalFFTTranslation
        self.assert_current()
        if type(row) is not int or not 0 <= row < self.orders.numel():
            raise ValueError('Directional translation row is out of range')
        result = CartesianDirectionalFFTTranslation(self.displacements[row], self.directions[row],
            self.wave_number, self.normalization_lengths[row], int(self.orders[row]), rhs_batch=rhs_batch,
            retained_budget_bytes=retained_budget_bytes, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes,
            test_radius=self.test_radii[row], source_radius=self.source_radii[row],
            kernel_error_budget=self.kernel_error_budget)
        self.assert_current()
        return result


class AutomaticDirectionalRegionPlan:
    """Complete support partition and material-weighted sparse cone catalog.

    Each geometric FAR rectangle receives B/sqrt(F) of the ideal balanced
    regional budget. Channel budgets split its square equally between G and
    grad G. This is conservative admission, not a native or physical error
    certificate. Failed orders or family capacity never become near blocks.
    Tree construction and integer traversal use the shared physical octree.
    """
    def __init__(self, basis, medium, *, geometry=None, kernel_error_budget,
                 ideal_rwg_error_budget, balancing_impedance, leaf_size=32,
                 separation_ratio=.25, min_order=0, max_order=12,
                 cone_subdivisions=(1, 2, 4, 8, 16, 32, 64), max_pairs=100000,
                 max_families=32, retained_budget_bytes=64*1024**2,
                 tensor_workspace_budget_bytes=256*1024**2):
        from .nearfield import ScalarSystemBlocks, _validate_scalar_channel_medium
        from .operator import PreparedRWGGeometry
        from .cluster import build_support_octree
        from .fmm_directional_moments import _require_sparse_fixed_geometry
        from .rwg_interpolation_error import (rwg_integral_logs,
            integral_statistics_from_logs, block_error_log, _receipt)
        from .direct_capacity import storage_bytes
        _validate_scalar_channel_medium(medium)
        if not isinstance(kernel_error_budget, KernelErrorBudget):
            raise TypeError('Automatic region requires KernelErrorBudget')
        KernelErrorBudget(**asdict(kernel_error_budget))
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
               for v in (ideal_rwg_error_budget, balancing_impedance)):
            raise ValueError('Automatic region ideal budget and common impedance must be positive finite')
        if any(type(v) is not int or v<1 for v in
               (leaf_size,max_pairs,max_families,retained_budget_bytes,tensor_workspace_budget_bytes)):
            raise ValueError('Automatic region requires positive native capacities')
        if type(min_order) is not int or type(max_order) is not int or not 0<=min_order<=max_order<=12:
            raise ValueError('Automatic region order range must lie in0..12')
        grids=tuple(cone_subdivisions)
        if not grids or any(type(v) is not int or not 1<=v<=2**20 for v in grids) or any(a>=b for a,b in zip(grids,grids[1:])):
            raise ValueError('Automatic region cone grids must be increasing bounded integers')
        if type(separation_ratio) not in (int,float) or not math.isfinite(separation_ratio) or not 0<separation_ratio<1:
            raise ValueError('Automatic region separation ratio must lie in(0,1)')
        if isinstance(basis,ScalarSystemBlocks):
            basis.assert_current()
            if geometry is not None or medium is not basis.exterior:
                raise ValueError('Automatic global region requires its actual exterior medium and geometry')
            geometry=basis
            size,device,dtype=basis.system.total_edges,basis.system.device,basis.system.dtype
            prepared=basis.geometry
            bodies=tuple(b.basis for b in basis.system.bodies)
        else:
            if not isinstance(geometry,PreparedRWGGeometry):
                raise TypeError('Automatic body region requires its prepared physical geometry')
            geometry.assert_current(basis)
            size,device,dtype=basis.n_edges,basis.mesh.device,basis.mesh.dtype
            prepared,bodies=(geometry,),(basis,)
        _require_sparse_fixed_geometry(basis,geometry)
        if device.type!='cuda' or dtype!=torch.float64:
            raise ValueError('Automatic region requires fixed CUDA float64 physical geometry')
        faces=sum(len(b.mesh.faces) for b in bodies)
        # Geometry/statistics allowance excludes allocator/library state. The
        # tree and catalog are counted exactly after geometry construction and
        # before any numerical moment or translation factory can be reached.
        admission_workspace=4096+faces*4096+size*4096+directional_bound_workspace_bytes(1,max_order-min_order+1)
        if admission_workspace>tensor_workspace_budget_bytes:
            raise MemoryError('Automatic region admission workspace exceeded before support/statistics')
        if 8*size>retained_budget_bytes:
            raise MemoryError('Automatic region tree capacity exceeded before support allocation')
        if isinstance(basis,ScalarSystemBlocks):lower,upper=basis.support_boxes()
        else:
            supports=torch.cat((geometry.triangles[basis.plus_faces],geometry.triangles[basis.minus_faces]),1)
            lower,upper=supports.amin(1),supports.amax(1)
            del supports
        tree=build_support_octree(lower,upper,leaf_size=leaf_size)
        centers=tree.lower*.5+tree.upper*.5
        radii=[]
        for start,stop,center in zip(tree.starts.tolist(),tree.stops.tolist(),centers,strict=True):
            edges=tree.permutation[start:stop]
            corners=torch.maximum((lower[edges]-center).abs(),(upper[edges]-center).abs())
            radii.append(corners.norm(dim=-1).amax()*(1+512*torch.finfo(torch.float64).eps))
        radii=torch.stack(radii)
        if not bool(torch.isfinite(radii).all() & (radii>0).all()):
            raise ValueError('Automatic complete support radii exceed native geometry range')
        del lower,upper
        base_bytes=storage_bytes((*tree.tensors(),centers,radii))
        if base_bytes>retained_budget_bytes:
            raise MemoryError('Automatic region owned tree capacity exceeded before contour selection')
        near,far=[],[]
        for aa,bb,flags in sphere_block_partition(tree,separation_ratio=separation_ratio,pair_batch=32):
            if len(near)+len(far)+aa.numel()>max_pairs:
                raise MemoryError('Automatic region physical pair capacity exceeded before contours')
            for a,b,flag in zip(aa.tolist(),bb.tolist(),flags.tolist(),strict=True):
                (far if flag else near).append((a,b))
        # Generated declared inputs own40 bytes and each one-row plan144 bytes.
        # At most24 bytes per canonical family are retained here; Python integer
        # topology/catalog metadata is excluded from the native tensor inventory.
        if base_bytes+184*len(far)+24*min(max_families,len(far))>retained_budget_bytes:
            raise MemoryError('Automatic region catalog capacity exceeded before contours')
        logs=torch.cat([rwg_integral_logs(b,g) for b,g in zip(bodies,prepared,strict=True)])
        statistics=integral_statistics_from_logs(tree,logs)
        per_pair=ideal_rwg_error_budget/math.sqrt(len(far)) if far else ideal_rwg_error_budget
        plans,inputs,bounds,families,orders,family_ids=[],[],[],[],[],[]
        for a,b in far:
            weights=torch.stack([block_error_log(statistics[a],statistics[b],medium,
                dict(green_absolute_bound=g,gradient_component_absolute_bound=h),balancing_impedance)
                for g,h in ((1.,0.),(0.,1.))])
            limits=torch.minimum(torch.exp(math.log(per_pair)-.5*math.log(2)-weights),
                centers.new_tensor([kernel_error_budget.green_absolute,kernel_error_budget.gradient_component_absolute]))
            if not bool(torch.isfinite(limits).all() & (limits>0).all()):
                raise ValueError('Automatic region effective pair budget is not natively representable')
            declared=((centers[a]-centers[b]).reshape(1,3),radii[a:a+1].clone(),radii[b:b+1].clone())
            plan=DirectionalConeOrderPlan(*declared,medium.k,KernelErrorBudget(*limits.tolist()),
                min_order=min_order,max_order=max_order,cone_subdivisions=grids,
                retained_budget_bytes=retained_budget_bytes,tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
            order=int(plan.orders[0])
            family=next((i for i,(c,p) in enumerate(zip(families,orders,strict=True))
                if p==order and torch.equal(c.contiguous().view(torch.int64),plan.directions[0].contiguous().view(torch.int64))),None)
            if family is None:
                if len(families)==max_families:
                    raise MemoryError('Automatic region carrier/order family capacity exceeded before maps/coefficients')
                family=len(families);families.append(plan.directions[0]);orders.append(order)
            bounds.append(block_error_log(statistics[a],statistics[b],medium,
                dict(green_absolute_bound=float(plan.green_bounds[0]),
                    gradient_component_absolute_bound=float(plan.gradient_bounds[0])),balancing_impedance))
            plans.append(plan);inputs.extend(declared);family_ids.append(family)
        logged=.5*torch.logsumexp(2*torch.stack(bounds),0) if bounds else centers.new_tensor(-torch.inf)
        bound=_receipt(logged,balancing_impedance)
        if bound['absolute_bound']>ideal_rwg_error_budget:
            raise ValueError('Automatic region selected ideal RSS exceeds unchanged regional budget')
        self.basis,self.geometry,self.medium,self.tree=basis,geometry,medium,tree
        self.centers,self.radii=centers,radii
        self.carriers=torch.stack(families) if families else centers.new_empty((0,3))
        self.orders,self.far_nodes,self.near_nodes=tuple(orders),tuple(far),tuple(near)
        self.family_ids,self.plans,self._inputs=tuple(family_ids),tuple(plans),tuple(inputs)
        self.requests=tuple(sorted({(node,family) for pair,family in zip(far,family_ids,strict=True) for node in pair}))
        self.kernel_error_budget,self.ideal_rwg_error_budget=kernel_error_budget,ideal_rwg_error_budget
        self.balancing_impedance,self.per_pair_budget=balancing_impedance,per_pair
        self.leaf_size,self.separation_ratio,self.min_order,self.max_order=leaf_size,separation_ratio,min_order,max_order
        self.cone_subdivisions,self.max_pairs,self.max_families=grids,max_pairs,max_families
        self.retained_budget_bytes,self.tensor_workspace_budget_bytes=retained_budget_bytes,tensor_workspace_budget_bytes
        self.retained_bytes=storage_bytes(self.tensors())
        self.planned_component_workspace_bytes=admission_workspace+max((p.planned_workspace_bytes for p in plans),default=0)+512*tree.node_count
        if self.retained_bytes>retained_budget_bytes or self.planned_component_workspace_bytes>tensor_workspace_budget_bytes:
            raise MemoryError('Automatic region plan component capacity exceeded before numerical factories')
        self._ideal_bound=bound
        self._binding=self._state()
        self.assert_current()

    def tensors(self):
        return (*self.tree.tensors(),self.centers,self.radii,self.carriers,*self._inputs,
            *(getattr(p,name) for p in self.plans for name in _DIRECTIONAL_PLAN_TENSORS))

    def _state(self):
        return (id(self.basis),id(self.geometry),id(self.medium),
            self.medium.frequency,self.medium.epsilon_r,self.medium.mu_r,self.medium.k,self.medium.eta,
            id(self.tree),self.tree.version,self.tree.leaf_size,self.tree.morton_bits,
            self.orders,self.far_nodes,self.near_nodes,self.family_ids,self.requests,
            self.leaf_size,self.separation_ratio,self.min_order,self.max_order,self.cone_subdivisions,
            self.max_pairs,self.max_families,self.retained_budget_bytes,self.tensor_workspace_budget_bytes,
            self.retained_bytes,self.planned_component_workspace_bytes,self.per_pair_budget,
            self.ideal_rwg_error_budget,self.balancing_impedance,
            asdict(self.kernel_error_budget),tuple(self._ideal_bound.items()),tuple(map(id,self.plans)),
            tuple((id(t),t._version,t.requires_grad) for t in self.tensors()))

    def assert_current(self):
        from .nearfield import ScalarSystemBlocks
        from .fmm_directional_moments import _require_sparse_fixed_geometry
        if self._state()!=self._binding:
            raise ValueError('Automatic directional physical partition/catalog changed')
        self.tree.assert_current()
        if isinstance(self.basis,ScalarSystemBlocks):self.basis.assert_current()
        else:self.geometry.assert_current(self.basis)
        _require_sparse_fixed_geometry(self.basis,self.geometry)
        for p in self.plans:p.assert_current()

    def receipt(self):
        self.assert_current()
        return dict(version='automatic_complete_sparse_directional_region_plan_v1',
            physical_edges=self.tree.size,physical_nodes=self.tree.node_count,
            far_pairs=len(self.far_nodes),near_pairs=len(self.near_nodes),carrier_order_families=len(self.orders),
            orders=list(self.orders),selection_policy='minimum_order_then_coarsest_configured_cube_grid',
            pair_ideal_budget=self.per_pair_budget,ideal_bound=dict(self._ideal_bound),
            owned_native_tensor_bytes=self.retained_bytes,
            planned_component_workspace_bytes=self.planned_component_workspace_bytes,
            caller_geometry_and_python_metadata_excluded=True,
            automatic_complete_physical_support_partition=True,phase_truncation_exact_zero=True,
            native_error_certified=False,quadrature_certified=False,whole_resource_certified=False,
            complexity_qualification=False,physical_qualification=False,production_qualification=False)

    def build_region(self, *, integration, quadrature_order, face_batch=32, rhs_batch=4,
                     projection_retained_budget_bytes=1024**3, table_retained_budget_bytes=512*1024**2,
                     near_retained_budget_bytes=1024**3, tensor_workspace_budget_bytes=1024**3,near_storage='scalar_dk_v1',
                     near_coefficient_workspace_budget_bytes=128*1024**2,
                     near_coefficient_cache_budget_bytes=128*1024**2):
        """Use existing sparse maps, actual-medium M2L and original near blocks.

        Budgets are separate component inventories, not an aggregate whole
        operator peak/crossover promise. The cumulative table budget is checked
        before every table factory; individual maps/near own their preflights.
        """
        from .nearfield import ScalarSystemBlocks
        from .fmm_directional_moments import SparseDirectionalRWGProjection
        from .fmm_directional_rwg import DirectionalFFTElectromagneticM2L,DirectionalSparseFarActions
        from .fmm_regional import DirectionalRegionalOperator
        self.assert_current()
        if any(type(v) is not int or v<1 for v in (quadrature_order,face_batch,rhs_batch,
                projection_retained_budget_bytes,table_retained_budget_bytes,near_retained_budget_bytes,tensor_workspace_budget_bytes)):
            raise ValueError('Automatic region numerical capacities must be positive integers')
        if self.orders and quadrature_order<(max(self.orders)+4)//2:
            raise ValueError('Automatic region quadrature underresolves its selected polynomial order')
        table_bytes=sum(m2l_storage_bytes(int(p.orders[0]),8,'fft_v1')+40 for p in self.plans)
        if table_bytes>table_retained_budget_bytes:
            raise MemoryError('Automatic region cumulative table capacity exceeded before maps/coefficients')
        if near_storage not in ('scalar_dk_v1','scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            raise ValueError('Unsupported directional scalar near storage')
        streamed=near_storage in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1')
        if type(near_coefficient_workspace_budget_bytes) is not int or near_coefficient_workspace_budget_bytes<1:
            raise ValueError('Directional near coefficient workspace must be a positive integer')
        if not streamed and near_coefficient_workspace_budget_bytes!=128*1024**2:
            raise ValueError('Explicit directional coefficient workspace requires streamed reciprocal storage')
        if (type(near_coefficient_cache_budget_bytes) is not int
                or near_coefficient_cache_budget_bytes < 0):
            raise ValueError('Directional near coefficient cache budget must be a nonnegative integer')
        if (near_storage != 'scalar_reciprocal_cached_dk_v1'
                and near_coefficient_cache_budget_bytes != 128*1024**2):
            raise ValueError('Explicit directional coefficient cache budget requires cached reciprocal storage')
        if near_storage in ('scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            from .fmm_capacity import plan_directional_reciprocal_near_storage
            plan_directional_reciprocal_near_storage(self.tree,self.near_nodes,budget_bytes=near_retained_budget_bytes,
                **({} if not streamed else dict(near_storage=near_storage,
                    near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes)),
                **({} if near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                    near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)))
        geometry=None if isinstance(self.basis,ScalarSystemBlocks) else self.geometry
        owner=SparseDirectionalRWGProjection(self.basis,self.tree,self.carriers,self.orders,self.requests,self.medium.k,
            geometry=geometry,quadrature_order=quadrature_order,face_batch=face_batch,rhs_batch=rhs_batch,
            retained_budget_bytes=projection_retained_budget_bytes,tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
        contexts={key:i for i,key in enumerate(owner.context_keys)}
        couplings=[]
        for (a,b),family,p in zip(self.far_nodes,self.family_ids,self.plans,strict=True):
            couplings.append(DirectionalFFTElectromagneticM2L.from_sparse_contexts(owner,contexts[a,family],
                owner,contexts[b,family],self.medium,p.normalization_lengths[0],
                kernel_error_budget=p.kernel_error_budget,ideal_rwg_error_budget=self.per_pair_budget,
                balancing_impedance=self.balancing_impedance,retained_budget_bytes=table_retained_budget_bytes,
                tensor_workspace_budget_bytes=tensor_workspace_budget_bytes))
        far=DirectionalSparseFarActions(owner,owner,self.medium,tuple(couplings),
            ideal_rwg_error_budget=self.ideal_rwg_error_budget,balancing_impedance=self.balancing_impedance,
            rhs_batch=rhs_batch,tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
        region=DirectionalRegionalOperator(far,self.near_nodes,integration=integration,
            near_retained_budget_bytes=near_retained_budget_bytes,tensor_workspace_budget_bytes=tensor_workspace_budget_bytes,near_storage=near_storage,
            **({} if not streamed else dict(near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes)),
                **({} if near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                    near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)))
        self.assert_current()
        return region


class AutomaticDirectionalDAGPlan:
    """Finite greedy sharing over an admitted physical directional partition.

    Root cones/orders and physical near/FAR rectangles stay fixed. Candidates
    share complete child closures, including positive changing-carrier links
    when phase padding, every root budget and the complete component cost admit
    them. This prototype has the same order/catalog limits as its map owner.
    """
    def __init__(self, region_plan, *, quadrature_order, face_batch=32, rhs_batch=4,
                 phase_relative_budget=1e-12, max_candidates=10000,
                 max_contexts=100000, retained_budget_bytes=1024**3,
                 tensor_workspace_budget_bytes=1024**3):
        from itertools import combinations
        from .nearfield import ScalarSystemBlocks
        from .rwg_interpolation_error import rwg_integral_logs, integral_statistics_from_logs
        from .fmm_error import directional_multipole_truncation_bound
        if not isinstance(region_plan, AutomaticDirectionalRegionPlan):
            raise TypeError('Automatic DAG requires an admitted actual regional plan')
        region_plan.assert_current()
        if any(type(v) is not int or v<1 for v in (quadrature_order,face_batch,rhs_batch,
                max_candidates,max_contexts,retained_budget_bytes,tensor_workspace_budget_bytes)):
            raise ValueError('Automatic DAG requires positive integer batches and capacities')
        if type(phase_relative_budget) not in (int,float) or not math.isfinite(phase_relative_budget) or not 0<phase_relative_budget<1:
            raise ValueError('Automatic DAG phase budget must lie in(0,1)')
        if region_plan.orders and quadrature_order<(max(region_plan.orders)+4)//2:
            raise ValueError('Automatic DAG quadrature underresolves an original root order')
        self.region_plan=region_plan
        self.quadrature_order,self.face_batch,self.rhs_batch=quadrature_order,face_batch,rhs_batch
        self.phase_relative_budget,self.max_candidates,self.max_contexts=phase_relative_budget,max_candidates,max_contexts
        self.retained_budget_bytes,self.tensor_workspace_budget_bytes=retained_budget_bytes,tensor_workspace_budget_bytes
        # Candidate scratch has bounded CUDA path arrays and shared physical
        # statistics; host integer adjacency is recorded separately in receipts.
        faces=(sum(len(b.basis.mesh.faces) for b in region_plan.basis.system.bodies)
               if isinstance(region_plan.basis,ScalarSystemBlocks) else len(region_plan.basis.mesh.faces))
        self.planned_candidate_workspace_bytes=(4096+max_contexts*128+faces*4096
            +region_plan.tree.size*4096+directional_bound_workspace_bytes(1,1))
        if self.planned_candidate_workspace_bytes>tensor_workspace_budget_bytes:
            raise MemoryError('Automatic DAG candidate capacity exceeded before path/statistic allocation')
        physical=tuple(tuple(c for c in row if c>=0) for row in region_plan.tree.children.tolist())
        self._physical_children=physical
        roots=region_plan.requests
        carriers=list(region_plan.carriers.unbind(0));orders=list(region_plan.orders)
        links={};pending=list(roots);seen=set()
        while pending:
            key=pending.pop()
            if key in seen:continue
            seen.add(key)
            if len(seen)>max_contexts:
                raise MemoryError('Automatic DAG exact closure context capacity exceeded')
            for child in physical[key[0]]:
                target=(child,key[1]);links[key,target]=0;pending.append(target)
        if isinstance(region_plan.basis,ScalarSystemBlocks):
            bases=tuple(b.basis for b in region_plan.basis.system.bodies);geometries=region_plan.basis.geometry
        else:bases,geometries=(region_plan.basis,),(region_plan.geometry,)
        logs=torch.cat([rwg_integral_logs(b,g) for b,g in zip(bases,geometries,strict=True)])
        self._statistics=integral_statistics_from_logs(region_plan.tree,logs)
        self._kernels=tuple(directional_multipole_truncation_bound(
            region_plan.centers[a]-region_plan.centers[b],region_plan.carriers[f],
            region_plan.radii[a],region_plan.radii[b],region_plan.medium.k,region_plan.orders[f],
            tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
            for (a,b),f in zip(region_plan.far_nodes,region_plan.family_ids,strict=True))
        state=self._canonical(carriers,orders,roots,links)
        current=self._evaluate(state)
        if current is None:
            raise ValueError('Automatic DAG exact root closure failed unchanged admission')
        initial_cost=current['cost'];attempts=0;selected=0;positive=0;limit=False
        # Physical topology is visited once. Candidate evaluation intentionally
        # recomputes all ancestors, root budgets and the entire live closure.
        for node in range(region_plan.tree.node_count):
            active=sorted({child for _,child in state[3] if child[0]==node and child not in state[2]})
            if len(active)<2:continue
            groups=[tuple(active)] if len(active)>2 else []
            groups.extend(combinations(active,2))
            best=None
            for group in groups:
                for family in sorted({key[1] for key in group}):
                    if attempts==max_candidates:limit=True;break
                    attempts+=1
                    candidate=self._merge(state,group,family)
                    if candidate is None:continue
                    evaluated=self._evaluate(candidate)
                    if evaluated is None or evaluated['cost']>=current['cost']:continue
                    if best is None or evaluated['cost']<best[1]['cost']:
                        best=(candidate,evaluated)
                if limit:break
            if best is not None:
                state,current=best;selected+=1
                if current['positive_links']>positive:positive=current['positive_links']
            if limit:break
        carriers,orders,roots,links=state
        self.carriers=torch.stack(carriers) if carriers else region_plan.centers.new_empty((0,3))
        self.orders,self.requests=tuple(orders),tuple(roots)
        self.links=tuple((a,b,L) for (a,b),L in sorted(links.items()))
        self.family_ids=tuple(self.requests[self._root_position(node,old_family)][1]
            for (node,_),old_family in zip(region_plan.far_nodes,region_plan.family_ids,strict=True))
        self.context_keys=current['keys'];self.path_products=current['path_products']
        self._bounds=current['bounds'];self._ideal_bound=current['ideal_bound']
        self.initial_component_cost=initial_cost;self.selected_component_cost=current['cost']
        self.planned_projection_retained_bytes=current['retained']
        self.planned_projection_workspace_bytes=current['workspace']
        self.candidate_attempts,self.selected_merges,self.search_limit_reached=attempts,selected,limit
        self.selected_positive_links=current['positive_links']
        self._binding=self._state()
        self.assert_current()

    def _root_position(self,node,family):
        return self.region_plan.requests.index((node,family))

    def _canonical(self,carriers,orders,roots,links):
        """Drop unreachable closures/layout references and unused catalog slots."""
        adjacency={}
        for parent,child in links:adjacency.setdefault(parent,[]).append(child)
        seen=set();pending=list(roots)
        while pending:
            key=pending.pop()
            if key in seen:continue
            seen.add(key)
            if len(seen)>self.max_contexts:return None
            pending.extend(adjacency.get(key,()))
        used=sorted({family for _,family in seen})
        if len(used)>self.region_plan.max_families:return None
        remap={f:i for i,f in enumerate(used)}
        rekey=lambda key:(key[0],remap[key[1]])
        return ([carriers[f] for f in used],[orders[f] for f in used],
            tuple(rekey(key) for key in roots),
            {(rekey(a),rekey(b)):L for (a,b),L in links.items() if a in seen})

    def _merge(self,state,group,family):
        carriers,orders,roots,links=state
        incoming=[(a,b) for a,b in links if b in group]
        phases={};required=0;node=group[0][0];direction=carriers[family]
        for parent,_ in incoming:
            exact=torch.equal(carriers[parent[1]],direction)
            x=(carriers[parent[1]]-direction).norm()*abs(self.region_plan.medium.k)*self.region_plan.radii[node]
            if not bool(torch.isfinite(x)) or (not exact and not bool(x>0)):return None
            L=0 if exact else next((L for L in range(13-orders[parent[1]])
                if bool(2*x+(L+1)*x.log()-math.lgamma(L+2)<=math.log(self.phase_relative_budget))),None)
            if L is None:return None
            phases[parent]=L;required=max(required,orders[parent[1]]+L)
        if required>12 or self.quadrature_order<(required+4)//2:return None
        target_family=next((i for i,(c,p) in enumerate(zip(carriers,orders,strict=True))
            if p==required and torch.equal(c,direction)),None)
        new_carriers,new_orders=list(carriers),list(orders)
        if target_family is None:
            target_family=len(carriers);new_carriers.append(direction);new_orders.append(required)
        target=(node,target_family)
        new_links={pair:L for pair,L in links.items() if pair[1] not in group}
        for parent in phases:new_links[parent,target]=phases[parent]
        # Existing shared descendants keep their admitted links. New contexts
        # get complete exact-carrier closures; full evaluation prices both.
        pending=[target];seen=set();parents={a for a,_ in new_links}
        while pending:
            key=pending.pop()
            if key in seen:continue
            seen.add(key)
            if len(seen)>self.max_contexts:return None
            if key in parents:continue
            for child in self._physical_children[key[0]]:
                next_key=(child,target_family);new_links[key,next_key]=0;pending.append(next_key)
        return self._canonical(new_carriers,new_orders,roots,new_links)

    def _evaluate(self,state):
        from .fmm_directional_moments import _directional_dag_topology,_directional_context_capacity
        from .fmm_directional_rwg import _directional_phase_perturbed_kernel
        from .rwg_interpolation_error import block_error_log,_receipt
        if state is None:return None
        carriers,orders,roots,links=state
        p=self.region_plan
        keys,children,phases=_directional_dag_topology(self._physical_children,roots,orders,
            tuple((a,b,L) for (a,b),L in links.items()))
        index={key:i for i,key in enumerate(keys)}
        _,_,_,_,_,retained,workspace=_directional_context_capacity(p.basis,p.tree,keys,children,phases,
            orders,len(carriers),self.quadrature_order,self.face_batch,self.rhs_batch,dag=True)
        if retained>self.retained_budget_bytes or workspace>self.tensor_workspace_budget_bytes:return None
        products=p.centers.new_zeros(len(keys));positive=0
        for a in reversed(range(len(keys))):
            values=[]
            for c in children[a]:
                family,child_family=keys[a][1],keys[c][1]
                exact=torch.equal(carriers[family],carriers[child_family]);L=phases[a,c]
                if exact:delta=products.new_zeros(())
                else:
                    x=(carriers[family]-carriers[child_family]).norm()*abs(p.medium.k)*p.radii[keys[c][0]]
                    if not bool(torch.isfinite(x) & (x>0)):return None
                    logged=2*x+(L+1)*x.log()-math.lgamma(L+2)
                    if bool(logged>math.log(self.phase_relative_budget)):return None
                    delta=logged.clamp_min(math.log(torch.finfo(torch.float64).tiny)).exp();positive+=1
                values.append(torch.log1p(delta)+products[c])
            if values:products[a]=torch.stack(values).amax()
        if not bool(torch.isfinite(products).all()):return None
        # The shared kernel formula needs only fixed context centers and the
        # complete path array, so planning does not allocate maps or transfers.
        class Paths:
            def _ideal_phase_path_log_products(self,workspace):return products
        view=Paths();view.centers=torch.stack([p.centers[n] for n,_ in keys]) if keys else p.centers.new_empty((0,3))
        bounds=[];logged_bounds=[]
        for (a,b),family,kernel,root_plan in zip(p.far_nodes,p.family_ids,self._kernels,p.plans,strict=True):
            ra=roots[self._root_position(a,family)];rb=roots[self._root_position(b,family)]
            effective,_,_,_=_directional_phase_perturbed_kernel(view,index[ra],view,index[rb],p.medium,
                (p.radii[a],p.radii[b]),kernel,self.tensor_workspace_budget_bytes)
            if (effective['green_absolute']>root_plan.kernel_error_budget.green_absolute
                    or effective['gradient_component_absolute']>root_plan.kernel_error_budget.gradient_component_absolute):return None
            logged=block_error_log(self._statistics[a],self._statistics[b],p.medium,
                dict(green_absolute_bound=effective['green_absolute'],
                    gradient_component_absolute_bound=effective['gradient_component_absolute']),p.balancing_impedance)
            bound=_receipt(logged,p.balancing_impedance)
            if bound['absolute_bound']>p.per_pair_budget:return None
            bounds.append(bound|dict(test_path_log=float(products[index[ra]]),source_path_log=float(products[index[rb]])))
            logged_bounds.append(logged)
        logged=.5*torch.logsumexp(2*torch.stack(logged_bounds),0) if logged_bounds else p.centers.new_tensor(-torch.inf)
        ideal=_receipt(logged,p.balancing_impedance)
        if ideal['absolute_bound']>p.ideal_rwg_error_budget:return None
        return dict(keys=keys,path_products=products,bounds=tuple(bounds),ideal_bound=ideal,
            retained=retained,workspace=workspace,cost=(retained+workspace,retained,workspace),positive_links=positive)

    def _state(self):
        return (id(self.region_plan),self.orders,self.requests,self.links,self.family_ids,self.context_keys,
            self.quadrature_order,self.face_batch,self.rhs_batch,self.phase_relative_budget,self.max_candidates,
            self.max_contexts,self.retained_budget_bytes,self.tensor_workspace_budget_bytes,
            self.planned_candidate_workspace_bytes,self.initial_component_cost,self.selected_component_cost,
            self.planned_projection_retained_bytes,self.planned_projection_workspace_bytes,
            self.candidate_attempts,self.selected_merges,self.selected_positive_links,self.search_limit_reached,
            tuple(tuple(b.items()) for b in self._bounds),tuple(self._ideal_bound.items()),
            tuple((id(t),t._version,t.requires_grad) for t in (self.carriers,self.path_products,self._statistics)))

    def assert_current(self):
        self.region_plan.assert_current()
        if self._state()!=self._binding or self.carriers.requires_grad or self.path_products.requires_grad:
            raise ValueError('Automatic directional DAG selection or root binding changed')

    def receipt(self):
        self.assert_current()
        indegrees={}
        for _,child,_ in self.links:indegrees[child]=indegrees.get(child,0)+1
        return dict(version='automatic_shared_directional_DAG_plan_v1',
            physical_far_pairs=len(self.region_plan.far_nodes),physical_near_pairs=len(self.region_plan.near_nodes),
            root_cones_orders_and_physical_partition_preserved=True,active_contexts=len(self.context_keys),
            shared_children=sum(v>1 for v in indegrees.values()),selected_positive_links=self.selected_positive_links,
            selected_merges=self.selected_merges,candidate_attempts=self.candidate_attempts,
            search_limit_reached=self.search_limit_reached,
            selection_policy='finite_physical_node_group_carrier_order_with_whole_closure_cost',
            cost_rule='retained_plus_component_workspace_then_retained_then_workspace',
            initial_component_cost=self.initial_component_cost,selected_component_cost=self.selected_component_cost,
            planned_projection_retained_bytes=self.planned_projection_retained_bytes,
            planned_projection_workspace_bytes=self.planned_projection_workspace_bytes,
            planned_candidate_workspace_bytes=self.planned_candidate_workspace_bytes,
            host_integer_metadata_excluded=True,all_roots_recomputed_for_each_candidate=True,
            ideal_bound=dict(self._ideal_bound),native_error_certified=False,quadrature_certified=False,
            whole_resource_certified=False,complexity_qualification=False,production_qualification=False)

    def build_far(self, *, table_retained_budget_bytes=512*1024**2):
        from .nearfield import ScalarSystemBlocks
        from .fmm_directional_moments import DirectionalRWGDAGProjection
        from .fmm_directional_rwg import DirectionalFFTElectromagneticM2L,DirectionalDAGFarActions
        self.assert_current();p=self.region_plan
        if any(type(v) is not int or v<1 for v in (table_retained_budget_bytes,)):
            raise ValueError('Automatic DAG table/near capacities must be positive integers')
        if sum(m2l_storage_bytes(int(q.orders[0]),8,'fft_v1')+40 for q in p.plans)>table_retained_budget_bytes:
            raise MemoryError('Automatic DAG cumulative tables exceed capacity before maps/coefficients')
        owner=DirectionalRWGDAGProjection(p.basis,p.tree,self.carriers,self.orders,self.requests,self.links,p.medium.k,
            geometry=None if isinstance(p.basis,ScalarSystemBlocks) else p.geometry,
            quadrature_order=self.quadrature_order,face_batch=self.face_batch,rhs_batch=self.rhs_batch,
            phase_relative_budget=self.phase_relative_budget,retained_budget_bytes=self.retained_budget_bytes,
            tensor_workspace_budget_bytes=self.tensor_workspace_budget_bytes,plan_guard=self)
        if (owner.retained_bytes!=self.planned_projection_retained_bytes
                or owner.planned_tensor_workspace_bytes!=self.planned_projection_workspace_bytes):
            raise RuntimeError('Automatic DAG factory inventory differs from shared structural plan')
        indices={key:i for i,key in enumerate(owner.context_keys)};couplings=[]
        for (a,b),family,q in zip(p.far_nodes,self.family_ids,p.plans,strict=True):
            couplings.append(DirectionalFFTElectromagneticM2L.from_dag_contexts(owner,indices[a,family],owner,indices[b,family],
                p.medium,q.normalization_lengths[0],kernel_error_budget=q.kernel_error_budget,
                ideal_rwg_error_budget=p.per_pair_budget,balancing_impedance=p.balancing_impedance,
                retained_budget_bytes=table_retained_budget_bytes,tensor_workspace_budget_bytes=self.tensor_workspace_budget_bytes))
        far=DirectionalDAGFarActions(owner,owner,p.medium,tuple(couplings),ideal_rwg_error_budget=p.ideal_rwg_error_budget,
            balancing_impedance=p.balancing_impedance,rhs_batch=self.rhs_batch,
            tensor_workspace_budget_bytes=self.tensor_workspace_budget_bytes)
        self.assert_current();return far

    def build_region(self, *, integration, table_retained_budget_bytes=512*1024**2,
                     near_retained_budget_bytes=1024**3,near_storage='scalar_dk_v1',
                     near_coefficient_workspace_budget_bytes=128*1024**2,
                     near_coefficient_cache_budget_bytes=128*1024**2):
        from .fmm_regional import DirectionalRegionalOperator
        self.assert_current()
        if type(near_retained_budget_bytes) is not int or near_retained_budget_bytes<1:
            raise ValueError('Automatic DAG near capacity must be a positive integer')
        if near_storage not in ('scalar_dk_v1','scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            raise ValueError('Unsupported directional scalar near storage')
        streamed=near_storage in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1')
        if type(near_coefficient_workspace_budget_bytes) is not int or near_coefficient_workspace_budget_bytes<1:
            raise ValueError('Directional near coefficient workspace must be a positive integer')
        if not streamed and near_coefficient_workspace_budget_bytes!=128*1024**2:
            raise ValueError('Explicit directional coefficient workspace requires streamed reciprocal storage')
        if (type(near_coefficient_cache_budget_bytes) is not int
                or near_coefficient_cache_budget_bytes < 0):
            raise ValueError('Directional near coefficient cache budget must be a nonnegative integer')
        if (near_storage != 'scalar_reciprocal_cached_dk_v1'
                and near_coefficient_cache_budget_bytes != 128*1024**2):
            raise ValueError('Explicit directional coefficient cache budget requires cached reciprocal storage')
        if near_storage in ('scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            from .fmm_capacity import plan_directional_reciprocal_near_storage
            plan_directional_reciprocal_near_storage(self.region_plan.tree,self.region_plan.near_nodes,
                budget_bytes=near_retained_budget_bytes,
                **({} if not streamed else dict(near_storage=near_storage,
                    near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes)),
                **({} if near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                    near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)))
        far=self.build_far(table_retained_budget_bytes=table_retained_budget_bytes)
        region=DirectionalRegionalOperator(far,self.region_plan.near_nodes,integration=integration,
            near_retained_budget_bytes=near_retained_budget_bytes,tensor_workspace_budget_bytes=self.tensor_workspace_budget_bytes,near_storage=near_storage,
            **({} if not streamed else dict(near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes)),
                **({} if near_storage != 'scalar_reciprocal_cached_dk_v1' else dict(
                    near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes)))
        self.assert_current();return region
