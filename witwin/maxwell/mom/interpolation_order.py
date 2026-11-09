"""Preallocation uniform-order planning over streamed regional tree partitions."""
from dataclasses import asdict
import math
import torch
from .cluster import support_block_partition
from .physics import Medium
from .interpolation_error import KernelErrorBudget, KernelBoundUnavailable, scalar_box_interpolation_bound
from .rwg_interpolation_error import block_error_log


class KernelOrderSelectionError(ValueError):
    def __init__(self, receipt):
        super().__init__('No admissible interpolation order within the declared error and table budgets')
        self.receipt = receipt


def plan_uniform_kernel_order(regions, budget, *, min_order=2, max_order=12,
                              admissibility=1., kernel_budget_bytes=512*1024**2,
                              rwg_statistics=None, rwg_absolute_budget=None, balancing_impedance=None):
    """Select the first fully passing order; no nodes, moments or tables exist.

    Each region is (support tree, scalar medium), with the same tree on both
    sides. Every accepted order checks every far block; no monotonic truncation
    assumption or empirical sampling replaces the analytic budget.
    With RWG statistics, region zero is the common exterior and all remaining
    regions must occupy disjoint body-interior coefficient spaces.
    """
    regions = tuple(regions)
    if not regions or not isinstance(budget, KernelErrorBudget):
        raise ValueError('Order planning requires regional trees and KernelErrorBudget')
    if (any(type(v) is not int or v < 1 for v in (min_order, max_order, kernel_budget_bytes))
            or max_order < min_order):
        raise ValueError('Order range and table budget must be positive and ordered')
    for tree, medium in regions:
        tree.assert_current()
        if (not isinstance(medium, Medium) or not math.isfinite(medium.k.real)
                or not math.isfinite(medium.k.imag) or medium.k.imag < 0):
            raise ValueError('Order planning requires finite outgoing passive scalar media')
    use_rwg = rwg_absolute_budget is not None
    if use_rwg:
        if (type(rwg_absolute_budget) not in (int, float) or not math.isfinite(rwg_absolute_budget)
                or rwg_absolute_budget <= 0 or type(balancing_impedance) not in (int, float)
                or not math.isfinite(balancing_impedance) or balancing_impedance <= 0):
            raise ValueError('RWG absolute budget and balancing impedance must be positive and finite')
        if not isinstance(rwg_statistics, (tuple, list)) or len(rwg_statistics) != len(regions):
            raise ValueError('RWG planning requires one statistic array per region')
        for (tree, _), stat in zip(regions, rwg_statistics):
            if (not isinstance(stat, torch.Tensor) or stat.shape != (tree.node_count, 3)
                    or stat.device != tree.device or stat.device != regions[0][0].device
                    or stat.dtype != torch.float64 or stat.requires_grad or not bool(torch.isfinite(stat).all())):
                raise ValueError('RWG planning statistics must be finite fixed CUDA float64 node arrays')
        stat_versions = tuple((id(s), s._version) for s in rwg_statistics)
    elif rwg_statistics is not None or balancing_impedance is not None:
        raise ValueError('RWG planning arguments require an absolute budget')
    attempts = []
    for order in range(min_order, max_order+1):
        table_bytes, far_count, exact_count = 0, 0, 0
        maximum_green, maximum_gradient = 0., 0.
        bound_available = True
        regional_errors = []
        for region_index, (tree, medium) in enumerate(regions):
            if use_rwg:
                stat = rwg_statistics[region_index]
                squared_error = stat.new_tensor(-torch.inf)
            itemsize = 16 if tree.lower.dtype == torch.float64 else 8
            active_axes = ((tree.upper-tree.lower)*.5 > 0).sum(-1).cpu().tolist()
            for rows, cols, flags in support_block_partition(tree, admissibility=admissibility):
                for a, b, far in zip(rows.cpu().tolist(), cols.cpu().tolist(), flags.cpu().tolist()):
                    if not far:
                        exact_count += 1
                        continue
                    far_count += 1
                    table_bytes += 4*itemsize*order**(active_axes[a]+active_axes[b])
                    if table_bytes > kernel_budget_bytes:
                        # Node counts cannot decrease at higher uniform orders.
                        raise KernelOrderSelectionError({'version': 'uniform_kernel_order_plan_v1',
                            'reason': 'kernel_table_budget', 'attempts': attempts, 'order': order,
                            'required_table_bytes_at_least': table_bytes, 'kernel_budget_bytes': kernel_budget_bytes,
                            'rwg_absolute_budget': rwg_absolute_budget, 'balancing_impedance': balancing_impedance,
                            'budget': asdict(budget), 'physical_qualification': False})
                    try:
                        bound = scalar_box_interpolation_bound(tree.lower[a], tree.upper[a],
                            tree.lower[b], tree.upper[b], medium.k, order, order)
                    except KernelBoundUnavailable:
                        bound_available = False
                        continue
                    maximum_green = max(maximum_green, bound['green_absolute_bound'])
                    maximum_gradient = max(maximum_gradient, bound['gradient_component_absolute_bound'])
                    if use_rwg:
                        block_log = block_error_log(stat[a], stat[b], medium, bound, balancing_impedance)
                        squared_error = torch.logaddexp(squared_error, 2*block_log)
            if use_rwg:
                regional_errors.append(.5*squared_error)
        rwg_bound, rwg_passed = None, not use_rwg
        if use_rwg:
            if tuple((id(s), s._version) for s in rwg_statistics) != stat_versions:
                raise ValueError('RWG planning statistics changed during selection')
            # First region is the common exterior; the rest are disjoint body interiors.
            regional_errors = torch.stack(regional_errors)
            interior_log = .5*torch.logsumexp(2*regional_errors[1:], dim=0)
            total_log = torch.logaddexp(regional_errors[0], interior_log)
            if bound_available and not bool(torch.isnan(total_log) | torch.isposinf(total_log)):
                value = (total_log.new_zeros(()) if bool(torch.isneginf(total_log)) else
                         torch.exp(total_log.clamp_min(math.log(torch.finfo(torch.float64).tiny))))
                if bool(torch.isfinite(value)):
                    rwg_bound = float(value)
                    rwg_passed = rwg_bound <= rwg_absolute_budget
        attempt = {'order': order, 'kernel_table_bytes': table_bytes, 'far_tables': far_count,
                   'exact_blocks': exact_count, 'maximum_green_bound': maximum_green,
                   'maximum_gradient_component_bound': maximum_gradient,
                   'all_bounds_finite': bound_available}
        if use_rwg:
            attempt['rwg_absolute_bound'] = rwg_bound
            attempt['rwg_absolute_budget_passed'] = rwg_passed
        attempts.append(attempt)
        if bound_available and rwg_passed and maximum_green <= budget.green_absolute and maximum_gradient <= budget.gradient_component_absolute:
            return {'version': 'uniform_kernel_order_plan_v1', 'selected_order': order,
                    'attempts': attempts, 'kernel_table_bytes': table_bytes, 'budget': asdict(budget),
                    'rwg_absolute_budget': rwg_absolute_budget, 'balancing_impedance': balancing_impedance,
                    'rwg_error_scope': 'ideal_exactly_integrated_interpolant_only',
                    'scope': ('ideal_rwg_and_kernel_interpolation_plus_retained_tables' if use_rwg else
                              'exact_arithmetic_kernel_truncation_and_retained_tables_only'),
                    'peak_memory_qualified': False, 'operator_error_certified': False,
                    'physical_qualification': False}
    raise KernelOrderSelectionError({'version': 'uniform_kernel_order_plan_v1',
        'reason': 'maximum_order', 'attempts': attempts, 'budget': asdict(budget),
        'rwg_absolute_budget': rwg_absolute_budget, 'balancing_impedance': balancing_impedance,
        'physical_qualification': False})
