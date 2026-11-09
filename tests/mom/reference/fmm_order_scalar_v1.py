"""Plan uniform Cartesian FMM order before moment or coefficient allocation."""
from dataclasses import asdict
import math
import torch
from .cluster import sphere_block_partition
from .fmm_storage import m2l_storage_bytes
from .fmm_error import multipole_truncation_bound
from .fmm_rwg_error import multipole_block_error_log, multipole_error_receipt
from .interpolation_error import KernelErrorBudget, KernelBoundUnavailable


class MultipoleOrderError(ValueError):
    def __init__(self, receipt):
        super().__init__('No Cartesian FMM order satisfies the declared kernel/table budgets')
        self.receipt = receipt


def plan_multipole_order(regions, budget, *, min_order=0, max_order=12,
                         separation_ratio=.5, table_budget_bytes=512*1024**2,
                         rwg_statistics=None, rwg_absolute_budget=None, balancing_impedance=None, table_storage='dense'):
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
    regions = tuple(regions)
    if not regions:
        raise ValueError('FMM planning requires regional trees and media')
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
        raise MultipoleOrderError({'version': 'cartesian_fmm_order_plan_v1', 'table_storage': table_storage, 'reason': reason,
            'attempts': attempts, 'kernel_budget': asdict(budget), 'table_budget_bytes': table_budget_bytes,
            'rwg_absolute_budget': rwg_absolute_budget, 'balancing_impedance': balancing_impedance})
    for order in range(min_order, max_order+1):
        count = math.comb(order+3, 3)
        total, far_count, unavailable = 0, 0, 0
        green, gradient = 0., 0.
        region_logs = []
        for region_index, (tree, medium) in enumerate(regions):
            region_log_squared = tree.lower.new_tensor(-torch.inf, dtype=torch.float64)
            tree.assert_current()
            centers = tree.lower*.5+tree.upper*.5
            radii = torch.linalg.vector_norm((tree.upper-tree.lower)*.5, dim=-1)
            for aa, bb, flags in sphere_block_partition(tree, separation_ratio=separation_ratio, pair_batch=32):
                for a, b in zip(aa[flags].tolist(), bb[flags].tolist()):
                    far_count += 1
                    total += m2l_storage_bytes(order, tree.lower.element_size(), table_storage)
                    try:
                        bound = multipole_truncation_bound(centers[a]-centers[b], radii[a], radii[b], medium.k, order)
                    except KernelBoundUnavailable:
                        unavailable += 1
                        continue
                    if rwg_absolute_budget is not None:
                        statistics = rwg_statistics[region_index]
                        block_log = multipole_block_error_log(statistics[a], statistics[b], medium, bound, balancing_impedance)
                        region_log_squared = torch.logaddexp(region_log_squared, 2*block_log)
                    green = max(green, bound['green_absolute'])
                    gradient = max(gradient, bound['gradient_component_absolute'])
            tree.assert_current()
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
            return {'version': 'cartesian_fmm_order_plan_v1', 'table_storage': table_storage, 'selected_order': order,
                    'table_bytes': total, 'kernel_budget': asdict(budget), 'attempts': attempts,
                    'rwg_absolute_budget': rwg_absolute_budget, 'rwg_bound': rwg_receipt,
                    'scope': 'ideal_kernel_and_exactly_integrated_rwg' if rwg_absolute_budget is not None else 'ideal_exact_arithmetic_kernel_truncation_only',
                    'implemented_operator_certified': False, 'physical_qualification': False}
    fail('maximum_order_or_unavailable_bound')
