"""Fixed-image-cube order selection with complete exterior and interior gates."""
import math
from dataclasses import asdict
from .interpolation_error import KernelErrorBudget, KernelBoundUnavailable
from .fmm_order import plan_multipole_order, MultipoleOrderError
from .fmm_periodic_error import periodic_hierarchy_error_bound
from .fmm_periodic_capacity import plan_periodic_storage
from .fmm_storage import m2l_storage_bytes


class PeriodicOrderError(ValueError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('No periodic FMM order satisfies the declared error and storage budgets')


def plan_periodic_order(provider, trees, lattice, *, min_order, max_order, shells,
                        exterior_budget, interior_budget, separation_ratio=.5,
                        table_storage='dense', action_budget_bytes=1024**3,
                        table_budget_bytes=512*1024**2, near_budget_bytes=1024**3,
                        projection_budget_bytes=1024**3, remote_budget=None,shift_storage='dense'):
    """Select the first admitted order; never increase budgets or change shells."""
    if table_storage in ('fft_v1','fft_ewald_v1'):
        raise NotImplementedError('Periodic FFT order planning requires separately admitted screened coefficients')
    if (any(type(p) is not int or p < 0 for p in (min_order, max_order))
            or max_order < min_order):
        raise ValueError('Periodic order range must be nonnegative and ordered')
    if not isinstance(exterior_budget, KernelErrorBudget) or not isinstance(interior_budget, KernelErrorBudget):
        raise TypeError('Periodic order planning requires exterior and interior KernelErrorBudget')
    if remote_budget is not None and not isinstance(remote_budget, KernelErrorBudget):
        raise TypeError('Periodic remote budget must be KernelErrorBudget')
    for value in (action_budget_bytes, table_budget_bytes, near_budget_bytes, projection_budget_bytes):
        if type(value) is not int or value < 1:
            raise ValueError('Periodic planning storage budgets must be positive integers')
    trees = tuple(trees)
    if len(trees) != len(provider.system.bodies)+1:
        raise ValueError('Periodic planning requires exterior then one tree per body')
    attempts = []
    def satisfies(bound, budget):
        return (bound['green_absolute'] <= budget.green_absolute and
                bound['gradient_component_absolute'] <= budget.gradient_component_absolute)
    metadata = dict(version='periodic_fixed_cube_order_plan_v1', shells=shells,
        exterior_budget=asdict(exterior_budget), interior_budget=asdict(interior_budget),
        remote_budget=None if remote_budget is None else asdict(remote_budget),
        scope='ideal_complete_exterior_kernel_and_each_interior_kernel_plus_retained_storage',
        implemented_operator_certified=False, physical_qualification=False, peak_memory_qualified=False)
    for order in range(min_order, max_order+1):
        capacity = plan_periodic_storage(provider, trees, lattice, order=order, shells=shells,
            separation_ratio=separation_ratio, budget_bytes=action_budget_bytes, table_storage=table_storage,shift_storage=shift_storage)
        real = trees[0].lower.element_size()
        root_retained = m2l_storage_bytes(order, real, table_storage)
        root_setup = root_retained+(root_retained if table_storage == 'dense' else math.comb(2*order+4, 3)*2*real)
        resource_pass = (capacity['admitted'] and max(capacity['categories']['m2l'], root_setup) <= table_budget_bytes
            and capacity['categories']['near'] <= near_budget_bytes
            and capacity['categories']['projections'] <= projection_budget_bytes)
        exterior, unavailable = None, None
        try:
            exterior = periodic_hierarchy_error_bound(trees[0], trees[0], lattice, provider.exterior.k,
                order=order, shells=shells, separation_ratio=separation_ratio)
        except KernelBoundUnavailable as exc:
            unavailable = str(exc)
        exterior_pass = exterior is not None and satisfies(exterior, exterior_budget)
        remote_pass = remote_budget is None or (exterior is not None and satisfies(exterior['remote'], remote_budget))
        try:
            interior = plan_multipole_order(tuple((tree, body.interior) for tree, body in
                zip(trees[1:], provider.system.bodies)), interior_budget, min_order=order, max_order=order,
                separation_ratio=separation_ratio, table_budget_bytes=table_budget_bytes, table_storage=table_storage)
            interior_pass = True
        except MultipoleOrderError as exc:
            interior, interior_pass = exc.receipt, False
        attempt = dict(order=order, exterior_bound=exterior, exterior_bound_unavailable=unavailable,
            exterior_pass=exterior_pass, interior_plan=interior, interior_pass=interior_pass,
            remote_pass=remote_pass, capacity_plan=capacity, root_setup_table_bytes=root_setup,
            resource_pass=resource_pass)
        attempts.append(attempt)
        if exterior_pass and interior_pass and remote_pass and resource_pass:
            return dict(**metadata, selected_order=order, attempts=attempts, capacity_plan=capacity)
        # With fixed partition, retained sizes and root setup storage are
        # nondecreasing in order. Higher order cannot rescue a resource failure.
        if not resource_pass:
            raise PeriodicOrderError(dict(**metadata, attempts=attempts, reason='storage_budget'))
    raise PeriodicOrderError(dict(**metadata, attempts=attempts, reason='maximum_order_or_unavailable_bound'))


def plan_periodic_parameters(provider, trees, lattice, *, min_shells, max_shells, **order_options):
    """Choose the smallest admitted cube, then its first admitted order.

    A larger cube may reduce its omitted tail enough to permit lower order,
    so a failed high-order storage attempt does not stop the shell search.
    """
    if order_options.get('table_storage') in ('fft_v1','fft_ewald_v1'):
        raise NotImplementedError('Periodic FFT shell planning requires separately admitted screened coefficients')
    if (any(type(s) is not int or s < 1 for s in (min_shells, max_shells))
            or max_shells < min_shells):
        raise ValueError('Periodic shell range must be positive and ordered')
    trees = tuple(trees)
    attempts = []
    metadata = dict(version='periodic_shell_order_plan_v1', minimum_shells=min_shells,
        maximum_shells=max_shells, selection='smallest_shell_then_smallest_admitted_order',
        implemented_operator_certified=False, physical_qualification=False, peak_memory_qualified=False)
    for shells in range(min_shells, max_shells+1):
        try:
            plan = plan_periodic_order(provider, trees, lattice, shells=shells, **order_options)
        except PeriodicOrderError as exc:
            attempts.append(dict(shells=shells, admitted=False, plan=exc.receipt))
            continue
        attempts.append(dict(shells=shells, admitted=True, plan=plan))
        return dict(**metadata, selected_shells=shells, selected_order=plan['selected_order'],
            capacity_plan=plan['capacity_plan'], attempts=attempts)
    raise PeriodicOrderError(dict(**metadata, attempts=attempts, reason='shell_order_range_exhausted'))
