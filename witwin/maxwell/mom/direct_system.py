"""Global exterior tree plus body-local interiors for direct scalar actions."""
import torch
import json
from .direct_capacity import plan_direct_storage, DirectCapacityError, direct_action_tensors, storage_bytes
from .nearfield import ScalarSystemBlocks
from .cluster import build_support_octree
from .nested_moments import NestedRWGMoments
from .direct_regional import DirectRegionalOperator
from .interpolation_order import plan_uniform_kernel_order
from .rwg_interpolation_error import rwg_integral_logs, integral_statistics_from_logs


class _FixedDirectAction(torch.autograd.Function):
    """Linear current derivative without saving the internal action graph."""
    @staticmethod
    def forward(ctx, value, actions, adjoint):
        ctx.actions, ctx.adjoint = actions, adjoint
        return actions._action(value, adjoint=adjoint)

    @staticmethod
    def backward(ctx, gradient):
        # Re-enter the guarded linear primitive, preserving higher current
        # derivatives without retaining forward currents or projection graphs.
        actions = ctx.actions
        actions.assert_current()
        result = actions.apply(gradient) if ctx.adjoint else actions.adjoint(gradient)
        return result, None, None


class DirectSystemActions:
    def __init__(self, provider, *, order=6, quadrature_order=None, leaf_size=32,
                 admissibility=1., kernel_budget_bytes=512*1024**2, kernel_error_budget=None,
                 max_order=None, rwg_absolute_budget=None, action_retained_budget_bytes=1024**3):
        if not isinstance(provider, ScalarSystemBlocks):
            raise TypeError('Direct system actions require scalar system blocks')
        provider.assert_current()
        if type(kernel_budget_bytes) is not int or kernel_budget_bytes < 1:
            raise ValueError('Kernel table budget must be a positive integer')
        if rwg_absolute_budget is not None and max_order is None:
            raise ValueError('RWG absolute admission requires automatic order planning')
        self.provider = provider
        self.device = provider.system.device
        self.dtype = torch.complex128 if provider.system.dtype == torch.float64 else torch.complex64
        self.shape = (2*provider.system.total_edges,)*2
        lower, upper = provider.support_boxes()
        global_tree = build_support_octree(lower, upper, leaf_size=leaf_size)
        body_trees = tuple(build_support_octree(lower[b.edge_offset:b.edge_offset+b.basis.n_edges],
            upper[b.edge_offset:b.edge_offset+b.basis.n_edges], leaf_size=leaf_size) for b in provider.system.bodies)
        order_plan = None
        if max_order is not None:
            regions = ((global_tree, provider.exterior), *tuple((tree, body.interior)
                       for tree, body in zip(body_trees, provider.system.bodies)))
            statistics, impedance = None, None
            if rwg_absolute_budget is not None:
                logs = tuple(rwg_integral_logs(body.basis, geometry)
                             for body, geometry in zip(provider.system.bodies, provider.geometry))
                statistics = (integral_statistics_from_logs(global_tree, torch.cat(logs)),
                              *tuple(integral_statistics_from_logs(tree, values) for tree, values in zip(body_trees, logs)))
                impedance = abs(provider.exterior.eta)
            order_plan = plan_uniform_kernel_order(regions, kernel_error_budget, min_order=order,
                max_order=max_order, admissibility=admissibility, kernel_budget_bytes=kernel_budget_bytes,
                rwg_statistics=statistics, rwg_absolute_budget=rwg_absolute_budget, balancing_impedance=impedance)
            order = order_plan['selected_order']
            if rwg_absolute_budget is not None:
                required = max((int((((tree.upper-tree.lower)*.5 > 0).sum(-1)
                    [(tree.children < 0).all(-1)]).max())*(order-1)+4)//2
                    for tree in (global_tree, *body_trees))
                order_plan['required_moment_quadrature_order'] = required
                if quadrature_order is not None and (type(quadrature_order) is not int or quadrature_order < required):
                    raise ValueError('RWG absolute admission requires polynomial-exact moment quadrature')
                del statistics, logs
        capacity = plan_direct_storage((global_tree, *body_trees), order=order,
            admissibility=admissibility, budget_bytes=action_retained_budget_bytes)
        if not capacity['admitted']:
            raise DirectCapacityError(capacity)
        self._capacity_text = json.dumps(capacity, sort_keys=True, allow_nan=False)
        global_moments = NestedRWGMoments(provider, global_tree, order=order, quadrature_order=quadrature_order)
        self.exterior = DirectRegionalOperator(global_moments, global_moments, (provider.exterior,),
            integration=provider.config, admissibility=admissibility, kernel_budget_bytes=kernel_budget_bytes,
            kernel_error_budget=kernel_error_budget)
        kernel_bytes = self.exterior.kernel_bytes
        projections, indices, interiors = [], [], []
        for body, geometry, tree in zip(provider.system.bodies, provider.geometry, body_trees):
            start, stop = body.edge_offset, body.edge_offset+body.basis.n_edges
            moments = NestedRWGMoments(body.basis, tree, order=order,
                quadrature_order=quadrature_order, geometry=geometry)
            projections.append(moments)
            edges = torch.arange(start, stop, device=self.device)
            indices.append(torch.cat((edges, edges+provider.system.total_edges)))
            remaining = kernel_budget_bytes-kernel_bytes
            interior = DirectRegionalOperator(moments, moments, (body.interior,),
                integration=provider.config, admissibility=admissibility,
                kernel_budget_bytes=max(1, remaining), kernel_error_budget=kernel_error_budget)
            if interior.kernel_bytes > remaining:
                raise MemoryError('Cumulative system kernel table budget exceeded')
            kernel_bytes += interior.kernel_bytes
            interiors.append(interior)
        self.projections, self.indices, self.interiors = tuple(projections), tuple(indices), tuple(interiors)
        self.kernel_bytes = kernel_bytes
        if order_plan is not None and kernel_bytes != order_plan['kernel_table_bytes']:
            raise RuntimeError('Constructed kernel tables differ from the preallocation plan')
        self._order_plan_text = json.dumps(order_plan, sort_keys=True, allow_nan=False)
        self._binding = (id(provider), self.shape, self.device, self.dtype, id(self.exterior),
                         tuple(map(id, self.projections)), tuple(map(id, self.interiors)), self._order_plan_text, self._capacity_text)
        self._versions = tuple((id(t), t._version) for t in self.indices)
        if storage_bytes(direct_action_tensors(self)) != capacity['planned_bytes']:
            raise RuntimeError('Direct action storage differs from preallocation plan')

    def assert_current(self):
        if ((id(self.provider), self.shape, self.device, self.dtype, id(self.exterior),
             tuple(map(id, self.projections)), tuple(map(id, self.interiors)), self._order_plan_text, self._capacity_text) != self._binding
                or tuple((id(t), t._version) for t in self.indices) != self._versions):
            raise ValueError('Direct system actions changed')
        self.provider.assert_current()
        self.exterior.assert_current()
        for block in self.interiors:
            block.assert_current()

    @property
    def order_plan(self):
        return json.loads(self._order_plan_text)

    @property
    def capacity_plan(self):
        return json.loads(self._capacity_text)

    def _action(self, value, *, adjoint):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.shape[0]
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('System actions require finite matching complex coefficients')
        result = self.exterior.adjoint(value) if adjoint else self.exterior.apply(value)
        for indices, interior in zip(self.indices, self.interiors):
            local = interior.adjoint(value[..., indices]) if adjoint else interior.apply(value[..., indices])
            result = result.index_add(-1, indices, local)
        return result

    def apply(self, value):
        if torch.is_grad_enabled() and isinstance(value, torch.Tensor) and value.requires_grad:
            return _FixedDirectAction.apply(value, self, False)
        return self._action(value, adjoint=False)

    def adjoint(self, value):
        if torch.is_grad_enabled() and isinstance(value, torch.Tensor) and value.requires_grad:
            return _FixedDirectAction.apply(value, self, True)
        return self._action(value, adjoint=True)

    def receipt(self):
        self.assert_current()
        return {'version': 'direct_global_exterior_scalar_system_v2', 'shape': list(self.shape),
                'bodies': len(self.projections), 'kernel_table_bytes': self.kernel_bytes,
                'order_plan': self.order_plan, 'capacity_plan': self.capacity_plan,
                'moment_quadrature': {'exterior': self.exterior.test.moment_quadrature_receipt(),
                    'interiors': [p.moment_quadrature_receipt() for p in self.projections]},
                'far_projection_passes': 'one_global_exterior_plus_body_local_interiors',
                'exterior': self.exterior.receipt(), 'interiors': [b.receipt() for b in self.interiors],
                'current_backward': 'fixed_operator_adjoint_no_saved_action_intermediates',
                'reference_h_conversion': False, 'interpolation_error_certified': False,
                'physical_qualification': False, 'complexity_qualification': False}
