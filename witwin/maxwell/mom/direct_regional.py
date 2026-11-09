"""Nested direct-kernel regional actions with exact near-block integration."""
import torch
from dataclasses import asdict
from .interpolation_error import KernelErrorBudget
from .nested_moments import NestedRWGMoments
from .interpolation_kernel import InterpolationKernel
from .cluster import support_block_partition
from .nearfield import scalar_regional_block, ScalarSystemBlocks
from .packed_near import PackedNearBlocks
from .operator import IntegrationConfig
from .physics import Medium


class DirectRegionalOperator:
    """Regional sum on body or shared exterior trees, in all-J/all-M ordering.

    Fixed-order interpolation is explicit and not an error certificate. Near
    blocks reuse original weak quadrature. Kernel budgets cover cumulative far
    tables only; tree moments, exact blocks and setup/solve workspaces are separate.
    """
    def __init__(self, test, source, media, *, integration=IntegrationConfig(),
                 admissibility=1., pair_batch=32, kernel_row_batch=32,
                 kernel_budget_bytes=512*1024**2, kernel_error_budget=None):
        if not isinstance(test, NestedRWGMoments) or not isinstance(source, NestedRWGMoments):
            raise TypeError('Direct regional operators require nested RWG moments')
        test.assert_current()
        source.assert_current()
        if test.device != source.device or test.dtype != source.dtype:
            raise ValueError('Regional moment spaces must share device and precision')
        media = tuple(media)
        if not media or any(not isinstance(m, Medium) for m in media):
            raise ValueError('Regional actions require sampled scalar media')
        if any(m.frequency != media[0].frequency for m in media):
            raise ValueError('Regional media must share one frequency')
        if test.basis is not source.basis and len(media) != 1:
            raise ValueError('Cross-body blocks contain only the common exterior medium')
        global_space = isinstance(test.basis, ScalarSystemBlocks) or isinstance(source.basis, ScalarSystemBlocks)
        if global_space and (test.basis is not source.basis or len(media) != 1
                             or media[0] is not test.basis.exterior or integration is not test.basis.config):
            raise ValueError('Global region requires one shared exterior provider and its integration')
        if type(kernel_budget_bytes) is not int or kernel_budget_bytes < 1:
            raise ValueError('Kernel table budget must be a positive integer')
        if kernel_error_budget is not None and not isinstance(kernel_error_budget, KernelErrorBudget):
            raise TypeError('Regional kernel admission requires KernelErrorBudget')
        self.kernel_error_budget = kernel_error_budget
        self.test, self.source, self.media, self.integration = test, source, media, integration
        self.device, self.dtype = test.device, test.dtype
        self.shape = (2*test.tree.size, 2*source.tree.size)
        far, exact, kernel_bytes = [], [], 0
        for aa, bb, flags in support_block_partition(test.tree, source.tree,
                admissibility=admissibility, pair_batch=pair_batch):
            for a, b, admissible in zip(aa.cpu().tolist(), bb.cpu().tolist(), flags.cpu().tolist()):
                if admissible:
                    for medium in media:
                        remaining = kernel_budget_bytes-kernel_bytes
                        if remaining <= 0:
                            raise MemoryError('Cumulative far kernel table budget exceeded')
                        kernel = InterpolationKernel(test.boxes[a], source.boxes[b], medium,
                            admissibility=admissibility, kernel_row_batch=kernel_row_batch,
                            kernel_budget_bytes=remaining, error_budget=kernel_error_budget)
                        far.append((a, b, kernel))
                        kernel_bytes += kernel.kernel_bytes
                else:
                    rows = test.tree.permutation[test.starts[a]:test.stops[a]]
                    cols = source.tree.permutation[source.starts[b]:source.stops[b]]
                    # A rectangular map is embedded into the shared square packed
                    # action primitive. Node namespaces and coefficient spaces are disjoint.
                    rr = torch.cat((rows, rows+test.tree.size))
                    cc = torch.cat((cols, cols+source.tree.size))+self.shape[0]
                    exact.append((a, b+test.tree.node_count, rr, cc))
        self.far = tuple(far)
        def near_block(a, b, rows, cols):
            rows = rows[:len(rows)//2]
            cols = cols[:len(cols)//2]-self.shape[0]
            if global_space:
                return test.basis.exterior_block(rows, cols)
            block, _ = scalar_regional_block(test.basis, source.basis, rows, cols,
                media, integration, geometry=(test.geometry, source.geometry))
            return block
        self.near = PackedNearBlocks.from_layout(exact, sum(self.shape), near_block,
            device=self.device, dtype=self.dtype) if exact else None
        self.kernel_bytes = kernel_bytes
        self._binding = (id(test), id(source), tuple(map(id, media)), integration,
                         self.shape, self.device, self.dtype, id(self.near),
                         tuple((a, b, id(k)) for a, b, k in self.far), self.kernel_error_budget)
        self._media = tuple((m.frequency, m.epsilon_r, m.mu_r, m.k, m.eta) for m in media)

    def assert_current(self):
        if ((id(self.test), id(self.source), tuple(map(id, self.media)), self.integration,
             self.shape, self.device, self.dtype, id(self.near),
             tuple((a, b, id(k)) for a, b, k in self.far), self.kernel_error_budget) != self._binding
                or tuple((m.frequency, m.epsilon_r, m.mu_r, m.k, m.eta) for m in self.media) != self._media):
            raise ValueError('Direct regional operator changed')
        self.test.assert_current()
        self.source.assert_current()
        if self.near is not None:
            self.near.assert_current()
        for _, _, kernel in self.far:
            kernel.assert_current()

    def _action(self, value, *, adjoint):
        self.assert_current()
        source, test = (self.test, self.source) if adjoint else (self.source, self.test)
        ns, nt = source.tree.size, test.tree.size
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != 2*ns
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Regional actions require finite matching complex coefficients')
        if self.far:
            moments = source.upward(torch.stack((value[..., :ns], value[..., ns:]), dim=-2))
            coefficients = self._local_coefficients(test, value)
            self._accumulate_far(moments, coefficients, adjoint=adjoint)
            result = test.downward(coefficients)
            result = torch.cat((result[..., 0, :], result[..., 1, :]), dim=-1)
        else:
            result = value.new_zeros((*value.shape[:-1], 2*nt))
        return result+self._exact_action(value, adjoint=adjoint)

    def _local_coefficients(self, test, value):
        return [value.new_zeros((*value.shape[:-1], 2, len(box.nodes), 4)) for box in test.boxes]

    def _accumulate_far(self, moments, coefficients, *, adjoint):
        # Internal composition hook: the caller validates all regional state.
        for a, b, kernel in self.far:
            target, origin = (b, a) if adjoint else (a, b)
            coefficients[target] = coefficients[target]+kernel.action(moments[origin], adjoint=adjoint)

    def _exact_action(self, value, *, adjoint):
        width = self.shape[1] if adjoint else self.shape[0]
        zeros = value.new_zeros((*value.shape[:-1], width))
        if self.near is not None:
            embedded = torch.cat((value, zeros), dim=-1) if adjoint else torch.cat((zeros, value), dim=-1)
            exact = self.near.action(embedded, adjoint=adjoint)
            return exact[..., self.shape[0]:] if adjoint else exact[..., :self.shape[0]]
        return zeros

    def apply(self, value):
        return self._action(value, adjoint=False)

    def adjoint(self, value):
        return self._action(value, adjoint=True)

    def receipt(self):
        self.assert_current()
        return {'version': 'direct_nested_scalar_regional_v1', 'shape': list(self.shape),
                'far_regional_tables': len(self.far), 'kernel_table_bytes': self.kernel_bytes,
                'kernel_truncation_budget': None if self.kernel_error_budget is None else asdict(self.kernel_error_budget),
                'kernel_truncation_scope': 'exact_arithmetic_kernel_only',
                'near': None if self.near is None else self.near.receipt(),
                'near_construction': 'stream_one_block_into_final_groups',
                'reference_h_conversion': False, 'interpolation_error_certified': False,
                'physical_qualification': False, 'complexity_qualification': False}
