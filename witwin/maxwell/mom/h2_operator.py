"""Nested H2 adapter for the shared surface Krylov and Riesz protocol."""
from dataclasses import asdict
import copy
import hashlib
import torch
from .h2 import NestedH2Actions, H2Config
from .hmatrix import HierarchicalOperator
from .operator_protocol import IterativeState


class HierarchicalH2Operator:
    """Convert a verified H partition and release its per-block far factors.

    The provider, local Riesz metric, tree and near blocks are shared explicitly.
    The source H object is not retained. Only this partition-preserving factory
    enables disjoint-block norm sums; arbitrary overlapping far actions must not
    be wrapped as though their squared norms were additive.
    """
    supports_direct = False
    # Residual sampling, Krylov admission and scaling conventions are identical
    # across H and H2; reuse their common algorithms, not copied implementations.
    _vector = HierarchicalOperator._vector
    read_access = HierarchicalOperator.read_access
    riesz_inverse = HierarchicalOperator.riesz_inverse
    preconditioner_scales = HierarchicalOperator.preconditioner_scales
    residual_report = HierarchicalOperator.residual_report
    solve = HierarchicalOperator.solve

    def __init__(self, operator, config=H2Config()):
        if not isinstance(operator, HierarchicalOperator):
            raise TypeError('H2 solver conversion requires a verified hierarchical partition')
        operator.refresh()
        self.actions = NestedH2Actions.from_hmatrix(operator, config)
        self.provider, self.metric = operator.provider, operator.metric
        self.config, self.source_configuration = config, operator.config
        self.tree = self.actions.tree
        self.shape, self.device, self.dtype = self.actions.shape, self.actions.device, self.actions.dtype
        self.iterative_state = IterativeState()
        self._bindings = (id(self.actions), id(self.provider), id(self.metric), id(self.tree),
                          self.config, self.source_configuration, self.shape, self.device, self.dtype)
        self._apply_calls = self._adjoint_calls = 0
        self.metadata = {'version': 'scalar_nested_h2_operator_v1', 'operator_backend': 'h2',
            'configuration': config.payload(), 'source_h_configuration': self.source_configuration.payload(),
            'quadrature': asdict(self.provider.config), 'physical_qualification': False,
            'source_h_diagnostics': copy.deepcopy({key: operator.metadata[key] for key in
                ('exact_blocks', 'far_blocks', 'compression_rejections')}),
            'global_riesz_error_verified': False, 'setup_complexity_qualified': False,
            'residual_metric': 'impedance_balanced_L2_dual_proxy', 'trace_norm_certified': False}
        operator.refresh()
        HierarchicalH2Operator.refresh(self)

    def refresh(self):
        if ((id(self.actions), id(self.provider), id(self.metric), id(self.tree), self.config,
             self.source_configuration, self.shape, self.device, self.dtype) != self._bindings
                or self.actions.config != self.config or self.actions.tree is not self.tree
                or self.metric.system is not self.provider.system):
            raise ValueError('H2 solver provider, metric or numerical binding changed')
        self.actions.refresh()
        self.provider.assert_current()
        self.metric.assert_current()

    def apply(self, value):
        with self.read_access():
            result = self.actions.apply(value)
            self._apply_calls += 1
            return result

    def adjoint(self, value):
        with self.read_access():
            result = self.actions.adjoint(value)
            self._adjoint_calls += 1
            return result

    def _squared_norms(self, *, columns=False, column_scale=None):
        self.refresh()
        h2 = self.actions
        if column_scale is None:
            column_scale = torch.ones_like(h2.column_scale)
        trial_weight = (column_scale/h2.column_scale).square()
        test_weight = h2.row_scale.reciprocal().square()
        incoming, outgoing = (h2.test_basis, h2.trial_basis) if columns else (h2.trial_basis, h2.test_basis)
        grams = incoming.weighted_grams(test_weight if columns else trial_weight)
        forms = [torch.zeros((rank, rank), device=self.device, dtype=self.dtype) for rank in outgoing.ranks]
        for a, b, coupling in h2.couplings:
            if columns:
                forms[b] = forms[b]+coupling.mH@grams[a]@coupling
            else:
                forms[a] = forms[a]+coupling@grams[b]@coupling.mH
        result = outgoing.diagonal_forms(forms)*(trial_weight if columns else test_weight)
        if h2.near is not None:
            for rows, cols, blocks in h2.near.groups:
                for start in range(0, len(blocks), h2.near.action_batch):
                    stop = start+h2.near.action_batch
                    values = blocks[start:stop]*column_scale[cols[start:stop]][:, None, :]
                    values = values.abs().square().sum(1 if columns else 2)
                    target = cols[start:stop] if columns else rows[start:stop]
                    result.index_add_(0, target.flatten(), values.flatten())
        if not bool(torch.isfinite(result).all()):
            raise ValueError('H2 equilibration norms overflowed native precision')
        return result.clamp_min(0)

    def storage_breakdown(self):
        self.refresh()
        action = self.actions.receipt()
        metric_bytes = sum(t.numel()*t.element_size() for t in self.metric.tensors())
        geometry_bytes = sum(t.numel()*t.element_size() for t in self.provider.geometry_tensors())
        inputs = self.provider.input_tensors()
        input_bytes = sum(t.untyped_storage().nbytes() for t in inputs)
        return {'action': action, 'shared_metric_bytes': metric_bytes,
                'shared_prepared_geometry_bytes': geometry_bytes, 'shared_input_bytes': input_bytes,
                'total_referenced_tensor_bytes': action['total_referenced_tensor_bytes']+metric_bytes+geometry_bytes+input_bytes,
                'excludes': ['allocator workspace/cache', 'host control/metadata objects', 'solver vectors',
                             'provider spectral caches under separate kernel budgets'],
                'peak_memory_qualified': False}

    def persistence_receipt(self):
        self.refresh()
        def digest(value):
            return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        h2 = self.actions
        groups = {'action': h2.tensors(), 'test_basis': h2.test_basis.tensors(),
                  'trial_basis': h2.trial_basis.tensors(), 'tree': self.tree.tensors(),
                  'metric': self.metric.tensors(), 'near': () if h2.near is None else h2.near.tensors()}
        return {'version': self.metadata['version'], 'configuration': self.config.payload(),
                'source_h_configuration': self.source_configuration.payload(),
                'coupling_nodes': [(a, b) for a, b, _ in h2.couplings],
                'tensor_hashes': {name: [digest(t) for t in tensors] for name, tensors in groups.items()},
                'executable_snapshot': False}

    def operator_stats(self):
        self.refresh()
        return {**self.metadata, 'apply_calls': self._apply_calls, 'adjoint_calls': self._adjoint_calls,
                'storage': self.storage_breakdown()}
