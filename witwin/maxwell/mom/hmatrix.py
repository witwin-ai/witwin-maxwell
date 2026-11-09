"""Experimental scalar H-matrix with exact leaves and verified far factors."""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import math
import torch
from .cluster import build_support_octree, support_block_partition
from .nearfield import ScalarEdgeBlockSampler
from .lowrank import streamed_cross_approximation, CompressionFailure, LowRankBlock
from .riesz import LocalRWGRiesz
from .packed_near import PackedNearBlocks
from .operator_protocol import IterativeState


@dataclass(frozen=True)
class HMatrixConfig:
    leaf_size: int = 32
    max_rank: int = 32
    tolerance: float = 1e-8
    admissibility: float = 1.
    row_batch: int = 8
    solve_dual_tolerance: float = 1e-8
    original_sample_tolerance: float = 1e-8
    retained_budget_bytes: int = 1024**3
    pivoting: str = 'complete'

    def __post_init__(self):
        if self.pivoting not in ('complete', 'adaptive'):
            raise ValueError('pivoting must be complete or adaptive')
        for name in ('leaf_size', 'max_rank', 'row_batch', 'retained_budget_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name+' must be a positive integer')
        for name in ('tolerance', 'admissibility', 'solve_dual_tolerance', 'original_sample_tolerance'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be positive and finite')


    def payload(self):
        return {'version': 'scalar_hmatrix_config_v2', 'parameters': asdict(self)}

    @classmethod
    def restore(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {'version', 'parameters'}
                or payload['version'] != 'scalar_hmatrix_config_v2'
                or not isinstance(payload['parameters'], dict)
                or set(payload['parameters']) != set(cls.__dataclass_fields__)):
            raise ValueError('Unsupported or incomplete hierarchical configuration')
        return cls(**payload['parameters'])


class HierarchicalOperator:
    """A shared fixed scalar block object for forward, adjoint and Krylov solves.

    Setup is the streamed reference cross algorithm, not a fast setup claim.
    The current byte budget bounds retained data; full setup/allocator peak
    preflight is a separate, unfinished production requirement. No public
    Scene backend is enabled by constructing this internal reference object.
    """
    supports_direct = False

    def __init__(self, provider, config=HMatrixConfig(), *, metric_impedance=None):
        if not isinstance(config, HMatrixConfig):
            raise TypeError('Hierarchical operator requires HMatrixConfig')
        provider.assert_current()
        self.provider, self.config = provider, config
        self._bindings = (provider, config)
        self.device = provider.system.device
        self.dtype = torch.complex128 if provider.system.dtype == torch.float64 else torch.complex64
        self.shape = (2*provider.system.total_edges,)*2
        self.tree = build_support_octree(*provider.support_boxes(), leaf_size=config.leaf_size)
        if self.tree.receipt()['oversized_leaves']:
            raise ValueError('Coincident Morton supports exceed the exact leaf limit; refine the geometric partition')
        if metric_impedance is None:
            metric_impedance = getattr(provider.exterior, 'eta', None)
        if metric_impedance is None:
            raise ValueError('Tensor exterior hierarchy requires an explicit residual proxy impedance')
        self.metric = LocalRWGRiesz(provider.system, metric_impedance)
        self.iterative_state = IterativeState()
        self.blocks = []
        self.block_nodes = []
        near_records = []
        self._apply_calls = self._adjoint_calls = 0
        geometry_bytes = sum(t.numel()*t.element_size() for t in provider.geometry_tensors())
        retained = self.tree.receipt()['retained_bytes']+self.metric.receipt()['retained_bytes']+geometry_bytes
        failures = []
        total = provider.system.total_edges
        for aa, bb, admissible in support_block_partition(self.tree, admissibility=config.admissibility):
            # Host values here control tree traversal; all field/quadrature
            # and compression operations remain on CUDA.
            pending = [(int(a), int(b), bool(f)) for a, b, f in zip(aa, bb, admissible)]
            while pending:
                a, b, far = pending.pop()
                rows = self.tree.permutation[self.tree.starts[a]:self.tree.stops[a]]
                cols = self.tree.permutation[self.tree.starts[b]:self.tree.stops[b]]
                value = None
                if far:
                    sampler = ScalarEdgeBlockSampler(provider, rows, cols)
                    eta = self.metric.impedance_squared**.5
                    rs = torch.cat((torch.full((len(rows),), eta, device=self.device, dtype=provider.system.dtype),
                                    torch.ones(len(rows), device=self.device, dtype=provider.system.dtype)))
                    cs = torch.cat((torch.ones(len(cols), device=self.device, dtype=provider.system.dtype),
                                    torch.full((len(cols),), eta, device=self.device, dtype=provider.system.dtype)))
                    try:
                        value = streamed_cross_approximation(sampler.sample_rows, sampler.sample_column,
                            sampler.shape, device=self.device, dtype=self.dtype, max_rank=config.max_rank,
                            rtol=config.tolerance, row_batch=config.row_batch, row_scale=rs, column_scale=cs,
                            pivoting=config.pivoting)
                    except CompressionFailure as failure:
                        failures.append({'test_node': a, 'source_node': b, 'reason': str(failure), **failure.receipt})
                        ac = self.tree.children[a][self.tree.children[a] >= 0]
                        bc = self.tree.children[b][self.tree.children[b] >= 0]
                        if len(ac) or len(bc):
                            if len(ac) and (not len(bc) or len(rows) >= len(cols)):
                                pending.extend((int(child), b, True) for child in ac)
                            else:
                                pending.extend((a, int(child), True) for child in bc)
                            continue
                ri, ci = torch.cat((rows, rows+total)), torch.cat((cols, cols+total))
                index_bytes = (ri.numel()+ci.numel())*ri.element_size()
                if value is None:
                    if len(rows) > config.leaf_size or len(cols) > config.leaf_size:
                        raise RuntimeError('Exact block exceeds the declared leaf bound')
                    block_bytes = 4*len(rows)*len(cols)*torch.empty((), dtype=self.dtype).element_size()
                    if retained+index_bytes+block_bytes > config.retained_budget_bytes:
                        raise MemoryError('Hierarchical retained-data budget exceeded before exact leaf allocation')
                    value = provider(rows, cols)
                else:
                    block_bytes = value.left.numel()*value.left.element_size()+value.right.numel()*value.right.element_size()
                retained += index_bytes+block_bytes
                if retained > config.retained_budget_bytes:
                    raise MemoryError('Hierarchical retained-data budget exceeded by compressed factors')
                self.blocks.append((ri, ci, value))
                self.block_nodes.append((a, b))
                if not isinstance(value, LowRankBlock):
                    near_records.append((a, b, ri, ci, value))
        self.packed_near = None
        if near_records:
            near_bytes = sum(r.numel()*r.element_size()+c.numel()*c.element_size()+v.numel()*v.element_size()
                             for _, _, r, c, v in near_records)
            # This repacking transiently holds both assembly inputs and packed
            # tensors. The retained budget is not a peak workspace guarantee.
            self.packed_near = PackedNearBlocks(near_records, self.shape[0],
                retained_budget_bytes=config.retained_budget_bytes-(retained-near_bytes))
            exact_views = []
            for _, _, gi, bi in self.packed_near.entries:
                rows, cols, values = self.packed_near.groups[gi]
                exact_views.append((rows[bi], cols[bi], values[bi]))
            self.block_nodes = [(a, b) for a, b, _, _ in self.packed_near.entries]+[
                nodes for nodes, entry in zip(self.block_nodes, self.blocks) if isinstance(entry[2], LowRankBlock)]
            self.blocks = exact_views+[entry for entry in self.blocks if isinstance(entry[2], LowRankBlock)]
            retained += self.packed_near.receipt()['retained_bytes']-near_bytes
        self.blocks = tuple(self.blocks)
        self.block_nodes = tuple(self.block_nodes)
        self._block_nodes = self.block_nodes
        self._packed_near = self.packed_near
        self._tree, self._metric = self.tree, self.metric
        self._versions = tuple((id(t), t._version) for t in self._block_tensors())
        self._layout = tuple((id(r), id(c), id(v)) for r, c, v in self.blocks)
        self.metadata = {'version': 'scalar_hmatrix_reference_v1', 'operator_backend': 'hmatrix',
            'configuration': asdict(config), 'retained_bytes': retained, 'prepared_geometry_bytes': geometry_bytes,
            'exact_blocks': sum(not isinstance(v, LowRankBlock) for _, _, v in self.blocks),
            'far_blocks': sum(isinstance(v, LowRankBlock) for _, _, v in self.blocks),
            'compression_rejections': failures, 'physical_qualification': False,
            'exact_storage': None if self.packed_near is None else self.packed_near.receipt(),
            'global_riesz_error_verified': False, 'setup_complexity_qualified': False}

    def _block_tensors(self):
        if self.packed_near is not None:
            yield from self.packed_near.tensors()[:4]
        for rows, cols, value in self.blocks:
            yield rows
            yield cols
            if isinstance(value, LowRankBlock):
                yield value.left
                yield value.right
            else:
                yield value

    def refresh(self):
        if self.provider is not self._bindings[0] or self.config is not self._bindings[1]:
            raise ValueError('Hierarchical provider or configuration changed')
        if self.tree is not self._tree or self.metric is not self._metric:
            raise ValueError('Hierarchical tree or metric changed')
        self.provider.assert_current()
        self.tree.assert_current()
        self.metric.assert_current()
        if self.packed_near is not self._packed_near:
            raise ValueError('Hierarchical packed near storage changed')
        if self.packed_near is not None:
            self.packed_near.assert_current()
        if (tuple((id(r), id(c), id(v)) for r, c, v in self.blocks) != self._layout
                or self.block_nodes != self._block_nodes
                or tuple((id(t), t._version) for t in self._block_tensors()) != self._versions):
            raise ValueError('Hierarchical blocks changed after construction')

    @contextmanager
    def read_access(self):
        self.refresh()
        try:
            yield
        finally:
            self.refresh()

    def _vector(self, value):
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.shape[0]
                or value.dtype != self.dtype or value.device != self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('Hierarchical vectors require finite matching complex CUDA coefficients')

    def apply(self, value):
        with self.read_access():
            self._vector(value)
            result = torch.zeros_like(value) if self.packed_near is None else self.packed_near.apply(value)
            for rows, cols, block in self.blocks:
                if not isinstance(block, LowRankBlock):
                    continue
                contribution = block.apply(value[..., cols])
                result.index_add_(-1, rows, contribution)
            self._apply_calls += 1
            return result

    def adjoint(self, value):
        with self.read_access():
            self._vector(value)
            result = torch.zeros_like(value) if self.packed_near is None else self.packed_near.adjoint(value)
            for rows, cols, block in self.blocks:
                if not isinstance(block, LowRankBlock):
                    continue
                contribution = block.adjoint(value[..., rows])
                result.index_add_(-1, cols, contribution)
            self._adjoint_calls += 1
            return result

    def riesz_inverse(self, value):
        with self.read_access():
            return self.metric.inverse(value)

    def _squared_norms(self, *, columns=False, column_scale=None):
        result = torch.zeros(self.shape[0], device=self.device, dtype=self.tree.lower.dtype)
        for rows, cols, block in self.blocks:
            if isinstance(block, LowRankBlock):
                left, right = block.left, block.right
                if column_scale is not None:
                    right = right*column_scale[cols][None, :]
                if columns:
                    values = (right.conj()*((left.mH@left)@right)).sum(0).real
                else:
                    values = (left.conj()*(left@(right@right.mH))).sum(1).real
            else:
                values = block if column_scale is None else block*column_scale[cols][None, :]
                values = values.abs().square().sum(0 if columns else 1)
            result.index_add_(0, cols if columns else rows, values.clamp_min(0))
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Hierarchical norm scaling exceeds native precision')
        return result

    def preconditioner_scales(self, kind):
        with self.read_access():
            one = torch.ones(self.shape[0], device=self.device, dtype=self.tree.lower.dtype)
            def inverse_norm(squared):
                norms = squared.sqrt()
                return torch.where(norms > 0, norms.reciprocal(), one)
            if kind == 'none':
                return one, one, False
            if kind == 'row_norm':
                return inverse_norm(self._squared_norms()), one, False
            if kind == 'row_column_norm':
                col = inverse_norm(self._squared_norms(columns=True))
                return inverse_norm(self._squared_norms(column_scale=col)), col, False
            raise ValueError('Unsupported hierarchical preconditioner')

    def residual_report(self, value, rhs, *, original_edges=None, source_batch=64):
        """Separate full compressed dual residual from sampled original rows.

        Original sampled rows use explicit impedance-balanced Euclidean norms;
        they cannot certify a global Riesz residual or unobserved rows.
        """
        with self.read_access():
            self._vector(value)
            self._vector(rhs)
            if rhs.shape != value.shape:
                raise ValueError('Residual report requires matching solution/RHS shapes')
            residual = self.apply(value)-rhs
            def dual(x):
                return (x.conj()*self.metric.inverse(x)).sum(-1).real.clamp_min(0).sqrt()
            own_abs, own_rhs = dual(residual), dual(rhs)
            total = self.shape[0]//2
            if original_edges is None:
                original_edges = torch.linspace(0, total-1, min(8, total), device=self.device).round().to(torch.int64).unique()
            self.provider._validate_selection(original_edges)
            indices = torch.cat((original_edges, original_edges+total))
            original = self.provider.sampled_action(original_edges, value, source_batch=source_batch)-rhs[..., indices]
            scale = torch.cat((torch.full((len(original_edges),), self.metric.impedance_squared**.5,
                device=self.device, dtype=self.tree.lower.dtype), torch.ones(len(original_edges),
                device=self.device, dtype=self.tree.lower.dtype)))
            absolute = torch.linalg.vector_norm(original*scale, dim=-1)
            reference = torch.linalg.vector_norm(rhs[..., indices]*scale, dim=-1)
            def record(a, b):
                aa, bb = a.detach().flatten().cpu().tolist(), b.detach().flatten().cpu().tolist()
                return {'absolute': aa, 'rhs_norm': bb,
                        'relative': [x/y if y > 0 else (0. if x == 0 else None) for x, y in zip(aa, bb)]}
            return {'version': 'hierarchical_original_residual_v1',
                'compressed_dual': record(own_abs, own_rhs),
                'original_sampled': record(absolute, reference),
                'original_edges': original_edges.detach().cpu().tolist(),
                'original_metric': 'sampled_impedance_balanced_euclidean',
                'original_global_residual_certified': False, 'physical_qualification': False}

    def solve(self, rhs, stopping=None, *, original_edges=None, source_batch=64, right_inverse=None,
              right_preconditioner=None):
        """Admit an inner solution only after both declared residual checks.

        A passing original-row sample is explicitly not a certificate for the
        full original operator. Physical qualification remains separate.
        """
        from .linalg import solve_reference, StoppingRule
        if stopping is None:
            stopping = StoppingRule(method='gmres', preconditioner='row_column_norm', rtol=1e-12, atol=0.)
        target = self
        if right_inverse is not None and right_preconditioner is not None:
            raise ValueError('Use either the fixed leaf right inverse or a flexible right preconditioner')
        if right_inverse is not None:
            from .block_preconditioner import RightPreconditionedOperator
            target = RightPreconditionedOperator(self, right_inverse)
        if right_preconditioner is None:
            value, inner = solve_reference(target, rhs, stopping)
        else:
            value, inner = solve_reference(target, rhs, stopping, right_preconditioner=right_preconditioner)
        if right_inverse is not None:
            value = right_inverse.apply(value)
            inner['leaf_preconditioner'] = right_inverse.receipt()
            inner['krylov_coordinates'] = 'right_inverse_equation_coordinates'
            inner['returned_coordinates'] = 'physical_scattered_currents'
        log = {**inner, 'inner_converged': inner['converged'],
            'original_global_residual_certified': False, 'physical_qualification': False,
            'error_budget': {'version': 'hierarchical_residual_admission_v1',
                'compressed_dual_rtol': self.config.solve_dual_tolerance,
                'original_sampled_rtol': self.config.original_sample_tolerance,
                'checked': False}}
        if not inner['converged']:
            return value, log
        report = self.residual_report(value, rhs, original_edges=original_edges, source_batch=source_batch)
        def accepted(record, tolerance):
            return all(v is not None and math.isfinite(v) and v <= tolerance for v in record['relative'])
        compressed_pass = accepted(report['compressed_dual'], self.config.solve_dual_tolerance)
        original_pass = accepted(report['original_sampled'], self.config.original_sample_tolerance)
        log['error_budget'].update(checked=True, compressed_dual_pass=compressed_pass,
                                  original_sampled_pass=original_pass, residual_report=report)
        log['converged'] = compressed_pass and original_pass
        if not compressed_pass:
            log['reason'] = 'compressed_dual_residual_above_budget'
        elif not original_pass:
            log['reason'] = 'original_sampled_residual_above_budget'
        return value, log

    def persistence_receipt(self):
        self.refresh()
        def digest(t):
            return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        return {'version': self.metadata['version'], 'configuration': asdict(self.config),
            'exact_storage': None if self.packed_near is None else self.packed_near.receipt(),
            'near_csr_hashes': [] if self.packed_near is None else
                [digest(t) for t in self.packed_near.tensors()[:4]],
            'blocks': [[digest(t) for t in ((r, c, v.left, v.right) if isinstance(v, LowRankBlock) else (r, c, v))]
                       for r, c, v in self.blocks], 'metric': self.metric.receipt(),
            'metric_tensor_hashes': [digest(t) for t in self.metric.tensors()],
            'tree_tensor_hashes': [digest(t) for t in self.tree.tensors()]}

    def storage_breakdown(self):
        self.refresh()
        def size(tensors):
            return sum(t.numel()*t.element_size() for t in tensors)
        exact = [value for _, _, value in self.blocks if not isinstance(value, LowRankBlock)]
        factors = [t for _, _, value in self.blocks if isinstance(value, LowRankBlock)
                   for t in (value.left, value.right)]
        indices = [t for rows, cols, _ in self.blocks for t in (rows, cols)]
        categories = {'exact_blocks': exact, 'far_factors': factors, 'block_indices': indices,
            'near_csr': [] if self.packed_near is None else self.packed_near.tensors()[:4],
            'tree': self.tree.tensors(), 'metric': self.metric.tensors(),
            'prepared_geometry': self.provider.geometry_tensors()}
        own = {name: {'tensors': len(tensors), 'bytes': size(tensors)} for name, tensors in categories.items()}
        inputs = self.provider.input_tensors()
        shared = sum(t.untyped_storage().nbytes() for t in inputs)
        owned = sum(record['bytes'] for record in own.values())
        return {'owned': own, 'owned_retained_bytes': owned,
            'shared_input_tensors': len(inputs), 'shared_input_bytes': shared,
            'total_referenced_tensor_bytes': owned+shared,
            'budget_scope': 'owned_retained_bytes',
            'excludes': ['allocator workspace/cache', 'host control/metadata objects', 'solver vectors',
                             'provider spectral caches under separate kernel budgets']}

    def operator_stats(self):
        self.refresh()
        return {**self.metadata, 'apply_calls': self._apply_calls, 'adjoint_calls': self._adjoint_calls,
                'tree': self.tree.receipt(), 'metric': self.metric.receipt(), 'storage': self.storage_breakdown()}
