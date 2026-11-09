"""Experimental algebraic H-to-H2 conversion with genuine nested actions.

Only leaf bases, parent/child transfers, small couplings and exact near blocks
are retained. Construction inherits the reference H setup cost; this module
does not advertise a fast setup, a solver backend or physical qualification.
"""
from dataclasses import dataclass, asdict
import math
import torch
from .h2_basis import NestedClusterBasis, NestedBasisConfig
from .lowrank import LowRankBlock
from .h2_capacity import plan_h2_storage, H2CapacityError


@dataclass(frozen=True)
class H2Config:
    max_rank: int = 64
    tolerance: float = 1e-10
    verification_row_batch: int = 16
    solve_dual_tolerance: float = 1e-8
    original_sample_tolerance: float = 1e-8
    retained_budget_bytes: int = 1024**3

    def __post_init__(self):
        NestedBasisConfig(self.max_rank, self.tolerance)
        if type(self.verification_row_batch) is not int or self.verification_row_batch < 1:
            raise ValueError('H2 verification batch must be a positive integer')
        if type(self.retained_budget_bytes) is not int or self.retained_budget_bytes < 1:
            raise ValueError('H2 retained_budget_bytes must be a positive integer')
        for name in ('solve_dual_tolerance', 'original_sample_tolerance'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be positive and finite')

    def payload(self):
        return {'version': 'nested_h2_config_v2', 'parameters': asdict(self)}

    @classmethod
    def restore(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {'version', 'parameters'}
                or payload['version'] != 'nested_h2_config_v2'
                or not isinstance(payload['parameters'], dict)
                or set(payload['parameters']) != set(cls.__dataclass_fields__)):
            raise ValueError('Unsupported or incomplete H2 configuration')
        return cls(**payload['parameters'])


class H2ConversionFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


class NestedH2Actions:
    """Balanced nested test/trial spaces and one shared forward/adjoint object.

    Far records are (test_node, source_node, LowRankBlock) in each cluster's
    component-major permutation order. Near storage is shared explicitly and
    guarded; the source H object and its far factors are not retained.
    """
    def __init__(self, tree, far_records, *, near=None, components=2,
                 dtype=torch.complex128, row_scale=None, column_scale=None,
                 config=H2Config()):
        tree.assert_current()
        if not isinstance(config, H2Config):
            raise TypeError('H2 actions require H2Config')
        if type(components) is not int or components < 1:
            raise ValueError('H2 components must be a positive integer')
        records = tuple(far_records)
        self.capacity_plan = plan_h2_storage(tree, records, components=components, dtype=dtype,
            max_rank=config.max_rank, budget_bytes=config.retained_budget_bytes)
        if not self.capacity_plan['admitted']:
            raise H2CapacityError(self.capacity_plan)
        self.tree, self.config, self.near = tree, config, near
        self.components, self.device, self.dtype = components, tree.device, dtype
        self.shape = (components*tree.size,)*2
        if near is not None:
            near.assert_current()
            if near.size != self.shape[0] or near.dtype != dtype or near.device != tree.device:
                raise ValueError('H2 exact near storage does not match its coefficient space')
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        def scale(value):
            if value is None:
                return torch.ones(self.shape[0], device=self.device, dtype=real_dtype)
            if (not isinstance(value, torch.Tensor) or value.shape != (self.shape[0],)
                    or value.device != self.device or value.dtype != real_dtype or value.requires_grad
                    or not bool(torch.isfinite(value).all() & (value > 0).all())):
                raise ValueError('H2 scaling requires positive finite fixed real CUDA vectors')
            return value.clone()
        self.row_scale, self.column_scale = scale(row_scale), scale(column_scale)
        starts, stops = tree.starts.cpu().tolist(), tree.stops.cpu().tolist()
        def indices(node):
            if type(node) is not int or not 0 <= node < tree.node_count:
                raise ValueError('H2 far node is outside the tree')
            points = tree.permutation[starts[node]:stops[node]]
            return torch.cat([points+k*tree.size for k in range(components)])
        test_seeds, trial_seeds, balanced = [], [], []
        seen = set()
        for a, b, block in records:
            if (a, b) in seen:
                raise ValueError('H2 far node pairs must be unique')
            seen.add((a, b))
            ri, ci = indices(a), indices(b)
            if not isinstance(block, LowRankBlock):
                raise TypeError('H2 conversion requires fixed low-rank input blocks')
            block.assert_current()
            left, right = block.left, block.right
            if (left.ndim != 2 or right.ndim != 2 or left.shape[0] != len(ri)
                    or right.shape[1] != len(ci) or left.shape[1] != right.shape[0]):
                raise ValueError('H2 far factors do not match their node maps')
            u = self.row_scale[ri, None]*left
            v = right*self.column_scale[ci][None, :]
            test_seeds.append((a, u))
            trial_seeds.append((b, v.mH))
            balanced.append((a, b, u, v))
        basis_config = NestedBasisConfig(config.max_rank, config.tolerance/8)
        self.test_basis = NestedClusterBasis(tree, test_seeds, components=components, dtype=dtype, config=basis_config)
        self.trial_basis = NestedClusterBasis(tree, trial_seeds, components=components, dtype=dtype, config=basis_config)
        couplings, errors = [], []
        for a, b, u, v in balanced:
            coupling = self.test_basis.project(a, u)@self.trial_basis.project(b, v.mH).mH
            # Verify the actual coupled approximation in bounded row batches;
            # checking only separate seed errors cannot bound an ill-conditioned
            # factor product. Expanded data here are tall rank-sized workspaces.
            test_factor = self.test_basis.expand(a, coupling)
            trial_factor = self.trial_basis.expand(b, torch.eye(self.trial_basis.ranks[b], device=self.device, dtype=dtype))
            norm = torch.zeros((), device=self.device, dtype=real_dtype)
            error = torch.zeros_like(norm)
            batch = min(config.verification_row_batch, max(1, len(u)//2))
            for start in range(0, len(u), batch):
                reference = u[start:start+batch]@v
                residual = reference-test_factor[start:start+batch]@trial_factor.mH
                norm = torch.hypot(norm, torch.linalg.vector_norm(reference))
                error = torch.hypot(error, torch.linalg.vector_norm(residual))
            error, norm = float(error), float(norm)
            receipt = {'test_node': a, 'source_node': b, 'absolute_error': error,
                       'balanced_block_norm': norm, 'relative_error': error/norm if norm else (0. if error == 0 else None),
                       'maximum_verification_rows': batch, 'tolerance': config.tolerance}
            if not math.isfinite(error) or not math.isfinite(norm) or error > config.tolerance*norm:
                raise H2ConversionFailure('Nested coupling exceeds the H-to-H2 error budget', receipt)
            couplings.append((a, b, coupling.clone()))
            errors.append(receipt)
        self.couplings, self.conversion_errors = tuple(couplings), tuple(errors)
        self._binding = (id(tree), id(near), id(self.test_basis), id(self.trial_basis),
                         config, components, self.device, dtype, self.shape)
        self._layout = tuple((a, b, id(c)) for a, b, c in self.couplings)
        self._versions = tuple((id(t), t._version) for t in self.tensors())
        retained = (sum(t.numel()*t.element_size() for t in self.tensors())
                    +self.test_basis.receipt()['retained_bytes']+self.trial_basis.receipt()['retained_bytes'])
        if retained > self.capacity_plan['retained_bytes_bound']:
            raise RuntimeError('H2 actual retained storage exceeded its structural bound')

    @classmethod
    def from_hmatrix(cls, operator, config=H2Config()):
        operator.refresh()
        if not isinstance(config, H2Config):
            raise TypeError('H2 conversion requires H2Config')
        far = [(a, b, block) for (a, b), (_, _, block) in zip(operator.block_nodes, operator.blocks)
               if isinstance(block, LowRankBlock)]
        # Check before even the input scaling workspaces are allocated. The
        # constructor independently checks admission for direct callers too.
        plan = plan_h2_storage(operator.tree, far, components=2, dtype=operator.dtype,
            max_rank=config.max_rank, budget_bytes=config.retained_budget_bytes)
        if not plan['admitted']:
            raise H2CapacityError(plan)
        total = operator.provider.system.total_edges
        real_dtype = operator.tree.lower.dtype
        eta = operator.metric.impedance_squared**.5
        row = torch.cat((torch.full((total,), eta, device=operator.device, dtype=real_dtype),
                         torch.ones(total, device=operator.device, dtype=real_dtype)))
        column = row.flip(0)
        result = cls(operator.tree, far, near=operator.packed_near, dtype=operator.dtype,
                     row_scale=row, column_scale=column, config=config)
        operator.refresh()
        return result

    def tensors(self):
        return (self.row_scale, self.column_scale, *(c for _, _, c in self.couplings))

    def refresh(self):
        if ((id(self.tree), id(self.near), id(self.test_basis), id(self.trial_basis), self.config,
             self.components, self.device, self.dtype, self.shape) != self._binding
                or tuple((a, b, id(c)) for a, b, c in self.couplings) != self._layout
                or tuple((id(t), t._version) for t in self.tensors()) != self._versions):
            raise ValueError('Nested H2 action binding or coupling changed')
        self.test_basis.assert_current()
        self.trial_basis.assert_current()
        if self.near is not None:
            self.near.assert_current()

    def _action(self, value, *, adjoint):
        self.refresh()
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.shape[0]
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Nested H2 actions require matching finite complex CUDA vectors')
        incoming, outgoing = (self.test_basis, self.trial_basis) if adjoint else (self.trial_basis, self.test_basis)
        inscale, outscale = (self.row_scale, self.column_scale) if adjoint else (self.column_scale, self.row_scale)
        upward = incoming.upward(value/inscale)
        local = [value.new_zeros((*value.shape[:-1], rank)) for rank in outgoing.ranks]
        for a, b, coupling in self.couplings:
            if adjoint:
                local[b] = local[b]+upward[a]@coupling.conj()
            else:
                local[a] = local[a]+upward[b]@coupling.mT
        result = outgoing.downward(local)/outscale
        if self.near is not None:
            result = result+(self.near.adjoint(value) if adjoint else self.near.apply(value))
        self.refresh()
        return result

    def apply(self, value):
        return self._action(value, adjoint=False)

    def adjoint(self, value):
        return self._action(value, adjoint=True)

    def receipt(self):
        self.refresh()
        test, trial = self.test_basis.receipt(), self.trial_basis.receipt()
        owned = sum(t.numel()*t.element_size() for t in self.tensors())+test['retained_bytes']+trial['retained_bytes']
        tree_bytes = self.tree.receipt()['retained_bytes']
        near_bytes = 0 if self.near is None else self.near.receipt()['retained_bytes']
        return {'version': 'algebraic_nested_h2_actions_v1', 'configuration': asdict(self.config),
                'test_basis': test, 'trial_basis': trial, 'far_couplings': len(self.couplings),
                'conversion_errors': list(self.conversion_errors), 'owned_retained_bytes': owned,
                'capacity_plan': self.capacity_plan,
                'shared_tree_bytes': tree_bytes, 'shared_near_bytes': near_bytes,
                'total_referenced_tensor_bytes': owned+tree_bytes+near_bytes,
                'source_h_far_factors_retained': False, 'global_riesz_certified': False,
                'setup_complexity_qualified': False, 'physical_qualification': False}
