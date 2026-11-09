"""Direct nested-kernel adapter for shared surface Krylov and Riesz protocols."""
from dataclasses import dataclass, asdict
import math
import torch
from .direct_system import DirectSystemActions
from .hmatrix import HierarchicalOperator
from .riesz import LocalRWGRiesz
from .operator_protocol import IterativeState


@dataclass(frozen=True)
class DirectSolveConfig:
    solve_dual_tolerance: float = 1e-8
    original_sample_tolerance: float = 1e-8

    def __post_init__(self):
        for value in (self.solve_dual_tolerance, self.original_sample_tolerance):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Direct-kernel residual tolerances must be finite positive scalars')


class DirectHierarchicalOperator:
    """Internal iterative backend; no H conversion or direct/dense fallback.

    Residual admission is shared with H/H2. The original-row residual remains a
    sample, and no global original/physical error certificate is implied.
    """
    supports_direct = False
    _vector = HierarchicalOperator._vector
    read_access = HierarchicalOperator.read_access
    riesz_inverse = HierarchicalOperator.riesz_inverse
    residual_report = HierarchicalOperator.residual_report

    def __init__(self, provider, *, solve_config=DirectSolveConfig(), **construction):
        if not isinstance(solve_config, DirectSolveConfig):
            raise TypeError('Direct-kernel solves require DirectSolveConfig')
        self.actions = self._build_actions(provider, **construction)
        reference = self.actions.provider
        if (reference.system is not provider.system or reference.exterior is not provider.exterior
                or reference.config is not provider.config):
            raise ValueError('Constructed original reference must preserve system, medium and integration')
        provider = reference
        self.provider, self.config = provider, solve_config
        self.metric = LocalRWGRiesz(provider.system, provider.exterior.eta)
        self.tree = self.actions.exterior.test.tree
        self.shape, self.device, self.dtype = self.actions.shape, self.actions.device, self.actions.dtype
        self.iterative_state = IterativeState()
        self._apply_calls = self._adjoint_calls = 0
        self._binding = (id(self.actions), id(provider), id(self.metric), id(self.tree),
                         self.config, self.shape, self.device, self.dtype)
        self.metadata = {'version': 'direct_nested_scalar_krylov_v1', 'operator_backend': 'direct_h2_internal',
            'reference_h_conversion': False, 'physical_qualification': False,
            'original_global_residual_certified': False, 'peak_memory_qualified': False,
            'residual_metric': 'impedance_balanced_L2_dual_proxy', 'trace_norm_certified': False,
            'rwg_diagonal_scaling': 'left=diag(eta/sqrt(Gii),1/sqrt(Gii)); right=diag(1/sqrt(Gii),eta/sqrt(Gii))'}
        DirectHierarchicalOperator.refresh(self)

    @staticmethod
    def _build_actions(provider, **construction):
        return DirectSystemActions(provider, **construction)

    def refresh(self):
        if ((id(self.actions), id(self.provider), id(self.metric), id(self.tree), self.config,
             self.shape, self.device, self.dtype) != self._binding
                or self.actions.provider is not self.provider or self.metric.system is not self.provider.system
                or self.actions.exterior.test.tree is not self.tree):
            raise ValueError('Direct-kernel solver binding changed')
        self.actions.assert_current()
        self.metric.assert_current()
        self.provider.assert_current()

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

    def preconditioner_scales(self, kind):
        with self.read_access():
            if kind == 'none':
                one = torch.ones_like(self.metric.diagonal)
                return one, one, False
            if kind != 'rwg_diagonal':
                raise ValueError('Direct-kernel backend supports none or rwg_diagonal scaling')
            left = self.metric.diagonal.rsqrt()
            n = self.shape[0]//2
            right = torch.cat((left[n:], left[:n]))
            if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
                raise ValueError('RWG diagonal scaling exceeds native precision')
            return left, right, False

    def solve(self, rhs, stopping=None, *, original_edges=None, source_batch=64, right_inverse=None,
              right_preconditioner=None):
        from .linalg import StoppingRule
        if stopping is None:
            stopping = StoppingRule(method='gmres', preconditioner='rwg_diagonal', rtol=1e-12, atol=0.)
        return HierarchicalOperator.solve(self, rhs, stopping, original_edges=original_edges, source_batch=source_batch,
                                          right_inverse=right_inverse, right_preconditioner=right_preconditioner)

    def persistence_receipt(self):
        self.refresh()
        return {'version': self.metadata['version'], 'solve_configuration': asdict(self.config),
                'construction': self.actions.receipt(), 'executable_snapshot': False,
                'compressed_tensor_snapshot_included': False}

    def operator_stats(self):
        self.refresh()
        return {**self.metadata, 'solve_configuration': asdict(self.config),
                'apply_calls': self._apply_calls, 'adjoint_calls': self._adjoint_calls,
                'construction': self.actions.receipt(), 'metric': self.metric.receipt()}
