"""Versioned configuration for direct scalar nested interpolation construction."""
from dataclasses import dataclass, asdict
import math
from .interpolation_error import KernelErrorBudget


@dataclass(frozen=True)
class DirectH2Config:
    order: int = 6
    max_order: int | None = None
    quadrature_order: int | None = None
    leaf_size: int = 32
    admissibility: float = 1.
    kernel_budget_bytes: int = 512*1024**2
    action_retained_budget_bytes: int = 1024**3
    kernel_error_budget: KernelErrorBudget | None = None
    rwg_absolute_budget: float | None = None
    solve_dual_tolerance: float = 1e-8
    original_sample_tolerance: float = 1e-8

    def __post_init__(self):
        for name in ('order', 'leaf_size', 'kernel_budget_bytes', 'action_retained_budget_bytes', 'max_order', 'quadrature_order'):
            value = getattr(self, name)
            if value is None and name in ('max_order', 'quadrature_order'):
                continue
            if type(value) is not int or value < 1:
                raise ValueError(name+' must be a positive integer')
        for name in ('admissibility', 'solve_dual_tolerance', 'original_sample_tolerance', 'rwg_absolute_budget'):
            value = getattr(self, name)
            if value is None and name == 'rwg_absolute_budget':
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be finite and positive')
        if self.kernel_error_budget is not None and not isinstance(self.kernel_error_budget, KernelErrorBudget):
            raise TypeError('kernel_error_budget must be KernelErrorBudget')
        if self.max_order is not None:
            if self.max_order < self.order or self.kernel_error_budget is None:
                raise ValueError('Automatic order selection requires max_order >= order and kernel_error_budget')
        if self.rwg_absolute_budget is not None and self.max_order is None:
            raise ValueError('RWG absolute admission requires automatic order selection')

    def construction(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__
                if name not in ('solve_dual_tolerance', 'original_sample_tolerance')}

    def solve_config(self):
        from .direct_operator import DirectSolveConfig
        return DirectSolveConfig(self.solve_dual_tolerance, self.original_sample_tolerance)

    def payload(self):
        return {'version': 'direct_nested_h2_config_v2', 'parameters': asdict(self)}

    @classmethod
    def restore(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {'version', 'parameters'}
                or payload['version'] != 'direct_nested_h2_config_v2'
                or not isinstance(payload['parameters'], dict)
                or set(payload['parameters']) != set(cls.__dataclass_fields__)):
            raise ValueError('Unsupported or incomplete direct H2 configuration')
        values = dict(payload['parameters'])
        budget = values['kernel_error_budget']
        if budget is not None:
            if not isinstance(budget, dict) or set(budget) != set(KernelErrorBudget.__dataclass_fields__):
                raise ValueError('Invalid direct H2 kernel error budget')
            values['kernel_error_budget'] = KernelErrorBudget(**budget)
        return cls(**values)
