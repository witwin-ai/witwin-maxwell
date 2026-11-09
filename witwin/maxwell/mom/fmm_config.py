"""Versioned experimental Cartesian FMM construction and solve configuration."""
from dataclasses import dataclass, asdict
import math
from .interpolation_error import KernelErrorBudget
from .fmm_error import validate_kernel_bound_rule


@dataclass(frozen=True)
class FMMConfig:
    order: int = 6
    table_storage: str = 'compact_shared_displacement_v1'
    shift_storage: str = 'separable_v1'
    near_storage: str = 'scalar_reciprocal_dk_v1'
    near_coefficient_workspace_budget_bytes: int = 128*1024**2
    near_coefficient_cache_budget_bytes: int = 128*1024**2
    kernel_bound_rule: str = 'bivariate_cauchy_v1'
    m2l_row_batch: int = 16
    far_action_batch: int = 1
    expansion_center_policy: str = 'support_box_midpoint_v1'
    # Bounds both block entries and RHS vectors per near-action chunk.
    near_action_batch: int = 32
    max_order: int | None = None
    rwg_absolute_budget: float | None = None
    kernel_error_budget: KernelErrorBudget | None = None
    quadrature_order: int | None = None
    leaf_size: int = 32
    separation_ratio: float = .5
    action_retained_budget_bytes: int = 1024**3
    table_budget_bytes: int = 512*1024**2
    near_budget_bytes: int = 1024**3
    projection_budget_bytes: int = 1024**3
    solve_dual_tolerance: float = 1e-8
    original_sample_tolerance: float = 1e-8
    leaf_factor_kernel: str = 'none'
    leaf_factor_max_edges: int = 32
    leaf_factor_budget_bytes: int = 256*1024**2

    def __post_init__(self):
        if self.expansion_center_policy not in ('support_box_midpoint_v1', 'dyadic_cell_v1', 'binary_lattice_v1'):
            raise ValueError('Unsupported Cartesian FMM expansion center policy')
        if self.expansion_center_policy in ('dyadic_cell_v1','binary_lattice_v1') and (
                self.table_storage not in ('compact', 'compact_shared_displacement_v1', 'compact_shared_signed_axis_v1','radial_series_shared_radius_v1')
                or self.order > 12 or self.max_order is not None and self.max_order > 12):
            raise ValueError('Dyadic cell centers require free-space compact order0..12')
        if type(self.far_action_batch) is not int or not 1 <= self.far_action_batch <= 32:
            raise ValueError('Compact FAR action batch must lie between one and32')
        if self.far_action_batch > 1 and (self.table_storage not in ('compact','compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1')
                or self.order > 12 or self.max_order is not None and self.max_order > 12):
            raise ValueError('Batched compact FAR requires compact order0..12')
        validate_kernel_bound_rule(self.kernel_bound_rule)
        if self.near_storage not in ('full_matrix', 'scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1'):
            raise ValueError('Unsupported FMM near storage')
        if self.near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1') and self.table_storage == 'fft_ewald_v1':
            raise NotImplementedError('Scalar D/K near storage currently requires free space')
        if (type(self.near_coefficient_workspace_budget_bytes) is not int
                or self.near_coefficient_workspace_budget_bytes < 1):
            raise ValueError('Near coefficient workspace budget must be a positive integer')
        if (self.near_storage not in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1')
                and self.near_coefficient_workspace_budget_bytes != 128*1024**2):
            raise ValueError('Explicit near coefficient workspace requires streamed reciprocal storage')
        if (type(self.near_coefficient_cache_budget_bytes) is not int
                or self.near_coefficient_cache_budget_bytes < 0):
            raise ValueError('Near coefficient cache budget must be a nonnegative integer')
        if (self.near_storage != 'scalar_reciprocal_cached_dk_v1'
                and self.near_coefficient_cache_budget_bytes != 128*1024**2):
            raise ValueError('Explicit near coefficient cache budget requires cached reciprocal storage')
        if self.shift_storage not in ('dense','separable_v1'):
            raise ValueError('FMM shift_storage must be dense or separable_v1')
        if self.leaf_factor_kernel not in ('none', 'original', 'primary'):
            raise ValueError('FMM leaf_factor_kernel must be none, original or primary')
        if self.table_storage not in ('dense', 'compact', 'fft_v1', 'fft_ewald_v1', 'compact_shared_displacement_v1', 'compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            raise ValueError('FMM table_storage must be dense, compact, fft_v1 or fft_ewald_v1')
        if type(self.order) is not int or self.order < 0:
            raise ValueError('FMM order must be a nonnegative integer')
        if self.table_storage in ('compact_shared_signed_axis_v1','radial_series_shared_radius_v1') and (
                self.order > 12 or self.max_order is not None and self.max_order > 12
                or type(self.m2l_row_batch) is not int or not 1 <= self.m2l_row_batch <= 16):
            raise ValueError('Signed-axis FMM requires order0..12 and rows1..16')
        for name in ('near_action_batch', 'leaf_factor_max_edges', 'leaf_factor_budget_bytes', 'm2l_row_batch', 'quadrature_order', 'leaf_size', 'action_retained_budget_bytes', 'table_budget_bytes', 'near_budget_bytes', 'projection_budget_bytes'):
            value = getattr(self, name)
            if value is None and name == 'quadrature_order':
                continue
            if type(value) is not int or value < 1:
                raise ValueError(name+' must be a positive integer')
        for name in ('separation_ratio', 'solve_dual_tolerance', 'original_sample_tolerance'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be finite and positive')
        if self.kernel_error_budget is not None and not isinstance(self.kernel_error_budget, KernelErrorBudget):
            raise TypeError('FMM kernel_error_budget must be KernelErrorBudget')
        if self.max_order is not None:
            if type(self.max_order) is not int or self.max_order < self.order or self.kernel_error_budget is None:
                raise ValueError('Automatic FMM order requires max_order >= order and a kernel error budget')
        if self.table_storage in ('fft_v1','fft_ewald_v1') and (self.order > 12 or self.max_order is not None and self.max_order > 12):
            raise ValueError('FFT FMM native order range must lie between zero and twelve')
        if self.rwg_absolute_budget is not None:
            value = self.rwg_absolute_budget
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or self.max_order is None:
                raise ValueError('RWG absolute budget requires a positive finite value and automatic FMM order')
        if self.separation_ratio >= 1:
            raise ValueError('FMM separation_ratio must be less than one')

    def construction(self):
        parameters = {name: getattr(self, name) for name in self.__dataclass_fields__
                if name not in ('solve_dual_tolerance', 'original_sample_tolerance',
                                'leaf_factor_kernel', 'leaf_factor_max_edges', 'leaf_factor_budget_bytes')}
        if self.far_action_batch == 1:
            parameters.pop('far_action_batch')
        if self.expansion_center_policy == 'support_box_midpoint_v1':
            parameters.pop('expansion_center_policy')
        if self.near_storage == 'full_matrix':
            parameters.pop('near_storage')
        if self.near_storage not in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            parameters.pop('near_coefficient_workspace_budget_bytes')
        if self.near_storage != 'scalar_reciprocal_cached_dk_v1':
            parameters.pop('near_coefficient_cache_budget_bytes')
        return parameters

    def solve_config(self):
        from .direct_operator import DirectSolveConfig
        return DirectSolveConfig(self.solve_dual_tolerance, self.original_sample_tolerance)

    def payload(self):
        parameters=asdict(self)
        if self.table_storage == 'radial_series_shared_radius_v1':
            return {'version':'cartesian_fmm_radial_series_config_v1','parameters':parameters}
        if self.near_storage == 'scalar_reciprocal_cached_dk_v1':
            return {'version':'cartesian_fmm_cached_reciprocal_dk_config_v1','parameters':parameters}
        if self.table_storage == 'compact_shared_signed_axis_v1':
            return {'version':'cartesian_fmm_signed_axis_config_v1','parameters':parameters}
        if self.near_storage == 'scalar_reciprocal_streamed_dk_v1':
            return {'version':'cartesian_fmm_streamed_reciprocal_dk_config_v1', 'parameters':parameters}
        if self.expansion_center_policy == 'binary_lattice_v1':
            return {'version': 'cartesian_fmm_binary_lattice_config_v1', 'parameters': parameters}
        if self.expansion_center_policy == 'dyadic_cell_v1':
            return {'version': 'cartesian_fmm_dyadic_centers_config_v1', 'parameters': parameters}
        parameters.pop('expansion_center_policy')
        if self.far_action_batch > 1:
            return {'version':'cartesian_fmm_batched_compact_config_v1','parameters':parameters}
        parameters.pop('far_action_batch')
        if self.table_storage=='compact_shared_displacement_v1':
            return {'version':'cartesian_fmm_shared_compact_config_v1','parameters':parameters}
        if self.near_storage == 'scalar_reciprocal_dk_v1':
            return {'version': 'cartesian_fmm_reciprocal_dk_config_v1', 'parameters': parameters}
        if self.near_storage == 'scalar_dk_v1':
            return {'version': 'cartesian_fmm_scalar_dk_config_v1', 'parameters': parameters}
        if self.table_storage == 'fft_ewald_v1':
            return {'version': 'cartesian_fmm_ewald_fft_config_v1', 'parameters': parameters}
        if self.table_storage == 'fft_v1':
            # This version fixes physical RHS chunks at4, scalar chunks at4
            # and visible scalar workspace at128MiB, including EM chunk overhead.
            return {'version': 'cartesian_fmm_fft_config_v1', 'parameters': parameters}
        if self.kernel_bound_rule == 'direct_first_wave_min_v1':
            return {'version': 'cartesian_fmm_direct_first_wave_config_v1', 'parameters': parameters}
        if self.kernel_bound_rule == 'direct_positive_tail_min_v1':
            return {'version': 'cartesian_fmm_direct_positive_tail_config_v1', 'parameters': parameters}
        if self.kernel_bound_rule == 'first_wave_min_v1':
            return {'version': 'cartesian_fmm_first_wave_config_v1', 'parameters': parameters}
        if self.kernel_bound_rule == 'positive_tail_min_v1':
            return {'version': 'cartesian_fmm_positive_tail_config_v1', 'parameters': parameters}
        if self.kernel_bound_rule == 'real_support_min_v1':
            return {'version': 'cartesian_fmm_real_support_config_v1', 'parameters': parameters}
        parameters.pop('kernel_bound_rule')
        if self.shift_storage=='dense':
            return {'version': 'cartesian_fmm_config_v7', 'parameters': parameters}
        return {'version': 'cartesian_fmm_separable_config_v1', 'parameters': parameters}

    @classmethod
    def restore(cls, payload):
        if (isinstance(payload,dict) and isinstance(payload.get('parameters'),dict)
                and payload['parameters'].get('table_storage') == 'radial_series_shared_radius_v1'
                and payload.get('version') != 'cartesian_fmm_radial_series_config_v1'):
            raise ValueError('Radial series storage requires its explicit configuration version')
        if isinstance(payload,dict) and payload.get('version') == 'cartesian_fmm_radial_series_config_v1':
            if (set(payload) != {'version','parameters'} or not isinstance(payload['parameters'],dict)
                    or set(payload['parameters']) != set(cls.__dataclass_fields__)
                    or payload['parameters']['table_storage'] != 'radial_series_shared_radius_v1'):
                raise ValueError('Unsupported or incomplete radial series FMM configuration')
            values=dict(payload['parameters']);budget=values['kernel_error_budget']
            if budget is not None:
                if not isinstance(budget,dict) or set(budget)!=set(KernelErrorBudget.__dataclass_fields__):
                    raise ValueError('Invalid FMM kernel budget payload')
                values['kernel_error_budget']=KernelErrorBudget(**budget)
            return cls(**values)
        if isinstance(payload, dict) and payload.get('version') == 'automatic_directional_cached_reciprocal_dag_fmm_config_v1':
            return DirectionalFMMConfig.restore(payload)
        if (isinstance(payload,dict) and isinstance(payload.get('parameters'),dict)
                and payload['parameters'].get('near_storage') == 'scalar_reciprocal_cached_dk_v1'
                and payload.get('version') != 'cartesian_fmm_cached_reciprocal_dk_config_v1'):
            raise ValueError('Cached original near storage requires its explicit configuration version')
        if isinstance(payload,dict) and payload.get('version') == 'cartesian_fmm_cached_reciprocal_dk_config_v1':
            if (set(payload) != {'version','parameters'}
                    or not isinstance(payload['parameters'],dict)
                    or set(payload['parameters']) != set(cls.__dataclass_fields__)
                    or payload['parameters']['near_storage'] != 'scalar_reciprocal_cached_dk_v1'):
                raise ValueError('Unsupported or incomplete cached original near FMM configuration')
            values = dict(payload['parameters'])
            budget = values['kernel_error_budget']
            if budget is not None:
                if not isinstance(budget,dict) or set(budget) != set(KernelErrorBudget.__dataclass_fields__):
                    raise ValueError('Invalid FMM kernel budget payload')
                values['kernel_error_budget'] = KernelErrorBudget(**budget)
            return cls(**values)
        if (isinstance(payload,dict) and isinstance(payload.get('parameters'),dict)
                and payload['parameters'].get('table_storage') == 'compact_shared_signed_axis_v1'
                and payload.get('version') != 'cartesian_fmm_signed_axis_config_v1'):
            raise ValueError('Signed-axis sharing requires its explicit configuration version')
        if isinstance(payload,dict) and payload.get('version') == 'cartesian_fmm_signed_axis_config_v1':
            if (set(payload) != {'version','parameters'}
                    or not isinstance(payload['parameters'],dict)
                    or set(payload['parameters']) != set(cls.__dataclass_fields__)
                    or payload['parameters']['table_storage'] != 'compact_shared_signed_axis_v1'):
                raise ValueError('Unsupported or incomplete signed-axis FMM configuration')
            values = dict(payload['parameters'])
            budget = values['kernel_error_budget']
            if budget is not None:
                if not isinstance(budget,dict) or set(budget) != set(KernelErrorBudget.__dataclass_fields__):
                    raise ValueError('Invalid FMM kernel budget payload')
                values['kernel_error_budget'] = KernelErrorBudget(**budget)
            return cls(**values)
        if isinstance(payload, dict) and payload.get('version') in ('automatic_directional_dag_fmm_config_v1','automatic_directional_reciprocal_dag_fmm_config_v1','automatic_directional_streamed_reciprocal_dag_fmm_config_v1'):
            return DirectionalFMMConfig.restore(payload)
        if (isinstance(payload,dict) and payload.get('version') != 'cartesian_fmm_streamed_reciprocal_dk_config_v1'
                and isinstance(payload.get('parameters'),dict)
                and payload['parameters'].get('near_storage') == 'scalar_reciprocal_streamed_dk_v1'):
            raise ValueError('Streamed near storage requires its explicit configuration version')
        if isinstance(payload,dict) and payload.get('version') == 'cartesian_fmm_streamed_reciprocal_dk_config_v1':
            if (set(payload) != {'version','parameters'}
                    or not isinstance(payload['parameters'],dict)
                    or set(payload['parameters']) != set(cls.__dataclass_fields__)
                    or payload['parameters']['near_storage'] != 'scalar_reciprocal_streamed_dk_v1'):
                raise ValueError('Unsupported or incomplete streamed near FMM configuration')
            values = dict(payload['parameters'])
            budget = values['kernel_error_budget']
            if budget is not None:
                if not isinstance(budget,dict) or set(budget) != set(KernelErrorBudget.__dataclass_fields__):
                    raise ValueError('Invalid FMM kernel budget payload')
                values['kernel_error_budget'] = KernelErrorBudget(**budget)
            return cls(**values)
        if isinstance(payload, dict) and payload.get('version') in ('cartesian_fmm_dyadic_centers_config_v1','cartesian_fmm_binary_lattice_config_v1'):
            if (set(payload) != {'version', 'parameters'}
                    or not isinstance(payload['parameters'], dict)
                    or set(payload['parameters']) != set(cls.__dataclass_fields__)
                    or payload['parameters']['expansion_center_policy'] != (
                        'binary_lattice_v1' if payload['version']=='cartesian_fmm_binary_lattice_config_v1' else 'dyadic_cell_v1')):
                raise ValueError('Unsupported or incomplete dyadic center FMM configuration')
            values = dict(payload['parameters'])
            budget = values['kernel_error_budget']
            if budget is not None:
                if not isinstance(budget, dict) or set(budget) != set(KernelErrorBudget.__dataclass_fields__):
                    raise ValueError('Invalid FMM kernel budget payload')
                values['kernel_error_budget'] = KernelErrorBudget(**budget)
            return cls(**values)
        fields=set(cls.__dataclass_fields__)
        fields.remove('expansion_center_policy')
        if not isinstance(payload,dict) or payload.get('version') != 'cartesian_fmm_batched_compact_config_v1':
            fields.remove('far_action_batch')
        if isinstance(payload, dict) and payload.get('version') not in ('cartesian_fmm_batched_compact_config_v1','cartesian_fmm_shared_compact_config_v1','cartesian_fmm_reciprocal_dk_config_v1','cartesian_fmm_scalar_dk_config_v1','cartesian_fmm_real_support_config_v1','cartesian_fmm_positive_tail_config_v1','cartesian_fmm_first_wave_config_v1','cartesian_fmm_direct_positive_tail_config_v1','cartesian_fmm_direct_first_wave_config_v1','cartesian_fmm_fft_config_v1','cartesian_fmm_ewald_fft_config_v1'):
            fields.remove('kernel_bound_rule')
        if (not isinstance(payload, dict) or set(payload) != {'version', 'parameters'}
                or payload['version'] not in ('cartesian_fmm_batched_compact_config_v1','cartesian_fmm_shared_compact_config_v1','cartesian_fmm_reciprocal_dk_config_v1','cartesian_fmm_scalar_dk_config_v1','cartesian_fmm_config_v7','cartesian_fmm_separable_config_v1','cartesian_fmm_real_support_config_v1','cartesian_fmm_positive_tail_config_v1','cartesian_fmm_first_wave_config_v1','cartesian_fmm_direct_positive_tail_config_v1','cartesian_fmm_direct_first_wave_config_v1','cartesian_fmm_fft_config_v1','cartesian_fmm_ewald_fft_config_v1')
                or not isinstance(payload['parameters'], dict)
                or set(payload['parameters']) != fields
                or (payload['version']!='cartesian_fmm_batched_compact_config_v1' and ((payload['version']=='cartesian_fmm_shared_compact_config_v1') != (payload['parameters'].get('table_storage')=='compact_shared_displacement_v1')))
                or (payload['version']=='cartesian_fmm_batched_compact_config_v1' and (type(payload['parameters'].get('far_action_batch')) is not int or not 2 <= payload['parameters']['far_action_batch'] <= 32))
                or (payload['version']=='cartesian_fmm_fft_config_v1' and payload['parameters'].get('table_storage')!='fft_v1')
                or (payload['parameters'].get('table_storage')=='fft_v1' and payload['version'] not in ('cartesian_fmm_fft_config_v1','cartesian_fmm_scalar_dk_config_v1','cartesian_fmm_reciprocal_dk_config_v1'))
                or (payload['version']=='cartesian_fmm_reciprocal_dk_config_v1' and payload['parameters'].get('near_storage')!='scalar_reciprocal_dk_v1')
                or (payload['version']=='cartesian_fmm_scalar_dk_config_v1' and payload['parameters'].get('near_storage')!='scalar_dk_v1')
                or (payload['version'] not in ('cartesian_fmm_batched_compact_config_v1','cartesian_fmm_shared_compact_config_v1','cartesian_fmm_reciprocal_dk_config_v1','cartesian_fmm_scalar_dk_config_v1') and payload['parameters']['near_storage']!='full_matrix')
                or ((payload['version']=='cartesian_fmm_ewald_fft_config_v1') != (payload['parameters'].get('table_storage')=='fft_ewald_v1'))
                or (payload['version']=='cartesian_fmm_config_v7' and payload['parameters']['shift_storage']!='dense')
                or (payload['version']=='cartesian_fmm_separable_config_v1'
                    and payload['parameters'].get('shift_storage')!='separable_v1')
                or (payload['version']=='cartesian_fmm_real_support_config_v1'
                    and payload['parameters'].get('kernel_bound_rule')!='real_support_min_v1')
                or (payload['version']=='cartesian_fmm_positive_tail_config_v1'
                    and payload['parameters'].get('kernel_bound_rule')!='positive_tail_min_v1')
                or (payload['version']=='cartesian_fmm_first_wave_config_v1'
                    and payload['parameters'].get('kernel_bound_rule')!='first_wave_min_v1')
                or (payload['version']=='cartesian_fmm_direct_positive_tail_config_v1'
                    and payload['parameters'].get('kernel_bound_rule')!='direct_positive_tail_min_v1')
                or (payload['version']=='cartesian_fmm_direct_first_wave_config_v1'
                    and payload['parameters'].get('kernel_bound_rule')!='direct_first_wave_min_v1')):
            raise ValueError('Unsupported or incomplete FMM configuration')
        values = dict(payload['parameters'])
        budget = values['kernel_error_budget']
        if budget is not None:
            if not isinstance(budget, dict) or set(budget) != set(KernelErrorBudget.__dataclass_fields__):
                raise ValueError('Invalid FMM kernel budget payload')
            values['kernel_error_budget'] = KernelErrorBudget(**budget)
        return cls(**values)


@dataclass(frozen=True)
class DirectionalFMMConfig:
    """Automatic scalar DAG configuration with one global ideal error budget.

    Regional native capacities are per region. They do not certify aggregate
    setup/action peaks, allocator state, converged physics or complexity.
    """
    kernel_error_budget: KernelErrorBudget = KernelErrorBudget(1e-6, 1e-4)
    global_ideal_rwg_error_budget: float = 1e-4
    phase_relative_budget: float = 1e-12
    min_order: int = 0
    max_order: int = 12
    quadrature_order: int = 12
    leaf_size: int = 8
    separation_ratio: float = .1
    cone_subdivisions: tuple[int, ...] = (2,)
    face_batch: int = 4
    rhs_batch: int = 4
    max_candidates: int = 10000
    max_contexts: int = 100000
    max_pairs: int = 100000
    max_families: int = 32
    regional_plan_budget_bytes: int = 256*1024**2
    regional_projection_budget_bytes: int = 1024**3
    regional_table_budget_bytes: int = 512*1024**2
    regional_near_budget_bytes: int = 1024**3
    embedding_budget_bytes: int = 1024**2
    tensor_workspace_budget_bytes: int = 1024**3
    solve_dual_tolerance: float = 1e-8
    original_sample_tolerance: float = 1e-8
    leaf_factor_kernel: str = 'none'
    leaf_factor_max_edges: int = 32
    leaf_factor_budget_bytes: int = 256*1024**2
    near_storage: str = 'scalar_dk_v1'
    near_coefficient_workspace_budget_bytes: int = 128*1024**2
    near_coefficient_cache_budget_bytes: int = 128*1024**2

    def __post_init__(self):
        if self.near_storage not in ('scalar_dk_v1','scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
            raise ValueError('Unsupported directional scalar near storage')
        if (type(self.near_coefficient_workspace_budget_bytes) is not int
                or self.near_coefficient_workspace_budget_bytes < 1):
            raise ValueError('Directional near coefficient workspace must be a positive integer')
        if (self.near_storage not in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1')
                and self.near_coefficient_workspace_budget_bytes != 128*1024**2):
            raise ValueError('Explicit directional coefficient workspace requires streamed reciprocal storage')
        if (type(self.near_coefficient_cache_budget_bytes) is not int
                or self.near_coefficient_cache_budget_bytes < 0):
            raise ValueError('Directional near coefficient cache budget must be a nonnegative integer')
        if (self.near_storage != 'scalar_reciprocal_cached_dk_v1'
                and self.near_coefficient_cache_budget_bytes != 128*1024**2):
            raise ValueError('Explicit directional coefficient cache budget requires cached reciprocal storage')
        if type(self.kernel_error_budget) is not KernelErrorBudget:
            raise TypeError('Directional FMM requires KernelErrorBudget')
        if (type(self.min_order) is not int or type(self.max_order) is not int
                or not 0 <= self.min_order <= self.max_order <= 12):
            raise ValueError('Directional FMM order range must lie in0..12')
        integers = ('quadrature_order', 'leaf_size', 'face_batch', 'rhs_batch',
            'max_candidates', 'max_contexts', 'max_pairs', 'max_families',
            'regional_plan_budget_bytes', 'regional_projection_budget_bytes',
            'regional_table_budget_bytes', 'regional_near_budget_bytes',
            'embedding_budget_bytes', 'tensor_workspace_budget_bytes',
            'leaf_factor_max_edges', 'leaf_factor_budget_bytes')
        for name in integers:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name+' must be a positive integer')
        if self.quadrature_order < (self.max_order+4)//2:
            raise ValueError('Directional FMM quadrature must resolve every configured order')
        for name in ('global_ideal_rwg_error_budget', 'phase_relative_budget',
                     'separation_ratio', 'solve_dual_tolerance', 'original_sample_tolerance'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be finite and positive')
        if self.separation_ratio >= 1 or self.phase_relative_budget >= 1:
            raise ValueError('Directional separation and phase budgets must be less than one')
        grids = self.cone_subdivisions
        if (type(grids) is not tuple or not grids
                or any(type(v) is not int or not 1 <= v <= 2**20 for v in grids)
                or any(a >= b for a, b in zip(grids, grids[1:]))):
            raise ValueError('Directional cone grids must be an increasing bounded integer tuple')
        if self.leaf_factor_kernel not in ('none', 'original'):
            raise ValueError('Directional FMM leaf factors must be none or original')

    def solve_config(self):
        from .direct_operator import DirectSolveConfig
        return DirectSolveConfig(self.solve_dual_tolerance, self.original_sample_tolerance)

    def payload(self):
        values = asdict(self)
        values['cone_subdivisions'] = list(self.cone_subdivisions)
        if self.near_storage == 'scalar_reciprocal_cached_dk_v1':
            return dict(version='automatic_directional_cached_reciprocal_dag_fmm_config_v1', parameters=values)
        values.pop('near_coefficient_cache_budget_bytes')
        if self.near_storage == 'scalar_reciprocal_streamed_dk_v1':
            return dict(version='automatic_directional_streamed_reciprocal_dag_fmm_config_v1',parameters=values)
        values.pop('near_coefficient_workspace_budget_bytes')
        if self.near_storage == 'scalar_reciprocal_dk_v1':
            return dict(version='automatic_directional_reciprocal_dag_fmm_config_v1',parameters=values)
        values.pop('near_storage')
        return dict(version='automatic_directional_dag_fmm_config_v1', parameters=values)

    @classmethod
    def restore(cls, payload):
        fields=set(cls.__dataclass_fields__)
        version=payload.get('version') if isinstance(payload,dict) else None
        cached=version=='automatic_directional_cached_reciprocal_dag_fmm_config_v1'
        if not cached:
            fields.remove('near_coefficient_cache_budget_bytes')
        streamed=version=='automatic_directional_streamed_reciprocal_dag_fmm_config_v1' or cached
        if not streamed:
            fields.remove('near_coefficient_workspace_budget_bytes')
        if version not in ('automatic_directional_reciprocal_dag_fmm_config_v1','automatic_directional_streamed_reciprocal_dag_fmm_config_v1','automatic_directional_cached_reciprocal_dag_fmm_config_v1'):
            fields.remove('near_storage')
        if (not isinstance(payload, dict) or set(payload) != {'version', 'parameters'}
                or version not in ('automatic_directional_dag_fmm_config_v1','automatic_directional_reciprocal_dag_fmm_config_v1','automatic_directional_streamed_reciprocal_dag_fmm_config_v1','automatic_directional_cached_reciprocal_dag_fmm_config_v1')
                or not isinstance(payload['parameters'], dict)
                or set(payload['parameters']) != fields
                or (streamed and payload['parameters'].get('near_storage')!=('scalar_reciprocal_cached_dk_v1' if cached else 'scalar_reciprocal_streamed_dk_v1'))
                or (payload['version']=='automatic_directional_reciprocal_dag_fmm_config_v1'
                    and payload['parameters'].get('near_storage')!='scalar_reciprocal_dk_v1')):
            raise ValueError('Unsupported or incomplete directional FMM configuration')
        values = dict(payload['parameters'])
        budget = values['kernel_error_budget']
        if not isinstance(budget, dict) or set(budget) != set(KernelErrorBudget.__dataclass_fields__):
            raise ValueError('Invalid directional FMM kernel budget payload')
        if type(values['cone_subdivisions']) is not list:
            raise ValueError('Directional FMM cone grid payload must be a list')
        values['kernel_error_budget'] = KernelErrorBudget(**budget)
        values['cone_subdivisions'] = tuple(values['cone_subdivisions'])
        return cls(**values)
