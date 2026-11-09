"""Configuration for the experimental finite-dielectric SIE-VIE HB path."""
from dataclasses import dataclass,asdict,fields
import math
from .volume_mesh import TetrahedralVolumeMesh
from .linalg import StoppingRule


@dataclass(frozen=True)
class HarmonicCutoffConfig:
    """Budgeted cell-field cutoff estimation; physical qualification is separate."""
    max_harmonics: int
    relative_tolerance: float = 1e-4
    absolute_tolerance: float = 1e-10
    tail_absolute_tolerance: float = 1e-8
    successive_passes: int = 2
    failure: str = 'raise'

    def __post_init__(self):
        for name in ('max_harmonics','successive_passes'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f'Cutoff {name} must be a positive integer')
        for name in ('relative_tolerance','absolute_tolerance','tail_absolute_tolerance'):
            value = getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Cutoff {name} must be finite and nonnegative')
        if self.relative_tolerance+self.absolute_tolerance == 0 or self.tail_absolute_tolerance == 0:
            raise ValueError('Cutoff retained and tail budgets must be nonzero')
        if self.failure not in ('raise','record'):
            raise ValueError('Cutoff failure must be raise or record')


@dataclass(frozen=True)
class HarmonicBalanceConfig:
    """Start at the given positive harmonic count with explicit volume meshes.

    Optional cutoff control extends this count using explicit local budgets.
    A fixed harmonic count does not certify truncation convergence. A branch
    name identifies the chosen linear-start continuation, not its stability.
    Source maxima bound outer surface testing and inner volume quadrature;
    increasing either maximum does not relax the fixed source error tolerances.
    Newton-Krylov uses an explicit fixed-tangent GMRES policy. Its dense final
    implicit check may be replaced by a complete streamed dominance check;
    inconclusive bounds reject. Streaming avoids the full HB Jacobian storage,
    not all dense DC work or all-column cost, and grants no qualification.
    """
    harmonics: int
    volumes: tuple[TetrahedralVolumeMesh, ...]
    branch_id: str
    method: str = 'newton'
    pump_scales: tuple[float, ...] = (1.,)
    relative_tolerance: float = 1e-9
    absolute_tolerance: float = 1e-10
    max_iterations: int = 30
    damping: float = 1.
    max_backtracks: int = 12
    dc_testing_order: int = 4
    source_max_order: int = 32
    source_max_inner_order: int = 64
    tensor_face_rule: str = 'triangle'
    tensor_kernel_batch: int = 128
    cutoff: HarmonicCutoffConfig | None = None
    tangent_rule: StoppingRule | None = None
    implicit_solver: str = 'dense'
    tangent_column_batch: int = 8
    dc_tangent_rule: StoppingRule | None = None
    dc_tangent_column_batch: int = 8
    dc_magnetic_volumes: tuple[TetrahedralVolumeMesh, ...] | None = None

    def __post_init__(self):
        if self.tensor_face_rule not in ('triangle','projected_duffy','distance_scaled_duffy'):
            raise ValueError('HB tensor face rule is not supported')
        if type(self.tensor_kernel_batch) is not int or self.tensor_kernel_batch<1:
            raise ValueError('HB tensor kernel batch must be a positive integer')
        if self.dc_magnetic_volumes is not None:
            if (type(self.dc_magnetic_volumes) is not tuple or not self.dc_magnetic_volumes
                    or any(not isinstance(v,TetrahedralVolumeMesh) for v in self.dc_magnetic_volumes)):
                raise TypeError('DC magnetic volumes must be a nonempty tuple of tetrahedral volumes')
            if tuple(v.region_id for v in self.dc_magnetic_volumes)!=tuple(range(1,len(self.dc_magnetic_volumes)+1)):
                raise ValueError('DC magnetic volumes must use complete increasing region order')
        if self.dc_tangent_rule is not None:
            rule=self.dc_tangent_rule
            if (type(rule) is not StoppingRule or rule.method!='gmres' or rule.preconditioner!='none'
                    or rule.recycle_size or rule.fallback!='none'):
                raise ValueError('DC tangent policy requires unpreconditioned unrecycled GMRES without fallback')
        if type(self.dc_tangent_column_batch) is not int or self.dc_tangent_column_batch<1:
            raise ValueError('DC tangent column batch must be a positive integer')
        if self.dc_tangent_rule is None and self.dc_tangent_column_batch!=8:
            raise ValueError('DC column batch requires an active tangent rule')
        if type(self.harmonics) is not int or self.harmonics < 1:
            raise ValueError('HB harmonics must be a positive integer')
        if self.cutoff is not None:
            if not isinstance(self.cutoff,HarmonicCutoffConfig):
                raise TypeError('HB cutoff must be a HarmonicCutoffConfig')
            if self.cutoff.max_harmonics < self.harmonics:
                raise ValueError('Cutoff maximum cannot be below the initial harmonic count')
        if (not isinstance(self.volumes,tuple) or not self.volumes
                or any(not isinstance(volume,TetrahedralVolumeMesh) for volume in self.volumes)):
            raise TypeError('HB volumes must be a nonempty tuple of TetrahedralVolumeMesh objects')
        if len({volume.region_id for volume in self.volumes}) != len(self.volumes):
            raise ValueError('HB volume regions must be unique')
        if not isinstance(self.branch_id,str) or not self.branch_id.strip():
            raise ValueError('HB requires an explicit nonempty branch_id')
        if self.method not in ('newton','picard','newton_krylov'):
            raise ValueError('HB method must be newton, picard or newton_krylov')
        if self.method=='newton_krylov':
            if self.tangent_rule is None:
                object.__setattr__(self,'tangent_rule',StoppingRule(preconditioner='none',rtol=1e-10,atol=1e-13))
            rule=self.tangent_rule
            if (type(rule) is not StoppingRule or rule.method!='gmres' or rule.preconditioner!='none'
                    or rule.recycle_size or rule.fallback!='none'):
                raise ValueError('HB Newton-Krylov requires unpreconditioned unrecycled GMRES without fallback')
        elif self.tangent_rule is not None or self.implicit_solver!='dense' or self.tangent_column_batch!=8:
            raise ValueError('HB tangent settings require newton_krylov')
        if self.implicit_solver not in ('dense','streamed_krylov','streamed_inverse_krylov'):
            raise ValueError('HB implicit solver must be dense, streamed_krylov or streamed_inverse_krylov')
        if type(self.tangent_column_batch) is not int or self.tangent_column_batch<1:
            raise ValueError('HB tangent column batch must be a positive integer')
        if (not isinstance(self.pump_scales,tuple) or not self.pump_scales or
                any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in self.pump_scales)):
            raise ValueError('HB pump_scales must be a nonempty tuple of finite nonnegative values')
        for name in ('relative_tolerance','absolute_tolerance','damping'):
            value = getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'HB {name} must be a finite nonnegative number')
        if self.relative_tolerance+self.absolute_tolerance == 0 or not 0 < self.damping <= 1:
            raise ValueError('HB requires nonzero tolerances and damping in (0,1]')
        for name,minimum in (('max_iterations',0),('max_backtracks',0),('dc_testing_order',2),
                             ('source_max_order',3),('source_max_inner_order',3)):
            if type(getattr(self,name)) is not int or getattr(self,name) < minimum:
                raise ValueError(f'HB {name} must be an integer >= {minimum}')

    def solve_settings(self):
        result={name:getattr(self,name) for name in ('method','relative_tolerance','absolute_tolerance',
                'max_iterations','damping','max_backtracks')}
        if self.method=='newton_krylov':
            result.update(tangent_rule=self.tangent_rule,implicit_solver=self.implicit_solver,
                tangent_column_batch=self.tangent_column_batch)
        return result

    def numerical_payload(self):
        """Canonical policy fields; inactive tangent settings are not serialized."""
        tangent={'tangent_rule','implicit_solver','tangent_column_batch','dc_tangent_rule','dc_tangent_column_batch'}
        result={f.name:getattr(self,f.name) for f in fields(self) if f.name not in {'volumes','dc_magnetic_volumes','tensor_face_rule','tensor_kernel_batch',*tangent}}
        result['cutoff']=None if self.cutoff is None else asdict(self.cutoff)
        if self.tensor_face_rule!='triangle' or self.tensor_kernel_batch!=128:
            result.update(tensor_face_rule=self.tensor_face_rule,tensor_kernel_batch=self.tensor_kernel_batch)
        if self.method=='newton_krylov':
            result.update(tangent_rule=asdict(self.tangent_rule),implicit_solver=self.implicit_solver,
                tangent_column_batch=self.tangent_column_batch)
        if self.dc_tangent_rule is not None:
            result.update(dc_tangent_rule=asdict(self.dc_tangent_rule),dc_tangent_column_batch=self.dc_tangent_column_batch)
        return result

    def dc_policy(self):
        return {'tangent_rule':self.dc_tangent_rule,'tangent_column_batch':self.dc_tangent_column_batch}
