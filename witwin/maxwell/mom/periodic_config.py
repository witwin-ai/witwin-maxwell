"""Immutable declarative Bravais/Ewald settings for the MoM runtime."""
from dataclasses import dataclass, asdict, field
import math
import torch


@dataclass(frozen=True)
class MoMPeriodicConfig:
    """Row lattice vectors [m], Bloch vector [rad/m] and explicit Ewald budgets.

    The lattice defines MoM repetition independently of a volumetric grid.
    CUDA compilation validates rank, the Bloch span and native precision.
    Anomalies and spectral poles retain the kernel's fail-closed contract.
    """
    vectors_m: tuple
    bloch_rad_per_m: tuple = (0., 0., 0.)
    eta_per_m: float = 3.
    real_shells: int = 3
    reciprocal_shells: int = 3
    quadrature_order: int = 64
    image_batch: int = 32
    query_batch: int = 256
    pole_relative_tolerance: float = 1e-10
    version: str = field(default='mom_bravais_ewald_config_v1', init=False)

    def __post_init__(self):
        def vector(value):
            if (not isinstance(value, (tuple, list)) or len(value) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
                raise ValueError('Periodic vectors must have three finite real coordinates')
            return tuple(float(v) for v in value)
        if not isinstance(self.vectors_m, (tuple, list)) or not 1 <= len(self.vectors_m) <= 3:
            raise ValueError('Periodic lattice requires one, two or three row vectors')
        object.__setattr__(self, 'vectors_m', tuple(vector(v) for v in self.vectors_m))
        object.__setattr__(self, 'bloch_rad_per_m', vector(self.bloch_rad_per_m))
        for name in ('eta_per_m', 'pole_relative_tolerance'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be positive and finite')
            object.__setattr__(self, name, float(value))
        for name in ('real_shells', 'reciprocal_shells', 'quadrature_order', 'image_batch', 'query_batch'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name+' must be a positive integer')

    def payload(self):
        result = asdict(self)
        result['vectors_m'] = [list(v) for v in self.vectors_m]
        result['bloch_rad_per_m'] = list(self.bloch_rad_per_m)
        return result

    @classmethod
    def restore(cls, payload):
        if not isinstance(payload, dict) or payload.get('version') != 'mom_bravais_ewald_config_v1':
            raise ValueError('Unsupported periodic MoM configuration version')
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError('Periodic configuration has missing or unknown fields')
        return cls(**{k: v for k, v in payload.items() if k != 'version'})

    def compile(self, *, device, dtype=torch.float64):
        from .periodic import LatticeSpec, EwaldHelmholtzKernel
        real_dtype = {torch.complex128: torch.float64, torch.complex64: torch.float32,
                      torch.float64: torch.float64, torch.float32: torch.float32}.get(dtype)
        if real_dtype is None:
            raise ValueError('Periodic MoM compilation requires float32/64 or complex64/128')
        lattice = LatticeSpec(torch.tensor(self.vectors_m, device=device, dtype=real_dtype),
                              torch.tensor(self.bloch_rad_per_m, device=device, dtype=real_dtype))
        return EwaldHelmholtzKernel(lattice, self.eta_per_m, real_shells=self.real_shells,
            reciprocal_shells=self.reciprocal_shells, quadrature_order=self.quadrature_order,
            image_batch=self.image_batch, query_batch=self.query_batch,
            pole_relative_tolerance=self.pole_relative_tolerance)


@dataclass(frozen=True)
class MoMDyadicPeriodicConfig:
    """Explicit checked periodic Maxwell budgets, distinct from scalar Ewald.

    The regular adapter owns primary subtraction and source-plane limits.
    These settings declare numerical admission tests, not accuracy certificates.
    """
    vectors_m: tuple
    shells: int
    height_m: float
    max_halvings: int
    absolute_tolerance: float
    max_shells: int
    bloch_rad_per_m: tuple = (0., 0., 0.)
    normal: tuple | None = None
    transverse_cutoff: float = 64.
    quadrature_order: int = 64
    transverse_map: str = 'linear'
    pole_relative_tolerance: float = 1e-10
    mode_batch: int = 4096
    query_batch: int = 32
    cache_max_bytes: int = 256*1024**2
    version: str = field(default='mom_bravais_dyadic_config_v1', init=False)

    def __post_init__(self):
        def vector(value):
            if (not isinstance(value, (tuple, list)) or len(value) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
                raise ValueError('Dyadic periodic vectors require three finite real coordinates')
            return tuple(float(v) for v in value)
        if not isinstance(self.vectors_m, (tuple, list)) or not 1 <= len(self.vectors_m) <= 3:
            raise ValueError('Dyadic periodic lattice requires one, two or three row vectors')
        object.__setattr__(self, 'vectors_m', tuple(vector(v) for v in self.vectors_m))
        object.__setattr__(self, 'bloch_rad_per_m', vector(self.bloch_rad_per_m))
        if self.normal is not None:
            object.__setattr__(self, 'normal', vector(self.normal))
        if len(self.vectors_m) == 1 and self.normal is None:
            raise ValueError('One-dimensional dyadic periodicity requires an explicit transverse normal')
        if len(self.vectors_m) != 1 and self.normal is not None:
            raise ValueError('Plane and stacked dyadic normals are determined by lattice rows')
        for name in ('height_m', 'absolute_tolerance', 'transverse_cutoff', 'pole_relative_tolerance'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be positive and finite')
            object.__setattr__(self, name, float(value))
        for name in ('shells', 'max_shells', 'max_halvings', 'quadrature_order', 'mode_batch', 'query_batch'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name+' must be a positive integer')
        if self.max_halvings < 3 or self.max_shells < 2*self.shells:
            raise ValueError('Dyadic periodic limits require three halvings and reciprocal doubling')
        if len(self.vectors_m) == 1 and self.max_shells != 2*self.shells:
            raise ValueError('One-dimensional dyadic spectra require an exact two-level reciprocal budget')
        if self.transverse_map not in ('linear', 'sinh'):
            raise ValueError('Dyadic transverse mapping must be linear or sinh')
        if type(self.cache_max_bytes) is not int or self.cache_max_bytes < 0:
            raise ValueError('Dyadic periodic cache budget must be a nonnegative integer')

    def payload(self):
        result = asdict(self)
        result['vectors_m'] = [list(v) for v in self.vectors_m]
        result['bloch_rad_per_m'] = list(self.bloch_rad_per_m)
        result['normal'] = None if self.normal is None else list(self.normal)
        return result

    @classmethod
    def restore(cls, payload):
        if (type(payload) is not dict or set(payload) != set(cls.__dataclass_fields__)
                or payload.get('version') != 'mom_bravais_dyadic_config_v1'):
            raise ValueError('Invalid or unsupported dyadic periodic configuration')
        return cls(**{key: value for key, value in payload.items() if key != 'version'})

    def compile(self, *, medium, integration, primary_cache_max_bytes=256*1024**2):
        from .periodic import LatticeSpec
        from .periodic_dyadic import QuasiperiodicDyadicGreenKernel, QuasiperiodicDyadicRegularKernel
        from .tensor_medium import TensorMedium
        if not isinstance(medium, TensorMedium):
            raise TypeError('Dyadic periodic compilation requires an exact TensorMedium binding')
        medium.assert_current()
        sample = medium.epsilon_r.real
        lattice = LatticeSpec(sample.new_tensor(self.vectors_m), sample.new_tensor(self.bloch_rad_per_m))
        spectral = QuasiperiodicDyadicGreenKernel(lattice, medium,
            pole_relative_tolerance=self.pole_relative_tolerance, mode_batch=self.mode_batch,
            query_batch=self.query_batch, cache_max_bytes=self.cache_max_bytes)
        return QuasiperiodicDyadicRegularKernel(spectral, self.shells, self.height_m,
            self.max_halvings, self.absolute_tolerance, self.max_shells,
            normal=None if self.normal is None else sample.new_tensor(self.normal),
            transverse_cutoff=self.transverse_cutoff, quadrature_order=self.quadrature_order,
            transverse_map=self.transverse_map, integration=integration,
            primary_cache_max_bytes=primary_cache_max_bytes)
