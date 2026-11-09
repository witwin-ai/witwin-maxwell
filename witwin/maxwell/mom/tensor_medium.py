"""Sampled homogeneous tensor contract for the dyadic MoM implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch
from ..constants import EPSILON_0, MU_0


@dataclass(frozen=True)
class TensorMedium:
    """Owned complex CUDA constitutive tensors under exp(-i omega t).

    Reciprocity is transpose symmetry, not Hermitian symmetry. Passivity is
    positive semidefiniteness of (T-T^H)/(2i); imaginary antisymmetric entries
    of a lossless gyrotropic tensor are not absorption. Validation tolerances
    cover floating-point roundoff, not a user-controlled gain allowance.
    """
    frequency: float
    epsilon_r: torch.Tensor
    mu_r: torch.Tensor
    reciprocal: bool = True
    _versions: tuple = field(init=False, repr=False)
    validation: dict = field(init=False)

    def __post_init__(self):
        if isinstance(self.frequency, bool) or not math.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError('Tensor-medium frequency must be finite and positive in Hz')
        if type(self.reciprocal) is not bool:
            raise ValueError('reciprocal must be an explicit boolean')
        values, records = [], {}
        for name in ('epsilon_r', 'mu_r'):
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cuda'
                    or value.dtype not in (torch.complex64, torch.complex128) or value.shape != (3, 3)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f'{name} must be a finite complex CUDA 3x3 tensor')
            if values and (value.device != values[0].device or value.dtype != values[0].dtype):
                raise ValueError('Constitutive tensors must share device and precision')
            value = value.clone()
            scale = torch.linalg.matrix_norm(value, ord=2)
            tolerance = 64*torch.finfo(value.real.dtype).eps*scale
            loss = (value-value.mH)/(2j)
            minimum_loss = torch.linalg.eigvalsh(loss).amin()
            if bool(minimum_loss < -tolerance):
                raise ValueError(f'{name} violates passive anti-Hermitian constitutive response')
            transpose_error = torch.linalg.matrix_norm(value-value.mT, ord=2)
            if self.reciprocal and bool(transpose_error > tolerance):
                raise ValueError(f'{name} violates reciprocal transpose symmetry')
            object.__setattr__(self, name, value)
            values.append(value)
            records[name] = {'minimum_loss_eigenvalue': float(minimum_loss.detach()),
                'roundoff_tolerance': float(tolerance.detach()), 'transpose_error': float(transpose_error.detach())}
        object.__setattr__(self, '_versions', tuple((id(value), value._version) for value in values))
        object.__setattr__(self, 'validation', records)

    def assert_current(self):
        if tuple((id(v), v._version) for v in (self.epsilon_r, self.mu_r)) != self._versions:
            raise ValueError('Tensor medium changed after constitutive validation; compile a new sample')

    @property
    def omega(self): return 2*math.pi*self.frequency
    @property
    def epsilon(self):
        self.assert_current()
        return EPSILON_0*self.epsilon_r
    @property
    def mu(self):
        self.assert_current()
        return MU_0*self.mu_r

    def dissipated_power_density(self, electric, magnetic):
        """Peak-phasor material dissipation [W/m^3], retaining native gradients."""
        self.assert_current()
        if (not isinstance(electric, torch.Tensor) or not isinstance(magnetic, torch.Tensor)
                or electric.shape != magnetic.shape or electric.shape[-1:] != (3,)
                or electric.device != self.epsilon_r.device or magnetic.device != electric.device
                or electric.dtype != self.epsilon_r.dtype or magnetic.dtype != electric.dtype
                or not bool(torch.isfinite(electric).all() & torch.isfinite(magnetic).all())):
            raise ValueError('E/H must be finite matching complex CUDA [...,3] fields')
        electric_loss = (self.epsilon-self.epsilon.mH)/(2j)
        magnetic_loss = (self.mu-self.mu.mH)/(2j)
        return .5*self.omega*(torch.einsum('...i,ij,...j->...', electric.conj(), electric_loss, electric).real
                            +torch.einsum('...i,ij,...j->...', magnetic.conj(), magnetic_loss, magnetic).real)



def scalar_medium_if_isotropic(medium):
    """Return the exact fixed scalar specialization, or None for a tensor sample."""
    import cmath
    from .physics import Medium
    medium.assert_current()
    eye = torch.eye(3, device=medium.epsilon_r.device, dtype=medium.epsilon_r.dtype)
    if not (torch.equal(medium.epsilon_r, medium.epsilon_r[0, 0]*eye)
            and torch.equal(medium.mu_r, medium.mu_r[0, 0]*eye)):
        return None
    if medium.epsilon_r.requires_grad or medium.mu_r.requires_grad:
        raise ValueError('Fixed scalar specialization cannot detach differentiable tensor materials')
    eps, mu = complex(medium.epsilon_r[0, 0]), complex(medium.mu_r[0, 0])
    if eps == 0 or mu == 0:
        raise ValueError('Zero constitutive parameter requires a separate limiting formulation')
    k = medium.omega*math.sqrt(EPSILON_0*MU_0)*cmath.sqrt(eps)*cmath.sqrt(mu)
    if k.imag < 0: k = -k
    eta = medium.omega*MU_0*mu/k
    if k.imag == 0 and eta.real < 0: k, eta = -k, -eta
    return Medium(medium.frequency, eps, mu, k, eta, {'origin': 'exact_isotropic_tensor_sample'})



def compile_tensor_medium(material, frequency, *, device, dtype=torch.complex128):
    """Compile existing public scalar, reciprocal-tensor and ferrite materials.

    Static tensor declarations replace the corresponding scalar base; scalar
    pole susceptibilities add isotropically. A declared conductivity tensor
    replaces scalar electric conductivity, matching Material.evaluate_static.
    """
    from witwin.core import PhysicalMaterial
    from ..media import Material, DiagonalTensor3, Tensor3x3, ComplexTensor3x3, GyromagneticFerrite
    from .physics import _finite_scalar, _pole_sum
    if type(material) not in (PhysicalMaterial, Material, GyromagneticFerrite):
        raise TypeError('Tensor MoM requires an explicitly supported material model')
    device = torch.device(device)
    if device.type != 'cuda' or dtype not in (torch.complex64, torch.complex128):
        raise ValueError('Tensor material compilation requires CUDA complex precision')
    if isinstance(material, Material) and (material.is_nonlinear or material.pec
            or material.modulation is not None or material.breakdown is not None or material.has_custom_poles):
        raise ValueError('Tensor MoM requires a homogeneous linear constitutive model')
    magnetic = isinstance(material, GyromagneticFerrite)
    sigma_tensor = getattr(material, 'sigma_e_tensor', None)
    kwargs = {name: getattr(material, name, ()) for name in
              ('debye_poles', 'drude_poles', 'lorentz_poles', 'mu_debye_poles', 'mu_drude_poles', 'mu_lorentz_poles')}
    if isinstance(frequency, bool) or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError('Tensor-medium frequency must be finite and positive in Hz')
    omega = 2*math.pi*frequency
    if any(getattr(pole, 'allow_gain', False) for family in kwargs.values() for pole in family):
        raise ValueError('Tensor MoM excludes gain-enabled material poles')
    electric_poles = sum((kwargs[name] for name in ('debye_poles', 'drude_poles', 'lorentz_poles')), ())
    magnetic_poles = sum((kwargs[name] for name in ('mu_debye_poles', 'mu_drude_poles', 'mu_lorentz_poles')), ())
    sigma_e = _finite_scalar(0. if sigma_tensor is not None else material.sigma_e, 'sigma_e')
    sigma_m = _finite_scalar(getattr(material, 'sigma_m', 0.), 'sigma_m')
    for conductivity in (sigma_e, sigma_m):
        if conductivity.imag != 0 or conductivity.real < 0:
            raise ValueError('Scalar conductivity must be real and nonnegative')
    electric_increment = _pole_sum(electric_poles, omega)+1j*sigma_e.real/(omega*EPSILON_0)
    magnetic_increment = _pole_sum(magnetic_poles, omega)+1j*sigma_m.real/(omega*MU_0)
    eye = torch.eye(3, device=device, dtype=dtype)

    def tensor(declaration, fallback):
        if declaration is None: return complex(fallback)*eye
        if isinstance(declaration, DiagonalTensor3):
            return torch.diag(torch.tensor(declaration.as_tuple(), device=device, dtype=dtype))
        if isinstance(declaration, (Tensor3x3, ComplexTensor3x3)):
            value = torch.tensor(declaration.rows, device=device, dtype=dtype)
            if isinstance(declaration, ComplexTensor3x3) and declaration.reciprocal:
                tolerance = 64*torch.finfo(value.real.dtype).eps*torch.linalg.matrix_norm(value, ord=2)
                if bool(torch.linalg.matrix_norm(value-value.mT, ord=2) > tolerance):
                    raise ValueError('Constitutive declaration violates reciprocal transpose symmetry')
            return value
        raise TypeError('Unsupported constitutive tensor declaration')

    epsilon = tensor(getattr(material, 'epsilon_tensor', None), material.eps_r)
    epsilon = epsilon+electric_increment*eye
    if sigma_tensor is not None:
        sigma = tensor(sigma_tensor, 0.)
        if not bool(torch.isfinite(sigma).all()) or bool((torch.linalg.eigvalsh(sigma).real < 0).any()):
            raise ValueError('Electric conductivity tensor must be finite and positive semidefinite')
        epsilon = epsilon+1j*sigma/(omega*EPSILON_0)
    if magnetic:
        frequency_tensor = torch.tensor(float(frequency), device=device, dtype=eye.real.dtype)
        mu = material.permeability_tensor_at_freq(frequency_tensor, dtype=dtype)
    else:
        mu = tensor(getattr(material, 'mu_tensor', None), material.mu_r)
        mu = mu+magnetic_increment*eye
    reciprocal = not magnetic and all(getattr(value, 'reciprocal', True) for value in
        (getattr(material, 'epsilon_tensor', None), getattr(material, 'mu_tensor', None)))
    return TensorMedium(float(frequency), epsilon, mu, reciprocal=reciprocal)


def sampled_medium_payload(medium):
    """Serialize sampled constitutive data, excluding caches and tensor versions.

    Scalar payloads retain their existing schema. Tensor data is a versioned,
    device-independent, exact real/imaginary metadata record; serialization is
    an explicit host transfer, not a numerical CPU execution path.
    """
    from dataclasses import asdict
    from .physics import Medium
    if isinstance(medium, Medium):
        return asdict(medium)
    if not isinstance(medium, TensorMedium):
        raise TypeError('Unsupported sampled MoM medium')
    medium.assert_current()
    def components(value):
        return torch.view_as_real(value.detach().resolve_conj()).cpu().tolist()
    return {'schema_version': 1, 'medium_kind': 'homogeneous_passive_tensor',
        'phasor_convention': 'exp(-i omega t)', 'frequency_Hz': float(medium.frequency),
        'dtype': str(medium.epsilon_r.dtype), 'reciprocal': medium.reciprocal,
        'epsilon_r_real_imag': components(medium.epsilon_r),
        'mu_r_real_imag': components(medium.mu_r)}


def restore_tensor_medium(payload, *, device):
    """Restore exact tensor samples on CUDA and repeat all physical validation."""
    keys = {'schema_version', 'medium_kind', 'phasor_convention', 'frequency_Hz',
            'dtype', 'reciprocal', 'epsilon_r_real_imag', 'mu_r_real_imag'}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError('Tensor medium payload has missing or unsupported fields')
    if (type(payload['schema_version']) is not int or payload['schema_version'] != 1
            or payload['medium_kind'] != 'homogeneous_passive_tensor'
            or payload['phasor_convention'] != 'exp(-i omega t)'):
        raise ValueError('Unsupported tensor medium schema or phasor convention')
    if not isinstance(payload['dtype'], str):
        raise ValueError('Tensor medium payload precision must be an explicit string')
    dtype = {'torch.complex64': torch.complex64, 'torch.complex128': torch.complex128}.get(payload['dtype'])
    if dtype is None or type(payload['reciprocal']) is not bool:
        raise ValueError('Tensor medium payload requires explicit precision and reciprocity')
    frequency = payload['frequency_Hz']
    if type(frequency) not in (float, int) or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError('Tensor medium payload frequency must be finite and positive')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Tensor medium restoration requires CUDA')
    values = []
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    for name in ('epsilon_r_real_imag', 'mu_r_real_imag'):
        rows = payload[name]
        if (not isinstance(rows, list) or len(rows) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in rows)
                or any(not isinstance(pair, list) or len(pair) != 2 for row in rows for pair in row)
                or any(type(v) not in (int, float) or not math.isfinite(v) for row in rows for pair in row for v in pair)):
            raise ValueError(f'{name} must contain finite numeric real/imaginary pairs in a 3x3 array')
        value = torch.view_as_complex(torch.tensor(rows, device=device, dtype=real_dtype))
        # Prevent malformed precision declarations from silently rounding values.
        if torch.view_as_real(value).cpu().tolist() != rows:
            raise ValueError('Tensor sample values are not exact in their declared precision')
        values.append(value)
    return TensorMedium(float(frequency), *values, reciprocal=payload['reciprocal'])


def sampled_tensor_identity(medium):
    """Stable sample identity for binding material records to native results."""
    import hashlib
    import json
    if not isinstance(medium, TensorMedium):
        raise TypeError('Tensor sample identity requires TensorMedium')
    encoded = json.dumps(sampled_medium_payload(medium), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()
