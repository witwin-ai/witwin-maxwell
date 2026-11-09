"""Owned analytic Maxwell plane modes for homogeneous tensor backgrounds."""
from dataclasses import dataclass, field
import hashlib
import json
import math
import torch
from .tensor_medium import TensorMedium


@dataclass(frozen=True)
class TensorIncidentCarrier:
    """Validated mode amplitudes, not arbitrary scalar polarization vectors.

    Each row supplies physical q [1/m], E [V/m] and H [A/m] satisfying both
    homogeneous tensor Maxwell curl equations. Mode selection/source compilation
    is a separate responsibility; this internal carrier preserves their phase.
    """
    medium: TensorMedium
    wavevectors: torch.Tensor
    electric: torch.Tensor
    magnetic: torch.Tensor
    origin: torch.Tensor | None = None
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.medium, TensorMedium):
            raise TypeError('Tensor carrier requires TensorMedium')
        self.medium.assert_current()
        template = self.medium.epsilon_r
        for name in ('wavevectors', 'electric', 'magnetic'):
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 3
                    or not len(value) or value.device != template.device or value.dtype != template.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Tensor modes require finite matching complex CUDA [N,3] arrays')
            object.__setattr__(self, name, value.clone())
        if self.wavevectors.shape != self.electric.shape or self.magnetic.shape != self.electric.shape:
            raise ValueError('Every tensor mode must supply q, E and H')
        origin = template.real.new_zeros(3) if self.origin is None else self.origin
        if (not isinstance(origin, torch.Tensor) or origin.shape != (3,)
                or origin.device != template.device or origin.dtype != template.real.dtype
                or not bool(torch.isfinite(origin).all())):
            raise ValueError('Mode origin must be a finite matching real CUDA vector')
        object.__setattr__(self, 'origin', origin.clone())
        for left, right in ((torch.linalg.cross(self.wavevectors, self.electric),
                             self.medium.omega*(self.magnetic@self.medium.mu.T)),
                            (torch.linalg.cross(self.wavevectors, self.magnetic),
                             -self.medium.omega*(self.electric@self.medium.epsilon.T))):
            error = torch.linalg.vector_norm(left-right, dim=-1)
            scale = torch.maximum(left.norm(dim=-1), right.norm(dim=-1))
            if (not bool(torch.isfinite(error).all() & torch.isfinite(scale).all())
                    or bool((error > 4096*torch.finfo(scale.dtype).eps*scale).any())):
                raise ValueError('Incident tensor mode violates homogeneous Maxwell curl equations')
        if bool(((self.electric.abs().sum(-1)+self.magnetic.abs().sum(-1)) == 0).any()):
            raise ValueError('Incident tensor modes must have nonzero fields')
        object.__setattr__(self, '_versions', self._state())

    def _state(self):
        return tuple((id(value), value._version) for value in
                     (self.wavevectors, self.electric, self.magnetic, self.origin))

    def validate_medium(self, exterior):
        self.medium.assert_current()
        if exterior is not self.medium or self._state() != self._versions:
            raise ValueError('Tensor incident carrier or its bound medium changed')

    def validate_lattice(self, lattice):
        self.validate_medium(self.medium)
        lattice.assert_current()
        if lattice.vectors.device != self.origin.device or lattice.vectors.dtype != self.origin.dtype:
            raise ValueError('Tensor incident carrier and lattice must share device and precision')
        cycles = (self.wavevectors-lattice.bloch)@lattice.vectors.T.to(self.wavevectors.dtype)/(2*math.pi)
        tolerance = 4096*torch.finfo(self.origin.dtype).eps
        if bool((cycles.imag.abs() > tolerance).any() or
                ((cycles.real-cycles.real.round()).abs() > tolerance).any()):
            raise ValueError('Every tensor incident mode must share the real Bloch sector without axial attenuation')

    def evaluate(self, points, exterior):
        self.validate_medium(exterior)
        if (not isinstance(points, torch.Tensor) or points.ndim < 1 or points.shape[-1] != 3
                or points.device != self.origin.device or points.dtype != self.origin.dtype
                or not bool(torch.isfinite(points).all())):
            raise ValueError('Tensor carrier queries must be finite matching real CUDA vectors')
        phase = torch.exp(1j*((points-self.origin).to(self.wavevectors.dtype)@self.wavevectors.T))
        e, h = phase@self.electric, phase@self.magnetic
        if not bool(torch.isfinite(e).all() & torch.isfinite(h).all()):
            raise ValueError('Tensor incident mode evaluation overflowed')
        return e, h

    def payload(self):
        self.validate_medium(self.medium)
        def data(value):
            return torch.view_as_real(value.detach().resolve_conj()).cpu().tolist()
        return {'version': 'analytic_tensor_plane_modes_v1', 'phasor': 'exp(-i omega t)',
                'frequency_Hz': self.medium.frequency, 'dtype': str(self.electric.dtype),
                'reciprocal': self.medium.reciprocal,
                'epsilon_r': data(self.medium.epsilon_r), 'mu_r': data(self.medium.mu_r),
                'wavevectors_per_m': data(self.wavevectors), 'electric_V_per_m': data(self.electric),
                'magnetic_A_per_m': data(self.magnetic), 'origin_m': self.origin.detach().cpu().tolist()}

    @classmethod
    def from_payload(cls, payload, *, device, medium=None):
        """Restore fixed mode state on CUDA, optionally binding an exact exterior.

        Restoration rechecks constitutive passivity and both Maxwell equations.
        It preserves saved precision and values, but not a saved autograd graph.
        Bloch-sector validation remains the consuming lattice's responsibility.
        """
        keys = {'version', 'phasor', 'frequency_Hz', 'dtype', 'reciprocal',
                'epsilon_r', 'mu_r', 'wavevectors_per_m', 'electric_V_per_m',
                'magnetic_A_per_m', 'origin_m'}
        if type(payload) is not dict or set(payload) != keys:
            raise ValueError('Tensor carrier payload has missing or unknown fields')
        if (payload['version'] != 'analytic_tensor_plane_modes_v1'
                or payload['phasor'] != 'exp(-i omega t)'):
            raise ValueError('Unsupported tensor carrier version or phasor convention')
        dtypes = {'torch.complex64': torch.complex64, 'torch.complex128': torch.complex128}
        if type(payload['dtype']) is not str or payload['dtype'] not in dtypes:
            raise ValueError('Tensor carrier payload requires exact complex precision')
        frequency = payload['frequency_Hz']
        if (type(frequency) not in (int, float) or not math.isfinite(frequency)
                or frequency <= 0 or type(payload['reciprocal']) is not bool):
            raise ValueError('Tensor carrier payload has invalid frequency or reciprocity')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('Tensor carrier restoration requires CUDA')
        dtype = dtypes[payload['dtype']]
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

        def array(name, shape):
            def validate(value, dimensions):
                if not dimensions:
                    return type(value) in (int, float) and math.isfinite(value)
                return (type(value) is list and len(value) == dimensions[0]
                        and all(validate(item, dimensions[1:]) for item in value))
            if not validate(payload[name], shape):
                raise ValueError(f'Tensor carrier payload has invalid {name} shape or values')
            value = torch.tensor(payload[name], device=device, dtype=real_dtype)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Tensor carrier payload overflows saved precision in {name}')
            return value

        epsilon = torch.view_as_complex(array('epsilon_r', (3, 3, 2)))
        mu = torch.view_as_complex(array('mu_r', (3, 3, 2)))
        if medium is None:
            medium = TensorMedium(frequency, epsilon, mu, reciprocal=payload['reciprocal'])
        else:
            if not isinstance(medium, TensorMedium):
                raise TypeError('Restored tensor carrier requires a TensorMedium binding')
            medium.assert_current()
            if (medium.frequency != frequency or medium.reciprocal != payload['reciprocal']
                    or medium.epsilon_r.device != epsilon.device or medium.epsilon_r.dtype != dtype
                    or not torch.equal(medium.epsilon_r, epsilon) or not torch.equal(medium.mu_r, mu)):
                raise ValueError('Tensor carrier payload does not match the bound exterior medium')
        rows = payload['wavevectors_per_m']
        if type(rows) is not list or not rows:
            raise ValueError('Tensor carrier payload requires at least one mode')
        modes = [torch.view_as_complex(array(name, (len(rows), 3, 2))) for name in
                 ('wavevectors_per_m', 'electric_V_per_m', 'magnetic_A_per_m')]
        return cls(medium, *modes, origin=array('origin_m', (3,)))

    @property
    def identity(self):
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()


def compile_tensor_carrier(sources, exterior):
    """Bind physical mode declarations to one sampled exterior on CUDA.

    Each declaration is validated before combining modes. Different phase
    origins are translated to the first declaration's origin without detaching
    amplitudes, wavevectors or origins from their native PyTorch graphs.
    """
    from ..sources import MaxwellPlaneModes
    if not isinstance(exterior, TensorMedium):
        raise TypeError('Physical Maxwell mode compilation requires TensorMedium')
    exterior.assert_current()
    sources = tuple(sources)
    if not sources or any(not isinstance(source, MaxwellPlaneModes) for source in sources):
        raise TypeError('Tensor mode compilation requires nonempty MaxwellPlaneModes declarations')
    template = exterior.epsilon_r
    carriers = []
    for source in sources:
        if source.frequency != exterior.frequency:
            raise ValueError('Maxwell mode source and sampled exterior frequencies must match')
        carriers.append(TensorIncidentCarrier(exterior,
            source.wavevectors.to(device=template.device, dtype=template.dtype),
            source.electric.to(device=template.device, dtype=template.dtype),
            source.magnetic.to(device=template.device, dtype=template.dtype),
            origin=source.origin.to(device=template.device, dtype=template.real.dtype)))
    origin = carriers[0].origin
    electric, magnetic = [], []
    for carrier in carriers:
        phase = torch.exp(1j*(carrier.wavevectors@(origin-carrier.origin).to(template.dtype)))[:, None]
        electric.append(carrier.electric*phase)
        magnetic.append(carrier.magnetic*phase)
    return TensorIncidentCarrier(exterior,
        torch.cat([carrier.wavevectors for carrier in carriers]),
        torch.cat(electric), torch.cat(magnetic), origin=origin)
