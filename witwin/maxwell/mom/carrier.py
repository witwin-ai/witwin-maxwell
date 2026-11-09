"""Immutable, versioned incident-field identities for one physical RHS."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from ..constants import complex_dtype_for
from ..sources import PointDipole
from .physics import source_phasor

CARRIER_VERSION = 'analytic_plane_wave_carrier_v1'
# Carriers with at least one exterior electric dipole use a separate version;
# plane-wave-only payloads and identities are unchanged.
POINT_DIPOLE_CARRIER_VERSION = 'analytic_plane_wave_point_dipole_carrier_v1'


def _complex_pair(value, name):
    try:
        result = complex(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite complex scalar.') from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f'{name} must be finite.')
    return (result.real, result.imag)


def _vector(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f'{name} must contain three finite coordinates.')
    result = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f'{name} must contain three finite coordinates.')
    return result


@dataclass(frozen=True)
class CarrierWave:
    direction: tuple
    polarization: tuple
    amplitude: tuple
    phase_origin: tuple = (0., 0., 0.)

    def __post_init__(self):
        direction = _vector(self.direction, 'Carrier direction')
        polarization = _vector(self.polarization, 'Carrier polarization')
        origin = _vector(self.phase_origin, 'Carrier phase origin')
        if (not math.isclose(sum(x*x for x in direction), 1., rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(sum(x*x for x in polarization), 1., rel_tol=1e-12, abs_tol=1e-12)
                or abs(sum(a*b for a, b in zip(direction, polarization))) > 1e-6):
            raise ValueError('Carrier direction and polarization must be orthonormal.')
        # Preserve the shared PlaneWave transverse tolerance and exact declared
        # vectors. Projecting here would change the physical source identity.
        if not isinstance(self.amplitude, (tuple, list)) or len(self.amplitude) != 2:
            raise ValueError('Carrier amplitude must have explicit real and imaginary components.')
        amplitude = _complex_pair(complex(*self.amplitude), 'Carrier amplitude')
        for name, value in (('direction', direction), ('polarization', polarization),
                            ('phase_origin', origin), ('amplitude', amplitude)):
            object.__setattr__(self, name, value)

    def payload(self):
        return {name: list(getattr(self, name)) for name in ('direction', 'polarization', 'amplitude', 'phase_origin')}


@dataclass(frozen=True)
class CarrierDipole:
    """Ideal exterior electric current moment Il [A*m] at a fixed position."""
    position: tuple
    moment: tuple

    def __post_init__(self):
        position = _vector(self.position, 'Carrier dipole position')
        if not isinstance(self.moment, (tuple, list)) or len(self.moment) != 3:
            raise ValueError('Carrier dipole moment must have three complex components.')
        moment = []
        for component in self.moment:
            if not isinstance(component, (tuple, list)) or len(component) != 2:
                raise ValueError('Carrier dipole moment components require real and imaginary parts.')
            moment.append(_complex_pair(complex(*component), 'Carrier dipole moment'))
        object.__setattr__(self, 'position', position)
        object.__setattr__(self, 'moment', tuple(moment))

    def payload(self):
        return {'position': list(self.position), 'moment': [list(component) for component in self.moment]}


@dataclass(frozen=True)
class IncidentCarrier:
    frequency: float
    epsilon_r: tuple
    mu_r: tuple
    k: tuple
    eta: tuple
    waves: tuple
    dipoles: tuple = ()

    def __post_init__(self):
        frequency = float(self.frequency)
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError('Carrier frequency must be finite and positive.')
        object.__setattr__(self, 'frequency', frequency)
        for name in ('epsilon_r', 'mu_r', 'k', 'eta'):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError('Carrier medium scalars require real and imaginary components.')
            object.__setattr__(self, name, _complex_pair(complex(*value), 'Carrier '+name))
        if not isinstance(self.waves, (tuple, list)) or any(type(w) is not CarrierWave for w in self.waves):
            raise ValueError('Carrier waves must be immutable CarrierWave records.')
        object.__setattr__(self, 'waves', tuple(self.waves))
        if not isinstance(self.dipoles, (tuple, list)) or any(type(d) is not CarrierDipole for d in self.dipoles):
            raise ValueError('Carrier dipoles must be immutable CarrierDipole records.')
        object.__setattr__(self, 'dipoles', tuple(self.dipoles))

    @classmethod
    def compile(cls, sources, exterior):
        waves, dipoles = [], []
        for source in sources:
            phasor = source_phasor(source, exterior.frequency)
            if isinstance(source, PointDipole):
                moment = tuple(_complex_pair(phasor*component, 'Carrier dipole moment')
                               for component in source.polarization)
                dipoles.append(CarrierDipole(tuple(source.position), moment))
                continue
            amplitude = _complex_pair(phasor, 'Carrier amplitude')
            waves.append(CarrierWave(tuple(source.direction), tuple(source.polarization), amplitude))
        return cls(exterior.frequency, _complex_pair(exterior.epsilon_r, 'epsilon_r'),
                   _complex_pair(exterior.mu_r, 'mu_r'), _complex_pair(exterior.k, 'k'),
                   _complex_pair(exterior.eta, 'eta'), waves, dipoles)

    def payload(self):
        payload = {'version': CARRIER_VERSION, 'phasor_convention': 'exp(-i omega t)',
                   'frequency': self.frequency, 'epsilon_r': list(self.epsilon_r),
                   'mu_r': list(self.mu_r), 'k': list(self.k), 'eta': list(self.eta),
                   'coordinate_units': 'm', 'electric_amplitude_units': 'V/m',
                   'waves': [wave.payload() for wave in self.waves]}
        if self.dipoles:
            from .dipole import CLEARANCE_RULE
            payload.update(version=POINT_DIPOLE_CARRIER_VERSION, dipole_moment_units='A*m',
                           dipole_kind='ideal_electric_current_moment', dipole_clearance_rule=CLEARANCE_RULE,
                           dipoles=[dipole.payload() for dipole in self.dipoles])
        return payload

    @property
    def identity(self):
        raw = json.dumps(self.payload(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def restore(cls, payload, *, expected_identity, sources, exterior):
        expected = cls.compile(sources, exterior)
        try:
            supplied_bytes = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            supplied_identity = hashlib.sha256(supplied_bytes).hexdigest()
        except (TypeError, ValueError) as error:
            raise ValueError('Incident carrier payload is not canonical finite data.') from error
        if (not isinstance(payload, dict) or payload != expected.payload()
                or supplied_identity != expected_identity or expected_identity != expected.identity):
            raise ValueError('Incident carrier does not match the declared sources, medium and frequency.')
        return expected

    def validate_medium(self, exterior):
        actual = (float(exterior.frequency), _complex_pair(exterior.epsilon_r, 'epsilon_r'),
                  _complex_pair(exterior.mu_r, 'mu_r'), _complex_pair(exterior.k, 'k'),
                  _complex_pair(exterior.eta, 'eta'))
        if actual != (self.frequency, self.epsilon_r, self.mu_r, self.k, self.eta):
            raise ValueError('Incident carrier background identity changed.')
        if (complex(*self.epsilon_r).imag != 0 or complex(*self.mu_r).imag != 0
                or complex(*self.k).imag != 0 or complex(*self.eta).real <= 0):
            raise ValueError('Incident carrier requires a lossless propagating exterior medium with positive power impedance.')

    def evaluate(self, points, exterior):
        """Evaluate one RHS's coherent superposition on the caller's CUDA device."""
        electric, magnetic = self.evaluate_many((self,), points, exterior)
        return electric[0], magnetic[0]

    @staticmethod
    def evaluate_many(carriers, points, exterior):
        """Evaluate several carriers on shared CUDA points: E/H [S,...,3].

        All plane waves share one phase evaluation and one host-to-device
        transfer of their declared vectors; each carrier sums only its own.
        """
        carriers = tuple(carriers)
        if not carriers or any(type(carrier) is not IncidentCarrier for carrier in carriers):
            raise TypeError('evaluate_many requires IncidentCarrier records.')
        for carrier in carriers:
            carrier.validate_medium(exterior)
        if (not isinstance(points, torch.Tensor) or points.device.type != 'cuda'
                or points.dtype not in (torch.float32, torch.float64)
                or points.ndim < 1 or points.shape[-1] != 3
                or not bool(torch.isfinite(points).all())):
            raise ValueError('Incident carrier points must be finite CUDA real vectors.')
        dtype = complex_dtype_for(points.dtype)
        electric = torch.zeros((len(carriers), *points.shape), dtype=dtype, device=points.device)
        magnetic = torch.zeros_like(electric)
        waves = [(index, wave) for index, carrier in enumerate(carriers) for wave in carrier.waves]
        if waves:
            table = points.new_tensor([[*w.direction, *w.polarization, *w.phase_origin] for _, w in waves])
            phases = torch.exp(1j*exterior.k*torch.stack(
                [(points-table[i, 6:9]) @ table[i, :3] for i in range(len(waves))]))
            # Waves accumulate elementwise in declaration order, so equal and
            # opposite waves cancel exactly (no fused multiply-add reductions).
            for i, (index, wave) in enumerate(waves):
                e = (complex(*wave.amplitude)*phases[i][..., None]*table[i, 3:6]).to(dtype)
                h = torch.linalg.cross(table[i, :3].to(dtype).expand_as(e), e)/exterior.eta
                electric[index], magnetic[index] = electric[index]+e, magnetic[index]+h
        dipoles = [(index, dipole) for index, carrier in enumerate(carriers) for dipole in carrier.dipoles]
        if dipoles:
            from .dipole import point_dipole_fields
            for index, dipole in dipoles:
                e, h = point_dipole_fields(points, dipole.position,
                                           tuple(complex(*c) for c in dipole.moment), exterior)
                electric[index], magnetic[index] = electric[index]+e, magnetic[index]+h
        return electric, magnetic
