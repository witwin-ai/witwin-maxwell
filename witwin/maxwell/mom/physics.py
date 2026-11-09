"""SI physical contract for the homogeneous scalar electromagnetic MoM backend.

All phasors multiply ``exp(-i omega t)``. The radiating scalar Green function
therefore uses ``exp(+i k r)`` and passive media have ``Im(k) >= 0``.
Material/source configuration is host metadata; field evaluation stays on CUDA.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math

import torch
from witwin.core import PhysicalMaterial as CoreMaterial

from ..constants import EPSILON_0, MU_0, complex_dtype_for
from ..media import Material
from ..scene import Scene
from ..sources import CW, PlaneWave, PointDipole


def _finite_scalar(value, name: str) -> complex:
    if torch.is_tensor(value) and value.ndim != 0:
        raise ValueError(f"MoM {name} must be a homogeneous scalar.")
    try:
        scalar = complex(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"MoM {name} must be a homogeneous scalar.") from error
    if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
        raise ValueError(f"MoM {name} must be finite.")
    return scalar


@dataclass(frozen=True)
class Medium:
    """One homogeneous region, with the outgoing branch fixed before impedance."""

    frequency: float
    epsilon_r: complex
    mu_r: complex
    k: complex
    eta: complex
    metadata: dict

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.frequency

    @property
    def epsilon(self) -> complex:
        return EPSILON_0 * self.epsilon_r

    @property
    def mu(self) -> complex:
        return MU_0 * self.mu_r


def _pole_sum(poles, omega):
    try:
        result = sum(pole.susceptibility(omega) for pole in poles)
    except (ZeroDivisionError, OverflowError) as error:
        raise ValueError('MoM sampled an exact material pole; a finite constitutive response is required') from error
    return _finite_scalar(result, 'pole susceptibility')


def compile_medium(material, frequency: float, *, background: bool = False) -> Medium:
    """Compile known scalar materials without double-counting conductivity.

    The core static material's frequency sampler uses a different loss sign.
    Read its explicit constitutive parameters here instead. Maxwell poles
    already use this module's phasor convention. Unknown material subclasses
    require their own explicit adapter, so cannot silently lose their physics.
    """
    frequency = float(frequency)
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("MoM frequency must be finite and positive in Hz.")
    if type(material) not in (CoreMaterial, Material):
        raise TypeError("MoM requires a supported homogeneous scalar Material model.")
    if isinstance(material, Material):
        unsupported = (
            material.is_anisotropic or material.is_nonlinear or material.pec
            or material.modulation is not None or material.breakdown is not None
            or material.has_custom_poles
        )
        if unsupported:
            raise ValueError("Core MoM requires linear homogeneous isotropic material.")
    eps_base = _finite_scalar(material.eps_r, "eps_r")
    mu_base = _finite_scalar(material.mu_r, "mu_r")
    sigma_e = _finite_scalar(material.sigma_e, "sigma_e")
    sigma_m = _finite_scalar(getattr(material, 'sigma_m', 0.), "sigma_m")
    for name, sigma in (('sigma_e', sigma_e), ('sigma_m', sigma_m)):
        if sigma.imag != 0 or sigma.real < 0:
            raise ValueError(f"MoM {name} must be real and nonnegative")
    if eps_base.imag < 0 or mu_base.imag < 0:
        raise ValueError("MoM requires passive electric and magnetic parameters under exp(-i omega t)")
    omega = 2.0 * math.pi * frequency
    electric_poles, magnetic_poles = (), ()
    if isinstance(material, Material):
        electric_poles = material.debye_poles + material.drude_poles + material.lorentz_poles
        magnetic_poles = material.mu_debye_poles + material.mu_drude_poles + material.mu_lorentz_poles
        if any(getattr(pole, "allow_gain", False) for pole in (*electric_poles, *magnetic_poles)):
            raise ValueError("Core MoM excludes gain-enabled material models")
    epsilon_r = _finite_scalar(eps_base+1j*sigma_e.real/(omega*EPSILON_0)
        +_pole_sum(electric_poles, omega), 'sampled epsilon_r')
    mu_r = _finite_scalar(mu_base+1j*sigma_m.real/(omega*MU_0)
        +_pole_sum(magnetic_poles, omega), 'sampled mu_r')
    if epsilon_r.imag < 0 or mu_r.imag < 0:
        raise ValueError("MoM requires passive sampled epsilon_r and mu_r")
    if background and (epsilon_r.imag != 0 or mu_r.imag != 0):
        raise ValueError("Core MoM background must be homogeneous and lossless")
    if epsilon_r == 0 or mu_r == 0:
        raise ValueError("Exactly zero constitutive parameters require a static-limit formulation; nonzero near-zero values are supported")
    k, eta = _outgoing_parameters(frequency, epsilon_r, mu_r)
    if background and (k.imag != 0 or eta.real <= 0):
        raise ValueError('MoM incident background must support propagating waves with positive power impedance')
    return Medium(frequency, epsilon_r, mu_r, k, eta, {
        'units': 'SI', 'input_convention': 'exp(-i omega t)',
        'loss_accounting': 'epsilon_r=eps_base+chi_e+i*sigma_e/(omega*epsilon_0); mu_r=mu_base+chi_m+i*sigma_m/(omega*mu_0); each conductivity counted once',
        'model_id': f'{type(material).__module__}.{type(material).__qualname__}', 'model_name': material.name,
        'constitutive_parameters': {'eps_base': eps_base, 'mu_base': mu_base,
            'sigma_e': sigma_e.real, 'sigma_m': sigma_m.real,
            'poles': tuple(repr(pole) for pole in electric_poles),
            'magnetic_poles': tuple(repr(pole) for pole in magnetic_poles)},
        'evaluation_frequency_hz': frequency, 'validated_frequency_band_hz': (frequency, frequency),
        'k_branch': 'passive_sqrt_epsilon_times_sqrt_mu_v1; Im(k)>=0; positive power for propagating waves',
        'impedance_definition': 'eta=omega*mu/k'})


def _outgoing_parameters(frequency, epsilon_r, mu_r):
    """Common sampled-medium passive branch for compilation and exact restore."""
    omega = 2.0 * math.pi * frequency
    # The product's principal square root alone loses the passive branch for
    # double-negative media. Multiplying separately passive roots preserves the
    # limiting-absorption branch, including negative real k and positive power.
    k = omega*math.sqrt(EPSILON_0*MU_0)*cmath.sqrt(epsilon_r)*cmath.sqrt(mu_r)
    if k.imag < 0:
        k = -k
    eta = omega*MU_0*mu_r/k
    if k.imag == 0 and eta.real < 0:
        k, eta = -k, -eta
    if not all(math.isfinite(v) for z in (k, eta) for v in (z.real, z.imag)):
        raise ValueError('MoM constitutive sample produced a nonfinite wavenumber or impedance')
    return k, eta


def restore_scalar_medium(payload):
    """Validate an exact sampled scalar medium without resampling a declaration."""
    import copy
    if type(payload) is not dict or set(payload)!={'frequency','epsilon_r','mu_r','k','eta','metadata'}:
        raise ValueError('Invalid sampled scalar medium fields')
    frequency=payload['frequency']
    if type(frequency) not in (int,float) or not math.isfinite(frequency) or frequency<=0:
        raise ValueError('Invalid sampled scalar frequency')
    epsilon,mu=(_finite_scalar(payload[k],k) for k in ('epsilon_r','mu_r'))
    if epsilon.imag<0 or mu.imag<0 or epsilon==0 or mu==0:
        raise ValueError('Sampled scalar medium requires passive nonzero constitutive parameters')
    k,eta=_outgoing_parameters(frequency,epsilon,mu)
    if payload['k']!=k or payload['eta']!=eta or type(payload['metadata']) is not dict:
        raise ValueError('Sampled scalar outgoing parameters or metadata differ')
    return Medium(frequency,epsilon,mu,k,eta,copy.deepcopy(payload['metadata']))


def source_phasor(source: PlaneWave | PointDipole, frequency: float) -> complex:
    """Map the shared CW convention A*cos(omega*t+phase) to this phasor.

    For a PlaneWave the phasor is the electric amplitude [V/m]; for an
    electric PointDipole it scales the unit polarization to the ideal current
    moment Il [A*m].
    """
    if isinstance(source, PointDipole):
        kind = 'PointDipole'
        if source.profile != 'ideal':
            raise ValueError("MoM PointDipole requires profile='ideal': the surface solver radiates an exact point "
                             "current moment, while a Gaussian profile is a finite-width grid current distribution "
                             "with a different exterior field.")
    elif isinstance(source, PlaneWave):
        kind = 'PlaneWave'
    else:
        raise ValueError("Core MoM supports external PlaneWave and electric PointDipole sources only; magnetic "
                         "dipoles, current distributions, beams and mode sources are not implemented.")
    waveform = source.source_time
    if waveform is None:
        return 1.0 + 0j
    if not isinstance(waveform, CW):
        raise ValueError(f"MoM {kind} source_time must be CW or None; pulse normalization is not implicit.")
    if not math.isclose(waveform.frequency, frequency, rel_tol=1e-12, abs_tol=0):
        raise ValueError("MoM and CW source frequencies must match.")
    amplitude = _finite_scalar(waveform.amplitude, "source amplitude")
    phase = _finite_scalar(waveform.phase, "source phase")
    return amplitude * cmath.exp(-1j * phase.real)


def validate_scene_structure(scene: Scene, frequency: float, *, tensor_exterior=False, point_dipoles=False,
                             formulation='pmchwt'):
    """Shared scene scope/source checks, before linear or harmonic sampling.

    ``point_dipoles`` is enabled only by the linear dense free-space scalar
    runtimes, whose analytic incident fields support exterior dipoles.
    ``formulation`` names the surface formulation; every enabled body kind
    must belong to it (``mom.topology``), so PEC and sheet bodies never reach
    the dielectric material compiler and vice versa.
    """
    if not isinstance(scene, Scene):
        raise TypeError("MoM preparation requires the shared Scene object.")
    if getattr(scene.background, 'perturbation', None) is not None:
        raise NotImplementedError('MoM homogeneous backgrounds do not support material perturbation fields')
    if scene.background is not None and not tensor_exterior:
        raise NotImplementedError('An explicit homogeneous background requires the dedicated dyadic exterior runtime')
    structures = [structure for structure in scene.structures if structure.enabled]
    if not structures:
        raise ValueError("MoM requires at least one enabled homogeneous structure.")
    from .topology import require_formulation
    require_formulation(structures, formulation)
    for feature in ("material_regions", "thin_wires", "ports", "networks",
                    "lumped_elements", "circuits", "electrostatic_terminals", "charge_densities"):
        if getattr(scene, feature, ()):
            raise ValueError(f"Core MoM does not support Scene.{feature}.")
    if any(kind not in ("none", "pml") for pair in scene.boundary.face_kinds for kind in pair):
        raise ValueError("Core MoM requires an unbounded exterior, without periodic or reflecting boundaries.")
    if any(axis is not None for axis in scene.symmetry):
        raise ValueError("Core MoM does not support symmetry-reduced scenes.")
    for source in scene.sources:
        if tensor_exterior:
            from ..sources import MaxwellPlaneModes
            if not isinstance(source, MaxwellPlaneModes):
                raise TypeError('Dyadic exterior scenes require physical MaxwellPlaneModes sources')
            if source.frequency != frequency:
                raise ValueError('Maxwell mode source and simulation frequencies must match')
        else:
            if isinstance(source, PointDipole) and not point_dipoles:
                raise ValueError("MoM PointDipole sources require the linear dense free-space scalar runtime "
                                 "(operator_backend='dense' without periodic, dyadic, harmonic-balance or "
                                 "linear-coupled configuration); this path supports PlaneWave only.")
            source_phasor(source, frequency)
    return structures


def validate_scene_system(scene: Scene, frequency: float, *, tensor_materials=False,
                          tensor_exterior=False, dtype=torch.complex128, point_dipoles=False):
    """Validate shared scene scope and compile its linear scalar materials."""
    structures = validate_scene_structure(scene, frequency, tensor_exterior=tensor_exterior,
                                          point_dipoles=point_dipoles)
    if tensor_exterior:
        from .tensor_medium import compile_tensor_medium
        exterior = compile_tensor_medium(Material() if scene.background is None else scene.background,
            frequency, device=scene.device, dtype=dtype)
    else:
        exterior = compile_medium(Material(), frequency, background=True)
    if tensor_materials or tensor_exterior:
        from .tensor_medium import compile_tensor_medium
        from ..media import GyromagneticFerrite
        interiors = tuple(compile_tensor_medium(s.material, frequency, device=scene.device, dtype=dtype)
            if tensor_exterior or isinstance(s.material, GyromagneticFerrite) or getattr(s.material, 'is_anisotropic', False)
            else compile_medium(s.material, frequency) for s in structures)
    else:
        interiors = tuple(compile_medium(structure.material, frequency) for structure in structures)
    return exterior, interiors


def validate_scene(scene: Scene, frequency: float) -> tuple[Medium, Medium]:
    """Single-body compiler entry; multi-body callers use validate_scene_system."""
    exterior, interiors = validate_scene_system(scene, frequency)
    if len(interiors) != 1:
        raise ValueError("Core MoM requires exactly one enabled homogeneous dielectric structure.")
    return exterior, interiors[0]


def incident_fields(
    source: PlaneWave | PointDipole, points: torch.Tensor, medium: Medium, *,
    amplitude: complex | None = None, phase_origin=(0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exterior-medium incident E [V/m] and H [A/m].

    Plane waves use ``H = direction x E / eta``. Electric point dipoles use the
    exact near- and far-field expressions of ``mom.dipole``; their phase is
    referenced to the dipole itself, so ``phase_origin`` must stay default.
    An explicit amplitude replaces the source CW phasor, enabling complex source
    fixtures or linear combinations without changing operator parameters q.
    Coordinates and phase origin are in metres. No numerical CPU fallback.
    """
    default_amplitude = source_phasor(source, medium.frequency)
    if not isinstance(points, torch.Tensor) or points.device.type != "cuda":
        raise ValueError("MoM incident field evaluation requires CUDA tensor points.")
    if points.dtype not in (torch.float32, torch.float64) or points.shape[-1:] != (3,):
        raise ValueError("MoM query points must be real float32/float64 with shape (..., 3).")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("MoM query points must be finite.")
    if medium.epsilon_r.imag != 0 or medium.mu_r.imag != 0 or medium.k.imag != 0 or medium.eta.real <= 0:
        raise ValueError("Core MoM incident waves require a lossless background.")
    phasor = default_amplitude if amplitude is None else _finite_scalar(amplitude, "source amplitude")
    if isinstance(source, PointDipole):
        if tuple(float(x) for x in phase_origin) != (0.0, 0.0, 0.0):
            raise ValueError("MoM PointDipole fields are referenced to the dipole position; phase_origin is undefined.")
        from .dipole import point_dipole_fields
        return point_dipole_fields(points, source.position,
                                   tuple(phasor*component for component in source.polarization), medium)
    origin = torch.as_tensor(phase_origin, dtype=points.dtype, device=points.device)
    if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
        raise ValueError("MoM phase_origin must contain three finite coordinates in metres.")
    direction = torch.tensor(source.direction, dtype=points.dtype, device=points.device)
    polarization = torch.tensor(source.polarization, dtype=points.dtype, device=points.device)
    if not bool(torch.isfinite(direction).all() & torch.isfinite(polarization).all()):
        raise ValueError("MoM source direction and polarization must be finite.")
    phase = torch.exp(1j * medium.k * ((points - origin) @ direction))
    electric = (phasor * phase[..., None] * polarization).to(complex_dtype_for(points.dtype))
    magnetic = torch.linalg.cross(direction.to(electric.dtype).expand_as(electric), electric) / medium.eta
    return electric, magnetic
