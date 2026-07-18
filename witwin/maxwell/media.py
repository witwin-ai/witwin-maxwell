from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from witwin.core import (
    FrequencyMaterialSample,
    Material as CoreMaterial,
    MaterialCapabilities,
    StaticMaterialSample,
)
from witwin.core.material import VACUUM_PERMITTIVITY

_SPEED_OF_LIGHT = 299_792_458.0
_DEFAULT_GYROMAGNETIC_RATIO = 1.760859e11  # rad/(s*T), electron gyromagnetic ratio magnitude
_OERSTED_TO_A_PER_M = 1.0e-4 / (4.0e-7 * np.pi)  # 1 Oe of H -> A/m (= 1000/(4*pi))


def _coerce_frequency(value: float, *, name: str) -> float:
    frequency = float(value)
    if frequency <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return frequency


def _coerce_nonnegative(value: float, *, name: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return number


def _coerce_positive(value: float, *, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return number


def _normalize_poles(value, pole_types, *, name: str):
    if not isinstance(pole_types, tuple):
        pole_types = (pole_types,)
    label = " or ".join(pole_type.__name__ for pole_type in pole_types)
    if value is None:
        return ()
    if isinstance(value, pole_types):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        normalized = tuple(value)
        if any(not isinstance(item, pole_types) for item in normalized):
            raise TypeError(f"{name} entries must be {label} instances.")
        return normalized
    raise TypeError(f"{name} must be a {label} or an iterable of them.")


def _normalize_tensor_rows(value, *, name: str):
    rows = tuple(tuple(float(component) for component in row) for row in value)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"{name} must be a 3x3 tensor.")
    return rows


def _coerce_real_scalar(value, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


_TENSOR_SYMMETRY_RTOL = 1.0e-6


def _validate_symmetric_positive_definite(tensor: "Tensor3x3", *, name: str) -> None:
    """Reject full 3x3 tensors that are not symmetric positive-definite.

    A lossless anisotropic permittivity must be an SPD tensor; asymmetric or
    indefinite tensors would make the FDTD update unconditionally unstable, so
    they fail loudly at construction time.
    """
    matrix = np.asarray(tensor.rows, dtype=np.float64)
    scale = max(float(np.abs(matrix).max()), 1.0)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_TENSOR_SYMMETRY_RTOL * scale):
        raise ValueError(f"{name} must be a symmetric 3x3 tensor.")
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if float(eigenvalues.min()) <= 0.0:
        raise ValueError(
            f"{name} must be positive-definite; got eigenvalues {tuple(float(v) for v in eigenvalues)}."
        )


@dataclass(frozen=True)
class DebyePole:
    delta_eps: float
    tau: float

    def __post_init__(self):
        object.__setattr__(self, "delta_eps", _coerce_nonnegative(self.delta_eps, name="delta_eps"))
        object.__setattr__(self, "tau", _coerce_positive(self.tau, name="tau"))

    def susceptibility(self, angular_frequency: float) -> complex:
        return self.delta_eps / (1.0 - 1j * float(angular_frequency) * self.tau)

    def susceptibility_at_freq(self, frequency: float) -> complex:
        return self.susceptibility(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


@dataclass(frozen=True)
class DrudePole:
    plasma_frequency: float
    gamma: float

    def __post_init__(self):
        object.__setattr__(self, "plasma_frequency", _coerce_frequency(self.plasma_frequency, name="plasma_frequency"))
        object.__setattr__(self, "gamma", _coerce_nonnegative(self.gamma, name="gamma"))

    @property
    def plasma_angular_frequency(self) -> float:
        return 2.0 * np.pi * self.plasma_frequency

    @property
    def gamma_angular_frequency(self) -> float:
        return 2.0 * np.pi * self.gamma

    def susceptibility(self, angular_frequency: float) -> complex:
        angular_frequency = float(angular_frequency)
        if angular_frequency <= 0.0:
            raise ValueError("Drude susceptibility is undefined at zero frequency.")
        omega_p = self.plasma_angular_frequency
        gamma = self.gamma_angular_frequency
        return -(omega_p * omega_p) / (
            angular_frequency * angular_frequency + 1j * gamma * angular_frequency
        )

    def susceptibility_at_freq(self, frequency: float) -> complex:
        return self.susceptibility(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


@dataclass(frozen=True)
class LorentzPole:
    delta_eps: float
    resonance_frequency: float
    gamma: float
    allow_gain: bool = False

    def __post_init__(self):
        object.__setattr__(self, "allow_gain", bool(self.allow_gain))
        # A negative oscillator strength (delta_eps < 0) inverts the resonance and
        # models linear gain instead of absorption. This is only permitted behind an
        # explicit opt-in so an accidental sign flip still fails validation loudly.
        delta_eps = _coerce_real_scalar(self.delta_eps, name="delta_eps")
        if delta_eps < 0.0 and not self.allow_gain:
            raise ValueError(
                "LorentzPole delta_eps < 0 encodes a linear-gain (negative oscillator "
                "strength) resonance; pass allow_gain=True to opt in explicitly."
            )
        object.__setattr__(self, "delta_eps", delta_eps)
        object.__setattr__(self, "resonance_frequency", _coerce_frequency(self.resonance_frequency, name="resonance_frequency"))
        object.__setattr__(self, "gamma", _coerce_nonnegative(self.gamma, name="gamma"))
        if delta_eps < 0.0:
            warnings.warn(
                "LorentzPole with negative delta_eps models linear gain; gain media can "
                "violate the usual FDTD Courant/stability margins and may cause the "
                "time-domain solution to diverge. Verify run stability and keep the "
                "single-pass gain small.",
                stacklevel=2,
            )

    @property
    def resonance_angular_frequency(self) -> float:
        return 2.0 * np.pi * self.resonance_frequency

    @property
    def gamma_angular_frequency(self) -> float:
        return 2.0 * np.pi * self.gamma

    def susceptibility(self, angular_frequency: float) -> complex:
        angular_frequency = float(angular_frequency)
        omega_0 = self.resonance_angular_frequency
        gamma = self.gamma_angular_frequency
        return self.delta_eps * omega_0 * omega_0 / (
            omega_0 * omega_0 - angular_frequency * angular_frequency - 1j * gamma * angular_frequency
        )

    def susceptibility_at_freq(self, frequency: float) -> complex:
        return self.susceptibility(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


def _coerce_parameter_grid(value, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 3:
        raise ValueError(f"{name} must be a 3D tensor, got ndim={value.ndim}.")
    with torch.no_grad():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite everywhere.")
        if torch.any(value < 0):
            raise ValueError(f"{name} must be >= 0 everywhere.")
        if float(value.max()) <= 0.0:
            raise ValueError(f"{name} must contain at least one positive value.")
    return value


class CustomPole:
    """Marker base for spatially-varying (per-cell) dispersive pole descriptors.

    A custom pole carries a 3D parameter grid instead of a scalar oscillator
    strength. The grid spans the axis-aligned extent of the ``Box`` geometry of
    the ``Structure`` the material is attached to, using the same node-coverage
    convention as ``MaterialRegion.density``: box faces select grid nodes
    lower-inclusive / upper-exclusive, and the grid is trilinearly resampled
    when its shape differs from the covered node count. The rasterized grid is
    composed multiplicatively with the structure's soft geometry occupancy, so
    it blends and overlaps with other structures exactly like a scalar pole.

    Only the oscillator-strength parameter may vary per cell (``delta_eps`` for
    Debye/Lorentz, ``plasma_frequency`` for Drude); the rate parameters
    (``tau``, ``gamma``, ``resonance_frequency``) stay spatially uniform so the
    ADE recursion constants remain scalar and the existing native CUDA
    dispersive kernels are reused unchanged.
    """


@dataclass(frozen=True, eq=False)
class CustomDebyePole(CustomPole):
    """Debye pole with per-cell oscillator strength ``delta_eps(x) >= 0``."""

    delta_eps: torch.Tensor
    tau: float

    def __post_init__(self):
        object.__setattr__(self, "delta_eps", _coerce_parameter_grid(self.delta_eps, name="delta_eps"))
        object.__setattr__(self, "tau", _coerce_positive(self.tau, name="tau"))

    @property
    def peak_delta_eps(self) -> float:
        return float(self.delta_eps.detach().max())

    def reference_pole(self) -> DebyePole:
        """Scalar Debye pole at the peak oscillator strength."""
        return DebyePole(delta_eps=self.peak_delta_eps, tau=self.tau)

    def amplitude(self) -> torch.Tensor:
        """Per-cell strength normalized by the peak, in [0, 1]."""
        return self.delta_eps / self.peak_delta_eps

    def susceptibility(self, angular_frequency: float) -> torch.Tensor:
        return self.delta_eps / (1.0 - 1j * float(angular_frequency) * self.tau)

    def susceptibility_at_freq(self, frequency: float) -> torch.Tensor:
        return self.susceptibility(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


@dataclass(frozen=True, eq=False)
class CustomDrudePole(CustomPole):
    """Drude pole with per-cell plasma frequency ``plasma_frequency(x) >= 0`` [Hz]."""

    plasma_frequency: torch.Tensor
    gamma: float

    def __post_init__(self):
        object.__setattr__(
            self,
            "plasma_frequency",
            _coerce_parameter_grid(self.plasma_frequency, name="plasma_frequency"),
        )
        object.__setattr__(self, "gamma", _coerce_nonnegative(self.gamma, name="gamma"))

    @property
    def peak_plasma_frequency(self) -> float:
        return float(self.plasma_frequency.detach().max())

    def reference_pole(self) -> DrudePole:
        """Scalar Drude pole at the peak plasma frequency."""
        return DrudePole(plasma_frequency=self.peak_plasma_frequency, gamma=self.gamma)

    def amplitude(self) -> torch.Tensor:
        """Per-cell ``(f_p / f_p_peak)^2`` weight in [0, 1] (chi scales with f_p^2)."""
        normalized = self.plasma_frequency / self.peak_plasma_frequency
        return normalized * normalized

    def susceptibility(self, angular_frequency: float) -> torch.Tensor:
        angular_frequency = float(angular_frequency)
        if angular_frequency <= 0.0:
            raise ValueError("Drude susceptibility is undefined at zero frequency.")
        omega_p = 2.0 * np.pi * self.plasma_frequency
        gamma = 2.0 * np.pi * self.gamma
        return -(omega_p * omega_p) / (
            angular_frequency * angular_frequency + 1j * gamma * angular_frequency
        )

    def susceptibility_at_freq(self, frequency: float) -> torch.Tensor:
        return self.susceptibility(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


@dataclass(frozen=True, eq=False)
class CustomLorentzPole(CustomPole):
    """Lorentz pole with per-cell oscillator strength ``delta_eps(x) >= 0``."""

    delta_eps: torch.Tensor
    resonance_frequency: float
    gamma: float

    def __post_init__(self):
        object.__setattr__(self, "delta_eps", _coerce_parameter_grid(self.delta_eps, name="delta_eps"))
        object.__setattr__(
            self,
            "resonance_frequency",
            _coerce_frequency(self.resonance_frequency, name="resonance_frequency"),
        )
        object.__setattr__(self, "gamma", _coerce_nonnegative(self.gamma, name="gamma"))

    @property
    def peak_delta_eps(self) -> float:
        return float(self.delta_eps.detach().max())

    def reference_pole(self) -> LorentzPole:
        """Scalar Lorentz pole at the peak oscillator strength."""
        return LorentzPole(
            delta_eps=self.peak_delta_eps,
            resonance_frequency=self.resonance_frequency,
            gamma=self.gamma,
        )

    def amplitude(self) -> torch.Tensor:
        """Per-cell strength normalized by the peak, in [0, 1]."""
        return self.delta_eps / self.peak_delta_eps

    def susceptibility(self, angular_frequency: float) -> torch.Tensor:
        angular_frequency = float(angular_frequency)
        omega_0 = 2.0 * np.pi * self.resonance_frequency
        gamma = 2.0 * np.pi * self.gamma
        return self.delta_eps * omega_0 * omega_0 / (
            omega_0 * omega_0 - angular_frequency * angular_frequency - 1j * gamma * angular_frequency
        )

    def susceptibility_at_freq(self, frequency: float) -> torch.Tensor:
        return self.susceptibility(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


@dataclass(frozen=True)
class NonlinearSusceptibility:
    """Instantaneous nonlinear susceptibility descriptor.

    ``chi2`` [m/V] adds the second-order polarization ``P_i = eps0 * chi2 * E_i^2``
    per field component (diagonal scalar model), which drives second-harmonic
    generation and optical rectification. ``chi3`` [m^2/V^2] adds the isotropic
    third-order polarization ``P_i = eps0 * chi3 * |E|^2 * E_i`` and lowers to the
    same runtime channel as ``Material(kerr_chi3=...)`` (the two are strictly
    equivalent and additive). Both are scalars; tensorial susceptibilities are
    not supported. Composes with a ``Material`` through the ``nonlinearity=``
    argument.
    """

    chi2: float = 0.0
    chi3: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "chi2", _coerce_real_scalar(self.chi2, name="chi2"))
        object.__setattr__(self, "chi3", _coerce_real_scalar(self.chi3, name="chi3"))
        if float(self.chi2) == 0.0 and float(self.chi3) == 0.0:
            raise ValueError("NonlinearSusceptibility requires a nonzero chi2 or chi3.")


@dataclass(frozen=True)
class TwoPhotonAbsorption:
    """Two-photon absorption (TPA): intensity-dependent nonlinear loss.

    ``beta`` [m/W] is the TPA coefficient of ``dI/dz = -beta * I^2``. At runtime
    the loss enters the update as a field-dependent conductivity
    ``sigma_NL(x, t) = sigma_scale * |E(x, t)|^2`` folded into the semi-implicit
    lossy decay term, with ``sigma_scale = (4/3) * beta * (n0 * eps0 * c0)^2``.
    The 4/3 factor makes the cycle-averaged dissipation of a CW field
    (``<E^4> = 3/8 |E0|^4``) reproduce ``alpha = beta * I`` with
    ``I = 0.5 * n0 * eps0 * c0 * |E0|^2``. ``n0`` is the linear refractive index
    used in the intensity conversion and defaults to ``sqrt(eps_r)`` of the host
    material. Composes with a ``Material`` through the ``nonlinearity=`` argument.
    """

    beta: float
    n0: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "beta", _coerce_positive(self.beta, name="beta"))
        if self.n0 is not None:
            object.__setattr__(self, "n0", _coerce_positive(self.n0, name="n0"))

    def sigma_scale(self, base_index: float) -> float:
        """The ``sigma_NL / |E|^2`` coefficient [S*m/V^2] for a host index."""
        index = float(base_index) if self.n0 is None else float(self.n0)
        return (4.0 / 3.0) * self.beta * (index * VACUUM_PERMITTIVITY * _SPEED_OF_LIGHT) ** 2


_NONLINEAR_SPEC_TYPES = (NonlinearSusceptibility, TwoPhotonAbsorption)

# Harmonic modulation depth cap. The FDTD time step is sized against the static
# permittivity, and a modulated cell momentarily speeds the wave up by
# 1/sqrt(1 - amplitude); capping the amplitude at 0.5 keeps that inflation below
# sqrt(2), well inside the 2x headroom of the default Courant factor.
_MODULATION_AMPLITUDE_LIMIT = 0.5


def _coerce_modulation_amplitude(value):
    if torch.is_tensor(value):
        if value.ndim != 3:
            raise ValueError(f"ModulationSpec amplitude must be a 3D tensor, got ndim={value.ndim}.")
        with torch.no_grad():
            if not torch.isfinite(value).all():
                raise ValueError("ModulationSpec amplitude must be finite everywhere.")
            if torch.any(value < 0):
                raise ValueError("ModulationSpec amplitude must be >= 0 everywhere.")
            if float(value.max()) <= 0.0:
                raise ValueError("ModulationSpec amplitude must contain at least one positive value.")
            if float(value.max()) >= _MODULATION_AMPLITUDE_LIMIT:
                raise ValueError(
                    f"ModulationSpec amplitude must be < {_MODULATION_AMPLITUDE_LIMIT} everywhere to keep "
                    "the time-varying permittivity positive and Courant-stable."
                )
        return value
    amplitude = float(value)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("ModulationSpec amplitude must be > 0.")
    if amplitude >= _MODULATION_AMPLITUDE_LIMIT:
        raise ValueError(
            f"ModulationSpec amplitude must be < {_MODULATION_AMPLITUDE_LIMIT} to keep the "
            "time-varying permittivity positive and Courant-stable."
        )
    return amplitude


def _coerce_modulation_phase(value):
    if torch.is_tensor(value):
        if value.ndim != 3:
            raise ValueError(f"ModulationSpec phase must be a 3D tensor, got ndim={value.ndim}.")
        with torch.no_grad():
            if not torch.isfinite(value).all():
                raise ValueError("ModulationSpec phase must be finite everywhere.")
        return value
    return _coerce_real_scalar(value, name="phase")


@dataclass(frozen=True, eq=False)
class ModulationSpec:
    """Harmonic space-time permittivity modulation.

    Attached to a ``Material`` through ``Material(modulation=...)`` it makes the
    static permittivity time-varying:

    ``eps(x, t) = eps_static(x) * (1 + amplitude(x) * cos(2*pi*frequency*t + phase(x)))``

    ``frequency`` [Hz] is the modulation frequency (a single frequency per
    scene). ``amplitude`` is the dimensionless modulation depth, a scalar or a
    3D grid in ``[0, 0.5)`` spanning the Box extent of the structure the
    material is attached to (same node-coverage and trilinear-resampling
    convention as ``MaterialRegion.density``). ``phase`` [rad] is a scalar or a
    3D grid with the same spatial convention, enabling traveling-wave
    modulation profiles. The runtime applies the modulation inside a dedicated
    E-update kernel from precompiled per-cell quadrature fields, so no
    coefficient tensors are rebuilt per step.
    """

    frequency: float
    amplitude: float | torch.Tensor
    phase: float | torch.Tensor = 0.0

    def __post_init__(self):
        object.__setattr__(self, "frequency", _coerce_frequency(self.frequency, name="frequency"))
        object.__setattr__(self, "amplitude", _coerce_modulation_amplitude(self.amplitude))
        object.__setattr__(self, "phase", _coerce_modulation_phase(self.phase))

    @property
    def angular_frequency(self) -> float:
        return 2.0 * np.pi * self.frequency


@dataclass(frozen=True)
class DiagonalTensor3:
    xx: float
    yy: float
    zz: float

    def __post_init__(self):
        object.__setattr__(self, "xx", float(self.xx))
        object.__setattr__(self, "yy", float(self.yy))
        object.__setattr__(self, "zz", float(self.zz))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.xx, self.yy, self.zz)


@dataclass(frozen=True)
class Tensor3x3:
    rows: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

    def __init__(self, rows):
        object.__setattr__(self, "rows", _normalize_tensor_rows(rows, name="rows"))


def _shift_permittivity_real(value, susceptibility: complex):
    """Add the real part of a homogeneous susceptibility to a permittivity sample.

    A ``DiagonalTensor3`` stays a real diagonal tensor with each axis shifted by
    ``Re(chi)``; a scalar background shifts to ``eps_inf + Re(chi)``. The
    frequency-domain anisotropic permittivity sample is real because
    ``DiagonalTensor3`` carries no imaginary channel, and the only consumer of the
    sample (AutoGrid meshing) needs the real refractive index; the dispersive loss
    is recovered from the compiled scene material model when the imaginary part is
    required.
    """
    shift = float(susceptibility.real)
    if isinstance(value, DiagonalTensor3):
        return DiagonalTensor3(value.xx + shift, value.yy + shift, value.zz + shift)
    return float(value) + shift


@dataclass(frozen=True, init=False)
class Material(CoreMaterial):
    debye_poles: tuple[DebyePole, ...]
    drude_poles: tuple[DrudePole, ...]
    lorentz_poles: tuple[LorentzPole, ...]
    mu_debye_poles: tuple[DebyePole, ...]
    mu_drude_poles: tuple[DrudePole, ...]
    mu_lorentz_poles: tuple[LorentzPole, ...]
    epsilon_tensor: DiagonalTensor3 | Tensor3x3 | None
    mu_tensor: DiagonalTensor3 | Tensor3x3 | None
    sigma_e_tensor: DiagonalTensor3 | Tensor3x3 | None
    sigma_m: float
    orientation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None
    kerr_chi3: float | None
    nonlinearity: tuple
    modulation: ModulationSpec | None
    pec: bool

    def __init__(
        self,
        eps_r: float = 1.0,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
        *,
        sigma_m: float = 0.0,
        debye_poles=(),
        drude_poles=(),
        lorentz_poles=(),
        mu_debye_poles=(),
        mu_drude_poles=(),
        mu_lorentz_poles=(),
        epsilon_tensor: DiagonalTensor3 | Tensor3x3 | None = None,
        mu_tensor: DiagonalTensor3 | Tensor3x3 | None = None,
        sigma_e_tensor: DiagonalTensor3 | Tensor3x3 | None = None,
        orientation=None,
        kerr_chi3: float | None = None,
        nonlinearity=None,
        modulation: ModulationSpec | None = None,
        pec: bool = False,
    ):
        super().__init__(eps_r=eps_r, mu_r=mu_r, sigma_e=sigma_e, name=name)
        object.__setattr__(self, "pec", bool(pec))
        object.__setattr__(self, "debye_poles", _normalize_poles(debye_poles, (DebyePole, CustomDebyePole), name="debye_poles"))
        object.__setattr__(self, "drude_poles", _normalize_poles(drude_poles, (DrudePole, CustomDrudePole), name="drude_poles"))
        object.__setattr__(self, "lorentz_poles", _normalize_poles(lorentz_poles, (LorentzPole, CustomLorentzPole), name="lorentz_poles"))
        object.__setattr__(self, "mu_debye_poles", _normalize_poles(mu_debye_poles, (DebyePole, CustomDebyePole), name="mu_debye_poles"))
        object.__setattr__(self, "mu_drude_poles", _normalize_poles(mu_drude_poles, (DrudePole, CustomDrudePole), name="mu_drude_poles"))
        object.__setattr__(self, "mu_lorentz_poles", _normalize_poles(mu_lorentz_poles, (LorentzPole, CustomLorentzPole), name="mu_lorentz_poles"))
        object.__setattr__(self, "epsilon_tensor", epsilon_tensor)
        object.__setattr__(self, "mu_tensor", mu_tensor)
        object.__setattr__(self, "sigma_e_tensor", sigma_e_tensor)
        # Static magnetic conductivity [Ohm/m]: the magnetic dual of sigma_e. It
        # adds a magnetic conduction current sigma_m * H to Faraday's law, folded
        # semi-implicitly into the H-update Da/Db coefficients exactly as sigma_e
        # folds into the E-update Ca/Cb coefficients.
        object.__setattr__(self, "sigma_m", _coerce_nonnegative(sigma_m, name="sigma_m"))
        object.__setattr__(self, "orientation", None if orientation is None else _normalize_tensor_rows(orientation, name="orientation"))
        object.__setattr__(self, "kerr_chi3", None if kerr_chi3 is None else _coerce_real_scalar(kerr_chi3, name="kerr_chi3"))
        object.__setattr__(self, "nonlinearity", _normalize_poles(nonlinearity, _NONLINEAR_SPEC_TYPES, name="nonlinearity"))
        if modulation is not None and not isinstance(modulation, ModulationSpec):
            raise TypeError("Material.modulation must be a ModulationSpec instance.")
        object.__setattr__(self, "modulation", modulation)

        if self.orientation is not None:
            raise NotImplementedError(
                "Material.orientation (a separate crystal-frame rotation matrix) is not consumed: the "
                "tensor material path assembles per-edge Yee coefficients directly from the lab-frame "
                "permittivity/permeability/conductivity tensor, so a standalone rotation would need the "
                "tensor rotated into the lab frame before compilation. Pass the already-rotated tensor as "
                "Tensor3x3 rows (or an axis-aligned DiagonalTensor3) instead of an orientation matrix."
            )
        for tensor_name, tensor_value in (
            ("mu_tensor", self.mu_tensor),
            ("sigma_e_tensor", self.sigma_e_tensor),
        ):
            if isinstance(tensor_value, Tensor3x3):
                side = "H-field" if tensor_name == "mu_tensor" else "conduction-current"
                raise NotImplementedError(
                    f"Material.{tensor_name} accepts a DiagonalTensor3 (axis-aligned) tensor only. A full "
                    f"off-diagonal Tensor3x3 would require a coupled per-edge 3x3 inverse on the {side} "
                    "update, which only the electric permittivity update forms; supply an axis-aligned "
                    "DiagonalTensor3, or move the off-diagonal coupling to epsilon_tensor."
                )
        if isinstance(self.epsilon_tensor, Tensor3x3):
            _validate_symmetric_positive_definite(self.epsilon_tensor, name="epsilon_tensor")
        # A full (off-diagonal) Tensor3x3 permittivity combined with (diagonal)
        # electric conductivity is supported: the FDTD update folds the loss through
        # the exact semi-implicit tensor inverse B = dt * (eps_inf + dt/2 * diag(sigma))^-1
        # applied to both curl(H) and the conduction current sigma . E, which
        # diagonalizes in the crystal principal frame and reproduces the per-axis
        # complex index of a lossy anisotropic crystal (see build_full_anisotropy).

        # A full (off-diagonal) Tensor3x3 permittivity combined with electric poles is
        # supported: the poles enter isotropically (chi(omega) * I), so the frequency
        # response is eps_inf_tensor + chi(omega) * I and the FDTD update applies the
        # single instantaneous tensor inverse eps_inf^-1 to both curl(H) and the ADE
        # polarization current. This diagonalizes exactly in the crystal principal
        # frame, reproducing n_o(omega) and n_e(omega) for a rotated birefringent
        # dispersive crystal.
        if self.is_nonlinear and self.is_anisotropic:
            raise NotImplementedError(
                "A nonlinear Material cannot carry an anisotropic permittivity tensor: the instantaneous "
                "Kerr/chi2/TPA channel updates the permittivity as a per-component field-dependent scalar, "
                "whereas an anisotropic tensor applies a coupled per-edge tensor inverse. Composing them "
                "would need a per-step re-inversion of a field-amplitude-dependent 3x3 tensor, which the "
                "scalar nonlinear coefficient update does not form. Use an isotropic (scalar) permittivity "
                "with the nonlinearity, or drop the tensor."
            )

        if self.modulation is not None:
            # A time-modulated Material may now carry dispersive poles (electric or
            # magnetic) and the instantaneous nonlinear channels (Kerr / chi2 / TPA):
            # the modulation scales the eps_inf background while the ADE polarization
            # current and the field-dependent coefficients are folded through the same
            # per-step modulation factor (electro-optic-modulator physics). Two edges
            # remain physically out of reach for the single instantaneous-tensor path:
            if self.is_anisotropic:
                raise NotImplementedError(
                    "A time-modulated Material cannot carry anisotropic tensors: modulating the "
                    "crystal-frame permittivity tensor needs a per-step re-inversion of the coupled "
                    "3x3 tensor, which the diagonalized effective-permittivity update does not provide."
                )
            if float(self.sigma_e) != 0.0:
                raise NotImplementedError(
                    "A time-modulated Material cannot carry static electric conductivity: the "
                    "semi-implicit loss fold 0.5*sigma*dt/eps must see the modulated eps_inf*m(t), "
                    "which the static decay coefficient does not track. Model loss with a Drude/Debye "
                    "pole (dispersive path) instead."
                )

        if self.pec:
            if (
                self.is_dispersive
                or self.is_anisotropic
                or self.is_nonlinear
                or self.modulation is not None
                or float(self.eps_r) != 1.0
                or float(self.mu_r) != 1.0
                or float(self.sigma_e) != 0.0
                or float(self.sigma_m) != 0.0
            ):
                raise ValueError(
                    "A PEC Material must not carry dispersion, anisotropy, Kerr, modulation, or "
                    "non-default eps/mu/sigma; its permittivity is not a finite number."
                )

    @property
    def is_dispersive(self) -> bool:
        return bool(
            self.debye_poles
            or self.drude_poles
            or self.lorentz_poles
            or self.mu_debye_poles
            or self.mu_drude_poles
            or self.mu_lorentz_poles
        )

    @property
    def is_electric_dispersive(self) -> bool:
        return bool(self.debye_poles or self.drude_poles or self.lorentz_poles)

    @property
    def is_magnetic_dispersive(self) -> bool:
        return bool(self.mu_debye_poles or self.mu_drude_poles or self.mu_lorentz_poles)

    @property
    def has_custom_electric_poles(self) -> bool:
        return any(
            isinstance(pole, CustomPole)
            for pole in (*self.debye_poles, *self.drude_poles, *self.lorentz_poles)
        )

    @property
    def has_custom_magnetic_poles(self) -> bool:
        return any(
            isinstance(pole, CustomPole)
            for pole in (*self.mu_debye_poles, *self.mu_drude_poles, *self.mu_lorentz_poles)
        )

    @property
    def has_custom_poles(self) -> bool:
        return self.has_custom_electric_poles or self.has_custom_magnetic_poles

    @property
    def is_anisotropic(self) -> bool:
        return any(value is not None for value in (self.epsilon_tensor, self.mu_tensor, self.sigma_e_tensor))

    @property
    def has_full_epsilon_tensor(self) -> bool:
        """Whether the permittivity is a full (potentially off-diagonal) 3x3 tensor."""
        return isinstance(self.epsilon_tensor, Tensor3x3)

    @property
    def nonlinear_chi2(self) -> float:
        """Total instantaneous second-order susceptibility [m/V]."""
        return float(
            sum(spec.chi2 for spec in self.nonlinearity if isinstance(spec, NonlinearSusceptibility))
        )

    @property
    def nonlinear_chi3(self) -> float:
        """Total instantaneous third-order (Kerr) susceptibility [m^2/V^2]."""
        return float(self.kerr_chi3 or 0.0) + float(
            sum(spec.chi3 for spec in self.nonlinearity if isinstance(spec, NonlinearSusceptibility))
        )

    @property
    def tpa_sigma_scale(self) -> float:
        """Total two-photon-absorption ``sigma_NL / |E|^2`` coefficient [S*m/V^2]."""
        base_index = float(np.sqrt(float(self.eps_r)))
        return float(
            sum(
                spec.sigma_scale(base_index)
                for spec in self.nonlinearity
                if isinstance(spec, TwoPhotonAbsorption)
            )
        )

    @property
    def is_nonlinear(self) -> bool:
        return self.nonlinear_chi2 != 0.0 or self.nonlinear_chi3 != 0.0 or self.tpa_sigma_scale != 0.0

    @property
    def is_modulated(self) -> bool:
        return self.modulation is not None

    @property
    def is_pec(self) -> bool:
        return self.pec

    @classmethod
    def pec(cls, name: str | None = None) -> "Material":
        """Construct a perfect-electric-conductor marker material."""
        return cls(pec=True, name=name)

    def capabilities(self) -> MaterialCapabilities:
        base = super().capabilities()
        return MaterialCapabilities(
            conductive=base.conductive or self.sigma_e_tensor is not None,
            magnetic=(
                base.magnetic
                or self.mu_tensor is not None
                or self.is_magnetic_dispersive
                or float(self.sigma_m) != 0.0
            ),
            anisotropic=self.is_anisotropic,
            dispersive=self.is_dispersive,
        )

    def evaluate_static(self) -> StaticMaterialSample:
        return StaticMaterialSample(
            eps_r=self.epsilon_tensor if self.epsilon_tensor is not None else self.eps_r,
            mu_r=self.mu_tensor if self.mu_tensor is not None else self.mu_r,
            sigma_e=self.sigma_e_tensor if self.sigma_e_tensor is not None else self.sigma_e,
        )

    def _electric_pole_susceptibility(self, frequency: float) -> complex:
        """Summed homogeneous electric-pole susceptibility ``chi_e(f)``.

        Defined only for scalar (non-custom) poles; spatially-varying custom poles
        are rejected before this is reached.
        """
        total = 0.0 + 0.0j
        for pole in (*self.debye_poles, *self.drude_poles, *self.lorentz_poles):
            total += pole.susceptibility_at_freq(frequency)
        return complex(total)

    def _magnetic_pole_susceptibility(self, frequency: float) -> complex:
        """Summed homogeneous magnetic-pole susceptibility ``chi_m(f)``."""
        total = 0.0 + 0.0j
        for pole in (*self.mu_debye_poles, *self.mu_drude_poles, *self.mu_lorentz_poles):
            total += pole.susceptibility_at_freq(frequency)
        return complex(total)

    def evaluate_at_frequency(self, frequency: float) -> FrequencyMaterialSample:
        if self.is_nonlinear:
            raise NotImplementedError("Nonlinear Material frequency evaluation is not defined without a field amplitude.")
        if self.is_dispersive and (self.has_full_epsilon_tensor or self.has_custom_poles):
            raise NotImplementedError(
                "Frequency evaluation of a dispersive Material is only defined for an axis-aligned "
                "DiagonalTensor3 permittivity with homogeneous poles: a fully anisotropic Tensor3x3 "
                "permittivity needs a coupled tensor ADE, and spatially-varying custom poles have no "
                "homogeneous sample. Evaluate the compiled scene material model "
                "(Scene.compile_relative_materials) instead."
            )
        if self.is_anisotropic:
            # Diagonal (axis-aligned) anisotropy composes with homogeneous
            # dispersion by shifting each per-axis background permittivity by the
            # isotropic pole susceptibility; the sample stays a real DiagonalTensor3.
            static = self.evaluate_static()
            eps_r = static.eps_r
            mu_r = static.mu_r
            if self.is_electric_dispersive:
                eps_r = _shift_permittivity_real(eps_r, self._electric_pole_susceptibility(frequency))
            if self.is_magnetic_dispersive:
                mu_r = _shift_permittivity_real(mu_r, self._magnetic_pole_susceptibility(frequency))
            return FrequencyMaterialSample(
                eps_r=eps_r,
                mu_r=mu_r,
                sigma_e=static.sigma_e,
            )
        return FrequencyMaterialSample(
            eps_r=self.relative_permittivity(frequency),
            mu_r=self.relative_permeability(frequency),
            sigma_e=self.sigma_e,
        )

    def relative_permittivity(self, frequency: float) -> complex:
        if self.is_anisotropic:
            raise NotImplementedError("relative_permittivity() currently supports isotropic Material only.")
        if self.is_nonlinear:
            raise NotImplementedError("relative_permittivity() is not defined for nonlinear Material without a field amplitude.")
        if self.has_custom_electric_poles:
            raise NotImplementedError(
                "relative_permittivity() is not defined for spatially-varying custom dispersive poles; "
                "evaluate the compiled scene material model (Scene.compile_relative_materials) instead."
            )
        frequency = _coerce_frequency(frequency, name="frequency")
        angular_frequency = 2.0 * np.pi * frequency
        epsilon = complex(self.eps_r, self.sigma_e / (angular_frequency * VACUUM_PERMITTIVITY))
        for pole in self.debye_poles:
            epsilon += pole.susceptibility_at_freq(frequency)
        for pole in self.drude_poles:
            epsilon += pole.susceptibility_at_freq(frequency)
        for pole in self.lorentz_poles:
            epsilon += pole.susceptibility_at_freq(frequency)
        return epsilon

    def relative_permeability(self, frequency: float) -> complex:
        if self.is_anisotropic:
            raise NotImplementedError("relative_permeability() currently supports isotropic Material only.")
        if self.is_nonlinear:
            raise NotImplementedError("relative_permeability() is not defined for nonlinear Material without a field amplitude.")
        if self.has_custom_magnetic_poles:
            raise NotImplementedError(
                "relative_permeability() is not defined for spatially-varying custom dispersive poles; "
                "evaluate the compiled scene material model (Scene.compile_relative_materials) instead."
            )
        frequency = _coerce_frequency(frequency, name="frequency")
        permeability = complex(self.mu_r, 0.0)
        for pole in self.mu_debye_poles:
            permeability += pole.susceptibility_at_freq(frequency)
        for pole in self.mu_drude_poles:
            permeability += pole.susceptibility_at_freq(frequency)
        for pole in self.mu_lorentz_poles:
            permeability += pole.susceptibility_at_freq(frequency)
        return permeability

    @classmethod
    def debye(
        cls,
        *,
        eps_inf: float = 1.0,
        delta_eps: float,
        tau: float,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
    ) -> "Material":
        return cls(
            eps_r=eps_inf,
            mu_r=mu_r,
            sigma_e=sigma_e,
            name=name,
            debye_poles=(DebyePole(delta_eps=delta_eps, tau=tau),),
        )

    @classmethod
    def drude(
        cls,
        *,
        eps_inf: float = 1.0,
        plasma_frequency: float,
        gamma: float,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
    ) -> "Material":
        return cls(
            eps_r=eps_inf,
            mu_r=mu_r,
            sigma_e=sigma_e,
            name=name,
            drude_poles=(DrudePole(plasma_frequency=plasma_frequency, gamma=gamma),),
        )

    @classmethod
    def lorentz(
        cls,
        *,
        eps_inf: float = 1.0,
        delta_eps: float,
        resonance_frequency: float,
        gamma: float,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
        allow_gain: bool = False,
    ) -> "Material":
        return cls(
            eps_r=eps_inf,
            mu_r=mu_r,
            sigma_e=sigma_e,
            name=name,
            lorentz_poles=(
                LorentzPole(
                    delta_eps=delta_eps,
                    resonance_frequency=resonance_frequency,
                    gamma=gamma,
                    allow_gain=allow_gain,
                ),
            ),
        )

    @classmethod
    def sellmeier(
        cls,
        *,
        b_coefficients,
        c_coefficients,
        eps_inf: float = 1.0,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
    ) -> "Material":
        """Construct a lossless Sellmeier dielectric.

        The Sellmeier dispersion ``n(lambda)^2 = eps_inf + sum_i B_i * lambda^2 /
        (lambda^2 - C_i)`` is a sum of lossless Lorentz oscillators. Each ``(B_i, C_i)``
        term maps exactly to a zero-damping ``LorentzPole`` with ``delta_eps = B_i`` and
        resonance ``omega0_i = 2*pi*c / sqrt(C_i)`` (i.e. ``resonance_frequency =
        c / sqrt(C_i)``), because ``omega0_i^2 / (omega0_i^2 - omega^2) =
        lambda^2 / (lambda^2 - C_i)``.

        ``c_coefficients`` carry units of length squared and MUST be expressed in SI
        ``meters^2``, consistent with the rest of the simulation. Coefficient tables
        quoted in ``micron^2`` (for example the BK7 Schott coefficients) must be scaled
        by ``1e-12`` before being passed in. Each ``B_i`` must be non-negative and each
        ``C_i`` strictly positive.
        """
        b_values = tuple(float(b) for b in b_coefficients)
        c_values = tuple(float(c) for c in c_coefficients)
        if len(b_values) != len(c_values):
            raise ValueError("Sellmeier b_coefficients and c_coefficients must have equal length.")
        if not b_values:
            raise ValueError("Sellmeier material requires at least one (B, C) coefficient pair.")
        poles = []
        for b_value, c_value in zip(b_values, c_values):
            if c_value <= 0.0:
                raise ValueError("Sellmeier c_coefficients must be > 0 (squared resonance wavelength in meters^2).")
            resonance_frequency = _SPEED_OF_LIGHT / np.sqrt(c_value)
            poles.append(
                LorentzPole(
                    delta_eps=b_value,
                    resonance_frequency=resonance_frequency,
                    gamma=0.0,
                )
            )
        return cls(
            eps_r=eps_inf,
            mu_r=mu_r,
            sigma_e=sigma_e,
            name=name,
            lorentz_poles=tuple(poles),
        )


def _gyromagnetic_cross_matrix(bias_unit_vector: torch.Tensor) -> torch.Tensor:
    bx, by, bz = bias_unit_vector[0], bias_unit_vector[1], bias_unit_vector[2]
    zero = torch.zeros_like(bx)
    return torch.stack(
        [
            torch.stack([zero, -bz, by]),
            torch.stack([bz, zero, -bx]),
            torch.stack([-by, bx, zero]),
        ]
    )


def gyromagnetic_polder_tensor(
    omega,
    *,
    omega_0,
    omega_m,
    gilbert_damping,
    mu_infinity,
    bias_unit_vector,
    dtype=torch.complex128,
) -> torch.Tensor:
    """Lab-frame 3x3 complex Polder permeability tensor.

    Frozen convention (``exp(-i*omega*t)``, contract section 2.3/2.4):

    ``W = omega_0 - i*alpha*omega``, ``D = W^2 - omega^2``,
    ``mu = mu_infinity + omega_m*W/D``, ``kappa = omega_m*omega/D``, and

    ``mu_r = mu*(I - b b^T) + mu_infinity*(b b^T) + i*kappa*[b]_x``.

    Torch-native and differentiable in ``omega`` and any keyword argument that is
    a leaf tensor (used for material-parameter autograd).
    """
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    def _c(value):
        if isinstance(value, torch.Tensor):
            return value.to(dtype)
        return torch.as_tensor(value, dtype=real_dtype).to(dtype)

    w = _c(omega)
    w0 = _c(omega_0)
    wm = _c(omega_m)
    alpha = _c(gilbert_damping)
    mu_inf = _c(mu_infinity)
    i = torch.tensor(1j, dtype=dtype)
    W = w0 - i * alpha * w
    D = W * W - w * w
    mu = mu_inf + wm * W / D
    kappa = wm * w / D

    if isinstance(bias_unit_vector, torch.Tensor):
        b = bias_unit_vector.to(real_dtype)
    else:
        b = torch.as_tensor(bias_unit_vector, dtype=real_dtype)
    b = (b / torch.linalg.vector_norm(b)).to(dtype)
    eye = torch.eye(3, dtype=dtype)
    bbT = torch.outer(b, b)
    return mu * (eye - bbT) + mu_inf * bbT + i * kappa * _gyromagnetic_cross_matrix(b)


@dataclass(frozen=True, init=False)
class GyromagneticFerrite(Material):
    """DC-biased ferrite with a gyromagnetic (Polder) permeability tensor.

    The first non-reciprocal material in the framework. Its frequency-domain
    permeability is the tensor

    ``mu_r(omega) = mu*(I - b b^T) + mu_infinity*(b b^T) + i*kappa*[b]_x``

    with the scalar Polder components in
    :func:`gyromagnetic_polder_tensor`. The gyrotropy is carried by a local
    linearized Landau-Lifshitz-Gilbert magnetization ADE state (see the FDTD
    runtime slices), never by widening ``mu_tensor`` -- the off-diagonal
    ``mu_tensor`` guard stays in force. The full derivation, sign/unit
    conventions, discretization, and acceptance budget are frozen in
    ``docs/reference/ferrite-physics-contract.md``.

    All parameters are SI. Datasheet CGS quantities enter only through
    :meth:`from_cgs`, which records the conversion in :attr:`cgs_conversion`.

    Parameters
    ----------
    eps_r:
        Relative permittivity of the ferrite host (real, ``> 0``).
    saturation_magnetization:
        ``Ms`` [A/m], ``> 0``.
    bias_field:
        Static internal bias ``H0`` [A/m], a 3-vector (``!= 0``). The user
        supplies the internal field; there is no magnetostatic solve.
    gilbert_damping:
        Gilbert damping ``alpha`` (dimensionless, ``>= 0``).
    gyromagnetic_ratio:
        ``gamma`` [rad/(s*T)], ``> 0`` (default the electron value).
    mu_infinity:
        High-frequency background permeability (``> 0``, default ``1.0``).
    sigma_e:
        Electric conductivity [S/m] (``>= 0``).
    """

    saturation_magnetization: float
    bias_field: tuple[float, float, float]
    gilbert_damping: float
    gyromagnetic_ratio: float
    mu_infinity: float
    cgs_conversion: tuple

    def __init__(
        self,
        *,
        eps_r: float = 1.0,
        saturation_magnetization: float,
        bias_field,
        gilbert_damping: float = 0.0,
        gyromagnetic_ratio: float = _DEFAULT_GYROMAGNETIC_RATIO,
        mu_infinity: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
        cgs_conversion: tuple = (),
    ):
        saturation = _coerce_positive(saturation_magnetization, name="saturation_magnetization")
        damping = _coerce_nonnegative(gilbert_damping, name="gilbert_damping")
        gamma = _coerce_positive(gyromagnetic_ratio, name="gyromagnetic_ratio")
        mu_inf = _coerce_positive(mu_infinity, name="mu_infinity")
        _coerce_positive(eps_r, name="eps_r")

        bias = tuple(_coerce_real_scalar(component, name="bias_field") for component in bias_field)
        if len(bias) != 3:
            raise ValueError("bias_field must be a 3-vector [A/m].")
        bias_norm = float(np.sqrt(sum(component * component for component in bias)))
        if bias_norm <= 0.0:
            raise ValueError(
                "bias_field must be non-zero: a zero-bias ferrite has no gyrotropy and the local "
                "precession frame is degenerate."
            )

        super().__init__(eps_r=eps_r, mu_r=mu_inf, sigma_e=sigma_e, name=name)
        object.__setattr__(self, "saturation_magnetization", saturation)
        object.__setattr__(self, "bias_field", bias)
        object.__setattr__(self, "gilbert_damping", damping)
        object.__setattr__(self, "gyromagnetic_ratio", gamma)
        object.__setattr__(self, "mu_infinity", mu_inf)
        object.__setattr__(self, "cgs_conversion", tuple(cgs_conversion))

    # --- Derived physical quantities -----------------------------------------

    @property
    def bias_magnitude(self) -> float:
        """Static bias magnitude ``|H0|`` [A/m]."""
        return float(np.sqrt(sum(component * component for component in self.bias_field)))

    @property
    def bias_unit_vector(self) -> tuple[float, float, float]:
        magnitude = self.bias_magnitude
        return tuple(component / magnitude for component in self.bias_field)

    @property
    def omega_0(self) -> float:
        """Larmor precession frequency ``omega_0 = gamma*mu_0*|H0|`` [rad/s]."""
        return self.gyromagnetic_ratio * _VACUUM_PERMEABILITY * self.bias_magnitude

    @property
    def omega_m(self) -> float:
        """Magnetization frequency ``omega_m = gamma*mu_0*Ms`` [rad/s]."""
        return self.gyromagnetic_ratio * _VACUUM_PERMEABILITY * self.saturation_magnetization

    @property
    def resonance_frequency(self) -> float:
        """Gyromagnetic resonance frequency ``omega_0/(2*pi)`` [Hz]."""
        return self.omega_0 / (2.0 * np.pi)

    # --- Torch-native Polder tensor accessors --------------------------------

    def polder_tensor(self, angular_frequency, *, dtype=torch.complex128) -> torch.Tensor:
        """Lab-frame complex 3x3 permeability tensor at ``omega`` (angular frequency).

        Torch-native and differentiable in ``angular_frequency``.
        """
        return gyromagnetic_polder_tensor(
            angular_frequency,
            omega_0=self.omega_0,
            omega_m=self.omega_m,
            gilbert_damping=self.gilbert_damping,
            mu_infinity=self.mu_infinity,
            bias_unit_vector=self.bias_unit_vector,
            dtype=dtype,
        )

    def permeability_tensor_at_freq(self, frequency, *, dtype=torch.complex128) -> torch.Tensor:
        """Lab-frame complex 3x3 permeability tensor at ordinary ``frequency`` [Hz]."""
        if isinstance(frequency, torch.Tensor):
            omega = 2.0 * np.pi * frequency
        else:
            omega = 2.0 * np.pi * _coerce_frequency(frequency, name="frequency")
        return self.polder_tensor(omega, dtype=dtype)

    def scalar_polder_components(self, frequency):
        """Analytic scalar ``(mu, kappa)`` (complex) at ordinary ``frequency`` [Hz].

        Frame-invariant: valid for any bias orientation, not only ``b = z_hat``.
        ``mu`` is recovered from the isotropic transverse block via
        ``trace(mu_r) = 2*mu + mu_infinity`` (since ``tr(I - b b^T) = 2``,
        ``tr(b b^T) = 1``, ``tr([b]_x) = 0``), and ``kappa`` from the gyrotropic
        (antisymmetric) part contracted against the bias cross matrix ``[b]_x``,
        for which ``sum_ij (mu_r)_ij ([b]_x)_ij = 2*i*kappa`` (the symmetric part
        contracts to zero and ``sum_ij ([b]_x)_ij^2 = 2`` for a unit bias).
        """
        tensor = self.permeability_tensor_at_freq(frequency)
        b = torch.as_tensor(self.bias_unit_vector, dtype=torch.float64)
        cross = _gyromagnetic_cross_matrix(b).to(tensor.dtype)
        mu = complex((torch.trace(tensor) - self.mu_infinity) / 2.0)
        kappa = complex((tensor * cross).sum() / 2.0j)
        return mu, kappa

    # --- Material-family overrides -------------------------------------------

    def capabilities(self) -> MaterialCapabilities:
        return MaterialCapabilities(
            conductive=float(self.sigma_e) != 0.0,
            magnetic=True,
            anisotropic=True,
            dispersive=True,
        )

    def relative_permeability(self, frequency: float) -> complex:
        raise NotImplementedError(
            "relative_permeability() is not defined for a GyromagneticFerrite: its permeability is a "
            "non-reciprocal complex 3x3 Polder tensor, not a scalar. Use permeability_tensor_at_freq() "
            "(or polder_tensor() for the angular-frequency form)."
        )

    def evaluate_at_frequency(self, frequency: float) -> FrequencyMaterialSample:
        raise NotImplementedError(
            "evaluate_at_frequency() is not defined for a GyromagneticFerrite: a FrequencyMaterialSample "
            "carries a scalar/diagonal mu, which cannot represent the off-diagonal gyromagnetic Polder "
            "tensor. Use permeability_tensor_at_freq() for the full 3x3 permeability."
        )

    @classmethod
    def from_cgs(
        cls,
        *,
        saturation_4piMs_gauss: float,
        bias_Oe: float,
        bias_direction=(0.0, 0.0, 1.0),
        eps_r: float = 1.0,
        gilbert_damping: float = 0.0,
        gyromagnetic_ratio: float = _DEFAULT_GYROMAGNETIC_RATIO,
        mu_infinity: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
    ) -> "GyromagneticFerrite":
        """Construct from CGS datasheet quantities, recording the SI conversion.

        ``saturation_4piMs_gauss`` is ``4*pi*Ms`` in Gauss and ``bias_Oe`` is the
        internal bias in Oersted. Both convert as ``x[A/m] = x_cgs * 1e-4 / mu_0``
        (i.e. ``1000/(4*pi) = 79.57747`` A/m per Gauss / per Oersted). The exact
        factors are stored in :attr:`cgs_conversion`.
        """
        four_pi_ms = _coerce_positive(saturation_4piMs_gauss, name="saturation_4piMs_gauss")
        bias_oe = _coerce_positive(bias_Oe, name="bias_Oe")
        saturation = four_pi_ms * _OERSTED_TO_A_PER_M
        bias_magnitude = bias_oe * _OERSTED_TO_A_PER_M
        direction = tuple(_coerce_real_scalar(component, name="bias_direction") for component in bias_direction)
        if len(direction) != 3:
            raise ValueError("bias_direction must be a 3-vector.")
        direction_norm = float(np.sqrt(sum(component * component for component in direction)))
        if direction_norm <= 0.0:
            raise ValueError("bias_direction must be non-zero.")
        bias_field = tuple(component / direction_norm * bias_magnitude for component in direction)
        conversion = (
            ("saturation_4piMs_gauss", four_pi_ms),
            ("bias_Oe", bias_oe),
            ("saturation_magnetization_A_per_m", saturation),
            ("bias_magnitude_A_per_m", bias_magnitude),
            ("cgs_to_A_per_m", _OERSTED_TO_A_PER_M),
        )
        return cls(
            eps_r=eps_r,
            saturation_magnetization=saturation,
            bias_field=bias_field,
            gilbert_damping=gilbert_damping,
            gyromagnetic_ratio=gyromagnetic_ratio,
            mu_infinity=mu_infinity,
            sigma_e=sigma_e,
            name=name,
            cgs_conversion=conversion,
        )

    @classmethod
    def from_resonance(
        cls,
        *,
        resonance_frequency: float,
        saturation_magnetization: float,
        linewidth: float = 0.0,
        bias_direction=(0.0, 0.0, 1.0),
        eps_r: float = 1.0,
        gyromagnetic_ratio: float = _DEFAULT_GYROMAGNETIC_RATIO,
        mu_infinity: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
    ) -> "GyromagneticFerrite":
        """Construct from the gyromagnetic resonance frequency and FMR linewidth.

        The bias magnitude is back-computed from ``omega_0 = 2*pi*resonance_frequency
        = gamma*mu_0*|H0|``. ``linewidth`` is the full-width (FWHM) resonance
        linewidth in Hz; it maps to Gilbert damping ``alpha = linewidth /
        (2*resonance_frequency)`` (``Delta_omega = 2*alpha*omega_0``).
        """
        f_res = _coerce_frequency(resonance_frequency, name="resonance_frequency")
        gamma = _coerce_positive(gyromagnetic_ratio, name="gyromagnetic_ratio")
        width = _coerce_nonnegative(linewidth, name="linewidth")
        omega_0 = 2.0 * np.pi * f_res
        bias_magnitude = omega_0 / (gamma * _VACUUM_PERMEABILITY)
        damping = width / (2.0 * f_res)
        direction = tuple(_coerce_real_scalar(component, name="bias_direction") for component in bias_direction)
        if len(direction) != 3:
            raise ValueError("bias_direction must be a 3-vector.")
        direction_norm = float(np.sqrt(sum(component * component for component in direction)))
        if direction_norm <= 0.0:
            raise ValueError("bias_direction must be non-zero.")
        bias_field = tuple(component / direction_norm * bias_magnitude for component in direction)
        return cls(
            eps_r=eps_r,
            saturation_magnetization=saturation_magnetization,
            bias_field=bias_field,
            gilbert_damping=damping,
            gyromagnetic_ratio=gamma,
            mu_infinity=mu_infinity,
            sigma_e=sigma_e,
            name=name,
        )


class Medium2D(Material):
    """Zero-thickness conductive sheet (2D material).

    ``Medium2D`` models an infinitesimally thin conductive layer through its
    sheet conductivity ``sigma_s`` [S] (siemens; a surface quantity, not the
    volumetric S/m). The sheet carries the surface current
    ``J_s = sigma_s * E_t`` in the two tangential field components and leaves
    the normal component untouched.

    A ``Medium2D`` must be attached to a ``Structure`` whose geometry is an
    axis-aligned ``Box`` with exactly one zero-size axis (the sheet normal).
    The compiler snaps the sheet to the nearest node plane along the normal
    axis and lowers ``sigma_s`` to the effective volumetric conductivity
    ``sigma_s / dcell`` on that single Yee layer, where ``dcell`` is the local
    dual-cell width at the plane: integrating the discrete Ampere law across
    one dual cell shows a sheet current distributed over that cell is exactly
    equivalent to this volumetric conductivity. Sheet contributions are
    additive with bulk conductivity and with other overlapping sheets.
    """

    def __init__(self, *, sigma_s: float = 0.0, name: str | None = None):
        super().__init__(eps_r=1.0, mu_r=1.0, sigma_e=0.0, name=name)
        object.__setattr__(self, "sigma_s", _coerce_nonnegative(sigma_s, name="sigma_s"))

    @property
    def is_medium2d(self) -> bool:
        return True

    def sheet_pole_terms(self) -> tuple[tuple[float, float], ...]:
        """Dispersive surface-conductivity terms ``(weight, rate)``.

        Each term contributes ``weight / (rate - i*omega)`` [S] to the sheet
        conductivity (``weight`` in S/s, ``rate`` in 1/s), i.e. a Drude-like
        relaxation of the surface current ``dJ_s/dt + rate * J_s = weight * E_t``.
        The static base class carries none; frequency-dependent sheets such as
        ``Graphene`` override this.
        """
        return ()

    def sheet_lorentz_terms(self) -> tuple[tuple[float, float, float], ...]:
        """Resonant surface-conductivity terms ``(strength, omega_0, gamma)``.

        Each term contributes
        ``-i*omega * strength * omega_0^2 / (omega_0^2 - omega^2 - i*gamma*omega)``
        [S] to the sheet conductivity (``strength`` in S*s, ``omega_0`` and
        ``gamma`` in rad/s), i.e. the sheet lowering of one volumetric Lorentz
        pole. A Drude term (``sheet_pole_terms``) can only reproduce an inductive
        surface reactance (``Im(sigma) > 0``); a resonant absorption edge whose
        below-edge reactance is capacitive (``Im(sigma) < 0``), such as the
        graphene interband transition, needs a Lorentz term. The static base
        class carries none.
        """
        return ()

    def sheet_conductivity(self, angular_frequency: float) -> complex:
        """Complex sheet conductivity ``sigma_s(omega)`` [S] (e^{-i*omega*t} convention)."""
        omega = float(angular_frequency)
        sigma = complex(self.sigma_s)
        for weight, rate in self.sheet_pole_terms():
            sigma += weight / complex(rate, -omega)
        for strength, omega_0, gamma in self.sheet_lorentz_terms():
            sigma += -1j * omega * strength * omega_0 * omega_0 / (
                omega_0 * omega_0 - omega * omega - 1j * gamma * omega
            )
        return sigma

    def sheet_conductivity_at_freq(self, frequency: float) -> complex:
        return self.sheet_conductivity(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


_ELEMENTARY_CHARGE = 1.602176634e-19  # [C]
_REDUCED_PLANCK = 1.054571817e-34  # [J*s]
_BOLTZMANN = 1.380649e-23  # [J/K]

_GRAPHENE_INTERBAND_POLES = 4


def _kubo_interband_sigma(angular_frequency: float, mu_c_ev: float, tau: float, temperature: float) -> complex:
    """Analytic Kubo interband sheet conductivity ``sigma_inter(omega)`` [S].

    T > 0 principal-value form in the e^{-i*omega*t} convention:

    ``sigma_inter = i*e^2*Omega/(pi*hbar^2) * INT_0^inf [f(-xi) - f(xi)]
                    / (Omega^2 - (2*xi/hbar)^2) dxi``

    where ``Omega = omega + i/tau`` is the collision-broadened complex
    frequency and ``f(xi) = 1/(exp((xi - mu_c)/(kB*T)) + 1)`` is the Fermi-Dirac
    occupation. The finite scattering rate ``1/tau`` moves the integrand pole
    off the real ``xi`` axis, so the "principal value" is a convergent integral.

    The integrand has a sharp resonant peak of width ``hbar/(2*tau)`` at
    ``xi = hbar*omega/2`` and a Fermi step of width ``kB*T`` at ``xi = mu_c``,
    and its ``1/xi^2`` tail decays too slowly to truncate. It is therefore split
    at ``xi = split`` (chosen above both features, where ``f(-xi) - f(xi) = 1``
    to machine precision): the near part is integrated adaptively with the peak
    and step passed as subdivision hints, and the far part is added in closed
    form, ``INT_split^inf dxi / (Omega^2 - 4*xi^2/hbar^2) = (hbar/(4*Omega)) *
    ln((2*split - hbar*Omega)/(2*split + hbar*Omega))``. This converges to the
    exact T = 0 log form as ``T -> 0``.
    """
    from scipy.integrate import quad

    mu = mu_c_ev * _ELEMENTARY_CHARGE
    thermal_energy = _BOLTZMANN * temperature
    omega_complex = angular_frequency + 1j / tau
    prefactor = 1j * _ELEMENTARY_CHARGE * _ELEMENTARY_CHARGE * omega_complex / (
        np.pi * _REDUCED_PLANCK * _REDUCED_PLANCK
    )

    def fermi(energy: float) -> float:
        return 1.0 / (np.exp(np.clip((energy - mu) / thermal_energy, -500.0, 500.0)) + 1.0)

    def integrand(energy: float) -> complex:
        numerator = fermi(-energy) - fermi(energy)
        denom = omega_complex * omega_complex - 4.0 * (energy / _REDUCED_PLANCK) ** 2
        return numerator / denom

    split = mu + 40.0 * thermal_energy + 6.0 * _REDUCED_PLANCK * abs(angular_frequency)
    peak = _REDUCED_PLANCK * angular_frequency / 2.0
    peak_width = _REDUCED_PLANCK / (2.0 * tau)
    hints = (peak - 2.0 * peak_width, peak, peak + 2.0 * peak_width,
             mu - 4.0 * thermal_energy, mu, mu + 4.0 * thermal_energy)
    points = sorted(p for p in hints if 0.0 < p < split) or None
    real_part, _ = quad(lambda e: integrand(e).real, 0.0, split, limit=2000, points=points)
    imag_part, _ = quad(lambda e: integrand(e).imag, 0.0, split, limit=2000, points=points)
    tail = (_REDUCED_PLANCK / (4.0 * omega_complex)) * np.log(
        (2.0 * split - _REDUCED_PLANCK * omega_complex) / (2.0 * split + _REDUCED_PLANCK * omega_complex)
    )
    return prefactor * (real_part + 1j * imag_part + tail)


def _fit_graphene_interband_lorentz(
    mu_c_ev: float, tau: float, temperature: float, intraband_weight: float
) -> tuple[tuple[float, float, float], ...]:
    """Fit the Kubo interband sheet conductivity with Lorentz sheet terms.

    Returns ``(strength, omega_0, gamma)`` terms (see ``sheet_lorentz_terms``)
    that reproduce the interband conductivity below the band edge
    ``hbar*omega = 2*|mu_c|``. Below the edge the interband response is a
    capacitive reactance (``Im(sigma) < 0``) that a Drude pole cannot represent,
    but a Lorentz oscillator seated near the edge can. Strengths are constrained
    non-negative so every term lowers to a passive (``delta_eps >= 0``) pole and
    the time-domain ADE stays stable.

    The fit is a deterministic non-negative least-squares seed on a fixed
    ``(omega_0, gamma)`` grid, refined by a bounded Levenberg-Marquardt step,
    weighted by ``1/|sigma_total|`` so the relative error of the *total*
    (intraband + interband) conductivity is minimized across the band.
    """
    from scipy.optimize import least_squares, nnls

    omega_gap = 2.0 * mu_c_ev * _ELEMENTARY_CHARGE / _REDUCED_PLANCK
    n_samples = 140
    omega = np.linspace(0.02 * omega_gap, 0.99 * omega_gap, n_samples)
    intra = intraband_weight / (1.0 / tau - 1j * omega)
    inter = np.array([_kubo_interband_sigma(w, mu_c_ev, tau, temperature) for w in omega])
    total = intra + inter
    fit_weight = 1.0 / np.maximum(np.abs(total), 0.1 * np.abs(total).max())

    def basis(w, omega_0, gamma):
        return -1j * w * omega_0 * omega_0 / (omega_0 * omega_0 - w * w - 1j * gamma * w)

    omega_0_grid = np.linspace(0.5, 2.0, 25) * omega_gap
    gamma_grid = np.array([0.05, 0.1, 0.2, 0.35, 0.6, 1.0]) * omega_gap
    candidates = [(o, g) for o in omega_0_grid for g in gamma_grid]
    design = np.zeros((2 * n_samples, len(candidates)))
    scales = np.zeros(len(candidates))
    for k, (o, g) in enumerate(candidates):
        column = basis(omega, o, g) * fit_weight
        stacked = np.concatenate([column.real, column.imag])
        scales[k] = np.linalg.norm(stacked)
        design[:, k] = stacked / scales[k]
    target = np.concatenate([inter.real * fit_weight, inter.imag * fit_weight])
    seed_solution, _ = nnls(design, target, maxiter=20000)
    seed_strengths = seed_solution / scales
    seeds = sorted(
        (
            (seed_strengths[k], candidates[k][0], candidates[k][1])
            for k in range(len(candidates))
            if seed_strengths[k] > 0.0
        ),
        key=lambda term: -term[0],
    )[:_GRAPHENE_INTERBAND_POLES]
    while len(seeds) < _GRAPHENE_INTERBAND_POLES:
        seeds.append((1e-24, 1.3 * omega_gap, 0.3 * omega_gap))

    initial, lower, upper = [], [], []
    for strength, omega_0, gamma in seeds:
        initial += [np.log10(max(strength, 1e-30)), omega_0 / omega_gap, np.log10(max(gamma / omega_gap, 0.03))]
        lower += [-33.0, 0.3, -1.6]
        upper += [-13.0, 2.5, 0.6]

    def residual(params):
        model = np.zeros(n_samples, dtype=complex)
        for k in range(_GRAPHENE_INTERBAND_POLES):
            model += 10 ** params[3 * k] * basis(
                omega, params[3 * k + 1] * omega_gap, 10 ** params[3 * k + 2] * omega_gap
            )
        residual_complex = (model - inter) * fit_weight
        return np.concatenate([residual_complex.real, residual_complex.imag])

    solution = least_squares(
        residual,
        np.array(initial),
        bounds=(np.array(lower), np.array(upper)),
        method="trf",
        max_nfev=8000,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )
    terms = tuple(
        (
            float(10 ** solution.x[3 * k]),
            float(solution.x[3 * k + 1] * omega_gap),
            float(10 ** solution.x[3 * k + 2] * omega_gap),
        )
        for k in range(_GRAPHENE_INTERBAND_POLES)
    )

    fitted = intra + sum(strength * basis(omega, omega_0, gamma) for strength, omega_0, gamma in terms)
    l2_error = float(np.linalg.norm(fitted - total) / np.linalg.norm(total))
    if l2_error > 0.03:
        raise ValueError(
            f"Graphene interband Lorentz fit reached only {l2_error * 100:.1f}% band error "
            f"(> 3%) for chemical_potential={mu_c_ev} eV, scattering_time={tau} s, "
            f"temperature={temperature} K; this operating point lies outside the validated "
            f"interband range (the fit assumes a well-defined edge with |mu_c| >~ kB*T)."
        )
    return terms


class Graphene(Medium2D):
    """Graphene sheet with the Kubo surface conductivity.

    The intraband (Drude-like) term of the Kubo conductivity is

    ``sigma_intra(omega) = A / (Gamma - i*omega)``  [S]

    with the Drude weight (e^{-i*omega*t} convention)

    ``A = (e^2 * kB * T / (pi * hbar^2)) * (mu_c/(kB*T) + 2*ln(1 + exp(-mu_c/(kB*T))))``

    and the scattering rate ``Gamma = 1 / scattering_time``. ``chemical_potential``
    is given in eV, ``temperature`` in K, and ``scattering_time`` (the intraband
    relaxation time tau) in seconds. The DC sheet conductivity is
    ``A * scattering_time``.

    At runtime the sheet lowers to a surface-current ADE on the snapped Yee
    layer: the relaxation ``dJ_s/dt + Gamma*J_s = A*E_t`` distributed over the
    local dual-cell width ``dcell`` is exactly a Drude pole with
    ``eps0*omega_p^2 = A/dcell``, so the existing native Drude current kernels
    advance the sheet current with no new per-step work beyond one pole.

    With ``include_interband=True`` the interband term of the Kubo conductivity
    (the T > 0 principal-value form, ``_kubo_interband_sigma``) is added as a
    small set of Lorentz sheet terms fitted at construction. The interband
    transition is a reactive absorption edge at ``hbar*omega = 2*|mu_c|`` whose
    below-edge conductivity is capacitive (``Im(sigma) < 0``); this does not
    lower onto a Drude pole (always inductive) but does lower onto Lorentz
    poles, which the existing native Lorentz current kernels advance. The fitted
    conductivity matches the analytic Kubo model to within a few percent across
    the optical band up to the edge. The intraband-only model (the default)
    stays a single Drude pole and dominates for ``|mu_c| >> hbar*omega``.
    """

    def __init__(
        self,
        *,
        chemical_potential: float,
        scattering_time: float,
        temperature: float = 300.0,
        include_interband: bool = False,
        name: str | None = None,
    ):
        super().__init__(sigma_s=0.0, name=name)
        object.__setattr__(
            self,
            "chemical_potential",
            _coerce_nonnegative(chemical_potential, name="chemical_potential"),
        )
        object.__setattr__(
            self, "scattering_time", _coerce_positive(scattering_time, name="scattering_time")
        )
        object.__setattr__(self, "temperature", _coerce_positive(temperature, name="temperature"))
        object.__setattr__(self, "include_interband", bool(include_interband))
        interband_terms: tuple[tuple[float, float, float], ...] = ()
        if include_interband:
            interband_terms = _fit_graphene_interband_lorentz(
                float(self.chemical_potential),
                float(self.scattering_time),
                float(self.temperature),
                self.intraband_drude_weight,
            )
        object.__setattr__(self, "_interband_lorentz_terms", interband_terms)

    @property
    def intraband_drude_weight(self) -> float:
        """The Kubo intraband Drude weight ``A`` [S/s]."""
        thermal_energy = _BOLTZMANN * self.temperature
        mu_over_kt = self.chemical_potential * _ELEMENTARY_CHARGE / thermal_energy
        bracket = mu_over_kt + 2.0 * float(np.log1p(np.exp(-mu_over_kt)))
        prefactor = (
            _ELEMENTARY_CHARGE * _ELEMENTARY_CHARGE * thermal_energy
            / (np.pi * _REDUCED_PLANCK * _REDUCED_PLANCK)
        )
        return float(prefactor * bracket)

    @property
    def scattering_rate(self) -> float:
        """Intraband relaxation rate ``Gamma = 1/tau`` [1/s]."""
        return 1.0 / self.scattering_time

    @property
    def characteristic_frequency(self) -> float:
        """Material rate [Hz] folded into the FDTD auto-dt bound.

        The intraband relaxation rate and, when interband is enabled, the
        highest interband Lorentz resonance and linewidth all set the fastest
        material timescale the time step must resolve.
        """
        characteristic = self.scattering_rate / (2.0 * np.pi)
        for _strength, omega_0, gamma in self._interband_lorentz_terms:
            characteristic = max(characteristic, omega_0 / (2.0 * np.pi), gamma / (2.0 * np.pi))
        return characteristic

    def sheet_pole_terms(self) -> tuple[tuple[float, float], ...]:
        return ((self.intraband_drude_weight, self.scattering_rate),)

    def sheet_lorentz_terms(self) -> tuple[tuple[float, float, float], ...]:
        return self._interband_lorentz_terms


_VACUUM_PERMEABILITY = 4.0e-7 * np.pi  # [H/m]


class LossyMetalMedium(Material):
    """Good-conductor metal intended for a surface-impedance boundary condition (SIBC).

    ``conductivity`` [S/m] is the bulk metal conductivity. The intended runtime
    treatment replaces the resolved metal interior with the first-order Leontovich
    boundary condition ``E_t = Z_s(omega) * (n x H)`` on the metal surface, where
    (with the ``e^{-i*omega*t}`` convention)

    ``Z_s(omega) = (1 - i) / (conductivity * skin_depth(omega))
                 = (1 - i) * sqrt(omega * mu0 / (2 * conductivity))``

    so the skin-depth interior never needs to be meshed.

    The v1 runtime is scoped to normal incidence on a single axis-aligned planar
    face: the metal must be a ``Box`` slab that spans the full transverse
    cross-section and sits flush against one domain boundary. The surface
    impedance is realized as a narrowband series R-L evaluated at the operating
    frequency (``Z_s(omega0) = R + i*omega0*L_s`` with ``L_s = R/omega0``), so it
    reproduces the exact Leontovich value at the source frequency; the metal
    interior is masked and the two tangential E faces are updated each step from
    the vacuum-side tangential H (see ``compiler/materials.py`` and
    ``fdtd/runtime/materials.py``). Laterally finite blocks, oblique/curved
    surfaces, mid-domain slabs, and Bloch runs raise ``NotImplementedError`` with
    a physical reason; resolve those metals volumetrically with
    ``Material(sigma_e=...)`` (or ``Material.pec()`` for a lossless shortcut). The
    analytic helpers ``surface_impedance`` / ``surface_impedance_at_freq`` /
    ``skin_depth`` are exposed for validation and design work.
    """

    def __init__(self, *, conductivity: float, name: str | None = None):
        super().__init__(eps_r=1.0, mu_r=1.0, sigma_e=0.0, name=name)
        object.__setattr__(
            self, "conductivity", _coerce_positive(conductivity, name="conductivity")
        )

    @property
    def is_lossy_metal(self) -> bool:
        return True

    def skin_depth(self, frequency: float) -> float:
        """Skin depth ``sqrt(2 / (omega * mu0 * sigma))`` [m] at ``frequency`` [Hz]."""
        omega = 2.0 * np.pi * _coerce_frequency(frequency, name="frequency")
        return float(np.sqrt(2.0 / (omega * _VACUUM_PERMEABILITY * self.conductivity)))

    def surface_impedance(self, angular_frequency: float) -> complex:
        """Leontovich surface impedance ``Z_s(omega)`` [ohm] (e^{-i*omega*t} convention)."""
        omega = float(angular_frequency)
        if omega <= 0.0:
            raise ValueError("surface_impedance requires angular_frequency > 0.")
        magnitude = np.sqrt(omega * _VACUUM_PERMEABILITY / (2.0 * self.conductivity))
        return complex(magnitude, -magnitude)

    def surface_impedance_at_freq(self, frequency: float) -> complex:
        return self.surface_impedance(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


class PerturbationMedium(Material):
    """A base material whose permittivity is shifted by an external perturbation field.

    ``eps(x) = eps_base + eps_sensitivity * perturbation(x)`` where
    ``perturbation`` is a 3D torch tensor (for example a temperature or
    carrier-density change from an external multiphysics solve). The
    perturbation grid maps onto the ``Box`` extent of the structure the
    material is attached to with the same lower-inclusive / upper-exclusive
    node coverage and trilinear resampling as ``MaterialRegion.density``, is
    composed multiplicatively with the structure's soft geometry occupancy,
    and is applied at compile time before ADE templating, so dispersive base
    materials see the shifted ``eps_inf``.

    The map is PyTorch-native and differentiable: a ``perturbation`` tensor
    with ``requires_grad=True`` is discovered as a trainable material input by
    the gradient bridges, so simulation outputs can be differentiated with
    respect to the perturbation field.
    """

    def __init__(
        self,
        base: Material,
        *,
        perturbation: torch.Tensor,
        eps_sensitivity: float = 1.0,
        name: str | None = None,
    ):
        if not isinstance(base, CoreMaterial):
            raise TypeError("PerturbationMedium base must be a Material.")
        if isinstance(base, GyromagneticFerrite):
            raise NotImplementedError(
                "PerturbationMedium cannot wrap a GyromagneticFerrite: it perturbs a scalar permittivity "
                "background, but a ferrite carries a non-reciprocal gyromagnetic permeability in a local "
                "magnetization state that this scalar-eps perturbation would silently discard. Perturb the "
                "ferrite parameters directly instead."
            )
        if bool(getattr(base, "is_pec", False)):
            raise ValueError("PerturbationMedium cannot wrap a PEC base material.")
        if not torch.is_tensor(perturbation):
            raise TypeError("PerturbationMedium perturbation must be a torch.Tensor.")
        if perturbation.ndim != 3:
            raise ValueError(
                f"PerturbationMedium perturbation must be a 3D tensor, got ndim={perturbation.ndim}."
            )
        with torch.no_grad():
            if not torch.isfinite(perturbation).all():
                raise ValueError("PerturbationMedium perturbation must be finite everywhere.")
        sensitivity = _coerce_real_scalar(eps_sensitivity, name="eps_sensitivity")

        super().__init__(
            eps_r=base.eps_r,
            mu_r=base.mu_r,
            sigma_e=base.sigma_e,
            name=name if name is not None else base.name,
            debye_poles=getattr(base, "debye_poles", ()),
            drude_poles=getattr(base, "drude_poles", ()),
            lorentz_poles=getattr(base, "lorentz_poles", ()),
            mu_debye_poles=getattr(base, "mu_debye_poles", ()),
            mu_drude_poles=getattr(base, "mu_drude_poles", ()),
            mu_lorentz_poles=getattr(base, "mu_lorentz_poles", ()),
            epsilon_tensor=getattr(base, "epsilon_tensor", None),
            mu_tensor=getattr(base, "mu_tensor", None),
            sigma_e_tensor=getattr(base, "sigma_e_tensor", None),
            kerr_chi3=getattr(base, "kerr_chi3", None),
            nonlinearity=getattr(base, "nonlinearity", ()),
            modulation=getattr(base, "modulation", None),
        )
        object.__setattr__(self, "perturbation", perturbation)
        object.__setattr__(self, "eps_sensitivity", sensitivity)

        if isinstance(self.epsilon_tensor, Tensor3x3):
            raise NotImplementedError(
                "PerturbationMedium applies its scalar sensitivity field as an isotropic delta_eps to a "
                "scalar or axis-aligned (DiagonalTensor3) base permittivity; a full off-diagonal Tensor3x3 "
                "base has no single principal-axis eps to perturb, so the perturbation direction in the "
                "coupled tensor is ambiguous. Use a scalar or DiagonalTensor3 base permittivity."
            )
        base_eps = self.epsilon_tensor if self.epsilon_tensor is not None else self.eps_r
        eps_floor = min(base_eps.as_tuple()) if isinstance(base_eps, DiagonalTensor3) else float(base_eps)
        with torch.no_grad():
            delta_floor = float((sensitivity * perturbation).min())
        if eps_floor + delta_floor <= 0.0:
            raise ValueError(
                "PerturbationMedium would produce a non-positive permittivity: base eps "
                f"{eps_floor} + minimum perturbation contribution {delta_floor} <= 0."
            )

    def evaluate_at_frequency(self, frequency: float) -> FrequencyMaterialSample:
        raise NotImplementedError(
            "PerturbationMedium frequency evaluation is spatially varying; evaluate the "
            "compiled scene material model (Scene.compile_relative_materials) instead."
        )

    def relative_permittivity(self, frequency: float) -> complex:
        raise NotImplementedError(
            "relative_permittivity() is not defined for PerturbationMedium; evaluate the "
            "compiled scene material model (Scene.compile_relative_materials) instead."
        )
