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
        object.__setattr__(self, "orientation", None if orientation is None else _normalize_tensor_rows(orientation, name="orientation"))
        object.__setattr__(self, "kerr_chi3", None if kerr_chi3 is None else _coerce_real_scalar(kerr_chi3, name="kerr_chi3"))
        object.__setattr__(self, "nonlinearity", _normalize_poles(nonlinearity, _NONLINEAR_SPEC_TYPES, name="nonlinearity"))
        if modulation is not None and not isinstance(modulation, ModulationSpec):
            raise TypeError("Material.modulation must be a ModulationSpec instance.")
        object.__setattr__(self, "modulation", modulation)

        if self.orientation is not None:
            raise NotImplementedError("Material.orientation is not supported yet; use axis-aligned DiagonalTensor3 media only.")
        for tensor_name, tensor_value in (
            ("mu_tensor", self.mu_tensor),
            ("sigma_e_tensor", self.sigma_e_tensor),
        ):
            if isinstance(tensor_value, Tensor3x3):
                raise NotImplementedError(
                    f"Material.{tensor_name} currently supports DiagonalTensor3 only; Tensor3x3 is not implemented yet."
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
            raise NotImplementedError("A nonlinear Material cannot carry anisotropic tensors in v1.")

        if self.modulation is not None:
            if self.is_dispersive:
                raise NotImplementedError("A time-modulated Material cannot carry dispersive poles in v1.")
            if self.is_anisotropic:
                raise NotImplementedError("A time-modulated Material cannot carry anisotropic tensors in v1.")
            if self.is_nonlinear:
                raise NotImplementedError("A time-modulated Material cannot carry nonlinear channels in v1.")
            if float(self.sigma_e) != 0.0:
                raise NotImplementedError("A time-modulated Material cannot carry electric conductivity in v1.")

        if self.pec:
            if (
                self.is_dispersive
                or self.is_anisotropic
                or self.is_nonlinear
                or self.modulation is not None
                or float(self.eps_r) != 1.0
                or float(self.mu_r) != 1.0
                or float(self.sigma_e) != 0.0
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
            magnetic=base.magnetic or self.mu_tensor is not None or self.is_magnetic_dispersive,
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
        epsilon = complex(self.eps_r, -self.sigma_e / (angular_frequency * VACUUM_PERMITTIVITY))
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

    def sheet_conductivity(self, angular_frequency: float) -> complex:
        """Complex sheet conductivity ``sigma_s(omega)`` [S] (e^{-i*omega*t} convention)."""
        sigma = complex(self.sigma_s)
        for weight, rate in self.sheet_pole_terms():
            sigma += weight / complex(rate, -float(angular_frequency))
        return sigma

    def sheet_conductivity_at_freq(self, frequency: float) -> complex:
        return self.sheet_conductivity(2.0 * np.pi * _coerce_frequency(frequency, name="frequency"))


_ELEMENTARY_CHARGE = 1.602176634e-19  # [C]
_REDUCED_PLANCK = 1.054571817e-34  # [J*s]
_BOLTZMANN = 1.380649e-23  # [J/K]


class Graphene(Medium2D):
    """Graphene sheet with the Kubo intraband surface conductivity.

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

    The interband term of the Kubo conductivity is not implemented; it does
    not lower onto the real-coefficient pole machinery without a Pade/pole fit
    and ``include_interband=True`` raises ``NotImplementedError`` explicitly.
    Around and below the THz range the intraband term dominates for typical
    ``|mu_c| >> hbar*omega`` operating points.
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
        if include_interband:
            raise NotImplementedError(
                "Graphene interband conductivity is not implemented yet; only the Kubo "
                "intraband (Drude-like) term is supported."
            )
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
        """Material rate [Hz] folded into the FDTD auto-dt bound."""
        return self.scattering_rate / (2.0 * np.pi)

    def sheet_pole_terms(self) -> tuple[tuple[float, float], ...]:
        return ((self.intraband_drude_weight, self.scattering_rate),)


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

    The SIBC runtime is **not implemented yet**: ``Z_s ~ sqrt(-i*omega)`` is not
    rational in ``omega``, so a time-domain implementation needs a vector-fitted
    pole expansion of ``Z_s`` with per-face recursive-convolution state and a
    dedicated boundary-side update kernel on the tangential E faces adjacent to
    the metal surface. Compiling a Scene that contains a ``LossyMetalMedium``
    structure raises ``NotImplementedError``; resolve the metal volumetrically
    with ``Material(sigma_e=...)`` (or use ``Material.pec()`` for a lossless
    shortcut) in the meantime. The analytic helpers ``surface_impedance`` /
    ``surface_impedance_at_freq`` / ``skin_depth`` are exposed for validation
    and design work.
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
                "PerturbationMedium does not support a fully anisotropic (Tensor3x3) base material yet."
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
