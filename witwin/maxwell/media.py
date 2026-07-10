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

        if self.orientation is not None:
            raise NotImplementedError("Material.orientation is not supported yet; use axis-aligned DiagonalTensor3 media only.")
        for tensor_name, tensor_value in (
            ("epsilon_tensor", self.epsilon_tensor),
            ("mu_tensor", self.mu_tensor),
            ("sigma_e_tensor", self.sigma_e_tensor),
        ):
            if isinstance(tensor_value, Tensor3x3):
                raise NotImplementedError(
                    f"Material.{tensor_name} currently supports DiagonalTensor3 only; Tensor3x3 is not implemented yet."
                )

        if self.is_electric_dispersive and self.epsilon_tensor is not None:
            raise NotImplementedError("Anisotropic electric dispersion is not supported yet.")
        if self.is_magnetic_dispersive and self.mu_tensor is not None:
            raise NotImplementedError("Magnetic dispersion with mu_tensor is not supported yet.")
        if self.is_nonlinear and self.is_dispersive:
            raise NotImplementedError("Kerr media cannot be combined with dispersive poles in v1.")
        if self.is_nonlinear and self.is_anisotropic:
            raise NotImplementedError("Kerr media cannot be combined with anisotropic tensors in v1.")

        if self.pec:
            if (
                self.is_dispersive
                or self.is_anisotropic
                or self.is_nonlinear
                or float(self.eps_r) != 1.0
                or float(self.mu_r) != 1.0
                or float(self.sigma_e) != 0.0
            ):
                raise ValueError(
                    "A PEC Material must not carry dispersion, anisotropy, Kerr, or non-default "
                    "eps/mu/sigma; its permittivity is not a finite number."
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
    def is_nonlinear(self) -> bool:
        return self.kerr_chi3 is not None and float(self.kerr_chi3) != 0.0

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

    def evaluate_at_frequency(self, frequency: float) -> FrequencyMaterialSample:
        if self.is_nonlinear:
            raise NotImplementedError("Nonlinear Material frequency evaluation is not defined without a field amplitude.")
        if self.is_anisotropic and self.is_dispersive:
            raise NotImplementedError("Anisotropic dispersive Material frequency evaluation is not implemented yet.")
        if self.is_anisotropic:
            static = self.evaluate_static()
            return FrequencyMaterialSample(
                eps_r=static.eps_r,
                mu_r=static.mu_r,
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
