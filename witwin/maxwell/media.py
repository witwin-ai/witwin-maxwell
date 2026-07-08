from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from witwin.core import (
    FrequencyMaterialSample,
    Material as CoreMaterial,
    MaterialCapabilities,
    StaticMaterialSample,
)
from witwin.core.material import VACUUM_PERMITTIVITY


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


def _normalize_poles(value, pole_type, *, name: str):
    if value is None:
        return ()
    if isinstance(value, pole_type):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        normalized = tuple(value)
        if any(not isinstance(item, pole_type) for item in normalized):
            raise TypeError(f"{name} entries must be {pole_type.__name__} instances.")
        return normalized
    raise TypeError(f"{name} must be a {pole_type.__name__} or an iterable of them.")


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

    def __post_init__(self):
        object.__setattr__(self, "delta_eps", _coerce_nonnegative(self.delta_eps, name="delta_eps"))
        object.__setattr__(self, "resonance_frequency", _coerce_frequency(self.resonance_frequency, name="resonance_frequency"))
        object.__setattr__(self, "gamma", _coerce_nonnegative(self.gamma, name="gamma"))

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
        object.__setattr__(self, "debye_poles", _normalize_poles(debye_poles, DebyePole, name="debye_poles"))
        object.__setattr__(self, "drude_poles", _normalize_poles(drude_poles, DrudePole, name="drude_poles"))
        object.__setattr__(self, "lorentz_poles", _normalize_poles(lorentz_poles, LorentzPole, name="lorentz_poles"))
        object.__setattr__(self, "mu_debye_poles", _normalize_poles(mu_debye_poles, DebyePole, name="mu_debye_poles"))
        object.__setattr__(self, "mu_drude_poles", _normalize_poles(mu_drude_poles, DrudePole, name="mu_drude_poles"))
        object.__setattr__(self, "mu_lorentz_poles", _normalize_poles(mu_lorentz_poles, LorentzPole, name="mu_lorentz_poles"))
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
                ),
            ),
        )
