"""Conductor surface-roughness corrections for good-conductor surface impedances.

A rough conductor surface dissipates more power than a smooth one once the skin depth
``delta`` becomes comparable to the roughness scale. Each model here is a frozen,
opt-in descriptor that returns a frequency-dependent multiplicative correction ``K``
applied to the smooth Leontovich impedance,

``Z_rough(omega) = K(omega) * Z_smooth(omega)``,
``Z_smooth(omega) = (1 - i) * sqrt(omega * mu0 / (2 * sigma))``

(repository ``e^{-i omega t}`` convention). ``K`` depends on frequency only through
the skin depth ``delta = sqrt(2 / (omega * mu0 * sigma))``, so every model is evaluated
from a skin-depth tensor.

Complex (causal) form
---------------------
The classic roughness formulas are *power-loss ratios*: they scale the surface
resistance only. Scaling the whole complex impedance by that real ratio breaks the
Kramers-Kronig relation between resistance and internal reactance. The Hammerstad and
Huray models therefore use their causal complex forms (Dmitriev-Zdorov et al. for the
modified Hammerstad model, Bracken for the Huray model): ``K`` is complex and analytic
in the upper half ``omega`` plane, and applied to the complex smooth impedance it
reproduces the classic loss ratio *exactly* in the resistance,

``Re(Z_rough) / Re(Z_smooth) = Re(K * (1 - i)) = Re(K) + Im(K) = classic loss ratio``,

while also adding the matching causal internal reactance. :meth:`SurfaceRoughness.loss_factor`
returns this resistance ratio. The Groisse model has no published causal partner; it is
kept as a real factor applied to the whole impedance (resistance and reactance scale
together) and is documented as non-causal.

All evaluations are PyTorch-native (they run on the device and dtype of the supplied
skin-depth tensor, in complex128 / float64 precision).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch


def _positive_finite(value, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return number


def _skin_depth_tensor(skin_depth) -> torch.Tensor:
    tensor = (
        skin_depth if torch.is_tensor(skin_depth)
        else torch.as_tensor(skin_depth, dtype=torch.float64)
    )
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError("skin_depth must be real.")
    tensor = tensor.to(dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(tensor))) or not bool(torch.all(tensor > 0.0)):
        raise ValueError("skin_depth must be finite and strictly positive.")
    return tensor


class SurfaceRoughness(ABC):
    """Contract for a multiplicative conductor-roughness correction ``K(delta)``.

    :meth:`impedance_factor` returns the complex factor that multiplies the smooth
    good-conductor surface impedance; :meth:`loss_factor` returns the resulting surface
    resistance (power-loss) ratio ``Re(K * Z_smooth) / Re(Z_smooth)``. A physical
    roughness model only adds loss, so ``loss_factor >= 1`` and the rough impedance stays
    passive (``Re(Z_rough) > 0``). ``causal`` states whether ``K`` is analytic in the
    upper half ``omega`` plane (a broadband rational fit requires it).
    """

    causal = True

    @abstractmethod
    def _factor(self, skin_depth: torch.Tensor) -> torch.Tensor:
        """Complex correction for a validated float64 skin-depth tensor."""

    def impedance_factor(self, skin_depth) -> torch.Tensor:
        """Complex correction ``K`` (complex128) with the shape of ``skin_depth`` [m]."""

        return self._factor(_skin_depth_tensor(skin_depth))

    def loss_factor(self, skin_depth) -> torch.Tensor:
        """Surface-resistance ratio ``Re(K * (1 - i))`` (float64) at ``skin_depth`` [m]."""

        factor = self.impedance_factor(skin_depth)
        ratio = factor.real + factor.imag
        if not bool(torch.all(ratio > 0.0)):
            raise ValueError("Surface roughness produced a non-passive (Re(Z) <= 0) impedance.")
        return ratio


@dataclass(frozen=True)
class HammerstadRoughness(SurfaceRoughness):
    """Modified Hammerstad roughness in its causal complex form.

    ``rms_height`` [m] is the RMS peak-to-valley roughness height ``Rq`` and
    ``roughness_factor`` (``> 1``) the high-frequency saturation of the loss ratio
    (``2`` is the classic Hammerstad equation). The resistance ratio is exactly the
    modified Hammerstad expression

    ``1 + (RF - 1) * (2/pi) * atan(1.4 * (Rq / delta)^2)``,

    obtained from the causal factor ``K = 1 + (RF - 1)/pi * [ln(1 + 2 sqrt(p)/(1 + p))
    + 2 atan(sqrt(p))]`` with the normalized Laplace variable ``p = -1.4 i (Rq/delta)^2``
    (``e^{-i omega t}``; principal square root). ``K -> 1`` as ``Rq/delta -> 0`` and
    ``K -> RF`` as ``Rq/delta -> inf``.
    """

    rms_height: float
    roughness_factor: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, "rms_height", _positive_finite(self.rms_height, name="rms_height"))
        factor = float(self.roughness_factor)
        if not math.isfinite(factor) or factor <= 1.0:
            raise ValueError("roughness_factor must be finite and > 1.")
        object.__setattr__(self, "roughness_factor", factor)

    def _factor(self, skin_depth: torch.Tensor) -> torch.Tensor:
        ratio_sq = (self.rms_height / skin_depth) ** 2
        laplace = torch.complex(torch.zeros_like(ratio_sq), -1.4 * ratio_sq)
        root = torch.sqrt(laplace)
        response = torch.log(1.0 + 2.0 * root / (1.0 + laplace)) + 2.0 * torch.atan(root)
        return 1.0 + (self.roughness_factor - 1.0) / math.pi * response


@dataclass(frozen=True)
class HurayRoughness(SurfaceRoughness):
    """Huray "snowball" roughness in its causal complex form (one sphere population).

    The rough surface is a matte base of relative area ``relative_area`` (``A_matte /
    A_flat``) carrying spheres of radius ``sphere_radius`` [m] at ``sphere_density``
    spheres per unit flat area [1/m^2]. With the sphere-to-flat surface ratio
    ``f = sphere_density * 4 pi r^2`` the resistance ratio is exactly the Huray
    expression

    ``A_matte/A_flat + (3/2) f / (1 + delta/r + delta^2 / (2 r^2))``,

    obtained from the causal factor ``K = A_matte/A_flat + (3/2) f / (1 + (1 + i) delta /
    (2 r))`` (``e^{-i omega t}``). ``K -> relative_area`` as ``r/delta -> 0`` and
    ``K -> relative_area + 1.5 f`` as ``r/delta -> inf``.
    """

    sphere_radius: float
    sphere_density: float
    relative_area: float = 1.0

    def __post_init__(self):
        object.__setattr__(
            self, "sphere_radius", _positive_finite(self.sphere_radius, name="sphere_radius")
        )
        object.__setattr__(
            self, "sphere_density", _positive_finite(self.sphere_density, name="sphere_density")
        )
        object.__setattr__(
            self, "relative_area", _positive_finite(self.relative_area, name="relative_area")
        )

    @classmethod
    def from_surface_ratio(
        cls, sphere_radius: float, surface_ratio: float, relative_area: float = 1.0
    ) -> "HurayRoughness":
        """Build from the sphere-to-flat surface ratio ``f = N 4 pi r^2 / A_flat``."""

        radius = _positive_finite(sphere_radius, name="sphere_radius")
        ratio = _positive_finite(surface_ratio, name="surface_ratio")
        return cls(
            sphere_radius=radius,
            sphere_density=ratio / (4.0 * math.pi * radius * radius),
            relative_area=relative_area,
        )

    @classmethod
    def cannonball(cls, sphere_radius: float) -> "HurayRoughness":
        """Cannonball-Huray stacking: 14 spheres per ``(6 r)^2`` tile on a flat base.

        The surface ratio is ``f = 14 pi / 9`` and the saturated loss ratio is
        ``1 + 7 pi / 3``.
        """

        radius = _positive_finite(sphere_radius, name="sphere_radius")
        return cls(sphere_radius=radius, sphere_density=14.0 / (36.0 * radius * radius))

    @property
    def surface_ratio(self) -> float:
        """Total sphere surface area over flat area, ``N 4 pi r^2 / A_flat``."""

        return self.sphere_density * 4.0 * math.pi * self.sphere_radius ** 2

    def _factor(self, skin_depth: torch.Tensor) -> torch.Tensor:
        half = 0.5 * skin_depth / self.sphere_radius
        denominator = torch.complex(1.0 + half, half)
        return self.relative_area + 1.5 * self.surface_ratio / denominator


@dataclass(frozen=True)
class GroisseRoughness(SurfaceRoughness):
    """Groisse roughness ``K = 1 + exp(-(delta / (2 Rq))^1.6)`` (real, non-causal).

    ``rms_height`` [m] is the RMS roughness height ``Rq``. The factor is real and is
    applied to the whole complex impedance, so resistance and internal reactance scale
    together; it has no causal reactance partner. The narrowband surface path uses only
    the resistance, where the factor is exact; a broadband rational fit can only
    not reproduce it and is rejected by
    ``LossyMetalMedium.rational_surface_impedance``.
    """

    rms_height: float
    causal = False

    def __post_init__(self):
        object.__setattr__(self, "rms_height", _positive_finite(self.rms_height, name="rms_height"))

    def _factor(self, skin_depth: torch.Tensor) -> torch.Tensor:
        real = 1.0 + torch.exp(-((skin_depth / (2.0 * self.rms_height)) ** 1.6))
        return torch.complex(real, torch.zeros_like(real))


__all__ = [
    "GroisseRoughness",
    "HammerstadRoughness",
    "HurayRoughness",
    "SurfaceRoughness",
]
