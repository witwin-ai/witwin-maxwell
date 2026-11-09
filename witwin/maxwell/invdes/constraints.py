"""Binarization penalty and geometric minimum length-scale constraints.

The length-scale constraints follow Zhou, Lazarov, Wang and Sigmund, "Minimum
length scale in topology optimization by geometric constraints", CMAME 293
(2015). With the filtered field ``x_f`` and the projected design ``x_p``:

* solid indicator ``I_s = x_p * exp(-c |grad x_f|^2)``
* void indicator  ``I_v = (1 - x_p) * exp(-c |grad x_f|^2)``
* line width ``g_s = mean(I_s * min(x_f - eta_e, 0)^2)``
* gap        ``g_v = mean(I_v * min(eta_d - x_f, 0)^2)``

The exponential concentrates the indicators on the skeleton of each phase (the
inflection region where the filtered gradient vanishes). A solid (void) feature
that is wide enough with respect to the filter radius reaches ``x_f >= eta_e``
(``x_f <= eta_d``) on its skeleton, so the constraint vanishes; thinner features
leave a positive residual. Impose them as ``g <= epsilon`` (a small tolerance
such as ``1e-8``) or add them as weighted penalties.
"""

from __future__ import annotations

import math

import torch

from .filters import _normalize_spacing
from .projection import _gradient_norm


def gray_level_penalty(x: torch.Tensor) -> torch.Tensor:
    """Measure of non-discreteness ``mean(4 x (1 - x))`` in ``[0, 1]``.

    Zero for a binary design, one for a uniform ``0.5`` design (Sigmund, 2007).
    """
    return (4.0 * x * (1.0 - x)).mean()


def length_scale_coefficient(radius: float, spacing) -> float:
    """Default indicator decay ``c`` (m^2) for a filter of physical ``radius``.

    Reproduces the recommended ``c = R^4`` of Zhou et al. (filter radius and
    gradients in element units) for physical gradients:
    ``c = radius^4 / h^2`` with ``h`` the smallest design-cell spacing.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite.")
    spacing = (float(spacing),) if isinstance(spacing, (int, float)) else tuple(float(h) for h in spacing)
    h = min(spacing)
    return radius**4 / (h * h)


def _indicator_decay(filtered: torch.Tensor, spacing, radius, c) -> torch.Tensor:
    if filtered.ndim not in (2, 3):
        raise ValueError("length-scale constraints support 2D and 3D design grids only.")
    spacing = _normalize_spacing(spacing, filtered.ndim)
    if c is None:
        if radius is None:
            raise ValueError("pass either the filter radius or the indicator coefficient c.")
        c = length_scale_coefficient(radius, spacing)
    return torch.exp(-float(c) * _gradient_norm(filtered, spacing))


def solid_indicator(filtered, projected, *, spacing, radius=None, c=None) -> torch.Tensor:
    """Solid-skeleton indicator ``projected * exp(-c |grad filtered|^2)``."""
    return projected * _indicator_decay(filtered, spacing, radius, c)


def void_indicator(filtered, projected, *, spacing, radius=None, c=None) -> torch.Tensor:
    """Void-skeleton indicator ``(1 - projected) * exp(-c |grad filtered|^2)``."""
    return (1.0 - projected) * _indicator_decay(filtered, spacing, radius, c)


def minimum_linewidth_constraint(
    filtered: torch.Tensor,
    projected: torch.Tensor,
    *,
    spacing,
    radius: float | None = None,
    eta_e: float = 0.75,
    c: float | None = None,
) -> torch.Tensor:
    """Solid minimum line-width constraint value ``g_s >= 0`` (zero when satisfied).

    ``filtered`` is the filtered density, ``projected`` its projection (usually
    the nominal design with ``eta = 0.5``) and ``eta_e`` the eroded threshold
    that sets the imposed solid length scale relative to the filter radius.
    """
    indicator = solid_indicator(filtered, projected, spacing=spacing, radius=radius, c=c)
    return (indicator * torch.clamp(filtered - eta_e, max=0.0) ** 2).mean()


def minimum_gap_constraint(
    filtered: torch.Tensor,
    projected: torch.Tensor,
    *,
    spacing,
    radius: float | None = None,
    eta_d: float = 0.25,
    c: float | None = None,
) -> torch.Tensor:
    """Void minimum gap constraint value ``g_v >= 0`` (zero when satisfied).

    ``eta_d`` is the dilated threshold that sets the imposed void length scale.
    """
    indicator = void_indicator(filtered, projected, spacing=spacing, radius=radius, c=c)
    return (indicator * torch.clamp(eta_d - filtered, max=0.0) ** 2).mean()
