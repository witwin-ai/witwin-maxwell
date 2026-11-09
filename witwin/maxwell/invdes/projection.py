"""Threshold projections, robust erode/dilate variants and beta continuation.

All projections map a filtered density ``x`` in ``[0, 1]`` to a (nearly) binary
density with threshold ``eta`` and sharpness ``beta``. ``beta`` may be a Python
float (``math.inf`` gives the exact step), and ``eta`` may be a float or a tensor
broadcastable to ``x`` (spatially varying thresholds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from .filters import _normalize_spacing


def _check_beta(beta) -> None:
    if isinstance(beta, torch.Tensor):
        return
    beta = float(beta)
    if math.isnan(beta) or beta <= 0.0:
        raise ValueError("projection beta must be > 0 (math.inf selects the exact step).")


def tanh_projection(x: torch.Tensor, beta, eta=0.5) -> torch.Tensor:
    """Smoothed Heaviside projection.

    ``(tanh(beta * eta) + tanh(beta * (x - eta))) / (tanh(beta * eta) + tanh(beta * (1 - eta)))``

    maps ``0 -> 0`` and ``1 -> 1`` exactly for every ``beta``, and approaches the
    step ``x > eta`` as ``beta -> inf``. ``beta = math.inf`` returns that step
    (``0.5`` exactly at ``x == eta``), which carries no gradient.
    """
    _check_beta(beta)
    if not isinstance(beta, torch.Tensor) and math.isinf(float(beta)):
        step = (x > eta).to(x.dtype)
        return torch.where(x == eta, torch.full_like(x, 0.5), step)
    eta_t = torch.as_tensor(eta, device=x.device, dtype=x.dtype)
    beta_t = torch.as_tensor(beta, device=x.device, dtype=x.dtype)
    low = torch.tanh(beta_t * eta_t)
    high = torch.tanh(beta_t * (1.0 - eta_t))
    return (low + torch.tanh(beta_t * (x - eta_t))) / (low + high)


class RobustProjections(NamedTuple):
    """Eroded, nominal and dilated projections of one filtered density."""

    eroded: torch.Tensor
    nominal: torch.Tensor
    dilated: torch.Tensor


def robust_projections(x: torch.Tensor, beta, eta: float = 0.5, delta: float = 0.1) -> RobustProjections:
    """Return threshold-shifted projections for robust (erode/dilate) design.

    The eroded design uses ``eta + delta`` (less solid), the dilated design uses
    ``eta - delta`` (more solid). Optimizing the worst of the three objectives
    makes the nominal design tolerant to uniform edge deviations and imposes a
    minimum length scale on both phases (Wang, Lazarov and Sigmund, 2011).
    """
    eta = float(eta)
    delta = float(delta)
    if not 0.0 <= delta < min(eta, 1.0 - eta) + 1e-12:
        raise ValueError("robust_projections requires 0 <= delta < min(eta, 1 - eta).")
    return RobustProjections(
        eroded=tanh_projection(x, beta, eta + delta),
        nominal=tanh_projection(x, beta, eta),
        dilated=tanh_projection(x, beta, eta - delta),
    )


def _gradient_norm(x: torch.Tensor, spacing: tuple[float, ...]) -> torch.Tensor:
    """Squared physical gradient magnitude, skipping length-one axes."""
    total = torch.zeros_like(x)
    for axis, h in enumerate(spacing):
        if int(x.shape[axis]) < 2:
            continue
        (grad,) = torch.gradient(x, spacing=(h,), dim=axis)
        total = total + grad * grad
    return total


def _fill_fraction(t: torch.Tensor) -> torch.Tensor:
    # Solid fraction of a smoothing ball cut at normalized signed distance t in
    # [-1, 1]; a quintic with zero slope at t = +/-1 so the smoothing switch is C1.
    return 0.5 - (15.0 / 16.0) * t + (5.0 / 8.0) * t**3 - (3.0 / 16.0) * t**5


def smoothed_projection(
    x: torch.Tensor,
    beta,
    eta: float = 0.5,
    *,
    spacing,
    smoothing_radius: float | None = None,
) -> torch.Tensor:
    """Subpixel-smoothed projection (SSP) of a filtered density.

    Follows the subpixel-smoothed projection of Hammond, Oskooi, Johnson et al.
    ("Unifying and accelerating level-set and density-based topology
    optimization by subpixel-smoothed projection"). Each design cell is modeled
    as a ball of radius ``R = smoothing_radius`` (default ``0.55 * min(spacing)``)
    across which the filtered field is linear. With the signed interface distance
    ``d = (eta - x) / |grad x|`` and ``t = d / R``, cells with ``|t| < 1`` blend

    ``(1 - F(t)) * P(x - R |grad x| F(t)) + F(t) * P(x + R |grad x| (1 - F(t)))``

    where ``P`` is :func:`tanh_projection` and ``F`` is a smooth fill fraction;
    all other cells use ``P(x)``. As ``beta -> inf`` the result becomes the fill
    fraction itself, so the projection stays differentiable with nonzero
    interface gradients even for an exactly binary (``beta = math.inf``) design.

    ``spacing`` is the physical design-cell size (metres); gradients are taken
    with ``torch.gradient`` (one-sided at the boundary), and length-one axes are
    treated as extruded.
    """
    if x.ndim not in (2, 3):
        raise ValueError("smoothed_projection supports 2D and 3D design grids only.")
    spacing = _normalize_spacing(spacing, x.ndim)
    if smoothing_radius is None:
        active = [h for h, n in zip(spacing, x.shape) if n > 1]
        radius = 0.55 * min(active) if active else math.inf
    else:
        radius = float(smoothing_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("smoothing_radius must be positive and finite.")

    projected = tanh_projection(x, beta, eta)
    grad_sq = _gradient_norm(x, spacing)
    nonzero = grad_sq > 0
    # Double-where keeps sqrt and the division differentiable at flat cells.
    grad_norm = torch.sqrt(torch.where(nonzero, grad_sq, torch.ones_like(grad_sq)))
    t = (eta - x) / (radius * grad_norm)
    smooth = nonzero & (t.abs() < 1.0)
    t = torch.where(smooth, t, torch.zeros_like(t))
    fill = _fill_fraction(t)
    span = radius * grad_norm
    lower = tanh_projection(x - span * fill, beta, eta)
    upper = tanh_projection(x + span * (1.0 - fill), beta, eta)
    blended = (1.0 - fill) * lower + fill * upper
    return torch.where(smooth, blended, projected)


@dataclass(frozen=True)
class BetaSchedule:
    """Deterministic geometric beta continuation.

    ``beta(i) = min(maximum, initial * factor ** ((i - warmup) // interval))`` for
    ``i >= warmup`` and ``initial`` before. With ``factor=2, interval=25`` beta
    doubles every 25 iterations. ``maximum=math.inf`` removes the cap (beta stays
    finite); pass ``math.inf`` to the projection explicitly for an exact final step,
    which keeps nonzero gradients only with :func:`smoothed_projection`.
    """

    initial: float = 1.0
    factor: float = 2.0
    interval: int = 25
    maximum: float = 64.0
    warmup: int = 0

    def __post_init__(self):
        if not self.initial > 0.0 or math.isinf(self.initial):
            raise ValueError("BetaSchedule.initial must be positive and finite.")
        if not self.factor >= 1.0 or math.isinf(self.factor):
            raise ValueError("BetaSchedule.factor must be finite and >= 1.")
        if isinstance(self.interval, bool) or int(self.interval) != self.interval or self.interval < 1:
            raise ValueError("BetaSchedule.interval must be a positive integer.")
        if isinstance(self.warmup, bool) or int(self.warmup) != self.warmup or self.warmup < 0:
            raise ValueError("BetaSchedule.warmup must be a nonnegative integer.")
        if not self.maximum >= self.initial:
            raise ValueError("BetaSchedule.maximum must be >= initial.")

    def __call__(self, iteration: int) -> float:
        iteration = int(iteration)
        if iteration < 0:
            raise ValueError("iteration must be nonnegative.")
        stage = max(0, iteration - self.warmup) // self.interval
        # Compare in log space so very late iterations cannot overflow.
        if math.log(self.initial) + stage * math.log(self.factor) >= math.log(self.maximum):
            return float(self.maximum)
        return float(self.initial * self.factor**stage)

    def values(self, iterations: int) -> list[float]:
        """Return ``[beta(0), ..., beta(iterations - 1)]``."""
        return [self(i) for i in range(int(iterations))]

    def milestones(self, iterations: int) -> list[int]:
        """Iterations below ``iterations`` at which beta changes value."""
        values = self.values(iterations)
        return [i for i in range(1, len(values)) if values[i] != values[i - 1]]


__all__ = (
    "BetaSchedule",
    "RobustProjections",
    "robust_projections",
    "smoothed_projection",
    "tanh_projection",
)
