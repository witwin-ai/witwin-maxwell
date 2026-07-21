"""Differentiable smooth breakdown-risk surrogate (NON-PHYSICAL, NON-REGULATORY).

This module is deliberately kept separate from :mod:`witwin.maxwell.breakdown`
(the deterministic dynamic dielectric-breakdown feedback model) and from
:mod:`witwin.maxwell.breakdown_stress` (the auditable non-feedback stress
accumulator). Nothing here is a physical breakdown model, a failure predictor,
or a regulatory quantity: :class:`SmoothBreakdownRisk` is an **optimization
surrogate** whose only purpose is to expose a differentiable, monotone scalar
that increases as a recorded ``|E|(t)`` field approaches and exceeds a declared
critical field, so a trainable :class:`~witwin.maxwell.scene.SceneModule` can
push a design away from over-stress.

The hard field-duration/latching breakdown model is non-differentiable at the
trigger time and stays that way: trainable scenes that enable hard breakdown
feedback are still rejected at ``prepare()``. This surrogate is a **separate
non-feedback path** -- it never touches the field solve and never claims to
predict a breakdown event.

Definitions (all differentiable in ``torch``). Given a recorded, colocated
cell-center magnitude series ``e_mag`` of shape ``(T, *region)`` sampled at a
uniform step ``dt``, a critical field ``Ecrit`` and a margin width ``w``:

    margin(t, cell) = (|E|(t, cell) - Ecrit) / w
    p(t, cell)      = sigmoid(margin(t, cell))          # soft exceedance in (0, 1)
    soft_duration(cell) = sum_t p(t, cell) * dt         # soft dwell above threshold
    risk = reduce_cells( occupancy(cell) * soft_duration(cell) )

The default cell reduction is an occupancy-weighted sum (a "soft over-stress
dose", in seconds), which is strictly monotone in the field amplitude and decays
to zero far below the threshold. A ``"softmax"`` reduction is offered for a
differentiable worst-cell emphasis (temperature-controlled), mirroring the SAR
``soft_peak`` surrogate convention.

The scalar magnitude is colocated with the exact energy-consistent averaging the
:class:`~witwin.maxwell.monitors.BreakdownMonitor` uses
(:func:`~witwin.maxwell.breakdown_stress.colocate_electric_magnitude`), so the
surrogate reads "the same ``|E|``" as the physical stress accumulator -- only the
reduction is soft instead of a hard Heaviside/argmax.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from .breakdown_stress import colocate_electric_magnitude


SMOOTH_BREAKDOWN_RISK_VERSION = "smooth-breakdown-risk-1"
# The capability tag is intentionally verbose so a downstream consumer can never
# mistake a surrogate value for a physical or regulatory breakdown result.
SMOOTH_BREAKDOWN_CAPABILITY_LEVEL = "differentiable-surrogate (non-physical, non-regulatory)"

_REDUCTIONS = ("sum", "mean", "softmax")


@dataclass(frozen=True)
class SmoothBreakdownRisk:
    """Configuration of the differentiable smooth breakdown-risk surrogate.

    Capability level: **differentiable-surrogate (non-physical, non-regulatory)**.
    This is NOT a breakdown model. It defines a smooth, monotone functional of a
    recorded ``|E|(t)`` series used only as an optimization objective.

    Parameters
    ----------
    critical_field:
        Reference field ``Ecrit`` (V/m) about which the sigmoid margin is
        centered. Must be positive and finite. This is the same threshold a
        :class:`BreakdownMonitor` would compare against, reused here only to
        place the soft margin -- no exceedance decision is made.
    width:
        Margin width ``w`` (V/m) of the sigmoid. Smaller ``w`` sharpens the soft
        exceedance toward the hard Heaviside; larger ``w`` spreads the gradient.
        Must be positive and finite.
    reduction:
        Cell reduction of the per-cell soft dwell into the scalar risk:
        ``"sum"`` (default, occupancy-weighted over-stress dose in seconds),
        ``"mean"`` (occupancy-weighted mean dwell), or ``"softmax"``
        (temperature-weighted differentiable worst-cell dwell).
    temperature:
        Softmax temperature (seconds) for ``reduction="softmax"``; ignored
        otherwise. Must be positive when used.
    damage_exponent:
        Optional soft-damage weighting ``k``: when set, an auxiliary
        ``soft_damage(cell) = sum_t p * (|E|/Ecrit)^k * dt`` map is also
        produced. Does not change the primary ``risk`` scalar.
    """

    critical_field: float
    width: float
    reduction: str = "sum"
    temperature: float | None = None
    damage_exponent: float | None = None

    def __post_init__(self):
        ecrit = float(self.critical_field)
        if not math.isfinite(ecrit) or ecrit <= 0.0:
            raise ValueError("critical_field must be a positive finite field strength.")
        object.__setattr__(self, "critical_field", ecrit)
        w = float(self.width)
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError("width must be a positive finite field width.")
        object.__setattr__(self, "width", w)
        reduction = str(self.reduction)
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}; got {reduction!r}.")
        object.__setattr__(self, "reduction", reduction)
        if reduction == "softmax":
            if self.temperature is None:
                raise ValueError("reduction='softmax' requires a positive temperature.")
            temp = float(self.temperature)
            if not math.isfinite(temp) or temp <= 0.0:
                raise ValueError("temperature must be a positive finite value.")
            object.__setattr__(self, "temperature", temp)
        elif self.temperature is not None:
            object.__setattr__(self, "temperature", float(self.temperature))
        if self.damage_exponent is not None:
            k = float(self.damage_exponent)
            if not math.isfinite(k) or k <= 0.0:
                raise ValueError("damage_exponent must be a positive finite exponent or None.")
            object.__setattr__(self, "damage_exponent", k)

    @property
    def capability_level(self) -> str:
        return SMOOTH_BREAKDOWN_CAPABILITY_LEVEL

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        e_magnitude_series: torch.Tensor,
        dt: float,
        *,
        occupancy: torch.Tensor | None = None,
        name: str = "",
        provenance_extra: Mapping[str, Any] | None = None,
    ) -> "SmoothBreakdownRiskData":
        """Reduce a colocated ``|E|(t)`` series to a differentiable risk scalar.

        ``e_magnitude_series`` has shape ``(T, *region)`` and MUST already be the
        energy-consistent cell-center magnitude (see :meth:`evaluate_from_components`
        for the raw-Yee entry point). Autograd flows through ``e_magnitude_series``
        back to whatever produced it (source amplitude, material parameters), so
        the returned :attr:`SmoothBreakdownRiskData.risk` is trainable. No host
        synchronization is performed on the tensors.
        """

        if not torch.is_tensor(e_magnitude_series):
            raise TypeError("e_magnitude_series must be a torch tensor.")
        if e_magnitude_series.dim() < 2:
            raise ValueError("e_magnitude_series must have a leading time axis and >=1 cell axis.")
        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be a positive finite time step.")

        e_mag = e_magnitude_series
        region_shape = tuple(e_mag.shape[1:])
        if occupancy is None:
            weight = None
        else:
            if tuple(occupancy.shape) != region_shape:
                raise ValueError("occupancy must match the region (non-time) shape of the series.")
            weight = occupancy.to(device=e_mag.device, dtype=e_mag.dtype)
            if bool(torch.any(weight < 0.0)) or bool(torch.any(weight > 1.0)):
                raise ValueError("occupancy must lie in [0, 1].")

        margin = (e_mag - self.critical_field) / self.width
        p = torch.sigmoid(margin)  # (T, *region), soft exceedance in (0, 1)
        soft_duration = p.sum(dim=0) * step  # (*region)

        soft_damage = None
        if self.damage_exponent is not None:
            ratio = torch.clamp(e_mag / self.critical_field, min=0.0)
            soft_damage = (p * torch.pow(ratio, self.damage_exponent)).sum(dim=0) * step

        if weight is None:
            weighted_duration = soft_duration
        else:
            weighted_duration = soft_duration * weight

        risk = self._reduce(weighted_duration)

        # Diagnostic scalars (detached from the reduction path is unnecessary --
        # they stay in-graph, but callers typically read them as monitors).
        peak_soft_duration = weighted_duration.max() if weighted_duration.numel() else weighted_duration.sum()
        peak_instant_risk = p.max() if p.numel() else p.sum()

        provenance = {
            "kind": "smooth_breakdown_risk",
            "name": str(name),
            "capability_level": SMOOTH_BREAKDOWN_CAPABILITY_LEVEL,
            "model_version": SMOOTH_BREAKDOWN_RISK_VERSION,
            "non_physical": True,
            "non_regulatory": True,
            "note": (
                "Optimization surrogate only: soft sigmoid field-margin dwell. "
                "Not a breakdown prediction, failure probability, or regulatory metric."
            ),
            "definition": "p = sigmoid((|E| - Ecrit)/w); soft_duration = sum_t p dt; "
            "risk = reduce_cells(occupancy * soft_duration)",
            "critical_field": self.critical_field,
            "width": self.width,
            "reduction": self.reduction,
            "temperature": self.temperature,
            "damage_exponent": self.damage_exponent,
            "dt": step,
            "colocation": "cell-center energy-consistent (shared with BreakdownMonitor)",
            "units": {"field": "V/m", "time": "s", "risk": "s (soft over-stress dose)"},
        }
        if provenance_extra:
            provenance.update(dict(provenance_extra))

        return SmoothBreakdownRiskData(
            name=str(name),
            risk=risk,
            soft_duration_map=soft_duration,
            soft_damage_map=soft_damage,
            peak_soft_duration=peak_soft_duration,
            peak_instant_risk=peak_instant_risk,
            critical_field=self.critical_field,
            width=self.width,
            dt=step,
            reduction=self.reduction,
            provenance=provenance,
        )

    def evaluate_from_components(
        self,
        ex_series: torch.Tensor,
        ey_series: torch.Tensor,
        ez_series: torch.Tensor,
        dt: float,
        *,
        occupancy: torch.Tensor | None = None,
        name: str = "",
        provenance_extra: Mapping[str, Any] | None = None,
    ) -> "SmoothBreakdownRiskData":
        """Colocate raw Yee ``E`` component series then evaluate the surrogate.

        ``ex_series``/``ey_series``/``ez_series`` are ``(T, X, Y, Z)`` node-overhang
        blocks in the same staggering the observer feeds
        :func:`colocate_electric_magnitude` (each component one node larger along
        its two node-staggered axes). This reuses the exact energy-consistent
        colocation the physical stress accumulator uses, so the surrogate reads the
        same ``|E|``. Autograd flows through the components.
        """

        e_mag = colocate_electric_magnitude(ex_series, ey_series, ez_series)
        return self.evaluate(
            e_mag,
            dt,
            occupancy=occupancy,
            name=name,
            provenance_extra=provenance_extra,
        )

    def _reduce(self, weighted_duration: torch.Tensor) -> torch.Tensor:
        flat = weighted_duration.reshape(-1)
        if flat.numel() == 0:
            return flat.sum()
        if self.reduction == "sum":
            return flat.sum()
        if self.reduction == "mean":
            return flat.mean()
        # softmax: temperature-weighted differentiable worst-cell dwell.
        weights = torch.softmax(flat / self.temperature, dim=0)
        return (weights * flat).sum()


@dataclass(frozen=True)
class SmoothBreakdownRiskData:
    """Typed differentiable surrogate output -- NON-PHYSICAL, NON-REGULATORY.

    Capability level: **differentiable-surrogate (non-physical, non-regulatory)**.
    :attr:`risk` is the trainable scalar objective (occupancy-weighted soft
    over-stress dose, in seconds, or the softmax worst-cell dwell); it is NOT a
    breakdown probability, a failure prediction, or a regulatory metric. The
    per-cell :attr:`soft_duration_map` and the diagnostic peaks stay on device and
    in the autograd graph.
    """

    name: str
    risk: torch.Tensor
    soft_duration_map: torch.Tensor
    soft_damage_map: torch.Tensor | None
    peak_soft_duration: torch.Tensor
    peak_instant_risk: torch.Tensor
    critical_field: float
    width: float
    dt: float
    reduction: str
    provenance: Mapping[str, Any]

    @property
    def capability_level(self) -> str:
        return SMOOTH_BREAKDOWN_CAPABILITY_LEVEL


__all__ = [
    "SMOOTH_BREAKDOWN_CAPABILITY_LEVEL",
    "SMOOTH_BREAKDOWN_RISK_VERSION",
    "SmoothBreakdownRisk",
    "SmoothBreakdownRiskData",
]
