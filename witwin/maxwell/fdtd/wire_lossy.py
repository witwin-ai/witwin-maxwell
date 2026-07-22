"""Passive lossy-wire current recurrence (thin-wire finite-conductivity B2).

A PEC thin-wire segment advances its current with the lossless leapfrog

    L (I^{n+1/2} - I^{n-1/2}) / dt = e^n

where ``e^n = emf + V_tail - V_head`` is the driving voltage sampled at the
integer step and ``L`` is the per-segment external inductance. A finite round
conductor adds a per-unit-length internal series impedance
``Z'(omega) = R_dc + Z_excess(omega)`` (compiler/wire_impedance.py): the exact DC
resistance plus a *passive* rational skin-effect model realized as a discrete
state space ``(Ad, Bd, Cd, Dd)`` (the shared bilinear/trapezoidal discretization).

This module folds that series impedance into the current update so the segment
equation becomes

    L dI/dt + R0 I + w(t) = e(t),   dx/dt = A x + B I,   w = C x + D I

with ``R0 = R_dc * length`` the segment series resistance and ``w`` the excess
voltage from the ADE state ``x``. The excess state space is length-scaled
(``Cs = length * Cd``, ``direct = length * Dd``) so ``w`` is a segment voltage.

Companion discretization (dissipativity asserted via checked positive-real
conditions, not by construction). Keep the leapfrog current at half steps and
evaluate every loss term with the trapezoidal integer-step current
``I_bar = (I^{n+1/2} + I^{n-1/2}) / 2``:

    L (I^+ - I^-) / dt = e - R0 I_bar - (Cs x + direct I_bar)
                       = e - Cs x - G I_bar,     G = R0 + direct

    x^+ = Ad x + Bd I_bar

Because the excess state ``x`` is known from the previous step, ``I_bar`` couples
only ``I^+`` and ``I^-`` and the current update stays explicit:

    (L/dt + G/2) I^+ = (L/dt - G/2) I^- + (e - Cs x)

For ``G = 0`` (PEC) the update reduces bitwise to the lossless leapfrog and no
ADE state is carried. The discrete inductor-energy identity is

    Delta(1/2 L I^2)/dt = e I_bar - G I_bar^2 - Cs x I_bar,

so ``e I_bar`` drives the segment and ``G I_bar^2 + Cs x I_bar`` is the power into
the loss/storage branch.

Stability certificate (important — not positive-real by construction). ``G`` is
*not* guaranteed non-negative. ``G = R0 + length * Dd`` with ``Dd = D + (dt/2) Cd B``
the discrete direct term of the shared rational excess fit, and ``Dd`` is not
sign-definite: the skin-effect internal impedance is improper, so the fit carries
an out-of-band direct term whose sign is not controlled. The realized loss-branch
resistance ``Re(Z_branch)`` is positive across the fitting band (where the fit
reproduces the passive analytic internal impedance) but can go negative out of
band — the discrete companion is an active (negative-resistance) one-port outside
the fitted band. The recurrence is therefore *not* certified passive by a
positive-real realization argument.

Instead, stability is certified numerically at build time: the combined ``[I; x]``
linear transition of each segment companion is formed (``_companion_spectral_radius``)
and the highest fit order whose spectral radius is ``< 1 - margin`` on every sharing
segment is selected (``_fit_stable_order``); the build fails closed when no order in
range qualifies. This is an *isolated* per-segment certificate (the wire decoupled
from the Maxwell field). It bounds the homogeneous growth of each segment's own
recurrence but does not by itself prove the field-coupled system (the wire exchanges
energy with the grid every step via emf sampling and current deposition) is passive;
empirical closed-run boundedness against the PEC reference is used to check the
coupled behavior. The reported ohmic dissipation ``0.5 Re(Z') |I|^2`` uses the
in-band series AC resistance and is additionally clamped non-negative
(``ac_resistance_per_length``), so the monitored energy channel stays physical.

The realized excess impedance ``Cs (zI - Ad)^{-1} Bd + direct`` equals the fitted
continuous ``length * Z_excess(s)`` under the bilinear map ``s = (2/dt)(z-1)/(z+1)``,
so the recurrence reproduces the analytic internal impedance within the fit error
and the (in-band, small) bilinear frequency prewarp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..compiler.wire_impedance import (
    SeriesImpedanceModel,
    dc_resistance,
    fit_series_impedance,
    internal_impedance,
    internal_impedance_conductivity_gradient,
)
from ..constants import MU_0


class _AnalyticACResistance(torch.autograd.Function):
    """PyTorch-native analytic AC resistance ``Re(Z'(f; sigma))`` of a round wire.

    The forward evaluates the exact scaled-Bessel per-unit-length AC resistance at
    the requested frequencies; the backward returns the closed-form conductivity
    sensitivity ``Re(d Z'/d sigma)`` (compiler/wire_impedance.py), so a differentiable
    conductivity leaf flows through the wire's dissipation channel exactly. Only the
    scalar ``conductivity`` carries a gradient (radius/permeability/frequencies are
    fixed geometry passed as plain values); the analytic model is deterministic, so
    this path is free of the shared rational fit's nondeterminism (B1) that blocks
    the field-coupled current sensitivity.
    """

    @staticmethod
    def forward(ctx, conductivity, radius, permeability, frequencies):
        sigma = float(conductivity.detach().reshape(()).item())
        resistance = internal_impedance(
            radius, sigma, permeability, frequencies
        ).real.to(dtype=torch.float64)
        gradient = internal_impedance_conductivity_gradient(
            radius, sigma, permeability, frequencies
        ).real.to(dtype=torch.float64)
        ctx.save_for_backward(gradient.to(device=conductivity.device))
        return resistance.to(device=conductivity.device, dtype=conductivity.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (gradient,) = ctx.saved_tensors
        grad_sigma = torch.sum(
            grad_output.to(dtype=gradient.dtype) * gradient
        ).to(dtype=grad_output.dtype)
        return grad_sigma.reshape(()), None, None, None


def analytic_ac_resistance(
    conductivity: torch.Tensor,
    *,
    radius: float,
    permeability: float,
    frequencies,
) -> torch.Tensor:
    """Differentiable per-unit-length AC resistance ``Re(Z'(f; sigma))`` in ``[F]``.

    ``conductivity`` is a scalar tensor (a trainable conductivity leaf). The result
    is differentiable with respect to it via the closed-form analytic gradient, so
    ``0.5 * analytic_ac_resistance(...) * length * |I|^2`` gives an ohmic-dissipation
    objective whose conductivity gradient matches a central difference of the
    analytic internal impedance (the deterministic conductivity adjoint channel).
    """

    if not isinstance(conductivity, torch.Tensor) or conductivity.numel() != 1:
        raise ValueError("conductivity must be a scalar tensor.")
    freqs = tuple(float(value) for value in frequencies)
    if not freqs:
        raise ValueError("frequencies must be non-empty.")
    return _AnalyticACResistance.apply(
        conductivity, float(radius), float(permeability), freqs
    )


@dataclass(frozen=True)
class LossySegmentModel:
    """Per-segment passive series-impedance companion for the wire recurrence.

    All tensors are shape ``[S]`` (or ``[S, k]`` / ``[S, k, k]`` for the ADE
    state) over the network's physical segments. PEC segments carry ``R0 = 0``,
    ``G = 0`` and zero ADE blocks, so they advance exactly like the lossless path.
    """

    inductance: torch.Tensor  # [S] external per-segment inductance L
    resistance_dc: torch.Tensor  # [S] segment series DC resistance R0
    companion_conductance: torch.Tensor  # [S] G = R0 + length * Dd
    ade_transition: torch.Tensor  # [S, k, k] discrete Ad
    ade_input: torch.Tensor  # [S, k] discrete Bd (single input)
    ade_output: torch.Tensor  # [S, k] length-scaled Cd
    is_lossy: torch.Tensor  # [S] bool
    length: torch.Tensor  # [S] segment length
    segment_model_index: torch.Tensor  # [S] int64, -1 for PEC
    dt: float
    band: tuple[float, float]
    order: int
    models: tuple[SeriesImpedanceModel, ...]
    spectral_radius: float  # max certified combined companion spectral radius
    model_orders: tuple[int, ...]  # adaptively selected fit order per unique model

    @property
    def state_count(self) -> int:
        return int(self.ade_transition.shape[1])

    @property
    def segment_count(self) -> int:
        return int(self.inductance.shape[0])

    @property
    def any_lossy(self) -> bool:
        return bool(torch.any(self.is_lossy).item())

    def initial_state(self) -> torch.Tensor:
        """Return a zeroed ADE state ``[S, k]`` matching device/dtype."""

        return torch.zeros(
            self.segment_count,
            self.state_count,
            device=self.inductance.device,
            dtype=self.inductance.dtype,
        )

    def advance_current(
        self,
        current: torch.Tensor,
        ade_state: torch.Tensor,
        drive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance ``I^{n-1/2} -> I^{n+1/2}`` and the ADE state one step.

        ``drive`` is the integer-step driving voltage ``e^n = emf + V_tail - V_head``.
        Returns ``(current_next, ade_state_next)``. The map is the passive companion
        derived in the module docstring; ``G = 0`` segments recover the lossless
        leapfrog ``I^+ = I^- + dt * e / L``.
        """

        inductive = self.inductance / self.dt
        half_g = 0.5 * self.companion_conductance
        history = (self.ade_output * ade_state).sum(dim=-1)
        numerator = (inductive - half_g) * current + drive - history
        current_next = numerator / (inductive + half_g)
        current_bar = 0.5 * (current_next + current)
        ade_next = torch.einsum(
            "skj,sj->sk", self.ade_transition, ade_state
        ) + self.ade_input * current_bar.unsqueeze(-1)
        return current_next, ade_next

    def instantaneous_dissipation(
        self,
        current: torch.Tensor,
        current_next: torch.Tensor,
        ade_state: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-segment loss-branch power ``v_loss * I_bar`` this step.

        ``v_loss = G I_bar + Cs x`` is the total voltage across the series
        resistance plus excess-impedance branch, and ``I_bar`` the trapezoidal
        current. Its cycle average at a single tone equals ``0.5 Re(Z') |I|^2``
        (the reported ohmic dissipation); the instantaneous value also carries the
        reactive exchange with the ADE storage.
        """

        current_bar = 0.5 * (current_next + current)
        history = (self.ade_output * ade_state).sum(dim=-1)
        v_loss = self.companion_conductance * current_bar + history
        return v_loss * current_bar

    def ac_resistance_per_length(self, frequencies: torch.Tensor) -> torch.Tensor:
        """Return ``Re(Z'(f))`` per unit length, shape ``[F, S]``.

        PEC segments return zeros. Finite segments evaluate the fitted total series
        AC resistance ``R_dc + Re(excess_fit(f))`` (the resistance the recurrence
        realizes, matching the analytic curve within the fit tolerance).
        """

        freqs = torch.as_tensor(frequencies, dtype=torch.float64).reshape(-1)
        resistance = torch.zeros(
            freqs.numel(),
            self.segment_count,
            device=self.inductance.device,
            dtype=self.inductance.dtype,
        )
        index = self.segment_model_index.tolist()
        for segment, model_index in enumerate(index):
            if model_index < 0:
                continue
            model = self.models[model_index]
            values = model.ac_resistance(freqs).to(dtype=resistance.dtype)
            resistance[:, segment] = values.to(device=resistance.device)
        # The physical series AC resistance is non-negative; clamp the shared fit's
        # occasional sub-tolerance undershoot so the reported ohmic dissipation
        # 0.5 Re(Z') |I|^2 stays a valid (non-negative) energy accounting.
        return resistance.clamp_min(0.0)

    def conductivity_ac_resistance_gradient(
        self, frequencies: torch.Tensor
    ) -> torch.Tensor:
        """Return the analytic ``d Re(Z'(f)) / d sigma`` per length, shape ``[F, S]``.

        The deterministic conductivity-adjoint channel: the closed-form derivative
        of the exact scaled-Bessel AC resistance (compiler/wire_impedance.py),
        evaluated at each segment's built ``(radius, sigma, mu)``. PEC segments
        return zero. This is the sensitivity the reported ohmic dissipation
        ``0.5 Re(Z') length |I|^2`` differentiates through with the current held
        fixed; it does not require refitting the (nondeterministic) rational model.
        """

        freqs = torch.as_tensor(frequencies, dtype=torch.float64).reshape(-1)
        gradient = torch.zeros(
            freqs.numel(),
            self.segment_count,
            device=self.inductance.device,
            dtype=self.inductance.dtype,
        )
        index = self.segment_model_index.tolist()
        for segment, model_index in enumerate(index):
            if model_index < 0:
                continue
            model = self.models[model_index]
            values = internal_impedance_conductivity_gradient(
                model.radius, model.conductivity, model.permeability, freqs
            ).real.to(dtype=gradient.dtype)
            gradient[:, segment] = values.to(device=gradient.device)
        return gradient


def _companion_spectral_radius(
    discrete,
    resistance_dc_per_length: float,
    inductance: float,
    length: float,
    dt: float,
) -> float:
    """Return the exact linear transition spectral radius of one segment companion.

    The combined ``[I; x]`` update is linear; its spectral radius certifies whether
    the passive lossy recurrence grows. A value ``< 1`` guarantees no numerical
    growth over arbitrarily long runs (passivity). The leapfrog inductor and the
    bilinear ADE realization are not jointly a positive-real realization when the
    shared fit carries a large out-of-band direct term, so this certificate is the
    build-time stability gate that drives the adaptive order selection.
    """

    ad = discrete.A.to(dtype=torch.float64)
    k = ad.shape[0]
    bd = discrete.B.reshape(-1).to(dtype=torch.float64)
    cs = length * discrete.C.reshape(-1).to(dtype=torch.float64)
    r0 = resistance_dc_per_length * length
    direct = length * float(discrete.D.reshape(()))
    companion = r0 + direct
    inductive = inductance / dt
    half_g = 0.5 * companion
    denom = inductive + half_g
    a_ii = (inductive - half_g) / denom
    a_ix = (-cs) / denom
    bar_i = 0.5 * (a_ii + 1.0)
    bar_x = 0.5 * a_ix
    transition = torch.zeros(k + 1, k + 1, dtype=torch.float64)
    transition[0, 0] = a_ii
    transition[0, 1:] = a_ix
    transition[1:, 0] = bd * bar_i
    transition[1:, 1:] = ad + torch.outer(bd, bar_x)
    return float(torch.linalg.eigvals(transition).abs().max().item())


def _segment_conductor(metadata, segment_wire_ids: torch.Tensor):
    conductor = metadata.get("conductor")
    if conductor is None:
        return None
    kinds = conductor["kinds"]
    conductivity = conductor["conductivity"]
    permeability = conductor["permeability"]
    wire_ids = segment_wire_ids.detach().cpu().tolist()
    seg_kinds = [kinds[wire_id] for wire_id in wire_ids]
    seg_sigma = [conductivity[wire_id] for wire_id in wire_ids]
    seg_mu = [permeability[wire_id] for wire_id in wire_ids]
    return seg_kinds, seg_sigma, seg_mu


def _fit_stable_order(
    a: float,
    sigma: float,
    mu: float,
    *,
    band: tuple[float, float],
    dt: float,
    samples: int,
    device,
    dtype,
    segment_inductance: list[float],
    segment_length: list[float],
    order_max: int,
    order_min: int,
    stability_margin: float,
) -> tuple[SeriesImpedanceModel, int, float]:
    """Fit the highest order whose companion is certified stable on every segment.

    Higher order improves the in-band impedance accuracy but degrades the joint
    leapfrog/ADE conditioning (the shared fit's out-of-band direct term grows), so
    the highest order with combined spectral radius ``< 1 - margin`` on all sharing
    segments is the best accuracy that is provably non-growing. Fails closed when
    no order in range qualifies (a genuine positive-real-realization blocker).
    """

    resistance_dc_per_length = float(dc_resistance(a, sigma).item())
    best: tuple[SeriesImpedanceModel, int, float] | None = None
    last_error: str | None = None
    for order in range(order_max, order_min - 1, -1):
        try:
            # relative_tolerance here is deliberately loose (0.5, vs the fitter
            # default 0.1). This is NOT the accuracy contract for the shipped
            # model: the binding acceptance gates are downstream — (1) the in-band
            # AC-resistance positivity check below, (2) the combined-companion
            # spectral-radius stability certificate, and (3) the analytic
            # AC-resistance sweep test (gate (a), ~8% on the certified config).
            # We loosen the fitter's own fail-closed error ceiling so that when
            # accuracy and stability trade off (high order fits accurately but can
            # exceed the spectral-radius bound; low order is stable but less
            # accurate), a stable lower-order fit is not rejected purely on the
            # fitter's raw in-band error before the stability selection runs. The
            # cost is that a config whose only stable fit has 10-50% in-band error
            # runs instead of failing closed; such configs are surfaced by the
            # analytic AC sweep gate rather than by this ceiling.
            model = fit_series_impedance(
                a,
                sigma,
                band=band,
                permeability=mu,
                order=order,
                dt=dt,
                samples=samples,
                relative_tolerance=0.5,
                device=device,
                dtype=dtype,
            )
        except (RuntimeError, ValueError) as error:
            last_error = f"order {order}: {error}"
            continue
        if model.discrete is None:
            raise RuntimeError("Lossy wire fit did not produce a discrete ADE; dt must be set.")
        # Physical quality gate: the realized in-band series AC resistance must be
        # strictly positive (a passive conductor never has negative loss). The
        # shared vector fit is nondeterministic (B1) and can occasionally dip
        # non-positive in band; reject such an order so both the recurrence and the
        # reported ohmic_loss stay physical.
        band_samples = torch.logspace(
            math.log10(band[0]), math.log10(band[1]), 24, dtype=torch.float64
        )
        min_ac = float(model.ac_resistance(band_samples).min().item())
        if min_ac <= 0.0:
            last_error = f"order {order}: in-band AC resistance dipped to {min_ac:.4g} <= 0"
            continue
        radius_spectral = max(
            _companion_spectral_radius(
                model.discrete,
                resistance_dc_per_length,
                inductance,
                length,
                dt,
            )
            for inductance, length in zip(segment_inductance, segment_length)
        )
        if radius_spectral < 1.0 - stability_margin:
            return model, order, radius_spectral
        last_error = f"order {order}: combined spectral radius {radius_spectral:.7g} >= 1"
    raise RuntimeError(
        "Lossy thin-wire recurrence could not be certified passive: no fit order in "
        f"[{order_min}, {order_max}] yields a non-growing companion "
        f"(radius a={a:g}, sigma={sigma:g}). The skin-effect impedance is improper and "
        "the shared rational fit does not guarantee a positive-real realization; a "
        "positive-real-preserving fit refinement is required. Last: "
        f"{last_error}."
    )


def build_lossy_segment_model(
    *,
    inductance: torch.Tensor,
    radius: torch.Tensor,
    length: torch.Tensor,
    segment_wire_ids: torch.Tensor,
    metadata,
    band: tuple[float, float],
    dt: float,
    order: int = 13,
    order_min: int = 6,
    samples: int = 240,
    stability_margin: float = 1.0e-6,
) -> LossySegmentModel | None:
    """Build the per-segment companion, or ``None`` when every segment is PEC.

    One passive rational ADE is fitted per unique ``(radius, sigma, mu)`` triple
    (per the no-duplicate-pole-fitting rule the fit reuses the shared rational
    stack) and shared across segments; the per-segment matrices are gathered from
    those fits and length-scaled. ``order`` is the maximum fit order; the highest
    order with a certified non-growing companion (see ``_fit_stable_order``) is
    selected per unique model. ``band`` is the fitting/validity band and must
    bracket the frequency content the run cares about.
    """

    conductor = _segment_conductor(metadata, segment_wire_ids)
    if conductor is None:
        return None
    seg_kinds, seg_sigma, seg_mu = conductor
    if not any(kind == "finite" for kind in seg_kinds):
        return None

    device = inductance.device
    dtype = inductance.dtype
    segment_count = int(inductance.shape[0])
    radius_cpu = radius.detach().cpu().to(dtype=torch.float64).tolist()
    inductance_cpu = inductance.detach().cpu().to(dtype=torch.float64).tolist()
    length_cpu = length.detach().cpu().to(dtype=torch.float64).tolist()

    unique_keys: dict[tuple[float, float, float], int] = {}
    key_segments: dict[tuple[float, float, float], list[int]] = {}
    for segment in range(segment_count):
        if seg_kinds[segment] != "finite":
            continue
        key = (
            round(float(radius_cpu[segment]), 15),
            round(float(seg_sigma[segment]), 6),
            round(float(seg_mu[segment]), 12),
        )
        key_segments.setdefault(key, []).append(segment)

    models: list[SeriesImpedanceModel] = []
    model_orders: list[int] = []
    model_spectral: list[float] = []
    segment_model_index = [-1] * segment_count
    for segment in range(segment_count):
        if seg_kinds[segment] != "finite":
            continue
        key = (
            round(float(radius_cpu[segment]), 15),
            round(float(seg_sigma[segment]), 6),
            round(float(seg_mu[segment]), 12),
        )
        model_index = unique_keys.get(key)
        if model_index is None:
            members = key_segments[key]
            model, selected_order, radius_spectral = _fit_stable_order(
                float(radius_cpu[segment]),
                float(seg_sigma[segment]),
                float(seg_mu[segment]),
                band=band,
                dt=dt,
                samples=samples,
                device=device,
                dtype=dtype,
                segment_inductance=[inductance_cpu[member] for member in members],
                segment_length=[length_cpu[member] for member in members],
                order_max=int(order),
                order_min=int(order_min),
                stability_margin=float(stability_margin),
            )
            model_index = len(models)
            models.append(model)
            model_orders.append(selected_order)
            model_spectral.append(radius_spectral)
            unique_keys[key] = model_index
        segment_model_index[segment] = model_index

    state_count = max((model.discrete.A.shape[0] for model in models), default=0)
    state_count = max(state_count, 1)

    resistance_dc = torch.zeros(segment_count, device=device, dtype=dtype)
    companion = torch.zeros(segment_count, device=device, dtype=dtype)
    ade_transition = torch.zeros(
        segment_count, state_count, state_count, device=device, dtype=dtype
    )
    ade_input = torch.zeros(segment_count, state_count, device=device, dtype=dtype)
    ade_output = torch.zeros(segment_count, state_count, device=device, dtype=dtype)
    is_lossy = torch.zeros(segment_count, device=device, dtype=torch.bool)
    length_dtype = length.to(device=device, dtype=dtype)

    for segment in range(segment_count):
        model_index = segment_model_index[segment]
        if model_index < 0:
            continue
        model = models[model_index]
        discrete = model.discrete
        k = discrete.A.shape[0]
        seg_length = length_dtype[segment]
        r0 = float(model.resistance_dc) * seg_length
        direct = seg_length * discrete.D.reshape(())
        resistance_dc[segment] = r0
        companion[segment] = r0 + direct
        ade_transition[segment, :k, :k] = discrete.A.to(device=device, dtype=dtype)
        ade_input[segment, :k] = discrete.B.reshape(-1).to(device=device, dtype=dtype)
        ade_output[segment, :k] = (
            seg_length * discrete.C.reshape(-1).to(device=device, dtype=dtype)
        )
        is_lossy[segment] = True

    return LossySegmentModel(
        inductance=inductance,
        resistance_dc=resistance_dc,
        companion_conductance=companion,
        ade_transition=ade_transition,
        ade_input=ade_input,
        ade_output=ade_output,
        is_lossy=is_lossy,
        length=length_dtype,
        segment_model_index=torch.tensor(
            segment_model_index, device=device, dtype=torch.int64
        ),
        dt=float(dt),
        band=(float(band[0]), float(band[1])),
        order=int(order),
        models=tuple(models),
        spectral_radius=max(model_spectral) if model_spectral else 0.0,
        model_orders=tuple(model_orders),
    )


def resolve_lossy_band(
    frequencies: tuple[float, ...],
    *,
    margin: float = 2.0,
) -> tuple[float, float]:
    """Return the fitting band bracketing ``frequencies`` with a factor margin.

    The lossy recurrence needs a finite validity band. It is derived from the
    frequencies the run monitors so the passive fit is accurate where the answer
    is read out. A lossy wire with no frequency content is a fail-closed error at
    the call site.
    """

    positive = sorted(float(value) for value in frequencies if float(value) > 0.0)
    if not positive:
        raise ValueError(
            "A lossy thin wire requires monitored frequencies to set its "
            "series-impedance fitting band; add a WireMonitor (or DFT) frequency."
        )
    f_min = positive[0] / margin
    f_max = positive[-1] * margin
    if not math.isfinite(f_max) or f_min <= 0.0 or f_min >= f_max:
        raise ValueError("Could not derive a valid lossy-wire fitting band.")
    return (f_min, f_max)


__all__ = [
    "LossySegmentModel",
    "analytic_ac_resistance",
    "build_lossy_segment_model",
    "resolve_lossy_band",
    "dc_resistance",
    "MU_0",
]
