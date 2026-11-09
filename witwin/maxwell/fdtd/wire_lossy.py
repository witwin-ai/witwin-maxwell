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

The half-step current uses ``I_bar = (I_next + I)/2``. The shared bilinear
realization supplies the midpoint loss voltage from the pre-step ADE state:

    (L/dt + G/2) I_next = (L/dt - G/2) I + e - Cs x
    x_next = Ad x + Bd I_bar,  G = R0 + length * Dd

Then the node charge advances with the NEW current, exactly as in the PEC
leapfrog: ``q_next = q - dt B I_next``, with grounded charge fixed to zero.
The sampling/deposition pair and the pure-PEC CUDA path are unchanged.

At the final solver dt, preparation checks the installed branch coefficients,
not just the continuous fit. For each diagonal RL state, ``|Ad| < 1`` and
``Cs*Bd <= 0`` make its unit-circle resistance monotone in ``cos(theta)``.
Its minimum is at DC; checking that endpoint certifies nonnegative resistance
on the ENTIRE digital band ``0 <= theta <= pi`` (including Nyquist), with the
actual ``I_bar`` centering. This is an analytic interval certificate, not a
finite-band sample test.

Preparation also forms the exact homogeneous ``[I; voltage; ADE]`` transition
of the installed wire graph, including node capacitances, grounding, incidence,
and the new-current charge update. Its spectral radius must be at most one
(up to 1e-10 eigensolver roundoff), or preparation fails. Conserved charge and
cycle modes can lie on the unit circle. This graph certificate excludes Maxwell
and port states; their energy-paired coupling retains the existing joint CFL
bound. Long-run coupled pulse decay is validated separately. No isolated
current/ADE certificate is presented as a proof of full field/port stability.

The realized branch is the continuous Foster impedance under the bilinear map
``s = (2/dt)(z-1)/(z+1)``. In-band accuracy therefore includes the usual frequency
warping. Reported phasor loss uses the continuous fitted AC resistance; at finite
dt, the midpoint current also differs from the half-step current by a cosine
factor. Instantaneous branch power includes reversible ADE storage exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    this analytic reference path is independent of the passive fit. The realized
    fit sensitivity is available on ``LossySegmentModel``; the field-coupled
    current/ADE reverse replay remains unimplemented.
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
    ``G = 0`` and zero ADE blocks, so their mathematical update is lossless; pure PEC uses the original CUDA path.
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
    spectral_radius: float  # certified current/charge/ADE graph spectral radius
    model_orders: tuple[int, ...]  # fixed RL pole count per unique model
    minimum_digital_resistance: float  # full-band lower bound, segment ohms

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
        current. At a single digital frequency its cycle average is
        ``0.5 Re(Z_discrete) |I_bar|^2``, including bilinear warping and midpoint
        centering. The instantaneous value also includes ADE storage exchange.
        """

        current_bar = 0.5 * (current_next + current)
        history = (self.ade_output * ade_state).sum(dim=-1)
        v_loss = self.companion_conductance * current_bar + history
        return v_loss * current_bar

    def ac_resistance_per_length(self, frequencies: torch.Tensor) -> torch.Tensor:
        """Return ``Re(Z'(f))`` per unit length, shape ``[F, S]``.

        PEC segments return zeros. Finite segments evaluate the fitted total series
        AC resistance ``R_dc + Re(excess_fit(f))`` at continuous frequency f.
        The recurrence realizes this curve at the bilinear-warped frequency.
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
        return resistance

    def conductivity_ac_resistance_gradient(
        self, frequencies: torch.Tensor
    ) -> torch.Tensor:
        """Return ``d Re(Z_fit'(f))/d sigma``, shape ``[F, S]``.

        The analytic Bessel derivative is projected through the fixed-pole NNLS
        active-set equations, including relative-weight and exact DC derivatives.
        Current is held fixed. This differentiates the realized fit's phasor
        dissipation, rather than substituting the analytic reference sensitivity.
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
            values = model.conductivity_gradient(freqs).real.to(dtype=gradient.dtype)
            gradient[:, segment] = values.to(device=gradient.device)
        return gradient


def _certify_discrete_graph(model, *, tail, head, node_capacitance, grounded):
    """Certify the installed digital branch and actual charge-coupled graph.

    Voltage coordinates eliminate unobservable constant node charge. ADE states
    are similarity-scaled by Bd to avoid ill-conditioning from SI units. All
    certificate algebra stays on the runtime device in float64.
    """

    ad = model.ade_transition.double()
    poles = ad.diagonal(dim1=-2, dim2=-1)
    bd = model.ade_input.double()
    cb = model.ade_output.double() * bd
    lossy = model.is_lossy
    minimum = model.companion_conductance.double() + (cb / (1.0 - poles)).sum(-1)
    if not bool(torch.all(poles.abs() < 1.0) & torch.all(cb <= 0.0)
                & torch.all(minimum >= 0.0)):
        raise RuntimeError("Lossy wire installed branch failed full-digital-band passivity.")

    count, states = bd.shape
    device = bd.device
    incidence = torch.zeros(node_capacitance.numel(), count, device=device, dtype=torch.float64)
    columns = torch.arange(count, device=device)
    incidence[tail, columns] += 1.0
    incidence[head, columns] -= 1.0
    inverse_c = torch.where(grounded, 0.0, node_capacitance.double().reciprocal())
    stiffness = incidence.T @ (inverse_c[:, None] * incidence)
    # v = (V_tail - V_head)/(L/dt), y = x/Bd. Charge is updated AFTER I.
    inductive = model.inductance.double() / model.dt
    half_g = model.companion_conductance.double() / 2.0
    denominator = inductive + half_g
    dimension = count * (states + 2)
    basis = torch.eye(dimension, device=device, dtype=torch.float64)
    current, voltage = basis[:count], basis[count:2*count]
    ade = basis[2*count:].reshape(count, states, dimension)
    next_current = (
        (inductive - half_g)[:, None] * current + inductive[:, None] * voltage
        - torch.einsum("sk,skd->sd", cb, ade)
    ) / denominator[:, None]
    next_voltage = voltage - model.dt / inductive[:, None] * (stiffness @ next_current)
    next_ade = torch.einsum("skj,sjd->skd", ad, ade) + (
        lossy[:, None, None] * 0.5 * (next_current + current)[:, None, :]
    )
    transition = torch.cat((next_current, next_voltage, next_ade.flatten(0, 1)))
    spectral_radius = float(torch.linalg.eigvals(transition).abs().max())
    if not math.isfinite(spectral_radius) or spectral_radius > 1.0 + 1.0e-10:
        raise RuntimeError(
            f"Lossy wire current/charge/ADE spectral radius {spectral_radius:.12g} exceeds one."
        )
    return replace(model, spectral_radius=spectral_radius,
                   minimum_digital_resistance=float(minimum[lossy].min()))


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


def build_lossy_segment_model(
    *,
    inductance: torch.Tensor,
    radius: torch.Tensor,
    length: torch.Tensor,
    segment_wire_ids: torch.Tensor,
    tail: torch.Tensor,
    head: torch.Tensor,
    node_capacitance: torch.Tensor,
    grounded: torch.Tensor,
    metadata,
    band: tuple[float, float],
    dt: float,
    order: int = 13,
    samples: int = 240,
) -> LossySegmentModel | None:
    """Build the per-segment companion, or ``None`` when every segment is PEC.

    One fixed-pole passive RL fit is shared per ``(radius, sigma, mu)`` triple.
    The installed coefficients are certified over DC through Nyquist, and the
    complete charge-coupled graph is checked at ``dt`` before returning. ``band``
    sets the impedance accuracy band; it does not limit passivity.
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

    unique_keys: dict[tuple[float, float, float], int] = {}
    models: list[SeriesImpedanceModel] = []
    model_orders: list[int] = []
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
            model = fit_series_impedance(
                float(radius_cpu[segment]),
                float(seg_sigma[segment]),
                permeability=float(seg_mu[segment]),
                band=band,
                dt=dt,
                samples=samples,
                device=device,
                dtype=dtype,
                order=order,
            )
            model_index = len(models)
            models.append(model)
            model_orders.append(order)
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

    model = LossySegmentModel(
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
        spectral_radius=0.0,
        minimum_digital_resistance=0.0,
        model_orders=tuple(model_orders),
    )
    return _certify_discrete_graph(
        model, tail=tail, head=head, node_capacitance=node_capacitance, grounded=grounded
    )


def resolve_lossy_band(
    frequencies: tuple[float, ...],
    *,
    margin: float = 2.0,
) -> tuple[float, float]:
    """Return the fitting band bracketing ``frequencies`` with a factor margin.

    The lossy fit needs a finite accuracy band. It is derived from the
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
