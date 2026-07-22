"""Finite-conductivity thin-wire series impedance model.

This module turns a solid round conductor (bulk conductivity and permeability)
into the per-unit-length series impedance ``Z'(omega)`` that a lossy thin wire
adds to its current recurrence. It has two layers:

* An analytic layer (:func:`dc_resistance`, :func:`internal_impedance`,
  :func:`surface_resistance`, :func:`ohmic_loss_density`) that evaluates the
  exact round-wire skin-effect impedance. This is the reference curve the
  acceptance gates compare against; it is prepare/validation-time work, not a
  solver hot path, and is deliberately not differentiable.
* A fitting layer (:func:`fit_series_impedance`) that represents the
  frequency-dependent excess impedance ``Z'(omega) - R_dc`` as a *passive*
  rational auxiliary-differential-equation (ADE) model.

The fitting layer does **not** implement its own pole fitting. Per the thin-wire
plan's explicit no-duplicate-pole-fitting rule (section 6.3), it reuses the
shared network/embedding rational-fitting stack:

* :func:`witwin.maxwell.rational.fit_rational` — the stable/passive vector fit,
* :meth:`RationalModel.to_state_space` — the same realization the embedded
  N-port networks use,
* :meth:`StateSpaceNetwork.discretize` — the bilinear (trapezoidal, passivity
  preserving) discretization,
* :meth:`RationalModel.check_passivity` — the pole-aware positive-real
  certificate.

The analytic impedance uses the exponentially scaled complex modified Bessel
function ``scipy.special.ive`` (where ``ive(n, z) = iv(n, z) * exp(-|Re z|)``),
in the same prepare-time spirit as the mode solver and the material dispersion
fits. Only the ratio ``I0(m a) / I1(m a)`` is needed, so the shared scaling
cancels exactly and the evaluation stays finite even when the unscaled ``iv``
would overflow at large ``Re(m a)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..rational import (
    DiscreteStateSpaceNetwork,
    FitReport,
    RationalFitConfig,
    RationalModel,
    StateSpaceNetwork,
    fit_rational,
)
from ..constants import MU_0


def _as_1d_frequencies(frequencies) -> torch.Tensor:
    tensor = (
        frequencies
        if isinstance(frequencies, torch.Tensor)
        else torch.as_tensor(frequencies, dtype=torch.float64)
    )
    if tensor.is_complex() or not tensor.dtype.is_floating_point:
        raise TypeError("frequencies must be real floating point.")
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if not bool(torch.all(torch.isfinite(tensor))) or not bool(torch.all(tensor > 0.0)):
        raise ValueError("frequencies must be finite and strictly positive.")
    return tensor


def _radius_vector(radius) -> tuple[torch.Tensor, bool]:
    """Return ``(radius[S], per_segment)`` as a positive double tensor."""

    tensor = (
        radius
        if isinstance(radius, torch.Tensor)
        else torch.as_tensor(radius, dtype=torch.float64)
    )
    if tensor.is_complex() or tensor.dtype == torch.bool or not tensor.dtype.is_floating_point:
        raise TypeError("radius must be real floating point.")
    tensor = tensor.to(dtype=torch.float64)
    if tensor.ndim == 0:
        return tensor.reshape(1), False
    if tensor.ndim == 1 and tensor.numel() >= 1:
        return tensor, True
    raise ValueError("radius must be a scalar or a shape [S] tensor.")


def _positive(value: float, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def dc_resistance(radius, conductivity) -> torch.Tensor:
    """Return the exact per-unit-length DC resistance ``1 / (pi a^2 sigma)``.

    The output preserves the input radius shape (scalar radius yields a scalar).
    """

    sigma = _positive(conductivity, "conductivity")
    radius_vector, per_segment = _radius_vector(radius)
    if not bool(torch.all(radius_vector > 0.0)):
        raise ValueError("radius must be positive.")
    resistance = 1.0 / (math.pi * radius_vector.square() * sigma)
    return resistance if per_segment else resistance.reshape(())


def surface_resistance(conductivity, permeability, frequencies) -> torch.Tensor:
    """Return the high-frequency surface resistance ``sqrt(omega mu / (2 sigma))``."""

    sigma = _positive(conductivity, "conductivity")
    mu = _positive(permeability, "permeability")
    omega = 2.0 * math.pi * _as_1d_frequencies(frequencies)
    return torch.sqrt(omega * mu / (2.0 * sigma))


def internal_impedance(
    radius,
    conductivity,
    permeability,
    frequencies,
) -> torch.Tensor:
    """Analytic per-unit-length internal impedance of a solid round wire.

    ``Z'(omega) = m / (2 pi a sigma) * I0(m a) / I1(m a)`` with
    ``m = sqrt(j omega mu sigma)``. Returns shape ``[F]`` for a scalar radius or
    ``[F, S]`` for a per-segment radius tensor. Complex dtype ``complex128``.

    The DC limit is ``R_dc`` and the high-frequency limit is
    ``(1 + j) R_s / (2 pi a)`` with ``R_s`` the surface resistance.
    """

    from scipy import special

    sigma = _positive(conductivity, "conductivity")
    mu = _positive(permeability, "permeability")
    freqs = _as_1d_frequencies(frequencies).to(dtype=torch.float64)
    radius_vector, per_segment = _radius_vector(radius)
    if not bool(torch.all(radius_vector > 0.0)):
        raise ValueError("radius must be positive.")

    omega = (2.0 * math.pi * freqs).numpy()
    a = radius_vector.numpy()
    # m = sqrt(j omega mu sigma), principal root (positive real part).
    m = (1j * omega * mu * sigma) ** 0.5  # [F]
    ma = m[:, None] * a[None, :]  # [F, S]
    # The impedance only needs the ratio I0(ma) / I1(ma). Evaluating the
    # unscaled iv overflows once Re(ma) grows large (both I0 and I1 blow up
    # past |Re| ~ 700), so use the exponentially scaled ive, where
    # ive(n, z) = iv(n, z) * exp(-|Re z|). Numerator and denominator share the
    # same argument ``ma``, so the exp(-|Re ma|) scaling cancels exactly and the
    # ratio equals iv(0, ma) / iv(1, ma) for every z, including the small-z DC
    # limit, while staying finite for arbitrarily large arguments.
    ratio = special.ive(0, ma) / special.ive(1, ma)
    prefactor = m[:, None] / (2.0 * math.pi * a[None, :] * sigma)
    impedance = prefactor * ratio
    result = torch.as_tensor(impedance, dtype=torch.complex128)
    if not bool(torch.all(torch.isfinite(result.real) & torch.isfinite(result.imag))):
        raise ValueError(
            "internal_impedance produced non-finite values; check radius/conductivity."
        )
    return result if per_segment else result.reshape(-1)


def internal_impedance_conductivity_gradient(
    radius,
    conductivity,
    permeability,
    frequencies,
) -> torch.Tensor:
    """Analytic derivative ``d Z'(omega) / d sigma`` of the internal impedance.

    Differentiating ``Z'(omega) = m / (2 pi a sigma) * I0(m a) / I1(m a)`` with
    ``m = sqrt(j omega mu sigma)`` in closed form collapses to

        d Z' / d sigma = (j omega mu) / (4 pi sigma) * (1 - R^2),   R = I0(m a) / I1(m a),

    using the Bessel recurrences ``I0'(z) = I1(z)`` and ``I1'(z) = I0(z) - I1(z)/z``
    together with ``d(m a)/d sigma = (m a) / (2 sigma)``. The bulky ``z R'(z) - R``
    term telescopes to ``z (1 - R^2)`` and ``m^2 = j omega mu sigma`` cancels the
    radius dependence, so the derivative needs only the same scaled-Bessel ratio
    ``R`` the impedance itself uses. Its DC limit is exactly ``d R_dc / d sigma =
    -1 / (pi a^2 sigma^2)`` (verified against the closed-form DC resistance), and
    ``Re`` of the result is the per-unit-length AC-resistance sensitivity that the
    reported ohmic dissipation differentiates through.

    Returns shape ``[F]`` for a scalar radius or ``[F, S]`` for a per-segment
    radius tensor, complex ``complex128`` (same layout as :func:`internal_impedance`).
    """

    from scipy import special

    sigma = _positive(conductivity, "conductivity")
    mu = _positive(permeability, "permeability")
    freqs = _as_1d_frequencies(frequencies).to(dtype=torch.float64)
    radius_vector, per_segment = _radius_vector(radius)
    if not bool(torch.all(radius_vector > 0.0)):
        raise ValueError("radius must be positive.")

    omega = (2.0 * math.pi * freqs).numpy()
    a = radius_vector.numpy()
    m = (1j * omega * mu * sigma) ** 0.5  # [F]
    ma = m[:, None] * a[None, :]  # [F, S]
    # Same exponentially scaled ratio as internal_impedance: ive(n, z) = iv(n, z)
    # * exp(-|Re z|), so the scaling cancels in the ratio and stays finite for
    # large Re(ma).
    ratio = special.ive(0, ma) / special.ive(1, ma)
    gradient = (1j * omega[:, None] * mu) / (4.0 * math.pi * sigma) * (1.0 - ratio * ratio)
    result = torch.as_tensor(gradient, dtype=torch.complex128)
    if not bool(torch.all(torch.isfinite(result.real) & torch.isfinite(result.imag))):
        raise ValueError(
            "internal_impedance_conductivity_gradient produced non-finite values; "
            "check radius/conductivity."
        )
    return result if per_segment else result.reshape(-1)


def ohmic_loss_density(
    current: torch.Tensor,
    resistance: torch.Tensor,
) -> torch.Tensor:
    """Time-averaged ohmic power per unit length for a current phasor.

    ``P = 0.5 * Re(Z') * |I|^2``. ``resistance`` is the real part of the
    per-unit-length series impedance ``Re(Z'(omega))`` (the AC resistance).
    """

    if not current.is_complex():
        raise TypeError("current must be a complex phasor tensor.")
    if resistance.is_complex() or not resistance.dtype.is_floating_point:
        raise TypeError("resistance must be a real AC-resistance tensor.")
    return 0.5 * resistance * current.abs().square()


@dataclass(frozen=True)
class SeriesImpedanceModel:
    """Passive rational model of a lossy wire's per-unit-length series impedance.

    ``resistance_dc`` is the exact analytic DC resistance. ``model`` fits the
    frequency-dependent *excess* impedance ``Z'(omega) - R_dc`` (positive-real,
    zero at DC), realized as ``state_space`` and discretized as ``discrete`` for
    time stepping. ``fit_report`` is the shared fitter's auditable report.
    """

    radius: float
    conductivity: float
    permeability: float
    band: tuple[float, float]
    resistance_dc: float
    model: RationalModel
    state_space: StateSpaceNetwork
    discrete: DiscreteStateSpaceNetwork | None
    fit_report: FitReport | None
    sample_frequencies: torch.Tensor

    @property
    def state_count(self) -> int:
        return self.state_space.state_count

    def evaluate(self, frequencies) -> torch.Tensor:
        """Return the fitted total series impedance ``R_dc + excess`` in ``[F]``."""

        freqs = _as_1d_frequencies(frequencies)
        excess = self.model.evaluate(freqs).reshape(freqs.numel(), -1)[:, 0]
        return excess + self.resistance_dc

    def ac_resistance(self, frequencies) -> torch.Tensor:
        """Return the fitted AC resistance ``Re(R_dc + excess)`` in ``[F]``."""

        return self.evaluate(frequencies).real

    def analytic_ac_resistance_error(self, frequencies) -> torch.Tensor:
        """Relative error of the fitted AC resistance vs. the analytic curve."""

        freqs = _as_1d_frequencies(frequencies)
        analytic = internal_impedance(
            self.radius, self.conductivity, self.permeability, freqs
        ).real
        fitted = self.ac_resistance(freqs).to(dtype=analytic.dtype)
        return (fitted - analytic).abs() / analytic.abs()


def fit_series_impedance(
    radius,
    conductivity,
    *,
    band: tuple[float, float],
    permeability: float = MU_0,
    order: int = 12,
    dt: float | None = None,
    samples: int = 160,
    relative_weighting: bool = True,
    relative_tolerance: float = 0.1,
    iterations: int = 18,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> SeriesImpedanceModel:
    """Fit a passive rational ADE for a lossy round wire's series impedance.

    Reuses the shared rational-fitting stack (:func:`fit_rational`,
    :meth:`RationalModel.to_state_space`, :meth:`StateSpaceNetwork.discretize`)
    on the analytic excess impedance ``Z'(omega) - R_dc``. The DC resistance is
    kept exact and applied separately, so the discrete model only has to capture
    the skin-effect excess (zero at DC, growing toward the surface-resistance
    asymptote).

    ``radius`` must be a scalar for this fit (a per-segment fit loops over unique
    radii). ``band`` is the fitting/validity band. When ``dt`` is given the
    continuous realization is discretized with the bilinear transform for time
    stepping; otherwise ``discrete`` is ``None``.
    """

    if not isinstance(band, (tuple, list)) or len(band) != 2:
        raise ValueError("band must be an increasing (f_min, f_max) pair.")
    f_min, f_max = float(band[0]), float(band[1])
    if not (0.0 < f_min < f_max) or not math.isfinite(f_max):
        raise ValueError("band must satisfy 0 < f_min < f_max.")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("order must be a positive integer.")
    if not isinstance(samples, int) or samples < order + 2:
        raise ValueError("samples must be an integer of at least order + 2.")

    radius_vector, per_segment = _radius_vector(radius)
    if per_segment and radius_vector.numel() != 1:
        raise ValueError(
            "fit_series_impedance fits one scalar radius; fit unique radii separately."
        )
    radius_value = float(radius_vector.reshape(()).item())
    sigma = _positive(conductivity, "conductivity")
    mu = _positive(permeability, "permeability")

    fit_frequencies = torch.logspace(
        math.log10(f_min), math.log10(f_max), samples, dtype=torch.float64
    )
    analytic = internal_impedance(radius_value, sigma, mu, fit_frequencies)
    resistance_dc = float(dc_resistance(radius_value, sigma).item())
    excess = analytic - resistance_dc

    config_weights = None
    if relative_weighting:
        config_weights = (1.0 / excess.abs().clamp_min(torch.finfo(torch.float64).tiny))
    config = RationalFitConfig(
        order=order,
        band=(f_min, f_max),
        iterations=iterations,
        proportional=False,
        relative_tolerance=relative_tolerance,
        weights=config_weights,
    )
    model = fit_rational(
        fit_frequencies,
        excess.to(dtype=torch.complex128),
        config,
        representation="Z",
    )

    state_space = model.to_state_space()
    discrete = None if dt is None else state_space.discretize(float(dt))

    target_dtype = dtype
    target_device = torch.device("cpu" if device is None else device)

    def _move_rational(value: RationalModel) -> RationalModel:
        complex_dtype = (
            torch.complex64 if target_dtype == torch.float32 else torch.complex128
        )
        return RationalModel(
            poles=value.poles.to(device=target_device, dtype=complex_dtype),
            residues=value.residues.to(device=target_device, dtype=complex_dtype),
            direct=value.direct.to(device=target_device, dtype=complex_dtype),
            proportional=value.proportional.to(
                device=target_device, dtype=complex_dtype
            ),
            representation=value.representation,
            report=value.report,
        )

    moved_model = _move_rational(model)
    moved_state = StateSpaceNetwork(
        A=state_space.A.to(device=target_device, dtype=target_dtype),
        B=state_space.B.to(device=target_device, dtype=target_dtype),
        C=state_space.C.to(device=target_device, dtype=target_dtype),
        D=state_space.D.to(device=target_device, dtype=target_dtype),
        representation=state_space.representation,
        port_order=state_space.port_order,
        passivity_margin=state_space.passivity_margin,
        report=state_space.report,
    )
    moved_discrete = None
    if discrete is not None:
        moved_discrete = DiscreteStateSpaceNetwork(
            A=discrete.A.to(device=target_device, dtype=target_dtype),
            B=discrete.B.to(device=target_device, dtype=target_dtype),
            C=discrete.C.to(device=target_device, dtype=target_dtype),
            D=discrete.D.to(device=target_device, dtype=target_dtype),
            dt=discrete.dt,
            representation=discrete.representation,
            port_order=discrete.port_order,
            pole_radius=discrete.pole_radius,
            passivity_margin=discrete.passivity_margin,
            report=discrete.report,
        )

    return SeriesImpedanceModel(
        radius=radius_value,
        conductivity=sigma,
        permeability=mu,
        band=(f_min, f_max),
        resistance_dc=resistance_dc,
        model=moved_model,
        state_space=moved_state,
        discrete=moved_discrete,
        fit_report=model.report,
        sample_frequencies=fit_frequencies.to(device=target_device),
    )


__all__ = [
    "SeriesImpedanceModel",
    "dc_resistance",
    "fit_series_impedance",
    "internal_impedance",
    "internal_impedance_conductivity_gradient",
    "ohmic_loss_density",
    "surface_resistance",
]
