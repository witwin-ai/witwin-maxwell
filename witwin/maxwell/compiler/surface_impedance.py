"""Shared passive rational fitter for surface impedance/admittance boundaries.

This is the compiler-layer surface fitter for the generalized surface-impedance
subsystem. It represents a broadband tangential surface response (a good conductor's
``Z_s(omega)``, a roughness-corrected conductor, or user frequency samples) as a
*passive* rational auxiliary-differential-equation (ADE) model discretized for time
stepping.

Like :mod:`witwin.maxwell.compiler.wire_impedance`, it does **not** implement its own
pole fitting. It reuses the shared network/embedding rational-fitting stack:

* :func:`witwin.maxwell.rational.fit_rational` -- the stable/passive vector fit,
* :meth:`RationalModel.to_state_space` -- the same realization the embedded networks
  and the thin-wire series impedance use,
* :meth:`StateSpaceNetwork.discretize` -- the bilinear (trapezoidal, passivity- and
  stability-preserving, ``|z| < 1``) discretization used as the surface update, an
  implicitly A-stable recurrence rather than an explicit stiff ``R + L/dt`` term,
* :meth:`RationalModel.check_passivity` -- the pole-aware positive-real certificate,
  applied here as a compile **exit gate** (a fit that is accurate but non-passive is
  rejected, never run).

The thin-wire series-impedance fit is the scalar ``Z``-form special case of this
contract: fitting the same excess-impedance samples through
:func:`fit_surface_impedance` with ``representation="Z"`` and the same order/config
reproduces the wire fit. Phase 2 adds the tangential 2x2 case.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from ..rational import (
    DiscreteStateSpaceNetwork,
    FitReport,
    PassivityReport,
    RationalFitConfig,
    RationalModel,
    StateSpaceNetwork,
    fit_rational,
)


def _as_band(band) -> tuple[float, float]:
    if not isinstance(band, (tuple, list)) or len(band) != 2:
        raise ValueError("band must be an increasing (f_min, f_max) pair.")
    f_min, f_max = float(band[0]), float(band[1])
    if not (0.0 < f_min < f_max) or not math.isfinite(f_max):
        raise ValueError("band must satisfy 0 < f_min < f_max.")
    return (f_min, f_max)


def _resolve_samples(
    target: Callable | tuple | list,
    band: tuple[float, float],
    samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(frequencies[F], values[F, ...])`` from a callable or explicit samples."""

    if callable(target):
        if not isinstance(samples, int) or samples < 2:
            raise ValueError("samples must be an integer of at least 2.")
        frequencies = torch.logspace(
            math.log10(band[0]), math.log10(band[1]), samples, dtype=torch.float64
        )
        values = torch.as_tensor(target(frequencies))
    else:
        if not isinstance(target, (tuple, list)) or len(target) != 2:
            raise ValueError(
                "target must be a callable or an (frequencies, values) pair."
            )
        frequencies = torch.as_tensor(target[0], dtype=torch.float64)
        values = torch.as_tensor(target[1])
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if frequencies.is_complex() or not frequencies.dtype.is_floating_point:
        raise TypeError("frequencies must be real floating point.")
    if not bool(torch.all(torch.isfinite(frequencies))) or not bool(
        torch.all(frequencies > 0.0)
    ):
        raise ValueError("frequencies must be finite and strictly positive.")
    values = values.to(dtype=torch.complex128)
    if values.shape[0] != frequencies.numel():
        raise ValueError("values must have a leading dimension matching frequencies.")
    return frequencies, values


@dataclass(frozen=True)
class SurfaceImpedanceRationalModel:
    """Compiled passive rational surface model with a discretized ADE realization.

    ``model`` is the shared rational realization of the fitted ``representation``
    (``"Y"`` admittance or ``"Z"`` impedance). ``state_space`` is its continuous
    A/B/C/D realization and ``discrete`` its bilinear time-stepping form (``None`` when
    no ``dt`` was given). ``fit_report`` is the shared fitter's auditable report and
    ``passivity`` the certified positive-real report over the fitting band.
    """

    model: RationalModel
    state_space: StateSpaceNetwork
    discrete: DiscreteStateSpaceNetwork | None
    fit_report: FitReport | None
    passivity: PassivityReport
    sample_frequencies: torch.Tensor
    representation: str
    band: tuple[float, float]

    @property
    def state_count(self) -> int:
        return self.state_space.state_count

    @property
    def port_count(self) -> int:
        return int(self.model.output_count)

    @property
    def pole_radius(self) -> float | None:
        return None if self.discrete is None else self.discrete.pole_radius

    def evaluate(self, frequencies) -> torch.Tensor:
        """Return the represented transfer (``Y`` or ``Z``), shape ``[F, P, P]``."""

        return self.model.evaluate(frequencies)

    def surface_impedance(self, frequencies) -> torch.Tensor:
        """Return ``Z_s(omega)`` in ohms, shape ``[F]`` (scalar) or ``[F, P, P]``."""

        response = self.model.evaluate(frequencies)
        if self.representation == "Z":
            impedance = response
        elif self.port_count == 1:
            impedance = 1.0 / response
        else:
            impedance = torch.linalg.inv(response)
        if self.port_count == 1:
            return impedance.reshape(impedance.shape[0])
        return impedance


def _move_rational(model: RationalModel, device, complex_dtype) -> RationalModel:
    return RationalModel(
        poles=model.poles.to(device=device, dtype=complex_dtype),
        residues=model.residues.to(device=device, dtype=complex_dtype),
        direct=model.direct.to(device=device, dtype=complex_dtype),
        proportional=model.proportional.to(device=device, dtype=complex_dtype),
        representation=model.representation,
        report=model.report,
    )


def _move_state_space(state: StateSpaceNetwork, device, dtype) -> StateSpaceNetwork:
    return StateSpaceNetwork(
        A=state.A.to(device=device, dtype=dtype),
        B=state.B.to(device=device, dtype=dtype),
        C=state.C.to(device=device, dtype=dtype),
        D=state.D.to(device=device, dtype=dtype),
        representation=state.representation,
        port_order=state.port_order,
        passivity_margin=state.passivity_margin,
        report=state.report,
    )


def _move_discrete(
    discrete: DiscreteStateSpaceNetwork, device, dtype
) -> DiscreteStateSpaceNetwork:
    return DiscreteStateSpaceNetwork(
        A=discrete.A.to(device=device, dtype=dtype),
        B=discrete.B.to(device=device, dtype=dtype),
        C=discrete.C.to(device=device, dtype=dtype),
        D=discrete.D.to(device=device, dtype=dtype),
        dt=discrete.dt,
        representation=discrete.representation,
        port_order=discrete.port_order,
        pole_radius=discrete.pole_radius,
        passivity_margin=discrete.passivity_margin,
        report=discrete.report,
    )


def fit_surface_impedance(
    target,
    *,
    band,
    order: int,
    dt: float | None = None,
    device: str | torch.device | None = None,
    representation: str = "Y",
    samples: int = 200,
    iterations: int = 20,
    relative_tolerance: float = 1.0e-3,
    relative_weighting: bool = True,
    passivity_tolerance: float = 1.0e-9,
    config: RationalFitConfig | None = None,
    dtype: torch.dtype = torch.float64,
) -> SurfaceImpedanceRationalModel:
    """Fit a passive rational surface impedance/admittance and discretize it.

    ``target`` is either a callable ``f(frequencies) -> values`` (sampled on a
    log-spaced grid over ``band``) or an explicit ``(frequencies, values)`` pair. The
    fit represents ``representation`` (``"Y"`` fits the admittance ``Y_s``, ``"Z"``
    fits the impedance ``Z_s``). Passivity is a compile exit gate: a fit that is
    accurate but not certified passive over ``band`` is rejected.

    When ``dt`` is given, the continuous realization is discretized with the bilinear
    transform for time stepping (``|z| < 1`` guaranteed); otherwise ``discrete`` is
    ``None``. A ``config`` overrides the sampling weights/order/iterations, which makes
    the thin-wire scalar ``Z``-form fit an exact special case of this contract.
    """

    band = _as_band(band)
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("order must be a positive integer.")
    frequencies, values = _resolve_samples(target, band, samples)

    if config is None:
        weights = None
        if relative_weighting:
            flat = values.reshape(frequencies.numel(), -1)
            magnitude = flat.abs().amax(dim=1)
            weights = 1.0 / magnitude.clamp_min(torch.finfo(torch.float64).tiny)
        config = RationalFitConfig(
            order=int(order),
            band=band,
            iterations=int(iterations),
            proportional=False,
            relative_tolerance=float(relative_tolerance),
            passivity_tolerance=float(passivity_tolerance),
            weights=weights,
        )

    model = fit_rational(frequencies, values, config, representation=representation)
    if model.output_count != model.input_count:
        raise ValueError("A surface impedance transfer matrix must be square.")
    if model.output_count not in (1, 2):
        raise ValueError(
            "A surface impedance model must be scalar (1x1) or tangential (2x2)."
        )

    endpoints = torch.tensor(band, dtype=torch.float64, device=model.poles.device)
    certificate = torch.unique(
        torch.cat((frequencies.to(device=model.poles.device), endpoints)), sorted=True
    )
    passivity = model.check_passivity(certificate, tolerance=float(passivity_tolerance))
    if not passivity.passive or not passivity.certified:
        raise ValueError(
            "A surface impedance fit must be certified passive over its band; "
            f"maximum violation is {passivity.max_violation:.6g}, "
            f"certificate={passivity.certified}."
        )

    state_space = model.to_state_space()
    discrete = None if dt is None else state_space.discretize(float(dt))

    target_device = torch.device("cpu" if device is None else device)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return SurfaceImpedanceRationalModel(
        model=_move_rational(model, target_device, complex_dtype),
        state_space=_move_state_space(state_space, target_device, dtype),
        discrete=None if discrete is None else _move_discrete(discrete, target_device, dtype),
        fit_report=model.report,
        passivity=passivity,
        sample_frequencies=frequencies.to(device=target_device),
        representation=model.representation,
        band=band,
    )


__all__ = [
    "SurfaceImpedanceRationalModel",
    "fit_surface_impedance",
]
