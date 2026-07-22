"""Conductivity-adjoint gates (B3): analytic d Z'/d sigma of a lossy thin wire.

The finite-conductor recurrence coefficients come from a shared rational vector
fit that is nondeterministic (B1) and not a differentiable map of sigma, so the
field-coupled current sensitivity dI/dsigma cannot be certified by an exact
reverse replay and fails closed. The deterministic conductivity adjoint that DOES
ship is the closed-form sensitivity of the exact scaled-Bessel internal impedance
-- the dissipation channel d Re(Z'(f))/d sigma that the reported ohmic loss
differentiates through. These gates check that closed form against a float64
central difference of the analytic model (no fit involved) and exercise the
PyTorch-native autograd path.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.maxwell.compiler.wire_impedance import (
    dc_resistance,
    internal_impedance,
    internal_impedance_conductivity_gradient,
)
from witwin.maxwell.fdtd.wire_lossy import MU_0, analytic_ac_resistance

from tests.fdtd.thin_wire.test_wire_lossy_recurrence import _single_segment_model


_MU = MU_0
_CASES = (
    (1.0e-3, 5.8e7),   # copper, 1 mm
    (2.0e-3, 3.5e7),   # aluminium-ish, 2 mm
    (5.0e-4, 1.0e6),   # low-conductivity, sub-grid radius
)
_FREQS = (1.0e3, 1.0e6, 1.0e8, 1.5e9, 5.0e9, 2.0e10)


def _central_difference(fn, sigma, *, rel_step=1.0e-7):
    h = sigma * rel_step
    return (fn(sigma + h) - fn(sigma - h)) / (2.0 * h)


def test_internal_impedance_conductivity_gradient_matches_central_difference():
    """Gate: closed-form d Z'/d sigma == float64 central difference of Z'(sigma)."""

    freqs = torch.tensor(_FREQS, dtype=torch.float64)
    for radius, sigma in _CASES:
        closed = internal_impedance_conductivity_gradient(radius, sigma, _MU, freqs)

        def evaluate(value):
            return internal_impedance(radius, value, _MU, freqs)

        fd = _central_difference(evaluate, sigma)
        # Analytic vs central difference of the same analytic curve: tight.
        rel = (closed - fd).abs() / fd.abs().clamp_min(1.0e-300)
        assert float(rel.max()) < 1.0e-6, (radius, sigma, float(rel.max()))


def test_dc_resistance_conductivity_gradient_is_exact():
    """Gate (DC limit): Re(d Z'/d sigma) -> d R_dc/d sigma = -R_dc / sigma exactly."""

    low = torch.tensor([1.0], dtype=torch.float64)
    for radius, sigma in _CASES:
        rdc = float(dc_resistance(radius, sigma).item())
        expected = -rdc / sigma
        grad = float(
            internal_impedance_conductivity_gradient(radius, sigma, _MU, low).real.item()
        )
        assert abs(grad - expected) <= 1.0e-6 * abs(expected)


def test_analytic_ac_resistance_autograd_matches_central_difference():
    """Gate: PyTorch autograd through analytic_ac_resistance == central difference.

    A scalar conductivity leaf differentiates the summed AC resistance; the custom
    backward returns the closed-form sensitivity. Validated against a float64
    central difference of the same forward evaluation.
    """

    for radius, sigma in _CASES:
        sigma_leaf = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
        resistance = analytic_ac_resistance(
            sigma_leaf, radius=radius, permeability=_MU, frequencies=_FREQS
        )
        # Forward equals the analytic AC resistance.
        reference = internal_impedance(
            radius, sigma, _MU, torch.tensor(_FREQS, dtype=torch.float64)
        ).real
        torch.testing.assert_close(resistance.detach(), reference, rtol=0.0, atol=1.0e-18)

        weights = torch.linspace(0.3, 1.7, len(_FREQS), dtype=torch.float64)
        objective = torch.sum(weights * resistance)
        (grad,) = torch.autograd.grad(objective, sigma_leaf)

        def evaluate(value):
            r = internal_impedance(
                radius, value, _MU, torch.tensor(_FREQS, dtype=torch.float64)
            ).real
            return float(torch.sum(weights * r))

        fd = _central_difference(evaluate, sigma)
        assert abs(float(grad) - fd) <= 1.0e-6 * abs(fd), (radius, sigma)


def test_ohmic_loss_conductivity_gradient_matches_central_difference():
    """Gate: d(ohmic_loss)/d sigma of 0.5 Re(Z') L |I|^2 (current held fixed).

    This is the reported dissipation channel's conductivity adjoint. The realized
    current |I| is a fixed recurrence output with respect to which sigma has no
    differentiable path (dI/dsigma is the fail-closed field-coupled channel), so
    the objective differentiates sigma only through Re(Z'(sigma)).
    """

    radius, sigma = 1.0e-3, 5.8e7
    length = 0.02
    current_abs2 = torch.tensor(
        [4.0, 2.5, 1.0, 0.6, 0.25, 0.1], dtype=torch.float64
    )  # |I(f)|^2 per frequency, treated as constant

    def loss_from_resistance(resistance):
        return 0.5 * torch.sum(resistance * length * current_abs2)

    sigma_leaf = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
    resistance = analytic_ac_resistance(
        sigma_leaf, radius=radius, permeability=_MU, frequencies=_FREQS
    )
    objective = loss_from_resistance(resistance)
    (grad,) = torch.autograd.grad(objective, sigma_leaf)

    def evaluate(value):
        r = internal_impedance(
            radius, value, _MU, torch.tensor(_FREQS, dtype=torch.float64)
        ).real
        return float(loss_from_resistance(r))

    fd = _central_difference(evaluate, sigma)
    assert abs(float(grad) - fd) <= 1.0e-6 * abs(fd)


def test_lossy_segment_model_conductivity_gradient_matches_analytic():
    """The built model's per-segment d Re(Z')/d sigma equals the analytic closed form."""

    model = _single_segment_model()
    freqs = torch.tensor([5.0e8, 1.0e9, 2.5e9], dtype=torch.float64)
    grad = model.conductivity_ac_resistance_gradient(freqs)
    assert grad.shape == (3, 1)
    expected = internal_impedance_conductivity_gradient(
        model.models[0].radius,
        model.models[0].conductivity,
        model.models[0].permeability,
        freqs,
    ).real
    torch.testing.assert_close(grad[:, 0], expected.to(grad.dtype), rtol=1.0e-12, atol=0.0)


def test_pec_segment_conductivity_gradient_is_zero():
    """A PEC conductor carries no lossy model, hence no conductivity sensitivity."""

    model = _single_segment_model(kind="pec")
    assert model is None


def test_analytic_ac_resistance_requires_scalar_conductivity():
    with pytest.raises(ValueError, match="scalar tensor"):
        analytic_ac_resistance(
            torch.tensor([5.8e7, 1.0e6], dtype=torch.float64),
            radius=1.0e-3,
            permeability=_MU,
            frequencies=_FREQS,
        )
