"""Phase 0 rational surface-impedance model and shared passive fitter (S0.2).

Covers the public model layer (``SurfaceImpedanceModel`` /
``RationalSurfaceImpedance`` / ``SurfaceImpedanceMedium``) and the compiler-layer
fitter (``compiler/surface_impedance.py``), which reuse the shared rational stack
(``fit_rational`` + ``to_state_space`` + ``discretize`` + ``check_passivity``). No
runtime path is exercised: this slice adds the contract, not the kernel.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.surface_impedance import (
    SurfaceImpedanceRationalModel,
    fit_surface_impedance,
)
from witwin.maxwell.compiler.wire_impedance import (
    fit_series_impedance,
    internal_impedance,
)
from witwin.maxwell.fdtd.surface_impedance_reference import (
    SURFACE_ACCEPTANCE_BUDGET as BUDGET,
    good_conductor_surface_impedance,
)
from witwin.maxwell.media import (
    RationalSurfaceImpedance,
    SurfaceImpedanceMedium,
    SurfaceImpedanceModel,
)
from witwin.maxwell.rational import (
    DiscreteStateSpaceNetwork,
    RationalFitConfig,
    RationalModel,
    StateSpaceNetwork,
)

_COPPER = 5.8e7
_MU0 = 1.25663706212e-6
_BAND = (1.0e9, 40.0e9)


def _good_conductor_admittance(frequencies):
    return (1.0 / good_conductor_surface_impedance(_COPPER, frequencies)).to(torch.complex128)


# --------------------------------------------------------------------------- #
# RationalSurfaceImpedance: narrowband order-1 R-L and broadband good-conductor fit
# --------------------------------------------------------------------------- #


def test_narrowband_rl_order_one_reproduces_leontovich():
    """The narrowband series R-L is the order-1 admittance special case (R2 tie-in)."""
    sigma, f0 = 50.0, 2.0e9
    omega0 = 2.0 * math.pi * f0
    r = math.sqrt(omega0 * _MU0 / (2.0 * sigma))
    inductance = r / omega0  # L_s = R / omega0
    # Y_s(s) = (1/L) / (s + omega0), a single stable pole (Re < 0) that is positive-real.
    model = RationalSurfaceImpedance(
        poles=torch.tensor([-omega0], dtype=torch.complex128),
        residues=torch.tensor([[[1.0 / inductance]]], dtype=torch.complex128),
        frequency_range=(1.0e8, 1.0e10),
        representation="Y",
    )
    z_s = complex(model.surface_impedance([f0])[0])
    assert z_s.real == pytest.approx(r, rel=1e-9)
    assert z_s.imag == pytest.approx(-r, rel=1e-9)
    assert model.port_count == 1
    assert model.passivity.passive and model.passivity.certified
    assert isinstance(model, SurfaceImpedanceModel)


@pytest.mark.parametrize("representation", ["Y", "Z"])
def test_broadband_good_conductor_fit_meets_budget(representation):
    frequencies = torch.logspace(
        math.log10(_BAND[0]), math.log10(_BAND[1]), 200, dtype=torch.float64
    )
    if representation == "Y":
        target = _good_conductor_admittance(frequencies)
    else:
        target = good_conductor_surface_impedance(_COPPER, frequencies).to(torch.complex128)
    model = RationalSurfaceImpedance.fit(
        frequencies, target, order=10, band=_BAND, representation=representation
    )
    assert model.fit_report.relative_max_error <= BUDGET.fit_max_complex_error
    assert model.passivity.passive and model.passivity.certified
    assert model.passivity.margin >= BUDGET.fit_min_passivity_margin
    # Z_s always returns the impedance regardless of the internal representation.
    z_fit = model.surface_impedance(frequencies)
    z_ref = good_conductor_surface_impedance(_COPPER, frequencies)
    rel = float(((z_fit - z_ref).abs() / z_ref.abs()).max())
    assert rel <= BUDGET.fit_max_complex_error


def test_fit_report_is_inspectable():
    frequencies = torch.logspace(math.log10(_BAND[0]), math.log10(_BAND[1]), 128, dtype=torch.float64)
    model = RationalSurfaceImpedance.fit(frequencies, _good_conductor_admittance(frequencies), order=8, band=_BAND)
    report = model.fit_report
    assert report.order == 8
    assert report.differentiable_parameters == ("residues", "direct")
    assert model.sample_frequencies.numel() == 128


# --------------------------------------------------------------------------- #
# RationalSurfaceImpedance: fail-closed validation
# --------------------------------------------------------------------------- #


def test_non_passive_model_is_rejected():
    """Falsification: a stable but non-passive (Re(Y) < 0) surface must fail closed."""
    with pytest.raises(ValueError, match="passive"):
        RationalSurfaceImpedance(
            poles=torch.tensor([-1.0e10], dtype=torch.complex128),
            residues=torch.tensor([[[-1.0e12]]], dtype=torch.complex128),  # Re(Y) << 0 in band
            frequency_range=_BAND,
            representation="Y",
        )


def test_non_square_model_is_rejected():
    with pytest.raises(ValueError, match="square"):
        RationalSurfaceImpedance(
            poles=torch.tensor([-1.0e10], dtype=torch.complex128),
            residues=torch.ones((2, 1, 1), dtype=torch.complex128),
            frequency_range=_BAND,
            representation="Y",
        )


def test_higher_than_tangential_rank_is_rejected():
    residues = torch.eye(3, dtype=torch.complex128).reshape(3, 3, 1)
    with pytest.raises(ValueError, match="scalar .1x1. or tangential"):
        RationalSurfaceImpedance(
            poles=torch.tensor([-1.0e10], dtype=torch.complex128),
            residues=residues,
            frequency_range=_BAND,
            representation="Y",
        )


def test_bad_frequency_range_and_representation_are_rejected():
    good_poles = torch.tensor([-1.0e10], dtype=torch.complex128)
    good_res = torch.tensor([[[1.0e9]]], dtype=torch.complex128)
    with pytest.raises(ValueError, match="f_min < f_max"):
        RationalSurfaceImpedance(good_poles, good_res, frequency_range=(4.0e10, 1.0e9))
    with pytest.raises(ValueError, match="'Y' or 'Z'"):
        RationalSurfaceImpedance(good_poles, good_res, frequency_range=_BAND, representation="S")


def test_representation_kwarg_contradicting_passed_model_is_rejected():
    """Fail closed: a representation kwarg that contradicts a passed RationalModel's own
    representation must raise, not silently reinterpret Z-samples as admittance."""
    model = RationalModel(
        poles=torch.tensor([-1.0e10], dtype=torch.complex128),
        residues=torch.tensor([[[1.0e9]]], dtype=torch.complex128),
        representation="Y",
    )
    with pytest.raises(ValueError, match="contradicts"):
        RationalSurfaceImpedance(model, None, frequency_range=_BAND, representation="Z")
    # Omitting the kwarg (or passing the matching one) keeps the model's representation.
    assert RationalSurfaceImpedance(model, None, frequency_range=_BAND).representation == "Y"
    assert (
        RationalSurfaceImpedance(model, None, frequency_range=_BAND, representation="Y").representation
        == "Y"
    )


# --------------------------------------------------------------------------- #
# SurfaceImpedanceMedium
# --------------------------------------------------------------------------- #


def test_surface_impedance_medium_wraps_model():
    frequencies = torch.logspace(math.log10(_BAND[0]), math.log10(_BAND[1]), 96, dtype=torch.float64)
    model = RationalSurfaceImpedance.fit(frequencies, _good_conductor_admittance(frequencies), order=8, band=_BAND)
    medium = SurfaceImpedanceMedium(impedance=model, name="coating")
    assert medium.is_surface_impedance is True
    assert medium.frequency_range == _BAND
    assert bool(getattr(medium, "is_lossy_metal", False)) is False
    with pytest.raises(TypeError):
        SurfaceImpedanceMedium(impedance=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# compiler/surface_impedance.py fitter
# --------------------------------------------------------------------------- #


def test_fitter_reuses_shared_stack_and_is_stable_passive():
    model = fit_surface_impedance(
        lambda f: (1.0 / good_conductor_surface_impedance(_COPPER, f)).to(torch.complex128),
        band=_BAND,
        order=10,
        representation="Y",
        dt=1.0e-13,
    )
    assert isinstance(model, SurfaceImpedanceRationalModel)
    assert isinstance(model.model, RationalModel)
    assert isinstance(model.state_space, StateSpaceNetwork)
    assert isinstance(model.discrete, DiscreteStateSpaceNetwork)
    assert model.fit_report.relative_max_error <= BUDGET.fit_max_complex_error
    assert model.passivity.passive and model.passivity.certified
    assert model.pole_radius < BUDGET.discrete_pole_radius_max
    assert model.state_count > 0


def test_fitter_rejects_active_non_passive_target():
    """Falsification: a negative-conductance (active) surface must fail the exit gate."""
    with pytest.raises(ValueError, match="passive"):
        fit_surface_impedance(
            lambda f: -torch.ones(f.numel(), dtype=torch.complex128),
            band=_BAND,
            order=2,
            representation="Y",
        )


def test_fitter_matches_wire_impedance_excess_fit_scalar_z_form():
    """Shared-fitter parity: the wire excess-Z fit is the scalar Z-form special case."""
    band, order = (4.0e8, 3.0e9), 10
    wire = fit_series_impedance(
        5.0e-4, _COPPER, band=band, order=order, dt=8.0e-13, samples=240, iterations=20
    )
    frequencies = wire.sample_frequencies
    excess = internal_impedance(5.0e-4, _COPPER, _MU0, frequencies) - wire.resistance_dc
    weights = 1.0 / excess.abs().clamp_min(torch.finfo(torch.float64).tiny)
    config = RationalFitConfig(
        order=order, band=band, iterations=20, proportional=False,
        relative_tolerance=0.1, weights=weights,
    )
    surface = fit_surface_impedance(
        (frequencies, excess), band=band, order=order, representation="Z",
        config=config, dt=8.0e-13,
    )
    # Vector fitting is not bit-reproducible in its pole placement, but the two callers
    # of the one shared fitter must agree on the fitted transfer function.
    wire_response = wire.model.evaluate(frequencies).reshape(frequencies.numel(), -1)[:, 0]
    surface_response = surface.model.evaluate(frequencies).reshape(frequencies.numel(), -1)[:, 0]
    rel = float(((surface_response - wire_response).abs() / excess.abs()).max())
    assert rel < 1.0e-4
    assert isinstance(surface.discrete, DiscreteStateSpaceNetwork)
