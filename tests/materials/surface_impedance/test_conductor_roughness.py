"""Analytic and fitted checks of the opt-in conductor surface-roughness models."""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.surface_impedance import fit_surface_impedance

_MU0 = 4.0e-7 * math.pi
_SIGMA = 5.8e7
_BAND = (1.0e9, 40.0e9)


def _delta(frequency, sigma=_SIGMA):
    return math.sqrt(2.0 / (2.0 * math.pi * frequency * _MU0 * sigma))


def _classic_hammerstad(ratio, factor=2.0):
    return 1.0 + (factor - 1.0) * (2.0 / math.pi) * torch.atan(1.4 * ratio**2)


def _classic_huray(delta, radius, surface_ratio, relative_area=1.0):
    return relative_area + 1.5 * surface_ratio / (1.0 + delta / radius + delta**2 / (2.0 * radius**2))


_ROUGH_MODELS = [
    mw.HammerstadRoughness(rms_height=1.0e-6),
    mw.HurayRoughness.cannonball(0.5e-6),
    mw.HurayRoughness(sphere_radius=0.4e-6, sphere_density=1.5e11, relative_area=1.3),
]


def test_exports_and_validation():
    assert issubclass(mw.HammerstadRoughness, mw.SurfaceRoughness)
    assert issubclass(mw.HurayRoughness, mw.SurfaceRoughness)
    assert issubclass(mw.GroisseRoughness, mw.SurfaceRoughness)
    with pytest.raises(ValueError):
        mw.HammerstadRoughness(rms_height=0.0)
    with pytest.raises(ValueError):
        mw.HammerstadRoughness(rms_height=1e-6, roughness_factor=1.0)
    with pytest.raises(ValueError):
        mw.HurayRoughness(sphere_radius=1e-6, sphere_density=-1.0)
    with pytest.raises(ValueError):
        mw.GroisseRoughness(rms_height=float("nan"))
    with pytest.raises(TypeError):
        mw.LossyMetalMedium(conductivity=_SIGMA, roughness=2.0)
    with pytest.raises(ValueError):
        mw.HammerstadRoughness(rms_height=1e-6).impedance_factor(torch.tensor([0.0], dtype=torch.float64))
    model = mw.HammerstadRoughness(rms_height=1e-6)
    with pytest.raises(Exception):
        model.rms_height = 2e-6  # frozen
    assert mw.LossyMetalMedium(conductivity=_SIGMA).roughness is None


def test_hammerstad_loss_factor_is_exactly_the_classic_formula():
    model = mw.HammerstadRoughness(rms_height=1.0e-6)
    ratio = torch.logspace(-3, 3, 241, dtype=torch.float64)
    delta = model.rms_height / ratio
    torch.testing.assert_close(model.loss_factor(delta), _classic_hammerstad(ratio), rtol=1e-12, atol=0)
    triple = mw.HammerstadRoughness(rms_height=1.0e-6, roughness_factor=3.0)
    torch.testing.assert_close(triple.loss_factor(delta), _classic_hammerstad(ratio, 3.0), rtol=1e-12, atol=0)


def test_hammerstad_limits():
    model = mw.HammerstadRoughness(rms_height=1.0e-6)
    # Rq/delta -> 0: smooth conductor.
    small = model.impedance_factor(torch.tensor([1.0e-2], dtype=torch.float64))  # delta = 1 cm >> Rq
    assert abs(complex(small[0]) - 1.0) < 1e-3
    tiny = model.impedance_factor(torch.tensor([10.0], dtype=torch.float64))
    assert abs(complex(tiny[0]) - 1.0) < 1e-6
    # Rq/delta -> inf: saturates at the roughness factor (2 is classic Hammerstad).
    high = model.impedance_factor(torch.tensor([1.0e-10], dtype=torch.float64))
    assert abs(complex(high[0]) - 2.0) < 1e-4
    assert float(model.loss_factor(torch.tensor([1.0e-10], dtype=torch.float64))[0]) == pytest.approx(2.0, abs=1e-6)
    # Monotone increasing loss with frequency (decreasing skin depth).
    delta = torch.logspace(-4, -9, 200, dtype=torch.float64)
    loss = model.loss_factor(delta)
    assert bool(torch.all(loss[1:] >= loss[:-1]))
    assert bool(torch.all((loss >= 1.0) & (loss <= 2.0)))


def test_huray_matches_hand_computed_value():
    radius = 0.5e-6
    model = mw.HurayRoughness.cannonball(radius)
    assert model.surface_ratio == pytest.approx(14.0 * math.pi / 9.0, rel=1e-14)
    # At delta == r: denominator 1 + (1 + i)/2, so K = 1 + f (0.9 - 0.3 i) and the loss
    # ratio is 1 + f / (1 + 1 + 1/2) * 3/2 = 1 + 0.6 f = 1 + 14 pi / 15.
    factor = complex(model.impedance_factor(torch.tensor([radius], dtype=torch.float64))[0])
    f = 14.0 * math.pi / 9.0
    assert factor.real == pytest.approx(1.0 + 0.9 * f, rel=1e-14)
    assert factor.imag == pytest.approx(-0.3 * f, rel=1e-14)
    assert float(model.loss_factor(torch.tensor([radius], dtype=torch.float64))[0]) == pytest.approx(1.0 + 14.0 * math.pi / 15.0, rel=1e-14)


def test_huray_loss_factor_is_exactly_the_classic_formula_and_limits():
    model = mw.HurayRoughness(sphere_radius=0.4e-6, sphere_density=1.5e11, relative_area=1.3)
    assert model.surface_ratio == pytest.approx(1.5e11 * 4.0 * math.pi * 0.16e-12)
    same = mw.HurayRoughness.from_surface_ratio(0.4e-6, model.surface_ratio, relative_area=1.3)
    assert same.sphere_density == pytest.approx(model.sphere_density, rel=1e-14)
    delta = torch.logspace(-9, -3, 241, dtype=torch.float64)
    expected = _classic_huray(delta, 0.4e-6, model.surface_ratio, 1.3)
    torch.testing.assert_close(model.loss_factor(delta), expected, rtol=1e-12, atol=0)
    # Vanishing spheres (r/delta -> 0) leave only the matte-area ratio; at r/delta -> inf
    # the loss saturates at relative_area + 3/2 f.
    low = complex(model.impedance_factor(torch.tensor([1.0], dtype=torch.float64))[0])
    assert abs(low - 1.3) < 1e-5
    high = complex(model.impedance_factor(torch.tensor([1.0e-14], dtype=torch.float64))[0])
    assert abs(high - (1.3 + 1.5 * model.surface_ratio)) < 1e-6
    smooth_base = mw.HurayRoughness(sphere_radius=1.0e-12, sphere_density=1.0e20)
    assert abs(complex(smooth_base.impedance_factor(torch.tensor([1.0e-3], dtype=torch.float64))[0]) - 1.0) < 1e-7


def test_groisse_formula_and_limits():
    model = mw.GroisseRoughness(rms_height=1.0e-6)
    assert not model.causal
    at = complex(model.impedance_factor(torch.tensor([2.0e-6], dtype=torch.float64))[0])
    assert at == pytest.approx(1.0 + math.exp(-1.0), rel=1e-14)
    assert float(model.loss_factor(torch.tensor([2.0e-6], dtype=torch.float64))[0]) == pytest.approx(1.0 + math.exp(-1.0), rel=1e-14)
    assert abs(complex(model.impedance_factor(torch.tensor([1.0e-2], dtype=torch.float64))[0]) - 1.0) < 1e-12
    assert abs(complex(model.impedance_factor(torch.tensor([1.0e-12], dtype=torch.float64))[0]) - 2.0) < 1e-6


def test_causal_factors_add_internal_reactance():
    # A causal complex correction must add internal reactance as well as resistance.
    for model in (mw.HammerstadRoughness(rms_height=1e-6), mw.HurayRoughness.cannonball(0.5e-6)):
        assert model.causal
        factor = model.impedance_factor(torch.tensor([1.0e-6], dtype=torch.float64))
        assert float(factor.imag[0]) < 0.0


@pytest.mark.parametrize("roughness", _ROUGH_MODELS + [mw.GroisseRoughness(rms_height=1e-6)])
def test_lossy_metal_applies_roughness_to_complex_impedance(roughness):
    smooth = mw.LossyMetalMedium(conductivity=_SIGMA)
    rough = mw.LossyMetalMedium(conductivity=_SIGMA, roughness=roughness)
    frequencies = torch.logspace(8, 11, 13, dtype=torch.float64)
    for frequency in frequencies.tolist():
        z_smooth = smooth.surface_impedance_at_freq(frequency)
        factor = complex(roughness.impedance_factor(torch.tensor([_delta(frequency)], dtype=torch.float64))[0])
        z_rough = rough.surface_impedance_at_freq(frequency)
        assert z_rough == pytest.approx(factor * z_smooth, rel=1e-12)
        assert z_rough.real > 0.0
        assert rough.roughness_loss_factor(frequency) == pytest.approx(z_rough.real / z_smooth.real, rel=1e-12)
        assert smooth.roughness_loss_factor(frequency) == 1.0
    samples = rough.surface_impedance_samples(frequencies)
    expected = torch.tensor([rough.surface_impedance_at_freq(f) for f in frequencies.tolist()], dtype=torch.complex128)
    torch.testing.assert_close(samples, expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(
        smooth.surface_impedance_samples(frequencies),
        torch.tensor([smooth.surface_impedance_at_freq(f) for f in frequencies.tolist()], dtype=torch.complex128),
        rtol=1e-14, atol=0,
    )


# Rough models inside the validated broadband envelope (refit direct-term ratio <= 3.1).
_BROADBAND_MODELS = [
    mw.HammerstadRoughness(rms_height=1.0e-6),
    mw.HammerstadRoughness(rms_height=1.0e-6, roughness_factor=3.0),
    mw.HurayRoughness(sphere_radius=0.4e-6, sphere_density=1.5e11, relative_area=1.3),
]


@pytest.mark.parametrize("roughness", [None] + _BROADBAND_MODELS)
@pytest.mark.parametrize("representation", ["Y", "Z"])
def test_rational_fit_matches_analytic_rough_impedance_and_is_passive(roughness, representation):
    metal = mw.LossyMetalMedium(conductivity=_SIGMA, roughness=roughness)
    model = metal.rational_surface_impedance(_BAND, order=6, representation=representation)
    assert model.passivity.passive and model.passivity.certified
    dense = torch.logspace(math.log10(_BAND[0]), math.log10(_BAND[1]), 1601, dtype=torch.float64)
    target = metal.surface_impedance_samples(dense)
    fitted = model.surface_impedance(dense)
    relative = ((fitted - target).abs() / target.abs()).max()
    assert float(relative) < 2.0e-3
    # Positive surface resistance throughout the band (passive surface).
    assert bool(torch.all(fitted.real > 0.0))
    # The time-stepping runtime refits the model in Z form and discretizes it; that
    # realization must stay certified passive, stable and band-accurate as well.
    samples = model.sample_frequencies
    refit = fit_surface_impedance(
        (samples, model.surface_impedance(samples)), band=_BAND, order=6, dt=1.0e-13, representation="Z"
    )
    assert refit.passivity.passive and refit.passivity.certified
    assert refit.pole_radius < 1.0
    refit_relative = ((refit.surface_impedance(dense) - target).abs() / target.abs()).max()
    assert float(refit_relative) < 2.0e-3


def test_rough_fit_carries_the_roughness_loss():
    smooth = mw.LossyMetalMedium(conductivity=_SIGMA)
    roughness = mw.HammerstadRoughness(rms_height=1.0e-6)
    rough = mw.LossyMetalMedium(conductivity=_SIGMA, roughness=roughness)
    f = torch.tensor([10.0e9], dtype=torch.float64)
    ratio = rough.rational_surface_impedance(_BAND).surface_impedance(f).real / smooth.rational_surface_impedance(_BAND).surface_impedance(f).real
    assert float(ratio[0]) == pytest.approx(rough.roughness_loss_factor(10.0e9), rel=3e-3)
    assert float(ratio[0]) > 1.5


def test_rational_fit_fails_closed():
    metal = mw.LossyMetalMedium(conductivity=_SIGMA, roughness=mw.GroisseRoughness(rms_height=1e-6))
    with pytest.raises(ValueError, match="non-causal"):
        metal.rational_surface_impedance(_BAND)
    hammerstad = mw.LossyMetalMedium(conductivity=_SIGMA, roughness=mw.HammerstadRoughness(rms_height=1e-6))
    with pytest.raises((ValueError, RuntimeError)):
        hammerstad.rational_surface_impedance(_BAND, order=2)
    with pytest.raises(ValueError, match="representation"):
        hammerstad.rational_surface_impedance(_BAND, representation="S")


@pytest.mark.parametrize(
    "sigma,band,roughness",
    [
        # Diverged in the broadband surface update at 45 and 60 cells/wavelength.
        (50.0, (0.5e9, 5.0e9), mw.HurayRoughness.cannonball(0.3e-3)),
        (50.0, (0.5e9, 5.0e9), mw.HurayRoughness.cannonball(0.1e-3)),
        # Saturating copper Huray: refit direct-term ratio ~8.2, outside the envelope.
        (_SIGMA, _BAND, mw.HurayRoughness.cannonball(0.5e-6)),
    ],
)
def test_broadband_fit_rejects_rough_surfaces_outside_the_validated_envelope(sigma, band, roughness):
    metal = mw.LossyMetalMedium(conductivity=sigma, roughness=roughness)
    with pytest.raises(ValueError, match="narrowband surface"):
        metal.rational_surface_impedance(band)
    # The narrowband surface stays available for the same rough conductor.
    assert metal.roughness_loss_factor(band[0]) > 1.0


@pytest.mark.parametrize(
    "roughness",
    [mw.HammerstadRoughness(rms_height=1.5e-3), mw.HammerstadRoughness(rms_height=1.5e-3, roughness_factor=3.0)],
)
def test_broadband_fit_accepts_the_validated_stable_cases(roughness):
    # FDTD-stable at 45 and 60 cells/wavelength (refit direct-term ratios 2.01 and 3.04).
    metal = mw.LossyMetalMedium(conductivity=50.0, roughness=roughness)
    model = metal.rational_surface_impedance((0.5e9, 5.0e9))
    assert model.passivity.passive and model.passivity.certified


def test_rough_rational_model_builds_a_surface_impedance_medium():
    metal = mw.LossyMetalMedium(
        conductivity=_SIGMA,
        roughness=mw.HurayRoughness(sphere_radius=0.4e-6, sphere_density=1.5e11, relative_area=1.3),
    )
    medium = mw.SurfaceImpedanceMedium(impedance=metal.rational_surface_impedance(_BAND))
    assert medium.is_surface_impedance
    assert medium.frequency_range == _BAND
