"""Physical power normalization is independent of aperture size and excitation units."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0
from witwin.maxwell.fdtd.observers import _safe_divide_source_spectrum
from witwin.maxwell.postprocess import compute_s_parameters
from witwin.maxwell.postprocess.diffraction import compute_diffraction_from_payload
from witwin.maxwell.postprocess import modal


def _plane(length):
    coords = torch.linspace(-.5, .5, 5, dtype=torch.float64) * length
    shape = (5, 5)
    return dict(axis="x", position=0., frequency=C_0/length, normal_direction="+", y=coords, z=coords,
                Ey=torch.full(shape, 1e-12, dtype=torch.complex128),
                Hz=torch.full(shape, 1e-12/200, dtype=torch.complex128))


@pytest.mark.parametrize("length", [1., 1e-6])
def test_scattering_ratios_accept_small_positive_si_power(length):
    power = 1e-24 / 400 * length**2
    payloads = {name: dict(frequencies=(C_0/length,), flux=[flux]) for name, flux in (("in", .9*power), ("out", .8*power))}
    result = SimpleNamespace(frequencies=(C_0/length,), monitor=payloads.__getitem__)
    data = compute_s_parameters(result, "in", transmitted_monitor="out", incident_power=power)
    assert data["S11_mag"].item() == pytest.approx(.1**.5)
    assert data["S21_mag"].item() == pytest.approx(.8**.5)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_s_parameters(result, "in", incident_power=0.)


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("explicit", [False, True])
def test_diffraction_efficiencies_scale_with_aperture(length, explicit):
    result = SimpleNamespace(solver=SimpleNamespace(scene=SimpleNamespace(domain_range=(-length/2, length/2)*3)))
    payload = _plane(length)
    payload["diffraction_spec"] = {"orders": 0}
    power = 1e-24 / 400 * length**2
    data = compute_diffraction_from_payload(result, "plane", payload, incident_power=power if explicit else None)
    assert data["propagating_power"] == pytest.approx(power, rel=1e-12, abs=0)
    assert data["orders"][0]["efficiency"] == pytest.approx(1.)


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("backward", [False, True])
def test_modal_overlap_has_no_absolute_power_floor(monkeypatch, length, backward):
    payload = _plane(length)
    source = mw.ModeSource(position=(0.,0.,0.), size=(0.,length,length), polarization="Ey", source_time=mw.CW(frequency=C_0/length))
    scene = SimpleNamespace(sources=[source])
    result = SimpleNamespace(scene=scene, prepared_scene=scene,
                             solver=SimpleNamespace(scene=scene, _compiled_material_model={}))
    fields = modal._monitor_vector_fields(payload, (5,5), device=torch.device("cpu"), dtype=torch.complex128)
    # Supply an analytic reference, keeping the actual reciprocal overlap,
    # cell quadrature, normalization and fraction calculation under test.
    monkeypatch.setattr(modal, "_reference_mode_fields", lambda *a, **k: (
        payload["y"], payload["z"], *fields, {"effective_index":1., "beta":2*torch.pi/length}))
    if backward:
        payload["Hz"] = -payload["Hz"]
    data = modal._compute_mode_overlap_from_payload(result, "plane", payload)
    assert data["power_fraction_backward" if backward else "power_fraction_forward"].item() == pytest.approx(1.)
    assert data["power_fraction_forward" if backward else "power_fraction_backward"].item() == pytest.approx(0., abs=1e-20)


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("tensor", [False, True])
def test_monitor_source_normalization_preserves_tiny_voltages(length, tensor):
    # At fixed E, voltage scales with length. Both spectra are below the old floor.
    spectrum = np.array([1e-28*length*(1+2j), 2e-28*length*(1-1j)])
    expected = np.array([3+1j, 2-1j])
    if tensor:
        spectrum, expected = torch.as_tensor(spectrum), torch.as_tensor(expected)
    actual = _safe_divide_source_spectrum(expected*spectrum, spectrum)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=0)
    scalar = _safe_divide_source_spectrum(expected[0]*spectrum[0], spectrum[0])
    np.testing.assert_allclose(np.asarray(scalar), np.asarray(expected[0]), rtol=1e-6)
    with pytest.raises(ValueError, match="zero source spectrum"):
        _safe_divide_source_spectrum(expected, spectrum*0)


@pytest.mark.parametrize("length", [1., 1e-6])
def test_small_background_impedance_retains_relative_loss(length):
    from witwin.maxwell.postprocess.stratton_chu import _background_wavenumber_and_impedance

    # Opposite material losses give real k and a small complex eta; its loss
    # must survive even when the numerical value of eta is far below one ohm.
    k, eta = _background_wavenumber_and_impedance(
        1e12*(1-1e-8j), 1e-12*(1+1e-8j), omega=2*torch.pi*C_0/length,
        c=C_0, eta_vacuum=200.,
    )
    assert k*length == pytest.approx(2*torch.pi)
    assert isinstance(eta, complex)
    assert eta.imag/eta.real == pytest.approx(1e-8, rel=1e-12)
