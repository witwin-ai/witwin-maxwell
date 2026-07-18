"""Slice 1a: GyromagneticFerrite material type (SI + from_cgs, validation, autograd)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell import media
from witwin.maxwell.compiler import materials as material_compiler

MU_0 = 4.0e-7 * math.pi


def _ferrite(**kwargs):
    base = dict(
        eps_r=14.5, saturation_magnetization=1.40e5, bias_field=(0.0, 0.0, 1.75e5),
        gilbert_damping=2.0e-3,
    )
    base.update(kwargs)
    return mw.GyromagneticFerrite(**base)


def test_public_export():
    assert mw.GyromagneticFerrite is media.GyromagneticFerrite
    assert issubclass(mw.GyromagneticFerrite, mw.Material)


def test_derived_frequencies():
    f = _ferrite(gyromagnetic_ratio=1.760859e11)
    assert f.omega_0 == pytest.approx(1.760859e11 * MU_0 * 1.75e5, rel=1e-12)
    assert f.omega_m == pytest.approx(1.760859e11 * MU_0 * 1.40e5, rel=1e-12)
    assert f.bias_magnitude == pytest.approx(1.75e5)
    assert f.bias_unit_vector == pytest.approx((0.0, 0.0, 1.0))
    assert f.resonance_frequency == pytest.approx(f.omega_0 / (2 * math.pi))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(saturation_magnetization=0.0), "saturation_magnetization"),
        (dict(saturation_magnetization=-1.0), "saturation_magnetization"),
        (dict(bias_field=(0.0, 0.0, 0.0)), "non-zero"),
        (dict(gilbert_damping=-1e-3), "gilbert_damping"),
        (dict(gyromagnetic_ratio=0.0), "gyromagnetic_ratio"),
        (dict(mu_infinity=0.0), "mu_infinity"),
        (dict(eps_r=0.0), "eps_r"),
        (dict(bias_field=(0.0, 0.0)), "3-vector"),
    ],
)
def test_validation_rejects(kwargs, message):
    with pytest.raises((ValueError, TypeError)) as exc:
        _ferrite(**kwargs)
    assert message in str(exc.value)


def test_from_cgs_conversion_recorded():
    f = mw.GyromagneticFerrite.from_cgs(saturation_4piMs_gauss=1750.0, bias_Oe=2199.0, eps_r=14.5)
    factor = 1.0e-4 / MU_0
    assert f.saturation_magnetization == pytest.approx(1750.0 * factor)
    assert f.bias_magnitude == pytest.approx(2199.0 * factor)
    conversion = dict(f.cgs_conversion)
    assert conversion["saturation_4piMs_gauss"] == 1750.0
    assert conversion["bias_Oe"] == 2199.0
    assert conversion["cgs_to_A_per_m"] == pytest.approx(factor)
    # SI constructor records no CGS conversion.
    assert _ferrite().cgs_conversion == ()


def test_from_resonance_backcomputes_bias_and_damping():
    f = mw.GyromagneticFerrite.from_resonance(
        resonance_frequency=6.0e9, saturation_magnetization=1.40e5, linewidth=2.0e8, eps_r=14.5,
    )
    assert f.resonance_frequency == pytest.approx(6.0e9, rel=1e-10)
    assert f.gilbert_damping == pytest.approx(2.0e8 / (2.0 * 6.0e9))


def test_polder_tensor_frozen_form():
    """mu_r = [[mu, -i kappa, 0],[i kappa, mu, 0],[0,0,mu_par]] for bias +z."""
    f = _ferrite()
    t = f.polder_tensor(2.0 * math.pi * 9.0e9)
    assert t.shape == (3, 3)
    assert t.dtype == torch.complex128
    # Off-diagonal anti-symmetry: mu_xy = -mu_yx, both purely +/- i*kappa.
    assert torch.allclose(t[0, 1], -t[1, 0])
    assert torch.allclose(t[0, 0], t[1, 1])
    # Bias axis (z): pure background, decoupled row/col.
    assert torch.allclose(t[2, 2], torch.tensor(1.0 + 0j, dtype=torch.complex128))
    for idx in [(0, 2), (2, 0), (1, 2), (2, 1)]:
        assert torch.allclose(t[idx], torch.zeros((), dtype=torch.complex128))


def _scalar_polder_closed_form(f, frequency):
    """Independent scalar (mu, kappa) from the frozen omega_0/omega_m formula."""
    omega = 2.0 * math.pi * frequency
    w0, wm, alpha = f.omega_0, f.omega_m, f.gilbert_damping
    W = w0 - 1j * alpha * omega
    D = W * W - omega * omega
    mu = f.mu_infinity + wm * W / D
    kappa = wm * omega / D
    return mu, kappa


def test_scalar_polder_components_z_bias():
    """z-aligned bias: extracted (mu, kappa) match the closed-form scalars."""
    f = _ferrite()
    mu, kappa = f.scalar_polder_components(9.0e9)
    mu_t, kappa_t = _scalar_polder_closed_form(f, 9.0e9)
    assert abs(mu - mu_t) <= 1e-12 * abs(mu_t)
    assert abs(kappa - kappa_t) <= 1e-12 * abs(kappa_t)


@pytest.mark.parametrize(
    "bias_field",
    [
        (1.75e5, 0.0, 0.0),
        (0.0, 1.75e5, 0.0),
        (1.0e5, 1.0e5, 1.0e5),
        (-0.7e5, 1.2e5, -0.9e5),
    ],
)
def test_scalar_polder_components_frame_invariant(bias_field):
    """Non-z bias must yield the same frame-invariant scalars as z-bias.

    Falsification of the old fail-open extraction (mu = tensor[0,0],
    kappa = tensor[1,0]/1j only valid for b = z_hat): those hardcoded entries
    report near-zero gyrotropy for an x-aligned bias.
    """
    f = _ferrite(bias_field=bias_field)
    mu, kappa = f.scalar_polder_components(9.0e9)
    mu_t, kappa_t = _scalar_polder_closed_form(f, 9.0e9)
    assert abs(mu - mu_t) <= 1e-12 * abs(mu_t)
    assert abs(kappa - kappa_t) <= 1e-12 * abs(kappa_t)
    # Guard against the degenerate old behavior: real gyrotropy is resolved.
    assert abs(kappa) > 1e-3


def test_polder_hermitian_lossless_antihermitian_lossy():
    omega = 2.0 * math.pi * 9.0e9
    lossless = _ferrite(gilbert_damping=0.0).polder_tensor(omega)
    assert torch.allclose(lossless, lossless.conj().T, atol=1e-9)
    lossy = _ferrite(gilbert_damping=5.0e-2).polder_tensor(omega)
    assert not torch.allclose(lossy, lossy.conj().T, atol=1e-6)
    # Anti-Hermitian part is positive semidefinite (passive absorption) for exp(-i w t).
    anti = (lossy - lossy.conj().T) / (2j)
    eigs = torch.linalg.eigvalsh(anti)
    assert bool((eigs >= -1e-6).all())


def test_permeability_tensor_at_freq_matches_angular():
    f = _ferrite()
    freq = 9.0e9
    assert torch.allclose(f.permeability_tensor_at_freq(freq), f.polder_tensor(2 * math.pi * freq))


def test_bias_reversal_property():
    omega = 2.0 * math.pi * 9.0e9
    up = _ferrite(bias_field=(0.0, 0.0, 1.75e5)).polder_tensor(omega)
    down = _ferrite(bias_field=(0.0, 0.0, -1.75e5)).polder_tensor(omega)
    assert torch.allclose(up[0, 1], -down[0, 1])
    assert torch.allclose(up[0, 0], down[0, 0])


def test_polder_tensor_autograd_frequency():
    f = _ferrite()
    omega = torch.tensor(2.0 * math.pi * 9.0e9, dtype=torch.float64, requires_grad=True)
    f.polder_tensor(omega).abs().sum().backward()
    assert omega.grad is not None and torch.isfinite(omega.grad).all() and float(omega.grad) != 0.0


def test_polder_tensor_autograd_parameters():
    """Material-parameter differentiability through the module-level Polder function."""
    omega = 2.0 * math.pi * 9.0e9
    omega_0 = torch.tensor(_ferrite().omega_0, dtype=torch.float64, requires_grad=True)
    omega_m = torch.tensor(_ferrite().omega_m, dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor(2.0e-3, dtype=torch.float64, requires_grad=True)
    t = media.gyromagnetic_polder_tensor(
        omega, omega_0=omega_0, omega_m=omega_m, gilbert_damping=alpha,
        mu_infinity=1.0, bias_unit_vector=(0.0, 0.0, 1.0),
    )
    t.abs().sum().backward()
    for grad in (omega_0.grad, omega_m.grad, alpha.grad):
        assert grad is not None and torch.isfinite(grad).all() and float(grad) != 0.0


def test_polder_gradient_matches_finite_difference():
    """Frequency gradient of a scalar readout vs. central finite difference (<2%)."""
    f = _ferrite()

    def readout(w):
        return f.polder_tensor(torch.as_tensor(w, dtype=torch.float64))[0, 0].real

    w0 = 2.0 * math.pi * 9.0e9
    w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    readout(w).backward()
    analytic = float(w.grad)
    h = w0 * 1e-6
    fd = (float(readout(w0 + h)) - float(readout(w0 - h))) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 2.0e-2


def test_scalar_permeability_and_frequency_eval_rejected():
    f = _ferrite()
    with pytest.raises(NotImplementedError):
        f.relative_permeability(9.0e9)
    with pytest.raises(NotImplementedError):
        f.evaluate_at_frequency(9.0e9)


def test_permittivity_still_scalar():
    """A ferrite permittivity is an ordinary scalar; eps evaluation still works."""
    f = _ferrite(eps_r=14.5, sigma_e=1.0)
    eps = f.relative_permittivity(9.0e9)
    assert eps.real == pytest.approx(14.5)
    assert eps.imag > 0.0


def test_capabilities_flags():
    caps = _ferrite().capabilities()
    assert caps.magnetic and caps.anisotropic and caps.dispersive


def test_evaluate_static_background_for_meshing():
    """Static sample carries the real eps and the mu_infinity background (no gyrotropy)."""
    f = _ferrite(mu_infinity=1.0)
    sample = f.evaluate_static()
    assert float(sample.eps_r) == pytest.approx(14.5)
    assert float(sample.mu_r) == pytest.approx(1.0)


def test_off_diagonal_mu_guard_still_in_force():
    """The ferrite must not widen the rejected off-diagonal mu_tensor path."""
    with pytest.raises(NotImplementedError):
        mw.Material(mu_tensor=mw.Tensor3x3([[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]]))


def test_perturbation_medium_rejects_ferrite():
    grid = torch.ones((2, 2, 2), dtype=torch.float64)
    with pytest.raises(NotImplementedError) as exc:
        mw.PerturbationMedium(_ferrite(), perturbation=grid)
    assert "GyromagneticFerrite" in str(exc.value)


def test_compiler_fails_closed_on_ferrite():
    """Fail-closed: the material compiler rejects a ferrite rather than dropping gyrotropy."""
    box = mw.Box(size=(1e-3, 1e-3, 1e-3))
    structure = mw.Structure(geometry=box, material=_ferrite())
    with pytest.raises(NotImplementedError) as exc:
        material_compiler._static_structure_material(structure)
    assert "GyromagneticFerrite" in str(exc.value)
