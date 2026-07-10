import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from tests.materials.sheet.sheet_measurement import Z0 as _Z0, measure_sheet_scattering
from witwin.core import Box
from witwin.core.material import VACUUM_PERMITTIVITY
from witwin.maxwell.scene import prepare_scene

_ELEMENTARY_CHARGE = 1.602176634e-19  # [C]
_REDUCED_PLANCK = 1.054571817e-34  # [J*s]
_BOLTZMANN = 1.380649e-23  # [J/K]


def _kubo_intraband_sigma(mu_c_ev, tau, temperature, frequency):
    """Reference Kubo intraband sheet conductivity, written out independently."""
    kt = _BOLTZMANN * temperature
    mu_over_kt = mu_c_ev * _ELEMENTARY_CHARGE / kt
    weight = (
        _ELEMENTARY_CHARGE**2 * kt / (np.pi * _REDUCED_PLANCK**2)
    ) * (mu_over_kt + 2.0 * np.log1p(np.exp(-mu_over_kt)))
    omega = 2.0 * np.pi * frequency
    return weight / (1.0 / tau - 1j * omega)


def _kubo_interband_sigma_pv(mu_c_ev, tau, temperature, omega):
    """Independent reference: Kubo interband conductivity (T>0 principal value).

    e^{-i*omega*t} convention, written from scratch so the fitted-pole
    conductivity is validated against an analytic evaluation. The slowly
    decaying ``1/xi^2`` tail is split off in closed form at ``xi = split``
    (where the Fermi factor is unity); the near part, holding the resonant peak
    at ``xi = hbar*omega/2`` and the Fermi step at ``xi = mu_c``, is integrated
    adaptively with both passed as subdivision hints.
    """
    from scipy.integrate import quad

    mu = mu_c_ev * _ELEMENTARY_CHARGE
    kt = _BOLTZMANN * temperature
    omega_c = omega + 1j / tau

    def fermi(energy):
        return 1.0 / (np.exp(np.clip((energy - mu) / kt, -500.0, 500.0)) + 1.0)

    def integrand(energy):
        return (fermi(-energy) - fermi(energy)) / (omega_c**2 - 4.0 * (energy / _REDUCED_PLANCK) ** 2)

    split = mu + 40.0 * kt + 6.0 * _REDUCED_PLANCK * abs(omega)
    peak = _REDUCED_PLANCK * omega / 2.0
    width = _REDUCED_PLANCK / (2.0 * tau)
    hints = (peak - 2.0 * width, peak, peak + 2.0 * width, mu - 4.0 * kt, mu, mu + 4.0 * kt)
    pts = sorted(p for p in hints if 0.0 < p < split) or None
    real, _ = quad(lambda e: integrand(e).real, 0.0, split, limit=2000, points=pts)
    imag, _ = quad(lambda e: integrand(e).imag, 0.0, split, limit=2000, points=pts)
    tail = (_REDUCED_PLANCK / (4.0 * omega_c)) * np.log(
        (2.0 * split - _REDUCED_PLANCK * omega_c) / (2.0 * split + _REDUCED_PLANCK * omega_c)
    )
    return 1j * _ELEMENTARY_CHARGE**2 * omega_c / (np.pi * _REDUCED_PLANCK**2) * (real + 1j * imag + tail)


def _kubo_interband_sigma_t0(mu_c_ev, tau, omega):
    """Independent reference: closed-form T=0 interband conductivity (log form)."""
    mu = mu_c_ev * _ELEMENTARY_CHARGE
    omega_c = omega + 1j / tau
    ratio = (2.0 * mu - _REDUCED_PLANCK * omega_c) / (2.0 * mu + _REDUCED_PLANCK * omega_c)
    return 1j * _ELEMENTARY_CHARGE**2 / (4.0 * np.pi * _REDUCED_PLANCK) * np.log(ratio)


def test_graphene_construction_and_kubo_intraband_conductivity():
    material = mw.Graphene(chemical_potential=0.2, scattering_time=1.0e-13, temperature=300.0)
    assert material.is_medium2d
    assert material.sigma_s == 0.0

    frequency = 1.0e12
    sigma = material.sheet_conductivity_at_freq(frequency)
    sigma_ref = _kubo_intraband_sigma(0.2, 1.0e-13, 300.0, frequency)
    assert sigma == pytest.approx(sigma_ref, rel=1e-9)

    # DC limit: sigma_dc = A * tau.
    sigma_dc = material.sheet_conductivity(0.0)
    assert sigma_dc.real == pytest.approx(material.intraband_drude_weight * 1.0e-13, rel=1e-9)
    assert sigma_dc.imag == pytest.approx(0.0, abs=1e-15)

    # One intraband Drude-like relaxation term.
    ((weight, rate),) = material.sheet_pole_terms()
    assert weight == pytest.approx(material.intraband_drude_weight, rel=1e-12)
    assert rate == pytest.approx(1.0e13, rel=1e-12)
    assert material.characteristic_frequency == pytest.approx(1.0e13 / (2.0 * np.pi), rel=1e-12)

    # The default intraband-only model carries no resonant sheet terms.
    assert material.sheet_lorentz_terms() == ()

    with pytest.raises(ValueError):
        mw.Graphene(chemical_potential=0.2, scattering_time=0.0)
    with pytest.raises(ValueError):
        mw.Graphene(chemical_potential=-0.1, scattering_time=1.0e-13)


def _build_cpu_graphene_scene(material):
    return prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-50.0e-6, 50.0e-6), (-20.0e-6, 20.0e-6), (-20.0e-6, 20.0e-6))),
            grid=mw.GridSpec.uniform(5.0e-6),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cpu",
            structures=[
                mw.Structure(
                    geometry=Box(position=(0.0, 0.0, 0.0), size=(0.0, 1.0e-3, 1.0e-3)),
                    material=material,
                )
            ],
        )
    )


def test_graphene_compiles_to_tangential_sheet_drude_pole():
    material = mw.Graphene(chemical_potential=0.2, scattering_time=1.0e-13)
    scene = _build_cpu_graphene_scene(material)
    model = scene.compile_materials()

    assert len(model["drude_poles"]) == 1
    entry = model["drude_poles"][0]
    assert entry["axes"] == ("y", "z")

    node_index = int(np.argmin(np.abs(scene.x_nodes64 - 0.0)))
    dual = float(scene.dx_dual64[node_index])
    pole = entry["pole"]
    omega_p = 2.0 * np.pi * pole.plasma_frequency
    assert VACUUM_PERMITTIVITY * omega_p**2 == pytest.approx(
        material.intraband_drude_weight / dual, rel=1e-6
    )
    assert 2.0 * np.pi * pole.gamma == pytest.approx(material.scattering_rate, rel=1e-9)

    weight = entry["weight"]
    plane = weight[node_index]
    assert torch.all(plane == 1.0)
    off_plane = torch.cat([weight[:node_index], weight[node_index + 1 :]])
    assert torch.all(off_plane == 0)

    # The static conductivity channel stays untouched (graphene sigma_s = 0).
    for axis in ("x", "y", "z"):
        assert torch.all(model["sigma_e_components"][axis] == 0)

    # Frequency evaluation applies the sheet susceptibility to tangential axes only.
    eps_components, _ = scene.compile_material_components(frequency=1.0e12)
    assert torch.all(eps_components["x"].imag == 0)
    assert bool(torch.any(eps_components["y"][node_index].imag != 0))
    assert bool(torch.any(eps_components["z"][node_index].imag != 0))


def test_graphene_characteristic_frequency_feeds_auto_dt_bound():
    from witwin.maxwell.fdtd.runtime.initialization import _material_characteristic_frequency

    material = mw.Graphene(chemical_potential=0.2, scattering_time=1.0e-13)
    assert _material_characteristic_frequency(material) == pytest.approx(
        material.scattering_rate / (2.0 * np.pi), rel=1e-12
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_graphene_sheet_transmission_matches_kubo_fresnel():
    # Free-standing graphene sheet at normal incidence with the Kubo intraband
    # sigma_s(omega): t = 2 / (2 + Z0*sigma_s(omega)), r = t - 1.
    material = mw.Graphene(chemical_potential=0.2, scattering_time=1.0e-13, temperature=300.0)
    for frequency in (1.0e12, 3.0e12):
        t, r = measure_sheet_scattering(
            material,
            frequency,
            dl=4.0e-6,
            half_length=300.0e-6,
            half_width=20.0e-6,
            fit_gap=20.0e-6,
            fit_extent=200.0e-6,
        )
        sigma = material.sheet_conductivity_at_freq(frequency)
        t_exact = 2.0 / (2.0 + _Z0 * sigma)
        r_exact = -_Z0 * sigma / (2.0 + _Z0 * sigma)
        assert abs(abs(t) - abs(t_exact)) < 0.02
        assert abs(abs(r) - abs(r_exact)) < 0.02


def test_kubo_interband_reference_pv_converges_to_t0_log():
    # The T>0 principal-value integral must converge to the closed-form T=0 log
    # expression as T -> 0. This validates the reference integrator (sign,
    # prefactor, the slowly-decaying tail) independently of the fit. At room
    # temperature the two forms genuinely differ near the edge (thermal
    # smearing), so the numeric-agreement check is made in the degenerate limit.
    mu_c, tau = 0.6, 1.0e-13
    omega_gap = 2.0 * mu_c * _ELEMENTARY_CHARGE / _REDUCED_PLANCK
    for frac in (0.3, 0.5, 0.7, 0.9):
        omega = frac * omega_gap
        # Below the edge Im(sigma_inter) is capacitive (negative) at room
        # temperature -- the reason a Drude pole cannot represent it.
        assert _kubo_interband_sigma_pv(mu_c, tau, 300.0, omega).imag < 0.0
        pv_cold = _kubo_interband_sigma_pv(mu_c, tau, 20.0, omega)
        t0 = _kubo_interband_sigma_t0(mu_c, tau, omega)
        assert abs(pv_cold - t0) <= 0.01 * abs(t0)


def test_graphene_interband_construction_and_fit():
    intraband = mw.Graphene(chemical_potential=0.4, scattering_time=1.0e-13, temperature=300.0)
    assert intraband.include_interband is False
    assert intraband.sheet_lorentz_terms() == ()

    material = mw.Graphene(
        chemical_potential=0.4, scattering_time=1.0e-13, temperature=300.0, include_interband=True
    )
    assert material.is_medium2d
    assert material.include_interband is True

    # Intraband stays a single Drude relaxation term, unchanged by interband.
    ((weight, rate),) = material.sheet_pole_terms()
    assert weight == pytest.approx(material.intraband_drude_weight, rel=1e-12)
    assert rate == pytest.approx(material.scattering_rate, rel=1e-12)

    # Interband adds passive Lorentz terms (positive strength, resonance, linewidth).
    terms = material.sheet_lorentz_terms()
    assert len(terms) >= 1
    for strength, omega_0, gamma in terms:
        assert strength > 0.0
        assert omega_0 > 0.0
        assert gamma > 0.0

    # The interband resonances raise the auto-dt characteristic frequency above
    # the intraband relaxation rate.
    assert material.characteristic_frequency > material.scattering_rate / (2.0 * np.pi)


def test_graphene_interband_sheet_conductivity_matches_kubo():
    for mu_c, tau, temperature in ((0.5, 1.0e-13, 300.0), (0.3, 1.0e-13, 300.0), (0.2, 5.0e-13, 300.0)):
        material = mw.Graphene(
            chemical_potential=mu_c, scattering_time=tau, temperature=temperature, include_interband=True
        )
        omega_gap = 2.0 * mu_c * _ELEMENTARY_CHARGE / _REDUCED_PLANCK
        omega = np.linspace(0.03 * omega_gap, 0.95 * omega_gap, 80)
        fitted = np.array([material.sheet_conductivity(w) for w in omega])
        reference = np.array(
            [
                _kubo_intraband_sigma(mu_c, tau, temperature, w / (2.0 * np.pi))
                + _kubo_interband_sigma_pv(mu_c, tau, temperature, w)
                for w in omega
            ]
        )
        # L2-relative agreement over the band (the total conductivity Im part
        # crosses zero mid-band, so a whole-band norm is the meaningful metric).
        l2_rel = np.linalg.norm(fitted - reference) / np.linalg.norm(reference)
        assert l2_rel < 0.03
        # Pointwise agreement away from the mid-band Im zero-crossing, where
        # |sigma| collapses to the tiny real part and relative error is undefined.
        scale = np.maximum(np.abs(reference), 0.12 * np.abs(reference).max())
        assert np.max(np.abs(fitted - reference) / scale) < 0.03


def test_graphene_interband_compiles_to_tangential_lorentz_poles():
    material = mw.Graphene(
        chemical_potential=0.4, scattering_time=1.0e-13, temperature=300.0, include_interband=True
    )
    scene = _build_cpu_graphene_scene(material)
    model = scene.compile_materials()

    # One intraband Drude pole; one Lorentz pole per fitted interband term.
    assert len(model["drude_poles"]) == 1
    assert len(model["lorentz_poles"]) == len(material.sheet_lorentz_terms())

    node_index = int(np.argmin(np.abs(scene.x_nodes64 - 0.0)))
    dual = float(scene.dx_dual64[node_index])
    for entry, (strength, omega_0, gamma) in zip(
        model["lorentz_poles"], material.sheet_lorentz_terms()
    ):
        assert entry["axes"] == ("y", "z")
        pole = entry["pole"]
        assert 2.0 * np.pi * pole.resonance_frequency == pytest.approx(omega_0, rel=1e-6)
        assert 2.0 * np.pi * pole.gamma == pytest.approx(gamma, rel=1e-6)
        # delta_eps = strength / (eps0 * dual_spacing).
        assert pole.delta_eps == pytest.approx(strength / (VACUUM_PERMITTIVITY * dual), rel=1e-6)
        plane = entry["weight"][node_index]
        assert torch.all(plane == 1.0)
        off_plane = torch.cat([entry["weight"][:node_index], entry["weight"][node_index + 1 :]])
        assert torch.all(off_plane == 0)

    # Frequency evaluation applies the interband susceptibility to tangential
    # axes only; the normal (x) axis stays real.
    frequency = 0.6 * (2.0 * 0.4 * _ELEMENTARY_CHARGE / _REDUCED_PLANCK) / (2.0 * np.pi)
    eps_components, _ = scene.compile_material_components(frequency=frequency)
    assert torch.all(eps_components["x"].imag == 0)
    assert bool(torch.any(eps_components["y"][node_index].imag != 0))
    assert bool(torch.any(eps_components["z"][node_index].imag != 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_graphene_interband_sheet_scattering_matches_realized_sigma():
    # Optical-band spot check: an interband-enabled graphene sheet must realize
    # its full (intraband Drude + interband Lorentz) sheet conductivity in the
    # time-domain ADE. The measured complex (t, r) are compared against the
    # Fresnel sheet relations at the descriptor sigma_s(omega) the poles
    # reproduce, and the sheet conductivity is inverted back from t.
    mu_c, tau = 0.3, 1.0e-13
    material = mw.Graphene(
        chemical_potential=mu_c, scattering_time=tau, temperature=300.0, include_interband=True
    )
    omega_gap = 2.0 * mu_c * _ELEMENTARY_CHARGE / _REDUCED_PLANCK
    frequency = 0.9 * omega_gap / (2.0 * np.pi)  # below the edge, interband-dominated
    wavelength = 299792458.0 / frequency
    dl = wavelength / 28.0

    sigma = material.sheet_conductivity_at_freq(frequency)
    # Below the edge the interband term flips Im(sigma_s) from inductive
    # (intraband) to capacitive, so the intraband-only sheet is a completely
    # different scatterer here -- the spot check genuinely exercises interband.
    intraband_only = material.intraband_drude_weight / (1.0 / tau - 1j * 2.0 * np.pi * frequency)
    assert abs(sigma - intraband_only) > 0.5 * abs(sigma)

    t, r = measure_sheet_scattering(
        material,
        frequency,
        dl=dl,
        half_length=12.0 * wavelength,
        half_width=3.0 * dl,
        fit_gap=2.0 * wavelength,
        fit_extent=7.0 * wavelength,
        steady_cycles=24,
        transient_cycles=28,
    )
    t_exact = 2.0 / (2.0 + _Z0 * sigma)
    r_exact = -_Z0 * sigma / (2.0 + _Z0 * sigma)
    assert abs(t - t_exact) < 0.005
    assert abs(r - r_exact) < 0.005

    # Invert the measured transmission back to the realized sheet conductivity
    # and require it to match the fitted Kubo sigma_s to a few percent.
    sigma_measured = 2.0 * (1.0 - t) / (_Z0 * t)
    assert abs(sigma_measured - sigma) < 0.05 * abs(sigma)
