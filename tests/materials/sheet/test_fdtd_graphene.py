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

    with pytest.raises(NotImplementedError, match="interband"):
        mw.Graphene(chemical_potential=0.2, scattering_time=1.0e-13, include_interband=True)
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
