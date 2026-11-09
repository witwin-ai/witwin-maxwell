import math

import pytest

import witwin.maxwell as mw

_C0 = 299_792_458.0


def _index(key: str, wavelength: float) -> complex:
    return mw.library_material(key).relative_permittivity(_C0 / wavelength) ** 0.5


def test_every_entry_is_an_ordinary_cited_material():
    assert set(mw.MATERIAL_LIBRARY) == {"SiO2", "Si3N4", "Al2O3", "cSi", "Ag", "Au", "Cu", "Al"}
    for key, entry in mw.MATERIAL_LIBRARY.items():
        assert isinstance(entry, mw.MaterialLibraryEntry)
        assert entry.key == key
        assert entry.reference and entry.doi.startswith("10.")
        low, high = entry.wavelength_range
        assert 0.0 < low < high
        assert entry.frequency_range == pytest.approx((_C0 / high, _C0 / low))
        assert entry.contains_wavelength(math.sqrt(low * high))
        assert not entry.contains_wavelength(0.5 * low)
        material = entry.material()
        assert isinstance(material, mw.Material)
        assert material.is_dispersive
        assert material.name == f"{key} ({entry.variant})"
        assert entry.material(name="custom").name == "custom"
        # Passive across the valid range (exp(-i*omega*t): absorption is Im(eps) >= 0).
        for step in range(9):
            wavelength = low * (high / low) ** (step / 8)
            assert material.relative_permittivity(_C0 / wavelength).imag >= 0.0


def test_sellmeier_entries_are_lossless_and_use_si_coefficients():
    for key in ("SiO2", "Si3N4", "Al2O3", "cSi"):
        entry = mw.MATERIAL_LIBRARY[key]
        material = entry.material()
        assert entry.model == "sellmeier"
        assert float(material.eps_r) == 1.0
        assert all(pole.gamma == 0.0 for pole in material.lorentz_poles)
        # Resonances lie outside the transparent window: UV above, IR below it.
        low_f, high_f = entry.frequency_range
        assert all(not low_f <= pole.resonance_frequency <= high_f for pole in material.lorentz_poles)
        assert material.relative_permittivity(_C0 / math.sqrt(entry.wavelength_range[0] * entry.wavelength_range[1])).imag == 0.0


def test_dielectric_indices_at_reference_wavelengths():
    # Malitson (1965): fused silica n = 1.4440 at 1.55 um and the sodium-D value 1.4584.
    assert _index("SiO2", 1.55e-6).real == pytest.approx(1.4440, abs=1e-4)
    assert _index("SiO2", 0.5893e-6).real == pytest.approx(1.4584, abs=1e-4)
    # Malitson (1962): sapphire ordinary-ray sodium-D index 1.7681.
    assert _index("Al2O3", 0.5893e-6).real == pytest.approx(1.7681, abs=1e-4)
    assert _index("Al2O3", 1.55e-6).real == pytest.approx(1.7462, abs=1e-4)
    # Luke et al. (2015): LPCVD Si3N4 n ~ 1.996 at 1.55 um.
    assert _index("Si3N4", 1.55e-6).real == pytest.approx(1.9963, abs=1e-4)
    # Salzberg and Villa (1957) silicon: n ~ 3.48 near 1.55 um, ~ 3.418 at 10 um.
    assert _index("cSi", 1.55e-6).real == pytest.approx(3.4777, abs=1e-3)
    assert _index("cSi", 10.0e-6).real == pytest.approx(3.4179, abs=1e-3)


def test_metals_are_lorentz_drude_with_metallic_response():
    for key in ("Ag", "Au", "Cu", "Al"):
        entry = mw.MATERIAL_LIBRARY[key]
        material = entry.material()
        assert entry.model == "lorentz-drude"
        assert "Rakic" in entry.reference and entry.doi == "10.1364/AO.37.005271"
        assert float(material.eps_r) == 1.0
        assert len(material.drude_poles) == 1
        assert len(material.lorentz_poles) in (4, 5)
        epsilon = material.relative_permittivity(_C0 / 1.55e-6)
        assert epsilon.real < -50.0
        assert epsilon.imag > 0.0
    # Rakic et al. (1998) Table 2: gold plasma energy 9.03 eV with f0 = 0.760, Gamma0 = 0.053 eV.
    ev_to_hz = 1.602176634e-19 / 6.62607015e-34
    gold = mw.library_material("Au")
    assert gold.drude_poles[0].plasma_frequency == pytest.approx(math.sqrt(0.760) * 9.03 * ev_to_hz)
    assert gold.drude_poles[0].gamma == pytest.approx(0.053 * ev_to_hz)


def test_unknown_key_lists_available_entries():
    with pytest.raises(KeyError, match="available"):
        mw.library_material("Unobtainium")
