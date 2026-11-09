"""Curated optical materials from published closed-form dispersion models.

Every entry is built through the ordinary ``Material`` constructors
(``Material.sellmeier`` or explicit ``DrudePole`` / ``LorentzPole`` lists) from
coefficients quoted verbatim in the cited source, and records that citation plus
the wavelength range over which the source model is valid. Outside that range the
model is an extrapolation and should not be trusted.

Sellmeier tables are quoted in ``micron^2`` and converted here to the SI
``meter^2`` that ``Material.sellmeier`` expects. Lorentz-Drude metal parameters
are quoted in eV and converted with ``f = E * e / h``. All models follow the
repository ``exp(-i*omega*t)`` convention (absorption is ``Im(eps) > 0``).

Usage::

    import witwin.maxwell as mw

    silica = mw.library_material("SiO2")
    entry = mw.MATERIAL_LIBRARY["Au"]
    entry.reference, entry.wavelength_range  # citation and valid range [m]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from .constants import C_0
from .media import DrudePole, LorentzPole, Material

_MICRON = 1.0e-6
_ELEMENTARY_CHARGE = 1.602176634e-19  # [C], exact SI
_PLANCK = 6.62607015e-34  # [J*s], exact SI
_EV_TO_HZ = _ELEMENTARY_CHARGE / _PLANCK


@dataclass(frozen=True)
class MaterialLibraryEntry:
    """One literature dispersion model.

    ``wavelength_range`` is the vacuum-wavelength validity window ``(min, max)``
    in meters; ``frequency_range`` is the matching ``(min, max)`` in Hz.
    """

    key: str
    description: str
    variant: str
    model: str
    reference: str
    doi: str
    wavelength_range: tuple[float, float]
    _builder: Callable[[str], Material]

    @property
    def frequency_range(self) -> tuple[float, float]:
        return (C_0 / self.wavelength_range[1], C_0 / self.wavelength_range[0])

    def contains_wavelength(self, wavelength: float) -> bool:
        return self.wavelength_range[0] <= float(wavelength) <= self.wavelength_range[1]

    def material(self, name: str | None = None) -> Material:
        """A fresh ``Material`` for this model (named ``"<key> (<variant>)"`` by default)."""

        return self._builder(name if name is not None else f"{self.key} ({self.variant})")


def _sellmeier_builder(b_coefficients, c_microns) -> Callable[[str], Material]:
    """``n^2 - 1 = sum B_i lambda^2 / (lambda^2 - C_i^2)`` with ``C_i`` in microns."""

    c_coefficients = tuple((c * _MICRON) ** 2 for c in c_microns)

    def build(name: str) -> Material:
        return Material.sellmeier(
            b_coefficients=b_coefficients,
            c_coefficients=c_coefficients,
            name=name,
        )

    return build


def _lorentz_drude_builder(plasma_ev, drude, oscillators) -> Callable[[str], Material]:
    """Rakic et al. (1998) Lorentz-Drude model.

    ``eps(omega) = 1 - f0 wp^2 / (omega (omega + i G0))
    + sum_j f_j wp^2 / (w_j^2 - omega^2 - i omega G_j)``; ``drude = (f0, G0)`` and
    each oscillator is ``(f_j, G_j, w_j)``, all energies in eV. The Drude term maps
    to ``DrudePole(sqrt(f0) * wp, G0)`` and oscillator ``j`` to
    ``LorentzPole(delta_eps = f_j wp^2 / w_j^2, w_j, G_j)``.
    """

    f0, gamma0 = drude

    def build(name: str) -> Material:
        drude_pole = DrudePole(
            plasma_frequency=math.sqrt(f0) * plasma_ev * _EV_TO_HZ,
            gamma=gamma0 * _EV_TO_HZ,
        )
        lorentz_poles = tuple(
            LorentzPole(
                delta_eps=strength * plasma_ev**2 / resonance**2,
                resonance_frequency=resonance * _EV_TO_HZ,
                gamma=damping * _EV_TO_HZ,
            )
            for strength, damping, resonance in oscillators
        )
        return Material(eps_r=1.0, name=name, drude_poles=(drude_pole,), lorentz_poles=lorentz_poles)

    return build


_RAKIC_1998 = (
    "A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski, \"Optical properties "
    "of metallic films for vertical-cavity optoelectronic devices,\" Appl. Opt. 37, "
    "5271-5283 (1998), Table 2 (Lorentz-Drude model)"
)
_RAKIC_1998_DOI = "10.1364/AO.37.005271"


def _entries() -> tuple[MaterialLibraryEntry, ...]:
    return (
        MaterialLibraryEntry(
            key="SiO2",
            description="Fused silica, lossless three-term Sellmeier at 20 C",
            variant="Malitson1965",
            model="sellmeier",
            reference=(
                "I. H. Malitson, \"Interspecimen comparison of the refractive index of fused "
                "silica,\" J. Opt. Soc. Am. 55, 1205-1209 (1965)"
            ),
            doi="10.1364/JOSA.55.001205",
            wavelength_range=(0.21 * _MICRON, 3.71 * _MICRON),
            _builder=_sellmeier_builder(
                (0.6961663, 0.4079426, 0.8974794),
                (0.0684043, 0.1162414, 9.896161),
            ),
        ),
        MaterialLibraryEntry(
            key="Si3N4",
            description="Stoichiometric LPCVD silicon nitride, lossless two-term Sellmeier",
            variant="Luke2015",
            model="sellmeier",
            reference=(
                "K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, and M. Lipson, \"Broadband "
                "mid-infrared frequency comb generation in a Si3N4 microresonator,\" Opt. Lett. "
                "40, 4823-4826 (2015)"
            ),
            doi="10.1364/OL.40.004823",
            wavelength_range=(0.310 * _MICRON, 5.504 * _MICRON),
            _builder=_sellmeier_builder((3.0249, 40314.0), (0.1353406, 1239.842)),
        ),
        MaterialLibraryEntry(
            key="Al2O3",
            description="Synthetic sapphire, ordinary ray, lossless three-term Sellmeier at 24 C",
            variant="Malitson1962",
            model="sellmeier",
            reference=(
                "I. H. Malitson, \"Refraction and dispersion of synthetic sapphire,\" J. Opt. "
                "Soc. Am. 52, 1377-1379 (1962)"
            ),
            doi="10.1364/JOSA.52.001377",
            wavelength_range=(0.2652 * _MICRON, 5.577 * _MICRON),
            _builder=_sellmeier_builder(
                (1.023798, 1.058264, 5.280792),
                (0.06144821, 0.1106997, 17.92656),
            ),
        ),
        MaterialLibraryEntry(
            key="cSi",
            description=(
                "Crystalline silicon in its transparent infrared band at 26 C, lossless "
                "three-term Sellmeier fitted by Tatian (1984) to the Salzberg-Villa data"
            ),
            variant="SalzbergVilla1957",
            model="sellmeier",
            reference=(
                "C. D. Salzberg and J. J. Villa, \"Infrared refractive indexes of silicon, "
                "germanium and modified selenium glass,\" J. Opt. Soc. Am. 47, 244-246 (1957); "
                "Sellmeier fit: B. Tatian, \"Fitting refractive-index data with the Sellmeier "
                "dispersion formula,\" Appl. Opt. 23, 4477-4485 (1984)"
            ),
            doi="10.1364/JOSA.47.000244",
            wavelength_range=(1.357 * _MICRON, 11.04 * _MICRON),
            _builder=_sellmeier_builder(
                (10.6684293, 0.0030434748, 1.54133408),
                (0.301516485, 1.13475115, 1104.0),
            ),
        ),
        MaterialLibraryEntry(
            key="Ag",
            description="Silver film, Lorentz-Drude model",
            variant="RakicLorentzDrude1998",
            model="lorentz-drude",
            reference=_RAKIC_1998,
            doi=_RAKIC_1998_DOI,
            wavelength_range=(0.24797 * _MICRON, 12.398 * _MICRON),
            _builder=_lorentz_drude_builder(
                9.01,
                (0.845, 0.048),
                (
                    (0.065, 3.886, 0.816),
                    (0.124, 0.452, 4.481),
                    (0.011, 0.065, 8.185),
                    (0.840, 0.916, 9.083),
                    (5.646, 2.419, 20.29),
                ),
            ),
        ),
        MaterialLibraryEntry(
            key="Au",
            description="Gold film, Lorentz-Drude model",
            variant="RakicLorentzDrude1998",
            model="lorentz-drude",
            reference=_RAKIC_1998,
            doi=_RAKIC_1998_DOI,
            wavelength_range=(0.24797 * _MICRON, 6.1992 * _MICRON),
            _builder=_lorentz_drude_builder(
                9.03,
                (0.760, 0.053),
                (
                    (0.024, 0.241, 0.415),
                    (0.010, 0.345, 0.830),
                    (0.071, 0.870, 2.969),
                    (0.601, 2.494, 4.304),
                    (4.384, 2.214, 13.32),
                ),
            ),
        ),
        MaterialLibraryEntry(
            key="Cu",
            description="Copper film, Lorentz-Drude model",
            variant="RakicLorentzDrude1998",
            model="lorentz-drude",
            reference=_RAKIC_1998,
            doi=_RAKIC_1998_DOI,
            wavelength_range=(0.20664 * _MICRON, 12.398 * _MICRON),
            _builder=_lorentz_drude_builder(
                10.83,
                (0.575, 0.030),
                (
                    (0.061, 0.378, 0.291),
                    (0.104, 1.056, 2.957),
                    (0.723, 3.213, 5.300),
                    (0.638, 4.305, 11.18),
                ),
            ),
        ),
        MaterialLibraryEntry(
            key="Al",
            description="Aluminum film, Lorentz-Drude model",
            variant="RakicLorentzDrude1998",
            model="lorentz-drude",
            reference=_RAKIC_1998,
            doi=_RAKIC_1998_DOI,
            wavelength_range=(0.061992 * _MICRON, 247.97 * _MICRON),
            _builder=_lorentz_drude_builder(
                14.98,
                (0.523, 0.047),
                (
                    (0.227, 0.333, 0.162),
                    (0.050, 0.312, 1.544),
                    (0.166, 1.351, 1.808),
                    (0.030, 3.382, 3.473),
                ),
            ),
        ),
    )


MATERIAL_LIBRARY: Mapping[str, MaterialLibraryEntry] = MappingProxyType(
    {entry.key: entry for entry in _entries()}
)


def library_material(key: str, *, name: str | None = None) -> Material:
    """Build the ``Material`` for library entry ``key`` (see ``MATERIAL_LIBRARY``)."""

    try:
        entry = MATERIAL_LIBRARY[key]
    except KeyError:
        raise KeyError(f"Unknown library material {key!r}; available: {sorted(MATERIAL_LIBRARY)}.") from None
    return entry.material(name)
