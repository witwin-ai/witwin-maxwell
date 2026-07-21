"""Canonical tissue dielectric/mass catalogue for the phantom benchmark family.

All values are redistributable, published-class reference numbers for canonical
homogeneous tissues at 900 MHz. They come from the widely tabulated tissue
dielectric survey (Gabriel C., Gabriel S., Corthout E., "The dielectric
properties of biological tissues", Phys. Med. Biol. 41 (1996)) and the
mass-density tabulation used by the IEC/IEEE 62704-class canonical phantom
definitions. No licensed anatomical model or proprietary tissue database is
distributed with the repository; only these homogeneous canonical values.

The numbers below are a benchmark fixture, not a certified compliance dataset.
Each entry pins one homogeneous tissue at ``BENCHMARK_FREQUENCY``.
"""

from __future__ import annotations

import witwin.maxwell as mw

# The canonical SAR benchmark frequency (a common mobile-exposure band). All
# dielectric values are the published-class numbers at this frequency.
BENCHMARK_FREQUENCY = 900e6

# name -> (relative permittivity, conductivity [S/m], mass density [kg/m^3]).
TISSUE_PROPERTIES = {
    "skin_dry": (41.4, 0.87, 1109.0),
    "fat": (5.46, 0.051, 911.0),
    "muscle": (55.0, 0.94, 1090.0),
    # A generic single-tissue phantom filler in the muscle-equivalent class.
    "phantom": (42.0, 0.99, 1000.0),
}


def tissue_material(name: str) -> mw.Material:
    """Build a lossy :class:`Material` with mass density for the named tissue."""

    if name not in TISSUE_PROPERTIES:
        raise KeyError(f"Unknown tissue {name!r}; choices are {tuple(TISSUE_PROPERTIES)}.")
    eps_r, sigma_e, mass_density = TISSUE_PROPERTIES[name]
    return mw.Material(
        eps_r=eps_r,
        sigma_e=sigma_e,
        mass_density=mass_density,
        name=name,
    )
