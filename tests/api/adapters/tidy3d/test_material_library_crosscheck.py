"""Numerical cross-check of the literature material library against the reference package.

Narrowly scoped: it only compares permittivity values of the same literature
variant at in-range wavelengths and is skipped when the reference package is not
installed. No code is shared with the reference package.
"""

import math

import numpy as np
import pytest

import witwin.maxwell as mw

pytest.importorskip("tidy3d")
from tidy3d import material_library as td_library  # noqa: E402

_C0 = 299_792_458.0

# (library key, reference variant, relative tolerance on eps). Exact closed-form
# variants agree to rounding; the reference stores the Salzberg-Villa silicon data
# as a reduced-order pole fit instead of the three-term Sellmeier, so it agrees only
# to that fit's accuracy (~1e-4 in n).
_PAIRS = (
    ("Si3N4", "Luke2015Sellmeier", 1e-8),
    ("Ag", "RakicLorentzDrude1998", 1e-8),
    ("Au", "RakicLorentzDrude1998", 1e-8),
    ("Cu", "RakicLorentzDrude1998", 1e-8),
    ("Al", "RakicLorentzDrude1998", 1e-8),
    ("cSi", "SalzbergVilla1957", 3e-4),
)


@pytest.mark.parametrize(("key", "variant", "tolerance"), _PAIRS)
def test_library_matches_reference_variant(key, variant, tolerance):
    entry = mw.MATERIAL_LIBRARY[key]
    reference = td_library[key][variant]
    low = max(entry.frequency_range[0], reference.frequency_range[0])
    high = min(entry.frequency_range[1], reference.frequency_range[1])
    assert low < high
    frequencies = np.geomspace(low * 1.001, high / 1.001, 9)
    material = entry.material()
    ours = np.array([material.relative_permittivity(float(f)) for f in frequencies])
    theirs = np.asarray(reference.eps_model(frequencies), dtype=np.complex128)
    relative = np.abs(ours - theirs) / np.abs(theirs)
    assert float(np.max(relative)) < tolerance, (key, variant, float(np.max(relative)))
    assert math.isfinite(float(np.max(relative)))
