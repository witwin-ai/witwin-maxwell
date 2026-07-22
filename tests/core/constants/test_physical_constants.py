"""Pin the shared vacuum constants in ``witwin.maxwell.constants``.

``ETA_0`` feeds every absolute normalization in the package: soft plane-wave and
Gaussian-beam unit-power scales, modal/TEM impedances, and the far-field
constants. A refactor that "derives" it as ``MU_0 * C_0`` or ``1 / (EPSILON_0 *
C_0)`` looks equivalent but is not: both ``MU_0`` and ``EPSILON_0`` are quoted
to only twelve significant digits, so either product lands on
``376.730313667...`` instead of the CODATA 2018 recommended ``376.730313668``.
That is a 3e-12 relative shift applied silently to every derived amplitude.

These assertions are exact-value pins, not tolerance checks: their whole purpose
is to catch a change at the last quoted digit.
"""

from __future__ import annotations

import pytest
from witwin.core.material import VACUUM_PERMITTIVITY

from witwin.maxwell.constants import C_0, EPSILON_0, ETA_0, MU_0


def test_vacuum_constants_are_codata_literals():
    assert C_0 == 299792458.0  # exact by SI definition
    assert MU_0 == 1.25663706212e-6
    assert ETA_0 == 376.730313668
    # EPSILON_0 must stay bit-identical to witwin.core so material compilation
    # and the solver runtimes agree exactly.
    assert EPSILON_0 == VACUUM_PERMITTIVITY


def test_eta0_is_not_the_truncated_product_of_mu0_and_c0():
    # The excluded derivations, spelled out so the reason for the literal is
    # executable rather than only documented. Both are wrong in the last digit.
    assert MU_0 * C_0 != ETA_0
    assert 1.0 / (EPSILON_0 * C_0) != ETA_0
    # ... but only in the last quoted digit: the literal is still the physical
    # impedance to within the precision the inputs actually carry.
    assert MU_0 * C_0 == pytest.approx(ETA_0, rel=1e-11)
    assert 1.0 / (EPSILON_0 * C_0) == pytest.approx(ETA_0, rel=1e-11)
