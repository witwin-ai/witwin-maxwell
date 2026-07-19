"""Determinism of the coax current-contour half-grid snap (F7a).

``snap_contour_half`` maps the requested contour half-width onto the Yee
transverse half-grid for a given ``dx`` so the coax bench builds across refinement
tiers. The snap must be deterministic and the snapped edge must land exactly on a
half-grid coordinate (node + dx/2); the returned snap distance is what the
validation harness persists into the coax artifact per tier.
"""

from __future__ import annotations

import pytest

from benchmark.scenes.rf.coax_thru import (
    CONTOUR_HALF,
    DOMAIN_TRANSVERSE,
    snap_contour_half,
)


@pytest.mark.parametrize("dx", [0.0025, 0.005, 0.01])
def test_snap_is_deterministic(dx: float) -> None:
    first = snap_contour_half(dx)
    second = snap_contour_half(dx)
    assert first == second


@pytest.mark.parametrize("dx", [0.0025, 0.005, 0.01])
def test_snapped_edge_on_half_grid(dx: float) -> None:
    half, distance = snap_contour_half(dx)
    # The snapped edge +half must equal a half-grid coordinate lo + (n + 0.5)*dx.
    lo = -DOMAIN_TRANSVERSE
    offset = (half - lo) / dx - 0.5
    assert offset == pytest.approx(round(offset), abs=1e-6)
    # The reported distance is the true displacement from the requested target.
    assert distance == pytest.approx(abs(half - CONTOUR_HALF), abs=1e-12)
    assert distance <= 0.5 * dx + 1e-12
