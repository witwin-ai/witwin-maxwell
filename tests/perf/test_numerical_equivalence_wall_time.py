"""Wall time of the 85000-leaf numerical-equivalence comparison, on an idle host or device."""

import pytest

from tests.support.test_numerical_equivalence import DEVICES, compare_85000_leaves

pytestmark = pytest.mark.perf


@pytest.mark.parametrize("device", DEVICES)
def test_85000_leaves_wall_time(device, monkeypatch):
    elapsed = compare_85000_leaves(device, monkeypatch)
    assert elapsed < 5, f"85000 leaves took {elapsed:.3f}s on {device}"
