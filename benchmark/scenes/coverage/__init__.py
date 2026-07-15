"""Feature-coverage scenarios: one family module per exportable feature group.

Imports are unconditional on purpose: a missing or broken family module must fail
loudly at import time rather than silently shrink the campaign.
"""

from __future__ import annotations

from benchmark.scenes.coverage.boundaries import BOUNDARY_COVERAGE_SCENARIOS
from benchmark.scenes.coverage.grid_geometry import GRID_GEOMETRY_COVERAGE_SCENARIOS
from benchmark.scenes.coverage.media import MEDIA_COVERAGE_SCENARIOS
from benchmark.scenes.coverage.postprocess import POSTPROCESS_COVERAGE_SCENARIOS
from benchmark.scenes.coverage.sources import SOURCE_COVERAGE_SCENARIOS


COVERAGE_SCENARIOS = (
    *SOURCE_COVERAGE_SCENARIOS,
    *MEDIA_COVERAGE_SCENARIOS,
    *BOUNDARY_COVERAGE_SCENARIOS,
    *GRID_GEOMETRY_COVERAGE_SCENARIOS,
    *POSTPROCESS_COVERAGE_SCENARIOS,
)

__all__ = [
    "BOUNDARY_COVERAGE_SCENARIOS",
    "COVERAGE_SCENARIOS",
    "GRID_GEOMETRY_COVERAGE_SCENARIOS",
    "MEDIA_COVERAGE_SCENARIOS",
    "POSTPROCESS_COVERAGE_SCENARIOS",
    "SOURCE_COVERAGE_SCENARIOS",
]
