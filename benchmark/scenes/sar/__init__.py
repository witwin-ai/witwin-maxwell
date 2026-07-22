"""Canonical phantom SAR benchmark family (redistributable geometry only).

Each module exposes a ``build_scene(...)`` builder and a ``SCENARIO``
:class:`ScenarioDefinition`. Tissue dielectric/mass values are the published-class
canonical numbers documented in ``_tissue.py``; no licensed anatomical model is
distributed.
"""

from __future__ import annotations

from benchmark.scenes.sar import (
    antenna_near_phantom,
    layered_slab,
    one_gram_cube,
    uniform_lossy_cube,
)

SAR_SCENARIOS = (
    uniform_lossy_cube.SCENARIO,
    layered_slab.SCENARIO,
    one_gram_cube.SCENARIO,
    antenna_near_phantom.SCENARIO,
)

__all__ = [
    "SAR_SCENARIOS",
    "antenna_near_phantom",
    "layered_slab",
    "one_gram_cube",
    "uniform_lossy_cube",
]
