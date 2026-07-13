from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import witwin.maxwell as mw


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str
    builder: Callable[[], mw.Scene]
    frequencies: tuple[float, ...]
    display_monitor: str
    display_component: str
    run_time_factor: float = 8.0
    maxwell_alignment: dict | None = None
    solver: str = "fdtd"


@dataclass
class ScenarioMetrics:
    name: str
    description: str
    frequencies: tuple[float, ...]
    maxwell_time_s: float
    tidy3d_cache_hit: bool
    field_l2: float
    field_linf: float
    field_corr: float
    flux_error: float | None
    compared_monitor: str
    compared_component: str
    material_source_plot: Path
    field_plot: Path
    updated_at: str
    notes: list[str] = field(default_factory=list)
    per_frequency: list[dict[str, float]] = field(default_factory=list)
