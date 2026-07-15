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
    reference_solver: str = "tidy3d"
    normalize_source: bool | None = None
    scalar_observable: str | None = None
    compare_flux: bool = True
    spectral_reference_index: int | None = None
    compare_magnitude: bool = False
    comparison_components: tuple[str, ...] | None = None


@dataclass
class ScenarioMetrics:
    name: str
    description: str
    frequencies: tuple[float, ...]
    maxwell_time_s: float
    tidy3d_cache_hit: bool | None
    field_l2: float
    field_shape_l2: float
    field_linf: float
    field_corr: float
    flux_error: float | None
    compared_monitor: str
    compared_component: str
    material_source_plot: Path
    field_plot: Path
    updated_at: str
    maxwell_ms_per_step: float | None = None
    maxwell_steps_per_second: float | None = None
    maxwell_dft_samples: int | None = None
    maxwell_peak_gpu_memory_mb: float | None = None
    diagnostic_plot: Path | None = None
    scalar_plot: Path | None = None
    notes: list[str] = field(default_factory=list)
    per_frequency: list[dict[str, float]] = field(default_factory=list)
    scalar_metrics: list[dict[str, object]] = field(default_factory=list)
