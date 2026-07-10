from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


_BACKWARD_PROFILE_SECTION_NAMES = (
    "seed_build",
    "segment_replay",
    "seed_injection",
    "state_clone",
    "step_forward",
    "step_vjp",
    "material_pullback",
)


@dataclass(frozen=True)
class _ReverseStepResult:
    pre_step_adjoint: dict[str, torch.Tensor]
    grad_eps_ex: torch.Tensor
    grad_eps_ey: torch.Tensor
    grad_eps_ez: torch.Tensor
    backend: str
    source_adjoint_state: dict[str, torch.Tensor] | None = None
    magnetic_output_adjoint: dict[str, torch.Tensor] | None = None
    # Kerr chi3 gradient channel; populated only when the reverse step ran with
    # chi3 leaves (Kerr-enabled solvers on the torch-VJP backend).
    grad_chi3_ex: torch.Tensor | None = None
    grad_chi3_ey: torch.Tensor | None = None
    grad_chi3_ez: torch.Tensor | None = None


@dataclass
class _BackwardProfiler:
    enabled: bool
    device: torch.device | None
    uses_cuda_events: bool = False
    sections_ms: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _BACKWARD_PROFILE_SECTION_NAMES}
    )
    counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _BACKWARD_PROFILE_SECTION_NAMES}
    )
    _pending_events: list[tuple[str, Any, Any]] = field(default_factory=list)
    _total_wall_start: float | None = None
    total_wall_ms: float = 0.0
    reverse_backend_counts: dict[str, int] = field(default_factory=dict)
    seed_injection_backend: str = "none"
    seed_batch_counts: dict[str, int] = field(default_factory=lambda: {"dense": 0, "point": 0, "plane": 0})
    material_pullback_backend: str = "none"

    def __post_init__(self):
        self.uses_cuda_events = (
            self.enabled
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    def start_total(self):
        if self.enabled:
            self._total_wall_start = time.perf_counter()

    def stop_total(self):
        if self.enabled and self._total_wall_start is not None:
            self.total_wall_ms = (time.perf_counter() - self._total_wall_start) * 1000.0
            self._total_wall_start = None

    def section(self, name: str):
        return _BackwardProfileSection(self, name)

    def record_reverse_backend(self, backend: str):
        if not self.enabled:
            return
        self.reverse_backend_counts[backend] = int(self.reverse_backend_counts.get(backend, 0) + 1)

    def record_seed_runtime(self, backend: str, *, dense: int, point: int, plane: int):
        if not self.enabled:
            return
        self.seed_injection_backend = str(backend)
        self.seed_batch_counts = {
            "dense": int(dense),
            "point": int(point),
            "plane": int(plane),
        }

    def record_material_pullback_backend(self, backend: str):
        if not self.enabled:
            return
        self.material_pullback_backend = str(backend)

    def finalize(self):
        if not self.enabled or not self.uses_cuda_events or not self._pending_events:
            return
        torch.cuda.synchronize(self.device)
        for name, start_event, end_event in self._pending_events:
            self.sections_ms[name] += float(start_event.elapsed_time(end_event))
        self._pending_events.clear()

    def summary(
        self,
        *,
        steps: int,
        segments: int,
        checkpoint_stride: int,
    ) -> dict[str, Any]:
        self.finalize()
        mean_section_ms = {
            name: (self.sections_ms[name] / self.counts[name] if self.counts[name] else 0.0)
            for name in _BACKWARD_PROFILE_SECTION_NAMES
        }
        return {
            "timer": "cuda_event" if self.uses_cuda_events else "perf_counter",
            "steps": int(steps),
            "segments": int(segments),
            "checkpoint_stride": int(checkpoint_stride),
            "total_wall_ms": float(self.total_wall_ms),
            "sections_ms": {name: float(self.sections_ms[name]) for name in _BACKWARD_PROFILE_SECTION_NAMES},
            "counts": {name: int(self.counts[name]) for name in _BACKWARD_PROFILE_SECTION_NAMES},
            "mean_section_ms": {name: float(mean_section_ms[name]) for name in _BACKWARD_PROFILE_SECTION_NAMES},
            "reverse_backend_counts": {name: int(count) for name, count in self.reverse_backend_counts.items()},
            "seed_injection_backend": self.seed_injection_backend,
            "seed_batch_counts": {name: int(count) for name, count in self.seed_batch_counts.items()},
            "material_pullback_backend": self.material_pullback_backend,
        }


class _BackwardProfileSection:
    def __init__(self, profiler: _BackwardProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self._start_event = None
        self._end_event = None
        self._wall_start = None

    def __enter__(self):
        if not self.profiler.enabled:
            return self
        if self.profiler.uses_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._wall_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.profiler.enabled:
            return False
        self.profiler.counts[self.name] += 1
        if self.profiler.uses_cuda_events:
            self._end_event.record()
            self.profiler._pending_events.append((self.name, self._start_event, self._end_event))
        else:
            self.profiler.sections_ms[self.name] += (time.perf_counter() - self._wall_start) * 1000.0
        return False


def _empty_backward_profile(*, timer: str, steps: int, checkpoint_stride: int) -> dict[str, Any]:
    return {
        "timer": timer,
        "steps": int(steps),
        "segments": 0,
        "checkpoint_stride": int(checkpoint_stride),
        "total_wall_ms": 0.0,
        "sections_ms": {name: 0.0 for name in _BACKWARD_PROFILE_SECTION_NAMES},
        "counts": {name: 0 for name in _BACKWARD_PROFILE_SECTION_NAMES},
        "mean_section_ms": {name: 0.0 for name in _BACKWARD_PROFILE_SECTION_NAMES},
        "reverse_backend_counts": {},
        "seed_injection_backend": "none",
        "seed_batch_counts": {"dense": 0, "point": 0, "plane": 0},
        "material_pullback_backend": "none",
    }


def _clone_backward_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    cloned = {}
    for key, value in profile.items():
        cloned[key] = dict(value) if isinstance(value, dict) else value
    return cloned
