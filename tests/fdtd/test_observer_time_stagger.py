import math
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.adjoint.core import _build_spectral_weight_schedule
from witwin.maxwell.fdtd.adjoint.seeds import (
    _schedule_to_tensor_pack,
    _shift_observer_schedule_for_field,
)
from witwin.maxwell.fdtd.observers import accumulate_observers


class _Launch:
    def launchRaw(self):
        return None


class _Recorder:
    def __init__(self):
        self.calls = []

    def accumulatePointObservers3D(self, **kwargs):
        self.calls.append(kwargs)
        return _Launch()

    def accumulatePlaneObserver3D(self, **kwargs):
        self.calls.append(kwargs)
        return _Launch()


def test_spectral_observers_use_physical_yee_time_locations():
    recorder = _Recorder()
    frequency = 2.0
    dt = 0.025
    entry = {
        "frequency": frequency,
        "start_step": 0,
        "end_step": 1,
        "window_normalization": 0.0,
        "sample_count": 0,
        "phase_cos": 1.0,
        "phase_sin": 0.0,
        "phase_step_cos": math.cos(2.0 * math.pi * frequency * dt),
        "phase_step_sin": math.sin(2.0 * math.pi * frequency * dt),
    }
    groups = []
    for field_name in ("Ex", "Hy"):
        groups.append(
            (
                {
                    "field_name": field_name,
                    "real": torch.zeros((1, 2, 2)),
                    "imag": torch.zeros((1, 2, 2)),
                    "axis_code": 2,
                    "plane_index": 0,
                },
                0,
            )
        )
    solver = SimpleNamespace(
        observers_enabled=True,
        _normalize_source=False,
        _accumulate_source_spectrum=False,
        _observer_spectral_entries=[entry],
        _observer_point_groups_by_frequency=[[]],
        _observer_plane_groups_by_frequency=[groups],
        observer_window_type="rectangular",
        dt=dt,
        fdtd_module=recorder,
        Ex=torch.zeros((1, 1, 1)),
        Hy=torch.zeros((1, 1, 1)),
        complex_fields_enabled=False,
        _compute_window_weight=lambda *args, **kwargs: 1.0,
        _advance_phase=lambda cos, sin, step_cos, step_sin: (
            cos * step_cos - sin * step_sin,
            sin * step_cos + cos * step_sin,
        ),
        _sync_observer_primary_state=lambda: None,
    )

    accumulate_observers(solver, 0)

    electric, magnetic = recorder.calls
    electric_phase = 2.0 * math.pi * frequency * dt
    magnetic_phase = 0.5 * electric_phase
    assert electric["weightedCos"] == pytest.approx(math.cos(electric_phase))
    assert electric["weightedSin"] == pytest.approx(math.sin(electric_phase))
    assert magnetic["weightedCos"] == pytest.approx(math.cos(magnetic_phase))
    assert magnetic["weightedSin"] == pytest.approx(math.sin(magnetic_phase))
    assert entry["sample_count"] == 1


@pytest.mark.parametrize(
    ("observer_kind", "field_name", "time_offset"),
    (
        ("point", "Ex", 1.0),
        ("point", "Hy", 0.5),
        ("plane", "Ez", 1.0),
        ("plane", "Hx", 0.5),
    ),
)
def test_observer_adjoint_schedule_is_exact_transpose_of_forward_yee_dft(
    observer_kind,
    field_name,
    time_offset,
):
    recorder = _Recorder()
    frequency = 2.75
    dt = 0.037
    time_steps = 4
    phase_step = 2.0 * math.pi * frequency * dt
    entry = {
        "frequency": frequency,
        "start_step": 0,
        "end_step": time_steps,
        "window_normalization": 0.0,
        "sample_count": 0,
        "phase_cos": 1.0,
        "phase_sin": 0.0,
        "phase_step_cos": math.cos(phase_step),
        "phase_step_sin": math.sin(phase_step),
    }
    group = {
        "field_name": field_name,
        "real": torch.zeros((1, 2, 2)),
        "imag": torch.zeros((1, 2, 2)),
        "axis_code": 2,
        "plane_index": 0,
    }
    point_groups = [[] for _ in range(1)]
    plane_groups = [[] for _ in range(1)]
    if observer_kind == "point":
        group.update(
            point_i=torch.tensor([0], dtype=torch.int32),
            point_j=torch.tensor([0], dtype=torch.int32),
            point_k=torch.tensor([0], dtype=torch.int32),
        )
        point_groups[0].append((group, 0))
    else:
        plane_groups[0].append((group, 0))

    solver = SimpleNamespace(
        observers_enabled=True,
        _normalize_source=False,
        _accumulate_source_spectrum=False,
        _observer_spectral_entries=[entry],
        _observer_point_groups_by_frequency=point_groups,
        _observer_plane_groups_by_frequency=plane_groups,
        observer_window_type="none",
        dt=dt,
        fdtd_module=recorder,
        complex_fields_enabled=False,
        _compute_window_weight=lambda *args, **kwargs: 1.0,
        _advance_phase=lambda cos, sin, step_cos, step_sin: (
            cos * step_cos - sin * step_sin,
            sin * step_cos + cos * step_sin,
        ),
        _sync_observer_primary_state=lambda: None,
        **{field_name: torch.zeros((1, 1, 1))},
    )

    base_schedule = _build_spectral_weight_schedule(
        [entry],
        time_steps=time_steps,
        window_type="none",
    )
    schedule_pack = _schedule_to_tensor_pack(
        base_schedule,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    entry_indices = torch.tensor([0], dtype=torch.long)
    adjoint_cos, adjoint_sin = _shift_observer_schedule_for_field(
        schedule_pack.cos,
        schedule_pack.sin,
        solver=solver,
        entry_indices=entry_indices,
        state_field_name=field_name,
    )

    for step in range(time_steps):
        accumulate_observers(solver, step)
    forward_cos = torch.tensor(
        [call["weightedCos"] for call in recorder.calls], dtype=torch.float64
    )
    forward_sin = torch.tensor(
        [call["weightedSin"] for call in recorder.calls], dtype=torch.float64
    )

    physical_times = (
        torch.arange(time_steps, dtype=torch.float64) + time_offset
    ) * dt
    expected_phase = 2.0 * math.pi * frequency * physical_times
    torch.testing.assert_close(
        forward_cos, torch.cos(expected_phase), rtol=2.0e-14, atol=2.0e-14
    )
    torch.testing.assert_close(
        forward_sin, torch.sin(expected_phase), rtol=2.0e-14, atol=2.0e-14
    )
    torch.testing.assert_close(adjoint_cos[0], forward_cos, rtol=2.0e-14, atol=2.0e-14)
    torch.testing.assert_close(adjoint_sin[0], forward_sin, rtol=2.0e-14, atol=2.0e-14)

    field_samples = torch.tensor((0.25, -0.5, 0.75, -1.0), dtype=torch.float64)
    output_seed = torch.tensor((0.7, -0.4), dtype=torch.float64)
    forward_output = torch.stack(
        (torch.dot(field_samples, forward_cos), torch.dot(field_samples, forward_sin))
    )
    transposed_seed = output_seed[0] * adjoint_cos[0] + output_seed[1] * adjoint_sin[0]
    torch.testing.assert_close(
        torch.dot(forward_output, output_seed),
        torch.dot(field_samples, transposed_seed),
        rtol=0.0,
        atol=2.0e-15,
    )
