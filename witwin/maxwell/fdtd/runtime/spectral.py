from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from ...sources import evaluate_source_time
from ..boundary import has_complex_fields


def normalize_target_frequencies(solver, frequencies) -> tuple[float, ...]:
    del solver
    if frequencies is None:
        return ()
    if isinstance(frequencies, Iterable) and not isinstance(frequencies, (str, bytes)):
        values = tuple(float(freq) for freq in frequencies)
    else:
        values = (float(frequencies),)
    if not values:
        raise ValueError("At least one target frequency is required.")
    return values


def reset_dft_runtime_state(solver):
    solver._dft_batched_fields = {}
    solver._dft_phase_cos = None
    solver._dft_phase_sin = None
    solver._dft_phase_step_cos_values = None
    solver._dft_phase_step_sin_values = None
    solver._dft_start_steps = None
    solver._dft_end_steps = None
    solver._dft_window_normalization_values = None
    solver._dft_sample_count_values = None
    solver._dft_source_dft_real_values = None
    solver._dft_source_dft_imag_values = None


def sync_dft_entries_from_runtime_state(solver):
    if not solver._dft_entries or solver._dft_phase_cos is None:
        return

    for index, entry in enumerate(solver._dft_entries):
        entry["start_step"] = int(solver._dft_start_steps[index])
        end_step = int(solver._dft_end_steps[index])
        entry["end_step"] = None if end_step < 0 else end_step
        entry["window_normalization"] = float(solver._dft_window_normalization_values[index])
        entry["sample_count"] = int(solver._dft_sample_count_values[index])
        entry["phase_cos"] = float(solver._dft_phase_cos[index])
        entry["phase_sin"] = float(solver._dft_phase_sin[index])
        entry["phase_step_cos"] = float(solver._dft_phase_step_cos_values[index])
        entry["phase_step_sin"] = float(solver._dft_phase_step_sin_values[index])
        entry["source_dft_real"] = float(solver._dft_source_dft_real_values[index])
        entry["source_dft_imag"] = float(solver._dft_source_dft_imag_values[index])


def sync_dft_primary_runtime_state(solver):
    if not solver._dft_entries or solver._dft_phase_cos is None:
        solver.dft_sample_count = 0
        solver.dft_window_normalization = 0.0
        solver.dft_phase_step_cos = None
        solver.dft_phase_step_sin = None
        return

    solver.dft_sample_count = int(solver._dft_sample_count_values[0])
    solver.dft_window_normalization = float(solver._dft_window_normalization_values[0])
    solver.dft_phase_step_cos = float(solver._dft_phase_cos[0])
    solver.dft_phase_step_sin = float(solver._dft_phase_sin[0])


def sync_dft_legacy_state(solver):
    sync_dft_entries_from_runtime_state(solver)
    solver.dft_frequencies = tuple(entry["frequency"] for entry in solver._dft_entries)
    solver.dft_sample_counts = tuple(entry["sample_count"] for entry in solver._dft_entries)
    solver.dft_start_steps = tuple(entry["start_step"] for entry in solver._dft_entries)
    solver.dft_end_steps = tuple(entry["end_step"] for entry in solver._dft_entries)
    solver.dft_window_normalizations = tuple(entry["window_normalization"] for entry in solver._dft_entries)

    if not solver._dft_entries:
        solver.dft_frequency = None
        solver.dft_Ex_real = None
        solver.dft_Ex_imag = None
        solver.dft_Ey_real = None
        solver.dft_Ey_imag = None
        solver.dft_Ez_real = None
        solver.dft_Ez_imag = None
        solver.dft_Ex_aux_real = None
        solver.dft_Ex_aux_imag = None
        solver.dft_Ey_aux_real = None
        solver.dft_Ey_aux_imag = None
        solver.dft_Ez_aux_real = None
        solver.dft_Ez_aux_imag = None
        solver.dft_sample_count = 0
        solver.dft_start_step = None
        solver.dft_end_step = None
        solver.dft_window_normalization = 0.0
        solver.dft_phase_step_cos = None
        solver.dft_phase_step_sin = None
        reset_dft_runtime_state(solver)
        return

    primary = solver._dft_entries[0]
    primary_fields = primary["fields"]
    solver.dft_frequency = primary["frequency"]
    solver.dft_Ex_real = primary_fields["Ex"]["real"]
    solver.dft_Ex_imag = primary_fields["Ex"]["imag"]
    solver.dft_Ey_real = primary_fields["Ey"]["real"]
    solver.dft_Ey_imag = primary_fields["Ey"]["imag"]
    solver.dft_Ez_real = primary_fields["Ez"]["real"]
    solver.dft_Ez_imag = primary_fields["Ez"]["imag"]
    solver.dft_Ex_aux_real = primary_fields["Ex"]["aux_real"]
    solver.dft_Ex_aux_imag = primary_fields["Ex"]["aux_imag"]
    solver.dft_Ey_aux_real = primary_fields["Ey"]["aux_real"]
    solver.dft_Ey_aux_imag = primary_fields["Ey"]["aux_imag"]
    solver.dft_Ez_aux_real = primary_fields["Ez"]["aux_real"]
    solver.dft_Ez_aux_imag = primary_fields["Ez"]["aux_imag"]
    solver.dft_sample_count = primary["sample_count"]
    solver.dft_start_step = primary["start_step"]
    solver.dft_end_step = primary["end_step"]
    solver.dft_window_normalization = primary["window_normalization"]
    solver.dft_phase_step_cos = primary["phase_cos"]
    solver.dft_phase_step_sin = primary["phase_sin"]


def sync_observer_legacy_state(solver):
    solver.observer_frequencies = tuple(entry["frequency"] for entry in solver._observer_spectral_entries)
    solver.observer_sample_counts = tuple(entry["sample_count"] for entry in solver._observer_spectral_entries)
    solver.observer_start_steps = tuple(entry["start_step"] for entry in solver._observer_spectral_entries)
    solver.observer_end_steps = tuple(entry["end_step"] for entry in solver._observer_spectral_entries)

    if not solver._observer_spectral_entries:
        solver.observer_frequency = None
        solver.observer_start_step = None
        solver.observer_end_step = None
        solver.observer_window_normalization = 0.0
        solver.observer_sample_count = 0
        solver.observer_phase_step_cos = None
        solver.observer_phase_step_sin = None
        return

    primary = solver._observer_spectral_entries[0]
    solver.observer_frequency = primary["frequency"]
    solver.observer_start_step = primary["start_step"]
    solver.observer_end_step = primary["end_step"]
    solver.observer_window_normalization = primary["window_normalization"]
    solver.observer_sample_count = primary["sample_count"]
    solver.observer_phase_step_cos = primary["phase_step_cos"]
    solver.observer_phase_step_sin = primary["phase_step_sin"]


def sync_observer_primary_state(solver):
    if not solver._observer_spectral_entries:
        solver.observer_sample_count = 0
        solver.observer_window_normalization = 0.0
        solver.observer_phase_step_cos = None
        solver.observer_phase_step_sin = None
        return

    primary = solver._observer_spectral_entries[0]
    solver.observer_window_normalization = primary["window_normalization"]
    solver.observer_sample_count = primary["sample_count"]
    solver.observer_phase_step_cos = primary["phase_cos"]
    solver.observer_phase_step_sin = primary["phase_sin"]


def source_time_kind(solver):
    source_time = getattr(solver, "_source_time", None)
    if source_time is None:
        return ""
    if isinstance(source_time, dict):
        return str(source_time.get("kind", "")).lower()
    return str(getattr(source_time, "kind", "")).lower()


def resolve_spectral_window_type(solver, window_type):
    if source_time_kind(solver) and source_time_kind(solver) != "cw":
        return "none"
    return window_type


def compute_spectral_start_step(solver, frequency, *, window_type=None):
    if source_time_kind(solver) and source_time_kind(solver) != "cw":
        return 0
    if window_type is None:
        window_type = getattr(solver, "dft_window_type", "hanning")
    if str(window_type).lower() == "none":
        return 0

    period = 1.0 / frequency
    domain_range = getattr(solver.scene, "physical_domain_range", solver.scene.domain_range)
    domain_size = max(
        domain_range[1] - domain_range[0],
        domain_range[3] - domain_range[2],
        domain_range[5] - domain_range[4],
    )
    propagation_time = 2 * domain_size / solver.c
    transient_time = max(15 * period, propagation_time * 5)
    return int(transient_time / solver.dt)


def compute_window_weight(solver, n, start_step=None, end_step=None, window_type=None):
    if start_step is None:
        start_step = solver.dft_start_step
    if end_step is None:
        end_step = solver.dft_end_step
    if window_type is None:
        window_type = getattr(solver, "dft_window_type", "hanning")

    if start_step is None or end_step is None:
        return 1.0
    if n < start_step or n >= end_step:
        return 0.0

    total_samples = end_step - start_step
    pos = (n - start_step) / total_samples
    if window_type == "none":
        return 1.0
    if window_type == "hanning":
        return 0.5 * (1.0 - np.cos(2 * np.pi * pos))
    if window_type == "ramp":
        ramp_fraction = 0.1
        if pos < ramp_fraction:
            return 0.5 * (1.0 - np.cos(np.pi * pos / ramp_fraction))
        return 1.0
    return 1.0


def advance_phase(solver, cos_phase, sin_phase, step_cos, step_sin):
    del solver
    next_cos = cos_phase * step_cos - sin_phase * step_sin
    next_sin = sin_phase * step_cos + cos_phase * step_sin
    return next_cos, next_sin


def synchronize_device(solver):
    if torch.cuda.is_available() and str(solver.device).startswith("cuda"):
        torch.cuda.synchronize()


def enable_dft(solver, frequencies, window_type="hanning", end_step=None):
    frequency_values = normalize_target_frequencies(solver, frequencies)
    resolved_window_type = resolve_spectral_window_type(solver, window_type)
    solver.dft_enabled = True
    solver.dft_window_type = resolved_window_type
    solver._dft_entries = []
    complex_enabled = has_complex_fields(solver)
    frequency_count = len(frequency_values)
    solver._dft_batched_fields = {
        "Ex": {
            "real": torch.zeros((frequency_count,) + tuple(solver.Ex.shape), device=solver.device, dtype=solver.Ex.dtype),
            "imag": torch.zeros((frequency_count,) + tuple(solver.Ex.shape), device=solver.device, dtype=solver.Ex.dtype),
            "aux_real": (
                torch.zeros((frequency_count,) + tuple(solver.Ex.shape), device=solver.device, dtype=solver.Ex.dtype)
                if complex_enabled
                else None
            ),
            "aux_imag": (
                torch.zeros((frequency_count,) + tuple(solver.Ex.shape), device=solver.device, dtype=solver.Ex.dtype)
                if complex_enabled
                else None
            ),
        },
        "Ey": {
            "real": torch.zeros((frequency_count,) + tuple(solver.Ey.shape), device=solver.device, dtype=solver.Ey.dtype),
            "imag": torch.zeros((frequency_count,) + tuple(solver.Ey.shape), device=solver.device, dtype=solver.Ey.dtype),
            "aux_real": (
                torch.zeros((frequency_count,) + tuple(solver.Ey.shape), device=solver.device, dtype=solver.Ey.dtype)
                if complex_enabled
                else None
            ),
            "aux_imag": (
                torch.zeros((frequency_count,) + tuple(solver.Ey.shape), device=solver.device, dtype=solver.Ey.dtype)
                if complex_enabled
                else None
            ),
        },
        "Ez": {
            "real": torch.zeros((frequency_count,) + tuple(solver.Ez.shape), device=solver.device, dtype=solver.Ez.dtype),
            "imag": torch.zeros((frequency_count,) + tuple(solver.Ez.shape), device=solver.device, dtype=solver.Ez.dtype),
            "aux_real": (
                torch.zeros((frequency_count,) + tuple(solver.Ez.shape), device=solver.device, dtype=solver.Ez.dtype)
                if complex_enabled
                else None
            ),
            "aux_imag": (
                torch.zeros((frequency_count,) + tuple(solver.Ez.shape), device=solver.device, dtype=solver.Ez.dtype)
                if complex_enabled
                else None
            ),
        },
    }
    solver._dft_phase_cos = np.ones(frequency_count, dtype=np.float32)
    solver._dft_phase_sin = np.zeros(frequency_count, dtype=np.float32)
    solver._dft_phase_step_cos_values = np.empty(frequency_count, dtype=np.float32)
    solver._dft_phase_step_sin_values = np.empty(frequency_count, dtype=np.float32)
    solver._dft_start_steps = np.empty(frequency_count, dtype=np.int64)
    solver._dft_end_steps = np.empty(frequency_count, dtype=np.int64)
    solver._dft_window_normalization_values = np.zeros(frequency_count, dtype=np.float32)
    solver._dft_sample_count_values = np.zeros(frequency_count, dtype=np.int64)
    solver._dft_source_dft_real_values = np.zeros(frequency_count, dtype=np.float64)
    solver._dft_source_dft_imag_values = np.zeros(frequency_count, dtype=np.float64)

    for index, frequency in enumerate(frequency_values):
        omega_dt = 2 * np.pi * frequency * solver.dt
        solver._dft_phase_step_cos_values[index] = np.cos(omega_dt)
        solver._dft_phase_step_sin_values[index] = np.sin(omega_dt)
        solver._dft_start_steps[index] = compute_spectral_start_step(
            solver,
            frequency,
            window_type=resolved_window_type,
        )
        solver._dft_end_steps[index] = -1 if end_step is None else int(end_step)
        fields = {
            "Ex": {
                "real": solver._dft_batched_fields["Ex"]["real"][index],
                "imag": solver._dft_batched_fields["Ex"]["imag"][index],
                "aux_real": (
                    None
                    if solver._dft_batched_fields["Ex"]["aux_real"] is None
                    else solver._dft_batched_fields["Ex"]["aux_real"][index]
                ),
                "aux_imag": (
                    None
                    if solver._dft_batched_fields["Ex"]["aux_imag"] is None
                    else solver._dft_batched_fields["Ex"]["aux_imag"][index]
                ),
            },
            "Ey": {
                "real": solver._dft_batched_fields["Ey"]["real"][index],
                "imag": solver._dft_batched_fields["Ey"]["imag"][index],
                "aux_real": (
                    None
                    if solver._dft_batched_fields["Ey"]["aux_real"] is None
                    else solver._dft_batched_fields["Ey"]["aux_real"][index]
                ),
                "aux_imag": (
                    None
                    if solver._dft_batched_fields["Ey"]["aux_imag"] is None
                    else solver._dft_batched_fields["Ey"]["aux_imag"][index]
                ),
            },
            "Ez": {
                "real": solver._dft_batched_fields["Ez"]["real"][index],
                "imag": solver._dft_batched_fields["Ez"]["imag"][index],
                "aux_real": (
                    None
                    if solver._dft_batched_fields["Ez"]["aux_real"] is None
                    else solver._dft_batched_fields["Ez"]["aux_real"][index]
                ),
                "aux_imag": (
                    None
                    if solver._dft_batched_fields["Ez"]["aux_imag"] is None
                    else solver._dft_batched_fields["Ez"]["aux_imag"][index]
                ),
            },
        }
        solver._dft_entries.append(
            {
                "frequency": float(frequency),
                "start_step": int(solver._dft_start_steps[index]),
                "end_step": None if solver._dft_end_steps[index] < 0 else int(solver._dft_end_steps[index]),
                "window_normalization": 0.0,
                "sample_count": 0,
                "phase_cos": 1.0,
                "phase_sin": 0.0,
                "phase_step_cos": float(solver._dft_phase_step_cos_values[index]),
                "phase_step_sin": float(solver._dft_phase_step_sin_values[index]),
                "fields": fields,
                "source_dft_real": 0.0,
                "source_dft_imag": 0.0,
            }
        )
    sync_dft_legacy_state(solver)
    if solver.verbose:
        freq_text = ", ".join(f"{frequency / 1e9:.3f} GHz" for frequency in frequency_values)
        print(f"DFT enabled, target frequencies: {freq_text}, window: {resolved_window_type}")


def accumulate_dft(solver, n, phase_cos=None, phase_sin=None):
    del phase_cos, phase_sin
    if not solver.dft_enabled or not solver._dft_entries:
        return

    source_signal = None
    if getattr(solver, "_normalize_source", False) and solver._source_time is not None:
        source_signal = evaluate_source_time(solver._source_time, n * solver.dt)
    active = n >= solver._dft_start_steps
    active &= (solver._dft_end_steps < 0) | (n < solver._dft_end_steps)
    if solver.dft_window_type == "none":
        window_weight = active.astype(np.float32)
    else:
        total_samples = solver._dft_end_steps - solver._dft_start_steps
        if solver.dft_window_type == "hanning":
            pos = np.zeros_like(solver._dft_phase_cos, dtype=np.float32)
            valid = active & (total_samples > 0)
            pos[valid] = (n - solver._dft_start_steps[valid]) / total_samples[valid]
            window_weight = np.zeros_like(solver._dft_phase_cos, dtype=np.float32)
            window_weight[active & (solver._dft_end_steps < 0)] = 1.0
            window_weight[valid] = 0.5 * (1.0 - np.cos(2 * np.pi * pos[valid]))
        elif solver.dft_window_type == "ramp":
            pos = np.zeros_like(solver._dft_phase_cos, dtype=np.float32)
            valid = active & (total_samples > 0)
            pos[valid] = (n - solver._dft_start_steps[valid]) / total_samples[valid]
            window_weight = active.astype(np.float32)
            ramp_fraction = 0.1
            ramp_mask = valid & (pos < ramp_fraction)
            window_weight[ramp_mask] = 0.5 * (1.0 - np.cos(np.pi * pos[ramp_mask] / ramp_fraction))
        else:
            window_weight = active.astype(np.float32)

    weighted_cos = window_weight * solver._dft_phase_cos
    weighted_sin = window_weight * solver._dft_phase_sin
    weighted_cos_tensor = torch.as_tensor(weighted_cos, device=solver.device, dtype=torch.float32)
    weighted_sin_tensor = torch.as_tensor(weighted_sin, device=solver.device, dtype=torch.float32)

    if source_signal is not None:
        solver._dft_source_dft_real_values += float(source_signal) * weighted_cos
        solver._dft_source_dft_imag_values += float(source_signal) * weighted_sin

    solver.fdtd_module.accumulateRunningDftYee3DBatched(
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        ExRealAccum=solver._dft_batched_fields["Ex"]["real"],
        ExImagAccum=solver._dft_batched_fields["Ex"]["imag"],
        EyRealAccum=solver._dft_batched_fields["Ey"]["real"],
        EyImagAccum=solver._dft_batched_fields["Ey"]["imag"],
        EzRealAccum=solver._dft_batched_fields["Ez"]["real"],
        EzImagAccum=solver._dft_batched_fields["Ez"]["imag"],
        weightedCos=weighted_cos_tensor,
        weightedSin=weighted_sin_tensor,
    ).launchRaw()

    if has_complex_fields(solver):
        solver.fdtd_module.accumulateRunningDftYee3DBatched(
            Ex=solver.Ex_imag,
            Ey=solver.Ey_imag,
            Ez=solver.Ez_imag,
            ExRealAccum=solver._dft_batched_fields["Ex"]["aux_real"],
            ExImagAccum=solver._dft_batched_fields["Ex"]["aux_imag"],
            EyRealAccum=solver._dft_batched_fields["Ey"]["aux_real"],
            EyImagAccum=solver._dft_batched_fields["Ey"]["aux_imag"],
            EzRealAccum=solver._dft_batched_fields["Ez"]["aux_real"],
            EzImagAccum=solver._dft_batched_fields["Ez"]["aux_imag"],
            weightedCos=weighted_cos_tensor,
            weightedSin=weighted_sin_tensor,
        ).launchRaw()

    solver._dft_window_normalization_values += window_weight
    solver._dft_sample_count_values += active.astype(np.int64)

    next_cos = (
        solver._dft_phase_cos * solver._dft_phase_step_cos_values
        - solver._dft_phase_sin * solver._dft_phase_step_sin_values
    )
    next_sin = (
        solver._dft_phase_sin * solver._dft_phase_step_cos_values
        + solver._dft_phase_cos * solver._dft_phase_step_sin_values
    )
    solver._dft_phase_cos = next_cos.astype(np.float32, copy=False)
    solver._dft_phase_sin = next_sin.astype(np.float32, copy=False)
    sync_dft_primary_runtime_state(solver)


def _dft_window_weight(n, window_type, active, start, end):
    """The per-step spectral window weight vector, matching accumulate_dft."""
    if window_type == "none":
        return active.astype(np.float32)
    total = end - start
    valid = active & (total > 0)
    pos = np.zeros_like(start, dtype=np.float32)
    pos[valid] = (n - start[valid]) / total[valid]
    if window_type == "hanning":
        weight = np.zeros_like(start, dtype=np.float32)
        weight[active & (end < 0)] = 1.0
        weight[valid] = 0.5 * (1.0 - np.cos(2 * np.pi * pos[valid]))
        return weight
    if window_type == "ramp":
        weight = active.astype(np.float32)
        ramp_fraction = 0.1
        ramp_mask = valid & (pos < ramp_fraction)
        weight[ramp_mask] = 0.5 * (1.0 - np.cos(np.pi * pos[ramp_mask] / ramp_fraction))
        return weight
    return active.astype(np.float32)


def build_dft_step_tables(solver, time_steps):
    """Precompute the full per-step running-DFT weight table on the GPU.

    Each row ``table[n]`` is the ``window_weight(n) * phase(n)`` vector that
    accumulate_dft computes at step ``n``; it is built with the identical float32
    phase recurrence and window, so a GPU-driven accumulation that gathers rows
    by a device step counter is bit-identical to the per-step host path -- but
    with no per-step host arithmetic or host->device transfer (which makes the
    DFT accumulation capturable and cheaper). Also fixes the final window
    normalization / sample count / source-DFT to their full-run sums (the values
    the host path would reach, and what an early shutoff restores anyway).

    Returns False (caller keeps the host path) for the complex-field DFT, which
    is out of the Option-A/Stage-1 graph scope.
    """
    if not solver.dft_enabled or not solver._dft_entries or has_complex_fields(solver):
        return False
    frequency_count = len(solver._dft_entries)
    steps = int(time_steps)
    weighted_cos = np.zeros((steps, frequency_count), dtype=np.float32)
    weighted_sin = np.zeros((steps, frequency_count), dtype=np.float32)
    window_norm = np.zeros(frequency_count, dtype=np.float32)
    sample_count = np.zeros(frequency_count, dtype=np.int64)
    src_real = np.zeros(frequency_count, dtype=np.float64)
    src_imag = np.zeros(frequency_count, dtype=np.float64)
    normalize = bool(getattr(solver, "_normalize_source", False)) and solver._source_time is not None

    phase_cos = np.ones(frequency_count, dtype=np.float32)
    phase_sin = np.zeros(frequency_count, dtype=np.float32)
    step_cos = solver._dft_phase_step_cos_values
    step_sin = solver._dft_phase_step_sin_values
    start = solver._dft_start_steps
    end = solver._dft_end_steps
    window_type = solver.dft_window_type
    for n in range(steps):
        active = (n >= start) & ((end < 0) | (n < end))
        weight = _dft_window_weight(n, window_type, active, start, end)
        wc = weight * phase_cos
        ws = weight * phase_sin
        weighted_cos[n] = wc
        weighted_sin[n] = ws
        window_norm += weight
        sample_count += active.astype(np.int64)
        if normalize:
            signal = float(evaluate_source_time(solver._source_time, n * solver.dt))
            src_real += signal * wc
            src_imag += signal * ws
        next_cos = phase_cos * step_cos - phase_sin * step_sin
        next_sin = phase_sin * step_cos + phase_cos * step_sin
        phase_cos = next_cos.astype(np.float32, copy=False)
        phase_sin = next_sin.astype(np.float32, copy=False)

    solver._dft_weighted_cos_table = torch.as_tensor(weighted_cos, device=solver.device)
    solver._dft_weighted_sin_table = torch.as_tensor(weighted_sin, device=solver.device)
    solver._dft_weighted_cos = torch.zeros(frequency_count, device=solver.device, dtype=torch.float32)
    solver._dft_weighted_sin = torch.zeros(frequency_count, device=solver.device, dtype=torch.float32)
    solver._dft_step = torch.zeros(1, device=solver.device, dtype=torch.int64)
    solver._dft_window_normalization_values = window_norm
    solver._dft_sample_count_values = sample_count
    if normalize:
        solver._dft_source_dft_real_values = src_real
        solver._dft_source_dft_imag_values = src_imag
    sync_dft_primary_runtime_state(solver)
    return True


def accumulate_dft_gpu(solver):
    """GPU-driven running-DFT accumulation: gather the current step's weight row
    from the precomputed table by the device step counter and accumulate. No host
    arithmetic, no host->device transfer, and capturable into a CUDA graph."""
    if not solver.dft_enabled or not solver._dft_entries:
        return
    solver._dft_weighted_cos.copy_(
        solver._dft_weighted_cos_table.index_select(0, solver._dft_step).squeeze(0)
    )
    solver._dft_weighted_sin.copy_(
        solver._dft_weighted_sin_table.index_select(0, solver._dft_step).squeeze(0)
    )
    solver.fdtd_module.accumulateRunningDftYee3DBatched(
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        ExRealAccum=solver._dft_batched_fields["Ex"]["real"],
        ExImagAccum=solver._dft_batched_fields["Ex"]["imag"],
        EyRealAccum=solver._dft_batched_fields["Ey"]["real"],
        EyImagAccum=solver._dft_batched_fields["Ey"]["imag"],
        EzRealAccum=solver._dft_batched_fields["Ez"]["real"],
        EzImagAccum=solver._dft_batched_fields["Ez"]["imag"],
        weightedCos=solver._dft_weighted_cos,
        weightedSin=solver._dft_weighted_sin,
    ).launchRaw()
    solver._dft_step.add_(1)
