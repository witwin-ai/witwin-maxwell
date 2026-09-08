from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

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
    for component in ("Ex", "Ey", "Ez"):
        for part in ("real", "imag", "aux_real", "aux_imag"):
            setattr(solver, f"dft_{component}_{part}", primary_fields.get(component, {}).get(part))
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


def build_observer_spectral_entry(solver, frequency, *, end_step, window_type):
    """Running-DFT state for one observer frequency.

    Carries the per-step phase recurrence step ``(cos, sin)(omega*dt)`` and the
    constant magnetic half-step retard rotation ``(cos, sin)(-0.5*omega*dt)``
    (see :func:`observer_stagger_weights`), both precomputed once here so the
    per-step accumulation performs no trigonometric evaluation.
    """
    frequency = float(frequency)
    omega_dt = 2 * np.pi * frequency * solver.dt
    h_phase_offset = -0.5 * 2.0 * np.pi * frequency * solver.dt
    return {
        "frequency": frequency,
        "start_step": compute_spectral_start_step(
            solver,
            frequency,
            window_type=window_type,
        ),
        "end_step": end_step,
        "window_normalization": 0.0,
        "sample_count": 0,
        "phase_cos": 1.0,
        "phase_sin": 0.0,
        "phase_step_cos": np.cos(omega_dt),
        "phase_step_sin": np.sin(omega_dt),
        "h_offset_cos": np.cos(h_phase_offset),
        "h_offset_sin": np.sin(h_phase_offset),
        "source_dft_real": 0.0,
        "source_dft_imag": 0.0,
    }


def observer_stagger_weights(entry, field_name, weighted_cos, weighted_sin):
    """Yee time-stagger convention for spectral observers.

    The electric observers accumulate at the plain running-DFT step phase
    (offset 0), colocated with the full-field DFT accumulator and the
    source-current DFT: E^(n+1) is labelled at (n+1)dt. The magnetic
    observers carry an extra -0.5*omega*dt phase (a half-step retard), so
    H^(n+1/2) is labelled at (n+1/2)dt. The E-H *relative* label offset is
    therefore exactly +1/2 step -- the physical Yee stagger the
    time-averaged Poynting cross term S = 1/2 Re(E x H*) requires. The
    common step-phase cancels in S, so only the +1/2 relative offset is
    observable; retarding H (rather than advancing E) keeps E colocated
    with the plain DFT slices used everywhere else. The retard rotation
    ``(h_offset_cos, h_offset_sin)`` is an entry constant precomputed by
    :func:`build_observer_spectral_entry`.
    """
    if field_name.startswith("E"):
        return weighted_cos, weighted_sin
    offset_cos = entry["h_offset_cos"]
    offset_sin = entry["h_offset_sin"]
    return (
        weighted_cos * offset_cos - weighted_sin * offset_sin,
        weighted_sin * offset_cos + weighted_cos * offset_sin,
    )


def advance_observer_entry_phases(solver):
    """Advance every observer entry's running-DFT phase by one time step."""
    for entry in solver._observer_spectral_entries:
        entry["phase_cos"], entry["phase_sin"] = advance_phase(
            solver,
            entry["phase_cos"],
            entry["phase_sin"],
            entry["phase_step_cos"],
            entry["phase_step_sin"],
        )


def source_time_kind(solver):
    plan = getattr(solver, "_time_plan", None)
    if plan is not None:
        return "cw" if plan["response"] == "cw_steady_state" else "full_response"
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

    plan = getattr(solver, "_time_plan", None)
    if plan is not None:
        return plan["spectral_start_step"]

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


def _dft_buffer(solver, previous, shape, dtype, *, value=None):
    if (getattr(solver, "_dft_reuse_buffers", False) and isinstance(previous, torch.Tensor)
            and tuple(previous.shape) == tuple(shape) and previous.dtype == dtype
            and previous.device == solver.Ex.device):
        target = previous
    else:
        target = torch.empty(shape, device=solver.device, dtype=dtype)
        solver._dft_buffer_generation = getattr(solver, "_dft_buffer_generation", 0) + 1
    if value is None:
        target.zero_()
    else:
        target.copy_(value)
    return target


def enable_dft(solver, frequencies, window_type="hanning", end_step=None, components=("Ex", "Ey", "Ez"), reuse_buffers=False):
    frequency_values = normalize_target_frequencies(solver, frequencies)
    resolved_window_type = resolve_spectral_window_type(solver, window_type)
    solver.dft_enabled = True
    solver.dft_window_type = resolved_window_type
    solver._dft_entries = []
    complex_enabled = has_complex_fields(solver)
    frequency_count = len(frequency_values)
    components = tuple(components)
    if not components or len(set(components)) != len(components) or any(c not in ("Ex", "Ey", "Ez") for c in components):
        raise ValueError("DFT components must be a nonempty unique selection of Ex, Ey, Ez.")
    solver._dft_reuse_buffers = reuse_buffers
    previous = solver._dft_batched_fields
    solver._dft_batched_fields = {
        component: {
            part: (_dft_buffer(solver, previous.get(component, {}).get(part),
                              (frequency_count,) + tuple(getattr(solver, component).shape),
                              getattr(solver, component).dtype)
                   if not part.startswith("aux_") or complex_enabled else None)
            for part in ("real", "imag", "aux_real", "aux_imag")
        } for component in components
    }
    # Zero-extent descriptors omit a component in the existing fused CUDA
    # kernel without allocating an unwanted full-domain accumulator.
    solver._dft_empty_field = torch.empty(0, device=solver.device, dtype=solver.Ex.dtype)
    solver._dft_empty_accum = solver._dft_empty_field.view(1, 0).expand(frequency_count, 0)
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
        fields = {component: {part: None if value is None else value[index]
                              for part, value in buffers.items()}
                  for component, buffers in solver._dft_batched_fields.items()}
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
    if getattr(solver, "_normalize_source", False) and solver._normalization_source_time is not None:
        from ...sources import evaluate_source_time_normalization

        source_signal = evaluate_source_time_normalization(solver._normalization_source_time, n * solver.dt)
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

    _launch_dft_accumulation(solver, weighted_cos_tensor, weighted_sin_tensor)
    if has_complex_fields(solver):
        _launch_dft_accumulation(solver, weighted_cos_tensor, weighted_sin_tensor, auxiliary=True)

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
    DFT accumulation capturable and cheaper). Eager and graph-captured runs both
    use it whenever it can be built. Also fixes the final window normalization /
    sample count / source-DFT to their full-run sums (the values the host path
    would reach, and what an early shutoff restores anyway).

    Returns False (caller keeps the host path) for the complex-field DFT, whose
    real/imaginary split accumulation the table path does not express.
    """
    if not solver.dft_enabled or not solver._dft_entries or has_complex_fields(solver):
        return False
    frequency_count = len(solver._dft_entries)
    steps = int(time_steps)
    # Cache only scalar waveform declarations. Arbitrary tensor-valued
    # waveforms rebuild through the existing table compiler.
    normalize = bool(getattr(solver, "_normalize_source", False)) and solver._normalization_source_time is not None
    waveform = getattr(solver._normalization_source_time, "__dict__", None) if normalize else {}
    scalar_waveform = waveform is not None and all(
        value is None or isinstance(value, (str, int, float, bool)) for value in waveform.values())
    key = (steps, tuple(entry["frequency"] for entry in solver._dft_entries), solver.dt,
           solver.dft_window_type, tuple(solver._dft_start_steps), tuple(solver._dft_end_steps),
           type(solver._normalization_source_time) if normalize else None,
           tuple(sorted(waveform.items()))) if scalar_waveform else None
    cached = getattr(solver, "_dft_table_cache", None)
    if getattr(solver, "_dft_reuse_buffers", False) and key is not None and cached is not None and cached[0] == key:
        solver._dft_step.zero_()
        solver._dft_weighted_cos.zero_()
        solver._dft_weighted_sin.zero_()
        for name, value in cached[1].items():
            setattr(solver, name, value.copy())
        sync_dft_primary_runtime_state(solver)
        return True
    weighted_cos = np.zeros((steps, frequency_count), dtype=np.float32)
    weighted_sin = np.zeros((steps, frequency_count), dtype=np.float32)
    window_norm = np.zeros(frequency_count, dtype=np.float32)
    sample_count = np.zeros(frequency_count, dtype=np.int64)
    src_real = np.zeros(frequency_count, dtype=np.float64)
    src_imag = np.zeros(frequency_count, dtype=np.float64)

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
            from ...sources import evaluate_source_time_normalization

            signal = float(evaluate_source_time_normalization(solver._normalization_source_time, n * solver.dt))
            src_real += signal * wc
            src_imag += signal * ws
        next_cos = phase_cos * step_cos - phase_sin * step_sin
        next_sin = phase_sin * step_cos + phase_cos * step_sin
        phase_cos = next_cos.astype(np.float32, copy=False)
        phase_sin = next_sin.astype(np.float32, copy=False)

    for name, values in (("_dft_weighted_cos_table", weighted_cos), ("_dft_weighted_sin_table", weighted_sin)):
        setattr(solver, name, _dft_buffer(solver, getattr(solver, name, None), values.shape,
                                        torch.float32, value=torch.from_numpy(values)))
    for name, shape, dtype in (("_dft_weighted_cos", (frequency_count,), torch.float32),
                               ("_dft_weighted_sin", (frequency_count,), torch.float32),
                               ("_dft_step", (1,), torch.int64)):
        setattr(solver, name, _dft_buffer(solver, getattr(solver, name, None), shape, dtype))
    solver._dft_window_normalization_values = window_norm
    solver._dft_sample_count_values = sample_count
    if normalize:
        solver._dft_source_dft_real_values = src_real
        solver._dft_source_dft_imag_values = src_imag
    solver._dft_table_cache = (key, {name: getattr(solver, name).copy() for name in (
        "_dft_window_normalization_values", "_dft_sample_count_values",
        "_dft_source_dft_real_values", "_dft_source_dft_imag_values")})
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
    _launch_dft_accumulation(solver, solver._dft_weighted_cos, solver._dft_weighted_sin)
    solver._dft_step.add_(1)


def _launch_dft_accumulation(solver, weighted_cos, weighted_sin, *, auxiliary=False):
    arguments = dict(weightedCos=weighted_cos, weightedSin=weighted_sin)
    prefix = "aux_" if auxiliary else ""
    for component in ("Ex", "Ey", "Ez"):
        buffers = solver._dft_batched_fields.get(component)
        arguments[component] = (getattr(solver, component + ("_imag" if auxiliary else ""))
                                if buffers is not None else solver._dft_empty_field)
        arguments[component + "RealAccum"] = (buffers[prefix + "real"] if buffers is not None else solver._dft_empty_accum)
        arguments[component + "ImagAccum"] = (buffers[prefix + "imag"] if buffers is not None else solver._dft_empty_accum)
    solver.fdtd_module.accumulateRunningDftYee3DBatched(**arguments).launchRaw()
