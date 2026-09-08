"""Order-independent source support and measurement planning."""
from __future__ import annotations

import math

from ...sources import compile_source_time


def _step_at_or_after(duration, dt):
    steps = duration / dt
    return math.ceil(steps - 4 * math.ulp(steps))


def source_spectrum_key(source_time):
    if source_time is None:
        return None
    if source_time.get("times") is not None:
        return ("custom", tuple(source_time["times"]), tuple(source_time["amplitudes"]),
                float(source_time["amplitude"]))
    if source_time.get("fn") is not None:
        return ("custom_callable", id(source_time["fn"]), float(source_time["amplitude"]))
    return tuple(source_time.get(field) for field in
                 ("kind", "frequency", "amplitude", "phase", "delay", "fwidth"))


def sources_share_spectrum(compiled_sources):
    return len({source_spectrum_key(source.get("source_time")) for source in compiled_sources}) <= 1


def _circuit_times(scene):
    from ...circuits import CurrentSource, VoltageSource, PulseWaveform, SineWaveform, PiecewiseLinearWaveform

    times = []
    for circuit in getattr(scene, "circuits", ()):
        for device in circuit.devices:
            if not isinstance(device, (CurrentSource, VoltageSource)):
                continue
            waveform = device.waveform
            if isinstance(waveform, SineWaveform):
                damping = float(waveform.damping)
                times.append(dict(kind="circuit", frequency=float(waveform.frequency),
                    settling_time=float(waveform.delay) + (20 / damping if damping > 0 else 0),
                    ongoing=damping == 0 or float(waveform.offset) != 0))
            elif isinstance(waveform, PulseWaveform):
                period = float(waveform.period)
                times.append(dict(kind="circuit", frequency=1 / period if period > 0 else 0,
                    settling_time=sum(float(getattr(waveform, key)) for key in ("delay", "rise", "width", "fall")),
                    ongoing=period > 0 or float(waveform.initial) != 0))
            elif isinstance(waveform, PiecewiseLinearWaveform):
                times.append(dict(kind="circuit", settling_time=float(waveform.times[-1]),
                    ongoing=float(waveform.values[-1]) != 0))
            elif float(device.voltage if isinstance(device, VoltageSource) else device.current) != 0:
                times.append(dict(kind="circuit", settling_time=0, ongoing=True))
    return times


def compile_fdtd_time_plan(solver, scene, config, frequencies, *, excitations=(), normalize_source=False, window="hanning"):
    compiled = getattr(solver, "_compiled_sources", None)
    if compiled is None:
        from ...compiler.sources import compile_fdtd_sources
        compiled = compile_fdtd_sources(scene, default_frequency=min(frequencies))
    times = [source["source_time"] for source in compiled]
    times.extend(compile_source_time(excitation.source_time, default_frequency=min(frequencies))
                 for excitation in excitations)
    if normalize_source and not times:
        raise NotImplementedError("normalize_source requires an incident source spectrum.")
    if normalize_source and len({source_spectrum_key(value) for value in times}) > 1:
        raise NotImplementedError("normalize_source requires a single incident source spectrum; "
                                  "give all sources a common source_time or disable normalize_source.")
    circuit_times = _circuit_times(scene)
    if normalize_source and circuit_times:
        raise NotImplementedError("normalize_source has no unique incident spectrum for circuit-driven fields.")
    times.extend(circuit_times)
    unknown_support = any(value.get("fn") is not None for value in times)
    if unknown_support and config.time_steps is None:
        raise ValueError("Callable source-time support is unknown; provide explicit TimeConfig.time_steps.")
    pure_cw = bool(times) and all(value.get("kind") == "cw" for value in times)
    ongoing = any(value.get("kind") == "cw" or value.get("ongoing", False) for value in times)
    slow_frequency = min(tuple(float(f) for f in frequencies) + tuple(
        float(value["frequency"]) for value in times if float(value.get("frequency", 0)) > 0))
    period = 1 / slow_frequency
    propagation = 10 * max(high - low for low, high in scene.domain.bounds) / solver.c
    settling = max((float(value.get("settling_time", 0)) for value in times), default=0)
    transient = max(config.transient_cycles * period, propagation)
    measurement_start = transient if pure_cw and window != "none" else 0.0
    required_time = max(transient, settling + propagation) + config.steady_cycles * period
    required_steps = max(1, _step_at_or_after(required_time, solver.dt))
    total_steps = required_steps if config.time_steps is None else config.time_steps
    plan = dict(total_steps=total_steps, required_steps=required_steps,
        spectral_start_step=_step_at_or_after(measurement_start, solver.dt),
        response="cw_steady_state" if pure_cw and window != "none" else "full_response",
        window=window if pure_cw else "none",
        latest_source_settling_step=_step_at_or_after(settling, solver.dt),
        shutoff_allowed=not ongoing and not unknown_support,
        shutoff_min_step=_step_at_or_after(settling + propagation / 5, solver.dt),
        source_count=len(times), unknown_source_support=unknown_support,
        truncated=total_steps < required_steps,
        sources_complete=not unknown_support and total_steps * solver.dt >= settling,
        transient_cycles=config.transient_cycles, steady_cycles=config.steady_cycles,
        reference_frequency=slow_frequency)
    solver._time_plan = plan
    for shard in getattr(solver, "shards", ()):
        shard.solver._time_plan = dict(plan)
    return plan
