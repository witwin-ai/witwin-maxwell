"""Order-independent source support and measurement planning."""
from __future__ import annotations

import math

from ...sources import compile_source_time
from ..excitation.temporal import injection_time


def _step_at_or_after(duration, dt):
    steps = duration / dt
    return math.ceil(steps - 4 * math.ulp(steps))


def source_spectrum_key(source_time):
    """Waveform parameters excluding the scalar physical amplitude."""
    if source_time is None:
        return None
    if source_time.get("times") is not None:
        return ("custom", tuple(source_time["times"]), tuple(source_time["amplitudes"]),
                source_time.get("interpolation", "linear"), source_time["frequency"], source_time["delay"])
    if source_time.get("fn") is not None:
        return ("custom_callable", id(source_time["fn"]), source_time["frequency"], source_time["delay"])
    return tuple(source_time.get(field) for field in
                 ("kind", "frequency", "phase", "delay", "fwidth", "pedestal", "pedestal_fwidth"))


def sources_share_spectrum(compiled_sources):
    return len({source_spectrum_key(source.get("source_time")) for source in compiled_sources}) <= 1


def source_time_offset(source):
    """Injection offset in steps for a source's unit-envelope spectrum."""
    if source is None:
        return 0.0
    if source.get("reference_provider_kind") in {"plane_wave_ref_x_ez", "plane_wave_axis_aligned", "plane_wave_aux"}:
        return 1.0
    if source["kind"] in {"point_dipole", "uniform_current"}:
        return injection_time("electric_current", 0.0, 1.0)
    if source["kind"] == "custom_current":
        components = source["dataset"].components
        if all(name.startswith("J") for name in components):
            return injection_time("electric_current", 0.0, 1.0)
    # Combined J/M sources (custom field, TFSF, modes, or mixed currents)
    # share one incident-envelope reference at n*dt, not one per sheet.
    return 0.0


# Scalar reference descriptors are derived from prepared provider dispatch.
REFERENCE_FIELDS = (
    "reference_provider_kind", "reference_waveform_kind", "reference_incident_kind",
    "reference_incident_projection", "reference_incident_direction", "reference_incident_bounds",
)


def source_reference_label_offset(source):
    """Fourier label beta for the compiled provider, independent of sampling alpha."""
    provider = source["reference_provider_kind"]
    waveform = source["reference_waveform_kind"]
    if provider == "plane_wave_fixed_k":
        # Retain the line's n*dt soft drive reference. Its measured entry-plane
        # E spectrum has a separate physical Yee label at (n+1)*dt.
        return 0.0
    if provider in {"plane_wave_ref_x_ez", "plane_wave_axis_aligned", "plane_wave_aux"}:
        if waveform != "cw":
            if (source.get("reference_incident_kind") != "physical_upstream_yee_entry_plane"
                    or source.get("reference_incident_projection") is None):
                raise ValueError("Auxiliary pulse reference requires its physical Yee entry plane.")
        return 1.0
    # Analytic/discrete carriers already use physical M/J Yee instants.
    if provider == "plane_wave_discrete_cw":
        return 0.0 if waveform == "cw" else 1.0
    if provider in {"analytic_profile", "plane_wave_grating_slab_cw"}:
        return 0.0
    if provider in {"additive_J", "terminal_drive"}:
        return 0.5
    if provider in {"magnetic_M", "mixed_JM", "soft_equivalent_sheet", "mode_surface", "custom_field_surface"}:
        return 0.0
    raise ValueError(f"Unknown compiled source reference provider {provider!r}.")


def source_reference_descriptor(source):
    return {key: source[key] for key in REFERENCE_FIELDS if key in source}


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


def compile_fdtd_time_plan(solver, scene, config, frequencies, *, default_source_frequency, excitations=(), normalize_source=False, window="hanning"):
    compiled = solver._compiled_sources
    times = [source["source_time"] for source in compiled]
    times.extend(compile_source_time(excitation.source_time, default_frequency=default_source_frequency)
                 for excitation in excitations)
    if normalize_source and not times:
        raise NotImplementedError("normalize_source requires an incident source spectrum.")
    if normalize_source and len({source_spectrum_key(value) for value in times}) > 1:
        raise ValueError("normalize_source requires a single incident source spectrum; "
                         "source_time waveforms may differ only in amplitude.")
    # Keep the incident normalization reference independent of the field source
    # used to initialize injection. A terminal-only drive has no field source.
    normalization_source_time = times[0] if times else None
    normalization_time_offset = source_time_offset(compiled[0]) if compiled else (
        injection_time("port_drive", 0.0, 1.0) if excitations else 0.0)
    reference = compiled[0] if compiled else (dict(
        reference_provider_kind="terminal_drive",
        reference_waveform_kind=normalization_source_time["kind"],
    ) if excitations else None)
    descriptor = source_reference_descriptor(reference) if reference is not None else {}
    beta = source_reference_label_offset(reference) if reference is not None else None
    solver._normalization_reference = descriptor
    solver._normalization_reference_provider = descriptor.get("reference_provider_kind")
    solver._normalization_reference_waveform_kind = descriptor.get("reference_waveform_kind")
    solver._normalization_label_offset = beta
    circuit_times = _circuit_times(scene)
    if normalize_source and circuit_times:
        raise NotImplementedError("normalize_source has no unique incident spectrum for circuit-driven fields.")
    times.extend(circuit_times)
    unknown_support = any(value.get("fn") is not None for value in times)
    if unknown_support and config.time_steps is None:
        raise ValueError("Callable source-time support is unknown; provide explicit TimeConfig.time_steps.")
    pure_cw = bool(times) and all(value.get("kind") == "cw" for value in times)
    all_ongoing = all(value.get("kind") == "cw" or value.get("ongoing") is True for value in times)
    mixed_sources = not all_ongoing and any(
        value.get("kind") == "cw" or value.get("ongoing") is True for value in times)
    # Keep common internal samples for ratios; mixed absolute spectra are undefined.
    solver._spectral_integration_dt = None if all_ongoing else solver.dt
    from ..boundary import has_complex_fields
    complex_fields = has_complex_fields(solver)
    if complex_fields and not pure_cw and any(value.get("kind") == "cw" for value in times):
        raise NotImplementedError("Complex FDTD cannot mix analytic CW and real-pulse source times; "
                                  "use a pure CW or a pure non-CW run.")
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
        source_count=len(times), mixed_sources=mixed_sources, unknown_source_support=unknown_support,
        truncated=total_steps < required_steps,
        sources_complete=not unknown_support and total_steps * solver.dt >= settling,
        transient_cycles=config.transient_cycles, steady_cycles=config.steady_cycles,
        reference_frequency=slow_frequency,
        source_representation="analytic_cw" if complex_fields and pure_cw else "real_projected",
        spectral_phasor_factor=1.0 if complex_fields and pure_cw else 2.0)
    solver._time_plan = plan
    solver._normalization_source_time = normalization_source_time
    solver._normalization_time_offset = normalization_time_offset
    for shard in getattr(solver, "shards", ()):
        shard.solver._time_plan = dict(plan)
        shard.solver._normalization_source_time = normalization_source_time
        shard.solver._normalization_time_offset = normalization_time_offset
        shard.solver._normalization_reference = dict(descriptor)
        shard.solver._normalization_reference_provider = solver._normalization_reference_provider
        shard.solver._normalization_reference_waveform_kind = solver._normalization_reference_waveform_kind
        shard.solver._normalization_label_offset = beta
        shard.solver._spectral_integration_dt = solver._spectral_integration_dt
        shard.solver._spectral_reference_frequency = solver._spectral_reference_frequency
    return plan
