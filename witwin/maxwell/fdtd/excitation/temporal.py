from __future__ import annotations

import numpy as np

from ...sources import evaluate_source_time


def build_source_term(
    solver,
    *,
    field_name,
    offsets,
    patch=None,
    phase_real=1.0,
    phase_imag=0.0,
    delay_patch=None,
    activation_delay_patch=None,
    cw_cos_patch=None,
    cw_sin_patch=None,
    source_index=None,
    source_time=None,
    omega=None,
):
    grid_tensor = patch
    if grid_tensor is None:
        grid_tensor = delay_patch
    if grid_tensor is None:
        grid_tensor = cw_cos_patch
    if grid_tensor is None:
        grid_tensor = cw_sin_patch
    if grid_tensor is None:
        raise ValueError("source term requires at least one tensor payload.")

    return {
        "field_name": field_name,
        "offsets": offsets,
        "patch": None if patch is None else patch.contiguous(),
        "grid": solver._compute_linear_launch_shape(int(grid_tensor.numel())),
        "phase_real": float(phase_real),
        "phase_imag": float(phase_imag),
        "delay_patch": None if delay_patch is None else delay_patch.contiguous(),
        "activation_delay_patch": (
            None if activation_delay_patch is None else activation_delay_patch.contiguous()
        ),
        "cw_cos_patch": None if cw_cos_patch is None else cw_cos_patch.contiguous(),
        "cw_sin_patch": None if cw_sin_patch is None else cw_sin_patch.contiguous(),
        "source_index": None if source_index is None else int(source_index),
        "source_time": source_time,
        "omega": None if omega is None else float(omega),
    }


def append_source_term(term_list, solver, **kwargs):
    term_list.append(build_source_term(solver, **kwargs))


def _resolve_time_shift_patch(term):
    activation_delay_patch = term.get("activation_delay_patch")
    return activation_delay_patch if activation_delay_patch is not None else term["delay_patch"]


def _resolve_cw_signal(omega, source_time, time_value):
    phase = float(omega) * float(time_value) + float(source_time["phase"])
    amplitude = float(source_time["amplitude"])
    return amplitude * np.cos(phase), amplitude * np.sin(phase)


def _resolve_term_source(term, default_source_time, default_omega):
    term_source_time = term.get("source_time") or default_source_time
    term_omega = float(default_omega if term.get("omega") is None else term["omega"])
    return term_source_time, term_omega


def _resolve_term_cache_key(term, term_source_time, term_omega):
    source_index = term.get("source_index")
    if source_index is not None:
        return int(source_index)
    return (id(term_source_time), float(term_omega))


def _launch_uniform_patch(solver, *, field_name, source_patch, offsets, signal, grid):
    offset_i, offset_j, offset_k = offsets
    solver.fdtd_module.addSourcePatch3D(
        field=getattr(solver, field_name),
        sourcePatch=source_patch,
        offsetI=int(offset_i),
        offsetJ=int(offset_j),
        offsetK=int(offset_k),
        signal=float(signal),
    ).launchRaw(
        blockSize=solver.kernel_block_size,
        gridSize=grid,
    )


_FIELD_TANGENTIAL_AXES = {
    "Ex": ("y", "z"),
    "Ey": ("x", "z"),
    "Ez": ("x", "y"),
}


def _wrap_axes_for_kind(solver, field_name, kind):
    tangential_axes = _FIELD_TANGENTIAL_AXES[field_name]
    return tuple(
        1 if solver.scene.boundary.axis_kind(axis) == kind else 0
        for axis in tangential_axes
    )


def _periodic_wrap_axes(solver, field_name):
    return _wrap_axes_for_kind(solver, field_name, "periodic")


def _bloch_wrap_axes(solver, field_name):
    return _wrap_axes_for_kind(solver, field_name, "bloch")


def _launch_cw_patch(solver, *, field_name, term, signal_cos, signal_sin):
    offset_i, offset_j, offset_k = term["offsets"]
    solver.fdtd_module.addCwPhasedSourcePatch3D(
        field=getattr(solver, field_name),
        sourcePatchCos=term["cw_cos_patch"],
        sourcePatchSin=term["cw_sin_patch"],
        offsetI=int(offset_i),
        offsetJ=int(offset_j),
        offsetK=int(offset_k),
        signalCos=float(signal_cos),
        signalSin=float(signal_sin),
    ).launchRaw(
        blockSize=solver.kernel_block_size,
        gridSize=term["grid"],
    )


def _launch_time_shifted_patch(solver, *, field_name, term, source_time, time_value):
    offset_i, offset_j, offset_k = term["offsets"]
    solver.fdtd_module.addTimeShiftedSourcePatch3D(
        field=getattr(solver, field_name),
        sourcePatch=term["patch"],
        delayPatch=term["delay_patch"],
        activationDelayPatch=_resolve_time_shift_patch(term),
        offsetI=int(offset_i),
        offsetJ=int(offset_j),
        offsetK=int(offset_k),
        timeKind=int(source_time["kind_code"]),
        time=float(time_value),
        frequency=float(source_time["frequency"]),
        fwidth=float(source_time["fwidth"]),
        amplitude=float(source_time["amplitude"]),
        phase=float(source_time["phase"]),
        delay=float(source_time["delay"]),
        causalGate=1 if term.get("activation_delay_patch") is not None else 0,
    ).launchRaw(
        blockSize=solver.kernel_block_size,
        gridSize=term["grid"],
    )


def apply_generic_source_terms(solver, terms, *, source_time, omega, time_value, clamp_pec=True):
    if not terms:
        return

    signal_cache = {}
    cw_cache = {}
    for term in terms:
        field_name = term["field_name"]
        term_source_time, term_omega = _resolve_term_source(term, source_time, omega)
        cache_key = _resolve_term_cache_key(term, term_source_time, term_omega)
        if term["cw_cos_patch"] is not None:
            if cache_key not in cw_cache:
                cw_cache[cache_key] = _resolve_cw_signal(term_omega, term_source_time, time_value)
            signal_cos, signal_sin = cw_cache[cache_key]
            _launch_cw_patch(
                solver,
                field_name=field_name,
                term=term,
                signal_cos=signal_cos,
                signal_sin=signal_sin,
            )
            continue
        if term["delay_patch"] is not None:
            _launch_time_shifted_patch(
                solver,
                field_name=field_name,
                term=term,
                source_time=term_source_time,
                time_value=time_value,
            )
            continue
        if cache_key not in signal_cache:
            signal_cache[cache_key] = evaluate_source_time(term_source_time, time_value)
        _launch_uniform_patch(
            solver,
            field_name=field_name,
            source_patch=term["patch"],
            offsets=term["offsets"],
            signal=signal_cache[cache_key],
            grid=term["grid"],
        )

    if clamp_pec:
        solver._clamp_pec_boundaries()


def _launch_periodic_patch(solver, *, field_name, term, signal):
    wrap_axis_a, wrap_axis_b = _periodic_wrap_axes(solver, field_name)
    scaled_signal = float(signal) * float(term["phase_real"])
    if not wrap_axis_a and not wrap_axis_b:
        _launch_uniform_patch(
            solver,
            field_name=field_name,
            source_patch=term["patch"],
            offsets=term["offsets"],
            signal=scaled_signal,
            grid=term["grid"],
        )
        return
    offset_i, offset_j, offset_k = term["offsets"]
    kernel_name = {
        "Ex": "addSourcePatchExPeriodic3D",
        "Ey": "addSourcePatchEyPeriodic3D",
        "Ez": "addSourcePatchEzPeriodic3D",
    }[field_name]
    getattr(solver.fdtd_module, kernel_name)(
        **{
            field_name: getattr(solver, field_name),
            "sourcePatch": term["patch"],
            "offsetI": int(offset_i),
            "offsetJ": int(offset_j),
            "offsetK": int(offset_k),
            "signal": scaled_signal,
            "wrapAxisA": int(wrap_axis_a),
            "wrapAxisB": int(wrap_axis_b),
        }
    ).launchRaw(
        blockSize=solver.kernel_block_size,
        gridSize=term["grid"],
    )


def _launch_bloch_patch(solver, *, field_name, term, signal):
    wrap_axis_a, wrap_axis_b = _bloch_wrap_axes(solver, field_name)
    offset_i, offset_j, offset_k = term["offsets"]
    kernel_kwargs = {
        "sourcePatch": term["patch"],
        "offsetI": int(offset_i),
        "offsetJ": int(offset_j),
        "offsetK": int(offset_k),
        "signalReal": float(signal) * float(term["phase_real"]),
        "signalImag": float(signal) * float(term["phase_imag"]),
        "axisCode": {"Ex": 0, "Ey": 1, "Ez": 2}[field_name],
        "wrapAxisA": int(wrap_axis_a),
        "wrapAxisB": int(wrap_axis_b),
        "ExReal": solver.Ex,
        "ExImag": solver.Ex_imag,
        "EyReal": solver.Ey,
        "EyImag": solver.Ey_imag,
        "EzReal": solver.Ez,
        "EzImag": solver.Ez_imag,
    }
    if field_name == "Ex":
        kernel_kwargs.update(
            {
                "phaseCosA": solver.boundary_phase_cos[1],
                "phaseSinA": solver.boundary_phase_sin[1],
                "phaseCosB": solver.boundary_phase_cos[2],
                "phaseSinB": solver.boundary_phase_sin[2],
            }
        )
    elif field_name == "Ey":
        kernel_kwargs.update(
            {
                "phaseCosA": solver.boundary_phase_cos[0],
                "phaseSinA": solver.boundary_phase_sin[0],
                "phaseCosB": solver.boundary_phase_cos[2],
                "phaseSinB": solver.boundary_phase_sin[2],
            }
        )
    else:
        kernel_kwargs.update(
            {
                "phaseCosA": solver.boundary_phase_cos[0],
                "phaseSinA": solver.boundary_phase_sin[0],
                "phaseCosB": solver.boundary_phase_cos[1],
                "phaseSinB": solver.boundary_phase_sin[1],
            }
        )
    solver.fdtd_module.addSourcePatchBloch3D(**kernel_kwargs).launchRaw(
        blockSize=solver.kernel_block_size,
        gridSize=term["grid"],
    )


def apply_compiled_source_terms(solver, terms, *, source_time, omega, time_value, signal=None):
    if not terms:
        return

    default_key = _resolve_term_cache_key(
        {"source_index": None},
        source_time,
        float(omega),
    )
    signal_cache = {}
    cw_cache = {}
    for term in terms:
        field_name = term["field_name"]
        term_source_time, term_omega = _resolve_term_source(term, source_time, omega)
        cache_key = _resolve_term_cache_key(term, term_source_time, term_omega)
        if term["cw_cos_patch"] is not None:
            if cache_key not in cw_cache:
                cw_cache[cache_key] = _resolve_cw_signal(term_omega, term_source_time, time_value)
            signal_cos, signal_sin = cw_cache[cache_key]
            _launch_cw_patch(
                solver,
                field_name=field_name,
                term=term,
                signal_cos=signal_cos,
                signal_sin=signal_sin,
            )
            continue
        if term["delay_patch"] is not None:
            _launch_time_shifted_patch(
                solver,
                field_name=field_name,
                term=term,
                source_time=term_source_time,
                time_value=time_value,
            )
            continue
        if cache_key not in signal_cache:
            if signal is not None and cache_key == default_key:
                signal_cache[cache_key] = signal
            else:
                signal_cache[cache_key] = evaluate_source_time(term_source_time, time_value)
        if solver.scene.boundary.uses_kind("periodic"):
            _launch_periodic_patch(
                solver,
                field_name=field_name,
                term=term,
                signal=signal_cache[cache_key],
            )
            continue
        if solver.scene.boundary.uses_kind("bloch"):
            _launch_bloch_patch(
                solver,
                field_name=field_name,
                term=term,
                signal=signal_cache[cache_key],
            )
            continue
        _launch_uniform_patch(
            solver,
            field_name=field_name,
            source_patch=term["patch"],
            offsets=term["offsets"],
            signal=float(signal_cache[cache_key]) * float(term["phase_real"]),
            grid=term["grid"],
        )

    solver._clamp_pec_boundaries()
