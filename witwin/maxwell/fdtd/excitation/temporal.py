from __future__ import annotations

import numpy as np
import torch

from ...sources import (
    SOURCE_TIME_KIND_CW,
    SOURCE_TIME_KIND_GAUSSIAN_PULSE,
    evaluate_source_time,
)


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
    source_position=None,
    source_symmetry_scale=None,
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
        "phase_real": float(phase_real),
        "phase_imag": float(phase_imag),
        "delay_patch": None if delay_patch is None else delay_patch.contiguous(),
        "activation_delay_patch": (
            None if activation_delay_patch is None else activation_delay_patch.contiguous()
        ),
        "cw_cos_patch": None if cw_cos_patch is None else cw_cos_patch.contiguous(),
        "cw_sin_patch": None if cw_sin_patch is None else cw_sin_patch.contiguous(),
        "source_index": None if source_index is None else int(source_index),
        "source_position": (
            None if source_position is None else tuple(float(value) for value in source_position)
        ),
        "source_symmetry_scale": (
            None if source_symmetry_scale is None else float(source_symmetry_scale)
        ),
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


def _launch_uniform_patch(solver, *, field_name, source_patch, offsets, signal):
    offset_i, offset_j, offset_k = offsets
    solver.fdtd_module.addSourcePatch3D(
        field=getattr(solver, field_name),
        sourcePatch=source_patch,
        offsetI=int(offset_i),
        offsetJ=int(offset_j),
        offsetK=int(offset_k),
        signal=float(signal),
    ).launchRaw()


_COMPONENT_TANGENTIAL_AXES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}
_COMPONENT_AXIS_CODE = {"x": 0, "y": 1, "z": 2}


def _component_name(field_name):
    return field_name[-1].lower()


def _wrap_axes_for_kind(solver, field_name, kind):
    tangential_axes = _COMPONENT_TANGENTIAL_AXES[_component_name(field_name)]
    return tuple(
        1 if solver.scene.boundary.axis_kind(axis) == kind else 0
        for axis in tangential_axes
    )


def _periodic_wrap_axes(solver, field_name):
    return _wrap_axes_for_kind(solver, field_name, "periodic")


def _complex_wrap_axes(solver, field_name):
    tangential_axes = _COMPONENT_TANGENTIAL_AXES[_component_name(field_name)]
    return tuple(
        1 if solver.scene.boundary.axis_kind(axis) in {"bloch", "periodic"} else 0
        for axis in tangential_axes
    )


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
    ).launchRaw()


def _field_triplet_kwargs(solver, field_name):
    if field_name[0] == "H":
        return {
            "ExReal": solver.Hx,
            "ExImag": solver.Hx_imag,
            "EyReal": solver.Hy,
            "EyImag": solver.Hy_imag,
            "EzReal": solver.Hz,
            "EzImag": solver.Hz_imag,
        }
    return {
        "ExReal": solver.Ex,
        "ExImag": solver.Ex_imag,
        "EyReal": solver.Ey,
        "EyImag": solver.Ey_imag,
        "EzReal": solver.Ez,
        "EzImag": solver.Ez_imag,
    }


def _bloch_phase_kwargs(solver, field_name):
    component = _component_name(field_name)
    if component == "x":
        return {
            "phaseCosA": _phase_cos_for_axis(solver, "y"),
            "phaseSinA": _phase_sin_for_axis(solver, "y"),
            "phaseCosB": _phase_cos_for_axis(solver, "z"),
            "phaseSinB": _phase_sin_for_axis(solver, "z"),
        }
    if component == "y":
        return {
            "phaseCosA": _phase_cos_for_axis(solver, "x"),
            "phaseSinA": _phase_sin_for_axis(solver, "x"),
            "phaseCosB": _phase_cos_for_axis(solver, "z"),
            "phaseSinB": _phase_sin_for_axis(solver, "z"),
        }
    return {
        "phaseCosA": _phase_cos_for_axis(solver, "x"),
        "phaseSinA": _phase_sin_for_axis(solver, "x"),
        "phaseCosB": _phase_cos_for_axis(solver, "y"),
        "phaseSinB": _phase_sin_for_axis(solver, "y"),
    }


def _phase_cos_for_axis(solver, axis):
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    if solver.scene.boundary.axis_kind(axis) == "periodic":
        return 1.0
    return solver.boundary_phase_cos[axis_index]


def _phase_sin_for_axis(solver, axis):
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    if solver.scene.boundary.axis_kind(axis) == "periodic":
        return 0.0
    return solver.boundary_phase_sin[axis_index]


def _bloch_patch_common_kwargs(solver, field_name, term):
    wrap_axis_a, wrap_axis_b = _complex_wrap_axes(solver, field_name)
    offset_i, offset_j, offset_k = term["offsets"]
    kernel_kwargs = {
        "offsetI": int(offset_i),
        "offsetJ": int(offset_j),
        "offsetK": int(offset_k),
        "axisCode": _COMPONENT_AXIS_CODE[_component_name(field_name)],
        "wrapAxisA": int(wrap_axis_a),
        "wrapAxisB": int(wrap_axis_b),
    }
    kernel_kwargs.update(_field_triplet_kwargs(solver, field_name))
    kernel_kwargs.update(_bloch_phase_kwargs(solver, field_name))
    return kernel_kwargs


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
    ).launchRaw()


def _evaluate_time_shifted_signal_tensor(source_time, sample_time):
    # Per-cell source-time waveform for a time-shifted patch, matching the
    # ``evaluate_source_time`` device kernel branch-for-branch (kind 0 = CW,
    # kind 1 = Gaussian pulse, anything else = Ricker) so a Bloch pulse injected
    # through this Torch path is numerically identical to the non-Bloch kernel.
    kind_code = int(source_time["kind_code"])
    amplitude = float(source_time["amplitude"])
    frequency = float(source_time["frequency"])
    phase = float(source_time.get("phase", 0.0))
    delay = float(source_time.get("delay", 0.0))
    two_pi = 2.0 * np.pi
    if kind_code == SOURCE_TIME_KIND_CW:
        return amplitude * torch.cos(two_pi * frequency * sample_time + phase)
    if kind_code == SOURCE_TIME_KIND_GAUSSIAN_PULSE:
        inv_sigma = max(two_pi * float(source_time["fwidth"]), 1.0e-30)
        tau = sample_time - delay
        envelope = torch.exp(-0.5 * (tau * inv_sigma) ** 2)
        return amplitude * envelope * torch.cos(two_pi * frequency * tau + phase)
    tau = sample_time - delay
    alpha = np.pi * frequency * tau
    alpha_sq = alpha * alpha
    return amplitude * (1.0 - 2.0 * alpha_sq) * torch.exp(-alpha_sq)


def _time_shifted_signal_patch(term, source_time, time_value):
    # Collapse the delayed pulse to a real per-cell amplitude patch on-device.
    # The scatter into the split real/imag Bloch field (interior injection plus
    # the phase-rotated wrap copies) is still the native ``addSourcePatchBloch3D``
    # kernel; only the envelope evaluation is Torch, exactly as the CW-phased
    # Bloch path already precomputes its cos/sin combination before scattering.
    sample_time = float(time_value) - term["delay_patch"]
    signal_patch = _evaluate_time_shifted_signal_tensor(source_time, sample_time) * term["patch"]
    activation_delay_patch = term.get("activation_delay_patch")
    if activation_delay_patch is not None:
        signal_patch = torch.where(
            activation_delay_patch > float(time_value),
            torch.zeros_like(signal_patch),
            signal_patch,
        )
    return signal_patch.contiguous()


def _launch_bloch_time_shifted_patch(solver, *, field_name, term, source_time, time_value):
    signal_patch = _time_shifted_signal_patch(term, source_time, time_value)
    kernel_kwargs = _bloch_patch_common_kwargs(solver, field_name, term)
    kernel_kwargs.update(
        {
            "sourcePatch": signal_patch,
            "signalReal": 1.0,
            "signalImag": 0.0,
        }
    )
    solver.fdtd_module.addSourcePatchBloch3D(**kernel_kwargs).launchRaw()


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
            if solver.scene.boundary.uses_kind("bloch"):
                _launch_bloch_cw_patch(
                    solver,
                    field_name=field_name,
                    term=term,
                    signal_cos=signal_cos,
                    signal_sin=signal_sin,
                )
                continue
            _launch_cw_patch(
                solver,
                field_name=field_name,
                term=term,
                signal_cos=signal_cos,
                signal_sin=signal_sin,
            )
            continue
        if term["delay_patch"] is not None:
            if solver.scene.boundary.uses_kind("bloch"):
                _launch_bloch_time_shifted_patch(
                    solver,
                    field_name=field_name,
                    term=term,
                    source_time=term_source_time,
                    time_value=time_value,
                )
                continue
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
        if solver.scene.boundary.uses_kind("bloch"):
            _launch_bloch_patch(
                solver,
                field_name=field_name,
                term=term,
                signal=signal_cache[cache_key],
            )
            continue
        if solver.scene.boundary.uses_kind("periodic"):
            _launch_periodic_patch(
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
            signal=signal_cache[cache_key],
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
        )
        return
    offset_i, offset_j, offset_k = term["offsets"]
    component = _component_name(field_name)
    kernel_name = f"addSourcePatchE{component}Periodic3D"
    field_arg_name = f"E{component}"
    getattr(solver.fdtd_module, kernel_name)(
        **{
            field_arg_name: getattr(solver, field_name),
            "sourcePatch": term["patch"],
            "offsetI": int(offset_i),
            "offsetJ": int(offset_j),
            "offsetK": int(offset_k),
            "signal": scaled_signal,
            "wrapAxisA": int(wrap_axis_a),
            "wrapAxisB": int(wrap_axis_b),
        }
    ).launchRaw()


def _launch_bloch_patch(solver, *, field_name, term, signal):
    kernel_kwargs = _bloch_patch_common_kwargs(solver, field_name, term)
    kernel_kwargs.update(
        {
            "sourcePatch": term["patch"],
            "signalReal": float(signal) * float(term["phase_real"]),
            "signalImag": float(signal) * float(term["phase_imag"]),
        }
    )
    solver.fdtd_module.addSourcePatchBloch3D(**kernel_kwargs).launchRaw()


def _launch_bloch_cw_patch(solver, *, field_name, term, signal_cos, signal_sin):
    kernel_kwargs = _bloch_patch_common_kwargs(solver, field_name, term)
    kernel_kwargs.update(
        {
            "sourcePatchCos": term["cw_cos_patch"],
            "sourcePatchSin": term["cw_sin_patch"],
            "signalCos": float(signal_cos),
            "signalSin": float(signal_sin),
        }
    )
    solver.fdtd_module.addCwPhasedSourcePatchBloch3D(**kernel_kwargs).launchRaw()


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
            if solver.scene.boundary.uses_kind("bloch"):
                _launch_bloch_cw_patch(
                    solver,
                    field_name=field_name,
                    term=term,
                    signal_cos=signal_cos,
                    signal_sin=signal_sin,
                )
                continue
            _launch_cw_patch(
                solver,
                field_name=field_name,
                term=term,
                signal_cos=signal_cos,
                signal_sin=signal_sin,
            )
            continue
        if term["delay_patch"] is not None:
            if solver.scene.boundary.uses_kind("bloch"):
                _launch_bloch_time_shifted_patch(
                    solver,
                    field_name=field_name,
                    term=term,
                    source_time=term_source_time,
                    time_value=time_value,
                )
                continue
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
        if solver.scene.boundary.uses_kind("bloch"):
            _launch_bloch_patch(
                solver,
                field_name=field_name,
                term=term,
                signal=signal_cache[cache_key],
            )
            continue
        if solver.scene.boundary.uses_kind("periodic"):
            _launch_periodic_patch(
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
        )

    solver._clamp_pec_boundaries()
