from __future__ import annotations

import numpy as np

from ..sources import (
    AstigmaticGaussianBeam,
    CustomCurrentSource,
    CustomFieldSource,
    GaussianBeam,
    ModeSource,
    PlaneWave,
    PointDipole,
    TFSF,
    UniformCurrentSource,
    compile_source_time,
)
from ..waveforms import CALLABLE_TIME_SHIFT_MESSAGE


def _compile_injection(injection) -> dict[str, object]:
    if isinstance(injection, TFSF):
        if injection.mode == "slab":
            return {
                "kind": "tfsf",
                "mode": "slab",
                "axis": injection.axis,
                "axis_bounds": injection.axis_bounds,
            }
        return {
            "kind": "tfsf",
            "mode": "box",
            "bounds": injection.bounds,
        }
    return {"kind": str(injection).lower()}


def _compile_fdfd_point_dipole(source: PointDipole, *, default_frequency: float) -> dict:
    source_time = compile_source_time(source.source_time, default_frequency=default_frequency)
    if source_time["kind"] != "cw":
        raise ValueError("FDFD sources currently support CW source_time only.")
    # CW is A*cos(omega*t+phase); FDFD phasors multiply exp(-i*omega*t).
    phase = np.exp(-1j * float(source_time["phase"]))
    amplitude = float(source_time["amplitude"])
    polarization = tuple(amplitude * phase * component for component in source.polarization)
    return {
        "name": source.name,
        "position": source.position,
        "width": source.width,
        "profile": source.profile,
        "polarization": polarization,
    }


def compile_fdfd_sources(scene, *, default_frequency: float):
    from ..fdfd_response import LumpedCurrentSource
    compiled = []
    for source in scene.sources:
        if isinstance(source, LumpedCurrentSource):
            compiled.append({'kind': 'lumped_current', 'start': source.start,
                             'end': source.end, 'current': complex(source.current)})
            continue
        if isinstance(source, UniformCurrentSource):
            waveform = compile_source_time(source.source_time, default_frequency=default_frequency)
            if waveform['kind'] != 'cw':
                raise ValueError("FDFD current sources require CW waveforms.")
            amplitude = float(waveform['amplitude']) * np.exp(-1j*float(waveform['phase']))
            compiled.append({'kind': 'uniform_current', 'center': source.center,
                             'size': source.size, 'polarization': tuple(amplitude*p for p in source.polarization)})
            continue
        if not isinstance(source, PointDipole):
            raise ValueError(f"Unsupported FDFD source type: {type(source).__name__}")
        compiled.append(_compile_fdfd_point_dipole(source, default_frequency=default_frequency))
    return compiled


def _compile_point_dipole(source: PointDipole, *, default_frequency: float) -> dict:
    return {
        "kind": "point_dipole",
        "name": source.name,
        "position": source.position,
        "width": source.width,
        "profile": source.profile,
        "polarization": source.polarization,
        "source_time": compile_source_time(source.source_time, default_frequency=default_frequency),
    }


def _compile_time_shifted_source_time(source, *, default_frequency: float) -> dict:
    """Source time of a surface source whose cells sample it at per-cell delays."""
    source_time = compile_source_time(source.source_time, default_frequency=default_frequency)
    if source_time["kind"] == "custom" and source_time["table"] is None:
        raise ValueError(f"{type(source).__name__}: {CALLABLE_TIME_SHIFT_MESSAGE}")
    return source_time


def _compile_plane_wave(source: PlaneWave, *, default_frequency: float) -> dict:
    return {
        "kind": "plane_wave",
        "name": source.name,
        "direction": source.direction,
        "polarization": source.polarization,
        "injection": _compile_injection(source.injection),
        "injection_axis": source.injection_axis,
        # A TFSF plane wave drives its auxiliary incident line on the solver time
        # grid, so only the soft surface injection needs a tabulated waveform.
        "source_time": (
            compile_source_time(source.source_time, default_frequency=default_frequency)
            if isinstance(source.injection, TFSF)
            else _compile_time_shifted_source_time(source, default_frequency=default_frequency)
        ),
    }


def _compile_gaussian_beam(source: GaussianBeam, *, default_frequency: float) -> dict:
    return {
        "kind": "gaussian_beam",
        "name": source.name,
        "direction": source.direction,
        "polarization": source.polarization,
        "beam_waist": source.beam_waist,
        "focus": source.focus,
        "injection": _compile_injection(source.injection),
        "injection_axis": source.injection_axis,
        "source_time": _compile_time_shifted_source_time(source, default_frequency=default_frequency),
    }


def _compile_astigmatic_gaussian_beam(source: AstigmaticGaussianBeam, *, default_frequency: float) -> dict:
    return {
        "kind": "astigmatic_gaussian_beam",
        "name": source.name,
        "direction": source.direction,
        "polarization": source.polarization,
        "beam_waist_u": source.beam_waist[0],
        "beam_waist_v": source.beam_waist[1],
        "focus": source.focus,
        "focus_u": source.focus_u,
        "focus_v": source.focus_v,
        "injection": _compile_injection(source.injection),
        "injection_axis": source.injection_axis,
        "source_time": _compile_time_shifted_source_time(source, default_frequency=default_frequency),
    }


def _compile_uniform_current_source(source: UniformCurrentSource, *, default_frequency: float) -> dict:
    return {
        "kind": "uniform_current",
        "name": source.name,
        "center": source.center,
        "size": source.size,
        "polarization": source.polarization,
        "source_time": compile_source_time(source.source_time, default_frequency=default_frequency),
    }


def _compile_custom_current_source(source: CustomCurrentSource, *, default_frequency: float) -> dict:
    return {
        "kind": "custom_current",
        "name": source.name,
        "dataset": source.current_dataset,
        "source_time": compile_source_time(source.source_time, default_frequency=default_frequency),
    }


def _compile_custom_field_source(source: CustomFieldSource, *, default_frequency: float) -> dict:
    return {
        "kind": "custom_field",
        "name": source.name,
        "dataset": source.field_dataset,
        "normal_axis": source.normal_axis,
        "source_time": _compile_time_shifted_source_time(source, default_frequency=default_frequency),
    }


def _compile_mode_source(source: ModeSource, *, default_frequency: float) -> dict:
    # Broadband injection: the guided profile and effective index are solved once
    # at the waveform's center frequency and driven by the temporal envelope, so a
    # pulsed or sampled-table mode source excites the whole band from one run.
    source_time = _compile_time_shifted_source_time(source, default_frequency=default_frequency)

    direction_sign = 1 if source.direction == "+" else -1
    direction_vector = {
        "x": (float(direction_sign), 0.0, 0.0),
        "y": (0.0, float(direction_sign), 0.0),
        "z": (0.0, 0.0, float(direction_sign)),
    }[source.normal_axis]
    compiled = {
        "kind": "mode_source",
        "name": source.name,
        "position": source.position,
        "size": source.size,
        "mode_index": int(source.mode_index),
        "direction": source.direction,
        "direction_sign": int(direction_sign),
        "direction_vector": direction_vector,
        "normal_axis": source.normal_axis,
        "polarization": source.polarization,
        "polarization_axis": source.polarization_axis,
        "bend_radius": source.bend_radius,
        "bend_axis": source.bend_axis,
        "wave_family": getattr(source, "_wave_family", None),
        "conductor_model": getattr(source,"_conductor_model",None),
        "injection": {"kind": "soft"},
        "source_time": source_time,
    }
    terminal_mode = getattr(source, "_terminal_mode", None)
    if terminal_mode is not None:
        compiled["terminal_mode"] = terminal_mode
    return compiled


# Sources injected along a single carrier direction. Under fixed k the incident
# direction must vary with frequency, so their broadband injection is not exact,
# except for TFSF-slab plane waves, which the complex fixed-k incident line drives
# exactly at every frequency. Point and current sources are fixed-k exact at every
# frequency (the Bloch boundary supplies the per-frequency phase) and stay admitted.
_DIRECTIONAL_SOURCE_KINDS = frozenset(
    ("plane_wave", "gaussian_beam", "astigmatic_gaussian_beam", "mode_source", "custom_field"))


def _fixed_k_exact_broadband(source):
    injection = source.get("injection", {})
    return source["kind"] == "plane_wave" and injection.get("kind") == "tfsf" and injection.get("mode") == "slab"


def validate_fdtd_source_times(solver):
    if solver.complex_fields_enabled and any(
        k != 0.0 and solver.scene.boundary.axis_kind(axis) == "bloch"
        for axis, k in zip("xyz", solver.resolved_bloch_wavevector)
    ) and any(source["kind"] in _DIRECTIONAL_SOURCE_KINDS and source["source_time"]["kind"] != "cw"
              and not _fixed_k_exact_broadband(source)
              for source in getattr(solver, "_compiled_sources", ())):
        raise NotImplementedError(
            "Broadband fixed-k injection of beams, mode and field sources and of non-slab plane "
            "waves is not yet exact off the carrier; use a CW source per frequency. TFSF-slab plane "
            "waves, point and current sources are admitted. See "
            "docs/memory/bloch-broadband-tfsf-injection-2026-10-31.md."
        )


def compile_fdtd_sources(scene, *, default_frequency: float):
    compiled = []
    sources = scene.resolved_sources() if hasattr(scene, "resolved_sources") else scene.sources
    for source in sources:
        if isinstance(source, PointDipole):
            compiled.append(_compile_point_dipole(source, default_frequency=default_frequency))
        elif isinstance(source, PlaneWave):
            compiled.append(_compile_plane_wave(source, default_frequency=default_frequency))
        elif isinstance(source, AstigmaticGaussianBeam):
            compiled.append(_compile_astigmatic_gaussian_beam(source, default_frequency=default_frequency))
        elif isinstance(source, GaussianBeam):
            compiled.append(_compile_gaussian_beam(source, default_frequency=default_frequency))
        elif isinstance(source, ModeSource):
            compiled.append(_compile_mode_source(source, default_frequency=default_frequency))
        elif isinstance(source, UniformCurrentSource):
            compiled.append(_compile_uniform_current_source(source, default_frequency=default_frequency))
        elif isinstance(source, CustomCurrentSource):
            compiled.append(_compile_custom_current_source(source, default_frequency=default_frequency))
        elif isinstance(source, CustomFieldSource):
            compiled.append(_compile_custom_field_source(source, default_frequency=default_frequency))
        else:
            raise ValueError(f"Unsupported FDTD source type: {type(source).__name__}")
    return compiled
