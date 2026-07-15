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
    phase = np.exp(1j * float(source_time["phase"]))
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
    compiled = []
    for source in scene.sources:
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


def _reject_custom_source_time(source_time: dict, *, source_kind: str) -> dict:
    if source_time["kind"] == "custom":
        raise ValueError(
            f"{source_kind} does not support CustomSourceTime; it requires the native time-shifted "
            "injection kernel. Use PointDipole for arbitrary custom waveforms."
        )
    return source_time


def _compile_plane_wave(source: PlaneWave, *, default_frequency: float) -> dict:
    return {
        "kind": "plane_wave",
        "name": source.name,
        "direction": source.direction,
        "polarization": source.polarization,
        "injection": _compile_injection(source.injection),
        "injection_axis": source.injection_axis,
        "source_time": _reject_custom_source_time(
            compile_source_time(source.source_time, default_frequency=default_frequency),
            source_kind="PlaneWave",
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
        "source_time": _reject_custom_source_time(
            compile_source_time(source.source_time, default_frequency=default_frequency),
            source_kind="GaussianBeam",
        ),
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
        "source_time": _reject_custom_source_time(
            compile_source_time(source.source_time, default_frequency=default_frequency),
            source_kind="AstigmaticGaussianBeam",
        ),
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
        "source_time": compile_source_time(source.source_time, default_frequency=default_frequency),
    }


def _compile_mode_source(source: ModeSource, *, default_frequency: float) -> dict:
    # Broadband injection: the guided profile and effective index are solved once
    # at the waveform's center frequency and driven by the temporal envelope, so a
    # GaussianPulse/Ricker mode source excites the whole band from one run. Only
    # CustomSourceTime is rejected because the native time-shifted surface kernel
    # evaluates the analytic CW/Gaussian/Ricker forms on-device and has no table path.
    source_time = _reject_custom_source_time(
        compile_source_time(source.source_time, default_frequency=default_frequency),
        source_kind="ModeSource",
    )

    direction_sign = 1 if source.direction == "+" else -1
    direction_vector = {
        "x": (float(direction_sign), 0.0, 0.0),
        "y": (0.0, float(direction_sign), 0.0),
        "z": (0.0, 0.0, float(direction_sign)),
    }[source.normal_axis]
    return {
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
        "injection": {"kind": "soft"},
        "source_time": source_time,
    }


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
