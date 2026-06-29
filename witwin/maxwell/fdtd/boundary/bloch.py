from __future__ import annotations

import math

from ..dispersion import solve_numerical_wavenumber

_AXES = ("x", "y", "z")


def _domain_lengths(solver) -> tuple[float, float, float]:
    domain_range = solver.scene.domain_range
    return (
        float(domain_range[1] - domain_range[0]),
        float(domain_range[3] - domain_range[2]),
        float(domain_range[5] - domain_range[4]),
    )


def _bloch_axes(solver) -> tuple[str, ...]:
    return tuple(axis for axis in _AXES if solver.scene.boundary.axis_kind(axis) == "bloch")


def _tfsf_slab_sources(solver) -> list[dict]:
    return [
        source
        for source in getattr(solver, "_compiled_sources", ()) or ()
        if source.get("injection", {}).get("kind") == "tfsf"
        and source.get("injection", {}).get("mode") == "slab"
    ]


def validate_grating_tfsf_slab_topology(solver) -> dict:
    sources = _tfsf_slab_sources(solver)
    if len(sources) != 1:
        raise ValueError("Mixed Bloch/PML TFSF slab support requires exactly one TFSF slab source.")
    source = sources[0]
    if source.get("kind") != "plane_wave":
        raise ValueError("Mixed Bloch/PML TFSF slab support requires a PlaneWave source.")

    injection = source["injection"]
    normal_axis = injection["axis"]
    boundary = solver.scene.boundary
    normal_faces = boundary.axis_face_kinds(normal_axis)
    if boundary.axis_kind(normal_axis) == "bloch" or normal_faces != ("pml", "pml"):
        raise ValueError("Mixed Bloch/PML TFSF slab support requires the TFSF slab normal axis to use PML.")

    expected_bloch_axes = tuple(axis for axis in _AXES if axis != normal_axis)
    if _bloch_axes(solver) != expected_bloch_axes:
        raise ValueError("Mixed Bloch/PML TFSF slab support requires Bloch axes transverse to the TFSF slab.")
    return source


def _require_auto_bloch_source(solver) -> dict:
    source = validate_grating_tfsf_slab_topology(solver)
    if source["source_time"]["kind"] != "cw":
        raise ValueError("Automatic Bloch wavevector resolution requires a CW PlaneWave source_time.")
    return source


def resolve_bloch_wavevector(solver) -> tuple[float, float, float]:
    boundary = solver.scene.boundary
    wavevector = boundary.bloch_wavevector
    if wavevector != "auto":
        return tuple(float(component) for component in wavevector)

    source = _require_auto_bloch_source(solver)
    k_numeric = solve_numerical_wavenumber(
        solver,
        source["direction"],
        {"x": "dx", "y": "dy", "z": "dz"},
    )
    bloch_axes = set(_bloch_axes(solver))
    return tuple(
        float(k_numeric * float(component)) if axis in bloch_axes else 0.0
        for axis, component in zip(_AXES, source["direction"])
    )


def resolve_bloch_phase_factors(solver) -> tuple[complex, complex, complex]:
    wavevector = getattr(solver, "resolved_bloch_wavevector", None)
    if wavevector is None:
        wavevector = resolve_bloch_wavevector(solver)
    phases = []
    for axis, wave_number, length in zip(_AXES, wavevector, _domain_lengths(solver)):
        if solver.scene.boundary.axis_kind(axis) == "bloch":
            angle = float(wave_number) * float(length)
            phases.append(complex(math.cos(angle), math.sin(angle)))
        else:
            phases.append(1.0 + 0.0j)
    return tuple(phases)
