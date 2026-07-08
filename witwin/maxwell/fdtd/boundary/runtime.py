from __future__ import annotations

from .common import BOUNDARY_KIND_TO_CODE, BOUNDARY_PEC, initialize_complex_fields
from .bloch import resolve_bloch_phase_factors, resolve_bloch_wavevector
from .cpml import (
    initialize_absorber_state,
    initialize_cpml_state,
    initialize_neutral_boundary_state,
    initialize_simple_pml_state,
    initialize_stable_pml_state,
)


def _configure_face_boundary_codes(solver):
    symmetry = getattr(solver.scene, "symmetry", (None, None, None))
    boundary = solver.scene.boundary
    face_codes = {
        axis: [
            BOUNDARY_KIND_TO_CODE[boundary.face_kind(axis, "low")],
            BOUNDARY_KIND_TO_CODE[boundary.face_kind(axis, "high")],
        ]
        for axis in ("x", "y", "z")
    }
    for axis, mode in zip(("x", "y", "z"), symmetry):
        if mode is not None:
            if boundary.face_kind(axis, "low") not in {"none", "pml"}:
                raise ValueError(
                    f"Scene.symmetry on the {axis}-low face requires BoundarySpec.none() or BoundarySpec.pml(...)."
                )
            face_codes[axis][0] = BOUNDARY_KIND_TO_CODE[mode.lower()]

    for axis in ("x", "y", "z"):
        low_code, high_code = (int(face_codes[axis][0]), int(face_codes[axis][1]))
        setattr(solver, f"boundary_{axis}_low_code", low_code)
        setattr(solver, f"boundary_{axis}_high_code", high_code)

    solver.has_pec_faces = any(
        code == BOUNDARY_PEC
        for codes in face_codes.values()
        for code in codes
    )
    solver.has_pml_faces = any(
        code == BOUNDARY_KIND_TO_CODE["pml"]
        for codes in face_codes.values()
        for code in codes
    )
    solver.has_periodic_axes = tuple(
        axis
        for axis in ("x", "y", "z")
        if boundary.axis_kind(axis) == "periodic"
    )
    solver.has_bloch_axes = tuple(
        axis
        for axis in ("x", "y", "z")
        if boundary.axis_kind(axis) == "bloch"
    )
    solver.mur_faces = tuple(
        (axis, side)
        for axis in ("x", "y", "z")
        for side in ("low", "high")
        if boundary.face_kind(axis, side) == "mur"
    )
    solver.has_mur_faces = bool(solver.mur_faces)


def initialize_boundary_state(solver):
    solver.boundary_kind = solver.scene.boundary.kind
    solver.boundary_code = (
        BOUNDARY_KIND_TO_CODE[solver.boundary_kind]
        if solver.boundary_kind in BOUNDARY_KIND_TO_CODE
        else None
    )
    _configure_face_boundary_codes(solver)

    solver.complex_fields_enabled = solver.scene.boundary.uses_kind("bloch")
    if solver.complex_fields_enabled:
        initialize_complex_fields(solver)

    solver.resolved_bloch_wavevector = resolve_bloch_wavevector(solver)
    phases = resolve_bloch_phase_factors(solver)
    solver.boundary_phase_cos = tuple(float(phase.real) for phase in phases)
    solver.boundary_phase_sin = tuple(float(phase.imag) for phase in phases)
    solver.active_absorber_type = solver.absorber_type if solver.scene.boundary.uses_kind("pml") else "none"
    solver.uses_cpml = solver.active_absorber_type in ("cpml", "stablepml")

    if solver.scene.boundary.uses_kind("pml"):
        if solver.absorber_type == "cpml":
            initialize_cpml_state(solver)
        elif solver.absorber_type == "stablepml":
            initialize_stable_pml_state(solver)
        elif solver.absorber_type == "pml":
            initialize_simple_pml_state(solver)
        elif solver.absorber_type == "absorber":
            initialize_absorber_state(solver)
        else:
            raise ValueError(
                "PML boundaries require absorber_type to be 'cpml', 'stablepml', 'pml', or 'absorber'."
            )
    else:
        initialize_neutral_boundary_state(solver)

    initialize_mur_state(solver)


# Tangential electric-field components on each outer face and the Yee grid axis
# they are indexed along for the normal direction of that face.
_MUR_FACE_COMPONENTS = {
    "x": (("Ey", 0), ("Ez", 0)),
    "y": (("Ex", 1), ("Ez", 1)),
    "z": (("Ex", 2), ("Ey", 2)),
}


def initialize_mur_state(solver):
    solver._mur_state = []
    if not getattr(solver, "has_mur_faces", False):
        return

    axis_delta = {"x": solver.dx, "y": solver.dy, "z": solver.dz}
    for axis, side in solver.mur_faces:
        delta = float(axis_delta[axis])
        boundary_index = 0 if side == "low" else -1
        adjacent_index = 1 if side == "low" else -2
        for field_name, field_axis in _MUR_FACE_COMPONENTS[axis]:
            field = getattr(solver, field_name)
            solver._mur_state.append(
                {
                    "field": field_name,
                    "axis": int(field_axis),
                    "boundary_index": boundary_index,
                    "adjacent_index": adjacent_index,
                    "delta": delta,
                    # Previous-step boundary and first-interior slices (start at zero).
                    "prev_boundary": field.select(field_axis, boundary_index).clone(),
                    "prev_adjacent": field.select(field_axis, adjacent_index).clone(),
                }
            )
