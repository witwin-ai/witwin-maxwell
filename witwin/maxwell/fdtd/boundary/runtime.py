from __future__ import annotations

import torch

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
    for axis, entry in zip(("x", "y", "z"), symmetry):
        if entry is None:
            continue
        mode, face = entry
        if boundary.face_kind(axis, face) not in {"none", "pml"}:
            raise ValueError(
                f"Scene.symmetry on the {axis}-{face} face requires "
                "BoundarySpec.none() or BoundarySpec.pml(...)."
            )
        face_index = 0 if face == "low" else 1
        face_codes[axis][face_index] = BOUNDARY_KIND_TO_CODE[mode.lower()]

    _validate_symmetry_source_placement(solver.scene, symmetry)

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


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _source_center(source):
    """Return a point-like source center coordinate tuple, or ``None`` if the
    source spans the domain (e.g. plane waves) and has no single position."""
    for attr in ("position", "center"):
        value = getattr(source, attr, None)
        if value is not None:
            return tuple(float(v) for v in value)
    return None


def _validate_symmetry_source_placement(scene, symmetry):
    """Reject point-like sources located in the folded-away half of the domain.

    A source may sit on the symmetry plane or anywhere in the kept half; a source
    beyond the plane lives in the removed half and cannot be represented on the
    folded grid, so we raise clearly instead of silently dropping it.
    """
    if all(entry is None for entry in symmetry):
        return
    resolver = getattr(scene, "resolved_sources", None)
    sources = resolver() if callable(resolver) else getattr(scene, "sources", ())
    bounds = scene.domain.bounds
    tol = 1e-9
    for source in sources:
        center = _source_center(source)
        if center is None:
            continue
        for axis, entry in zip(("x", "y", "z"), symmetry):
            if entry is None:
                continue
            _mode, face = entry
            axis_idx = _AXIS_INDEX[axis]
            low, high = float(bounds[axis_idx][0]), float(bounds[axis_idx][1])
            coord = center[axis_idx]
            span = max(high - low, tol)
            plane = low if face == "low" else high
            if face == "low" and coord < plane - tol * span:
                raise ValueError(
                    f"Source at {axis}={coord:g} lies in the folded-away half below the "
                    f"{axis}-low symmetry plane at {plane:g}; place it on the plane or in "
                    "the kept half of the domain."
                )
            if face == "high" and coord > plane + tol * span:
                raise ValueError(
                    f"Source at {axis}={coord:g} lies in the folded-away half above the "
                    f"{axis}-high symmetry plane at {plane:g}; place it on the plane or in "
                    "the kept half of the domain."
                )


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

    scene = solver.scene
    axis_primal = {"x": scene.dx_primal64, "y": scene.dy_primal64, "z": scene.dz_primal64}
    speed_dt = solver.c * solver.dt
    for axis, side in solver.mur_faces:
        # Mur advection uses the local boundary-cell spacing of each face. The
        # coefficient depends only on c, dt, and that spacing, so it is constant
        # across time steps and precomputed once here.
        primal = axis_primal[axis]
        delta = float(primal[0] if side == "low" else primal[-1])
        coef = (speed_dt - delta) / (speed_dt + delta)
        for field_name, field_axis in _MUR_FACE_COMPONENTS[axis]:
            field = getattr(solver, field_name)
            size = int(field.shape[field_axis])
            boundary_index = 0 if side == "low" else size - 1
            adjacent_index = 1 if side == "low" else size - 2
            # Persistent boundary / first-interior plane buffers, updated in place
            # by the Mur kernel each step (no per-step allocation). They start at
            # zero, matching the initial field state.
            plane_shape = tuple(field.shape[:field_axis] + field.shape[field_axis + 1:])
            solver._mur_state.append(
                {
                    "field": field_name,
                    "axis": int(field_axis),
                    "boundary_index": int(boundary_index),
                    "adjacent_index": int(adjacent_index),
                    "coef": float(coef),
                    "prev_boundary": torch.zeros(plane_shape, device=solver.device, dtype=field.dtype),
                    "prev_adjacent": torch.zeros(plane_shape, device=solver.device, dtype=field.dtype),
                }
            )
