from __future__ import annotations

from .common import BOUNDARY_BLOCH, BOUNDARY_KIND_TO_CODE, BOUNDARY_PEC, initialize_complex_fields
from .bloch import resolve_bloch_phase_factors, resolve_bloch_wavevector, validate_grating_tfsf_slab_topology
from .cpml import initialize_cpml_state, initialize_neutral_boundary_state, initialize_simple_pml_state


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


def initialize_boundary_state(solver):
    solver.boundary_kind = solver.scene.boundary.kind
    has_tfsf_slab = any(
        source.get("injection", {}).get("kind") == "tfsf"
        and source.get("injection", {}).get("mode") == "slab"
        for source in getattr(solver, "_compiled_sources", ()) or ()
    )
    if solver.scene.boundary.uses_kind("bloch") and solver.boundary_kind != "bloch":
        if not has_tfsf_slab:
            raise NotImplementedError(
                "FDTD mixed Bloch boundaries are not implemented yet. Use BoundarySpec.bloch(...) or remove Bloch faces."
            )
        validate_grating_tfsf_slab_topology(solver)
    solver.boundary_code = (
        BOUNDARY_KIND_TO_CODE[solver.boundary_kind]
        if solver.boundary_kind in BOUNDARY_KIND_TO_CODE
        else None
    )
    _configure_face_boundary_codes(solver)

    solver.complex_fields_enabled = solver.boundary_code == BOUNDARY_BLOCH
    if solver.complex_fields_enabled:
        initialize_complex_fields(solver)

    solver.resolved_bloch_wavevector = resolve_bloch_wavevector(solver)
    phases = resolve_bloch_phase_factors(solver)
    solver.boundary_phase_cos = tuple(float(phase.real) for phase in phases)
    solver.boundary_phase_sin = tuple(float(phase.imag) for phase in phases)
    solver.active_absorber_type = solver.absorber_type if solver.scene.boundary.uses_kind("pml") else "none"
    solver.uses_cpml = solver.active_absorber_type == "cpml"

    if solver.scene.boundary.uses_kind("pml"):
        if solver.absorber_type == "cpml":
            initialize_cpml_state(solver)
        elif solver.absorber_type == "pml":
            initialize_simple_pml_state(solver)
        else:
            raise ValueError("PML boundaries require absorber_type to be 'cpml' or 'pml'.")
    else:
        initialize_neutral_boundary_state(solver)
