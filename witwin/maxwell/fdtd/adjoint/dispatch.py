from __future__ import annotations

from enum import Enum, auto


from ..boundary import (
    BOUNDARY_BLOCH,
    BOUNDARY_NONE,
    BOUNDARY_PERIODIC,
    BOUNDARY_PML,
    BOUNDARY_PMC,
    has_complex_fields,
)
from ..checkpoint import checkpoint_schema


_VALID_ADJOINT_BACKENDS = {"auto", "python"}

class _ReverseBackend(Enum):
    GRATING_TFSF = auto()
    TFSF = auto()
    PYTHON_BLOCH = auto()
    PYTHON_DISPERSIVE = auto()
    PYTHON_STANDARD = auto()
    PYTHON_CPML = auto()
    TORCH_VJP = auto()


def _runtime():
    from . import core as _adjoint

    return _adjoint


def _face_codes(solver) -> tuple[int, int, int, int, int, int]:
    return (
        int(getattr(solver, "boundary_x_low_code", BOUNDARY_NONE)),
        int(getattr(solver, "boundary_x_high_code", BOUNDARY_NONE)),
        int(getattr(solver, "boundary_y_low_code", BOUNDARY_NONE)),
        int(getattr(solver, "boundary_y_high_code", BOUNDARY_NONE)),
        int(getattr(solver, "boundary_z_low_code", BOUNDARY_NONE)),
        int(getattr(solver, "boundary_z_high_code", BOUNDARY_NONE)),
    )


def _matches_checkpoint_layout(solver, forward_state) -> bool:
    return tuple(forward_state.keys()) == checkpoint_schema(solver).state_names


def resolve_fdtd_adjoint_backend_name(requested: str | None = None) -> str:
    import os

    backend = (requested or os.environ.get("WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND", "auto")).strip().lower()
    if backend not in _VALID_ADJOINT_BACKENDS:
        choices = ", ".join(sorted(_VALID_ADJOINT_BACKENDS))
        raise ValueError(f"WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND must be one of: {choices}.")
    return "python"


def _has_open_face_conflicts(face_codes: tuple[int, int, int, int, int, int]) -> bool:
    return any(code in {BOUNDARY_PERIODIC, BOUNDARY_PMC, BOUNDARY_BLOCH} for code in face_codes)


def _supports_explicit_source_step(runtime, solver, resolved_source_terms) -> bool:
    return runtime._can_use_explicit_source_term_reverse_step(solver, resolved_source_terms)


def _supports_tfsf(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if not getattr(solver, "tfsf_enabled", False):
        return False
    if has_complex_fields(solver):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
        return False
    if runtime._has_resolved_source_terms(resolved_source_terms):
        return False

    tfsf_state = getattr(solver, "_tfsf_state", None)
    if tfsf_state is None:
        return False
    provider = tfsf_state.get("provider")
    if provider not in {
        "plane_wave_ref_x_ez",
        "plane_wave_axis_aligned",
        "plane_wave_aux",
        "plane_wave_discrete_cw",
        "analytic_profile",
    }:
        return False

    auxiliary_grid = tfsf_state.get("auxiliary_grid")
    has_auxiliary_state = "tfsf_aux_electric" in forward_state and "tfsf_aux_magnetic" in forward_state
    if auxiliary_grid is None:
        return not has_auxiliary_state
    return has_auxiliary_state


def _supports_grating_tfsf(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if not getattr(solver, "tfsf_enabled", False):
        return False
    if not has_complex_fields(solver):
        return False
    if not getattr(solver, "uses_cpml", False):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if runtime._has_resolved_source_terms(resolved_source_terms):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if tuple(getattr(solver, "has_bloch_axes", ())) != ("x", "y"):
        return False
    face_codes = _face_codes(solver)
    if face_codes != (
        BOUNDARY_BLOCH,
        BOUNDARY_BLOCH,
        BOUNDARY_BLOCH,
        BOUNDARY_BLOCH,
        BOUNDARY_PML,
        BOUNDARY_PML,
    ):
        return False

    tfsf_state = getattr(solver, "_tfsf_state", None)
    if tfsf_state is None:
        return False
    return (
        tfsf_state.get("provider") == "plane_wave_grating_slab_cw"
        and tfsf_state.get("mode") == "slab"
        and tfsf_state.get("axis") == "z"
        and "tfsf_aux_electric" not in forward_state
        and "tfsf_aux_magnetic" not in forward_state
    )


def _supports_standard(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if tuple(forward_state.keys()) != ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        return False
    if getattr(solver, "uses_cpml", False):
        return False
    if has_complex_fields(solver):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _with_profile_sections(profiler, fn):
    if profiler is None:
        return fn()
    with profiler.section("state_clone"):
        pass
    with profiler.section("step_forward"):
        result = fn()
    with profiler.section("step_vjp"):
        pass
    return result


def _supports_cpml(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if not getattr(solver, "uses_cpml", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if has_complex_fields(solver):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _supports_dispersive(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if not getattr(solver, "dispersive_enabled", False):
        return False
    if has_complex_fields(solver):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _supports_bloch(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "uses_cpml", False):
        return False
    if not has_complex_fields(solver):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if not all(code == BOUNDARY_BLOCH for code in _face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _select_reverse_backend(
    solver,
    forward_state,
    *,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
) -> _ReverseBackend:
    runtime = _runtime()

    supports_tfsf = _supports_tfsf(runtime, solver, forward_state, resolved_source_terms)
    supports_grating_tfsf = _supports_grating_tfsf(runtime, solver, forward_state, resolved_source_terms)
    supports_cpml = _supports_cpml(runtime, solver, forward_state, resolved_source_terms)
    supports_standard = _supports_standard(runtime, solver, forward_state, resolved_source_terms)
    supports_dispersive = _supports_dispersive(runtime, solver, forward_state, resolved_source_terms)
    supports_bloch = _supports_bloch(runtime, solver, forward_state, resolved_source_terms)

    decision_table = (
        (_ReverseBackend.GRATING_TFSF, supports_grating_tfsf),
        (_ReverseBackend.TFSF, supports_tfsf),
        (_ReverseBackend.PYTHON_BLOCH, supports_bloch),
        (_ReverseBackend.PYTHON_DISPERSIVE, supports_dispersive),
        (_ReverseBackend.PYTHON_STANDARD, supports_standard),
        (_ReverseBackend.PYTHON_CPML, supports_cpml),
        (_ReverseBackend.TORCH_VJP, True),
    )
    for backend, enabled in decision_table:
        if enabled:
            return backend
    raise RuntimeError("Reverse backend decision table did not produce a backend.")


def _accumulate_source_term_gradients(
    runtime,
    step_result,
    *,
    solver,
    adjoint_state,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
):
    return runtime._accumulate_source_term_gradients(
        step_result,
        solver=solver,
        adjoint_state=adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def reverse_step(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    chi3_ex=None,
    chi3_ey=None,
    chi3_ez=None,
    chi2_ex=None,
    chi2_ey=None,
    chi2_ez=None,
    tpa_ex=None,
    tpa_ey=None,
    tpa_ez=None,
    profiler=None,
):
    runtime = _runtime()
    from . import reference as _adjoint_reference

    state_names = tuple(forward_state.keys())
    if tuple(adjoint_state.keys()) != state_names:
        raise RuntimeError(
            "Reverse step expects forward and adjoint states to share the same frozen checkpoint layout."
        )

    resolved_source_terms = runtime._resolved_source_term_lists(solver, eps_ex, eps_ey, eps_ez)
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )

    def finish(step_result):
        return _accumulate_source_term_gradients(
            runtime,
            step_result,
            solver=solver,
            adjoint_state=adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=resolved_source_terms,
        )

    if backend is _ReverseBackend.TFSF:
        return _with_profile_sections(
            profiler,
            lambda: _adjoint_reference.reverse_step_tfsf(
                solver,
                forward_state,
                adjoint_state,
                time_value=time_value,
                eps_ex=eps_ex,
                eps_ey=eps_ey,
                eps_ez=eps_ez,
                resolved_source_terms=resolved_source_terms,
                profiler=None,
            ),
        )
    if backend is _ReverseBackend.GRATING_TFSF:
        return _adjoint_reference.reverse_step_grating_tfsf(
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            profiler=profiler,
        )
    if backend is _ReverseBackend.PYTHON_BLOCH:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: _adjoint_reference.reverse_step_bloch_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                ),
            )
        )
    if backend is _ReverseBackend.PYTHON_DISPERSIVE:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: _adjoint_reference.reverse_step_dispersive_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                ),
            )
        )
    if backend is _ReverseBackend.PYTHON_STANDARD:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: _adjoint_reference.reverse_step_standard_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                ),
            )
        )
    if backend is _ReverseBackend.PYTHON_CPML:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: _adjoint_reference.reverse_step_cpml_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                ),
            )
        )
    if backend is _ReverseBackend.TORCH_VJP:
        return _adjoint_reference.reverse_step_torch_vjp(
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            chi3_ex=chi3_ex,
            chi3_ey=chi3_ey,
            chi3_ez=chi3_ez,
            chi2_ex=chi2_ex,
            chi2_ey=chi2_ey,
            chi2_ez=chi2_ez,
            tpa_ex=tpa_ex,
            tpa_ey=tpa_ey,
            tpa_ez=tpa_ez,
            profiler=profiler,
        )
    raise RuntimeError(f"Unsupported reverse backend selection: {backend!r}")
