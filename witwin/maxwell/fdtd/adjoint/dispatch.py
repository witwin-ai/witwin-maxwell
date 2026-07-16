from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto


from ..boundary import (
    BOUNDARY_BLOCH,
    BOUNDARY_NONE,
    BOUNDARY_PERIODIC,
    BOUNDARY_PMC,
    has_complex_fields,
)
from ..checkpoint import checkpoint_schema


class _ReverseBackend(Enum):
    GENERAL_NONLINEAR = auto()
    MIXED_BLOCH_CPML = auto()
    GRATING_TFSF = auto()
    TFSF = auto()
    BLOCH_DISPERSIVE = auto()
    BLOCH = auto()
    DISPERSIVE = auto()
    STANDARD = auto()
    CPML = auto()
    CONDUCTIVE = auto()
    KERR = auto()
    FULL_ANISO = auto()
    WIRE_STANDARD = auto()
    WIRE_CPML = auto()


# Native CUDA reverse-execution backends mirror the native CUDA
# variants one-for-one. Each label is the value recorded into
# ``_BackwardProfiler.reverse_backend_counts`` so a native reverse step stays
# attributable per configuration (``native_standard`` mirrors
# ``native_standard`` and so on). The runner registry is populated by
# later P6 items as the fused CUDA reverse kernels land; until an entry exists
# ``auto`` mode never selects native and an explicit ``native`` override raises.
_NATIVE_REVERSE_LABELS: dict[_ReverseBackend, str] = {
    _ReverseBackend.GENERAL_NONLINEAR: "native_general_nonlinear",
    _ReverseBackend.MIXED_BLOCH_CPML: "native_mixed_bloch_cpml",
    _ReverseBackend.STANDARD: "native_standard",
    _ReverseBackend.CPML: "native_cpml",
    _ReverseBackend.CONDUCTIVE: "native_conductive",
    _ReverseBackend.KERR: "native_kerr",
    _ReverseBackend.FULL_ANISO: "native_full_aniso",
    _ReverseBackend.WIRE_STANDARD: "native_wire_standard",
    _ReverseBackend.WIRE_CPML: "native_wire_cpml",
    _ReverseBackend.BLOCH: "native_bloch",
    _ReverseBackend.BLOCH_DISPERSIVE: "native_bloch_dispersive",
    _ReverseBackend.DISPERSIVE: "native_dispersive",
    _ReverseBackend.TFSF: "native_tfsf",
    _ReverseBackend.GRATING_TFSF: "native_grating_tfsf",
}

# A registered runner takes the same (solver, forward_state, adjoint_state,
# time_value, eps_*, resolved_source_terms, profiler) contract as the analytic
# reference backends and returns a fully-formed ``_ReverseStepResult`` whose
# ``backend`` field is the matching ``_NATIVE_REVERSE_LABELS`` value.
_NativeReverseRunner = Callable[..., object]
_NATIVE_REVERSE_RUNNERS: dict[_ReverseBackend, _NativeReverseRunner] = {}

# A runner may carry a ``(solver, forward_state) -> bool`` qualifier stashed on
# the callable itself (``_native_reverse_qualifier``). The real native runners
# require a CUDA scene with the compiled extension present, so on CPU or a
# missing-extension host ``auto`` mode falls back to the native implementation and a
# forced ``native`` override raises. The qualifier travels with the runner so a
# test that swaps in its own runner (no qualifier attribute) qualifies anywhere.
_NATIVE_QUALIFIER_ATTR = "_native_reverse_qualifier"


def register_native_reverse_backend(
    backend: _ReverseBackend,
    runner: _NativeReverseRunner,
    *,
    qualifier: Callable[..., bool] | None = None,
) -> None:
    """Register a native CUDA reverse-step runner for an analytic backend variant."""
    if backend not in _NATIVE_REVERSE_LABELS:
        raise ValueError(f"No native reverse label is defined for backend {backend!r}.")
    if qualifier is not None:
        setattr(runner, _NATIVE_QUALIFIER_ATTR, qualifier)
    _NATIVE_REVERSE_RUNNERS[backend] = runner


def unregister_native_reverse_backend(backend: _ReverseBackend) -> None:
    """Drop a previously registered native runner (primarily for tests)."""
    _NATIVE_REVERSE_RUNNERS.pop(backend, None)


def _native_backend_available(backend: _ReverseBackend, solver=None, forward_state=None) -> bool:
    runner = _NATIVE_REVERSE_RUNNERS.get(backend)
    if runner is None:
        return False
    qualifier = getattr(runner, _NATIVE_QUALIFIER_ATTR, None)
    if qualifier is None:
        return True
    if solver is None:
        # No scene context to evaluate against: report registry membership only.
        return True
    return bool(qualifier(solver, forward_state))


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


def _has_open_face_conflicts(face_codes: tuple[int, int, int, int, int, int]) -> bool:
    return any(code in {BOUNDARY_PERIODIC, BOUNDARY_PMC, BOUNDARY_BLOCH} for code in face_codes)


def _supports_explicit_source_step(runtime, solver, resolved_source_terms) -> bool:
    return runtime._can_use_explicit_source_term_reverse_step(solver, resolved_source_terms)


def _supports_tfsf(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
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
    if getattr(solver, "full_aniso_enabled", False):
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

    # A grating slab is a single absorbing (PML) normal axis with the two
    # transverse axes periodic (Bloch), for any choice of normal axis. The reverse
    # split-field replay carries the wrap phase on the two Bloch axes and the
    # recursive-convolution stretch on the single PML axis; ``_bloch_cpml_pml_axis``
    # recognizes exactly that layout and returns the absorbing axis.
    from ..runtime.stepping import _bloch_cpml_pml_axis

    pml_axis = _bloch_cpml_pml_axis(solver)
    if pml_axis is None:
        return False

    tfsf_state = getattr(solver, "_tfsf_state", None)
    if tfsf_state is None:
        return False
    return (
        tfsf_state.get("provider") == "plane_wave_grating_slab_cw"
        and tfsf_state.get("mode") == "slab"
        and tfsf_state.get("axis") == pml_axis
        and "tfsf_aux_electric" not in forward_state
        and "tfsf_aux_magnetic" not in forward_state
    )


def _supports_mixed_bloch_cpml(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if not has_complex_fields(solver) or not getattr(solver, "uses_cpml", False):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if getattr(solver, "nonlinear_enabled", False) or getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    from ..runtime.stepping import _bloch_cpml_pml_axis

    return (
        _bloch_cpml_pml_axis(solver) is not None
        and _supports_explicit_source_step(runtime, solver, resolved_source_terms)
    )


def _supports_standard(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
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
    if getattr(solver, "full_aniso_enabled", False):
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


def _supports_wire(runtime, solver, forward_state, resolved_source_terms, *, cpml: bool) -> bool:
    if getattr(solver, "_wire_runtime", None) is None:
        return False
    if bool(getattr(solver, "uses_cpml", False)) != bool(cpml):
        return False
    if any(
        bool(getattr(solver, name, False))
        for name in (
            "nonlinear_enabled",
            "conductive_enabled",
            "full_aniso_enabled",
            "dispersive_enabled",
            "magnetic_dispersive_enabled",
            "tfsf_enabled",
        )
    ):
        return False
    if has_complex_fields(solver) or not _matches_checkpoint_layout(solver, forward_state):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _supports_dispersive(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
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
    if getattr(solver, "full_aniso_enabled", False):
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


def _supports_bloch_dispersive(runtime, solver, forward_state, resolved_source_terms) -> bool:
    """Analytic native reverse for a Bloch (complex-field) + electric-dispersive medium.

    The complex-field solver advances a real and an imaginary FDTD copy that couple
    only through the Bloch boundary wrap, and the electric ADE poles advance on each
    copy with the same real coefficients. The reverse is the complex Bloch base
    reverse plus the electric-dispersive correction / ADE-state VJP applied to both
    halves, so this variant carries the imaginary-ADE replica the plain Bloch and
    plain dispersive backends each drop. Gated to the pure Bloch (all-faces-Bloch,
    non-CPML) electric-dispersive class; combined with an absorbing (CPML) axis,
    conduction, nonlinearity, full anisotropy, or TFSF combinations are handled
    by another native specialization or rejected during preparation."""
    if not has_complex_fields(solver):
        return False
    if not getattr(solver, "dispersive_enabled", False):
        return False
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
        return False
    if getattr(solver, "uses_cpml", False):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if not all(code == BOUNDARY_BLOCH for code in _face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _supports_conductive(runtime, solver, forward_state, resolved_source_terms) -> bool:
    """Analytic native reverse for a static-conductive (sigma_e) CPML medium.

    The semi-implicit conduction loss makes the electric ``decay`` and ``curl``
    coefficients eps-dependent, so this variant carries the extra eps sensitivity
    the linear CPML reverse drops. Gated to the pure real, single-material
    conductive class on an absorbing (CPML) grid; conduction combined with an
    open boundary, ADE dispersion, nonlinearity, full anisotropy, complex fields,
    or TFSF is rejected during preparation when no native specialization exists."""
    if not getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
        return False
    if not getattr(solver, "uses_cpml", False):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if getattr(solver, "magnetic_dispersive_enabled", False):
        return False
    if has_complex_fields(solver):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _supports_kerr(runtime, solver, forward_state, resolved_source_terms) -> bool:
    """Analytic native reverse for an instantaneous Kerr (chi3) CPML medium.

    The Kerr forward recomputes only the electric ``curl`` coefficient each step
    from the pre-update fields (``curl = (dt / eff) * decay`` with
    ``eff = eps + eps0 * chi3 * |E|^2``), so the reverse carries the extra field /
    chi3 sensitivity the linear CPML reverse drops. Gated to the pure real,
    curl-only Kerr class on an absorbing (CPML) grid; the general nonlinear kernel
    (chi2 / two-photon absorption, which also rewrites ``decay``) uses the general
    nonlinear native variant. Unsupported combinations fail during preparation."""
    if not getattr(solver, "kerr_enabled", False):
        return False
    if getattr(solver, "nonlinear_general_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
        return False
    if not getattr(solver, "uses_cpml", False):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if getattr(solver, "magnetic_dispersive_enabled", False):
        return False
    if has_complex_fields(solver):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _supports_general_nonlinear(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if not getattr(solver, "nonlinear_general_enabled", False):
        return False
    if has_complex_fields(solver):
        return False
    if getattr(solver, "full_aniso_enabled", False) or getattr(solver, "tfsf_enabled", False):
        return False
    if getattr(solver, "magnetic_dispersive_enabled", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
        return False
    return _supports_explicit_source_step(runtime, solver, resolved_source_terms)


def _supports_full_aniso(runtime, solver, forward_state, resolved_source_terms) -> bool:
    """Analytic native reverse for a full (off-diagonal) anisotropic CPML medium.

    Full anisotropy adds the off-diagonal coupling ``E_i += coeff_ij * <curlH_j>``
    on top of the diagonal CPML electric update. The reverse reuses the linear CPML
    reverse for the diagonal step and folds the transpose of the off-diagonal
    collocated curl(H) into the mid-step H adjoint, so the extra coupling flows the
    cotangent through the field (H) path; the off-diagonal coefficients themselves
    carry no material gradient channel (trainable geometry on a full-anisotropic
    structure is guarded). Gated to the pure real, CPML class with the anisotropic
    structure clear of the absorber (enforced by the bridge ``_full_aniso_cpml_overlap``
    guard, so the un-stretched collocation matches the forward). Full anisotropy on
    an open boundary, or unsupported combinations with conduction, dispersion,
    nonlinearity, complex fields, or TFSF fail during preparation."""
    if not getattr(solver, "full_aniso_enabled", False):
        return False
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if not getattr(solver, "uses_cpml", False):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if getattr(solver, "magnetic_dispersive_enabled", False):
        return False
    if has_complex_fields(solver):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if not _matches_checkpoint_layout(solver, forward_state):
        return False
    if _has_open_face_conflicts(_face_codes(solver)):
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
    supports_mixed_bloch_cpml = _supports_mixed_bloch_cpml(
        runtime, solver, forward_state, resolved_source_terms
    )
    supports_cpml = _supports_cpml(runtime, solver, forward_state, resolved_source_terms)
    supports_standard = _supports_standard(runtime, solver, forward_state, resolved_source_terms)
    supports_dispersive = _supports_dispersive(runtime, solver, forward_state, resolved_source_terms)
    supports_bloch = _supports_bloch(runtime, solver, forward_state, resolved_source_terms)
    supports_bloch_dispersive = _supports_bloch_dispersive(runtime, solver, forward_state, resolved_source_terms)
    supports_conductive = _supports_conductive(runtime, solver, forward_state, resolved_source_terms)
    supports_kerr = _supports_kerr(runtime, solver, forward_state, resolved_source_terms)
    supports_general_nonlinear = _supports_general_nonlinear(
        runtime, solver, forward_state, resolved_source_terms
    )
    supports_full_aniso = _supports_full_aniso(runtime, solver, forward_state, resolved_source_terms)
    supports_wire_standard = _supports_wire(
        runtime, solver, forward_state, resolved_source_terms, cpml=False
    )
    supports_wire_cpml = _supports_wire(
        runtime, solver, forward_state, resolved_source_terms, cpml=True
    )

    if getattr(solver, "_wire_runtime", None) is not None:
        # A non-wire backend would silently omit I/q and coefficient cotangents.
        # Unsupported compositions must fail closed instead of falling through.
        decision_table = (
            (_ReverseBackend.WIRE_STANDARD, supports_wire_standard),
            (_ReverseBackend.WIRE_CPML, supports_wire_cpml),
        )
    else:
        decision_table = (
            (_ReverseBackend.GENERAL_NONLINEAR, supports_general_nonlinear),
            (_ReverseBackend.GRATING_TFSF, supports_grating_tfsf),
            (_ReverseBackend.MIXED_BLOCH_CPML, supports_mixed_bloch_cpml),
            (_ReverseBackend.TFSF, supports_tfsf),
            (_ReverseBackend.BLOCH_DISPERSIVE, supports_bloch_dispersive),
            (_ReverseBackend.BLOCH, supports_bloch),
            (_ReverseBackend.DISPERSIVE, supports_dispersive),
            (_ReverseBackend.STANDARD, supports_standard),
            (_ReverseBackend.CPML, supports_cpml),
            (_ReverseBackend.CONDUCTIVE, supports_conductive),
            (_ReverseBackend.KERR, supports_kerr),
            (_ReverseBackend.FULL_ANISO, supports_full_aniso),
        )
    for backend, enabled in decision_table:
        if enabled:
            return backend
    raise RuntimeError(
        "This differentiable FDTD configuration has no native CUDA adjoint variant. "
        "Adjust the scene to a supported native capability combination."
    )


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


def _run_native_reverse_step(
    backend,
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
    profiler,
):
    """Invoke the registered native CUDA reverse runner for ``backend``.

    The caller guarantees ``backend`` has a registered runner. The runner owns
    the full reverse-step contract (including any source-term gradient
    accumulation) and returns a ``_ReverseStepResult`` labelled with the matching
    ``_NATIVE_REVERSE_LABELS`` value.
    """
    if getattr(solver, "_fdtd_cuda_extension", None) is None:
        from ..cuda.backend import get_compiled_extension

        solver._fdtd_cuda_extension = get_compiled_extension()
    runner = _NATIVE_REVERSE_RUNNERS[backend]
    return runner(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
        profiler=profiler,
    )


def validate_native_adjoint_preparation(solver) -> _ReverseBackend:
    """Require a complete native CUDA reverse variant before time marching."""
    from ..checkpoint import capture_checkpoint_state
    from .capabilities import NATIVE_ADJOINT_CAPABILITIES

    state = capture_checkpoint_state(solver, step=0).tensors
    if not _scene_is_cuda_state(state):
        raise RuntimeError("Differentiable FDTD requires a CUDA scene.")
    runtime = _runtime()
    resolved = runtime._resolved_source_term_lists(
        solver, solver.eps_Ex, solver.eps_Ey, solver.eps_Ez
    )
    backend = _select_reverse_backend(
        solver, state, eps_ex=solver.eps_Ex, eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez, resolved_source_terms=resolved,
    )
    if backend.name not in NATIVE_ADJOINT_CAPABILITIES:
        raise RuntimeError(
            f"Native CUDA adjoint variant {backend.name} is missing from the capability contract."
        )
    if not _native_backend_available(backend, solver, state):
        raise RuntimeError(
            "Differentiable FDTD requires the compiled native CUDA adjoint extension; "
            f"variant {backend.name} is unavailable."
        )
    return backend


def _scene_is_cuda_state(state) -> bool:
    reference = state.get("Ex")
    return bool(reference is not None and reference.is_cuda)


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
    forward_magnetic_fields=None,
    resolved_source_terms=None,
    profiler=None,
):
    runtime = _runtime()

    state_names = tuple(forward_state.keys())
    if tuple(adjoint_state.keys()) != state_names:
        raise RuntimeError(
            "Reverse step expects forward and adjoint states to share the same frozen checkpoint layout."
        )

    # The FDTD backward bridge resolves the source-term lists once per pass and
    # threads them in; direct callers (parity tests) omit them and resolve here.
    if resolved_source_terms is None:
        resolved_source_terms = runtime._resolved_source_term_lists(solver, eps_ex, eps_ey, eps_ez)
    analytic_backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )
    if not _native_backend_available(analytic_backend, solver, forward_state):
        raise RuntimeError(
            "The prepared differentiable FDTD scene lost its native CUDA adjoint "
            f"variant {analytic_backend.name}; backward cannot continue."
        )
    return _with_profile_sections(
        profiler,
        lambda: _run_native_reverse_step(
            analytic_backend, solver, forward_state, adjoint_state,
            time_value=time_value, eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez,
            resolved_source_terms=resolved_source_terms, profiler=profiler,
        ),
    )
