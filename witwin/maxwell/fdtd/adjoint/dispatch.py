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


_VALID_ADJOINT_BACKENDS = {"auto", "native", "torch_reference", "torch_vjp"}

class _ReverseBackend(Enum):
    GRATING_TFSF = auto()
    TFSF = auto()
    PYTHON_BLOCH_DISPERSIVE = auto()
    PYTHON_BLOCH = auto()
    PYTHON_DISPERSIVE = auto()
    PYTHON_STANDARD = auto()
    PYTHON_CPML = auto()
    PYTHON_CONDUCTIVE = auto()
    PYTHON_KERR = auto()
    PYTHON_FULL_ANISO = auto()
    TORCH_VJP = auto()


# Native CUDA reverse-execution backends mirror the analytic torch reference
# variants one-for-one. Each label is the value recorded into
# ``_BackwardProfiler.reverse_backend_counts`` so a native reverse step stays
# attributable per configuration (``native_standard`` mirrors
# ``python_reference_standard`` and so on). The runner registry is populated by
# later P6 items as the fused CUDA reverse kernels land; until an entry exists
# ``auto`` mode never selects native and an explicit ``native`` override raises.
_NATIVE_REVERSE_LABELS: dict[_ReverseBackend, str] = {
    _ReverseBackend.PYTHON_STANDARD: "native_standard",
    _ReverseBackend.PYTHON_CPML: "native_cpml",
    _ReverseBackend.PYTHON_CONDUCTIVE: "native_conductive",
    _ReverseBackend.PYTHON_KERR: "native_kerr",
    _ReverseBackend.PYTHON_FULL_ANISO: "native_full_aniso",
    _ReverseBackend.PYTHON_BLOCH: "native_bloch",
    _ReverseBackend.PYTHON_BLOCH_DISPERSIVE: "native_bloch_dispersive",
    _ReverseBackend.PYTHON_DISPERSIVE: "native_dispersive",
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
# missing-extension host ``auto`` mode falls back to the analytic reference and a
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


def resolve_fdtd_adjoint_backend_name(requested: str | None = None) -> str:
    """Resolve which FDTD adjoint reverse-execution backend :func:`reverse_step` uses.

    The value comes from the explicit ``requested`` argument, otherwise the
    ``WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND`` environment variable, otherwise
    ``"auto"``. The returned (lower-cased, stripped) name is one of:

    - ``auto``: prefer the native CUDA reverse backend when one is registered for
      the resolved per-step configuration, otherwise the analytic torch reference
      backend, otherwise the torch-autograd VJP fallback.
    - ``native``: force the native CUDA reverse backend; error when none is
      registered for the configuration.
    - ``torch_reference``: force the analytic torch reference backend; error when
      the configuration only supports the VJP fallback.
    - ``torch_vjp``: force the torch-autograd VJP fallback.
    """
    import os

    backend = (requested or os.environ.get("WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND", "auto")).strip().lower()
    if backend not in _VALID_ADJOINT_BACKENDS:
        choices = ", ".join(sorted(_VALID_ADJOINT_BACKENDS))
        raise ValueError(f"WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND must be one of: {choices}.")
    return backend


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


def _supports_standard(runtime, solver, forward_state, resolved_source_terms) -> bool:
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
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
    magnetic dispersion, conduction, nonlinearity, full anisotropy, or TFSF it still
    routes to the torch-VJP fallback."""
    if not has_complex_fields(solver):
        return False
    if not getattr(solver, "electric_dispersive_enabled", False):
        return False
    if getattr(solver, "magnetic_dispersive_enabled", False):
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
    or TFSF still falls to the torch-VJP fallback."""
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
    (chi2 / two-photon absorption, which also rewrites ``decay``) still routes to
    the torch-VJP fallback, as does Kerr combined with an open boundary, ADE
    dispersion, conduction, full anisotropy, complex fields, or TFSF."""
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
    an open boundary, or combined with conduction/dispersion/nonlinearity/complex
    fields/TFSF, still routes to the torch-VJP fallback."""
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
    supports_cpml = _supports_cpml(runtime, solver, forward_state, resolved_source_terms)
    supports_standard = _supports_standard(runtime, solver, forward_state, resolved_source_terms)
    supports_dispersive = _supports_dispersive(runtime, solver, forward_state, resolved_source_terms)
    supports_bloch = _supports_bloch(runtime, solver, forward_state, resolved_source_terms)
    supports_bloch_dispersive = _supports_bloch_dispersive(runtime, solver, forward_state, resolved_source_terms)
    supports_conductive = _supports_conductive(runtime, solver, forward_state, resolved_source_terms)
    supports_kerr = _supports_kerr(runtime, solver, forward_state, resolved_source_terms)
    supports_full_aniso = _supports_full_aniso(runtime, solver, forward_state, resolved_source_terms)

    decision_table = (
        (_ReverseBackend.GRATING_TFSF, supports_grating_tfsf),
        (_ReverseBackend.TFSF, supports_tfsf),
        (_ReverseBackend.PYTHON_BLOCH_DISPERSIVE, supports_bloch_dispersive),
        (_ReverseBackend.PYTHON_BLOCH, supports_bloch),
        (_ReverseBackend.PYTHON_DISPERSIVE, supports_dispersive),
        (_ReverseBackend.PYTHON_STANDARD, supports_standard),
        (_ReverseBackend.PYTHON_CPML, supports_cpml),
        (_ReverseBackend.PYTHON_CONDUCTIVE, supports_conductive),
        (_ReverseBackend.PYTHON_KERR, supports_kerr),
        (_ReverseBackend.PYTHON_FULL_ANISO, supports_full_aniso),
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


def _execute_reference_backend(
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
    adjoint_reference,
    finish,
    forward_magnetic_fields=None,
):
    """Run one of the analytic torch reference reverse backends.

    This is the exact per-variant dispatch the ``auto`` and ``torch_reference``
    modes share; ``TORCH_VJP`` is handled by the caller and never reaches here.

    ``forward_magnetic_fields`` is the mid-step H the checkpoint replay already
    reconstructed; when present it is handed to the standard / CPML backends so
    they skip their own magnetic half-step recompute. The replay only supplies it
    for the pure real configuration those backends serve (see
    ``_replay_can_capture_mid_magnetic``), where it matches the recompute exactly.
    """
    if backend is _ReverseBackend.TFSF:
        return _with_profile_sections(
            profiler,
            lambda: adjoint_reference.reverse_step_tfsf(
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
        return adjoint_reference.reverse_step_grating_tfsf(
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
                lambda: adjoint_reference.reverse_step_bloch_python_reference(
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
    if backend is _ReverseBackend.PYTHON_BLOCH_DISPERSIVE:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: adjoint_reference.reverse_step_bloch_dispersive_python_reference(
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
                lambda: adjoint_reference.reverse_step_dispersive_python_reference(
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
                lambda: adjoint_reference.reverse_step_standard_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                    magnetic_fields=forward_magnetic_fields,
                ),
            )
        )
    if backend is _ReverseBackend.PYTHON_CPML:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: adjoint_reference.reverse_step_cpml_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                    magnetic_fields=forward_magnetic_fields,
                ),
            )
        )
    if backend is _ReverseBackend.PYTHON_CONDUCTIVE:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: adjoint_reference.reverse_step_conductive_cpml_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                    magnetic_fields=forward_magnetic_fields,
                ),
            )
        )
    if backend is _ReverseBackend.PYTHON_KERR:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: adjoint_reference.reverse_step_kerr_cpml_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                    magnetic_fields=forward_magnetic_fields,
                ),
            )
        )
    if backend is _ReverseBackend.PYTHON_FULL_ANISO:
        return finish(
            _with_profile_sections(
                profiler,
                lambda: adjoint_reference.reverse_step_full_aniso_cpml_python_reference(
                    solver,
                    forward_state,
                    adjoint_state,
                    time_value=time_value,
                    eps_ex=eps_ex,
                    eps_ey=eps_ey,
                    eps_ez=eps_ez,
                    resolved_source_terms=resolved_source_terms,
                    magnetic_fields=forward_magnetic_fields,
                ),
            )
        )
    raise RuntimeError(f"Unsupported reference reverse backend selection: {backend!r}")


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
    from . import reference as _adjoint_reference

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
    mode = resolve_fdtd_adjoint_backend_name()

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

    def run_reference():
        return _execute_reference_backend(
            analytic_backend,
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=resolved_source_terms,
            profiler=profiler,
            adjoint_reference=_adjoint_reference,
            finish=finish,
            forward_magnetic_fields=forward_magnetic_fields,
        )

    def run_torch_vjp():
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

    def run_native():
        # Account the native reverse step under the same per-step profiler
        # sections the analytic reference path records, so the backward profile
        # stays consistent regardless of which reverse backend auto mode selects.
        return _with_profile_sections(
            profiler,
            lambda: _run_native_reverse_step(
                analytic_backend,
                solver,
                forward_state,
                adjoint_state,
                time_value=time_value,
                eps_ex=eps_ex,
                eps_ey=eps_ey,
                eps_ez=eps_ez,
                resolved_source_terms=resolved_source_terms,
                profiler=profiler,
            ),
        )

    # The torch-VJP fallback has no native mirror; native applies only to the
    # analytic reference variants that a fused CUDA reverse kernel can replace,
    # and only when the runner's qualifier accepts this scene (CUDA + extension).
    native_available = (
        analytic_backend is not _ReverseBackend.TORCH_VJP
        and _native_backend_available(analytic_backend, solver, forward_state)
    )

    if mode == "torch_vjp":
        return run_torch_vjp()
    if mode == "native":
        if not native_available:
            raise ValueError(
                "WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND='native' was requested but no "
                "native CUDA reverse backend is registered for this configuration "
                f"(analytic backend {analytic_backend.name})."
            )
        return run_native()
    if mode == "torch_reference":
        if analytic_backend is _ReverseBackend.TORCH_VJP:
            raise ValueError(
                "WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND='torch_reference' was requested "
                "but this configuration only supports the torch-autograd VJP fallback."
            )
        return run_reference()

    # mode == "auto": prefer native, then the analytic reference, then the VJP fallback.
    if native_available:
        return run_native()
    if analytic_backend is _ReverseBackend.TORCH_VJP:
        return run_torch_vjp()
    return run_reference()
