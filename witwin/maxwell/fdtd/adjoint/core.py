from __future__ import annotations

import math
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ...sources import evaluate_source_time
from ..boundary import (
    BOUNDARY_BLOCH,
    BOUNDARY_NONE,
    BOUNDARY_PEC,
    BOUNDARY_PERIODIC,
    BOUNDARY_PMC,
    BOUNDARY_PML,
    has_complex_fields,
)
from .profiler import (
    _BackwardProfiler,
    _ReverseStepResult,
)
from ..checkpoint import (
    FDTDCheckpointSchema,
    checkpoint_schema,
    clone_checkpoint_tensors,
    dispersive_state_name,
    iter_dispersive_state_specs,
    iter_magnetic_dispersive_state_specs,
    validate_checkpoint_state,
)
from ..excitation.injection import initialize_source_terms
from ..excitation.temporal import _resolve_term_source
from ..excitation.tfsf_specs import reference_sample_axis_code
from ..observers import (
    _align_plane_monitor_payload,
    _compute_plane_flux,
    _monitor_payload_is_point,
    _plane_coord_names,
)

_TFSF_REFERENCE_PROVIDERS = {"plane_wave_ref_x_ez", "plane_wave_axis_aligned"}
_TFSF_AUXILIARY_PROVIDERS = {"plane_wave_ref_x_ez", "plane_wave_axis_aligned", "plane_wave_aux"}


def _monitor_template_is_point(template):
    if "field_indices" in template:
        return True
    if "axis" in template:
        return False
    kind = template.get("kind")
    if kind is not None:
        return kind == "point"
    return False


def _prepare_point_monitor_template(payload, start_index):
    component_indices = {}
    next_index = start_index
    for component_name in payload["fields"]:
        component_indices[component_name] = next_index
        next_index += 1
    template = {
        "kind": "point",
        "fields": tuple(payload["fields"]),
        "field_indices": dict(payload.get("field_indices", {})),
        "samples": payload.get("samples"),
        "frequency": payload.get("frequency"),
        "frequencies": tuple(payload.get("frequencies", (payload.get("frequency"),))),
        "position": payload.get("position"),
        "component_indices": component_indices,
    }
    return template, next_index


def _prepare_plane_monitor_template(payload, start_index):
    component_templates = {}
    next_index = start_index
    for component_name in payload["fields"]:
        component_payload = payload["components"][component_name]
        component_templates[component_name] = {
            "output_index": next_index,
            "coords": component_payload["coords"],
            "plane_index": component_payload["plane_index"],
            "plane_indices": tuple(component_payload.get("plane_indices", (component_payload["plane_index"],))),
            "plane_weights": tuple(component_payload.get("plane_weights", (1.0,))),
            "plane_positions": tuple(component_payload.get("plane_positions", (float(payload["position"]),))),
        }
        next_index += 1
    template = {
        "kind": "plane",
        "monitor_type": payload.get("monitor_type", "plane"),
        "fields": tuple(payload["fields"]),
        "plane_indices": dict(payload.get("plane_indices", {})),
        "samples": payload.get("samples"),
        "frequency": payload.get("frequency"),
        "frequencies": tuple(payload.get("frequencies", (payload.get("frequency"),))),
        "axis": payload["axis"],
        "position": payload["position"],
        "compute_flux": bool(payload.get("compute_flux", False)),
        "normal_direction": payload.get("normal_direction", "+"),
        "mode_spec": payload.get("mode_spec"),
        "components": component_templates,
    }
    return template, next_index


def _prepare_forward_pack(raw_output):
    output_tensors = []
    field_names = []
    for field_name in ("Ex", "Ey", "Ez"):
        value = raw_output.get(field_name)
        if isinstance(value, torch.Tensor):
            field_names.append(field_name)
            output_tensors.append(value)

    monitor_templates = {}
    next_index = len(output_tensors)
    for monitor_name, payload in raw_output.get("observers", {}).items():
        if _monitor_payload_is_point(payload):
            template, next_index = _prepare_point_monitor_template(payload, next_index)
        else:
            template, next_index = _prepare_plane_monitor_template(payload, next_index)
        monitor_templates[monitor_name] = template
        if _monitor_template_is_point(template):
            for component_name in template["fields"]:
                output_tensors.append(payload["components"][component_name])
        else:
            for component_name in template["fields"]:
                output_tensors.append(payload["components"][component_name]["data"])

    return _ForwardPack(
        field_names=tuple(field_names),
        monitor_templates=monitor_templates,
        output_tensors=tuple(output_tensors),
    )


def _finalize_point_monitor_payload(payload):
    primary_component = payload["fields"][0]
    for component_name, value in payload["components"].items():
        payload[component_name] = value
    if len(payload["components"]) == 1:
        payload["component"] = primary_component.lower()
        payload["data"] = payload["components"][primary_component]
        payload["field_index"] = payload["field_indices"][primary_component]
    return payload


def _finalize_plane_monitor_payload(payload):
    axis = payload["axis"]
    coord_names = _plane_coord_names(axis)
    primary_component = payload["fields"][0]

    if len(payload["components"]) == 1:
        component_payload = payload["components"][primary_component]
        payload["component"] = primary_component.lower()
        payload["data"] = component_payload["data"]
        payload["coords"] = component_payload["coords"]
        payload["plane_index"] = component_payload["plane_index"]
        payload["plane_sample_indices"] = component_payload["plane_indices"]
        payload["plane_sample_weights"] = component_payload["plane_weights"]
        payload["plane_sample_positions"] = component_payload["plane_positions"]
        payload[coord_names[0]] = component_payload["coords"][0]
        payload[coord_names[1]] = component_payload["coords"][1]
        payload[primary_component] = component_payload["data"]
        if payload.get("compute_flux"):
            flux = _compute_plane_flux(payload)
            payload["flux"] = flux
            payload["power"] = flux
        return payload

    aligned = _align_plane_monitor_payload(axis, payload["components"])
    if aligned is None:
        if payload.get("compute_flux"):
            raise ValueError(
                f"Monitor {payload!r} is missing aligned tangential fields required for flux integration."
            )
        return payload

    payload[coord_names[0]] = aligned[coord_names[0]]
    payload[coord_names[1]] = aligned[coord_names[1]]
    payload["coords"] = (aligned[coord_names[0]], aligned[coord_names[1]])
    for component_name, value in aligned["fields"].items():
        payload[component_name] = value
    if payload.get("compute_flux"):
        flux = _compute_plane_flux(payload)
        payload["flux"] = flux
        payload["power"] = flux
    return payload


def _rebuild_monitors(monitor_templates, output_tensors, field_offset):
    monitors = {}
    cursor = field_offset
    for monitor_name, template in monitor_templates.items():
        if _monitor_template_is_point(template):
            payload = {
                "kind": "point",
                "fields": template["fields"],
                "components": {},
                "field_indices": dict(template["field_indices"]),
                "samples": template["samples"],
                "frequency": template["frequency"],
                "frequencies": template["frequencies"],
                "position": template["position"],
            }
            for component_name in template["fields"]:
                payload["components"][component_name] = output_tensors[cursor]
                cursor += 1
            monitors[monitor_name] = _finalize_point_monitor_payload(payload)
            continue

        payload = {
            "kind": "plane",
            "monitor_type": template.get("monitor_type", "plane"),
            "fields": template["fields"],
            "components": {},
            "plane_indices": dict(template["plane_indices"]),
            "samples": template["samples"],
            "frequency": template["frequency"],
            "frequencies": template["frequencies"],
            "axis": template["axis"],
            "position": template["position"],
            "compute_flux": template["compute_flux"],
            "normal_direction": template["normal_direction"],
            "mode_spec": template.get("mode_spec"),
        }
        for component_name in template["fields"]:
            component_template = template["components"][component_name]
            payload["components"][component_name] = {
                "data": output_tensors[cursor],
                "coords": component_template["coords"],
                "plane_index": component_template["plane_index"],
                "plane_indices": component_template["plane_indices"],
                "plane_weights": component_template["plane_weights"],
                "plane_positions": component_template["plane_positions"],
            }
            cursor += 1
        monitors[monitor_name] = _finalize_plane_monitor_payload(payload)
    return monitors


def _term_grid_tensor(term):
    for key in ("patch", "delay_patch", "cw_cos_patch", "cw_sin_patch"):
        value = term.get(key)
        if value is not None:
            return value
    raise ValueError("source term does not include a tensor payload.")


def _slice_from_offsets_shape(offsets, shape):
    return tuple(slice(int(offset), int(offset) + int(size)) for offset, size in zip(offsets, shape))


def _safe_grad(grad, reference):
    if grad is None:
        return torch.zeros_like(reference)
    return grad.to(device=reference.device, dtype=reference.dtype)


def _validate_checkpoint_sequence(checkpoints) -> FDTDCheckpointSchema:
    if not checkpoints:
        raise RuntimeError("Adjoint replay requires at least one checkpoint.")
    schema = validate_checkpoint_state(checkpoints[0])
    for checkpoint in checkpoints[1:]:
        validate_checkpoint_state(checkpoint, expected_schema=schema)
    return schema


def _adjoint_kernel_block_size(solver) -> tuple[int, int, int]:
    return tuple(getattr(solver, "kernel_block_size", (256, 1, 1)))


def _adjoint_launch_shape(solver, field_name: str, tensor: torch.Tensor) -> tuple[int, int, int]:
    launch_shapes = getattr(solver, "_field_launch_shapes", {})
    if field_name in launch_shapes:
        return tuple(launch_shapes[field_name])
    compute_shape = getattr(solver, "_compute_linear_launch_shape", None)
    if compute_shape is not None:
        return tuple(compute_shape(int(tensor.numel())))
    block_size = _adjoint_kernel_block_size(solver)
    threads_per_block = int(block_size[0]) * int(block_size[1]) * int(block_size[2])
    grid_x = max(1, (int(tensor.numel()) + threads_per_block - 1) // threads_per_block)
    return (grid_x, 1, 1)


def _checkpoint_stride(simulation, time_steps: int) -> int:
    configured = getattr(simulation.config, "adjoint_checkpoint_stride", None)
    if configured is not None:
        return max(1, int(configured))
    return max(1, int(math.sqrt(max(time_steps, 1))))


def _source_time_kind_code(source_time) -> int:
    return int(source_time["kind_code"])


def _custom_source_time_tensor(source_time, sample_time: torch.Tensor) -> torch.Tensor:
    amplitude = float(source_time["amplitude"])
    times = source_time.get("times")
    if times is None:
        raise NotImplementedError(
            "Callable CustomSourceTime(fn) is not supported in the FDTD adjoint; "
            "provide a sampled (times, amplitudes) table instead."
        )
    xp = sample_time.new_tensor(times)
    fp = sample_time.new_tensor(source_time["amplitudes"])
    upper = torch.searchsorted(xp, sample_time, right=True).clamp(1, xp.numel() - 1)
    lower = upper - 1
    x0 = xp[lower]
    x1 = xp[upper]
    weight = (sample_time - x0) / (x1 - x0)
    interpolated = fp[lower] + weight * (fp[upper] - fp[lower])
    inside = (sample_time >= xp[0]) & (sample_time <= xp[-1])
    interpolated = torch.where(inside, interpolated, torch.zeros_like(interpolated))
    return amplitude * interpolated


def _evaluate_source_time_tensor(source_time, sample_time: torch.Tensor) -> torch.Tensor:
    kind_code = _source_time_kind_code(source_time)
    two_pi = 2.0 * math.pi
    amplitude = float(source_time["amplitude"])
    frequency = float(source_time["frequency"])
    phase = float(source_time.get("phase", 0.0))
    delay = float(source_time.get("delay", 0.0))

    if kind_code == 3:
        return _custom_source_time_tensor(source_time, sample_time)

    if kind_code == 0:
        return amplitude * torch.cos(two_pi * frequency * sample_time + phase)

    if kind_code == 1:
        fwidth = float(source_time["fwidth"])
        sigma_t = 1.0 / max(two_pi * fwidth, 1.0e-30)
        tau = sample_time - delay
        envelope = torch.exp(-0.5 * (tau / sigma_t) ** 2)
        return amplitude * envelope * torch.cos(two_pi * frequency * tau + phase)

    tau = sample_time - delay
    alpha = math.pi * frequency * tau
    alpha_sq = alpha * alpha
    return amplitude * (1.0 - 2.0 * alpha_sq) * torch.exp(-alpha_sq)


def _resolve_cw_signal(source_time, omega, time_value):
    phase = float(omega) * float(time_value) + float(source_time["phase"])
    amplitude = float(source_time["amplitude"])
    return amplitude * math.cos(phase), amplitude * math.sin(phase)


def _dynamic_electric_curl(base_curl: torch.Tensor, base_eps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return ((base_curl * base_eps) / eps).contiguous()


def _build_source_replay_solver(solver, compiled_sources, eps_ex, eps_ey, eps_ez):
    compiled_sources = tuple(compiled_sources)
    primary = compiled_sources[0]
    return SimpleNamespace(
        scene=solver.scene,
        min_dx=solver.min_dx,
        min_dy=solver.min_dy,
        min_dz=solver.min_dz,
        Nx=solver.Nx,
        Ny=solver.Ny,
        Nz=solver.Nz,
        device=solver.device,
        Ex=eps_ex,
        Ey=eps_ey,
        Ez=eps_ez,
        Hx=solver.Hx,
        Hy=solver.Hy,
        Hz=solver.Hz,
        dt=solver.dt,
        c=solver.c,
        eps0=solver.eps0,
        source_frequency=solver.source_frequency,
        source_omega=solver.source_omega,
        eps_Ex=eps_ex,
        eps_Ey=eps_ey,
        eps_Ez=eps_ez,
        cex_curl=_dynamic_electric_curl(solver.cex_curl, solver.eps_Ex, eps_ex),
        cey_curl=_dynamic_electric_curl(solver.cey_curl, solver.eps_Ey, eps_ey),
        cez_curl=_dynamic_electric_curl(solver.cez_curl, solver.eps_Ez, eps_ez),
        chx_curl=solver.chx_curl,
        chy_curl=solver.chy_curl,
        chz_curl=solver.chz_curl,
        boundary_kind=solver.boundary_kind,
        tfsf_enabled=bool(getattr(solver, "tfsf_enabled", False)),
        _compiled_sources=compiled_sources,
        _compiled_source=primary,
        _compiled_material_model=getattr(solver, "_compiled_material_model", None),
        _mode_source_rebuild_from_fields=True,
        _source_time=primary["source_time"],
        _compute_linear_launch_shape=solver._compute_linear_launch_shape,
        _iter_source_images=solver._iter_source_images,
        _component_plane_spec=solver._component_plane_spec,
        _component_positions=solver._component_positions,
        _plane_coordinate=solver._plane_coordinate,
    )


# Source kinds whose eps-leaf-dependent patches are re-derived on the replay
# solver so the mode-source objective and the torch-VJP reverse can flow
# gradients into eps. Other kinds keep their forward-built terms.
_SOURCE_REPLAY_REBUILD_KINDS = frozenset(
    {"point_dipole", "plane_wave", "gaussian_beam", "mode_source"}
)


def _compiled_sources_for_replay(solver):
    compiled_sources = tuple(getattr(solver, "_compiled_sources", ()) or ())
    if compiled_sources:
        return compiled_sources
    compiled_source = getattr(solver, "_compiled_source", None)
    return (compiled_source,) if compiled_source is not None else ()


def _scene_has_mode_source(solver) -> bool:
    return any(
        source.get("kind") == "mode_source"
        for source in _compiled_sources_for_replay(solver)
    )


def _resolved_source_term_lists(solver, eps_ex, eps_ey, eps_ez):
    default_source_terms = getattr(solver, "_source_terms", ())
    default_electric_source_terms = getattr(solver, "_electric_source_terms", ())
    default_magnetic_source_terms = getattr(solver, "_magnetic_source_terms", ())
    compiled_sources = _compiled_sources_for_replay(solver)
    if not compiled_sources:
        return default_source_terms, default_electric_source_terms, default_magnetic_source_terms
    # A kind outside the rebuild set (e.g. astigmatic beams) keeps the exact
    # forward-built terms, which already reproduce the injection for the explicit
    # reverse path; if any source in the scene is such a kind, fall back to the
    # forward terms for the whole scene so the per-source term indexing stays
    # consistent with the forward solve.
    if not all(source.get("kind") in _SOURCE_REPLAY_REBUILD_KINDS for source in compiled_sources):
        return default_source_terms, default_electric_source_terms, default_magnetic_source_terms

    has_mode_source = any(source.get("kind") == "mode_source" for source in compiled_sources)
    temp_solver = _build_source_replay_solver(solver, compiled_sources, eps_ex, eps_ey, eps_ez)
    if has_mode_source:
        # Build the mode-source patches under grad tracking so the eigensolve
        # profile keeps its eps-graph; the FDTD backward runs under no_grad, and
        # _mode_source_profile_pullback needs that graph for its once-per-pass VJP.
        with torch.enable_grad():
            initialize_source_terms(temp_solver)
    else:
        initialize_source_terms(temp_solver)
    return (
        tuple(temp_solver._source_terms),
        tuple(temp_solver._electric_source_terms),
        tuple(temp_solver._magnetic_source_terms),
    )


def _has_resolved_source_terms(resolved_source_terms) -> bool:
    if resolved_source_terms is None:
        return False
    source_terms, electric_source_terms, magnetic_source_terms = resolved_source_terms
    return bool(source_terms or electric_source_terms or magnetic_source_terms)


def _can_use_explicit_source_term_reverse_step(solver, resolved_source_terms) -> bool:
    if not _has_resolved_source_terms(resolved_source_terms):
        return True
    if getattr(solver, "tfsf_enabled", False):
        return False
    if bool(getattr(solver, "has_pec_faces", False)):
        return False
    if has_complex_fields(solver):
        source_terms, electric_source_terms, magnetic_source_terms = resolved_source_terms
        if getattr(solver, "boundary_kind", None) != "bloch":
            return False
        if electric_source_terms or magnetic_source_terms:
            return False
        return all(
            term["cw_cos_patch"] is None
            and term["delay_patch"] is None
            and term.get("activation_delay_patch") is None
            for term in source_terms
        )
    return True


def _apply_resolved_magnetic_source_terms(
    magnetic_fields,
    *,
    solver,
    time_value,
    resolved_source_terms,
):
    if resolved_source_terms is None:
        return magnetic_fields
    _source_terms, _electric_source_terms, magnetic_source_terms = resolved_source_terms
    if not magnetic_source_terms:
        return magnetic_fields
    return _apply_source_term_list(
        magnetic_fields,
        terms=magnetic_source_terms,
        source_time=solver._source_time,
        omega=solver.source_omega,
        time_value=time_value,
        solver=None,
    )


def _source_term_component_patches(term):
    """Ordered eps-dependent patches whose signal-weighted sum is the injected payload.

    Each patch carries the term's permittivity dependence (the ``1/eps`` injection
    coefficient and, for mode sources, the eigensolve profile). The time-signal
    weights come from :func:`_source_term_signal_factors` in the same order.
    """
    literal_patch = term.get("literal_patch")
    if literal_patch is not None:
        return (literal_patch,)
    if term["cw_cos_patch"] is not None:
        return (term["cw_cos_patch"], term["cw_sin_patch"])
    return (term["patch"],)


def _source_term_signal_factors(term, *, source_time, omega, time_value):
    """Time-signal weights (scalar or per-cell tensor) for the term's patches.

    The weights are permittivity-independent, so the injected payload is
    ``sum_k patch_k * factor_k`` and its eps-derivative is carried entirely by the
    patches. Order matches :func:`_source_term_component_patches`.
    """
    # Each compiled source carries its own source-time waveform / CW frequency;
    # resolve per-term so a second source at a different frequency is not driven
    # by the primary source's spectrum.
    source_time, omega = _resolve_term_source(term, source_time, omega)
    if term.get("literal_patch") is not None:
        return (1.0,)
    if term["cw_cos_patch"] is not None:
        signal_cos, signal_sin = _resolve_cw_signal(source_time, omega, time_value)
        return (float(signal_cos), float(signal_sin))
    if term["delay_patch"] is not None:
        sample_time = float(time_value) - term["delay_patch"]
        factor = _evaluate_source_time_tensor(source_time, sample_time)
        activation_delay_patch = term.get("activation_delay_patch")
        if activation_delay_patch is not None:
            factor = torch.where(
                activation_delay_patch > float(time_value),
                torch.zeros_like(factor),
                factor,
            )
        return (factor,)
    scalar_signal = evaluate_source_time(source_time, float(time_value))
    return (float(scalar_signal) * float(term["phase_real"]),)


def _source_term_signal_patch(term, *, source_time, omega, time_value):
    patches = _source_term_component_patches(term)
    factors = _source_term_signal_factors(term, source_time=source_time, omega=omega, time_value=time_value)
    payload = None
    for patch, factor in zip(patches, factors):
        contribution = patch * factor
        payload = contribution if payload is None else payload + contribution
    return payload


def _bloch_source_term_signal_patch(term, *, source_time, omega, time_value) -> torch.Tensor:
    source_time, omega = _resolve_term_source(term, source_time, omega)
    if term["cw_cos_patch"] is not None or term["delay_patch"] is not None:
        raise NotImplementedError("Bloch source-term reverse currently supports uniform point-source patches only.")
    scalar_signal = float(evaluate_source_time(source_time, float(time_value)))
    patch = term["patch"]
    patch_real = patch.to(dtype=torch.float32 if not torch.is_floating_point(patch) else patch.dtype)
    return torch.complex(
        patch_real * (scalar_signal * float(term["phase_real"])),
        patch_real * (scalar_signal * float(term["phase_imag"])),
    )


def _accumulate_mode_source_cotangents(
    solver,
    *,
    electric_terms,
    magnetic_terms,
    electric_adjoint,
    magnetic_adjoint,
    source_time,
    omega,
    time_value,
) -> None:
    """Accumulate the time-weighted adjoint cotangent for profile source terms.

    Mode-source patches carry an eigensolve eps-graph: the aperture mode profile
    (and its normalization) depends on the source-plane permittivity. The per-step
    source objective ``sum_t sum_k <patch_k, signal_k(t) * adjoint(t)>`` is linear
    in the detached adjoint, and grad_eps is summed over the backward sweep anyway,
    so accumulate ``sum_t signal_k(t) * adjoint(t)`` per patch here with a native
    scatter-add and defer the single eigensolve VJP to
    :func:`_mode_source_profile_pullback`. This keeps ``torch.autograd.grad`` off
    the per-step hot path while reproducing the exact profile eps-dependence.
    """
    accum = getattr(solver, "_mode_source_cotangent_accum", None)
    if accum is None:
        accum = {}
        solver._mode_source_cotangent_accum = accum
    groups = (
        ("E", electric_terms, electric_adjoint),
        ("H", magnetic_terms, magnetic_adjoint),
    )
    for group_name, terms, adjoint_by_field in groups:
        for term_pos, term in enumerate(terms):
            field_name = term["field_name"]
            if field_name not in adjoint_by_field:
                continue
            offsets = term["offsets"]
            patches = _source_term_component_patches(term)
            factors = _source_term_signal_factors(
                term, source_time=source_time, omega=omega, time_value=time_value
            )
            for comp_idx, (patch, factor) in enumerate(zip(patches, factors)):
                region = _slice_from_offsets_shape(offsets, patch.shape)
                adjoint_patch = adjoint_by_field[field_name][region].detach().to(
                    device=patch.device, dtype=patch.dtype
                )
                contribution = (adjoint_patch * factor).detach()
                key = (group_name, term_pos, comp_idx)
                existing = accum.get(key)
                if existing is None:
                    accum[key] = contribution.clone()
                else:
                    existing.add_(contribution)


def _mode_source_profile_pullback(solver, resolved_source_terms, eps_ex, eps_ey, eps_ez):
    """Apply the accumulated profile cotangent through the eigensolve eps-graph once.

    Runs a single ``torch.autograd.grad`` over the retained source-term graph at the
    end of the backward pass (once-per-run material-pullback stage), returning the
    per-axis eps gradients, or ``None`` when there is nothing to pull back.
    """
    accum = getattr(solver, "_mode_source_cotangent_accum", None)
    if not accum:
        return None
    source_terms, electric_source_terms, magnetic_source_terms = resolved_source_terms
    grouped = (
        ("E", tuple(electric_source_terms) + tuple(source_terms)),
        ("H", tuple(magnetic_source_terms)),
    )
    with torch.enable_grad():
        objective = None
        for group_name, terms in grouped:
            for term_pos, term in enumerate(terms):
                patches = _source_term_component_patches(term)
                for comp_idx, patch in enumerate(patches):
                    cotangent = accum.get((group_name, term_pos, comp_idx))
                    if cotangent is None:
                        continue
                    contribution = torch.sum(
                        patch * cotangent.to(device=patch.device, dtype=patch.dtype)
                    )
                    objective = contribution if objective is None else objective + contribution
        if objective is None:
            return None
        return torch.autograd.grad(objective, (eps_ex, eps_ey, eps_ez), allow_unused=True)


def _accumulate_source_term_gradients(
    step_result: _ReverseStepResult,
    *,
    solver,
    adjoint_state,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
) -> _ReverseStepResult:
    if not _has_resolved_source_terms(resolved_source_terms):
        return step_result
    if not _can_use_explicit_source_term_reverse_step(solver, resolved_source_terms):
        return step_result

    source_terms, electric_source_terms, magnetic_source_terms = resolved_source_terms
    source_adjoint_state = step_result.source_adjoint_state or adjoint_state
    if _scene_has_mode_source(solver):
        # Mode-source patches (and any other rebuilt patches sharing the scene)
        # carry an eigensolve eps-graph. Instead of running torch.autograd.grad on
        # the per-step hot path, accumulate the time-weighted adjoint cotangent with
        # a native scatter-add here; the single eigensolve VJP is applied once per
        # backward pass by _mode_source_profile_pullback (bridge material-pullback
        # stage). The per-step step_result eps-grad is returned unchanged.
        _accumulate_mode_source_cotangents(
            solver,
            electric_terms=tuple(electric_source_terms) + tuple(source_terms),
            magnetic_terms=tuple(magnetic_source_terms),
            electric_adjoint=source_adjoint_state,
            magnetic_adjoint=step_result.magnetic_output_adjoint or {},
            source_time=solver._source_time,
            omega=solver.source_omega,
            time_value=time_value,
        )
        return step_result

    grad_eps_by_field = {
        "Ex": step_result.grad_eps_ex.detach().clone(),
        "Ey": step_result.grad_eps_ey.detach().clone(),
        "Ez": step_result.grad_eps_ez.detach().clone(),
    }
    eps_by_field = {
        "Ex": eps_ex,
        "Ey": eps_ey,
        "Ez": eps_ez,
    }

    for term in tuple(electric_source_terms) + tuple(source_terms):
        field_name = term["field_name"]
        if field_name not in grad_eps_by_field:
            continue
        if has_complex_fields(solver):
            payload = _bloch_source_term_signal_patch(
                term,
                source_time=solver._source_time,
                omega=solver.source_omega,
                time_value=time_value,
            )
            offsets = term["offsets"]
            region = _slice_from_offsets_shape(offsets, payload.shape)
            adjoint_patch = _bloch_source_term_adjoint_patch(
                source_adjoint_state,
                solver=solver,
                field_name=field_name,
                offsets=offsets,
                patch_shape=payload.shape,
            )
            eps_patch = eps_by_field[field_name][region]
            grad_eps_by_field[field_name][region] = (
                grad_eps_by_field[field_name][region] - _complex_inner_real(adjoint_patch, payload) / eps_patch
            )
            continue
        payload = _source_term_signal_patch(
            term,
            source_time=solver._source_time,
            omega=solver.source_omega,
            time_value=time_value,
        )
        offsets = term["offsets"]
        region = _slice_from_offsets_shape(offsets, payload.shape)
        adjoint_patch = source_adjoint_state[field_name][region]
        eps_patch = eps_by_field[field_name][region]
        grad_eps_by_field[field_name][region] = (
            grad_eps_by_field[field_name][region] - (adjoint_patch * payload) / eps_patch
        )

    return _ReverseStepResult(
        pre_step_adjoint=step_result.pre_step_adjoint,
        grad_eps_ex=grad_eps_by_field["Ex"],
        grad_eps_ey=grad_eps_by_field["Ey"],
        grad_eps_ez=grad_eps_by_field["Ez"],
        backend=step_result.backend,
        source_adjoint_state=step_result.source_adjoint_state,
        magnetic_output_adjoint=step_result.magnetic_output_adjoint,
        # Preserve the nonlinear gradient channels across the source-term eps
        # accumulation (the source injection carries no chi2/chi3/tpa dependence,
        # so these pass through untouched); the analytic Kerr / general-nonlinear
        # reverses populate them and the bridge requires them per step.
        grad_chi3_ex=step_result.grad_chi3_ex,
        grad_chi3_ey=step_result.grad_chi3_ey,
        grad_chi3_ez=step_result.grad_chi3_ez,
        grad_chi2_ex=step_result.grad_chi2_ex,
        grad_chi2_ey=step_result.grad_chi2_ey,
        grad_chi2_ez=step_result.grad_chi2_ez,
        grad_tpa_ex=step_result.grad_tpa_ex,
        grad_tpa_ey=step_result.grad_tpa_ey,
        grad_tpa_ez=step_result.grad_tpa_ez,
    )


@dataclass(frozen=True)
class _ForwardPack:
    field_names: tuple[str, ...]
    monitor_templates: dict[str, dict[str, Any]]
    output_tensors: tuple[torch.Tensor, ...]


def _add_patch(field: torch.Tensor, offsets, patch: torch.Tensor) -> torch.Tensor:
    region = _slice_from_offsets_shape(offsets, patch.shape)
    updated = field.clone()
    updated[region] = updated[region] + patch.to(device=field.device, dtype=field.dtype)
    return updated


def _complex_phase_positive(value: torch.Tensor, phase_cos: float, phase_sin: float) -> torch.Tensor:
    return torch.complex(
        float(phase_cos) * value.real - float(phase_sin) * value.imag,
        float(phase_sin) * value.real + float(phase_cos) * value.imag,
    )


def _complex_phase_negative(value: torch.Tensor, phase_cos: float, phase_sin: float) -> torch.Tensor:
    return torch.complex(
        float(phase_cos) * value.real + float(phase_sin) * value.imag,
        float(phase_cos) * value.imag - float(phase_sin) * value.real,
    )


def _complex_inner_real(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return lhs.real * rhs.real + lhs.imag * rhs.imag


def _axis_uses_complex_source_wrap(solver, axis: int) -> bool:
    scene = getattr(solver, "scene", None)
    boundary = None if scene is None else getattr(scene, "boundary", None)
    if boundary is None:
        return getattr(solver, "boundary_kind", None) == "bloch"
    return boundary.axis_kind("xyz"[int(axis)]) in {"bloch", "periodic"}


def _bloch_phase_specs(solver, field_name: str):
    component = field_name[-1]
    if component == "x":
        return (
            (1, float(solver.boundary_phase_cos[1]), float(solver.boundary_phase_sin[1])),
            (2, float(solver.boundary_phase_cos[2]), float(solver.boundary_phase_sin[2])),
        )
    if component == "y":
        return (
            (0, float(solver.boundary_phase_cos[0]), float(solver.boundary_phase_sin[0])),
            (2, float(solver.boundary_phase_cos[2]), float(solver.boundary_phase_sin[2])),
        )
    if component == "z":
        return (
            (0, float(solver.boundary_phase_cos[0]), float(solver.boundary_phase_sin[0])),
            (1, float(solver.boundary_phase_cos[1]), float(solver.boundary_phase_sin[1])),
        )
    raise ValueError(f"Unsupported Bloch source field {field_name!r}.")


def _bloch_source_boundary_terms(solver, field_name: str, offsets, patch_shape, field_shape):
    boundary_terms = []
    for axis, phase_cos, phase_sin in _bloch_phase_specs(solver, field_name):
        if not _axis_uses_complex_source_wrap(solver, axis):
            continue
        axis_size = int(field_shape[axis])
        start = int(offsets[axis])
        end = start + int(patch_shape[axis])
        if start == 0:
            boundary_terms.append((axis, "low", axis_size - 1, phase_cos, phase_sin))
        elif end == axis_size:
            boundary_terms.append((axis, "high", 0, phase_cos, phase_sin))
    return boundary_terms


def _bloch_source_term_adjoint_patch(adjoint_state, *, solver, field_name: str, offsets, patch_shape):
    adjoint_field = torch.complex(adjoint_state[field_name], adjoint_state[f"{field_name}_imag"])
    region = _slice_from_offsets_shape(offsets, patch_shape)
    adjoint_patch = adjoint_field[region].clone()
    boundary_terms = _bloch_source_boundary_terms(
        solver,
        field_name,
        offsets,
        patch_shape,
        adjoint_field.shape,
    )

    for axis, side, pair_index, phase_cos, phase_sin in boundary_terms:
        paired_offsets = list(offsets)
        paired_offsets[axis] = pair_index
        paired_shape = list(patch_shape)
        paired_shape[axis] = 1
        paired_patch = adjoint_field[_slice_from_offsets_shape(tuple(paired_offsets), tuple(paired_shape))]
        boundary_adjoint = (
            _complex_phase_negative(paired_patch, phase_cos, phase_sin)
            if side == "low"
            else _complex_phase_positive(paired_patch, phase_cos, phase_sin)
        )
        boundary_index = 0 if side == "low" else int(adjoint_patch.shape[axis] - 1)
        adjoint_patch.narrow(axis, boundary_index, 1).add_(boundary_adjoint)

    if len(boundary_terms) == 2:
        (axis_a, side_a, pair_a, phase_cos_a, phase_sin_a), (axis_b, side_b, pair_b, phase_cos_b, phase_sin_b) = boundary_terms
        paired_offsets = list(offsets)
        paired_offsets[axis_a] = pair_a
        paired_offsets[axis_b] = pair_b
        paired_shape = list(patch_shape)
        paired_shape[axis_a] = 1
        paired_shape[axis_b] = 1
        corner_patch = adjoint_field[_slice_from_offsets_shape(tuple(paired_offsets), tuple(paired_shape))]
        corner_patch = (
            _complex_phase_negative(corner_patch, phase_cos_b, phase_sin_b)
            if side_b == "low"
            else _complex_phase_positive(corner_patch, phase_cos_b, phase_sin_b)
        )
        corner_patch = (
            _complex_phase_negative(corner_patch, phase_cos_a, phase_sin_a)
            if side_a == "low"
            else _complex_phase_positive(corner_patch, phase_cos_a, phase_sin_a)
        )
        corner_index_a = 0 if side_a == "low" else int(adjoint_patch.shape[axis_a] - 1)
        corner_index_b = 0 if side_b == "low" else int(adjoint_patch.shape[axis_b] - 1)
        adjoint_patch.narrow(axis_a, corner_index_a, 1).narrow(axis_b, corner_index_b, 1).add_(corner_patch)

    return adjoint_patch


def _add_complex_bloch_source_patch(field_mapping, *, solver, field_name: str, offsets, delta: torch.Tensor):
    real_name = field_name
    imag_name = f"{field_name}_imag"
    if imag_name not in field_mapping:
        raise KeyError(f"Bloch source injection requires {imag_name!r} in the replay state.")

    field = torch.complex(field_mapping[real_name], field_mapping[imag_name])
    delta = delta.to(device=field.device, dtype=field.dtype)
    field = _add_patch(field, offsets, delta)

    boundary_terms = _bloch_source_boundary_terms(
        solver,
        field_name,
        offsets,
        delta.shape,
        field.shape,
    )
    for axis, side, pair_index, phase_cos, phase_sin in boundary_terms:
        boundary_patch = _boundary_slice(delta, axis, side)
        phased_patch = (
            _complex_phase_positive(boundary_patch, phase_cos, phase_sin)
            if side == "low"
            else _complex_phase_negative(boundary_patch, phase_cos, phase_sin)
        )
        paired_offsets = list(offsets)
        paired_offsets[axis] = pair_index
        field = _add_patch(field, tuple(paired_offsets), phased_patch)

    if len(boundary_terms) == 2:
        (axis_a, side_a, pair_a, phase_cos_a, phase_sin_a), (axis_b, side_b, pair_b, phase_cos_b, phase_sin_b) = boundary_terms
        corner_patch = _boundary_slice(_boundary_slice(delta, axis_a, side_a), axis_b, side_b)
        corner_patch = (
            _complex_phase_positive(corner_patch, phase_cos_a, phase_sin_a)
            if side_a == "low"
            else _complex_phase_negative(corner_patch, phase_cos_a, phase_sin_a)
        )
        corner_patch = (
            _complex_phase_positive(corner_patch, phase_cos_b, phase_sin_b)
            if side_b == "low"
            else _complex_phase_negative(corner_patch, phase_cos_b, phase_sin_b)
        )
        paired_offsets = list(offsets)
        paired_offsets[axis_a] = pair_a
        paired_offsets[axis_b] = pair_b
        field = _add_patch(field, tuple(paired_offsets), corner_patch)

    updated = dict(field_mapping)
    updated[real_name] = field.real
    updated[imag_name] = field.imag
    return updated


def _boundary_slice(patch: torch.Tensor, axis: int, side: str) -> torch.Tensor:
    index = 0 if side == "low" else int(patch.shape[axis] - 1)
    return patch.narrow(axis, index, 1)


def _apply_bloch_uniform_source_term(
    field_mapping,
    *,
    solver,
    field_name: str,
    offsets,
    patch: torch.Tensor,
    signal_real: float,
    signal_imag: float,
):
    real_name = field_name
    imag_name = f"{field_name}_imag"
    if imag_name not in field_mapping:
        raise KeyError(f"Bloch source injection requires {imag_name!r} in the replay state.")

    patch_real = patch.to(device=field_mapping[real_name].device, dtype=field_mapping[real_name].dtype)
    delta = torch.complex(patch_real * float(signal_real), patch_real * float(signal_imag))
    return _add_complex_bloch_source_patch(
        field_mapping,
        solver=solver,
        field_name=field_name,
        offsets=offsets,
        delta=delta,
    )


def _apply_bloch_cw_source_term(
    field_mapping,
    *,
    solver,
    field_name: str,
    offsets,
    cw_cos_patch: torch.Tensor,
    cw_sin_patch: torch.Tensor,
    signal_cos: float,
    signal_sin: float,
):
    patch_cos = cw_cos_patch.to(device=field_mapping[field_name].device, dtype=field_mapping[field_name].dtype)
    patch_sin = cw_sin_patch.to(device=field_mapping[field_name].device, dtype=field_mapping[field_name].dtype)
    delta = torch.complex(
        float(signal_cos) * patch_cos + float(signal_sin) * patch_sin,
        float(signal_sin) * patch_cos - float(signal_cos) * patch_sin,
    )
    return _add_complex_bloch_source_patch(
        field_mapping,
        solver=solver,
        field_name=field_name,
        offsets=offsets,
        delta=delta,
    )


def _sample_auxiliary_field(field: torch.Tensor, *, origin: float, ds: float, positions: torch.Tensor) -> torch.Tensor:
    coord = torch.clamp((positions.to(device=field.device, dtype=field.dtype) - float(origin)) / float(ds), min=0.0)
    coord = torch.clamp(coord, max=max(field.numel() - 1, 0))
    lower = torch.floor(coord).to(torch.int64)
    upper = torch.clamp(lower + 1, max=max(field.numel() - 1, 0))
    frac = coord - lower.to(dtype=field.dtype)
    return field[lower] + (field[upper] - field[lower]) * frac


def _apply_tfsf_reference_terms(field_mapping, terms, *, auxiliary_field: torch.Tensor):
    updated = dict(field_mapping)
    for term in terms:
        if term["scalar_sample_index"] is not None:
            incident = auxiliary_field[int(term["scalar_sample_index"])]
            patch = term["coeff_patch"] * incident
        else:
            incident_line = auxiliary_field[term["sample_indices"].to(dtype=torch.long)]
            patch = term["coeff_patch"] * incident_line.view(term["sample_view"])
        patch = patch * float(term["component_scale"])
        updated[term["field_name"]] = _add_patch(updated[term["field_name"]], term["offsets"], patch)
    return updated


def _apply_tfsf_aux_terms(field_mapping, terms, *, auxiliary_field: torch.Tensor, origin: float, ds: float):
    updated = dict(field_mapping)
    for term in terms:
        incident = _sample_auxiliary_field(auxiliary_field, origin=origin, ds=ds, positions=term["sample_positions"])
        patch = term["coeff_patch"] * incident
        patch = patch * float(term["component_scale"])
        updated[term["field_name"]] = _add_patch(updated[term["field_name"]], term["offsets"], patch)
    return updated


def _extract_tfsf_auxiliary_state(state):
    if "tfsf_aux_electric" not in state or "tfsf_aux_magnetic" not in state:
        return None
    return {
        "electric": state["tfsf_aux_electric"],
        "magnetic": state["tfsf_aux_magnetic"],
    }


def _apply_tfsf_terms(field_mapping, *, solver, auxiliary_state, term_key: str, sample_kind: str, time_value):
    if not getattr(solver, "tfsf_enabled", False):
        return field_mapping

    tfsf_state = getattr(solver, "_tfsf_state", None)
    if tfsf_state is None:
        return field_mapping

    provider = tfsf_state.get("provider")
    terms = tfsf_state.get(term_key, ())
    if provider in _TFSF_REFERENCE_PROVIDERS:
        if auxiliary_state is None:
            return field_mapping
        auxiliary_field = auxiliary_state["electric" if sample_kind == "electric" else "magnetic"]
        return _apply_tfsf_reference_terms(field_mapping, terms, auxiliary_field=auxiliary_field)

    if provider == "plane_wave_aux":
        if auxiliary_state is None:
            return field_mapping
        auxiliary_grid = tfsf_state["auxiliary_grid"]
        if sample_kind == "electric":
            auxiliary_field = auxiliary_state["electric"]
            origin = float(auxiliary_grid.s_min)
        else:
            auxiliary_field = auxiliary_state["magnetic"]
            origin = float(auxiliary_grid.s_min + 0.5 * auxiliary_grid.ds)
        return _apply_tfsf_aux_terms(
            field_mapping,
            terms,
            auxiliary_field=auxiliary_field,
            origin=origin,
            ds=float(auxiliary_grid.ds),
        )

    return _apply_source_term_list(
        field_mapping,
        terms=terms,
        source_time=solver._source_time,
        omega=solver.source_omega,
        time_value=time_value,
        solver=solver if has_complex_fields(solver) else None,
    )


def _advance_tfsf_auxiliary_magnetic_state(solver, auxiliary_state):
    if auxiliary_state is None:
        return None
    auxiliary_grid = getattr(solver, "_tfsf_state", {}).get("auxiliary_grid")
    if auxiliary_grid is None:
        return auxiliary_state
    electric = auxiliary_state["electric"]
    magnetic = auxiliary_state["magnetic"]
    if magnetic.numel() == 0:
        return {"electric": electric, "magnetic": magnetic}
    curl_e = electric[1:] - electric[:-1]
    next_magnetic = auxiliary_grid.magnetic_decay * magnetic - auxiliary_grid.magnetic_curl * curl_e
    return {"electric": electric, "magnetic": next_magnetic}


def _advance_tfsf_auxiliary_electric_state(solver, auxiliary_state, *, time_value):
    if auxiliary_state is None:
        return None
    auxiliary_grid = getattr(solver, "_tfsf_state", {}).get("auxiliary_grid")
    if auxiliary_grid is None:
        return auxiliary_state
    electric = auxiliary_state["electric"]
    magnetic = auxiliary_state["magnetic"]
    next_electric = electric.clone()
    if electric.numel() > 2:
        curl_h = magnetic[1:] - magnetic[:-1]
        next_electric[1:-1] = (
            auxiliary_grid.electric_decay[1:-1] * electric[1:-1]
            - auxiliary_grid.electric_curl[1:-1] * curl_h
        )
    next_electric[int(auxiliary_grid.source_index)] = evaluate_source_time(auxiliary_grid.source_time, float(time_value))
    next_electric[-1] = 0.0
    return {"electric": next_electric, "magnetic": magnetic}


def _literal_source_term(*, field_name: str, offsets, patch: torch.Tensor) -> dict[str, Any]:
    return {
        "field_name": field_name,
        "offsets": offsets,
        "patch": None,
        "grid": None,
        "phase_real": 1.0,
        "phase_imag": 0.0,
        "delay_patch": None,
        "activation_delay_patch": None,
        "cw_cos_patch": None,
        "cw_sin_patch": None,
        "literal_patch": patch.detach().contiguous(),
    }


def _materialize_tfsf_term_patch(
    solver,
    term,
    *,
    auxiliary_state,
    sample_kind: str,
    time_value,
) -> torch.Tensor:
    if "coeff_patch" not in term:
        return _source_term_signal_patch(
            term,
            source_time=solver._source_time,
            omega=solver.source_omega,
            time_value=time_value,
        )

    if auxiliary_state is None:
        raise RuntimeError("TFSF auxiliary terms require frozen auxiliary checkpoint state.")

    auxiliary_field = auxiliary_state[sample_kind]
    if "sample_positions" in term:
        auxiliary_grid = getattr(solver, "_tfsf_state", {}).get("auxiliary_grid")
        if auxiliary_grid is None:
            raise RuntimeError("Auxiliary-position TFSF terms require an auxiliary grid.")
        if sample_kind == "electric":
            origin = float(auxiliary_grid.s_min)
        else:
            origin = float(auxiliary_grid.s_min + 0.5 * auxiliary_grid.ds)
        incident = _sample_auxiliary_field(
            auxiliary_field,
            origin=origin,
            ds=float(auxiliary_grid.ds),
            positions=term["sample_positions"],
        )
        patch = term["coeff_patch"] * incident
    elif term["scalar_sample_index"] is not None:
        patch = term["coeff_patch"] * auxiliary_field[int(term["scalar_sample_index"])]
    else:
        incident_line = auxiliary_field[term["sample_indices"].to(dtype=torch.long)]
        patch = term["coeff_patch"] * incident_line.view(term["sample_view"])
    return patch * float(term["component_scale"])


def _materialize_tfsf_source_terms(
    solver,
    *,
    terms,
    auxiliary_state,
    sample_kind: str,
    time_value,
) -> list[dict[str, Any]]:
    materialized_terms = []
    for term in terms:
        patch = _materialize_tfsf_term_patch(
            solver,
            term,
            auxiliary_state=auxiliary_state,
            sample_kind=sample_kind,
            time_value=time_value,
        )
        materialized_terms.append(
            _literal_source_term(
                field_name=term["field_name"],
                offsets=term["offsets"],
                patch=patch,
            )
        )
    return materialized_terms


def _merge_resolved_source_terms(
    resolved_source_terms,
    *,
    source_terms=(),
    electric_source_terms=(),
    magnetic_source_terms=(),
):
    if resolved_source_terms is None:
        base_source_terms = ()
        base_electric_source_terms = ()
        base_magnetic_source_terms = ()
    else:
        base_source_terms, base_electric_source_terms, base_magnetic_source_terms = resolved_source_terms
    return (
        tuple(base_source_terms) + tuple(source_terms),
        tuple(base_electric_source_terms) + tuple(electric_source_terms),
        tuple(base_magnetic_source_terms) + tuple(magnetic_source_terms),
    )


def _tfsf_magnetic_source_terms(solver, forward_state, *, time_value, resolved_source_terms=None):
    tfsf_state = getattr(solver, "_tfsf_state", None)
    if tfsf_state is None:
        return resolved_source_terms
    auxiliary_state = _extract_tfsf_auxiliary_state(forward_state)
    magnetic_terms = _materialize_tfsf_source_terms(
        solver,
        terms=tfsf_state.get("magnetic_terms", ()),
        auxiliary_state=auxiliary_state,
        sample_kind="electric",
        time_value=time_value,
    )
    return _merge_resolved_source_terms(
        resolved_source_terms,
        magnetic_source_terms=magnetic_terms,
    )


def _accumulate_tfsf_sample_adjoint_torch(adj_aux, terms, adjoint_fields, *, origin, ds):
    """Transpose of the per-step TFSF incident injection into ``adj_aux``.

    For every injection term this reads the adjoint of the field the forward step
    wrote the incident patch into (``adjoint_fields[field_name]`` at the term
    offsets), contracts it with the term coefficient patch and component scale,
    and scatters the result back onto the sampled 1D auxiliary grid. This mirrors
    the native ``accumulateTfsf*SampleAdjoint3D`` kernels: scalar terms fold onto a
    single index, line terms reduce the two non-sample axes onto their index line,
    and interpolated terms distribute linearly onto the two straddling indices.
    """
    for term in terms:
        field = adjoint_fields[term["field_name"]]
        offset_i, offset_j, offset_k = (int(offset) for offset in term["offsets"])
        coeff_patch = term["coeff_patch"]
        shape_i, shape_j, shape_k = (int(length) for length in coeff_patch.shape)
        adj_field_patch = field[
            offset_i : offset_i + shape_i,
            offset_j : offset_j + shape_j,
            offset_k : offset_k + shape_k,
        ]
        component_scale = float(term["component_scale"])
        weighted = component_scale * adj_field_patch * coeff_patch
        if "sample_positions" in term:
            last_index = max(adj_aux.numel() - 1, 0)
            inv_ds = 1.0 / ds if ds > 0.0 else 0.0
            coord = torch.clamp(
                (term["sample_positions"] - origin) * inv_ds, min=0.0, max=float(last_index)
            )
            lower = torch.floor(coord).to(torch.int64)
            upper = torch.clamp(lower + 1, max=last_index)
            frac = coord - lower.to(coord.dtype)
            value = weighted.reshape(-1)
            lower_flat = lower.reshape(-1)
            upper_flat = upper.reshape(-1)
            frac_flat = frac.reshape(-1)
            adj_aux.index_add_(0, lower_flat, value * (1.0 - frac_flat))
            distinct = upper_flat != lower_flat
            if bool(distinct.any()):
                adj_aux.index_add_(0, upper_flat[distinct], (value * frac_flat)[distinct])
            continue
        if term["scalar_sample_index"] is not None:
            adj_aux[int(term["scalar_sample_index"])] += torch.sum(weighted)
            continue
        axis = int(reference_sample_axis_code(term))
        reduce_axes = tuple(other for other in (0, 1, 2) if other != axis)
        per_index = weighted.sum(dim=reduce_axes)
        sample_indices = term["sample_indices"].to(device=adj_aux.device, dtype=torch.int64).reshape(-1)
        adj_aux.index_add_(0, sample_indices, per_index)


def _reverse_tfsf_auxiliary_electric_state_adjoint(
    adj_electric_prev,
    adj_magnetic_after,
    adj_electric_post,
    electric_decay,
    electric_curl,
    source_index,
):
    """Analytic transpose of ``_advance_tfsf_auxiliary_electric_state``.

    Accumulates the pre-step electric adjoint (decay pullback on the interior, an
    identity passthrough at index 0) and the advanced-magnetic adjoint (the curl-H
    difference) exactly as ``reverse_tfsf_auxiliary_electric_kernel`` does; the
    source-driven and clamped-tail entries carry no state dependence.
    """
    electric_total = adj_electric_prev.numel()
    magnetic_total = adj_magnetic_after.numel()
    index = torch.arange(electric_total, device=adj_electric_prev.device)
    overwritten = (index == int(source_index)) | (index == electric_total - 1)
    interior = (index >= 1) & (index + 1 < electric_total) & (~overwritten)
    passthrough = (index == 0) & (~overwritten)
    electric_coeff = torch.where(interior, electric_decay, torch.zeros_like(electric_decay))
    electric_coeff = torch.where(passthrough, torch.ones_like(electric_decay), electric_coeff)
    adj_electric_prev.add_(electric_coeff * adj_electric_post)

    if magnetic_total == 0:
        return
    lower = torch.arange(magnetic_total, device=adj_electric_prev.device)
    upper = lower + 1
    lower_valid = (lower > 0) & (lower + 1 < electric_total) & (lower != int(source_index))
    upper_valid = (upper > 0) & (upper + 1 < electric_total) & (upper != int(source_index))
    zeros = torch.zeros(magnetic_total, device=adj_electric_prev.device, dtype=adj_electric_prev.dtype)
    lower_term = torch.where(
        lower_valid, electric_curl[:magnetic_total] * adj_electric_post[:magnetic_total], zeros
    )
    upper_term = torch.where(
        upper_valid, electric_curl[1 : magnetic_total + 1] * adj_electric_post[1 : magnetic_total + 1], zeros
    )
    adj_magnetic_after.add_(upper_term - lower_term)


def _reverse_tfsf_auxiliary_magnetic_state_adjoint(
    adj_electric_prev,
    adj_magnetic_prev,
    adj_magnetic_after,
    magnetic_decay,
    magnetic_curl,
):
    """Analytic transpose of ``_advance_tfsf_auxiliary_magnetic_state``.

    Assigns the pre-step magnetic adjoint (decay pullback) and folds the curl-E
    difference into the pre-step electric adjoint, matching
    ``reverse_tfsf_auxiliary_magnetic_kernel``.
    """
    magnetic_total = adj_magnetic_after.numel()
    adj_magnetic_prev.copy_(magnetic_decay * adj_magnetic_after)
    if magnetic_total == 0:
        return
    curl_flux = magnetic_curl * adj_magnetic_after
    adj_electric_prev[:magnetic_total].add_(curl_flux)
    adj_electric_prev[1:].add_(-curl_flux)


def _reverse_tfsf_auxiliary_state_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    magnetic_output_adjoint,
):
    """Analytic (autograd-free) transpose of the per-step TFSF auxiliary update.

    Replays the reverse of the one-dimensional auxiliary electric/magnetic advance
    and the incident sample-and-inject in the exact order the forward step runs
    them, so no ``torch.autograd`` graph is built on the reverse hot path. Mirrors
    the native TFSF reverse kernels bit-for-bit (parity-tested), and stays equal to
    the full ``torch_vjp`` auxiliary gradient (``test_reverse_step_tfsf_python_
    reference_matches_torch_vjp``).

    The magnetic incident correction is injected into the mid-step H that both the
    post-step H seed and the electric update consume, so its sample-adjoint reads
    the base reverse's ``magnetic_output_adjoint`` (the full mid-H adjoint), not the
    raw post-step H seed.
    """
    auxiliary_state = _extract_tfsf_auxiliary_state(forward_state)
    if auxiliary_state is None:
        return {}
    tfsf_state = getattr(solver, "_tfsf_state", None)
    if tfsf_state is None:
        return {}
    auxiliary_grid = tfsf_state.get("auxiliary_grid")
    if auxiliary_grid is None:
        return {}

    adj_electric_prev = torch.zeros_like(auxiliary_state["electric"])
    adj_magnetic_prev = torch.zeros_like(auxiliary_state["magnetic"])
    adj_magnetic_after = adjoint_state["tfsf_aux_magnetic"].detach().clone()
    adj_electric_post = adjoint_state["tfsf_aux_electric"]

    ds = float(auxiliary_grid.ds)
    origin_electric = float(auxiliary_grid.s_min)
    origin_magnetic = float(auxiliary_grid.s_min + 0.5 * ds)

    # (1) Electric incident injection samples the advanced magnetic grid.
    _accumulate_tfsf_sample_adjoint_torch(
        adj_magnetic_after,
        tfsf_state.get("electric_terms", ()),
        adjoint_state,
        origin=origin_magnetic,
        ds=ds,
    )
    # (2) Electric auxiliary advance reverse (reads adj E1, writes adj E0 / adj H1).
    _reverse_tfsf_auxiliary_electric_state_adjoint(
        adj_electric_prev,
        adj_magnetic_after,
        adj_electric_post,
        auxiliary_grid.electric_decay,
        auxiliary_grid.electric_curl,
        int(auxiliary_grid.source_index),
    )
    # (3) Magnetic auxiliary advance reverse (reads adj H1, writes adj H0 / adj E0).
    _reverse_tfsf_auxiliary_magnetic_state_adjoint(
        adj_electric_prev,
        adj_magnetic_prev,
        adj_magnetic_after,
        auxiliary_grid.magnetic_decay,
        auxiliary_grid.magnetic_curl,
    )
    # (4) Magnetic incident injection samples the pre-step electric grid; it lands
    # on the mid-step H, so read the full mid-H adjoint from the base reverse.
    _accumulate_tfsf_sample_adjoint_torch(
        adj_electric_prev,
        tfsf_state.get("magnetic_terms", ()),
        magnetic_output_adjoint,
        origin=origin_electric,
        ds=ds,
    )

    return {
        "tfsf_aux_electric": adj_electric_prev,
        "tfsf_aux_magnetic": adj_magnetic_prev,
    }


def _bloch_backward_diff(field: torch.Tensor, *, axis: int, inv_delta: torch.Tensor, phase_cos: float, phase_sin: float):
    # inv_delta: 1D dual spacing reciprocals, length == field size + 1 along
    # axis; wrap entries live at [0] and [-1] (equal on a Bloch axis).
    shape = list(field.shape)
    shape[axis] += 1
    diff = field.new_zeros(shape)
    if field.shape[axis] == 0:
        return diff

    low = field.select(axis, 0)
    high = field.select(axis, field.shape[axis] - 1)
    low_value = (low - _complex_phase_negative(high, phase_cos, phase_sin)) * inv_delta[0]
    low_region = [slice(None)] * field.ndim
    low_region[axis] = 0
    diff[tuple(low_region)] = low_value

    if field.shape[axis] == 1:
        return diff

    high_value = (_complex_phase_positive(low, phase_cos, phase_sin) - high) * inv_delta[-1]
    high_region = [slice(None)] * diff.ndim
    high_region[axis] = -1
    diff[tuple(high_region)] = high_value

    if field.shape[axis] > 1:
        interior = [slice(None)] * diff.ndim
        interior[axis] = slice(1, -1)
        current = [slice(None)] * field.ndim
        current[axis] = slice(1, None)
        previous = [slice(None)] * field.ndim
        previous[axis] = slice(0, -1)
        diff[tuple(interior)] = (field[tuple(current)] - field[tuple(previous)]) * _broadcast_vector(inv_delta[1:-1], axis)
    return diff


def _apply_source_term_list(field_mapping, *, terms, source_time, omega, time_value, solver=None):
    if not terms:
        return field_mapping

    updated = dict(field_mapping)
    signal_cache = {}
    cw_cache = {}
    for term in terms:
        term_source_time = term.get("source_time") or source_time
        term_omega = float(omega if term.get("omega") is None else term["omega"])
        cache_key = term.get("source_index")
        if cache_key is None:
            cache_key = (id(term_source_time), term_omega)
        literal_patch = term.get("literal_patch")
        if literal_patch is not None:
            updated[term["field_name"]] = _add_patch(updated[term["field_name"]], term["offsets"], literal_patch)
            continue
        if term["cw_cos_patch"] is not None:
            if cache_key not in cw_cache:
                cw_cache[cache_key] = _resolve_cw_signal(term_source_time, term_omega, time_value)
            signal_cos, signal_sin = cw_cache[cache_key]
            if solver is not None and has_complex_fields(solver):
                updated = _apply_bloch_cw_source_term(
                    updated,
                    solver=solver,
                    field_name=term["field_name"],
                    offsets=term["offsets"],
                    cw_cos_patch=term["cw_cos_patch"],
                    cw_sin_patch=term["cw_sin_patch"],
                    signal_cos=signal_cos,
                    signal_sin=signal_sin,
                )
                continue
            patch = float(signal_cos) * term["cw_cos_patch"] + float(signal_sin) * term["cw_sin_patch"]
        elif term["delay_patch"] is not None:
            sample_time = time_value - term["delay_patch"]
            patch = _evaluate_source_time_tensor(term_source_time, sample_time) * term["patch"]
            activation_delay_patch = term.get("activation_delay_patch")
            if activation_delay_patch is not None:
                patch = torch.where(
                    time_value < activation_delay_patch,
                    torch.zeros_like(patch),
                    patch,
                )
            if solver is not None and has_complex_fields(solver):
                # The delayed pulse is a real current; scatter it into the split
                # real/imag Bloch field with the same wrap-phase rule the forward
                # ``addSourcePatchBloch3D`` kernel uses, so the reverse-replay
                # reconstruction (and the eps VJP flowing through it) matches the
                # forward injection exactly.
                real_patch = patch.to(
                    device=updated[term["field_name"]].device,
                    dtype=updated[term["field_name"]].dtype,
                )
                updated = _add_complex_bloch_source_patch(
                    updated,
                    solver=solver,
                    field_name=term["field_name"],
                    offsets=term["offsets"],
                    delta=torch.complex(real_patch, torch.zeros_like(real_patch)),
                )
                continue
        else:
            if cache_key not in signal_cache:
                signal_cache[cache_key] = evaluate_source_time(term_source_time, float(time_value))
            scalar_signal = signal_cache[cache_key]
            signal_real = float(scalar_signal) * float(term["phase_real"])
            signal_imag = float(scalar_signal) * float(term["phase_imag"])
            if solver is not None and has_complex_fields(solver):
                updated = _apply_bloch_uniform_source_term(
                    updated,
                    solver=solver,
                    field_name=term["field_name"],
                    offsets=term["offsets"],
                    patch=term["patch"],
                    signal_real=signal_real,
                    signal_imag=signal_imag,
                )
                continue
            patch = signal_real * term["patch"]
        field_name = term["field_name"]
        updated[field_name] = _add_patch(updated[field_name], term["offsets"], patch)
    return updated


def _update_debye_state(field, polarization, *, drive, decay, dt):
    next_polarization = polarization * float(decay) + drive * field
    next_current = (next_polarization - polarization) / float(dt)
    return next_polarization, next_current


def _update_drude_state(field, current, *, drive, decay):
    return current * float(decay) + drive * field


def _update_lorentz_state(field, polarization, current, *, drive, decay, restoring, dt):
    next_current = current * float(decay) - polarization * float(restoring) + drive * field
    next_polarization = polarization + float(dt) * next_current
    return next_polarization, next_current


def _advance_ade_state(solver, state, specs, *, imag=False):
    # ``imag`` advances the imaginary-field ADE replica the forward solver keeps
    # for Bloch (complex-field) runs: same real pole coefficients, but driven by
    # the imaginary field component and stored under ``*_imag`` state names.
    suffix = "_imag" if imag else ""
    updated = {}
    for component_name, model_name, index, _tensor_names, entry in specs or ():
        field = state[component_name + suffix]
        if model_name == "debye":
            polarization_name = dispersive_state_name(component_name, model_name, index, "polarization") + suffix
            current_name = dispersive_state_name(component_name, model_name, index, "current") + suffix
            polarization, current = _update_debye_state(
                field,
                state[polarization_name],
                drive=entry["drive"],
                decay=entry["decay"],
                dt=solver.dt,
            )
            updated[polarization_name] = polarization
            updated[current_name] = current
            continue
        if model_name == "drude":
            current_name = dispersive_state_name(component_name, model_name, index, "current") + suffix
            updated[current_name] = _update_drude_state(
                field,
                state[current_name],
                drive=entry["drive"],
                decay=entry["decay"],
            )
            continue

        polarization_name = dispersive_state_name(component_name, model_name, index, "polarization") + suffix
        current_name = dispersive_state_name(component_name, model_name, index, "current") + suffix
        polarization, current = _update_lorentz_state(
            field,
            state[polarization_name],
            state[current_name],
            drive=entry["drive"],
            decay=entry["decay"],
            restoring=entry["restoring"],
            dt=solver.dt,
        )
        updated[polarization_name] = polarization
        updated[current_name] = current
    return updated


def _advance_dispersive_state(solver, state):
    updated = _advance_ade_state(solver, state, iter_dispersive_state_specs(solver))
    if has_complex_fields(solver):
        # Bloch runs propagate a second real FDTD copy (the imaginary field);
        # the electric ADE poles advance on it identically, so mirror the real
        # pass with the imaginary drive to keep the replay bit-consistent with
        # the forward kernels.
        updated.update(
            _advance_ade_state(solver, state, iter_dispersive_state_specs(solver), imag=True)
        )
    return updated


def _advance_magnetic_dispersive_state(solver, state):
    return _advance_ade_state(solver, state, iter_magnetic_dispersive_state_specs(solver))


def _magnetic_inverse_permeabilities(solver):
    templates = getattr(solver, "_magnetic_dispersive_templates", {}) or {}
    return {
        component_name: templates[component_name]["inv_mu"]
        for component_name in ("Hx", "Hy", "Hz")
        if component_name in templates
    }


def _apply_magnetic_dispersive_corrections(solver, magnetic_fields, magnetic_dispersive_state):
    """H -= dt * J_m / mu with the same edge-averaged inv_mu the runtime kernels use.

    mu carries no gradient channel (it is a constant coefficient tensor in the
    adjoint), matching the forward `applyPolarizationCurrent3D` launches exactly.
    """
    if not getattr(solver, "magnetic_dispersive_enabled", False):
        return magnetic_fields

    inv_mu = _magnetic_inverse_permeabilities(solver)
    updated = dict(magnetic_fields)
    for component_name, model_name, index, _tensor_names, _entry in iter_magnetic_dispersive_state_specs(solver) or ():
        current_name = dispersive_state_name(component_name, model_name, index, "current")
        updated[component_name] = updated[component_name] - float(solver.dt) * magnetic_dispersive_state[
            current_name
        ] * inv_mu[component_name]
    return updated


def _apply_dispersive_corrections(solver, electric_fields, dispersive_state, *, eps_ex, eps_ey, eps_ez):
    if not getattr(solver, "dispersive_enabled", False):
        return electric_fields

    inv_eps = {
        "Ex": 1.0 / eps_ex,
        "Ey": 1.0 / eps_ey,
        "Ez": 1.0 / eps_ez,
    }
    updated = dict(electric_fields)
    dt = float(solver.dt)
    # On Bloch runs the imaginary field carries its own polarization current, so
    # subtract dt * J_imag / eps from the imaginary component too (matching the
    # forward apply_component_dispersive_currents(imag=True) launches).
    complex_fields = has_complex_fields(solver) and all(
        f"{component_name}_imag" in electric_fields for component_name in ("Ex", "Ey", "Ez")
    )
    for component_name, model_name, index, _tensor_names, _entry in iter_dispersive_state_specs(solver) or ():
        current_name = dispersive_state_name(component_name, model_name, index, "current")
        updated[component_name] = updated[component_name] - dt * dispersive_state[current_name] * inv_eps[
            component_name
        ]
        if complex_fields:
            imag_field = f"{component_name}_imag"
            updated[imag_field] = updated[imag_field] - dt * dispersive_state[current_name + "_imag"] * inv_eps[
                component_name
            ]
    return updated


def _reverse_dispersive_corrections(
    solver,
    adjoint_state,
    dispersive_state,
    *,
    eps_ex,
    eps_ey,
    eps_ez,
):
    electric_source_adjoint = {
        "Ex": adjoint_state["Ex"].detach().clone(),
        "Ey": adjoint_state["Ey"].detach().clone(),
        "Ez": adjoint_state["Ez"].detach().clone(),
    }
    dispersive_output_adjoint = {
        name: adjoint_state[name].detach().clone()
        for name in checkpoint_schema(solver).dispersive_state_names
    }
    grad_eps_by_field = {
        "Ex": torch.zeros_like(eps_ex),
        "Ey": torch.zeros_like(eps_ey),
        "Ez": torch.zeros_like(eps_ez),
    }
    eps_by_field = {
        "Ex": eps_ex,
        "Ey": eps_ey,
        "Ez": eps_ez,
    }
    dt = float(solver.dt)

    for component_name, model_name, index, _tensor_names, _entry in iter_dispersive_state_specs(solver) or ():
        current_name = dispersive_state_name(component_name, model_name, index, "current")
        electric_adjoint = electric_source_adjoint[component_name]
        eps_field = eps_by_field[component_name]
        dispersive_output_adjoint[current_name] = (
            dispersive_output_adjoint[current_name] - dt * electric_adjoint / eps_field
        )
        grad_eps_by_field[component_name] = (
            grad_eps_by_field[component_name] + dt * dispersive_state[current_name] * electric_adjoint / (eps_field * eps_field)
        )

    source_adjoint_state = dict(adjoint_state)
    source_adjoint_state.update(electric_source_adjoint)
    return electric_source_adjoint, dispersive_output_adjoint, grad_eps_by_field, source_adjoint_state


def _reverse_ade_state_python_reference(solver, forward_state, output_adjoint, *, specs, component_names, state_names):
    field_prev_adjoint = {
        name: torch.zeros_like(forward_state[name])
        for name in component_names
    }
    pre_step_dispersive_adjoint = {
        name: torch.zeros_like(forward_state[name])
        for name in state_names
    }
    dt = float(solver.dt)

    for component_name, model_name, index, _tensor_names, entry in specs or ():
        drive = entry["drive"]
        if model_name == "debye":
            polarization_name = dispersive_state_name(component_name, model_name, index, "polarization")
            current_name = dispersive_state_name(component_name, model_name, index, "current")
            adj_polarization_post = output_adjoint[polarization_name]
            adj_current_post = output_adjoint[current_name]
            adj_polarization_internal = adj_polarization_post + adj_current_post / dt
            field_prev_adjoint[component_name] = field_prev_adjoint[component_name] + drive * adj_polarization_internal
            pre_step_dispersive_adjoint[polarization_name] = (
                pre_step_dispersive_adjoint[polarization_name]
                + float(entry["decay"]) * adj_polarization_internal
                - adj_current_post / dt
            )
            continue

        if model_name == "drude":
            current_name = dispersive_state_name(component_name, model_name, index, "current")
            adj_current_post = output_adjoint[current_name]
            field_prev_adjoint[component_name] = field_prev_adjoint[component_name] + drive * adj_current_post
            pre_step_dispersive_adjoint[current_name] = (
                pre_step_dispersive_adjoint[current_name] + float(entry["decay"]) * adj_current_post
            )
            continue

        polarization_name = dispersive_state_name(component_name, model_name, index, "polarization")
        current_name = dispersive_state_name(component_name, model_name, index, "current")
        adj_polarization_post = output_adjoint[polarization_name]
        adj_current_post = output_adjoint[current_name]
        adj_current_internal = adj_current_post + dt * adj_polarization_post
        field_prev_adjoint[component_name] = field_prev_adjoint[component_name] + drive * adj_current_internal
        pre_step_dispersive_adjoint[polarization_name] = (
            pre_step_dispersive_adjoint[polarization_name]
            + adj_polarization_post
            - float(entry["restoring"]) * adj_current_internal
        )
        pre_step_dispersive_adjoint[current_name] = (
            pre_step_dispersive_adjoint[current_name] + float(entry["decay"]) * adj_current_internal
        )

    return field_prev_adjoint, pre_step_dispersive_adjoint


def _reverse_dispersive_state_python_reference(solver, forward_state, dispersive_output_adjoint):
    return _reverse_ade_state_python_reference(
        solver,
        forward_state,
        dispersive_output_adjoint,
        specs=iter_dispersive_state_specs(solver),
        component_names=("Ex", "Ey", "Ez"),
        state_names=checkpoint_schema(solver).dispersive_state_names,
    )


def _reverse_magnetic_dispersive_state_python_reference(solver, forward_state, magnetic_output_adjoint):
    return _reverse_ade_state_python_reference(
        solver,
        forward_state,
        magnetic_output_adjoint,
        specs=iter_magnetic_dispersive_state_specs(solver),
        component_names=("Hx", "Hy", "Hz"),
        state_names=checkpoint_schema(solver).magnetic_dispersive_state_names,
    )


def _reverse_magnetic_dispersive_corrections(solver, adjoint_state, magnetic_field_adjoint):
    """VJP of ``H -= dt * J_m * inv_mu`` given the adjoint of the corrected H.

    ``inv_mu`` is a constant coefficient tensor (there is no magnetic material
    gradient channel), so the correction only seeds the post-step magnetic
    dispersive-state adjoint; the field adjoint itself passes through unchanged.
    """
    output_adjoint = {
        name: adjoint_state[name].detach().clone()
        for name in checkpoint_schema(solver).magnetic_dispersive_state_names
    }
    inv_mu = _magnetic_inverse_permeabilities(solver)
    dt = float(solver.dt)
    for component_name, model_name, index, _tensor_names, _entry in iter_magnetic_dispersive_state_specs(solver) or ():
        current_name = dispersive_state_name(component_name, model_name, index, "current")
        output_adjoint[current_name] = (
            output_adjoint[current_name] - dt * magnetic_field_adjoint[component_name] * inv_mu[component_name]
        )
    return output_adjoint


def _broadcast_vector(vector: torch.Tensor, axis: int) -> torch.Tensor:
    shape = [1, 1, 1]
    shape[axis] = int(vector.shape[0])
    return vector.view(*shape)


def _forward_diff(field: torch.Tensor, axis: int, inv_delta: torch.Tensor) -> torch.Tensor:
    # inv_delta: 1D primal spacing reciprocals, length == field size - 1 along axis.
    slicer_lo = [slice(None)] * field.ndim
    slicer_hi = [slice(None)] * field.ndim
    slicer_lo[axis] = slice(0, -1)
    slicer_hi[axis] = slice(1, None)
    return (field[tuple(slicer_hi)] - field[tuple(slicer_lo)]) * _broadcast_vector(inv_delta, axis)


def _backward_diff(field: torch.Tensor, axis: int, inv_delta: torch.Tensor) -> torch.Tensor:
    # inv_delta: 1D dual spacing reciprocals, length == field size + 1 along axis.
    shape = list(field.shape)
    shape[axis] += 1
    diff = field.new_zeros(shape)
    interior = [slice(None)] * diff.ndim
    field_hi = [slice(None)] * field.ndim
    field_lo = [slice(None)] * field.ndim
    interior[axis] = slice(1, -1)
    field_hi[axis] = slice(1, None)
    field_lo[axis] = slice(0, -1)
    diff[tuple(interior)] = (field[tuple(field_hi)] - field[tuple(field_lo)]) * _broadcast_vector(inv_delta[1:-1], axis)
    return diff


def _scatter_diff_adjoint(field_grad: torch.Tensor, scaled_grad: torch.Tensor, axis: int):
    field_lo = [slice(None)] * field_grad.ndim
    field_hi = [slice(None)] * field_grad.ndim
    field_lo[axis] = slice(0, -1)
    field_hi[axis] = slice(1, None)
    field_grad[tuple(field_lo)] = field_grad[tuple(field_lo)] - scaled_grad
    field_grad[tuple(field_hi)] = field_grad[tuple(field_hi)] + scaled_grad


def _accumulate_forward_diff_adjoint(field_grad: torch.Tensor, diff_grad: torch.Tensor, *, axis: int, inv_delta: torch.Tensor):
    _scatter_diff_adjoint(field_grad, _broadcast_vector(inv_delta, axis) * diff_grad, axis)


def _accumulate_backward_diff_adjoint(field_grad: torch.Tensor, diff_grad: torch.Tensor, *, axis: int, inv_delta: torch.Tensor):
    interior = [slice(None)] * diff_grad.ndim
    interior[axis] = slice(1, -1)
    _scatter_diff_adjoint(field_grad, _broadcast_vector(inv_delta[1:-1], axis) * diff_grad[tuple(interior)], axis)


def _accumulate_bloch_backward_diff_adjoint(
    field_grad: torch.Tensor,
    diff_grad: torch.Tensor,
    *,
    axis: int,
    inv_delta: torch.Tensor,
    phase_cos: float,
    phase_sin: float,
):
    _accumulate_backward_diff_adjoint(field_grad, diff_grad, axis=axis, inv_delta=inv_delta)
    # Wrap terms transpose the forward wrap differences: contributions from the
    # low diff entry carry inv_delta[0], those from the high entry inv_delta[-1].
    low_scale = inv_delta[0]
    high_scale = inv_delta[-1]
    low_grad = diff_grad.select(axis, 0)
    high_grad = diff_grad.select(axis, int(diff_grad.shape[axis] - 1))
    field_grad.select(axis, 0).add_(low_scale * low_grad + high_scale * _complex_phase_negative(high_grad, phase_cos, phase_sin))
    field_grad.select(axis, int(field_grad.shape[axis] - 1)).add_(
        -low_scale * _complex_phase_positive(low_grad, phase_cos, phase_sin) - high_scale * high_grad
    )


# The face codes and grid shape are fixed for the whole reverse pass, so the
# boolean boundary masks the per-step electric-adjoint update reads (six rebuilds
# per reverse step) are config-level constants. Memoize them: the returned masks
# are consumed read-only (|, &, ~, torch.where) at every call site, so a shared
# cached tensor is safe, and the key (shape/axis/face-codes/device) recurs
# identically across every step and every optimization iteration on a scene.
_BOUNDARY_AXIS_MASK_CACHE: dict = {}


def _boundary_axis_masks(shape, axis: int, low_mode: int, high_mode: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (tuple(int(dim) for dim in shape), int(axis), int(low_mode), int(high_mode), str(device))
    cached = _BOUNDARY_AXIS_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    inactive = torch.zeros(shape, device=device, dtype=torch.bool)
    pec = torch.zeros(shape, device=device, dtype=torch.bool)
    low_face = [slice(None)] * len(shape)
    low_face[axis] = 0
    high_face = [slice(None)] * len(shape)
    high_face[axis] = -1
    if low_mode in {BOUNDARY_NONE, BOUNDARY_PML}:
        inactive[tuple(low_face)] = True
    elif low_mode == BOUNDARY_PEC:
        pec[tuple(low_face)] = True
    if high_mode in {BOUNDARY_NONE, BOUNDARY_PML}:
        inactive[tuple(high_face)] = True
    elif high_mode == BOUNDARY_PEC:
        pec[tuple(high_face)] = True
    _BOUNDARY_AXIS_MASK_CACHE[key] = (inactive, pec)
    return inactive, pec


def _update_magnetic_component(field, *, d_pos, d_neg, decay, curl, psi_pos=None, psi_neg=None, b_pos=None, c_pos=None, inv_kappa_pos=None, b_neg=None, c_neg=None, inv_kappa_neg=None, axis_pos=None, axis_neg=None):
    if psi_pos is not None:
        psi_pos_new = _broadcast_vector(b_pos, axis_pos) * psi_pos + _broadcast_vector(c_pos, axis_pos) * d_pos
        psi_neg_new = _broadcast_vector(b_neg, axis_neg) * psi_neg + _broadcast_vector(c_neg, axis_neg) * d_neg
        curl_term = (
            d_pos * _broadcast_vector(inv_kappa_pos, axis_pos) + psi_pos_new
        ) - (
            d_neg * _broadcast_vector(inv_kappa_neg, axis_neg) + psi_neg_new
        )
        updated = field * decay - curl * curl_term
        return updated, psi_pos_new, psi_neg_new

    updated = field * decay - curl * (d_pos - d_neg)
    return updated, None, None


def _update_electric_component(field, *, d_pos, d_neg, decay, curl_prefactor, eps, low_mode_pos, high_mode_pos, low_mode_neg, high_mode_neg, axis_pos, axis_neg, psi_pos=None, psi_neg=None, b_pos=None, c_pos=None, inv_kappa_pos=None, b_neg=None, c_neg=None, inv_kappa_neg=None):
    curl = curl_prefactor / eps
    if psi_pos is not None:
        psi_pos_candidate = _broadcast_vector(b_pos, axis_pos) * psi_pos + _broadcast_vector(c_pos, axis_pos) * d_pos
        psi_neg_candidate = _broadcast_vector(b_neg, axis_neg) * psi_neg + _broadcast_vector(c_neg, axis_neg) * d_neg
        curl_term = (
            d_pos * _broadcast_vector(inv_kappa_pos, axis_pos) + psi_pos_candidate
        ) - (
            d_neg * _broadcast_vector(inv_kappa_neg, axis_neg) + psi_neg_candidate
        )
    else:
        psi_pos_candidate = None
        psi_neg_candidate = None
        curl_term = d_pos - d_neg

    inactive_pos, pec_pos = _boundary_axis_masks(field.shape, axis_pos, low_mode_pos, high_mode_pos, field.device)
    inactive_neg, pec_neg = _boundary_axis_masks(field.shape, axis_neg, low_mode_neg, high_mode_neg, field.device)
    pec_mask = pec_pos | pec_neg
    inactive_mask = (~pec_mask) & (inactive_pos | inactive_neg)

    candidate = field * decay + curl * curl_term
    updated = torch.where(pec_mask, torch.zeros_like(candidate), torch.where(inactive_mask, field, candidate))
    if psi_pos_candidate is None:
        return updated, None, None

    keep_mask = inactive_mask | pec_mask
    psi_pos_new = torch.where(keep_mask, psi_pos, psi_pos_candidate)
    psi_neg_new = torch.where(keep_mask, psi_neg, psi_neg_candidate)
    return updated, psi_pos_new, psi_neg_new


def _reverse_magnetic_component_standard(adj_updated, *, decay, curl):
    adj_field = adj_updated * decay
    adj_d_pos = -curl * adj_updated
    adj_d_neg = curl * adj_updated
    return adj_field, adj_d_pos, adj_d_neg


def _reverse_electric_component_standard(
    adj_updated,
    field,
    *,
    d_pos,
    d_neg,
    decay,
    curl_prefactor,
    eps,
    low_mode_pos,
    high_mode_pos,
    low_mode_neg,
    high_mode_neg,
    axis_pos,
    axis_neg,
):
    inactive_pos, pec_pos = _boundary_axis_masks(field.shape, axis_pos, low_mode_pos, high_mode_pos, field.device)
    inactive_neg, pec_neg = _boundary_axis_masks(field.shape, axis_neg, low_mode_neg, high_mode_neg, field.device)
    pec_mask = pec_pos | pec_neg
    inactive_mask = (~pec_mask) & (inactive_pos | inactive_neg)
    active_mask = (~pec_mask) & (~inactive_mask)
    active = active_mask.to(device=field.device, dtype=field.dtype)
    inactive = inactive_mask.to(device=field.device, dtype=field.dtype)

    adj_field = inactive * adj_updated + active * (adj_updated * decay)
    curl_scale = active * (curl_prefactor / eps)
    adj_curl_term = adj_updated * curl_scale
    grad_eps = -adj_updated * active * curl_prefactor * (d_pos - d_neg) / (eps * eps)
    adj_d_pos = adj_curl_term
    adj_d_neg = -adj_curl_term
    return adj_field, adj_d_pos, adj_d_neg, grad_eps


def _reverse_electric_component_bloch(
    adj_updated: torch.Tensor,
    *,
    d_pos: torch.Tensor,
    d_neg: torch.Tensor,
    decay: torch.Tensor,
    curl_prefactor: torch.Tensor,
    eps: torch.Tensor,
):
    adj_field = adj_updated * decay
    curl_scale = curl_prefactor / eps
    adj_curl_term = adj_updated * curl_scale
    grad_eps = -curl_prefactor * _complex_inner_real(adj_updated, d_pos - d_neg) / (eps * eps)
    adj_d_pos = adj_curl_term
    adj_d_neg = -adj_curl_term
    return adj_field, adj_d_pos, adj_d_neg, grad_eps


def _reverse_magnetic_component_cpml(
    adj_updated,
    adj_psi_pos_post,
    adj_psi_neg_post,
    *,
    decay,
    curl,
    b_pos,
    c_pos,
    inv_kappa_pos,
    b_neg,
    c_neg,
    inv_kappa_neg,
    axis_pos,
    axis_neg,
):
    adj_curl_term = -curl * adj_updated
    adj_psi_pos_candidate = adj_psi_pos_post + adj_curl_term
    adj_psi_neg_candidate = adj_psi_neg_post - adj_curl_term
    adj_field = adj_updated * decay
    adj_psi_pos = _broadcast_vector(b_pos, axis_pos) * adj_psi_pos_candidate
    adj_psi_neg = _broadcast_vector(b_neg, axis_neg) * adj_psi_neg_candidate
    adj_d_pos = (
        _broadcast_vector(inv_kappa_pos, axis_pos) * adj_curl_term
        + _broadcast_vector(c_pos, axis_pos) * adj_psi_pos_candidate
    )
    adj_d_neg = (
        -_broadcast_vector(inv_kappa_neg, axis_neg) * adj_curl_term
        + _broadcast_vector(c_neg, axis_neg) * adj_psi_neg_candidate
    )
    return adj_field, adj_d_pos, adj_d_neg, adj_psi_pos, adj_psi_neg


def _reverse_electric_component_cpml(
    adj_updated,
    adj_psi_pos_post,
    adj_psi_neg_post,
    field,
    *,
    d_pos,
    d_neg,
    decay,
    curl_prefactor,
    eps,
    low_mode_pos,
    high_mode_pos,
    low_mode_neg,
    high_mode_neg,
    axis_pos,
    axis_neg,
    psi_pos,
    psi_neg,
    b_pos,
    c_pos,
    inv_kappa_pos,
    b_neg,
    c_neg,
    inv_kappa_neg,
):
    inactive_pos, pec_pos = _boundary_axis_masks(field.shape, axis_pos, low_mode_pos, high_mode_pos, field.device)
    inactive_neg, pec_neg = _boundary_axis_masks(field.shape, axis_neg, low_mode_neg, high_mode_neg, field.device)
    pec_mask = pec_pos | pec_neg
    inactive_mask = (~pec_mask) & (inactive_pos | inactive_neg)
    active_mask = (~pec_mask) & (~inactive_mask)
    keep_mask = inactive_mask | pec_mask

    active = active_mask.to(device=field.device, dtype=field.dtype)
    inactive = inactive_mask.to(device=field.device, dtype=field.dtype)
    keep = keep_mask.to(device=field.device, dtype=field.dtype)

    b_pos_term = _broadcast_vector(b_pos, axis_pos)
    c_pos_term = _broadcast_vector(c_pos, axis_pos)
    inv_kappa_pos_term = _broadcast_vector(inv_kappa_pos, axis_pos)
    b_neg_term = _broadcast_vector(b_neg, axis_neg)
    c_neg_term = _broadcast_vector(c_neg, axis_neg)
    inv_kappa_neg_term = _broadcast_vector(inv_kappa_neg, axis_neg)

    psi_pos_candidate = b_pos_term * psi_pos + c_pos_term * d_pos
    psi_neg_candidate = b_neg_term * psi_neg + c_neg_term * d_neg
    curl_term = (d_pos * inv_kappa_pos_term + psi_pos_candidate) - (d_neg * inv_kappa_neg_term + psi_neg_candidate)

    adj_field = inactive * adj_updated + active * (adj_updated * decay)
    adj_curl_term = active * adj_updated * (curl_prefactor / eps)
    grad_eps = -adj_updated * active * curl_prefactor * curl_term / (eps * eps)
    adj_psi_pos_candidate = active * adj_psi_pos_post + adj_curl_term
    adj_psi_neg_candidate = active * adj_psi_neg_post - adj_curl_term
    adj_psi_pos = keep * adj_psi_pos_post + active * (b_pos_term * adj_psi_pos_candidate)
    adj_psi_neg = keep * adj_psi_neg_post + active * (b_neg_term * adj_psi_neg_candidate)
    adj_d_pos = inv_kappa_pos_term * adj_curl_term + c_pos_term * adj_psi_pos_candidate
    adj_d_neg = -inv_kappa_neg_term * adj_curl_term + c_neg_term * adj_psi_neg_candidate
    return adj_field, adj_d_pos, adj_d_neg, grad_eps, adj_psi_pos, adj_psi_neg


def _enforce_pec_boundaries(solver, fields):
    updated = dict(fields)
    face_specs = (
        (solver.boundary_x_low_code, "low", (("Ey", 0), ("Ez", 0))),
        (solver.boundary_x_high_code, "high", (("Ey", 0), ("Ez", 0))),
        (solver.boundary_y_low_code, "low", (("Ex", 1), ("Ez", 1))),
        (solver.boundary_y_high_code, "high", (("Ex", 1), ("Ez", 1))),
        (solver.boundary_z_low_code, "low", (("Ex", 2), ("Ey", 2))),
        (solver.boundary_z_high_code, "high", (("Ex", 2), ("Ey", 2))),
    )
    for boundary_code, side, targets in face_specs:
        if boundary_code != BOUNDARY_PEC:
            continue
        face_index = 0 if side == "low" else -1
        for field_name, axis in targets:
            tensor = updated[field_name].clone()
            region = [slice(None)] * tensor.ndim
            region[axis] = face_index
            tensor[tuple(region)] = 0.0
            updated[field_name] = tensor
    return updated


def _complex_state_field(mapping, field_name: str) -> torch.Tensor:
    return torch.complex(mapping[field_name], mapping[f"{field_name}_imag"])


def _apply_complex_axis_boundary(candidate, previous, *, axis: int, low_mode: int, high_mode: int):
    inactive_mask, pec_mask = _boundary_axis_masks(candidate.shape, axis, low_mode, high_mode, candidate.device)
    return torch.where(
        pec_mask,
        torch.zeros_like(candidate),
        torch.where(inactive_mask, previous, candidate),
    )


def _cpml_correction_mask(shape, *, normal_axis: int, tangent_axis: int, tangent_low_mode: int, tangent_high_mode: int, device):
    mask = torch.ones(shape, device=device, dtype=torch.bool)
    low_face = [slice(None)] * len(shape)
    high_face = [slice(None)] * len(shape)
    low_face[normal_axis] = 0
    high_face[normal_axis] = -1
    mask[tuple(low_face)] = False
    mask[tuple(high_face)] = False
    inactive_tangent, pec_tangent = _boundary_axis_masks(
        shape,
        tangent_axis,
        tangent_low_mode,
        tangent_high_mode,
        device,
    )
    return mask & (~inactive_tangent) & (~pec_tangent)


def _apply_complex_cpml_correction(
    field: torch.Tensor,
    psi: torch.Tensor,
    derivative: torch.Tensor,
    *,
    curl: torch.Tensor,
    inv_kappa: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    normal_axis: int,
    tangent_axis: int,
    tangent_low_mode: int,
    tangent_high_mode: int,
    sign: float,
):
    b_term = _broadcast_vector(b, normal_axis)
    c_term = _broadcast_vector(c, normal_axis)
    inv_kappa_term = _broadcast_vector(inv_kappa, normal_axis)
    psi_candidate = b_term * psi + c_term * derivative
    correction = curl * (derivative * (inv_kappa_term - 1.0) + psi_candidate)
    candidate = field + float(sign) * correction
    mask = _cpml_correction_mask(
        field.shape,
        normal_axis=normal_axis,
        tangent_axis=tangent_axis,
        tangent_low_mode=tangent_low_mode,
        tangent_high_mode=tangent_high_mode,
        device=field.device,
    )
    return torch.where(mask, candidate, field), torch.where(mask, psi_candidate, psi)


def _update_mixed_bloch_cpml_electric_fields(solver, state, magnetic_fields, *, ex_curl, ey_curl, ez_curl):
    hx_complex = _complex_state_field(magnetic_fields, "Hx")
    hy_complex = _complex_state_field(magnetic_fields, "Hy")
    hz_complex = _complex_state_field(magnetic_fields, "Hz")

    previous_ex = _complex_state_field(state, "Ex")
    d_hz_dy = _bloch_backward_diff(
        hz_complex,
        axis=1,
        inv_delta=solver.inv_dy_e,
        phase_cos=float(solver.boundary_phase_cos[1]),
        phase_sin=float(solver.boundary_phase_sin[1]),
    )
    d_hy_dz = _backward_diff(hy_complex, axis=2, inv_delta=solver.inv_dz_e)
    ex_complex = previous_ex * solver.cex_decay + ex_curl * (d_hz_dy - d_hy_dz)
    ex_complex = _apply_complex_axis_boundary(
        ex_complex,
        previous_ex,
        axis=2,
        low_mode=solver.boundary_z_low_code,
        high_mode=solver.boundary_z_high_code,
    )
    psi_ex_z = torch.complex(state["psi_ex_z"], state["psi_ex_z_imag"])
    ex_complex, psi_ex_z = _apply_complex_cpml_correction(
        ex_complex,
        psi_ex_z,
        d_hy_dz,
        curl=ex_curl,
        inv_kappa=solver.cpml_inv_kappa_e_z,
        b=solver.cpml_b_e_z,
        c=solver.cpml_c_e_z,
        normal_axis=2,
        tangent_axis=1,
        tangent_low_mode=solver.boundary_y_low_code,
        tangent_high_mode=solver.boundary_y_high_code,
        sign=-1.0,
    )

    previous_ey = _complex_state_field(state, "Ey")
    d_hx_dz = _backward_diff(hx_complex, axis=2, inv_delta=solver.inv_dz_e)
    d_hz_dx = _bloch_backward_diff(
        hz_complex,
        axis=0,
        inv_delta=solver.inv_dx_e,
        phase_cos=float(solver.boundary_phase_cos[0]),
        phase_sin=float(solver.boundary_phase_sin[0]),
    )
    ey_complex = previous_ey * solver.cey_decay + ey_curl * (d_hx_dz - d_hz_dx)
    ey_complex = _apply_complex_axis_boundary(
        ey_complex,
        previous_ey,
        axis=2,
        low_mode=solver.boundary_z_low_code,
        high_mode=solver.boundary_z_high_code,
    )
    psi_ey_z = torch.complex(state["psi_ey_z"], state["psi_ey_z_imag"])
    ey_complex, psi_ey_z = _apply_complex_cpml_correction(
        ey_complex,
        psi_ey_z,
        d_hx_dz,
        curl=ey_curl,
        inv_kappa=solver.cpml_inv_kappa_e_z,
        b=solver.cpml_b_e_z,
        c=solver.cpml_c_e_z,
        normal_axis=2,
        tangent_axis=0,
        tangent_low_mode=solver.boundary_x_low_code,
        tangent_high_mode=solver.boundary_x_high_code,
        sign=1.0,
    )

    previous_ez = _complex_state_field(state, "Ez")
    d_hy_dx = _bloch_backward_diff(
        hy_complex,
        axis=0,
        inv_delta=solver.inv_dx_e,
        phase_cos=float(solver.boundary_phase_cos[0]),
        phase_sin=float(solver.boundary_phase_sin[0]),
    )
    d_hx_dy = _bloch_backward_diff(
        hx_complex,
        axis=1,
        inv_delta=solver.inv_dy_e,
        phase_cos=float(solver.boundary_phase_cos[1]),
        phase_sin=float(solver.boundary_phase_sin[1]),
    )
    ez_complex = previous_ez * solver.cez_decay + ez_curl * (d_hy_dx - d_hx_dy)

    electric_fields = {
        "Ex": ex_complex.real,
        "Ey": ey_complex.real,
        "Ez": ez_complex.real,
        "Ex_imag": ex_complex.imag,
        "Ey_imag": ey_complex.imag,
        "Ez_imag": ez_complex.imag,
    }
    electric_cpml_state = {
        "psi_ex_y": state["psi_ex_y"],
        "psi_ex_z": psi_ex_z.real,
        "psi_ey_x": state["psi_ey_x"],
        "psi_ey_z": psi_ey_z.real,
        "psi_ez_x": state["psi_ez_x"],
        "psi_ez_y": state["psi_ez_y"],
        "psi_ex_y_imag": state["psi_ex_y_imag"],
        "psi_ex_z_imag": psi_ex_z.imag,
        "psi_ey_x_imag": state["psi_ey_x_imag"],
        "psi_ey_z_imag": psi_ey_z.imag,
        "psi_ez_x_imag": state["psi_ez_x_imag"],
        "psi_ez_y_imag": state["psi_ez_y_imag"],
    }
    return electric_fields, electric_cpml_state


def _update_general_bloch_cpml_electric_fields(
    solver, state, magnetic_fields, *, pml_axis, ex_curl, ey_curl, ez_curl
):
    """Reverse replay of the general single-PML-axis (x or y) Bloch+CPML electric
    update. The forward runs the full-Bloch base (``update_electric_fields_bloch``)
    with the true wrap phase on the two periodic axes and the natural unit phase on
    the absorbing axis, then folds the recursive-convolution stretch into the two
    components with an absorbing-axis derivative. This mirrors that step exactly so
    the field, eps, and source VJPs stay consistent with the forward kernels."""
    hx = _complex_state_field(magnetic_fields, "Hx")
    hy = _complex_state_field(magnetic_fields, "Hy")
    hz = _complex_state_field(magnetic_fields, "Hz")
    phase_cos = solver.boundary_phase_cos
    phase_sin = solver.boundary_phase_sin

    def _bdiff(field, axis, inv_delta):
        return _bloch_backward_diff(
            field,
            axis=axis,
            inv_delta=inv_delta,
            phase_cos=float(phase_cos[axis]),
            phase_sin=float(phase_sin[axis]),
        )

    d_hz_dy = _bdiff(hz, 1, solver.inv_dy_e)
    d_hy_dz = _bdiff(hy, 2, solver.inv_dz_e)
    ex = _complex_state_field(state, "Ex") * solver.cex_decay + ex_curl * (d_hz_dy - d_hy_dz)

    d_hx_dz = _bdiff(hx, 2, solver.inv_dz_e)
    d_hz_dx = _bdiff(hz, 0, solver.inv_dx_e)
    ey = _complex_state_field(state, "Ey") * solver.cey_decay + ey_curl * (d_hx_dz - d_hz_dx)

    d_hy_dx = _bdiff(hy, 0, solver.inv_dx_e)
    d_hx_dy = _bdiff(hx, 1, solver.inv_dy_e)
    ez = _complex_state_field(state, "Ez") * solver.cez_decay + ez_curl * (d_hy_dx - d_hx_dy)

    psi = {
        name: torch.complex(state[name], state[f"{name}_imag"])
        for name in ("psi_ex_y", "psi_ex_z", "psi_ey_x", "psi_ey_z", "psi_ez_x", "psi_ez_y")
    }

    if pml_axis == "y":
        ex, psi["psi_ex_y"] = _apply_complex_cpml_correction(
            ex,
            psi["psi_ex_y"],
            d_hz_dy,
            curl=ex_curl,
            inv_kappa=solver.cpml_inv_kappa_e_y,
            b=solver.cpml_b_e_y,
            c=solver.cpml_c_e_y,
            normal_axis=1,
            tangent_axis=2,
            tangent_low_mode=solver.boundary_z_low_code,
            tangent_high_mode=solver.boundary_z_high_code,
            sign=1.0,
        )
        ez, psi["psi_ez_y"] = _apply_complex_cpml_correction(
            ez,
            psi["psi_ez_y"],
            d_hx_dy,
            curl=ez_curl,
            inv_kappa=solver.cpml_inv_kappa_e_y,
            b=solver.cpml_b_e_y,
            c=solver.cpml_c_e_y,
            normal_axis=1,
            tangent_axis=0,
            tangent_low_mode=solver.boundary_x_low_code,
            tangent_high_mode=solver.boundary_x_high_code,
            sign=-1.0,
        )
    else:  # pml_axis == "x"
        ey, psi["psi_ey_x"] = _apply_complex_cpml_correction(
            ey,
            psi["psi_ey_x"],
            d_hz_dx,
            curl=ey_curl,
            inv_kappa=solver.cpml_inv_kappa_e_x,
            b=solver.cpml_b_e_x,
            c=solver.cpml_c_e_x,
            normal_axis=0,
            tangent_axis=2,
            tangent_low_mode=solver.boundary_z_low_code,
            tangent_high_mode=solver.boundary_z_high_code,
            sign=-1.0,
        )
        ez, psi["psi_ez_x"] = _apply_complex_cpml_correction(
            ez,
            psi["psi_ez_x"],
            d_hy_dx,
            curl=ez_curl,
            inv_kappa=solver.cpml_inv_kappa_e_x,
            b=solver.cpml_b_e_x,
            c=solver.cpml_c_e_x,
            normal_axis=0,
            tangent_axis=1,
            tangent_low_mode=solver.boundary_y_low_code,
            tangent_high_mode=solver.boundary_y_high_code,
            sign=1.0,
        )

    electric_fields = {
        "Ex": ex.real,
        "Ey": ey.real,
        "Ez": ez.real,
        "Ex_imag": ex.imag,
        "Ey_imag": ey.imag,
        "Ez_imag": ez.imag,
    }
    electric_cpml_state = {}
    for name, value in psi.items():
        electric_cpml_state[name] = value.real
        electric_cpml_state[f"{name}_imag"] = value.imag
    return electric_fields, electric_cpml_state


def _clamped_center_pair_sum(field: torch.Tensor, axis: int) -> torch.Tensor:
    """Sum of ``field[clamp(i-1)] + field[clamp(i)]`` for ``i in 0..size`` along ``axis``.

    Replicates the index-clamped two-point gather of the CUDA collocation helper:
    the output is one element longer than the input along ``axis``.
    """
    first = field.narrow(axis, 0, 1)
    last = field.narrow(axis, field.shape[axis] - 1, 1)
    padded = torch.cat([first, field, last], dim=axis)
    length = field.shape[axis] + 1
    return padded.narrow(axis, 0, length) + padded.narrow(axis, 1, length)


def _forward_pair_sum(field: torch.Tensor, axis: int) -> torch.Tensor:
    """Sum of ``field[i] + field[i+1]`` for ``i in 0..size-2`` along ``axis``."""
    length = field.shape[axis] - 1
    return field.narrow(axis, 0, length) + field.narrow(axis, 1, length)


def _collocated_field_square(ex, ey, ez):
    """|E|^2 collocated onto each electric Yee edge, matching the CUDA
    ``collocate_electric_components`` stencil (4-point off-axis average with
    index clamping at the domain faces)."""
    ey_on_ex = 0.25 * _forward_pair_sum(_clamped_center_pair_sum(ey, axis=1), axis=0)
    ez_on_ex = 0.25 * _forward_pair_sum(_clamped_center_pair_sum(ez, axis=2), axis=0)
    ex_on_ey = 0.25 * _forward_pair_sum(_clamped_center_pair_sum(ex, axis=0), axis=1)
    ez_on_ey = 0.25 * _forward_pair_sum(_clamped_center_pair_sum(ez, axis=2), axis=1)
    ex_on_ez = 0.25 * _forward_pair_sum(_clamped_center_pair_sum(ex, axis=0), axis=2)
    ey_on_ez = 0.25 * _forward_pair_sum(_clamped_center_pair_sum(ey, axis=1), axis=2)
    return {
        "Ex": ex * ex + ey_on_ex * ey_on_ex + ez_on_ex * ez_on_ex,
        "Ey": ex_on_ey * ex_on_ey + ey * ey + ez_on_ey * ez_on_ey,
        "Ez": ex_on_ez * ex_on_ez + ey_on_ez * ey_on_ez + ez * ez,
    }


def _kerr_dynamic_electric_curls(solver, state, *, eps_ex, eps_ey, eps_ez, chi3_ex, chi3_ey, chi3_ez):
    """Differentiable replica of ``updateKerrElectricField*Curl3D``:
    ``curl = dt / max(eps + eps0 * chi3 * |E|^2, floor) * decay``."""
    field_square = _collocated_field_square(state["Ex"], state["Ey"], state["Ez"])
    eps0 = float(solver.eps0)
    floor = 1.0e-12 * eps0
    dt = float(solver.dt)
    curls = {}
    for component_name, eps, chi3, decay in (
        ("Ex", eps_ex, chi3_ex, solver.cex_decay),
        ("Ey", eps_ey, chi3_ey, solver.cey_decay),
        ("Ez", eps_ez, chi3_ez, solver.cez_decay),
    ):
        effective = torch.clamp_min(eps + eps0 * chi3 * field_square[component_name], floor)
        curls[component_name] = (dt / effective) * decay
    return curls


def _conductive_electric_coefficients(solver, *, eps_ex, eps_ey, eps_ez):
    """Differentiable replica of the semi-implicit lossy ``_electric_update_coefficients``.

    The forward bakes ``decay = (1 - h)/(1 + h)`` and ``curl = (dt/eps)/(1 + h)``
    (with ``h = 0.5*sigma_e*dt/eps``, times an eps-independent PML/PEC factor)
    into ``c*_decay`` / ``c*_curl``. Both carry an ``eps`` dependence that the
    linear-dielectric reverse rule drops, so recompute them from the
    differentiable ``eps`` leaves and the frozen per-edge conductivity. The
    eps-independent factor (PML split-field decay and any PEC open fraction) is
    recovered from the frozen forward curl so those scalings are preserved
    exactly. Returns per-component ``(decay, curl)`` where ``curl`` is the full
    curl coefficient (fed with ``eps = 1`` in the update, mirroring the Kerr
    dynamic-curl path)."""
    dt = float(solver.dt)
    coeffs = {}
    for name, eps, eps_fwd, sigma, curl_fwd in (
        ("Ex", eps_ex, solver.eps_Ex, solver.sigma_e_Ex, solver.cex_curl),
        ("Ey", eps_ey, solver.eps_Ey, solver.sigma_e_Ey, solver.cey_curl),
        ("Ez", eps_ez, solver.eps_Ez, solver.sigma_e_Ez, solver.cez_curl),
    ):
        half_fwd = 0.5 * sigma * dt / eps_fwd
        # curl_fwd = (dt / eps_fwd) / (1 + half_fwd) * factor  ->  solve for factor.
        factor = curl_fwd * eps_fwd * (1.0 + half_fwd) / dt
        half = 0.5 * sigma * dt / eps
        denom = 1.0 + half
        decay = ((1.0 - half) / denom) * factor
        curl = (dt / eps / denom) * factor
        coeffs[name] = (decay, curl)
    return coeffs


def _conductive_reverse_coefficients(solver, *, eps_ex, eps_ey, eps_ez):
    """Per-component ``(decay, curl, half)`` for the static-conductive reverse.

    Identical bake to :func:`_conductive_electric_coefficients` but also returns
    the semi-implicit ``half = 0.5*sigma_e*dt/eps`` (recomputed from the
    differentiable ``eps`` leaf) so the reverse can form the analytic
    ``d(decay)/d(eps)`` and ``d(curl)/d(eps)`` sensitivities without an autograd
    VJP: ``d(decay)/d(eps) = 2*half*curl/(dt*(1+half))`` and
    ``d(curl)/d(eps) = -curl/(eps*(1+half))``. The eps-independent PML/PEC
    ``factor`` is recovered from the frozen forward curl so those scalings are
    preserved exactly."""
    dt = float(solver.dt)
    coeffs = {}
    for name, eps, eps_fwd, sigma, curl_fwd in (
        ("Ex", eps_ex, solver.eps_Ex, solver.sigma_e_Ex, solver.cex_curl),
        ("Ey", eps_ey, solver.eps_Ey, solver.sigma_e_Ey, solver.cey_curl),
        ("Ez", eps_ez, solver.eps_Ez, solver.sigma_e_Ez, solver.cez_curl),
    ):
        half_fwd = 0.5 * sigma * dt / eps_fwd
        factor = curl_fwd * eps_fwd * (1.0 + half_fwd) / dt
        half = 0.5 * sigma * dt / eps
        denom = 1.0 + half
        decay = ((1.0 - half) / denom) * factor
        curl = (dt / eps / denom) * factor
        coeffs[name] = (decay, curl, half)
    return coeffs


def _reverse_electric_component_cpml_conductive(
    adj_updated,
    adj_psi_pos_post,
    adj_psi_neg_post,
    field,
    *,
    d_pos,
    d_neg,
    decay,
    curl,
    half,
    eps,
    dt,
    low_mode_pos,
    high_mode_pos,
    low_mode_neg,
    high_mode_neg,
    axis_pos,
    axis_neg,
    psi_pos,
    psi_neg,
    b_pos,
    c_pos,
    inv_kappa_pos,
    b_neg,
    c_neg,
    inv_kappa_neg,
):
    """Static-conductive analytic reverse of the semi-implicit CPML electric update.

    Structurally identical to :func:`_reverse_electric_component_cpml` (the psi
    recursion, ``adj_d_pos``/``adj_d_neg`` folds, and pre-step field pullback all
    reuse the CPML stretch unchanged, because the conductive update only rescales
    the ``decay``/``curl`` coefficient pair). The one difference is the eps
    gradient: the semi-implicit ``decay`` and ``curl`` both depend on ``eps``
    through the loss denominator, so the linear rule ``-curl*curl_term/eps^2`` is
    replaced by the exact
    ``adj_updated*(E_prev*d(decay)/d(eps) + curl_term*d(curl)/d(eps))``."""
    inactive_pos, pec_pos = _boundary_axis_masks(field.shape, axis_pos, low_mode_pos, high_mode_pos, field.device)
    inactive_neg, pec_neg = _boundary_axis_masks(field.shape, axis_neg, low_mode_neg, high_mode_neg, field.device)
    pec_mask = pec_pos | pec_neg
    inactive_mask = (~pec_mask) & (inactive_pos | inactive_neg)
    active_mask = (~pec_mask) & (~inactive_mask)
    keep_mask = inactive_mask | pec_mask

    active = active_mask.to(device=field.device, dtype=field.dtype)
    inactive = inactive_mask.to(device=field.device, dtype=field.dtype)
    keep = keep_mask.to(device=field.device, dtype=field.dtype)

    b_pos_term = _broadcast_vector(b_pos, axis_pos)
    c_pos_term = _broadcast_vector(c_pos, axis_pos)
    inv_kappa_pos_term = _broadcast_vector(inv_kappa_pos, axis_pos)
    b_neg_term = _broadcast_vector(b_neg, axis_neg)
    c_neg_term = _broadcast_vector(c_neg, axis_neg)
    inv_kappa_neg_term = _broadcast_vector(inv_kappa_neg, axis_neg)

    psi_pos_candidate = b_pos_term * psi_pos + c_pos_term * d_pos
    psi_neg_candidate = b_neg_term * psi_neg + c_neg_term * d_neg
    curl_term = (d_pos * inv_kappa_pos_term + psi_pos_candidate) - (d_neg * inv_kappa_neg_term + psi_neg_candidate)

    denom = 1.0 + half
    d_decay_d_eps = 2.0 * half * curl / (float(dt) * denom)
    d_curl_d_eps = -curl / (eps * denom)

    adj_field = inactive * adj_updated + active * (adj_updated * decay)
    adj_curl_term = active * adj_updated * curl
    grad_eps = active * adj_updated * (field * d_decay_d_eps + curl_term * d_curl_d_eps)
    adj_psi_pos_candidate = active * adj_psi_pos_post + adj_curl_term
    adj_psi_neg_candidate = active * adj_psi_neg_post - adj_curl_term
    adj_psi_pos = keep * adj_psi_pos_post + active * (b_pos_term * adj_psi_pos_candidate)
    adj_psi_neg = keep * adj_psi_neg_post + active * (b_neg_term * adj_psi_neg_candidate)
    adj_d_pos = inv_kappa_pos_term * adj_curl_term + c_pos_term * adj_psi_pos_candidate
    adj_d_neg = -inv_kappa_neg_term * adj_curl_term + c_neg_term * adj_psi_neg_candidate
    return adj_field, adj_d_pos, adj_d_neg, grad_eps, adj_psi_pos, adj_psi_neg


def _reverse_electric_component_cpml_nonlinear(
    adj_updated,
    adj_psi_pos_post,
    adj_psi_neg_post,
    field,
    *,
    d_pos,
    d_neg,
    decay,
    curl,
    low_mode_pos,
    high_mode_pos,
    low_mode_neg,
    high_mode_neg,
    axis_pos,
    axis_neg,
    psi_pos,
    psi_neg,
    b_pos,
    c_pos,
    inv_kappa_pos,
    b_neg,
    c_neg,
    inv_kappa_neg,
):
    """Reverse of the CPML electric update with GIVEN per-edge decay/curl pair.

    Unlike :func:`_reverse_electric_component_cpml` (which bakes the linear
    ``curl_prefactor/eps`` rule and returns ``grad_eps``), this returns the raw
    cotangents ``adj_decay`` and ``adj_curl`` to the field-dependent coefficient
    pair so a nonlinear coefficient reverse (Kerr / general) can push them onto the
    ``eps`` / ``chi2`` / ``chi3`` / ``tpa`` leaves and (via the collocation
    transpose) the pre-step fields. The psi recursion, the ``adj_d_pos``/
    ``adj_d_neg`` curl(H) folds, and the pre-step field pullback are the CPML
    machinery unchanged (a field-dependent coefficient only rescales the update)."""
    inactive_pos, pec_pos = _boundary_axis_masks(field.shape, axis_pos, low_mode_pos, high_mode_pos, field.device)
    inactive_neg, pec_neg = _boundary_axis_masks(field.shape, axis_neg, low_mode_neg, high_mode_neg, field.device)
    pec_mask = pec_pos | pec_neg
    inactive_mask = (~pec_mask) & (inactive_pos | inactive_neg)
    active_mask = (~pec_mask) & (~inactive_mask)
    keep_mask = inactive_mask | pec_mask

    active = active_mask.to(device=field.device, dtype=field.dtype)
    inactive = inactive_mask.to(device=field.device, dtype=field.dtype)
    keep = keep_mask.to(device=field.device, dtype=field.dtype)

    b_pos_term = _broadcast_vector(b_pos, axis_pos)
    c_pos_term = _broadcast_vector(c_pos, axis_pos)
    inv_kappa_pos_term = _broadcast_vector(inv_kappa_pos, axis_pos)
    b_neg_term = _broadcast_vector(b_neg, axis_neg)
    c_neg_term = _broadcast_vector(c_neg, axis_neg)
    inv_kappa_neg_term = _broadcast_vector(inv_kappa_neg, axis_neg)

    psi_pos_candidate = b_pos_term * psi_pos + c_pos_term * d_pos
    psi_neg_candidate = b_neg_term * psi_neg + c_neg_term * d_neg
    curl_term = (d_pos * inv_kappa_pos_term + psi_pos_candidate) - (d_neg * inv_kappa_neg_term + psi_neg_candidate)

    adj_field = inactive * adj_updated + active * (adj_updated * decay)
    adj_decay = active * (adj_updated * field)
    adj_curl = active * (adj_updated * curl_term)
    adj_curl_term = active * (adj_updated * curl)
    adj_psi_pos_candidate = active * adj_psi_pos_post + adj_curl_term
    adj_psi_neg_candidate = active * adj_psi_neg_post - adj_curl_term
    adj_psi_pos = keep * adj_psi_pos_post + active * (b_pos_term * adj_psi_pos_candidate)
    adj_psi_neg = keep * adj_psi_neg_post + active * (b_neg_term * adj_psi_neg_candidate)
    adj_d_pos = inv_kappa_pos_term * adj_curl_term + c_pos_term * adj_psi_pos_candidate
    adj_d_neg = -inv_kappa_neg_term * adj_curl_term + c_neg_term * adj_psi_neg_candidate
    return adj_field, adj_d_pos, adj_d_neg, adj_decay, adj_curl, adj_psi_pos, adj_psi_neg


def _kerr_reverse_coefficients(solver, state, *, eps_ex, eps_ey, eps_ez, chi3_ex, chi3_ey, chi3_ez):
    """Per-component ``(decay, curl, eff, fsq, clamp_mask)`` for the Kerr reverse.

    Mirrors :func:`_kerr_dynamic_electric_curls` (the differentiable forward
    replica already parity-tested against ``updateKerrElectricField*Curl3D``) but
    also returns the effective permittivity ``eff``, the collocated ``|E|^2``, and
    the ``clamp_min`` gradient mask so the reverse can form the analytic coefficient
    sensitivities without an autograd VJP. ``decay`` is the frozen linear PML decay
    (constant in the Kerr update); the eps/chi3/field dependence lives entirely in
    ``curl = (dt / eff) * decay`` through ``eff = eps + eps0 * chi3 * |E|^2``."""
    field_square = _collocated_field_square(state["Ex"], state["Ey"], state["Ez"])
    eps0 = float(solver.eps0)
    floor = 1.0e-12 * eps0
    dt = float(solver.dt)
    coeffs = {}
    for name, eps, chi3, decay in (
        ("Ex", eps_ex, chi3_ex, solver.cex_decay),
        ("Ey", eps_ey, chi3_ey, solver.cey_decay),
        ("Ez", eps_ez, chi3_ez, solver.cez_decay),
    ):
        fsq = field_square[name]
        raw = eps + eps0 * chi3 * fsq
        eff = torch.clamp_min(raw, floor)
        clamp_mask = (raw >= floor).to(dtype=eff.dtype)
        curl = (dt / eff) * decay
        coeffs[name] = (decay, curl, eff, fsq, clamp_mask)
    return coeffs


def _general_nonlinear_electric_coefficients(
    solver,
    state,
    *,
    eps_ex,
    eps_ey,
    eps_ez,
    chi2_ex,
    chi2_ey,
    chi2_ez,
    chi3_ex,
    chi3_ey,
    chi3_ez,
    tpa_ex,
    tpa_ey,
    tpa_ez,
):
    """Differentiable replica of ``updateNonlinearElectricCoefficients3D``.

    The general nonlinear kernel rewrites both the decay and the curl coefficient
    every step from the pre-update fields, exactly as in the CUDA source::

        eps_eff = max(eps_lin + eps0 * (chi2 * E_own + chi3 * |E|^2), floor)
        sigma   = max(sigma_static + tpa_sigma * |E|^2, 0)
        half    = 0.5 * sigma * dt / eps_eff
        decay   = external * (1 - half) / (1 + half)
        curl    = external * (dt / eps_eff) / (1 + half)

    ``external`` (``c*_decay_external``) carries the PML split-field decay and the
    PEC open fraction. ``E_own`` is the component's own edge field (the CUDA
    collocation returns the direct value for the own axis), while ``|E|^2`` is the
    4-point off-axis collocation shared with the Kerr replica. Flowing the VJP
    through the ``eps`` / ``chi2`` / ``chi3`` / ``tpa`` leaves and the pre-step
    ``state`` fields is what makes chi2 and two-photon absorption differentiable.
    Returns per-component ``(decay, curl)`` with ``dt / eps_eff`` already folded
    into ``curl`` (fed with ``eps = 1`` in the update, like the Kerr path).
    """
    field_square = _collocated_field_square(state["Ex"], state["Ey"], state["Ez"])
    eps0 = float(solver.eps0)
    floor = 1.0e-12 * eps0
    dt = float(solver.dt)
    coeffs = {}
    for name, own_field, eps, chi2, chi3, tpa, sigma_static, external in (
        ("Ex", state["Ex"], eps_ex, chi2_ex, chi3_ex, tpa_ex, solver.sigma_e_Ex, solver.cex_decay_external),
        ("Ey", state["Ey"], eps_ey, chi2_ey, chi3_ey, tpa_ey, solver.sigma_e_Ey, solver.cey_decay_external),
        ("Ez", state["Ez"], eps_ez, chi2_ez, chi3_ez, tpa_ez, solver.sigma_e_Ez, solver.cez_decay_external),
    ):
        effective = torch.clamp_min(
            eps + eps0 * (chi2 * own_field + chi3 * field_square[name]), floor
        )
        sigma = torch.clamp_min(sigma_static + tpa * field_square[name], 0.0)
        half = 0.5 * sigma * dt / effective
        inv_denom = 1.0 / (1.0 + half)
        decay = external * (1.0 - half) * inv_denom
        curl = external * (dt / effective) * inv_denom
        coeffs[name] = (decay, curl)
    return coeffs


def _forward_magnetic_fields(solver, state, *, time_value, resolved_source_terms):
    """Recompute the post-source real magnetic fields of one forward step.

    Mirrors the magnetic half of ``_step_state`` for real fields (standard or
    CPML, selected by the solver's psi/coefficient attributes) and applies the
    resolved magnetic source terms. Used by the reverse steps, which need the
    same mid-step H the electric update consumed.
    """
    uses_cpml = getattr(solver, "uses_cpml", False)

    d_ez_dy = _forward_diff(state["Ez"], axis=1, inv_delta=solver.inv_dy_h)
    d_ey_dz = _forward_diff(state["Ey"], axis=2, inv_delta=solver.inv_dz_h)
    hx, _, _ = _update_magnetic_component(
        state["Hx"],
        d_pos=d_ez_dy,
        d_neg=d_ey_dz,
        decay=solver.chx_decay,
        curl=solver.chx_curl,
        psi_pos=state.get("psi_hx_y") if uses_cpml else None,
        psi_neg=state.get("psi_hx_z") if uses_cpml else None,
        b_pos=getattr(solver, "cpml_b_h_y", None),
        c_pos=getattr(solver, "cpml_c_h_y", None),
        inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_y", None),
        b_neg=getattr(solver, "cpml_b_h_z", None),
        c_neg=getattr(solver, "cpml_c_h_z", None),
        inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_z", None),
        axis_pos=1,
        axis_neg=2,
    )

    d_ex_dz = _forward_diff(state["Ex"], axis=2, inv_delta=solver.inv_dz_h)
    d_ez_dx = _forward_diff(state["Ez"], axis=0, inv_delta=solver.inv_dx_h)
    hy, _, _ = _update_magnetic_component(
        state["Hy"],
        d_pos=d_ex_dz,
        d_neg=d_ez_dx,
        decay=solver.chy_decay,
        curl=solver.chy_curl,
        psi_pos=state.get("psi_hy_z") if uses_cpml else None,
        psi_neg=state.get("psi_hy_x") if uses_cpml else None,
        b_pos=getattr(solver, "cpml_b_h_z", None),
        c_pos=getattr(solver, "cpml_c_h_z", None),
        inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_z", None),
        b_neg=getattr(solver, "cpml_b_h_x", None),
        c_neg=getattr(solver, "cpml_c_h_x", None),
        inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_x", None),
        axis_pos=2,
        axis_neg=0,
    )

    d_ey_dx = _forward_diff(state["Ey"], axis=0, inv_delta=solver.inv_dx_h)
    d_ex_dy = _forward_diff(state["Ex"], axis=1, inv_delta=solver.inv_dy_h)
    hz, _, _ = _update_magnetic_component(
        state["Hz"],
        d_pos=d_ey_dx,
        d_neg=d_ex_dy,
        decay=solver.chz_decay,
        curl=solver.chz_curl,
        psi_pos=state.get("psi_hz_x") if uses_cpml else None,
        psi_neg=state.get("psi_hz_y") if uses_cpml else None,
        b_pos=getattr(solver, "cpml_b_h_x", None),
        c_pos=getattr(solver, "cpml_c_h_x", None),
        inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_x", None),
        b_neg=getattr(solver, "cpml_b_h_y", None),
        c_neg=getattr(solver, "cpml_c_h_y", None),
        inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_y", None),
        axis_pos=0,
        axis_neg=1,
    )

    return _apply_resolved_magnetic_source_terms(
        {"Hx": hx, "Hy": hy, "Hz": hz},
        solver=solver,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )


def _forward_magnetic_fields_complex(solver, state, *, time_value, resolved_source_terms):
    """Recompute the post-source complex (split-field) magnetic fields of one Bloch step.

    Mirrors the magnetic half of the split-field Bloch ``_step_state``: the real and
    imaginary field halves each run the plain (non-CPML) magnetic update with the
    shared decay/curl coefficients, and the resolved magnetic source terms apply once
    to the combined six-key dict. Both the analytic Bloch reference and the native
    Bloch reverse runner consume this to recover the mid-step H the electric update
    read (the reverse *math* is native; this replay stays Torch, same bar as the
    standard/CPML runners).
    """

    def _magnetic_half(suffix):
        ex = state["Ex" + suffix]
        ey = state["Ey" + suffix]
        ez = state["Ez" + suffix]
        d_ez_dy = _forward_diff(ez, axis=1, inv_delta=solver.inv_dy_h)
        d_ey_dz = _forward_diff(ey, axis=2, inv_delta=solver.inv_dz_h)
        hx, _, _ = _update_magnetic_component(
            state["Hx" + suffix],
            d_pos=d_ez_dy,
            d_neg=d_ey_dz,
            decay=solver.chx_decay,
            curl=solver.chx_curl,
            axis_pos=1,
            axis_neg=2,
        )
        d_ex_dz = _forward_diff(ex, axis=2, inv_delta=solver.inv_dz_h)
        d_ez_dx = _forward_diff(ez, axis=0, inv_delta=solver.inv_dx_h)
        hy, _, _ = _update_magnetic_component(
            state["Hy" + suffix],
            d_pos=d_ex_dz,
            d_neg=d_ez_dx,
            decay=solver.chy_decay,
            curl=solver.chy_curl,
            axis_pos=2,
            axis_neg=0,
        )
        d_ey_dx = _forward_diff(ey, axis=0, inv_delta=solver.inv_dx_h)
        d_ex_dy = _forward_diff(ex, axis=1, inv_delta=solver.inv_dy_h)
        hz, _, _ = _update_magnetic_component(
            state["Hz" + suffix],
            d_pos=d_ey_dx,
            d_neg=d_ex_dy,
            decay=solver.chz_decay,
            curl=solver.chz_curl,
            axis_pos=0,
            axis_neg=1,
        )
        return hx, hy, hz

    hx, hy, hz = _magnetic_half("")
    hx_imag, hy_imag, hz_imag = _magnetic_half("_imag")
    magnetic_fields = {
        "Hx": hx,
        "Hy": hy,
        "Hz": hz,
        "Hx_imag": hx_imag,
        "Hy_imag": hy_imag,
        "Hz_imag": hz_imag,
    }
    return _apply_resolved_magnetic_source_terms(
        magnetic_fields,
        solver=solver,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )


def _sum_adjacent(field: torch.Tensor, axis: int) -> torch.Tensor:
    """Sum of each adjacent pair along ``axis`` (length shrinks by one)."""
    lo = [slice(None)] * field.ndim
    hi = [slice(None)] * field.ndim
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    return field[tuple(lo)] + field[tuple(hi)]


def _spread_to_nodes(edge: torch.Tensor, axis: int) -> torch.Tensor:
    """Transpose of two-point averaging along ``axis`` (length grows by one).

    Scatters each edge value onto its two bounding node positions with zero-fill
    at the ends, matching the four-neighbor collocation the full-anisotropy
    kernel performs when it maps an off-axis curl(H) sample onto the target edge
    (out-of-range neighbours contribute zero while the fixed 1/4 weight stays).
    """
    zero_shape = [1 if i == axis else size for i, size in enumerate(edge.shape)]
    zero = edge.new_zeros(zero_shape)
    below = torch.cat((edge, zero), dim=axis)
    above = torch.cat((zero, edge), dim=axis)
    return below + above


def _collocate_curl(curl: torch.Tensor, *, edge_axis: int, curl_axis: int) -> torch.Tensor:
    """Neighbor-average an off-axis curl(H) component onto a target E edge.

    ``edge_axis`` is the Yee-edge axis of the *target* E component (summed over
    the two straddling cells) and ``curl_axis`` is the edge axis of the source
    curl component (spread onto the target's two bounding nodes).
    """
    return _spread_to_nodes(_sum_adjacent(curl, edge_axis), curl_axis)


def _full_aniso_electric_correction(solver, magnetic_fields):
    """Off-diagonal (full-tensor) anisotropic coupling added to the E update.

    Differentiable Torch replica of ``apply_full_aniso_corrections`` (the native
    ``updateElectricFieldE{x,y,z}FullAniso3D`` kernels): each off-axis curl(H)
    component is four-neighbor-averaged onto the target Yee edge and scaled by
    the off-diagonal inverse-permittivity coefficient. It consumes the same
    post-source magnetic fields the diagonal update used, so the adjoint VJP
    flows the cotangent through the coupling into the H (and thence E) fields.
    """
    hx = magnetic_fields["Hx"]
    hy = magnetic_fields["Hy"]
    hz = magnetic_fields["Hz"]
    # curl(H) on the native Yee E edges, identical backward differences to the
    # diagonal update (curl_x -> Ex edge, curl_y -> Ey edge, curl_z -> Ez edge).
    curl_x = _backward_diff(hz, axis=1, inv_delta=solver.inv_dy_e) - _backward_diff(hy, axis=2, inv_delta=solver.inv_dz_e)
    curl_y = _backward_diff(hx, axis=2, inv_delta=solver.inv_dz_e) - _backward_diff(hz, axis=0, inv_delta=solver.inv_dx_e)
    curl_z = _backward_diff(hy, axis=0, inv_delta=solver.inv_dx_e) - _backward_diff(hx, axis=1, inv_delta=solver.inv_dy_e)

    ex = 0.25 * (
        solver.cex_aniso_y * _collocate_curl(curl_y, edge_axis=0, curl_axis=1)
        + solver.cex_aniso_z * _collocate_curl(curl_z, edge_axis=0, curl_axis=2)
    )
    ey = 0.25 * (
        solver.cey_aniso_x * _collocate_curl(curl_x, edge_axis=1, curl_axis=0)
        + solver.cey_aniso_z * _collocate_curl(curl_z, edge_axis=1, curl_axis=2)
    )
    ez = 0.25 * (
        solver.cez_aniso_x * _collocate_curl(curl_x, edge_axis=2, curl_axis=0)
        + solver.cez_aniso_y * _collocate_curl(curl_y, edge_axis=2, curl_axis=1)
    )
    return {"Ex": ex, "Ey": ey, "Ez": ez}


def _step_state(
    solver,
    state,
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
    capture_magnetic=None,
):
    if getattr(solver, "modulation_enabled", False):
        raise NotImplementedError("FDTD adjoint replay does not support time-modulated media.")
    source_terms, electric_source_terms, magnetic_source_terms = _resolved_source_term_lists(
        solver,
        eps_ex,
        eps_ey,
        eps_ez,
    )
    auxiliary_state = _extract_tfsf_auxiliary_state(state)
    complex_fields = has_complex_fields(solver)
    psi_hx_y_imag = psi_hx_z_imag = psi_hy_x_imag = psi_hy_z_imag = psi_hz_x_imag = psi_hz_y_imag = None

    d_ez_dy = _forward_diff(state["Ez"], axis=1, inv_delta=solver.inv_dy_h)
    d_ey_dz = _forward_diff(state["Ey"], axis=2, inv_delta=solver.inv_dz_h)
    hx, psi_hx_y, psi_hx_z = _update_magnetic_component(
        state["Hx"],
        d_pos=d_ez_dy,
        d_neg=d_ey_dz,
        decay=solver.chx_decay,
        curl=solver.chx_curl,
        psi_pos=state.get("psi_hx_y"),
        psi_neg=state.get("psi_hx_z"),
        b_pos=getattr(solver, "cpml_b_h_y", None),
        c_pos=getattr(solver, "cpml_c_h_y", None),
        inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_y", None),
        b_neg=getattr(solver, "cpml_b_h_z", None),
        c_neg=getattr(solver, "cpml_c_h_z", None),
        inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_z", None),
        axis_pos=1,
        axis_neg=2,
    )

    d_ex_dz = _forward_diff(state["Ex"], axis=2, inv_delta=solver.inv_dz_h)
    d_ez_dx = _forward_diff(state["Ez"], axis=0, inv_delta=solver.inv_dx_h)
    hy, psi_hy_x, psi_hy_z = _update_magnetic_component(
        state["Hy"],
        d_pos=d_ex_dz,
        d_neg=d_ez_dx,
        decay=solver.chy_decay,
        curl=solver.chy_curl,
        psi_pos=state.get("psi_hy_z"),
        psi_neg=state.get("psi_hy_x"),
        b_pos=getattr(solver, "cpml_b_h_z", None),
        c_pos=getattr(solver, "cpml_c_h_z", None),
        inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_z", None),
        b_neg=getattr(solver, "cpml_b_h_x", None),
        c_neg=getattr(solver, "cpml_c_h_x", None),
        inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_x", None),
        axis_pos=2,
        axis_neg=0,
    )

    d_ey_dx = _forward_diff(state["Ey"], axis=0, inv_delta=solver.inv_dx_h)
    d_ex_dy = _forward_diff(state["Ex"], axis=1, inv_delta=solver.inv_dy_h)
    hz, psi_hz_x, psi_hz_y = _update_magnetic_component(
        state["Hz"],
        d_pos=d_ey_dx,
        d_neg=d_ex_dy,
        decay=solver.chz_decay,
        curl=solver.chz_curl,
        psi_pos=state.get("psi_hz_x"),
        psi_neg=state.get("psi_hz_y"),
        b_pos=getattr(solver, "cpml_b_h_x", None),
        c_pos=getattr(solver, "cpml_c_h_x", None),
        inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_x", None),
        b_neg=getattr(solver, "cpml_b_h_y", None),
        c_neg=getattr(solver, "cpml_c_h_y", None),
        inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_y", None),
        axis_pos=0,
        axis_neg=1,
    )

    magnetic_fields = {
        "Hx": hx,
        "Hy": hy,
        "Hz": hz,
    }
    if complex_fields:
        d_ez_imag_dy = _forward_diff(state["Ez_imag"], axis=1, inv_delta=solver.inv_dy_h)
        d_ey_imag_dz = _forward_diff(state["Ey_imag"], axis=2, inv_delta=solver.inv_dz_h)
        hx_imag, psi_hx_y_imag, psi_hx_z_imag = _update_magnetic_component(
            state["Hx_imag"],
            d_pos=d_ez_imag_dy,
            d_neg=d_ey_imag_dz,
            decay=solver.chx_decay,
            curl=solver.chx_curl,
            psi_pos=state.get("psi_hx_y_imag") if getattr(solver, "uses_cpml", False) else None,
            psi_neg=state.get("psi_hx_z_imag") if getattr(solver, "uses_cpml", False) else None,
            b_pos=getattr(solver, "cpml_b_h_y", None),
            c_pos=getattr(solver, "cpml_c_h_y", None),
            inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_y", None),
            b_neg=getattr(solver, "cpml_b_h_z", None),
            c_neg=getattr(solver, "cpml_c_h_z", None),
            inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_z", None),
            axis_pos=1,
            axis_neg=2,
        )

        d_ex_imag_dz = _forward_diff(state["Ex_imag"], axis=2, inv_delta=solver.inv_dz_h)
        d_ez_imag_dx = _forward_diff(state["Ez_imag"], axis=0, inv_delta=solver.inv_dx_h)
        hy_imag, psi_hy_z_imag, psi_hy_x_imag = _update_magnetic_component(
            state["Hy_imag"],
            d_pos=d_ex_imag_dz,
            d_neg=d_ez_imag_dx,
            decay=solver.chy_decay,
            curl=solver.chy_curl,
            psi_pos=state.get("psi_hy_z_imag") if getattr(solver, "uses_cpml", False) else None,
            psi_neg=state.get("psi_hy_x_imag") if getattr(solver, "uses_cpml", False) else None,
            b_pos=getattr(solver, "cpml_b_h_z", None),
            c_pos=getattr(solver, "cpml_c_h_z", None),
            inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_z", None),
            b_neg=getattr(solver, "cpml_b_h_x", None),
            c_neg=getattr(solver, "cpml_c_h_x", None),
            inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_x", None),
            axis_pos=2,
            axis_neg=0,
        )

        d_ey_imag_dx = _forward_diff(state["Ey_imag"], axis=0, inv_delta=solver.inv_dx_h)
        d_ex_imag_dy = _forward_diff(state["Ex_imag"], axis=1, inv_delta=solver.inv_dy_h)
        hz_imag, psi_hz_x_imag, psi_hz_y_imag = _update_magnetic_component(
            state["Hz_imag"],
            d_pos=d_ey_imag_dx,
            d_neg=d_ex_imag_dy,
            decay=solver.chz_decay,
            curl=solver.chz_curl,
            psi_pos=state.get("psi_hz_x_imag") if getattr(solver, "uses_cpml", False) else None,
            psi_neg=state.get("psi_hz_y_imag") if getattr(solver, "uses_cpml", False) else None,
            b_pos=getattr(solver, "cpml_b_h_x", None),
            c_pos=getattr(solver, "cpml_c_h_x", None),
            inv_kappa_pos=getattr(solver, "cpml_inv_kappa_h_x", None),
            b_neg=getattr(solver, "cpml_b_h_y", None),
            c_neg=getattr(solver, "cpml_c_h_y", None),
            inv_kappa_neg=getattr(solver, "cpml_inv_kappa_h_y", None),
            axis_pos=0,
            axis_neg=1,
        )
        magnetic_fields.update(
            {
                "Hx_imag": hx_imag,
                "Hy_imag": hy_imag,
                "Hz_imag": hz_imag,
            }
        )

    if getattr(solver, "tfsf_enabled", False):
        magnetic_fields = _apply_tfsf_terms(
            magnetic_fields,
            solver=solver,
            auxiliary_state=auxiliary_state,
            term_key="magnetic_terms",
            sample_kind="electric",
            time_value=time_value,
        )
        auxiliary_state = _advance_tfsf_auxiliary_magnetic_state(solver, auxiliary_state)
    else:
        magnetic_fields = _apply_source_term_list(
            magnetic_fields,
            terms=magnetic_source_terms,
            source_time=solver._source_time,
            omega=solver.source_omega,
            time_value=time_value,
            solver=None,
        )
    # Magnetic ADE state advances from the pre-step H; its correction lands on
    # the post-source H (before the electric update consumes it), matching the
    # forward per-step kernel order.
    magnetic_dispersive_state = _advance_magnetic_dispersive_state(solver, state)
    if magnetic_dispersive_state:
        if complex_fields:
            raise NotImplementedError(
                "FDTD adjoint replay does not support magnetic dispersive media with complex fields."
            )
        magnetic_fields = _apply_magnetic_dispersive_corrections(
            solver,
            magnetic_fields,
            magnetic_dispersive_state,
        )
    dispersive_state = _advance_dispersive_state(solver, state)

    if capture_magnetic is not None:
        # Post-magnetic, post-source real H that the electric update consumes this
        # step: the checkpoint replay hands it to the reverse reference backend so
        # it does not recompute the magnetic half-step (see ``_forward_magnetic_fields``).
        # The replay only requests capture for the pure real standard/CPML path
        # (see ``_replay_can_capture_mid_magnetic``), where this equals the reverse
        # recompute bit-for-bit; the tensors are never mutated after this point.
        capture_magnetic.append(
            {
                "Hx": magnetic_fields["Hx"].detach(),
                "Hy": magnetic_fields["Hy"].detach(),
                "Hz": magnetic_fields["Hz"].detach(),
            }
        )

    if complex_fields:
        ex_curl = _dynamic_electric_curl(solver.cex_curl, solver.eps_Ex, eps_ex)
        ey_curl = _dynamic_electric_curl(solver.cey_curl, solver.eps_Ey, eps_ey)
        ez_curl = _dynamic_electric_curl(solver.cez_curl, solver.eps_Ez, eps_ez)
        if getattr(solver, "uses_cpml", False):
            from ..runtime.stepping import _bloch_cpml_pml_axis

            pml_axis = _bloch_cpml_pml_axis(solver)
            if pml_axis is None:
                raise NotImplementedError(
                    "Complex-field CPML adjoint replay requires exactly one PML axis and two Bloch axes: "
                    "the reverse split-field update carries the transverse wrap phase on the two periodic "
                    "axes and the recursive-convolution stretch on the single absorbing axis, so a face "
                    "layout mixing Bloch and PML on one axis or leaving more than one absorbing axis is "
                    "not expressible on that split."
                )
            if pml_axis == "z":
                electric_fields, electric_cpml_state = _update_mixed_bloch_cpml_electric_fields(
                    solver,
                    state,
                    magnetic_fields,
                    ex_curl=ex_curl,
                    ey_curl=ey_curl,
                    ez_curl=ez_curl,
                )
            else:
                electric_fields, electric_cpml_state = _update_general_bloch_cpml_electric_fields(
                    solver,
                    state,
                    magnetic_fields,
                    pml_axis=pml_axis,
                    ex_curl=ex_curl,
                    ey_curl=ey_curl,
                    ez_curl=ez_curl,
                )
            psi_ex_y = electric_cpml_state["psi_ex_y"]
            psi_ex_z = electric_cpml_state["psi_ex_z"]
            psi_ey_x = electric_cpml_state["psi_ey_x"]
            psi_ey_z = electric_cpml_state["psi_ey_z"]
            psi_ez_x = electric_cpml_state["psi_ez_x"]
            psi_ez_y = electric_cpml_state["psi_ez_y"]
            psi_ex_y_imag = electric_cpml_state["psi_ex_y_imag"]
            psi_ex_z_imag = electric_cpml_state["psi_ex_z_imag"]
            psi_ey_x_imag = electric_cpml_state["psi_ey_x_imag"]
            psi_ey_z_imag = electric_cpml_state["psi_ey_z_imag"]
            psi_ez_x_imag = electric_cpml_state["psi_ez_x_imag"]
            psi_ez_y_imag = electric_cpml_state["psi_ez_y_imag"]
        else:
            hy_complex = torch.complex(magnetic_fields["Hy"], magnetic_fields["Hy_imag"])
            hz_complex = torch.complex(magnetic_fields["Hz"], magnetic_fields["Hz_imag"])
            d_hz_dy = _bloch_backward_diff(
                hz_complex,
                axis=1,
                inv_delta=solver.inv_dy_e,
                phase_cos=float(solver.boundary_phase_cos[1]),
                phase_sin=float(solver.boundary_phase_sin[1]),
            )
            d_hy_dz = _bloch_backward_diff(
                hy_complex,
                axis=2,
                inv_delta=solver.inv_dz_e,
                phase_cos=float(solver.boundary_phase_cos[2]),
                phase_sin=float(solver.boundary_phase_sin[2]),
            )
            ex_complex = torch.complex(state["Ex"], state["Ex_imag"]) * solver.cex_decay + ex_curl * (d_hz_dy - d_hy_dz)

            hx_complex = torch.complex(magnetic_fields["Hx"], magnetic_fields["Hx_imag"])
            d_hx_dz = _bloch_backward_diff(
                hx_complex,
                axis=2,
                inv_delta=solver.inv_dz_e,
                phase_cos=float(solver.boundary_phase_cos[2]),
                phase_sin=float(solver.boundary_phase_sin[2]),
            )
            d_hz_dx = _bloch_backward_diff(
                hz_complex,
                axis=0,
                inv_delta=solver.inv_dx_e,
                phase_cos=float(solver.boundary_phase_cos[0]),
                phase_sin=float(solver.boundary_phase_sin[0]),
            )
            ey_complex = torch.complex(state["Ey"], state["Ey_imag"]) * solver.cey_decay + ey_curl * (d_hx_dz - d_hz_dx)

            d_hy_dx = _bloch_backward_diff(
                hy_complex,
                axis=0,
                inv_delta=solver.inv_dx_e,
                phase_cos=float(solver.boundary_phase_cos[0]),
                phase_sin=float(solver.boundary_phase_sin[0]),
            )
            d_hx_dy = _bloch_backward_diff(
                hx_complex,
                axis=1,
                inv_delta=solver.inv_dy_e,
                phase_cos=float(solver.boundary_phase_cos[1]),
                phase_sin=float(solver.boundary_phase_sin[1]),
            )
            ez_complex = torch.complex(state["Ez"], state["Ez_imag"]) * solver.cez_decay + ez_curl * (d_hy_dx - d_hx_dy)

            electric_fields = {
                "Ex": ex_complex.real,
                "Ey": ey_complex.real,
                "Ez": ez_complex.real,
                "Ex_imag": ex_complex.imag,
                "Ey_imag": ey_complex.imag,
                "Ez_imag": ez_complex.imag,
            }
            psi_ex_y = psi_ex_z = psi_ey_x = psi_ey_z = psi_ez_x = psi_ez_y = None
    else:
        general_coeffs = None
        kerr_curls = None
        conductive_coeffs = None
        if getattr(solver, "nonlinear_general_enabled", False):
            # chi2 / two-photon absorption (and any static conduction) rewrite
            # BOTH the decay and curl coefficients from the pre-update E fields
            # each step; replicate the general coefficient kernel differentiably
            # so the VJP flows into the fields, eps, chi2, chi3, and the TPA
            # conductivity. This branch matches the forward dispatch order, which
            # prefers the general kernel over the curl-only Kerr kernel when a
            # chi2 or TPA channel is present.
            general_coeffs = _general_nonlinear_electric_coefficients(
                solver,
                state,
                eps_ex=eps_ex,
                eps_ey=eps_ey,
                eps_ez=eps_ez,
                chi2_ex=chi2_ex if chi2_ex is not None else solver.nonlinear_chi2_Ex,
                chi2_ey=chi2_ey if chi2_ey is not None else solver.nonlinear_chi2_Ey,
                chi2_ez=chi2_ez if chi2_ez is not None else solver.nonlinear_chi2_Ez,
                chi3_ex=chi3_ex if chi3_ex is not None else solver.kerr_chi3_Ex,
                chi3_ey=chi3_ey if chi3_ey is not None else solver.kerr_chi3_Ey,
                chi3_ez=chi3_ez if chi3_ez is not None else solver.kerr_chi3_Ez,
                tpa_ex=tpa_ex if tpa_ex is not None else solver.tpa_sigma_Ex,
                tpa_ey=tpa_ey if tpa_ey is not None else solver.tpa_sigma_Ey,
                tpa_ez=tpa_ez if tpa_ez is not None else solver.tpa_sigma_Ez,
            )
        elif getattr(solver, "kerr_enabled", False):
            # The Kerr coefficient kernel recomputes the curl factor from the
            # pre-update E fields each step; replicate it differentiably so the
            # VJP flows into the fields, eps, and chi3.
            kerr_curls = _kerr_dynamic_electric_curls(
                solver,
                state,
                eps_ex=eps_ex,
                eps_ey=eps_ey,
                eps_ez=eps_ez,
                chi3_ex=chi3_ex if chi3_ex is not None else solver.kerr_chi3_Ex,
                chi3_ey=chi3_ey if chi3_ey is not None else solver.kerr_chi3_Ey,
                chi3_ez=chi3_ez if chi3_ez is not None else solver.kerr_chi3_Ez,
            )
        elif getattr(solver, "conductive_enabled", False):
            # Static conduction makes both decay and curl eps-dependent through
            # the semi-implicit denominator; recompute them differentiably so the
            # VJP carries that dependence (the linear rule drops it).
            conductive_coeffs = _conductive_electric_coefficients(
                solver, eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez
            )
        if general_coeffs is not None:
            ex_decay_c, ex_curl_c, ex_eps_c = general_coeffs["Ex"][0], general_coeffs["Ex"][1], 1.0
            ey_decay_c, ey_curl_c, ey_eps_c = general_coeffs["Ey"][0], general_coeffs["Ey"][1], 1.0
            ez_decay_c, ez_curl_c, ez_eps_c = general_coeffs["Ez"][0], general_coeffs["Ez"][1], 1.0
        elif kerr_curls is not None:
            ex_decay_c, ex_curl_c, ex_eps_c = solver.cex_decay, kerr_curls["Ex"], 1.0
            ey_decay_c, ey_curl_c, ey_eps_c = solver.cey_decay, kerr_curls["Ey"], 1.0
            ez_decay_c, ez_curl_c, ez_eps_c = solver.cez_decay, kerr_curls["Ez"], 1.0
        elif conductive_coeffs is not None:
            ex_decay_c, ex_curl_c, ex_eps_c = conductive_coeffs["Ex"][0], conductive_coeffs["Ex"][1], 1.0
            ey_decay_c, ey_curl_c, ey_eps_c = conductive_coeffs["Ey"][0], conductive_coeffs["Ey"][1], 1.0
            ez_decay_c, ez_curl_c, ez_eps_c = conductive_coeffs["Ez"][0], conductive_coeffs["Ez"][1], 1.0
        else:
            ex_decay_c, ex_curl_c, ex_eps_c = solver.cex_decay, solver.cex_curl * solver.eps_Ex, eps_ex
            ey_decay_c, ey_curl_c, ey_eps_c = solver.cey_decay, solver.cey_curl * solver.eps_Ey, eps_ey
            ez_decay_c, ez_curl_c, ez_eps_c = solver.cez_decay, solver.cez_curl * solver.eps_Ez, eps_ez
        d_hz_dy = _backward_diff(magnetic_fields["Hz"], axis=1, inv_delta=solver.inv_dy_e)
        d_hy_dz = _backward_diff(magnetic_fields["Hy"], axis=2, inv_delta=solver.inv_dz_e)
        ex, psi_ex_y, psi_ex_z = _update_electric_component(
            state["Ex"],
            d_pos=d_hz_dy,
            d_neg=d_hy_dz,
            decay=ex_decay_c,
            curl_prefactor=ex_curl_c,
            eps=ex_eps_c,
            low_mode_pos=solver.boundary_y_low_code,
            high_mode_pos=solver.boundary_y_high_code,
            low_mode_neg=solver.boundary_z_low_code,
            high_mode_neg=solver.boundary_z_high_code,
            axis_pos=1,
            axis_neg=2,
            psi_pos=state.get("psi_ex_y"),
            psi_neg=state.get("psi_ex_z"),
            b_pos=getattr(solver, "cpml_b_e_y", None),
            c_pos=getattr(solver, "cpml_c_e_y", None),
            inv_kappa_pos=getattr(solver, "cpml_inv_kappa_e_y", None),
            b_neg=getattr(solver, "cpml_b_e_z", None),
            c_neg=getattr(solver, "cpml_c_e_z", None),
            inv_kappa_neg=getattr(solver, "cpml_inv_kappa_e_z", None),
        )

        d_hx_dz = _backward_diff(magnetic_fields["Hx"], axis=2, inv_delta=solver.inv_dz_e)
        d_hz_dx = _backward_diff(magnetic_fields["Hz"], axis=0, inv_delta=solver.inv_dx_e)
        ey, psi_ey_x, psi_ey_z = _update_electric_component(
            state["Ey"],
            d_pos=d_hx_dz,
            d_neg=d_hz_dx,
            decay=ey_decay_c,
            curl_prefactor=ey_curl_c,
            eps=ey_eps_c,
            low_mode_pos=solver.boundary_z_low_code,
            high_mode_pos=solver.boundary_z_high_code,
            low_mode_neg=solver.boundary_x_low_code,
            high_mode_neg=solver.boundary_x_high_code,
            axis_pos=2,
            axis_neg=0,
            psi_pos=state.get("psi_ey_z"),
            psi_neg=state.get("psi_ey_x"),
            b_pos=getattr(solver, "cpml_b_e_z", None),
            c_pos=getattr(solver, "cpml_c_e_z", None),
            inv_kappa_pos=getattr(solver, "cpml_inv_kappa_e_z", None),
            b_neg=getattr(solver, "cpml_b_e_x", None),
            c_neg=getattr(solver, "cpml_c_e_x", None),
            inv_kappa_neg=getattr(solver, "cpml_inv_kappa_e_x", None),
        )

        d_hy_dx = _backward_diff(magnetic_fields["Hy"], axis=0, inv_delta=solver.inv_dx_e)
        d_hx_dy = _backward_diff(magnetic_fields["Hx"], axis=1, inv_delta=solver.inv_dy_e)
        ez, psi_ez_x, psi_ez_y = _update_electric_component(
            state["Ez"],
            d_pos=d_hy_dx,
            d_neg=d_hx_dy,
            decay=ez_decay_c,
            curl_prefactor=ez_curl_c,
            eps=ez_eps_c,
            low_mode_pos=solver.boundary_x_low_code,
            high_mode_pos=solver.boundary_x_high_code,
            low_mode_neg=solver.boundary_y_low_code,
            high_mode_neg=solver.boundary_y_high_code,
            axis_pos=0,
            axis_neg=1,
            psi_pos=state.get("psi_ez_x"),
            psi_neg=state.get("psi_ez_y"),
            b_pos=getattr(solver, "cpml_b_e_x", None),
            c_pos=getattr(solver, "cpml_c_e_x", None),
            inv_kappa_pos=getattr(solver, "cpml_inv_kappa_e_x", None),
            b_neg=getattr(solver, "cpml_b_e_y", None),
            c_neg=getattr(solver, "cpml_c_e_y", None),
            inv_kappa_neg=getattr(solver, "cpml_inv_kappa_e_y", None),
        )
        if getattr(solver, "full_aniso_enabled", False):
            # Full (off-diagonal) anisotropy couples each E component to the two
            # off-axis curl(H) components; the diagonal entry is already carried
            # by the base curl above, so add the neighbor-averaged off-diagonal
            # correction here to match the forward per-step kernel order.
            aniso_correction = _full_aniso_electric_correction(solver, magnetic_fields)
            ex = ex + aniso_correction["Ex"]
            ey = ey + aniso_correction["Ey"]
            ez = ez + aniso_correction["Ez"]
        electric_fields = {"Ex": ex, "Ey": ey, "Ez": ez}

    if getattr(solver, "tfsf_enabled", False):
        electric_fields = _apply_tfsf_terms(
            electric_fields,
            solver=solver,
            auxiliary_state=auxiliary_state,
            term_key="electric_terms",
            sample_kind="magnetic",
            time_value=time_value,
        )
        auxiliary_state = _advance_tfsf_auxiliary_electric_state(
            solver,
            auxiliary_state,
            time_value=time_value,
        )
    else:
        electric_fields = _apply_source_term_list(
            electric_fields,
            terms=electric_source_terms,
            source_time=solver._source_time,
            omega=solver.source_omega,
            time_value=time_value,
            solver=None,
        )
        electric_fields = _apply_source_term_list(
            electric_fields,
            terms=source_terms,
            source_time=solver._source_time,
            omega=solver.source_omega,
            time_value=time_value,
            solver=solver if complex_fields else None,
        )

    dispersive_input_fields = {name: electric_fields[name] for name in ("Ex", "Ey", "Ez")}
    if complex_fields:
        # Bloch dispersive replay corrects the imaginary field with its own
        # polarization current; hand both halves to the shared correction.
        dispersive_input_fields.update(
            {name: electric_fields[name] for name in ("Ex_imag", "Ey_imag", "Ez_imag")}
        )
    corrected_electric_fields = _apply_dispersive_corrections(
        solver,
        dispersive_input_fields,
        dispersive_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    electric_fields.update(corrected_electric_fields)
    if not getattr(solver, "tfsf_enabled", False) and getattr(solver, "has_pec_faces", False):
        real_electric_fields = _enforce_pec_boundaries(
            solver,
            {name: electric_fields[name] for name in ("Ex", "Ey", "Ez")},
        )
        electric_fields.update(real_electric_fields)

    next_state = {
        "Ex": electric_fields["Ex"],
        "Ey": electric_fields["Ey"],
        "Ez": electric_fields["Ez"],
        "Hx": magnetic_fields["Hx"],
        "Hy": magnetic_fields["Hy"],
        "Hz": magnetic_fields["Hz"],
    }
    if complex_fields:
        next_state.update(
            {
                "Ex_imag": electric_fields["Ex_imag"],
                "Ey_imag": electric_fields["Ey_imag"],
                "Ez_imag": electric_fields["Ez_imag"],
                "Hx_imag": magnetic_fields["Hx_imag"],
                "Hy_imag": magnetic_fields["Hy_imag"],
                "Hz_imag": magnetic_fields["Hz_imag"],
            }
        )
    if solver.uses_cpml:
        next_state.update(
            {
                "psi_ex_y": psi_ex_y,
                "psi_ex_z": psi_ex_z,
                "psi_ey_x": psi_ey_x,
                "psi_ey_z": psi_ey_z,
                "psi_ez_x": psi_ez_x,
                "psi_ez_y": psi_ez_y,
                "psi_hx_y": psi_hx_y,
                "psi_hx_z": psi_hx_z,
                "psi_hy_x": psi_hy_x,
                "psi_hy_z": psi_hy_z,
                "psi_hz_x": psi_hz_x,
                "psi_hz_y": psi_hz_y,
            }
        )
        if complex_fields:
            next_state.update(
                {
                    "psi_ex_y_imag": psi_ex_y_imag,
                    "psi_ex_z_imag": psi_ex_z_imag,
                    "psi_ey_x_imag": psi_ey_x_imag,
                    "psi_ey_z_imag": psi_ey_z_imag,
                    "psi_ez_x_imag": psi_ez_x_imag,
                    "psi_ez_y_imag": psi_ez_y_imag,
                    "psi_hx_y_imag": psi_hx_y_imag,
                    "psi_hx_z_imag": psi_hx_z_imag,
                    "psi_hy_x_imag": psi_hy_x_imag,
                    "psi_hy_z_imag": psi_hy_z_imag,
                    "psi_hz_x_imag": psi_hz_x_imag,
                    "psi_hz_y_imag": psi_hz_y_imag,
                }
            )
    if auxiliary_state is not None:
        next_state["tfsf_aux_electric"] = auxiliary_state["electric"]
        next_state["tfsf_aux_magnetic"] = auxiliary_state["magnetic"]
    next_state.update(dispersive_state)
    next_state.update(magnetic_dispersive_state)
    return next_state


def _build_spectral_weight_schedule(entries, *, time_steps, window_type):
    schedules = []
    for entry in entries:
        phase_cos = 1.0
        phase_sin = 0.0
        start_step = int(entry["start_step"])
        end_step = None if entry["end_step"] is None else int(entry["end_step"])
        weights = []
        for n in range(time_steps):
            if n < start_step or (end_step is not None and n >= end_step):
                weights.append((0.0, 0.0))
            else:
                if window_type == "none" or start_step is None or end_step is None:
                    window_weight = 1.0
                else:
                    total_samples = max(end_step - start_step, 1)
                    position = (n - start_step) / total_samples
                    if window_type == "hanning":
                        window_weight = 0.5 * (1.0 - math.cos(2.0 * math.pi * position))
                    elif window_type == "ramp":
                        ramp_fraction = 0.1
                        if position < ramp_fraction:
                            window_weight = 0.5 * (1.0 - math.cos(math.pi * position / ramp_fraction))
                        else:
                            window_weight = 1.0
                    else:
                        window_weight = 1.0
                weights.append((window_weight * phase_cos, window_weight * phase_sin))
            next_cos = phase_cos * entry["phase_step_cos"] - phase_sin * entry["phase_step_sin"]
            next_sin = phase_sin * entry["phase_step_cos"] + phase_cos * entry["phase_step_sin"]
            phase_cos = next_cos
            phase_sin = next_sin
        schedules.append(tuple(weights))
    return tuple(schedules)


def _replay_can_capture_mid_magnetic(solver) -> bool:
    """Whether the replay may hand its mid-step H to the reverse reference backend.

    True only for the pure real standard / CPML configuration whose magnetic
    half-step is a plain ``_update_magnetic_component`` (no TFSF injection, no
    magnetic ADE correction, no magnetic surface source, no complex split field,
    no electric dispersion / nonlinear / conductive / full-anisotropic coupling).
    In that regime the H captured on the way to the electric update equals the
    reverse backend's ``_forward_magnetic_fields`` recompute bit-for-bit, and the
    per-step magnetic update carries no dependence on the (leaf) permittivity, so
    threading the captured tensor changes nothing numerically while removing one
    torch magnetic half-step per reverse step.
    """
    if has_complex_fields(solver):
        return False
    if getattr(solver, "tfsf_enabled", False):
        return False
    if getattr(solver, "dispersive_enabled", False):
        return False
    if getattr(solver, "magnetic_dispersive_enabled", False):
        return False
    if getattr(solver, "nonlinear_enabled", False):
        return False
    if getattr(solver, "conductive_enabled", False):
        return False
    if getattr(solver, "full_aniso_enabled", False):
        return False
    if getattr(solver, "_magnetic_source_terms", ()):
        return False
    return True


def _replay_segment_states(solver, checkpoint, start_step, end_step, *, mid_magnetic_out=None):
    """Replay the forward field states of one checkpoint segment.

    ``mid_magnetic_out``, when a list, collects the post-magnetic real H of each
    replayed step (one entry per step, aligned with ``states[offset]``) so the
    reverse pass can consume the mid-step H instead of recomputing it. It is only
    populated for configurations ``_replay_can_capture_mid_magnetic`` accepts.
    """
    validate_checkpoint_state(checkpoint)
    capture = mid_magnetic_out is not None and _replay_can_capture_mid_magnetic(solver)
    with torch.no_grad():
        current = clone_checkpoint_tensors(checkpoint)
        states = [current]
        for step_index in range(start_step, end_step):
            current = _step_state(
                solver,
                current,
                time_value=step_index * solver.dt,
                eps_ex=solver.eps_Ex,
                eps_ey=solver.eps_Ey,
                eps_ez=solver.eps_Ez,
                capture_magnetic=mid_magnetic_out if capture else None,
            )
            states.append({name: tensor.detach() for name, tensor in current.items()})
    return states


from .dispatch import reverse_step


from .bridge import _FDTDGradientBridge, run_fdtd_with_gradient_bridge
