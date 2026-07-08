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
    validate_checkpoint_state,
)
from ..excitation.injection import initialize_source_terms
from ..observers import (
    _align_plane_monitor_payload,
    _compute_plane_flux,
    _monitor_payload_is_point,
    _plane_coord_names,
)
from .reverse_common import (
    allocate_cpml_reverse_context,
    allocate_reverse_buffers,
    dynamic_electric_curls,
    finalize_dispersive_reverse_step,
    finalize_reverse_step_result,
    prepare_dispersive_reverse_context,
    replay_cpml_magnetic_step,
    replay_standard_magnetic_step,
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


def _evaluate_source_time_tensor(source_time, sample_time: torch.Tensor) -> torch.Tensor:
    kind_code = _source_time_kind_code(source_time)
    two_pi = 2.0 * math.pi
    amplitude = float(source_time["amplitude"])
    frequency = float(source_time["frequency"])
    phase = float(source_time.get("phase", 0.0))
    delay = float(source_time.get("delay", 0.0))

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


def _build_source_replay_solver(solver, compiled_source, eps_ex, eps_ey, eps_ez):
    return SimpleNamespace(
        scene=solver.scene,
        dx=solver.dx,
        dy=solver.dy,
        dz=solver.dz,
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
        _compiled_source=compiled_source,
        _compiled_material_model=getattr(solver, "_compiled_material_model", None),
        _mode_source_rebuild_from_fields=True,
        _source_time=compiled_source["source_time"],
        _compute_linear_launch_shape=solver._compute_linear_launch_shape,
        _iter_source_images=solver._iter_source_images,
        _component_plane_spec=solver._component_plane_spec,
        _component_positions=solver._component_positions,
        _plane_coordinate=solver._plane_coordinate,
    )


def _resolved_source_term_lists(solver, eps_ex, eps_ey, eps_ez):
    compiled_source = getattr(solver, "_compiled_source", None)
    default_source_terms = getattr(solver, "_source_terms", ())
    default_electric_source_terms = getattr(solver, "_electric_source_terms", ())
    default_magnetic_source_terms = getattr(solver, "_magnetic_source_terms", ())
    if compiled_source is None:
        return default_source_terms, default_electric_source_terms, default_magnetic_source_terms
    if compiled_source["kind"] not in {"point_dipole", "plane_wave", "gaussian_beam", "mode_source"}:
        return default_source_terms, default_electric_source_terms, default_magnetic_source_terms

    if compiled_source["kind"] == "mode_source":
        cache = getattr(solver, "_source_replay_term_cache", None)
        cache_key = (
            id(compiled_source),
            int(eps_ex.data_ptr()),
            int(eps_ey.data_ptr()),
            int(eps_ez.data_ptr()),
        )
        if isinstance(cache, dict):
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

    temp_solver = _build_source_replay_solver(solver, compiled_source, eps_ex, eps_ey, eps_ez)
    if compiled_source["kind"] == "mode_source":
        with torch.enable_grad():
            initialize_source_terms(temp_solver)
    else:
        initialize_source_terms(temp_solver)
    resolved = (
        tuple(temp_solver._source_terms),
        tuple(temp_solver._electric_source_terms),
        tuple(temp_solver._magnetic_source_terms),
    )
    if compiled_source["kind"] == "mode_source" and isinstance(getattr(solver, "_source_replay_term_cache", None), dict):
        solver._source_replay_term_cache.clear()
        solver._source_replay_term_cache[cache_key] = resolved
    return resolved


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


def _source_term_signal_patch(term, *, source_time, omega, time_value):
    literal_patch = term.get("literal_patch")
    if literal_patch is not None:
        return literal_patch
    if term["cw_cos_patch"] is not None:
        signal_cos, signal_sin = _resolve_cw_signal(source_time, omega, time_value)
        return float(signal_cos) * term["cw_cos_patch"] + float(signal_sin) * term["cw_sin_patch"]
    if term["delay_patch"] is not None:
        sample_time = float(time_value) - term["delay_patch"]
        patch = _evaluate_source_time_tensor(source_time, sample_time) * term["patch"]
        activation_delay_patch = term.get("activation_delay_patch")
        if activation_delay_patch is not None:
            patch = torch.where(
                activation_delay_patch > float(time_value),
                torch.zeros_like(patch),
                patch,
            )
        return patch
    scalar_signal = evaluate_source_time(source_time, float(time_value))
    return (float(scalar_signal) * float(term["phase_real"])) * term["patch"]


def _bloch_source_term_signal_patch(term, *, source_time, omega, time_value) -> torch.Tensor:
    if term["cw_cos_patch"] is not None or term["delay_patch"] is not None:
        raise NotImplementedError("Bloch source-term reverse currently supports uniform point-source patches only.")
    scalar_signal = float(evaluate_source_time(source_time, float(time_value)))
    patch = term["patch"]
    patch_real = patch.to(dtype=torch.float32 if not torch.is_floating_point(patch) else patch.dtype)
    return torch.complex(
        patch_real * (scalar_signal * float(term["phase_real"])),
        patch_real * (scalar_signal * float(term["phase_imag"])),
    )


def _mode_source_retain_graph(solver) -> bool:
    remaining = getattr(solver, "_mode_source_explicit_vjp_remaining", None)
    if remaining is None:
        return False
    remaining = max(int(remaining), 0)
    retain_graph = remaining > 1
    solver._mode_source_explicit_vjp_remaining = max(remaining - 1, 0)
    return retain_graph


def _mode_source_term_objective(
    terms,
    *,
    adjoint_by_field,
    source_time,
    omega,
    time_value,
) -> torch.Tensor | None:
    objective = None
    for term in terms:
        field_name = term["field_name"]
        if field_name not in adjoint_by_field:
            continue
        payload = _source_term_signal_patch(
            term,
            source_time=source_time,
            omega=omega,
            time_value=time_value,
        )
        offsets = term["offsets"]
        region = _slice_from_offsets_shape(offsets, payload.shape)
        adjoint_patch = adjoint_by_field[field_name][region].detach().to(device=payload.device, dtype=payload.dtype)
        contribution = torch.sum(payload * adjoint_patch)
        objective = contribution if objective is None else objective + contribution
    return objective


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
    compiled_source = getattr(solver, "_compiled_source", None)
    if compiled_source is not None and compiled_source.get("kind") == "mode_source":
        magnetic_output_adjoint = step_result.magnetic_output_adjoint or {}
        with torch.enable_grad():
            objective = _mode_source_term_objective(
                tuple(electric_source_terms) + tuple(source_terms),
                adjoint_by_field=source_adjoint_state,
                source_time=solver._source_time,
                omega=solver.source_omega,
                time_value=time_value,
            )
            magnetic_objective = _mode_source_term_objective(
                tuple(magnetic_source_terms),
                adjoint_by_field=magnetic_output_adjoint,
                source_time=solver._source_time,
                omega=solver.source_omega,
                time_value=time_value,
            )
            if magnetic_objective is not None:
                objective = magnetic_objective if objective is None else objective + magnetic_objective
            if objective is not None:
                gradients = torch.autograd.grad(
                    objective,
                    (eps_ex, eps_ey, eps_ez),
                    allow_unused=True,
                    retain_graph=_mode_source_retain_graph(solver),
                )
                for field_name, grad in zip(("Ex", "Ey", "Ez"), gradients):
                    if grad is not None:
                        grad_eps_by_field[field_name] = grad_eps_by_field[field_name] + grad.detach()
        return _ReverseStepResult(
            pre_step_adjoint=step_result.pre_step_adjoint,
            grad_eps_ex=grad_eps_by_field["Ex"],
            grad_eps_ey=grad_eps_by_field["Ey"],
            grad_eps_ez=grad_eps_by_field["Ez"],
            backend=step_result.backend,
            source_adjoint_state=step_result.source_adjoint_state,
            magnetic_output_adjoint=step_result.magnetic_output_adjoint,
        )

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


def _reverse_tfsf_auxiliary_state_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
):
    auxiliary_state = _extract_tfsf_auxiliary_state(forward_state)
    if auxiliary_state is None:
        return {}

    aux_electric = auxiliary_state["electric"].detach().clone().requires_grad_(True)
    aux_magnetic = auxiliary_state["magnetic"].detach().clone().requires_grad_(True)
    state_inputs = {name: tensor.detach() for name, tensor in forward_state.items()}
    state_inputs["tfsf_aux_electric"] = aux_electric
    state_inputs["tfsf_aux_magnetic"] = aux_magnetic

    with torch.enable_grad():
        next_state = _step_state(
            solver,
            state_inputs,
            time_value=time_value,
            eps_ex=eps_ex.detach(),
            eps_ey=eps_ey.detach(),
            eps_ez=eps_ez.detach(),
        )
        objective = aux_electric.new_zeros(())
        for name, tensor in next_state.items():
            adjoint_tensor = adjoint_state.get(name)
            if adjoint_tensor is None:
                continue
            objective = objective + torch.sum(tensor * adjoint_tensor.detach())
        aux_grads = torch.autograd.grad(
            objective,
            (aux_electric, aux_magnetic),
            allow_unused=True,
        )

    return {
        "tfsf_aux_electric": _safe_grad(aux_grads[0], aux_electric).detach(),
        "tfsf_aux_magnetic": _safe_grad(aux_grads[1], aux_magnetic).detach(),
    }


def _bloch_backward_diff(field: torch.Tensor, *, axis: int, inv_delta: float, phase_cos: float, phase_sin: float):
    shape = list(field.shape)
    shape[axis] += 1
    diff = field.new_zeros(shape)
    if field.shape[axis] == 0:
        return diff

    low = field.select(axis, 0)
    high = field.select(axis, field.shape[axis] - 1)
    low_value = (low - _complex_phase_negative(high, phase_cos, phase_sin)) * float(inv_delta)
    low_region = [slice(None)] * field.ndim
    low_region[axis] = 0
    diff[tuple(low_region)] = low_value

    if field.shape[axis] == 1:
        return diff

    high_value = (_complex_phase_positive(low, phase_cos, phase_sin) - high) * float(inv_delta)
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
        diff[tuple(interior)] = (field[tuple(current)] - field[tuple(previous)]) * float(inv_delta)
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
            if solver is not None and has_complex_fields(solver):
                raise NotImplementedError("Bloch-boundary source replay requires CW phased source terms.")
            sample_time = time_value - term["delay_patch"]
            patch = _evaluate_source_time_tensor(term_source_time, sample_time) * term["patch"]
            activation_delay_patch = term.get("activation_delay_patch")
            if activation_delay_patch is not None:
                patch = torch.where(
                    time_value < activation_delay_patch,
                    torch.zeros_like(patch),
                    patch,
                )
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


def _advance_dispersive_state(solver, state):
    updated = {}
    for component_name, model_name, index, _tensor_names, entry in iter_dispersive_state_specs(solver) or ():
        field = state[component_name]
        if model_name == "debye":
            polarization_name = dispersive_state_name(component_name, model_name, index, "polarization")
            current_name = dispersive_state_name(component_name, model_name, index, "current")
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
            current_name = dispersive_state_name(component_name, model_name, index, "current")
            updated[current_name] = _update_drude_state(
                field,
                state[current_name],
                drive=entry["drive"],
                decay=entry["decay"],
            )
            continue

        polarization_name = dispersive_state_name(component_name, model_name, index, "polarization")
        current_name = dispersive_state_name(component_name, model_name, index, "current")
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


def _apply_dispersive_corrections(solver, electric_fields, dispersive_state, *, eps_ex, eps_ey, eps_ez):
    if not getattr(solver, "dispersive_enabled", False):
        return electric_fields

    inv_eps = {
        "Ex": 1.0 / eps_ex,
        "Ey": 1.0 / eps_ey,
        "Ez": 1.0 / eps_ez,
    }
    updated = dict(electric_fields)
    for component_name, model_name, index, _tensor_names, _entry in iter_dispersive_state_specs(solver) or ():
        current_name = dispersive_state_name(component_name, model_name, index, "current")
        updated[component_name] = updated[component_name] - float(solver.dt) * dispersive_state[current_name] * inv_eps[
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


def _reverse_dispersive_state_python_reference(solver, forward_state, dispersive_output_adjoint):
    electric_prev_adjoint = {
        "Ex": torch.zeros_like(forward_state["Ex"]),
        "Ey": torch.zeros_like(forward_state["Ey"]),
        "Ez": torch.zeros_like(forward_state["Ez"]),
    }
    pre_step_dispersive_adjoint = {
        name: torch.zeros_like(forward_state[name])
        for name in checkpoint_schema(solver).dispersive_state_names
    }
    dt = float(solver.dt)

    for component_name, model_name, index, _tensor_names, entry in iter_dispersive_state_specs(solver) or ():
        drive = entry["drive"]
        if model_name == "debye":
            polarization_name = dispersive_state_name(component_name, model_name, index, "polarization")
            current_name = dispersive_state_name(component_name, model_name, index, "current")
            adj_polarization_post = dispersive_output_adjoint[polarization_name]
            adj_current_post = dispersive_output_adjoint[current_name]
            adj_polarization_internal = adj_polarization_post + adj_current_post / dt
            electric_prev_adjoint[component_name] = electric_prev_adjoint[component_name] + drive * adj_polarization_internal
            pre_step_dispersive_adjoint[polarization_name] = (
                pre_step_dispersive_adjoint[polarization_name]
                + float(entry["decay"]) * adj_polarization_internal
                - adj_current_post / dt
            )
            continue

        if model_name == "drude":
            current_name = dispersive_state_name(component_name, model_name, index, "current")
            adj_current_post = dispersive_output_adjoint[current_name]
            electric_prev_adjoint[component_name] = electric_prev_adjoint[component_name] + drive * adj_current_post
            pre_step_dispersive_adjoint[current_name] = (
                pre_step_dispersive_adjoint[current_name] + float(entry["decay"]) * adj_current_post
            )
            continue

        polarization_name = dispersive_state_name(component_name, model_name, index, "polarization")
        current_name = dispersive_state_name(component_name, model_name, index, "current")
        adj_polarization_post = dispersive_output_adjoint[polarization_name]
        adj_current_post = dispersive_output_adjoint[current_name]
        adj_current_internal = adj_current_post + dt * adj_polarization_post
        electric_prev_adjoint[component_name] = electric_prev_adjoint[component_name] + drive * adj_current_internal
        pre_step_dispersive_adjoint[polarization_name] = (
            pre_step_dispersive_adjoint[polarization_name]
            + adj_polarization_post
            - float(entry["restoring"]) * adj_current_internal
        )
        pre_step_dispersive_adjoint[current_name] = (
            pre_step_dispersive_adjoint[current_name] + float(entry["decay"]) * adj_current_internal
        )

    return electric_prev_adjoint, pre_step_dispersive_adjoint


def _broadcast_vector(vector: torch.Tensor, axis: int) -> torch.Tensor:
    shape = [1, 1, 1]
    shape[axis] = int(vector.shape[0])
    return vector.view(*shape)


def _forward_diff(field: torch.Tensor, axis: int, inv_delta: float) -> torch.Tensor:
    slicer_lo = [slice(None)] * field.ndim
    slicer_hi = [slice(None)] * field.ndim
    slicer_lo[axis] = slice(0, -1)
    slicer_hi[axis] = slice(1, None)
    return (field[tuple(slicer_hi)] - field[tuple(slicer_lo)]) * float(inv_delta)


def _backward_diff(field: torch.Tensor, axis: int, inv_delta: float) -> torch.Tensor:
    shape = list(field.shape)
    shape[axis] += 1
    diff = field.new_zeros(shape)
    interior = [slice(None)] * diff.ndim
    field_hi = [slice(None)] * field.ndim
    field_lo = [slice(None)] * field.ndim
    interior[axis] = slice(1, -1)
    field_hi[axis] = slice(1, None)
    field_lo[axis] = slice(0, -1)
    diff[tuple(interior)] = (field[tuple(field_hi)] - field[tuple(field_lo)]) * float(inv_delta)
    return diff


def _accumulate_forward_diff_adjoint(field_grad: torch.Tensor, diff_grad: torch.Tensor, *, axis: int, inv_delta: float):
    scale = float(inv_delta)
    field_lo = [slice(None)] * field_grad.ndim
    field_hi = [slice(None)] * field_grad.ndim
    field_lo[axis] = slice(0, -1)
    field_hi[axis] = slice(1, None)
    field_grad[tuple(field_lo)] = field_grad[tuple(field_lo)] - scale * diff_grad
    field_grad[tuple(field_hi)] = field_grad[tuple(field_hi)] + scale * diff_grad


def _accumulate_backward_diff_adjoint(field_grad: torch.Tensor, diff_grad: torch.Tensor, *, axis: int, inv_delta: float):
    interior = [slice(None)] * diff_grad.ndim
    interior[axis] = slice(1, -1)
    interior_grad = diff_grad[tuple(interior)]
    _accumulate_forward_diff_adjoint(field_grad, interior_grad, axis=axis, inv_delta=inv_delta)


def _accumulate_bloch_backward_diff_adjoint(
    field_grad: torch.Tensor,
    diff_grad: torch.Tensor,
    *,
    axis: int,
    inv_delta: float,
    phase_cos: float,
    phase_sin: float,
):
    _accumulate_backward_diff_adjoint(field_grad, diff_grad, axis=axis, inv_delta=inv_delta)
    scale = float(inv_delta)
    low_grad = diff_grad.select(axis, 0)
    high_grad = diff_grad.select(axis, int(diff_grad.shape[axis] - 1))
    field_grad.select(axis, 0).add_(scale * low_grad + scale * _complex_phase_negative(high_grad, phase_cos, phase_sin))
    field_grad.select(axis, int(field_grad.shape[axis] - 1)).add_(
        -scale * _complex_phase_positive(low_grad, phase_cos, phase_sin) - scale * high_grad
    )


def _boundary_axis_masks(shape, axis: int, low_mode: int, high_mode: int, device) -> tuple[torch.Tensor, torch.Tensor]:
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


def _apply_complex_z_cpml_correction(
    field: torch.Tensor,
    psi: torch.Tensor,
    derivative: torch.Tensor,
    *,
    curl: torch.Tensor,
    inv_kappa: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    tangent_axis: int,
    tangent_low_mode: int,
    tangent_high_mode: int,
    sign: float,
):
    b_term = _broadcast_vector(b, 2)
    c_term = _broadcast_vector(c, 2)
    inv_kappa_term = _broadcast_vector(inv_kappa, 2)
    psi_candidate = b_term * psi + c_term * derivative
    correction = curl * (derivative * (inv_kappa_term - 1.0) + psi_candidate)
    candidate = field + float(sign) * correction
    mask = _cpml_correction_mask(
        field.shape,
        normal_axis=2,
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
        inv_delta=solver.inv_dy,
        phase_cos=float(solver.boundary_phase_cos[1]),
        phase_sin=float(solver.boundary_phase_sin[1]),
    )
    d_hy_dz = _backward_diff(hy_complex, axis=2, inv_delta=solver.inv_dz)
    ex_complex = previous_ex * solver.cex_decay + ex_curl * (d_hz_dy - d_hy_dz)
    ex_complex = _apply_complex_axis_boundary(
        ex_complex,
        previous_ex,
        axis=2,
        low_mode=solver.boundary_z_low_code,
        high_mode=solver.boundary_z_high_code,
    )
    psi_ex_z = torch.complex(state["psi_ex_z"], state["psi_ex_z_imag"])
    ex_complex, psi_ex_z = _apply_complex_z_cpml_correction(
        ex_complex,
        psi_ex_z,
        d_hy_dz,
        curl=ex_curl,
        inv_kappa=solver.cpml_inv_kappa_e_z,
        b=solver.cpml_b_e_z,
        c=solver.cpml_c_e_z,
        tangent_axis=1,
        tangent_low_mode=solver.boundary_y_low_code,
        tangent_high_mode=solver.boundary_y_high_code,
        sign=-1.0,
    )

    previous_ey = _complex_state_field(state, "Ey")
    d_hx_dz = _backward_diff(hx_complex, axis=2, inv_delta=solver.inv_dz)
    d_hz_dx = _bloch_backward_diff(
        hz_complex,
        axis=0,
        inv_delta=solver.inv_dx,
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
    ey_complex, psi_ey_z = _apply_complex_z_cpml_correction(
        ey_complex,
        psi_ey_z,
        d_hx_dz,
        curl=ey_curl,
        inv_kappa=solver.cpml_inv_kappa_e_z,
        b=solver.cpml_b_e_z,
        c=solver.cpml_c_e_z,
        tangent_axis=0,
        tangent_low_mode=solver.boundary_x_low_code,
        tangent_high_mode=solver.boundary_x_high_code,
        sign=1.0,
    )

    previous_ez = _complex_state_field(state, "Ez")
    d_hy_dx = _bloch_backward_diff(
        hy_complex,
        axis=0,
        inv_delta=solver.inv_dx,
        phase_cos=float(solver.boundary_phase_cos[0]),
        phase_sin=float(solver.boundary_phase_sin[0]),
    )
    d_hx_dy = _bloch_backward_diff(
        hx_complex,
        axis=1,
        inv_delta=solver.inv_dy,
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


def _step_state(solver, state, *, time_value, eps_ex, eps_ey, eps_ez):
    source_terms, electric_source_terms, magnetic_source_terms = _resolved_source_term_lists(
        solver,
        eps_ex,
        eps_ey,
        eps_ez,
    )
    auxiliary_state = _extract_tfsf_auxiliary_state(state)
    complex_fields = has_complex_fields(solver)
    psi_hx_y_imag = psi_hx_z_imag = psi_hy_x_imag = psi_hy_z_imag = psi_hz_x_imag = psi_hz_y_imag = None

    d_ez_dy = _forward_diff(state["Ez"], axis=1, inv_delta=solver.inv_dy)
    d_ey_dz = _forward_diff(state["Ey"], axis=2, inv_delta=solver.inv_dz)
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

    d_ex_dz = _forward_diff(state["Ex"], axis=2, inv_delta=solver.inv_dz)
    d_ez_dx = _forward_diff(state["Ez"], axis=0, inv_delta=solver.inv_dx)
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

    d_ey_dx = _forward_diff(state["Ey"], axis=0, inv_delta=solver.inv_dx)
    d_ex_dy = _forward_diff(state["Ex"], axis=1, inv_delta=solver.inv_dy)
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
        d_ez_imag_dy = _forward_diff(state["Ez_imag"], axis=1, inv_delta=solver.inv_dy)
        d_ey_imag_dz = _forward_diff(state["Ey_imag"], axis=2, inv_delta=solver.inv_dz)
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

        d_ex_imag_dz = _forward_diff(state["Ex_imag"], axis=2, inv_delta=solver.inv_dz)
        d_ez_imag_dx = _forward_diff(state["Ez_imag"], axis=0, inv_delta=solver.inv_dx)
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

        d_ey_imag_dx = _forward_diff(state["Ey_imag"], axis=0, inv_delta=solver.inv_dx)
        d_ex_imag_dy = _forward_diff(state["Ex_imag"], axis=1, inv_delta=solver.inv_dy)
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
    dispersive_state = _advance_dispersive_state(solver, state)

    if complex_fields:
        ex_curl = _dynamic_electric_curl(solver.cex_curl, solver.eps_Ex, eps_ex)
        ey_curl = _dynamic_electric_curl(solver.cey_curl, solver.eps_Ey, eps_ey)
        ez_curl = _dynamic_electric_curl(solver.cez_curl, solver.eps_Ez, eps_ez)
        if getattr(solver, "uses_cpml", False):
            if tuple(getattr(solver, "has_bloch_axes", ())) != ("x", "y"):
                raise NotImplementedError("Complex-field CPML adjoint replay currently supports x/y Bloch with z PML only.")
            electric_fields, electric_cpml_state = _update_mixed_bloch_cpml_electric_fields(
                solver,
                state,
                magnetic_fields,
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
                inv_delta=solver.inv_dy,
                phase_cos=float(solver.boundary_phase_cos[1]),
                phase_sin=float(solver.boundary_phase_sin[1]),
            )
            d_hy_dz = _bloch_backward_diff(
                hy_complex,
                axis=2,
                inv_delta=solver.inv_dz,
                phase_cos=float(solver.boundary_phase_cos[2]),
                phase_sin=float(solver.boundary_phase_sin[2]),
            )
            ex_complex = torch.complex(state["Ex"], state["Ex_imag"]) * solver.cex_decay + ex_curl * (d_hz_dy - d_hy_dz)

            hx_complex = torch.complex(magnetic_fields["Hx"], magnetic_fields["Hx_imag"])
            d_hx_dz = _bloch_backward_diff(
                hx_complex,
                axis=2,
                inv_delta=solver.inv_dz,
                phase_cos=float(solver.boundary_phase_cos[2]),
                phase_sin=float(solver.boundary_phase_sin[2]),
            )
            d_hz_dx = _bloch_backward_diff(
                hz_complex,
                axis=0,
                inv_delta=solver.inv_dx,
                phase_cos=float(solver.boundary_phase_cos[0]),
                phase_sin=float(solver.boundary_phase_sin[0]),
            )
            ey_complex = torch.complex(state["Ey"], state["Ey_imag"]) * solver.cey_decay + ey_curl * (d_hx_dz - d_hz_dx)

            d_hy_dx = _bloch_backward_diff(
                hy_complex,
                axis=0,
                inv_delta=solver.inv_dx,
                phase_cos=float(solver.boundary_phase_cos[0]),
                phase_sin=float(solver.boundary_phase_sin[0]),
            )
            d_hx_dy = _bloch_backward_diff(
                hx_complex,
                axis=1,
                inv_delta=solver.inv_dy,
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
        d_hz_dy = _backward_diff(magnetic_fields["Hz"], axis=1, inv_delta=solver.inv_dy)
        d_hy_dz = _backward_diff(magnetic_fields["Hy"], axis=2, inv_delta=solver.inv_dz)
        ex, psi_ex_y, psi_ex_z = _update_electric_component(
            state["Ex"],
            d_pos=d_hz_dy,
            d_neg=d_hy_dz,
            decay=solver.cex_decay,
            curl_prefactor=solver.cex_curl * solver.eps_Ex,
            eps=eps_ex,
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

        d_hx_dz = _backward_diff(magnetic_fields["Hx"], axis=2, inv_delta=solver.inv_dz)
        d_hz_dx = _backward_diff(magnetic_fields["Hz"], axis=0, inv_delta=solver.inv_dx)
        ey, psi_ey_x, psi_ey_z = _update_electric_component(
            state["Ey"],
            d_pos=d_hx_dz,
            d_neg=d_hz_dx,
            decay=solver.cey_decay,
            curl_prefactor=solver.cey_curl * solver.eps_Ey,
            eps=eps_ey,
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

        d_hy_dx = _backward_diff(magnetic_fields["Hy"], axis=0, inv_delta=solver.inv_dx)
        d_hx_dy = _backward_diff(magnetic_fields["Hx"], axis=1, inv_delta=solver.inv_dy)
        ez, psi_ez_x, psi_ez_y = _update_electric_component(
            state["Ez"],
            d_pos=d_hy_dx,
            d_neg=d_hx_dy,
            decay=solver.cez_decay,
            curl_prefactor=solver.cez_curl * solver.eps_Ez,
            eps=eps_ez,
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

    real_electric_fields = _apply_dispersive_corrections(
        solver,
        {name: electric_fields[name] for name in ("Ex", "Ey", "Ez")},
        dispersive_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    electric_fields.update(real_electric_fields)
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


def _replay_segment_states(solver, checkpoint, start_step, end_step):
    validate_checkpoint_state(checkpoint)
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
            )
            states.append({name: tensor.detach() for name, tensor in current.items()})
    return states


from .dispatch import reverse_step


from .bridge import _FDTDGradientBridge, run_fdtd_with_gradient_bridge
