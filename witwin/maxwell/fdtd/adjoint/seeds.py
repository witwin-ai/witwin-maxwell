from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch

from ..boundary import has_complex_fields
from ..observers import get_observer_results as get_observer_results_impl
from ..postprocess import get_frequency_solution as get_frequency_solution_impl


def _runtime():
    from . import core as _adjoint

    return _adjoint


@dataclass(frozen=True)
class _FieldAccumulatorPair:
    entry_index: int
    state_field_name: str
    real_index: int
    imag_index: int


@dataclass(frozen=True)
class _PointAccumulatorPair:
    state_field_name: str
    global_indices: tuple[int, ...]
    point_i: torch.Tensor
    point_j: torch.Tensor
    point_k: torch.Tensor
    real_index: int
    imag_index: int


@dataclass(frozen=True)
class _PlaneAccumulatorPair:
    state_field_name: str
    global_indices: tuple[int, ...]
    axis_code: int
    plane_index: int
    real_index: int
    imag_index: int


@dataclass(frozen=True)
class _ScheduleTensorPack:
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass(frozen=True)
class _DenseSeedBatch:
    state_field_name: str
    entry_indices: torch.Tensor
    grad_real: torch.Tensor
    grad_imag: torch.Tensor
    cos_pack: torch.Tensor
    sin_pack: torch.Tensor


@dataclass(frozen=True)
class _PointSeedBatch:
    state_field_name: str
    entry_indices: torch.Tensor
    point_i: torch.Tensor
    point_j: torch.Tensor
    point_k: torch.Tensor
    grad_real: torch.Tensor
    grad_imag: torch.Tensor
    cos_pack: torch.Tensor
    sin_pack: torch.Tensor
    point_i32: torch.Tensor
    point_j32: torch.Tensor
    point_k32: torch.Tensor


@dataclass(frozen=True)
class _PlaneSeedBatch:
    state_field_name: str
    axis_code: int
    plane_index: int
    entry_indices: torch.Tensor
    grad_real: torch.Tensor
    grad_imag: torch.Tensor
    cos_pack: torch.Tensor
    sin_pack: torch.Tensor


@dataclass(frozen=True)
class _PortSeedBatch:
    port_index: int
    voltage_samples: torch.Tensor
    current_samples: torch.Tensor
    drive_samples: torch.Tensor


@dataclass(frozen=True)
class _CircuitSeedBatch:
    circuit_index: int
    node_samples: torch.Tensor
    branch_samples: torch.Tensor
    device_power_samples: torch.Tensor
    balance_increment_samples: torch.Tensor
    port_power_samples: torch.Tensor
    field_energy_samples: torch.Tensor
    dc_condition: torch.Tensor
    last_condition: torch.Tensor


@dataclass(frozen=True)
class _WireSeedBatch:
    current_samples: torch.Tensor
    charge_samples: torch.Tensor


@dataclass(frozen=True)
class _SeedRuntime:
    dft_schedule: _ScheduleTensorPack
    observer_schedule: _ScheduleTensorPack
    dense_batches: tuple[_DenseSeedBatch, ...]
    point_batches: tuple[_PointSeedBatch, ...]
    plane_batches: tuple[_PlaneSeedBatch, ...]
    port_batches: tuple[_PortSeedBatch, ...]
    circuit_batches: tuple[_CircuitSeedBatch, ...]
    wire_batches: tuple[_WireSeedBatch, ...] = ()
    backend: str = "device_batched"


def _schedule_to_tensor_pack(schedules, *, device, dtype) -> _ScheduleTensorPack:
    if not schedules:
        empty = torch.zeros((0, 0), device=device, dtype=dtype)
        return _ScheduleTensorPack(cos=empty, sin=empty)
    cos = torch.tensor(
        [[float(weight_cos) for weight_cos, _weight_sin in entry] for entry in schedules],
        device=device,
        dtype=dtype,
    )
    sin = torch.tensor(
        [[float(weight_sin) for _weight_cos, weight_sin in entry] for entry in schedules],
        device=device,
        dtype=dtype,
    )
    return _ScheduleTensorPack(cos=cos, sin=sin)


def _reshape_entry_major_grad(grad: torch.Tensor, entry_count: int) -> torch.Tensor:
    if entry_count <= 0:
        raise ValueError("Seed batch must contain at least one entry.")
    detached = grad.detach()
    if entry_count == 1:
        if detached.ndim == 0 or detached.shape[0] != 1:
            detached = detached.unsqueeze(0)
        return detached.contiguous()
    if detached.ndim == 0 or detached.shape[0] != entry_count:
        raise RuntimeError(
            "Seed gradient layout does not match the observer/global frequency index structure."
        )
    return detached.contiguous()


def _stack_dense_seed_records(records) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entry_indices = torch.tensor(
        [int(entry_index) for entry_index, _grad_real, _grad_imag in records],
        device=records[0][1].device,
        dtype=torch.long,
    )
    grad_real = torch.stack([grad_real.detach() for _entry_index, grad_real, _grad_imag in records], dim=0).contiguous()
    grad_imag = torch.stack([grad_imag.detach() for _entry_index, _grad_real, grad_imag in records], dim=0).contiguous()
    return entry_indices, grad_real, grad_imag


def _seed_batch_contribution(
    grad_real: torch.Tensor,
    grad_imag: torch.Tensor,
    weight_cos: torch.Tensor,
    weight_sin: torch.Tensor,
) -> torch.Tensor:
    view_shape = (int(weight_cos.shape[0]),) + (1,) * max(grad_real.ndim - 1, 0)
    return torch.sum(
        grad_real * weight_cos.view(view_shape) + grad_imag * weight_sin.view(view_shape),
        dim=0,
    )


def _clone_seed_solver(solver):
    seed_solver = SimpleNamespace()
    seed_solver.scene = solver.scene
    seed_solver.verbose = False
    seed_solver.complex_fields_enabled = has_complex_fields(solver)
    seed_solver._normalize_source = getattr(solver, "_normalize_source", False)
    seed_solver._source_time = getattr(solver, "_source_time", None)
    seed_solver.dft_window_type = getattr(solver, "dft_window_type", "none")
    seed_solver.observer_window_type = getattr(solver, "observer_window_type", "none")
    seed_solver.observers_enabled = bool(getattr(solver, "observers_enabled", False))
    seed_solver.observers = list(getattr(solver, "observers", ()))
    seed_solver._point_observer_groups = {}
    seed_solver._plane_observer_groups = {}

    leaves = []
    field_pairs = []
    point_pairs = []
    plane_pairs = []

    seed_solver._dft_entries = []
    for entry_index, entry in enumerate(getattr(solver, "_dft_entries", ())):
        cloned_entry = dict(entry)
        cloned_fields = {}
        for field_name in ("Ex", "Ey", "Ez"):
            payload = entry["fields"][field_name]
            real = payload["real"].detach().clone().requires_grad_(True)
            imag = payload["imag"].detach().clone().requires_grad_(True)
            real_index = len(leaves)
            imag_index = len(leaves) + 1
            leaves.extend([real, imag])
            field_pairs.append(
                _FieldAccumulatorPair(
                    entry_index=entry_index,
                    state_field_name=field_name,
                    real_index=real_index,
                    imag_index=imag_index,
                )
            )
            aux_real = None
            aux_imag = None
            if payload.get("aux_real") is not None and payload.get("aux_imag") is not None:
                aux_real = payload["aux_real"].detach().clone().requires_grad_(True)
                aux_imag = payload["aux_imag"].detach().clone().requires_grad_(True)
                aux_real_index = len(leaves)
                aux_imag_index = len(leaves) + 1
                leaves.extend([aux_real, aux_imag])
                field_pairs.append(
                    _FieldAccumulatorPair(
                        entry_index=entry_index,
                        state_field_name=f"{field_name}_imag",
                        real_index=aux_real_index,
                        imag_index=aux_imag_index,
                    )
                )
            cloned_fields[field_name] = {
                "real": real,
                "imag": imag,
                "aux_real": aux_real,
                "aux_imag": aux_imag,
            }
        cloned_entry["fields"] = cloned_fields
        seed_solver._dft_entries.append(cloned_entry)

    seed_solver._observer_spectral_entries = [dict(entry) for entry in getattr(solver, "_observer_spectral_entries", ())]

    for group_key, group in getattr(solver, "_point_observer_groups", {}).items():
        cloned_group = dict(group)
        real = group["real"].detach().clone().requires_grad_(True)
        imag = group["imag"].detach().clone().requires_grad_(True)
        real_index = len(leaves)
        imag_index = len(leaves) + 1
        leaves.extend([real, imag])
        point_pairs.append(
            _PointAccumulatorPair(
                state_field_name=group["field_name"],
                global_indices=tuple(group["global_freq_indices"]),
                point_i=group["point_i"].to(dtype=torch.long),
                point_j=group["point_j"].to(dtype=torch.long),
                point_k=group["point_k"].to(dtype=torch.long),
                real_index=real_index,
                imag_index=imag_index,
            )
        )
        cloned_group["real"] = real
        cloned_group["imag"] = imag
        if "aux_real" in group and "aux_imag" in group:
            aux_real = group["aux_real"].detach().clone().requires_grad_(True)
            aux_imag = group["aux_imag"].detach().clone().requires_grad_(True)
            aux_real_index = len(leaves)
            aux_imag_index = len(leaves) + 1
            leaves.extend([aux_real, aux_imag])
            point_pairs.append(
                _PointAccumulatorPair(
                    state_field_name=f"{group['field_name']}_imag",
                    global_indices=tuple(group["global_freq_indices"]),
                    point_i=group["point_i"].to(dtype=torch.long),
                    point_j=group["point_j"].to(dtype=torch.long),
                    point_k=group["point_k"].to(dtype=torch.long),
                    real_index=aux_real_index,
                    imag_index=aux_imag_index,
                )
            )
            cloned_group["aux_real"] = aux_real
            cloned_group["aux_imag"] = aux_imag
        seed_solver._point_observer_groups[group_key] = cloned_group

    seed_solver._plane_observer_groups = {}
    for group_key, group in getattr(solver, "_plane_observer_groups", {}).items():
        cloned_group = dict(group)
        real = group["real"].detach().clone().requires_grad_(True)
        imag = group["imag"].detach().clone().requires_grad_(True)
        real_index = len(leaves)
        imag_index = len(leaves) + 1
        leaves.extend([real, imag])
        plane_pairs.append(
            _PlaneAccumulatorPair(
                state_field_name=group["field_name"],
                global_indices=tuple(group["global_freq_indices"]),
                axis_code=int(group["axis_code"]),
                plane_index=int(group["plane_index"]),
                real_index=real_index,
                imag_index=imag_index,
            )
        )
        cloned_group["real"] = real
        cloned_group["imag"] = imag
        if "aux_real" in group and "aux_imag" in group:
            aux_real = group["aux_real"].detach().clone().requires_grad_(True)
            aux_imag = group["aux_imag"].detach().clone().requires_grad_(True)
            aux_real_index = len(leaves)
            aux_imag_index = len(leaves) + 1
            leaves.extend([aux_real, aux_imag])
            plane_pairs.append(
                _PlaneAccumulatorPair(
                    state_field_name=f"{group['field_name']}_imag",
                    global_indices=tuple(group["global_freq_indices"]),
                    axis_code=int(group["axis_code"]),
                    plane_index=int(group["plane_index"]),
                    real_index=aux_real_index,
                    imag_index=aux_imag_index,
                )
            )
            cloned_group["aux_real"] = aux_real
            cloned_group["aux_imag"] = aux_imag
        seed_solver._plane_observer_groups[group_key] = cloned_group

    return seed_solver, tuple(leaves), field_pairs, point_pairs, plane_pairs


def _dense_seed_output_pairs(seed_solver):
    runtime = _runtime()
    raw_output = {}
    if getattr(seed_solver, "_dft_entries", ()):
        raw_output.update(
            get_frequency_solution_impl(
                seed_solver,
                all_frequencies=True,
            )
        )
    if getattr(seed_solver, "observers_enabled", False):
        raw_output["observers"] = get_observer_results_impl(seed_solver)
    return runtime._prepare_forward_pack(raw_output)


def _build_output_seeds(
    solver,
    pack,
    grad_outputs,
    *,
    dft_schedule: _ScheduleTensorPack,
    observer_schedule: _ScheduleTensorPack,
) -> _SeedRuntime:
    runtime = _runtime()
    with torch.enable_grad():
        seed_solver, leaves, field_pairs, point_pairs, plane_pairs = _clone_seed_solver(solver)
        seed_pack = _dense_seed_output_pairs(seed_solver)
        if len(seed_pack.output_tensors) != pack.wire_offset:
            raise RuntimeError("Adjoint output pack layout changed between forward and backward.")
        output_grads = tuple(
            torch.zeros_like(output) if grad_output is None else grad_output.to(device=output.device, dtype=output.dtype)
            for output, grad_output in zip(
                seed_pack.output_tensors,
                grad_outputs[: pack.wire_offset],
            )
        )
        leaf_grads = (
            torch.autograd.grad(
                seed_pack.output_tensors,
                leaves,
                grad_outputs=output_grads,
                allow_unused=True,
            )
            if leaves and seed_pack.output_tensors
            else tuple(None for _leaf in leaves)
        )

    dense_seed_records: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
    for pair in field_pairs:
        real_leaf = leaves[pair.real_index]
        imag_leaf = leaves[pair.imag_index]
        dense_seed_records.setdefault(pair.state_field_name, []).append(
            (
                int(pair.entry_index),
                runtime._safe_grad(leaf_grads[pair.real_index], real_leaf),
                runtime._safe_grad(leaf_grads[pair.imag_index], imag_leaf),
            )
        )

    dense_batches = []
    for state_field_name, records in dense_seed_records.items():
        entry_indices, grad_real, grad_imag = _stack_dense_seed_records(records)
        cos_pack, sin_pack = _schedule_pack_for(dft_schedule, entry_indices)
        dense_batches.append(
            _DenseSeedBatch(
                state_field_name=state_field_name,
                entry_indices=entry_indices,
                grad_real=grad_real,
                grad_imag=grad_imag,
                cos_pack=cos_pack,
                sin_pack=sin_pack,
            )
        )

    point_batches = []
    for pair in point_pairs:
        entry_count = len(pair.global_indices)
        real_grad = _reshape_entry_major_grad(
            runtime._safe_grad(leaf_grads[pair.real_index], leaves[pair.real_index]),
            entry_count,
        )
        imag_grad = _reshape_entry_major_grad(
            runtime._safe_grad(leaf_grads[pair.imag_index], leaves[pair.imag_index]),
            entry_count,
        )
        entry_indices = torch.tensor(pair.global_indices, device=real_grad.device, dtype=torch.long)
        cos_pack, sin_pack = _schedule_pack_for(observer_schedule, entry_indices)
        cos_pack, sin_pack = _shift_observer_schedule_for_field(
            cos_pack,
            sin_pack,
            solver=solver,
            entry_indices=entry_indices,
            state_field_name=pair.state_field_name,
        )
        point_i = pair.point_i.to(device=real_grad.device, dtype=torch.long)
        point_j = pair.point_j.to(device=real_grad.device, dtype=torch.long)
        point_k = pair.point_k.to(device=real_grad.device, dtype=torch.long)
        point_batches.append(
            _PointSeedBatch(
                state_field_name=pair.state_field_name,
                entry_indices=entry_indices,
                point_i=point_i,
                point_j=point_j,
                point_k=point_k,
                grad_real=real_grad,
                grad_imag=imag_grad,
                cos_pack=cos_pack,
                sin_pack=sin_pack,
                point_i32=point_i.to(dtype=torch.int32),
                point_j32=point_j.to(dtype=torch.int32),
                point_k32=point_k.to(dtype=torch.int32),
            )
        )

    plane_batches = []
    for pair in plane_pairs:
        entry_count = len(pair.global_indices)
        real_grad = _reshape_entry_major_grad(
            runtime._safe_grad(leaf_grads[pair.real_index], leaves[pair.real_index]),
            entry_count,
        )
        imag_grad = _reshape_entry_major_grad(
            runtime._safe_grad(leaf_grads[pair.imag_index], leaves[pair.imag_index]),
            entry_count,
        )
        entry_indices = torch.tensor(pair.global_indices, device=real_grad.device, dtype=torch.long)
        cos_pack, sin_pack = _schedule_pack_for(observer_schedule, entry_indices)
        cos_pack, sin_pack = _shift_observer_schedule_for_field(
            cos_pack,
            sin_pack,
            solver=solver,
            entry_indices=entry_indices,
            state_field_name=pair.state_field_name,
        )
        plane_batches.append(
            _PlaneSeedBatch(
                state_field_name=pair.state_field_name,
                axis_code=pair.axis_code,
                plane_index=pair.plane_index,
                entry_indices=entry_indices,
                grad_real=real_grad,
                grad_imag=imag_grad,
                cos_pack=cos_pack,
                sin_pack=sin_pack,
            )
        )

    wire_batches = []
    wire_runtime = getattr(solver, "_wire_runtime", None)
    if pack.wire_monitor_templates:
        if wire_runtime is None:
            raise RuntimeError("Wire output seeds require an initialized wire runtime.")
        monitor_state = {
            state["compiled"].name: state for state in wire_runtime.monitor_state
        }
        entries = [
            entry
            for state in wire_runtime.monitor_state
            for entry in state["entries"]
        ]
        time_steps = max((int(entry["end_step"]) for entry in entries), default=0)
        current_samples = torch.zeros(
            (time_steps, wire_runtime.current.numel()),
            device=solver.device,
            dtype=solver.Ex.dtype,
        )
        charge_samples = torch.zeros(
            (time_steps, wire_runtime.charge.numel()),
            device=solver.device,
            dtype=solver.Ex.dtype,
        )
        for monitor_name, template in pack.wire_monitor_templates.items():
            state = monitor_state[monitor_name]
            quantity_indices = template["quantity_indices"]
            for quantity, target, indices_key, shift in (
                ("current", current_samples, "segment_indices", 0.5),
                ("charge", charge_samples, "node_indices", 1.0),
            ):
                output_index = quantity_indices.get(quantity)
                if output_index is None:
                    continue
                output = pack.output_tensors[output_index]
                gradient = grad_outputs[output_index]
                gradient = (
                    torch.zeros_like(output)
                    if gradient is None
                    else gradient.to(device=output.device, dtype=output.dtype)
                )
                cos_rows = []
                sin_rows = []
                for step_index in range(time_steps):
                    cos_values = []
                    sin_values = []
                    for entry in state["entries"]:
                        active = int(entry["start_step"]) <= step_index < int(entry["end_step"])
                        window = (
                            solver._compute_window_weight(
                                step_index,
                                start_step=int(entry["start_step"]),
                                end_step=int(entry["end_step"]),
                                window_type=solver.observer_window_type,
                            )
                            if active
                            else 0.0
                        )
                        normalization = float(entry["window_normalization"])
                        scale = 2.0 * window / normalization if normalization > 0.0 else 0.0
                        omega_dt = (
                            2.0
                            * math.pi
                            * float(entry["frequency"])
                            * float(solver.dt)
                        )
                        angle = omega_dt * (step_index + shift)
                        cos_values.append(scale * math.cos(angle))
                        sin_values.append(scale * math.sin(angle))
                    cos_rows.append(cos_values)
                    sin_rows.append(sin_values)
                cos_weights = torch.tensor(
                    cos_rows, device=solver.device, dtype=solver.Ex.dtype
                )
                sin_weights = torch.tensor(
                    sin_rows, device=solver.device, dtype=solver.Ex.dtype
                )
                sample_gradient = (
                    cos_weights @ gradient.real
                    + sin_weights @ gradient.imag
                )
                indices = state[indices_key].to(device=solver.device, dtype=torch.long)
                target[:, indices] = target[:, indices] + sample_gradient
        wire_batches.append(
            _WireSeedBatch(
                current_samples=current_samples,
                charge_samples=charge_samples,
            )
        )

    port_batches = []
    port_runtime_by_name = {
        runtime.port.name: (index, runtime)
        for index, runtime in enumerate(getattr(solver, "_port_runtimes", ()))
    }
    cursor = int(pack.port_offset)
    for port_name in pack.port_templates:
        port_index, port_runtime = port_runtime_by_name[port_name]
        voltage_output = pack.output_tensors[cursor]
        current_output = pack.output_tensors[cursor + 1]
        voltage_grad = (
            torch.zeros_like(voltage_output)
            if grad_outputs[cursor] is None
            else grad_outputs[cursor].to(
                device=voltage_output.device,
                dtype=voltage_output.dtype,
            )
        )
        current_grad = (
            torch.zeros_like(current_output)
            if grad_outputs[cursor + 1] is None
            else grad_outputs[cursor + 1].to(
                device=current_output.device,
                dtype=current_output.dtype,
            )
        )
        cursor += 2
        available_power_grad = None
        if pack.port_templates[port_name].available_power is not None:
            available_power_output = pack.output_tensors[cursor]
            available_power_grad = (
                torch.zeros_like(available_power_output)
                if grad_outputs[cursor] is None
                else grad_outputs[cursor].to(
                    device=available_power_output.device,
                    dtype=available_power_output.dtype,
                )
            )
            cursor += 1
        weights = port_runtime.window_weights
        if weights is None or port_runtime.accumulator is None:
            raise RuntimeError("Port DFT seed construction requires a completed forward accumulator.")
        steps = torch.arange(
            weights.shape[0],
            device=weights.device,
            dtype=weights.dtype,
        )
        if port_runtime.lumped is not None:
            # Lumped and wire-bound ports: honor the substep macro step and the
            # Yee E/H stagger. Current samples live on the magnetic half-step;
            # wire-bound voltage samples advance to the full electric step.
            macro_dt = port_runtime.lumped.dt * float(port_runtime.substeps)
            magnetic_times = (steps + 0.5) * macro_dt.to(dtype=weights.dtype)
            electric_times = (
                (steps + 1.0) * macro_dt.to(dtype=weights.dtype)
                if port_runtime.wire_provider is not None
                else magnetic_times
            )
            voltage_angles = (
                2.0
                * torch.pi
                * electric_times[:, None]
                * port_runtime.frequencies[None, :]
            )
            magnetic_angles = (
                2.0
                * torch.pi
                * magnetic_times[:, None]
                * port_runtime.frequencies[None, :]
            )
        else:
            # Circuit-coupled ports sample voltage and current at the shared
            # integration midpoint dictated by the circuit runtime.
            coupling = port_runtime.circuit_port.field
            circuit_runtime = next(
                runtime
                for runtime in getattr(solver, "_circuit_runtimes", ())
                if port_runtime.circuit_port in runtime.ports
            )
            fractions = torch.as_tensor(
                tuple(
                    0.5 if integration == "trapezoidal" else 1.0
                    for integration in circuit_runtime.integration_keys
                ),
                device=weights.device,
                dtype=weights.dtype,
            )
            sample_times = (steps + fractions) * coupling.dt.to(dtype=weights.dtype)
            angles = (
                2.0
                * torch.pi
                * sample_times[:, None]
                * port_runtime.frequencies[None, :]
            )
            voltage_angles = angles
            magnetic_angles = angles
        scale = 2.0 / port_runtime.accumulator._window_weight_sum
        weighted_scale = weights * scale[None, :]
        voltage_samples = torch.sum(
            weighted_scale
            * (
                voltage_grad.real[None, :] * torch.cos(voltage_angles)
                + voltage_grad.imag[None, :] * torch.sin(voltage_angles)
            ),
            dim=1,
        )
        current_samples = torch.sum(
            weighted_scale
            * (
                current_grad.real[None, :] * torch.cos(magnetic_angles)
                + current_grad.imag[None, :] * torch.sin(magnetic_angles)
            ),
            dim=1,
        )
        drive_samples = torch.zeros_like(voltage_samples)
        if available_power_grad is not None:
            if port_runtime.drive_accumulator is None:
                raise RuntimeError("Active-port power seed requires a drive DFT accumulator.")
            source_voltage, _ = port_runtime.drive_accumulator.phasors(
                normalization="peak"
            )
            source_voltage_grad = (
                available_power_grad
                * source_voltage
                / (4.0 * port_runtime.lumped.resistance)
            )
            drive_samples = torch.sum(
                weighted_scale
                * (
                    source_voltage_grad.real[None, :] * torch.cos(magnetic_angles)
                    + source_voltage_grad.imag[None, :] * torch.sin(magnetic_angles)
                ),
                dim=1,
            )
        port_batches.append(
            _PortSeedBatch(
                port_index=port_index,
                voltage_samples=voltage_samples,
                current_samples=current_samples,
                drive_samples=drive_samples,
            )
        )

    circuit_batches = []
    if cursor != int(pack.circuit_offset):
        raise RuntimeError("Adjoint port output layout changed before circuit seed construction.")
    for circuit_index, circuit_name in enumerate(pack.circuit_templates):
        template = pack.circuit_templates[circuit_name]
        node_output = pack.output_tensors[cursor]
        branch_output = pack.output_tensors[cursor + 1]
        node_grad = (
            torch.zeros_like(node_output)
            if grad_outputs[cursor] is None
            else grad_outputs[cursor].to(device=node_output.device, dtype=node_output.dtype)
        )
        branch_grad = (
            torch.zeros_like(branch_output)
            if grad_outputs[cursor + 1] is None
            else grad_outputs[cursor + 1].to(
                device=branch_output.device,
                dtype=branch_output.dtype,
            )
        )
        cursor += 2
        device_power_grads = []
        for _device_name in template.device_powers:
            output = pack.output_tensors[cursor]
            gradient = grad_outputs[cursor]
            device_power_grads.append(
                torch.zeros_like(output)
                if gradient is None
                else gradient.to(device=output.device, dtype=output.dtype)
            )
            cursor += 1
        device_power_grad = (
            torch.stack(device_power_grads, dim=1)
            if device_power_grads
            else node_output.new_zeros((node_output.shape[0], 0))
        )
        if template.energy_balance is None:
            energy_grad = node_output.new_zeros((node_output.shape[0],))
        else:
            energy_output = pack.output_tensors[cursor]
            energy_grad = (
                torch.zeros_like(energy_output)
                if grad_outputs[cursor] is None
                else grad_outputs[cursor].to(
                    device=energy_output.device,
                    dtype=energy_output.dtype,
                )
            )
            cursor += 1
        diagnostic_grads = {}
        for diagnostic_name in pack.circuit_diagnostic_names[circuit_name]:
            output = pack.output_tensors[cursor]
            gradient = grad_outputs[cursor]
            diagnostic_grads[diagnostic_name] = (
                torch.zeros_like(output)
                if gradient is None
                else gradient.to(device=output.device, dtype=output.dtype)
            )
            cursor += 1
        port_power_grad = diagnostic_grads["port_powers"]
        field_energy_grad = diagnostic_grads["field_energy_changes"]
        field_energy_grad = field_energy_grad + diagnostic_grads[
            "field_energy_change_total"
        ].unsqueeze(1)
        initial_seeds = (
            node_grad[0],
            branch_grad[0],
            device_power_grad[0],
            energy_grad[0],
            port_power_grad[0],
            field_energy_grad[0],
        )
        if any(bool(torch.any(seed != 0.0)) for seed in initial_seeds):
            raise RuntimeError(
                f"Differentiable CircuitData for {circuit_name!r} does not accept a t=0 "
                "tensor seed; the coupled adjoint starts from the explicitly zero "
                "companion state. Slice samples from index 1 or use port/field outputs."
            )
        balance_increment_grad = torch.flip(
            torch.cumsum(torch.flip(energy_grad, dims=(0,)), dim=0),
            dims=(0,),
        )
        circuit_batches.append(
            _CircuitSeedBatch(
                circuit_index=circuit_index,
                node_samples=node_grad,
                branch_samples=branch_grad,
                device_power_samples=device_power_grad,
                balance_increment_samples=balance_increment_grad,
                port_power_samples=port_power_grad,
                field_energy_samples=field_energy_grad,
                dc_condition=diagnostic_grads["dc_condition"],
                last_condition=diagnostic_grads["last_condition"],
            )
        )

    return _SeedRuntime(
        dft_schedule=dft_schedule,
        observer_schedule=observer_schedule,
        dense_batches=tuple(dense_batches),
        point_batches=tuple(point_batches),
        plane_batches=tuple(plane_batches),
        port_batches=tuple(port_batches),
        circuit_batches=tuple(circuit_batches),
        wire_batches=tuple(wire_batches),
    )


def _port_sample_adjoints(seed_runtime: _SeedRuntime, step_index: int):
    return {
        batch.port_index: (
            batch.voltage_samples[step_index],
            batch.current_samples[step_index],
            batch.drive_samples[step_index],
        )
        for batch in seed_runtime.port_batches
    }


def _circuit_sample_adjoints(seed_runtime: _SeedRuntime, step_index: int):
    return {
        batch.circuit_index: (
            batch.node_samples[step_index + 1],
            batch.branch_samples[step_index + 1],
            batch.device_power_samples[step_index + 1],
            batch.balance_increment_samples[step_index + 1],
            batch.port_power_samples[step_index + 1],
            batch.field_energy_samples[step_index + 1],
            (
                batch.dc_condition
                if step_index == 0
                else torch.zeros_like(batch.dc_condition)
            ),
            (
                batch.last_condition
                if step_index + 2 == batch.node_samples.shape[0]
                else torch.zeros_like(batch.last_condition)
            ),
        )
        for batch in seed_runtime.circuit_batches
    }


def _schedule_pack_for(
    schedule: _ScheduleTensorPack, entry_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the per-entry schedule rows ``(E, T)`` for a seed batch once.

    The native seed-injection kernels read column ``step`` of these packs per
    reverse step, so the (small) per-entry ``index_select`` runs once at seed
    build instead of every step. Degenerate empty schedules collapse to an
    ``(E, 0)`` pack so the native apply skips the batch (zero contribution),
    matching the Torch reference's empty-weight guard.
    """
    if entry_indices.numel() == 0 or schedule.cos.numel() == 0:
        empty = torch.zeros((int(entry_indices.numel()), 0), device=entry_indices.device, dtype=schedule.cos.dtype)
        return empty, empty
    cos = schedule.cos.index_select(0, entry_indices).contiguous()
    sin = schedule.sin.index_select(0, entry_indices).contiguous()
    return cos, sin


def _shift_observer_schedule_for_field(
    cos_pack: torch.Tensor,
    sin_pack: torch.Tensor,
    *,
    solver,
    entry_indices: torch.Tensor,
    state_field_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjoint transpose of the forward observer running-DFT schedule.

    The forward pass accumulates both electric and magnetic observer samples at
    the plain running-DFT step phase (the same phase used by the full-field DFT
    accumulator), so the exact adjoint transpose applies no per-field time
    shift.  The signature is preserved so the seed builder and the stagger
    regression test can pin this identity against a re-introduced offset.
    """

    del solver, entry_indices, state_field_name
    return cos_pack, sin_pack


def _apply_dense_seeds_native(_cuda_backend, adj_state, seed_runtime: _SeedRuntime, step_index):
    for batch in seed_runtime.dense_batches:
        if batch.cos_pack.shape[1] == 0:
            continue
        _cuda_backend._seed_inject_dense(
            AdjField=adj_state[batch.state_field_name],
            GradReal=batch.grad_real,
            GradImag=batch.grad_imag,
            CosPack=batch.cos_pack,
            SinPack=batch.sin_pack,
            step=step_index,
        )


def _apply_point_seeds_native(_cuda_backend, adj_state, seed_runtime: _SeedRuntime, step_index):
    for batch in seed_runtime.point_batches:
        if batch.cos_pack.shape[1] == 0:
            continue
        _cuda_backend._seed_inject_point(
            AdjField=adj_state[batch.state_field_name],
            GradReal=batch.grad_real,
            GradImag=batch.grad_imag,
            PointI=batch.point_i32,
            PointJ=batch.point_j32,
            PointK=batch.point_k32,
            CosPack=batch.cos_pack,
            SinPack=batch.sin_pack,
            step=step_index,
        )


def _apply_plane_seeds_native(_cuda_backend, adj_state, seed_runtime: _SeedRuntime, step_index):
    for batch in seed_runtime.plane_batches:
        if batch.cos_pack.shape[1] == 0:
            continue
        _cuda_backend._seed_inject_plane(
            AdjField=adj_state[batch.state_field_name],
            GradReal=batch.grad_real,
            GradImag=batch.grad_imag,
            CosPack=batch.cos_pack,
            SinPack=batch.sin_pack,
            axis=batch.axis_code,
            planeIndex=batch.plane_index,
            step=step_index,
        )


def _seed_runtime_native_backend(adj_state):
    """Require the CUDA module used by adjoint seed-injection kernels."""
    reference = next(iter(adj_state.values()), None)
    if reference is None or not reference.is_cuda:
        raise RuntimeError("FDTD adjoint seed injection requires CUDA tensors.")
    from ..cuda import backend as _cuda_backend

    if not _cuda_backend.is_available():
        raise RuntimeError("FDTD adjoint seed injection requires the native CUDA extension.")
    return _cuda_backend


def _apply_seed_runtime(adj_state, seed_runtime: _SeedRuntime, step_index):
    cuda_backend = _seed_runtime_native_backend(adj_state)
    _apply_dense_seeds_native(cuda_backend, adj_state, seed_runtime, step_index)
    _apply_point_seeds_native(cuda_backend, adj_state, seed_runtime, step_index)
    _apply_plane_seeds_native(cuda_backend, adj_state, seed_runtime, step_index)
    for batch in seed_runtime.wire_batches:
        cuda_backend._accumulate_in_place(
            dst=adj_state["wire_current"],
            src=batch.current_samples[step_index].contiguous(),
        )
        cuda_backend._accumulate_in_place(
            dst=adj_state["wire_charge"],
            src=batch.charge_samples[step_index].contiguous(),
        )
