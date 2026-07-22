from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import (
    FDTDCheckpointSchema,
    FDTDCheckpointState,
    capture_checkpoint_state,
    restore_checkpoint_state,
    validate_checkpoint_state,
)


_RESUME_SCHEMA_VERSION = 1
_SCHEMA_TUPLE_FIELDS = (
    "field_names",
    "complex_field_names",
    "cpml_state_names",
    "tfsf_auxiliary_state_names",
    "dispersive_state_names",
    "magnetic_dispersive_state_names",
    "lumped_state_names",
    "circuit_state_names",
    "gyromagnetic_state_names",
    "surface_impedance_state_names",
)


def _detach_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "FDTD resume checkpoints only support tensors and primitive containers; "
        f"got {type(value).__name__}."
    )


def _schema_snapshot(schema: FDTDCheckpointSchema) -> dict[str, object]:
    payload: dict[str, object] = {"version": schema.version}
    payload.update({name: getattr(schema, name) for name in _SCHEMA_TUPLE_FIELDS})
    return payload


def _schema_from_snapshot(payload: Mapping[str, Any]) -> FDTDCheckpointSchema:
    if not isinstance(payload, Mapping):
        raise ValueError("FDTD resume physics schema must be a mapping.")
    missing = {"version", *_SCHEMA_TUPLE_FIELDS}.difference(payload)
    if missing:
        raise ValueError(
            "FDTD resume physics schema is missing keys: "
            + ", ".join(sorted(missing))
        )
    return FDTDCheckpointSchema(
        version=int(payload["version"]),
        **{name: tuple(payload[name]) for name in _SCHEMA_TUPLE_FIELDS},
    )


@dataclass(frozen=True)
class FDTDResumeCheckpoint:
    """Detached forward-execution state for resuming a prepared FDTD run."""

    step: int
    total_steps: int
    dt: float
    physics: FDTDCheckpointState
    signature: dict[str, object]
    auxiliary: dict[str, object]

    schema_version = _RESUME_SCHEMA_VERSION

    def _snapshot(self) -> dict[str, object]:
        validate_checkpoint_state(self.physics)
        return _detach_to_cpu(
            {
                "schema_version": self.schema_version,
                "data_type": "FDTDResumeCheckpoint",
                "step": self.step,
                "total_steps": self.total_steps,
                "dt": self.dt,
                "physics_schema": _schema_snapshot(self.physics.schema),
                "physics_tensors": self.physics.tensors,
                "signature": self.signature,
                "auxiliary": self.auxiliary,
            }
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        payload = self._snapshot()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "FDTDResumeCheckpoint":
        payload = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("FDTD resume checkpoint must contain a mapping payload.")
        if payload.get("data_type") != "FDTDResumeCheckpoint":
            raise ValueError("FDTD resume checkpoint has an invalid data_type.")
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError(
                "Unsupported FDTD resume checkpoint schema_version "
                f"{payload.get('schema_version')!r}."
            )
        required = {
            "step",
            "total_steps",
            "dt",
            "physics_schema",
            "physics_tensors",
            "signature",
            "auxiliary",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                "FDTD resume checkpoint is missing required keys: "
                + ", ".join(sorted(missing))
            )
        step = int(payload["step"])
        physics = FDTDCheckpointState(
            step=step,
            schema=_schema_from_snapshot(payload["physics_schema"]),
            tensors=dict(payload["physics_tensors"]),
        )
        validate_checkpoint_state(physics)
        return cls(
            step=step,
            total_steps=int(payload["total_steps"]),
            dt=float(payload["dt"]),
            physics=physics,
            signature=dict(payload["signature"]),
            auxiliary=dict(payload["auxiliary"]),
        )


def _solver_signature(solver) -> dict[str, object]:
    return {
        "field_shapes": {
            name: tuple(getattr(solver, name).shape)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        },
        "field_dtype": str(solver.Ex.dtype),
        "boundary_kind": str(getattr(solver, "boundary_kind", "")),
        "absorber_type": str(getattr(solver, "active_absorber_type", "")),
        "cpml_memory_mode": str(getattr(solver, "_cpml_memory_mode", "none")),
        "requested_port_frequencies": tuple(
            float(value)
            for value in getattr(solver, "_requested_port_frequencies", ())
        ),
        "ports": tuple(
            runtime.port.name for runtime in getattr(solver, "_port_runtimes", ())
        ),
        "circuits": tuple(
            (
                runtime.circuit.name,
                tuple(port.binding.port_name for port in runtime.ports),
                tuple(device.name for device in runtime.circuit.devices),
            )
            for runtime in getattr(solver, "_circuit_runtimes", ())
        ),
        "configuration_fingerprint": _configuration_fingerprint(solver),
    }


def _hash_tensor(hasher, label: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    hasher.update(label.encode("utf-8"))
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(repr(tuple(tensor.shape)).encode("ascii"))
    hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())


def _hash_value(hasher, label: str, value) -> None:
    if isinstance(value, torch.Tensor):
        _hash_tensor(hasher, label, value)
        return
    if isinstance(value, Mapping):
        hasher.update(f"{label}:mapping".encode("utf-8"))
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(hasher, f"{label}.{key!r}", value[key])
        return
    if isinstance(value, (tuple, list)):
        hasher.update(f"{label}:{type(value).__name__}:{len(value)}".encode("utf-8"))
        for index, item in enumerate(value):
            _hash_value(hasher, f"{label}[{index}]", item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        hasher.update(
            f"{label}:dataclass:{type(value).__module__}.{type(value).__qualname__}".encode(
                "utf-8"
            )
        )
        for descriptor in fields(value):
            _hash_value(
                hasher,
                f"{label}.{descriptor.name}",
                getattr(value, descriptor.name),
            )
        return
    hasher.update(f"{label}:{value!r}".encode("utf-8"))


def _configuration_fingerprint(solver) -> str:
    """Hash immutable coefficients and declarative execution inputs."""

    hasher = hashlib.sha256()
    _hash_value(hasher, "dt", float(solver.dt))
    _hash_value(
        hasher,
        "requested_port_frequencies",
        tuple(getattr(solver, "_requested_port_frequencies", ())),
    )
    _hash_value(hasher, "compiled_sources", getattr(solver, "_compiled_sources", ()))
    _hash_value(hasher, "port_excitations", getattr(solver, "_port_excitations", ()))
    static_tensor_prefixes = (
        "eps_",
        "mu_",
        "ce",
        "ch",
        "inv_",
        "cpml_",
        "boundary_phase_",
        "material_",
        "sigma_",
        "modulation_",
    )
    for name, value in sorted(vars(solver).items()):
        if isinstance(value, torch.Tensor) and name.startswith(static_tensor_prefixes):
            _hash_tensor(hasher, f"solver.{name}", value)
    surface = getattr(solver, "_surface_impedance", None)
    if surface is not None:
        for index, write in enumerate(surface["writes"]):
            ade = write.get("ade")
            # Hash the stable descriptor scalars plus, for a generic rational face, the
            # discrete state-space coefficients (A, B, C, D). The per-edge ADE *state*
            # is dynamic and travels in the physics checkpoint, but the coefficients are
            # part of the resume identity: a checkpoint captured against a different
            # rational surface model with the same geometry must fail this fingerprint
            # rather than resume its state under mismatched dynamics.
            descriptor = (
                write["e_name"],
                write["h_name"],
                write["sign"],
                write["axis"],
                write["electric_index"],
                write["magnetic_index"],
                write["full_plane"],
                write["surface_r"],
                ade is not None,
            )
            _hash_value(hasher, f"surface_impedance[{index}]", descriptor)
            if ade is not None:
                for key in ("A", "B", "C", "D"):
                    _hash_value(hasher, f"surface_impedance[{index}].{key}", ade[key])
            # A staircased face carries its exposed-footprint edge mask; it is part of the
            # resume identity so a checkpoint from a different voxelized geometry with the
            # same plane index fails the fingerprint rather than resuming under the wrong
            # footprint.
            mask = write.get("mask")
            if mask is not None:
                _hash_tensor(hasher, f"surface_impedance[{index}].mask", mask)
    for index, runtime in enumerate(getattr(solver, "_port_runtimes", ())):
        _hash_value(hasher, f"port[{index}].definition", runtime.port)
        _hash_value(hasher, f"port[{index}].frequencies", runtime.frequencies)
        _hash_value(hasher, f"port[{index}].source_kind", runtime.source_kind)
        _hash_value(hasher, f"port[{index}].source_frequency", runtime.source_frequency)
        _hash_value(hasher, f"port[{index}].source_fwidth", runtime.source_fwidth)
        _hash_value(hasher, f"port[{index}].source_phase", runtime.source_phase)
        _hash_value(hasher, f"port[{index}].source_delay", runtime.source_delay)
        _hash_value(hasher, f"port[{index}].source_amplitude", runtime.source_amplitude)
        coupling = runtime.circuit_port.field if runtime.circuit_port is not None else None
        if coupling is not None:
            for name in (
                "linear_indices",
                "voltage_weights",
                "injection",
                "coupling_impedance",
                "discrete_port_impedance",
            ):
                _hash_value(
                    hasher,
                    f"port[{index}].coupling.{name}",
                    getattr(coupling, name),
                )
    for index, runtime in enumerate(getattr(solver, "_circuit_runtimes", ())):
        circuit = runtime.circuit
        _hash_value(hasher, f"circuit[{index}].parameters", circuit.parameters)
        _hash_value(
            hasher,
            f"circuit[{index}].initial_conditions",
            circuit.initial_conditions,
        )
        _hash_value(hasher, f"circuit[{index}].devices", circuit.devices)
        _hash_value(hasher, f"circuit[{index}].config", circuit.config)
        _hash_value(
            hasher,
            f"circuit[{index}].bindings",
            tuple(
                (binding.port_name, binding.positive.name, binding.negative.name)
                for binding in circuit.bindings
            ),
        )
    return hasher.hexdigest()


def _accumulator_snapshot(accumulator) -> dict[str, object] | None:
    if accumulator is None:
        return None
    return {
        "voltage_sum": accumulator._voltage_sum.detach().clone(),
        "current_sum": accumulator._current_sum.detach().clone(),
        "window_weight_sum": accumulator._window_weight_sum.detach().clone(),
        "sample_count": int(accumulator._sample_count),
    }


def _circuit_snapshot(runtime, step: int) -> dict[str, object]:
    count = step + 1
    return {
        "name": runtime.circuit.name,
        "step_index": int(runtime.step_index),
        "previous_stored_energy": runtime.previous_stored_energy.detach().clone(),
        "last_condition": getattr(runtime, "last_condition", runtime.dc_condition)
        .detach()
        .clone(),
        "node_samples": runtime.node_samples[:count].detach().clone(),
        "branch_samples": runtime.branch_samples[:count].detach().clone(),
        "device_power_samples": {
            name: values[:count].detach().clone()
            for name, values in runtime.device_power_samples.items()
        },
        "energy_balance_samples": runtime.energy_balance_samples[:count]
        .detach()
        .clone(),
        "port_power_samples": runtime.port_power_samples[:count].detach().clone(),
        "field_energy_change_samples": runtime.field_energy_change_samples[:count]
        .detach()
        .clone(),
        "integration_keys": runtime.integration_keys,
        "switch_keys": runtime.switch_keys,
        "source_values": {
            name: values.detach().clone()
            for name, values in runtime.source_values.items()
        },
    }


def capture_resume_checkpoint(
    solver,
    *,
    step: int,
    total_steps: int,
) -> FDTDResumeCheckpoint:
    if not 0 <= int(step) < int(total_steps):
        raise ValueError("Resume checkpoint step must satisfy 0 <= step < total_steps.")
    ports = []
    for runtime in getattr(solver, "_port_runtimes", ()):
        if runtime.sample_index != step:
            raise RuntimeError(
                f"Port {runtime.port.name!r} recorded {runtime.sample_index} samples "
                f"at checkpoint step {step}."
            )
        ports.append(
            {
                "name": runtime.port.name,
                "sample_index": int(runtime.sample_index),
                "electric_time": runtime.electric_time.detach().clone(),
                "magnetic_time": runtime.magnetic_time.detach().clone(),
                "drive_buffer": runtime.drive_buffer.detach().clone(),
                "accumulator": _accumulator_snapshot(runtime.accumulator),
                "drive_accumulator": _accumulator_snapshot(runtime.drive_accumulator),
            }
        )
    circuits = []
    for runtime in getattr(solver, "_circuit_runtimes", ()):
        if runtime.step_index != step:
            raise RuntimeError(
                f"Circuit {runtime.circuit.name!r} completed {runtime.step_index} "
                f"steps at checkpoint step {step}."
            )
        circuits.append(_circuit_snapshot(runtime, step))
    mur = [
        {
            "field": entry["field"],
            "axis": entry["axis"],
            "boundary_index": entry["boundary_index"],
            "adjacent_index": entry["adjacent_index"],
            "prev_boundary": entry["prev_boundary"].detach().clone(),
            "prev_adjacent": entry["prev_adjacent"].detach().clone(),
        }
        for entry in getattr(solver, "_mur_state", ())
    ]
    modulation_time = getattr(solver, "_modulation_time", None)
    auxiliary = {
        "ports": ports,
        "circuits": circuits,
        "mur": mur,
        "modulation_time": (
            None if modulation_time is None else modulation_time.detach().clone()
        ),
        "shutoff_peak": solver._shutoff_peak.detach().clone(),
        "shutoff_triggered": bool(solver._shutoff_triggered),
        "shutoff_step": solver._shutoff_step,
    }
    physics = capture_checkpoint_state(solver, step=step)
    return FDTDResumeCheckpoint(
        step=int(step),
        total_steps=int(total_steps),
        dt=float(solver.dt),
        physics=physics,
        signature=_solver_signature(solver),
        auxiliary=auxiliary,
    )


def _require_tensor_compatible(name: str, source, target) -> None:
    if not isinstance(source, torch.Tensor):
        raise TypeError(f"Resume state {name!r} must be a tensor.")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"Resume state {name!r} has shape {tuple(source.shape)}, "
            f"expected {tuple(target.shape)}."
        )
    if source.dtype != target.dtype:
        raise TypeError(
            f"Resume state {name!r} has dtype {source.dtype}, expected {target.dtype}."
        )


def _preflight_accumulator(
    name: str,
    payload,
    accumulator,
    copy_ops: list[tuple[torch.Tensor, torch.Tensor]],
) -> int | None:
    if payload is None:
        if accumulator is not None:
            raise ValueError(f"Resume state {name!r} is missing an accumulator.")
        return None
    if accumulator is None or not isinstance(payload, Mapping):
        raise ValueError(f"Resume state {name!r} has an incompatible accumulator.")
    for key, target in (
        ("voltage_sum", accumulator._voltage_sum),
        ("current_sum", accumulator._current_sum),
        ("window_weight_sum", accumulator._window_weight_sum),
    ):
        source = payload.get(key)
        _require_tensor_compatible(f"{name}.{key}", source, target)
        copy_ops.append((target, source))
    count = int(payload.get("sample_count", -1))
    if count < 0:
        raise ValueError(f"Resume state {name!r} has an invalid sample_count.")
    return count


def _same_schedule(current: torch.Tensor, saved) -> bool:
    return (
        isinstance(saved, torch.Tensor)
        and saved.dtype == current.dtype
        and tuple(saved.shape) == tuple(current.shape)
        and torch.equal(current, saved.to(device=current.device))
    )


def _preflight_auxiliary(solver, checkpoint: FDTDResumeCheckpoint):
    auxiliary = checkpoint.auxiliary
    if not isinstance(auxiliary, Mapping):
        raise ValueError("FDTD resume auxiliary state must be a mapping.")
    copy_ops: list[tuple[torch.Tensor, torch.Tensor]] = []
    host_updates: list[tuple[object, str, object]] = []

    port_payloads = auxiliary.get("ports")
    ports = tuple(getattr(solver, "_port_runtimes", ()))
    if not isinstance(port_payloads, (list, tuple)) or len(port_payloads) != len(ports):
        raise ValueError("FDTD resume port layout does not match the prepared solver.")
    for index, (runtime, payload) in enumerate(zip(ports, port_payloads)):
        if not isinstance(payload, Mapping) or payload.get("name") != runtime.port.name:
            raise ValueError(f"FDTD resume port ordering mismatch at index {index}.")
        if int(payload.get("sample_index", -1)) != checkpoint.step:
            raise ValueError(
                f"FDTD resume port {runtime.port.name!r} has an invalid sample index."
            )
        for key, target in (
            ("electric_time", runtime.electric_time),
            ("magnetic_time", runtime.magnetic_time),
            ("drive_buffer", runtime.drive_buffer),
        ):
            source = payload.get(key)
            _require_tensor_compatible(f"ports.{index}.{key}", source, target)
            copy_ops.append((target, source))
        sample_count = _preflight_accumulator(
            f"ports.{index}.accumulator",
            payload.get("accumulator"),
            runtime.accumulator,
            copy_ops,
        )
        drive_count = _preflight_accumulator(
            f"ports.{index}.drive_accumulator",
            payload.get("drive_accumulator"),
            runtime.drive_accumulator,
            copy_ops,
        )
        if sample_count != checkpoint.step or (
            drive_count is not None and drive_count != checkpoint.step
        ):
            raise ValueError(
                f"FDTD resume port {runtime.port.name!r} accumulator count mismatch."
            )
        host_updates.append((runtime, "sample_index", checkpoint.step))
        host_updates.append((runtime.accumulator, "_sample_count", sample_count))
        if runtime.drive_accumulator is not None:
            host_updates.append(
                (runtime.drive_accumulator, "_sample_count", drive_count)
            )

    circuit_payloads = auxiliary.get("circuits")
    circuits = tuple(getattr(solver, "_circuit_runtimes", ()))
    if not isinstance(circuit_payloads, (list, tuple)) or len(circuit_payloads) != len(circuits):
        raise ValueError("FDTD resume circuit layout does not match the prepared solver.")
    count = checkpoint.step + 1
    for index, (runtime, payload) in enumerate(zip(circuits, circuit_payloads)):
        if not isinstance(payload, Mapping) or payload.get("name") != runtime.circuit.name:
            raise ValueError(f"FDTD resume circuit ordering mismatch at index {index}.")
        if int(payload.get("step_index", -1)) != checkpoint.step:
            raise ValueError(
                f"FDTD resume circuit {runtime.circuit.name!r} has an invalid step index."
            )
        if tuple(payload.get("integration_keys", ())) != runtime.integration_keys:
            raise ValueError(
                f"FDTD resume circuit {runtime.circuit.name!r} integration schedule changed."
            )
        if tuple(tuple(item) for item in payload.get("switch_keys", ())) != runtime.switch_keys:
            raise ValueError(
                f"FDTD resume circuit {runtime.circuit.name!r} switch schedule changed."
            )
        source_values = payload.get("source_values")
        if not isinstance(source_values, Mapping) or set(source_values) != set(runtime.source_values):
            raise ValueError(
                f"FDTD resume circuit {runtime.circuit.name!r} source schedule changed."
            )
        if any(
            not _same_schedule(runtime.source_values[name], source_values[name])
            for name in runtime.source_values
        ):
            raise ValueError(
                f"FDTD resume circuit {runtime.circuit.name!r} source schedule changed."
            )
        tensor_pairs = (
            ("node_samples", runtime.node_samples[:count]),
            ("branch_samples", runtime.branch_samples[:count]),
            ("energy_balance_samples", runtime.energy_balance_samples[:count]),
            ("port_power_samples", runtime.port_power_samples[:count]),
            (
                "field_energy_change_samples",
                runtime.field_energy_change_samples[:count],
            ),
        )
        for key, target in tensor_pairs:
            source = payload.get(key)
            _require_tensor_compatible(f"circuits.{index}.{key}", source, target)
            copy_ops.append((target, source))
        power_payload = payload.get("device_power_samples")
        if not isinstance(power_payload, Mapping) or set(power_payload) != set(runtime.device_power_samples):
            raise ValueError(
                f"FDTD resume circuit {runtime.circuit.name!r} device-power layout changed."
            )
        for name, values in runtime.device_power_samples.items():
            target = values[:count]
            source = power_payload[name]
            _require_tensor_compatible(
                f"circuits.{index}.device_power_samples.{name}", source, target
            )
            copy_ops.append((target, source))
        previous_energy = payload.get("previous_stored_energy")
        last_condition = payload.get("last_condition")
        _require_tensor_compatible(
            f"circuits.{index}.previous_stored_energy",
            previous_energy,
            runtime.previous_stored_energy,
        )
        _require_tensor_compatible(
            f"circuits.{index}.last_condition",
            last_condition,
            runtime.dc_condition,
        )
        # Keep the persistent tensor identity captured by circuit CUDA Graphs.
        copy_ops.append((runtime.previous_stored_energy, previous_energy))
        host_updates.extend(
            (
                (runtime, "step_index", checkpoint.step),
                (
                    runtime,
                    "last_condition",
                    last_condition.to(
                        device=runtime.system.device, dtype=runtime.system.dtype
                    ).clone(),
                ),
            )
        )

    mur_payloads = auxiliary.get("mur")
    mur = tuple(getattr(solver, "_mur_state", ()))
    if not isinstance(mur_payloads, (list, tuple)) or len(mur_payloads) != len(mur):
        raise ValueError("FDTD resume Mur boundary layout changed.")
    for index, (entry, payload) in enumerate(zip(mur, mur_payloads)):
        identity = ("field", "axis", "boundary_index", "adjacent_index")
        if not isinstance(payload, Mapping) or any(payload.get(key) != entry[key] for key in identity):
            raise ValueError(f"FDTD resume Mur boundary ordering mismatch at index {index}.")
        for key in ("prev_boundary", "prev_adjacent"):
            source = payload.get(key)
            target = entry[key]
            _require_tensor_compatible(f"mur.{index}.{key}", source, target)
            copy_ops.append((target, source))

    modulation_source = auxiliary.get("modulation_time")
    modulation_target = getattr(solver, "_modulation_time", None)
    if (modulation_source is None) != (modulation_target is None):
        raise ValueError("FDTD resume modulation layout changed.")
    if modulation_target is not None:
        _require_tensor_compatible(
            "modulation_time", modulation_source, modulation_target
        )
        copy_ops.append((modulation_target, modulation_source))

    shutoff_peak = auxiliary.get("shutoff_peak")
    _require_tensor_compatible("shutoff_peak", shutoff_peak, solver._shutoff_peak)
    copy_ops.append((solver._shutoff_peak, shutoff_peak))
    host_updates.extend(
        (
            (solver, "_shutoff_triggered", bool(auxiliary.get("shutoff_triggered"))),
            (solver, "_shutoff_step", auxiliary.get("shutoff_step")),
        )
    )
    return copy_ops, host_updates


def preflight_resume_checkpoint(
    solver,
    checkpoint: FDTDResumeCheckpoint,
    *,
    total_steps: int,
) -> None:
    if not isinstance(checkpoint, FDTDResumeCheckpoint):
        raise TypeError("resume_from must be an FDTDResumeCheckpoint.")
    if checkpoint.total_steps != int(total_steps):
        raise ValueError(
            f"Resume checkpoint planned {checkpoint.total_steps} steps, "
            f"but this simulation plans {total_steps}."
        )
    if checkpoint.physics.step != checkpoint.step:
        raise ValueError("FDTD resume physics and execution step indices disagree.")
    if not 0 <= checkpoint.step < checkpoint.total_steps:
        raise ValueError("FDTD resume checkpoint step is outside the planned run.")
    if checkpoint.dt != float(solver.dt):
        raise ValueError(
            f"FDTD resume checkpoint dt={checkpoint.dt!r} does not match "
            f"the prepared solver dt={float(solver.dt)!r}."
        )
    if checkpoint.signature != _solver_signature(solver):
        raise ValueError("FDTD resume checkpoint does not match the prepared solver layout.")
    if bool(checkpoint.auxiliary.get("shutoff_triggered")):
        raise ValueError("A checkpoint captured after field shutoff cannot be resumed.")


def restore_resume_checkpoint(
    solver,
    checkpoint: FDTDResumeCheckpoint,
    *,
    total_steps: int,
) -> int:
    preflight_resume_checkpoint(
        solver,
        checkpoint,
        total_steps=total_steps,
    )
    copy_ops, host_updates = _preflight_auxiliary(solver, checkpoint)
    restore_checkpoint_state(solver, checkpoint.physics)
    for target, source in copy_ops:
        target.copy_(source)
    for target, name, value in host_updates:
        setattr(target, name, value)
    return checkpoint.step


__all__ = [
    "FDTDResumeCheckpoint",
    "capture_resume_checkpoint",
    "preflight_resume_checkpoint",
    "restore_resume_checkpoint",
]
