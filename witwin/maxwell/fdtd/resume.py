from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..media import Material
from .checkpoint import (
    FDTDCheckpointSchema,
    FDTDCheckpointState,
    capture_checkpoint_state,
    restore_checkpoint_state,
    validate_checkpoint_state,
)


_RESUME_SCHEMA_VERSION = 6
_SCHEMA_TUPLE_FIELDS = tuple(field.name for field in fields(FDTDCheckpointSchema) if field.name != "version")


def _detach_to_cpu(value):
    if isinstance(value, Enum):
        return _detach_to_cpu(value.value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    raise TypeError(
        "FDTD resume checkpoints only support tensors and primitive containers; "
        f"got {type(value).__name__}."
    )


_SCHEMA_ADDED_FIELDS = ("thin_sheet_state_names",)


def _schema_snapshot(schema: FDTDCheckpointSchema) -> dict[str, object]:
    payload: dict[str, object] = {"version": schema.version}
    payload.update({name: getattr(schema, name) for name in _SCHEMA_TUPLE_FIELDS})
    return payload


def _schema_from_snapshot(payload: Mapping[str, Any]) -> FDTDCheckpointSchema:
    if not isinstance(payload, Mapping):
        raise ValueError("FDTD resume physics schema must be a mapping.")
    # Fields added after checkpoints already existed. A checkpoint written before
    # a field existed cannot hold that state (conformal-sheet resume was rejected
    # before sheet feeds), so its absence means exactly the empty default.
    payload = {**{name: () for name in _SCHEMA_ADDED_FIELDS}, **payload}
    missing = {"version", *_SCHEMA_TUPLE_FIELDS}.difference(payload)
    if missing:
        raise ValueError(
            "FDTD resume physics schema is missing keys: "
            + ", ".join(sorted(missing))
        )
    unexpected = set(payload).difference({"version", *_SCHEMA_TUPLE_FIELDS})
    if unexpected:
        raise ValueError(f"FDTD resume physics schema has unexpected keys: {sorted(unexpected)}.")
    for name in _SCHEMA_TUPLE_FIELDS:
        if not isinstance(payload[name], (tuple, list)) or not all(isinstance(value, str) for value in payload[name]):
            raise ValueError(f"FDTD resume physics schema {name!r} must be a sequence of names.")
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
        auxiliary = dict(payload["auxiliary"])
        modulation_time = auxiliary.get("modulation_time")
        if modulation_time is not None and "modulation_step" not in auxiliary:
            # The saved step counts completed updates. Old FP32 accumulated times
            # are lossy; reconstruct the intended endpoints from that exact index.
            dt = float(payload["dt"])
            auxiliary["modulation_time"] = modulation_time.new_tensor(
                [((step - 2) if step else -1) * dt, (step - 1) * dt],
                dtype=torch.float64,
            )
            auxiliary["modulation_step"] = modulation_time.new_tensor([step - 1], dtype=torch.int64)
        return cls(
            step=step,
            total_steps=int(payload["total_steps"]),
            dt=float(payload["dt"]),
            physics=physics,
            signature=dict(payload["signature"]),
            auxiliary=auxiliary,
        )


def _additive_j_signature(solver) -> dict[str, str]:
    from .distributed.resume import is_distributed_solver

    if is_distributed_solver(solver):
        present = any(_additive_j_signature(shard.solver) for shard in solver.shards)
    else:
        present = any(
            source["kind"] in {"point_dipole", "uniform_current"}
            or (source["kind"] == "custom_current"
                and any(key.startswith("J") for key in source["dataset"].components))
            for source in getattr(solver, "_compiled_sources", ())
        )
    return {"additive_j_injection": "n+1/2"} if present else {}


def _preflight_additive_j_timing(solver, signature) -> None:
    # Before this metadata existed, additive electric currents sampled at n*dt.
    if _additive_j_signature(solver) and signature.get("additive_j_injection", "n") != "n+1/2":
        raise NotImplementedError(
            "checkpoint predates the additive-J n+1/2 source timing; re-run from the start"
        )


def _pulsed_tfsf_signature(solver) -> dict[str, str]:
    from .distributed.resume import is_distributed_solver

    if is_distributed_solver(solver):
        present = any(_pulsed_tfsf_signature(shard.solver) for shard in solver.shards)
    else:
        present = (getattr(solver, "_tfsf_state", None) or {}).get("incident_reference") is not None
    return {"pulsed_tfsf_reference": "physical_entry_plane"} if present else {}


def _preflight_pulsed_tfsf_reference(solver, signature) -> None:
    if (_pulsed_tfsf_signature(solver)
            and signature.get("pulsed_tfsf_reference") != "physical_entry_plane"):
        raise NotImplementedError(
            "checkpoint predates the pulsed TFSF physical entry-plane reference; re-run from the start"
        )


def _observer_origin_signature(solver):
    return dict(observer_spectral_origin="physical_yee_v1",
                source_reference_label="provider_reference_v1",
                normalization_reference=getattr(solver, "_normalization_reference", {}),
                normalization_label_offset=getattr(solver, "_normalization_label_offset", None))


def _tfsf_termination_signature(solver):
    from .distributed.resume import is_distributed_solver

    if is_distributed_solver(solver):
        present = any(_tfsf_termination_signature(shard.solver) for shard in solver.shards)
    else:
        present = (getattr(solver, "_tfsf_state", None) or {}).get("auxiliary_grid") is not None
    return {"tfsf_auxiliary_termination": "appended_320_p2_r1e-8_v1"} if present else {}


def _preflight_tfsf_termination(solver, signature):
    expected = _tfsf_termination_signature(solver)
    if any(signature.get(key) != value for key, value in expected.items()):
        raise NotImplementedError(
            "checkpoint predates the qualified TFSF auxiliary-line termination; re-run from the start"
        )


def _tfsf_timing_signature(solver):
    from .distributed.resume import is_distributed_solver

    if is_distributed_solver(solver):
        present = any(_tfsf_timing_signature(shard.solver) for shard in solver.shards)
    else:
        present = any(
            source.get("reference_provider_kind") in {
                "plane_wave_ref_x_ez", "plane_wave_axis_aligned", "plane_wave_aux"}
            or (source.get("reference_provider_kind") in {
                "plane_wave_discrete_cw", "plane_wave_grating_slab_cw", "analytic_profile"}
                and source["source_time"]["kind"] == "cw")
            for source in getattr(solver, "_compiled_sources", ()))
    return {"tfsf_incident_timing": "physical_yee_v2"} if present else {}


def _preflight_tfsf_timing(solver, signature):
    if (_tfsf_timing_signature(solver)
            and signature.get("tfsf_incident_timing") != "physical_yee_v2"):
        raise NotImplementedError(
            "checkpoint predates the physical TFSF incident timing; re-run from the start")


def _preflight_observer_origin(solver, signature):
    expected = _observer_origin_signature(solver)
    if any(signature.get(key) != value for key, value in expected.items()):
        raise NotImplementedError(
            "checkpoint lacks the physical Yee observer origin or compiled provider reference; re-run from the start")


def _solver_signature(solver) -> dict[str, object]:
    return {
        **_additive_j_signature(solver),
        **_pulsed_tfsf_signature(solver),
        **_tfsf_termination_signature(solver),
        **_tfsf_timing_signature(solver),
        **_observer_origin_signature(solver),
        "field_shapes": {
            name: tuple(getattr(solver, name).shape)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        },
        "field_dtype": str(solver.Ex.dtype),
        **({"observer_dtype": str(solver._observer_dtype)}
           if getattr(solver, "_observer_dtype", torch.float32) != torch.float32 else {}),
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


def _is_unset_boundary_extension(value, name: str) -> bool:
    """One-way ABC fields added to ``BoundarySpec`` are hashed only when set.

    Omitting them at their defaults keeps the fingerprint of every configuration
    that predates them byte-identical, so earlier disk checkpoints still resume.
    """
    from ..absorbers import MurABC
    from ..scene import BoundarySpec

    if not isinstance(value, BoundarySpec):
        return False
    if name == "mur_abc":
        return value.mur_abc == MurABC()
    if name == "mode_abc":
        return value.mode_abc is None
    return False


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
            # Registry IDs depend on construction order, not constitutive physics.
            # Surface ADE coefficients are fingerprinted separately below.
            if isinstance(value, Material) and descriptor.name == "material_id":
                continue
            if _is_unset_boundary_extension(value, descriptor.name):
                continue
            _hash_value(
                hasher,
                f"{label}.{descriptor.name}",
                getattr(value, descriptor.name),
            )
        return
    hasher.update(f"{label}:{value!r}".encode("utf-8"))


# Compiled material-model keys introduced after resume checkpoints were already in
# use. They are hashed only when set, so scenes without the feature keep their
# earlier configuration fingerprint and existing disk checkpoints still resume.
_OPTIONAL_MATERIAL_MODEL_KEYS = ("pmc_occupancy",)


def _without_unset_material_keys(model):
    if not isinstance(model, Mapping):
        return model
    unset = [key for key in _OPTIONAL_MATERIAL_MODEL_KEYS if key in model and model[key] is None]
    if not unset:
        return model
    return {key: value for key, value in model.items() if key not in unset}


# Read-side classifications derived from the fingerprinted sources; excluding them
# keeps checkpoints written before the flag existed resumable.
_DERIVED_TIME_PLAN_KEYS = ("mixed_sources",)


def _without_derived_time_plan_keys(plan):
    if not isinstance(plan, Mapping):
        return plan
    return {key: value for key, value in plan.items() if key not in _DERIVED_TIME_PLAN_KEYS}


def _configuration_fingerprint(solver) -> str:
    """Hash immutable coefficients and declarative execution inputs."""

    hasher = hashlib.sha256()
    _hash_value(hasher, "dt", float(solver.dt))
    timing = _tfsf_timing_signature(solver)
    if timing:
        _hash_value(hasher, "tfsf_incident_timing", timing)
    termination = _tfsf_termination_signature(solver)
    if termination:
        _hash_value(hasher, "tfsf_auxiliary_termination", termination)
    chart=getattr(solver.scene,"coordinates",None)
    _hash_value(hasher, "coordinate_frame",None if chart is None else chart.snapshot())
    _hash_value(hasher, "time_plan", _without_derived_time_plan_keys(getattr(solver, "_time_plan", None)))
    _hash_value(
        hasher,
        "requested_port_frequencies",
        tuple(getattr(solver, "_requested_port_frequencies", ())),
    )
    _hash_value(hasher, "compiled_sources", tuple(
        {key: value for key, value in source.items() if not key.startswith("reference_")}
        for source in getattr(solver, "_compiled_sources", ())))
    _hash_value(hasher, "port_excitations", getattr(solver, "_port_excitations", ()))
    _hash_value(hasher, "material_model", _without_unset_material_keys(getattr(solver, "_compiled_material_model", None)))
    _hash_value(hasher, "gyromagnetic_layout", getattr(solver, "_gyromagnetic_layout", None))
    _hash_value(hasher, "boundary", solver.scene.boundary)
    if getattr(solver.scene, "internal_absorbers", ()):
        _hash_value(hasher, "internal_absorbers", tuple(solver.scene.internal_absorbers))
    _hash_value(hasher, "monitors", solver.scene.monitors)
    breakdown = getattr(solver, "breakdown_runtime", None)
    if breakdown is not None:
        _hash_value(hasher, "breakdown.layout", breakdown["layout"])
        _hash_value(hasher, "breakdown.edge_sets", breakdown["edge_sets"])
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
        if isinstance(value, torch.Tensor) and name.startswith(static_tensor_prefixes) and not name.endswith("_dynamic"):
            if breakdown is not None and name in {
                "cex_decay", "cex_curl", "cey_decay", "cey_curl", "cez_decay", "cez_curl"
            }:
                # Conductivity history rewrites only capable edges. Hash their
                # original coefficients while preserving every other edge value.
                data = breakdown["edge_sets"].get("E" + name[2])
                if data is not None:
                    value = value.clone()
                    value.reshape(-1).index_copy_(0, data["edge_flat"], data["base_" + name[4:]])
            _hash_tensor(hasher, f"solver.{name}", value)
    for index, runtime in enumerate(getattr(solver, "_network_runtimes", ())):
        _hash_value(hasher, f"network[{index}].compiled", runtime.compiled)
        for name in ("A", "B", "C", "D", "gain_state", "gain_voltage", "reference_impedance"):
            _hash_value(hasher, f"network[{index}].{name}", getattr(runtime, name))
    wire = getattr(solver, "_wire_runtime", None)
    if wire is not None:
        _hash_value(hasher, "wire.network", wire.network)
        _hash_value(hasher, "wire.coefficients", wire.coefficients)
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
    cut_surface = getattr(solver, "_cut_surface", None)
    if cut_surface is not None:
        projection = cut_surface["projection"]
        _hash_tensor(hasher, "cut_surface.indices", projection.matrix.indices())
        # Coalesced GPU sums can differ in their last float64 bit. Identity is
        # the exact physical geometry and metric, not a reduction's summation
        # order. This still rejects every changed physical vertex/topology.
        for index, structure in enumerate(solver.scene.structures):
            if getattr(structure.geometry, "kind", None) == "surface_mesh":
                for name in ("vertices", "triangles", "position", "rotation", "physical_thickness"):
                    _hash_value(hasher, f"cut_surface.geometry[{index}].{name}", getattr(structure.geometry, name))
        _hash_tensor(hasher, "cut_surface.metric", projection.sqrt_weight)
        for name, group in cut_surface["groups"].items():
            _hash_tensor(hasher, f"cut_surface.{name}.indices", group["linear_indices"])
    for index, entry in enumerate(getattr(solver, "_copper_foils", ())):
        for key in ("face", "indices", "rates", "B", "D", "inverse", "rhs_matrix", "area"):
            _hash_value(hasher, f"copper_foil[{index}].{key}", entry[key])
    for index, sheet in enumerate(getattr(solver, "_thin_sheets", ())):
        if sheet.complex.feeds:
            for name, value in sorted(vars(sheet.complex).items()):
                if isinstance(value, np.ndarray):
                    _hash_tensor(hasher, f"thin_sheet[{index}].{name}", torch.from_numpy(value))
            _hash_value(hasher, f"thin_sheet[{index}].material", sheet.complex.material)
            for feed_index, feed in enumerate(sheet.feeds):
                _hash_tensor(hasher, f"thin_sheet[{index}].feed[{feed_index}].resistance", feed["binding"].resistance)
                for column_index, column in enumerate(feed["columns"]):
                    label = f"thin_sheet[{index}].feed[{feed_index}].column[{column_index}]"
                    _hash_tensor(hasher, label+".ids", column["ids"])
                    _hash_tensor(hasher, label+".u", column["u"])
                    _hash_value(hasher, label+".conductance", column["conductance"])
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
    from .distributed.resume import is_distributed_solver, capture_distributed_resume
    if is_distributed_solver(solver):
        return capture_distributed_resume(solver,step=step,total_steps=total_steps)
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
        if getattr(solver, "complex_fields_enabled", False):
            ports[-1]["accumulator_imag"] = _accumulator_snapshot(runtime.accumulator_imag)
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
        }
        for entry in getattr(solver, "_mur_state", ())
    ]
    modulation_step = getattr(solver, "_modulation_step", None)
    auxiliary = {
        "ports": ports,
        "circuits": circuits,
        "mur": mur,
        "modulation_step": (
            None if modulation_step is None else modulation_step.detach().clone()
        ),
        "nonlinear_peak": (
            None if getattr(solver, "_nonlinear_peak", None) is None else solver._nonlinear_peak.detach().clone()
        ),
        "shutoff_peak": solver._shutoff_peak.detach().clone(),
        "shutoff_triggered": bool(solver._shutoff_triggered),
        "shutoff_step": solver._shutoff_step,
    }
    from .runtime.passivity import capture_passivity
    auxiliary["passivity"] = capture_passivity(solver)
    from .observation_resume import capture_observations
    auxiliary["observations"] = capture_observations(solver)
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
        if getattr(solver, "complex_fields_enabled", False):
            imaginary_count = _preflight_accumulator(
                f"ports.{index}.accumulator_imag",
                payload.get("accumulator_imag"), runtime.accumulator_imag, copy_ops,
            )
            if imaginary_count != checkpoint.step:
                raise ValueError(f"FDTD resume port {runtime.port.name!r} imaginary accumulator count mismatch.")
            host_updates.append((runtime.accumulator_imag, "_sample_count", imaginary_count))
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

    modulation_source = auxiliary.get("modulation_step")
    modulation_target = getattr(solver, "_modulation_step", None)
    if (modulation_source is None) != (modulation_target is None):
        raise ValueError("FDTD resume modulation layout changed.")
    if modulation_target is not None:
        _require_tensor_compatible(
            "modulation_step", modulation_source, modulation_target
        )
        copy_ops.append((modulation_target, modulation_source))
        # Rebuild both FP64 endpoints from the exact counter, including the
        # initial (-dt, -dt) clock before any update has completed.
        endpoints = torch.cat(((modulation_source - 1).clamp_min(-1), modulation_source))
        copy_ops.append((solver._modulation_time, endpoints.to(dtype=torch.float64) * solver.dt))

    nonlinear_source = auxiliary.get("nonlinear_peak")
    nonlinear_target = getattr(solver, "_nonlinear_peak", None)
    if (nonlinear_source is None) != (nonlinear_target is None):
        raise ValueError("FDTD resume nonlinear stability envelope changed.")
    if nonlinear_target is not None:
        _require_tensor_compatible("nonlinear_peak", nonlinear_source, nonlinear_target)
        copy_ops.append((nonlinear_target, nonlinear_source))

    shutoff_peak = auxiliary.get("shutoff_peak")
    _require_tensor_compatible("shutoff_peak", shutoff_peak, solver._shutoff_peak)
    copy_ops.append((solver._shutoff_peak, shutoff_peak))
    host_updates.extend(
        (
            (solver, "_shutoff_triggered", bool(auxiliary.get("shutoff_triggered"))),
            (solver, "_shutoff_step", auxiliary.get("shutoff_step")),
        )
    )
    from .runtime.passivity import preflight_passivity
    preflight_passivity(solver, auxiliary.get("passivity"), copy_ops, host_updates)
    from .observation_resume import preflight_observations
    preflight_observations(solver, auxiliary.get("observations"), checkpoint.step, copy_ops, host_updates)
    return copy_ops, host_updates


def preflight_resume_checkpoint(
    solver,
    checkpoint: FDTDResumeCheckpoint,
    *,
    total_steps: int,
) -> None:
    from .distributed.resume import is_distributed_solver, preflight_distributed_resume
    if is_distributed_solver(solver):
        return preflight_distributed_resume(solver,checkpoint,total_steps=total_steps)
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
    _preflight_additive_j_timing(solver, checkpoint.signature)
    _preflight_pulsed_tfsf_reference(solver, checkpoint.signature)
    _preflight_tfsf_timing(solver, checkpoint.signature)
    _preflight_observer_origin(solver, checkpoint.signature)
    _preflight_tfsf_termination(solver, checkpoint.signature)
    if checkpoint.signature != _solver_signature(solver):
        raise ValueError("FDTD resume checkpoint configuration does not match the prepared solver layout.")
    if bool(checkpoint.auxiliary.get("shutoff_triggered")):
        raise ValueError("A checkpoint captured after field shutoff cannot be resumed.")


def restore_resume_checkpoint(
    solver,
    checkpoint: FDTDResumeCheckpoint,
    *,
    total_steps: int,
) -> int:
    from .distributed.resume import is_distributed_solver, restore_distributed_resume
    if is_distributed_solver(solver):
        return restore_distributed_resume(solver,checkpoint,total_steps=total_steps)
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
        if isinstance(target, dict):
            target[name] = value
        else:
            setattr(target, name, value)
    solver._sync_dft_legacy_state()
    solver._sync_observer_legacy_state()
    return checkpoint.step


__all__ = [
    "FDTDResumeCheckpoint",
    "capture_resume_checkpoint",
    "preflight_resume_checkpoint",
    "restore_resume_checkpoint",
]
