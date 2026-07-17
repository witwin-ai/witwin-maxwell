from __future__ import annotations

from dataclasses import dataclass

import torch

from .boundary import expand_cpml_memory_tensor

_FIELD_STATE_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_COMPLEX_FIELD_STATE_NAMES = ("Ex_imag", "Ey_imag", "Ez_imag", "Hx_imag", "Hy_imag", "Hz_imag")
_CPML_STATE_NAMES = (
    "psi_ex_y",
    "psi_ex_z",
    "psi_ey_x",
    "psi_ey_z",
    "psi_ez_x",
    "psi_ez_y",
    "psi_hx_y",
    "psi_hx_z",
    "psi_hy_x",
    "psi_hy_z",
    "psi_hz_x",
    "psi_hz_y",
)
_CPML_COMPLEX_STATE_NAMES = tuple(f"{name}_imag" for name in _CPML_STATE_NAMES)
_TFSF_AUXILIARY_STATE_NAMES = (
    "tfsf_aux_electric",
    "tfsf_aux_magnetic",
)
_DISPERSIVE_STATE_TENSORS = {
    "debye": ("polarization", "current"),
    "drude": ("current",),
    "lorentz": ("polarization", "current"),
}
_CHECKPOINT_SCHEMA_VERSION = 2


def dispersive_state_name(component_name: str, model_name: str, index: int, tensor_name: str) -> str:
    return f"{model_name}_{component_name}_{int(index)}_{tensor_name}"


def iter_dispersive_state_specs(solver):
    if not getattr(solver, "dispersive_enabled", False):
        return
    templates = getattr(solver, "_dispersive_templates", {})
    for component_name in ("Ex", "Ey", "Ez"):
        component_templates = templates.get(component_name, {})
        for model_name in ("debye", "drude", "lorentz"):
            tensor_names = _DISPERSIVE_STATE_TENSORS[model_name]
            for index, entry in enumerate(component_templates.get(model_name, ())):
                yield component_name, model_name, index, tensor_names, entry


def iter_magnetic_dispersive_state_specs(solver):
    if not getattr(solver, "magnetic_dispersive_enabled", False):
        return
    templates = getattr(solver, "_magnetic_dispersive_templates", {})
    for component_name in ("Hx", "Hy", "Hz"):
        component_templates = templates.get(component_name, {})
        for model_name in ("debye", "drude", "lorentz"):
            tensor_names = _DISPERSIVE_STATE_TENSORS[model_name]
            for index, entry in enumerate(component_templates.get(model_name, ())):
                yield component_name, model_name, index, tensor_names, entry


@dataclass(frozen=True)
class FDTDCheckpointSchema:
    version: int
    field_names: tuple[str, ...]
    complex_field_names: tuple[str, ...]
    cpml_state_names: tuple[str, ...]
    tfsf_auxiliary_state_names: tuple[str, ...]
    dispersive_state_names: tuple[str, ...]
    magnetic_dispersive_state_names: tuple[str, ...] = ()
    lumped_state_names: tuple[str, ...] = ()
    circuit_state_names: tuple[str, ...] = ()

    @property
    def state_names(self) -> tuple[str, ...]:
        return (
            self.field_names
            + self.complex_field_names
            + self.cpml_state_names
            + self.tfsf_auxiliary_state_names
            + self.dispersive_state_names
            + self.magnetic_dispersive_state_names
            + self.lumped_state_names
            + self.circuit_state_names
        )


def lumped_state_name(kind: str, index: int, tensor_name: str) -> str:
    return f"lumped_{kind}_{int(index)}_{tensor_name}"


def iter_lumped_state_specs(solver):
    """Yield the frozen auxiliary state for each coupled single-device branch."""

    for index, port_runtime in enumerate(getattr(solver, "_port_runtimes", ())):
        runtime = getattr(port_runtime, "lumped", None)
        if runtime is None:
            continue
        for tensor_name in ("inductor_current", "capacitor_voltage"):
            yield (
                lumped_state_name("port", index, tensor_name),
                runtime,
                tensor_name,
                port_runtime.field_name,
                "port",
                index,
            )
    for index, (runtime, field_name) in enumerate(
        getattr(solver, "_lumped_element_runtimes", ())
    ):
        for tensor_name in ("inductor_current", "capacitor_voltage"):
            yield (
                lumped_state_name("element", index, tensor_name),
                runtime,
                tensor_name,
                field_name,
                "element",
                index,
            )


def circuit_state_name(circuit_index: int, tensor_name: str) -> str:
    return f"circuit_{int(circuit_index)}_{tensor_name}"


def iter_circuit_state_specs(solver):
    """Yield the fixed companion history required by coupled MNA replay/resume."""

    for circuit_index, runtime in enumerate(getattr(solver, "_circuit_runtimes", ())):
        for tensor_name, tensor in runtime.checkpoint_tensors().items():
            yield circuit_state_name(circuit_index, tensor_name), tensor


def checkpoint_schema(solver) -> FDTDCheckpointSchema:
    complex_field_names = _COMPLEX_FIELD_STATE_NAMES if bool(getattr(solver, "complex_fields_enabled", False)) else ()
    cpml_state_names = ()
    if getattr(solver, "uses_cpml", False):
        cpml_state_names = _CPML_STATE_NAMES
        if bool(getattr(solver, "complex_fields_enabled", False)):
            cpml_state_names = cpml_state_names + _CPML_COMPLEX_STATE_NAMES

    tfsf_auxiliary_state_names = ()
    if getattr(solver, "tfsf_enabled", False):
        tfsf_state = getattr(solver, "_tfsf_state", None)
        if tfsf_state is not None and tfsf_state.get("auxiliary_grid") is not None:
            tfsf_auxiliary_state_names = _TFSF_AUXILIARY_STATE_NAMES

    dispersive_state_names = []
    for component_name, model_name, index, tensor_names, _entry in iter_dispersive_state_specs(solver) or ():
        for tensor_name in tensor_names:
            dispersive_state_names.append(dispersive_state_name(component_name, model_name, index, tensor_name))
    if bool(getattr(solver, "complex_fields_enabled", False)):
        # Bloch (complex-field) runs keep an imaginary ADE replica per electric
        # pole; the reverse replay differentiates through it, so it must be part
        # of the frozen checkpoint layout too.
        for component_name, model_name, index, tensor_names, _entry in iter_dispersive_state_specs(solver) or ():
            for tensor_name in tensor_names:
                dispersive_state_names.append(
                    dispersive_state_name(component_name, model_name, index, tensor_name) + "_imag"
                )

    magnetic_dispersive_state_names = []
    for component_name, model_name, index, tensor_names, _entry in iter_magnetic_dispersive_state_specs(solver) or ():
        for tensor_name in tensor_names:
            magnetic_dispersive_state_names.append(
                dispersive_state_name(component_name, model_name, index, tensor_name)
            )
    if bool(getattr(solver, "complex_fields_enabled", False)):
        magnetic_dispersive_state_names.extend(
            name + "_imag" for name in tuple(magnetic_dispersive_state_names)
        )
    lumped_state_names = tuple(
        name for name, _runtime, _tensor_name, _field_name, _kind, _index in iter_lumped_state_specs(solver)
    )
    circuit_state_names = tuple(name for name, _tensor in iter_circuit_state_specs(solver))

    return FDTDCheckpointSchema(
        version=_CHECKPOINT_SCHEMA_VERSION,
        field_names=_FIELD_STATE_NAMES,
        complex_field_names=tuple(complex_field_names),
        cpml_state_names=tuple(cpml_state_names),
        tfsf_auxiliary_state_names=tuple(tfsf_auxiliary_state_names),
        dispersive_state_names=tuple(dispersive_state_names),
        magnetic_dispersive_state_names=tuple(magnetic_dispersive_state_names),
        lumped_state_names=lumped_state_names,
        circuit_state_names=circuit_state_names,
    )


def checkpoint_state_names(solver) -> tuple[str, ...]:
    return checkpoint_schema(solver).state_names


@dataclass(frozen=True)
class FDTDCheckpointState:
    step: int
    schema: FDTDCheckpointSchema
    tensors: dict[str, torch.Tensor]


def validate_checkpoint_state(
    state: FDTDCheckpointState,
    *,
    expected_schema: FDTDCheckpointSchema | None = None,
) -> FDTDCheckpointSchema:
    state_names = tuple(state.tensors.keys())
    if state_names != state.schema.state_names:
        raise RuntimeError(
            "Checkpoint tensor layout drifted from the frozen adjoint state schema. "
            f"Expected {state.schema.state_names}, got {state_names}."
        )
    if expected_schema is not None and state.schema != expected_schema:
        raise RuntimeError(
            "Checkpoint schema mismatch detected during adjoint replay. "
            f"Expected version {expected_schema.version} with names {expected_schema.state_names}, "
            f"got version {state.schema.version} with names {state.schema.state_names}."
        )
    return state.schema


def capture_checkpoint_state(solver, step: int) -> FDTDCheckpointState:
    schema = checkpoint_schema(solver)
    tensors = {
        name: getattr(solver, name).detach().clone()
        for name in _FIELD_STATE_NAMES
    }
    if bool(getattr(solver, "complex_fields_enabled", False)):
        tensors.update(
            {
                name: getattr(solver, name).detach().clone()
                for name in _COMPLEX_FIELD_STATE_NAMES
            }
        )
    if getattr(solver, "uses_cpml", False):
        tensors.update(
            {
                name: expand_cpml_memory_tensor(solver, name).detach().clone()
                for name in schema.cpml_state_names
            }
        )
    if getattr(solver, "tfsf_enabled", False):
        tfsf_state = getattr(solver, "_tfsf_state", None)
        auxiliary_grid = None if tfsf_state is None else tfsf_state.get("auxiliary_grid")
        if auxiliary_grid is not None:
            tensors["tfsf_aux_electric"] = auxiliary_grid.electric.detach().clone()
            tensors["tfsf_aux_magnetic"] = auxiliary_grid.magnetic.detach().clone()
    for component_name, model_name, index, tensor_names, entry in iter_dispersive_state_specs(solver) or ():
        for tensor_name in tensor_names:
            tensors[dispersive_state_name(component_name, model_name, index, tensor_name)] = (
                entry[tensor_name].detach().clone()
            )
    if bool(getattr(solver, "complex_fields_enabled", False)):
        for component_name, model_name, index, tensor_names, entry in iter_dispersive_state_specs(solver) or ():
            for tensor_name in tensor_names:
                tensors[dispersive_state_name(component_name, model_name, index, tensor_name) + "_imag"] = (
                    entry[f"{tensor_name}_imag"].detach().clone()
                )
    for component_name, model_name, index, tensor_names, entry in iter_magnetic_dispersive_state_specs(solver) or ():
        for tensor_name in tensor_names:
            tensors[dispersive_state_name(component_name, model_name, index, tensor_name)] = (
                entry[tensor_name].detach().clone()
            )
    if bool(getattr(solver, "complex_fields_enabled", False)):
        for component_name, model_name, index, tensor_names, entry in iter_magnetic_dispersive_state_specs(solver) or ():
            for tensor_name in tensor_names:
                tensors[dispersive_state_name(component_name, model_name, index, tensor_name) + "_imag"] = (
                    entry[f"{tensor_name}_imag"].detach().clone()
                )
    for name, runtime, tensor_name, _field_name, _kind, _index in iter_lumped_state_specs(solver):
        tensors[name] = getattr(runtime, tensor_name).detach().clone()
    for name, tensor in iter_circuit_state_specs(solver):
        tensors[name] = tensor.detach().clone()
    state = FDTDCheckpointState(step=int(step), schema=schema, tensors=tensors)
    validate_checkpoint_state(state)
    return state


def _checkpoint_tensor_targets(solver, schema: FDTDCheckpointSchema):
    for name in _FIELD_STATE_NAMES:
        yield name, getattr(solver, name), None
    if bool(getattr(solver, "complex_fields_enabled", False)):
        for name in _COMPLEX_FIELD_STATE_NAMES:
            yield name, getattr(solver, name), None
    if getattr(solver, "uses_cpml", False):
        for name in schema.cpml_state_names:
            yield name, getattr(solver, name), getattr(
                solver, "_cpml_memory_layouts", {}
            ).get(name)
    if getattr(solver, "tfsf_enabled", False):
        auxiliary_grid = getattr(solver, "_tfsf_state", {}).get("auxiliary_grid")
        if auxiliary_grid is not None:
            yield "tfsf_aux_electric", auxiliary_grid.electric, None
            yield "tfsf_aux_magnetic", auxiliary_grid.magnetic, None
    for component_name, model_name, index, tensor_names, entry in (
        iter_dispersive_state_specs(solver) or ()
    ):
        for tensor_name in tensor_names:
            name = dispersive_state_name(
                component_name, model_name, index, tensor_name
            )
            yield name, entry[tensor_name], None
    if bool(getattr(solver, "complex_fields_enabled", False)):
        for component_name, model_name, index, tensor_names, entry in (
            iter_dispersive_state_specs(solver) or ()
        ):
            for tensor_name in tensor_names:
                name = (
                    dispersive_state_name(
                        component_name, model_name, index, tensor_name
                    )
                    + "_imag"
                )
                yield name, entry[f"{tensor_name}_imag"], None
    for component_name, model_name, index, tensor_names, entry in (
        iter_magnetic_dispersive_state_specs(solver) or ()
    ):
        for tensor_name in tensor_names:
            name = dispersive_state_name(
                component_name, model_name, index, tensor_name
            )
            yield name, entry[tensor_name], None
    if bool(getattr(solver, "complex_fields_enabled", False)):
        for component_name, model_name, index, tensor_names, entry in (
            iter_magnetic_dispersive_state_specs(solver) or ()
        ):
            for tensor_name in tensor_names:
                name = (
                    dispersive_state_name(
                        component_name, model_name, index, tensor_name
                    )
                    + "_imag"
                )
                yield name, entry[f"{tensor_name}_imag"], None
    for name, runtime, tensor_name, _field_name, _kind, _index in (
        iter_lumped_state_specs(solver)
    ):
        yield name, getattr(runtime, tensor_name), None
    for name, tensor in iter_circuit_state_specs(solver):
        yield name, tensor, None


def _checkpoint_expected_shape(target: torch.Tensor, layout) -> tuple[int, ...]:
    if layout is None:
        return tuple(target.shape)
    return tuple(layout["field_shape"])


def _copy_checkpoint_tensor(target: torch.Tensor, source: torch.Tensor, layout) -> None:
    if layout is None:
        target.copy_(source)
        return
    axis = int(layout["axis"])
    for region in layout["regions"]:
        length = int(region["length"])
        if length <= 0:
            continue
        target.narrow(axis, int(region["local_start"]), length).copy_(
            source.narrow(axis, int(region["global_start"]), length)
        )


def restore_checkpoint_state(solver, state: FDTDCheckpointState) -> None:
    """Restore a captured physical FDTD state into an initialized solver.

    The full layout is validated before the first tensor is mutated. Checkpoints
    loaded on CPU may be restored into a CUDA solver; dtype and shape must still
    match exactly. CPML slab storage is populated from the dense checkpoint
    representation without allocating persistent dense memory.
    """

    expected_schema = checkpoint_schema(solver)
    validate_checkpoint_state(state, expected_schema=expected_schema)
    targets = tuple(_checkpoint_tensor_targets(solver, expected_schema))
    if tuple(name for name, _target, _layout in targets) != expected_schema.state_names:
        raise RuntimeError("Initialized solver checkpoint targets drifted from its schema.")
    for name, target, layout in targets:
        source = state.tensors[name]
        expected_shape = _checkpoint_expected_shape(target, layout)
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                f"Checkpoint tensor {name!r} has shape {tuple(source.shape)}, "
                f"expected {expected_shape}."
            )
        if source.dtype != target.dtype:
            raise TypeError(
                f"Checkpoint tensor {name!r} has dtype {source.dtype}, "
                f"expected {target.dtype}."
            )
    for name, target, layout in targets:
        _copy_checkpoint_tensor(target, state.tensors[name], layout)


def clone_checkpoint_tensors(state: FDTDCheckpointState) -> dict[str, torch.Tensor]:
    validate_checkpoint_state(state)
    return {
        name: tensor.detach().clone()
        for name, tensor in state.tensors.items()
    }
