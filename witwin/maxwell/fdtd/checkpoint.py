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
_CHECKPOINT_SCHEMA_VERSION = 1


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


@dataclass(frozen=True)
class FDTDCheckpointSchema:
    version: int
    field_names: tuple[str, ...]
    complex_field_names: tuple[str, ...]
    cpml_state_names: tuple[str, ...]
    tfsf_auxiliary_state_names: tuple[str, ...]
    dispersive_state_names: tuple[str, ...]

    @property
    def state_names(self) -> tuple[str, ...]:
        return (
            self.field_names
            + self.complex_field_names
            + self.cpml_state_names
            + self.tfsf_auxiliary_state_names
            + self.dispersive_state_names
        )


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

    return FDTDCheckpointSchema(
        version=_CHECKPOINT_SCHEMA_VERSION,
        field_names=_FIELD_STATE_NAMES,
        complex_field_names=tuple(complex_field_names),
        cpml_state_names=tuple(cpml_state_names),
        tfsf_auxiliary_state_names=tuple(tfsf_auxiliary_state_names),
        dispersive_state_names=tuple(dispersive_state_names),
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
    state = FDTDCheckpointState(step=int(step), schema=schema, tensors=tensors)
    validate_checkpoint_state(state)
    return state


def clone_checkpoint_tensors(state: FDTDCheckpointState) -> dict[str, torch.Tensor]:
    validate_checkpoint_state(state)
    return {
        name: tensor.detach().clone()
        for name, tensor in state.tensors.items()
    }
