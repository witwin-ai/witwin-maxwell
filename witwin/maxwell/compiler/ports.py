from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ..ports import LumpedPort, ModePort, TerminalPort, _resolve_terminal_port
from ..scene import prepare_scene


_AXES = "xyz"
_TANGENTIAL_AXES = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}


def _integrate_sparse_term(
    fields: Mapping[str, torch.Tensor],
    component: str,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if component not in fields:
        raise KeyError(f"Field component {component!r} is required by the compiled port geometry.")
    field = fields[component]
    if field.device != indices.device:
        raise ValueError(
            f"Field {component} is on {field.device}, but port weights are on {indices.device}."
        )
    values = field[indices[:, 0], indices[:, 1], indices[:, 2]]
    return torch.sum(values * weights.to(dtype=values.dtype))


@dataclass(frozen=True)
class CompiledPortGeometry:
    """Sparse Yee-grid weights for one lumped-port voltage and current integral."""

    port_name: str
    axis: str
    direction: int
    voltage_component: str
    voltage_indices: torch.Tensor
    voltage_weights: torch.Tensor
    current_components: tuple[str, ...]
    current_indices: tuple[torch.Tensor, ...]
    current_weights: tuple[torch.Tensor, ...]
    reference_impedance: Any
    reference_plane: float | None = None
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"

    def integrate_voltage(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return _integrate_sparse_term(
            fields,
            self.voltage_component,
            self.voltage_indices,
            self.voltage_weights,
        )

    def integrate_current(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        result = None
        for component, indices, weights in zip(
            self.current_components,
            self.current_indices,
            self.current_weights,
        ):
            contribution = _integrate_sparse_term(fields, component, indices, weights)
            result = contribution if result is None else result + contribution
        if result is None:
            raise RuntimeError("Compiled port current contour is empty.")
        return result

    def average_power(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        voltage = self.integrate_voltage(fields)
        current = self.integrate_current(fields)
        return 0.5 * torch.real(voltage * torch.conj(current))


@dataclass(frozen=True)
class CompiledWirePortGeometry:
    """Sparse generalized coordinate for one port bound to a wire graph."""

    port_name: str
    binding_kind: str
    negative_node_id: int
    positive_node_id: int
    edge_components: torch.Tensor
    edge_offsets: torch.Tensor
    edge_weights: torch.Tensor
    binding_index: int
    gap_offset: int
    reference_impedance: Any
    reference_plane: float | None = None
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"
    metadata: Mapping[str, Any] | None = None

    @property
    def is_wire_bound(self) -> bool:
        return True


def _axis_values(scene, axis: str, *, half: bool) -> np.ndarray:
    suffix = "half64" if half else "nodes64"
    return np.asarray(getattr(scene, f"{axis}_{suffix}"), dtype=np.float64)


def _axis_widths(scene, axis: str, *, dual: bool) -> np.ndarray:
    suffix = "dual64" if dual else "primal64"
    return np.asarray(getattr(scene, f"d{axis}_{suffix}"), dtype=np.float64)


def _grid_tolerance(coords: np.ndarray) -> float:
    spacings = np.diff(coords)
    minimum_spacing = float(np.min(np.abs(spacings))) if spacings.size else 1.0
    return max(1.0e-12, minimum_spacing * 1.0e-6)


def _snap_index(coords: np.ndarray, value: float, *, label: str) -> int:
    index = int(np.argmin(np.abs(coords - float(value))))
    if abs(float(coords[index]) - float(value)) > _grid_tolerance(coords):
        raise ValueError(f"{label}={value} must lie on the Yee {label} grid.")
    return index


def _box_values(box) -> tuple[np.ndarray, np.ndarray]:
    position = torch.as_tensor(box.position).detach().cpu().to(dtype=torch.float64).numpy()
    size = torch.as_tensor(box.size).detach().cpu().to(dtype=torch.float64).numpy()
    rotation = torch.as_tensor(box.rotation).detach().cpu().to(dtype=torch.float64)
    identity = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
    if not torch.allclose(rotation, identity, rtol=0.0, atol=1.0e-12):
        raise ValueError("LumpedPort current_surface must be an unrotated axis-aligned Box.")
    return position, size


def _line_indices(
    varying_axis: str,
    varying_indices: np.ndarray,
    fixed_indices: Mapping[str, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    indices = np.empty((varying_indices.size, 3), dtype=np.int64)
    for axis_index, axis in enumerate(_AXES):
        if axis == varying_axis:
            indices[:, axis_index] = varying_indices
        else:
            indices[:, axis_index] = fixed_indices[axis]
    return torch.as_tensor(indices, device=device, dtype=torch.int64)


def _compile_voltage_path(
    scene,
    *,
    positive,
    negative,
    axis: str,
    device: torch.device,
):
    axis_index = _AXES.index(axis)
    nodes = _axis_values(scene, axis, half=False)
    negative_index = _snap_index(nodes, negative[axis_index], label=f"{axis} node")
    positive_index = _snap_index(nodes, positive[axis_index], label=f"{axis} node")
    direction = 1 if positive_index > negative_index else -1
    lower_index, upper_index = sorted((negative_index, positive_index))

    fixed_indices = {}
    for transverse_axis in _AXES:
        if transverse_axis == axis:
            continue
        transverse_index = _AXES.index(transverse_axis)
        fixed_indices[transverse_axis] = _snap_index(
            _axis_values(scene, transverse_axis, half=False),
            negative[transverse_index],
            label=f"{transverse_axis} node",
        )

    edge_indices = np.arange(lower_index, upper_index, dtype=np.int64)
    indices = np.empty((edge_indices.size, 3), dtype=np.int64)
    for component_axis_index, component_axis in enumerate(_AXES):
        if component_axis == axis:
            indices[:, component_axis_index] = edge_indices
        else:
            indices[:, component_axis_index] = fixed_indices[component_axis]
    weights = direction * _axis_widths(scene, axis, dual=False)[edge_indices]
    return (
        direction,
        f"E{axis}",
        torch.as_tensor(indices, device=device, dtype=torch.int64),
        torch.as_tensor(weights, device=device, dtype=torch.float64),
    )


def _compile_voltage(port: LumpedPort | TerminalPort, scene, *, device: torch.device):
    return _compile_voltage_path(
        scene,
        positive=port.positive,
        negative=port.negative,
        axis=port.voltage_path.axis,
        device=device,
    )


def _compile_current(
    port: LumpedPort | TerminalPort,
    scene,
    direction: int,
    *,
    device: torch.device,
):
    normal_axis = port.voltage_path.axis
    normal_index = _AXES.index(normal_axis)
    surface_position, surface_size = _box_values(port.current_surface)
    normal_tolerance = _grid_tolerance(_axis_values(scene, normal_axis, half=True))
    zero_axes = tuple(index for index, extent in enumerate(surface_size) if abs(float(extent)) <= normal_tolerance)
    if len(zero_axes) != 1:
        raise ValueError("LumpedPort current_surface must be planar.")
    if zero_axes[0] != normal_index:
        raise ValueError("LumpedPort current_surface normal must match voltage_path.axis.")

    if any(float(surface_size[index]) <= 0.0 for index in range(3) if index != normal_index):
        raise ValueError("LumpedPort current_surface must have positive tangential extents.")
    normal_half_index = _snap_index(
        _axis_values(scene, normal_axis, half=True),
        float(surface_position[normal_index]),
        label=f"{normal_axis} half-grid plane",
    )
    path_lower, path_upper = sorted((port.negative[normal_index], port.positive[normal_index]))
    if not path_lower < float(surface_position[normal_index]) < path_upper:
        raise ValueError("LumpedPort current_surface plane must lie between the two terminals.")

    u_axis, v_axis = _TANGENTIAL_AXES[normal_axis]
    half_indices = {}
    for tangent_axis in (u_axis, v_axis):
        tangent_index = _AXES.index(tangent_axis)
        lower = float(surface_position[tangent_index] - 0.5 * surface_size[tangent_index])
        upper = float(surface_position[tangent_index] + 0.5 * surface_size[tangent_index])
        half_coords = _axis_values(scene, tangent_axis, half=True)
        lower_index = _snap_index(
            half_coords,
            lower,
            label=f"{tangent_axis} half-grid boundary",
        )
        upper_index = _snap_index(
            half_coords,
            upper,
            label=f"{tangent_axis} half-grid boundary",
        )
        if lower_index >= upper_index:
            raise ValueError("LumpedPort current_surface tangential bounds must enclose at least one dual edge.")
        half_indices[tangent_axis] = (lower_index, upper_index)

    u_low, u_high = half_indices[u_axis]
    v_low, v_high = half_indices[v_axis]
    u_nodes = np.arange(u_low + 1, u_high + 1, dtype=np.int64)
    v_nodes = np.arange(v_low + 1, v_high + 1, dtype=np.int64)
    u_weights = _axis_widths(scene, u_axis, dual=True)[u_nodes]
    v_weights = _axis_widths(scene, v_axis, dual=True)[v_nodes]

    common = {normal_axis: normal_half_index}
    current_components = (f"H{u_axis}", f"H{v_axis}", f"H{u_axis}", f"H{v_axis}")
    current_indices = (
        _line_indices(u_axis, u_nodes, {**common, v_axis: v_low}, device=device),
        _line_indices(v_axis, v_nodes, {**common, u_axis: u_high}, device=device),
        _line_indices(u_axis, u_nodes, {**common, v_axis: v_high}, device=device),
        _line_indices(v_axis, v_nodes, {**common, u_axis: u_low}, device=device),
    )
    orientation = float(direction)
    current_weights = (
        torch.as_tensor(orientation * u_weights, device=device, dtype=torch.float64),
        torch.as_tensor(orientation * v_weights, device=device, dtype=torch.float64),
        torch.as_tensor(-orientation * u_weights, device=device, dtype=torch.float64),
        torch.as_tensor(-orientation * v_weights, device=device, dtype=torch.float64),
    )
    return current_components, current_indices, current_weights


def compile_port_geometry(
    scene,
    port: LumpedPort | TerminalPort,
    *,
    device: str | torch.device | None = None,
) -> CompiledPortGeometry:
    """Compile one axis-aligned lumped port into sparse global Yee indices."""

    if not isinstance(port, (LumpedPort, TerminalPort)):
        raise TypeError("compile_port_geometry expects a LumpedPort or TerminalPort.")
    resolved_scene = prepare_scene(scene)
    resolved_port = (
        _resolve_terminal_port(port, resolved_scene.structures, resolved_scene.domain.bounds)
        if isinstance(port, TerminalPort) and port.wire_binding is None
        else port
    )
    target_device = torch.device(resolved_scene.device if device is None else device)
    if getattr(resolved_port, "wire_binding", None) is not None:
        network = resolved_scene.compile_thin_wires(device=target_device)
        matches = tuple(
            index
            for index, name in enumerate(network.port_binding_names)
            if name == resolved_port.name
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Wire-bound port {resolved_port.name!r} was not lowered exactly once "
                "by the thin-wire compiler."
            )
        binding_index = matches[0]
        start = int(network.port_gap_offsets[binding_index])
        end = int(network.port_gap_offsets[binding_index + 1])
        negative_node = int(network.port_negative_node_ids[binding_index])
        positive_node = int(network.port_positive_node_ids[binding_index])
        records = tuple(network.metadata.get("port_bindings", ()))
        record = records[binding_index] if binding_index < len(records) else {}
        return CompiledWirePortGeometry(
            port_name=resolved_port.name,
            binding_kind=network.port_binding_kinds[binding_index],
            negative_node_id=negative_node,
            positive_node_id=positive_node,
            edge_components=network.port_gap_edge_components[start:end],
            edge_offsets=network.port_gap_edge_offsets[start:end],
            edge_weights=network.port_gap_weights[start:end],
            binding_index=binding_index,
            gap_offset=start,
            reference_impedance=resolved_port.reference_impedance,
            reference_plane=resolved_port.reference_plane,
            phasor_convention=resolved_port.phasor_convention,
            power_convention=resolved_port.power_convention,
            metadata=dict(record),
        )
    try:
        direction, voltage_component, voltage_indices, voltage_weights = _compile_voltage(
            resolved_port,
            resolved_scene,
            device=target_device,
        )
        current_components, current_indices, current_weights = _compile_current(
            resolved_port,
            resolved_scene,
            direction,
            device=target_device,
        )
    except (TypeError, ValueError) as error:
        if isinstance(port, TerminalPort):
            raise type(error)(f"TerminalPort {port.name!r}: {error}") from error
        raise
    return CompiledPortGeometry(
        port_name=resolved_port.name,
        axis=resolved_port.voltage_path.axis,
        direction=direction,
        voltage_component=voltage_component,
        voltage_indices=voltage_indices,
        voltage_weights=voltage_weights,
        current_components=current_components,
        current_indices=current_indices,
        current_weights=current_weights,
        reference_impedance=resolved_port.reference_impedance,
        reference_plane=resolved_port.reference_plane,
        phasor_convention=resolved_port.phasor_convention,
        power_convention=resolved_port.power_convention,
    )


def compile_ports(
    scene,
    ports=None,
    *,
    device: str | torch.device | None = None,
) -> tuple[CompiledPortGeometry, ...]:
    """Compile lumped ports while leaving existing ModePort expansion unchanged."""

    selected_ports = tuple(scene.ports if ports is None else ports)
    compiled = []
    for port in selected_ports:
        if isinstance(port, ModePort):
            continue
        if not isinstance(port, (LumpedPort, TerminalPort)):
            raise TypeError(f"Unsupported port type: {type(port).__name__}.")
        compiled.append(compile_port_geometry(scene, port, device=device))
    return tuple(compiled)


__all__ = [
    "CompiledPortGeometry",
    "CompiledWirePortGeometry",
    "compile_port_geometry",
    "compile_ports",
]
