from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from ..ports import CoaxialLumpedPort, LumpedPort, ModePort, TerminalPort, _coordinate_roundoff, _resolve_terminal_port
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

    metadata: Mapping[str, Any] = field(default_factory=dict,compare=False)

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


def _snap_index(coords: np.ndarray, value: float, *, label: str, roundoff: float = 0.0) -> int:
    """Normalize unique representation error without merging nearby grid planes.

    Input uncertainty comes from the original dtype, including the cancellation
    in a Box center +/- half size. A local spacing cap rejects coordinates whose
    source precision cannot distinguish nearby planes. Float64 arithmetic error
    is accounted for separately; no absolute SI-coordinate tolerance is used.
    """
    if not np.isfinite(value) or not np.isfinite(roundoff):
        raise ValueError(f"{label}={value} must be finite.")
    index = int(np.argmin(np.abs(coords - float(value))))
    neighbor_spacings = [
        abs(float(coords[neighbor] - coords[index]))
        for neighbor in (index - 1, index + 1)
        if 0 <= neighbor < coords.size
    ]
    local_spacing = min(neighbor_spacings) if neighbor_spacings else float("inf")
    precision_cap = local_spacing / 32.0
    arithmetic_error = 8.0 * max(
        abs(float(np.spacing(coords[index]))), abs(float(np.spacing(value))),
        # Uniform grids add an origin to an integer spacing product. Near zero,
        # cancellation is bounded by those construction operands, not ULP(0).
        abs(float(np.spacing(coords[0]))), abs(float(np.spacing(coords[-1]))),
        local_spacing * np.finfo(np.float64).eps if np.isfinite(local_spacing) else 0.0,
    )
    if roundoff > precision_cap or arithmetic_error > precision_cap:
        raise ValueError(
            f"{label}={value} has ambiguous coordinate precision for local Yee spacing "
            f"{local_spacing}; supply float64 coordinates."
        )
    tolerance = min(roundoff + arithmetic_error, precision_cap)
    residual = abs(float(coords[index]) - float(value))
    if residual > tolerance:
        raise ValueError(
            f"{label}={value} must lie on the Yee {label} grid "
            f"(off-grid; nearest={coords[index]}, residual={residual}, roundoff={tolerance})."
        )
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
    positive_roundoff=(0.0, 0.0, 0.0),
    negative_roundoff=(0.0, 0.0, 0.0),
):
    axis_index = _AXES.index(axis)
    nodes = _axis_values(scene, axis, half=False)
    negative_index = _snap_index(nodes, negative[axis_index], label=f"{axis} node", roundoff=negative_roundoff[axis_index])
    positive_index = _snap_index(nodes, positive[axis_index], label=f"{axis} node", roundoff=positive_roundoff[axis_index])
    if negative_index == positive_index:
        raise ValueError("LumpedPort terminals must occupy distinct Yee nodes.")
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
            roundoff=negative_roundoff[transverse_index],
        )
        positive_fixed_index = _snap_index(
            _axis_values(scene, transverse_axis, half=False),
            positive[transverse_index],
            label=f"{transverse_axis} node",
            roundoff=positive_roundoff[transverse_index],
        )
        if positive_fixed_index != fixed_indices[transverse_axis]:
            raise ValueError("LumpedPort terminals must define one axis-aligned Yee path.")

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
    compiled = _compile_voltage_path(
        scene,
        positive=port.positive,
        negative=port.negative,
        axis=port.voltage_path.axis,
        device=device,
        positive_roundoff=getattr(port, "_positive_roundoff", ()) or (0.0, 0.0, 0.0),
        negative_roundoff=getattr(port, "_negative_roundoff", ()) or (0.0, 0.0, 0.0),
    )
    aperture = getattr(port, "voltage_aperture", None)
    if aperture is None:
        return compiled
    position, size = _box_values(aperture)
    if not np.isfinite(position).all() or not np.isfinite(size).all() or np.any(size < 0):
        raise ValueError(f"Port {port.name!r} voltage_aperture must have finite nonnegative extents.")
    axis = _AXES.index(port.voltage_path.axis)
    uncertainty = np.asarray(_coordinate_roundoff(aperture.position)) + .5 * np.asarray(
        _coordinate_roundoff(aperture.size))
    selections, quadratures = [], []
    if not any(size[d] > 0 for d in range(3) if d != axis):
        raise ValueError(f"Port {port.name!r} voltage_aperture requires a finite transverse width.")
    for dim, name in enumerate(_AXES):
        nodes = _axis_values(scene, name, half=False)
        low, high = position[dim] - .5 * size[dim], position[dim] + .5 * size[dim]
        lower = _snap_index(nodes, low, label=f"{name} aperture lower node", roundoff=uncertainty[dim])
        upper = _snap_index(nodes, high, label=f"{name} aperture upper node", roundoff=uncertainty[dim])
        if dim == axis:
            endpoints = compiled[2][:, dim].detach().cpu().numpy()
            if lower != int(endpoints.min()) or upper != int(endpoints.max()) + 1:
                raise ValueError(f"Port {port.name!r} aperture must span exactly the voltage terminals.")
            selections.append(np.arange(lower, upper))
            quadratures.append(_axis_widths(scene, name, dual=False)[lower:upper] * compiled[0])
        else:
            if lower == upper:
                selections.append(np.array([lower]))
                quadratures.append(np.ones(1))
            else:
                chosen = np.arange(lower, upper + 1)
                halves = .5 * (nodes[:-1] + nodes[1:])
                left = np.maximum(nodes[lower], halves[np.maximum(0, chosen - 1)])
                right = np.minimum(nodes[upper], halves[np.minimum(len(halves) - 1, chosen)])
                left[0], right[-1] = nodes[lower], nodes[upper]
                weight = (right - left) / (nodes[upper] - nodes[lower])
                chart=getattr(scene,"coordinates",None)
                if chart is not None and chart.curvature and axis==2 and dim==1:
                    # A physical z-normal aperture has area h_s ds du.
                    # Integrate the linear metric over each clipped dual
                    # interval, then normalize the physical transverse area.
                    physical=(right-left)*(1-chart.curvature*(left+right)/2)
                    weight=physical/physical.sum()
                selections.append(chosen)
                quadratures.append(weight)
    indices = np.stack(np.meshgrid(*selections, indexing="ij"), axis=-1).reshape(-1, 3)
    weights = np.prod(np.stack(np.meshgrid(*quadratures, indexing="ij"), axis=-1), axis=-1).reshape(-1)
    contour_position, contour_size = _box_values(port.current_surface)
    for dim in range(3):
        if dim != axis and (position[dim] - .5 * size[dim] < contour_position[dim] - .5 * contour_size[dim]
                            or position[dim] + .5 * size[dim] > contour_position[dim] + .5 * contour_size[dim]):
            raise ValueError(f"Port {port.name!r} current contour must enclose the voltage aperture.")
    return compiled[0], compiled[1], torch.as_tensor(indices, device=device, dtype=torch.int64), torch.as_tensor(
        weights, device=device, dtype=torch.float64)


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
    position_roundoff = np.asarray(_coordinate_roundoff(port.current_surface.position))
    size_roundoff = np.asarray(_coordinate_roundoff(port.current_surface.size))
    zero_axes = tuple(index for index, extent in enumerate(surface_size) if float(extent) == 0.0)
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
        roundoff=float(position_roundoff[normal_index]),
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
        bound_roundoff = float(position_roundoff[tangent_index] + 0.5 * size_roundoff[tangent_index])
        inherited_roundoff = getattr(port, "_current_bound_roundoff", ())
        if inherited_roundoff:
            bound_roundoff += inherited_roundoff[tangent_index]
        lower_index = _snap_index(
            half_coords,
            lower,
            label=f"{tangent_axis} half-grid boundary",
            roundoff=bound_roundoff,
        )
        upper_index = _snap_index(
            half_coords,
            upper,
            label=f"{tangent_axis} half-grid boundary",
            roundoff=bound_roundoff,
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
        record = network.metadata["port_bindings"][binding_index]
        start, end = network.metadata["port_gap_ranges"][binding_index]
        negative_node = record["negative_node_id"]
        positive_node = record["positive_node_id"]
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
            # V is +integral(E) along the terminal arrow. Ampere circulation
            # measures impressed current along that arrow, whose delivered work
            # is -V*I. RF current into the field therefore has the opposite sign.
            -direction,
            device=target_device,
        )
    except (TypeError, ValueError) as error:
        raise type(error)(f"{type(port).__name__} {port.name!r}: {error}") from error
    geometry = CompiledPortGeometry(
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

    if isinstance(resolved_port, TerminalPort):
        from ..geometry.terminal_support import audit_compiled_terminal
        try:
            audit=audit_compiled_terminal(resolved_scene,resolved_port,geometry)
        except (ValueError,NotImplementedError) as error:
            raise type(error)(f"TerminalPort {resolved_port.name!r}: {error}") from error
        from dataclasses import replace
        geometry=replace(geometry,metadata=audit)
    return geometry


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
        if isinstance(port, CoaxialLumpedPort):
            from .coaxial import compile_coaxial_geometry

            compiled.append(compile_coaxial_geometry(scene, port, device=device))
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
