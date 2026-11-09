from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from ..monitors import PowerLossMonitor


_ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")


@dataclass(frozen=True)
class CompiledPowerLossMonitor:
    """Sparse Yee-edge selection and static conductivity for a loss monitor."""

    monitor: PowerLossMonitor
    component_masks: Mapping[str, torch.Tensor]
    component_volumes: Mapping[str, torch.Tensor]
    global_ids: Mapping[str, torch.Tensor]
    conductivity: Mapping[str, torch.Tensor] | None
    full_component_shapes: Mapping[str, tuple[int, int, int]]
    component_cell_bounds: Mapping[str, tuple] | None = None

    @property
    def device(self) -> torch.device:
        return next(iter(self.component_masks.values())).device


def _component_geometry(scene, component: str):
    if component == "Ex":
        return (
            (scene.x_half, scene.y, scene.z),
            (scene.dx_primal64, scene.dy_dual64, scene.dz_dual64),
        )
    if component == "Ey":
        return (
            (scene.x, scene.y_half, scene.z),
            (scene.dx_dual64, scene.dy_primal64, scene.dz_dual64),
        )
    if component == "Ez":
        return (
            (scene.x, scene.y, scene.z_half),
            (scene.dx_dual64, scene.dy_dual64, scene.dz_primal64),
        )
    raise ValueError(f"Unsupported electric component {component!r}.")


def _selection_mask(scene, monitor: PowerLossMonitor, component: str) -> torch.Tensor:
    coordinates, _ = _component_geometry(scene, component)
    selections = []
    for coordinate, (lower, upper) in zip(coordinates, monitor.bounds):
        selections.append((coordinate >= lower) & (coordinate <= upper))
    return (
        selections[0][:, None, None]
        & selections[1][None, :, None]
        & selections[2][None, None, :]
    )


def _component_cell_bounds(scene, component):
    """Physical primal/dual rectangles from the original float64 Yee nodes."""
    result = []
    for axis in "xyz":
        nodes = torch.tensor(getattr(scene, axis+"_nodes64").copy(), device=scene.device, dtype=torch.float64)
        if axis == component[-1].lower():
            lower, upper = nodes[:-1], nodes[1:]
        else:
            middle = (nodes[:-1]+nodes[1:])/2
            lower, upper = torch.cat((nodes[:1], middle)), torch.cat((middle, nodes[-1:]))
        result.append((lower, upper))
    return tuple(result)


def _intersected_geometry(bounds, window):
    widths = [torch.clamp(torch.minimum(upper, upper.new_tensor(hi))
              - torch.maximum(lower, lower.new_tensor(lo)), min=0.)
              for (lower, upper), (lo, hi) in zip(bounds, window)]
    mask = ((widths[0] > 0)[:, None, None] & (widths[1] > 0)[None, :, None]
            & (widths[2] > 0)[None, None, :])
    volume = widths[0][:, None, None]*widths[1][None, :, None]*widths[2][None, None, :]
    return mask, volume[mask]


def _selected_control_volumes(
    scene,
    component: str,
    mask: torch.Tensor,
) -> torch.Tensor:
    _, widths = _component_geometry(scene, component)
    dtype = scene.x.dtype
    device = scene.device
    x = torch.as_tensor(widths[0], device=device, dtype=dtype)
    y = torch.as_tensor(widths[1], device=device, dtype=dtype)
    z = torch.as_tensor(widths[2], device=device, dtype=dtype)
    volumes = x[:, None, None] * y[None, :, None] * z[None, None, :]
    return volumes[mask]


def _node_to_electric_component(
    node_tensor: torch.Tensor, component: str
) -> torch.Tensor:
    if component == "Ex":
        return 0.5 * (node_tensor[:-1, :, :] + node_tensor[1:, :, :])
    if component == "Ey":
        return 0.5 * (node_tensor[:, :-1, :] + node_tensor[:, 1:, :])
    if component == "Ez":
        return 0.5 * (node_tensor[:, :, :-1] + node_tensor[:, :, 1:])
    raise ValueError(f"Unsupported electric component {component!r}.")


def _scene_has_sheet_material(scene) -> bool:
    return any(
        bool(getattr(getattr(structure, "material", None), "is_medium2d", False))
        for structure in scene.structures
        if getattr(structure, "enabled", True)
    )


def compile_power_loss_monitor(
    scene, monitor: PowerLossMonitor
) -> CompiledPowerLossMonitor:
    """Compile an axis-aligned loss volume onto sparse electric Yee edges.

    The automatic ``conduction`` channel uses only the compiled static bulk
    electric conductivity. Surface sheets are rejected because their lowering
    into an effective volume conductivity no longer retains surface identity.
    Physical intersections integrate this compiled homogenized cell density,
    rather than reinterpreting its unresolved material fill fraction.
    """

    if not isinstance(monitor, PowerLossMonitor):
        raise TypeError("monitor must be a PowerLossMonitor instance.")
    if not all(hasattr(scene, name) for name in ("x_half", "y_half", "z_half")):
        from ..scene import prepare_scene

        scene = prepare_scene(scene)
    if monitor.quadrature == "physical_intersection" and getattr(scene, "coordinates", None) is not None:
        raise NotImplementedError("PowerLossMonitor physical_intersection currently requires Cartesian control volumes; mapped partial-volume metrics are not implemented.")
    if monitor.quadrature == "physical_intersection" and any(
        scene.boundary.axis_kind(axis) in ("periodic", "bloch") for axis in "xyz"
    ):
        raise NotImplementedError("PowerLossMonitor physical_intersection requires nonperiodic axes; wrapped control-volume fragments are not implemented.")

    masks = {}
    component_volumes = {}
    global_ids = {}
    full_shapes = {}
    cell_bounds = {}
    offset = 0
    for component in _ELECTRIC_COMPONENTS:
        if monitor.quadrature == "physical_intersection":
            cell_bounds[component] = _component_cell_bounds(scene, component)
            mask, volumes = _intersected_geometry(cell_bounds[component], monitor.bounds)
        else:
            mask = _selection_mask(scene, monitor, component)
            volumes = _selected_control_volumes(scene, component, mask)
        if not bool(torch.any(mask)):
            raise ValueError(
                f"PowerLossMonitor {monitor.name!r} contains no {component} Yee edges; "
                "increase its size or refine the grid."
            )
        masks[component] = mask
        full_shapes[component] = tuple(mask.shape)
        component_volumes[component] = volumes
        full_ids = torch.arange(
            offset,
            offset + mask.numel(),
            device=scene.device,
            dtype=torch.int64,
        ).reshape(mask.shape)
        global_ids[component] = full_ids[mask]
        offset += mask.numel()

    conductivity = None
    if "conduction" in monitor.channels:
        if _scene_has_sheet_material(scene):
            raise NotImplementedError(
                "Automatic conduction loss does not accept 2D sheet materials: the current material "
                "compiler lowers sheet conductance into volume conductivity and no longer retains the "
                "surface identity required for a surface-loss channel."
            )
        material_model = scene.compile_materials()
        sigma_m_components = material_model["sigma_m_components"]
        if any(
            bool(torch.any(sigma_m_components[axis] != 0.0)) for axis in ("x", "y", "z")
        ):
            raise NotImplementedError(
                "Automatic conduction loss currently supports static electric conductivity only; "
                "magnetic conductivity requires full-frequency H fields and a separate channel."
            )
        sigma_e_components = material_model["sigma_e_components"]
        edge_components = material_model.get("edge_components")
        conductivity = {}
        for component, axis in zip(_ELECTRIC_COMPONENTS, ("x", "y", "z")):
            # Match the actual FDTD material path. Re-averaging node samples
            # changes conductivity at interfaces even when native edges exist.
            edge_conductivity = (edge_components["sigma_e"][component]
                if edge_components is not None else
                _node_to_electric_component(sigma_e_components[axis], component))
            conductivity[component] = edge_conductivity[masks[component]]
        conductivity = MappingProxyType(conductivity)

    return CompiledPowerLossMonitor(
        monitor=monitor,
        component_masks=MappingProxyType(masks),
        component_volumes=MappingProxyType(component_volumes),
        global_ids=MappingProxyType(global_ids),
        conductivity=conductivity,
        full_component_shapes=MappingProxyType(full_shapes),
        component_cell_bounds=MappingProxyType(cell_bounds) if cell_bounds else None,
    )


__all__ = ["CompiledPowerLossMonitor", "compile_power_loss_monitor"]
