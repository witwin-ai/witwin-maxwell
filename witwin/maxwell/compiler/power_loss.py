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
    """

    if not isinstance(monitor, PowerLossMonitor):
        raise TypeError("monitor must be a PowerLossMonitor instance.")
    if not all(hasattr(scene, name) for name in ("x_half", "y_half", "z_half")):
        from ..scene import prepare_scene

        scene = prepare_scene(scene)

    masks = {}
    component_volumes = {}
    global_ids = {}
    full_shapes = {}
    offset = 0
    for component in _ELECTRIC_COMPONENTS:
        mask = _selection_mask(scene, monitor, component)
        if not bool(torch.any(mask)):
            raise ValueError(
                f"PowerLossMonitor {monitor.name!r} contains no {component} Yee edges; "
                "increase its size or refine the grid."
            )
        masks[component] = mask
        full_shapes[component] = tuple(mask.shape)
        component_volumes[component] = _selected_control_volumes(scene, component, mask)
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
        conductivity = {}
        for component, axis in zip(_ELECTRIC_COMPONENTS, ("x", "y", "z")):
            edge_conductivity = _node_to_electric_component(
                sigma_e_components[axis],
                component,
            )
            conductivity[component] = edge_conductivity[masks[component]]
        conductivity = MappingProxyType(conductivity)

    return CompiledPowerLossMonitor(
        monitor=monitor,
        component_masks=MappingProxyType(masks),
        component_volumes=MappingProxyType(component_volumes),
        global_ids=MappingProxyType(global_ids),
        conductivity=conductivity,
        full_component_shapes=MappingProxyType(full_shapes),
    )


__all__ = ["CompiledPowerLossMonitor", "compile_power_loss_monitor"]
