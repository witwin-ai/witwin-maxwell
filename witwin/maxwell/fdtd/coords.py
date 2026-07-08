from __future__ import annotations

import numpy as np

from ..monitors import normalize_component
from ..scene import prepare_scene


def component_coords(scene, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_scene = prepare_scene(scene)
    component_name = normalize_component(component)
    x, y, z = resolved_scene.x_nodes64, resolved_scene.y_nodes64, resolved_scene.z_nodes64
    x_half, y_half, z_half = resolved_scene.x_half64, resolved_scene.y_half64, resolved_scene.z_half64

    if component_name == "ex":
        return x_half, y, z
    if component_name == "ey":
        return x, y_half, z
    if component_name == "ez":
        return x, y, z_half
    if component_name == "hx":
        return x, y_half, z_half
    if component_name == "hy":
        return x_half, y, z_half
    if component_name == "hz":
        return x_half, y_half, z
    raise ValueError(f"Unsupported field component: {component!r}")


def centered_cell_coords(scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_scene = prepare_scene(scene)
    return resolved_scene.x_half64, resolved_scene.y_half64, resolved_scene.z_half64
