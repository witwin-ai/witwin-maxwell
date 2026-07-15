from __future__ import annotations

import numpy as np
import torch

from .interpolation import ideal_axis_weight


def _sample_control_widths(coords: torch.Tensor) -> torch.Tensor:
    count = int(coords.numel())
    if count <= 1:
        return torch.ones_like(coords)
    diffs = coords[1:] - coords[:-1]
    widths = torch.empty_like(coords)
    widths[0] = 0.5 * diffs[0]
    widths[-1] = 0.5 * diffs[-1]
    if count > 2:
        widths[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return widths


def rebuild_ideal_ex_interface_plane(
    term: dict,
    source: dict,
    solver,
    *,
    local_index: int,
    global_index: int,
    global_x_half64: np.ndarray,
) -> None:
    """Rebuild one artificial endpoint plane using global interpolation arithmetic."""

    offset_x, offset_y, offset_z = (int(value) for value in term["offsets"])
    relative_x = local_index - offset_x
    patch = term["patch"]
    if not (0 <= relative_x < int(patch.shape[0])):
        return

    coords = np.asarray(global_x_half64, dtype=np.float32)
    left = np.float32(coords[global_index] - coords[global_index - 1])
    right = np.float32(coords[global_index + 1] - coords[global_index])
    global_x_width = np.float32(0.5) * np.float32(left + right)
    position = term.get("source_position")
    if position is None:
        raise RuntimeError("Compiled ideal point-source term is missing its image position.")
    x_weight = ideal_axis_weight(coords, position[0], global_index)

    dtype = solver.Ex.dtype
    device = solver.device
    y_widths = _sample_control_widths(solver.scene.y.to(device=device, dtype=dtype))
    z_widths = _sample_control_widths(solver.scene.z.to(device=device, dtype=dtype))
    y_stop = offset_y + int(patch.shape[1])
    z_stop = offset_z + int(patch.shape[2])
    x_width = torch.as_tensor(global_x_width, device=device, dtype=dtype)
    volume = x_width * y_widths[offset_y:y_stop, None] * z_widths[None, offset_z:z_stop]
    eps = solver.eps_Ex[local_index, offset_y:y_stop, offset_z:z_stop]
    denominator = eps * volume
    symmetry_scale = term.get("source_symmetry_scale")
    if symmetry_scale is None:
        raise RuntimeError("Compiled ideal point-source term is missing its symmetry scale.")
    source_scale = (
        -float(solver.dt) * float(source["polarization"][0]) * float(symmetry_scale)
    )
    y_coords = np.asarray(solver.scene.y_nodes64, dtype=np.float64)
    z_coords = np.asarray(solver.scene.z_nodes64, dtype=np.float64)
    for local_y, global_y in enumerate(range(offset_y, y_stop)):
        y_weight = ideal_axis_weight(y_coords, position[1], global_y)
        for local_z, global_z in enumerate(range(offset_z, z_stop)):
            z_weight = ideal_axis_weight(z_coords, position[2], global_z)
            numerator = source_scale * x_weight * y_weight * z_weight
            patch[relative_x, local_y, local_z].copy_(
                numerator / denominator[local_y, local_z]
            )


__all__ = ["rebuild_ideal_ex_interface_plane"]
