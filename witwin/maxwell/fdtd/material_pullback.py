from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..compiler.materials import _density_kernel_size
from ..scene import prepare_scene
from witwin.core import Box


@dataclass(frozen=True)
class _RegionLayout:
    x_slice: slice
    y_slice: slice
    z_slice: slice
    target_shape: tuple[int, int, int]

    @property
    def region(self) -> tuple[slice, slice, slice]:
        return (self.x_slice, self.y_slice, self.z_slice)


def _region_axis_slice(axis_coords: torch.Tensor, lower: float, upper: float) -> slice | None:
    indices = torch.nonzero((axis_coords >= lower) & (axis_coords <= upper), as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    return slice(int(indices[0].item()), int(indices[-1].item()) + 1)


def _resolve_region_layout(scene, region) -> _RegionLayout | None:
    geometry = region.geometry
    if not isinstance(geometry, Box):
        raise ValueError(
            f"MaterialRegion currently supports Box geometry only, got {type(geometry).__name__}."
        )

    pos = geometry.position
    sz_vec = geometry.size
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    sx, sy, sz = float(sz_vec[0]), float(sz_vec[1]), float(sz_vec[2])
    x_slice = _region_axis_slice(scene.x, cx - sx / 2.0, cx + sx / 2.0)
    y_slice = _region_axis_slice(scene.y, cy - sy / 2.0, cy + sy / 2.0)
    z_slice = _region_axis_slice(scene.z, cz - sz / 2.0, cz + sz / 2.0)
    if x_slice is None or y_slice is None or z_slice is None:
        return None

    return _RegionLayout(
        x_slice=x_slice,
        y_slice=y_slice,
        z_slice=z_slice,
        target_shape=(
            x_slice.stop - x_slice.start,
            y_slice.stop - y_slice.start,
            z_slice.stop - z_slice.start,
        ),
    )


def _resolved_density_kernel(region, density_shape: tuple[int, int, int], scene) -> tuple[int, int, int]:
    if region.filter_radius is None:
        return (1, 1, 1)
    kernel = _density_kernel_size(scene, region.filter_radius)
    return tuple(
        min(size, shape if shape % 2 == 1 else max(1, shape - 1))
        for size, shape in zip(kernel, density_shape)
    )


def _projection_derivative(density: torch.Tensor, beta: float) -> torch.Tensor:
    midpoint = density.new_tensor(0.5)
    denominator = torch.tanh(beta * midpoint) + torch.tanh(beta * (1.0 - midpoint))
    sech = torch.cosh(beta * (density - midpoint)).reciprocal()
    return beta * sech * sech / denominator


def _clamp_backward_mask(tensor: torch.Tensor, *, lower: float = 0.0, upper: float = 1.0) -> torch.Tensor:
    return ((tensor >= lower) & (tensor <= upper)).to(dtype=tensor.dtype)


def _pullback_region_density(scene, region, *, layout: _RegionLayout, grad_eps_patch: torch.Tensor) -> torch.Tensor:
    if not torch.count_nonzero(grad_eps_patch):
        return torch.zeros_like(region.density)

    density = region.density.to(device=scene.device, dtype=grad_eps_patch.dtype)
    lower, upper = (float(region.bounds[0]), float(region.bounds[1]))
    bound_scale = upper - lower
    if np.isclose(bound_scale, 0.0):
        return torch.zeros_like(region.density)

    normalized = density if np.isclose(lower, 0.0) and np.isclose(upper, 1.0) else (density - lower) / bound_scale
    normalized_clamped = normalized.clamp(0.0, 1.0)

    kernel = _resolved_density_kernel(region, tuple(int(v) for v in density.shape), scene)
    if kernel == (1, 1, 1):
        filtered = normalized_clamped
    else:
        filtered = F.avg_pool3d(
            normalized_clamped[None, None, ...],
            kernel_size=kernel,
            stride=1,
            padding=tuple(size // 2 for size in kernel),
        )[0, 0]

    if region.projection_beta is None:
        projected = filtered
    else:
        beta = float(region.projection_beta)
        midpoint = filtered.new_tensor(0.5)
        numerator = torch.tanh(beta * midpoint) + torch.tanh(beta * (filtered - midpoint))
        denominator = torch.tanh(beta * midpoint) + torch.tanh(beta * (1.0 - midpoint))
        projected = numerator / denominator
    projected_clamped = projected.clamp(0.0, 1.0)

    grad_density = grad_eps_patch * float(region.eps_bounds[1] - region.eps_bounds[0])
    if projected_clamped.shape != layout.target_shape:
        grad_density = torch.ops.aten.upsample_trilinear3d_backward(
            grad_density[None, None, ...].contiguous(),
            list(layout.target_shape),
            list(projected_clamped[None, None, ...].shape),
            False,
            None,
            None,
            None,
        )[0, 0]

    grad_density = grad_density * _clamp_backward_mask(projected)
    if region.projection_beta is not None:
        grad_density = grad_density * _projection_derivative(filtered, float(region.projection_beta))

    if kernel != (1, 1, 1):
        grad_density = torch.ops.aten.avg_pool3d_backward(
            grad_density[None, None, ...].contiguous(),
            normalized_clamped[None, None, ...].contiguous(),
            list(kernel),
            [1, 1, 1],
            [size // 2 for size in kernel],
            False,
            True,
            None,
        )[0, 0]

    grad_density = grad_density * _clamp_backward_mask(normalized)
    if not (np.isclose(lower, 0.0) and np.isclose(upper, 1.0)):
        grad_density = grad_density / bound_scale

    return grad_density.to(device=region.density.device, dtype=region.density.dtype)


def component_node_gradients_from_yee_permittivity(
    scene,
    *,
    grad_eps_ex: torch.Tensor,
    grad_eps_ey: torch.Tensor,
    grad_eps_ez: torch.Tensor,
    eps0: float,
) -> dict[str, torch.Tensor]:
    """Per-axis node gradients transposing the node->Yee-edge permittivity build.

    The forward path builds ``eps_E<axis> = 0.5 * (node_a + node_b) * eps0`` from
    the *per-axis* relative-permittivity node component, so each Yee-edge gradient
    scatters half onto its two endpoint nodes of that same axis component. Keeping
    the three axes separate is what makes diagonal-anisotropic media differentiable;
    for isotropic media summing the three components reproduces the scalar pullback
    exactly.
    """
    shape = (scene.Nx, scene.Ny, scene.Nz)
    grad_x = torch.zeros(shape, device=scene.device, dtype=grad_eps_ex.dtype)
    grad_y = torch.zeros(shape, device=scene.device, dtype=grad_eps_ey.dtype)
    grad_z = torch.zeros(shape, device=scene.device, dtype=grad_eps_ez.dtype)
    grad_x[:-1, :, :] += 0.5 * grad_eps_ex
    grad_x[1:, :, :] += 0.5 * grad_eps_ex
    grad_y[:, :-1, :] += 0.5 * grad_eps_ey
    grad_y[:, 1:, :] += 0.5 * grad_eps_ey
    grad_z[:, :, :-1] += 0.5 * grad_eps_ez
    grad_z[:, :, 1:] += 0.5 * grad_eps_ez
    scale = float(eps0)
    return {"x": grad_x * scale, "y": grad_y * scale, "z": grad_z * scale}


def node_gradient_from_yee_permittivity(
    scene,
    *,
    grad_eps_ex: torch.Tensor,
    grad_eps_ey: torch.Tensor,
    grad_eps_ez: torch.Tensor,
    eps0: float,
) -> torch.Tensor:
    component_gradients = component_node_gradients_from_yee_permittivity(
        scene,
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        eps0=eps0,
    )
    return component_gradients["x"] + component_gradients["y"] + component_gradients["z"]


def pullback_density_gradients(
    scene,
    *,
    density_tensors,
    trainable_region_indices,
    grad_eps_ex: torch.Tensor,
    grad_eps_ey: torch.Tensor,
    grad_eps_ez: torch.Tensor,
    eps0: float,
) -> tuple[torch.Tensor, ...]:
    if not density_tensors:
        return ()
    prepared_scene = prepare_scene(scene)

    grad_eps_r = node_gradient_from_yee_permittivity(
        prepared_scene,
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        eps0=eps0,
    )

    region_layouts = [
        _resolve_region_layout(prepared_scene, region)
        for region in prepared_scene.material_regions
    ]
    outputs = [torch.zeros_like(density) for density in density_tensors]
    output_positions = {
        int(region_index): position
        for position, region_index in enumerate(trainable_region_indices)
    }
    covered_mask = torch.zeros(
        (prepared_scene.Nx, prepared_scene.Ny, prepared_scene.Nz),
        device=prepared_scene.device,
        dtype=torch.bool,
    )

    for region_index in range(len(prepared_scene.material_regions) - 1, -1, -1):
        layout = region_layouts[region_index]
        if layout is None:
            continue

        region = prepared_scene.material_regions[region_index]
        region_slice = layout.region
        effective_mask = ~covered_mask[region_slice]

        output_position = output_positions.get(region_index)
        if output_position is not None:
            grad_patch = torch.where(
                effective_mask,
                grad_eps_r[region_slice],
                torch.zeros_like(grad_eps_r[region_slice]),
            )
            outputs[output_position] = _pullback_region_density(
                prepared_scene,
                region,
                layout=layout,
                grad_eps_patch=grad_patch,
            )

        covered_mask[region_slice] = True

    return tuple(outputs)


def pullback_material_input_gradients(
    scene,
    *,
    inputs,
    grad_eps_ex: torch.Tensor,
    grad_eps_ey: torch.Tensor,
    grad_eps_ez: torch.Tensor,
    eps0: float,
    grad_chi3_ex: torch.Tensor | None = None,
    grad_chi3_ey: torch.Tensor | None = None,
    grad_chi3_ez: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if not inputs:
        return ()
    prepared_scene = prepare_scene(scene)

    component_gradients = component_node_gradients_from_yee_permittivity(
        prepared_scene,
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        eps0=eps0,
    )
    # Back-propagate through the per-axis relative-permittivity node components
    # rather than their scalar average so diagonal-anisotropic sensitivities are
    # attributed to the correct axis. For isotropic media this is exactly the
    # previous scalar-eps pullback.
    model = prepared_scene.compile_materials()
    eps_components = model["eps_components"]
    outputs = []
    grad_outputs = []
    for axis in ("x", "y", "z"):
        component = eps_components[axis]
        if not component.requires_grad:
            continue
        outputs.append(component)
        grad_outputs.append(component_gradients[axis].to(device=component.device, dtype=component.dtype))

    if grad_chi3_ex is not None:
        # Kerr channel: the compiled chi3 node field feeds all three Yee-edge
        # chi3 tensors through the same two-point averaging (no eps0 scale), so
        # its node gradient is the sum of the three per-axis edge scatters.
        kerr_field = model["kerr_chi3"]
        if kerr_field.requires_grad:
            chi3_node_gradients = component_node_gradients_from_yee_permittivity(
                prepared_scene,
                grad_eps_ex=grad_chi3_ex,
                grad_eps_ey=grad_chi3_ey,
                grad_eps_ez=grad_chi3_ez,
                eps0=1.0,
            )
            chi3_node_gradient = (
                chi3_node_gradients["x"] + chi3_node_gradients["y"] + chi3_node_gradients["z"]
            )
            outputs.append(kerr_field)
            grad_outputs.append(chi3_node_gradient.to(device=kerr_field.device, dtype=kerr_field.dtype))

    gradients = (
        torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            allow_unused=True,
            retain_graph=False,
        )
        if outputs
        else (None,) * len(inputs)
    )
    prepared_scene.release_meshgrid()
    return tuple(
        torch.zeros_like(tensor)
        if grad is None
        else grad.to(device=tensor.device, dtype=tensor.dtype)
        for tensor, grad in zip(inputs, gradients)
    )
