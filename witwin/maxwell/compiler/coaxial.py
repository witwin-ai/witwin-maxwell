"""Annular (coaxial) lumped voltage rows and current loops on the Yee grid."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np
import torch

from .ports import _AXES, _TANGENTIAL_AXES, _axis_values, _axis_widths, _integrate_sparse_term, _snap_index


@dataclass(frozen=True)
class CompiledCoaxialGeometry:
    """Sparse multi-component voltage row and H loop of one annular lumped object.

    ``voltage_weights`` integrate ``r_hat . E / (2 pi r)`` exactly along every
    tangential edge clipped to the annulus, so ``V = sum w E`` is the azimuthal
    mean of the inner-to-outer radial voltage. The same row distributes the
    coupled current, giving the radial surface density ``I / (2 pi r)``.
    ``current_*`` encode the H circulation around the inner conductor on the
    ``direction`` side of the plane, signed as inner-conductor current flowing
    along ``direction``; resistors carry no current loop.
    """

    port_name: str
    axis: str
    direction: int
    plane_index: int
    voltage_components: tuple[str, ...]
    voltage_indices: tuple[torch.Tensor, ...]
    voltage_weights: tuple[torch.Tensor, ...]
    current_components: tuple[str, ...] = ()
    current_indices: tuple[torch.Tensor, ...] = ()
    current_weights: tuple[torch.Tensor, ...] = ()
    reference_impedance: Any = None
    reference_plane: float | None = None
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_coaxial(self) -> bool:
        return True

    def integrate_voltage(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        result = None
        for component, indices, weights in zip(self.voltage_components, self.voltage_indices, self.voltage_weights):
            term = _integrate_sparse_term(fields, component, indices, weights)
            result = term if result is None else result + term
        return result

    def integrate_current(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not self.current_components:
            raise RuntimeError(f"Coaxial object {self.port_name!r} has no current loop.")
        result = None
        for component, indices, weights in zip(self.current_components, self.current_indices, self.current_weights):
            term = _integrate_sparse_term(fields, component, indices, weights)
            result = term if result is None else result + term
        return result


def _radial_log_weight(lower, upper, offset, inner, outer):
    """Exact ``int U / (U^2 + offset^2) dU`` over ``[lower, upper]`` clipped to the annulus."""
    offset_sq = offset * offset
    outer_half = np.sqrt(np.maximum(outer * outer - offset_sq, 0.0))
    inner_half = np.sqrt(np.maximum(inner * inner - offset_sq, 0.0))
    total = np.zeros_like(lower)
    # Positive and negative annulus chords; an offset beyond the inner radius
    # makes the chords meet at U = 0, which the zero-width inner chord handles.
    for chord_low, chord_high in ((inner_half, outer_half), (-outer_half, -inner_half)):
        low = np.maximum(lower, chord_low)
        high = np.minimum(upper, chord_high)
        valid = high > low
        safe_low = np.where(valid, low, 1.0)
        safe_high = np.where(valid, high, 1.0)
        value = 0.5 * (np.log(safe_high * safe_high + offset_sq) - np.log(safe_low * safe_low + offset_sq))
        total = total + np.where(valid, value, 0.0)
    return total / (2.0 * math.pi)


def _component_row(scene, axis_along, axis_across, normal_axis, plane_index, center, inner, outer, device):
    along = _AXES.index(axis_along)
    across = _AXES.index(axis_across)
    along_nodes = _axis_values(scene, axis_along, half=False)
    across_nodes = _axis_values(scene, axis_across, half=False)
    across_dual = _axis_widths(scene, axis_across, dual=True)
    edge_low = along_nodes[:-1, None] - center[along]
    edge_high = along_nodes[1:, None] - center[along]
    offset = across_nodes[None, :] - center[across]
    weights = _radial_log_weight(edge_low, edge_high, offset, inner, outer) * across_dual[None, :]
    keep = np.nonzero(weights != 0.0)
    indices = np.empty((keep[0].size, 3), dtype=np.int64)
    indices[:, along] = keep[0]
    indices[:, across] = keep[1]
    indices[:, _AXES.index(normal_axis)] = plane_index
    return (
        f"E{axis_along}",
        torch.as_tensor(indices, device=device, dtype=torch.int64),
        torch.as_tensor(weights[keep], device=device, dtype=torch.float64),
    )


def _current_loop(scene, normal_axis, plane_index, direction, center, radius, device):
    u_axis, v_axis = _TANGENTIAL_AXES[normal_axis]
    u, v, n = (_AXES.index(axis) for axis in (u_axis, v_axis, normal_axis))
    half = _axis_values(scene, normal_axis, half=True)
    half_index = plane_index if direction > 0 else plane_index - 1
    if not 0 <= half_index < half.size:
        raise ValueError("the current loop has no magnetic half plane on the direction side inside the grid.")
    u_nodes = _axis_values(scene, u_axis, half=False)
    v_nodes = _axis_values(scene, v_axis, half=False)
    du = _axis_widths(scene, u_axis, dual=True)
    dv = _axis_widths(scene, v_axis, dual=True)
    inside = ((u_nodes[:, None] - center[u]) ** 2 + (v_nodes[None, :] - center[v]) ** 2) < radius * radius
    if not inside.any():
        raise ValueError("the mid-radius current loop encloses no Yee node; refine the grid.")
    padded = np.pad(inside, 1)
    # Boundary dual edges of the union of enclosed node cells, right-handed about +n.
    # H_u[i, j] sits between nodes (i, j) and (i, j + 1); H_v[i, j] between (i, j) and (i + 1, j).
    hu = padded[1:-1, 2:-1].astype(np.int8) - padded[1:-1, 1:-2].astype(np.int8)
    hv = padded[1:-2, 1:-1].astype(np.int8) - padded[2:-1, 1:-1].astype(np.int8)
    components, indices, weights = [], [], []
    for name, orientation, cell_width in ((f"H{u_axis}", hu, du[:, None]), (f"H{v_axis}", hv, dv[None, :])):
        keep = np.nonzero(orientation)
        entries = np.empty((keep[0].size, 3), dtype=np.int64)
        entries[:, u] = keep[0]
        entries[:, v] = keep[1]
        entries[:, n] = half_index
        width = np.broadcast_to(cell_width, orientation.shape)[keep]
        components.append(name)
        indices.append(torch.as_tensor(entries, device=device, dtype=torch.int64))
        weights.append(torch.as_tensor(direction * orientation[keep] * width, device=device, dtype=torch.float64))
    return tuple(components), tuple(indices), tuple(weights), half_index


def compile_coaxial_geometry(scene, item, *, device=None, with_current: bool = True) -> CompiledCoaxialGeometry:
    """Compile a CoaxialLumpedPort or CoaxialLumpedResistor annulus into sparse Yee weights."""
    from ..scene import prepare_scene

    resolved = prepare_scene(scene)
    target = torch.device(resolved.device if device is None else device)
    label = f"{type(item).__name__} {item.name!r}"
    normal_axis = item.normal_axis
    n = _AXES.index(normal_axis)
    center = item.center
    roundoff = getattr(item, "_center_roundoff", ()) or (0.0, 0.0, 0.0)
    inner, outer = 0.5 * item.inner_diameter, 0.5 * item.outer_diameter
    try:
        plane_index = _snap_index(
            _axis_values(resolved, normal_axis, half=False), center[n],
            label=f"{normal_axis} coaxial node plane", roundoff=roundoff[n],
        )
        u_axis, v_axis = _TANGENTIAL_AXES[normal_axis]
        for axis in (u_axis, v_axis):
            nodes = _axis_values(resolved, axis, half=False)
            c = center[_AXES.index(axis)]
            if c - outer < nodes[0] or c + outer > nodes[-1]:
                raise ValueError(f"the annulus leaves the grid along {axis}.")
            if np.count_nonzero((np.abs(nodes - c) > inner) & (np.abs(nodes - c) < outer)) < 4:
                raise ValueError(f"the annulus spans fewer than two Yee cells per side along {axis}; refine the grid.")
        # Staircased conductor surfaces deviate from the circles by up to a cell.
        # The row extends two local cells past each radius so the radial sheet
        # reaches both discrete PEC surfaces; edges inside a conductor are
        # dropped at runtime preparation. ln(r) is harmonic, so the extended
        # sheet stays divergence-free at every vacuum node.
        slack = 2.0 * max(
            float(np.max(_axis_widths(resolved, axis, dual=False)[
                np.abs(0.5 * (_axis_values(resolved, axis, half=False)[:-1]
                              + _axis_values(resolved, axis, half=False)[1:]) - center[_AXES.index(axis)])
                <= outer + 1.0e-300]))
            for axis in (u_axis, v_axis)
        )
        row_inner, row_outer = max(inner - slack, 0.5 * inner), outer + slack
        rows = (
            _component_row(resolved, u_axis, v_axis, normal_axis, plane_index, center, row_inner, row_outer, target),
            _component_row(resolved, v_axis, u_axis, normal_axis, plane_index, center, row_inner, row_outer, target),
        )
        direction = int(getattr(item, "direction_sign", 1))
        loop = ((), (), (), None)
        if with_current:
            loop = _current_loop(resolved, normal_axis, plane_index, direction, center, 0.5 * (inner + outer), target)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error
    return CompiledCoaxialGeometry(
        port_name=item.name,
        axis=normal_axis,
        direction=direction,
        plane_index=plane_index,
        voltage_components=tuple(row[0] for row in rows),
        voltage_indices=tuple(row[1] for row in rows),
        voltage_weights=tuple(row[2] for row in rows),
        current_components=loop[0],
        current_indices=loop[1],
        current_weights=loop[2],
        reference_impedance=getattr(item, "reference_impedance", None),
        reference_plane=getattr(item, "reference_plane", None),
        metadata={"center": tuple(float(value) for value in center), "inner_radius": inner, "outer_radius": outer, "row_inner_radius": row_inner,
                  "row_outer_radius": row_outer, "current_half_index": loop[3]},
    )


__all__ = ["CompiledCoaxialGeometry", "compile_coaxial_geometry"]
