"""Compile TerminalWavePort apertures, terminal node masks and custom integrals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..path_integrals import AxisSegment, CustomVoltageIntegral
from ..ports import TerminalWavePort
from ..scene import prepare_scene
from .path_integrals import (
    CompiledPathIntegral,
    PortPlane,
    compile_current_integral,
    compile_voltage_integral,
)
from .ports import _axis_values, _snap_index

_AXES = "xyz"


@dataclass(frozen=True)
class CompiledTerminalWavePort:
    """Grid-resolved TerminalWavePort aperture (no mode solve)."""

    port_name: str
    plane: PortPlane
    aperture_lower_indices: tuple[int, int, int]
    aperture_upper_indices: tuple[int, int, int]
    terminal_masks: tuple[tuple[int, ...], ...]
    reference_mask: tuple[int, ...]
    voltage_integrals: tuple[CompiledPathIntegral, ...] | None
    current_integrals: tuple[CompiledPathIntegral, ...] | None
    default_voltage_path: AxisSegment | None

    @property
    def electric_plane_index(self) -> int:
        return self.plane.electric_plane_index

    @property
    def transverse_shape(self) -> tuple[int, int]:
        return self.plane.shape

    def terminal_mode(self, potentials=None) -> dict:
        """Private mode-solver payload; ``potentials=None`` requests the basis."""
        return {
            "terminal_masks": self.terminal_masks,
            "reference_mask": self.reference_mask,
            "plane_shape": self.plane.shape,
            "potentials": None if potentials is None else tuple(float(value) for value in potentials),
        }


def _error(port, message):
    return ValueError(f"TerminalWavePort {port.name!r}: {message}")


def _compile_plane(scene, port: TerminalWavePort) -> tuple[PortPlane, tuple, tuple]:
    position = np.asarray(port.position, dtype=np.float64)
    size = np.asarray(port.size, dtype=np.float64)
    lower = position - 0.5 * size
    upper = position + 0.5 * size
    roundoff = np.asarray(port._position_roundoff) + .5 * np.asarray(port._size_roundoff)
    scale = max(1.0, *(abs(float(value)) for bounds in scene.domain.bounds for value in bounds))
    for axis, low, high, bounds, uncertainty in zip(_AXES, lower, upper, scene.domain.bounds, roundoff):
        if low < float(bounds[0]) - 1e-12 * scale - uncertainty or high > float(bounds[1]) + 1e-12 * scale + uncertainty:
            raise _error(port, f"aperture lies outside the Scene domain along {axis}.")
    normal_index = _AXES.index(port.normal_axis)
    tangential = tuple(axis for axis in _AXES if axis != port.normal_axis)
    plane_index = _snap_index(
        _axis_values(scene, port.normal_axis, half=False),
        port.reference_plane,
        label=f"{port.normal_axis} aperture plane",
        roundoff=port._reference_roundoff,
    )
    lower_indices = [plane_index] * 3
    upper_indices = [plane_index] * 3
    for axis in tangential:
        index = _AXES.index(axis)
        coords = _axis_values(scene, axis, half=False)
        low = _snap_index(coords, lower[index], label=f"{axis} aperture lower bound", roundoff=roundoff[index])
        high = _snap_index(coords, upper[index], label=f"{axis} aperture upper bound", roundoff=roundoff[index])
        if high - low < 2:
            raise _error(port, f"aperture must span at least two cells along {axis}.")
        lower_indices[index], upper_indices[index] = low, high
    half_coords = _axis_values(scene, port.normal_axis, half=True)
    magnetic_index = plane_index if port.direction_sign > 0 else plane_index - 1
    if magnetic_index < 0 or magnetic_index >= len(half_coords):
        raise _error(port, "aperture direction has no adjacent magnetic half-grid plane inside the grid.")
    plane = PortPlane(
        normal_axis=port.normal_axis,
        tangential_axes=tangential,
        electric_plane_index=plane_index,
        magnetic_plane_index=magnetic_index,
        lower=tuple(lower_indices[_AXES.index(axis)] for axis in tangential),
        upper=tuple(upper_indices[_AXES.index(axis)] for axis in tangential),
        reference_plane=float(_axis_values(scene, port.normal_axis, half=False)[plane_index]),
        direction_sign=port.direction_sign,
    )
    return plane, tuple(lower_indices), tuple(upper_indices)


def _support_mask(scene, plane: PortPlane, support) -> tuple[int, ...]:
    coords = [
        _axis_values(scene, axis, half=False)[plane.lower[slot]:plane.upper[slot] + 1]
        for slot, axis in enumerate(plane.tangential_axes)
    ]
    normal_index = _AXES.index(plane.normal_axis)
    spacing = min(float(np.min(np.diff(values))) for values in coords)
    tolerance = 1e-6 * spacing
    flat = []
    for i, u in enumerate(coords[0]):
        for j, v in enumerate(coords[1]):
            point = np.zeros(3)
            point[normal_index] = plane.reference_plane
            point[_AXES.index(plane.tangential_axes[0])] = u
            point[_AXES.index(plane.tangential_axes[1])] = v
            if support.contains(point, tolerance):
                flat.append(i * len(coords[1]) + j)
    return tuple(flat)


def _default_voltage_path(scene, plane: PortPlane, record: dict, port) -> AxisSegment:
    """Snap the resolved default segment to Yee nodes inside both conductors."""
    path_axis, offset_axis = record["path_axis"], record["offset_axis"]
    nodes_path = _axis_values(scene, path_axis, half=False)
    nodes_offset = _axis_values(scene, offset_axis, half=False)
    spacing = float(np.min(np.diff(nodes_path)))
    tolerance = 1e-6 * spacing

    def nearest(nodes, target, interval):
        candidates = np.flatnonzero((nodes >= interval[0] - tolerance) & (nodes <= interval[1] + tolerance))
        if candidates.size == 0:
            raise _error(port, f"default voltage path: no Yee node lies within the conductor span {interval}.")
        return float(nodes[candidates[np.argmin(np.abs(nodes[candidates] - target))]])

    start = [0.0, 0.0, 0.0]
    end = [0.0, 0.0, 0.0]
    normal_index = _AXES.index(plane.normal_axis)
    start[normal_index] = end[normal_index] = plane.reference_plane
    offset = nearest(nodes_offset, record["offset"], record["offset_interval"])
    start[_AXES.index(offset_axis)] = end[_AXES.index(offset_axis)] = offset
    start[_AXES.index(path_axis)] = nearest(nodes_path, record["start"], record["start_interval"])
    end[_AXES.index(path_axis)] = nearest(nodes_path, record["end"], record["end_interval"])
    return AxisSegment(tuple(start), tuple(end))


def compile_terminal_waveport(scene, port: TerminalWavePort, *, device=None) -> CompiledTerminalWavePort:
    """Compile one TerminalWavePort without invoking a mode solve."""
    from ..geometry.terminal_support import wave_terminal_supports

    if not isinstance(port, TerminalWavePort):
        raise TypeError("compile_terminal_waveport expects a TerminalWavePort.")
    resolved = prepare_scene(scene)
    target_device = torch.device(resolved.device if device is None else device)
    try:
        plane, lower_indices, upper_indices = _compile_plane(resolved, port)
    except ValueError as error:
        if str(error).startswith("TerminalWavePort"):
            raise
        raise _error(port, str(error)) from error
    terminal_supports, reference_supports = wave_terminal_supports(port, resolved.structures)
    terminal_masks = tuple(_support_mask(resolved, plane, support) for support in terminal_supports)
    reference_mask = tuple(sorted({index for support in reference_supports
                                   for index in _support_mask(resolved, plane, support)}))
    for index, mask in enumerate(terminal_masks):
        if not mask:
            raise _error(port, f"terminal {index + 1} ({port.terminals[index].structure_name!r}) covers no aperture node.")
        if set(mask) & set(reference_mask):
            raise _error(port, f"terminal {index + 1} overlaps the reference conductor on the port plane.")
    default_path = None
    voltage_specs = port.voltage_integrals
    if voltage_specs is None:
        if port._default_voltage_path is None:
            raise _error(port, "the default voltage path is unresolved; add the port to its Scene first.")
        default_path = _default_voltage_path(resolved, plane, port._default_voltage_path, port)
        voltage_specs = (CustomVoltageIntegral(default_path),)
    try:
        voltages = tuple(compile_voltage_integral(resolved, spec, plane, device=target_device) for spec in voltage_specs)
        currents = None
        if port.current_integrals is not None:
            currents = tuple(compile_current_integral(resolved, spec, plane, device=target_device)
                             for spec in port.current_integrals)
    except ValueError as error:
        raise _error(port, str(error)) from error
    return CompiledTerminalWavePort(
        port_name=port.name,
        plane=plane,
        aperture_lower_indices=lower_indices,
        aperture_upper_indices=upper_indices,
        terminal_masks=terminal_masks,
        reference_mask=reference_mask,
        voltage_integrals=voltages,
        current_integrals=currents,
        default_voltage_path=default_path,
    )


def compile_terminal_waveports(scene, ports=None, *, device=None) -> tuple[CompiledTerminalWavePort, ...]:
    selected = tuple(port for port in scene.ports if isinstance(port, TerminalWavePort)) if ports is None else tuple(ports)
    return tuple(compile_terminal_waveport(scene, port, device=device) for port in selected)


__all__ = ["CompiledTerminalWavePort", "compile_terminal_waveport", "compile_terminal_waveports"]
