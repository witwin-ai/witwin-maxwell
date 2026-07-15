from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from ..ports import AxisPath, WaveModeSpec, WavePort
from ..scene import prepare_scene
from ..sources import _resolve_mode_source_polarization_axis, polarization_vector
from .ports import (
    _axis_values,
    _compile_current,
    _compile_voltage_path,
    _snap_index,
)


_AXES = "xyz"


@dataclass(frozen=True)
class CompiledWaveModeSpec:
    """Grid-resolved modal identity and impedance-normalization geometry."""

    mode_name: str
    tracking_id: str
    family: str
    mode_index: int
    polarization_axis: str
    polarization: tuple[float, float, float]
    impedance_definition: str
    impedance_formula: str
    voltage_component: str | None
    voltage_direction: int | None
    voltage_indices: torch.Tensor | None
    voltage_weights: torch.Tensor | None
    current_components: tuple[str, ...]
    current_indices: tuple[torch.Tensor, ...]
    current_weights: tuple[torch.Tensor, ...]
    current_plane_index: int | None
    current_plane_coordinate: float | None


@dataclass(frozen=True)
class CompiledWavePortCrossSection:
    """One axis-aligned WavePort aperture prepared for a later mode solve."""

    port_name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    normal_axis: str
    tangential_axes: tuple[str, str]
    direction: str
    direction_sign: int
    reference_plane: float
    electric_plane_index: int
    aperture_lower_indices: tuple[int, int, int]
    aperture_upper_indices: tuple[int, int, int]
    transverse_shape: tuple[int, int]
    aperture_indices: torch.Tensor
    modes: tuple[CompiledWaveModeSpec, ...]
    phasor_convention: str = "peak"
    power_convention: str = "0.5*Re(V*conj(I))"

    @property
    def device(self) -> torch.device:
        return self.aperture_indices.device


def _port_error(port: WavePort, message: str) -> ValueError:
    return ValueError(f"WavePort {port.name!r}: {message}")


def _geometry_values(port: WavePort, mode: WaveModeSpec):
    contour = mode.current_contour
    if contour is None:
        return None
    position = torch.as_tensor(contour.position).detach().cpu().to(dtype=torch.float64)
    size = torch.as_tensor(contour.size).detach().cpu().to(dtype=torch.float64)
    rotation = torch.as_tensor(contour.rotation).detach().cpu().to(dtype=torch.float64)
    identity = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
    if not torch.allclose(rotation, identity, rtol=0.0, atol=1.0e-12):
        raise _port_error(
            port,
            f"mode {mode.name!r} current_contour must be an unrotated axis-aligned Box.",
        )
    if not bool(torch.all(torch.isfinite(position))) or not bool(torch.all(torch.isfinite(size))):
        raise _port_error(port, f"mode {mode.name!r} current_contour must be finite.")
    return position.numpy(), size.numpy()


def _validate_in_domain(port: WavePort, lower: np.ndarray, upper: np.ndarray, domain_bounds) -> None:
    scale = max(1.0, *(abs(float(value)) for bounds in domain_bounds for value in bounds))
    tolerance = 1.0e-12 * scale
    for axis, value_lower, value_upper, bounds in zip(_AXES, lower, upper, domain_bounds):
        domain_lower, domain_upper = (float(value) for value in bounds)
        if value_lower < domain_lower - tolerance or value_upper > domain_upper + tolerance:
            raise _port_error(port, f"aperture lies outside the Scene domain along {axis}.")


def _compile_aperture(scene, port: WavePort, *, device: torch.device):
    position = np.asarray(port.position, dtype=np.float64)
    size = np.asarray(port.size, dtype=np.float64)
    lower = position - 0.5 * size
    upper = position + 0.5 * size
    _validate_in_domain(port, lower, upper, scene.domain.bounds)

    normal_index = _AXES.index(port.normal_axis)
    tangential_axes = tuple(axis for axis in _AXES if axis != port.normal_axis)
    try:
        plane_index = _snap_index(
            _axis_values(scene, port.normal_axis, half=False),
            port.reference_plane,
            label=f"{port.normal_axis} aperture plane",
        )
        lower_indices = [plane_index, plane_index, plane_index]
        upper_indices = [plane_index, plane_index, plane_index]
        transverse_ranges = []
        for axis in tangential_axes:
            axis_index = _AXES.index(axis)
            coords = _axis_values(scene, axis, half=False)
            lower_index = _snap_index(coords, lower[axis_index], label=f"{axis} aperture lower bound")
            upper_index = _snap_index(coords, upper[axis_index], label=f"{axis} aperture upper bound")
            if lower_index >= upper_index:
                raise _port_error(port, f"aperture must span at least one cell along {axis}.")
            lower_indices[axis_index] = lower_index
            upper_indices[axis_index] = upper_index
            transverse_ranges.append(
                torch.arange(lower_index, upper_index + 1, device=device, dtype=torch.int64)
            )
    except ValueError as error:
        if str(error).startswith(f"WavePort {port.name!r}:"):
            raise
        raise _port_error(port, str(error)) from error

    first, second = torch.meshgrid(*transverse_ranges, indexing="ij")
    indices = torch.empty((*first.shape, 3), device=device, dtype=torch.int64)
    indices[..., normal_index] = plane_index
    indices[..., _AXES.index(tangential_axes[0])] = first
    indices[..., _AXES.index(tangential_axes[1])] = second
    return (
        lower,
        upper,
        plane_index,
        tuple(lower_indices),
        tuple(upper_indices),
        tangential_axes,
        indices,
    )


def _compile_voltage_geometry(
    scene,
    port: WavePort,
    mode: WaveModeSpec,
    aperture_lower: np.ndarray,
    aperture_upper: np.ndarray,
    *,
    device: torch.device,
):
    if mode.voltage_path is None:
        return None, None, None, None
    negative, positive = mode.voltage_path
    negative_array = np.asarray(negative, dtype=np.float64)
    positive_array = np.asarray(positive, dtype=np.float64)
    normal_index = _AXES.index(port.normal_axis)
    tolerance = 1.0e-12 * max(1.0, abs(port.reference_plane))
    if (
        abs(negative_array[normal_index] - port.reference_plane) > tolerance
        or abs(positive_array[normal_index] - port.reference_plane) > tolerance
    ):
        raise _port_error(
            port,
            f"mode {mode.name!r} voltage_path must lie in the aperture plane.",
        )
    varying_axis_index = next(
        index
        for index, (negative_value, positive_value) in enumerate(zip(negative, positive))
        if not np.isclose(negative_value, positive_value, rtol=0.0, atol=1.0e-12)
    )
    if varying_axis_index == normal_index:
        raise _port_error(
            port,
            f"mode {mode.name!r} voltage_path axis must be tangential to the aperture.",
        )
    if bool(
        np.any(negative_array < aperture_lower - tolerance)
        or np.any(negative_array > aperture_upper + tolerance)
        or np.any(positive_array < aperture_lower - tolerance)
        or np.any(positive_array > aperture_upper + tolerance)
    ):
        raise _port_error(port, f"mode {mode.name!r} voltage_path lies outside the aperture.")
    try:
        return _compile_voltage_path(
            scene,
            positive=positive,
            negative=negative,
            axis=_AXES[varying_axis_index],
            device=device,
        )
    except ValueError as error:
        raise _port_error(port, f"mode {mode.name!r} voltage_path: {error}") from error


def _compile_current_geometry(
    scene,
    port: WavePort,
    mode: WaveModeSpec,
    aperture_lower: np.ndarray,
    aperture_upper: np.ndarray,
    electric_plane_index: int,
    *,
    device: torch.device,
):
    values = _geometry_values(port, mode)
    if values is None:
        return (), (), (), None, None
    position, size = values
    normal_index = _AXES.index(port.normal_axis)
    grid_scale = max(
        1.0,
        *(float(np.max(np.abs(_axis_values(scene, axis, half=False)))) for axis in _AXES),
    )
    tolerance = 1.0e-12 * grid_scale
    zero_axes = tuple(index for index, extent in enumerate(size) if abs(float(extent)) <= tolerance)
    if len(zero_axes) != 1:
        raise _port_error(port, f"mode {mode.name!r} current_contour must be planar.")
    if zero_axes[0] != normal_index:
        raise _port_error(
            port,
            f"mode {mode.name!r} current_contour normal must match the WavePort normal.",
        )
    for axis_index, axis in enumerate(_AXES):
        if axis_index == normal_index:
            continue
        if size[axis_index] <= 0.0:
            raise _port_error(
                port,
                f"mode {mode.name!r} current_contour must have positive tangential extents.",
            )
        lower = position[axis_index] - 0.5 * size[axis_index]
        upper = position[axis_index] + 0.5 * size[axis_index]
        if lower < aperture_lower[axis_index] - tolerance or upper > aperture_upper[axis_index] + tolerance:
            raise _port_error(
                port,
                f"mode {mode.name!r} current_contour lies outside the aperture along {axis}.",
            )

    half_coords = _axis_values(scene, port.normal_axis, half=True)
    expected_half_index = electric_plane_index if port.direction_sign > 0 else electric_plane_index - 1
    if expected_half_index < 0 or expected_half_index >= len(half_coords):
        raise _port_error(
            port,
            "aperture direction has no adjacent magnetic half-grid plane inside the grid.",
        )
    try:
        actual_half_index = _snap_index(
            half_coords,
            float(position[normal_index]),
            label=f"mode {mode.name} current contour plane",
        )
    except ValueError as error:
        raise _port_error(port, str(error)) from error
    if actual_half_index != expected_half_index:
        expected = float(half_coords[expected_half_index])
        raise _port_error(
            port,
            f"mode {mode.name!r} current_contour plane must be the direction-adjacent "
            f"magnetic half-grid plane at {expected:g}.",
        )

    span = max(
        1.0,
        float(scene.domain.bounds[normal_index][1] - scene.domain.bounds[normal_index][0]),
    )
    negative = list(port.position)
    positive = list(port.position)
    negative[normal_index] = float(position[normal_index] - span)
    positive[normal_index] = float(position[normal_index] + span)
    proxy = SimpleNamespace(
        voltage_path=AxisPath(port.normal_axis),
        negative=tuple(negative),
        positive=tuple(positive),
        current_surface=mode.current_contour,
    )
    try:
        components, indices, weights = _compile_current(
            proxy,
            scene,
            port.direction_sign,
            device=device,
        )
    except ValueError as error:
        raise _port_error(port, f"mode {mode.name!r} current_contour: {error}") from error
    return components, indices, weights, actual_half_index, float(position[normal_index])


def _compile_mode(
    scene,
    port: WavePort,
    mode: WaveModeSpec,
    aperture_lower: np.ndarray,
    aperture_upper: np.ndarray,
    electric_plane_index: int,
    *,
    device: torch.device,
) -> CompiledWaveModeSpec:
    polarization_axis = _resolve_mode_source_polarization_axis(
        port.normal_axis,
        port.size,
        mode.polarization,
    )
    voltage_direction, voltage_component, voltage_indices, voltage_weights = (
        _compile_voltage_geometry(
            scene,
            port,
            mode,
            aperture_lower,
            aperture_upper,
            device=device,
        )
    )
    current_components, current_indices, current_weights, current_plane_index, current_plane = (
        _compile_current_geometry(
            scene,
            port,
            mode,
            aperture_lower,
            aperture_upper,
            electric_plane_index,
            device=device,
        )
    )
    return CompiledWaveModeSpec(
        mode_name=mode.name,
        tracking_id=port.mode_name(mode),
        family=mode.family,
        mode_index=mode.mode_index,
        polarization_axis=polarization_axis,
        polarization=polarization_vector(f"E{polarization_axis}"),
        impedance_definition=mode.impedance_definition,
        impedance_formula=mode.impedance_formula,
        voltage_component=voltage_component,
        voltage_direction=voltage_direction,
        voltage_indices=voltage_indices,
        voltage_weights=voltage_weights,
        current_components=current_components,
        current_indices=current_indices,
        current_weights=current_weights,
        current_plane_index=current_plane_index,
        current_plane_coordinate=current_plane,
    )


def compile_waveport_cross_section(
    scene,
    port: WavePort,
    *,
    device: str | torch.device | None = None,
) -> CompiledWavePortCrossSection:
    """Compile one WavePort aperture and its normalization geometry only."""

    if not isinstance(port, WavePort):
        raise TypeError("compile_waveport_cross_section expects a WavePort.")
    resolved_scene = prepare_scene(scene)
    target_device = torch.device(resolved_scene.device if device is None else device)
    (
        aperture_lower,
        aperture_upper,
        plane_index,
        lower_indices,
        upper_indices,
        tangential_axes,
        aperture_indices,
    ) = _compile_aperture(resolved_scene, port, device=target_device)
    modes = tuple(
        _compile_mode(
            resolved_scene,
            port,
            mode,
            aperture_lower,
            aperture_upper,
            plane_index,
            device=target_device,
        )
        for mode in port.modes
    )
    return CompiledWavePortCrossSection(
        port_name=port.name,
        position=port.position,
        size=port.size,
        normal_axis=port.normal_axis,
        tangential_axes=tangential_axes,
        direction=port.direction,
        direction_sign=port.direction_sign,
        reference_plane=port.reference_plane,
        electric_plane_index=plane_index,
        aperture_lower_indices=lower_indices,
        aperture_upper_indices=upper_indices,
        transverse_shape=tuple(int(value) for value in aperture_indices.shape[:2]),
        aperture_indices=aperture_indices,
        modes=modes,
        phasor_convention=port.phasor_convention,
        power_convention=port.power_convention,
    )


def compile_waveports(
    scene,
    ports=None,
    *,
    device: str | torch.device | None = None,
) -> tuple[CompiledWavePortCrossSection, ...]:
    """Compile WavePort declarations without invoking the mode solver."""

    if ports is None:
        selected_ports = tuple(port for port in scene.ports if isinstance(port, WavePort))
    else:
        selected_ports = tuple(ports)
        if any(not isinstance(port, WavePort) for port in selected_ports):
            raise TypeError("compile_waveports expects only WavePort instances.")
    return tuple(
        compile_waveport_cross_section(scene, port, device=device)
        for port in selected_ports
    )


__all__ = [
    "CompiledWaveModeSpec",
    "CompiledWavePortCrossSection",
    "compile_waveport_cross_section",
    "compile_waveports",
]
