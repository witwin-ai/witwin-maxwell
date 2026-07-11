from __future__ import annotations

from types import SimpleNamespace

import torch

from ..compiler.sources import _compile_mode_source
from ..fdtd.excitation.modes import sample_mode_source_component, solve_mode_source_profile
from ..scene import prepare_scene
from ..sources import CW, ModeSource
from .stratton_chu import (
    _as_1d_coords,
    _resolve_complex_dtype,
    _resolve_real_dtype,
    _resolve_tensor_device,
    _to_complex_tensor,
    _trapz_weights_1d,
    build_plane_points,
)


_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
_INDEX_TO_AXIS = ("x", "y", "z")
_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_POWER_FLOOR = 1e-30


def _mode_source_from_mode_spec(mode_spec: dict[str, object], *, frequency: float) -> ModeSource:
    polarization_axis = mode_spec.get("polarization_axis")
    if polarization_axis is not None:
        polarization = f"E{str(polarization_axis)}"
    else:
        polarization = tuple(float(component) for component in mode_spec["polarization"])
    return ModeSource(
        position=tuple(float(value) for value in mode_spec["position"]),
        size=tuple(float(value) for value in mode_spec["size"]),
        mode_index=int(mode_spec["mode_index"]),
        direction=str(mode_spec["direction"]),
        polarization=polarization,
        source_time=CW(frequency=float(frequency), amplitude=1.0),
        name=mode_spec.get("name"),
        bend_radius=mode_spec.get("bend_radius"),
        bend_axis=mode_spec.get("bend_axis"),
    )


def _resolve_mode_source(scene, monitor=None, *, mode_source=None, frequency: float) -> ModeSource:
    if mode_source is None and monitor is not None and monitor.get("mode_spec") is not None:
        return _mode_source_from_mode_spec(monitor["mode_spec"], frequency=float(frequency))

    mode_sources = [source for source in getattr(scene, "sources", ()) if isinstance(source, ModeSource)]
    for port in getattr(scene, "ports", ()):
        port_source = port.to_mode_source()
        if port_source is not None:
            mode_sources.append(port_source)
    if mode_source is None:
        if len(mode_sources) != 1:
            raise ValueError("mode_source must be provided when the scene does not contain exactly one ModeSource.")
        return mode_sources[0]
    if isinstance(mode_source, ModeSource):
        return mode_source
    if isinstance(mode_source, str):
        matches = [source for source in mode_sources if source.name == mode_source]
        if len(matches) != 1:
            raise ValueError(f"ModeSource {mode_source!r} was not uniquely found on result.scene.")
        return matches[0]
    raise TypeError("mode_source must be None, a ModeSource instance, or a ModeSource name.")


def _mode_overlap_context(result):
    solver = getattr(result, "solver", None)
    if solver is not None and hasattr(solver, "_compiled_material_model"):
        return solver

    scene = result.prepared_scene if hasattr(result, "prepared_scene") else prepare_scene(result.scene)
    device = torch.device(scene.device)
    return SimpleNamespace(
        scene=scene,
        Ex=torch.empty((1,), device=device, dtype=torch.float32),
        c=299792458.0,
        boundary_kind=scene.boundary.kind,
        _compiled_material_model=scene.compile_materials(),
    )


def _monitor_payload(result, monitor_name: str, *, frequency: float | None, freq_index: int | None):
    if hasattr(result, "raw_monitor"):
        monitor = result.raw_monitor(monitor_name, frequency=frequency, freq_index=freq_index)
    else:
        monitor = result.monitor(monitor_name, frequency=frequency, freq_index=freq_index, resolve_modal=False)
    if "axis" not in monitor:
        raise ValueError(f"Monitor {monitor_name!r} is not a plane monitor.")
    return monitor


def _monitor_coords(monitor):
    axis = str(monitor["axis"]).lower()
    tangential_axes = tuple(label for label in "xyz" if label != axis)
    coord_a = monitor.get(tangential_axes[0])
    coord_b = monitor.get(tangential_axes[1])
    if coord_a is None or coord_b is None:
        coords = monitor.get("coords")
        if coords is None or len(coords) != 2:
            raise ValueError("Plane monitor does not expose aligned tangential coordinates.")
        coord_a, coord_b = coords
    return axis, tangential_axes, coord_a, coord_b


def _surface_normal(axis: str, normal_direction: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    normal = torch.zeros((3,), device=device, dtype=dtype)
    normal[_AXIS_TO_INDEX[axis]] = 1.0 if str(normal_direction) != "-" else -1.0
    return normal


def _monitor_vector_fields(monitor, shape, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    electric = torch.zeros(shape + (3,), device=device, dtype=dtype)
    magnetic = torch.zeros(shape + (3,), device=device, dtype=dtype)
    for index, axis_label in enumerate(_INDEX_TO_AXIS):
        electric_name = f"E{axis_label}"
        magnetic_name = f"H{axis_label}"
        if electric_name in monitor:
            electric[..., index] = _to_complex_tensor(monitor[electric_name], device=device, dtype=dtype)
        if magnetic_name in monitor:
            magnetic[..., index] = _to_complex_tensor(monitor[magnetic_name], device=device, dtype=dtype)
    return electric, magnetic


def _reference_mode_fields(context, mode_source: ModeSource, monitor, *, frequency: float, direction: str | None):
    axis, tangential_axes, coord_a, coord_b = _monitor_coords(monitor)
    public_source = mode_source
    if axis != public_source.normal_axis:
        raise ValueError(
            f"Monitor axis {axis!r} does not match ModeSource normal axis {public_source.normal_axis!r}."
        )

    compiled = _compile_mode_source(public_source, default_frequency=float(frequency))
    if direction is not None:
        direction_sign = 1 if str(direction) == "+" else -1
        compiled["direction"] = str(direction)
        compiled["direction_sign"] = int(direction_sign)
        compiled["direction_vector"] = {
            "x": (float(direction_sign), 0.0, 0.0),
            "y": (0.0, float(direction_sign), 0.0),
            "z": (0.0, 0.0, float(direction_sign)),
        }[axis]
    compiled["source_time"] = _compile_mode_source(
        ModeSource(
            position=public_source.position,
            size=public_source.size,
            mode_index=public_source.mode_index,
            direction=compiled["direction"],
            polarization=f"E{public_source.polarization_axis}",
            source_time=CW(frequency=float(frequency), amplitude=1.0),
            name=public_source.name,
        ),
        default_frequency=float(frequency),
    )["source_time"]

    position = list(compiled["position"])
    position[_AXIS_TO_INDEX[axis]] = float(monitor["position"])
    compiled["position"] = tuple(position)
    mode_data = solve_mode_source_profile(context, compiled)

    device = _resolve_tensor_device(coord_a, coord_b, mode_data["profile"])
    real_dtype = _resolve_real_dtype(coord_a, coord_b, mode_data["profile"])
    complex_dtype = _resolve_complex_dtype(mode_data["profile"])
    coords_a = _as_1d_coords(coord_a, tangential_axes[0], device=device, dtype=real_dtype)
    coords_b = _as_1d_coords(coord_b, tangential_axes[1], device=device, dtype=real_dtype)
    points = build_plane_points(axis, float(monitor["position"]), coords_a, coords_b).to(
        device=device,
        dtype=real_dtype,
    )
    electric = torch.zeros((coords_a.numel(), coords_b.numel(), 3), device=device, dtype=complex_dtype)
    magnetic = torch.zeros((coords_a.numel(), coords_b.numel(), 3), device=device, dtype=complex_dtype)
    for index, field_name in enumerate(("Ex", "Ey", "Ez")):
        electric[..., index] = sample_mode_source_component(mode_data, points, field_name).to(
            device=device,
            dtype=complex_dtype,
        )
    for index, field_name in enumerate(("Hx", "Hy", "Hz")):
        magnetic[..., index] = sample_mode_source_component(mode_data, points, field_name).to(
            device=device,
            dtype=complex_dtype,
        )
    return coords_a, coords_b, electric, magnetic, mode_data


def _integrated_power(electric: torch.Tensor, magnetic: torch.Tensor, normal: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    density = 0.5 * torch.sum(torch.cross(electric, torch.conj(magnetic), dim=-1) * normal, dim=-1)
    return torch.sum(density * weights)


def _compute_mode_overlap_from_payload(
    result,
    monitor_name: str,
    monitor,
    *,
    mode_source: ModeSource | str | None = None,
    direction: str | None = None,
) -> dict[str, object]:
    context = _mode_overlap_context(result)
    resolved_frequency = float(monitor.get("frequency", getattr(result, "frequency", None)))
    resolved_mode_source = _resolve_mode_source(
        result.scene,
        monitor,
        mode_source=mode_source,
        frequency=resolved_frequency,
    )
    axis, tangential_axes, coord_a, coord_b = _monitor_coords(monitor)

    device = _resolve_tensor_device(coord_a, coord_b, *[monitor.get(name) for name in _FIELD_NAMES])
    real_dtype = _resolve_real_dtype(coord_a, coord_b)
    complex_dtype = _resolve_complex_dtype(*[monitor.get(name) for name in _FIELD_NAMES])
    coords_a = _as_1d_coords(coord_a, tangential_axes[0], device=device, dtype=real_dtype)
    coords_b = _as_1d_coords(coord_b, tangential_axes[1], device=device, dtype=real_dtype)
    weights = _trapz_weights_1d(coords_a)[:, None] * _trapz_weights_1d(coords_b)[None, :]
    normal = _surface_normal(
        axis,
        monitor.get("normal_direction", "+"),
        device=device,
        dtype=real_dtype,
    ).to(dtype=complex_dtype)

    electric_total, magnetic_total = _monitor_vector_fields(
        monitor,
        (coords_a.numel(), coords_b.numel()),
        device=device,
        dtype=complex_dtype,
    )
    coords_ref_a, coords_ref_b, electric_ref, magnetic_ref, mode_data = _reference_mode_fields(
        context,
        resolved_mode_source,
        monitor,
        frequency=resolved_frequency,
        direction=direction,
    )
    coords_ref_a = coords_ref_a.to(device=device, dtype=coords_a.dtype)
    coords_ref_b = coords_ref_b.to(device=device, dtype=coords_b.dtype)
    electric_ref = electric_ref.to(device=device, dtype=complex_dtype)
    magnetic_ref = magnetic_ref.to(device=device, dtype=complex_dtype)
    if coords_ref_a.shape != coords_a.shape or coords_ref_b.shape != coords_b.shape:
        raise ValueError("Reference mode coordinates do not match the monitor coordinates.")
    if not torch.allclose(coords_ref_a, coords_a, atol=1e-6, rtol=0.0) or not torch.allclose(
        coords_ref_b,
        coords_b,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("Reference mode coordinates do not match the monitor coordinates.")

    mode_power = _integrated_power(electric_ref, magnetic_ref, normal, weights)
    mode_power_abs = torch.abs(torch.real(mode_power))
    if float(mode_power_abs.item()) <= _POWER_FLOOR:
        raise ValueError("Reference mode power is numerically zero on the supplied monitor plane.")

    plus_density = torch.sum(
        (
            torch.cross(electric_total, torch.conj(magnetic_ref), dim=-1)
            + torch.cross(torch.conj(electric_ref), magnetic_total, dim=-1)
        )
        * normal,
        dim=-1,
    )
    minus_density = torch.sum(
        (
            torch.cross(electric_total, torch.conj(magnetic_ref), dim=-1)
            - torch.cross(torch.conj(electric_ref), magnetic_total, dim=-1)
        )
        * normal,
        dim=-1,
    )
    amplitude_forward = torch.sum(plus_density * weights) / (4.0 * mode_power)
    amplitude_backward = torch.sum(minus_density * weights) / (4.0 * mode_power)

    total_power = _integrated_power(electric_total, magnetic_total, normal, weights)
    forward_power = torch.abs(amplitude_forward) ** 2 * mode_power_abs.to(dtype=amplitude_forward.real.dtype)
    backward_power = torch.abs(amplitude_backward) ** 2 * mode_power_abs.to(dtype=amplitude_backward.real.dtype)
    total_power_abs = torch.abs(torch.real(total_power))
    if float(total_power_abs.item()) > _POWER_FLOOR:
        forward_fraction = forward_power / total_power_abs
        backward_fraction = backward_power / total_power_abs
    else:
        forward_fraction = None
        backward_fraction = None

    return {
        "monitor": monitor_name,
        "frequency": resolved_frequency,
        "axis": axis,
        "position": float(monitor["position"]),
        "direction": direction or resolved_mode_source.direction,
        "mode_index": int(resolved_mode_source.mode_index),
        "polarization_axis": resolved_mode_source.polarization_axis,
        "effective_index": float(mode_data["effective_index"]),
        "beta": float(mode_data["beta"]),
        "mode_power": torch.real(mode_power),
        "total_power": torch.real(total_power),
        "amplitude_forward": amplitude_forward,
        "amplitude_backward": amplitude_backward,
        "power_forward": forward_power,
        "power_backward": backward_power,
        "power_fraction_forward": forward_fraction,
        "power_fraction_backward": backward_fraction,
        "coords": (coords_a, coords_b),
    }


def compute_mode_overlap(
    result,
    monitor_name: str,
    *,
    mode_source: ModeSource | str | None = None,
    direction: str | None = None,
    frequency: float | None = None,
    freq_index: int | None = None,
) -> dict[str, object]:
    monitor = _monitor_payload(result, monitor_name, frequency=frequency, freq_index=freq_index)
    return _compute_mode_overlap_from_payload(
        result,
        monitor_name,
        monitor,
        mode_source=mode_source,
        direction=direction,
    )
