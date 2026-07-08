from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .monitors import ClosedSurfaceMonitor, FinitePlaneMonitor, MediumMonitor, PermittivityMonitor
from .scene import prepare_scene
from .visualization import extract_orthogonal_slice, plot_slice_image

_UNSET = object()


def _clone_mapping(data: dict[str, Any]) -> dict[str, Any]:
    return dict(data) if data is not None else {}


def _cpu_serializable(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_serializable(item) for item in value)
    return value


def _resolve_result_frequencies(*, frequency: float | None, frequencies) -> tuple[float, ...]:
    if frequencies is not None:
        resolved = tuple(float(freq) for freq in frequencies)
    elif frequency is not None:
        resolved = (float(frequency),)
    else:
        raise ValueError("Result requires frequency or frequencies.")
    if not resolved:
        raise ValueError("Result frequencies must not be empty.")
    return resolved


def _resolve_frequency_index(
    frequencies: tuple[float, ...],
    *,
    frequency: float | None = None,
    freq_index: int | None = None,
) -> int | None:
    if frequency is not None and freq_index is not None:
        raise ValueError("Pass either frequency or freq_index, not both.")
    if not frequencies:
        return None
    if freq_index is not None:
        index = int(freq_index)
        if index < 0 or index >= len(frequencies):
            raise IndexError(f"freq_index={index} is out of range for {len(frequencies)} frequencies.")
        return index
    if frequency is None:
        return None

    target = float(frequency)
    for index, candidate in enumerate(frequencies):
        if np.isclose(candidate, target, rtol=1e-9, atol=1e-12):
            return index
    raise KeyError(f"Frequency {target!r} is not available. Choices: {frequencies}.")


def _slice_frequency_axis(value: Any, freq_count: int, index: int):
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == freq_count:
        return value[index]
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == freq_count:
        return value[index]
    return value


def _slice_monitor_samples(samples: Any, freq_count: int, index: int):
    if isinstance(samples, torch.Tensor) and samples.ndim > 0 and samples.shape[0] == freq_count:
        return int(samples[index].item())
    if isinstance(samples, np.ndarray) and samples.ndim > 0 and samples.shape[0] == freq_count:
        return int(samples[index])
    if isinstance(samples, (list, tuple)) and len(samples) == freq_count:
        return int(samples[index])
    return samples


def _monitor_frequencies(payload: dict[str, Any]) -> tuple[float, ...]:
    if "frequencies" in payload:
        return tuple(float(freq) for freq in payload["frequencies"])
    if "frequency" in payload:
        return (float(payload["frequency"]),)
    return ()


def _select_monitor_frequency(payload: dict[str, Any], frequencies: tuple[float, ...], index: int) -> dict[str, Any]:
    selected = dict(payload)
    selected_frequency = frequencies[index]
    selected["frequencies"] = (selected_frequency,)
    selected["frequency"] = selected_frequency

    if "samples" in payload:
        selected["samples"] = _slice_monitor_samples(payload["samples"], len(frequencies), index)
    if "data" in payload:
        selected["data"] = _slice_frequency_axis(payload["data"], len(frequencies), index)
    if "flux" in payload:
        selected["flux"] = _slice_frequency_axis(payload["flux"], len(frequencies), index)
    if "power" in payload:
        selected["power"] = _slice_frequency_axis(payload["power"], len(frequencies), index)

    if _monitor_payload_is_point(payload):
        components = {}
        for component_name, component_value in payload.get("components", {}).items():
            sliced = _slice_frequency_axis(component_value, len(frequencies), index)
            components[component_name] = sliced
            if component_name in payload:
                selected[component_name] = _slice_frequency_axis(payload[component_name], len(frequencies), index)
        selected["components"] = components
        return selected

    components = {}
    for component_name, component_payload in payload.get("components", {}).items():
        updated_payload = dict(component_payload)
        updated_payload["data"] = _slice_frequency_axis(component_payload["data"], len(frequencies), index)
        components[component_name] = updated_payload
        if component_name in payload:
            selected[component_name] = _slice_frequency_axis(payload[component_name], len(frequencies), index)
    selected["components"] = components
    return selected


_FIELD_NORMAL_AXIS = {
    "EX": 0,
    "EY": 1,
    "EZ": 2,
    "HX": 0,
    "HY": 1,
    "HZ": 2,
}


def _mirror_sign(component: str | None, symmetry: str, axis: int) -> int:
    if component is None:
        return 1
    normal_axis = _FIELD_NORMAL_AXIS.get(component.upper())
    if normal_axis is None:
        return 1
    is_normal = normal_axis == axis
    if symmetry == "PEC":
        return 1 if is_normal else -1
    if symmetry == "PMC":
        return -1 if is_normal else 1
    return 1


def _expand_tensor_with_symmetry(tensor: torch.Tensor, scene, component: str | None = None) -> torch.Tensor:
    expanded = tensor
    half_sizes = (int(scene.Nx), int(scene.Ny), int(scene.Nz))
    spatial_axis_offset = expanded.ndim - 3
    if spatial_axis_offset < 0:
        raise ValueError(f"Expected tensor with at least 3 dimensions, got shape {tuple(expanded.shape)}.")
    for axis, symmetry in enumerate(getattr(scene, "symmetry", (None, None, None))):
        if symmetry is None:
            continue
        tensor_axis = spatial_axis_offset + axis
        half_size = half_sizes[axis]
        dim_size = int(expanded.shape[tensor_axis])
        if dim_size == half_size:
            mirrored_source = expanded.narrow(tensor_axis, 1, max(dim_size - 1, 0))
        elif dim_size == half_size - 1:
            mirrored_source = expanded
        else:
            raise ValueError(
                f"Cannot expand symmetry axis {axis} for tensor shape {tuple(expanded.shape)} and half-domain size {half_size}."
            )
        mirrored = torch.flip(mirrored_source, dims=(tensor_axis,))
        sign = _mirror_sign(component, symmetry, axis)
        if sign < 0:
            mirrored = -mirrored
        expanded = torch.cat((mirrored, expanded), dim=tensor_axis)
    return expanded


def _monitor_payload_is_point(payload: dict[str, Any]) -> bool:
    if "field_indices" in payload:
        return True
    if "axis" in payload:
        return False
    kind = payload.get("kind")
    if kind is not None:
        return kind == "point"
    return False


def _monitor_payload_is_mode(payload: dict[str, Any]) -> bool:
    if payload.get("mode_spec") is not None:
        return True
    monitor_type = payload.get("monitor_type")
    if monitor_type is not None:
        return monitor_type == "mode"
    return False


def _monitor_payload_is_diffraction(payload: dict[str, Any]) -> bool:
    return payload.get("monitor_type") == "diffraction"


def _monitor_payload_is_closed_surface(payload: dict[str, Any]) -> bool:
    return payload.get("kind") == "closed_surface"


def _find_scene_monitor(scene, name: str):
    for monitor in getattr(scene, "monitors", ()):
        if getattr(monitor, "name", None) == name:
            return monitor
    return None


def _find_resolved_scene_monitor(scene, name: str):
    if not hasattr(scene, "resolved_monitors"):
        return None
    for monitor in scene.resolved_monitors():
        if getattr(monitor, "name", None) == name:
            return monitor
    return None


def _plane_coord_names(axis: str) -> tuple[str, str]:
    axis_name = str(axis).lower()
    if axis_name == "x":
        return "y", "z"
    if axis_name == "y":
        return "x", "z"
    return "x", "y"


def _select_coord_indices(coords, lower: float, upper: float):
    if isinstance(coords, torch.Tensor):
        coord_tensor = coords.to(dtype=coords.real.dtype)
        tolerance = 1e-12 * max(
            abs(lower),
            abs(upper),
            float(torch.max(torch.abs(coord_tensor)).item()),
            1.0,
        )
        mask = (coord_tensor >= lower - tolerance) & (coord_tensor <= upper + tolerance)
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if indices.numel() < 2:
            raise ValueError(f"Finite monitor bounds [{lower}, {upper}] select fewer than two samples.")
        if indices.numel() == coord_tensor.numel():
            return None
        return indices.to(device=coords.device)

    coord_array = np.asarray(coords, dtype=float)
    tolerance = 1e-12 * max(abs(lower), abs(upper), float(np.max(np.abs(coord_array))), 1.0)
    indices = np.nonzero((coord_array >= lower - tolerance) & (coord_array <= upper + tolerance))[0]
    if indices.size < 2:
        raise ValueError(f"Finite monitor bounds [{lower}, {upper}] select fewer than two samples.")
    if indices.size == coord_array.size:
        return None
    return indices


def _slice_plane_samples(values, u_indices, v_indices):
    if isinstance(values, torch.Tensor):
        sliced = values
        if u_indices is not None:
            if not isinstance(u_indices, torch.Tensor):
                u_indices = torch.as_tensor(u_indices, device=values.device, dtype=torch.long)
            sliced = sliced.index_select(sliced.ndim - 2, u_indices)
        if v_indices is not None:
            if not isinstance(v_indices, torch.Tensor):
                v_indices = torch.as_tensor(v_indices, device=values.device, dtype=torch.long)
            sliced = sliced.index_select(sliced.ndim - 1, v_indices)
        return sliced

    array = np.asarray(values)
    if u_indices is not None:
        array = np.take(array, u_indices, axis=array.ndim - 2)
    if v_indices is not None:
        array = np.take(array, v_indices, axis=array.ndim - 1)
    return array


def _crop_plane_monitor_payload(payload: dict[str, Any], monitor: FinitePlaneMonitor) -> dict[str, Any]:
    coord_u_name, coord_v_name = _plane_coord_names(monitor.axis)
    if coord_u_name not in payload or coord_v_name not in payload:
        return dict(payload)

    u_indices = _select_coord_indices(payload[coord_u_name], *monitor.tangential_bounds[coord_u_name])
    v_indices = _select_coord_indices(payload[coord_v_name], *monitor.tangential_bounds[coord_v_name])
    if u_indices is None and v_indices is None:
        selected = dict(payload)
        selected["monitor_type"] = "finite_plane"
        selected["center"] = monitor.position
        selected["size"] = monitor.size
        selected["tangential_bounds"] = dict(monitor.tangential_bounds)
        if monitor.face_label is not None:
            selected["face_label"] = monitor.face_label
        if monitor.surface_name is not None:
            selected["surface_name"] = monitor.surface_name
        return selected

    coord_u = payload[coord_u_name]
    coord_v = payload[coord_v_name]
    selected_u = coord_u if u_indices is None else (
        coord_u.index_select(0, u_indices) if isinstance(coord_u, torch.Tensor) else np.asarray(coord_u)[u_indices]
    )
    selected_v = coord_v if v_indices is None else (
        coord_v.index_select(0, v_indices) if isinstance(coord_v, torch.Tensor) else np.asarray(coord_v)[v_indices]
    )

    selected = dict(payload)
    selected[coord_u_name] = selected_u
    selected[coord_v_name] = selected_v
    selected["coords"] = (selected_u, selected_v)
    selected["monitor_type"] = "finite_plane"
    selected["center"] = monitor.position
    selected["size"] = monitor.size
    selected["tangential_bounds"] = dict(monitor.tangential_bounds)
    if monitor.face_label is not None:
        selected["face_label"] = monitor.face_label
    if monitor.surface_name is not None:
        selected["surface_name"] = monitor.surface_name

    if "data" in payload:
        selected["data"] = _slice_plane_samples(payload["data"], u_indices, v_indices)

    components = {}
    for component_name, component_payload in payload.get("components", {}).items():
        updated_component = dict(component_payload)
        updated_component["data"] = _slice_plane_samples(component_payload["data"], u_indices, v_indices)
        updated_component["coords"] = (selected_u, selected_v)
        components[component_name] = updated_component
        if component_name in payload:
            selected[component_name] = _slice_plane_samples(payload[component_name], u_indices, v_indices)
    selected["components"] = components

    if selected.get("compute_flux"):
        from .fdtd.observers import _compute_plane_flux

        flux = _compute_plane_flux(selected)
        selected["flux"] = flux
        selected["power"] = flux
    return selected


def _material_monitor_axis_indices(coord: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    lower, upper = float(bounds[0]), float(bounds[1])
    center = 0.5 * (lower + upper)
    if abs(upper - lower) <= 1e-12:
        nearest = int(torch.argmin(torch.abs(coord - center)).item())
        return torch.tensor([nearest], device=coord.device, dtype=torch.long)
    scale = max(abs(lower), abs(upper), float(torch.max(torch.abs(coord)).item()), 1.0)
    tolerance = 1e-9 * scale
    mask = (coord >= lower - tolerance) & (coord <= upper + tolerance)
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        nearest = int(torch.argmin(torch.abs(coord - center)).item())
        return torch.tensor([nearest], device=coord.device, dtype=torch.long)
    return indices.to(device=coord.device, dtype=torch.long)


def _crop_material_grid(tensor: torch.Tensor, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(0, ix).index_select(1, iy).index_select(2, iz)


def _stack_material_frequencies(items: list[torch.Tensor]) -> torch.Tensor:
    if len(items) == 1:
        return items[0]
    return torch.stack(items, dim=0)


def _resolve_material_monitor_frequencies(
    result: "Result",
    monitor,
    *,
    frequency,
    freq_index,
) -> tuple[float, ...]:
    if frequency is not None and freq_index is not None:
        raise ValueError("Pass either frequency or freq_index, not both.")
    if freq_index is not None:
        index = _resolve_frequency_index(result.frequencies, freq_index=freq_index)
        return (result.frequencies[index],)
    if frequency is not None:
        return (float(frequency),)
    if monitor.frequencies is not None:
        return tuple(float(freq) for freq in monitor.frequencies)
    return tuple(result.frequencies)


def _build_material_monitor_payload(result: "Result", monitor, *, frequency, freq_index):
    prepared = result.prepared_scene
    ix = _material_monitor_axis_indices(prepared.x, monitor.bounds[0])
    iy = _material_monitor_axis_indices(prepared.y, monitor.bounds[1])
    iz = _material_monitor_axis_indices(prepared.z, monitor.bounds[2])

    eval_frequencies = _resolve_material_monitor_frequencies(
        result,
        monitor,
        frequency=frequency,
        freq_index=freq_index,
    )
    is_medium = isinstance(monitor, MediumMonitor)

    eps_x_list, eps_y_list, eps_z_list = [], [], []
    mu_x_list, mu_y_list, mu_z_list = [], [], []
    for freq in eval_frequencies:
        eps_components, mu_components = prepared.compile_material_components(frequency=freq)
        eps_x_list.append(_crop_material_grid(eps_components["x"], ix, iy, iz))
        eps_y_list.append(_crop_material_grid(eps_components["y"], ix, iy, iz))
        eps_z_list.append(_crop_material_grid(eps_components["z"], ix, iy, iz))
        if is_medium:
            mu_x_list.append(_crop_material_grid(mu_components["x"], ix, iy, iz))
            mu_y_list.append(_crop_material_grid(mu_components["y"], ix, iy, iz))
            mu_z_list.append(_crop_material_grid(mu_components["z"], ix, iy, iz))

    eps_x = _stack_material_frequencies(eps_x_list)
    eps_y = _stack_material_frequencies(eps_y_list)
    eps_z = _stack_material_frequencies(eps_z_list)

    payload: dict[str, Any] = {
        "kind": monitor.kind,
        "monitor_type": monitor.kind,
        "name": monitor.name,
        "bounds": monitor.bounds,
        "x": prepared.x.index_select(0, ix),
        "y": prepared.y.index_select(0, iy),
        "z": prepared.z.index_select(0, iz),
        "eps_x": eps_x,
        "eps_y": eps_y,
        "eps_z": eps_z,
        "eps": (eps_x + eps_y + eps_z) / 3.0,
    }

    if is_medium:
        mu_x = _stack_material_frequencies(mu_x_list)
        mu_y = _stack_material_frequencies(mu_y_list)
        mu_z = _stack_material_frequencies(mu_z_list)
        sigma_components = prepared.compile_materials()["sigma_e_components"]
        sigma_x = _crop_material_grid(sigma_components["x"], ix, iy, iz)
        sigma_y = _crop_material_grid(sigma_components["y"], ix, iy, iz)
        sigma_z = _crop_material_grid(sigma_components["z"], ix, iy, iz)
        # sigma_e is real and frequency-independent; broadcast it across evaluated frequencies.
        sigma_x = _stack_material_frequencies([sigma_x] * len(eval_frequencies))
        sigma_y = _stack_material_frequencies([sigma_y] * len(eval_frequencies))
        sigma_z = _stack_material_frequencies([sigma_z] * len(eval_frequencies))
        payload["mu_x"] = mu_x
        payload["mu_y"] = mu_y
        payload["mu_z"] = mu_z
        payload["mu"] = (mu_x + mu_y + mu_z) / 3.0
        payload["sigma_e_x"] = sigma_x
        payload["sigma_e_y"] = sigma_y
        payload["sigma_e_z"] = sigma_z
        payload["sigma_e"] = (sigma_x + sigma_y + sigma_z) / 3.0

    if len(eval_frequencies) == 1:
        payload["frequency"] = eval_frequencies[0]
    payload["frequencies"] = eval_frequencies
    return payload


def _build_closed_surface_payload(result: "Result", monitor: ClosedSurfaceMonitor, *, frequency, freq_index):
    faces = {}
    for face in monitor.faces:
        face_payload = result.raw_monitor(face.name, frequency=frequency, freq_index=freq_index)
        faces[face.face_label or face.name] = face_payload

    first_face = next(iter(faces.values()))
    frequencies = tuple(first_face.get("frequencies", (first_face.get("frequency"),)))
    payload = {
        "kind": "closed_surface",
        "monitor_type": "closed_surface",
        "name": monitor.name,
        "faces": faces,
        "face_monitor_names": monitor.face_monitor_names,
        "frequency": first_face.get("frequency"),
        "frequencies": frequencies,
        "bounds": monitor.bounds,
    }
    return payload


@dataclass(frozen=True)
class ResultFieldAccessor:
    selection: "ResultSelection"
    family: str

    @property
    def x(self) -> torch.Tensor:
        return self.selection.tensor(f"{self.family}x")

    @property
    def y(self) -> torch.Tensor:
        return self.selection.tensor(f"{self.family}y")

    @property
    def z(self) -> torch.Tensor:
        return self.selection.tensor(f"{self.family}z")

    def as_dict(self) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for axis in ("x", "y", "z"):
            field_name = f"{self.family}{axis}".upper()
            if field_name not in self.selection.result._fields:
                continue
            tensors[axis] = self.selection.tensor(field_name)
        return tensors


@dataclass(frozen=True)
class ResultMaterialTensorAccessor:
    selection: "ResultSelection"
    family: str

    @property
    def scalar(self) -> torch.Tensor:
        suffix = "eps_r" if self.family == "eps" else "mu_r"
        return self.selection.material(suffix)

    @property
    def r(self) -> torch.Tensor:
        return self.scalar

    @property
    def x(self) -> torch.Tensor:
        return self.selection.material(f"{self.family}_x")

    @property
    def y(self) -> torch.Tensor:
        return self.selection.material(f"{self.family}_y")

    @property
    def z(self) -> torch.Tensor:
        return self.selection.material(f"{self.family}_z")

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "scalar": self.scalar,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(frozen=True)
class ResultMaterialsAccessor:
    selection: "ResultSelection"

    @property
    def eps(self) -> ResultMaterialTensorAccessor:
        return ResultMaterialTensorAccessor(self.selection, "eps")

    @property
    def mu(self) -> ResultMaterialTensorAccessor:
        return ResultMaterialTensorAccessor(self.selection, "mu")

    @property
    def permittivity(self) -> ResultMaterialTensorAccessor:
        return self.eps

    @property
    def permeability(self) -> ResultMaterialTensorAccessor:
        return self.mu


@dataclass(frozen=True)
class ResultSelection:
    result: "Result"
    frequency: float | None = None
    freq_index: int | None = None
    expand_symmetry: bool = False
    resolve_modal: bool = True

    def __post_init__(self):
        if self.frequency is not None and self.freq_index is not None:
            raise ValueError("Pass either frequency or freq_index, not both.")

    @property
    def E(self) -> ResultFieldAccessor:
        return ResultFieldAccessor(self, "E")

    @property
    def H(self) -> ResultFieldAccessor:
        return ResultFieldAccessor(self, "H")

    @property
    def materials(self) -> ResultMaterialsAccessor:
        return ResultMaterialsAccessor(self)

    def field(self, name: str = "E"):
        return self.result.field(
            name,
            expand_symmetry=self.expand_symmetry,
            frequency=self.frequency,
            freq_index=self.freq_index,
        )

    def tensor(self, name: str) -> torch.Tensor:
        return self.result.tensor(
            name,
            expand_symmetry=self.expand_symmetry,
            frequency=self.frequency,
            freq_index=self.freq_index,
        )

    def material(self, name: str = "eps_r") -> torch.Tensor:
        return self.result.material(
            name,
            expand_symmetry=self.expand_symmetry,
        )

    def raw_monitor(self, name: str):
        return self.result.raw_monitor(
            name,
            frequency=self.frequency,
            freq_index=self.freq_index,
        )

    def monitor(self, name: str, *, resolve_modal: bool | None = None):
        should_resolve_modal = self.resolve_modal if resolve_modal is None else bool(resolve_modal)
        return self.result.monitor(
            name,
            frequency=self.frequency,
            freq_index=self.freq_index,
            resolve_modal=should_resolve_modal,
        )

    def at(
        self,
        *,
        frequency: float | None | object = _UNSET,
        freq_index: int | None | object = _UNSET,
        expand_symmetry: bool | object = _UNSET,
        resolve_modal: bool | object = _UNSET,
    ) -> "ResultSelection":
        selected_frequency = self.frequency if frequency is _UNSET else frequency
        selected_freq_index = self.freq_index if freq_index is _UNSET else freq_index
        selected_expand_symmetry = self.expand_symmetry if expand_symmetry is _UNSET else bool(expand_symmetry)
        selected_resolve_modal = self.resolve_modal if resolve_modal is _UNSET else bool(resolve_modal)
        return ResultSelection(
            self.result,
            frequency=selected_frequency,
            freq_index=selected_freq_index,
            expand_symmetry=selected_expand_symmetry,
            resolve_modal=selected_resolve_modal,
        )

    select = at


@dataclass
class ResultPlotter:
    result: "Result"

    def field(self, axis: str = "z", position: float = 0.0, component: str = "abs", **kwargs):
        solver = self.result.solver
        if solver is None or not hasattr(solver, "plot_cross_section"):
            raise RuntimeError("Plotting is only available for results created from a solver run.")
        return solver.plot_cross_section(
            axis=axis,
            position=position,
            component=component,
            **kwargs,
        )

    def material(
        self,
        name: str = "eps_r",
        axis: str = "z",
        position: float = 0.0,
        figsize: tuple[int, int] = (8, 6),
        cmap: str = "viridis",
    ):
        material = self.result.material(name)
        scene = self.result.prepared_scene
        slice_info = extract_orthogonal_slice(
            material.detach().cpu().numpy(),
            axis,
            position,
            scene.x.detach().cpu().numpy(),
            scene.y.detach().cpu().numpy(),
            scene.z.detach().cpu().numpy(),
        )
        return plot_slice_image(
            slice_info["slice"],
            extent=slice_info["extent"],
            xlabel=slice_info["xlabel"],
            ylabel=slice_info["ylabel"],
            title=f"{name} at {axis}={position:.3f}m",
            colorbar_label=name,
            figsize=figsize,
            cmap=cmap,
        )


class Result:
    def __init__(
        self,
        *,
        method: str,
        scene,
        prepared_scene=None,
        frequency: float | None = None,
        frequencies=None,
        solver=None,
        fields: dict[str, torch.Tensor] | None = None,
        monitors: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        solver_stats: dict[str, Any] | None = None,
        raw_output: Any = None,
    ):
        self.method = method
        self.scene = scene
        self._prepared_scene = prepared_scene
        self.frequencies = _resolve_result_frequencies(frequency=frequency, frequencies=frequencies)
        self.frequency = self.frequencies[0]
        self.solver = solver
        self._fields = _clone_mapping(fields)
        self._monitors = _clone_mapping(monitors)
        self._metadata = _clone_mapping(metadata)
        self._solver_stats = _clone_mapping(solver_stats)
        self.raw_output = raw_output
        self.plot = ResultPlotter(self)

    @property
    def prepared_scene(self):
        if self._prepared_scene is None:
            self._prepared_scene = prepare_scene(self.scene)
        return self._prepared_scene

    def at(
        self,
        *,
        frequency: float | None = None,
        freq_index: int | None = None,
        expand_symmetry: bool = False,
        resolve_modal: bool = True,
    ) -> ResultSelection:
        return ResultSelection(
            self,
            frequency=frequency,
            freq_index=freq_index,
            expand_symmetry=expand_symmetry,
            resolve_modal=resolve_modal,
        )

    select = at

    @property
    def fields(self) -> dict[str, torch.Tensor]:
        return dict(self._fields)

    @property
    def monitors(self) -> dict[str, Any]:
        return dict(self._monitors)

    @property
    def E(self) -> ResultFieldAccessor:
        return self.at().E

    @property
    def H(self) -> ResultFieldAccessor:
        return self.at().H

    @property
    def materials(self) -> ResultMaterialsAccessor:
        return self.at().materials

    def field(
        self,
        name: str = "E",
        *,
        expand_symmetry: bool = False,
        frequency: float | None = None,
        freq_index: int | None = None,
    ):
        key = name.upper()
        if key in {"E", "H"}:
            component_names = tuple(field_name for field_name in (f"{key}X", f"{key}Y", f"{key}Z") if field_name in self._fields)
            if not expand_symmetry:
                return {
                    field_name: self.tensor(field_name, frequency=frequency, freq_index=freq_index)
                    for field_name in component_names
                }
            return {
                field_name: self.tensor(
                    field_name,
                    expand_symmetry=True,
                    frequency=frequency,
                    freq_index=freq_index,
                )
                for field_name in component_names
            }
        return self.tensor(key, expand_symmetry=expand_symmetry, frequency=frequency, freq_index=freq_index)

    def tensor(
        self,
        name: str,
        *,
        expand_symmetry: bool = False,
        frequency: float | None = None,
        freq_index: int | None = None,
    ) -> torch.Tensor:
        key = name.upper()
        if key not in self._fields:
            raise KeyError(f"Field {name!r} is not available in this result.")
        tensor = self._fields[key]
        selected_index = _resolve_frequency_index(self.frequencies, frequency=frequency, freq_index=freq_index)
        if selected_index is not None and len(self.frequencies) > 1 and tensor.ndim > 0 and tensor.shape[0] == len(self.frequencies):
            tensor = tensor[selected_index]
        if not expand_symmetry:
            return tensor
        return _expand_tensor_with_symmetry(tensor, self.prepared_scene, component=key)

    def raw_monitor(self, name: str, *, frequency: float | None = None, freq_index: int | None = None):
        public_monitor = _find_scene_monitor(self.scene, name)
        if isinstance(public_monitor, ClosedSurfaceMonitor):
            return _build_closed_surface_payload(self, public_monitor, frequency=frequency, freq_index=freq_index)
        if isinstance(public_monitor, (PermittivityMonitor, MediumMonitor)):
            return _build_material_monitor_payload(self, public_monitor, frequency=frequency, freq_index=freq_index)

        if name not in self._monitors:
            raise KeyError(f"Monitor {name!r} is not available in this result.")

        payload = self._monitors[name]
        monitor_frequencies = _monitor_frequencies(payload)
        selected_index = _resolve_frequency_index(
            monitor_frequencies,
            frequency=frequency,
            freq_index=freq_index,
        )
        if selected_index is None or len(monitor_frequencies) <= 1:
            selected = dict(payload)
        else:
            selected = _select_monitor_frequency(payload, monitor_frequencies, selected_index)

        resolved_monitor = _find_resolved_scene_monitor(self.scene, name)
        if isinstance(resolved_monitor, FinitePlaneMonitor):
            return _crop_plane_monitor_payload(selected, resolved_monitor)
        return selected

    def monitor(
        self,
        name: str,
        *,
        frequency: float | None = None,
        freq_index: int | None = None,
        resolve_modal: bool = True,
    ):
        payload = self.raw_monitor(name, frequency=frequency, freq_index=freq_index)
        if not resolve_modal or _monitor_payload_is_closed_surface(payload):
            return payload

        if _monitor_payload_is_diffraction(payload):
            from .postprocess.diffraction import compute_diffraction_from_payload

            return compute_diffraction_from_payload(self, name, payload)

        if not _monitor_payload_is_mode(payload):
            return payload

        from .postprocess.modal import _compute_mode_overlap_from_payload

        modal = _compute_mode_overlap_from_payload(
            self,
            name,
            payload,
            mode_source=None,
            direction=None,
        )
        modal["kind"] = "mode"
        modal["fields"] = tuple(payload.get("fields", ()))
        modal["frequencies"] = tuple(payload.get("frequencies", (payload.get("frequency"),)))
        modal["frequency"] = payload.get("frequency")
        modal["normal_direction"] = payload.get("normal_direction", "+")
        modal["plane"] = payload
        return modal

    def material(self, name: str = "eps_r", *, expand_symmetry: bool = False) -> torch.Tensor:
        prepared_scene = self.prepared_scene
        key = name.lower()
        if key in {"eps", "eps_r", "permittivity"}:
            tensor = prepared_scene.permittivity
        elif key in {"eps_x", "epsilon_x", "permittivity_x"}:
            tensor = prepared_scene.permittivity_components["x"]
        elif key in {"eps_y", "epsilon_y", "permittivity_y"}:
            tensor = prepared_scene.permittivity_components["y"]
        elif key in {"eps_z", "epsilon_z", "permittivity_z"}:
            tensor = prepared_scene.permittivity_components["z"]
        elif key in {"mu", "mu_r", "permeability"}:
            tensor = prepared_scene.permeability
        elif key in {"mu_x", "permeability_x"}:
            tensor = prepared_scene.permeability_components["x"]
        elif key in {"mu_y", "permeability_y"}:
            tensor = prepared_scene.permeability_components["y"]
        elif key in {"mu_z", "permeability_z"}:
            tensor = prepared_scene.permeability_components["z"]
        else:
            raise KeyError(f"Material {name!r} is not supported.")
        if not expand_symmetry:
            return tensor
        return _expand_tensor_with_symmetry(tensor, prepared_scene, component=None)

    def stats(self) -> dict[str, Any]:
        stats = dict(self._solver_stats)
        stats["method"] = self.method
        stats["frequency"] = self.frequency
        stats["frequencies"] = self.frequencies
        stats["num_frequencies"] = len(self.frequencies)
        stats["num_fields"] = len(self._fields)
        stats["num_monitors"] = len(self._monitors)
        return stats

    def save(self, path: str | Path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": self.method,
                "frequency": self.frequency,
                "frequencies": self.frequencies,
                "fields": {name: tensor.detach().cpu() for name, tensor in self._fields.items()},
                "monitors": _cpu_serializable(self._monitors),
                "metadata": self._metadata,
                "solver_stats": self._solver_stats,
            },
            output_path,
        )
