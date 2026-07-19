from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .circuits import CircuitData
from .monitors import (
    ClosedSurfaceMonitor,
    FinitePlaneMonitor,
    MediumMonitor,
    PermittivityMonitor,
    PowerLossMonitor,
)
from .network import (
    EmbeddedNetworkData,
    NetworkData,
    PortData,
    _validate_safe_persistence,
)
from .rational import FitReport, NetworkFitReport
from .scene import prepare_scene
from .thin_wire import WireData
from .visualization import extract_orthogonal_slice, plot_slice_image

_UNSET = object()
RESULT_SNAPSHOT_SCHEMA_VERSION = 2
_EMBEDDED_NETWORK_SNAPSHOT_SCHEMA_VERSION = 1
_FIT_REPORT_SNAPSHOT_SCHEMA_VERSION = 1


def _clone_mapping(data: dict[str, Any]) -> dict[str, Any]:
    return dict(data) if data is not None else {}


def _cpu_serializable(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_serializable(item) for item in value)
    return value


def _normalize_port_data_mapping(ports) -> dict[str, PortData]:
    if ports is None:
        return {}
    if not isinstance(ports, Mapping):
        raise TypeError("ports must be a mapping from port names to PortData.")

    normalized = {}
    for name, data in ports.items():
        port_name = str(name)
        if not isinstance(data, PortData):
            raise TypeError(f"Result port {port_name!r} must be a PortData instance.")
        if port_name != data.port_name:
            raise ValueError(
                f"Result port key {port_name!r} does not match "
                f"PortData.port_name {data.port_name!r}."
            )
        normalized[port_name] = data
    return normalized


def _normalize_circuit_data_mapping(circuits) -> dict[str, CircuitData]:
    if circuits is None:
        return {}
    if not isinstance(circuits, Mapping):
        raise TypeError("circuits must be a mapping from circuit names to CircuitData.")
    normalized = {}
    for name, data in circuits.items():
        circuit_name = str(name)
        if circuit_name in normalized:
            raise ValueError(f"Duplicate normalized circuit key {circuit_name!r}.")
        if not isinstance(data, CircuitData):
            raise TypeError(
                f"Result circuit {circuit_name!r} must be a CircuitData instance."
            )
        if circuit_name != data.circuit_name:
            raise ValueError(
                f"Result circuit key {circuit_name!r} does not match "
                f"CircuitData.circuit_name {data.circuit_name!r}."
            )
        normalized[circuit_name] = data
    return normalized


def _normalize_embedded_network_mapping(networks) -> dict[str, EmbeddedNetworkData]:
    if networks is None:
        return {}
    if not isinstance(networks, Mapping):
        raise TypeError(
            "embedded_networks must map network names to EmbeddedNetworkData."
        )
    normalized = {}
    for name, data in networks.items():
        network_name = str(name)
        if not isinstance(data, EmbeddedNetworkData):
            raise TypeError(
                f"Embedded network {network_name!r} must be EmbeddedNetworkData."
            )
        if network_name != data.name:
            raise ValueError(
                f"Embedded network key {network_name!r} does not match data name "
                f"{data.name!r}."
            )
        normalized[network_name] = data
    return normalized


def _normalize_monitor_data_mapping(monitors) -> dict[str, Any]:
    normalized = _clone_mapping(monitors)
    for name, data in normalized.items():
        if isinstance(data, WireData) and name != data.monitor_name:
            raise ValueError(
                f"Result monitor key {name!r} does not match "
                f"WireData.monitor_name {data.monitor_name!r}."
            )
    return normalized


def _wire_data_snapshot(data: WireData) -> dict[str, Any]:
    _validate_safe_persistence(
        data.metadata,
        path=f"monitors[{data.monitor_name!r}].metadata",
    )
    return {
        "schema_version": data.schema_version,
        "data_type": "WireData",
        "monitor_name": data.monitor_name,
        "wire_name": data.wire_name,
        "frequencies": _cpu_serializable(data.frequencies),
        "current": _cpu_serializable(data.current),
        "charge": _cpu_serializable(data.charge),
        "ohmic_loss": _cpu_serializable(data.ohmic_loss),
        "metadata": _cpu_serializable(data.metadata),
    }


def _wire_data_from_snapshot(payload: Mapping[str, Any]) -> WireData:
    if not isinstance(payload, Mapping):
        raise ValueError("WireData snapshot must contain a mapping payload.")
    if payload.get("data_type") != "WireData":
        raise ValueError("WireData snapshot has an invalid data_type.")
    if payload.get("schema_version") != WireData.schema_version:
        raise ValueError(
            f"Unsupported WireData schema_version {payload.get('schema_version')!r}."
        )
    return WireData(
        monitor_name=payload["monitor_name"],
        wire_name=payload["wire_name"],
        frequencies=payload["frequencies"],
        current=payload["current"],
        charge=payload["charge"],
        ohmic_loss=payload["ohmic_loss"],
        metadata=payload.get("metadata", {}),
    )


def _monitor_mapping_snapshot(monitors: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = {}
    for name, data in monitors.items():
        if isinstance(data, WireData):
            snapshot[name] = {
                "__witwin_result_monitor_type__": "WireData",
                "payload": _wire_data_snapshot(data),
            }
        else:
            snapshot[name] = _cpu_serializable(data)
    return snapshot


def _monitor_mapping_from_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Result monitor snapshot must contain a mapping payload.")
    monitors = {}
    for name, data in payload.items():
        if (
            isinstance(data, Mapping)
            and data.get("__witwin_result_monitor_type__") == "WireData"
        ):
            monitors[name] = _wire_data_from_snapshot(data.get("payload"))
        else:
            monitors[name] = data
    return monitors


def _port_data_snapshot(data: PortData) -> dict[str, Any]:
    _validate_safe_persistence(data.metadata, path=f"ports[{data.port_name!r}].metadata")
    return {
        "schema_version": data.schema_version,
        "data_type": "PortData",
        "port_name": data.port_name,
        "frequencies": _cpu_serializable(data.frequencies),
        "voltage": _cpu_serializable(data.voltage),
        "current": _cpu_serializable(data.current),
        "z0": _cpu_serializable(data.z0),
        "direction": data.direction,
        "reference_plane": data.reference_plane,
        "available_power": _cpu_serializable(data.available_power),
        "mode_names": data.mode_names,
        "beta": _cpu_serializable(data.beta),
        "characteristic_impedance": _cpu_serializable(data.characteristic_impedance),
        "tracking_confidence": _cpu_serializable(data.tracking_confidence),
        "metadata": _cpu_serializable(data.metadata),
        "phasor_convention": data.phasor_convention,
        "power_wave_convention": data.power_wave_convention,
    }


def _network_data_snapshot(data: NetworkData | None):
    if data is None:
        return None
    _validate_safe_persistence(data.metadata, path="network.metadata")
    return {
        "schema_version": data.schema_version,
        "data_type": "NetworkData",
        "frequencies": _cpu_serializable(data.frequencies),
        "s": _cpu_serializable(data.s),
        "z0": _cpu_serializable(data.z0),
        "port_names": data.port_names,
        "valid_columns": _cpu_serializable(data.valid_columns),
        "metadata": _cpu_serializable(data.metadata),
        "phasor_convention": data.phasor_convention,
        "power_wave_convention": data.power_wave_convention,
        "matrix_order": "[frequency, output_port, input_port]",
    }


def _circuit_data_snapshot(data: CircuitData) -> dict[str, Any]:
    return data._snapshot()


def _fit_report_snapshot(report: FitReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    if type(report) not in (FitReport, NetworkFitReport):
        raise TypeError(
            "Embedded network fit_report must be a FitReport or NetworkFitReport."
        )
    return {
        "schema_version": _FIT_REPORT_SNAPSHOT_SCHEMA_VERSION,
        "data_type": type(report).__name__,
        "values": {
            field.name: _cpu_serializable(getattr(report, field.name))
            for field in fields(report)
        },
    }


def _embedded_network_snapshot(data: EmbeddedNetworkData) -> dict[str, Any]:
    _validate_safe_persistence(
        data.metadata,
        path=f"embedded_networks[{data.name!r}].metadata",
    )
    return {
        "schema_version": _EMBEDDED_NETWORK_SNAPSHOT_SCHEMA_VERSION,
        "data_type": "EmbeddedNetworkData",
        "name": data.name,
        "frequencies": _cpu_serializable(data.frequencies),
        "port_names": data.port_names,
        "voltage": _cpu_serializable(data.voltage),
        "current": _cpu_serializable(data.current),
        "port_power": _cpu_serializable(data.port_power),
        "absorbed_power": _cpu_serializable(data.absorbed_power),
        "generated_power": _cpu_serializable(data.generated_power),
        "state_norm": _cpu_serializable(data.state_norm),
        "model_id": data.model_id,
        "fit_report": _fit_report_snapshot(data.fit_report),
        "runtime_warnings": data.runtime_warnings,
        "metadata": _cpu_serializable(data.metadata),
    }


def _port_data_from_snapshot(payload: Mapping[str, Any]) -> PortData:
    if not isinstance(payload, Mapping):
        raise ValueError("Result port snapshot must contain a mapping payload.")
    if payload.get("data_type") != "PortData":
        raise ValueError("Result port snapshot has an invalid data_type.")
    if payload.get("schema_version") != PortData.schema_version:
        raise ValueError(
            f"Unsupported PortData schema_version {payload.get('schema_version')!r}."
        )
    return PortData(
        port_name=payload["port_name"],
        frequencies=payload["frequencies"],
        voltage=payload["voltage"],
        current=payload["current"],
        z0=payload["z0"],
        direction=payload["direction"],
        reference_plane=payload["reference_plane"],
        available_power=payload["available_power"],
        mode_names=payload.get("mode_names"),
        beta=payload.get("beta"),
        characteristic_impedance=payload.get("characteristic_impedance"),
        tracking_confidence=payload.get("tracking_confidence"),
        metadata=payload["metadata"],
        phasor_convention=payload["phasor_convention"],
        power_wave_convention=payload["power_wave_convention"],
    )


def _network_data_from_snapshot(payload: Mapping[str, Any] | None) -> NetworkData | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("Result network snapshot must contain a mapping payload.")
    if payload.get("data_type") != "NetworkData":
        raise ValueError("Result network snapshot has an invalid data_type.")
    if payload.get("schema_version") != NetworkData.schema_version:
        raise ValueError(
            f"Unsupported NetworkData schema_version {payload.get('schema_version')!r}."
        )
    return NetworkData(
        frequencies=payload["frequencies"],
        s=payload["s"],
        z0=payload["z0"],
        port_names=tuple(payload["port_names"]),
        valid_columns=payload["valid_columns"],
        metadata=payload["metadata"],
        phasor_convention=payload["phasor_convention"],
        power_wave_convention=payload["power_wave_convention"],
    )


def _circuit_data_from_snapshot(payload: Mapping[str, Any]) -> CircuitData:
    return CircuitData._from_snapshot(payload)


def _validate_result_snapshot_payload(
    payload: Mapping[str, Any],
    *,
    sharded: bool,
) -> None:
    label = "Sharded Result metadata" if sharded else "Result checkpoint"
    if payload.get("data_type") != "ResultSnapshot":
        raise ValueError(f"{label} has an invalid data_type.")
    version = payload.get("schema_version")
    if version != RESULT_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported {label.lower()} schema_version={version!r}.")
    if "circuits" not in payload:
        raise ValueError(f"{label} schema v2 is missing required key: circuits.")
    if not isinstance(payload["circuits"], Mapping):
        raise ValueError(f"{label} circuits must contain a mapping payload.")


def _fit_report_from_snapshot(payload: Mapping[str, Any] | None) -> FitReport | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("Embedded network fit report must contain a mapping payload.")
    version = payload.get("schema_version")
    if version != _FIT_REPORT_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported embedded network fit report schema_version {version!r}."
        )
    data_type = payload.get("data_type")
    report_type = {
        "FitReport": FitReport,
        "NetworkFitReport": NetworkFitReport,
    }.get(data_type)
    if report_type is None:
        raise ValueError(
            f"Embedded network fit report has an invalid data_type {data_type!r}."
        )
    values = payload.get("values")
    if not isinstance(values, Mapping):
        raise ValueError("Embedded network fit report values must be a mapping.")
    expected = {field.name for field in fields(report_type)}
    actual = set(values)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected {', '.join(sorted(unexpected))}")
        raise ValueError(
            "Embedded network fit report fields are invalid: " + "; ".join(details) + "."
        )
    try:
        return report_type(**dict(values))
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedded network fit report values are invalid.") from exc


def _embedded_network_from_snapshot(
    payload: Mapping[str, Any],
) -> EmbeddedNetworkData:
    if not isinstance(payload, Mapping):
        raise ValueError("Embedded network snapshot must contain a mapping payload.")
    if payload.get("data_type") != "EmbeddedNetworkData":
        raise ValueError("Embedded network snapshot has an invalid data_type.")
    version = payload.get("schema_version")
    if version != _EMBEDDED_NETWORK_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported EmbeddedNetworkData schema_version {version!r}."
        )
    required = {
        "name",
        "frequencies",
        "port_names",
        "voltage",
        "current",
        "port_power",
        "absorbed_power",
        "generated_power",
        "state_norm",
        "model_id",
        "fit_report",
        "runtime_warnings",
        "metadata",
    }
    missing = required.difference(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Embedded network snapshot is missing required keys: {names}.")
    _validate_safe_persistence(
        payload["metadata"],
        path=f"embedded_networks[{payload['name']!r}].metadata",
    )
    fit_report = _fit_report_from_snapshot(payload["fit_report"])
    try:
        return EmbeddedNetworkData(
            name=payload["name"],
            frequencies=payload["frequencies"],
            port_names=tuple(payload["port_names"]),
            voltage=payload["voltage"],
            current=payload["current"],
            port_power=payload["port_power"],
            absorbed_power=payload["absorbed_power"],
            generated_power=payload["generated_power"],
            state_norm=payload["state_norm"],
            model_id=payload["model_id"],
            fit_report=fit_report,
            runtime_warnings=tuple(payload["runtime_warnings"]),
            metadata=payload["metadata"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Embedded network snapshot values are invalid.") from exc


def _embedded_networks_from_snapshot(payload: Any) -> dict[str, EmbeddedNetworkData]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Result embedded_networks snapshot must contain a mapping.")
    restored = {}
    for key, value in payload.items():
        name = str(key)
        data = _embedded_network_from_snapshot(value)
        if name != data.name:
            raise ValueError(
                f"Embedded network snapshot key {name!r} does not match data name "
                f"{data.name!r}."
            )
        restored[name] = data
    return restored


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
    if payload.get("power_delivered") is not None:
        selected["power_delivered"] = _slice_frequency_axis(payload["power_delivered"], len(frequencies), index)
    if payload.get("current_spectrum") is not None:
        selected["current_spectrum"] = _slice_frequency_axis(payload["current_spectrum"], len(frequencies), index)

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
    for axis, entry in enumerate(getattr(scene, "symmetry", (None, None, None))):
        if entry is None:
            continue
        mode, face = entry
        tensor_axis = spatial_axis_offset + axis
        half_size = half_sizes[axis]
        dim_size = int(expanded.shape[tensor_axis])
        # Components with a grid node on the symmetry plane have ``half_size``
        # samples; that shared plane must not be duplicated when mirroring.
        # Components staggered off the plane have ``half_size - 1`` samples and
        # are mirrored in full. The shared node sits at index 0 for a low-face
        # plane and at the last index for a high-face plane.
        if dim_size == half_size:
            if face == "low":
                mirrored_source = expanded.narrow(tensor_axis, 1, max(dim_size - 1, 0))
            else:
                mirrored_source = expanded.narrow(tensor_axis, 0, max(dim_size - 1, 0))
        elif dim_size == half_size - 1:
            mirrored_source = expanded
        else:
            raise ValueError(
                f"Cannot expand symmetry axis {axis} for tensor shape {tuple(expanded.shape)} and half-domain size {half_size}."
            )
        mirrored = torch.flip(mirrored_source, dims=(tensor_axis,))
        sign = _mirror_sign(component, mode, axis)
        if sign < 0:
            mirrored = -mirrored
        if face == "low":
            expanded = torch.cat((mirrored, expanded), dim=tensor_axis)
        else:
            expanded = torch.cat((expanded, mirrored), dim=tensor_axis)
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


def _monitor_payload_is_dipole_emission(payload: dict[str, Any]) -> bool:
    return payload.get("monitor_type") == "dipole_emission"


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
    if "cell_widths" in payload:
        width_u = payload["cell_widths"][coord_u_name]
        width_v = payload["cell_widths"][coord_v_name]
        selected["cell_widths"] = {
            coord_u_name: (
                width_u
                if u_indices is None
                else (
                    width_u.index_select(0, u_indices)
                    if isinstance(width_u, torch.Tensor)
                    else np.asarray(width_u)[u_indices]
                )
            ),
            coord_v_name: (
                width_v
                if v_indices is None
                else (
                    width_v.index_select(0, v_indices)
                    if isinstance(width_v, torch.Tensor)
                    else np.asarray(width_v)[v_indices]
                )
            ),
        }
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
        ports: Mapping[str, PortData] | None = None,
        circuits: Mapping[str, CircuitData] | None = None,
        network: NetworkData | None = None,
        embedded_networks: Mapping[str, EmbeddedNetworkData] | None = None,
        array_run_data=None,
        metadata: dict[str, Any] | None = None,
        solver_stats: dict[str, Any] | None = None,
        raw_output: Any = None,
        breakdown=None,
    ):
        self.method = method
        self.scene = scene
        self._prepared_scene = prepared_scene
        self.frequencies = _resolve_result_frequencies(frequency=frequency, frequencies=frequencies)
        self.frequency = self.frequencies[0]
        self.solver = solver
        self._fields = _clone_mapping(fields)
        self._monitors = _normalize_monitor_data_mapping(monitors)
        self._ports = _normalize_port_data_mapping(ports)
        self._circuits = _normalize_circuit_data_mapping(circuits)
        self._embedded_networks = _normalize_embedded_network_mapping(embedded_networks)
        if network is not None and not isinstance(network, NetworkData):
            raise TypeError("network must be a NetworkData or None.")
        self._network = network
        self._array_run_data = array_run_data
        self._metadata = _clone_mapping(metadata)
        self._solver_stats = _clone_mapping(solver_stats)
        self._sharded_manifest = None
        self._shard_paths: tuple[Path, ...] = ()
        self.raw_output = raw_output
        self._breakdown = breakdown
        self.plot = ResultPlotter(self)

    @property
    def breakdown_data(self):
        """Deterministic dielectric-breakdown outputs, or ``None`` for scenes with no
        breakdown material. See :class:`witwin.maxwell.breakdown.BreakdownResultData`.

        Distinct from :meth:`breakdown`, which returns the non-feedback
        dielectric-stress record of a named :class:`BreakdownMonitor`."""
        return self._breakdown

    @property
    def breakdown_events(self):
        """The typed breakdown event log ordered by ``(step, cell_index)`` (empty tuple
        when breakdown is inactive)."""
        if self._breakdown is None:
            return ()
        return self._breakdown.events

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

    def _esd_sources(self):
        from .esd import ESDCurrentSource

        scene = self.scene
        sources = getattr(scene, "sources", ()) if scene is not None else ()
        return {
            source.name: source
            for source in sources
            if isinstance(source, ESDCurrentSource)
        }

    def esd_waveform_names(self) -> tuple[str, ...]:
        """Names of ESD current sources declared on the run scene."""

        return tuple(self._esd_sources())

    def esd_waveform(self, name: str):
        """Return the typed ESD injection record for source ``name``.

        Capability level: stress-only. Exposes the target waveform diagnostics,
        the charge-conserving projection of the injected current onto the run
        time grid, and full provenance (standard revision, level voltage,
        colocation-independent scalar metrics, model version).

        The record also carries a ``measured`` port record recovered from the
        run when terminal-port voltage/current was recorded for the bound port
        (the RF ``PortData``), so users can compare the injected/target current
        against the measured port. For the Phase-1 ideal-current injection path
        no terminal-port recorder runs (the ESD source lowers to a volumetric
        current source), so ``measured`` is ``None`` and the injected current on
        the run grid is the ``resampled`` record; see :class:`ESDPortRecord` for
        the documented target-vs-measured limitation and workaround.
        """

        from .esd import ESDPortRecord

        sources = self._esd_sources()
        if name not in sources:
            available = ", ".join(sorted(sources)) or "<none>"
            raise KeyError(
                f"ESD waveform {name!r} is not present; available ESD sources: {available}."
            )
        source = sources[name]
        waveform = source.waveform
        diagnostics = waveform.diagnostics()
        resampled = None
        dt = None
        metadata = self._metadata or {}
        if metadata.get("dt") is not None:
            dt = float(metadata["dt"])
        elif self.solver is not None and getattr(self.solver, "dt", None) is not None:
            dt = float(self.solver.dt)
        if dt is not None and dt > 0.0:
            time_steps = metadata.get("time_steps")
            t_end = None
            if time_steps is not None:
                t_end = min(float(time_steps) * dt, float(waveform.support[1]))
                if t_end <= float(waveform.support[0]):
                    t_end = float(waveform.support[1])
            resampled = waveform.resample_to_grid(dt, t_end=t_end)
        provenance = source.provenance()
        # Expose a measured terminal-port record if the run recorded one for this
        # port (RF PortData); None for the ideal-current injection path.
        measured = self._ports.get(source.port_name)
        return ESDPortRecord(
            name=source.name,
            port_name=source.port_name,
            diagnostics=diagnostics,
            resampled=resampled,
            provenance=provenance,
            measured=measured,
        )

    def breakdown_names(self) -> tuple[str, ...]:
        """Names of BreakdownMonitor stress records present in the result."""

        from .breakdown_stress import BreakdownStressData

        return tuple(
            name
            for name, data in self._monitors.items()
            if isinstance(data, BreakdownStressData)
        )

    def breakdown(self, name: str):
        """Return the typed non-feedback dielectric-stress record for ``name``.

        Capability level: stress-only. Exposes peak field, exceedance duration,
        longest contiguous exceedance, qualifying sustained-stress locations, and
        per-cell maps (kept on device), with full threshold/colocation provenance.
        """

        from .breakdown_stress import BreakdownStressData

        payload = self._monitors.get(name)
        if not isinstance(payload, BreakdownStressData):
            available = ", ".join(sorted(self.breakdown_names())) or "<none>"
            raise KeyError(
                f"Breakdown stress record {name!r} is not present; available: {available}."
            )
        return payload

    def _extract_time_series(self, monitor_name: str):
        payload = self._monitors.get(monitor_name)
        if payload is None:
            raise KeyError(
                f"Time series {monitor_name!r} is not present in the result monitors."
            )
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Monitor {monitor_name!r} does not expose a time-series payload."
            )
        time = payload.get("t")
        if time is None:
            raise ValueError(f"Monitor {monitor_name!r} has no time axis 't'.")
        if "flux" in payload:
            values = payload["flux"]
        elif "data" in payload:
            values = payload["data"]
        elif "field" in payload:
            values = payload["field"]
        else:
            raise ValueError(
                f"Monitor {monitor_name!r} does not expose a scalar time series "
                "('data', 'field', or 'flux')."
            )
        return time, values

    def component_stress_names(self) -> tuple[str, ...]:
        """Names of ComponentStressMonitor bindings declared on the run scene."""

        from .monitors import ComponentStressMonitor

        scene = self.scene
        monitors = getattr(scene, "resolved_monitors", None)
        monitor_iter = monitors() if callable(monitors) else getattr(scene, "monitors", ())
        return tuple(
            monitor.name
            for monitor in monitor_iter
            if isinstance(monitor, ComponentStressMonitor)
        )

    def component_stress(self, name: str):
        """Return the typed component port-stress and rating-exceedance summary.

        Capability level: stress-only. Reduces the bound voltage/current time
        series into ``P = V I``, cumulative ``integral P dt`` and an exceedance
        summary versus the declared :class:`ComponentRating` envelope.
        """

        from .monitors import ComponentStressMonitor
        from .breakdown_stress import ComponentStressData

        scene = self.scene
        monitors = getattr(scene, "resolved_monitors", None)
        monitor_iter = monitors() if callable(monitors) else getattr(scene, "monitors", ())
        binding = None
        for monitor in monitor_iter:
            if isinstance(monitor, ComponentStressMonitor) and monitor.name == name:
                binding = monitor
                break
        if binding is None:
            available = ", ".join(sorted(self.component_stress_names())) or "<none>"
            raise KeyError(
                f"ComponentStressMonitor {name!r} is not present; available: {available}."
            )
        v_time, voltage = self._extract_time_series(binding.voltage_series)
        i_time, current = self._extract_time_series(binding.current_series)
        if not isinstance(v_time, torch.Tensor):
            v_time = torch.as_tensor(v_time)
        if not isinstance(i_time, torch.Tensor):
            i_time = torch.as_tensor(i_time)
        if v_time.numel() != i_time.numel():
            raise ValueError(
                f"ComponentStressMonitor {name!r} voltage series "
                f"{binding.voltage_series!r} and current series "
                f"{binding.current_series!r} have mismatched sample counts."
            )
        i_time_cmp = i_time.to(device=v_time.device, dtype=v_time.dtype)
        span = torch.abs(v_time[-1] - v_time[0]) if v_time.numel() > 1 else torch.abs(v_time).max()
        atol = float(span) * 1e-6 if float(span) > 0.0 else 1e-12
        if not torch.allclose(v_time, i_time_cmp, rtol=1e-6, atol=atol):
            raise ValueError(
                f"ComponentStressMonitor {name!r} voltage series "
                f"{binding.voltage_series!r} and current series "
                f"{binding.current_series!r} are recorded on different time axes; "
                "voltage and current must share a common time grid for power and "
                "energy integration."
            )
        voltage = voltage if isinstance(voltage, torch.Tensor) else torch.as_tensor(voltage)
        current = current if isinstance(current, torch.Tensor) else torch.as_tensor(current)
        return ComponentStressData.from_time_series(
            v_time,
            voltage,
            current,
            binding.rating,
            name=binding.name,
            port_name=binding.port,
            provenance_extra={
                "voltage_series": binding.voltage_series,
                "current_series": binding.current_series,
            },
        )

    @property
    def ports(self) -> dict[str, PortData]:
        return dict(self._ports)

    @property
    def circuits(self) -> dict[str, CircuitData]:
        return dict(self._circuits)

    @property
    def network(self) -> NetworkData | None:
        return self._network

    @property
    def embedded_networks(self) -> dict[str, EmbeddedNetworkData]:
        return dict(self._embedded_networks)

    @property
    def solver_stats(self) -> dict[str, Any]:
        return dict(self._solver_stats)

    @property
    def is_sharded(self) -> bool:
        return self._sharded_manifest is not None

    @property
    def sharded_manifest(self):
        return self._sharded_manifest

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        return self._shard_paths

    @property
    def electrostatic(self):
        """Typed electrostatic solver output (``ElectrostaticResultData``)."""
        from .electrostatic.runtime import ElectrostaticResultData

        if self.method != "electrostatic" or not isinstance(self.raw_output, ElectrostaticResultData):
            raise AttributeError(
                "result.electrostatic is only available for Simulation.electrostatic(...) runs."
            )
        return self.raw_output

    @property
    def capacitance(self):
        """Typed capacitance-matrix output (``CapacitanceData``)."""
        from .electrostatic.capacitance import CapacitanceData

        if self.method != "capacitance" or not isinstance(self.raw_output, CapacitanceData):
            raise AttributeError(
                "result.capacitance is only available for Simulation.capacitance(...) runs."
            )
        return self.raw_output

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
        if isinstance(payload, WireData):
            if frequency is not None or freq_index is not None:
                raise ValueError(
                    "WireData preserves its explicit frequency axis; retrieve the "
                    "full typed result and select its tensors explicitly."
                )
            return payload
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
        payload = self._monitors.get(name)
        if isinstance(payload, WireData):
            if frequency is not None or freq_index is not None:
                raise ValueError(
                    "WireData preserves its explicit frequency axis; retrieve the "
                    "full typed result and select its tensors explicitly."
                )
            return payload
        public_monitor = _find_scene_monitor(self.scene, name)
        if isinstance(public_monitor, PowerLossMonitor):
            if frequency is not None or freq_index is not None:
                raise ValueError(
                    "PowerLossData preserves its explicit frequency axis; retrieve the "
                    "full typed result and select its tensors explicitly."
                )
            return self.power_loss(name)
        payload = self.raw_monitor(name, frequency=frequency, freq_index=freq_index)
        if not resolve_modal or _monitor_payload_is_closed_surface(payload):
            return payload

        if _monitor_payload_is_diffraction(payload):
            from .postprocess.diffraction import compute_diffraction_from_payload

            return compute_diffraction_from_payload(self, name, payload)

        if _monitor_payload_is_dipole_emission(payload):
            emission = {
                "kind": "dipole_emission",
                "monitor_type": "dipole_emission",
                "power_delivered": payload.get("power_delivered"),
                "current_spectrum": payload.get("current_spectrum"),
                "frequencies": tuple(payload.get("frequencies", (payload.get("frequency"),))),
                "frequency": payload.get("frequency"),
                "polarization": payload.get("dipole_polarization"),
                "position": payload.get("dipole_position", payload.get("position")),
                "source_name": payload.get("source_name"),
                # The Purcell factor requires a vacuum reference run; combine two
                # runs with postprocess.purcell_factor(structured, vacuum, name).
                "purcell_factor": None,
            }
            return emission

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

    def port(self, name: str) -> PortData:
        port_name = str(name)
        if port_name not in self._ports:
            choices = tuple(self._ports)
            raise KeyError(
                f"Port {port_name!r} is not available in this result. "
                f"Choices: {choices}."
            )
        return self._ports[port_name]

    def circuit(self, name: str) -> CircuitData:
        circuit_name = str(name)
        if circuit_name not in self._circuits:
            choices = tuple(self._circuits)
            raise KeyError(
                f"Circuit {circuit_name!r} is not available in this result. "
                f"Choices: {choices}."
            )
        return self._circuits[circuit_name]

    def embedded_network(self, name: str) -> EmbeddedNetworkData:
        network_name = str(name)
        if network_name not in self._embedded_networks:
            choices = tuple(self._embedded_networks)
            raise KeyError(
                f"Embedded network {network_name!r} is not available in this result. "
                f"Choices: {choices}."
            )
        return self._embedded_networks[network_name]

    def antenna(
        self,
        *,
        surface: str,
        driven_port: str | PortData,
        polarization=None,
        theta=None,
        phi=None,
        theta_points: int = 181,
        phi_points: int = 361,
        radius=1.0,
        phase_center=None,
        frame=None,
        batch_size: int = 1024,
    ):
        """Compute antenna engineering data from a closed near-field surface."""

        from .postprocess.antenna import antenna_data_from_result

        return antenna_data_from_result(
            self,
            surface=surface,
            driven_port=driven_port,
            polarization=polarization,
            theta=theta,
            phi=phi,
            theta_points=theta_points,
            phi_points=phi_points,
            radius=radius,
            phase_center=phase_center,
            frame=frame,
            batch_size=batch_size,
        )

    def array_basis(
        self,
        *,
        monitor: str,
        polarization=None,
        theta=None,
        phi=None,
        theta_points: int = 181,
        phi_points: int = 361,
        radius=1.0,
        phase_center=None,
        frame=None,
        batch_size: int = 1024,
    ):
        """Build a reusable power-wave/embedded-pattern basis from a PortSweep."""

        from .postprocess.array import array_basis_from_result

        return array_basis_from_result(
            self,
            monitor=monitor,
            polarization=polarization,
            theta=theta,
            phi=phi,
            theta_points=theta_points,
            phi_points=phi_points,
            radius=radius,
            phase_center=phase_center,
            frame=frame,
            batch_size=batch_size,
        )

    def power_loss(
        self,
        name: str,
        *,
        electric_fields=None,
        volume_channels=None,
        integrated_channels=None,
        surface_channels=None,
        face_areas=None,
        line_channels=None,
        line_lengths=None,
        occupancy=None,
        material_ids=None,
        geometry_ids=None,
    ):
        """Compute typed power-loss channels for a declared loss monitor."""

        monitor = _find_scene_monitor(self.scene, name)
        if not isinstance(monitor, PowerLossMonitor):
            raise KeyError(f"PowerLossMonitor {name!r} is not declared in this Scene.")
        if self.method != "fdtd":
            raise NotImplementedError(
                "Automatic PowerLossMonitor field colocation currently supports FDTD results only."
            )

        from .compiler.power_loss import compile_power_loss_monitor
        from .postprocess.power_loss import compute_power_loss_data

        compiled = compile_power_loss_monitor(self.prepared_scene, monitor)
        selected_frequencies = (
            tuple(self.frequencies)
            if monitor.frequencies is None
            else tuple(monitor.frequencies)
        )
        frequency_tensor = torch.as_tensor(
            selected_frequencies,
            device=compiled.device,
            dtype=torch.float64,
        )
        resolved_fields = electric_fields
        if resolved_fields is None and "conduction" in monitor.channels:
            missing = tuple(
                component
                for component in ("Ex", "Ey", "Ez")
                if component.upper() not in self._fields
            )
            if missing:
                raise RuntimeError(
                    "PowerLossMonitor conduction requires frequency-domain Ex/Ey/Ez fields; "
                    "run FDTD with full_field_dft=True."
                )
            resolved_fields = {
                component: torch.stack(
                    [
                        self.tensor(component, frequency=frequency)
                        for frequency in selected_frequencies
                    ],
                    dim=0,
                )
                for component in ("Ex", "Ey", "Ez")
            }
        return compute_power_loss_data(
            compiled,
            frequency_tensor,
            electric_fields=resolved_fields,
            volume_channels=volume_channels,
            integrated_channels=integrated_channels,
            surface_channels=surface_channels,
            face_areas=face_areas,
            line_channels=line_channels,
            line_lengths=line_lengths,
            occupancy=occupancy,
            material_ids=material_ids,
            geometry_ids=geometry_ids,
            source_result_fingerprint=f"runtime-result:{id(self):x}",
        )

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
        stats["num_ports"] = len(self._ports)
        stats["num_circuits"] = len(self._circuits)
        stats["has_network"] = self._network is not None
        stats["num_embedded_networks"] = len(self._embedded_networks)
        return stats

    def _snapshot_payload(self, *, fields: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Build and validate the detached payload before any I/O side effect."""

        return {
            "schema_version": RESULT_SNAPSHOT_SCHEMA_VERSION,
            "data_type": "ResultSnapshot",
            "format_version": 1,
            "method": self.method,
            "frequency": self.frequency,
            "frequencies": self.frequencies,
            "fields": {
                name: tensor.detach().cpu()
                for name, tensor in fields.items()
            },
            "monitors": _monitor_mapping_snapshot(self._monitors),
            "ports": {
                name: _port_data_snapshot(data)
                for name, data in self._ports.items()
            },
            "circuits": {
                name: _circuit_data_snapshot(data)
                for name, data in self._circuits.items()
            },
            "network": _network_data_snapshot(self._network),
            "embedded_networks": {
                name: _embedded_network_snapshot(data)
                for name, data in self._embedded_networks.items()
            },
            "metadata": _cpu_serializable(self._metadata),
            "solver_stats": _cpu_serializable(self._solver_stats),
        }

    def save(self, path: str | Path):
        """Save a detached CPU data snapshot.

        Port payloads use the same versioned schema as ``PortData.save``. Embedded
        network diagnostics use a nested schema that preserves the concrete fit
        report type. The snapshot intentionally omits the declarative Scene,
        prepared Scene, solver, raw runtime output, and every live autograd graph.
        Loading therefore requires the caller to supply the corresponding
        declarative Scene.

        Retained in-memory array sweep columns are also omitted. Call
        ``result.array_basis(...)`` and save the resulting ``ArrayBasisData`` before
        saving when delayed embedded-pattern reuse is required.
        """

        payload = self._snapshot_payload(fields=self._fields)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)

    def save_sharded(self, directory: str | Path):
        """Persist owned field shards without assembling global field tensors."""

        destination = Path(directory)
        if destination.exists():
            raise FileExistsError(
                "Sharded Result directory already exists; refusing non-atomic overwrite: "
                f"{destination}."
            )
        exporter = getattr(self.solver, "export_field_shards", None)
        if not callable(exporter):
            raise RuntimeError(
                "Result.save_sharded() requires solver.export_field_shards(); "
                "this Result has no shard field export provider."
            )
        from .fdtd.distributed.persistence import write_sharded_result

        payload = self._snapshot_payload(fields={})
        shard_artifacts = exporter()
        return write_sharded_result(
            destination,
            result_payload=payload,
            shard_artifacts=shard_artifacts,
            frequencies=self.frequencies,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        scene,
        prepared_scene=None,
        map_location: Any = "cpu",
    ) -> "Result":
        """Load detached inference data saved by :meth:`save`.

        The declarative scene is supplied by the caller because runtime solvers,
        prepared scenes, and transport state are intentionally not persisted.
        Result files use pickle-backed ``torch.save`` and must therefore only be
        loaded from trusted sources.
        """

        payload = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise ValueError("Result checkpoint must contain a mapping payload.")
        _validate_result_snapshot_payload(payload, sharded=False)
        missing = {"method", "fields", "monitors"}.difference(payload)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Result checkpoint is missing required keys: {names}.")

        return cls(
            method=payload["method"],
            scene=scene,
            prepared_scene=prepared_scene,
            frequency=payload.get("frequency"),
            frequencies=payload.get("frequencies"),
            fields=payload["fields"],
            monitors=_monitor_mapping_from_snapshot(payload["monitors"]),
            ports={
                name: _port_data_from_snapshot(data)
                for name, data in payload.get("ports", {}).items()
            },
            circuits={
                name: _circuit_data_from_snapshot(data)
                for name, data in payload["circuits"].items()
            },
            network=_network_data_from_snapshot(payload.get("network")),
            embedded_networks=_embedded_networks_from_snapshot(
                payload.get("embedded_networks")
            ),
            metadata=payload.get("metadata"),
            solver_stats=payload.get("solver_stats"),
        )

    @classmethod
    def load_sharded(
        cls,
        directory: str | Path,
        *,
        scene,
        prepared_scene=None,
        gather_fields: bool = False,
        map_location: Any = "cpu",
    ) -> "Result":
        """Load sharded metadata lazily, gathering owned fields only on request."""

        if not isinstance(gather_fields, bool):
            raise TypeError("gather_fields must be a bool.")
        from .fdtd.distributed.persistence import load_sharded_result

        loaded = load_sharded_result(
            directory,
            gather_fields=gather_fields,
            map_location=map_location,
        )
        payload = loaded.result_payload
        _validate_result_snapshot_payload(payload, sharded=True)
        missing = {"method", "monitors"}.difference(payload)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Sharded Result metadata is missing required keys: {names}.")
        result = cls(
            method=payload["method"],
            scene=scene,
            prepared_scene=prepared_scene,
            frequency=payload.get("frequency"),
            frequencies=payload.get("frequencies"),
            fields=loaded.fields,
            monitors=_monitor_mapping_from_snapshot(payload["monitors"]),
            ports={
                name: _port_data_from_snapshot(data)
                for name, data in payload.get("ports", {}).items()
            },
            circuits={
                name: _circuit_data_from_snapshot(data)
                for name, data in payload["circuits"].items()
            },
            network=_network_data_from_snapshot(payload.get("network")),
            embedded_networks=_embedded_networks_from_snapshot(
                payload.get("embedded_networks")
            ),
            metadata=payload.get("metadata"),
            solver_stats=payload.get("solver_stats"),
        )
        result._sharded_manifest = loaded.manifest
        result._shard_paths = loaded.shard_paths
        return result
