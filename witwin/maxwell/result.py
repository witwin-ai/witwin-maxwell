from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, fields
from pathlib import Path
from io import BytesIO
from typing import Any

import numpy as np
import torch

from .circuits import CircuitData
from .monitors import (
    ClosedSurfaceMonitor,
    FinitePlaneMonitor,
    IncidentPowerDensityMonitor,
    MediumMonitor,
    PermittivityMonitor,
    PowerLossMonitor,
)
from .monitors import GAUSSIAN_OVERLAP_MONITOR_TYPES, PROJECTION_MONITOR_TYPES
from .network import (
    EmbeddedNetworkData,
    NetworkData,
    PortData,
    _validate_safe_persistence,
)
from .rational import FitReport, NetworkFitReport
from .ports_postprocess import PortDrive
from .scene import prepare_scene
from .thin_wire import WireData
from .visualization import extract_orthogonal_slice, plot_slice_image

_UNSET = object()
RESULT_SNAPSHOT_SCHEMA_VERSION = 2
_EMBEDDED_NETWORK_SNAPSHOT_SCHEMA_VERSION = 1
_FIT_REPORT_SNAPSHOT_SCHEMA_VERSION = 1


def _clone_mapping(data: dict[str, Any]) -> dict[str, Any]:
    return dict(data) if data is not None else {}


def _mixed_absolute_spectral_stats(stats):
    """Sweep columns may group single-frequency runs into tuples."""
    if isinstance(stats, (tuple, list)):
        return any(_mixed_absolute_spectral_stats(column) for column in stats)
    return (stats.get("time_plan", {}).get("mixed_sources", False)
            and not stats.get("normalize_source", False)) or any(
        _mixed_absolute_spectral_stats(column) for column in stats.get("columns", ()))


class _RatioOnlyPortData:
    """Live port ratios computed from private common-scale V/I samples."""

    __slots__ = ("_port",)

    def __init__(self, port: PortData):
        self._port = port

    def __getattr__(self, name):
        if name == "metadata":
            return {key: value for key, value in self._port.metadata.items()
                    if key not in {"terminal_voltage", "terminal_current"}}
        if name in {
            "voltage", "current", "available_power", "a", "b",
            "incident_power", "reflected_power", "accepted_power", "delivered_power",
            "save",
        }:
            from .fdtd.runtime.representation import require_absolute_spectra
            require_absolute_spectra(True)
        if name in {
            "z_in", "reflection_coefficient", "return_loss_db", "return_loss", "vswr",
            "z0", "frequencies", "port_name", "mode_names", "direction",
            "reference_plane", "beta", "characteristic_impedance", "tracking_confidence",
            "phasor_convention", "power_wave_convention", "schema_version",
        }:
            return getattr(self._port, name)
        raise AttributeError(name)


def _copy_monitor_containers(value):
    """Give callers independent payload containers while sharing live tensors."""
    if isinstance(value, dict):
        return {key: _copy_monitor_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_monitor_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_monitor_containers(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _clone_conductor_diagnostics(value):
    """Return independent diagnostic ownership without detaching live tensors."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: _clone_conductor_diagnostics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_conductor_diagnostics(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_conductor_diagnostics(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _cpu_serializable(value: Any, *, _memo=None):
    # Monitor payloads expose the same GPU Tensor under several schema keys.
    # Preserve that identity while copying; distinct views stay distinct. Keep
    # source references alive so temporary objects cannot reuse a cached id.
    if _memo is None:
        _memo = {}
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in _memo:
            _memo[key] = (value, value.detach().cpu())
        return _memo[key][1]
    if isinstance(value, Mapping):
        return {key: _cpu_serializable(item, _memo=_memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_serializable(item, _memo=_memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_serializable(item, _memo=_memo) for item in value)
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
    if payload.get("kind") == "linear_projection" and "value" in payload:
        selected["value"] = _slice_frequency_axis(payload["value"], len(frequencies), index)
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


def _finite_plane_cell_selection(coords, widths, lower, upper, *, whole_cells=False):
    """Intersect physical rectangles, including cells with exterior centers.

    ``whole_cells`` applies to node-aligned (closed-surface) bounds: every cell
    lies entirely inside or outside them, so cells are selected by center and
    kept whole, without round-off slivers from reconstructed cell edges.
    """
    tensor = isinstance(coords, torch.Tensor)
    points = coords if tensor else np.asarray(coords, dtype=float)
    span = (torch.as_tensor(widths, device=points.device, dtype=points.dtype)
            if tensor else np.asarray(widths, dtype=float))
    if span.shape != points.shape:
        raise ValueError("Finite plane cell widths must match their coordinates.")
    if whole_cells:
        inside = (points > lower) & (points < upper)
        indices = (torch.nonzero(inside, as_tuple=False).flatten() if tensor else np.flatnonzero(inside))
        if len(indices) == 0:
            raise ValueError("Closed-surface face bounds do not contain any retained physical cells.")
        return indices, points[indices], span[indices]
    if tensor:
        left = torch.maximum(points - span / 2, points.new_tensor(lower))
        right = torch.minimum(points + span / 2, points.new_tensor(upper))
        indices = torch.nonzero(right > left, as_tuple=False).flatten()
    else:
        left = np.maximum(points - span / 2, lower)
        right = np.minimum(points + span / 2, upper)
        indices = np.flatnonzero(right > left)
    if len(indices) == 0:
        raise ValueError("Finite monitor bounds do not intersect any retained physical cells.")
    clipped_widths = (right - left)[indices]
    centers = ((right + left) / 2)[indices]
    # Preserve all unchanged native coordinates exactly, including roundoff.
    full = ((points - span / 2 >= lower) & (points + span / 2 <= upper))[indices]
    if tensor:
        centers = torch.where(full, points[indices], centers)
    else:
        centers = np.where(full, points[indices], centers)
    return indices, centers, clipped_widths


def _resample_plane_samples(values, native_coords, target_coords):
    """Differentiable bilinear evaluation of aligned plane fields."""
    result = values
    for offset, (native, target) in enumerate(zip(native_coords, target_coords)):
        axis = result.ndim - 2 + offset
        if isinstance(result, torch.Tensor):
            source = torch.as_tensor(native, device=result.device, dtype=torch.float64)
            points = torch.as_tensor(target, device=result.device, dtype=torch.float64)
            if len(source) == 1:
                low = high = torch.zeros(len(points), device=result.device, dtype=torch.long)
                fraction = torch.zeros_like(points)
            else:
                high = torch.searchsorted(source.contiguous(), points.contiguous()).clamp(1, len(source)-1)
                low = high - 1
                fraction = (points-source[low])/(source[high]-source[low])
            shape = [1]*result.ndim
            shape[axis] = len(points)
            weight = fraction.to(dtype=result.real.dtype).reshape(shape)
            result = result.index_select(axis, low)*(1-weight) + result.index_select(axis, high)*weight
        else:
            source, points = np.asarray(native), np.asarray(target)
            if len(source) == 1:
                low = high = np.zeros(len(points), dtype=int)
                fraction = np.zeros_like(points)
            else:
                high = np.clip(np.searchsorted(source, points), 1, len(source)-1)
                low = high-1
                fraction = (points-source[low])/(source[high]-source[low])
            shape = [1]*result.ndim
            shape[axis] = len(points)
            weight = fraction.reshape(shape)
            result = np.take(result, low, axis=axis)*(1-weight) + np.take(result, high, axis=axis)*weight
    return result


def _crop_cell_centered_plane_payload(payload, monitor, coord_names, bounds):
    native = tuple(payload[name] for name in coord_names)
    selections = [_finite_plane_cell_selection(payload[name], payload["cell_widths"][name],
                  *bounds[name], whole_cells=monitor.closure_planes is not None) for name in coord_names]
    coordinates = tuple(selection[1] for selection in selections)
    selected = dict(payload)
    selected['native_components'] = payload.get('native_components', payload.get('components', {}))
    selected.update(zip(coord_names, coordinates))
    selected["coords"] = coordinates
    selected["cell_widths"] = {name: selection[2] for name, selection in zip(coord_names, selections)}
    selected["finite_cell_selection"] = {
        "source_indices": dict(zip(coord_names, (selection[0] for selection in selections))),
        "source_coords": dict(zip(coord_names, native)),
        "rule": ("watertight_closure_at_snapped_adjacent_planes" if monitor.closure_planes is not None
                 else "intersected_rectangles_at_clipped_midpoints"),
    }
    # Top-level fields are already collocated; native component payloads retain
    # staggered support and cannot be sliced using aligned-grid indices.
    components = {}
    for name in payload.get("fields", ()):
        if name not in payload:
            continue
        value = _resample_plane_samples(payload[name], native, coordinates)
        selected[name] = value
        component = dict(payload.get("components", {}).get(name, {}))
        component.update(data=value, coords=coordinates, collocated=True)
        components[name] = component
    if "data" in payload:
        selected["data"] = _resample_plane_samples(payload["data"], native, coordinates)
    selected["components"] = components
    selected["collocated"] = True
    selected.update(monitor_type="finite_plane", center=monitor.position, size=monitor.size)
    _record_plane_bounds(selected, monitor, bounds)
    for name in ("face_label", "surface_name"):
        if getattr(monitor, name) is not None:
            selected[name] = getattr(monitor, name)
    if selected.get("compute_flux"):
        from .fdtd.observers import _compute_plane_flux
        selected["flux"] = selected["power"] = _compute_plane_flux(selected)
    return selected


def _record_plane_bounds(selected, monitor, bounds):
    selected["tangential_bounds"] = dict(bounds)
    if monitor.closure_planes is not None:
        selected["declared_tangential_bounds"] = dict(monitor.tangential_bounds)
        selected["closure_planes"] = dict(monitor.closure_planes)


def _crop_plane_monitor_payload(
    payload: dict[str, Any],
    monitor: FinitePlaneMonitor,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Restrict a plane payload to its integration rectangle.

    ``bounds`` defaults to the declared rectangle. Closed-surface faces pass the
    grid-resolved watertight rectangle from ``finite_plane_integration_bounds``.
    """
    coord_u_name, coord_v_name = _plane_coord_names(monitor.axis)
    if coord_u_name not in payload or coord_v_name not in payload:
        return dict(payload)
    bounds = dict(monitor.tangential_bounds) if bounds is None else dict(bounds)

    if "cell_widths" in payload:
        return _crop_cell_centered_plane_payload(payload, monitor, (coord_u_name, coord_v_name), bounds)

    u_indices = _select_coord_indices(payload[coord_u_name], *bounds[coord_u_name])
    v_indices = _select_coord_indices(payload[coord_v_name], *bounds[coord_v_name])
    if u_indices is None and v_indices is None:
        selected = dict(payload)
        selected["monitor_type"] = "finite_plane"
        selected["center"] = monitor.position
        selected["size"] = monitor.size
        _record_plane_bounds(selected, monitor, bounds)
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
    _record_plane_bounds(selected, monitor, bounds)
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
    complete_flux=True
    for face in monitor.faces:
        face_payload = result._native_raw_monitor(face.name, frequency=frequency, freq_index=freq_index)
        from .monitors import required_flux_fields
        if all(component in face_payload for component in required_flux_fields(face.axis)):
            if "flux" not in face_payload:
                from .fdtd.quadrature import _compute_plane_flux
                face_payload=dict(face_payload)
                face_payload["flux"]=face_payload["power"]=_compute_plane_flux(face_payload)
        else:
            complete_flux=False
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
    if complete_flux:
        payload["flux"]=payload["power"]=sum(face["flux"] for face in faces.values())
        payload["power_unit"]="W"
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
        self.selection.result._require_frequency_volume_fields()
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
        self.result._require_frequency_volume_fields()
        return ResultFieldAccessor(self, "E")

    @property
    def H(self) -> ResultFieldAccessor:
        self.result._require_frequency_volume_fields()
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
    @classmethod
    def from_fields(
        cls, scene, *, frequencies, kernel: str, fields=None, monitors=None,
        ports=None, network=None, metadata=None, prepared_scene=None,
    ) -> "Result":
        """Wrap externally computed Yee fields and payloads in a standard Result."""
        if not isinstance(kernel, str) or not kernel:
            raise ValueError("kernel must be a non-empty string.")
        metadata = dict(metadata) if metadata is not None else {}
        if "kernel" in metadata and metadata["kernel"] != kernel:
            raise ValueError("metadata kernel must match kernel.")
        metadata["kernel"] = kernel
        return cls(
            method="external", scene=scene, prepared_scene=prepared_scene,
            frequencies=frequencies, fields=fields, monitors=monitors,
            ports=ports, network=network, metadata=metadata,
        )

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
        surface_data=None,
        harmonic_data=None,
    ):
        self.method = method
        if surface_data is not None:
            from .mom.runtime import SURFACE_DATA_TYPES
            if method != "mom" or not isinstance(surface_data, SURFACE_DATA_TYPES):
                raise TypeError("surface_data requires an MoM Result and SurfaceData payload")
        if harmonic_data is not None:
            from .mom.harmonic_runtime import HarmonicData
            if method != "mom" or not isinstance(harmonic_data,HarmonicData) or surface_data is not None:
                raise TypeError("harmonic_data requires an MoM Result with only a HarmonicData payload")
        self._harmonic_data = harmonic_data
        if method == "mom" and ((surface_data is None and harmonic_data is None) or fields):
            raise ValueError("MoM Results require native surface data and cannot carry Yee field arrays")
        self._surface_data = surface_data
        self.scene = scene
        self._prepared_scene = prepared_scene
        self.frequencies = _resolve_result_frequencies(frequency=frequency, frequencies=frequencies)
        self.frequency = self.frequencies[0]
        if harmonic_data is not None:
            harmonic_data.state.assert_current()
            if self.frequencies != tuple(float(f) for f in harmonic_data.state.frequencies):
                raise ValueError("Result frequencies must match the actual coupled harmonic state")
        self.solver = solver
        self._fields = _clone_mapping(fields)
        self._monitors = _normalize_monitor_data_mapping(monitors)
        self._closed_surface_payload_cache = {}
        self._ports = _normalize_port_data_mapping(ports)
        self._circuits = _normalize_circuit_data_mapping(circuits)
        self._embedded_networks = _normalize_embedded_network_mapping(embedded_networks)
        if network is not None and not isinstance(network, NetworkData):
            raise TypeError("network must be a NetworkData or None.")
        self._network = network
        self._array_run_data = array_run_data
        self._metadata = _clone_mapping(metadata)
        if solver is not None and getattr(solver, "_port_drives", None):
            self._metadata["port_drives"] = _clone_mapping(solver._port_drives)
        coordinates=getattr(scene,"coordinates",None)
        if coordinates is not None:
            from .coordinates import FieldFrame
            frame=FieldFrame("covariant_chart",coordinates).snapshot()
            previous=self._metadata.get("field_frame")
            if previous is not None and previous!=frame:
                raise ValueError("Result field frame differs from the supplied physical coordinate chart.")
            self._metadata["field_frame"]=frame
        elif self._metadata.get("field_frame",{}).get("kind")=="covariant_chart":
            raise ValueError("Covariant chart field data requires its matching coordinate Scene.")
        if method == "fdtd" and (getattr(solver, "_copper_foils", ())
                                or raw_output is not None and "conductor_losses" in raw_output):
            from .fdtd.copper_foil import copper_loss_report
            self._metadata["conductor_losses"] = (
                _clone_mapping(raw_output["conductor_losses"])
                if raw_output is not None and "conductor_losses" in raw_output else copper_loss_report(solver)
            )
        self._solver_stats = _clone_mapping(solver_stats)
        if method == "fdtd":
            columns = self._solver_stats.get("columns", ())
            if ("execution_path" not in self._solver_stats and columns
                    and all("execution_path" in column for column in columns)):
                self._solver_stats.update(
                    execution_path="core" if all(column["execution_path"] == "core" for column in columns) else "existing",
                    fallback_reason=next((column["fallback_reason"] for column in columns
                                          if column.get("fallback_reason") is not None), None),
                )
            if "execution_path" in self._solver_stats:
                self._solver_stats.setdefault("fallback_reason", None)
                self._metadata.update(execution_path=self._solver_stats["execution_path"],
                                      fallback_reason=self._solver_stats["fallback_reason"])
        self._mixed_absolute_spectra = method == "fdtd" and _mixed_absolute_spectral_stats(self._solver_stats)
        self._allow_spectral_ratios = False
        self._sharded_manifest = None
        self._shard_paths: tuple[Path, ...] = ()
        self._raw_output = raw_output
        self._breakdown = breakdown

    def _require_absolute_spectra(self):
        from .fdtd.runtime.representation import require_absolute_spectra
        if not self._allow_spectral_ratios:
            require_absolute_spectra(self._mixed_absolute_spectra)

    def _spectral_ratio_view(self):
        """Internal zero-copy samples for reductions that return only ratios."""
        result = copy(self)
        result._allow_spectral_ratios = True
        return result

    def _public_port(self, port):
        if self._mixed_absolute_spectra and not self._allow_spectral_ratios:
            return _RatioOnlyPortData(port)
        return port

    def _require_monitor_spectra(self, name):
        if not self._mixed_absolute_spectra or self._allow_spectral_ratios:
            return
        payload = self._monitors.get(name)
        if isinstance(payload, dict) and payload.get("kind", "").endswith("_time"):
            return
        if isinstance(_find_scene_monitor(self.scene, name), (PermittivityMonitor, MediumMonitor)):
            return
        self._require_absolute_spectra()

    @property
    def raw_output(self):
        self._require_absolute_spectra()
        return self._raw_output

    @property
    def metadata(self) -> dict[str, Any]:
        """The Result's mutable metadata mapping."""
        return self._metadata

    def release_solver(self):
        """Free solver-held direct factors and drop the solver; dependent accessors behave as on a loaded Result.
        Trimming allocator pools is the caller's choice."""
        if hasattr(self.solver, "_release_direct_solver"):
            self.solver._release_direct_solver()
        self.solver = None

    @property
    def conductor_losses(self):
        """Independent copies of per-foil energy (J), spectral work (W) and geometry.

        Native spectral tile work is detached diagnostic data. Copying a tensor
        preserves any separately supplied live graph rather than detaching it.
        The energy_representation identifies physical real-field joules versus
        combined complex-bank algorithmic work/state, not real transient heat.
        """
        self._require_absolute_spectra()
        return _clone_conductor_diagnostics(self._metadata.get("conductor_losses", {}))

    @property
    def conductor_loss_columns(self):
        """Sweep losses keyed by excitation port, then foil name (joules/watts).

        Each column retains the single-run scalar energies and [F] surface
        powers; independent excitation runs are never summed into one loss.
        """
        self._require_absolute_spectra()
        return _clone_conductor_diagnostics(self._metadata.get("conductor_loss_columns", {}))

    @property
    def plot(self):
        # Construct the facade on demand: storing it on Result creates a cycle
        # that delays releasing streamed GPU results until cyclic GC runs.
        return ResultPlotter(self)

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
    def surface_currents(self):
        """Typed scattered RWG coefficients plus the exact incident carrier."""
        if self._surface_data is None:
            raise ValueError("This Result has no native surface currents")
        return self._surface_data

    @property
    def harmonic_state(self):
        """Coupled DC/AC surface and volume unknowns with their actual frequency axis."""
        if self._harmonic_data is None:
            raise ValueError("This Result has no coupled harmonic state")
        self._harmonic_data.state.assert_current()
        return self._harmonic_data.state

    def compare_harmonic_cutoff(self, previous, *, relative_tolerance=1e-4,
                                absolute_tolerance=1e-10, tail_absolute_tolerance=1e-8):
        """Compare fixed-grid cell fields and added harmonic amplitudes in V/m.

        This is an empirical cutoff estimate, not spatial or physical qualification.
        """
        if not isinstance(previous,Result) or previous._harmonic_data is None or self._harmonic_data is None:
            raise ValueError("Harmonic cutoff comparison requires two coupled harmonic Results")
        from .mom.harmonic_convergence import compare_harmonic_cells
        return compare_harmonic_cells(previous._harmonic_data,self._harmonic_data,
            relative_tolerance=relative_tolerance,absolute_tolerance=absolute_tolerance,
            tail_absolute_tolerance=tail_absolute_tolerance)

    def refine_harmonic_power(self, control_surface, *, surface_orders=(2,4,8), volume_orders=(2,4,8),
                              relative_tolerance=1e-4, absolute_tolerance=1e-20, query_config=None,
                              query_batch_size=None, progress_callback=None):
        """Refine power quadrature; optional callbacks receive detached CPU step reports."""
        if self._harmonic_data is None:
            raise ValueError("This Result has no coupled harmonic power balance")
        from .mom.harmonic_observables import harmonic_power_refinement
        return harmonic_power_refinement(self._harmonic_data,control_surface,surface_orders=surface_orders,
            volume_orders=volume_orders,relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,query_config=query_config,
            query_batch_size=query_batch_size,progress_callback=progress_callback)

    def harmonic_power_balance(self, control_surface, *, surface_order=3, volume_order=3, query_config=None,query_batch_size=None):
        """Integrate exterior flux and regional work for a coupled harmonic Result."""
        if self._harmonic_data is None:
            raise ValueError("This Result has no coupled harmonic power balance")
        from .mom.harmonic_observables import harmonic_power_balance
        return harmonic_power_balance(self._harmonic_data,control_surface,surface_order=surface_order,
                                      volume_order=volume_order,query_config=query_config,query_batch_size=query_batch_size)

    def query_dc(self, points, *, region_id):
        """Return the DC scalar potential part [V] and full real electric field [V/m]."""
        if self._harmonic_data is None:
            raise ValueError("This Result has no harmonic DC field")
        return self._harmonic_data.query_dc(points,region_id)

    def query_dc_magnetic(self, points, *, region_id):
        """Return recovered static H [A/m] and B [T] at physical points."""
        if self._harmonic_data is None:
            raise ValueError("This Result has no harmonic DC magnetic field")
        return self._harmonic_data.query_dc_magnetic(points,region_id)

    def query_dc_dual(self, points, *, region_id):
        """Return coupled DC E/H, scalar potential parts and magnetic flux data.

        Conductive interiors report the residue lim(omega*B)=i*sigma_m*H
        in T/s and no finite B value. Vacuum exterior B remains finite.
        """
        if self._harmonic_data is None:
            raise ValueError('This Result has no coupled harmonic DC state')
        return self._harmonic_data.query_dc_dual(points,region_id)

    def query_fields(self, points, *, region_id, field_kind, trace_side=None, reference=None, query_config=None,
                     harmonic=None, coordinates="world", components=None, frequencies=None,
                     owner=None, representation=None):
        """Evaluate complex E/H at physical points in metres.

        MoM decodes native currents. FDTD interpolates recorded spectral Yee
        values; all interpolation stencil samples must be retained as full
        fields or PointMonitors. FDTD incident/scattered queries require an
        explicit matched empty-background Result. Its running-DFT amplitude
        and phase are preserved, including the observer H half-step correction.
        CopperFoil interface points require trace_side='negative' or 'positive',
        relative to increasing chart/native normal coordinate. Tangential E uses
        the actual saved side spectrum. Near the sheet H_t and E_n use the two
        nearest native half nodes on that side, with linear extrapolation to the
        interface; this is a convergent reconstruction, not an exact charge jump.

        Explicit components=("Ey",) activates the selected-component contract:
        a dict of complex tensors [selected frequencies, points], including
        singleton axes. Selected geometry is fixed (points.requires_grad=False);
        field/head gradients remain live. frequencies selects recorded values in output
        order. This mode uses float64 geometry, raw source normalization and
        the none window, with full fields, Points or exact native Planes.
        Copper Et requires explicit owner, trace_side and actual saved sampling
        provenance from a real time-domain field bank. The spectra remain
        complex and mean E(t)=Re(a*exp(-i*omega*t)). The DFT accumulator uses
        exp(+i*omega*n*dt), which is a separate sign convention. Complex
        time-domain banks require a separate clock contract.
        representation="constitutive" uses the actual side head;
        "published_native" selects only the negative native primitive. Native
        C64 values may be promoted by float64 interpolation weights. Default
        components=None preserves the existing six-field/single-frequency tuple.
        """
        self._require_absolute_spectra()
        if components is None and any(value is not None for value in (frequencies, owner, representation)):
            raise ValueError("frequencies/owner/representation selectors require explicit components.")
        if components is not None and self.method != "fdtd":
            raise ValueError("components selects FDTD native fields; MoM uses its existing E/H query result.")
        if harmonic is not None and self._harmonic_data is None:
            raise ValueError("harmonic selects fields only on a coupled harmonic Result")
        if coordinates not in ("world","chart"):
            raise ValueError("Field query positions use coordinates='world' or 'chart'.")
        if self.method == "mom":
            if coordinates!="world":
                raise ValueError("MoM physical queries require world Cartesian coordinates.")
            if reference is not None:
                raise ValueError("MoM incident fields are defined by its source; reference is only for FDTD")
            if self._harmonic_data is not None:
                return self._harmonic_data.query_fields(points,region_id,field_kind,trace_side,
                    harmonic=harmonic,query_config=query_config)
            data = self.surface_currents
            return data.query_fields(points, region_id, field_kind, trace_side, query_config=query_config)
        if self.method != "fdtd":
            raise NotImplementedError("Physical E/H queries currently support MoM and FDTD Results")
        if query_config is not None:
            raise ValueError("query_config controls MoM quadrature and does not apply to FDTD interpolation")
        if trace_side not in (None, "negative", "positive"):
            raise ValueError("FDTD trace_side must be 'negative', 'positive', or None.")
        if region_id not in (0, 1, "exterior", "interior") or isinstance(region_id, bool):
            raise ValueError("region_id must identify exterior or interior")
        if field_kind not in ("total", "incident", "scattered"):
            raise ValueError("field_kind must be total, incident, or scattered")
        if region_id in (1, "interior") and field_kind != "total":
            raise ValueError("Interior field queries require field_kind='total'")
        if components is not None and field_kind in ("incident", "scattered"):
            raise NotImplementedError("Selected native incident or scattered fields need qualified matched-reference frequency and sampling records.")
        if components is None and len(self.frequencies) != 1:
            raise NotImplementedError("FDTD physical queries currently require one recorded frequency")
        if not isinstance(points, torch.Tensor) or points.device.type != "cuda" or points.dtype not in (torch.float32, torch.float64) or points.ndim != 2 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
            raise ValueError("Physical queries require finite CUDA real points of shape [N, 3]")
        input_points=points
        chart=getattr(self.scene,"coordinates",None)
        point_roundoff=0.
        if chart is not None and coordinates=="world":
            points=chart.inverse(points,bounds=self.scene.domain.bounds)
            scale=max(float(input_points.detach().abs().max()),
                      max(abs(value) for value in chart.origin),
                      max(abs(value) for bound in self.scene.domain.bounds for value in bound))
            point_roundoff=16*torch.finfo(input_points.dtype).eps*scale
            spacing=min(float(np.min(np.diff(getattr(self.prepared_scene,axis+"_nodes64")))) for axis in "xyz")
            if point_roundoff > spacing/32:
                raise ValueError("World-coordinate precision cannot resolve the native query grid; use float64 points or coordinates='chart'.")
        inside = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        for structure in self.scene.structures:
            if structure.enabled:
                from .geometry.surface_mesh import SurfaceMesh
                if isinstance(structure.geometry, SurfaceMesh):
                    interior, contact = structure.geometry.classify_query_points(points)
                    if bool(contact.any()):
                        raise ValueError("FDTD interpolation does not define a one-sided field on a SurfaceMesh triangle.")
                    inside |= interior
                else:
                    inside |= structure.geometry.signed_distance(*points.unbind(-1)) < 0
        if bool((inside != (region_id in (1, "interior"))).any()):
            raise ValueError("Query geometry does not match region_id")
        if field_kind in ("incident", "scattered"):
            _validate_fdtd_query_reference(self, reference)
            incident = reference.query_fields(input_points, region_id="exterior", field_kind="total",coordinates=coordinates)
            if field_kind == "incident":
                return incident
        if components is not None:
            from .postprocess.selected_query import query_selected
            return query_selected(self, points, components=components, frequencies=frequencies,
                trace_side=trace_side, owner=owner,
                representation="constitutive" if representation is None else representation,
                point_roundoff=point_roundoff)
        from .postprocess.yee_interpolation import copper_query_faces
        copper_faces=copper_query_faces(self,device=points.device if points.requires_grad else "cpu")
        values = [_interpolate_fdtd_component(self, points, component, trace_side=trace_side,
                      point_roundoff=point_roundoff, copper_faces=copper_faces)
                  for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
        result = (torch.stack(values[:3], -1), torch.stack(values[3:], -1))
        if chart is not None:
            result=tuple(chart.physical_field(points,value) for value in result)
        if field_kind == "scattered":
            return tuple(value - baseline for value, baseline in zip(result, incident))
        return result

    def far_field(self, directions, *, harmonic=None):
        """MoM complex amplitudes with E,H ~ exp(i k r) F / r."""
        if self._harmonic_data is not None:
            return self._harmonic_data.far_field(directions,harmonic=harmonic)
        if harmonic is not None:
            raise ValueError("harmonic selects far fields only on a coupled harmonic Result")
        data = self.surface_currents
        return data.far_field(directions)

    def rayleigh_orders(self, orders, *, side, plane_offset, quadrature_order=6):
        """2D periodic total plane amplitudes and signed outward order flux."""
        return self.surface_currents.rayleigh_orders(orders, side=side,
            plane_offset=plane_offset, quadrature_order=quadrature_order)

    def cylindrical_orders(self, orders, directions, *, quadrature_order=6):
        """1D periodic scattered cylindrical amplitudes and power per radian."""
        return self.surface_currents.cylindrical_orders(orders, directions,
            quadrature_order=quadrature_order)

    def bloch_spectrum(self, orders, *, quadrature_order=6):
        """3D homogeneous-exterior equivalent-current potential coefficients.

        These coefficients are not body-interior physical fields or incident
        fields. Spectral poles reject ordinary forced resolvent evaluation.
        """
        return self.surface_currents.bloch_spectrum(orders, quadrature_order=quadrature_order)

    def _require_volume_fields(self):
        if self.method == "mom":
            raise NotImplementedError("MoM Results contain RWG currents; use query_fields for E/H instead of Yee field access")

    def _require_frequency_volume_fields(self):
        self._require_absolute_spectra()
        if (self.method == "fdtd" and len(self.frequencies) > 1
                and self._solver_stats.get("full_field_dft") is False
                and self._solver_stats.get("full_field_dft_source") == "explicit"):
            raise RuntimeError(
                "This FDTD run did not record frequency-domain volume fields "
                "(full_field_dft=False); pass full_field_dft=True."
            )

    @property
    def prepared_scene(self):
        self._require_volume_fields()
        if self._prepared_scene is None:
            dtype = torch.float64 if self._solver_stats.get("field_dtype") == "torch.float64" else torch.float32
            self._prepared_scene = prepare_scene(self.scene, material_dtype=dtype)
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
        self._require_frequency_volume_fields()
        if getattr(self.scene,"coordinates",None) is not None:
            raise ValueError("Mapped arrays require the explicit result.chart_fields frame.")
        self._require_volume_fields()
        return dict(self._fields)

    @property
    def monitors(self) -> dict[str, Any]:
        for name in self._monitors:
            self._require_monitor_spectra(name)
        if getattr(self.scene,"coordinates",None) is not None:
            return {name:self.raw_monitor(name) for name in self._monitors}
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

    def _esd_generator_circuits(self):
        scene = self.scene
        circuits = getattr(scene, "circuits", ()) if scene is not None else ()
        return {
            circuit.name: dict(circuit.metadata["esd_generator"])
            for circuit in circuits
            if isinstance(getattr(circuit, "metadata", None), dict)
            and "esd_generator" in circuit.metadata
        }

    def esd_generator_names(self) -> tuple[str, ...]:
        """Names of circuit-driven ESD generator networks on the run scene.

        These are the circuits assembled by :meth:`ESDVoltageSource.build_circuit`
        (the source-impedance-network ESD excitation), identified by the
        generator provenance stamped on their circuit metadata.
        """

        return tuple(self._esd_generator_circuits())

    def esd_generator(self, name: str) -> dict[str, Any]:
        """Return the ESD generator-network provenance for circuit ``name``.

        Capability level: stress-only. The provenance carries the versioned
        waveform metadata (standard, revision, level voltage, model version) and
        the source-network element values (330 ohm discharge resistor / 150 pF
        storage capacitor by default), documenting that this is a circuit
        approximation of the standard network rather than gun/system
        certification. The coupled port voltage/current time series lives on the
        associated :class:`CircuitData` (``result.circuit(name)``).
        """

        generators = self._esd_generator_circuits()
        if name not in generators:
            available = ", ".join(sorted(generators)) or "<none>"
            raise KeyError(
                f"ESD generator {name!r} is not present; available: {available}."
            )
        return generators[name]

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
    def ports(self) -> dict[str, PortData | _RatioOnlyPortData]:
        return {name: self._public_port(port) for name, port in self._ports.items()}

    @property
    def circuits(self) -> dict[str, CircuitData]:
        return dict(self._circuits)

    @property
    def network(self) -> NetworkData | None:
        return self._network

    @property
    def embedded_networks(self) -> dict[str, EmbeddedNetworkData]:
        self._require_absolute_spectra()
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
    def electrostatic_prebias(self):
        """Provenance of the electrostatic pre-bias initial condition, or ``None``.

        Populated when the FDTD run was seeded with an
        :class:`~witwin.maxwell.ElectrostaticInitialCondition` (see
        ``Simulation.fdtd(initial_condition=...)``). Carries the DC solve
        diagnostics, the mapped-field discrete-Gauss residual, and the tolerance."""
        metadata = self._metadata or {}
        return metadata.get("electrostatic_prebias")

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
        self._require_frequency_volume_fields()
        self._require_volume_fields()
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
        if getattr(self.scene,"coordinates",None) is not None:
            raise ValueError("Mapped native arrays are covariant chart fields; use result.chart_fields.tensor('E_s') or world query_fields().")
        return self._native_tensor(name,expand_symmetry=expand_symmetry,frequency=frequency,freq_index=freq_index)

    @property
    def chart_fields(self):
        from .coordinates import FramedFieldData
        if getattr(self.scene,"coordinates",None) is None:
            raise ValueError("chart_fields requires an explicit Scene coordinate chart.")
        return FramedFieldData(self)

    def _native_tensor(self,name,*,expand_symmetry=False,frequency=None,freq_index=None):
        self._require_frequency_volume_fields()
        self._require_volume_fields()
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
        payload=self._native_raw_monitor(name,frequency=frequency,freq_index=freq_index)
        if getattr(self.scene,"coordinates",None) is not None and isinstance(payload,dict):
            from .coordinates import _frame_monitor_payload
            return _frame_monitor_payload(payload,self.chart_fields.frame)
        return payload

    def _native_raw_monitor(self, name: str, *, frequency: float | None = None, freq_index: int | None = None):
        self._require_monitor_spectra(name)
        public_monitor = _find_scene_monitor(self.scene, name)
        if isinstance(public_monitor, (ClosedSurfaceMonitor, *PROJECTION_MONITOR_TYPES)):
            # Projection monitors expose their recorded Huygens faces as one
            # closed-surface payload; monitor() projects it.
            # Keep selected-frequency reductions on their original scalar path;
            # the full-spectrum payload is shared by antenna frequency slices.
            if frequency is None and freq_index is None:
                if name not in self._closed_surface_payload_cache:
                    self._closed_surface_payload_cache[name] = _copy_monitor_containers(
                        _build_closed_surface_payload(self, public_monitor, frequency=None, freq_index=None)
                    )
                return _copy_monitor_containers(self._closed_surface_payload_cache[name])
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

        if getattr(self.scene, "coordinates", None) is not None:
            selected["native_product_disabled"] = True
        resolved_monitor = _find_resolved_scene_monitor(self.scene, name)
        if isinstance(resolved_monitor, FinitePlaneMonitor):
            bounds = None
            if resolved_monitor.closure_planes is not None and self.method != "mom":
                from .fdtd.observers import finite_plane_integration_bounds
                bounds = finite_plane_integration_bounds(self.prepared_scene, resolved_monitor)
            selected = _crop_plane_monitor_payload(selected, resolved_monitor, bounds)
        from .fdtd.plane_product import enrich_plane_flux
        return enrich_plane_flux(self, selected)

    def monitor(
        self,
        name: str,
        *,
        frequency: float | None = None,
        freq_index: int | None = None,
        resolve_modal: bool = True,
    ):
        self._require_monitor_spectra(name)
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
        if isinstance(public_monitor, PROJECTION_MONITOR_TYPES) and resolve_modal:
            from .postprocess.field_projection import project_monitor

            return project_monitor(self, public_monitor, frequency=frequency, freq_index=freq_index)
        if isinstance(public_monitor, GAUSSIAN_OVERLAP_MONITOR_TYPES) and resolve_modal:
            from .postprocess.gaussian_overlap import compute_gaussian_overlap

            return compute_gaussian_overlap(self, public_monitor, frequency=frequency, freq_index=freq_index)
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

    def port(self, name: str) -> PortData | _RatioOnlyPortData:
        port_name = str(name)
        if port_name not in self._ports:
            choices = tuple(self._ports)
            raise KeyError(
                f"Port {port_name!r} is not available in this result. "
                f"Choices: {choices}."
            )
        return self._public_port(self._ports[port_name])

    def port_drive(self, name: str) -> PortDrive:
        """Return the prescribed, run-normalized drive in this port's bin order.

        Raises ValueError for a port with no recorded excitation. The spectrum
        uses the same per-bin reference period and source normalization as V/I.
        """
        self._require_absolute_spectra()
        self.port(name)
        drives = self._metadata.get("port_drives", {})
        if name not in drives:
            raise ValueError(f"Port {name!r} has no recorded excitation.")
        return PortDrive(**drives[name])

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
        self._require_absolute_spectra()
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

        self._require_absolute_spectra()
        if self._solver_stats.get("field_dtype") == "torch.float64":
            raise ValueError("FP64 native forward does not yet retain frequency-domain volume fields for PowerLossMonitor; use its qualified spectral surface outputs.")
        monitor = _find_scene_monitor(self.scene, name)
        if not isinstance(monitor, PowerLossMonitor):
            raise KeyError(f"PowerLossMonitor {name!r} is not declared in this Scene.")
        if getattr(self.scene,"coordinates",None) is not None and any(value is not None for value in
                (electric_fields,volume_channels,integrated_channels,surface_channels,face_areas,line_channels,line_lengths)):
            raise NotImplementedError("Mapped PowerLossMonitor currently requires automatic native conduction fields; explicit density inputs need an explicit frame.")
        if self.method != "fdtd":
            raise NotImplementedError(
                "Automatic PowerLossMonitor field colocation currently supports FDTD results only."
            )

        from .compiler.power_loss import compile_power_loss_monitor
        from .postprocess.power_loss import compute_power_loss_data

        compiled = compile_power_loss_monitor(self.prepared_scene, monitor)
        loss_materials = (self.raw_output or {}).get("loss_materials", {})
        if compiled.conductivity is not None and loss_materials:
            # Static loss prefactors participate in the same custom VJP as the
            # fields, including the direct derivative of sigma times |E|^2.
            from dataclasses import replace
            compiled = replace(compiled, conductivity={component:values[compiled.component_masks[component]]
                for component,values in loss_materials.items()})
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
        automatic_dispersion = ("electric_dispersion" in monitor.channels
            and "electric_dispersion" not in (volume_channels or {})
            and "electric_dispersion" not in (integrated_channels or {}))
        if resolved_fields is None and ("conduction" in monitor.channels or automatic_dispersion):
            missing = tuple(
                component
                for component in ("Ex", "Ey", "Ez")
                if component.upper() not in self._fields
            )
            if missing:
                raise RuntimeError(
                    "Automatic electric PowerLossMonitor channels require frequency-domain Ex/Ey/Ez fields; "
                    "run FDTD with full_field_dft=True or pass electric_fields=... explicitly."
                )
            resolved_fields = {
                component: torch.stack(
                    [
                        self._native_tensor(component, frequency=frequency)
                        for frequency in selected_frequencies
                    ],
                    dim=0,
                )
                for component in ("Ex", "Ey", "Ez")
            }
        if automatic_dispersion:
            from .postprocess.power_loss import _validate_electric_fields
            resolved_fields = _validate_electric_fields(resolved_fields,
                frequency_count=len(selected_frequencies), compiled=compiled)
        records = self.conductor_losses
        if resolved_fields is not None and not any(record.get("bulk_electric_traces") for record in records.values()):
            from .compiler.copper_foil import compile_copper_faces
            for face in compile_copper_faces(self.prepared_scene, device=compiled.device):
                component = "E" + "xyz"[face.component]
                if bool(compiled.component_masks[component][tuple(face.indices.T)].any()):
                    raise NotImplementedError("Split-foil bulk loss requires retained side electric spectra, which are unavailable for this result.")
        from .postprocess.fdtd_loss import automatic_electric_dispersion_density, apply_split_trace_densities
        if automatic_dispersion:
            dt = self.stats().get("dt")
            if dt is None:
                raise RuntimeError("Automatic electric-dispersion loss requires the recorded FDTD time step.")
            volume_channels = dict(volume_channels or {})
            volume_channels["electric_dispersion"] = automatic_electric_dispersion_density(
                self.prepared_scene, compiled, frequency_tensor, resolved_fields, dt)
        data = compute_power_loss_data(
            compiled,
            frequency_tensor,
            electric_fields=resolved_fields if "conduction" in monitor.channels else None,
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
        if resolved_fields is not None:
            data = apply_split_trace_densities(data, compiled, resolved_fields, records,
                                              correct_dispersion=automatic_dispersion)
        chart=getattr(self.scene,"coordinates",None)
        if chart is not None:
            from dataclasses import replace
            from .fdtd.coords import component_coords
            metric={}
            for component,mask in compiled.component_masks.items():
                u=torch.as_tensor(component_coords(self.prepared_scene,component)[1],device=mask.device,dtype=torch.float64)
                metric[component]=(1-chart.curvature*u)[None,:,None].expand(mask.shape)[mask]
            data=replace(data,
                component_volumes={key:value*metric[key].to(value) for key,value in data.component_volumes.items()},
                volume_density={channel:{key:value/metric[key].to(value) for key,value in densities.items()}
                                for channel,densities in data.volume_density.items()},
                coordinate_frame=self.chart_fields.frame.snapshot(),
                collocation="Covariant s/u/z Yee samples; metric-weighted physical dual volumes; selector bounds in chart metres")
        return data

    def incident_power_density(
        self,
        monitor: str,
        *,
        spatial_average=_UNSET,
    ):
        """Reduce a declared IncidentPowerDensityMonitor to incident power density.

        Pure result-domain reduction: forms the time-averaged normal Poynting
        component ``S.n`` per cell, its magnitude ``|S.n|`` (exposure incident
        power density in W/m^2), the plane-integrated flux, and — when the monitor
        (or an explicit ``spatial_average`` override) requests a window area — the
        versioned ``spatial-average-v1`` moving-window average of ``|S.n|``. Fails
        closed when the named monitor is not an
        :class:`IncidentPowerDensityMonitor`.
        """

        from .postprocess.incident_power import compute_incident_power_density

        public_monitor = _find_scene_monitor(self.scene, monitor)
        if not isinstance(public_monitor, IncidentPowerDensityMonitor):
            raise KeyError(
                "Result.incident_power_density requires a declared "
                f"IncidentPowerDensityMonitor; {monitor!r} is not one."
            )
        if spatial_average is _UNSET:
            area = public_monitor.spatial_average
        elif spatial_average is None:
            area = None
        else:
            area = float(spatial_average)
        payload = self.raw_monitor(monitor)
        return compute_incident_power_density(
            payload,
            monitor_name=monitor,
            spatial_average_area=area,
        )

    def sar(
        self,
        monitor: str,
        *,
        averaging=None,
        normalization=None,
        electric_fields=None,
        volume_channels=None,
    ):
        """Reduce a PowerLossMonitor and the tissue mass model to point SAR.

        Pure result-domain reduction: it never runs a solver. It fails explicitly
        when the named ``PowerLossMonitor``, the frequency-domain fields, or the
        tissue mass densities required to form SAR are missing. ``averaging`` (a
        :class:`~witwin.maxwell.SARAveraging`) is recorded in the result
        provenance; the mass-averaging computation is delivered by a later stage.
        ``normalization`` defaults to unit source amplitude.
        """

        from .compiler.mass_density import compile_mass_density
        from .compiler.power_loss import compile_power_loss_monitor
        from .postprocess.sar import compute_sar
        from .sar import PowerNormalization

        public_monitor = _find_scene_monitor(self.scene, monitor)
        if not isinstance(public_monitor, PowerLossMonitor):
            raise KeyError(
                f"Result.sar requires a declared PowerLossMonitor; {monitor!r} is not one."
            )
        if self.method != "fdtd":
            raise NotImplementedError(
                "Result.sar currently consumes FDTD PowerLossData only."
            )
        resolved_normalization = (
            PowerNormalization.none() if normalization is None else normalization
        )

        power_loss = self.power_loss(
            monitor,
            electric_fields=electric_fields,
            volume_channels=volume_channels,
        )
        prepared = self.prepared_scene
        mass = compile_mass_density(prepared)
        compiled_loss = compile_power_loss_monitor(prepared, public_monitor)
        power_scale = resolved_normalization.resolve_scale(
            result=self, frequencies=power_loss.frequencies
        )
        return compute_sar(
            prepared_scene=prepared,
            monitor=public_monitor,
            power_loss=power_loss,
            mass=mass,
            compiled_loss=compiled_loss,
            normalization=resolved_normalization,
            averaging=averaging,
            power_scale=power_scale,
        )

    def material(self, name: str = "eps_r", *, expand_symmetry: bool = False) -> torch.Tensor:
        if getattr(self.scene,"coordinates",None) is not None:
            raise ValueError("Mapped constitutive arrays have a chart tensor frame; unframed scalar material views require a qualified physical material query.")
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

        self._require_absolute_spectra()
        return {
            "schema_version": RESULT_SNAPSHOT_SCHEMA_VERSION,
            "data_type": "ResultSnapshot",
            "format_version": 1,
            "method": self.method,
            "mixed_absolute_spectra": self._mixed_absolute_spectra,
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

        payload = self._spectral_ratio_view()._snapshot_payload(fields=self._fields)
        if self._surface_data is not None:
            payload["surface_data"] = self._surface_data.snapshot()
            if payload["surface_data"].get("schema_version") == 20:
                # Field queries can update native diagnostic metadata after the
                # solve. Persist one validated surface/volume representation.
                payload["metadata"] = dict(payload["metadata"],
                    **payload["surface_data"]["representation"])
        if self._harmonic_data is not None:
            payload["harmonic_data"] = self._harmonic_data.snapshot()
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

        payload = self._spectral_ratio_view()._snapshot_payload(fields={})
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
        path: str | Path | BytesIO,
        *,
        scene,
        prepared_scene=None,
        map_location: Any = "cpu",
    ) -> "Result":
        """Load detached inference data saved by :meth:`save`.

        A BytesIO stream allows restoration from caller-verified snapshot bytes
        without reopening a mutable filesystem path.

        The declarative scene is supplied by the caller because runtime solvers,
        prepared scenes, and transport state are intentionally not persisted.
        Result files use pickle-backed ``torch.save`` and must therefore only be
        loaded from trusted sources.
        """

        payload = torch.load(
            path if isinstance(path,BytesIO) else Path(path),
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise ValueError("Result checkpoint must contain a mapping payload.")
        _validate_result_snapshot_payload(payload, sharded=False)
        chart=getattr(scene,"coordinates",None)
        field_frame=payload.get("metadata",{}).get("field_frame")
        if chart is not None:
            from .coordinates import FieldFrame
            if field_frame!=FieldFrame("covariant_chart",chart).snapshot():
                raise ValueError("Mapped Result snapshot lacks the matching coordinate field frame.")
        elif field_frame is not None and field_frame.get("kind")=="covariant_chart":
            raise ValueError("Mapped Result snapshot requires its coordinate Scene.")
        missing = {"method", "fields", "monitors"}.difference(payload)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Result checkpoint is missing required keys: {names}.")

        surface_data,harmonic_data = None,None
        if payload["method"] == "mom":
            if ("surface_data" in payload) == ("harmonic_data" in payload):
                raise ValueError("MoM snapshot requires exactly one native surface or harmonic payload")
            if "harmonic_data" in payload:
                from .mom.harmonic_snapshot import restore
                harmonic_data = restore(payload["harmonic_data"],scene)
                if payload['harmonic_data'].get('schema_version') in (21,22,23):
                    from .mom.tensor_harmonic_runtime import validate_result_metadata
                    validate_result_metadata(harmonic_data,payload.get('metadata'),payload.get('solver_stats'))
            else:
                from .mom.runtime import SurfaceData
                surface_data = SurfaceData.restore(payload["surface_data"], scene, payload["frequency"])
                if payload['surface_data'].get('schema_version') == 20:
                    from .mom.runtime import _same_metadata
                    expected = surface_data.representation_metadata()
                    metadata = payload.get('metadata')
                    if not isinstance(metadata, dict) or any(
                            not _same_metadata(metadata.get(k), v) for k, v in expected.items()):
                        raise ValueError('Linear coupled Result metadata differs from its native identity')
                elif (surface_data.config.periodic is not None or surface_data.config.dyadic_integration is not None
                        or surface_data.config.operator_backend in ('hmatrix', 'h2', 'direct_h2', 'fmm')):
                    from .mom.runtime import _same_metadata, _representation_identity
                    metadata = payload.get('metadata')
                    expected = surface_data.handle.metadata
                    keys = (*_representation_identity(surface_data.config), 'operator_identity',
                            'rhs_identity', 'kernel_version', 'physical_qualification')
                    if not isinstance(metadata, dict) or any(
                            not _same_metadata(metadata.get(key), expected.get(key)) for key in keys):
                        raise ValueError('MoM Result metadata does not match its native solution identity')
        result = cls(
            surface_data=surface_data,harmonic_data=harmonic_data,
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
        result._mixed_absolute_spectra |= payload.get("mixed_absolute_spectra", False)
        return result

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
        result._mixed_absolute_spectra |= payload.get("mixed_absolute_spectra", False)
        result._sharded_manifest = loaded.manifest
        result._shard_paths = loaded.shard_paths
        return result


def _validate_fdtd_query_reference(result, reference):
    if not isinstance(reference, Result) or reference.method != 'fdtd':
        raise ValueError('FDTD incident/scattered queries require a matched empty-background reference Result')
    left,right=getattr(result.scene,"coordinates",None),getattr(reference.scene,"coordinates",None)
    if (None if left is None else left.snapshot())!=(None if right is None else right.snapshot()):
        raise ValueError("FDTD reference must share the same physical coordinate chart.")
    if reference.frequencies != result.frequencies or any(s.enabled for s in reference.scene.structures) or reference.scene.material_regions:
        raise ValueError('FDTD reference must be empty background at the same frequency')
    for name in ('thin_wires', 'ports', 'networks', 'lumped_elements', 'circuits', 'electrostatic_terminals', 'charge_densities'):
        if getattr(reference.scene, name, ()):
            raise ValueError(f'FDTD reference must be empty background without Scene.{name}')
    required = {'normalize_source', 'dt', 'time_steps', 'steps_run', 'dft_window', 'observer_sample_counts'}
    if not required.issubset(result._solver_stats) or not required.issubset(reference._solver_stats):
        raise ValueError('FDTD reference subtraction requires explicit recording and normalization provenance')
    for name in ('domain', 'grid', 'boundary', 'sources', 'symmetry'):
        if repr(getattr(reference.scene, name)) != repr(getattr(result.scene, name)):
            raise ValueError(f'FDTD reference must share Scene.{name}')
    for key in ('dt', 'time_steps', 'steps_run', 'time_plan', 'dft_window', 'normalize_source', 'reference_frequency',
                'dft_sample_counts', 'observer_sample_counts', 'full_field_dft'):
        if reference._solver_stats.get(key) != result._solver_stats.get(key):
            raise ValueError(f'FDTD reference must share {key}')


def _interpolate_fdtd_component(result, points, component, *, trace_side=None, point_roundoff=0., copper_faces=None,
                                selected_frequency_index=None, selected_representation=None):
    """Trilinear interpolation on each component's actual Yee coordinates."""
    from .fdtd.coords import component_coords
    axes = [torch.as_tensor(axis, device=points.device, dtype=points.dtype)
            for axis in component_coords(result.prepared_scene, component)]
    if any(len(axis) < 2 for axis in axes):
        raise ValueError('Physical field interpolation requires at least two nodes per axis')
    from .postprocess.yee_interpolation import interpolation_stencil, normalized_trace_values
    stencil = interpolation_stencil(result, points, component, axes, trace_side=trace_side, point_roundoff=point_roundoff, faces=copper_faces)
    if selected_frequency_index is not None:
        from .postprocess.selected_query import native_sampler
        selected_sampler = []
        def sample_native(indices, *, required=True):
            if not selected_sampler:
                selected_sampler.append(native_sampler(result, points, component, axes, selected_frequency_index))
            return selected_sampler[0](indices, required=required)
    else:
        dense = result._fields.get(component.upper())
        if dense is not None and not dense.is_complex():
            dense = None
        if dense is not None and dense.shape != tuple(len(axis) for axis in axes):
            raise ValueError('Stored spectral field shape does not match its Yee coordinates')
        samples = {}
        if dense is None:
            for monitor in result._monitors.values():
                if not isinstance(monitor, dict) or monitor.get('kind') != 'point':
                    continue
                index = monitor.get('field_indices', {}).get(component)
                value = monitor.get('components', {}).get(component)
                if index is not None and value is not None:
                    flat = (int(index[0]) * len(axes[1]) + int(index[1])) * len(axes[2]) + int(index[2])
                    samples.setdefault(flat, value)
            ids = sorted(samples)
            sample_ids = torch.tensor(ids, device=points.device, dtype=torch.long)
            sample_values = (torch.stack([torch.as_tensor(samples[i], device=points.device).reshape(()) for i in ids])
                             if ids else torch.empty(0, device=points.device, dtype=torch.complex128))
            if not sample_values.is_complex():
                raise ValueError('FDTD queries require complex spectral samples, not instantaneous fields')
        def sample_native(indices, *, required=True):
            indices=indices.to(device=points.device)
            if dense is not None:
                return dense[tuple(indices.T)], torch.ones(len(indices), device=points.device, dtype=torch.bool)
            if not len(sample_ids):
                if required:
                    raise ValueError(f'No spectral {component} samples are retained; record its PointMonitor interpolation stencil')
                return torch.zeros(len(indices), device=points.device, dtype=torch.complex128), torch.zeros(len(indices), device=points.device, dtype=torch.bool)
            flat = (indices[:, 0]*len(axes[1])+indices[:, 1])*len(axes[2])+indices[:, 2]
            position = torch.searchsorted(sample_ids, flat).clamp_max(len(sample_ids)-1)
            present = sample_ids[position] == flat
            if required and not bool(present.all()):
                raise ValueError(f'Missing {component} interpolation stencil samples; prepare PointMonitors at surrounding Yee nodes')
            return sample_values[position], present
    trace_values = {}
    values = torch.zeros(len(points), device=points.device, dtype=torch.complex128 if points.dtype == torch.float64 else torch.complex64)
    for indices, weight, overrides in stencil:
        index = tuple(indices.T)
        # A TFSF runtime stores scattered fields outside its injection box.
        # Do not interpolate across that representation discontinuity or label
        # those samples as a physical total field.
        from .sources import TFSF
        for source in result.scene.sources:
            injection = getattr(source, 'injection', None)
            if isinstance(injection, TFSF):
                bounds_by_axis = injection.bounds
                if bounds_by_axis is None:
                    bounds_by_axis = tuple(injection.axis_bounds if axis == injection.axis else None for axis in 'xyz')
                for dim, bounds in enumerate(bounds_by_axis):
                    if bounds is not None:
                        coordinate = axes[dim][index[dim]]
                        outside = (coordinate <= bounds[0]) | (coordinate >= bounds[1])
                        if bool((outside & (weight != 0)).any()):
                            raise NotImplementedError('FDTD query stencils must lie strictly inside the TFSF total-field box; exterior reconstruction is not implemented')
        replaced = torch.zeros(len(points), device=points.device, dtype=torch.bool)
        for selected, _, _ in overrides:
            replaced = replaced | selected
        need = (weight != 0) & ~replaced
        sampled = torch.zeros(len(points), device=points.device, dtype=values.dtype)
        if bool(need.any()):
            sampled[need] = sample_native(indices[need])[0].to(values.dtype)
        for selected, positive, face in overrides:
            if selected_frequency_index is not None:
                selected = selected & (weight != 0)
                if not bool(selected.any()):
                    continue
                if selected_representation == 'published_native':
                    # Gather only the requested primitive; no fictitious positive
                    # published bank or entire-plane requirement is introduced.
                    sampled = sampled.clone()
                    sampled[selected] = sample_native(indices[selected])[0].to(sampled.dtype)
                    continue
            key = id(face)
            if key not in trace_values:
                if selected_frequency_index is not None:
                    # Sampling provenance was validated before this gather.
                    # The actual side heads are already in the recorded units.
                    trace_values[key] = face['electric_fields'].to(device=points.device)
                else:
                    trace_values[key] = normalized_trace_values(face, sample_native,
                        normalized=bool(result.stats().get('normalize_source', False)))
            support = face['indices'].to(device=points.device)
            shape = tuple(len(axis) for axis in axes)
            def flatten(location):
                return (location[:, 0]*shape[1]+location[:, 1])*shape[2]+location[:, 2]
            ordered, order = torch.sort(flatten(support))
            position = torch.searchsorted(ordered, flatten(indices[selected]))
            offset = order[position]
            selected_values = trace_values[key][positive[selected].long(), offset]
            sampled = sampled.clone()
            sampled[selected] = selected_values.to(sampled.dtype)
        values = values + weight * sampled
    return values
