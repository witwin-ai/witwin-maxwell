from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Mapping

import torch

from ..array import _validate_angular_grid
from ..monitors import ClosedSurfaceMonitor


@dataclass(frozen=True)
class CompiledArrayBasisRequest:
    """Frozen closed-surface and angular contract for an array basis sweep."""

    monitor_name: str
    monitor_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    monitor_faces: tuple[tuple[str, str, float, tuple[float, float, float]], ...]
    frequencies: tuple[float, ...]
    theta: torch.Tensor
    phi: torch.Tensor
    phase_center: torch.Tensor
    frame: torch.Tensor
    phase_center_source: str
    port_names: tuple[str, ...]
    physical_port_names: tuple[str, ...]
    run_manifest_metadata: Mapping[str, object]
    run_manifest_fingerprint: str


def validate_array_superposition(scene) -> None:
    """Reject scene objects that invalidate linear time-invariant superposition."""

    violations: list[str] = []
    resolved_sources = (
        scene.resolved_sources() if hasattr(scene, "resolved_sources") else scene.sources
    )
    for index, source in enumerate(resolved_sources):
        label = getattr(source, "name", None) or f"sources[{index}]"
        violations.append(f"{label} (independent field source)")
    for index, structure in enumerate(scene.structures):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        label = getattr(structure, "name", None) or f"structures[{index}]"
        if bool(getattr(material, "is_nonlinear", False)):
            violations.append(f"{label} (nonlinear material)")
        if bool(getattr(material, "is_modulated", False)):
            violations.append(f"{label} (time-modulated material)")
    if violations:
        joined = ", ".join(violations)
        raise ValueError(
            "Array basis superposition requires a linear, time-invariant Scene; "
            f"incompatible objects: {joined}."
        )


def _resolve_monitor(scene, monitor: str | ClosedSurfaceMonitor) -> ClosedSurfaceMonitor:
    if isinstance(monitor, ClosedSurfaceMonitor):
        if monitor not in scene.monitors:
            raise ValueError("The requested ClosedSurfaceMonitor is not declared in the Scene.")
        return monitor
    if not isinstance(monitor, str) or not monitor:
        raise TypeError("monitor must be a non-empty name or ClosedSurfaceMonitor.")
    matches = [
        candidate
        for candidate in scene.monitors
        if isinstance(candidate, ClosedSurfaceMonitor) and candidate.name == monitor
    ]
    if not matches:
        raise ValueError(f"Scene has no ClosedSurfaceMonitor named {monitor!r}.")
    if len(matches) != 1:
        raise ValueError(f"Scene contains duplicate ClosedSurfaceMonitor name {monitor!r}.")
    return matches[0]


def _resolve_frequencies(surface: ClosedSurfaceMonitor, frequencies) -> tuple[float, ...]:
    requested = surface.frequencies if frequencies is None else tuple(float(v) for v in frequencies)
    if requested is None or not requested:
        raise ValueError("Array basis compilation requires explicit frequencies.")
    if surface.frequencies is not None and tuple(requested) != tuple(surface.frequencies):
        raise ValueError(
            "Array basis frequencies must exactly match the ClosedSurfaceMonitor frequencies; "
            "frequency interpolation is forbidden."
        )
    value = tuple(float(item) for item in requested)
    if any(not math.isfinite(item) or item <= 0.0 for item in value):
        raise ValueError("Array basis frequencies must be finite and strictly positive.")
    if any(current <= previous for previous, current in zip(value, value[1:])):
        raise ValueError("Array basis frequencies must be strictly increasing.")
    return value


def _resolve_angles(theta, phi, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    theta_value = (
        torch.linspace(0.0, math.pi, 181, device=device, dtype=dtype)
        if theta is None
        else torch.as_tensor(theta, device=device, dtype=dtype)
    )
    phi_value = (
        torch.linspace(0.0, 2.0 * math.pi, 361, device=device, dtype=dtype)
        if phi is None
        else torch.as_tensor(phi, device=device, dtype=dtype)
    )
    if theta_value.ndim == 1 and phi_value.ndim == 1:
        theta_value, phi_value = torch.meshgrid(theta_value, phi_value, indexing="ij")
    return _validate_angular_grid(theta_value, phi_value, device=torch.device(device))


def _geometry_bounds(geometry):
    if geometry is None:
        return None
    bounds = getattr(geometry, "bounds", None)
    if callable(bounds):
        bounds = bounds()
    if bounds is None:
        position = getattr(geometry, "position", None)
        size = getattr(geometry, "size", None)
        if position is not None and size is not None:
            center = torch.as_tensor(position).reshape(-1)
            extent = torch.as_tensor(size).reshape(-1)
            if center.numel() == 3 and extent.numel() == 3:
                return tuple(
                    (float(center[axis] - 0.5 * extent[axis]), float(center[axis] + 0.5 * extent[axis]))
                    for axis in range(3)
                )
        return None
    try:
        resolved = tuple((float(axis[0]), float(axis[1])) for axis in bounds)
    except (TypeError, ValueError, IndexError):
        return None
    return resolved if len(resolved) == 3 else None


def _port_bounds(port):
    position = getattr(port, "position", None)
    size = getattr(port, "size", None)
    if position is not None and size is not None:
        center = torch.as_tensor(position).reshape(-1)
        extent = torch.as_tensor(size).reshape(-1)
        if center.numel() == 3 and extent.numel() == 3:
            return tuple(
                (float(center[axis] - 0.5 * extent[axis]), float(center[axis] + 0.5 * extent[axis]))
                for axis in range(3)
            )
    negative = getattr(port, "negative", None)
    positive = getattr(port, "positive", None)
    if negative is not None and positive is not None:
        return tuple(
            (min(float(negative[axis]), float(positive[axis])), max(float(negative[axis]), float(positive[axis])))
            for axis in range(3)
        )
    return None


def _array_aabb_center(scene, physical_port_names, *, device, dtype) -> torch.Tensor:
    ports_by_name = {port.name: port for port in scene.ports}
    source = [
        bounds
        for name in physical_port_names
        if name in ports_by_name and (bounds := _port_bounds(ports_by_name[name])) is not None
    ]
    if not source:
        raise ValueError(
            "Automatic phase center requires selected port or aperture geometry; "
            "provide phase_center explicitly when that geometry has no finite bounds."
        )
    lower = [min(bounds[axis][0] for bounds in source) for axis in range(3)]
    upper = [max(bounds[axis][1] for bounds in source) for axis in range(3)]
    return torch.tensor(
        [(lo + hi) * 0.5 for lo, hi in zip(lower, upper)],
        device=device,
        dtype=dtype,
    )


def _resolve_frame(frame, *, device, dtype) -> torch.Tensor:
    value = (
        torch.eye(3, device=device, dtype=dtype)
        if frame is None
        else torch.as_tensor(frame, device=device, dtype=dtype)
    )
    if value.shape != (3, 3):
        raise ValueError("frame must have shape [3, 3].")
    identity = torch.eye(3, device=device, dtype=dtype)
    tolerance = 256.0 * torch.finfo(dtype).eps
    if not torch.allclose(value.transpose(0, 1) @ value, identity, atol=tolerance, rtol=tolerance):
        raise ValueError("frame columns must be orthonormal.")
    if not bool(torch.linalg.det(value) > 0.0):
        raise ValueError("frame must be right-handed.")
    return value


def _manifest_contract(scene, run_manifest, ports):
    rf_port_names = tuple(
        port.name
        for port in scene.ports
        if hasattr(port, "reference_impedance") and hasattr(port, "termination")
    )
    if run_manifest is None:
        selected = rf_port_names if ports is None else tuple(str(name) for name in ports)
        if not selected or any(not name for name in selected):
            raise ValueError("Array basis compilation requires at least one RF power-wave port.")
        if len(set(selected)) != len(selected):
            raise ValueError("ports must contain unique names.")
        missing = tuple(name for name in selected if name not in rf_port_names)
        if missing:
            raise ValueError(f"Array basis request references non-RF or missing ports: {missing}.")
        metadata = {
            "port_names": selected,
            "execution": "declarative_only",
        }
        return selected, selected, None, metadata

    from ..network_sweep import NetworkRunManifest, resolve_network_run_manifest
    from ..waveport_sweep import WavePortRunManifest
    from ..lumped import PortSweep

    if isinstance(run_manifest, WavePortRunManifest):
        manifest_frequencies = tuple(float(value) for value in run_manifest.frequencies)
        selected = tuple(run_manifest.channel_names)
        physical = tuple(run_manifest.physical_port_names)
        ports_by_name = {port.name: port for port in scene.ports}
        if any(name not in ports_by_name for name in physical):
            raise ValueError("WavePortRunManifest physical ports do not belong to this Scene.")
        if len(run_manifest.prepared_ports) != len(physical) or any(
            prepared.port is not ports_by_name[name]
            for prepared, name in zip(run_manifest.prepared_ports, physical)
        ):
            raise ValueError("WavePortRunManifest was not prepared from this Scene's port objects.")
    elif isinstance(run_manifest, NetworkRunManifest):
        manifest_frequencies = tuple(float(value) for value in run_manifest.frequencies)
        selected = tuple(run_manifest.port_names)
        physical = tuple(run_manifest.all_rf_port_names)
        current = resolve_network_run_manifest(
            scene,
            PortSweep(ports=selected, source_time=run_manifest.source_time),
            manifest_frequencies,
        )
        if (
            current.all_rf_port_names != run_manifest.all_rf_port_names
            or current.reference_impedances != run_manifest.reference_impedances
            or current.metadata() != run_manifest.metadata()
        ):
            raise ValueError("NetworkRunManifest does not match this Scene's RF port contract.")
    else:
        raise TypeError(
            "run_manifest must be a NetworkRunManifest or WavePortRunManifest contract."
        )
    if ports is not None and tuple(str(name) for name in ports) != selected:
        raise ValueError("ports must exactly match the run manifest basis order when both are supplied.")
    metadata = dict(run_manifest.metadata())
    return selected, physical, manifest_frequencies, metadata


def _manifest_fingerprint(metadata: Mapping[str, object]) -> str:
    return hashlib.sha256(repr(sorted(metadata.items())).encode("utf-8")).hexdigest()


def compile_array_basis_request(
    scene,
    *,
    monitor: str | ClosedSurfaceMonitor,
    frequencies=None,
    ports=None,
    theta=None,
    phi=None,
    phase_center=None,
    frame=None,
    device=None,
    dtype=torch.float64,
    run_manifest=None,
) -> CompiledArrayBasisRequest:
    validate_array_superposition(scene)
    resolved_device = torch.device(scene.device if device is None else device)
    surface = _resolve_monitor(scene, monitor)
    selected_ports, physical_ports, manifest_frequencies, manifest_metadata = _manifest_contract(
        scene, run_manifest, ports
    )
    requested_frequencies = manifest_frequencies if frequencies is None else frequencies
    if manifest_frequencies is not None and tuple(float(v) for v in requested_frequencies) != manifest_frequencies:
        raise ValueError("frequencies must exactly match the run manifest frequencies.")
    resolved_frequencies = _resolve_frequencies(surface, requested_frequencies)
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype.")
    resolved_dtype = dtype
    angle_tensors = tuple(value for value in (theta, phi) if isinstance(value, torch.Tensor))
    if angle_tensors:
        if any(
            value.is_complex() or not value.dtype.is_floating_point for value in angle_tensors
        ):
            raise TypeError("theta and phi must be real floating-point tensors.")
        if len({value.dtype for value in angle_tensors}) != 1:
            raise TypeError("theta and phi must have the same dtype.")
        resolved_dtype = angle_tensors[0].dtype
    if resolved_dtype not in {torch.float32, torch.float64}:
        raise TypeError("dtype must be torch.float32 or torch.float64.")
    resolved_theta, resolved_phi = _resolve_angles(
        theta,
        phi,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    if phase_center is None:
        resolved_center = _array_aabb_center(
            scene,
            physical_ports,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        center_source = "array_aabb"
    else:
        resolved_center = torch.as_tensor(
            phase_center,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        center_source = "explicit"
    if resolved_center.shape != (3,) or not bool(torch.all(torch.isfinite(resolved_center))):
        raise ValueError("phase_center must contain three finite coordinates.")
    resolved_frame = _resolve_frame(
        frame,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    if surface.bounds is None:
        raise ValueError("ClosedSurfaceMonitor must expose finite bounds for array basis compilation.")
    monitor_faces = tuple(
        (face.face_label, face.axis, float(face.plane_position), tuple(float(v) for v in face.size))
        for face in surface.faces
    )
    return CompiledArrayBasisRequest(
        monitor_name=surface.name,
        monitor_bounds=surface.bounds,
        monitor_faces=monitor_faces,
        frequencies=resolved_frequencies,
        theta=resolved_theta,
        phi=resolved_phi,
        phase_center=resolved_center,
        frame=resolved_frame,
        phase_center_source=center_source,
        port_names=selected_ports,
        physical_port_names=physical_ports,
        run_manifest_metadata=manifest_metadata,
        run_manifest_fingerprint=_manifest_fingerprint(manifest_metadata),
    )


def compile_array_monitors(scene, *, monitor=None, **kwargs) -> tuple[CompiledArrayBasisRequest, ...]:
    """Compile one or all declared closed surfaces for embedded-pattern sampling."""

    if monitor is not None:
        return (compile_array_basis_request(scene, monitor=monitor, **kwargs),)
    surfaces = tuple(
        candidate for candidate in scene.monitors if isinstance(candidate, ClosedSurfaceMonitor)
    )
    if not surfaces:
        raise ValueError("Scene has no ClosedSurfaceMonitor for array basis compilation.")
    return tuple(
        compile_array_basis_request(scene, monitor=surface, **kwargs) for surface in surfaces
    )


__all__ = [
    "CompiledArrayBasisRequest",
    "compile_array_basis_request",
    "compile_array_monitors",
    "validate_array_superposition",
]
