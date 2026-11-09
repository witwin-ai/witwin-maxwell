from __future__ import annotations

from .._gc import paused_cyclic_gc
from ..monitors import (
    BreakdownMonitor,
    ComponentStressMonitor,
    DiffractionMonitor,
    DipoleEmissionMonitor,
    FieldTimeMonitor,
    FinitePlaneMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    MediumMonitor,
    ModeMonitor,
    PermittivityMonitor,
    PlaneMonitor,
    PointMonitor,
    PowerLossMonitor,
    WireMonitor,
    normalize_component,
    normalize_axis,
)
from ..monitors import AUX_FIELD_COMPONENTS, AuxFieldTimeMonitor, SurfaceFieldTimeMonitor
from ..sources import PointDipole

_POLARIZATION_AXES = (("x", "Ex"), ("y", "Ey"), ("z", "Ez"))


def _find_scene_dipole(scene, source_name: str) -> PointDipole:
    for source in getattr(scene, "sources", ()):
        if isinstance(source, PointDipole) and getattr(source, "name", None) == source_name:
            return source
    raise ValueError(
        f"DipoleEmissionMonitor references PointDipole {source_name!r}, "
        "which is not present in the scene."
    )


def _point_observer_record(name, position, *, component: str) -> dict[str, object]:
    return {
        "name": str(name),
        "kind": "point",
        "position": tuple(float(value) for value in position),
        "component": normalize_component(component),
    }


def _plane_observer_record(name, axis, position, *, component: str) -> dict[str, object]:
    return {
        "name": str(name),
        "kind": "plane",
        "axis": normalize_axis(axis),
        "position": float(position),
        "component": normalize_component(component),
    }


def _internal_observer_name(monitor_name: str, component: str) -> str:
    return f"{monitor_name}::{normalize_component(component)}"


def compile_fdtd_observers(scene):
    with paused_cyclic_gc():
        compiled = []
        point_fields = {}
        point_components = {}

        def normalized_point_fields(fields):
            normalized = point_fields.get(fields)
            if normalized is None:
                for field in fields:
                    if field not in point_components:
                        point_components[field] = normalize_component(field)
                normalized = tuple(point_components[field] for field in fields)
                point_fields[fields] = normalized
            return normalized

        monitors = scene.resolved_monitors() if hasattr(scene, "resolved_monitors") else scene.monitors
        for monitor in monitors:
            if isinstance(monitor, (SurfaceFieldTimeMonitor, AuxFieldTimeMonitor)):
                # Surface and auxiliary-state time traces are compiled by
                # compile_fdtd_time_observers; they carry no spectral observer.
                continue
            if isinstance(
                monitor,
                (
                    FieldTimeMonitor,
                    FluxTimeMonitor,
                    PermittivityMonitor,
                    MediumMonitor,
                    PowerLossMonitor,
                    WireMonitor,
                    BreakdownMonitor,
                    ComponentStressMonitor,
                ),
            ):
                # Time-domain monitors are handled by compile_fdtd_time_observers,
                # breakdown monitors by compile_fdtd_breakdown_observers, component
                # stress monitors at result time, and material monitors from compiled
                # material tensors; none must become a running-DFT spectral observer.
                continue
            if isinstance(monitor, DipoleEmissionMonitor):
                dipole = _find_scene_dipole(scene, monitor.source_name)
                polarization = tuple(float(value) for value in dipole.polarization)
                position = tuple(float(value) for value in dipole.position)
                active = tuple(
                    component
                    for (axis, component), weight in zip(_POLARIZATION_AXES, polarization)
                    if weight != 0.0
                )
                monitor_fields = normalized_point_fields(active)
                for component in monitor_fields:
                    observer = {
                        "name": f"{monitor.name}::{component}",
                        "kind": "point",
                        "position": position,
                        "component": component,
                    }
                    observer["monitor_name"] = monitor.name
                    observer["monitor_fields"] = monitor_fields
                    observer["monitor_frequencies"] = monitor.frequencies
                    observer["monitor_type"] = "dipole_emission"
                    observer["dipole_polarization"] = polarization
                    observer["dipole_position"] = position
                    observer["source_name"] = monitor.source_name
                    compiled.append(observer)
                continue
            if isinstance(monitor, PointMonitor):
                fields = tuple(monitor.fields)
                normalized_fields = normalized_point_fields(fields)
                position = tuple(float(value) for value in monitor.position)
                for field in normalized_fields:
                    observer = {
                        "name": f"{monitor.name}::{field}",
                        "kind": "point",
                        "position": position,
                        "component": field,
                    }
                    observer["monitor_name"] = monitor.name
                    observer["monitor_fields"] = normalized_fields
                    compiled.append(observer)
                continue
            if isinstance(monitor, (PlaneMonitor, FinitePlaneMonitor, FluxMonitor, ModeMonitor, DiffractionMonitor)):
                compiled.extend(_compile_plane_observers(monitor))
                continue
            raise ValueError(f"Unsupported monitor type: {type(monitor).__name__}")
        return compiled


def _compile_plane_observers(monitor):
    normalized_fields = tuple(normalize_component(field) for field in monitor.fields)
    compiled = []
    monitor_plane_position = monitor.position if isinstance(monitor, PlaneMonitor) else monitor.plane_position
    for field in normalized_fields:
        observer = _plane_observer_record(
            _internal_observer_name(monitor.name, field),
            axis=monitor.axis,
            position=monitor_plane_position,
            component=field,
        )
        observer["monitor_name"] = monitor.name
        observer["monitor_fields"] = normalized_fields
        observer["monitor_frequencies"] = monitor.frequencies
        observer["compute_flux"] = bool(getattr(monitor, "compute_flux", False))
        observer["normal_direction"] = getattr(monitor, "normal_direction", "+")
        if isinstance(monitor, FinitePlaneMonitor) and monitor.closure_planes is not None:
            observer["closure_planes"] = dict(monitor.closure_planes)
        if isinstance(monitor, FinitePlaneMonitor) and monitor.sample_spacing is not None:
            observer["sample_spacing"] = monitor.sample_spacing
            observer["sample_x_indices"] = getattr(monitor,"_fdtd_sample_x_indices",None)
            observer["tangential_bounds"] = dict(monitor.tangential_bounds)
        if isinstance(monitor, ModeMonitor):
            observer["monitor_type"] = "mode"
            observer["mode_spec"] = monitor.mode_spec()
        elif isinstance(monitor, DiffractionMonitor):
            observer["monitor_type"] = "diffraction"
            observer["diffraction_spec"] = monitor.diffraction_spec()
        compiled.append(observer)
    return compiled


def compile_fdtd_time_observers(scene):
    compiled = []
    monitors = scene.resolved_monitors() if hasattr(scene, "resolved_monitors") else scene.monitors
    for monitor in monitors:
        if isinstance(monitor, FieldTimeMonitor):
            record = {
                "name": monitor.name,
                "kind": "field_time",
                "start": int(monitor.start),
                "stop": None if monitor.stop is None else int(monitor.stop),
                "interval": int(monitor.interval),
                "region_kind": monitor.region_kind,
                "components": tuple(normalize_component(field).capitalize() for field in monitor.components),
                "position": tuple(float(value) for value in monitor.position),
                "size": tuple(float(value) for value in monitor.size),
            }
            if monitor.region_kind == "plane":
                record["axis"] = normalize_axis(monitor.axis)
                record["plane_position"] = float(monitor.plane_position)
            compiled.append(record)
            continue
        if isinstance(monitor, FluxTimeMonitor):
            compiled.append(
                {
                    "name": monitor.name,
                    "kind": "flux_time",
                    "start": int(monitor.start),
                    "stop": None if monitor.stop is None else int(monitor.stop),
                    "interval": int(monitor.interval),
                    "axis": normalize_axis(monitor.axis),
                    "position": float(monitor.position),
                    "fields": tuple(normalize_component(field).capitalize() for field in monitor.fields),
                    "normal_direction": monitor.normal_direction,
                }
            )
            continue
        if isinstance(monitor, SurfaceFieldTimeMonitor):
            compiled.append(_surface_field_time_record(monitor))
            continue
        if isinstance(monitor, AuxFieldTimeMonitor):
            compiled.append(_aux_field_time_record(monitor))
            continue
    return compiled


def _surface_field_time_record(monitor: SurfaceFieldTimeMonitor) -> dict[str, object]:
    return {
        "name": monitor.name,
        "kind": "surface_field_time",
        "start": int(monitor.start),
        "stop": None if monitor.stop is None else int(monitor.stop),
        "interval": int(monitor.interval),
        "components": tuple(normalize_component(field).capitalize() for field in monitor.components),
        "position": tuple(float(value) for value in monitor.position),
        "size": tuple(float(value) for value in monitor.size),
        "surface_kind": monitor.surface_kind,
        "faces": tuple(
            {
                "label": label,
                "axis": normalize_axis(axis),
                "normal_direction": normal_direction,
                "position": tuple(float(value) for value in face_position),
                "size": tuple(float(value) for value in face_size),
            }
            for label, axis, normal_direction, face_position, face_size in monitor.face_specs
        ),
    }


def _aux_field_time_record(monitor: AuxFieldTimeMonitor) -> dict[str, object]:
    return {
        "name": monitor.name,
        "kind": "aux_field_time",
        "start": int(monitor.start),
        "stop": None if monitor.stop is None else int(monitor.stop),
        "interval": int(monitor.interval),
        "components": tuple(monitor.components),
        "component_sources": {component: AUX_FIELD_COMPONENTS[component] for component in monitor.components},
        "position": tuple(float(value) for value in monitor.position),
        "size": tuple(float(value) for value in monitor.size),
        "region_kind": monitor.region_kind,
    }


def validate_component_stress_ports(scene, monitors=None) -> None:
    """Fail closed when a ComponentStressMonitor names a non-existent port.

    ``ComponentStressMonitor.port`` is a free string at construction; the bound
    port must resolve to an actual scene port at prepare/compile time. Raising
    here turns a silent typo into an explicit error before the run.
    """

    if monitors is None:
        monitors = scene.resolved_monitors() if hasattr(scene, "resolved_monitors") else scene.monitors
    stress_monitors = [m for m in monitors if isinstance(m, ComponentStressMonitor)]
    if not stress_monitors:
        return
    port_names = {
        str(getattr(port, "name", ""))
        for port in getattr(scene, "ports", ())
    }
    for monitor in stress_monitors:
        if monitor.port not in port_names:
            available = ", ".join(sorted(name for name in port_names if name)) or "<none>"
            raise ValueError(
                f"ComponentStressMonitor {monitor.name!r} references port "
                f"{monitor.port!r}, which is not a port on the scene; available "
                f"ports: {available}."
            )


def compile_fdtd_breakdown_observers(scene):
    """Collect lightweight BreakdownMonitor records (grid-resolved at prepare).

    Only region bounds, thresholds, and requested quantities are captured here;
    the region cell mask, control volumes, and occupancy are resolved from the
    solver grid in ``prepare_breakdown_observers`` so no per-step Scene raster is
    needed. ComponentStressMonitor port bindings are validated here so a typo'd
    port name fails closed at compile time rather than binding silently.
    """

    compiled = []
    monitors = scene.resolved_monitors() if hasattr(scene, "resolved_monitors") else scene.monitors
    monitors = list(monitors)
    validate_component_stress_ports(scene, monitors)
    for monitor in monitors:
        if isinstance(monitor, BreakdownMonitor):
            compiled.append(
                {
                    "name": monitor.name,
                    "kind": "breakdown",
                    "bounds": tuple((float(lo), float(hi)) for lo, hi in monitor.bounds),
                    "quantities": tuple(monitor.quantities),
                    "critical_field": float(monitor.critical_field),
                    "minimum_duration": float(monitor.minimum_duration),
                    "damage_exponent": (
                        None if monitor.damage_exponent is None else float(monitor.damage_exponent)
                    ),
                }
            )
    return compiled
