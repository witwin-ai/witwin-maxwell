from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
from datetime import datetime
import hashlib
import json
import time
from typing import Any

import numpy as np
import torch

import witwin.maxwell as mw
from benchmark.cache import has_cache, load_tidy3d_result, save_tidy3d_result
from benchmark.metrics import (
    align_arrays,
    align_plane_fields,
    field_correlation,
    field_l2_error,
    field_max_error,
    flux_incident_normalized_error,
    phase_align_field,
    plane_coord_keys,
    significant_field_mask,
)
from benchmark.models import ScenarioMetrics
from benchmark.paths import ensure_directories
from benchmark.plotting import (
    save_complex_field_diagnostic_plot,
    save_field_comparison_plot,
    save_material_source_plot,
)
from benchmark.report import write_results_markdown
from benchmark.scenes import SCENARIOS, build_scene
from benchmark.tidy3d_scene import benchmark_physical_bounds, prepare_tidy3d_benchmark_scene
from witwin.maxwell.adapters.tidy3d import _M_TO_UM
from witwin.maxwell.fdtd.excitation.spatial import resolve_injection_axis, soft_plane_wave_coordinate
from witwin.maxwell.fdtd.observers import _compute_plane_flux
from witwin.maxwell.monitors import required_flux_fields
from witwin.maxwell.simulation import TimeConfig, Simulation


_PLANE_COORD_NAMES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}
MAX_TIDY3D_COST_PER_SCENARIO = 2.0
_MODE_EXPORT_CONTRACT_VERSION = 1


def _to_numpy(value, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if dtype is None:
        return np.asarray(value)
    return np.asarray(value, dtype=dtype)


def _zero_size_axis(size) -> int | None:
    size_array = np.asarray(size, dtype=np.float64).reshape(-1)
    zero_axes = np.flatnonzero(np.isclose(size_array, 0.0, atol=1e-12))
    if zero_axes.size != 1:
        return None
    return int(zero_axes[0])


def _stable_serialize(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if dataclasses.is_dataclass(value):
        return {
            field.name: _stable_serialize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_serialize(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_serialize(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _stable_serialize(value.detach().cpu().item())
        return _stable_serialize(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if hasattr(value, "__dict__"):
        data = {
            "__class__": value.__class__.__name__,
        }
        data.update(
            {
                str(key): _stable_serialize(val)
                for key, val in sorted(vars(value).items(), key=lambda item: str(item[0]))
                if not str(key).startswith("_")
            }
        )
        return data
    return value


def _benchmark_cache_key(scene: mw.Scene, frequencies: tuple[float, ...], run_time_factor: float) -> str:
    tidy_scene = prepare_tidy3d_benchmark_scene(scene)
    payload = {
        "version": 5,
        "courant": _maxwell_courant(scene, frequencies),
        "frequencies": [float(frequency) for frequency in frequencies],
        "run_time_factor": float(run_time_factor),
        "domain": _stable_serialize(tidy_scene.domain),
        "grid": _stable_serialize(tidy_scene.grid),
        "boundary": _stable_serialize(tidy_scene.boundary),
        "symmetry": _stable_serialize(tidy_scene.symmetry),
        "structures": _stable_serialize(tuple(tidy_scene.structures)),
        "sources": _stable_serialize(tuple(tidy_scene.sources)),
        "monitors": _stable_serialize(tuple(tidy_scene.monitors)),
    }
    if any(isinstance(source, mw.ModeSource) for source in tidy_scene.sources) or any(
        isinstance(monitor, mw.ModeMonitor) for monitor in tidy_scene.monitors
    ):
        # Mode candidate-count and polarization-ordering changes alter the SaaS result
        # without changing the declarative Scene, so track that export contract explicitly.
        payload["mode_export_contract_version"] = _MODE_EXPORT_CONTRACT_VERSION
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _component_plane_coords(monitor_data: dict[str, Any], component: str) -> tuple[np.ndarray, np.ndarray] | None:
    field_coords = monitor_data.get("field_coords", {})
    component_coords = field_coords.get(component)
    if component_coords:
        if "axis" in component_coords and component_coords["axis"] in _PLANE_COORD_NAMES:
            coord_keys = _PLANE_COORD_NAMES[component_coords["axis"]]
        else:
            coord_keys = plane_coord_keys(component_coords)
        return tuple(_to_numpy(component_coords[key], dtype=np.float64) for key in coord_keys)

    axis = monitor_data.get("axis")
    if axis in _PLANE_COORD_NAMES:
        coord_keys = _PLANE_COORD_NAMES[axis]
        if all(key in monitor_data for key in coord_keys):
            return tuple(_to_numpy(monitor_data[key], dtype=np.float64) for key in coord_keys)
        return None

    try:
        coord_keys = plane_coord_keys(monitor_data)
    except ValueError:
        return None
    return tuple(_to_numpy(monitor_data[key], dtype=np.float64) for key in coord_keys)


def _select_monitor_plane_field(
    monitor_data: dict[str, Any],
    component: str,
    values,
    *,
    freq_index: int = 0,
) -> np.ndarray:
    array = np.squeeze(_to_numpy(values))
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise TypeError(f"Expected a 2D plane field or stacked multi-frequency planes, got shape {array.shape}.")

    frequencies = tuple(float(freq) for freq in monitor_data.get("frequencies", ()))
    coords = _component_plane_coords(monitor_data, component)
    plane_shape = tuple(coord.size for coord in coords) if coords is not None else None

    if plane_shape is not None:
        if array.shape[-2:] == plane_shape:
            if array.shape[0] <= freq_index:
                raise IndexError(f"freq_index {freq_index} is out of range for field shape {array.shape}.")
            return np.asarray(array[freq_index])
        if array.shape[:2] == plane_shape:
            if array.shape[-1] <= freq_index:
                raise IndexError(f"freq_index {freq_index} is out of range for field shape {array.shape}.")
            return np.asarray(array[..., freq_index])

    if frequencies:
        if array.shape[0] == len(frequencies):
            return np.asarray(array[freq_index])
        if array.shape[-1] == len(frequencies):
            return np.asarray(array[..., freq_index])

    if 0 <= freq_index < array.shape[0]:
        return np.asarray(array[freq_index])
    if 0 <= freq_index < array.shape[-1]:
        return np.asarray(array[..., freq_index])

    raise TypeError(
        "Unable to identify the frequency axis for plane field "
        f"shape {array.shape} and component {component!r}."
    )


def _align_plane_monitor_fields(
    scene: mw.Scene,
    maxwell_monitor: dict[str, Any],
    tidy3d_monitor: dict[str, Any],
    *,
    component: str,
    maxwell_field,
    tidy3d_field,
    return_coords: bool = False,
):
    maxwell_coords = _component_plane_coords(maxwell_monitor, component)
    tidy3d_coords = _component_plane_coords(tidy3d_monitor, component)
    axis = maxwell_monitor.get("axis") or tidy3d_monitor.get("axis")
    if axis in _PLANE_COORD_NAMES:
        if maxwell_coords is not None:
            maxwell_field, maxwell_coords = _crop_plane_field_to_physical_bounds(
                maxwell_field,
                maxwell_coords,
                scene=scene,
                axis=axis,
            )
        if tidy3d_coords is not None:
            tidy3d_field, tidy3d_coords = _crop_plane_field_to_physical_bounds(
                tidy3d_field,
                tidy3d_coords,
                scene=scene,
                axis=axis,
            )
    if maxwell_coords is not None and tidy3d_coords is not None:
        try:
            aligned_maxwell, aligned_tidy3d, aligned_coords = align_plane_fields(
                maxwell_field,
                tidy3d_field,
                source_coords=maxwell_coords,
                reference_coords=tidy3d_coords,
            )
            if return_coords:
                return aligned_maxwell, aligned_tidy3d, aligned_coords
            return aligned_maxwell, aligned_tidy3d
        except ValueError:
            pass
    aligned_maxwell, aligned_tidy3d = align_arrays(maxwell_field, tidy3d_field)
    if return_coords:
        return aligned_maxwell, aligned_tidy3d, None
    return aligned_maxwell, aligned_tidy3d


def _comparison_fields(scene: mw.Scene, monitor_axis: str, coords, maxwell_field, tidy3d_field):
    """Compare the downstream field of one directional soft surface source."""
    directional_types = (
        mw.PlaneWave,
        mw.GaussianBeam,
        mw.AstigmaticGaussianBeam,
        mw.ModeSource,
        mw.CustomFieldSource,
    )
    directional_sources = [source for source in scene.sources if isinstance(source, directional_types)]
    if len(directional_sources) != 1:
        return maxwell_field, tidy3d_field

    source = directional_sources[0]
    if isinstance(source, (mw.PlaneWave, mw.GaussianBeam, mw.AstigmaticGaussianBeam)):
        if getattr(source, "injection", "soft") != "soft":
            return maxwell_field, tidy3d_field
        injection_axis = resolve_injection_axis(source.direction, source.injection_axis)
        direction_component = float(source.direction["xyz".index(injection_axis)])
        source_position = soft_plane_wave_coordinate(scene, injection_axis, direction_component)
    elif isinstance(source, mw.ModeSource):
        injection_axis = source.normal_axis
        direction_component = 1.0 if source.direction == "+" else -1.0
        source_position = float(source.position["xyz".index(injection_axis)])
    else:
        injection_axis = source.normal_axis
        direction_component = 1.0
        source_position = float(source.field_dataset.coords["xyz".index(injection_axis)][0])

    support = np.ones(np.asarray(tidy3d_field).shape, dtype=bool)
    coord_names = _PLANE_COORD_NAMES.get(monitor_axis, ())
    if coords is not None and injection_axis in coord_names:
        propagation_coords = np.asarray(coords[coord_names.index(injection_axis)], dtype=np.float64)
        downstream = (
            propagation_coords > source_position
            if direction_component >= 0.0
            else propagation_coords < source_position
        )
        shape = [1, 1]
        shape[coord_names.index(injection_axis)] = propagation_coords.size
        support &= downstream.reshape(shape)
    if not np.any(support):
        support = significant_field_mask(tidy3d_field)
    phase_aligned, _ = phase_align_field(maxwell_field, tidy3d_field, mask=support)
    return phase_aligned[support], np.asarray(tidy3d_field)[support]


def _take_plane_window(values, indices_a: np.ndarray, indices_b: np.ndarray) -> np.ndarray:
    array = _to_numpy(values)
    windowed = np.take(array, indices_a, axis=-2)
    windowed = np.take(windowed, indices_b, axis=-1)
    return windowed


def _coordinate_tolerance(coords: np.ndarray, bounds: tuple[float, float]) -> float:
    return 1e-7 * max(1.0, float(np.max(np.abs(coords))), abs(bounds[0]), abs(bounds[1]))


def _crop_plane_field_to_physical_bounds(
    values,
    coords: tuple[np.ndarray, np.ndarray],
    *,
    scene: mw.Scene,
    axis: str,
):
    coord_names = _PLANE_COORD_NAMES[axis]
    physical_bounds = benchmark_physical_bounds(scene)
    tangential_bounds = tuple(physical_bounds["xyz".index(coord_name)] for coord_name in coord_names)
    coords_a = _to_numpy(coords[0], dtype=np.float64)
    coords_b = _to_numpy(coords[1], dtype=np.float64)
    tolerance_a = _coordinate_tolerance(coords_a, tangential_bounds[0])
    tolerance_b = _coordinate_tolerance(coords_b, tangential_bounds[1])
    mask_a = (coords_a >= tangential_bounds[0][0] - tolerance_a) & (coords_a <= tangential_bounds[0][1] + tolerance_a)
    mask_b = (coords_b >= tangential_bounds[1][0] - tolerance_b) & (coords_b <= tangential_bounds[1][1] + tolerance_b)
    if not np.any(mask_a) or not np.any(mask_b):
        return _to_numpy(values), (coords_a, coords_b)

    indices_a = np.flatnonzero(mask_a)
    indices_b = np.flatnonzero(mask_b)
    return _take_plane_window(values, indices_a, indices_b), (coords_a[indices_a], coords_b[indices_b])


def _benchmark_flux_from_payload(payload: dict[str, Any], scene: mw.Scene):
    axis = payload.get("axis")
    if axis not in _PLANE_COORD_NAMES:
        return payload.get("flux")

    coord_names = _PLANE_COORD_NAMES[axis]
    if any(coord_name not in payload for coord_name in coord_names):
        return payload.get("flux")

    physical_bounds = benchmark_physical_bounds(scene)
    tangential_bounds = tuple(physical_bounds["xyz".index(coord_name)] for coord_name in coord_names)
    coords_a = _to_numpy(payload[coord_names[0]], dtype=np.float64)
    coords_b = _to_numpy(payload[coord_names[1]], dtype=np.float64)
    tolerance_a = _coordinate_tolerance(coords_a, tangential_bounds[0])
    tolerance_b = _coordinate_tolerance(coords_b, tangential_bounds[1])
    mask_a = (coords_a >= tangential_bounds[0][0] - tolerance_a) & (coords_a <= tangential_bounds[0][1] + tolerance_a)
    mask_b = (coords_b >= tangential_bounds[1][0] - tolerance_b) & (coords_b <= tangential_bounds[1][1] + tolerance_b)
    if not np.any(mask_a) or not np.any(mask_b):
        return payload.get("flux")

    indices_a = np.flatnonzero(mask_a)
    indices_b = np.flatnonzero(mask_b)
    flux_payload: dict[str, Any] = {
        "axis": axis,
        "normal_direction": payload.get("normal_direction", "+"),
        coord_names[0]: coords_a[indices_a],
        coord_names[1]: coords_b[indices_b],
        "frequency": payload.get("frequency"),
        "frequencies": payload.get("frequencies", ()),
    }
    for component in required_flux_fields(axis):
        if component not in payload:
            return payload.get("flux")
        flux_payload[component] = _take_plane_window(payload[component], indices_a, indices_b)

    return _compute_plane_flux(flux_payload)


def _extract_tidy3d_monitors(td_data, td_sim) -> dict[str, dict[str, Any]]:
    import tidy3d as td

    monitors_out: dict[str, dict[str, Any]] = {}
    for monitor in td_sim.monitors:
        data = td_data[monitor.name]
        monitor_data: dict[str, Any] = {
            "center": tuple(float(value) for value in monitor.center),
            "size": tuple(float(value) for value in monitor.size),
        }
        zero_axis = _zero_size_axis(monitor.size)
        if zero_axis is not None:
            monitor_data["axis"] = "xyz"[zero_axis]
            monitor_data["position"] = float(monitor.center[zero_axis])
        if isinstance(monitor, td.FluxMonitor):
            monitor_data["kind"] = "flux"
            monitor_data["flux"] = np.asarray(data.flux.values, dtype=np.float64)
            monitor_data["power"] = monitor_data["flux"].copy()
            monitor_data["frequencies"] = tuple(float(value) for value in data.flux.coords["f"].values)
            monitor_data["normal_direction"] = str(monitor.normal_dir)
        else:
            monitor_data["kind"] = "field"
            fields = {}
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                component_data = getattr(data, component, None)
                if component_data is None:
                    continue
                fields[component] = np.asarray(component_data.values)
            monitor_data["fields"] = fields
            first_component = next(iter(fields.keys()), None)
            if first_component is not None:
                xr_data = getattr(data, first_component)
                for coord_key in ("x", "y", "z"):
                    if coord_key in xr_data.coords:
                        monitor_data[coord_key] = xr_data.coords[coord_key].values
                if "f" in xr_data.coords:
                    monitor_data["frequencies"] = tuple(float(value) for value in xr_data.coords["f"].values)
        monitors_out[monitor.name] = monitor_data
    return monitors_out


def _extract_maxwell_monitors(result, scene) -> dict[str, dict[str, Any]]:
    from witwin.maxwell.monitors import FluxMonitor, PlaneMonitor, PointMonitor

    out: dict[str, dict[str, Any]] = {}
    if getattr(result, "method", None) == "fdfd":
        component_arrays = {
            name: _to_numpy(result.tensor(name))
            for name in ("EX", "EY", "EZ")
            if name in result.fields
        }
        for monitor in scene.monitors:
            if not isinstance(monitor, PlaneMonitor):
                continue
            axis_index = "xyz".index(monitor.axis)
            first = next(iter(component_arrays.values()))
            axis_size = first.shape[axis_index]
            lo, hi = scene.domain.bounds[axis_index]
            axis_coords = np.linspace(lo, hi, axis_size, endpoint=False) + (hi - lo) / (2 * axis_size)
            plane_index = int(np.argmin(np.abs(axis_coords - monitor.position)))
            coord_names = _PLANE_COORD_NAMES[monitor.axis]
            monitor_data = {
                "kind": "field", "axis": monitor.axis, "position": float(monitor.position),
                "frequencies": (float(result.frequency),), "fields": {},
            }
            for coord_name, size in zip(coord_names, (first.shape[i] for i in range(3) if i != axis_index)):
                bound_lo, bound_hi = scene.domain.bounds["xyz".index(coord_name)]
                monitor_data[coord_name] = np.linspace(bound_lo, bound_hi, size, endpoint=False) + (bound_hi - bound_lo) / (2 * size)
            for component in monitor.fields:
                key = component.upper()
                if key in component_arrays:
                    monitor_data["fields"][component] = np.take(component_arrays[key], plane_index, axis=axis_index)
            out[monitor.name] = monitor_data
        return out
    for monitor in scene.monitors:
        payload = result.monitor(monitor.name)
        monitor_data: dict[str, Any] = {}
        if isinstance(monitor, FluxMonitor) or (
            isinstance(monitor, PlaneMonitor) and monitor.compute_flux
        ):
            monitor_data["kind"] = "flux"
            if "flux" in payload:
                raw_flux = _to_numpy(payload["flux"], dtype=np.float64)
                monitor_data["raw_flux"] = raw_flux
                benchmark_flux = _benchmark_flux_from_payload(payload, scene)
                monitor_data["flux"] = _to_numpy(benchmark_flux, dtype=np.float64)
            if "power" in payload:
                monitor_data["power"] = _to_numpy(monitor_data.get("flux", payload["power"]), dtype=np.float64)
        elif isinstance(monitor, PlaneMonitor):
            monitor_data["kind"] = "field"
            monitor_data["axis"] = monitor.axis
            monitor_data["position"] = float(monitor.position)
            components = payload.get("components", {})
            fields = {}
            field_coords = {}
            coord_names = _PLANE_COORD_NAMES[monitor.axis]
            for component in monitor.fields:
                comp_payload = components.get(component)
                if isinstance(comp_payload, dict) and "data" in comp_payload:
                    fields[component] = _to_numpy(comp_payload["data"])
                    if "coords" in comp_payload:
                        field_coords[component] = {
                            "axis": monitor.axis,
                            coord_names[0]: _to_numpy(comp_payload["coords"][0]),
                            coord_names[1]: _to_numpy(comp_payload["coords"][1]),
                        }
                elif component in payload:
                    fields[component] = _to_numpy(payload[component])
                elif comp_payload is not None:
                    fields[component] = _to_numpy(comp_payload)
            monitor_data["fields"] = fields
            if field_coords:
                monitor_data["field_coords"] = field_coords
            for coord_key in ("x", "y", "z"):
                if coord_key in payload:
                    monitor_data[coord_key] = _to_numpy(payload[coord_key])
        elif isinstance(monitor, PointMonitor):
            monitor_data["kind"] = "field"
            monitor_data["fields"] = {
                name: _to_numpy(values)
                for name, values in payload.get("components", {}).items()
            }
        out[monitor.name] = monitor_data
    return out


def _maxwell_courant(scene: mw.Scene, frequencies: tuple[float, ...]) -> float:
    c0 = 299_792_458.0
    from witwin.maxwell.scene import prepare_scene

    prepared_scene = prepare_scene(scene.clone(device="cpu"))
    min_dx = float(prepared_scene.dx_primal64.min())
    min_dy = float(prepared_scene.dy_primal64.min())
    min_dz = float(prepared_scene.dz_primal64.min())
    dt_cfl = 1.0 / (c0 * np.sqrt(1.0 / min_dx**2 + 1.0 / min_dy**2 + 1.0 / min_dz**2))
    characteristic_frequency = max((float(value) for value in frequencies), default=0.0)
    for source in scene.resolved_sources():
        source_time = getattr(source, "source_time", None)
        if source_time is not None:
            characteristic_frequency = max(
                characteristic_frequency,
                float(source_time.characteristic_frequency),
            )
    from witwin.maxwell.fdtd.runtime.initialization import _scene_material_characteristic_frequency

    characteristic_frequency = max(
        characteristic_frequency,
        _scene_material_characteristic_frequency(scene),
    )
    if characteristic_frequency <= 0.0:
        return 0.99
    return min(0.99, (1.0 / (30.0 * characteristic_frequency)) / dt_cfl)


def _compute_num_steps(scene: mw.Scene, run_time_factor: float, *, dt: float) -> int:
    c0 = 299_792_458.0
    domain_size = max(bounds[1] - bounds[0] for bounds in scene.domain.bounds)
    run_time_s = run_time_factor * domain_size / c0
    return int(np.ceil(run_time_s / float(dt)))


def _clone_scene(scene: mw.Scene, *, device: str) -> mw.Scene:
    return scene.clone(device=device)


def _run_maxwell(scene: mw.Scene, *, frequencies: tuple[float, ...], run_time_factor: float, solver: str = "fdtd"):
    scene = _clone_scene(scene, device="cuda")
    sources = tuple(scene.resolved_sources())
    source_spectra = [
        _stable_serialize(getattr(source, "source_time", None))
        for source in sources
    ]
    normalize_source = bool(source_spectra) and all(
        spectrum == source_spectra[0] for spectrum in source_spectra[1:]
    )
    start = time.perf_counter()
    if solver == "fdfd":
        if len(frequencies) != 1:
            raise ValueError("FDFD benchmark scenarios require exactly one frequency.")
        result = Simulation.fdfd(
            scene, frequency=frequencies[0],
            solver=mw.GMRES(solver_type="sqmr", preconditioner="ssor", precision="double"),
        ).run()
    elif solver == "fdtd":
        simulation = Simulation.fdtd(
            scene, frequencies=frequencies, run_time=TimeConfig(time_steps=1),
            spectral_sampler=mw.SpectralSampler(normalize_source=normalize_source),
        )
        prepared = simulation.prepare()
        simulation.config.run_time = TimeConfig(
            time_steps=_compute_num_steps(scene, run_time_factor, dt=prepared.solver.dt)
        )
        result = prepared.run()
    else:
        raise ValueError(f"Unknown benchmark solver {solver!r}.")
    elapsed = time.perf_counter() - start
    return result, _extract_maxwell_monitors(result, scene), elapsed


def _rescale_tidy3d_fields(monitors: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for monitor_data in monitors.values():
        if "center" in monitor_data:
            monitor_data["center"] = tuple(float(value) / _M_TO_UM for value in monitor_data["center"])
        if "size" in monitor_data:
            monitor_data["size"] = tuple(float(value) / _M_TO_UM for value in monitor_data["size"])
        if "position" in monitor_data:
            monitor_data["position"] = float(monitor_data["position"]) / _M_TO_UM
        for coord_key in ("x", "y", "z"):
            if coord_key in monitor_data:
                monitor_data[coord_key] = np.asarray(monitor_data[coord_key], dtype=np.float64) / _M_TO_UM
        fields = monitor_data.get("fields")
        if fields is None:
            continue
        for component, values in list(fields.items()):
            fields[component] = np.asarray(values) * _M_TO_UM
    return monitors


def _load_or_run_tidy3d(name: str, scene: mw.Scene, frequencies: tuple[float, ...], run_time_factor: float):
    cache_key = _benchmark_cache_key(scene, frequencies, run_time_factor)
    cached = has_cache(name)
    if cached:
        try:
            print(f"  Tidy3D: using cache for {name}")
            return _rescale_tidy3d_fields(load_tidy3d_result(name, expected_cache_key=cache_key)), True
        except ValueError as exc:
            print(f"  Tidy3D: cache invalid for {name} ({exc}); regenerating with Tidy3D")
            cached = False

    import tidy3d.web as web

    c0 = 299_792_458.0
    domain_size = max(bounds[1] - bounds[0] for bounds in scene.domain.bounds)
    run_time = run_time_factor * domain_size / c0
    td_scene = prepare_tidy3d_benchmark_scene(scene)
    td_sim = td_scene.to_tidy3d(
        frequencies=frequencies,
        run_time=run_time,
        courant=_maxwell_courant(scene, frequencies),
    )
    print(f"  Tidy3D: estimating reference cost for {name}")
    job = web.Job(simulation=td_sim, task_name=f"maxwell_benchmark_{name}", verbose=False)
    estimated_cost = float(job.estimate_cost(verbose=False))
    print(f"  Tidy3D: estimated cost {estimated_cost:.4f} FlexCredits")
    if estimated_cost > MAX_TIDY3D_COST_PER_SCENARIO:
        raise RuntimeError(
            f"Tidy3D estimate for {name} is {estimated_cost:.4f} FlexCredits, above the "
            f"per-scenario budget {MAX_TIDY3D_COST_PER_SCENARIO:.4f}; reference was not run."
        )
    print(f"  Tidy3D: generating reference for {name} with cloud run")
    td_data = job.run()
    monitors = _extract_tidy3d_monitors(td_data, td_sim)
    save_tidy3d_result(name, frequencies=frequencies, monitors=monitors, cache_key=cache_key)
    print(f"  Tidy3D: saved cache for {name}")
    return _rescale_tidy3d_fields(monitors), False


def _pick_flux_error(maxwell_monitors: dict[str, dict], tidy3d_monitors: dict[str, dict]) -> float | None:
    pairs = []
    for name, monitor in maxwell_monitors.items():
        if "flux" not in monitor or name not in tidy3d_monitors or "flux" not in tidy3d_monitors[name]:
            continue
        maxwell_flux = np.asarray(monitor["flux"]).ravel()
        tidy3d_flux = np.asarray(tidy3d_monitors[name]["flux"]).ravel()
        common = min(len(maxwell_flux), len(tidy3d_flux))
        if common == 0:
            continue
        pairs.append((maxwell_flux[:common], tidy3d_flux[:common]))
    if not pairs:
        return None
    incident_power = max(float(np.max(np.abs(reference))) for _, reference in pairs)
    return max(
        flux_incident_normalized_error(actual, reference, incident_power=incident_power)
        for actual, reference in pairs
    )


def run_benchmarks(names: list[str] | None = None) -> list[ScenarioMetrics]:
    ensure_directories()
    selected_names = names if names else list(SCENARIOS.keys())
    results: list[ScenarioMetrics] = []

    for name in selected_names:
        scenario = SCENARIOS[name]
        scene = build_scene(name)
        tidy_scene = prepare_tidy3d_benchmark_scene(scene)
        print(f"\n=== {name} ===")
        print(f"{scenario.description}")

        _, maxwell_monitors, elapsed = _run_maxwell(
            scene,
            frequencies=scenario.frequencies,
            run_time_factor=scenario.run_time_factor,
            solver=scenario.solver,
        )
        tidy3d_monitors, cache_hit = _load_or_run_tidy3d(
            name,
            scene,
            scenario.frequencies,
            scenario.run_time_factor,
        )

        monitor_name = scenario.display_monitor
        component = scenario.display_component
        per_frequency = []
        for freq_index, frequency in enumerate(scenario.frequencies):
            maxwell_field = _select_monitor_plane_field(
                maxwell_monitors[monitor_name], component,
                maxwell_monitors[monitor_name]["fields"][component], freq_index=freq_index)
            tidy3d_field = _select_monitor_plane_field(
                tidy3d_monitors[monitor_name], component,
                tidy3d_monitors[monitor_name]["fields"][component], freq_index=freq_index)
            maxwell_field, tidy3d_field, comparison_coords = _align_plane_monitor_fields(
                scene, maxwell_monitors[monitor_name], tidy3d_monitors[monitor_name],
                component=component, maxwell_field=maxwell_field, tidy3d_field=tidy3d_field,
                return_coords=True)
            maxwell_field, tidy3d_field = _comparison_fields(
                scene,
                maxwell_monitors[monitor_name].get("axis") or tidy3d_monitors[monitor_name].get("axis"),
                comparison_coords,
                maxwell_field,
                tidy3d_field,
            )
            per_frequency.append({
                "frequency": float(frequency),
                "field_l2": field_l2_error(maxwell_field, tidy3d_field),
                "field_linf": field_max_error(maxwell_field, tidy3d_field),
                "field_corr": field_correlation(maxwell_field, tidy3d_field),
            })
        l2_error = max(item["field_l2"] for item in per_frequency)
        linf_error = max(item["field_linf"] for item in per_frequency)
        corr = min(item["field_corr"] for item in per_frequency)
        flux_error = _pick_flux_error(maxwell_monitors, tidy3d_monitors)

        material_source_plot = save_material_source_plot(
            scene=scene,
            tidy_scene=tidy_scene,
            scenario_name=name,
        )
        field_plot = save_field_comparison_plot(
            scene=scene,
            scenario_name=name,
            maxwell_monitors=maxwell_monitors,
            tidy3d_monitors=tidy3d_monitors,
        )
        diagnostic_plot = save_complex_field_diagnostic_plot(
            scene=scene,
            scenario_name=name,
            monitor_name=monitor_name,
            component=component,
            maxwell_monitor=maxwell_monitors[monitor_name],
            tidy3d_monitor=tidy3d_monitors[monitor_name],
        )

        result = ScenarioMetrics(
            name=name,
            description=scenario.description,
            frequencies=scenario.frequencies,
            maxwell_time_s=elapsed,
            tidy3d_cache_hit=cache_hit,
            field_l2=l2_error,
            field_linf=linf_error,
            field_corr=corr,
            flux_error=flux_error,
            compared_monitor=monitor_name,
            compared_component=component,
            material_source_plot=material_source_plot,
            field_plot=field_plot,
            diagnostic_plot=diagnostic_plot,
            updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            per_frequency=per_frequency,
        )
        results.append(result)
        print(
            f"  {monitor_name}/{component}: L2={l2_error:.4e} Linf={linf_error:.4e} Corr={corr:.4f}"
        )
        if flux_error is not None:
            print(f"  Flux error: {flux_error:.4e}")

    write_results_markdown(results)
    return results


def generate_tidy3d_references(names: list[str] | None = None) -> None:
    """Populate cache files without running either local Maxwell solver."""
    ensure_directories()
    selected_names = names if names else list(SCENARIOS.keys())
    for name in selected_names:
        scenario = SCENARIOS[name]
        scene = build_scene(name)
        print(f"\n=== reference: {name} ===")
        _, cache_hit = _load_or_run_tidy3d(
            name, scene, scenario.frequencies, scenario.run_time_factor
        )
        print(f"  reference cache: {'hit' if cache_hit else 'generated'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark scenarios against Tidy3D.")
    parser.add_argument("scenarios", nargs="*", help="Scenario names to run. Defaults to all.")
    parser.add_argument(
        "--references-only", action="store_true",
        help="Generate/reuse Tidy3D caches without running Maxwell.",
    )
    parser.add_argument(
        "--campaign-only", action="store_true",
        help="Select only the S1-S6 cases from the validation campaign.",
    )
    parser.add_argument(
        "--historical-only", action="store_true",
        help="Select registered scenarios outside the S1-S6 campaign.",
    )
    parser.add_argument(
        "--solver", choices=("fdtd", "fdfd"),
        help="Restrict selected scenarios to one Maxwell solver.",
    )
    args = parser.parse_args()

    selection_modes = int(bool(args.scenarios)) + int(args.campaign_only) + int(args.historical_only)
    if selection_modes > 1:
        raise SystemExit("Use explicit scenarios, --campaign-only, or --historical-only, not more than one.")
    if args.campaign_only:
        from benchmark.validation_catalog import VALIDATION_CASES

        selected = [case.name for case in VALIDATION_CASES]
    elif args.historical_only:
        from benchmark.validation_catalog import VALIDATION_CASES

        campaign_names = {case.name for case in VALIDATION_CASES}
        selected = [name for name in SCENARIOS if name not in campaign_names]
    else:
        selected = args.scenarios or list(SCENARIOS.keys())
    if args.solver is not None:
        selected = [name for name in selected if SCENARIOS[name].solver == args.solver]
    unknown = [name for name in selected if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown benchmark scenarios: {unknown}. Available: {list(SCENARIOS)}")
    if args.references_only:
        generate_tidy3d_references(selected)
    else:
        run_benchmarks(selected)
