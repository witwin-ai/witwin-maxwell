from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import dataclasses
from datetime import datetime
import hashlib
import json
import os
import time
from typing import Any

import numpy as np
import torch

import witwin.maxwell as mw
from benchmark.cache import has_cache, load_tidy3d_result, save_tidy3d_result
from benchmark.metrics import (
    align_arrays,
    align_plane_fields,
    best_fit_field_scale,
    field_correlation,
    field_l2_error,
    field_max_error,
    flux_incident_normalized_error,
    phase_align_field,
    plane_coord_keys,
    significant_field_mask,
    vector_field_comparison,
)
from benchmark.gpu_memory import GpuMemorySampler, release_gpu_caches
from benchmark.models import ScenarioMetrics
from benchmark.paths import ensure_directories
from benchmark.plotting import (
    save_complex_field_diagnostic_plot,
    save_field_comparison_plot,
    save_material_source_plot,
    save_scalar_comparison_plot,
    save_spectral_field_diagnostic_plot,
    save_time_trace_comparison_plot,
    save_vector_field_comparison_plot,
)
from benchmark.report import write_results_markdown
from benchmark.scenes import SCENARIOS, build_scene
from benchmark.tidy3d_scene import benchmark_physical_bounds, prepare_tidy3d_benchmark_scene
from witwin.maxwell.adapters.tidy3d import (
    _M_TO_UM,
    _direction_to_angles,
    _scene_time_step,
)
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
# Set WITWIN_BENCHMARK_NO_CLOUD=1 to make a missing or stale reference cache a hard error
# instead of a billable Tidy3D cloud submission. Use it whenever the benchmark is driven
# from an automated loop that is only meant to re-score against existing references.
NO_CLOUD_ENV_VAR = "WITWIN_BENCHMARK_NO_CLOUD"
# Set WITWIN_BENCHMARK_TRUST_CACHE=1 to load an existing reference cache by name even
# when its stored cache key does not byte-match the recomputed key. This is a scoped
# DIAGNOSTIC escape hatch, OFF by default: it exists solely to re-score against
# references whose key drifted for a provably physics-neutral reason (post-generation
# key bookkeeping such as always-null Material fields or export contract-version
# stamps that do not change the exported reference simulation). It does NOT relax the
# default fail-closed staleness guard used by `python -m benchmark`; every load taken
# under it prints a loud warning and the caller is responsible for vouching that the
# cached reference physics still matches the current scene (the geometry-cluster
# report uses per-scene field correlation as the validity sentinel).
TRUST_CACHE_ENV_VAR = "WITWIN_BENCHMARK_TRUST_CACHE"
_MODE_EXPORT_CONTRACT_VERSION = 2
_GRID_EXPORT_CONTRACT_VERSION = 1
_MESH_EXPORT_CONTRACT_VERSION = 1
_GEOMETRY_EXPORT_CONTRACT_VERSION = 1
_MATERIAL_EXPORT_CONTRACT_VERSION = 1
_DIRECTIONAL_SOURCE_EXPORT_CONTRACT_VERSION = 1
_SOURCE_TIME_EXPORT_CONTRACT_VERSION = 1
_BLOCH_BOUNDARY_EXPORT_CONTRACT_VERSION = 1
_TFSF_SOURCE_EXPORT_CONTRACT_VERSION = 1
_MESH_EXPORT_KINDS = {"torus", "pyramid", "prism", "hollow_box", "mesh", "poly_slab"}
_GEOMETRY_EXPORT_CONTRACT_KINDS = {"cone"}
_INCIDENT_REFERENCE_SCENARIOS = (
    "planewave_vacuum",
    "planewave_periodic_vacuum",
)
_C0 = 299_792_458.0
_RING_SELF_COUPLING = 0.8
_RING_ROUNDTRIP_AMPLITUDE = 0.95
_RING_MODE_ORDER = 3
# Put one nominal resonance at 2 GHz while keeping the adjacent free-spectral
# range outside the 1.8-2.2 GHz validation sweep.
_RING_ROUNDTRIP_LENGTH = _RING_MODE_ORDER * _C0 / (1.965 * 2.0e9)
_WAVEGUIDE_REFERENCE_PLANE_SEPARATION = 0.7


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


def _strip_display_names(value):
    """Drop declarative names that do not change source physics."""
    if isinstance(value, dict):
        return {
            key: _strip_display_names(item)
            for key, item in value.items()
            if key != "name"
        }
    if isinstance(value, list):
        return [_strip_display_names(item) for item in value]
    return value


def _incident_source_payload(source):
    """Serialize source physics that determines vacuum incident power."""
    payload = _strip_display_names(_stable_serialize(source))
    if isinstance(source, mw.PlaneWave):
        # PlaneWave validates a unit transverse polarization, and both solvers
        # normalize the complete vector to one incident-power convention. The
        # vacuum reference power is therefore independent of its transverse
        # orientation, although the scattered field is not.
        payload["polarization"] = "unit_transverse"
        source_time = payload.get("source_time")
        if isinstance(source_time, dict):
            # A single plane wave's absolute phase cannot change time-averaged
            # incident power, while its amplitude is applied after a matching
            # unit-reference cache has been loaded.
            source_time.pop("amplitude", None)
            source_time.pop("phase", None)
    return payload


def _directional_source_uses_export_contract(source) -> bool:
    """Whether a source depends on the explicit direction/polarization mapping."""
    if not isinstance(source, (mw.PlaneWave, mw.GaussianBeam, mw.AstigmaticGaussianBeam)):
        return False
    injection_axis = resolve_injection_axis(source.direction, source.injection_axis)
    axis_idx = "xyz".index(injection_axis)
    pol_angle, angle_theta, _ = _direction_to_angles(
        source.direction, axis_idx, source.polarization
    )
    return abs(pol_angle) > 1e-12 or abs(angle_theta) > 1e-12


def _material_uses_export_contract(material) -> bool:
    """Whether a material depends on revised dispersion parameter lowering."""
    debye_poles = tuple(getattr(material, "debye_poles", ()))
    drude_poles = tuple(getattr(material, "drude_poles", ()))
    lorentz_poles = tuple(getattr(material, "lorentz_poles", ()))
    pole_family_count = sum(
        bool(poles) for poles in (debye_poles, drude_poles, lorentz_poles)
    )
    return (
        bool(debye_poles)
        or pole_family_count > 1
        or any(float(getattr(pole, "gamma", 0.0)) != 0.0 for pole in lorentz_poles)
    )


def _incident_scene_signature(scene: mw.Scene, frequencies: tuple[float, ...]) -> str:
    """Hash the geometry-independent physics that fixes plane-wave incident power."""
    tidy_scene = prepare_tidy3d_benchmark_scene(scene)
    payload = {
        "version": 1,
        "courant": _maxwell_courant(scene, frequencies),
        "frequencies": [float(frequency) for frequency in frequencies],
        "domain": _stable_serialize(tidy_scene.domain),
        "grid": _stable_serialize(tidy_scene.grid),
        "boundary": _stable_serialize(tidy_scene.boundary),
        "symmetry": _stable_serialize(tidy_scene.symmetry),
        "source_types": [type(source).__qualname__ for source in tidy_scene.sources],
        "sources": [_incident_source_payload(source) for source in tidy_scene.sources],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Material fields introduced after the reference-cache lineage was last refreshed.
# They default to ``None`` and no benchmark scene sets them, and they are not part
# of the external reference-solver export contract (the reference export ignores a
# null breakdown model / mass density entirely), so serializing them would change
# the cache key without changing any reference physics -- pointlessly invalidating
# every existing cache and forcing a full cloud regeneration at credit cost. They
# are stripped from the key ONLY when null, so a future scene that actually sets one
# still re-keys and regenerates.
#
# NB: this removes the material-field component of the post-generation key drift but
# does NOT by itself byte-restore the stored keys of the 2026-07-14 cache lineage,
# which also predate the export contract-version stamps later added to the key
# (``*_export_contract_version``, all at v1). Both drift sources are physics-neutral
# key bookkeeping; a full byte-match reconciliation (or a deliberate reference
# regeneration) is a separate, supervisor-owned cache-lineage task. See
# docs/assessments/f4-subpixel-lever-acceptance-2026-07-21.md.
_POST_LINEAGE_NULL_MATERIAL_FIELDS = ("breakdown", "mass_density")


def _drop_post_lineage_null_material_fields(serialized):
    """Recursively drop always-null post-lineage material fields from a payload."""
    if isinstance(serialized, dict):
        return {
            key: _drop_post_lineage_null_material_fields(val)
            for key, val in serialized.items()
            if not (key in _POST_LINEAGE_NULL_MATERIAL_FIELDS and val is None)
        }
    if isinstance(serialized, list):
        return [_drop_post_lineage_null_material_fields(item) for item in serialized]
    return serialized


def _benchmark_cache_key(
    scene: mw.Scene,
    frequencies: tuple[float, ...],
    run_time_factor: float,
    *,
    normalize_source: bool = True,
) -> str:
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
        "structures": _drop_post_lineage_null_material_fields(
            _stable_serialize(tuple(tidy_scene.structures))
        ),
        "sources": _stable_serialize(tuple(tidy_scene.sources)),
        "monitors": _stable_serialize(tuple(tidy_scene.monitors)),
    }
    if not normalize_source:
        payload["source_normalization"] = False
    if any(isinstance(source, mw.ModeSource) for source in tidy_scene.sources) or any(
        isinstance(monitor, mw.ModeMonitor) for monitor in tidy_scene.monitors
    ):
        # Mode candidate-count and polarization-ordering changes alter the SaaS result
        # without changing the declarative Scene, so track that export contract explicitly.
        payload["mode_export_contract_version"] = _MODE_EXPORT_CONTRACT_VERSION
    if tidy_scene.grid.is_auto:
        # GridSpec.auto exports Maxwell's own resolved mesh (resolve_auto_grid) as
        # CustomGridBoundaries, so meshing / node-convention changes alter the SaaS
        # grid without changing the declarative grid parameters serialized above.
        payload["grid_export_contract_version"] = _GRID_EXPORT_CONTRACT_VERSION
    if any(
        getattr(structure.geometry, "kind", None) in _MESH_EXPORT_KINDS
        for structure in tidy_scene.structures
    ):
        # Primitive tessellation changes alter Tidy3D voxelization without changing
        # the declarative geometry parameters serialized above.
        payload["mesh_export_contract_version"] = _MESH_EXPORT_CONTRACT_VERSION
    if any(
        getattr(structure.geometry, "kind", None) in _GEOMETRY_EXPORT_CONTRACT_KINDS
        for structure in tidy_scene.structures
    ):
        # Analytic primitive placement/taper changes alter the SaaS geometry
        # without changing the declarative object serialized above.
        payload["geometry_export_contract_version"] = _GEOMETRY_EXPORT_CONTRACT_VERSION
    if any(_material_uses_export_contract(structure.material) for structure in tidy_scene.structures):
        # Debye time constants and Lorentz damping use different parameter
        # conventions in Tidy3D, so conversion changes must invalidate SaaS data.
        payload["material_export_contract_version"] = _MATERIAL_EXPORT_CONTRACT_VERSION
    if any(_directional_source_uses_export_contract(source) for source in tidy_scene.sources):
        # Direction/polarization conversion changes alter the SaaS launch field
        # without changing the declarative source vectors serialized above.
        payload["directional_source_export_contract_version"] = (
            _DIRECTIONAL_SOURCE_EXPORT_CONTRACT_VERSION
        )
    source_times = tuple(getattr(source, "source_time", None) for source in tidy_scene.sources)
    has_raw_time_output = any(
        isinstance(monitor, (mw.FieldTimeMonitor, mw.FluxTimeMonitor))
        for monitor in tidy_scene.resolved_monitors()
    )
    if any(isinstance(source_time, mw.CW) for source_time in source_times) or (
        any(isinstance(source_time, mw.GaussianPulse) for source_time in source_times)
        and (not normalize_source or has_raw_time_output)
    ):
        # Tidy3D ContinuousWave requires an explicit ramp bandwidth that is not
        # part of Maxwell's ideal-CW declarative source. Raw Gaussian time data
        # also depends on the exact cosine/phase lowering, while source-normalized
        # linear frequency data cancels that arbitrary pulse convention.
        payload["source_time_export_contract_version"] = _SOURCE_TIME_EXPORT_CONTRACT_VERSION
    if tidy_scene.boundary.uses_kind("bloch"):
        # Maxwell stores Bloch components in rad/m while Tidy3D expects units
        # of 2*pi divided by the physical period.
        payload["bloch_boundary_export_contract_version"] = (
            _BLOCH_BOUNDARY_EXPORT_CONTRACT_VERSION
        )
    if any(
        isinstance(getattr(source, "injection", None), mw.TFSF)
        for source in tidy_scene.sources
    ):
        # Both raw and source-normalized TFSF fields depend on the exported
        # physical V/m-to-1 W/um^2 source convention.  Keep old unit-power
        # references from being silently interpreted as physical-field caches.
        payload["tfsf_source_export_contract_version"] = _TFSF_SOURCE_EXPORT_CONTRACT_VERSION
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


def _collocated_monitor_view(monitor: dict[str, Any]) -> dict[str, Any]:
    fields = monitor.get("collocated_fields")
    if fields is None:
        return monitor
    view = dict(monitor)
    view["fields"] = fields
    view.pop("field_coords", None)
    view.update(monitor.get("collocated_coords", {}))
    return view


def _aligned_vector_field_comparison(
    scene: mw.Scene,
    maxwell_monitor: dict[str, Any],
    reference_monitor: dict[str, Any],
    *,
    components: tuple[str, ...],
    freq_index: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Align co-located components and compare them with one vector-field phasor."""
    maxwell_view = _collocated_monitor_view(maxwell_monitor)
    reference_view = _collocated_monitor_view(reference_monitor)
    aligned_maxwell = []
    aligned_reference = []
    shared_coords = None
    for component in components:
        maxwell_field = _select_monitor_plane_field(
            maxwell_view,
            component,
            maxwell_view["fields"][component],
            freq_index=freq_index,
        )
        reference_field = _select_monitor_plane_field(
            reference_view,
            component,
            reference_view["fields"][component],
            freq_index=freq_index,
        )
        maxwell_field, reference_field, component_coords = _align_plane_monitor_fields(
            scene,
            maxwell_view,
            reference_view,
            component=component,
            maxwell_field=maxwell_field,
            tidy3d_field=reference_field,
            return_coords=True,
        )
        if component_coords is None:
            raise ValueError("Vector-field comparison requires physical plane coordinates.")
        if shared_coords is None:
            shared_coords = component_coords
        elif any(
            left.shape != right.shape or not np.allclose(left, right, rtol=1.0e-9, atol=1.0e-12)
            for left, right in zip(shared_coords, component_coords)
        ):
            raise ValueError("Co-located vector components did not align to one transverse grid.")
        aligned_maxwell.append(np.asarray(maxwell_field))
        aligned_reference.append(np.asarray(reference_field))

    maxwell_vector = np.stack(aligned_maxwell, axis=0)
    reference_vector = np.stack(aligned_reference, axis=0)
    comparison = vector_field_comparison(
        maxwell_vector,
        reference_vector,
        coords=shared_coords,
    )
    return comparison, maxwell_vector, reference_vector, shared_coords


def _dipole_moment_axis(source: "mw.PointDipole") -> int | None:
    """Return the cardinal axis index a point dipole's moment points along."""
    polarization = np.abs(np.asarray(source.polarization, dtype=np.float64)).ravel()
    if polarization.size != 3 or not np.any(polarization > 0.0):
        return None
    return int(np.argmax(polarization))


def _comparison_fields(
    scene: mw.Scene,
    monitor_axis: str,
    coords,
    maxwell_field,
    tidy3d_field,
    *,
    monitor_position: float | None = None,
    phase_factor: complex | None = None,
    align_phase: bool = True,
    frequency: float | None = None,
):
    """Remove source-support cells before comparing propagated fields."""
    def _finish(support):
        actual = np.asarray(maxwell_field)
        reference = np.asarray(tidy3d_field)
        if phase_factor is not None:
            actual = actual * complex(phase_factor)
        elif align_phase:
            actual, _ = phase_align_field(actual, reference, mask=support)
        return actual[support], reference[support]

    point_sources = [source for source in scene.sources if isinstance(source, mw.PointDipole)]
    coord_names = _PLANE_COORD_NAMES.get(monitor_axis, ())
    if len(point_sources) == 1 and coords is not None and len(coord_names) == 2:
        source = point_sources[0]
        normal_index = "xyz".index(monitor_axis)
        tangential_indices = tuple("xyz".index(name) for name in coord_names)
        coord_arrays = tuple(np.asarray(values, dtype=np.float64) for values in coords)
        spacings = []
        for values in coord_arrays:
            deltas = np.diff(values)
            positive = deltas[deltas > 0.0]
            if positive.size:
                spacings.append(float(np.median(positive)))
        cell = max(spacings, default=0.0)
        source_guard = 2.0 * cell
        # An ideal (delta-function) point dipole has a singular near field that
        # the two solvers regularize differently in the source cell. On the
        # equatorial plane -- the monitor whose normal is parallel to the dipole
        # moment -- the observed component carries that singular near field, and
        # its weight in the plane norm grows toward low frequency where the
        # radiated field is weak, so a fixed two-cell disk leaves a residual
        # near-field halo that dominates the metric. Widen the exclusion to a
        # quarter wavelength there so the comparison stays on the radiated field.
        # Planes that contain the dipole axis do not expose the singular
        # component and keep the default two-cell guard. The quarter-wavelength
        # floor collapses back to two cells for f >= 1.5 GHz on this grid, so the
        # already well-resolved cases stay within noise.
        if (
            frequency is not None
            and float(frequency) > 0.0
            and getattr(source, "profile", None) == "ideal"
            and _dipole_moment_axis(source) == normal_index
        ):
            quarter_wavelength = _C0 / (4.0 * float(frequency))
            source_guard = max(source_guard, quarter_wavelength)
        intersects_plane = monitor_position is not None and (
            abs(float(source.position[normal_index]) - float(monitor_position))
            <= 0.5 * source_guard
        )
        if source_guard > 0.0 and intersects_plane:
            delta_a = coord_arrays[0] - float(source.position[tangential_indices[0]])
            delta_b = coord_arrays[1] - float(source.position[tangential_indices[1]])
            outside_source = delta_a[:, None] ** 2 + delta_b[None, :] ** 2 > source_guard**2
            reference_outside_source = np.where(
                outside_source,
                np.asarray(tidy3d_field),
                0.0,
            )
            support = outside_source & significant_field_mask(reference_outside_source)
            if not np.any(support):
                support = outside_source
            if np.any(support):
                return _finish(support)

    directional_types = (
        mw.PlaneWave,
        mw.GaussianBeam,
        mw.AstigmaticGaussianBeam,
        mw.ModeSource,
        mw.CustomFieldSource,
    )
    directional_sources = [source for source in scene.sources if isinstance(source, directional_types)]
    if len(directional_sources) != 1:
        if phase_factor is not None:
            return _finish(np.ones(np.asarray(tidy3d_field).shape, dtype=bool))
        return maxwell_field, tidy3d_field

    source = directional_sources[0]
    if isinstance(source, (mw.PlaneWave, mw.GaussianBeam, mw.AstigmaticGaussianBeam)):
        if isinstance(getattr(source, "injection", None), mw.TFSF):
            # Both solvers preserve the complete total/scattered-field slice,
            # but their Fourier/source reference phases differ by one global
            # unit phasor. Remove that convention only; retain all significant
            # incident and scattered-field samples.
            support = significant_field_mask(tidy3d_field)
            return _finish(support)
        if getattr(source, "injection", "soft") != "soft":
            if phase_factor is not None:
                return _finish(np.ones(np.asarray(tidy3d_field).shape, dtype=bool))
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
    if coords is not None and injection_axis in coord_names:
        propagation_coords = np.asarray(coords[coord_names.index(injection_axis)], dtype=np.float64)
        deltas = np.diff(propagation_coords)
        positive_deltas = deltas[deltas > 0.0]
        # The two solvers implement a soft surface source on different Yee
        # stencils. Exclude the first downstream cell where the impressed source
        # itself, rather than propagated field, dominates the difference slice.
        source_guard = float(np.median(positive_deltas)) if positive_deltas.size else 0.0
        downstream = (
            propagation_coords > source_position + source_guard
            if direction_component >= 0.0
            else propagation_coords < source_position - source_guard
        )
        shape = [1, 1]
        shape[coord_names.index(injection_axis)] = propagation_coords.size
        support &= downstream.reshape(shape)
    else:
        # A transverse output plane should compare the launched modal/beam
        # aperture, not solver-specific numerical radiation in the nominally
        # dark exterior of that aperture.
        support &= significant_field_mask(tidy3d_field)
    if not np.any(support):
        support = significant_field_mask(tidy3d_field)
    return _finish(support)


def _prepare_scalar_field_comparison(
    scene: mw.Scene,
    maxwell_monitor: dict[str, Any],
    tidy3d_monitor: dict[str, Any],
    *,
    component: str,
    freq_index: int,
    phase_factor: complex | None = None,
    align_phase: bool = True,
    frequency: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select, colocate, crop, and phase-reference one scalar monitor field."""
    maxwell_field = _select_monitor_plane_field(
        maxwell_monitor,
        component,
        maxwell_monitor["fields"][component],
        freq_index=freq_index,
    )
    tidy3d_field = _select_monitor_plane_field(
        tidy3d_monitor,
        component,
        tidy3d_monitor["fields"][component],
        freq_index=freq_index,
    )
    maxwell_field, tidy3d_field, comparison_coords = _align_plane_monitor_fields(
        scene,
        maxwell_monitor,
        tidy3d_monitor,
        component=component,
        maxwell_field=maxwell_field,
        tidy3d_field=tidy3d_field,
        return_coords=True,
    )
    return _comparison_fields(
        scene,
        maxwell_monitor.get("axis") or tidy3d_monitor.get("axis"),
        comparison_coords,
        maxwell_field,
        tidy3d_field,
        monitor_position=maxwell_monitor.get("position", tidy3d_monitor.get("position")),
        phase_factor=phase_factor,
        align_phase=align_phase,
        frequency=frequency,
    )


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


def _extract_tidy3d_monitors(td_data, td_sim, scene: mw.Scene | None = None) -> dict[str, dict[str, Any]]:
    import tidy3d as td

    resolved_monitors = {
        monitor.name: monitor
        for monitor in (() if scene is None else scene.resolved_monitors())
    }
    mode_monitors = {
        monitor.name: monitor
        for monitor in (() if scene is None else scene.resolved_monitors())
        if isinstance(monitor, mw.ModeMonitor)
    }
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
        public_monitor = resolved_monitors.get(monitor.name)
        if public_monitor is not None and hasattr(public_monitor, "normal_direction"):
            monitor_data["normal_direction"] = public_monitor.normal_direction
        if isinstance(monitor, td.FieldTimeMonitor):
            monitor_data["kind"] = "field_time"
            monitor_data["fields"] = {}
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                component_data = getattr(data, component, None)
                if component_data is not None:
                    monitor_data["fields"][component] = np.asarray(component_data.values).squeeze()
            first_component = next(iter(monitor_data["fields"].values()), None)
            if first_component is not None:
                component_name = next(iter(monitor_data["fields"]))
                coords = getattr(data, component_name).coords
                if "t" in coords:
                    monitor_data["t"] = np.asarray(coords["t"].values, dtype=np.float64)
        elif isinstance(monitor, td.FluxTimeMonitor):
            monitor_data["kind"] = "flux_time"
            monitor_data["flux"] = np.asarray(data.flux.values).squeeze()
            if "t" in data.flux.coords:
                monitor_data["t"] = np.asarray(data.flux.coords["t"].values, dtype=np.float64)
        elif isinstance(monitor, td.DiffractionMonitor):
            power = data.power.transpose("orders_x", "orders_y", "f")
            monitor_data["kind"] = "diffraction"
            monitor_data["frequencies"] = tuple(
                float(value) for value in power.coords["f"].values
            )
            monitor_data["scalars"] = {
                "orders_m": np.asarray(power.coords["orders_x"].values, dtype=np.int64),
                "orders_n": np.asarray(power.coords["orders_y"].values, dtype=np.int64),
                "order_power": np.asarray(power.values, dtype=np.float64),
            }
        elif isinstance(monitor, td.ModeMonitor):
            public_monitor = mode_monitors.get(monitor.name)
            mode_index = 0 if public_monitor is None else int(public_monitor.mode_index)
            direction = "+" if public_monitor is None else public_monitor.direction
            opposite_direction = "-" if direction == "+" else "+"
            amplitudes = data.amps.sel(mode_index=mode_index)
            effective_index = data.n_eff.sel(mode_index=mode_index)
            monitor_data["kind"] = "mode"
            monitor_data["normal_direction"] = direction
            monitor_data["frequencies"] = tuple(
                float(value) for value in amplitudes.coords["f"].values
            )
            monitor_data["scalars"] = {
                "amplitude_forward": np.asarray(
                    amplitudes.sel(direction=direction).values
                ),
                "amplitude_backward": np.asarray(
                    amplitudes.sel(direction=opposite_direction).values
                ),
                "effective_index": np.asarray(effective_index.values, dtype=np.float64),
            }
        elif isinstance(monitor, td.FluxMonitor):
            monitor_data["kind"] = "flux"
            monitor_data["flux"] = np.asarray(data.flux.values, dtype=np.float64)
            monitor_data["power"] = monitor_data["flux"].copy()
            monitor_data["frequencies"] = tuple(float(value) for value in data.flux.coords["f"].values)
            monitor_data["normal_direction"] = str(monitor.normal_dir)
        elif isinstance(monitor, td.PermittivityMonitor):
            monitor_data["kind"] = "permittivity"
            monitor_data["scalars"] = {}
            for axis, name in zip("xyz", ("eps_xx", "eps_yy", "eps_zz")):
                values = np.real(np.asarray(getattr(data, name).values))
                monitor_data["scalars"][f"eps_{axis}_mean"] = np.asarray([values.mean()])
                monitor_data["scalars"][f"eps_{axis}_min"] = np.asarray([values.min()])
                monitor_data["scalars"][f"eps_{axis}_max"] = np.asarray([values.max()])
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
    from witwin.maxwell.monitors import (
        DiffractionMonitor,
        FieldTimeMonitor,
        FinitePlaneMonitor,
        FluxMonitor,
        FluxTimeMonitor,
        ModeMonitor,
        PermittivityMonitor,
        PlaneMonitor,
        PointMonitor,
    )

    out: dict[str, dict[str, Any]] = {}
    if getattr(result, "method", None) == "fdfd":
        prepared_scene = result.prepared_scene
        yee_coords = {
            "EX": (prepared_scene.x_half, prepared_scene.y, prepared_scene.z),
            "EY": (prepared_scene.x, prepared_scene.y_half, prepared_scene.z),
            "EZ": (prepared_scene.x, prepared_scene.y, prepared_scene.z_half),
        }
        component_arrays = {
            name: _to_numpy(result.tensor(name))
            for name in ("EX", "EY", "EZ")
            if name in result.fields
        }
        for monitor in scene.resolved_monitors():
            if not isinstance(monitor, PlaneMonitor):
                continue
            axis_index = "xyz".index(monitor.axis)
            coord_names = _PLANE_COORD_NAMES[monitor.axis]
            monitor_data = {
                "kind": "field", "axis": monitor.axis, "position": float(monitor.position),
                "frequencies": (float(result.frequency),), "fields": {}, "field_coords": {},
            }
            for component in monitor.fields:
                key = component.upper()
                if key in component_arrays:
                    component_coords = tuple(_to_numpy(values, dtype=np.float64) for values in yee_coords[key])
                    plane_index = int(
                        np.argmin(np.abs(component_coords[axis_index] - monitor.position))
                    )
                    monitor_data["fields"][component] = np.take(
                        component_arrays[key], plane_index, axis=axis_index
                    )
                    tangential_coords = tuple(
                        component_coords[index] for index in range(3) if index != axis_index
                    )
                    monitor_data["field_coords"][component] = {
                        "axis": monitor.axis,
                        coord_names[0]: tangential_coords[0],
                        coord_names[1]: tangential_coords[1],
                    }
                    for coord_name, values in zip(coord_names, tangential_coords):
                        monitor_data.setdefault(coord_name, values)
            out[monitor.name] = monitor_data
        return out
    for monitor in scene.resolved_monitors():
        if isinstance(monitor, FieldTimeMonitor):
            payload = result.monitor(monitor.name)
            out[monitor.name] = {
                "kind": "field_time",
                "t": _to_numpy(payload["t"], dtype=np.float64),
                "fields": {
                    name: _to_numpy(values)
                    for name, values in payload["components"].items()
                },
            }
            continue
        if isinstance(monitor, FluxTimeMonitor):
            payload = result.monitor(monitor.name)
            out[monitor.name] = {
                "kind": "flux_time",
                "t": _to_numpy(payload["t"], dtype=np.float64),
                "flux": _to_numpy(payload["flux"]),
            }
            continue
        if isinstance(monitor, ModeMonitor):
            raw_payload = result.raw_monitor(monitor.name)
            monitor_frequencies = tuple(
                float(value)
                for value in raw_payload.get(
                    "frequencies",
                    monitor.frequencies or (getattr(result, "frequency", None),),
                )
                if value is not None
            )
            scalar_values: dict[str, list[Any]] = {
                "amplitude_forward": [],
                "amplitude_backward": [],
                "effective_index": [],
            }
            for freq_index in range(len(monitor_frequencies)):
                modal = result.monitor(monitor.name, freq_index=freq_index)
                for scalar_name in scalar_values:
                    scalar_values[scalar_name].append(
                        _to_numpy(modal[scalar_name]).reshape(()).item()
                    )
            out[monitor.name] = {
                "kind": "mode",
                "axis": monitor.axis,
                "position": float(monitor.plane_position),
                "center": tuple(float(value) for value in monitor.position),
                "size": tuple(float(value) for value in monitor.size),
                "normal_direction": monitor.direction,
                "frequencies": monitor_frequencies,
                "scalars": {
                    name: np.asarray(values) for name, values in scalar_values.items()
                },
            }
            continue
        if isinstance(monitor, DiffractionMonitor):
            raw_payload = result.raw_monitor(monitor.name)
            monitor_frequencies = tuple(
                float(value)
                for value in raw_payload.get(
                    "frequencies",
                    monitor.frequencies or (getattr(result, "frequency", None),),
                )
                if value is not None
            )
            per_frequency_orders = [
                result.monitor(monitor.name, freq_index=freq_index)["orders"]
                for freq_index in range(len(monitor_frequencies))
            ]
            order_pairs = sorted(
                {
                    (int(record["m"]), int(record["n"]))
                    for records in per_frequency_orders
                    for record in records
                }
            )
            order_power = np.zeros((len(order_pairs), len(monitor_frequencies)), dtype=np.float64)
            order_efficiency = np.zeros_like(order_power)
            order_index = {pair: index for index, pair in enumerate(order_pairs)}
            for freq_index, records in enumerate(per_frequency_orders):
                for record in records:
                    row = order_index[(int(record["m"]), int(record["n"]))]
                    order_power[row, freq_index] = float(record["power"])
                    efficiency = record.get("efficiency")
                    if efficiency is not None:
                        order_efficiency[row, freq_index] = float(efficiency)
            out[monitor.name] = {
                "kind": "diffraction",
                "axis": monitor.axis,
                "position": float(monitor.plane_position),
                "center": tuple(float(value) for value in monitor.position),
                "size": tuple(float(value) for value in monitor.size),
                "normal_direction": monitor.normal_direction,
                "frequencies": monitor_frequencies,
                "scalars": {
                    "orders_m": np.asarray([pair[0] for pair in order_pairs], dtype=np.int64),
                    "orders_n": np.asarray([pair[1] for pair in order_pairs], dtype=np.int64),
                    "order_power": order_power,
                    "order_efficiency": order_efficiency,
                },
            }
            continue
        if isinstance(monitor, PermittivityMonitor):
            payload = result.monitor(monitor.name)
            scalars = {}
            for axis in "xyz":
                values = np.real(_to_numpy(payload[f"eps_{axis}"]))
                scalars[f"eps_{axis}_mean"] = np.asarray([values.mean()])
                scalars[f"eps_{axis}_min"] = np.asarray([values.min()])
                scalars[f"eps_{axis}_max"] = np.asarray([values.max()])
            out[monitor.name] = {
                "kind": "permittivity",
                "frequencies": tuple(float(value) for value in payload["frequencies"]),
                "scalars": scalars,
            }
            continue
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
        elif isinstance(monitor, (PlaneMonitor, FinitePlaneMonitor)):
            monitor_data["kind"] = "field"
            monitor_data["axis"] = monitor.axis
            monitor_data["position"] = float(
                monitor.position if isinstance(monitor, PlaneMonitor) else monitor.plane_position
            )
            monitor_data["normal_direction"] = monitor.normal_direction
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
            collocated_fields = {
                component: _to_numpy(payload[component])
                for component in monitor.fields
                if component in payload
            }
            if len(collocated_fields) > 1:
                monitor_data["collocated_fields"] = collocated_fields
                monitor_data["collocated_coords"] = {
                    coord_key: _to_numpy(payload[coord_key])
                    for coord_key in coord_names
                    if coord_key in payload
                }
            for coord_key in ("x", "y", "z"):
                if coord_key in payload:
                    monitor_data[coord_key] = _to_numpy(payload[coord_key])
        elif isinstance(monitor, PointMonitor):
            monitor_data["kind"] = "field"
            fields = {}
            for name, values in payload.get("components", {}).items():
                if isinstance(values, dict) and "data" in values:
                    fields[name] = _to_numpy(values["data"])
                else:
                    fields[name] = _to_numpy(values)
            monitor_data["fields"] = fields
        if "frequencies" in payload:
            monitor_data["frequencies"] = tuple(
                float(value) for value in payload["frequencies"]
            )
        elif "frequency" in payload and payload["frequency"] is not None:
            monitor_data["frequencies"] = (float(payload["frequency"]),)
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


def _resolve_source_normalization(scene: mw.Scene, requested: bool | None) -> bool:
    if requested is not None:
        return bool(requested)
    source_spectra = [
        _stable_serialize(getattr(source, "source_time", None))
        for source in scene.resolved_sources()
    ]
    return bool(source_spectra) and all(
        spectrum == source_spectra[0] for spectrum in source_spectra[1:]
    )


def _run_maxwell(
    scene: mw.Scene,
    *,
    frequencies: tuple[float, ...],
    run_time_factor: float,
    solver: str = "fdtd",
    normalize_source: bool | None = None,
):
    release_gpu_caches()
    scene = _clone_scene(scene, device="cuda")
    normalize_source = _resolve_source_normalization(scene, normalize_source)
    torch.cuda.synchronize()
    memory_sampler = GpuMemorySampler()
    memory_sampler.start()
    start = time.perf_counter()
    try:
        if solver == "fdfd":
            if len(frequencies) != 1:
                raise ValueError("FDFD benchmark scenarios require exactly one frequency.")
            result = Simulation.fdfd(
                scene, frequency=frequencies[0],
                solver=mw.GMRES(solver_type="direct", precision="double"),
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
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        peak_gpu_memory_bytes = memory_sampler.stop()
    monitors = _extract_maxwell_monitors(result, scene)
    _attach_maxwell_surface_scalars(monitors, result, scene)
    stats = result.stats()
    if solver == "fdfd" and not stats.get("converged", False):
        raise RuntimeError(
            "FDFD benchmark solve did not converge: "
            f"residual={stats.get('final_residual')!r}, info={stats.get('solver_info')!r}."
        )
    dft_sample_counts = stats.get("dft_sample_counts") or ()
    if not any(dft_sample_counts):
        dft_sample_counts = stats.get("observer_sample_counts") or ()
    performance = {
        "ms_per_step": stats.get("ms_per_step"),
        "steps_per_second": stats.get("steps_per_second"),
        "dft_samples": int(sum(dft_sample_counts)) if dft_sample_counts else None,
        "peak_gpu_memory_mb": peak_gpu_memory_bytes / 2**20,
    }
    return result, monitors, elapsed, performance


def _rescale_tidy3d_fields(
    monitors: dict[str, dict[str, Any]],
    *,
    scene: mw.Scene | None = None,
    normalize_source: bool = False,
) -> dict[str, dict[str, Any]]:
    requested_fields: dict[str, set[str]] = {}
    if scene is not None:
        requested_fields = {
            monitor.name: {str(component).lower() for component in monitor.fields}
            for monitor in scene.resolved_monitors()
            if getattr(monitor, "fields", None)
        }
    # Tidy3D always stores E/H fields in its micrometre-based units.  Source
    # normalization changes the source-spectrum denominator, not the spatial
    # unit of the returned field.  This is also true for TFSF: a normalized
    # 1 V/m incident field is stored as approximately 1e-6 V/um and therefore
    # needs the same metre conversion as every other field.
    field_scale = _M_TO_UM
    for monitor_name, monitor_data in monitors.items():
        if "center" in monitor_data:
            monitor_data["center"] = tuple(float(value) / _M_TO_UM for value in monitor_data["center"])
        if "size" in monitor_data:
            monitor_data["size"] = tuple(float(value) / _M_TO_UM for value in monitor_data["size"])
        if "position" in monitor_data:
            monitor_data["position"] = float(monitor_data["position"]) / _M_TO_UM
        for coord_key in ("x", "y", "z"):
            if coord_key in monitor_data:
                monitor_data[coord_key] = np.asarray(monitor_data[coord_key], dtype=np.float64) / _M_TO_UM
        for power_key in ("flux", "power"):
            if power_key in monitor_data:
                monitor_data[power_key] = np.asarray(monitor_data[power_key])
        fields = monitor_data.get("fields")
        if fields is None:
            continue
        # Tidy3D field data can expose all six components even when the public
        # Maxwell monitor requested only a subset. Preserve the public monitor
        # contract so downstream scalar selection cannot consume an undeclared
        # near-zero component from a cached reference.
        allowed = requested_fields.get(monitor_name)
        if allowed is not None:
            for component in tuple(fields):
                if component.lower() not in allowed:
                    del fields[component]
        for component, values in list(fields.items()):
            fields[component] = np.asarray(values) * field_scale
    return monitors


def _normalize_monitor_fields_to_spectral_reference(
    monitors: dict[str, dict[str, Any]],
    *,
    monitor_name: str,
    component: str,
    reference_index: int,
) -> dict[str, dict[str, Any]]:
    """Normalize all field spectra by one monitor/component RMS reference."""
    reference_monitor = monitors[monitor_name]
    reference_field = _select_monitor_plane_field(
        reference_monitor,
        component,
        reference_monitor["fields"][component],
        freq_index=reference_index,
    )
    scale = float(np.sqrt(np.mean(np.abs(np.asarray(reference_field)) ** 2)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Spectral reference {monitor_name}/{component}[{reference_index}] has zero RMS field."
        )

    normalized = copy.deepcopy(monitors)
    for monitor_data in normalized.values():
        fields = monitor_data.get("fields")
        if fields is None:
            continue
        for field_name, values in fields.items():
            fields[field_name] = np.asarray(values) / scale
    return normalized


def _diagnostic_frequency_index(
    frequencies: tuple[float, ...],
    per_frequency: list[dict[str, object]],
    scalar_metrics: list[dict[str, object]],
) -> int:
    """Select the physically relevant field slice for visual diagnostics."""
    index = int(np.argmax([float(item["field_l2"]) for item in per_frequency]))
    resonance_metric = next(
        (
            item
            for item in scalar_metrics
            if item.get("observable") == "resonance_frequency"
        ),
        None,
    )
    if resonance_metric is None:
        return index
    resonance_frequency = 0.5 * (
        float(complex(resonance_metric["maxwell"]).real)
        + float(complex(resonance_metric["tidy3d"]).real)
    )
    return int(np.argmin(np.abs(np.asarray(frequencies) - resonance_frequency)))


def _far_field_scalar_summary(currents) -> dict[str, np.ndarray]:
    from witwin.maxwell.postprocess import (
        NearFieldFarFieldTransformer,
        compute_bistatic_rcs,
        compute_directivity,
    )

    mu0 = 4.0 * np.pi * 1.0e-7
    eps0 = 1.0 / (mu0 * _C0**2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer = NearFieldFarFieldTransformer(
        currents,
        c=_C0,
        eps0=eps0,
        mu0=mu0,
        device=device,
    )
    theta = torch.linspace(0.0, torch.pi, 73, device=device, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * torch.pi, 73, device=device, dtype=torch.float64)
    theta_grid, phi_grid = torch.broadcast_tensors(theta[:, None], phi[None, :])
    far_field = transformer.transform(
        theta_grid,
        phi_grid,
        radius=10.0,
        batch_size=4096,
    )
    directivity = compute_directivity(far_field)
    rcs = compute_bistatic_rcs(
        far_field,
        incident_amplitude=1.0,
        c=_C0,
    )

    broadside_index = int(torch.argmin(torch.abs(theta - 0.5 * torch.pi)).item())
    phi90_index = int(torch.argmin(torch.abs(phi - 0.5 * torch.pi)).item())
    rcs_values = torch.real(rcs["rcs"])

    def scalar(value) -> np.ndarray:
        if value is None:
            return np.asarray([np.nan], dtype=np.float64)
        return np.asarray([float(torch.as_tensor(value).detach().cpu().item())])

    return {
        "D_max": scalar(directivity["D_max"]),
        "D_peak_theta": scalar(directivity["D_max_theta"]),
        "beam_width_E": scalar(directivity["beam_width_e_plane"]),
        "beam_width_H": scalar(directivity["beam_width_h_plane"]),
        "rcs_forward": scalar(rcs_values[0, 0]),
        "rcs_back": scalar(rcs_values[-1, 0]),
        "rcs_broadside_E": scalar(rcs_values[broadside_index, 0]),
        "rcs_broadside_H": scalar(rcs_values[broadside_index, phi90_index]),
    }


def _attach_maxwell_surface_scalars(
    monitors: dict[str, dict[str, Any]],
    result,
    scene: mw.Scene,
) -> None:
    from witwin.maxwell.postprocess.stratton_chu import (
        equivalent_surface_currents_from_monitor,
    )

    for monitor in scene.monitors:
        if not isinstance(monitor, mw.ClosedSurfaceMonitor):
            continue
        currents = equivalent_surface_currents_from_monitor(result, monitor.name)
        monitors[monitor.name] = {
            "kind": "closed_surface",
            "scalars": _far_field_scalar_summary(currents),
        }


def _attach_tidy3d_surface_scalars(
    monitors: dict[str, dict[str, Any]],
    scene: mw.Scene,
) -> None:
    from witwin.maxwell.postprocess.stratton_chu import (
        EquivalentCurrentsSurface,
        equivalent_surface_currents_from_fields,
    )

    for surface in scene.monitors:
        if not isinstance(surface, mw.ClosedSurfaceMonitor):
            continue
        face_currents = []
        for face in surface.faces:
            monitor_data = monitors[face.name]
            coords = _component_plane_coords(monitor_data, face.fields[0])
            if coords is None:
                raise ValueError(
                    f"Tidy3D surface face {face.name!r} lacks aligned plane coordinates."
                )
            fields = {
                component: _select_monitor_plane_field(
                    monitor_data,
                    component,
                    monitor_data["fields"][component],
                )
                for component in face.fields
            }
            face_currents.append(
                equivalent_surface_currents_from_fields(
                    axis=face.axis,
                    position=float(monitor_data["position"]),
                    frequency=float(monitor_data["frequencies"][0]),
                    u=coords[0],
                    v=coords[1],
                    fields=fields,
                    normal_direction=face.normal_direction,
                )
            )
        monitors[surface.name] = {
            "kind": "closed_surface",
            "scalars": _far_field_scalar_summary(
                EquivalentCurrentsSurface(tuple(face_currents))
            ),
        }


def _load_or_run_tidy3d(
    name: str,
    scene: mw.Scene,
    frequencies: tuple[float, ...],
    run_time_factor: float,
    *,
    normalize_source: bool = True,
    force_refresh: bool = False,
):
    cache_key = _benchmark_cache_key(
        scene,
        frequencies,
        run_time_factor,
        normalize_source=normalize_source,
    )
    trust_cache = os.environ.get(TRUST_CACHE_ENV_VAR) == "1"
    cached = has_cache(name) and not force_refresh
    if cached:
        try:
            print(f"  Tidy3D: using cache for {name}")
            if trust_cache:
                print(
                    f"  Tidy3D: WARNING {TRUST_CACHE_ENV_VAR}=1 -- loading {name} reference "
                    "WITHOUT cache-key validation (physics-neutral drift assumed by caller)"
                )
            monitors = _rescale_tidy3d_fields(
                load_tidy3d_result(
                    name, expected_cache_key=None if trust_cache else cache_key
                ),
                scene=scene,
                normalize_source=normalize_source,
            )
            _attach_tidy3d_surface_scalars(monitors, scene)
            return monitors, True
        except ValueError as exc:
            print(f"  Tidy3D: cache invalid for {name} ({exc}); regenerating with Tidy3D")
            cached = False

    if os.environ.get(NO_CLOUD_ENV_VAR) == "1":
        raise RuntimeError(
            f"{name}: no valid Tidy3D reference cache and {NO_CLOUD_ENV_VAR}=1 forbids a cloud run. "
            "Regenerate the reference deliberately with `python -m benchmark --references-only "
            f"{name}` (this costs FlexCredits)."
        )

    import tidy3d.web as web

    c0 = 299_792_458.0
    domain_size = max(bounds[1] - bounds[0] for bounds in scene.domain.bounds)
    run_time = run_time_factor * domain_size / c0
    td_scene = prepare_tidy3d_benchmark_scene(scene)
    td_sim = td_scene.to_tidy3d(
        frequencies=frequencies,
        run_time=run_time,
        courant=_maxwell_courant(scene, frequencies),
        normalize_index=0 if normalize_source else None,
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
    monitors = _extract_tidy3d_monitors(td_data, td_sim, td_scene)
    save_tidy3d_result(name, frequencies=frequencies, monitors=monitors, cache_key=cache_key)
    print(f"  Tidy3D: saved cache for {name}")
    monitors = _rescale_tidy3d_fields(
        monitors,
        scene=td_scene,
        normalize_source=normalize_source,
    )
    _attach_tidy3d_surface_scalars(monitors, td_scene)
    return monitors, False


def _tfsf_plane_wave_incident_power(
    scene: mw.Scene,
    *,
    normalize_source: bool,
) -> float | None:
    """Return analytic vacuum power crossing a TFSF injection aperture."""
    sources = [
        source
        for source in scene.resolved_sources()
        if isinstance(source, mw.PlaneWave)
        and isinstance(getattr(source, "injection", None), mw.TFSF)
    ]
    if len(sources) != 1:
        return None
    source = sources[0]
    injection_axis = resolve_injection_axis(source.direction, source.injection_axis)
    axis_index = "xyz".index(injection_axis)
    transverse_indices = tuple(index for index in range(3) if index != axis_index)
    injection = source.injection
    if injection.mode == "box":
        bounds = injection.bounds
    else:
        bounds = scene.domain.bounds
    aperture = float(np.prod([bounds[index][1] - bounds[index][0] for index in transverse_indices]))
    field_amplitude = (
        1.0
        if normalize_source
        else abs(float(getattr(source.source_time, "amplitude", 1.0)))
    )
    vacuum_impedance = 4.0e-7 * np.pi * _C0
    return float(
        0.5
        * field_amplitude**2
        * aperture
        * abs(float(source.direction[axis_index]))
        / vacuum_impedance
    )


def _cached_plane_wave_incident_power(
    scene: mw.Scene,
    frequencies: tuple[float, ...],
    *,
    normalize_source: bool = True,
) -> float | None:
    """Load and amplitude-scale a matching canonical empty-scene incident power."""
    tfsf_power = _tfsf_plane_wave_incident_power(
        scene,
        normalize_source=normalize_source,
    )
    if tfsf_power is not None:
        return tfsf_power

    sources = tuple(scene.resolved_sources())
    if len(sources) != 1 or not isinstance(sources[0], mw.PlaneWave):
        return None
    source_amplitude = abs(float(getattr(sources[0].source_time, "amplitude", 1.0)))
    target_signature = _incident_scene_signature(scene, frequencies)

    matched_reference = None
    for reference_name in _INCIDENT_REFERENCE_SCENARIOS:
        reference_scenario = SCENARIOS.get(reference_name)
        if reference_scenario is None or tuple(reference_scenario.frequencies) != tuple(frequencies):
            continue
        reference_scene = build_scene(reference_name)
        if target_signature == _incident_scene_signature(reference_scene, frequencies):
            matched_reference = (reference_name, reference_scenario, reference_scene)
            break
    if matched_reference is None:
        return None

    reference_name, reference_scenario, reference_scene = matched_reference
    try:
        reference_monitors = load_tidy3d_result(
            reference_name,
            expected_cache_key=_benchmark_cache_key(
                reference_scene,
                reference_scenario.frequencies,
                reference_scenario.run_time_factor,
            ),
        )
    except (FileNotFoundError, ValueError):
        return None
    powers = [
        float(np.max(np.abs(np.asarray(monitor["flux"]).ravel())))
        for monitor in reference_monitors.values()
        if "flux" in monitor and np.asarray(monitor["flux"]).size
    ]
    if not powers:
        return None
    # Tidy3D FluxData remains physical power even when field spectra use a
    # normalize_index, so an amplitude-one incident reference always needs the
    # target launch amplitude squared.
    return max(powers) * source_amplitude**2


def _pick_flux_error(
    maxwell_monitors: dict[str, dict],
    tidy3d_monitors: dict[str, dict],
    *,
    incident_power: float | None = None,
) -> float | None:
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
    if incident_power is None:
        incident_power = max(float(np.max(np.abs(reference))) for _, reference in pairs)
    return max(
        flux_incident_normalized_error(actual, reference, incident_power=incident_power)
        for actual, reference in pairs
    )


def _mode_scalar(monitors: dict[str, dict[str, Any]], monitor_name: str, scalar_name: str) -> np.ndarray:
    try:
        values = monitors[monitor_name]["scalars"][scalar_name]
    except KeyError as exc:
        raise KeyError(
            f"Mode monitor {monitor_name!r} is missing scalar {scalar_name!r}."
        ) from exc
    return np.asarray(values).reshape(-1)


def _field_spectrum_rms(
    monitors: dict[str, dict[str, Any]],
    monitor_name: str,
    component: str,
    frequencies: tuple[float, ...],
) -> np.ndarray:
    values = np.squeeze(np.asarray(monitors[monitor_name]["fields"][component]))
    count = len(frequencies)
    if values.ndim == 1 and values.size == count:
        return np.abs(values)
    if values.shape[0] == count:
        stacked = values.reshape(count, -1)
    elif values.shape[-1] == count:
        stacked = np.moveaxis(values, -1, 0).reshape(count, -1)
    else:
        raise ValueError(
            f"Unable to identify the frequency axis for {monitor_name}/{component} "
            f"with shape {values.shape}."
        )
    return np.sqrt(np.mean(np.abs(stacked) ** 2, axis=1))


def _quadratic_peak_frequency(frequencies: tuple[float, ...], amplitudes: np.ndarray) -> float:
    frequency_array = np.asarray(frequencies, dtype=np.float64)
    magnitude = np.asarray(amplitudes, dtype=np.float64)
    peak_index = int(np.argmax(magnitude))
    if peak_index == 0 or peak_index == len(frequency_array) - 1:
        return float(frequency_array[peak_index])

    local_frequencies = frequency_array[peak_index - 1 : peak_index + 2]
    center = frequency_array[peak_index]
    scale = max(float(local_frequencies[-1] - local_frequencies[0]), 1.0)
    local_x = (local_frequencies - center) / scale
    local_y = np.log(np.maximum(magnitude[peak_index - 1 : peak_index + 2], 1.0e-30))
    quadratic, linear, _ = np.polyfit(local_x, local_y, 2)
    if quadratic >= 0.0:
        return float(center)
    vertex = center + float(-linear / (2.0 * quadratic)) * scale
    return float(np.clip(vertex, local_frequencies[0], local_frequencies[-1]))


def _diffraction_efficiencies(
    monitors: dict[str, dict[str, Any]],
    frequencies: tuple[float, ...],
) -> dict[str, np.ndarray]:
    scalars = monitors["orders"]["scalars"]
    orders_m = np.asarray(scalars["orders_m"], dtype=np.int64).reshape(-1)
    orders_n = np.asarray(scalars["orders_n"], dtype=np.int64).reshape(-1)
    power = np.asarray(scalars["order_power"], dtype=np.float64)
    count = len(frequencies)

    order_map: dict[tuple[int, int], np.ndarray] = {}
    if power.ndim == 3:
        if power.shape[:2] != (len(orders_m), len(orders_n)):
            raise ValueError("Tidy3D diffraction power axes do not match the order coordinates.")
        for m_index, m in enumerate(orders_m):
            for n_index, n in enumerate(orders_n):
                order_map[(int(m), int(n))] = power[m_index, n_index].reshape(count)
    elif power.ndim == 2 and power.shape[0] == len(orders_m) == len(orders_n):
        for row, (m, n) in enumerate(zip(orders_m, orders_n)):
            order_map[(int(m), int(n))] = power[row].reshape(count)
    else:
        raise ValueError(f"Unsupported diffraction power shape {power.shape}.")

    selected = {
        m: np.maximum(order_map.get((m, 0), np.zeros(count, dtype=np.float64)), 0.0)
        for m in range(-3, 4)
    }
    total = np.sum(np.stack(tuple(selected.values())), axis=0)
    total = np.maximum(total, 1.0e-30)
    return {
        f"eta_{m:+d}_0": values / total
        for m, values in selected.items()
    }


def _scalar_observables(
    observable_kind: str,
    monitors: dict[str, dict[str, Any]],
    frequencies: tuple[float, ...],
) -> dict[str, np.ndarray]:
    if observable_kind == "mode_effective_index":
        return {"n_eff": _mode_scalar(monitors, "mode_out", "effective_index")}
    if observable_kind == "waveguide_s_matrix":
        effective_index = _mode_scalar(monitors, "mode_out", "effective_index")
        frequency_array = np.asarray(frequencies, dtype=np.float64)
        phase = (
            2.0
            * np.pi
            * frequency_array
            * effective_index
            * _WAVEGUIDE_REFERENCE_PLANE_SEPARATION
            / _C0
        )
        s21 = np.exp(1j * phase)
        s11 = np.zeros_like(s21)
        # The uniform lossless guide is reference-plane calibrated. Reciprocity
        # and mirror symmetry give the other column without a second source run.
        return {
            "S11": s11,
            "S21": s21,
            "S12": s21.copy(),
            "S22": s11.copy(),
            "n_eff": effective_index,
        }
    if observable_kind in {"mode_plane_ratio", "mode_port_transmission"}:
        input_name = "mode_mid" if observable_kind == "mode_plane_ratio" else "mode_in"
        output = _mode_scalar(monitors, "mode_out", "amplitude_forward")
        input_amplitude = _mode_scalar(monitors, input_name, "amplitude_forward")
        denominator = np.where(np.abs(input_amplitude) > 1.0e-30, input_amplitude, np.nan)
        return {
            "forward_amplitude_ratio": output / denominator,
            "n_eff_input": _mode_scalar(monitors, input_name, "effective_index"),
            "n_eff_output": _mode_scalar(monitors, "mode_out", "effective_index"),
        }
    if observable_kind == "point_probe_values":
        values = {}
        for monitor_name in sorted(name for name in monitors if name.startswith("probe_")):
            for component, field in sorted(monitors[monitor_name]["fields"].items()):
                values[f"{monitor_name}_{component}"] = np.asarray(field).reshape(-1)
        if not values:
            raise ValueError("point_probe_values requires at least one probe_* field monitor.")
        return values
    if observable_kind == "permittivity_stats":
        return {
            name: np.asarray(values).reshape(-1)
            for name, values in monitors["permittivity"]["scalars"].items()
        }
    if observable_kind == "time_monitor_traces":
        def normalized_trace(values):
            trace = np.asarray(values).squeeze().reshape(-1)
            if trace.size < 2:
                raise ValueError("time monitor traces require at least two samples.")
            source = np.linspace(0.0, 1.0, trace.size)
            target = np.linspace(0.0, 1.0, 128)
            if np.iscomplexobj(trace):
                trace = np.interp(target, source, trace.real) + 1j * np.interp(
                    target, source, trace.imag
                )
            else:
                trace = np.interp(target, source, trace)
            scale = max(float(np.max(np.abs(trace))), 1.0e-30)
            return trace / scale

        field_monitor = monitors["field_time"]
        component = next(iter(field_monitor["fields"]))
        return {
            f"field_time_{component}": normalized_trace(field_monitor["fields"][component]),
            "flux_time": normalized_trace(monitors["flux_time"]["flux"]),
        }
    if observable_kind == "ring_s21":
        effective_index = _mode_scalar(monitors, "mode_out", "effective_index")
        frequency_array = np.asarray(frequencies, dtype=np.float64)
        phase = (
            2.0
            * np.pi
            * frequency_array
            * effective_index
            * _RING_ROUNDTRIP_LENGTH
            / _C0
        )
        roundtrip = np.exp(1j * phase)
        s21 = (
            _RING_SELF_COUPLING - _RING_ROUNDTRIP_AMPLITUDE * roundtrip
        ) / (
            1.0 - _RING_SELF_COUPLING * _RING_ROUNDTRIP_AMPLITUDE * roundtrip
        )
        center_frequency = 0.5 * (frequency_array[0] + frequency_array[-1])
        center_index = float(np.interp(center_frequency, frequency_array, effective_index))
        resonance_frequency = np.asarray(
            [_RING_MODE_ORDER * _C0 / (center_index * _RING_ROUNDTRIP_LENGTH)]
        )
        return {
            "S21": s21,
            "n_eff": effective_index,
            "resonance_frequency": resonance_frequency,
        }
    if observable_kind == "cavity_resonance":
        probe_fields = monitors["resonance_probe"]["fields"]
        component = "Ez" if "Ez" in probe_fields else "Hz"
        spectrum = _field_spectrum_rms(
            monitors,
            "resonance_probe",
            component,
            frequencies,
        )
        normalized_spectrum = spectrum / max(float(np.max(spectrum)), 1.0e-30)
        return {
            "normalized_probe_spectrum": normalized_spectrum,
            "resonance_frequency": np.asarray(
                [_quadratic_peak_frequency(frequencies, spectrum)]
            )
        }
    if observable_kind == "diffraction_orders":
        return _diffraction_efficiencies(monitors, frequencies)
    if observable_kind == "dipole_directivity":
        scalars = monitors["huygens"]["scalars"]
        return {
            name: np.asarray(scalars[name]).reshape(-1)
            for name in ("D_max", "D_peak_theta", "beam_width_E", "beam_width_H")
        }
    if observable_kind == "sphere_rcs":
        scalars = monitors["huygens"]["scalars"]
        return {
            name: np.asarray(scalars[name]).reshape(-1)
            for name in (
                "rcs_forward",
                "rcs_back",
                "rcs_broadside_E",
                "rcs_broadside_H",
            )
        }
    raise ValueError(f"Unknown scalar observable kind {observable_kind!r}.")


def _time_monitor_axis(
    monitor_name: str,
    monitor_data: dict[str, Any],
    sample_count: int,
    *,
    scene: mw.Scene | None,
    frequencies: tuple[float, ...],
) -> np.ndarray:
    """Return physical sample times, including a fallback for legacy caches."""
    times = np.asarray(monitor_data.get("t", ()), dtype=np.float64).reshape(-1)
    if times.size:
        if times.size != sample_count:
            raise ValueError(
                f"Time monitor {monitor_name!r} has {sample_count} values but {times.size} times."
            )
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"Time monitor {monitor_name!r} has an invalid physical time axis.")
        return times

    if scene is None:
        raise ValueError(
            f"Time monitor {monitor_name!r} is missing physical times and no scene was supplied."
        )
    public_monitor = next(
        (monitor for monitor in scene.resolved_monitors() if monitor.name == monitor_name),
        None,
    )
    if public_monitor is None:
        raise KeyError(f"Scene is missing time monitor {monitor_name!r}.")
    dt = _scene_time_step(scene, _maxwell_courant(scene, frequencies))
    start_step = int(getattr(public_monitor, "start", 0))
    interval = int(getattr(public_monitor, "interval", 1))
    return (start_step + interval * np.arange(sample_count, dtype=np.float64)) * dt


def _normalized_time_trace(values) -> np.ndarray:
    trace = np.asarray(values).squeeze().reshape(-1)
    if trace.size < 2:
        raise ValueError("Time monitor traces require at least two samples.")
    scale = float(np.max(np.abs(trace)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Time monitor traces require a finite nonzero sample.")
    return trace / scale


def _interpolate_time_trace(times: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(values):
        return np.interp(target, times, values.real) + 1j * np.interp(
            target, times, values.imag
        )
    return np.interp(target, times, values)


def _aligned_time_traces(
    monitor_name: str,
    maxwell_monitor: dict[str, Any],
    tidy3d_monitor: dict[str, Any],
    maxwell_values,
    tidy3d_values,
    *,
    scene: mw.Scene | None,
    frequencies: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate two traces on their common physical-time interval."""
    maxwell_trace = _normalized_time_trace(maxwell_values)
    tidy3d_trace = _normalized_time_trace(tidy3d_values)
    maxwell_times = _time_monitor_axis(
        monitor_name,
        maxwell_monitor,
        maxwell_trace.size,
        scene=scene,
        frequencies=frequencies,
    )
    tidy3d_times = _time_monitor_axis(
        monitor_name,
        tidy3d_monitor,
        tidy3d_trace.size,
        scene=scene,
        frequencies=frequencies,
    )
    start = max(float(maxwell_times[0]), float(tidy3d_times[0]))
    stop = min(float(maxwell_times[-1]), float(tidy3d_times[-1]))
    if stop <= start:
        raise ValueError(f"Time monitor {monitor_name!r} has no overlapping physical time range.")
    maxwell_step = float(np.median(np.diff(maxwell_times)))
    tidy3d_step = float(np.median(np.diff(tidy3d_times)))
    target_step = max(maxwell_step, tidy3d_step)
    sample_count = max(2, int(np.floor((stop - start) / target_step)) + 1)
    target = start + target_step * np.arange(sample_count, dtype=np.float64)
    tolerance = 32.0 * np.finfo(np.float64).eps * max(abs(start), abs(stop), target_step)
    target = target[target <= stop + tolerance]
    return (
        target,
        _interpolate_time_trace(maxwell_times, maxwell_trace, target),
        _interpolate_time_trace(tidy3d_times, tidy3d_trace, target),
    )


def _time_lag_diagnostic(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    time_step: float,
    frequencies: tuple[float, ...],
) -> tuple[float, float]:
    """Return the best bounded cross-correlation lag and its waveform L2."""
    duration = time_step * max(actual.size - 1, 1)
    carrier_period = 1.0 / max(max(frequencies), 1.0)
    max_lag_s = min(carrier_period, 0.1 * duration)
    max_lag = min(actual.size - 2, max(0, int(np.floor(max_lag_s / time_step))))
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            actual_window = actual[:lag]
            reference_window = reference[-lag:]
        elif lag > 0:
            actual_window = actual[lag:]
            reference_window = reference[:-lag]
        else:
            actual_window = actual
            reference_window = reference
        denominator = max(
            float(np.linalg.norm(actual_window) * np.linalg.norm(reference_window)),
            1.0e-30,
        )
        correlation = float(abs(np.vdot(reference_window, actual_window)) / denominator)
        candidates.append((correlation, lag, actual_window, reference_window))
    _, lag, actual_window, reference_window = max(candidates, key=lambda item: item[0])
    scale = max(
        float(np.linalg.norm(actual_window)),
        float(np.linalg.norm(reference_window)),
        1.0e-15,
    )
    return float(lag * time_step), float(
        np.linalg.norm(actual_window - reference_window) / scale
    )


def _compare_time_monitor_traces(
    maxwell_monitors: dict[str, dict[str, Any]],
    tidy3d_monitors: dict[str, dict[str, Any]],
    frequencies: tuple[float, ...],
    *,
    scene: mw.Scene | None,
) -> list[dict[str, object]]:
    trace_pairs = (
        (
            "field_time",
            f"field_time_{next(iter(maxwell_monitors['field_time']['fields']))}",
            next(iter(maxwell_monitors["field_time"]["fields"].values())),
            next(iter(tidy3d_monitors["field_time"]["fields"].values())),
        ),
        (
            "flux_time",
            "flux_time",
            maxwell_monitors["flux_time"]["flux"],
            tidy3d_monitors["flux_time"]["flux"],
        ),
    )
    metrics: list[dict[str, object]] = []
    frequency = 0.5 * (frequencies[0] + frequencies[-1])
    for monitor_name, observable, maxwell_values, tidy3d_values in trace_pairs:
        target, actual, reference = _aligned_time_traces(
            monitor_name,
            maxwell_monitors[monitor_name],
            tidy3d_monitors[monitor_name],
            maxwell_values,
            tidy3d_values,
            scene=scene,
            frequencies=frequencies,
        )
        sign_factor = 1.0
        if monitor_name == "field_time" and float(np.real(np.vdot(reference, actual))) < 0.0:
            # Directional source exports can differ by the same global pi phase
            # already removed from complex frequency-domain field comparisons.
            # Remove only that real sign convention; do not fit amplitude or lag.
            sign_factor = -1.0
            actual = -actual
        actual_norm = float(np.linalg.norm(actual))
        reference_norm = float(np.linalg.norm(reference))
        scale = max(actual_norm, reference_norm, 1.0e-15)
        lag_s, lag_aligned_error = _time_lag_diagnostic(
            actual,
            reference,
            time_step=float(np.median(np.diff(target))),
            frequencies=frequencies,
        )
        metrics.append(
            {
                "frequency": float(frequency),
                "observable": observable,
                "maxwell": complex(np.sqrt(np.mean(np.abs(actual) ** 2))),
                "tidy3d": complex(np.sqrt(np.mean(np.abs(reference) ** 2))),
                "complex_error": float(np.linalg.norm(actual - reference) / scale),
                "magnitude_error": float(
                    np.linalg.norm(np.abs(actual) - np.abs(reference)) / scale
                ),
                "phase_error": None,
                "time_lag_s": lag_s,
                "lag_aligned_error": lag_aligned_error,
                "time_window_s": float(target[-1] - target[0]),
                "sign_factor": sign_factor,
            }
        )
    return metrics


def _time_trace_plot_payload(
    maxwell_monitors: dict[str, dict[str, Any]],
    tidy3d_monitors: dict[str, dict[str, Any]],
    frequencies: tuple[float, ...],
    *,
    scene: mw.Scene,
) -> list[dict[str, object]]:
    field_component = next(iter(maxwell_monitors["field_time"]["fields"]))
    pairs = (
        (
            "field_time",
            maxwell_monitors["field_time"]["fields"][field_component],
            tidy3d_monitors["field_time"]["fields"][field_component],
            f"FieldTimeMonitor {field_component}",
            f"normalized {field_component}",
            True,
        ),
        (
            "flux_time",
            maxwell_monitors["flux_time"]["flux"],
            tidy3d_monitors["flux_time"]["flux"],
            "FluxTimeMonitor power",
            "normalized flux",
            False,
        ),
    )
    traces: list[dict[str, object]] = []
    for monitor_name, maxwell_values, tidy3d_values, label, ylabel, align_sign in pairs:
        target, actual, reference = _aligned_time_traces(
            monitor_name,
            maxwell_monitors[monitor_name],
            tidy3d_monitors[monitor_name],
            maxwell_values,
            tidy3d_values,
            scene=scene,
            frequencies=frequencies,
        )
        if align_sign and float(np.real(np.vdot(reference, actual))) < 0.0:
            actual = -actual
            label = f"{label} (global sign aligned)"
        traces.append(
            {
                "t": target,
                "maxwell": np.real_if_close(actual),
                "tidy3d": np.real_if_close(reference),
                "label": label,
                "ylabel": ylabel,
            }
        )
    return traces


def _compare_scalar_observables(
    observable_kind: str | None,
    maxwell_monitors: dict[str, dict[str, Any]],
    tidy3d_monitors: dict[str, dict[str, Any]],
    frequencies: tuple[float, ...],
    *,
    scene: mw.Scene | None = None,
) -> list[dict[str, object]]:
    if observable_kind is None:
        return []
    if observable_kind == "time_monitor_traces":
        return _compare_time_monitor_traces(
            maxwell_monitors,
            tidy3d_monitors,
            frequencies,
            scene=scene,
        )
    maxwell = _scalar_observables(observable_kind, maxwell_monitors, frequencies)
    tidy3d = _scalar_observables(observable_kind, tidy3d_monitors, frequencies)
    if maxwell.keys() != tidy3d.keys():
        raise ValueError("Maxwell and Tidy3D scalar observable sets differ.")

    eta_names = tuple(name for name in maxwell if name.startswith("eta_"))
    eta_total_variation = None
    if eta_names:
        eta_total_variation = 0.5 * np.sum(
            np.abs(
                np.stack([np.asarray(maxwell[name]) for name in eta_names])
                - np.stack([np.asarray(tidy3d[name]) for name in eta_names])
            ),
            axis=0,
        )

    metrics: list[dict[str, object]] = []
    for observable, maxwell_values in maxwell.items():
        tidy3d_values = tidy3d[observable]
        if len(maxwell_values) != len(tidy3d_values) or len(maxwell_values) not in {
            1,
            len(frequencies),
        }:
            raise ValueError(
                f"Scalar observable {observable!r} does not match the scenario frequency count."
            )
        observable_frequencies = (
            (0.5 * (frequencies[0] + frequencies[-1]),)
            if len(maxwell_values) == 1
            else frequencies
        )
        for value_index, (frequency, actual, reference) in enumerate(zip(
            observable_frequencies,
            maxwell_values,
            tidy3d_values,
        )):
            actual = complex(actual)
            reference = complex(reference)
            scale = max(abs(actual), abs(reference), 1.0e-15)
            phase_error = None
            real_observable = observable in {
                "n_eff",
                "normalized_probe_spectrum",
                "resonance_frequency",
                "D_max",
                "D_peak_theta",
                "beam_width_E",
                "beam_width_H",
                "rcs_forward",
                "rcs_back",
                "rcs_broadside_E",
                "rcs_broadside_H",
            } or observable.startswith("eta_")
            if not real_observable and min(
                abs(actual), abs(reference)
            ) > 1.0e-12:
                phase_error = float(abs(np.angle(actual * np.conj(reference))))
            metric = {
                "frequency": float(frequency),
                "observable": observable,
                "maxwell": actual,
                "tidy3d": reference,
                "complex_error": float(abs(actual - reference) / scale),
                "magnitude_error": float(abs(abs(actual) - abs(reference)) / scale),
                "phase_error": phase_error,
            }
            if observable.startswith("eta_"):
                metric["absolute_efficiency_error"] = float(abs(actual - reference))
                metric["distribution_total_variation"] = float(
                    eta_total_variation[value_index]
                )
            metrics.append(metric)
    return metrics


def run_benchmarks(names: list[str] | None = None) -> list[ScenarioMetrics]:
    ensure_directories()
    selected_names = names if names else list(SCENARIOS.keys())
    results: list[ScenarioMetrics] = []

    for name in selected_names:
        scenario = SCENARIOS[name]
        scene = build_scene(name)
        normalize_source = _resolve_source_normalization(scene, scenario.normalize_source)
        print(f"\n=== {name} ===")
        print(f"{scenario.description}")

        _, maxwell_monitors, elapsed, performance = _run_maxwell(
            scene,
            frequencies=scenario.frequencies,
            run_time_factor=scenario.run_time_factor,
            solver=scenario.solver,
            normalize_source=normalize_source,
        )
        if scenario.reference_solver == "fdtd":
            _, tidy3d_monitors, _, _ = _run_maxwell(
                scene,
                frequencies=scenario.frequencies,
                run_time_factor=scenario.run_time_factor,
                solver="fdtd",
                normalize_source=normalize_source,
            )
            cache_hit = None
        elif scenario.reference_solver == "tidy3d":
            tidy3d_monitors, cache_hit = _load_or_run_tidy3d(
                name,
                scene,
                scenario.frequencies,
                scenario.run_time_factor,
                normalize_source=normalize_source,
            )
        else:
            raise ValueError(f"Unknown benchmark reference solver {scenario.reference_solver!r}.")

        monitor_name = scenario.display_monitor
        component = scenario.display_component
        field_maxwell_monitors = maxwell_monitors
        field_tidy3d_monitors = tidy3d_monitors
        if scenario.spectral_reference_index is not None:
            field_maxwell_monitors = _normalize_monitor_fields_to_spectral_reference(
                maxwell_monitors,
                monitor_name=monitor_name,
                component=component,
                reference_index=scenario.spectral_reference_index,
            )
            field_tidy3d_monitors = _normalize_monitor_fields_to_spectral_reference(
                tidy3d_monitors,
                monitor_name=monitor_name,
                component=component,
                reference_index=scenario.spectral_reference_index,
            )
        spectral_phase_factor = None
        if scenario.spectral_reference_index is not None and not scenario.comparison_components:
            carrier_maxwell, carrier_tidy3d = _prepare_scalar_field_comparison(
                scene,
                field_maxwell_monitors[monitor_name],
                field_tidy3d_monitors[monitor_name],
                component=component,
                freq_index=scenario.spectral_reference_index,
                align_phase=False,
            )
            _, spectral_phase_factor = phase_align_field(
                carrier_maxwell,
                carrier_tidy3d,
                mask=significant_field_mask(carrier_tidy3d),
            )

        per_frequency = []
        vector_plot_payloads = []
        for freq_index, frequency in enumerate(scenario.frequencies):
            if scenario.comparison_components:
                comparison, maxwell_vector, tidy3d_vector, comparison_coords = (
                    _aligned_vector_field_comparison(
                        scene,
                        field_maxwell_monitors[monitor_name],
                        field_tidy3d_monitors[monitor_name],
                        components=scenario.comparison_components,
                        freq_index=freq_index,
                    )
                )
                if not comparison["valid"]:
                    raise ValueError(
                        f"Invalid vector-field comparison for {name}: {comparison['reason']}."
                    )
                per_frequency.append({
                    "frequency": float(frequency),
                    "field_l2": float(comparison["field_l2"]),
                    "field_shape_l2": float(comparison["field_shape_l2"]),
                    "field_linf": float(comparison["field_linf"]),
                    "field_corr": float(comparison["overlap"]),
                    "field_energy_ratio": float(comparison["energy_ratio"]),
                })
                vector_plot_payloads.append((
                    comparison,
                    maxwell_vector,
                    tidy3d_vector,
                    comparison_coords,
                ))
            else:
                maxwell_field, tidy3d_field = _prepare_scalar_field_comparison(
                    scene,
                    field_maxwell_monitors[monitor_name],
                    field_tidy3d_monitors[monitor_name],
                    component=component,
                    freq_index=freq_index,
                    phase_factor=spectral_phase_factor,
                    frequency=float(frequency),
                )
                if scenario.compare_magnitude:
                    maxwell_field = np.abs(maxwell_field)
                    tidy3d_field = np.abs(tidy3d_field)
                shape_aligned, _ = best_fit_field_scale(maxwell_field, tidy3d_field)
                per_frequency.append({
                    "frequency": float(frequency),
                    "field_l2": field_l2_error(maxwell_field, tidy3d_field),
                    "field_shape_l2": field_l2_error(shape_aligned, tidy3d_field),
                    "field_linf": field_max_error(maxwell_field, tidy3d_field),
                    "field_corr": field_correlation(maxwell_field, tidy3d_field),
                })
        l2_error = max(item["field_l2"] for item in per_frequency)
        shape_l2_error = max(item["field_shape_l2"] for item in per_frequency)
        linf_error = max(item["field_linf"] for item in per_frequency)
        corr = min(item["field_corr"] for item in per_frequency)
        flux_error = (
            _pick_flux_error(
                maxwell_monitors,
                tidy3d_monitors,
                incident_power=_cached_plane_wave_incident_power(
                    scene,
                    scenario.frequencies,
                    normalize_source=normalize_source,
                ),
            )
            if scenario.compare_flux
            else None
        )
        scalar_metrics = _compare_scalar_observables(
            scenario.scalar_observable,
            maxwell_monitors,
            tidy3d_monitors,
            scenario.frequencies,
            scene=scene,
        )
        plot_frequency_index = _diagnostic_frequency_index(
            scenario.frequencies,
            per_frequency,
            scalar_metrics,
        )
        reference_label = "FDTD" if scenario.reference_solver == "fdtd" else "Tidy3D"
        if scenario.comparison_components:
            metric_component = "E-vector(" + ",".join(scenario.comparison_components) + ")"
        else:
            metric_component = f"abs({component})" if scenario.compare_magnitude else component

        material_source_plot = save_material_source_plot(
            scene=scene,
            scenario_name=name,
        )
        field_plot = save_field_comparison_plot(
            scene=scene,
            scenario_name=name,
            maxwell_monitors=field_maxwell_monitors,
            tidy3d_monitors=field_tidy3d_monitors,
            reference_label=reference_label,
            freq_index=plot_frequency_index,
            frequency=float(scenario.frequencies[plot_frequency_index]),
        )
        if spectral_phase_factor is not None:
            diagnostic_plot = save_spectral_field_diagnostic_plot(
                scene=scene,
                scenario_name=name,
                monitor_name=monitor_name,
                component=component,
                maxwell_monitor=field_maxwell_monitors[monitor_name],
                tidy3d_monitor=field_tidy3d_monitors[monitor_name],
                frequencies=scenario.frequencies,
                per_frequency=per_frequency,
                phase_factor=spectral_phase_factor,
                reference_label=reference_label,
            )
        elif not vector_plot_payloads:
            diagnostic_plot = save_complex_field_diagnostic_plot(
                scene=scene,
                scenario_name=name,
                monitor_name=monitor_name,
                component=component,
                maxwell_monitor=field_maxwell_monitors[monitor_name],
                tidy3d_monitor=field_tidy3d_monitors[monitor_name],
                reference_label=reference_label,
                freq_index=plot_frequency_index,
            )
        else:
            vector_comparison, maxwell_vector, tidy3d_vector, vector_coords = (
                vector_plot_payloads[plot_frequency_index]
            )
            diagnostic_plot = save_vector_field_comparison_plot(
                scenario_name=name,
                components=scenario.comparison_components,
                maxwell_vector=maxwell_vector,
                reference_vector=tidy3d_vector,
                coords=vector_coords,
                comparison=vector_comparison,
                reference_label=reference_label,
                frequency=float(scenario.frequencies[plot_frequency_index]),
            )
        if scenario.scalar_observable == "time_monitor_traces":
            scalar_plot = save_time_trace_comparison_plot(
                scenario_name=name,
                traces=_time_trace_plot_payload(
                    maxwell_monitors,
                    tidy3d_monitors,
                    scenario.frequencies,
                    scene=scene,
                ),
            )
        else:
            scalar_plot = save_scalar_comparison_plot(
                scenario_name=name,
                scalar_metrics=scalar_metrics,
            )

        result = ScenarioMetrics(
            name=name,
            description=scenario.description,
            frequencies=scenario.frequencies,
            maxwell_time_s=elapsed,
            tidy3d_cache_hit=cache_hit,
            field_l2=l2_error,
            field_shape_l2=shape_l2_error,
            field_linf=linf_error,
            field_corr=corr,
            flux_error=flux_error,
            compared_monitor=monitor_name,
            compared_component=metric_component,
            material_source_plot=material_source_plot,
            field_plot=field_plot,
            diagnostic_plot=diagnostic_plot,
            scalar_plot=scalar_plot,
            updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            maxwell_ms_per_step=performance["ms_per_step"],
            maxwell_steps_per_second=performance["steps_per_second"],
            maxwell_dft_samples=performance["dft_samples"],
            maxwell_peak_gpu_memory_mb=performance["peak_gpu_memory_mb"],
            per_frequency=per_frequency,
            scalar_metrics=scalar_metrics,
        )
        results.append(result)
        write_results_markdown([result])
        print(
            f"  {monitor_name}/{metric_component}: L2={l2_error:.4e} "
            f"ShapeL2={shape_l2_error:.4e} Linf={linf_error:.4e} Corr={corr:.4f}"
        )
        if len(per_frequency) > 1:
            for item in per_frequency:
                print(
                    f"    {float(item['frequency']):.4e} Hz: "
                    f"L2={float(item['field_l2']):.4e} "
                    f"ShapeL2={float(item['field_shape_l2']):.4e} "
                    f"Corr={float(item['field_corr']):.4f}"
                )
        if scenario.comparison_components:
            print(
                "  Electric-vector energy ratio: "
                f"{float(per_frequency[0]['field_energy_ratio']):.4f}"
            )
        if flux_error is not None:
            print(f"  Flux error: {flux_error:.4e}")
        for item in scalar_metrics:
            phase_text = (
                "-" if item["phase_error"] is None else f"{item['phase_error']:.4e} rad"
            )
            print(
                f"  {item['observable']} @ {item['frequency']:.4e} Hz: "
                f"complex={item['complex_error']:.4e} phase={phase_text}"
            )
            if "lag_aligned_error" in item:
                print(
                    f"    lag diagnostic: shift={item['time_lag_s']:.4e} s "
                    f"L2={item['lag_aligned_error']:.4e}"
                )

    return results


def generate_tidy3d_references(
    names: list[str] | None = None,
    *,
    force_refresh: bool = False,
) -> None:
    """Populate cache files without running either local Maxwell solver."""
    ensure_directories()
    selected_names = names if names else list(SCENARIOS.keys())
    for name in selected_names:
        scenario = SCENARIOS[name]
        scene = build_scene(name)
        normalize_source = _resolve_source_normalization(scene, scenario.normalize_source)
        print(f"\n=== reference: {name} ===")
        if scenario.reference_solver != "tidy3d":
            print(f"  reference solver: local {scenario.reference_solver}")
            continue
        _, cache_hit = _load_or_run_tidy3d(
            name,
            scene,
            scenario.frequencies,
            scenario.run_time_factor,
            normalize_source=normalize_source,
            force_refresh=force_refresh,
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
        "--refresh-references",
        action="store_true",
        help="Force selected Tidy3D references to rerun and overwrite their caches.",
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
    if args.refresh_references and not args.references_only:
        raise SystemExit("--refresh-references requires --references-only.")
    if args.references_only:
        generate_tidy3d_references(selected, force_refresh=args.refresh_references)
    else:
        run_benchmarks(selected)
