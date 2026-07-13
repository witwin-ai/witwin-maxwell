import warnings

import numpy as np
import torch

from ..monitors import normalize_axis, normalize_component
from .coords import component_coords
from .boundary import combine_complex_spectral_components, has_complex_fields

_AXIS_CODES = {"x": 0, "y": 1, "z": 2}

_PLANE_COORD_NAMES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}

_PLANE_ALIGNMENT_RULES = {
    "x": {
        "coord_sources": {"y": ("Ey", 0, "Hz", 0), "z": ("Ez", 1, "Hy", 1)},
        "average_axis": {"Ex": (0, 1), "Ey": 1, "Ez": 0, "Hy": 0, "Hz": 1},
    },
    "y": {
        "coord_sources": {"x": ("Ex", 0, "Hz", 0), "z": ("Ez", 1, "Hx", 1)},
        "average_axis": {"Ex": 1, "Ey": (0, 1), "Ez": 0, "Hx": 0, "Hz": 1},
    },
    "z": {
        "coord_sources": {"x": ("Ex", 0, "Hy", 0), "y": ("Ey", 1, "Hx", 1)},
        "average_axis": {"Ex": 1, "Ey": 0, "Ez": (0, 1), "Hx": 0, "Hy": 1},
    },
}

# Tangential (Ea, Eb, Ha, Hb) ordering for the plane-normal Poynting component
# (E x H)_n = Ea*Hb - Eb*Ha, used by the native flux-time reduction kernel.
_FLUX_KERNEL_COMPONENTS = {
    "x": ("Ey", "Ez", "Hy", "Hz"),
    "y": ("Ez", "Ex", "Hz", "Hx"),
    "z": ("Ex", "Ey", "Hx", "Hy"),
}


def _field_name(component):
    return normalize_component(component).capitalize()


def _plane_coord_names(axis):
    return _PLANE_COORD_NAMES[normalize_axis(axis)]


def _observer_is_point(observer):
    if "field_index" in observer:
        return True
    if "axis" in observer:
        return False
    kind = observer.get("kind")
    if kind is not None:
        return kind == "point"
    return False


def _monitor_payload_is_point(payload):
    if "field_indices" in payload:
        return True
    if "axis" in payload:
        return False
    kind = payload.get("kind")
    if kind is not None:
        return kind == "point"
    return False


def _point_observer_record(name, position, component="Ez"):
    if len(position) != 3:
        raise ValueError("Point observer position must have three coordinates.")
    return {
        "name": str(name),
        "kind": "point",
        "position": tuple(float(value) for value in position),
        "component": normalize_component(component),
    }


def _plane_observer_record(name, axis="z", position=0.0, component="Ez"):
    return {
        "name": str(name),
        "kind": "plane",
        "axis": normalize_axis(axis),
        "position": float(position),
        "component": normalize_component(component),
    }


def _average_adjacent(data, axis):
    if data.ndim == 3:
        axis += 1
    if data.shape[axis] < 2:
        raise ValueError("Cannot average adjacent Yee samples on an axis with fewer than two points.")
    slicer_lo = [slice(None)] * data.ndim
    slicer_hi = [slice(None)] * data.ndim
    slicer_lo[axis] = slice(0, -1)
    slicer_hi[axis] = slice(1, None)
    return 0.5 * (data[tuple(slicer_lo)] + data[tuple(slicer_hi)])


def _average_adjacent_axes(data, axes):
    if isinstance(axes, int):
        return _average_adjacent(data, axes)

    averaged = data
    for axis in axes:
        averaged = _average_adjacent(averaged, axis)
    return averaged


def _resolve_coord_source(component_payloads, *candidates):
    for component_name, coord_index in zip(candidates[::2], candidates[1::2]):
        payload = component_payloads.get(component_name)
        if payload is not None:
            return payload["coords"][coord_index]
    return None


def _align_plane_monitor_payload(axis, component_payloads):
    axis_name = normalize_axis(axis)
    rules = _PLANE_ALIGNMENT_RULES[axis_name]
    coord_names = _plane_coord_names(axis_name)

    aligned = {}
    for component_name, payload in component_payloads.items():
        average_axis = rules["average_axis"].get(component_name)
        if average_axis is None:
            continue
        else:
            aligned[component_name] = _average_adjacent_axes(payload["data"], average_axis)

    coord_a = _resolve_coord_source(
        component_payloads,
        *rules["coord_sources"][coord_names[0]],
    )
    coord_b = _resolve_coord_source(
        component_payloads,
        *rules["coord_sources"][coord_names[1]],
    )
    if coord_a is None or coord_b is None:
        return None

    return {
        coord_names[0]: coord_a,
        coord_names[1]: coord_b,
        "fields": aligned,
    }


def _physical_plane_masks(solver, axis, coord_a, coord_b):
    coord_names = _plane_coord_names(axis)
    domain_range = getattr(solver.scene, "physical_domain_range", solver.scene.domain_range)
    bounds = {
        "x": (domain_range[0], domain_range[1]),
        "y": (domain_range[2], domain_range[3]),
        "z": (domain_range[4], domain_range[5]),
    }
    arrays = (np.asarray(coord_a, dtype=np.float64), np.asarray(coord_b, dtype=np.float64))
    masks = []
    for name, values in zip(coord_names, arrays):
        lo, hi = bounds[name]
        tolerance = 1e-7 * max(1.0, float(np.max(np.abs(values))), abs(lo), abs(hi))
        masks.append((values >= lo - tolerance) & (values <= hi + tolerance))
    return tuple(masks)


def _crop_aligned_plane_to_physical_bounds(solver, axis, aligned):
    coord_names = _plane_coord_names(axis)
    mask_a, mask_b = _physical_plane_masks(
        solver,
        axis,
        aligned[coord_names[0]],
        aligned[coord_names[1]],
    )
    indices_a = np.flatnonzero(mask_a)
    indices_b = np.flatnonzero(mask_b)
    if indices_a.size == 0 or indices_b.size == 0:
        raise ValueError("FluxMonitor plane has no samples inside the physical domain.")

    cropped_fields = {}
    for component_name, values in aligned["fields"].items():
        if isinstance(values, torch.Tensor):
            index_a = torch.as_tensor(indices_a, device=values.device, dtype=torch.long)
            index_b = torch.as_tensor(indices_b, device=values.device, dtype=torch.long)
            cropped = torch.index_select(values, values.ndim - 2, index_a)
            cropped = torch.index_select(cropped, cropped.ndim - 1, index_b)
        else:
            cropped = np.take(np.take(values, indices_a, axis=-2), indices_b, axis=-1)
        cropped_fields[component_name] = cropped
    return {
        coord_names[0]: np.asarray(aligned[coord_names[0]])[indices_a],
        coord_names[1]: np.asarray(aligned[coord_names[1]])[indices_b],
        "fields": cropped_fields,
    }


def _plane_shape(field, axis):
    if axis == "x":
        return (field.shape[1], field.shape[2])
    if axis == "y":
        return (field.shape[0], field.shape[2])
    return (field.shape[0], field.shape[1])


def _normalize_frequency_list(frequencies):
    if frequencies is None:
        return ()
    if isinstance(frequencies, (tuple, list, np.ndarray)):
        values = tuple(float(freq) for freq in frequencies)
    else:
        values = (float(frequencies),)
    if not values:
        raise ValueError("At least one observer frequency is required.")
    return values


def _merge_frequency_lists(*frequency_lists):
    ordered = []
    seen = set()
    for frequency_list in frequency_lists:
        for frequency in _normalize_frequency_list(frequency_list):
            if frequency in seen:
                continue
            ordered.append(frequency)
            seen.add(frequency)
    return tuple(ordered)


def _spectral_scale(entry):
    if entry["window_normalization"] > 0:
        return 2.0 / entry["window_normalization"]
    if entry["sample_count"] > 0:
        return 2.0 / entry["sample_count"]
    return 0.0


def _contains_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_torch_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_torch_tensor(item) for item in value)
    return False


def _infer_tensor_device_and_dtype(*values):
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device, value.real.dtype
    return None, torch.float32


def _to_torch_scalar_or_tensor(value, *, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _cell_center_weights_1d(points):
    """Control-volume widths for samples located at Yee cell centers."""
    if isinstance(points, torch.Tensor):
        coords = points.to(dtype=points.real.dtype)
        count = int(coords.numel())
        if count <= 1:
            return torch.ones((count,), device=coords.device, dtype=coords.dtype)
        diffs = coords[1:] - coords[:-1]
        weights = torch.empty((count,), device=coords.device, dtype=coords.dtype)
        weights[0] = diffs[0]
        weights[-1] = diffs[-1]
        if count > 2:
            weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
        return weights

    coords = np.asarray(points, dtype=float)
    count = coords.size
    if count <= 1:
        return np.ones((count,), dtype=float)
    diffs = np.diff(coords)
    weights = np.empty((count,), dtype=float)
    weights[0] = diffs[0]
    weights[-1] = diffs[-1]
    if count > 2:
        weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
    return weights


def _compute_plane_flux(result):
    axis = normalize_axis(result["axis"])
    axis_index = _AXIS_CODES[axis]
    coord_names = _plane_coord_names(axis)
    frequencies = tuple(float(freq) for freq in result.get("frequencies", (result["frequency"],)))
    has_multi_frequency = len(frequencies) > 1

    if _contains_torch_tensor(result):
        device, coord_dtype = _infer_tensor_device_and_dtype(
            result.get(coord_names[0]),
            result.get(coord_names[1]),
            result.get("Ex"),
            result.get("Ey"),
            result.get("Ez"),
            result.get("Hx"),
            result.get("Hy"),
            result.get("Hz"),
        )
        coord_a = _to_torch_scalar_or_tensor(result[coord_names[0]], device=device, dtype=coord_dtype)
        coord_b = _to_torch_scalar_or_tensor(result[coord_names[1]], device=device, dtype=coord_dtype)
        weights = _cell_center_weights_1d(coord_a)[:, None] * _cell_center_weights_1d(coord_b)[None, :]
        field_shape = ((len(frequencies),) if has_multi_frequency else ()) + (coord_a.numel(), coord_b.numel())
        complex_dtype = None
        for component_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            value = result.get(component_name)
            if isinstance(value, torch.Tensor):
                complex_dtype = value.dtype
                break
        if complex_dtype is None:
            complex_dtype = torch.complex64
        e_field = torch.zeros(field_shape + (3,), device=device, dtype=complex_dtype)
        h_field = torch.zeros(field_shape + (3,), device=device, dtype=complex_dtype)
        for component_name in ("Ex", "Ey", "Ez"):
            if component_name not in result:
                continue
            component_index = "xyz".index(component_name[1].lower())
            e_field[..., component_index] = _to_torch_scalar_or_tensor(
                result[component_name],
                device=device,
                dtype=coord_dtype,
            )
        for component_name in ("Hx", "Hy", "Hz"):
            if component_name not in result:
                continue
            component_index = "xyz".index(component_name[1].lower())
            h_field[..., component_index] = _to_torch_scalar_or_tensor(
                result[component_name],
                device=device,
                dtype=coord_dtype,
            )

        direction = 1.0 if result.get("normal_direction", "+") == "+" else -1.0
        poynting = 0.5 * torch.real(torch.cross(e_field, torch.conj(h_field), dim=-1)[..., axis_index]) * direction
        flux = torch.sum(poynting * weights, dim=(-2, -1))
        if not has_multi_frequency:
            return flux.reshape(())
        return flux

    coord_a = np.asarray(result[coord_names[0]], dtype=float)
    coord_b = np.asarray(result[coord_names[1]], dtype=float)
    weights = _cell_center_weights_1d(coord_a)[:, None] * _cell_center_weights_1d(coord_b)[None, :]
    leading_shape = (len(frequencies),) if has_multi_frequency else ()
    field_shape = leading_shape + (coord_a.size, coord_b.size)

    e_field = np.zeros(field_shape + (3,), dtype=np.complex128)
    h_field = np.zeros(field_shape + (3,), dtype=np.complex128)
    for component_name in ("Ex", "Ey", "Ez"):
        if component_name not in result:
            continue
        component_index = "xyz".index(component_name[1].lower())
        values = np.asarray(result[component_name], dtype=np.complex128)
        e_field[..., component_index] = values
    for component_name in ("Hx", "Hy", "Hz"):
        if component_name not in result:
            continue
        component_index = "xyz".index(component_name[1].lower())
        values = np.asarray(result[component_name], dtype=np.complex128)
        h_field[..., component_index] = values

    direction = 1.0 if result.get("normal_direction", "+") == "+" else -1.0
    poynting = 0.5 * np.real(np.cross(e_field, np.conj(h_field), axis=-1)[..., axis_index]) * direction
    flux = np.sum(poynting * weights, axis=(-2, -1))
    if not has_multi_frequency:
        return float(flux)
    return flux


def clear_observers(solver):
    solver.observers = []
    solver.observers_enabled = False
    solver.observer_frequency = None
    solver.observer_frequencies = ()
    solver.observer_start_step = None
    solver.observer_start_steps = ()
    solver.observer_end_step = None
    solver.observer_end_steps = ()
    solver.observer_window_normalization = 0.0
    solver.observer_sample_count = 0
    solver.observer_sample_counts = ()
    solver._point_observer_groups = {}
    solver._plane_observer_groups = {}
    solver._observer_spectral_entries = []
    solver._observer_point_groups_by_frequency = []
    solver._observer_plane_groups_by_frequency = []


def add_point_observer(solver, name, position, component="Ez"):
    solver.observers.append(_point_observer_record(name, position, component=component))


def add_plane_observer(solver, name, axis="z", position=0.0, component="Ez"):
    solver.observers.append(_plane_observer_record(name, axis=axis, position=position, component=component))


def get_component_coords(solver, component):
    return component_coords(solver.scene, component)


def resolve_point_observer(solver, observer):
    x_coords, y_coords, z_coords = get_component_coords(solver, observer["component"])
    px, py, pz = observer["position"]
    return (
        int(np.argmin(np.abs(x_coords - px))),
        int(np.argmin(np.abs(y_coords - py))),
        int(np.argmin(np.abs(z_coords - pz))),
    )


def _plane_field_slice(axis, plane_index):
    if axis == "x":
        return (slice(plane_index, plane_index + 1), slice(None), slice(None))
    if axis == "y":
        return (slice(None), slice(plane_index, plane_index + 1), slice(None))
    return (slice(None), slice(None), slice(plane_index, plane_index + 1))


def _resolve_plane_sample_positions(coords, position):
    axis_coords = np.asarray(coords, dtype=np.float64)
    if axis_coords.ndim != 1 or axis_coords.size == 0:
        raise ValueError("Plane observer axis coordinates must be a non-empty 1D array.")

    target = float(position)
    if axis_coords.size == 1:
        return (
            {
                "plane_index": 0,
                "plane_position": float(axis_coords[0]),
                "weight": 1.0,
            },
        )

    upper = int(np.searchsorted(axis_coords, target, side="left"))
    if upper <= 0:
        return (
            {
                "plane_index": 0,
                "plane_position": float(axis_coords[0]),
                "weight": 1.0,
            },
        )
    if upper >= axis_coords.size:
        last = int(axis_coords.size - 1)
        return (
            {
                "plane_index": last,
                "plane_position": float(axis_coords[last]),
                "weight": 1.0,
            },
        )

    lower = upper - 1
    lower_coord = float(axis_coords[lower])
    upper_coord = float(axis_coords[upper])
    tolerance = 1e-12 * max(abs(target), abs(lower_coord), abs(upper_coord), 1.0)
    if abs(target - lower_coord) <= tolerance:
        return (
            {
                "plane_index": lower,
                "plane_position": lower_coord,
                "weight": 1.0,
            },
        )
    if abs(target - upper_coord) <= tolerance:
        return (
            {
                "plane_index": upper,
                "plane_position": upper_coord,
                "weight": 1.0,
            },
        )

    span = upper_coord - lower_coord
    if span <= tolerance:
        nearest = lower if abs(target - lower_coord) <= abs(target - upper_coord) else upper
        return (
            {
                "plane_index": nearest,
                "plane_position": float(axis_coords[nearest]),
                "weight": 1.0,
            },
        )

    upper_weight = (target - lower_coord) / span
    lower_weight = 1.0 - upper_weight
    return (
        {
            "plane_index": lower,
            "plane_position": lower_coord,
            "weight": float(lower_weight),
        },
        {
            "plane_index": upper,
            "plane_position": upper_coord,
            "weight": float(upper_weight),
        },
    )


def resolve_plane_observer(solver, observer):
    axis = normalize_axis(observer["axis"])
    x_coords, y_coords, z_coords = get_component_coords(solver, observer["component"])
    coord_map = {"x": x_coords, "y": y_coords, "z": z_coords}
    if axis not in coord_map:
        raise ValueError("Observer axis must be 'x', 'y', or 'z'")

    plane_samples = _resolve_plane_sample_positions(coord_map[axis], observer["position"])
    if axis == "x":
        return plane_samples, (y_coords, z_coords)
    if axis == "y":
        return plane_samples, (x_coords, z_coords)
    return plane_samples, (x_coords, y_coords)


def _flux_yee_plane_position(solver, axis, requested_position):
    """Snap a flux plane to the nearest tangential-E Yee plane.

    Tangential electric components live on primal nodes along the plane normal;
    magnetic components then straddle that node symmetrically and are averaged by
    the ordinary plane sampler.  Sampling both families independently at an
    arbitrary coordinate would attenuate each complex phasor by a position-
    dependent interpolation factor.
    """
    axis_name = normalize_axis(axis)
    nodes = {
        "x": solver.scene.x_nodes64,
        "y": solver.scene.y_nodes64,
        "z": solver.scene.z_nodes64,
    }[axis_name]
    index = int(np.argmin(np.abs(np.asarray(nodes, dtype=np.float64) - float(requested_position))))
    return float(nodes[index])


def prepare_observers(solver, frequencies, window_type, time_steps):
    default_frequencies = _normalize_frequency_list(frequencies)
    if not solver.observers:
        solver.observers_enabled = False
        solver._point_observer_groups = {}
        solver._plane_observer_groups = {}
        solver._observer_spectral_entries = []
        solver._observer_point_groups_by_frequency = []
        solver._observer_plane_groups_by_frequency = []
        solver._sync_observer_legacy_state()
        return

    solver.observers_enabled = True
    solver.observer_window_type = solver._resolve_spectral_window_type(window_type)
    # DipoleEmissionMonitor needs the source current spectrum (the running DFT of
    # the injected source signal) to form the delivered power, independently of
    # the source-field normalization pass.
    solver._accumulate_source_spectrum = any(
        observer.get("monitor_type") == "dipole_emission" for observer in solver.observers
    )
    solver._point_observer_groups = {}
    solver._plane_observer_groups = {}
    requested_frequencies = _merge_frequency_lists(
        default_frequencies,
        *(
            observer.get("monitor_frequencies")
            for observer in solver.observers
            if observer.get("monitor_frequencies") is not None
        ),
    )
    solver._observer_spectral_entries = []
    for frequency in requested_frequencies:
        omega_dt = 2 * np.pi * frequency * solver.dt
        solver._observer_spectral_entries.append(
            {
                "frequency": float(frequency),
                "start_step": solver._compute_spectral_start_step(
                    frequency,
                    window_type=solver.observer_window_type,
                ),
                "end_step": time_steps,
                "window_normalization": 0.0,
                "sample_count": 0,
                "phase_cos": 1.0,
                "phase_sin": 0.0,
                "phase_step_cos": np.cos(omega_dt),
                "phase_step_sin": np.sin(omega_dt),
                "source_dft_real": 0.0,
                "source_dft_imag": 0.0,
            }
        )
    frequency_to_index = {entry["frequency"]: index for index, entry in enumerate(solver._observer_spectral_entries)}

    for observer in solver.observers:
        observer_frequencies = _normalize_frequency_list(observer.get("monitor_frequencies")) or default_frequencies
        observer["frequencies"] = observer_frequencies
        observer["global_freq_indices"] = tuple(frequency_to_index[freq] for freq in observer_frequencies)
        component = normalize_component(observer["component"])
        observer["component"] = component
        field = getattr(solver, _field_name(component))
        if _observer_is_point(observer):
            field_index = resolve_point_observer(solver, observer)
            observer["field_index"] = field_index
            group = solver._point_observer_groups.setdefault(
                component,
                {
                    "field_name": _field_name(component),
                    "point_i": [],
                    "point_j": [],
                    "point_k": [],
                    "observers": [],
                    "requested_global_freq_indices": [],
                },
            )
            observer["group_component"] = component
            observer["group_offset"] = len(group["observers"])
            group["observers"].append(observer)
            group["point_i"].append(field_index[0])
            group["point_j"].append(field_index[1])
            group["point_k"].append(field_index[2])
            group["requested_global_freq_indices"].extend(observer["global_freq_indices"])
        else:
            sampling_observer = observer
            if observer.get("compute_flux", False):
                sampling_observer = dict(observer)
                sampling_observer["position"] = _flux_yee_plane_position(
                    solver,
                    observer["axis"],
                    observer["position"],
                )
                observer["yee_plane_position"] = sampling_observer["position"]
            plane_samples, plane_coords = resolve_plane_observer(solver, sampling_observer)
            axis = normalize_axis(observer["axis"])
            observer["plane_samples"] = []
            for sample in plane_samples:
                plane_index = sample["plane_index"]
                group_key = (component, axis, plane_index)
                group = solver._plane_observer_groups.get(group_key)
                if group is None:
                    group = {
                        "field_name": _field_name(component),
                        "axis": axis,
                        "axis_code": _AXIS_CODES[axis],
                        "plane_index": plane_index,
                        "plane_shape": _plane_shape(field, axis),
                        "plane_coords": plane_coords,
                        "field_slice": _plane_field_slice(axis, plane_index),
                        "observers": [],
                        "requested_global_freq_indices": [],
                    }
                    solver._plane_observer_groups[group_key] = group
                group["observers"].append(observer)
                group["requested_global_freq_indices"].extend(observer["global_freq_indices"])
                observer["plane_samples"].append(
                    {
                        "group_key": group_key,
                        "plane_index": plane_index,
                        "plane_position": sample["plane_position"],
                        "weight": sample["weight"],
                    }
                )
            primary_sample = max(
                observer["plane_samples"],
                key=lambda sample: (sample["weight"], -abs(sample["plane_position"] - float(observer["position"]))),
            )
            observer["group_key"] = primary_sample["group_key"]
            observer["plane_index"] = primary_sample["plane_index"]
            observer["plane_indices"] = tuple(sample["plane_index"] for sample in observer["plane_samples"])
            observer["plane_weights"] = tuple(float(sample["weight"]) for sample in observer["plane_samples"])
            observer["plane_positions"] = tuple(float(sample["plane_position"]) for sample in observer["plane_samples"])
            observer["field_slice"] = _plane_field_slice(axis, observer["plane_index"])
            observer["plane_coords"] = plane_coords

    solver._observer_point_groups_by_frequency = [[] for _ in solver._observer_spectral_entries]
    for group in solver._point_observer_groups.values():
        point_count = len(group["observers"])
        field = getattr(solver, group["field_name"])
        group["global_freq_indices"] = tuple(dict.fromkeys(group.pop("requested_global_freq_indices")))
        group["frequencies"] = tuple(
            solver._observer_spectral_entries[index]["frequency"]
            for index in group["global_freq_indices"]
        )
        group["freq_local_lookup"] = {
            global_index: local_index
            for local_index, global_index in enumerate(group["global_freq_indices"])
        }
        group["point_i"] = torch.tensor(group["point_i"], device=solver.device, dtype=torch.int32)
        group["point_j"] = torch.tensor(group["point_j"], device=solver.device, dtype=torch.int32)
        group["point_k"] = torch.tensor(group["point_k"], device=solver.device, dtype=torch.int32)
        group["real"] = torch.zeros((len(group["global_freq_indices"]), point_count), device=solver.device, dtype=field.dtype)
        group["imag"] = torch.zeros((len(group["global_freq_indices"]), point_count), device=solver.device, dtype=field.dtype)
        if has_complex_fields(solver):
            group["aux_real"] = torch.zeros((len(group["global_freq_indices"]), point_count), device=solver.device, dtype=field.dtype)
            group["aux_imag"] = torch.zeros((len(group["global_freq_indices"]), point_count), device=solver.device, dtype=field.dtype)
        for global_index, local_index in group["freq_local_lookup"].items():
            solver._observer_point_groups_by_frequency[global_index].append((group, local_index))

    solver._observer_plane_groups_by_frequency = [[] for _ in solver._observer_spectral_entries]
    for group in solver._plane_observer_groups.values():
        field = getattr(solver, group["field_name"])
        group["global_freq_indices"] = tuple(dict.fromkeys(group.pop("requested_global_freq_indices")))
        group["frequencies"] = tuple(
            solver._observer_spectral_entries[index]["frequency"]
            for index in group["global_freq_indices"]
        )
        group["freq_local_lookup"] = {
            global_index: local_index
            for local_index, global_index in enumerate(group["global_freq_indices"])
        }
        group["real"] = torch.zeros((len(group["global_freq_indices"]),) + group["plane_shape"], device=solver.device, dtype=field.dtype)
        group["imag"] = torch.zeros((len(group["global_freq_indices"]),) + group["plane_shape"], device=solver.device, dtype=field.dtype)
        if has_complex_fields(solver):
            group["aux_real"] = torch.zeros((len(group["global_freq_indices"]),) + group["plane_shape"], device=solver.device, dtype=field.dtype)
            group["aux_imag"] = torch.zeros((len(group["global_freq_indices"]),) + group["plane_shape"], device=solver.device, dtype=field.dtype)
        for global_index, local_index in group["freq_local_lookup"].items():
            solver._observer_plane_groups_by_frequency[global_index].append((group, local_index))

    solver._sync_observer_legacy_state()


def accumulate_observers(solver, n, phase_cos=None, phase_sin=None):
    if not solver.observers_enabled:
        return

    source_signal = None
    accumulate_source = getattr(solver, '_normalize_source', False) or getattr(solver, '_accumulate_source_spectrum', False)
    if accumulate_source and getattr(solver, '_source_time', None) is not None:
        from ..sources import evaluate_source_time
        source_signal = evaluate_source_time(solver._source_time, n * solver.dt)

    for global_index, entry in enumerate(solver._observer_spectral_entries):
        if n < entry["start_step"]:
            continue
        if entry["end_step"] is not None and n >= entry["end_step"]:
            continue

        window_weight = solver._compute_window_weight(
            n,
            start_step=entry["start_step"],
            end_step=entry["end_step"],
            window_type=solver.observer_window_type,
        )
        weighted_cos = window_weight * entry["phase_cos"]
        weighted_sin = window_weight * entry["phase_sin"]

        if source_signal is not None:
            entry["source_dft_real"] += source_signal * weighted_cos
            entry["source_dft_imag"] += source_signal * weighted_sin

        for group, local_index in solver._observer_point_groups_by_frequency[global_index]:
            solver.fdtd_module.accumulatePointObservers3D(
                field=getattr(solver, group["field_name"]),
                pointI=group["point_i"],
                pointJ=group["point_j"],
                pointK=group["point_k"],
                realAccum=group["real"][local_index],
                imagAccum=group["imag"][local_index],
                weightedCos=weighted_cos,
                weightedSin=weighted_sin,
            ).launchRaw()
            if has_complex_fields(solver):
                solver.fdtd_module.accumulatePointObservers3D(
                    field=getattr(solver, f"{group['field_name']}_imag"),
                    pointI=group["point_i"],
                    pointJ=group["point_j"],
                    pointK=group["point_k"],
                    realAccum=group["aux_real"][local_index],
                    imagAccum=group["aux_imag"][local_index],
                    weightedCos=weighted_cos,
                    weightedSin=weighted_sin,
                ).launchRaw()

        for group, local_index in solver._observer_plane_groups_by_frequency[global_index]:
            solver.fdtd_module.accumulatePlaneObserver3D(
                field=getattr(solver, group["field_name"]),
                planeRealAccum=group["real"][local_index],
                planeImagAccum=group["imag"][local_index],
                axisCode=group["axis_code"],
                planeIndex=group["plane_index"],
                weightedCos=weighted_cos,
                weightedSin=weighted_sin,
            ).launchRaw()
            if has_complex_fields(solver):
                solver.fdtd_module.accumulatePlaneObserver3D(
                    field=getattr(solver, f"{group['field_name']}_imag"),
                    planeRealAccum=group["aux_real"][local_index],
                    planeImagAccum=group["aux_imag"][local_index],
                    axisCode=group["axis_code"],
                    planeIndex=group["plane_index"],
                    weightedCos=weighted_cos,
                    weightedSin=weighted_sin,
                ).launchRaw()

        entry["window_normalization"] += window_weight
        entry["sample_count"] += 1

    for entry in solver._observer_spectral_entries:
        entry["phase_cos"], entry["phase_sin"] = solver._advance_phase(
            entry["phase_cos"],
            entry["phase_sin"],
            entry["phase_step_cos"],
            entry["phase_step_sin"],
        )
    solver._sync_observer_primary_state()


# ---------------------------------------------------------------------------
# Time-domain observers (raw time-series fields and instantaneous flux).
#
# These are intentionally kept separate from the running-DFT spectral path:
# they preallocate GPU buffers and copy the current real field slice on a fixed
# sampling schedule, rather than accumulating phase-weighted spectral sums.
# ---------------------------------------------------------------------------


def _time_sample_steps(record, time_steps):
    start = int(record["start"])
    stop = time_steps if record["stop"] is None else int(record["stop"])
    stop = min(stop, int(time_steps))
    interval = int(record["interval"])
    return [n for n in range(start, stop) if (n - start) % interval == 0]


def _primary_plane_index(plane_samples):
    primary = max(plane_samples, key=lambda sample: sample["weight"])
    return int(primary["plane_index"])


def _resolve_time_plane_component(solver, axis, position, component):
    observer = {"axis": axis, "position": float(position), "component": component}
    plane_samples, plane_coords = resolve_plane_observer(solver, observer)
    axis_name = normalize_axis(axis)
    plane_index = _primary_plane_index(plane_samples)
    field_name = _field_name(component)
    field = getattr(solver, field_name)
    return {
        "field_name": field_name,
        "axis_code": _AXIS_CODES[axis_name],
        "plane_index": plane_index,
        "field_slice": _plane_field_slice(axis_name, plane_index),
        "plane_coords": plane_coords,
        "plane_shape": _plane_shape(field, axis_name),
    }


def _resolve_time_volume_box(solver, component, position, size):
    x_coords, y_coords, z_coords = get_component_coords(solver, component)
    axis_coords = (
        np.asarray(x_coords, dtype=np.float64),
        np.asarray(y_coords, dtype=np.float64),
        np.asarray(z_coords, dtype=np.float64),
    )
    slices = []
    coords_out = []
    for axis_index in range(3):
        coords = axis_coords[axis_index]
        center = float(position[axis_index])
        half = 0.5 * float(size[axis_index])
        if half <= 1e-12:
            index = int(np.argmin(np.abs(coords - center)))
            slices.append(slice(index, index + 1))
            coords_out.append(coords[index:index + 1])
            continue
        tolerance = 1e-12 * max(abs(center - half), abs(center + half), 1.0)
        mask = (coords >= center - half - tolerance) & (coords <= center + half + tolerance)
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            index = int(np.argmin(np.abs(coords - center)))
            slices.append(slice(index, index + 1))
            coords_out.append(coords[index:index + 1])
            continue
        lower = int(indices[0])
        upper = int(indices[-1]) + 1
        slices.append(slice(lower, upper))
        coords_out.append(coords[lower:upper])
    return tuple(slices), tuple(coords_out)


def clear_time_observers(solver):
    solver.time_observers = []
    solver.time_observers_enabled = False


def prepare_time_observers(solver, time_steps):
    records = getattr(solver, "time_observers", None)
    if not records:
        solver.time_observers_enabled = False
        return

    for record in records:
        sample_steps = _time_sample_steps(record, time_steps)
        record["sample_steps"] = sample_steps
        record["next_index"] = 0
        num_samples = len(sample_steps)

        if record["kind"] == "field_time":
            region_kind = record["region_kind"]
            record["buffers"] = {}
            record["component_meta"] = {}
            if region_kind == "point":
                for component in record["components"]:
                    field = getattr(solver, _field_name(component))
                    index = resolve_point_observer(
                        solver,
                        {"component": component, "position": record["position"]},
                    )
                    record["component_meta"][component] = {"field_index": index}
                    record["buffers"][component] = torch.zeros(
                        (num_samples,), device=solver.device, dtype=field.dtype
                    )
            elif region_kind == "plane":
                for component in record["components"]:
                    meta = _resolve_time_plane_component(
                        solver,
                        record["axis"],
                        record["plane_position"],
                        component,
                    )
                    record["component_meta"][component] = meta
                    field = getattr(solver, meta["field_name"])
                    record["buffers"][component] = torch.zeros(
                        (num_samples,) + meta["plane_shape"], device=solver.device, dtype=field.dtype
                    )
            else:
                warnings.warn(
                    f"FieldTimeMonitor {record['name']!r} records a volume time series; "
                    "this preallocates a dense (num_samples, Nx, Ny, Nz) GPU buffer and can be memory-heavy.",
                    stacklevel=2,
                )
                for component in record["components"]:
                    field = getattr(solver, _field_name(component))
                    slices, coords = _resolve_time_volume_box(
                        solver, component, record["position"], record["size"]
                    )
                    sub_shape = tuple(field[slices].shape)
                    record["component_meta"][component] = {
                        "field_name": _field_name(component),
                        "slices": slices,
                        "coords": coords,
                    }
                    record["buffers"][component] = torch.zeros(
                        (num_samples,) + sub_shape, device=solver.device, dtype=field.dtype
                    )
        else:
            record["flux_components"] = {
                component: _resolve_time_plane_component(
                    solver, record["axis"], record["position"], component
                )
                for component in record["fields"]
            }
            record["buffer"] = torch.zeros((num_samples,), device=solver.device, dtype=torch.float32)
            # Precompute the constant trapezoidal area weights on the aligned
            # plane grid plus the kernel's field mapping / sign, so the per-step
            # flux is a single native reduction with no host-side work. The plane
            # coordinates depend only on the grid, not on the evolving fields.
            axis_name = normalize_axis(record["axis"])
            rules = _PLANE_ALIGNMENT_RULES[axis_name]
            coord_names = _plane_coord_names(axis_name)
            coord_payloads = {
                component: {"coords": meta["plane_coords"]}
                for component, meta in record["flux_components"].items()
            }
            coord_a = _resolve_coord_source(coord_payloads, *rules["coord_sources"][coord_names[0]])
            coord_b = _resolve_coord_source(coord_payloads, *rules["coord_sources"][coord_names[1]])
            weights = (
                _cell_center_weights_1d(np.asarray(coord_a, dtype=float))[:, None]
                * _cell_center_weights_1d(np.asarray(coord_b, dtype=float))[None, :]
            )
            physical_a, physical_b = _physical_plane_masks(
                solver,
                axis_name,
                coord_a,
                coord_b,
            )
            weights *= physical_a[:, None] * physical_b[None, :]
            record["flux_weights"] = torch.as_tensor(
                weights, device=solver.device, dtype=torch.float32
            ).contiguous()
            record["flux_kernel_components"] = _FLUX_KERNEL_COMPONENTS[axis_name]
            # scale = 2.0 (undo the 0.5 time-average) * 0.5 (Poynting) * direction.
            record["flux_scale"] = 1.0 if record["normal_direction"] == "+" else -1.0

    solver.time_observers_enabled = any(records)


def accumulate_time_observers(solver, n):
    if not getattr(solver, "time_observers_enabled", False):
        return

    for record in solver.time_observers:
        sample_steps = record["sample_steps"]
        next_index = record["next_index"]
        if next_index >= len(sample_steps) or n != sample_steps[next_index]:
            continue
        k = next_index
        record["next_index"] = next_index + 1

        if record["kind"] == "field_time":
            region_kind = record["region_kind"]
            for component, meta in record["component_meta"].items():
                field = getattr(solver, _field_name(component))
                buffer = record["buffers"][component]
                if region_kind == "point":
                    i, j, l = meta["field_index"]
                    buffer[k] = field[i, j, l]
                elif region_kind == "plane":
                    buffer[k] = field[meta["field_slice"]].squeeze(meta["axis_code"])
                else:
                    buffer[k] = field[meta["slices"]]
            continue

        axis = record["axis"]
        component_payloads = {}
        for component, meta in record["flux_components"].items():
            field = getattr(solver, meta["field_name"])
            plane = field[meta["field_slice"]].squeeze(meta["axis_code"])
            component_payloads[component] = {"data": plane, "coords": meta["plane_coords"]}
        # Yee-average the tangential slices onto the common plane grid, then
        # reduce the instantaneous Poynting flux in a single native kernel that
        # writes the scalar straight into this step's buffer slot.
        aligned = _align_plane_monitor_payload(axis, component_payloads)["fields"]
        ea_name, eb_name, ha_name, hb_name = record["flux_kernel_components"]
        solver.fdtd_module.planeFluxReduce(
            ea=aligned[ea_name].contiguous(),
            eb=aligned[eb_name].contiguous(),
            ha=aligned[ha_name].contiguous(),
            hb=aligned[hb_name].contiguous(),
            weights=record["flux_weights"],
            out=record["buffer"],
            outIndex=k,
            scale=record["flux_scale"],
        ).launchRaw()


def get_time_observer_results(solver):
    results = {}
    for record in getattr(solver, "time_observers", ()):
        sample_steps = record.get("sample_steps", [])
        # An early auto-shutoff break freezes next_index below the planned sample
        # count; truncate t and the buffers to what was actually recorded so the
        # trace never carries zeros at time steps that were never simulated.
        count = int(record.get("next_index", len(sample_steps)))
        t = torch.tensor(sample_steps[:count], device=solver.device, dtype=torch.float64) * solver.dt
        if record["kind"] == "field_time":
            components = {name: buffer[:count] for name, buffer in record["buffers"].items()}
            payload = {
                "kind": "field_time",
                "name": record["name"],
                "t": t,
                "components": dict(components),
                "fields": tuple(record["components"]),
                "start": record["start"],
                "stop": record["stop"],
                "interval": record["interval"],
                "position": record["position"],
                "size": record["size"],
            }
            first_component = record["components"][0]
            first_meta = record["component_meta"][first_component]
            if record["region_kind"] == "plane":
                payload["coords"] = first_meta["plane_coords"]
            elif record["region_kind"] == "volume":
                payload["coords"] = first_meta["coords"]
            if len(record["components"]) == 1:
                buffer = components[first_component]
                payload["data"] = buffer
                payload["field"] = buffer
            results[record["name"]] = payload
        else:
            payload = {
                "kind": "flux_time",
                "name": record["name"],
                "t": t,
                "flux": record["buffer"][:count],
                "axis": record["axis"],
                "position": record["position"],
                "normal_direction": record["normal_direction"],
                "start": record["start"],
                "stop": record["stop"],
                "interval": record["interval"],
            }
            results[record["name"]] = payload
    return results


def _compute_source_spectrum(entries, global_freq_indices, scale_fn):
    """Compute complex source DFT values for the given frequency indices."""
    device, _ = _infer_tensor_device_and_dtype()
    use_torch = False
    for idx in global_freq_indices:
        entry = entries[idx]
        if isinstance(entry.get("source_dft_real"), torch.Tensor) or isinstance(entry.get("source_dft_imag"), torch.Tensor):
            use_torch = True
            device = entry.get("source_dft_real", entry.get("source_dft_imag")).device
            break

    if use_torch:
        spectrum = torch.empty((len(global_freq_indices),), device=device, dtype=torch.complex64)
        for i, idx in enumerate(global_freq_indices):
            entry = entries[idx]
            scale = float(scale_fn(entry))
            real = _to_torch_scalar_or_tensor(entry.get("source_dft_real", 0.0), device=device, dtype=torch.float32)
            imag = _to_torch_scalar_or_tensor(entry.get("source_dft_imag", 0.0), device=device, dtype=torch.float32)
            spectrum[i] = (real.to(dtype=torch.complex64) + 1j * imag.to(dtype=torch.complex64)) * scale
        return spectrum

    spectrum = np.empty(len(global_freq_indices), dtype=np.complex128)
    for i, idx in enumerate(global_freq_indices):
        entry = entries[idx]
        scale = scale_fn(entry)
        spectrum[i] = (entry["source_dft_real"] + 1j * entry["source_dft_imag"]) * scale
    return spectrum


def _combine_plane_sample_data(sample_data):
    if sample_data and any(isinstance(data, torch.Tensor) for _, data in sample_data):
        combined = torch.zeros_like(sample_data[0][1])
        for weight, data in sample_data:
            combined = combined + float(weight) * data
        return combined
    if len(sample_data) == 1:
        return sample_data[0][1]

    combined = np.zeros_like(np.asarray(sample_data[0][1]), dtype=np.complex128)
    for weight, data in sample_data:
        combined += float(weight) * np.asarray(data)
    return combined


def _safe_source_spectrum(spectrum):
    if isinstance(spectrum, torch.Tensor):
        spectrum_array = spectrum.to(dtype=torch.complex64)
        if spectrum_array.ndim == 0 or spectrum_array.numel() == 1:
            scalar = spectrum_array.reshape(-1)[0] if spectrum_array.numel() else spectrum_array.new_zeros(())
            abs_scalar = torch.abs(scalar)
            floor = max(float(abs_scalar.item()) * 1e-10, 1e-30)
            return scalar if float(abs_scalar.item()) > floor else torch.as_tensor(complex(floor, 0.0), device=spectrum_array.device, dtype=spectrum_array.dtype)

        abs_spectrum = torch.abs(spectrum_array)
        peak = float(torch.max(abs_spectrum).item()) if abs_spectrum.numel() > 0 else 0.0
        floor = 1e-10 * peak if peak > 0.0 else 1e-30
        floor_tensor = torch.full_like(spectrum_array, complex(floor, 0.0))
        return torch.where(abs_spectrum > floor, spectrum_array, floor_tensor)

    spectrum_array = np.asarray(spectrum, dtype=np.complex128)
    if spectrum_array.ndim == 0 or spectrum_array.size == 1:
        scalar = complex(spectrum_array.reshape(-1)[0]) if spectrum_array.size else 0.0j
        abs_scalar = abs(scalar)
        floor = max(abs_scalar * 1e-10, 1e-30)
        return scalar if abs_scalar > floor else complex(floor, 0.0)

    abs_spectrum = np.abs(spectrum_array)
    peak = float(np.max(abs_spectrum)) if abs_spectrum.size > 0 else 0.0
    floor = 1e-10 * peak if peak > 0.0 else 1e-30
    return np.where(abs_spectrum > floor, spectrum_array, complex(floor, 0.0))


def _safe_divide_source_spectrum(field_data, spectrum):
    """Divide field data by source spectrum with near-zero clamping."""
    safe_spectrum = _safe_source_spectrum(spectrum)
    if isinstance(field_data, torch.Tensor) or isinstance(safe_spectrum, torch.Tensor):
        field_tensor = field_data if isinstance(field_data, torch.Tensor) else torch.as_tensor(field_data)
        if not isinstance(safe_spectrum, torch.Tensor):
            safe_spectrum = torch.as_tensor(safe_spectrum, device=field_tensor.device, dtype=field_tensor.dtype)
        else:
            safe_spectrum = safe_spectrum.to(device=field_tensor.device, dtype=field_tensor.dtype)
        if safe_spectrum.ndim == 0:
            return field_tensor / safe_spectrum
        if field_tensor.ndim == 0 or (field_tensor.ndim == 1 and field_tensor.shape[0] == safe_spectrum.shape[0]):
            return field_tensor / safe_spectrum
        shape = (safe_spectrum.shape[0],) + (1,) * (field_tensor.ndim - 1)
        return field_tensor / safe_spectrum.reshape(shape)

    if np.isscalar(safe_spectrum) or np.asarray(safe_spectrum).ndim == 0:
        return field_data / safe_spectrum
    if field_data.ndim == 0 or (field_data.ndim == 1 and field_data.shape[0] == safe_spectrum.shape[0]):
        return field_data / safe_spectrum
    # Multi-dimensional: spectrum broadcasts over spatial dims
    shape = (safe_spectrum.shape[0],) + (1,) * (field_data.ndim - 1)
    return field_data / safe_spectrum.reshape(shape)


def _normalize_flux_by_source_spectrum(flux_data, spectrum):
    safe_spectrum = _safe_source_spectrum(spectrum)
    if isinstance(flux_data, torch.Tensor) or isinstance(safe_spectrum, torch.Tensor):
        flux_tensor = flux_data if isinstance(flux_data, torch.Tensor) else torch.as_tensor(flux_data)
        if not isinstance(safe_spectrum, torch.Tensor):
            safe_spectrum = torch.as_tensor(safe_spectrum, device=flux_tensor.device, dtype=torch.complex64)
        else:
            safe_spectrum = safe_spectrum.to(device=flux_tensor.device)
        normalized = flux_tensor / (torch.abs(safe_spectrum) ** 2)
        if normalized.ndim == 0:
            return normalized.reshape(())
        return normalized

    flux_array = np.asarray(flux_data, dtype=np.float64)
    if np.isscalar(safe_spectrum) or np.asarray(safe_spectrum).ndim == 0:
        normalized = flux_array / (abs(safe_spectrum) ** 2)
    else:
        normalized = flux_array / (np.abs(safe_spectrum) ** 2)
    if normalized.ndim == 0:
        return float(normalized)
    return normalized


def _normalize_monitor_result_inplace(result, spectrum):
    for comp_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        if comp_name in result:
            result[comp_name] = _safe_divide_source_spectrum(result[comp_name], spectrum)

    components = result.get("components", {})
    if _monitor_payload_is_point(result):
        for comp_name, value in list(components.items()):
            components[comp_name] = _safe_divide_source_spectrum(value, spectrum)
    else:
        for payload in components.values():
            if isinstance(payload, dict) and "data" in payload:
                payload["data"] = _safe_divide_source_spectrum(payload["data"], spectrum)

    primary_component = result.get("component")
    if primary_component and primary_component in result:
        result["data"] = result[primary_component]
    elif "data" in result:
        result["data"] = _safe_divide_source_spectrum(result["data"], spectrum)

    if "flux" in result:
        result["flux"] = _normalize_flux_by_source_spectrum(result["flux"], spectrum)
    if "power" in result:
        result["power"] = _normalize_flux_by_source_spectrum(result["power"], spectrum)


def get_observer_results(solver):
    if not solver.observers_enabled or not any(entry["sample_count"] > 0 for entry in solver._observer_spectral_entries):
        raise ValueError("No observer samples recorded. Configure observers and run solve().")

    point_cache = {}
    plane_cache = {}
    monitor_source_indices = {}
    results = {}
    for observer in solver.observers:
        monitor_name = observer.get("monitor_name", observer["name"])
        monitor_source_indices.setdefault(monitor_name, observer["global_freq_indices"])
        monitor_fields = tuple(_field_name(field) for field in observer.get("monitor_fields", (observer["component"],)))
        public_component = _field_name(observer["component"])
        monitor_frequencies = tuple(float(freq) for freq in observer["frequencies"])
        observer_samples = np.asarray(
            [solver._observer_spectral_entries[index]["sample_count"] for index in observer["global_freq_indices"]],
            dtype=int,
        )

        if _observer_is_point(observer):
            cache_key = (observer["group_component"], observer["group_offset"])
            if cache_key not in point_cache:
                group = solver._point_observer_groups[observer["group_component"]]
                offset = observer["group_offset"]
                scales = torch.as_tensor(
                    [_spectral_scale(solver._observer_spectral_entries[index]) for index in group["global_freq_indices"]],
                    device=group["real"].device,
                    dtype=group["real"].dtype,
                )
                point_cache[cache_key] = combine_complex_spectral_components(
                    group["real"][:, offset],
                    group["imag"][:, offset],
                    group["aux_real"][:, offset] if "aux_real" in group else None,
                    group["aux_imag"][:, offset] if "aux_imag" in group else None,
                ) * scales
            group = solver._point_observer_groups[observer["group_component"]]
            local_indices = [group["freq_local_lookup"][index] for index in observer["global_freq_indices"]]
            data = point_cache[cache_key][local_indices]
            if len(monitor_frequencies) == 1:
                data = data[0]
        else:
            plane_sample_data = []
            for sample in observer["plane_samples"]:
                cache_key = sample["group_key"]
                if cache_key not in plane_cache:
                    group = solver._plane_observer_groups[cache_key]
                    scales = torch.as_tensor(
                        [_spectral_scale(solver._observer_spectral_entries[index]) for index in group["global_freq_indices"]],
                        device=group["real"].device,
                        dtype=group["real"].dtype,
                    )
                    plane_cache[cache_key] = combine_complex_spectral_components(
                        group["real"],
                        group["imag"],
                        group["aux_real"] if "aux_real" in group else None,
                        group["aux_imag"] if "aux_imag" in group else None,
                    ) * scales[:, None, None]
                group = solver._plane_observer_groups[cache_key]
                local_indices = [group["freq_local_lookup"][index] for index in observer["global_freq_indices"]]
                sample_values = plane_cache[cache_key][local_indices]
                if len(monitor_frequencies) == 1:
                    sample_values = sample_values[0]
                plane_sample_data.append((sample["weight"], sample_values))
            data = _combine_plane_sample_data(plane_sample_data)

        if _observer_is_point(observer):
            entry = results.setdefault(
                monitor_name,
                {
                    "kind": "point",
                    "monitor_type": observer.get("monitor_type", "point"),
                    "fields": monitor_fields,
                    "components": {},
                    "field_indices": {},
                    "samples": observer_samples if len(monitor_frequencies) > 1 else int(observer_samples[0]),
                    "frequency": monitor_frequencies[0],
                    "frequencies": monitor_frequencies,
                    "position": observer["position"],
                    "dipole_polarization": observer.get("dipole_polarization"),
                    "dipole_position": observer.get("dipole_position"),
                    "source_name": observer.get("source_name"),
                },
            )
            entry["components"][public_component] = data
            entry["field_indices"][public_component] = observer["field_index"]
        else:
            entry = results.setdefault(
                monitor_name,
                {
                    "kind": "plane",
                    "monitor_type": observer.get("monitor_type", "plane"),
                    "fields": monitor_fields,
                    "components": {},
                    "plane_indices": {},
                    "samples": observer_samples if len(monitor_frequencies) > 1 else int(observer_samples[0]),
                    "frequency": monitor_frequencies[0],
                    "frequencies": monitor_frequencies,
                    "axis": observer["axis"],
                    "position": observer["position"],
                    "compute_flux": bool(observer.get("compute_flux", False)),
                    "normal_direction": observer.get("normal_direction", "+"),
                    "mode_spec": observer.get("mode_spec"),
                    "diffraction_spec": observer.get("diffraction_spec"),
                },
            )
            entry["components"][public_component] = {
                "data": data,
                "coords": observer["plane_coords"],
                "plane_index": observer["plane_index"],
                "plane_indices": observer.get("plane_indices", (observer["plane_index"],)),
                "plane_weights": observer.get("plane_weights", (1.0,)),
                "plane_positions": observer.get("plane_positions", (float(observer["position"]),)),
            }
            entry["plane_indices"][public_component] = observer["plane_index"]

    for result in results.values():
        primary_component = result["fields"][0]
        if _monitor_payload_is_point(result):
            for component_name, value in result["components"].items():
                result[component_name] = value
            if len(result["components"]) == 1:
                result["component"] = normalize_component(primary_component)
                result["data"] = result["components"][primary_component]
                result["field_index"] = result["field_indices"][primary_component]
            continue

        axis = result["axis"]
        coord_names = _plane_coord_names(axis)
        if len(result["components"]) == 1:
            payload = result["components"][primary_component]
            result["component"] = normalize_component(primary_component)
            result["data"] = payload["data"]
            result["coords"] = payload["coords"]
            result["plane_index"] = payload["plane_index"]
            result["plane_sample_indices"] = payload.get("plane_indices", (payload["plane_index"],))
            result["plane_sample_weights"] = payload.get("plane_weights", (1.0,))
            result["plane_sample_positions"] = payload.get("plane_positions", ())
            result[coord_names[0]] = payload["coords"][0]
            result[coord_names[1]] = payload["coords"][1]
            result[primary_component] = payload["data"]
            continue

        aligned = _align_plane_monitor_payload(axis, result["components"])
        if aligned is None:
            if result.get("compute_flux"):
                raise ValueError(
                    f"Monitor {result!r} is missing aligned tangential fields required for flux integration."
                )
            continue
        if result.get("compute_flux"):
            aligned = _crop_aligned_plane_to_physical_bounds(solver, axis, aligned)
        result[coord_names[0]] = aligned[coord_names[0]]
        result[coord_names[1]] = aligned[coord_names[1]]
        result["coords"] = (aligned[coord_names[0]], aligned[coord_names[1]])
        for component_name, value in aligned["fields"].items():
            result[component_name] = value
        if result.get("compute_flux"):
            flux = _compute_plane_flux(result)
            result["flux"] = flux
            result["power"] = flux

    # Dipole-emission power pass: P = -(1/2) Re(conj(J) . E) at the dipole cell.
    # The delivered power is formed from the co-located E DFT and the known
    # source current spectrum, then excluded from the field-normalization pass.
    for monitor_name in list(results.keys()):
        result = results[monitor_name]
        if result.get("monitor_type") != "dipole_emission":
            continue
        global_freq_indices = monitor_source_indices.pop(monitor_name, None)
        if global_freq_indices is None:
            continue
        current_spectrum = _compute_source_spectrum(
            solver._observer_spectral_entries,
            global_freq_indices,
            _spectral_scale,
        )
        polarization = result.get("dipole_polarization") or (0.0, 0.0, 0.0)
        components = result.get("components", {})
        e_projected = None
        for (comp_name, weight) in (("Ex", polarization[0]), ("Ey", polarization[1]), ("Ez", polarization[2])):
            if weight == 0.0 or comp_name not in components:
                continue
            term = float(weight) * components[comp_name]
            e_projected = term if e_projected is None else e_projected + term
        if e_projected is None:
            result["power_delivered"] = None
            continue
        if isinstance(e_projected, torch.Tensor):
            current = current_spectrum
            if not isinstance(current, torch.Tensor):
                current = torch.as_tensor(current, device=e_projected.device, dtype=e_projected.dtype)
            else:
                current = current.to(device=e_projected.device, dtype=e_projected.dtype)
            if current.ndim == 0 or e_projected.ndim == 0:
                current = current.reshape(e_projected.shape)
            power = -0.5 * torch.real(torch.conj(current) * e_projected)
        else:
            current = np.asarray(current_spectrum, dtype=np.complex128).reshape(np.asarray(e_projected).shape)
            power = -0.5 * np.real(np.conj(current) * np.asarray(e_projected))
        result["power_delivered"] = power
        result["current_spectrum"] = current_spectrum

    # Source-spectrum normalization pass
    if getattr(solver, '_normalize_source', False) and getattr(solver, '_source_time', None) is not None:
        for monitor_name, global_freq_indices in monitor_source_indices.items():
            result = results.get(monitor_name)
            if result is None:
                continue
            spectrum = _compute_source_spectrum(
                solver._observer_spectral_entries,
                global_freq_indices,
                _spectral_scale,
            )
            if spectrum.size == 1:
                spectrum = spectrum[0]
            _normalize_monitor_result_inplace(result, spectrum)

    return results
