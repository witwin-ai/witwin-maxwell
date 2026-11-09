"""Verbatim pre-vectorization oracles from 4911c0da. Test-only snapshots."""
# Function bodies below are copied verbatim; globals retain the original helpers.
from witwin.maxwell.fdtd import observers as _observers
from witwin.maxwell.compiler import monitors as _monitors

globals().update({k: v for k, v in vars(_observers).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_monitors).items() if not k.startswith("__")})

def prepare_observers(solver, frequencies, window_type, time_steps):
    observer_dtype = getattr(solver, "_observer_dtype", torch.float32)
    if observer_dtype == torch.float64 and has_complex_fields(solver):
        raise ValueError("Float64 spectral observers currently require real time-domain fields; complex replica ownership is not qualified.")
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
    solver._observer_spectral_entries = [
        build_observer_spectral_entry(
            solver,
            frequency,
            end_step=time_steps,
            window_type=solver.observer_window_type,
        )
        for frequency in requested_frequencies
    ]
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
            _prepare_plane_observer(
                solver.scene, observer, device=solver.device,
                dtype=getattr(solver, "_observer_dtype", solver.Ex.dtype),
            )
            axis = normalize_axis(observer["axis"])
            for sample in observer["plane_samples"]:
                group_key = sample["group_key"]
                plane_index = sample["plane_index"]
                group = solver._plane_observer_groups.get(group_key)
                if group is None:
                    group = {
                        "field_name": _field_name(component),
                        "axis": axis,
                        "axis_code": _AXIS_CODES[axis],
                        "plane_index": plane_index,
                        "plane_shape": _plane_shape(field, axis),
                        "plane_coords": observer["plane_coords"],
                        "field_slice": _plane_field_slice(axis, plane_index),
                        "observers": [],
                        "requested_global_freq_indices": [],
                    }
                    if "sample_weights" in sample:
                        points, weights = sample["sample_points"], sample["sample_weights"]
                        group.update(sample_points=points, sample_weights=weights,
                                     plane_shape=tuple(weights.shape))
                        for dim, label in enumerate("ijk"):
                            group["sample_"+label] = points[:, dim].contiguous()
                    solver._plane_observer_groups[group_key] = group
                group["observers"].append(observer)
                group["requested_global_freq_indices"].extend(observer["global_freq_indices"])

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
        group["real"] = torch.zeros((len(group["global_freq_indices"]), point_count), device=solver.device, dtype=observer_dtype)
        group["imag"] = torch.zeros((len(group["global_freq_indices"]), point_count), device=solver.device, dtype=observer_dtype)
        if has_complex_fields(solver):
            group["aux_real"] = torch.zeros((len(group["global_freq_indices"]), point_count), device=solver.device, dtype=observer_dtype)
            group["aux_imag"] = torch.zeros((len(group["global_freq_indices"]), point_count), device=solver.device, dtype=observer_dtype)
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
        group["real"] = torch.zeros((len(group["global_freq_indices"]),) + group["plane_shape"], device=solver.device, dtype=observer_dtype)
        group["imag"] = torch.zeros((len(group["global_freq_indices"]),) + group["plane_shape"], device=solver.device, dtype=observer_dtype)
        if has_complex_fields(solver):
            group["aux_real"] = torch.zeros((len(group["global_freq_indices"]),) + group["plane_shape"], device=solver.device, dtype=observer_dtype)
            group["aux_imag"] = torch.zeros((len(group["global_freq_indices"]),) + group["plane_shape"], device=solver.device, dtype=observer_dtype)
        for global_index, local_index in group["freq_local_lookup"].items():
            solver._observer_plane_groups_by_frequency[global_index].append((group, local_index))

    solver._sync_observer_legacy_state()

def get_observer_results(solver):
    if not solver.observers_enabled or not any(entry["sample_count"] > 0 for entry in solver._observer_spectral_entries):
        raise ValueError("No observer samples recorded. Configure observers and run solve().")

    point_cache = {}
    point_selection_cache = {}
    plane_cache = {}
    plane_components = {}
    monitor_source_indices = {}
    results = {}
    for observer in solver.observers:
        monitor_name = observer.get("monitor_name", observer["name"])
        monitor_source_indices.setdefault(monitor_name, observer["global_freq_indices"])
        if _observer_is_point(observer):
            cache_key = observer["group_component"]
            group = solver._point_observer_groups[cache_key]
            public_component = group["field_name"]
            if cache_key not in point_cache:
                scales = torch.as_tensor(
                    [_spectral_scale(solver, solver._observer_spectral_entries[index]) for index in group["global_freq_indices"]],
                    device=group["real"].device,
                    dtype=group["real"].dtype,
                )
                point_cache[cache_key] = combine_complex_spectral_components(
                    group["real"],
                    group["imag"],
                    group["aux_real"] if "aux_real" in group else None,
                    group["aux_imag"] if "aux_imag" in group else None,
                ) * scales[:, None]
            selection_key = (cache_key, observer["global_freq_indices"])
            if selection_key not in point_selection_cache:
                local_indices = torch.as_tensor(
                    [group["freq_local_lookup"][index] for index in observer["global_freq_indices"]],
                    device=group["real"].device,
                    dtype=torch.long,
                )
                selected = point_cache[cache_key][local_indices]
                if len(observer["frequencies"]) == 1:
                    selected = selected[0]
                # Split once to avoid a GPU indexing operation per point.
                point_selection_cache[selection_key] = selected.unbind(dim=-1)
            data = point_selection_cache[selection_key][observer["group_offset"]]
        else:
            monitor_fields = tuple(_field_name(field) for field in observer.get("monitor_fields", (observer["component"],)))
            public_component = _field_name(observer["component"])
            monitor_frequencies = tuple(float(freq) for freq in observer["frequencies"])
            observer_samples = np.asarray(
                [solver._observer_spectral_entries[index]["sample_count"] for index in observer["global_freq_indices"]],
                dtype=int,
            )
            plane_sample_data = []
            for sample in observer["plane_samples"]:
                cache_key = sample["group_key"]
                if cache_key not in plane_cache:
                    group = solver._plane_observer_groups[cache_key]
                    scales = torch.as_tensor(
                        [_spectral_scale(solver, solver._observer_spectral_entries[index]) for index in group["global_freq_indices"]],
                        device=group["real"].device,
                        dtype=group["real"].dtype,
                    )
                    values = combine_complex_spectral_components(
                        group["real"],
                        group["imag"],
                        group["aux_real"] if "aux_real" in group else None,
                        group["aux_imag"] if "aux_imag" in group else None,
                    )
                    if "sample_weights" in group:
                        values = (values*group["sample_weights"]).sum(-1)
                    plane_cache[cache_key] = values * scales[:, None, None]
                group = solver._plane_observer_groups[cache_key]
                local_indices = [group["freq_local_lookup"][index] for index in observer["global_freq_indices"]]
                sample_values = plane_cache[cache_key][local_indices]
                if len(monitor_frequencies) == 1:
                    sample_values = sample_values[0]
                plane_sample_data.append((sample["weight"], sample_values))
            data = _combine_plane_sample_data(plane_sample_data)

        if _observer_is_point(observer):
            if monitor_name not in results:
                monitor_fields = tuple(_field_name(field) for field in observer.get("monitor_fields", (observer["component"],)))
                monitor_frequencies = tuple(float(freq) for freq in observer["frequencies"])
                observer_samples = np.asarray(
                    [solver._observer_spectral_entries[index]["sample_count"] for index in observer["global_freq_indices"]],
                    dtype=int,
                )
                results[monitor_name] = {
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
                }
            entry = results[monitor_name]
            entry["components"][public_component] = data
            entry["field_indices"][public_component] = observer["field_index"]
        else:
            entry = results.setdefault(monitor_name, {
                "samples": observer_samples if len(monitor_frequencies) > 1 else int(observer_samples[0]),
                "fields": monitor_fields,
                "axis": observer["axis"],
            })
            plane_components.setdefault(monitor_name, []).append((observer, data))

    for monitor_name, result in results.items():
        primary_component = result["fields"][0]
        if _monitor_payload_is_point(result):
            for component_name, value in result["components"].items():
                result[component_name] = value
            if len(result["components"]) == 1:
                result["component"] = normalize_component(primary_component)
                result["data"] = result["components"][primary_component]
                result["field_index"] = result["field_indices"][primary_component]
            continue

        result.update(_assemble_plane_monitor_payload(
            solver.scene, plane_components[monitor_name], samples=result["samples"],
        ))

    # Dipole-emission power pass: P = -(1/2) Re(conj(J) . E) at the dipole cell.
    # The delivered power is formed from the co-located E DFT and the known
    # source current spectrum, before the common source-normalization pass.
    for monitor_name in list(results.keys()):
        result = results[monitor_name]
        if result.get("monitor_type") != "dipole_emission":
            continue
        global_freq_indices = monitor_source_indices.get(monitor_name)
        if global_freq_indices is None:
            continue
        current_spectrum = _compute_source_spectrum(
            solver._observer_spectral_entries,
            global_freq_indices,
            partial(_spectral_scale, solver),
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
    if getattr(solver, '_normalize_source', False) and getattr(solver, '_normalization_source_time', None) is not None:
        for monitor_name, global_freq_indices in monitor_source_indices.items():
            result = results.get(monitor_name)
            if result is None:
                continue
            spectrum = _compute_source_spectrum(
                solver._observer_spectral_entries,
                global_freq_indices,
                partial(_spectral_scale, solver),
            )
            if spectrum.size == 1:
                spectrum = spectrum[0]
            _normalize_monitor_result_inplace(result, spectrum)

    return results

def compile_fdtd_observers(scene):
    compiled = []
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
            monitor_fields = tuple(normalize_component(component) for component in active)
            for component in monitor_fields:
                observer = _point_observer_record(
                    _internal_observer_name(monitor.name, component),
                    position,
                    component=component,
                )
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
            normalized_fields = tuple(normalize_component(field) for field in monitor.fields)
            for field in normalized_fields:
                observer = _point_observer_record(
                    _internal_observer_name(monitor.name, field),
                    monitor.position,
                    component=field,
                )
                observer["monitor_name"] = monitor.name
                observer["monitor_fields"] = normalized_fields
                compiled.append(observer)
            continue
        if isinstance(monitor, (PlaneMonitor, FinitePlaneMonitor, FluxMonitor, ModeMonitor, DiffractionMonitor)):
            compiled.extend(_compile_plane_observers(monitor))
            continue
        raise ValueError(f"Unsupported monitor type: {type(monitor).__name__}")
    return compiled
