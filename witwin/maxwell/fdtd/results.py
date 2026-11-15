"""Typed FDTD results: monitor payloads and volume fields as physics-typed data.

The FDTD runtime records monitors as payload dictionaries (``Result.raw_monitor``);
this module wraps the same tensors in ``FieldData``, ``FluxData``,
``ModeAmplitudeData`` or ``TensorData`` without changing a value. Spectral data
are FDTD DFTs of an ``exp(+i*omega*t)`` sum, i.e. phasors with
``exp(-i*omega*t)`` time dependence (peak phasors for CW drives; the Fourier
transform times the reference frequency for pulses), and carry
``attrs["normalization"]`` ``"raw"`` or, with ``SpectralSampler(normalize_source=True)``,
``"per_unit_source_spectrum"``.

Component and propagation coordinates are plain strings (``"Ez"``,
``"+"``), the values of ``mw.monitors.FieldComponent`` and ``mw.monitors.Direction``;
both are ``StrEnum`` members, so ``sel(component=FieldComponent.EZ)`` and
``sel(propagation=Direction.PLUS)`` select them, as on FDFD results.

Frequency-domain data keep a leading ``f`` dimension, of length one for a
single-frequency monitor, unless a frequency was selected (``frequency=`` or
``freq_index=``), which drops it, as for FDFD. Time traces have a ``t``
dimension instead. Components that share sample points are stacked along a
``component`` dimension; components on different Yee grids come back as a
dict of per-component ``FieldData`` (as FDFD ``Result.fields``), each naming
its component in ``attrs["component"]``.
"""

from __future__ import annotations

import torch

from ..monitors import AUX_FIELD_COMPONENTS
from ..results import FieldData, FluxData, ModeAmplitudeData, TensorData
from ..results.typed import FIELD_KIND_UNITS

PHASOR_CONVENTION = "exp(-i omega t), peak"
_AXES = ("x", "y", "z")
_AUX_UNITS = {"polarization": {"electric": "C/m^2", "magnetic": "V*s/m^2"},
              "current": {"electric": "A/m^2", "magnetic": "V/m^2"}}
# Raw payload keys copied to attrs when present (geometry and recording provenance).
_PLANE_ATTRS = ("axis", "position", "yee_plane_position", "normal_direction", "center", "size", "bounds")
_TIME_ATTRS = ("start", "stop", "interval", "position", "size", "field_time_offsets_steps")
_FRAME_ATTRS = ("field_frame", "coordinate_kind", "coordinate_axes")


def spectral_attrs(normalized: bool) -> dict:
    return {"phasor_convention": PHASOR_CONVENTION,
            "normalization": "per_unit_source_spectrum" if normalized else "raw"}


def component_units(labels) -> tuple[str, dict[str, str]]:
    """``(units, component_units)`` for E/H field labels (also chart labels such as ``E_s``) and ADE state labels."""
    per_component = {}
    for label in labels:
        if label in AUX_FIELD_COMPONENTS:
            family, _field, quantity = AUX_FIELD_COMPONENTS[label]
            per_component[label] = _AUX_UNITS[quantity][family]
        else:
            per_component[label] = FIELD_KIND_UNITS[label[0]]
    kinds = set(per_component.values())
    return (kinds.pop() if len(kinds) == 1 else "mixed"), per_component


def yee_location(label: str) -> str:
    """Native Yee location of one E/H component (E on cell edges, H on cell faces); ADE state sits with its field."""
    label = AUX_FIELD_COMPONENTS[label][1] if label in AUX_FIELD_COMPONENTS else label
    return ("edge_" if label[0] == "E" else "face_") + label[-1].lower()


def _frequency_coordinate(frequencies) -> torch.Tensor:
    return torch.tensor(tuple(float(value) for value in frequencies), dtype=torch.float64)


def _tensor(value) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _spectral_layout(payload, selected: bool):
    """``(has_f, coords)`` for a spectral payload: ``f`` is kept unless a frequency was selected."""
    if selected:
        return False, {}
    return True, {"f": _frequency_coordinate(payload["frequencies"])}


def _with_f(value, has_f: bool, spatial_ndim: int) -> torch.Tensor:
    """A payload value with a leading ``f`` axis when ``has_f``, without one otherwise.

    Solver payloads drop the axis for a single frequency or a selected one;
    externally assembled payloads (``Result.from_fields``) may keep a length-1 axis.
    """
    value = _tensor(value)
    if value.ndim == spatial_ndim:
        return value.unsqueeze(0) if has_f else value
    return value if has_f else value.squeeze(0)


def _copy(payload, keys) -> dict:
    return {key: payload[key] for key in keys if payload.get(key) is not None}


def _field(values, dims, coords, labels, attrs) -> FieldData:
    units, per_component = component_units(labels)
    attrs = {**attrs, "component_units": per_component}
    if "component" not in dims:
        attrs["component"] = labels[0]
    return FieldData(values=values, dims=dims, coords=coords, units=units, attrs=attrs)


def _component_mapping(entries, attrs) -> dict[str, FieldData]:
    """``{label: FieldData}`` for components on different grids: ``entries`` maps label to (values, dims, coords)."""
    return {label: _field(values, dims, coords, (label,), {**attrs, "yee_location": yee_location(label)})
            for label, (values, dims, coords) in entries.items()}


# --- Spectral field monitors ----------------------------------------------------------------
def point_data(payload, *, selected: bool, normalized: bool) -> FieldData:
    """``PointMonitor``: dims ``("component", "f")``; each component at its nearest native Yee sample."""
    labels = tuple(map(str, payload["fields"]))
    has_f, coords = _spectral_layout(payload, selected)
    values = torch.stack([_with_f(payload["components"][label], has_f, 0) for label in labels])
    attrs = {**spectral_attrs(normalized), "yee_location": "native", "position": payload["position"],
             "sample_indices": dict(payload["field_indices"]), **_copy(payload, _FRAME_ATTRS)}
    return _field(values, ("component", "f") if has_f else ("component",),
                  {"component": labels, **coords}, labels, attrs)


def plane_data(payload, *, selected: bool, normalized: bool):
    """``PlaneMonitor``/``FinitePlaneMonitor`` fields: dims ``("component", "f", a, b)`` over the plane axes.

    One component keeps its native Yee samples; several components use the
    common cell-centred (or ``sample_spacing`` quadrature) grid the monitor
    aligns them on. Components the alignment does not cover (the plane-normal
    H component) make the monitor return per-component native ``FieldData``.
    """
    labels = tuple(map(str, payload["fields"]))
    axes = tuple(axis for axis in _AXES if axis != payload["axis"])
    has_f, coords = _spectral_layout(payload, selected)
    dims = (("f",) if has_f else ()) + axes
    attrs = {**spectral_attrs(normalized), **_copy(payload, _PLANE_ATTRS), **_copy(payload, _FRAME_ATTRS)}
    components = payload["components"]
    if len(labels) == 1:
        entry = components[labels[0]]
        values = _with_f(entry["data"], has_f, 2).unsqueeze(0)
        attrs["yee_location"] = "collocated" if entry.get("collocated") else yee_location(labels[0])
        return _field(values, ("component", *dims), {"component": labels, **coords, **dict(zip(axes, entry["coords"]))},
                      labels, attrs)
    if all(label in payload for label in labels) and all(axis in payload for axis in axes):
        values = torch.stack([_with_f(payload[label], has_f, 2) for label in labels])
        attrs["yee_location"] = "collocated" if payload.get("sample_spacing") is not None else "cell_center"
        return _field(values, ("component", *dims), {"component": labels, **coords, **{axis: payload[axis] for axis in axes}},
                      labels, attrs)
    return _component_mapping(
        {label: (_with_f(components[label]["data"], has_f, 2), dims,
                 {**coords, **dict(zip(axes, components[label]["coords"]))}) for label in labels},
        attrs)


def flux_data(payload, *, selected: bool, normalized: bool) -> FluxData:
    """Plane flux (``compute_flux=True``, ``FluxMonitor``) or closed-surface total: dims ``("f",)``, units W."""
    has_f, coords = _spectral_layout(payload, selected)
    values = _tensor(payload["flux"])
    values = values.reshape(-1) if has_f else values.reshape(())
    attrs = {**spectral_attrs(normalized), **_copy(payload, _PLANE_ATTRS)}
    return FluxData(values=values, dims=("f",) if has_f else (), coords=coords, units="W", attrs=attrs)


def dipole_emission_data(payload, *, selected: bool, normalized: bool) -> FluxData:
    """``DipoleEmissionMonitor``: power delivered by the dipole, ``P = -(1/2) Re(conj(J) . E)``, dims ``("f",)``."""
    if payload.get("power_delivered") is None:
        raise ValueError(
            f"DipoleEmissionMonitor {payload.get('source_name')!r} recorded no electric component along the "
            "dipole polarization, so no delivered power exists."
        )
    has_f, coords = _spectral_layout(payload, selected)
    values = _tensor(payload["power_delivered"])
    current = _tensor(payload["current_spectrum"])
    values, current = (values.reshape(-1), current.reshape(-1)) if has_f else (values.reshape(()), current.reshape(()))
    attrs = {**spectral_attrs(normalized), "current_spectrum": current, "polarization": payload.get("dipole_polarization"),
             "position": payload.get("dipole_position", payload.get("position")), "source_name": payload.get("source_name")}
    return FluxData(values=values, dims=("f",) if has_f else (), coords=coords, units="W", attrs=attrs)


def mode_data(modals, frequencies, *, selected: bool, normalized: bool) -> ModeAmplitudeData:
    """``ModeMonitor``: forward/backward amplitudes of the monitor's mode, dims ``("mode", "propagation", "f")``.

    ``modals`` holds one overlap record per frequency. Amplitudes are relative
    to the solved mode profile: ``attrs["power"]`` is ``|a|^2 * mode_power``
    (W) and ``attrs["power_fraction"]`` that power over the net plane power
    (``None`` when the net power is numerically zero at some frequency).
    """
    first = modals[0]
    fractions = [(modal["power_fraction_forward"], modal["power_fraction_backward"]) for modal in modals]
    attrs = {
        **spectral_attrs(normalized),
        "mode_index": first["mode_index"], "polarization_axis": first.get("polarization_axis"),
        "axis": first["axis"], "position": first["position"], "normal_direction": first["normal_direction"],
        "effective_index": tuple(modal["effective_index"] for modal in modals),
        "beta": tuple(modal["beta"] for modal in modals),
        "beta_complex": tuple(modal.get("beta_complex") for modal in modals),
        "mode_power": torch.stack([modal["mode_power"] for modal in modals]),
        "total_power": torch.stack([modal["total_power"] for modal in modals]),
        "power": _per_direction(modals, "power_forward", "power_backward"),
        "power_fraction": None if any(pair[0] is None for pair in fractions) else torch.stack(
            [torch.stack(column) for column in zip(*fractions)]),
    }
    values = _per_direction(modals, "amplitude_forward", "amplitude_backward")
    return _amplitudes(values, frequencies, units="1", attrs=attrs, selected=selected)


def gaussian_overlap_data(overlap, *, selected: bool, normalized: bool) -> ModeAmplitudeData:
    """``GaussianOverlapMonitor``: beam amplitudes in sqrt(W) against the unit-power beam, dims ``("mode", "propagation", "f")``."""
    attrs = {
        **spectral_attrs(normalized),
        "total_power": overlap["total_power"], "background_index": tuple(overlap["background_index"]),
        "power": torch.stack((overlap["power_forward"], overlap["power_backward"])),
        "power_fraction": torch.stack((overlap["power_fraction_forward"], overlap["power_fraction_backward"])),
        "axis": overlap["axis"], "position": overlap["position"], "normal_direction": overlap["normal_direction"],
        "beam_direction": overlap["direction"], "polarization": overlap["polarization"],
        "beam_waist": overlap["beam_waist"], "focus": overlap["focus"], "focus_offsets": overlap["focus_offsets"],
    }
    values = torch.stack((overlap["amplitude_forward"], overlap["amplitude_backward"]))
    return _amplitudes(values, overlap["frequencies"], units="sqrt(W)", attrs=attrs, selected=selected)


def _per_direction(records, forward, backward) -> torch.Tensor:
    # [propagation, f] from one record per frequency.
    return torch.stack((torch.stack([record[forward] for record in records]),
                        torch.stack([record[backward] for record in records])))


def _amplitudes(values, frequencies, *, units, attrs, selected) -> ModeAmplitudeData:
    """One mode: ``values`` and the value-shaped attrs are ``[propagation, f]``; selection drops ``f``."""
    shaped = [key for key in ModeAmplitudeData.VALUE_ATTRS if isinstance(attrs.get(key), torch.Tensor)]
    data = ModeAmplitudeData(
        values=values.unsqueeze(0), dims=("mode", "propagation", "f"),
        coords={"propagation": ("+", "-"), "f": _frequency_coordinate(frequencies)}, units=units,
        attrs={**attrs, **{key: attrs[key].unsqueeze(0) for key in shaped}},
    )
    return data.isel(f=0) if selected else data


def diffraction_data(diffraction, *, selected: bool, normalized: bool) -> FluxData:
    """``DiffractionMonitor``: power per Floquet order, dims ``("order", "f")``, units W.

    Per-order attributes (``m``, ``n``, ``kx``, ``ky``, ``kz``, ``theta``,
    ``phi``, ``propagating``, ``efficiency``, ``amplitude_E``, ``amplitude_H``)
    are selected with ``order``.
    """
    orders = diffraction["orders"]
    values = torch.stack([order["power"] for order in orders])
    attrs = {**spectral_attrs(normalized), **{key: diffraction[key] for key in (
        "axis", "normal_direction", "position", "tangential_axes", "periods", "reciprocal_vectors",
        "background_index", "incident_power", "total_power", "propagating_power", "num_propagating")}}
    for key in FluxData.INDEXED_ATTRS["order"]:
        column = [order[key] for order in orders]
        attrs[key] = torch.stack(column) if key in ("amplitude_E", "amplitude_H") else tuple(column)
    coords = {"order": torch.arange(len(orders))}
    if not selected:
        values = values.unsqueeze(-1)
        coords["f"] = _frequency_coordinate((diffraction["frequency"],))
    return FluxData(values=values, dims=("order", "f") if not selected else ("order",), coords=coords, units="W",
                    attrs=attrs)


# --- Time traces ---------------------------------------------------------------------------
def _region_axes(payload) -> tuple[str, ...]:
    size = payload.get("size") or (0.0, 0.0, 0.0)
    return tuple(axis for axis, extent in zip(_AXES, size) if abs(float(extent)) > 1e-12)


def field_time_data(payload):
    """``FieldTimeMonitor``/``AuxFieldTimeMonitor``: dims ``("component", "t", ...)`` (detached traces).

    Point traces and single-component traces are one ``FieldData``; plane and
    volume traces of several components come back as a dict of per-component
    ``FieldData`` because each component has its own Yee sub-grid.
    ``attrs["field_time_offsets_steps"]`` gives the E/H sample time offsets in
    steps relative to ``t``.
    """
    labels = tuple(map(str, payload["fields"]))
    t = payload["t"]
    attrs = {**_copy(payload, _TIME_ATTRS), "differentiable": False}
    if payload["kind"] == "aux_field_time":
        attrs.update(present=dict(payload["present"]), drive_lag_samples=payload["drive_lag_samples"])
    components = payload["components"]
    # Field traces list per-component coordinates in "component_coords", ADE traces in "coords".
    coordinates_by_label = payload.get("component_coords") if payload["kind"] == "field_time" else payload["coords"]
    entries = {}
    for label in labels:
        trace = components[label]
        # Volume boxes keep all three axes; planes keep their two tangential axes.
        axes = () if trace.ndim == 1 else _AXES if trace.ndim == 4 else _region_axes(payload)
        coordinates = coordinates_by_label[label] if axes else ()
        entries[label] = (trace, ("t", *axes), {"t": t, **dict(zip(axes, coordinates))})
    if len(labels) == 1 or all(trace.ndim == 1 for trace, _dims, _coords in entries.values()):
        trace, dims, coords = entries[labels[0]]
        location = yee_location(labels[0]) if len(labels) == 1 else "native"
        values = torch.stack([entries[label][0] for label in labels])
        return _field(values, ("component", *dims), {"component": labels, **coords}, labels,
                      {**attrs, "yee_location": location})
    return _component_mapping(entries, attrs)


def surface_field_time_data(payload) -> dict[str, dict[str, FieldData]]:
    """``SurfaceFieldTimeMonitor``: ``{face: {component: FieldData}}``, dims ``("t", a, b)`` per face."""
    attrs = {**_copy(payload, _TIME_ATTRS), "differentiable": False, "surface_kind": payload.get("surface_kind")}
    faces = {}
    for label, face in payload["faces"].items():
        axes = tuple(axis for axis in _AXES if axis != face["axis"])
        face_attrs = {**attrs, "axis": face["axis"], "normal_direction": face["normal_direction"]}
        faces[label] = {
            component: _field(trace, ("t", *axes), {"t": payload["t"], **dict(zip(axes, face["coords"][component]))},
                              (component,), {**face_attrs, "yee_location": yee_location(component),
                                             "position": face["plane_coordinate"][component]})
            for component, trace in face["components"].items()
        }
    return faces


def flux_time_data(payload) -> FluxData:
    """``FluxTimeMonitor``: instantaneous plane flux, dims ``("t",)``, units W (detached)."""
    attrs = {**_copy(payload, (*_TIME_ATTRS, "axis", "normal_direction", "yee_plane_position")), "differentiable": False}
    return FluxData(values=payload["flux"], dims=("t",), coords={"t": payload["t"]}, units="W", attrs=attrs)


# --- Material monitors ---------------------------------------------------------------------
_MATERIAL_UNITS = {"eps": "1", "mu": "1", "sigma_e": "S/m"}


def material_data(payload, *, selected: bool) -> dict[str, TensorData]:
    """``PermittivityMonitor`` (``eps``) / ``MediumMonitor`` (``eps``, ``mu``, ``sigma_e``):
    ``{quantity: TensorData}`` with dims ``("component", "f", "x", "y", "z")`` on the cropped cell grid."""
    has_f, coords = _spectral_layout(payload, selected)
    grid = {axis: payload[axis] for axis in _AXES}
    data = {}
    for quantity, units in _MATERIAL_UNITS.items():
        if f"{quantity}_x" not in payload:
            continue
        values = torch.stack([_with_f(payload[f"{quantity}_{axis}"], has_f, 3) for axis in _AXES])
        data[quantity] = TensorData(values=values, dims=("component", *(("f",) if has_f else ()), *_AXES),
                                    coords={"component": _AXES, **coords, **grid}, units=units,
                                    attrs={"bounds": payload["bounds"]})
    return data


# --- Volume fields -------------------------------------------------------------------------
# Physical Yee time of the field held when the DFT samples step ``m`` (weighted at ``m*dt``), in steps
# relative to ``m*dt``: the step updates H to ``(m - 1/2)*dt`` from E^(m-1), then E to ``m*dt``.
VOLUME_FIELD_TIME_OFFSET_STEPS = {"E": 0.0, "H": -0.5}


def volume_field_data(prepared_scene, frequencies, fields, *, normalized: bool,
                      solver_record: bool = True) -> dict[str, FieldData]:
    """``Result.fields`` for FDTD: ``{"EX", "EY", "EZ"}`` (and ``"HX", "HY", "HZ"`` when recorded) DFT
    phasors, dims ``(f, x, y, z)`` on each component's Yee grid.

    Values are views of the stored phasor tensors (no copy); coordinates are the
    prepared float64 nodes and cell midpoints, absorber layers included. A solver
    record (``solver_record``) also states the sample time label: every component
    is weighted at ``t = m*dt`` for the field held after step ``m``, which is E at
    ``m*dt`` (``attrs["field_time_offset_steps"] = 0.0``) and H at ``(m - 1/2)*dt``
    (``-0.5``; see docs/reference/fdtd-typed-results.md).
    """
    from .coords import component_coords

    f = _frequency_coordinate(frequencies)
    attrs = spectral_attrs(normalized)
    return {name: _field(tensor.reshape(f.numel(), *tensor.shape[-3:]), ("f", *_AXES),
                         {"f": f, **dict(zip(_AXES, component_coords(prepared_scene, name, device=prepared_scene.device)))},
                         (name[0] + name[1].lower(),),
                         {**attrs, **({"field_time_offset_steps": VOLUME_FIELD_TIME_OFFSET_STEPS[name[0]]}
                                      if solver_record else {}),
                          "yee_location": yee_location(name)})
            for name, tensor in fields.items()}


def time_snapshot_data(prepared_scene, fields, *, time: float, step: int) -> dict[str, FieldData]:
    """``Result.time_snapshot``: the real fields after the last step, dims ``(x, y, z)`` on each Yee grid."""
    from .coords import component_coords

    return {name: _field(tensor, _AXES, dict(zip(_AXES, component_coords(prepared_scene, name, device=prepared_scene.device))),
                         (name[0] + name[1].lower(),),
                         {"kind": "time_snapshot", "time": time, "step": step, "yee_location": yee_location(name)})
            for name, tensor in fields.items()}


def tfsf_h_mask(result) -> dict[str, torch.Tensor]:
    """Per H component, ``False`` where the H update adds a TFSF incident correction.

    Returns ``{"HX", "HY", "HZ"}`` boolean tensors on the result device, on each H
    component's full computational Yee grid (``HX`` is ``(Nx, Ny - 1, Nz - 1)`` for
    ``N`` primal nodes per axis, absorber layers included, the index space of
    ``Result.fields``). Everywhere else H follows the plain discrete Faraday update
    of the stored E; at the masked samples (tangential H half a cell outside each
    TFSF face) the update also used the incident field, which ``Result.fields``
    does not hold. A scene without a TFSF source returns all-``True`` masks. The
    snapped TFSF node indices are ``result.stats()["tfsf"]["node_indices"]``. Reads
    only the Result, so it works after ``Result.release_solver()`` and on a loaded
    Result.
    """
    stats = result.stats() if result.method == "fdtd" else {}
    if "tfsf" not in stats:
        raise ValueError("tfsf_h_mask requires an FDTD Result that records stats()['tfsf'].")
    prepared = result.prepared_scene
    counts = tuple(len(nodes) for nodes in (prepared.x_nodes64, prepared.y_nodes64, prepared.z_nodes64))
    device = prepared.device
    masks = {f"H{axis.upper()}": torch.ones(tuple(n if i == index else n - 1 for i, n in enumerate(counts)),
                                            dtype=torch.bool, device=device)
             for index, axis in enumerate(_AXES)}
    for block in (stats["tfsf"] or {}).get("h_corrections", ()):
        masks[block["component"].upper()][tuple(slice(s, s + n) for s, n in zip(block["start"], block["shape"]))] = False
    return masks
