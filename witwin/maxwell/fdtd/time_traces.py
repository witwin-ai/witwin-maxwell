"""Surface-field and auxiliary-state time traces for the FDTD time observers.

Both recorders follow the ``FieldTimeMonitor`` contract: samples are copied on
the device into preallocated buffers on the ``start``/``stop``/``interval``
step schedule, at the same point of the step loop as the other time
observers, and the returned traces are detached.
"""

from __future__ import annotations

import torch

from .boundary import has_complex_fields

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _recording_dtype(solver, dtype):
    if not has_complex_fields(solver):
        return dtype
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def _copy_sample(buffer_row, real, imag):
    if buffer_row.is_complex():
        buffer_row.real.copy_(real)
        buffer_row.imag.copy_(imag)
    else:
        buffer_row.copy_(real)


# ---------------------------------------------------------------------------
# SurfaceFieldTimeMonitor
# ---------------------------------------------------------------------------


def prepare_surface_field_time(solver, record, num_samples):
    from .observers import _resolve_time_volume_box

    face_meta = []
    for face in record["faces"]:
        axis_index = _AXIS_INDEX[face["axis"]]
        components = {}
        for component in record["components"]:
            field = getattr(solver, component)
            slices, coords = _resolve_time_volume_box(solver, component, face["position"], face["size"])
            sub_shape = tuple(field[slices].shape)
            plane_shape = sub_shape[:axis_index] + sub_shape[axis_index + 1:]
            components[component] = {
                "slices": slices,
                "axis_index": axis_index,
                "coords": tuple(coords[index] for index in range(3) if index != axis_index),
                "plane_coordinate": float(coords[axis_index][0]),
                "buffer": torch.zeros(
                    (num_samples,) + plane_shape,
                    device=solver.device,
                    dtype=_recording_dtype(solver, field.dtype),
                ),
            }
        face_meta.append(
            {
                "label": face["label"],
                "axis": face["axis"],
                "normal_direction": face["normal_direction"],
                "components": components,
            }
        )
    record["face_meta"] = face_meta


def accumulate_surface_field_time(solver, record, k):
    complex_fields = has_complex_fields(solver)
    for face in record["face_meta"]:
        for component, meta in face["components"].items():
            selection = meta["slices"]
            real = getattr(solver, component)[selection].squeeze(meta["axis_index"])
            imag = (
                getattr(solver, component + "_imag")[selection].squeeze(meta["axis_index"])
                if complex_fields
                else None
            )
            _copy_sample(meta["buffer"][k], real, imag)


def surface_field_time_payload(record, t, count):
    faces = {}
    for face in record["face_meta"]:
        faces[face["label"]] = {
            "axis": face["axis"],
            "normal_direction": face["normal_direction"],
            "components": {
                component: meta["buffer"][:count].detach() for component, meta in face["components"].items()
            },
            "coords": {component: meta["coords"] for component, meta in face["components"].items()},
            "plane_coordinate": {
                component: meta["plane_coordinate"] for component, meta in face["components"].items()
            },
        }
    return {
        "kind": "surface_field_time",
        "differentiable": False,
        "name": record["name"],
        "t": t,
        "fields": tuple(record["components"]),
        "faces": faces,
        "face_labels": tuple(faces),
        "surface_kind": record["surface_kind"],
        "field_time_offsets_steps": {"E": 0.0, "H": -0.5},
        "start": record["start"],
        "stop": record["stop"],
        "interval": record["interval"],
        "position": record["position"],
        "size": record["size"],
    }


# ---------------------------------------------------------------------------
# AuxFieldTimeMonitor
# ---------------------------------------------------------------------------

_FAMILY_TEMPLATES = {
    "electric": ("_dispersive_templates", "electric_dispersive_enabled"),
    "magnetic": ("_magnetic_dispersive_templates", "magnetic_dispersive_enabled"),
}


def aux_state_entries(solver, family, field_component, tensor_name):
    """Live ADE pole entries holding ``tensor_name`` for one Yee component."""

    templates_attr, enabled_attr = _FAMILY_TEMPLATES[family]
    if not getattr(solver, enabled_attr, False):
        return ()
    component_templates = getattr(solver, templates_attr, {}).get(field_component) or {}
    entries = []
    for model_name in ("debye", "drude", "lorentz"):
        for entry in component_templates.get(model_name, ()):
            if entry.get(tensor_name) is not None:
                entries.append(entry)
    return tuple(entries)


def prepare_aux_field_time(solver, record, num_samples):
    from .observers import _resolve_time_volume_box

    point = record["region_kind"] == "point"
    meta = {}
    for component in record["components"]:
        family, field_component, tensor_name = record["component_sources"][component]
        field = getattr(solver, field_component)
        slices, coords = _resolve_time_volume_box(solver, field_component, record["position"], record["size"])
        sub_shape = () if point else tuple(field[slices].shape)
        meta[component] = {
            "family": family,
            "field_component": field_component,
            "tensor_name": tensor_name,
            "slices": slices,
            "coords": coords,
            "buffer": torch.zeros(
                (num_samples,) + sub_shape,
                device=solver.device,
                dtype=_recording_dtype(solver, field.dtype),
            ),
        }
    record["aux_meta"] = meta


def accumulate_aux_field_time(solver, record, k):
    point = record["region_kind"] == "point"
    for component, meta in record["aux_meta"].items():
        entries = aux_state_entries(solver, meta["family"], meta["field_component"], meta["tensor_name"])
        if not entries:
            continue
        row = meta["buffer"][k]
        selection = meta["slices"]
        imag_name = meta["tensor_name"] + "_imag"
        real = entries[0][meta["tensor_name"]][selection]
        imag = entries[0].get(imag_name)
        imag = None if imag is None else imag[selection]
        for entry in entries[1:]:
            real = real + entry[meta["tensor_name"]][selection]
            if imag is not None:
                imag = imag + entry[imag_name][selection]
        if point:
            real = real.reshape(())
            imag = None if imag is None else imag.reshape(())
        _copy_sample(row, real, imag)


def aux_field_time_payload(solver, record, t, count):
    meta = record["aux_meta"]
    components = {component: item["buffer"][:count].detach() for component, item in meta.items()}
    payload = {
        "kind": "aux_field_time",
        "differentiable": False,
        "name": record["name"],
        "t": t,
        "fields": tuple(record["components"]),
        "components": components,
        "present": {
            component: bool(
                aux_state_entries(solver, item["family"], item["field_component"], item["tensor_name"])
            )
            for component, item in meta.items()
        },
        "coords": {component: item["coords"] for component, item in meta.items()},
        "region_kind": record["region_kind"],
        "field_time_offsets_steps": {"electric": 0.0, "magnetic": -0.5},
        # The ADE state of sample n was advanced with the co-located field of
        # sample n - 1 (it is updated before that step's field update).
        "drive_lag_samples": 1,
        "start": record["start"],
        "stop": record["stop"],
        "interval": record["interval"],
        "position": record["position"],
        "size": record["size"],
    }
    if len(components) == 1:
        payload["data"] = next(iter(components.values()))
    return payload


# ---------------------------------------------------------------------------
# Differentiable-run guard
# ---------------------------------------------------------------------------

_GUARDED_TRACE_KINDS = ("surface_field_time", "aux_field_time")


class _NonDifferentiableTrace(torch.autograd.Function):
    """Identity in forward; refuses to back-propagate into a detached trace."""

    @staticmethod
    def forward(ctx, trace, anchor, monitor_name, kind):
        ctx.monitor_name = monitor_name
        ctx.kind = kind
        return trace.view_as(trace)

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            f"Monitor {ctx.monitor_name!r} ({ctx.kind}) records detached time-domain traces and "
            "cannot appear in a differentiable FDTD objective; use a spectral monitor "
            "(PointMonitor, FinitePlaneMonitor, projection or Gaussian-overlap monitors) instead."
        )


def guard_nondifferentiable_traces(monitors):
    """Mark surface/auxiliary time traces of a differentiable run as non-differentiable.

    The traces stay numerically identical, but any objective that depends on
    them raises a clear error during ``backward()`` instead of silently
    contributing a zero gradient.
    """

    for name, payload in monitors.items():
        if not isinstance(payload, dict) or payload.get("kind") not in _GUARDED_TRACE_KINDS:
            continue
        anchor = torch.zeros((), requires_grad=True)
        kind = payload["kind"]

        def guard(value):
            return _NonDifferentiableTrace.apply(value, anchor, name, kind)

        guarded = dict(payload)
        if kind == "aux_field_time":
            guarded["components"] = {key: guard(value) for key, value in payload["components"].items()}
            if "data" in payload:
                guarded["data"] = next(iter(guarded["components"].values()))
        else:
            guarded["faces"] = {
                label: {**face, "components": {key: guard(value) for key, value in face["components"].items()}}
                for label, face in payload["faces"].items()
            }
        guarded["differentiable"] = False
        monitors[name] = guarded
    return monitors
