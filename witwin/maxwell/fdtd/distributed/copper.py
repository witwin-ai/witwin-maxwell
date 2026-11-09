"""Unique two-sided foil ownership and sparse peer trace halos for x slabs."""
from __future__ import annotations

import torch

from ...compiler.copper_foil import compile_copper_faces
from ..copper_foil import (
    begin_copper_step, correct_copper_magnetic, advance_copper_electric,
    accumulate_copper_spectra, prepare_copper_spectra, copper_loss_report,
    prepare_copper_magnetic_support, correct_copper_magnetic_support,
)


class DistributedCopperRuntime:
    """Keep every diffusion/ADE state on its Yee electric-edge owner.

    Native electric halos carry the low trace. A sparse peer copy supplies the
    high trace at a left shard's terminal node, whose normal-H curl reaches the
    last owned cell. Halo stencils allocate no diffusion or polarization history.
    """

    def __init__(self, shards, prepared_scene):
        self.shards = tuple(shards)
        self.transfers = []
        owners = {}
        for shard in self.shards:
            for entry in shard.solver._copper_foils:
                origin = shard.layout.component(entry["electric_name"].removesuffix("_imag")).global_origin[0]
                for index, point in enumerate(entry["indices"].cpu().tolist()):
                    point[0] += origin
                    key = (entry["face"].name, entry["electric_name"], tuple(point))
                    if key in owners:
                        raise RuntimeError("A distributed copper trace has multiple evolving state owners.")
                    owners[key] = (shard, entry, index)
        active = {(name, point) for _foil, name, point in owners}
        for geometry in prepared_scene.compile_ports(device="cpu"):
            for point in geometry.voltage_indices.tolist():
                if (geometry.voltage_component, tuple(point)) in active:
                    raise NotImplementedError(
                        f"Distributed port {geometry.port_name!r} overlaps a CopperFoil split trace; "
                        "joint two-trace port elimination is required.")
        for shard in self.shards[:-1]:
            solver, layout = shard.solver, shard.layout
            for face in compile_copper_faces(solver.scene, device=shard.device, include_halos=True):
                if face.axis == 0 or face.component == 0:
                    continue
                name = "E" + "xyz"[face.component]
                stop = layout.component(name).owned_local_slice[0].stop
                indices = face.indices[face.indices[:, 0] == stop]
                if not len(indices):
                    continue
                selector = tuple(indices[:, axis] for axis in range(3))
                indices = indices[getattr(solver, "c" + name.lower() + "_curl")[selector] != 0]
                grouped = {}
                origin = layout.component(name).global_origin[0]
                for destination_index, point in enumerate(indices.cpu().tolist()):
                    point[0] += origin
                    key = (face.name, name, tuple(point))
                    if key not in owners:
                        raise RuntimeError("A distributed copper halo has no matching trace owner.")
                    source, entry, source_index = owners[key]
                    group = grouped.setdefault(id(entry), (source, entry, [], []))
                    group[2].append(source_index)
                    group[3].append(destination_index)
                for source, entry, source_indices, destination_indices in grouped.values():
                    selected = indices[torch.tensor(destination_indices, device=shard.device)]
                    support = prepare_copper_magnetic_support(solver, face, selected)
                    widths = getattr(solver.scene, "d" + "xyz"[face.axis] + "_primal64")
                    support.update(left_width=float(widths[face.normal_index-1])/2,
                                   right_width=float(widths[face.normal_index])/2)
                    count = len(selected)
                    self.transfers.append(dict(source=source, destination=shard, entry=entry, support=support,
                        indices=torch.tensor(source_indices, device=source.device, dtype=torch.int64),
                        send=torch.empty(count, device=source.device, dtype=entry["right_e"].dtype),
                        receive=torch.empty(count, device=shard.device, dtype=entry["right_e"].dtype),
                        source_ready=torch.cuda.Event(), destination_ready=torch.cuda.Event(), received=torch.cuda.Event()))
        self.owned_edge_count = len(owners)
        self.shadow_halo_bytes_per_step = sum(transfer["send"].numel()*transfer["send"].element_size()
                                              for transfer in self.transfers)

    def begin(self):
        for shard in self.shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                begin_copper_step(shard.solver)
        for transfer in self.transfers:
            source, destination = transfer["source"], transfer["destination"]
            with torch.cuda.device(source.device), torch.cuda.stream(source.compute_stream):
                torch.index_select(transfer["entry"]["right_e"], 0, transfer["indices"], out=transfer["send"])
                transfer["source_ready"].record(source.compute_stream)
            with torch.cuda.device(destination.device), torch.cuda.stream(destination.compute_stream):
                transfer["destination_ready"].record(destination.compute_stream)
            with torch.cuda.device(destination.device), torch.cuda.stream(destination.communication_stream):
                destination.communication_stream.wait_event(transfer["source_ready"])
                destination.communication_stream.wait_event(transfer["destination_ready"])
                transfer["receive"].copy_(transfer["send"], non_blocking=True)
                transfer["received"].record(destination.communication_stream)

    def correct_magnetic(self):
        for shard in self.shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                correct_copper_magnetic(shard.solver)
        for transfer in self.transfers:
            shard, support = transfer["destination"], transfer["support"]
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                shard.compute_stream.wait_event(transfer["received"])
                left_trace = getattr(shard.solver, support["electric_name"])[support["index"]]
                correct_copper_magnetic_support(shard.solver, support,
                    transfer["receive"].double()-left_trace.double())

    def advance_electric(self):
        for shard in self.shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                advance_copper_electric(shard.solver)
                accumulate_copper_spectra(shard.solver)

    def prepare_spectra(self, steps, frequencies, window):
        for shard in self.shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                prepare_copper_spectra(shard.solver, steps, frequencies, window)

    def loss_report(self):
        reports = []
        for shard in self.shards:
            current = copper_loss_report(shard.solver)
            for name, report in current.items():
                contexts = []
                for entry in shard.solver._copper_foils:
                    if entry["face"].name != name or entry["imaginary"] or "spectral_real" not in entry:
                        continue
                    signature = entry["spectral_signature"]
                    context = dict(dt=float(shard.solver.dt), frequencies=tuple(signature[0]),
                        starts=tuple(signature[1]), total_steps=int(signature[2]), window=signature[3],
                        observed_steps=float(entry["spectral_step"].detach().cpu()),
                        normalization=entry["spectral_norm"].detach().cpu().clone())
                    if contexts:
                        _report_require(_same_report_value(contexts[0], context),
                            "Copper report entries have different actual sampling contexts.")
                    contexts.append(context)
                if contexts:
                    report = dict(report, spectral_context=contexts[0])
                for key in ("bulk_electric_traces", "surface_power_traces"):
                    if key in report:
                        report[key] = [_global_trace(trace, shard.layout.component(trace["component"]).global_origin)
                                       for trace in report[key]]
                reports.append((name, report))
        return _merge_owned_copper_reports(reports)


def _report_require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_report_value(first, second):
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
                and first.dtype == second.dtype and first.device == second.device
                and torch.equal(first, second))
    if isinstance(first, dict) or isinstance(second, dict):
        return (isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys()
                and all(_same_report_value(first[key], second[key]) for key in first))
    return type(first) is type(second) and first == second


def _global_trace(trace, origin):
    indices = trace["indices"]
    _report_require(isinstance(indices, torch.Tensor) and indices.dtype == torch.int64
                    and indices.device.type == "cpu" and indices.ndim == 2 and indices.shape[1] == 3,
                    "Copper report requires typed CPU globalizable [N,3] edge indices.")
    return dict(trace, indices=indices + indices.new_tensor(origin))


def _merge_trace_chunks(chunks, *, surface, frequency_count):
    """Concatenate unique owned physical edges without combining Et side values.

    Diagnostic tensors keep their dtype/device/graph. Only integer ownership
    geometry is sorted; overlapping owners are rejected, even if values match.
    """
    if not chunks:
        return []
    groups = {}
    for trace in chunks:
        groups.setdefault(trace["component"], []).append(trace)
    result = []
    edge_axes = ({"power": -1, "area": 0, "edge_impedance_factor": 0, "lower_bounds": 0, "upper_bounds": 0} if surface else
                 {"normal_half_widths": -1, "electric_fields": -1,
                  "conduction_coefficient": -1, "dispersion_coefficient": -1})
    for component, parts in groups.items():
        if not surface:
            # The native runtime retains one common [2,1] half-width pair.
            # Expand its exact broadcast to owned rows before canonical gather;
            # this is a view, preserves dtype/device/grad, and never infers a fit.
            expanded = []
            for part in parts:
                widths = part.get("normal_half_widths")
                count = len(part["indices"])
                _report_require(isinstance(widths, torch.Tensor) and widths.is_floating_point()
                    and (tuple(widths.shape) == (2, 1) or tuple(widths.shape) == (2, count))
                    and bool(torch.isfinite(widths).all()) and bool((widths > 0).all()),
                    "Copper half widths require finite positive typed [2,1] or [2,N] geometry.")
                if widths.shape == (2, 1):
                    part = dict(part, normal_half_widths=widths.expand(2, len(part["indices"])))
                expanded.append(part)
            parts = expanded
        first = parts[0]
        metadata = {key: value for key, value in first.items() if key not in (*edge_axes, "indices")}
        for part in parts:
            _report_require(part.keys() == first.keys() and
                _same_report_value(metadata, {key: value for key, value in part.items()
                                              if key not in (*edge_axes, "indices")}),
                "Copper trace chunks have incompatible component/normal/typed context.")
            idx = part["indices"]
            _report_require(isinstance(idx, torch.Tensor) and idx.dtype == torch.int64
                and idx.device.type == "cpu" and idx.ndim == 2 and idx.shape[1] == 3
                and bool((idx >= 0).all()), "Copper global trace indices must be nonnegative int64 [N,3].")
            expected_shapes = ({"power": (frequency_count, len(idx)), "area": (len(idx),),
                "edge_impedance_factor": (len(idx),),
                "lower_bounds": (len(idx), 3), "upper_bounds": (len(idx), 3)} if surface else
                {"normal_half_widths": (2, len(idx)), "electric_fields": (frequency_count, 2, len(idx)),
                 "conduction_coefficient": (frequency_count, 2, len(idx)),
                 "dispersion_coefficient": (frequency_count, 2, len(idx))})
            for key, axis in edge_axes.items():
                value, reference = part[key], first[key]
                _report_require(isinstance(value, torch.Tensor) and tuple(value.shape) == expected_shapes[key]
                    and (value.is_complex() if key == "electric_fields" else value.is_floating_point()),
                    "Copper trace array has an invalid typed frequency/side/edge shape.")
                _report_require(isinstance(value, torch.Tensor) and isinstance(reference, torch.Tensor)
                    and value.dtype == reference.dtype and value.device == reference.device
                    and value.ndim == reference.ndim and value.shape[axis] == len(idx)
                    and tuple(value.shape[:axis] + value.shape[axis+1:] if axis >= 0 else value.shape[:-1])
                        == tuple(reference.shape[:axis] + reference.shape[axis+1:] if axis >= 0 else reference.shape[:-1]),
                    "Copper trace arrays require identical dtype/device/non-edge shape and exact owned support.")
        indices = torch.cat([part["indices"] for part in parts], dim=0)
        order = torch.arange(len(indices), dtype=torch.int64)
        for axis in (2, 1, 0):
            order = order.index_select(0, torch.argsort(indices[order, axis], stable=True))
        indices = indices.index_select(0, order)
        _report_require(len(indices) < 2 or not bool((indices[1:] == indices[:-1]).all(-1).any()),
                        "Copper trace gather has duplicate evolving global edge owners.")
        merged = dict(first, indices=indices)
        for key, axis in edge_axes.items():
            value = torch.cat([part[key] for part in parts], dim=axis)
            merged[key] = value.index_select(axis, order.to(value.device))
        result.append(merged)
    return result


def _merge_owned_copper_reports(reports):
    result = {}
    invariant = ("frequency_range", "frequencies", "energy_representation", "coordinate_frame",
                 "periodic_axes", "spectral_context")
    for name, report in reports:
        if report.get("bulk_electric_traces") or report.get("surface_power_traces"):
            frequencies = report.get("frequencies")
            context = report.get("spectral_context")
            _report_require(isinstance(frequencies, torch.Tensor) and frequencies.ndim == 1
                and frequencies.numel() > 0 and frequencies.dtype == torch.float64
                and frequencies.device.type == "cpu" and isinstance(context, dict)
                and set(context) == {"dt", "frequencies", "starts", "total_steps", "window", "observed_steps", "normalization"},
                "Copper actual sampling context requires typed frequencies and complete clock/window/normalization.")
            norm = context["normalization"]
            _report_require(isinstance(norm, torch.Tensor) and norm.dtype == torch.float64
                and norm.device.type == "cpu" and norm.shape == frequencies.shape
                and bool(torch.isfinite(norm).all()) and bool((norm >= 0).all())
                and isinstance(context["frequencies"], tuple)
                and context["frequencies"] == tuple(frequencies.tolist())
                and isinstance(context["starts"], tuple) and len(context["starts"]) == len(frequencies)
                and type(context["total_steps"]) is int and context["total_steps"] >= 0
                and type(context["dt"]) is float and type(context["observed_steps"]) is float
                and bool(torch.isfinite(torch.tensor((context["dt"], context["observed_steps"]), dtype=torch.float64)).all())
                and context["dt"] > 0 and 0 <= context["observed_steps"] <= context["total_steps"]
                and isinstance(context["window"], str),
                "Copper sampling normalization/frequency/count/dt/window context is malformed.")
        if name not in result:
            result[name] = dict(report)
            for key in ("bulk_electric_traces", "surface_power_traces"):
                if key in report:
                    result[name][key] = list(report[key])
            continue
        target = result[name]
        for key in invariant:
            _report_require((key in target) == (key in report) and
                            (key not in target or _same_report_value(target[key], report[key])),
                            "Copper shard reports have incompatible frequency/clock/window/normalization context.")
        for key in ("bulk_electric_traces", "surface_power_traces"):
            _report_require((key in target) == (key in report),
                            "Copper shard reports have inconsistent actual retained trace availability.")
            if key in report:
                target[key].extend(report[key])
        for key in ("dissipated_energy", "stored_energy", "surface_power", "pec_contact_edge_count",
                    "active_tangential_edge_count", "free_edge_trace_count"):
            if key in report:
                target[key] = target[key] + report[key]
        if "spectral_valid" in report:
            target["spectral_valid"] = target["spectral_valid"] & report["spectral_valid"]
        for key in ("max_relative_model_error", "certified_relative_model_error_bound"):
            target[key] = max(target[key], report[key])
    for report in result.values():
        frequencies = report.get("frequencies", ())
        if "bulk_electric_traces" in report:
            report["bulk_electric_traces"] = _merge_trace_chunks(report["bulk_electric_traces"], surface=False, frequency_count=len(frequencies))
        if "surface_power_traces" in report:
            report["surface_power_traces"] = _merge_trace_chunks(report["surface_power_traces"], surface=True, frequency_count=len(frequencies))
    return result
