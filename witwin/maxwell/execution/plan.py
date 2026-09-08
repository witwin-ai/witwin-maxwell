from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class ExecutionTask:
    """One immutable unit of ensemble work in submission order.

    ``run`` executes the task on the leased device and returns its ``Result``.
    It must be self-contained: the shared execution layer knows nothing about
    electromagnetic semantics, only that a task runs on a device and returns a
    value or raises.
    """

    index: int
    run: Callable[[torch.device], object]
    estimated_bytes: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable ordered task list plus placement and concurrency policy."""

    tasks: tuple[ExecutionTask, ...]
    placement: str
    max_concurrency: int
    fail_fast: bool

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        for expected, task in enumerate(tasks):
            if task.index != expected:
                raise ValueError(
                    "ExecutionPlan tasks must be contiguously indexed in submission order."
                )
        object.__setattr__(self, "tasks", tasks)
        if self.placement not in {"round_robin", "memory_aware"}:
            raise ValueError("placement must be 'round_robin' or 'memory_aware'.")
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int):
            raise TypeError("max_concurrency must be an integer.")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0.")
        if not isinstance(self.fail_fast, bool):
            raise TypeError("fail_fast must be a bool.")

    def __len__(self) -> int:
        return len(self.tasks)


__all__ = ["ExecutionPlan", "ExecutionTask"]


@dataclass(frozen=True)
class BatchPlanReport:
    """Immutable JSON-backed planning snapshot with independent exported values."""

    _encoded: str

    @classmethod
    def from_dict(cls, payload):
        import json
        return cls(json.dumps(payload, sort_keys=True, allow_nan=False))

    def to_dict(self):
        import json
        return json.loads(self._encoded)

    def __getattr__(self, name):
        payload = self.to_dict()
        if name not in payload:
            raise AttributeError(name)
        return payload[name]

    def to_markdown(self):
        data = self.to_dict()
        lines = [
            f"Batch plan: {data['variant_count']} variants; resolved={data['resolved']}",
            "", "| Group | Variant indices | Capability |",
            "| --- | --- | --- |",
        ]
        for group in data["groups"]:
            lines.append(f"| {group['group_id']} | {group['variant_indices']} | {group['capability_status']} |")
        lines.extend(["", *[f"- {warning}" for warning in data["warnings"]]])
        return "\n".join(lines) + "\n"


def _batch_declaration(value, path, unresolved):
    """Serialize declarations only; tensor values require compile-time proof."""
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    from enum import Enum
    import numpy as np

    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        unresolved.append(f"{path}: tensor values need compile-time comparison")
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        # GridSpec owns small, read-only host coordinate masters.
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _batch_declaration(v, f"{path}.{key}", unresolved)
                for key, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_batch_declaration(v, f"{path}[{i}]", unresolved) for i, v in enumerate(value)]
    if is_dataclass(value):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {f.name: _batch_declaration(getattr(value, f.name), f"{path}.{f.name}", unresolved)
                           for f in fields(value)}}
    unresolved.append(f"{path}: {type(value).__name__} requires compilation")
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _batch_first_difference(a, b, path=""):
    if type(a) is not type(b):
        return path
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return path
        for key in a:
            difference = _batch_first_difference(a[key], b[key], f"{path}.{key}" if path else key)
            if difference:
                return difference
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}.length"
        for i, (left, right) in enumerate(zip(a, b)):
            difference = _batch_first_difference(left, right, f"{path}[{i}]")
            if difference:
                return difference
    elif a != b:
        return path
    return None


def _batch_output_plan(scene, outputs, frequencies):
    from ..monitors import PointMonitor

    selected = []
    for monitor in scene.resolved_monitors():
        if outputs.monitors != "all" and monitor.name not in outputs.monitors:
            continue
        monitor_frequencies = getattr(monitor, "frequencies", None) or frequencies
        selected.append({
            "name": monitor.name, "type": type(monitor).__name__,
            "frequencies": list(monitor_frequencies),
            "components": list(getattr(monitor, "fields", ())),
            "sample_count": 1 if isinstance(monitor, PointMonitor) else None,
        })
    return {
        "fields": list(outputs.fields), "full_field_dft": bool(outputs.fields),
        "frequencies": list(frequencies), "monitors": selected,
        "ports": [p.name for p in scene.ports if outputs.ports == "all" or p.name in outputs.ports],
        "materials": outputs.materials, "result_device": outputs.result_device,
        "max_pending_results": outputs.max_pending_results,
        "allocation_status": "requires_compilation",
    }


def build_batch_plan(batch):
    """Build provisional groups without preparing scenes or initializing CUDA.

    Compiled grid, dt, coefficient equality and native capabilities remain
    unresolved. A declaration digest is never used as a static artifact key.
    """
    import hashlib
    import json
    from dataclasses import fields

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

    groups = []
    declarations = []
    by_digest = {}
    warnings = [
        "Dry planning performs no solver allocation; prepare validates the available native forward tier.",
        "Declarative groups are provisional; compiled equality and memory admission remain unresolved.",
        "BatchExecution.graph owns batch capture policy; child cuda_graph flags are not used.",
    ]
    for index, simulation in enumerate(batch.simulations):
        scene = simulation.scene
        unresolved = ["compiled material equality and variation envelope/halo proof",
                      "resolved Yee layout, stability bound and native capability validation",
                      "slot, workspace, graph pool and retained output allocation sizes"]
        if scene.grid.is_auto:
            unresolved.append("adaptive grid requires compilation")
        if simulation.config.run_time.dt is None:
            unresolved.append("actual dt requires compiled material stability bounds")
        output_plan = _batch_output_plan(scene, batch.outputs, simulation.frequencies)
        config = {f.name: getattr(simulation.config, f.name) for f in fields(simulation.config)
                  if f.name not in ("cuda_graph", "enable_plot", "full_field_dft", "parallel")}
        sources = []
        changing_sources = set(batch.parameters.source_scales if batch.parameters is not None
                               else batch.variation.sources if batch.variation else ())
        for source in scene.resolved_sources():
            declaration = _batch_declaration(source, "sources", unresolved)
            if source.name in changing_sources:
                source_time = declaration.get("fields", {}).get("source_time")
                if isinstance(source_time, dict) and "fields" in source_time:
                    source_time["fields"].pop("amplitude", None)
            sources.append(declaration)
        declaration = {
            "domain": _batch_declaration(scene.domain, "domain", unresolved),
            "grid": _batch_declaration(scene.grid, "grid", unresolved),
            "boundary": _batch_declaration(scene.boundary, "boundary", unresolved),
            "symmetry": _batch_declaration(scene.symmetry, "symmetry", unresolved),
            "frequencies": list(simulation.frequencies),
            "config": _batch_declaration(config, "config", unresolved),
            "sources": sources,
            "monitors": _batch_declaration(scene.resolved_monitors(), "monitors", unresolved),
            "ports": _batch_declaration(scene.ports, "ports", unresolved),
            "excitations": _batch_declaration(simulation.excitations, "excitations", unresolved),
            "port_sweep": _batch_declaration(simulation.port_sweep, "port_sweep", unresolved),
            "output_plan": output_plan,
        }
        if declarations and batch.execution.grouping == "strict":
            difference = _batch_first_difference(declarations[0], declaration)
            if difference:
                raise ValueError(f"simulations[{index}] is incompatible with simulations[0] at {difference}.")
        declarations.append(declaration)
        signature = digest(declaration)
        if signature not in by_digest:
            by_digest[signature] = len(groups)
            groups.append({
                "group_id": len(groups), "variant_indices": [], "signature_digest": signature,
                "signature_kind": "provisional_declaration", "capability_status": "planning_only",
                "shared_artifacts": [], "output_plan": output_plan,
                "unresolved_reasons": list(dict.fromkeys(unresolved)),
            })
        group = groups[by_digest[signature]]
        group["variant_indices"].extend(range(len(batch)) if batch.template is not None else (index,))
    execution = _batch_declaration(batch.execution, "execution", [])
    if batch.parameters is not None and batch.parameters.requires_grad:
        warnings.append("Trainable parameters retain their PyTorch graph; native batch backward is pending P6.")
    devices = [{
        "device": device, "group_id": group["group_id"],
        "batch_size": None if batch.execution.batch_size == "auto" else batch.execution.batch_size,
        "slots": None if batch.execution.slots_per_device == "auto" else batch.execution.slots_per_device,
        "graph_steps": None if batch.execution.graph_steps == "auto" else batch.execution.graph_steps,
        "graph_mode": "unresolved", "estimated_peak_bytes": None, "memory_breakdown": {},
        "reserved_headroom_bytes": batch.execution.reserve_bytes,
        "tuned": False, "candidate_measurements": [], "rejected_candidates": [],
    } for group in groups for device in batch.execution.devices]
    payload = {
        "schema_version": 1, "resolved": False, "variant_count": len(batch),
        "variant_ids": list(batch.variant_ids), "groups": groups, "devices": devices,
        "input_copy_bytes": None, "output_bytes": None, "retained_tape_bytes": None,
        "warnings": warnings, "unsupported_features": ["batch_adjoint", "coordinated_batch_graph",
            "batch_byte_admission", "batch_multi_gpu"],
        "configuration_digest": digest({"declarations": declarations, "execution": execution,
                                         "ids": batch.variant_ids,
                                         "variation": _batch_declaration(batch.variation, "variation", []),
                                         "parameters": batch.parameters.schema if batch.parameters else None}),
    }
    return BatchPlanReport.from_dict(payload)
