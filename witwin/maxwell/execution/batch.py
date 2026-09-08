"""Declarative batch configuration and planning over ordinary Simulations.

Execution is enabled only as native runtime capabilities become available. A
plan never treats declarative agreement as proof of compiled storage sharing.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

import torch

from ..fdtd_parallel import _normalize_devices, _normalize_cuda_device


def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _names(value, name, *, all_allowed=False):
    if isinstance(value, str):
        if all_allowed and value == "all":
            return value
        raise TypeError(f"{name} must be an ordered collection of names.")
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple.")
    names = tuple(value)
    if any(not isinstance(v, str) or not v.strip() for v in names):
        raise ValueError(f"{name} must contain nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique names.")
    return names


def _mapping(value, name):
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    _names(tuple(value), name)
    return MappingProxyType(dict(value))


@dataclass(frozen=True, kw_only=True)
class BatchExecution:
    """Device and capacity policy; construction does not initialize CUDA."""

    devices: tuple[str, ...]
    batch_size: int | Literal["auto"] = "auto"
    max_batch_size: int = 32
    memory_limit_bytes: int | Mapping[str, int] | None = None
    reserve_bytes: int = 512 * 1024**2
    grouping: Literal["strict", "compatible"] = "strict"
    scheduling: Literal["static", "throughput"] = "throughput"
    slots_per_device: int | Literal["auto"] = 1
    graph: Literal["auto", "required", "off"] = "auto"
    graph_steps: int | Literal["auto"] = "auto"
    tune: bool = False
    fail_fast: bool = True

    def __post_init__(self):
        object.__setattr__(self, "devices", _normalize_devices(self.devices, minimum=1))
        _integer(self.max_batch_size, "max_batch_size")
        if self.batch_size != "auto":
            _integer(self.batch_size, "batch_size")
            if self.batch_size > self.max_batch_size:
                raise ValueError("batch_size must not exceed max_batch_size.")
        _integer(self.reserve_bytes, "reserve_bytes", minimum=0)
        for name, choices in (
            ("grouping", ("strict", "compatible")),
            ("scheduling", ("static", "throughput")),
            ("graph", ("auto", "required", "off")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}.")
        if self.graph_steps != "auto":
            _integer(self.graph_steps, "graph_steps")
        for name in ("tune", "fail_fast"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        if self.slots_per_device == "auto":
            if not self.tune:
                raise ValueError("slots_per_device='auto' requires tune=True.")
        else:
            _integer(self.slots_per_device, "slots_per_device")
            if self.slots_per_device not in (1, 2):
                raise ValueError("slots_per_device must be 1, 2 or 'auto'.")
        limit = self.memory_limit_bytes
        if isinstance(limit, Mapping):
            normalized = {}
            for device, value in limit.items():
                key = _normalize_cuda_device(device, name="memory_limit_bytes device")
                if key in normalized:
                    raise ValueError(f"Duplicate memory limit for {key}.")
                normalized[key] = _integer(value, f"memory_limit_bytes[{key}]")
            if set(normalized) != set(self.devices):
                raise ValueError("memory_limit_bytes must specify every execution device exactly once.")
            object.__setattr__(self, "memory_limit_bytes", MappingProxyType(normalized))
        elif limit is not None:
            _integer(limit, "memory_limit_bytes")


@dataclass(frozen=True, kw_only=True)
class BatchParameters:
    """Named tensor inputs with an explicit leading variant dimension."""

    region_densities: Mapping[str, torch.Tensor] | None = None
    source_scales: Mapping[str, torch.Tensor] | None = None
    variant_count: int = field(init=False)

    def __post_init__(self):
        counts = set()
        for name, ndim in (("region_densities", 4), ("source_scales", 1)):
            values = _mapping(getattr(self, name), name)
            object.__setattr__(self, name, values)
            for key, value in values.items():
                path = f"{name}[{key!r}]"
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{path} must be a torch.Tensor.")
                if value.layout != torch.strided or not value.is_floating_point():
                    raise TypeError(f"{path} must be a real floating-point strided tensor.")
                if value.device.type not in ("cpu", "cuda"):
                    raise ValueError(f"{path} must use a CPU or CUDA input device.")
                if value.ndim != ndim or any(n == 0 for n in value.shape):
                    raise ValueError(f"{path} must have {ndim} nonempty dimensions.")
                counts.add(value.shape[0])
        if len(counts) != 1:
            raise ValueError("BatchParameters needs nonempty inputs with the same leading dimension.")
        object.__setattr__(self, "variant_count", counts.pop())

    @property
    def requires_grad(self):
        return any(t.requires_grad for values in (self.region_densities, self.source_scales)
                   for t in values.values())

    @property
    def schema(self):
        return tuple((kind, name, tuple(t.shape), str(t.dtype))
                     for kind in ("region_densities", "source_scales")
                     for name, t in sorted(getattr(self, kind).items()))


@dataclass(frozen=True, kw_only=True)
class BatchVariation:
    material_regions: tuple[str, ...] = ()
    structures: Mapping[str, object] | None = None
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "material_regions", _names(self.material_regions, "material_regions"))
        object.__setattr__(self, "sources", _names(self.sources, "sources"))
        structures = _mapping(self.structures, "structures")
        from witwin.core import GeometryBase

        if any(not isinstance(g, GeometryBase) for g in structures.values()):
            raise TypeError("structures values must be geometry envelopes.")
        object.__setattr__(self, "structures", structures)


@dataclass(frozen=True, kw_only=True)
class BatchOutput:
    monitors: tuple[str, ...] | Literal["all"] = "all"
    ports: tuple[str, ...] | Literal["all"] = "all"
    fields: tuple[str, ...] = ()
    materials: bool = False
    result_device: str = "local"
    max_pending_results: int = 2

    def __post_init__(self):
        for name in ("monitors", "ports"):
            object.__setattr__(self, name, _names(getattr(self, name), name, all_allowed=True))
        fields = _names(self.fields, "fields")
        if any(f not in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") for f in fields):
            raise ValueError("fields must contain Yee component names Ex/Ey/Ez/Hx/Hy/Hz.")
        object.__setattr__(self, "fields", fields)
        if not isinstance(self.materials, bool):
            raise TypeError("materials must be a bool.")
        if self.result_device not in ("local", "cpu"):
            object.__setattr__(self, "result_device", _normalize_cuda_device(
                self.result_device, name="result_device"))
        _integer(self.max_pending_results, "max_pending_results")


def _named(objects, name, *, path):
    matches = [obj for obj in objects if getattr(obj, "name", None) == name]
    if len(matches) != 1:
        raise ValueError(f"{path}[{name!r}] must identify exactly one object; found {len(matches)}.")
    return matches[0]


class BatchSimulation:
    """Inspectable batch submission preserving the single-Scene object model."""

    def __init__(self, simulations, *, execution, parameters=None, variant_ids=None,
                 variation=None, outputs=None):
        from ..simulation import Simulation, SimulationMethod

        if not isinstance(execution, BatchExecution):
            raise TypeError("execution must be a BatchExecution.")
        if parameters is not None and not isinstance(parameters, BatchParameters):
            raise TypeError("parameters must be a BatchParameters.")
        if variation is not None and not isinstance(variation, BatchVariation):
            raise TypeError("variation must be a BatchVariation.")
        if outputs is not None and not isinstance(outputs, BatchOutput):
            raise TypeError("outputs must be a BatchOutput.")
        self.execution = execution
        self.outputs = outputs if outputs is not None else BatchOutput()
        self.parameters = parameters
        self.variation = variation
        self.template = simulations if isinstance(simulations, Simulation) else None
        if self.template is not None:
            if parameters is None:
                raise ValueError("A Simulation template requires nonempty BatchParameters.")
            self.simulations = (self.template,)
            count = parameters.variant_count
        else:
            if parameters is not None:
                raise ValueError("A Simulation sequence cannot also supply parameters.")
            if not isinstance(simulations, Sequence) or isinstance(simulations, (str, bytes)):
                raise TypeError("simulations must be a Simulation or nonempty sequence.")
            self.simulations = tuple(simulations)
            count = len(self.simulations)
            if count == 0:
                raise ValueError("simulations must not be empty.")
        ids = tuple(str(i) for i in range(count)) if variant_ids is None else _names(variant_ids, "variant_ids")
        if len(ids) != count:
            raise ValueError("variant_ids must match the variant count.")
        self.variant_ids = ids
        for index, simulation in enumerate(self.simulations):
            if not isinstance(simulation, Simulation):
                raise TypeError(f"simulations[{index}] must be a Simulation.")
            if simulation.method != SimulationMethod.FDTD:
                raise ValueError(f"simulations[{index}] must select FDTD.")
            if simulation.config.parallel is not None:
                raise ValueError(f"simulations[{index}] cannot combine batch with spatial parallel execution.")
            if simulation.config.full_field_dft and not self.outputs.fields:
                raise ValueError(f"simulations[{index}].full_field_dft conflicts with empty BatchOutput.fields.")
            scene = simulation.scene
            if variation is not None:
                for name in variation.material_regions:
                    _named(scene.material_regions, name, path=f"simulations[{index}].material_regions")
                for name in variation.structures:
                    _named(scene.structures, name, path=f"simulations[{index}].structures")
                for name in variation.sources:
                    _named(scene.resolved_sources(), name, path=f"simulations[{index}].sources")
            for kind, objects in (("monitors", scene.resolved_monitors()), ("ports", scene.ports)):
                names = getattr(self.outputs, kind)
                if names != "all":
                    for name in names:
                        _named(objects, name, path=f"outputs.{kind}")
        if self.template is not None:
            self._validate_parameters(parameters)
            if variation is not None:
                if (set(variation.material_regions) != set(parameters.region_densities)
                        or set(variation.sources) != set(parameters.source_scales)
                        or variation.structures):
                    raise ValueError("Template variation must match the parameter targets exactly.")
        if self.outputs.result_device == "cpu" and (
            (parameters is not None and parameters.requires_grad)
            or any(sim.has_trainable_parameters for sim in self.simulations)
        ):
            raise ValueError("CPU result export requires a detached forward input.")

    def _validate_parameters(self, parameters):
        scene = self.template.scene
        for name, value in parameters.region_densities.items():
            region = _named(scene.material_regions, name, path="parameters.region_densities")
            if tuple(value.shape[1:]) != tuple(region.density.shape):
                raise ValueError(f"parameters.region_densities[{name!r}] spatial shape differs from the template.")
        for name in parameters.source_scales:
            source = _named(scene.resolved_sources(), name, path="parameters.source_scales")
            if source not in scene.sources:
                raise NotImplementedError("source_scales currently requires an explicit Scene source, not a port-generated source.")
            if getattr(source, "source_time", None) is None or not hasattr(source.source_time, "amplitude"):
                raise ValueError(f"parameters.source_scales[{name!r}] requires an explicit source_time amplitude.")

    def __len__(self):
        return len(self.variant_ids)

    def scene(self, index):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.template is None:
            return self.simulations[index].scene.clone()
        scene = self.template.scene.clone()
        for name, value in self.parameters.region_densities.items():
            region = _named(scene.material_regions, name, path="material_regions")
            slot = next(i for i, obj in enumerate(scene.material_regions) if obj is region)
            scene.material_regions[slot] = replace(region, density=value[index])
        for name, value in self.parameters.source_scales.items():
            if value.requires_grad:
                raise NotImplementedError("Trainable source scale materialization requires the native source-amplitude contract (P6).")
            source = _named(scene.sources, name, path="sources")
            slot = next(i for i, obj in enumerate(scene.sources) if obj is source)
            scaled = copy.copy(source)
            object.__setattr__(scaled, "source_time", replace(
                source.source_time, amplitude=source.source_time.amplitude * float(value[index]),
            ))
            scene.sources[slot] = scaled
        return scene

    def plan(self):
        from .plan import build_batch_plan
        return build_batch_plan(self)

    def prepare(self):
        self.plan()
        raise NotImplementedError(
            "Native shared-state Batch execution is not implemented yet (P2-P4). "
            "Simulation.batch currently provides validation and planning only."
        )

    def run(self):
        with self.prepare() as prepared:
            return prepared.run()

    def iter_results(self, *, order="completion"):
        if order not in ("completion", "input"):
            raise ValueError("order must be 'completion' or 'input'.")
        return self.prepare().iter_results(order=order)
