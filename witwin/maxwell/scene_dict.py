"""Constructor-based scene dictionaries for ``witwin.scene/1``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from inspect import signature
import math

import numpy as np
import torch
from witwin.core import GeometryBase, Structure, geometry_from_dict, geometry_to_dict
from witwin.core.geometry.base import _rotation_quaternion

from .absorbers import ModeABC, MurABC
from .copper_foil import CopperFoil
from .geometry import PlanarPEC, PolySlab, SurfaceMesh
from .lumped import Capacitor, Inductor, ParallelRLC, Resistor, SeriesRLC
from .media import DebyePole, DrudePole, LorentzPole, LossyMetalMedium, Material
from .monitors import (
    AngleProjectionMonitor, CartesianProjectionMonitor, ClosedSurfaceMonitor,
    FinitePlaneMonitor, FluxMonitor, KSpaceProjectionMonitor, PermittivityMonitor,
    PlaneMonitor, PointMonitor,
)
from .ports import AxisPath, LumpedPort, TerminalPort, TerminalRef, WireNodeRef, WirePortBinding
from .scene import (
    BoundarySpec, CornerFinderSpec, Domain, GridRefinement, GridSpec,
    LayerRefinementSpec, MeshOverrideStructure, Scene, SubpixelSpec,
)
from .sources import CW, GaussianPulse, PlaneWave, PointDipole, RickerWavelet, TFSF
from .thin_wire import ThinWire, WireConductor, WireEnd


_TYPES = (
    Domain, GridSpec, BoundarySpec, MurABC, ModeABC, MeshOverrideStructure, LayerRefinementSpec,
    GridRefinement, CornerFinderSpec, Material, CopperFoil, LossyMetalMedium,
    DebyePole, DrudePole, LorentzPole, Structure, ThinWire, WireConductor, WireEnd,
    PointDipole, PlaneWave, CW, GaussianPulse, RickerWavelet, TFSF,
    LumpedPort, TerminalPort, TerminalRef, AxisPath, WireNodeRef, WirePortBinding,
    Resistor, Capacitor, Inductor, SeriesRLC, ParallelRLC,
    PointMonitor, PlaneMonitor, FinitePlaneMonitor, FluxMonitor, ClosedSurfaceMonitor,
    AngleProjectionMonitor, CartesianProjectionMonitor, KSpaceProjectionMonitor,
    PermittivityMonitor, SubpixelSpec,
)
_TYPES_BY_TAG = {cls.__name__: cls for cls in _TYPES}
_GEOMETRIES = {cls.kind: cls for cls in (PolySlab, PlanarPEC, SurfaceMesh)}
_MATERIALS = (Material, CopperFoil, LossyMetalMedium, WireConductor)
_EXCLUDED_SCENE_FIELDS = (
    "networks", "circuits", "material_regions", "electrostatic_terminals",
    "charge_densities", "internal_absorbers",
)


def _unsupported(what):
    raise TypeError(f"not in witwin.scene/1: {what}")


def _metadata(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            _unsupported("metadata keys")
        return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    _unsupported(f"metadata {type(value).__name__}")


def _check_fixed(value):
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            _unsupported("requires_grad tensor")
    elif isinstance(value, GeometryBase):
        for name in signature(type(value)).parameters:
            if name != "device":
                _check_fixed(getattr(value, name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_fixed(item)


def _field_value(obj, name):
    if isinstance(obj, TerminalPort) and name == "current_surface":
        return obj._declared_current_surface
    if isinstance(obj, BoundarySpec):
        if name == "kind" and obj.kind == "mixed":
            return "none"
        if name in {f"{axis}_{side}" for axis in "xyz" for side in ("low", "high")}:
            axis, side = name.split("_")
            return obj.face_kind(axis, side)
    return getattr(obj, name)


class _Writer:
    def __init__(self, arrays):
        self.arrays = arrays
        self.materials = {}
        self.material_ids = {}

    def material_id(self, material):
        if type(material) not in _MATERIALS:
            _unsupported(type(material).__name__)
        identity = id(material)
        if identity not in self.material_ids:
            key = f"m{len(self.material_ids)}"
            self.material_ids[identity] = key
            self.materials[key] = self.object(material, f"materials/{key}")
        return self.material_ids[identity]

    def object(self, obj, path):
        cls = type(obj)
        if cls not in _TYPES and cls not in _GEOMETRIES.values():
            _unsupported(cls.__name__)
        result = {"type": cls.kind if cls in _GEOMETRIES.values() else cls.__name__}
        for name in signature(cls).parameters:
            if name == "device":
                continue
            value = _field_value(obj, name)
            if cls in _GEOMETRIES.values() and name == "rotation":
                value = _rotation_quaternion(value)
            if (cls is Structure and name == "material") or (cls is ThinWire and name == "conductor"):
                result[name] = self.material_id(value)
            elif name == "metadata":
                result[name] = _metadata(value)
            else:
                result[name] = self.encode(value, f"{path}/{name}", geometry=cls in _GEOMETRIES.values())
        if cls is CopperFoil:
            result["kind"] = "copper_foil"
            result["roughness"] = "none"
        elif cls is LossyMetalMedium:
            result["roughness"] = "none"
            result["kind"] = "lossy_metal"
        elif cls is Material:
            result["kind"] = "pec" if obj.pec else "pmc" if obj.pmc else "medium"
        if cls is GridSpec:
            result["kind"] = "custom" if obj.is_custom else "auto" if obj.is_auto else "uniform"
        if cls is TFSF:
            for field in fields(cls):
                result[field.name] = self.encode(getattr(obj, field.name), f"{path}/{field.name}")
        return result

    def core_arrays(self, value, local_arrays, path):
        if isinstance(value, dict):
            if "array" in value:
                key = path + value["array"][len("geometry"):]
                self.arrays[key] = local_arrays[value["array"]]
                return {"array": key}
            return {key: self.core_arrays(item, local_arrays, path) for key, item in value.items()}
        if isinstance(value, list):
            return [self.core_arrays(item, local_arrays, path) for item in value]
        return value

    def encode(self, value, path, *, geometry=False):
        if isinstance(value, GeometryBase):
            _check_fixed(value)
            if type(value) in _GEOMETRIES.values():
                return self.object(value, path)
            local_arrays = None if self.arrays is None else {}
            result = geometry_to_dict(value, arrays=local_arrays)
            return self.core_arrays(result, local_arrays, path)
        if type(value) in _TYPES:
            return self.object(value, path)
        if isinstance(value, torch.Tensor):
            _check_fixed(value)
            if self.arrays is not None and value.numel() > 4096:
                self.arrays[path] = value
                return {"array": path}
            if geometry:
                return value.detach().cpu().tolist()
            return {"type": "Tensor", "dtype": str(value.dtype).removeprefix("torch."),
                    "values": self.encode(value.detach().cpu().tolist(), f"{path}/values")}
        if isinstance(value, np.ndarray):
            if self.arrays is not None and value.size > 4096:
                self.arrays[path] = torch.from_numpy(value)
                return {"array": path}
            return self.encode(value.tolist(), path)
        if isinstance(value, complex):
            return {"type": "complex", "real": value.real, "imag": value.imag}
        if isinstance(value, Mapping):
            return {key: self.encode(item, f"{path}/{key}") for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.encode(item, f"{path}/{index}", geometry=geometry) for index, item in enumerate(value)]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        _unsupported(type(value).__name__)


def scene_to_dict(scene, arrays=None):
    """Write a fixed scene with explicit constructor defaults and shared materials."""
    for name in _EXCLUDED_SCENE_FIELDS:
        if getattr(scene, name):
            _unsupported(name)
    if any(entry is not None for entry in scene.symmetry):
        _unsupported("symmetry")
    if scene.coordinates is not None:
        _unsupported("coordinates")
    writer = _Writer(arrays)
    background_id = None if scene.background is None else writer.material_id(scene.background)
    result = {
        "schema": "witwin.scene/1", "unit": {"length": "m", "frequency": "Hz"},
        "background": {"material": background_id}, "materials": writer.materials,
    }
    for name in ("domain", "grid", "boundary", "structures", "thin_wires", "sources",
                 "ports", "lumped_elements", "monitors", "subpixel"):
        result[name] = writer.encode(getattr(scene, name), name)
    result["metadata"] = _metadata(scene.metadata)
    return result


class _Reader:
    def __init__(self, arrays, device):
        self.arrays = arrays
        self.device = device
        self.materials = {}

    def decode(self, value):
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "array" in value:
            return self.arrays[value["array"]].to(device=self.device)
        if "type" not in value:
            return {key: self.decode(item) for key, item in value.items()}
        tag = value["type"]
        if tag == "complex":
            return complex(value["real"], value["imag"])
        if tag == "Tensor":
            return torch.tensor(self.decode(value["values"]), dtype=getattr(torch, value["dtype"]),
                                device=self.device)
        if tag not in _TYPES_BY_TAG and tag not in _GEOMETRIES:
            return geometry_from_dict(value, arrays=self.arrays, device=self.device)
        cls = _GEOMETRIES[tag] if tag in _GEOMETRIES else _TYPES_BY_TAG[tag]
        if cls is TFSF and value["mode"] == "slab":
            return TFSF.slab(axis=value["axis"], bounds=value["axis_bounds"])
        kwargs = {}
        for name in signature(cls).parameters:
            if name == "device":
                kwargs[name] = self.device
            elif (cls is Structure and name == "material") or (cls is ThinWire and name == "conductor"):
                kwargs[name] = self.materials[value[name]]
            elif name == "metadata":
                kwargs[name] = value[name]
            elif cls is GridSpec and name.endswith("_coords") and value[name] is not None:
                coordinates = value[name]
                if isinstance(coordinates, dict) and "array" in coordinates:
                    coordinates = self.arrays[coordinates["array"]].detach().cpu().numpy()
                kwargs[name] = np.asarray(coordinates, dtype=np.float64)
            else:
                kwargs[name] = self.decode(value[name])
        if cls is LossyMetalMedium:
            kwargs["roughness"] = None
        if cls is ThinWire and not kwargs["endpoints"]:
            kwargs["endpoints"] = None
        return cls(**kwargs)


def scene_from_dict(d, arrays=None, device="cuda"):
    """Rebuild a scene through its public constructors."""
    if d["schema"] != "witwin.scene/1":
        _unsupported(d["schema"])
    reader = _Reader(arrays, device)
    reader.materials = {key: reader.decode(value) for key, value in d["materials"].items()}
    kwargs = {name: reader.decode(d[name]) for name in (
        "domain", "grid", "boundary", "structures", "thin_wires", "sources",
        "ports", "lumped_elements", "monitors",
    )}
    background_id = d["background"]["material"]
    background = None if background_id is None else reader.materials[background_id]
    return Scene(**kwargs, background=background,
                 subpixel_samples=reader.decode(d["subpixel"]), metadata=d["metadata"], device=device)
