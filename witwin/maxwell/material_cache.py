"""Bounded device-resident compiled snapshots for static FDTD geometry."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import threading
import time

import numpy as np
import torch


def _identity(value):
    """Exact content identity; only small geometry/configuration inputs cross host."""
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("FDTDMaterialCache does not support trainable tensors")
        array = value.detach().contiguous().cpu().numpy()
        return {"tensor": str(value.dtype), "shape": list(value.shape),
                "device": str(value.device), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot enter a material cache identity")
        return {"array": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, Enum):
        return _identity(value.value)
    if is_dataclass(value):
        return {field.name: _identity(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _identity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_identity(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported material cache input: {type(value).__name__}")


def _clone_tree(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("Cannot cache a material model with an autograd graph")
        if id(value) not in memo:
            memo[id(value)] = value.detach().clone()
        return memo[id(value)]
    if isinstance(value, dict):
        return {key: _clone_tree(item, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item, memo) for item in value)
    if is_dataclass(value):
        return replace(value, **{field.name: _clone_tree(getattr(value, field.name), memo)
                                 for field in fields(value) if field.init})
    if value is None or type(value) in (bool, int, float, complex, str):
        return value
    raise TypeError(f"Unsupported compiled material entry: {type(value).__name__}")


def _tree_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tree_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_bytes(v, seen) for v in value)
    if is_dataclass(value):
        return sum(_tree_bytes(getattr(value, field.name), seen) for field in fields(value))
    return 0


def _digest(value):
    return hashlib.sha256(json.dumps(_identity(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _scene_key(scene, frequency, *, parts=False):
    """Content key; ``parts=True`` adds the non-structure frame digest and the
    per-structure digests used to align local recompiles."""
    from witwin.core import Box, Mesh, Structure
    from .media import Material
    from .geometry.planar_pec import PlanarPEC
    from .copper_foil import CopperFoil

    if torch.device(scene.device).type != "cuda":
        raise ValueError("FDTDMaterialCache requires a CUDA scene")
    for name in ("material_regions", "thin_wires", "networks", "lumped_elements",
                 "circuits", "electrostatic_terminals", "charge_densities"):
        if getattr(scene, name, ()):
            raise NotImplementedError(f"FDTDMaterialCache does not support scene.{name}")
    if getattr(scene, "_material_model_override", None) is not None:
        raise ValueError("Cannot cache a caller-supplied material model override")
    structures = []
    for structure in scene.structures:
        if (type(structure) is not Structure or type(structure.geometry) not in (Mesh, Box, PlanarPEC)
                or type(structure.material) not in (Material, CopperFoil)):
            raise NotImplementedError("FDTDMaterialCache supports static Mesh/Box/PlanarPEC with Material or Box CopperFoil")
        material, geometry = structure.material, structure.geometry
        if type(material) is CopperFoil and type(geometry) is not Box:
            raise NotImplementedError("Cached CopperFoil support requires planar Box geometry")
        for name in ("epsilon_tensor", "mu_tensor", "sigma_e_tensor", "orientation",
                     "kerr_chi3", "nonlinearity", "modulation", "breakdown", "mass_density"):
            value = getattr(material, name)
            if value is not None and value is not False and not (isinstance(value, tuple) and not value):
                raise NotImplementedError(f"FDTDMaterialCache does not support material.{name}")
        if any(isinstance(value, torch.Tensor) and value.requires_grad for value in vars(geometry).values()):
            raise ValueError("FDTDMaterialCache does not support trainable geometry")
        # A cache hit must not bypass the core SDF's stale-topology guard.
        from witwin.core.geometry.mesh import _tensor_revision
        revision = getattr(geometry, "_topology_diagnostic_revision", None)
        if type(geometry) is Mesh and revision is not None and revision != (
                _tensor_revision(geometry.vertices), _tensor_revision(geometry.faces)):
            raise RuntimeError("Mesh topology diagnostics are stale; reconstruct the Mesh before caching")
        # Exclude derived BVH/SDF caches, but include every input consumed by Mesh
        # distance/occupancy, including validation flags used by its fill policy.
        if type(geometry) is Mesh:
            mesh = {name: getattr(geometry, name) for name in (
                "position", "rotation", "scale", "recenter", "fill_mode", "surface_thickness",
                "is_watertight", "degenerate_face_count", "inconsistent_edge_orientation_count")}
            mesh.update(vertices=geometry.vertices, faces=geometry.faces)
        elif type(geometry) is Box:
            mesh = {"kind": "box", **{name: getattr(geometry, name) for name in ("position", "rotation", "size")}}
        else:
            mesh = {"kind": "planar_pec", **{name: getattr(geometry, name) for name in (
                "vertices", "holes", "normal_axis", "coordinate", "physical_thickness", "position", "rotation")},
                "diagnostic_name": structure.name}
        # Names and allocation-generated material IDs are aliases, not physics.
        static_material = {field.name: getattr(material, field.name) for field in fields(material)
                           if field.name not in ("name", "material_id")}
        if type(material) is CopperFoil:
            static_material.update(kind="copper_foil", **{name: getattr(material, name) for name in (
                "thickness", "conductivity", "frequency_range", "mode_count", "edge_model")})
            mesh["diagnostic_name"] = structure.name
        structures.append({"mesh": mesh, "material": static_material,
                           "priority": structure.priority, "enabled": structure.enabled})
    # Resolve an unindexed 'cuda' declaration to the actual prepared allocation.
    # Empty scenes have no geometry tensor whose identity could disambiguate it.
    frame = {"version": 2, "device": str(scene.x.device), "frequency": frequency,
             "torch_default_dtype": str(torch.get_default_dtype()),
             "domain": scene.domain, "boundary": scene.boundary, "subpixel": scene.subpixel,
             "symmetry": scene.symmetry, "nodes": [scene.x_nodes64, scene.y_nodes64, scene.z_nodes64]}
    payload = {**frame, "structures": structures}
    key = _digest(payload)
    return (key, _digest(frame), tuple(_digest(structure) for structure in structures)) if parts else key


def _unchanged_structures(base, current):
    """Map current structure indices to equal base ones: common prefix and suffix,
    or every equal position when the counts match."""
    if len(base) == len(current):
        return {i: i for i, (a, b) in enumerate(zip(base, current)) if a == b}
    prefix = 0
    while prefix < min(len(base), len(current)) and base[prefix] == current[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(base), len(current)) - prefix
           and base[len(base) - 1 - suffix] == current[len(current) - 1 - suffix]):
        suffix += 1
    unchanged = {i: i for i in range(prefix)}
    unchanged.update({len(current) - 1 - k: len(base) - 1 - k for k in range(suffix)})
    return unchanged


class FDTDMaterialCache:
    """Opt-in static material compilation cache shared by source solves.

    Pass one instance to ``Simulation.fdtd(..., material_cache=cache)``. Inputs
    are content-keyed; sources/monitors and dt do not affect the cached static
    material model. Frequency is conservatively part of the key. Every solver
    gets owned copies, never shared evolving or writable cached state.

    The cache retains GPU snapshots only, with LRU eviction by tensor bytes and
    entry count. The byte cap excludes live solvers, key temporaries and CUDA's
    allocator reserve. There is no disk cache, gradient reuse or CPU fallback.

    A miss whose frame (device, frequency, domain, grid nodes, boundary,
    subpixel and symmetry) matches the most recent full compilation, and whose
    changed structures are static Box volumes or PEC PlanarPEC sheets, is
    recompiled locally from that compilation (outcome ``"local"``): unchanged
    sheets keep their edge ownership and volume materials are resampled only
    around the old and new bounds of the changed boxes. The result is identical
    to a full compile; every other miss compiles in full and becomes the new
    local base. That base stays retained after LRU eviction until the next full
    compilation or ``clear()``.
    """

    def __init__(self, *, max_entries=2, max_bytes=1024**3):
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict()
        self._bytes = self._hits = self._misses = self._evictions = self._oversized = self._local_recompiles = 0
        self._local = None
        self._lock = threading.RLock()

    @property
    def stats(self):
        with self._lock:
            return {"entries": len(self._entries), "bytes": self._bytes, "hits": self._hits,
                    "misses": self._misses, "evictions": self._evictions, "oversized": self._oversized,
                    "local_recompiles": self._local_recompiles}

    def clear(self):
        """Release retained snapshots, preserving cumulative hit/miss counters."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._local = None

    def _local_model(self, scene, frame, structures):
        """Patch the most recent full compilation, or ``None`` without a locality proof."""
        if self._local is None or self._local[0] != frame:
            return None
        _, base_structures, local = self._local
        model = local.recompile(scene, _unchanged_structures(base_structures, structures))
        if model is not None:
            self._local_recompiles += 1
        return model

    def _track(self, scene, snapshot, frame, structures):
        from .compiler.materials import LocalMaterialCache
        try:
            self._local = (frame, structures, LocalMaterialCache(scene, base=snapshot))
        except (NotImplementedError, ValueError):
            self._local = None

    def _bind(self, scene, *, frequency):
        with self._lock:
            started = time.perf_counter()
            key, frame, structures = _scene_key(scene, frequency, parts=True)
            key_seconds = time.perf_counter() - started
            started = time.perf_counter()
            if key in self._entries:
                snapshot, _ = self._entries.pop(key)
                self._entries[key] = (snapshot, _)
                # Synchronous copies make entries safe across streams/threads and
                # immediate eviction; no borrowed cache tensor escapes this class.
                model = _clone_tree(snapshot)
                torch.cuda.synchronize(scene.x.device)
                self._hits += 1
                outcome = "hit"
            else:
                self._misses += 1
                local = self._local_model(scene, frame, structures)
                model = scene.compile_materials() if local is None else local
                size = _tree_bytes(model)
                outcome = "miss" if local is None else "local"
                if local is not None:
                    # A local model shares unchanged tensors with the immutable
                    # base snapshot; the solver receives its own copy.
                    model = _clone_tree(local)
                if size <= self.max_bytes:
                    while self._entries and (len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes):
                        _, (_, old_size) = self._entries.popitem(last=False)
                        self._bytes -= old_size
                        self._evictions += 1
                    snapshot = _clone_tree(model) if local is None else local
                    torch.cuda.synchronize(scene.x.device)
                    self._entries[key] = (snapshot, size)
                    self._bytes += size
                    if local is None:
                        self._track(scene, snapshot, frame, structures)
                else:
                    if local is not None:
                        torch.cuda.synchronize(scene.x.device)
                    self._oversized += 1
                    outcome = f"{outcome}_oversized_not_retained"
            scene._material_model_override = model
            return {"outcome": outcome, "key": key, "key_seconds": key_seconds,
                    "bind_seconds": time.perf_counter() - started, **self.stats}
