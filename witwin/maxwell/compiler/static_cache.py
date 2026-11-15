"""Identity-keyed reuse of the static structure prefix of the material compiler.

A ``SceneModule`` builds its static part once and returns ``self._base.updated(material_regions=...)``
from ``to_scene()``. Every preparation then holds the very same ``Structure`` objects, and only the
material regions change. The material compiler blends all bulk structures before any region, so the
accumulator state after the structure loop (the *static prefix*) of each node sample and of each
edge-component sample depends only on the structures, the grid and the sampling flags. This module
keeps those prefixes between preparations and hands the compiler owned copies, so the structure
signed distances and blends are not evaluated again. The regions and every later stage (PEC, sheets,
surface layout, averaging, reconstruction) run as before; a hit returns the same tensors the
uncached arithmetic produces, so the compiled model is ``torch.equal`` to an uncached compile.

Admission (per compile, decided before compiling). A preparation uses the cache only when

- it declares material regions (a changing part follows the prefix);
- every structure is static: no ``requires_grad`` tensor in its geometry or material and no
  modulation; no conductivity capture is active; the backgrounds are Python numbers;
- its prepared structures are the declared ones, or the PML-interface extrusions of them;
- the payload bound below fits the frame budget.

Key (a frame). The device (with index), the current CUDA stream, the material dtype, the
inference-mode flag, the grid (float64 nodes, primal, half and dual spacings per axis), the domain
and boundary (``==``), the coordinate chart (``is``) and the declared structures (``is``, in order).
Within a frame, prefixes are keyed by site, component, sample index, sample counts, averaging,
backgrounds and the sampling flags.

Identity safety. A frame holds weak references to its structures; when one dies the frame is
dropped and its tensors are released, and an ``id`` cannot be reused while the frame lives. Every
lookup also compares a content snapshot of the structures and the chart, so an in-place edit of a
geometry tensor (including through ``.data``) or a rebound attribute misses instead of reusing
stale tables.

Ownership. Prefixes are cloned when stored and cloned again on every hit (read-only constant-zero
channels are re-created), so no caller can mutate a tensor back into the cache.

Memory and eviction. Payload bound per frame, from the shapes before compiling, with
``S = prod(samples)``, ``N = Nx*Ny*Nz``, ``b`` the dtype bytes and ``P`` the pole entries::

    S * N * (b * (24 + P) + 1)      node samples: 12 components, 3 summaries, P pole weights,
                                    6 inverse components, 3 off-diagonal channels, 1 bool mask
  + 6 * S * N * (3 * b + 1)         edge samples: material, inverse, conductivity, bool mask

A frame holds at most ``FRAME_BYTES`` (512 MiB); a scene whose bound exceeds it is not admitted
(``stats["not_admitted_bytes"]``), and a prefix that would overflow a frame is not stored. At most
``MAX_FRAMES`` (2) frames are kept, evicted least recently used first. Eviction is deterministic: by
frame count and bytes, never by memory pressure. On CUDA these bytes are device memory held across
runs until the frame is evicted, its structures die, or ``STATIC_PREFIX_CACHE.clear()`` is called.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import is_dataclass
from enum import Enum
from collections.abc import Mapping
import math
import threading
import weakref

import numpy as np
import torch

FRAME_BYTES = 512 * 1024 ** 2
MAX_FRAMES = 2

_POLE_KEYS = ("debye_poles", "drude_poles", "lorentz_poles", "mu_debye_poles", "mu_drude_poles", "mu_lorentz_poles")
_ACTIVE: ContextVar[tuple | None] = ContextVar("maxwell_static_prefix_frame", default=None)


_IMMUTABLE = (str, bytes, int, float, complex, bool, type, Enum, torch.device, torch.dtype)


class _State:
    """The mutable state reachable from key objects: attribute and item bindings, and tensor contents.

    Objects with attributes, lists and mappings record their bindings (values compared with ``is``); tensors
    and arrays record their contents (compared exactly, one comparison per dtype and device).
    ``trainable`` is set when a tensor requires grad or a material is modulated.
    """

    def __init__(self, roots, *, snapshot):
        self.bindings, self.arrays, tensors = [], [], []
        self.trainable = False
        seen = set()
        stack = list(roots)
        while stack:
            value = stack.pop()
            if isinstance(value, torch.Tensor):
                tensors.append(value)
                self.trainable |= value.requires_grad
                continue
            if value is None or isinstance(value, _IMMUTABLE) or id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, np.ndarray):
                self.arrays.append((value, value.copy() if snapshot else value))
                continue
            if isinstance(value, tuple):
                stack.extend(value)
                continue
            if isinstance(value, list):
                keys, values = None, tuple(value)
            elif isinstance(value, Mapping):
                keys, values = tuple(value.keys()), tuple(value.values())
            elif hasattr(value, "__dict__"):
                keys, values = tuple(vars(value)), tuple(vars(value).values())
                self.trainable |= is_dataclass(value) and getattr(value, "modulation", None) is not None
            else:
                continue
            # The owner by id: owners are reached from the live key structures, and every replaced binding
            # stays referenced by its parent's recorded values, so no id is reused while a frame lives.
            self.bindings.append((id(value), keys, values))
            stack.extend(values)
        groups = {}
        for tensor in tensors:
            groups.setdefault((tensor.dtype, tensor.device), []).append(tensor.detach().reshape(-1))
        self.contents = {key: torch.cat(group) for key, group in groups.items()}
        if snapshot:
            self.contents = {key: value.clone() for key, value in self.contents.items()}
        self.bytes = (sum(value.numel() * value.element_size() for value in self.contents.values())
                      + sum(copy.nbytes for _, copy in self.arrays))

    def same(self, other) -> bool:
        return (len(self.bindings) == len(other.bindings) and all(
                    a == b and keys == other_keys and len(values) == len(other_values)
                    and all(x is y for x, y in zip(values, other_values))
                    for (a, keys, values), (b, other_keys, other_values) in zip(self.bindings, other.bindings))
                and len(self.arrays) == len(other.arrays)
                and all(np.array_equal(a, b) for (_, a), (_, b) in zip(self.arrays, other.arrays))
                and self.contents.keys() == other.contents.keys()
                and all(torch.equal(self.contents[key], other.contents[key]) for key in self.contents))


def _owned(value, memo):
    """A copy of a prefix that shares no tensor storage with ``value``."""
    from .materials import _constant_zero_field, _is_constant_zero

    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = (_constant_zero_field(value.shape, value.device)
                               if _is_constant_zero(value) else value.clone())
        return memo[id(value)]
    if isinstance(value, dict):
        return {key: _owned(item, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_owned(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_owned(item, memo) for item in value)
    return value


def _tree_bytes(value, seen):
    if isinstance(value, torch.Tensor):
        if id(value) in seen or value.stride() and 0 in value.stride():
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tree_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_bytes(item, seen) for item in value)
    return 0


def payload_bound(scene, samples) -> int:
    """Upper bound of the prefix bytes one compile of ``scene`` stores (module docstring)."""
    from .materials import _material_dtype

    count = math.prod(int(s) for s in samples)
    nodes = scene.Nx * scene.Ny * scene.Nz
    size = torch.empty((), dtype=_material_dtype(scene)).element_size()
    poles = sum(len(getattr(s.material, key, ()) or ()) for s in scene.scene.structures for key in _POLE_KEYS)
    return count * nodes * (size * (24 + poles) + 1) + 6 * count * nodes * (3 * size + 1)


class _Frame:
    def __init__(self, key, public, state, owner):
        self.key = key
        self.structures = tuple(weakref.ref(s, owner._dropper(self)) for s in public.structures)
        self.coordinates = public.coordinates
        self.state = state
        self.entries = {}
        self.bytes = 0
        self.dead = False


class StaticPrefixCache:
    """Process-wide frames of static structure prefixes (module docstring)."""

    def __init__(self):
        self._frames = OrderedDict()
        self._lock = threading.RLock()
        self._counts = dict.fromkeys(("hits", "misses", "frames", "evictions", "dropped", "stale",
                                      "not_admitted", "not_admitted_bytes", "not_stored_bytes"), 0)

    @property
    def stats(self):
        with self._lock:
            live = [frame for frame in self._frames.values() if not frame.dead]
            return {**self._counts, "entries": sum(len(frame.entries) for frame in live),
                    "bytes": sum(frame.bytes for frame in live), "live_frames": len(live)}

    def clear(self):
        """Release every frame, keeping the cumulative counters."""
        with self._lock:
            self._frames.clear()

    def _dropper(self, frame):
        frame_ref = weakref.ref(frame)

        def drop(_):
            # Runs from garbage collection: mark and release only, never mutate the frame table here.
            frame = frame_ref()
            if frame is not None and not frame.dead:
                frame.dead = True
                frame.entries = {}
                frame.bytes = 0
                # The snapshot holds the attribute values of the dead structures' geometry and materials.
                frame.state = None
                self._counts["dropped"] += 1
        return drop

    def frame(self, scene, samples, eps_background, mu_background):
        """The frame serving one compile of ``scene``, or ``None`` when it is not admitted."""
        from ..scene import PreparedScene
        from .materials import _material_dtype

        if (not isinstance(scene, PreparedScene) or not scene.scene.material_regions
                or getattr(scene, "_conductivity_capture", None) is not None
                or not all(type(v) in (int, float) for v in (eps_background, mu_background))):
            return None
        public = scene._public_scene
        prepared = scene.scene.structures
        extruded = {record["structure_index"] for record in scene.pml_extrusions
                    if record.get("kind") in ("box", "wrapper") and "structure_index" in record}
        if len(prepared) != len(public.structures) or not all(
                p is q or (index in extruded and p.material is q.material and p.name == q.name
                           and p.priority == q.priority)
                for index, (p, q) in enumerate(zip(prepared, public.structures))):
            self._count("not_admitted")
            return None
        state = _State((public.structures, public.coordinates), snapshot=False)
        if state.trainable:
            self._count("not_admitted")
            return None
        if payload_bound(scene, samples) > FRAME_BYTES:
            self._count("not_admitted_bytes")
            return None
        device = scene.x.device
        grid = tuple(getattr(scene, name.format(axis)).tobytes() for axis in "xyz"
                     for name in ("{}_nodes64", "d{}_primal64", "{}_half64", "d{}_dual64"))
        key = (str(device), torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None,
               _material_dtype(scene), torch.is_inference_mode_enabled(), grid,
               id(public.coordinates), tuple(id(s) for s in public.structures))
        with self._lock:
            frame = self._frames.get(key)
            if frame is not None and not frame.dead and frame.coordinates is public.coordinates and all(
                    ref() is s for ref, s in zip(frame.structures, public.structures)) and (
                    frame.domain == scene.scene.domain and frame.boundary == scene.scene.boundary
                    and frame.state.same(state)):
                self._frames.move_to_end(key)
                return frame
            if frame is not None:
                self._counts["stale"] += int(not frame.dead)
                del self._frames[key]
            for stale in [k for k, f in self._frames.items() if f.dead]:
                del self._frames[stale]
            while len(self._frames) >= MAX_FRAMES:
                self._frames.popitem(last=False)
                self._counts["evictions"] += 1
            frame = _Frame(key, public, _State((public.structures, public.coordinates), snapshot=True), self)
            frame.domain, frame.boundary = scene.scene.domain, scene.scene.boundary
            self._frames[key] = frame
            self._counts["frames"] += 1
            return frame

    def _count(self, name):
        with self._lock:
            self._counts[name] += 1

    def prefix(self, frame, key, compute):
        with self._lock:
            entry = frame.entries.get(key)
            if entry is not None:
                self._counts["hits"] += 1
        if entry is not None:
            return _owned(entry, {})
        value = compute()
        stored = _owned(value, {})
        size = _tree_bytes(stored, set())
        with self._lock:
            self._counts["misses"] += 1
            if frame.dead or frame.bytes + size > FRAME_BYTES:
                self._counts["not_stored_bytes"] += int(not frame.dead)
            else:
                frame.entries[key] = stored
                frame.bytes += size
        return value


STATIC_PREFIX_CACHE = StaticPrefixCache()


@contextmanager
def static_prefix_frame(scene, samples, eps_background, mu_background):
    """Serve the static prefixes of one compile of ``scene`` from its frame, when admitted."""
    frame = STATIC_PREFIX_CACHE.frame(scene, samples, eps_background, mu_background)
    token = _ACTIVE.set(None if frame is None else (scene, frame))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def static_prefix(scene, key, compute):
    """``compute()``, or an owned copy of its stored result when ``scene`` compiles in a frame.

    ``key`` is ``None`` for a call whose inputs the frame key does not cover.
    """
    active = _ACTIVE.get()
    if key is None or active is None or active[0] is not scene:
        return compute()
    return STATIC_PREFIX_CACHE.prefix(active[1], key, compute)


GEOMETRY_ENTRIES = 4096
GEOMETRY_BYTES = 64 * 1024 ** 2
GEOMETRY_ENTRY_BYTES = 8 * 1024 ** 2


class _GeometryEntry:
    def __init__(self, geometry, state, value):
        self.state, self.value, self.bytes, self.dead = state, value, state.bytes, False
        entry_ref = weakref.ref(self)

        def drop(_):
            # Runs from garbage collection: release the snapshot (it holds the geometry's attribute
            # values) and mark the entry; the table is purged under the lock by the next lookup.
            entry = entry_ref()
            if entry is not None:
                entry.dead, entry.state, entry.value = True, None, None
        self.geometry = weakref.ref(geometry, drop)


class GeometryMemo:
    """Results of pure per-geometry functions of a prepared grid, reused by geometry identity.

    An entry is keyed by the function tag, the geometry (``is``, through a weak reference) and the
    caller's grid key; a lookup also compares the geometry's state (attribute bindings and tensor
    contents) with the snapshot taken when the entry was stored. Geometry that requires grad is
    never memoized. The snapshot holds the geometry's attribute values and a copy of its tensor
    contents (on the geometry's device): an entry is released when its geometry dies (the weak
    reference callback frees the snapshot and value; the next lookup purges the entry), a geometry
    whose snapshot exceeds ``GEOMETRY_ENTRY_BYTES`` (8 MiB) is not memoized, and at most
    ``GEOMETRY_ENTRIES`` (4096) entries and ``GEOMETRY_BYTES`` (64 MiB) of snapshot contents are
    kept, evicted least recently used first (deterministic, never memory-pressure driven). Callers
    receive the stored value and must not mutate it.
    """

    def __init__(self):
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(("hits", "misses", "not_memoized", "evictions", "purged"), 0)

    @property
    def stats(self):
        with self._lock:
            live = [entry for entry in self._entries.values() if not entry.dead]
            return {**self._counts, "entries": len(live), "bytes": sum(entry.bytes for entry in live)}

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def _purge(self):
        for key in [key for key, entry in self._entries.items() if entry.dead]:
            self._bytes -= self._entries.pop(key).bytes
            self._counts["purged"] += 1

    def get(self, tag, geometry, grid_key, compute):
        state = _State((geometry,), snapshot=False)
        if state.trainable:
            return compute()
        key = (tag, id(geometry), grid_key)
        with self._lock:
            self._purge()
            entry = self._entries.get(key)
            if entry is not None and entry.geometry() is geometry and entry.state.same(state):
                self._entries.move_to_end(key)
                self._counts["hits"] += 1
                return entry.value
        value = compute()
        snapshot = _State((geometry,), snapshot=True)
        with self._lock:
            self._counts["misses"] += 1
            if snapshot.bytes > GEOMETRY_ENTRY_BYTES:
                self._counts["not_memoized"] += 1
                return value
            if key in self._entries:
                self._bytes -= self._entries.pop(key).bytes
            self._entries[key] = _GeometryEntry(geometry, snapshot, value)
            self._bytes += snapshot.bytes
            while len(self._entries) > GEOMETRY_ENTRIES or self._bytes > GEOMETRY_BYTES:
                self._bytes -= self._entries.popitem(last=False)[1].bytes
                self._counts["evictions"] += 1
        return value


GEOMETRY_MEMO = GeometryMemo()
