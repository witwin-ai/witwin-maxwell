"""Scoped mesh distance acceleration; winding retains the installed torch tree."""
from collections import OrderedDict
import weakref

import torch
from witwin.core.geometry import mesh_sdf as core

from .build import build_extension
from .dispatch import setup_enabled


class MeshAccelerator:
    def __init__(self):
        self._cache = OrderedDict()

    def supports(self, points, vertices):
        return (setup_enabled.get() and points.device.type == "cuda"
                and vertices.device == points.device
                and points.dtype == vertices.dtype == torch.float32
                and points.ndim == 2 and points.shape[1] == 3
                and vertices.ndim in (2, 3) and vertices.shape[-1] == 3
                and points.numel() > 0 and vertices.numel() > 0
                and not points.requires_grad and not vertices.requires_grad)

    def is_available(self):
        # Loading here would let core's optional-loader catch a build failure.
        # Build in the operation instead, where failures propagate to the caller.
        return setup_enabled.get()

    def _hierarchy(self, triangles, bvh):
        key = (triangles.device, triangles.dtype, triangles.untyped_storage().data_ptr(),
               triangles.storage_offset(), tuple(triangles.shape), tuple(triangles.stride()),
               triangles._version)
        cached = self._cache.get(key)
        if cached is not None and cached[0]() is triangles:
            self._cache.move_to_end(key)
            stream = torch.cuda.current_stream(triangles.device)
            stream.wait_event(cached[2])
            for tensor in cached[1]:
                tensor.record_stream(stream)
            return cached[1]
        extension = build_extension()
        if bvh is None:
            extent = torch.stack((triangles.amin((0, 1)), triangles.amax((0, 1))))
            keys = extension.mesh_keys(triangles.contiguous(), extent)
            ids = keys.argsort(stable=True).to(torch.int32)
            topology = extension.mesh_topology(keys[ids.long()].contiguous())
        else:
            # CPU-authored core trees already carry original triangle IDs.
            ids = bvh["triangle_indices"].to(device=triangles.device, dtype=torch.int32).contiguous()
            topology = torch.stack([bvh[name].to(device=triangles.device, dtype=torch.int32)
                                    for name in ("left", "right", "start", "count")])
        bounds = extension.mesh_bounds(triangles.contiguous(), topology, ids)
        hierarchy = (topology, ids, bounds)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(triangles.device))
        self._cache[key] = (weakref.ref(triangles), hierarchy, ready)
        # Mesh owns the transformed triangle tensor and invalidates it for every
        # transform/storage revision. Do not retain expired geometry allocations.
        for stale in [k for k, entry in self._cache.items() if entry[0]() is None]:
            del self._cache[stale]
        while len(self._cache) > 16:
            self._cache.popitem(last=False)
        return hierarchy

    def unsigned_distance(self, points, triangles):
        return self.unsigned_distance_static_bvh(points, triangles, None)

    def unsigned_distance_static_bvh(self, points, triangles, bvh):
        if not self.supports(points, triangles):
            return core._triangle_mesh_unsigned_distance_torch_from_triangles(points, triangles)
        hierarchy = self._hierarchy(triangles, bvh) if len(triangles) >= 32 else ()
        return build_extension().mesh_distance(points.contiguous(), triangles.contiguous(), list(hierarchy))

    def winding_angle(self, points, triangles):
        return core._triangle_mesh_winding_angle_torch_from_triangles(points, triangles)

    def smooth_signed_distance(self, points, triangles, *, winding_beta=0.5):
        return core._signed_distance_from_unsigned_and_winding(
            self.unsigned_distance(points, triangles), self.winding_angle(points, triangles), winding_beta)

    def signed_distance_static_bvh(self, points, triangles, bvh):
        return core._signed_distance_from_unsigned_and_winding(
            self.unsigned_distance_static_bvh(points, triangles, bvh),
            self.winding_angle(points, triangles), 0.5)


accelerator = MeshAccelerator()
core._register_mesh_sdf_accelerator(accelerator)
