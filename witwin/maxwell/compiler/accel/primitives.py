"""Exact distance pieces with torch transforms, taper and occupancy arithmetic."""
from types import SimpleNamespace

import torch
from witwin.core import GeometryBase

from ...geometry.polyslab import PolySlab, _axial_split
from .build import build_extension
from .geometry import descriptors, native_geometry


class _DistanceGeometry:
    def __init__(self, geometry):
        self.geometry = geometry
        self._native = native_geometry(geometry)
        self.packed = {}

    def _descriptor(self, device):
        if device not in self.packed:
            self.packed[device] = descriptors(
                [SimpleNamespace(geometry=self.geometry)], device=device, native_coverage=True)
        return self.packed[device]

    def signed_distance(self, xx, yy, zz):
        geometry = self.geometry
        if self._native:
            points = torch.stack((xx, yy, zz), -1).reshape(-1, 3)
            return build_extension().distance(self._descriptor(points.device), points)[0].reshape(xx.shape)
        # Preserve core's local frame, torch tan and all stored slab intermediates.
        dx, dy, dz = geometry._local_coords(xx, yy, zz)
        axial, pu, pv = _axial_split(dx, dy, dz, geometry.axis)
        vertices = geometry.vertices.to(device=pu.device, dtype=pu.dtype).contiguous()
        loops = torch.split(vertices, list(geometry.loop_sizes), dim=0)
        ends = torch.cat([torch.roll(loop, shifts=-1, dims=0) for loop in loops])
        points = torch.stack((pu, pv), -1).reshape(-1, 2)
        polygon_distance = build_extension().polygon_distance(points, vertices, ends).reshape(pu.shape)
        bounds = geometry.bounds.to(device=axial.device, dtype=axial.dtype)
        tan_angle = torch.tan(geometry.sidewall_angle.to(device=axial.device, dtype=axial.dtype))
        tapered_distance = polygon_distance + tan_angle * (axial - geometry._reference_position(bounds))
        axial_distance = torch.abs(axial - .5 * (bounds[0] + bounds[1])) - .5 * (bounds[1] - bounds[0])
        outside = torch.sqrt(torch.clamp(tapered_distance, min=0.) ** 2
                             + torch.clamp(axial_distance, min=0.) ** 2)
        inside = torch.clamp(torch.maximum(tapered_distance, axial_distance), max=0.)
        return outside + inside

    to_mask = GeometryBase.to_mask


class _CoverageGeometry(_DistanceGeometry):
    def __init__(self, geometry, spacing):
        super().__init__(geometry)
        self.spacing = spacing

    def to_mask(self, xx, yy, zz, offset=0., beta=None):
        from ..subpixel import _broadcast_spacing, _NORMAL_EPS

        shape = xx.shape
        spacing = [_broadcast_spacing(h, a, shape) for a, h in enumerate(self.spacing)]
        delta = .25*min(float(h.min()) if torch.is_tensor(h) else h for h in spacing)
        widths = [torch.as_tensor(h, device=xx.device, dtype=xx.dtype).reshape(-1).expand(shape[a]).contiguous()
                  for a, h in enumerate(spacing)]
        points = torch.stack((xx, yy, zz), -1).reshape(-1, 3)
        return build_extension().coverage(self._descriptor(points.device), points, widths, list(shape),
            [delta, _NORMAL_EPS*delta*delta, 1./(delta*delta)]).reshape(shape)


def sampling_geometry(geometry, *, spacing=None):
    # Mesh/mesh booleans and Pyramid already reach the scoped core distance hook.
    # Unknown geometry keeps its authored to_mask and signed_distance methods.
    if native_geometry(geometry) and spacing is not None:
        return _CoverageGeometry(geometry, spacing)
    if native_geometry(geometry) or type(geometry) is PolySlab:
        return _DistanceGeometry(geometry)
    return geometry
