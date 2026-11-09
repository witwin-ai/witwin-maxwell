"""Physical coordinate charts and explicit frames for native field data."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import numpy as np
from collections.abc import Mapping


def _has_gradient(value):
    if torch.is_tensor(value):
        return value.requires_grad
    if isinstance(value, np.ndarray) and value.dtype == object:
        return any(_has_gradient(item) for item in value.flat)
    if isinstance(value, Mapping):
        return any(_has_gradient(item) for item in value.values())
    return isinstance(value,(tuple,list)) and any(_has_gradient(item) for item in value)


class BendCoordinates:
    """Fixed orthogonal constant-curvature chart q=(s,u,z), in SI metres.

    Scene positions and geometry are declared in this chart. Scalar material
    properties remain physical. Native fields are covariant chart components;
    use Result.query_fields for world Cartesian vectors.
    """
    kind = "orthogonal_bend"

    def __init__(self, *, curvature, origin=(0.,0.,0.), frame=None, device=None):
        for value in (curvature, origin, frame):
            if _has_gradient(value):
                raise NotImplementedError("BendCoordinates currently requires fixed chart geometry.")
        self._curvature = float(curvature)
        if not math.isfinite(self.curvature):
            raise ValueError("BendCoordinates curvature must be finite.")
        self._origin = torch.as_tensor(origin, dtype=torch.float64, device=device).clone()
        self._frame = (torch.eye(3, dtype=torch.float64, device=self._origin.device) if frame is None
                      else torch.as_tensor(frame, dtype=torch.float64, device=self._origin.device).clone())
        if self._origin.shape != (3,) or not bool(torch.isfinite(self._origin).all()):
            raise ValueError("BendCoordinates origin must contain three finite lengths.")
        if (self._frame.shape != (3,3) or not bool(torch.isfinite(self._frame).all())
                or not bool(torch.allclose(self._frame.T@self._frame, torch.eye(3, dtype=torch.float64, device=self._frame.device), rtol=0, atol=128*torch.finfo(torch.float64).eps))
                or not bool(torch.linalg.det(self._frame)>0)):
            raise ValueError("BendCoordinates frame must be right-handed and orthonormal in float64.")

    @property
    def curvature(self):
        return self._curvature

    @property
    def origin(self):
        return self._origin.clone()

    @property
    def frame(self):
        return self._frame.clone()

    def to(self, device):
        return type(self)(curvature=self.curvature, origin=self._origin.to(device), frame=self._frame.to(device))

    def validate_bounds(self, bounds):
        if len(bounds)!=3 or any(not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or lo>=hi for lo,hi in bounds):
            raise ValueError("BendCoordinates bounds require three finite increasing intervals.")
        minimum = min(1-self.curvature*bounds[1][0],1-self.curvature*bounds[1][1])
        if minimum<=0:
            raise ValueError("BendCoordinates physical or padded domain crosses the focal singularity h_s=0.")
        if abs(self.curvature)*(bounds[0][1]-bounds[0][0])>=2*math.pi:
            raise ValueError("BendCoordinates open chart wraps or self-intersects over its domain.")
        return minimum

    def metric_scale(self, points):
        return 1-self.curvature*points[...,1]

    def position(self, points):
        s,u,z = points.unbind(-1)
        angle = self.curvature*s
        gx = s*torch.sinc(angle/math.pi)
        gy = self.curvature*s.square()/2*torch.sinc(angle/(2*math.pi)).square()
        local = torch.stack((gx-u*torch.sin(angle),gy+u*torch.cos(angle),z),-1)
        return local@self._frame.to(points).T+self._origin.to(points)

    def jacobian(self, points):
        angle = self.curvature*points[...,0]
        sine, cosine = torch.sin(angle),torch.cos(angle)
        zero = torch.zeros_like(angle)
        tangent = torch.stack((cosine,sine,zero),-1)
        normal = torch.stack((-sine,cosine,zero),-1)
        axial = torch.stack((zero,zero,torch.ones_like(angle)),-1)
        local = torch.stack((self.metric_scale(points)[...,None]*tangent,normal,axial),-1)
        return self._frame.to(points)@local

    def inverse(self, points, *, bounds):
        self.validate_bounds(bounds)
        local = (points-self._origin.to(points))@self._frame.to(points)
        if self.curvature==0:
            return local
        x,y,z = local.unbind(-1)
        k = self.curvature
        radial = torch.sqrt((k*x).square()+(1-k*y).square())
        s = torch.atan2(k*x,1-k*y)/k
        period = 2*math.pi/abs(k)
        middle = (bounds[0][0]+bounds[0][1])/2
        s = s+torch.round((middle-s)/period)*period
        u = (2*y-k*(x.square()+y.square()))/(1+radial)
        return torch.stack((s,u,z),-1)

    def physical_field(self, points, covariant):
        return torch.linalg.solve(self.jacobian(points).to(covariant.dtype).transpose(-1,-2),covariant[...,None])[...,0]

    def pullback_field(self, points, physical):
        return torch.einsum("...ji,...j->...i",self.jacobian(points).to(physical.dtype),physical)

    def physical_area_vector(self, points, chart_normal):
        return self.metric_scale(points)[...,None]*torch.linalg.solve(self.jacobian(points).transpose(-1,-2),chart_normal[...,None])[...,0]

    def snapshot(self):
        return dict(schema_version=1,kind=self.kind,curvature=self.curvature,
                    origin=self._origin.detach().cpu().tolist(),frame=self._frame.detach().cpu().tolist())


@dataclass(frozen=True)
class FieldFrame:
    kind: str
    coordinates: BendCoordinates

    def snapshot(self):
        return dict(schema_version=1,kind=self.kind,coordinates=self.coordinates.snapshot())


class FramedFieldData:
    """Native covariant chart arrays with explicit component/sample identity."""
    _NATIVE = dict(E_s="Ex",E_u="Ey",E_z="Ez",H_s="Hx",H_u="Hy",H_z="Hz")

    def __init__(self, result):
        self._result = result
        self.frame = FieldFrame("covariant_chart",result.scene.coordinates)

    @property
    def components(self):
        self._result._require_frequency_volume_fields()
        return tuple(label for label,name in self._NATIVE.items() if name.upper() in self._result._fields)

    def tensor(self, component, *, frequency=None, freq_index=None):
        if component not in self._NATIVE:
            raise ValueError("Chart fields use E_s/E_u/E_z/H_s/H_u/H_z component labels.")
        return self._result._native_tensor(self._NATIVE[component],frequency=frequency,freq_index=freq_index)

    def coordinate_axes(self, component):
        if component not in self._NATIVE:
            raise ValueError("Unknown covariant chart field component.")
        from .fdtd.coords import component_coords
        field = self.tensor(component)
        return tuple(torch.as_tensor(axis,dtype=torch.float64,device=field.device)
                     for axis in component_coords(self._result.prepared_scene,self._NATIVE[component]))

    def positions(self, component, *, coordinates="chart"):
        if coordinates not in ("chart","world"):
            raise ValueError("Field positions use coordinates='chart' or 'world'.")
        points = torch.stack(torch.meshgrid(*self.coordinate_axes(component),indexing="ij"),-1)
        return points if coordinates=="chart" else self.frame.coordinates.position(points)


def _frame_monitor_payload(payload, frame):
    """Label native chart samples without changing their retained values."""
    selected=dict(payload)
    inverse={native:label for label,native in FramedFieldData._NATIVE.items()}
    selected.update(field_frame=frame,coordinate_kind="chart",coordinate_axes={"x":"s","y":"u","z":"z"})
    if "faces" in selected:
        selected["faces"]={name:_frame_monitor_payload(face,frame) for name,face in selected["faces"].items()}
    if "components" in selected:
        selected["components"]={inverse.get(key,key):value for key,value in selected["components"].items()}
    for key in tuple(inverse):
        if key in selected:
            selected[inverse[key]]=selected.pop(key)
    if "field_indices" in selected:
        selected["field_indices"]={inverse.get(key,key):value for key,value in selected["field_indices"].items()}
    if selected.get("component") is not None:
        native=str(selected["component"])
        selected["component"]=inverse.get(native[0].upper()+native[1:].lower(),native)
    if "fields" in selected:
        selected["fields"]=tuple(inverse.get(key,key) for key in selected["fields"])
    if "flux" in selected:
        selected["power_unit"]="W"
        selected["flux_frame"]="physical integrated flux from invariant chart tangential fields"
    return selected
