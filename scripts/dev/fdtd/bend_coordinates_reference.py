"""Independent PyTorch metric identities for a possible bent-coordinate route.

Validation only. No public Scene mapping or solver feature is enabled here.
Coordinates q=(s,u,z) have physical SI length units; s is neutral-surface
arclength, u is signed normal distance, and z is the extrusion coordinate.
"""
from __future__ import annotations

import math
import torch


class ConstantCurvatureChart:
    def __init__(self, curvature, *, origin=(0.,0.,0.), frame=None):
        self.curvature = float(curvature)
        if not math.isfinite(self.curvature):
            raise ValueError("Chart curvature must be finite.")
        self.origin = origin
        self.frame = frame

    def _frame(self, q):
        frame = torch.eye(3,device=q.device,dtype=q.dtype) if self.frame is None else torch.as_tensor(self.frame,device=q.device,dtype=q.dtype)
        if frame.shape != (3,3) or not torch.allclose(frame.T@frame,torch.eye(3,device=q.device,dtype=q.dtype),rtol=0,atol=128*torch.finfo(q.dtype).eps) or not bool(torch.linalg.det(frame)>0):
            raise ValueError("Chart frame must be a right-handed orthonormal matrix.")
        return frame

    def metric(self, q):
        h = 1-self.curvature*q[...,1]
        if not bool(torch.isfinite(h).all() & (h>0).all()):
            raise ValueError("Orthogonal bend requires h_s=1-curvature*u>0.")
        return h

    def validate_bounds(self, bounds):
        if len(bounds)!=3 or any(not math.isfinite(float(a)) or not math.isfinite(float(b)) or a>=b for a,b in bounds):
            raise ValueError("Chart bounds must be three increasing finite intervals.")
        if min(1-self.curvature*bounds[1][0],1-self.curvature*bounds[1][1])<=0:
            raise ValueError("Chart domain crosses its focal singularity h_s=0.")
        if abs(self.curvature)*(bounds[0][1]-bounds[0][0])>=2*math.pi:
            raise ValueError("An open constant-curvature chart must not wrap or self-overlap.")

    def position(self, q):
        s,u,z=q.unbind(-1)
        angle=self.curvature*s
        # Entire functions make the flat limit exact without a 0/0 branch.
        gx=s*torch.sinc(angle/math.pi)
        gy=self.curvature*s.square()/2*torch.sinc(angle/(2*math.pi)).square()
        local=torch.stack((gx-u*torch.sin(angle),gy+u*torch.cos(angle),z),-1)
        return local@self._frame(q).T+q.new_tensor(self.origin)

    def jacobian(self,q):
        h=self.metric(q)
        angle=self.curvature*q[...,0]
        zero=torch.zeros_like(angle)
        t=torch.stack((torch.cos(angle),torch.sin(angle),zero),-1)
        n=torch.stack((-torch.sin(angle),torch.cos(angle),zero),-1)
        e=torch.stack((zero,zero,torch.ones_like(angle)),-1)
        local=torch.stack((h[...,None]*t,n,e),-1)
        return self._frame(q)@local

    def inverse(self, positions, *, bounds=None):
        local=(positions-positions.new_tensor(self.origin))@self._frame(positions)
        x,y,z=local.unbind(-1)
        if self.curvature==0:
            return local
        k=self.curvature
        radial=torch.sqrt((k*x).square()+(1-k*y).square())
        s=torch.atan2(k*x,1-k*y)/k
        if bounds is not None:
            self.validate_bounds(bounds)
            period=2*math.pi/abs(k)
            midpoint=(bounds[0][0]+bounds[0][1])/2
            s=s+torch.round((midpoint-s)/period)*period
        u=(2*y-k*(x.square()+y.square()))/(1+radial)
        return torch.stack((s,u,z),-1)

    def material_metric(self,q):
        h=self.metric(q)
        return torch.diag_embed(torch.stack((1/h,h,h),-1))

    def pullback_field(self,q,physical):
        return torch.einsum("...ji,...j->...i",self.jacobian(q).to(physical.dtype),physical)

    def physical_field(self,q,covariant):
        return torch.linalg.solve(self.jacobian(q).to(covariant.dtype).transpose(-1,-2),covariant[...,None])[...,0]

    def pullback_current(self,q,physical):
        return self.metric(q)[...,None]*torch.linalg.solve(self.jacobian(q).to(physical.dtype),physical[...,None])[...,0]

    def physical_area_vector(self,q,chart_normal):
        return self.metric(q)[...,None]*torch.linalg.solve(self.jacobian(q).transpose(-1,-2),chart_normal[...,None])[...,0]


def curl_from_autograd(field,q):
    derivatives=[]
    for component in range(3):
        real=torch.autograd.grad(field[...,component].real.sum(),q,create_graph=True,retain_graph=True)[0]
        if field.is_complex():
            imaginary=torch.autograd.grad(field[...,component].imag.sum(),q,create_graph=True,retain_graph=True)[0]
            real=torch.complex(real,imaginary)
        derivatives.append(real)
    return torch.stack((derivatives[2][...,1]-derivatives[1][...,2],
                        derivatives[0][...,2]-derivatives[2][...,0],
                        derivatives[1][...,0]-derivatives[0][...,1]),-1)
