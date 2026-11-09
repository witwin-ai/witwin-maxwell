"""Uniform-grid projection of RWG bases for FFT-accelerated surface operators.

Each RWG basis (its three Cartesian components and its surface divergence)
is replaced by point weights on an (M+1)^3 stencil of a uniform grid. The
weights match every tensor-product moment of order <= M per axis about the
stencil centre:

    sum_p w[p] (t_p)^alpha = integral f(r) ((r - c)/h)^alpha dS,

with integer node offsets t_p in [-M/2, M/2]. The stencil matrix is the same
for every basis, so the weights are one fixed Kronecker inverse applied to
moments computed by exact triangle quadrature. Testing uses the same weights
(Galerkin, bilinear), which makes the grid operator symmetric like the
original single-layer block.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .operator import local_geometry
from .quadrature import triangle_rule

CHANNELS = ('x', 'y', 'z', 'divergence')


@dataclass(frozen=True)
class PFFTGridSpec:
    """One grid: spacing [m], per-axis stencil order M ((M+1)^3 points), precision, near radius [m].

    ``mean_edge`` [m] and ``spacing_rule`` record how a configuration chose the
    spacing from the mesh (informational; ``None`` for directly specified grids).
    """
    spacing: float
    order: int = 3
    precision: str = 'complex128'
    precorrection_radius: float | None = None
    mean_edge: float | None = None
    spacing_rule: str | None = None

    def __post_init__(self):
        if type(self.spacing) not in (int, float) or not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError('pFFT spacing must be a positive finite length in metres')
        if type(self.order) is not int or not 1 <= self.order <= 6:
            raise ValueError('pFFT stencil order must be an integer in 1..6')
        if self.precision not in ('complex128', 'complex64'):
            raise ValueError('pFFT precision must be complex128 or complex64')
        if self.precorrection_radius is not None and (type(self.precorrection_radius) not in (int, float)
                                                      or self.precorrection_radius <= 0):
            raise ValueError('precorrection_radius must be a positive length in metres')

    @property
    def stencil_points(self):
        return (self.order+1)**3

    @property
    def near_radius(self):
        """Basis-centre distance below which grid interactions are replaced (default (M+2) h)."""
        return self.precorrection_radius if self.precorrection_radius is not None else (self.order+2)*self.spacing


def _inverse_vandermonde(order, dtype, device):
    offsets = torch.arange(order+1, device=device, dtype=dtype)-order/2
    vandermonde = offsets[None, :]**torch.arange(order+1, device=device, dtype=dtype)[:, None]
    return torch.linalg.inv(vandermonde), offsets


class PFFTProjection:
    """Stencil indices and moment-matching weights of every RWG basis of one body.

    ``indices`` [n, S] are flat indices into the unpadded grid of ``shape``;
    ``weights`` [n, S, 4] hold the x, y, z and divergence channels (real).
    """

    def __init__(self, basis, config, *, origin=None, shape=None):
        if not isinstance(config, PFFTGridSpec):
            raise TypeError('PFFTProjection requires PFFTGridSpec')
        mesh = basis.mesh
        dtype, device = mesh.dtype, mesh.device
        self.basis, self.config = basis, config
        order, h = config.order, float(config.spacing)
        midpoints = (mesh.vertices[basis.edges[:, 0]]+mesh.vertices[basis.edges[:, 1]])/2
        if origin is None:
            origin = midpoints.amin(0)-(order/2+1)*h
        origin = torch.as_tensor(origin, device=device, dtype=dtype)
        base = torch.round((midpoints-origin)/h-order/2).to(torch.int64)
        if bool((base < 0).any()):
            raise ValueError('pFFT grid origin does not cover every stencil')
        extent = base.amax(0)+order+1
        if shape is None:
            shape = tuple(int(v) for v in extent.tolist())
        elif any(e > s for e, s in zip(extent.tolist(), shape)):
            raise ValueError('pFFT grid shape does not cover every stencil')
        self.origin, self.shape, self.base = origin, tuple(shape), base
        centres = origin+(base.to(dtype)+order/2)*h
        self.centres = centres
        self.weights = self._weights(basis, centres, order, h)
        local = torch.arange(order+1, device=device)
        lx, ly, lz = torch.meshgrid(local, local, local, indexing='ij')
        offsets = torch.stack((lx.flatten(), ly.flatten(), lz.flatten()), -1)
        nodes = base[:, None, :]+offsets[None]
        self.indices = (nodes[..., 0]*shape[1]+nodes[..., 1])*shape[2]+nodes[..., 2]
        self.local_offsets = offsets
        self.count = basis.n_edges
        self.midpoints = midpoints
        self.device = device
        # Largest distance from an edge midpoint to a vertex of its two support triangles.
        corners = torch.stack((mesh.vertices[basis.edges[:, 0]], mesh.vertices[basis.edges[:, 1]],
                               mesh.vertices[basis.plus_vertices], mesh.vertices[basis.minus_vertices]), 1)
        self.support_radius = torch.linalg.vector_norm(corners-midpoints[:, None], dim=-1).amax(-1)

    @classmethod
    def for_bases(cls, bases, config):
        """Projections of several bodies on one common grid, concatenated in edge order."""
        order, h = config.order, float(config.spacing)
        midpoints = torch.cat([(b.mesh.vertices[b.edges[:, 0]]+b.mesh.vertices[b.edges[:, 1]])/2 for b in bases])
        origin = midpoints.amin(0)-(order/2+1)*h
        base = torch.round((midpoints-origin)/h-order/2).to(torch.int64)
        shape = tuple(int(v) for v in (base.amax(0)+order+1).tolist())
        parts = [cls(b, config, origin=origin, shape=shape) for b in bases]
        if len(parts) == 1:
            return parts[0]
        combined = cls.__new__(cls)
        combined.basis, combined.config = None, config
        combined.origin, combined.shape, combined.device = origin, shape, parts[0].device
        for name in ('base', 'centres', 'weights', 'indices', 'midpoints', 'support_radius'):
            setattr(combined, name, torch.cat([getattr(p, name) for p in parts]))
        combined.local_offsets = parts[0].local_offsets
        combined.count = sum(p.count for p in parts)
        return combined

    @staticmethod
    def _weights(basis, centres, order, h, *, face_local=False):
        """RWG stencil weights, optionally decomposed by support face for precorrection."""
        edges, triangles, slopes = local_geometry(basis)
        dtype, device = triangles.dtype, triangles.device
        rule = (3*order+4)//2
        points, weights = triangle_rule(triangles, rule)
        powers = torch.arange(order+1, device=device, dtype=dtype)
        layout = (len(triangles), 3) if face_local else (basis.n_edges,)
        moments = torch.zeros((*layout, 4, order+1, order+1, order+1), device=device, dtype=dtype)
        for local in range(3):
            edge = edges[:, local]
            u = (points-centres[edge][:, None])/h
            px, py, pz = (u[..., d, None]**powers for d in range(3))
            current = slopes[:, local, None, None]*(points-triangles[:, None, local])
            divergence = (2*slopes[:, local])[:, None, None].expand(-1, points.shape[1], 1)
            values = torch.cat((current, divergence), -1)
            contribution = torch.einsum('fq,fqc,fqa,fqb,fqd->fcabd', weights, values, px, py, pz)
            if face_local:
                moments[:, local] = contribution
            else:
                moments.index_add_(0, edge, contribution)
        inverse, _ = _inverse_vandermonde(order, dtype, device)
        moments = moments.reshape(-1, 4, order+1, order+1, order+1)
        stencil = torch.einsum('pa,qb,rd,ecabd->epqrc', inverse, inverse, inverse, moments)
        return stencil.reshape(*layout, (order+1)**3, 4)

    def receipt(self):
        values = sum(t.numel()*t.element_size() for t in (self.indices, self.weights))
        edge = self.config.mean_edge
        return {'version': 'pfft_moment_projection_v1', 'order': self.config.order,
                'spacing_m': self.config.spacing, 'mean_edge_m': edge, 'spacing_rule': self.config.spacing_rule,
                'spacing_to_mean_edge': None if edge is None else self.config.spacing/edge,
                'grid_shape': list(self.shape),
                'stencil_points': self.config.stencil_points, 'bases': self.count,
                'retained_bytes': values}
