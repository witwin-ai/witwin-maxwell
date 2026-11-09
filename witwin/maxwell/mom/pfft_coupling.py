"""Clustered exterior for sparse multi-body pFFT: per-cluster grids plus plane-wave coupling.

Bodies are merged into clusters until every pair of cluster bounding spheres
satisfies |X_ab| >= separation * (R_a + R_b). Inside a cluster the exterior D/K
channels use one ``PFFTRegion`` over that cluster only, with its own
precorrection, so empty volume between clusters costs no grid. Between
clusters the exact single-level plane-wave (diagonal translation) expansion

    G(r - r') = ik/(16 pi^2) int_{S^2} exp(ik s.d) T_L(s, X) ds,
    T_L(s, X) = sum_{l<=L} i^l (2l+1) h_l(k|X|) P_l(s . X/|X|),

with d = (r - c_a) - (r' - c_b) and X = c_a - c_b couples the radiation
patterns of RWG functions about their cluster centres. With
V_b(s) = sum_j x_j int f_j exp(-ik s.(r'-c_b)) and the analogous divergence
pattern S_b(s),

    (D x)_i = ik/(16 pi^2) sum_q w_q T_ab(s_q) [R_i . V_b - R_i^div S_b / k^2],
    (K x)_i = ik/(16 pi^2) sum_q w_q T_ab(s_q) R_i . (ik s_q x V_b),

with receive patterns R_i = conj(transmit pattern) for the real exterior k.
The quadrature is (L+1) Gauss-Legendre nodes in cos(theta) by 2L+2 uniform
azimuths, symmetric under s -> -s, so the coupling keeps D and K symmetric.
This is the top level of a later multilevel scheme; patterns are stored per
edge (N x Q x 4), which is the memory limit of the single level.
"""
from __future__ import annotations

import math

import numpy as np
import torch

PFFT_COUPLING_VERSION = 'pfft_single_level_plane_wave_coupling_v1'


def body_spheres(bases):
    """Bounding sphere (centre of the vertex box, radius to the farthest vertex) of each body."""
    spheres = []
    for basis in bases:
        vertices = basis.mesh.vertices
        centre = (vertices.amax(0)+vertices.amin(0))/2
        spheres.append((centre, float(torch.linalg.vector_norm(vertices-centre, dim=-1).max())))
    return spheres


def form_clusters(bases, separation):
    """Groups of body indices whose cluster spheres are pairwise separated by ``separation``."""
    groups = [[i] for i in range(len(bases))]
    while True:
        spheres = [body_spheres_union([bases[i] for i in group]) for group in groups]
        merged = False
        for a in range(len(groups)):
            for b in range(a+1, len(groups)):
                distance = float(torch.linalg.vector_norm(spheres[a][0]-spheres[b][0]))
                if distance < separation*(spheres[a][1]+spheres[b][1]):
                    groups[a] = sorted(groups[a]+groups[b])
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            return [sorted(g) for g in sorted(groups)], spheres


def body_spheres_union(bases):
    vertices = torch.cat([basis.mesh.vertices for basis in bases])
    centre = (vertices.amax(0)+vertices.amin(0))/2
    return centre, float(torch.linalg.vector_norm(vertices-centre, dim=-1).max())


def truncation_order(k, diameter, digits):
    """Excess-bandwidth rule L = kd + 1.8 d0^(2/3) (kd)^(1/3) for d0 digits."""
    kd = max(abs(k)*diameter, 1.)
    return int(math.ceil(kd+1.8*digits**(2/3)*kd**(1/3)))


def sphere_quadrature(order, device):
    """Directions [Q, 3] and weights [Q] integrating spherical harmonics of degree <= 2L+1 exactly."""
    nodes, weights = np.polynomial.legendre.leggauss(order+1)
    azimuths = 2*np.pi*np.arange(2*order+2)/(2*order+2)
    cos_t, phi = np.meshgrid(nodes, azimuths, indexing='ij')
    sin_t = np.sqrt(1-cos_t**2)
    directions = np.stack((sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t), -1).reshape(-1, 3)
    w = np.repeat(weights, len(azimuths))*2*np.pi/len(azimuths)
    return (torch.tensor(directions, device=device, dtype=torch.float64),
            torch.tensor(w, device=device, dtype=torch.float64))


def _spherical_hankel(order, x):
    """h_l^(1)(x) for l = 0..order by upward recurrence (stable for the outgoing function)."""
    values = [np.exp(1j*x)/(1j*x)]
    values.append(values[0]*(1/x-1j))
    for l in range(1, order):
        values.append((2*l+1)/x*values[l]-values[l-1])
    return np.array(values[:order+1])


def translation(order, k, displacement, directions):
    """T_L(s_q, X) [Q] for one cluster pair displacement X = c_a - c_b."""
    distance = float(torch.linalg.vector_norm(displacement))
    hankel = torch.tensor(_spherical_hankel(order, k*distance), device=directions.device)
    cosine = directions@(displacement/distance)
    previous, current = torch.ones_like(cosine), cosine
    result = hankel[0]*previous+3j*hankel[1]*current
    for l in range(1, order):
        previous, current = current, ((2*l+1)*cosine*current-l*previous)/(l+1)
        result = result+(1j**(l+1))*(2*l+3)*hankel[l+1]*current
    return result


def edge_patterns(basis, centre, k, directions, dtype, *, order=4, chunk=32):
    """Transmit patterns int f_i exp(-ik s.(r-c)) [E, Q, 3] and divergence patterns [E, Q]."""
    from .operator import local_geometry
    from .quadrature import triangle_rule
    _, triangles, slopes = local_geometry(basis)
    points, weights = triangle_rule(triangles, order)
    values = basis.local_values(points)
    relative = points-centre
    out = torch.zeros((basis.n_edges, len(directions), 4), device=points.device, dtype=dtype)
    edges = basis.face_edges.flatten()
    for start in range(0, len(directions), chunk):
        s = directions[start:start+chunk]
        phase = torch.exp(-1j*k*torch.einsum('tpc,qc->tpq', relative, s))*weights[..., None]
        vector = torch.einsum('tplc,tpq->tlqc', values.to(phase.dtype), phase)
        divergence = (2*slopes)[:, :, None]*phase.sum(1)[:, None, :]
        local = torch.cat((vector, divergence[..., None]), -1).to(dtype)
        out[:, start:start+chunk].index_add_(0, edges, local.reshape(-1, *local.shape[2:]))
    return out


class PlaneWaveCoupling:
    """Exterior D/K interactions between separated clusters (all ordered pairs a != b)."""

    def __init__(self, clusters, medium, *, digits, precision, device):
        k = complex(medium.k)
        if abs(k.imag) > 1e-12*abs(k.real):
            raise ValueError('Plane-wave cluster coupling requires a lossless exterior')
        self.k = k.real
        self.dtype = torch.complex128 if precision == 'complex128' else torch.complex64
        self.indices = [index for index, _, _, _ in clusters]
        centres = [centre for _, _, centre, _ in clusters]
        radii = [radius for _, _, _, radius in clusters]
        diameter = max(radii[a]+radii[b] for a in range(len(clusters)) for b in range(len(clusters)) if a != b)
        self.order = truncation_order(self.k, diameter, digits)
        self.directions, self.weights = sphere_quadrature(self.order, device)
        self.patterns = []
        for index, bases, centre, _ in clusters:
            self.patterns.append(torch.cat([edge_patterns(basis, centre, self.k, self.directions, self.dtype)
                                            for basis in bases]))
        prefactor = 1j*self.k/(16*math.pi**2)
        self.pairs, self.translations = [], []
        separations = []
        for a in range(len(clusters)):
            for b in range(len(clusters)):
                if a == b:
                    continue
                displacement = centres[a]-centres[b]
                separations.append(float(torch.linalg.vector_norm(displacement))/(radii[a]+radii[b]))
                t = translation(self.order, self.k, displacement.to(torch.float64), self.directions)
                self.pairs.append((a, b))
                self.translations.append((prefactor*self.weights*t).to(self.dtype))
        self.translations = torch.stack(self.translations)
        self.minimum_separation_ratio = min(separations)
        self.digits = digits

    def channels(self, value, count):
        """Far-field (D x, K x) [count] contributions from every other cluster."""
        x = value.to(self.dtype)
        spectra = [torch.einsum('eqc,e->qc', pattern, x[index])
                   for pattern, index in zip(self.patterns, self.indices)]
        incoming = [torch.zeros_like(spectra[0]) for _ in spectra]
        for (a, b), t in zip(self.pairs, self.translations):
            incoming[a] += t[:, None]*spectra[b]
        d = torch.zeros(count, device=value.device, dtype=torch.complex128)
        kk = torch.zeros_like(d)
        s = self.directions.to(self.dtype)
        for pattern, index, field in zip(self.patterns, self.indices, incoming):
            receive = pattern.conj()
            vector, scalar = field[:, :3], field[:, 3]
            d[index] = (torch.einsum('eqc,qc->e', receive[..., :3], vector)
                        - torch.einsum('eq,q->e', receive[..., 3], scalar)/self.k**2).to(torch.complex128)
            curl = 1j*self.k*torch.linalg.cross(s, vector, dim=-1)
            kk[index] = torch.einsum('eqc,qc->e', receive[..., :3], curl).to(torch.complex128)
        return d, kk

    def tensors(self):
        return (*self.patterns, self.translations, self.directions, self.weights)

    def receipt(self):
        return {'version': PFFT_COUPLING_VERSION, 'order': self.order, 'directions': len(self.directions),
                'digits': self.digits, 'cluster_pairs': len(self.pairs),
                'minimum_separation_ratio': self.minimum_separation_ratio,
                'pattern_bytes': sum(p.numel()*p.element_size() for p in self.patterns),
                'translation_bytes': self.translations.numel()*self.translations.element_size()}


class ClusteredExterior:
    """Exterior D/K channels as per-cluster precorrected grids plus plane-wave coupling.

    Drop-in for ``PFFTRegionOperator`` of the exterior: ``channels`` and
    ``apply`` act on the global edge order of all bodies.
    """

    def __init__(self, bodies, medium, config, integration):
        from .pfft_grid import PFFTProjection
        from .pfft_operator import PFFTRegion
        from .pfft_system import PFFTPrecorrection, PFFTRegionOperator
        self.medium = medium
        self.electric, self.magnetic = 1j*medium.omega*medium.epsilon, -1j*medium.omega*medium.mu
        bases = [basis for basis, _, _ in bodies]
        self.count = sum(basis.n_edges for basis in bases)
        device = bases[0].mesh.device
        groups, spheres = form_clusters(bases, config.cluster_separation)
        if len(groups) < 2:
            raise ValueError('Clustered pFFT exterior needs at least two separated clusters; '
                             'use exterior_grids=single')
        self.groups, self.regions, clusters = groups, [], []
        for group, (centre, radius) in zip(groups, spheres):
            members = [bodies[i] for i in group]
            index = torch.cat([torch.arange(offset, offset+basis.n_edges, device=device)
                               for basis, offset, _ in members])
            local, cursor = [], 0
            for basis, _, geometry in members:
                local.append((basis, cursor, geometry))
                cursor += basis.n_edges
            member_bases = [basis for basis, _, _ in members]
            region = PFFTRegion(PFFTProjection.for_bases(member_bases, config.grid_spec(medium, member_bases)), medium)
            correction = PFFTPrecorrection(region, local, medium, integration, batch=config.near_pair_batch)
            self.regions.append((index, PFFTRegionOperator(region, correction, medium)))
            clusters.append((index, member_bases, centre, radius))
        self.coupling = PlaneWaveCoupling(clusters, medium, digits=config.digits,
                                          precision=config.grid_precision, device=device)

    def channels(self, value):
        d, k = self.coupling.channels(value, self.count)
        for index, region in self.regions:
            dc, kc = region.channels(value[index])
            d[index] += dc
            k[index] += kc
        return d, k

    def apply(self, current_j, current_m, *, transpose=False):
        dj, kj = self.channels(current_j)
        dm, km = self.channels(current_m)
        electric, magnetic = (self.magnetic, self.electric) if transpose else (self.electric, self.magnetic)
        return kj+electric*dm, magnetic*dj+km

    def tensors(self):
        return tuple(t for _, region in self.regions for t in region.tensors())+self.coupling.tensors()

    def receipt(self):
        clusters = [{'bodies': group, **region.receipt()} for group, (_, region) in zip(self.groups, self.regions)]
        coupling = self.coupling.receipt()
        return {'version': 'pfft_clustered_exterior_v1', 'clusters': clusters, 'coupling': coupling,
                'retained_bytes': (sum(c['retained_bytes'] for c in clusters)+coupling['pattern_bytes']
                                   + coupling['translation_bytes'])}
