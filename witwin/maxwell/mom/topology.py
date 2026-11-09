"""Device-resident triangle-surface topology and per-body MoM kinds.

Edges are classified by their number of incident faces: interior (two),
boundary (one) or junction (three or more). Classification, orientation and
vertex-link checks use tensor operations on the faces' own device; nothing
iterates over faces on the host. Edge order is the lexicographic order of the
ascending vertex pair, identical to the closed-surface ``RWGBasis`` order.

Each enabled ``Structure`` also has a body kind. The kind selects the surface
formulation: closed dielectric bodies use PMCHWT and zero-thickness PEC sheets
use the open-surface EFIE. Other kinds are named here so every runtime rejects
them explicitly instead of reinterpreting their material.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

INTERIOR, BOUNDARY, JUNCTION = 0, 1, 2
EDGE_KIND_NAMES = ('interior', 'boundary', 'junction')


@dataclass(frozen=True)
class SurfaceTopology:
    """Edge classification of one triangle surface.

    ``face_edges[t, l]`` is the global edge opposite local vertex ``l`` and
    ``face_edge_signs[t, l]`` is +1 when the face traverses that edge from its
    smaller to its larger vertex index. ``incidence_faces/slots`` list, for
    each edge, its first two incidences ordered by descending sign (the plus
    face first); entries of boundary edges beyond their single face are -1.
    """
    edges: torch.Tensor
    face_count: torch.Tensor
    kind: torch.Tensor
    face_edges: torch.Tensor
    face_edge_signs: torch.Tensor
    incidence_faces: torch.Tensor
    incidence_slots: torch.Tensor
    interior_edges: int
    boundary_edges: int
    junction_edges: int
    components: int
    orientable: bool
    manifold_vertices: bool

    @property
    def closed(self):
        return self.boundary_edges == 0 and self.junction_edges == 0

    def summary(self):
        return {'edges': len(self.edges), 'interior_edges': self.interior_edges,
                'boundary_edges': self.boundary_edges, 'junction_edges': self.junction_edges,
                'components': self.components, 'orientable': self.orientable,
                'manifold_vertices': self.manifold_vertices, 'closed': self.closed}


def _component_labels(count, left, right):
    """Minimum-index labels of an undirected graph by hooking and pointer jumping."""
    labels = torch.arange(count, device=left.device)
    if not len(left):
        return labels
    while True:
        previous = labels
        smallest = torch.minimum(labels[left], labels[right])
        labels = labels.scatter_reduce(0, left, smallest, 'amin').scatter_reduce(0, right, smallest, 'amin')
        labels = labels[labels]
        if torch.equal(labels, previous):
            return labels


def classify_surface(faces, vertex_count):
    """Classify every edge and check orientation and vertex links of a triangle surface.

    ``faces`` is an int64 [T, 3] tensor on any device. Invalid indices, repeated
    face vertices and unused vertices raise; the topological properties are
    reported for the caller's formulation to accept or reject.
    """
    if (not isinstance(faces, torch.Tensor) or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype != torch.long or not len(faces)):
        raise ValueError('faces must be a nonempty int64 tensor with shape (T, 3)')
    if type(vertex_count) is not int or vertex_count < 3:
        raise ValueError('vertex_count must be an integer of at least three')
    if bool((faces < 0).any() | (faces >= vertex_count).any()):
        raise ValueError('Triangle indices are invalid or repeated')
    if bool(((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])).any()):
        raise ValueError('Triangle indices are invalid or repeated')
    used = torch.zeros(vertex_count, dtype=torch.bool, device=faces.device)
    used[faces.reshape(-1)] = True
    if not bool(used.all()):
        raise ValueError('Mesh contains unused vertices')
    count = len(faces)
    # Local slot l is the edge opposite vertex l, traversed (l+1) -> (l+2).
    start, end = faces[:, [1, 2, 0]], faces[:, [2, 0, 1]]
    low, high = torch.minimum(start, end), torch.maximum(start, end)
    signs = torch.where(start < end, 1, -1).to(torch.long)
    edges, inverse, face_count = torch.unique(torch.stack((low, high), -1).reshape(-1, 2), dim=0,
                                              return_inverse=True, return_counts=True)
    kind = torch.where(face_count == 2, INTERIOR, torch.where(face_count == 1, BOUNDARY, JUNCTION))
    # Group incidences by edge with the plus traversal first inside every group.
    flat_signs = signs.reshape(-1)
    order = torch.argsort(inverse*4 + (1-flat_signs), stable=True)
    first = torch.cumsum(face_count, 0)-face_count
    head = order[first]
    second = order[torch.where(face_count > 1, first+1, first)]
    incidence_faces = torch.stack((head//3, torch.where(face_count > 1, second//3, -1)), -1)
    incidence_slots = torch.stack((head % 3, torch.where(face_count > 1, second % 3, -1)), -1)
    interior = kind == INTERIOR
    sign_sum = torch.zeros(len(edges), dtype=torch.long, device=faces.device).index_add_(0, inverse, flat_signs)
    orientable = bool((sign_sum[interior] == 0).all())
    # Face components over interior edges.
    pair_faces = incidence_faces[interior]
    labels = _component_labels(count, pair_faces[:, 0], pair_faces[:, 1])
    components = int(torch.unique(labels).numel())
    # Vertex links: corner nodes (face, corner) joined across interior edges
    # through both shared vertices; one class per vertex is a single fan.
    slot_faces, slot_ids = pair_faces, incidence_slots[interior]
    endpoints = edges[interior]
    nodes_left, nodes_right = [], []
    for vertex in (endpoints[:, 0], endpoints[:, 1]):
        corners = []
        for side in range(2):
            face, slot = slot_faces[:, side], slot_ids[:, side]
            following = (slot+1) % 3
            corner = torch.where(faces[face, following] == vertex, following, (slot+2) % 3)
            corners.append(face*3+corner)
        nodes_left.append(corners[0])
        nodes_right.append(corners[1])
    corner_labels = _component_labels(count*3, torch.cat(nodes_left), torch.cat(nodes_right))
    vertex_classes = torch.unique(torch.stack((faces.reshape(-1), corner_labels), -1), dim=0)
    manifold_vertices = len(vertex_classes) == vertex_count
    return SurfaceTopology(
        edges=edges, face_count=face_count, kind=kind, face_edges=inverse.reshape(count, 3),
        face_edge_signs=signs, incidence_faces=incidence_faces, incidence_slots=incidence_slots,
        interior_edges=int(interior.sum()), boundary_edges=int((kind == BOUNDARY).sum()),
        junction_edges=int((kind == JUNCTION).sum()), components=components,
        orientable=orientable, manifold_vertices=manifold_vertices)


def require_open_sheet(topology):
    """Fail closed unless the surface is an oriented manifold sheet without junctions."""
    if topology.junction_edges:
        raise NotImplementedError(
            f'Surface has {topology.junction_edges} junction edges shared by three or more faces; '
            'junction RWG functions are not implemented, so T-junctions are rejected.')
    if not topology.manifold_vertices:
        raise ValueError('Surface has a non-manifold vertex (sheets touching at a single vertex)')
    if not topology.orientable:
        raise ValueError('Surface faces must have consistent orientation across every interior edge')
    return topology


class BodyKind(str, Enum):
    """Per-body surface model. Only dielectric closed and PEC open are implemented."""
    DIELECTRIC_CLOSED = 'dielectric_closed'
    PEC_CLOSED = 'pec_closed'
    PEC_OPEN = 'pec_open'
    IMPEDANCE_SHEET = 'impedance_sheet'


PMCHWT = 'pmchwt'
EFIE_OPEN_PEC = 'efie_open_pec'
FORMULATION_BODY_KIND = {PMCHWT: BodyKind.DIELECTRIC_CLOSED, EFIE_OPEN_PEC: BodyKind.PEC_OPEN}


def sheet_geometry(geometry):
    """True for zero-thickness Box/Cylinder or explicit surface Mesh geometry."""
    from witwin.core import Box, Cylinder, Mesh
    if isinstance(geometry, Mesh):
        return geometry.fill_mode == 'surface'
    if isinstance(geometry, Box):
        zero = int((geometry.size == 0).sum())
        if zero > 1:
            raise ValueError('A Box sheet may have exactly one zero dimension')
        return zero == 1
    if isinstance(geometry, Cylinder):
        return bool(geometry.height == 0)
    return False


def body_kind(structure):
    """Classify one Structure from its material and geometry topology."""
    from ..media import LossyMetalMedium, Medium2D
    material, geometry = structure.material, structure.geometry
    sheet = sheet_geometry(geometry)
    if isinstance(material, Medium2D):
        return BodyKind.IMPEDANCE_SHEET
    if getattr(material, 'pec', False):
        if not sheet:
            return BodyKind.PEC_CLOSED
        from witwin.core import Mesh
        if isinstance(geometry, Mesh):
            topology = classify_surface(geometry.faces.to(torch.long), geometry.vertex_count)
            if topology.closed:
                return BodyKind.PEC_CLOSED
        return BodyKind.PEC_OPEN
    if isinstance(material, LossyMetalMedium):
        raise NotImplementedError('MoM surface-impedance (Leontovich) conductors are not implemented')
    if sheet:
        raise ValueError('A zero-thickness dielectric sheet has no MoM model; use Material.pec() '
                         'or a closed dielectric body')
    return BodyKind.DIELECTRIC_CLOSED


_UNIMPLEMENTED = {
    BodyKind.PEC_CLOSED: 'closed PEC bodies require the combined-field formulation, which is not implemented',
    BodyKind.IMPEDANCE_SHEET: 'impedance sheets (Medium2D, Graphene) are not implemented',
}


def scene_formulation(structures):
    """Return the single surface formulation admitted by the enabled bodies."""
    kinds = [body_kind(structure) for structure in structures]
    for kind in kinds:
        if kind in _UNIMPLEMENTED:
            raise NotImplementedError(f'MoM body kind {kind.value!r}: {_UNIMPLEMENTED[kind]}')
    if kinds and all(kind is BodyKind.PEC_OPEN for kind in kinds):
        return EFIE_OPEN_PEC
    if BodyKind.PEC_OPEN in kinds:
        raise NotImplementedError('MoM scenes mixing open PEC sheets with dielectric bodies require a '
                                  'multiregion formulation, which is not implemented')
    return PMCHWT


def require_formulation(structures, formulation):
    """Fail closed when any body kind is outside ``formulation``."""
    if formulation not in FORMULATION_BODY_KIND:
        raise ValueError(f'Unknown MoM formulation {formulation!r}')
    actual = scene_formulation(structures)
    if actual != formulation:
        raise ValueError(f'MoM scene bodies require the {actual!r} formulation, not {formulation!r}')
