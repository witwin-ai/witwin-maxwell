"""Faceted primitive surfaces for the MoM runtime.

Builders produce closed, consistently outward-oriented, conforming
triangulations of convex primitives with sharp edges and corners: the
``witwin.core`` ``Box``, ``Cylinder``, ``Cone`` and ``Prism`` geometries, and
utility rounded boxes, convex polygonal prisms and thin closed slabs.

Meshing is structured. Every edge family carries a one-dimensional node
distribution; flat faces are tensor products of those distributions and
planar caps are homothetic rings around a centre joined by a zipper
triangulation. Shared boundaries use one global vertex identity, so meshes are
conforming by construction and pass ``SurfaceMesh`` validation.

Element size is uniform by default. Optional ``EdgeGrading`` uses the linear
size law ``h(d) = min(h, h_min + (q - 1) d)`` in the distance ``d`` to the
nearest sharp edge, corner or apex; its nodes form a geometric sequence of
element sizes ``h_min q^j`` (scaled down to an integer count) up to ``h``. Grading is perpendicular to edges, so elements
along an edge keep the uniform size and become anisotropic near it; corners
and the cone apex are refined in every direction.

Only 1D node distributions and integer incidence are built on the host.
Coordinates are evaluated with torch on the target device and keep autograd
ancestry of primitive dimensions. Coordinates are metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from witwin.core import Box, Cone, Cylinder, Prism

from ..constants import EPSILON_0, MU_0
from .mesh import SurfaceMesh, _cuda


FACETED_PRIMITIVES = (Box, Cylinder, Cone, Prism)
_FLOAT_DTYPES = (torch.float32, torch.float64)


def _positive_real(value, name, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite real number')
    return float(value)


@dataclass(frozen=True)
class EdgeGrading:
    """Geometric refinement toward sharp edges, corners and apexes.

    ``min_size`` is the element size in metres at an edge; ``ratio`` is the
    growth factor between neighbouring elements away from it.
    """
    min_size: float
    ratio: float = 1.5

    def __post_init__(self):
        _positive_real(self.min_size, 'EdgeGrading.min_size')
        if _positive_real(self.ratio, 'EdgeGrading.ratio') <= 1:
            raise ValueError('EdgeGrading.ratio must exceed one')

    def payload(self):
        return {'min_size': float(self.min_size), 'ratio': float(self.ratio)}


@dataclass(frozen=True)
class SurfaceMeshConfig:
    """Mesh-size control for faceted MoM primitives.

    The target edge length is ``lambda_min / elements_per_wavelength`` with
    ``lambda_min = 2 pi / max |k|`` over the exterior and interior media of a
    body, capped by ``max_edge_length`` when given. With
    ``elements_per_wavelength=None`` only the explicit length applies.
    ``max_arc_angle`` bounds the angular step on curved boundaries.
    ``max_triangles`` fails closed before an unexpectedly large mesh is built.
    """
    elements_per_wavelength: float | None = 10.0
    max_edge_length: float | None = None
    max_arc_angle: float = math.pi/12
    min_edge_divisions: int = 1
    edge_grading: EdgeGrading | None = None
    max_triangles: int = 200_000

    def __post_init__(self):
        _positive_real(self.elements_per_wavelength, 'elements_per_wavelength', allow_none=True)
        _positive_real(self.max_edge_length, 'max_edge_length', allow_none=True)
        if self.elements_per_wavelength is None and self.max_edge_length is None:
            raise ValueError('SurfaceMeshConfig requires elements_per_wavelength or max_edge_length')
        if _positive_real(self.max_arc_angle, 'max_arc_angle') > math.pi/2:
            raise ValueError('max_arc_angle must not exceed pi/2')
        for name in ('min_edge_divisions', 'max_triangles'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.edge_grading is not None and type(self.edge_grading) is not EdgeGrading:
            raise TypeError('edge_grading must be EdgeGrading or None')

    def edge_length(self, media=None):
        """Target edge length in metres for one body and its adjacent media."""
        candidates = [] if self.max_edge_length is None else [float(self.max_edge_length)]
        if self.elements_per_wavelength is not None:
            if not media:
                raise ValueError('Wavelength-based MoM meshing requires the body media; '
                                 'use elements_per_wavelength=None with max_edge_length on this path')
            wavenumber = max(medium_wavenumber(medium) for medium in media)
            candidates.append(2*math.pi/(wavenumber*self.elements_per_wavelength))
        return min(candidates)

    def payload(self):
        return {'elements_per_wavelength': self.elements_per_wavelength, 'max_edge_length': self.max_edge_length,
                'max_arc_angle': float(self.max_arc_angle), 'min_edge_divisions': self.min_edge_divisions,
                'edge_grading': None if self.edge_grading is None else self.edge_grading.payload(),
                'max_triangles': self.max_triangles}


def medium_wavenumber(medium):
    """Largest wavenumber magnitude [1/m] of a scalar or tensor sampled medium."""
    if hasattr(medium, 'k'):
        value = abs(complex(medium.k))
    else:
        # Spectral norms bound every eigen-wavenumber of a homogeneous tensor medium.
        omega = 2*math.pi*float(medium.frequency)
        scale = float(torch.linalg.matrix_norm(medium.epsilon_r, ord=2)*torch.linalg.matrix_norm(medium.mu_r, ord=2))
        value = omega*math.sqrt(EPSILON_0*MU_0*scale)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('MoM meshing requires a positive finite medium wavenumber')
    return value


# One-dimensional size law -----------------------------------------------------

def _size_integral(distance, h, grading):
    """Element-count function N(d) of the graded size law.

    For ``d`` below the knee ``(h - h_min)/(q - 1)``,
    ``N(d) = log(1 + (q - 1) d / h_min) / log(q)`` is integer exactly at the
    nodes of the geometric sequence ``h_min q^j``; beyond the knee the count
    grows uniformly as ``d / h``.
    """
    q1, h_min = grading.ratio-1, grading.min_size
    knee = (h-h_min)/q1
    if distance <= knee:
        return math.log1p(q1*distance/h_min)/math.log(grading.ratio)
    return math.log(h/h_min)/math.log(grading.ratio) + (distance-knee)/h


def _size_integral_inverse(value, h, grading):
    q1, h_min = grading.ratio-1, grading.min_size
    knee_value = math.log(h/h_min)/math.log(grading.ratio)
    if value <= knee_value:
        return h_min*math.expm1(value*math.log(grading.ratio))/q1
    return (h-h_min)/q1 + (value-knee_value)*h


def graded_size(distance, h, grading):
    """Local target size at a distance [m] from the nearest graded feature."""
    if grading is None:
        return h
    return min(h, grading.min_size+(grading.ratio-1)*distance)


def _divisions(value, minimum):
    # The relative slack absorbs float32-authored dimensions such as 0.3.
    return max(minimum, math.ceil(value*(1-1e-6)))


def graded_nodes(length, h, grading=None, *, start=True, end=True, min_divisions=1):
    """Nodes on ``[0, length]`` graded toward the selected ends.

    The element count is the smallest integer not below the count function,
    and nodes are equally spaced in it. Elements therefore start at most at
    ``h_min``, grow by a factor of at most ``q`` (up to about 12% more for
    ``q <= 2`` at the single element that joins the uniform region) and never
    exceed ``h``; elsewhere they stay within the local size ``h(d)`` at their
    edge-side end, up to the same transition margin. Two-sided distributions are
    exactly mirror symmetric.
    """
    length, h = float(length), float(h)
    if length <= 0:
        raise ValueError('Graded segment length must be positive')
    graded = grading is not None and grading.min_size < h and (start or end)
    if not graded:
        count = _divisions(length/h, min_divisions)
        return [length*j/count for j in range(count)] + [length]
    if start and end:
        total = 2*_size_integral(length/2, h, grading)
    else:
        total = _size_integral(length, h, grading)
    count = _divisions(total, min_divisions)
    nodes = []
    for j in range(count+1):
        value = j*total/count
        if start and end:
            x = (_size_integral_inverse(value, h, grading) if 2*j <= count
                 else length-_size_integral_inverse(total-value, h, grading))
        elif start:
            x = _size_integral_inverse(value, h, grading)
        else:
            x = length-_size_integral_inverse(total-value, h, grading)
        nodes.append(min(max(x, 0.), length))
    nodes[0], nodes[-1] = 0., length
    return nodes


# Topology helpers --------------------------------------------------------------

class _Registry:
    """One global vertex identity per structural key."""

    def __init__(self):
        self.index, self.records = {}, []

    def __call__(self, key, record):
        if key not in self.index:
            self.index[key] = len(self.records)
            self.records.append(record)
        return self.index[key]


def _thin_run(run, weights):
    """Subset of a node run with fixed ends and target-normalized segments at most one."""
    cumulative = [0.]
    for weight in weights:
        cumulative.append(cumulative[-1]+weight)
    total = cumulative[-1]
    count = max(1, math.ceil(total-1e-9))
    # Thin only by at least two boundary steps per segment on average; a mix
    # of single and double steps would double some segments near the boundary.
    if 2*count > len(run)-1:
        return run[:-1]
    kept, previous = [run[0]], 0
    for j in range(1, count):
        goal = j*total/count
        index = min(range(previous+1, len(run)-1-(count-1-j)), key=lambda i: abs(cumulative[i]-goal))
        kept.append(run[index])
        previous = index
    return kept


def _nested_rings(points, fractions, target, minimum=6):
    """Boundary positions of the rings between a centre and its closed boundary.

    Ring ``k`` lies at ``fractions[k] * point`` around the centre and uses a
    subset of the boundary nodes chosen at equal spacing in the
    target-normalized ring arc length, so no ring segment exceeds the local
    tangential target ``target(k, b)`` and densities change gradually from
    ring to ring. Returns rings for ``fractions[1:]``; the last ring is the
    full boundary.
    """
    total = len(points)
    run = list(range(total))+[0]
    rings = []
    for k in range(1, len(fractions)-1):
        weights = [2*fractions[k]*math.dist(points[p], points[q])/(target(k, p)+target(k, q))
                   for p, q in zip(run, run[1:])]
        kept = _thin_run(run, weights)
        if len(kept) < minimum:
            kept = sorted({math.floor(j*total/minimum+.5) % total for j in range(minimum)})
        rings.append(kept)
    return rings+[list(range(total))]


def _zip(inner, inner_positions, outer, outer_positions, total):
    """Triangulate the band between two nested rings sharing boundary positions."""
    faces, i, j = [], 0, 0
    ni, no = len(inner), len(outer)
    while i < ni or j < no:
        next_inner = inner_positions[i+1] if i+1 < ni else total
        next_outer = outer_positions[j+1] if j+1 < no else total
        # Shortest new diagonal in the shared boundary parameter; avoids
        # slivers that curvature could invert on revolved surfaces.
        advance_outer = next_outer-(inner_positions[i] if i < ni else total)
        advance_inner = next_inner-(outer_positions[j] if j < no else total)
        if j < no and (i >= ni or advance_outer <= advance_inner):
            faces.append((inner[i % ni], outer[j], outer[(j+1) % no]))
            j += 1
        else:
            faces.append((inner[i], outer[j % no], inner[(i+1) % ni]))
            i += 1
    return faces


def _fan(center, ring):
    return [(center, ring[j], ring[(j+1) % len(ring)]) for j in range(len(ring))]


def _quad(a, b, c, d, rising):
    """Split quad a-b-c-d (cyclic) along a-c when ``rising`` else along b-d."""
    return [(a, b, c), (a, c, d)] if rising else [(a, b, d), (b, c, d)]


def _cap(registry, name, boundary, fractions, positions, level):
    """Planar homothetic-ring cap; ``boundary`` maps boundary index -> vertex id."""
    center = registry((name, 'centre'), (0., 0, level))
    rings = [[registry((name, k, b), (fractions[k], b, level)) for b in ring]
             for k, ring in enumerate(positions[:-1], 1)]
    rings.append([boundary[b] for b in positions[-1]])
    faces = _fan(center, rings[0])
    for k in range(len(rings)-1):
        faces += _zip(rings[k], positions[k], rings[k+1], positions[k+1], len(boundary))
    return faces


def _check_count(faces, config_limit):
    if len(faces) > config_limit:
        raise ValueError(f'Faceted MoM mesh would contain {len(faces)} triangles, above max_triangles={config_limit}')


def _orient_outward(vertices, faces, interior):
    """Orient every triangle of a convex closed surface away from an interior point."""
    triangles = vertices[faces]
    normal = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    side = (normal*(triangles.mean(1)-interior)).sum(-1)
    if not bool((side != 0).all()):
        raise ValueError('Faceted mesh contains a triangle through its interior reference point')
    return torch.where((side < 0)[:, None], faces[:, [0, 2, 1]], faces)


def _faces_tensor(faces, device):
    return torch.tensor(faces, dtype=torch.long, device=device)


# Tensor-product boxes ---------------------------------------------------------

def _box_faces(counts):
    """Boundary vertices (grid indices) and triangles of a tensor-product box."""
    registry, faces = _Registry(), []
    for axis in range(3):
        u, v = (axis+1) % 3, (axis+2) % 3
        for side in (0, counts[axis]):
            def vertex(iu, iv):
                index = [0, 0, 0]
                index[axis], index[u], index[v] = side, iu, iv
                return registry(tuple(index), tuple(index))
            for i in range(counts[u]):
                for j in range(counts[v]):
                    # Diagonals point away from the face centre; with even
                    # division counts the mesh has every mirror of the face.
                    rising = (2*i+1-counts[u])*(2*j+1-counts[v]) >= 0
                    faces += _quad(vertex(i, j), vertex(i+1, j), vertex(i+1, j+1), vertex(i, j+1), rising)
    return registry.records, faces


def _box_arrays(axis_nodes, scale, *, max_triangles, device, dtype):
    """Box surface from per-axis node fractions in [-1, 1] times half sizes."""
    counts = [len(nodes)-1 for nodes in axis_nodes]
    records, faces = _box_faces(counts)
    _check_count(faces, max_triangles)
    index = torch.tensor(records, dtype=torch.long, device=device)
    coordinates = [torch.tensor(nodes, dtype=dtype, device=device) for nodes in axis_nodes]
    vertices = torch.stack([coordinates[a][index[:, a]] for a in range(3)], -1)*scale
    faces = _orient_outward(vertices, _faces_tensor(faces, device), vertices.new_zeros(3))
    return vertices, faces


def _symmetric_fractions(half, h, grading, min_divisions):
    nodes = graded_nodes(2*half, h, grading, min_divisions=min_divisions)
    fractions = [x/half-1 for x in nodes]
    count = len(fractions)-1
    for j in range(count//2+1):
        fractions[count-j] = -fractions[j]
    if count % 2 == 0:
        fractions[count//2] = 0.
    return fractions


def box_arrays(size, edge_length, *, grading=None, min_edge_divisions=1, max_triangles=200_000,
               device='cuda', dtype=torch.float64):
    """Vertices and faces of an origin-centred box on any device."""
    size = _size_tensor(size, 3, 'Box size', device, dtype)
    half = [float(value)/2 for value in size.detach().cpu()]
    nodes = [_symmetric_fractions(a, edge_length, grading, min_edge_divisions) for a in half]
    return _box_arrays(nodes, size/2, max_triangles=max_triangles, device=device, dtype=dtype)


def rounded_box_arrays(size, radius, edge_length, *, grading=None, max_arc_angle=math.pi/12, min_edge_divisions=1,
                       max_triangles=200_000, device='cuda', dtype=torch.float64):
    """Box with edges and corners rounded by ``radius`` (Minkowski sum with a ball).

    ``radius=0`` yields the sharp box exactly. Each quarter-circle edge arc is
    split at 45 degrees between the two adjacent face strips and sampled at
    uniform angles. With grading, flat regions are graded toward the arcs and
    arc steps follow the grading minimum size.
    """
    size = _size_tensor(size, 3, 'Rounded box size', device, dtype)
    half = [float(value)/2 for value in size.detach().cpu()]
    radius = float(radius)
    if isinstance(radius, bool) or not math.isfinite(radius) or radius < 0 or radius > min(half):
        raise ValueError('Rounded box radius must lie in [0, min(size)/2]')
    if radius == 0:
        return box_arrays(size, edge_length, grading=grading, min_edge_divisions=min_edge_divisions,
                          max_triangles=max_triangles, device=device, dtype=dtype)
    arc_size = edge_length if grading is None else min(edge_length, grading.min_size)
    steps = max(_divisions(math.pi/(4*max_arc_angle), 1), _divisions(math.pi*radius/(4*arc_size), 1))
    axis_nodes = []
    for a in half:
        inner = a-radius
        arc = [radius*math.tan(j*math.pi/(4*steps)) for j in range(1, steps)]
        flat = [0.] if inner <= 0 else [x-inner for x in graded_nodes(2*inner, edge_length, grading,
                                                                        min_divisions=min_edge_divisions)]
        nodes = [-(inner+radius)] + [-(inner+t) for t in reversed(arc)] + flat + [inner+t for t in arc] + [inner+radius]
        axis_nodes.append([x/a for x in nodes])
    vertices, faces = _box_arrays(axis_nodes, size/2, max_triangles=max_triangles, device=device, dtype=dtype)
    inner = size/2-radius
    centre = torch.maximum(torch.minimum(vertices, inner), -inner)
    offset = vertices-centre
    vertices = centre+radius*offset/torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
    return vertices, faces


def rounded_box_area(size, radius):
    """Exact area of the rounded box surface (Steiner formula)."""
    a, b, c = (float(s)-2*radius for s in size)
    return 2*(a*b+b*c+c*a)+2*math.pi*radius*(a+b+c)+4*math.pi*radius**2


def rounded_box_volume(size, radius):
    """Exact enclosed volume of the rounded box (Steiner formula)."""
    a, b, c = (float(s)-2*radius for s in size)
    return a*b*c+2*radius*(a*b+b*c+c*a)+math.pi*radius**2*(a+b+c)+4*math.pi*radius**3/3


# Extruded and revolved bodies -------------------------------------------------

def _size_tensor(value, length, name, device, dtype):
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float64)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.shape != ((length,) if length else ()) or not bool(torch.isfinite(tensor).all() & (tensor > 0).all()):
        raise ValueError(f'{name} must be positive and finite in metres')
    return tensor


def _evaluate(records, boundary, centre, axial):
    """Vertices from host records (ring fraction, boundary index, axial fraction)."""
    device, dtype = boundary.device, boundary.dtype
    fraction = torch.tensor([r[0] for r in records], dtype=dtype, device=device)
    index = torch.tensor([r[1] for r in records], dtype=torch.long, device=device)
    level = torch.tensor([r[2] for r in records], dtype=dtype, device=device)
    planar = centre+fraction[:, None]*(boundary[index]-centre)
    return torch.cat((planar, (level*axial)[:, None]), -1)


def _circle_count(radius, h, max_arc_angle):
    return max(6, _divisions(2*math.pi*radius/h, 3), _divisions(2*math.pi/max_arc_angle, 3))


def _circle_points(radius, total):
    return [(radius*math.cos(2*math.pi*b/total), radius*math.sin(2*math.pi*b/total)) for b in range(total)]


def _cap_rings(points, h, grading, min_edge_divisions):
    """Ring fractions and positions of an origin-centred planar disk cap.

    Radial nodes are graded toward the rim edge; the tangential target is
    the uniform size, since the rim is graded only perpendicular to itself.
    """
    reach = min(math.hypot(*point) for point in points)
    fractions = [x/reach for x in graded_nodes(reach, h, grading, start=False, end=True,
                                               min_divisions=min_edge_divisions)]
    return fractions, _nested_rings(points, fractions, lambda k, b: h, minimum=6)


def _side_faces(side, closed_count):
    """Tensor-product side band between consecutive axial levels of a loop."""
    faces, count = [], len(side)-1
    for k in range(count):
        for b in range(closed_count):
            n = (b+1) % closed_count
            faces += _quad(side[k][b], side[k][n], side[k+1][n], side[k+1][b], 2*k+1 < count)
    return faces


def cylinder_arrays(radius, height, edge_length, *, grading=None, max_arc_angle=math.pi/12, min_edge_divisions=1,
                    max_triangles=200_000, device='cuda', dtype=torch.float64):
    """Origin-centred circular cylinder along local z."""
    radius = _size_tensor(radius, 0, 'Cylinder radius', device, dtype)
    height = _size_tensor(height, 0, 'Cylinder height', device, dtype)
    r, length = float(radius), float(height)
    total = _circle_count(r, edge_length, max_arc_angle)
    points = _circle_points(r, total)
    levels = [x/length-.5 for x in graded_nodes(length, edge_length, grading, min_divisions=min_edge_divisions)]
    registry = _Registry()
    side = [[registry(('side', k, b), (1., b, level)) for b in range(total)] for k, level in enumerate(levels)]
    faces = _side_faces(side, total)
    fractions, positions = _cap_rings(points, edge_length, grading, min_edge_divisions)
    for name, k in (('bottom', 0), ('top', len(levels)-1)):
        faces += _cap(registry, name, side[k], fractions, positions, levels[k])
    _check_count(faces, max_triangles)
    angle = torch.arange(total, device=device, dtype=dtype)*(2*math.pi/total)
    boundary = torch.stack((torch.cos(angle), torch.sin(angle)), -1)*radius
    vertices = _evaluate(registry.records, boundary, boundary.new_zeros(2), height)
    return vertices, _orient_outward(vertices, _faces_tensor(faces, device), vertices.new_zeros(3))


def _polygon(corners):
    corners = [tuple(float(c) for c in corner) for corner in corners]
    if len(corners) < 3 or any(len(c) != 2 or not all(map(math.isfinite, c)) for c in corners):
        raise ValueError('Prism cross-section requires at least three finite 2D vertices')
    area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(corners, corners[1:]+corners[:1]))/2
    if area < 0:
        corners = corners[::-1]
    count = len(corners)
    scale = max(math.dist(a, b) for a in corners for b in corners)
    for j in range(count):
        a, b, c = corners[j-1], corners[j], corners[(j+1) % count]
        turn = (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
        if turn <= 1e-12*scale**2:
            raise ValueError('Prism cross-section must be a strictly convex polygon without repeated vertices')
    return corners


def _prism_arrays(corners_tensor, corners, height, h, grading, min_edge_divisions, max_triangles):
    """Convex prism; each cap is split into corner quads meshed as tensor grids.

    Quad ``k`` joins corner ``P_k``, the midpoints of its two edges and the
    vertex centroid. One normalized half-edge distribution, graded toward the
    corner and sized by the longest half-edge, is shared by every quad, so
    spokes and edge midpoints conform. Vertices are corner-weight
    combinations evaluated on the device.
    """
    device, dtype = corners_tensor.device, corners_tensor.dtype
    count = len(corners)
    half = max(math.dist(a, b) for a, b in zip(corners, corners[1:]+corners[:1]))/2
    steps = [x/half for x in graded_nodes(half, h, grading, start=True, end=False,
                                            min_divisions=max(1, math.ceil(min_edge_divisions/2)))]
    n = len(steps)-1
    centre = [1/count]*count

    def mix(*terms):
        weights = [0.]*count
        for scale, vector in terms:
            for c in range(count):
                weights[c] += scale*vector[c]
        return tuple(weights)

    unit = [tuple(1. if c == k else 0. for c in range(count)) for k in range(count)]
    midpoint = [mix((.5, unit[k]), (.5, unit[(k+1) % count])) for k in range(count)]
    # Boundary loop: edge k from P_k to P_{k+1}, graded symmetrically at both corners.
    boundary = []
    for k in range(count):
        ts = [w/2 for w in steps]+[1-w/2 for w in reversed(steps[:-1])]
        boundary += [mix((1-t, unit[k]), (t, unit[(k+1) % count])) for t in ts[:-1]]
    total = len(boundary)
    levels = [x/float(height)-.5 for x in graded_nodes(float(height), h, grading, min_divisions=min_edge_divisions)]
    registry = _Registry()
    side = [[registry(('side', k, b), (boundary[b], level)) for b in range(total)] for k, level in enumerate(levels)]
    faces = _side_faces(side, total)
    for name, level in (('bottom', 0), ('top', len(levels)-1)):
        z = levels[level]

        def vertex(k, i, j):
            if j == 0:
                return side[level][2*n*k+i]
            if i == 0:
                return side[level][(2*n*k-j) % total]
            if i == n or j == n:
                spoke, fraction = (k, steps[j]) if i == n else ((k-1) % count, steps[i])
                if fraction == 1:
                    return registry((name, 'centre'), (tuple(centre), z))
                start = midpoint[spoke]
                return registry((name, 'spoke', spoke, fraction),
                                (mix((1-fraction, start), (fraction, centre)), z))
            u, v = steps[i], steps[j]
            weights = mix(((1-u)*(1-v), unit[k]), (u*(1-v), midpoint[k]), (u*v, centre),
                          ((1-u)*v, midpoint[(k-1) % count]))
            return registry((name, 'quad', k, i, j), (weights, z))
        for k in range(count):
            for i in range(n):
                for j in range(n):
                    faces += _quad(vertex(k, i, j), vertex(k, i+1, j), vertex(k, i+1, j+1), vertex(k, i, j+1), True)
    _check_count(faces, max_triangles)
    weights = torch.tensor([r[0] for r in registry.records], dtype=dtype, device=device)
    level = torch.tensor([r[1] for r in registry.records], dtype=dtype, device=device)
    vertices = torch.cat((weights@corners_tensor, (level*height)[:, None]), -1)
    interior = torch.cat((corners_tensor.mean(0), corners_tensor.new_zeros(1)))
    return vertices, _orient_outward(vertices, _faces_tensor(faces, device), interior)


def prism_arrays(polygon, height, edge_length, *, grading=None, min_edge_divisions=1, max_triangles=200_000,
                 device='cuda', dtype=torch.float64):
    """Straight prism over a strictly convex polygon in local xy, centred in z."""
    corners = _polygon(polygon)
    height = _size_tensor(height, 0, 'Prism height', device, dtype)
    corners_tensor = torch.tensor(corners, dtype=dtype, device=device)
    return _prism_arrays(corners_tensor, corners, height, edge_length, grading, min_edge_divisions, max_triangles)


def cone_arrays(radius, height, edge_length, *, grading=None, max_arc_angle=math.pi/12, min_edge_divisions=1,
                max_triangles=200_000, device='cuda', dtype=torch.float64):
    """Right circular cone with apex at the origin and base disk at local z = height.

    Lateral rings are graded toward the apex and the rim; near the apex, a
    point singularity, the tangential size follows the apex grading too.
    """
    radius = _size_tensor(radius, 0, 'Cone radius', device, dtype)
    height = _size_tensor(height, 0, 'Cone height', device, dtype)
    r, length = float(radius), float(height)
    slant = math.hypot(r, length)
    total = _circle_count(r, edge_length, max_arc_angle)
    points = _circle_points(r, total)
    registry, faces = _Registry(), []
    rim = [registry(('rim', b), (1., b, 1.)) for b in range(total)]
    apex = registry(('apex',), (0., 0, 0.))
    lateral = [x/slant for x in graded_nodes(slant, edge_length, grading, min_divisions=min_edge_divisions)]
    # Ring chords scale with the slant fraction, exactly as in a planar cap.
    positions = _nested_rings(points, lateral, lambda k, b: graded_size(lateral[k]*slant, edge_length, grading),
                              minimum=6)
    rings = [[registry(('lateral', k, b), (lateral[k], b, lateral[k])) for b in ring]
             for k, ring in enumerate(positions[:-1], 1)]
    rings.append(rim)
    faces += _fan(apex, rings[0])
    for k in range(len(rings)-1):
        faces += _zip(rings[k], positions[k], rings[k+1], positions[k+1], total)
    fractions, cap_positions = _cap_rings(points, edge_length, grading, min_edge_divisions)
    faces += _cap(registry, 'base', rim, fractions, cap_positions, 1.)
    _check_count(faces, max_triangles)
    angle = torch.arange(total, device=device, dtype=dtype)*(2*math.pi/total)
    boundary = torch.stack((torch.cos(angle), torch.sin(angle)), -1)*radius
    vertices = _evaluate(registry.records, boundary, boundary.new_zeros(2), height)
    interior = torch.stack((height.new_zeros(()), height.new_zeros(()), .75*height))
    return vertices, _orient_outward(vertices, _faces_tensor(faces, device), interior)


# Public CUDA builders ---------------------------------------------------------

def _surface(arrays, center, device, dtype):
    device = _cuda(device)
    if dtype not in _FLOAT_DTYPES:
        raise ValueError('Mesh coordinates require float32 or float64')
    vertices, faces = arrays(device, dtype)
    offset = torch.as_tensor(center, dtype=dtype, device=device)
    if offset.shape != (3,) or not bool(torch.isfinite(offset).all()):
        raise ValueError('center must contain three finite coordinates')
    return SurfaceMesh(vertices+offset, faces)


def _edge(edge_length):
    return _positive_real(edge_length, 'edge_length')


def box_mesh(size, *, edge_length, grading=None, min_edge_divisions=1, center=(0., 0., 0.),
             device='cuda', dtype=torch.float64):
    """Axis-aligned sharp box with exact face planes, edges and corners."""
    h = _edge(edge_length)
    return _surface(lambda d, t: box_arrays(size, h, grading=grading, min_edge_divisions=min_edge_divisions,
                                            device=d, dtype=t), center, device, dtype)


def rounded_box_mesh(size, radius, *, edge_length, grading=None, max_arc_angle=math.pi/12, min_edge_divisions=1,
                     center=(0., 0., 0.), device='cuda', dtype=torch.float64):
    """Rounded-edge box; ``radius=0`` gives the sharp box mesh exactly."""
    h = _edge(edge_length)
    return _surface(lambda d, t: rounded_box_arrays(size, radius, h, grading=grading, max_arc_angle=max_arc_angle,
                                                    min_edge_divisions=min_edge_divisions, device=d, dtype=t),
                    center, device, dtype)


def slab_mesh(length, width, thickness, *, edge_length, grading=None, min_edge_divisions=1, center=(0., 0., 0.),
              device='cuda', dtype=torch.float64):
    """Thin closed rectangular slab in the local xy-plane with thickness along z."""
    if float(thickness) > min(float(length), float(width)):
        raise ValueError('Slab thickness must not exceed its in-plane dimensions')
    return box_mesh((length, width, thickness), edge_length=edge_length, grading=grading,
                    min_edge_divisions=min_edge_divisions, center=center, device=device, dtype=dtype)


def prism_mesh(polygon, height, *, edge_length, grading=None, min_edge_divisions=1, center=(0., 0., 0.),
               device='cuda', dtype=torch.float64):
    """Straight convex polygonal prism, e.g. a triangular prism for three vertices."""
    h = _edge(edge_length)
    return _surface(lambda d, t: prism_arrays(polygon, height, h, grading=grading,
                                              min_edge_divisions=min_edge_divisions, device=d, dtype=t),
                    center, device, dtype)


def triangular_prism_mesh(side, height, *, edge_length, grading=None, min_edge_divisions=1, center=(0., 0., 0.),
                          device='cuda', dtype=torch.float64):
    """Equilateral triangular prism with its cross-section centroid on the axis."""
    side = _positive_real(side, 'side')
    radius = side/math.sqrt(3)
    polygon = [(radius*math.cos(a), radius*math.sin(a)) for a in (math.pi/2, math.pi*7/6, math.pi*11/6)]
    return prism_mesh(polygon, height, edge_length=edge_length, grading=grading,
                      min_edge_divisions=min_edge_divisions, center=center, device=device, dtype=dtype)


def cylinder_mesh(radius, height, *, edge_length, grading=None, max_arc_angle=math.pi/12, min_edge_divisions=1,
                  center=(0., 0., 0.), device='cuda', dtype=torch.float64):
    h = _edge(edge_length)
    return _surface(lambda d, t: cylinder_arrays(radius, height, h, grading=grading, max_arc_angle=max_arc_angle,
                                                 min_edge_divisions=min_edge_divisions, device=d, dtype=t),
                    center, device, dtype)


def cone_mesh(radius, height, *, edge_length, grading=None, max_arc_angle=math.pi/12, min_edge_divisions=1,
              center=(0., 0., 0.), device='cuda', dtype=torch.float64):
    """Cone with apex at ``center`` and base disk at ``center + height * z``."""
    h = _edge(edge_length)
    return _surface(lambda d, t: cone_arrays(radius, height, h, grading=grading, max_arc_angle=max_arc_angle,
                                             min_edge_divisions=min_edge_divisions, device=d, dtype=t),
                    center, device, dtype)


# Core geometry dispatch -------------------------------------------------------

def primitive_arrays(geometry, config, media=None, *, device, dtype):
    """World-frame vertices and outward faces of a core faceted primitive.

    Local meshes follow the core conventions: Box, Cylinder and Prism are
    centred on ``position``; the Cone apex is at ``position`` with its base at
    ``+height`` along ``axis``. Axis mapping follows the core signed distance
    (the volume-solver definition), then the core rotation/position transform.
    """
    if type(geometry) not in FACETED_PRIMITIVES:
        raise ValueError(f'{type(geometry).__name__} is not a faceted MoM primitive')
    if type(config) is not SurfaceMeshConfig:
        raise TypeError('Faceted MoM primitives require a SurfaceMeshConfig')
    h = config.edge_length(media)
    options = dict(grading=config.edge_grading, min_edge_divisions=config.min_edge_divisions,
                   max_triangles=config.max_triangles, device=device, dtype=dtype)
    if type(geometry) is Box:
        vertices, faces = box_arrays(geometry.size, h, **options)
    elif type(geometry) is Cylinder:
        vertices, faces = cylinder_arrays(geometry.radius, geometry.height, h, max_arc_angle=config.max_arc_angle, **options)
    elif type(geometry) is Cone:
        vertices, faces = cone_arrays(geometry.radius, geometry.height, h, max_arc_angle=config.max_arc_angle, **options)
    else:
        radius = _size_tensor(geometry.radius, 0, 'Prism radius', device, dtype)
        angles = [2*math.pi*j/geometry.num_sides for j in range(geometry.num_sides)]
        unit = torch.tensor([(math.cos(a), math.sin(a)) for a in angles], dtype=dtype, device=device)
        corners = [(float(radius)*math.cos(a), float(radius)*math.sin(a)) for a in angles]
        height = _size_tensor(geometry.height, 0, 'Prism height', device, dtype)
        vertices, faces = _prism_arrays(unit*radius, corners, height, h, config.edge_grading,
                                        config.min_edge_divisions, config.max_triangles)
    axis = getattr(geometry, 'axis', 'z')
    # Map local (u, v, axial) exactly as the core signed distance splits
    # coordinates: axis x -> (axial, u, v), axis y -> (u, axial, v).
    if axis == 'x':
        vertices = vertices[:, [2, 0, 1]]
    elif axis == 'y':
        # An odd coordinate permutation reverses orientation.
        vertices, faces = vertices[:, [0, 2, 1]], faces[:, [0, 2, 1]]
    return geometry._transform_mesh_verts(vertices), faces


def primitive_surface(geometry, config, media=None, *, device, dtype=torch.float64):
    """Validated CUDA ``SurfaceMesh`` for a core Box, Cylinder, Cone or Prism."""
    device = _cuda(device)
    vertices, faces = primitive_arrays(geometry, config, media, device=device, dtype=dtype)
    return SurfaceMesh(vertices, faces)


def mesh_quality(vertices, faces):
    """Minimum angle [deg], aspect ratios (1 = equilateral) and edge-length range."""
    triangles = vertices[faces]
    edges = triangles.roll(-1, 1)-triangles
    lengths = torch.linalg.vector_norm(edges, dim=-1)
    area = torch.linalg.vector_norm(torch.linalg.cross(edges[:, 0], -edges[:, 2]), dim=-1)/2
    aspect = lengths.amax(-1)*lengths.sum(-1)/(4*math.sqrt(3)*area)
    cosine = (-(edges*edges.roll(1, 1)).sum(-1)/(lengths*lengths.roll(1, 1))).clamp(-1, 1)
    angle = torch.rad2deg(torch.arccos(cosine))
    return {'triangles': len(faces), 'min_angle_deg': float(angle.min()), 'max_aspect_ratio': float(aspect.max()),
            'mean_aspect_ratio': float(aspect.mean()), 'min_edge_length': float(lengths.min()),
            'max_edge_length': float(lengths.max())}
