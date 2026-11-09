"""Tree-pruned mesh validation equals the original all-pairs decisions.

The reference functions below are verbatim copies of the all-pairs scans
that the pruned paths replaced (commit c90c8346), kept only as test oracles.
"""
import math

import pytest
import torch

from witwin.maxwell.mom.mesh import (_has_coincident_vertices, _has_intersections, _minimum_triangle_distance,
                                     sphere_mesh, triangle_pair_distances)
from witwin.maxwell.mom.spatial import BoxTree, overlapping_pairs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
DEVICE = 'cuda'


def _reference_has_intersections(triangles):
    """Find improper triangle intersections using CUDA batched predicates."""
    count = len(triangles)
    scale = torch.linalg.vector_norm(triangles.amax((0, 1)) - triangles.amin((0, 1)))
    tol = 128 * torch.finfo(triangles.dtype).eps
    # Normalize geometry to make predicate tolerances independent of SI scale.
    tri = (triangles - triangles[0, 0]) / scale
    lows, highs = tri.amin(1), tri.amax(1)
    for start in range(0, count, 128):
        candidate = (lows[start:start + 128, None] <= highs[None] + tol).all(-1)
        candidate &= (highs[start:start + 128, None] >= lows[None] - tol).all(-1)
        left, right = candidate.nonzero(as_tuple=True)
        left = left + start
        keep = left < right
        left, right = left[keep], right[keep]
        if not len(left):
            continue
        a, b = tri[left], tri[right]
        for source, target in ((a, b), (b, a)):
            edge1, edge2 = target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]
            for k in range(3):
                origin = source[:, k]
                direction = source[:, (k + 1) % 3] - origin
                h = torch.linalg.cross(direction, edge2)
                determinant = (edge1 * h).sum(-1)
                safe = torch.where(determinant.abs() > tol, determinant, torch.ones_like(determinant))
                s = origin - target[:, 0]
                u = (s * h).sum(-1) / safe
                q = torch.linalg.cross(s, edge1)
                v = (direction * q).sum(-1) / safe
                t = (edge2 * q).sum(-1) / safe
                end = source[:, (k + 1) % 3]
                # A nonparallel segment has only one intersection with the
                # target plane. An exactly shared endpoint fixes that location
                # without the cancellation in the scalar triple product for t.
                # Only this segment is classified; other edges and coplanar
                # overlap predicates still detect improper shared-face contacts.
                exact_start = (target == origin[:, None]).all(-1).any(-1)
                exact_end = (target == end[:, None]).all(-1).any(-1)
                t = torch.where(exact_start, torch.zeros_like(t), t)
                t = torch.where(exact_end, torch.ones_like(t), t)
                hit = (determinant.abs() > tol) & (u >= -tol) & (v >= -tol) & (u + v <= 1 + tol)
                # A nonshared endpoint touching a face or edge is also an
                # invalid contact. Shared conforming mesh vertices are allowed.
                interior_hit = (t > tol) & (t < 1 - tol)
                at_start = (t >= -tol) & (t <= tol)
                at_end = (t >= 1-tol) & (t <= 1+tol)
                shared_start = ((target-origin[:, None]).abs().amax(-1) <= tol).any(-1)
                shared_end = ((target-end[:, None]).abs().amax(-1) <= tol).any(-1)
                hit &= interior_hit | (at_start & ~shared_start) | (at_end & ~shared_end)
                if bool(hit.any()):
                    return True
        # Coplanar intersections: strict edge crossings or interior vertices.
        normal = torch.linalg.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
        normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        coplanar = (((b - a[:, :1]) * normal[:, None]).sum(-1).abs() < tol).all(-1)
        if not bool(coplanar.any()):
            continue
        ac, bc, nc = a[coplanar], b[coplanar], normal[coplanar]
        def side(p, q, r):
            return (torch.linalg.cross(q - p, r - p) * nc).sum(-1)
        for i in range(3):
            p, q = ac[:, i], ac[:, (i + 1) % 3]
            for j in range(3):
                r, s = bc[:, j], bc[:, (j + 1) % 3]
                # Require both endpoints to be strictly off the other edge.
                # Product-only tests amplify shared-endpoint roundoff when the
                # other orientation is large, spuriously crossing conforming
                # coplanar children after uniform refinement.
                def opposite(first,second):
                    return ((first > tol) & (second < -tol)) | ((first < -tol) & (second > tol))
                crossing = opposite(side(p,q,r),side(p,q,s))
                crossing &= opposite(side(r,s,p),side(r,s,q))
                if bool(crossing.any()):
                    return True
        for source, target in ((ac, bc), (bc, ac)):
            for k in range(3):
                signs = torch.stack([side(target[:, j], target[:, (j + 1) % 3], source[:, k]) for j in range(3)], -1)
                if bool(((signs > tol).all(-1) | (signs < -tol).all(-1)).any()):
                    return True
                # Coplanar T contacts have neither a strict crossing nor an
                # interior vertex. Reject vertices in another edge's interior.
                for edge in range(3):
                    a, b = target[:, edge], target[:, (edge+1) % 3]
                    direction = b-a
                    fraction = ((source[:, k]-a)*direction).sum(-1)/direction.square().sum(-1)
                    projection = a+fraction[:, None]*direction
                    on_edge = (source[:, k]-projection).abs().amax(-1) <= tol
                    if bool((on_edge & (fraction > tol) & (fraction < 1-tol)).any()):
                        return True
    return False


def _reference_has_coincident_vertices(vertices, tolerance):
    vertex_ids = torch.arange(len(vertices), device=vertices.device)
    for start in range(0, len(vertices), 128):
        delta = vertices[start:start+128, None]-vertices[None]
        close = delta.abs().amax(-1) <= tolerance
        distinct = vertex_ids[start:start+128, None] < vertex_ids[None]
        if bool((close & distinct).any()):
            return True
    return False


def _reference_minimum(a, b):
    minimum = torch.tensor(torch.inf, dtype=a.dtype, device=a.device)
    for start in range(0, len(a)*len(b), 4096):
        ids = torch.arange(start, min(start+4096, len(a)*len(b)), device=a.device)
        minimum = torch.minimum(minimum, triangle_pair_distances(a[ids//len(b)], b[ids % len(b)]).min())
    return minimum


def _sphere(radius=1., frequency=6, centre=(0., 0., 0.), dtype=torch.float64):
    mesh = sphere_mesh(radius=radius, frequency=frequency, center=centre, device=DEVICE, dtype=dtype)
    return mesh.vertices, mesh.faces


def _box(size, divisions, centre=(0., 0., 0.), rotation=None):
    from benchmark.mom_large_scale_l1 import _box_surface
    v, f = _box_surface(size, max(size)/divisions, centre)
    v = torch.tensor(v, device=DEVICE, dtype=torch.float64)
    if rotation is not None:
        v = (v-torch.tensor(centre, device=DEVICE, dtype=torch.float64))@rotation.T+torch.tensor(centre, device=DEVICE, dtype=torch.float64)
    return v, torch.tensor(f, device=DEVICE)


def _rotation(seed):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    return (q*torch.sign(torch.linalg.det(q))).to(DEVICE)


def test_overlap_traversal_returns_every_overlapping_pair_once():
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    lower = torch.rand((3000, 3), generator=generator, device=DEVICE, dtype=torch.float64)
    upper = lower+0.02*torch.rand((3000, 3), generator=generator, device=DEVICE, dtype=torch.float64)
    upper[:5] += 0.8  # a few very large boxes stress the fixed-branching hierarchy
    tolerance = 1e-3
    found = torch.cat([a*len(lower)+b for a, b in overlapping_pairs(BoxTree(lower, upper), tolerance)])
    assert len(found.unique()) == len(found)
    overlap = ((lower[:, None] <= upper[None]+tolerance) & (upper[:, None] >= lower[None]-tolerance)).all(-1)
    i, j = overlap.nonzero(as_tuple=True)
    expected = (i*len(lower)+j)[i < j]
    assert torch.equal(found.sort().values, expected.sort().values)


@pytest.mark.parametrize('offset_factor', [0., .5, .999, 1., 1.001, 2., 1e3])
def test_coincident_vertex_decisions_match_all_pairs_near_tolerance(offset_factor):
    vertices, _ = _sphere(frequency=9)
    extent = torch.linalg.vector_norm(vertices.amax(0)-vertices.amin(0))
    tolerance = 128*torch.finfo(vertices.dtype).eps*extent
    planted = vertices.clone()
    direction = torch.tensor([1., -1., 1.], device=DEVICE, dtype=planted.dtype)
    planted = torch.cat((planted, planted[17:18]+offset_factor*tolerance*direction))
    assert _has_coincident_vertices(planted, tolerance) == _reference_has_coincident_vertices(planted, tolerance)
    assert _has_coincident_vertices(vertices, tolerance) is _reference_has_coincident_vertices(vertices, tolerance) is False


def _soup(*meshes):
    return torch.cat([v[f] for v, f in meshes])


def _intersection_cases():
    sphere = _sphere(frequency=7)
    cases = {'closed_sphere': _soup(sphere), 'cube': _soup(_box((1., 1., 1.), 6)),
             'rotated_cube': _soup(_box((1., .7, .4), 5, rotation=_rotation(1)))}
    radius = 1.
    for name, gap in (('spheres_overlap', -0.05), ('spheres_touch', 0.), ('spheres_gap_tiny', 1e-15),
                      ('spheres_gap_1e-13', 1e-13), ('spheres_gap', 0.05)):
        other = _sphere(frequency=7, centre=(2*radius+gap, 0., 0.))
        cases[name] = _soup(sphere, other)
    # Coplanar faces of two cubes: face-to-face contact, partial overlap and a T arrangement.
    for name, shift in (('cubes_face_contact', (1., 0., 0.)), ('cubes_partial_overlap', (.95, .3, .2)),
                        ('cubes_coplanar_offset', (1., .5, 0.)), ('cubes_separated', (1.001, .5, 0.))):
        cases[name] = _soup(_box((1., 1., 1.), 4), _box((1., 1., 1.), 3, centre=shift))
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    base = torch.rand((400, 1, 3), generator=generator, device=DEVICE, dtype=torch.float64)
    cases['random_soup'] = base+0.05*torch.rand((400, 3, 3), generator=generator, device=DEVICE, dtype=torch.float64)
    cases['graded_with_large_triangle'] = torch.cat((_soup(sphere), torch.tensor(
        [[[-3., -3., 0.], [3., -3., 0.], [0., 3., 0.]]], device=DEVICE, dtype=torch.float64)))
    return cases


@pytest.mark.parametrize('name', (
    'closed_sphere', 'cube', 'rotated_cube',
    'spheres_overlap', 'spheres_touch', 'spheres_gap_tiny', 'spheres_gap_1e-13', 'spheres_gap',
    'cubes_face_contact', 'cubes_partial_overlap', 'cubes_coplanar_offset', 'cubes_separated',
    'random_soup', 'graded_with_large_triangle'))
def test_self_intersection_decisions_match_all_pairs(name):
    triangles = _intersection_cases()[name]
    assert _has_intersections(triangles) == _reference_has_intersections(triangles)


def test_intersection_cases_cover_both_decisions():
    decisions = {name: _reference_has_intersections(t) for name, t in _intersection_cases().items()}
    assert not decisions['closed_sphere'] and not decisions['spheres_gap']
    assert decisions['spheres_overlap'] and decisions['random_soup'] and decisions['cubes_partial_overlap']


def _separation_cases():
    left = _sphere(frequency=6)
    cases = {}
    for gap in (0., 1e-14, 1e-12, 1e-6, .01, .5, 3.):
        cases[f'spheres_{gap:g}'] = (left, _sphere(.7, frequency=5, centre=(1.7+gap, .1, -.05)))
    cases['facing_cubes'] = (_box((1., 1., 1.), 5), _box((1., 1., 1.), 4, centre=(1.25, 0., 0.)))
    cases['facing_cubes_aligned_grid'] = (_box((1., 1., 1.), 4), _box((1., 1., 1.), 4, centre=(1.25, 0., 0.)))
    cases['rotated_cubes'] = (_box((1., 1., 1.), 4, rotation=_rotation(2)),
                              _box((.6, .9, .3), 4, centre=(1.3, .2, .1), rotation=_rotation(5)))
    cases['edge_near_face'] = (_box((1., 1., 1.), 3), _box((1., 1., 1.), 3, centre=(1.+1e-9, .5, .5),
                                                           rotation=_rotation(7)))
    cases['float32_spheres'] = (_sphere(frequency=5, dtype=torch.float32),
                                _sphere(.5, frequency=4, centre=(1.6, 0., 0.), dtype=torch.float32))
    return cases


@pytest.mark.parametrize('name', (
    *(f'spheres_{gap:g}' for gap in (0., 1e-14, 1e-12, 1e-6, .01, .5, 3.)),
    'facing_cubes', 'facing_cubes_aligned_grid', 'rotated_cubes', 'edge_near_face', 'float32_spheres'))
def test_body_minimum_distance_equals_all_pairs(name):
    (va, fa), (vb, fb) = _separation_cases()[name]
    a, b = va[fa], vb[fb]
    expected = _reference_minimum(a, b)
    actual = _minimum_triangle_distance(a, b)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected), (float(actual), float(expected))


def test_surface_system_still_rejects_touching_bodies():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh, SurfaceSystem
    left = SurfaceMesh(*_sphere(.5, frequency=3))
    right = SurfaceMesh(*_sphere(.5, frequency=3, centre=(.9, 0., 0.)))
    medium = object()
    with pytest.raises(ValueError, match='touch or intersect'):
        SurfaceSystem.create([(0, None, RWGBasis(left), medium), (1, None, RWGBasis(right), medium)])
    far = SurfaceMesh(*_sphere(.5, frequency=3, centre=(1.5, 0., 0.)))
    system = SurfaceSystem.create([(0, None, RWGBasis(left), medium), (1, None, RWGBasis(far), medium)])
    expected = float(_reference_minimum(left.triangles, far.triangles))
    assert system.pair_gaps == ((1, 2, expected),)
    del mw
