"""Source-batched lifting and field queries agree with one-source evaluation (CUDA)."""
import json
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom import operator
from witwin.maxwell.mom.carrier import IncidentCarrier
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.lifting import FORMULATION_ID, build_lifted_rhs, build_lifted_rhs_batch
from witwin.maxwell.mom.lifting_fields import decode_system, query_fields_batch
from witwin.maxwell.mom.linalg import StoppingRule, solve_reference
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.quadrature import classify_all_pairs, touching_pair_rule, triangle_rule
from witwin.maxwell.mom.runtime import SurfaceData, _signature

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

FREQUENCY = 6e8
INTEGRATION = IntegrationConfig(order=4, singular_order=4)


def source_groups():
    cw = lambda amplitude=1., phase=0.: mw.CW(frequency=FREQUENCY, amplitude=amplitude, phase=phase)
    waves = [mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0), source_time=cw()),
             mw.PlaneWave(direction=(0.6, 0, 0.8), polarization=(0, 1, 0), source_time=cw(.7, .4)),
             mw.PlaneWave(direction=(0, -1, 0), polarization=(0, 0, 1), source_time=cw(1.3, -1.1))]
    dipole = mw.PointDipole(position=(0., 0.05, 0.25), polarization=(0.3, 1., -0.4), profile='ideal',
                            source_time=cw(2., .3))
    return [(waves[0],), (waves[1],), (waves[2],), (dipole,), (waves[1], dipole)]


def prepared(groups, eps_r=3.5, sigma_e=0.02, subdivisions=2):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=eps_r, sigma_e=sigma_e))],
                     sources=list(groups[0]))
    config = mw.MoMConfig(subdivisions=subdivisions, riesz_storage='dense', integration=INTEGRATION,
                          stopping=StoppingRule(method='direct', rtol=1e-10),
                          query_config=FieldQueryConfig(relative_tolerance=1e-5, max_order=32))
    solver = mw.Simulation.mom(scene, frequency=FREQUENCY, config=config).prepare().solver
    return scene, config, solver


def relative(actual, expected):
    return float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))


def test_batched_lifted_rhs_matches_each_source_and_keeps_identities():
    groups = source_groups()
    scene, config, solver = prepared(groups)
    handle = solver.handle
    rhs, decoders = handle.build_rhs_batch(groups)
    assert rhs.shape == (len(groups), handle.shape[0])
    # The prepared single-source RHS is the first group.
    assert relative(rhs[0], solver.rhs) < 1e-12
    for row, group, decoder in zip(rhs, groups, decoders):
        local, record = build_lifted_rhs(decoder.body_handles[0])
        assert relative(row, local) < 1e-12
        expected = IncidentCarrier.compile(group, handle.exterior)
        canonical = lambda payload: json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        assert canonical(decoder.incident_carrier.payload()) == canonical(expected.payload())
        assert decoder.metadata['incident_carrier_identity'] == expected.identity == record['incident_carrier_identity']
        batched = decoder.metadata['rhs_bodies'][0]
        assert batched['formulation_id'] == record['formulation_id'] == FORMULATION_ID
        assert batched['rhs_source_batch'] == len(groups) and record['rhs_source_batch'] == 1
        near, single = batched['rhs_near_quadrature'], record['rhs_near_quadrature']
        assert near['maximum_order'] == single['maximum_order']
        assert near['pair_evaluations'] == single['pair_evaluations'] > 0
        assert near['maximum_accepted_relative_difference'] == pytest.approx(
            single['maximum_accepted_relative_difference'], rel=1e-6)
    assert handle.metadata['incident_carrier_identity'] == handle.incident_carrier.identity


def test_batched_lifted_rhs_rejects_mixed_geometry_and_keeps_zero_contrast_exact():
    groups = source_groups()[:3]
    _, _, solver = prepared(groups, eps_r=1., sigma_e=0.)
    rhs, _ = solver.handle.build_rhs_batch(groups)
    assert torch.count_nonzero(rhs) == 0
    _, _, other = prepared(groups, subdivisions=1)
    with pytest.raises(ValueError, match='shared basis'):
        build_lifted_rhs_batch((solver.handle.body_handles[0], other.handle.body_handles[0]))


def _reference_pair_integrals(basis, rows, cols, medium, category, order):
    """Explicit node-pair tensors: the pre-moment formulation."""
    from witwin.maxwell.mom.kernels import FREE_SPACE
    _, free, slopes = operator.local_geometry(basis)
    if category < 3:
        obs, src, weight = touching_pair_rule(free[rows], free[cols], basis.mesh.faces[rows], basis.mesh.faces[cols],
                                              category, order)
        g, grad = FREE_SPACE.evaluate(obs-src, medium.k)
        f = (slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])).to(grad.dtype)
        h = (slopes[cols, None, :, None]*(src[:, :, None]-free[cols, None])).to(grad.dtype)
        vector = torch.einsum('bqlc,bqmc,bq->blm', f, h, g*weight)
        scalar = (g*weight).sum(1)[:, None, None]*(2*slopes[rows, :, None])*(2*slopes[cols, None, :])
        magnetic = torch.einsum('bqlc,bqmc,bq->blm', f, torch.linalg.cross(grad[:, :, None], h, dim=-1),
                                weight.to(grad.dtype))
        return vector-scalar/medium.k**2, magnetic*(category != 0)
    obs, wo = triangle_rule(free[rows], order)
    src, ws = triangle_rule(free[cols], order)
    src, ws = src[:, None], ws[:, None]
    g, grad = FREE_SPACE.evaluate(obs[:, :, None]-src, medium.k)
    weight = wo[:, :, None]*ws
    f = (slopes[rows, None, :, None]*(obs[:, :, None]-free[rows, None])).to(grad.dtype)
    h = (slopes[cols, None, None, :, None]*(src[:, :, :, None]-free[cols, None, None])).to(grad.dtype)
    vector = torch.einsum('bqlc,bqsmc,bqs->blm', f, h, g*weight)
    scalar = (g*weight).sum((1, 2))[:, None, None]*(2*slopes[rows, :, None])*(2*slopes[cols, None, :])
    magnetic = torch.einsum('bqlc,bqsmc,bqs->blm', f, torch.linalg.cross(grad[:, :, :, None], h, dim=-1),
                            weight.to(grad.dtype))
    return vector-scalar/medium.k**2, magnetic


@pytest.mark.parametrize('category', range(5))
def test_shared_media_pair_integrals_match_explicit_node_pair_tensors(category):
    _, _, solver = prepared(source_groups()[:1])
    body = solver.handle.system.bodies[0]
    basis, nf = body.basis, len(body.mesh.faces)
    pairs = torch.nonzero(classify_all_pairs(basis.mesh) == category).flatten()[:96]
    rows, cols = pairs//nf, pairs % nf
    media = (solver.handle.exterior, body.interior)
    if category < 3:
        values = operator._touching_integrals(basis, rows, cols, media, 4, category)
    else:
        values = operator._product_integrals(basis, rows, cols, media, 4, 4)
    for medium, (d, k) in zip(media, values):
        reference_d, reference_k = _reference_pair_integrals(basis, rows, cols, medium, category, 4)
        assert relative(d, reference_d) < 1e-13
        if category:
            assert relative(k, reference_k) < 1e-13
        else:
            assert torch.count_nonzero(k) == 0


def test_batched_field_queries_match_each_source_and_logs():
    groups = source_groups()
    scene, config, solver = prepared(groups)
    handle = solver.handle
    rhs, decoders = handle.build_rhs_batch(groups)
    currents, _ = solve_reference(handle, rhs, config.stopping)
    material = scene.structures[0]
    datas = [SurfaceData(x, decoder, config, _signature(mw.Scene(structures=[material], sources=list(group)),
             handle.exterior, tuple(b.interior for b in handle.system.bodies)))
             for x, decoder, group in zip(currents, decoders, groups)]
    directions = torch.tensor([[.3, .4, .866], [-.6, .64, -.48], [0., 0., -1.]], device='cuda', dtype=torch.float64)
    directions = directions/torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    queries = {('exterior', 'total'): .17*directions, ('exterior', 'scattered'): .12*directions,
               ('exterior', 'incident'): .2*directions, ('interior', 'total'): .05*directions}
    for (region, kind), points in queries.items():
        electric, magnetic = query_fields_batch(datas, points, region, kind)
        batched_logs = [json.dumps(decoder.last_field_query_log, default=str) for decoder in decoders]
        for index, (decoder, x) in enumerate(zip(decoders, currents)):
            e, h = decode_system(decoder, x, points, region, kind, query_config=config.query_config)
            scale = max(float(e.abs().max()), 1e-300)
            assert float((electric[index]-e).abs().max()) <= 1e-12*scale
            assert float((magnetic[index]-h).abs().max()) <= 1e-12*float(h.abs().max())
            single = decoder.last_field_query_log
            batched = json.loads(batched_logs[index])
            assert [q.get('final_order') for q in batched['body_queries']] == [q.get('final_order') for q in single['body_queries']]
            assert [len(q.get('checks', ())) for q in batched['body_queries']] == [len(q.get('checks', ())) for q in single['body_queries']]
            assert batched['converged'] == single['converged']


def test_batched_lifted_rhs_applies_point_dipole_admission():
    groups = source_groups()[:1]
    _, _, solver = prepared(groups)
    close = mw.PointDipole(position=(0., 0., 0.102), polarization=(0., 0., 1.), profile='ideal',
                           source_time=mw.CW(frequency=FREQUENCY))
    with pytest.raises(ValueError):
        solver.handle.build_rhs_batch(((close,),))


def _box_surface(dims, counts, scale=.1):
    """Closed outward-oriented box surface with a structured grid per face."""
    import numpy as np
    keys, vertices, faces = {}, [], []

    def vertex(point):
        key = tuple(np.round(np.asarray(point)/scale*1e9).astype(np.int64))
        if key not in keys:
            keys[key] = len(vertices)
            vertices.append(point)
        return keys[key]

    size = np.asarray(dims, dtype=float)*scale
    for axis in range(3):
        b, c = (axis+1) % 3, (axis+2) % 3
        for side in (0., 1.):
            for u in range(counts[b]):
                for v in range(counts[c]):
                    corner = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        point = np.zeros(3)
                        point[axis] = side*size[axis]
                        point[b], point[c] = (u+du)/counts[b]*size[b], (v+dv)/counts[c]*size[c]
                        corner.append(vertex(point-size/2))
                    quad = [(corner[0], corner[1], corner[2]), (corner[0], corner[2], corner[3])]
                    faces.extend(quad if side else [t[::-1] for t in quad])
    from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh
    return RWGBasis(SurfaceMesh(torch.tensor(np.asarray(vertices), device='cuda', dtype=torch.float64),
                                torch.tensor(faces, device='cuda', dtype=torch.int64)))


def _reference_near(basis, rows, cols, medium, config):
    """The previous per-batch, per-medium near refinement loop."""
    order = config.singular_order
    d, k = _reference_pair_integrals(basis, rows, cols, medium, 3, order)
    pending = torch.arange(len(rows), device=rows.device)
    evaluations, maximum = len(rows), 0.
    while len(pending):
        order = min(order+2, config.near_max_order)
        ud, uk = _reference_pair_integrals(basis, rows[pending], cols[pending], medium, 3, order)
        evaluations += len(pending)
        d_scale = torch.linalg.vector_norm(ud, dim=(-2, -1)).clamp_min(torch.finfo(torch.float64).tiny)
        k_scale = torch.maximum(torch.linalg.vector_norm(uk, dim=(-2, -1)), d_scale*abs(medium.k))
        relative = torch.maximum(torch.linalg.vector_norm(ud-d[pending], dim=(-2, -1))/d_scale,
                                 torch.linalg.vector_norm(uk-k[pending], dim=(-2, -1))/k_scale)
        d[pending], k[pending] = ud, uk
        accepted = relative <= config.near_tolerance
        if bool(accepted.any()):
            maximum = max(maximum, float(relative[accepted].max()))
        pending = pending[~accepted]
    return d, k, {'maximum_order': order, 'pair_evaluations': evaluations, 'maximum_accepted_relative_difference': maximum}


def _reference_assembly(basis, media, config):
    """The previous dense traversal: explicit node-pair tensors per medium and batch."""
    from witwin.maxwell.constants import EPSILON_0, MU_0
    ne, nt = basis.n_edges, len(basis.mesh.faces)
    classes = classify_all_pairs(basis.mesh)
    matrix = torch.zeros((2*ne, 2*ne), device='cuda', dtype=torch.complex128)
    near = {'maximum_order': 0, 'pair_evaluations': 0, 'maximum_accepted_relative_difference': 0.}
    for category in range(5):
        selected = torch.nonzero(classes == category).flatten()
        for ids in selected.split(config.far_pair_batch if category == 4 else config.pair_batch):
            r, c = ids//nt, ids % nt
            ri = basis.face_edges[r, :, None].expand(-1, -1, 3).reshape(-1)
            ci = basis.face_edges[c, None, :].expand(-1, 3, -1).reshape(-1)
            for medium in media:
                if category == 3:
                    d, k, stats = _reference_near(basis, r, c, medium, config)
                    near['maximum_order'] = max(near['maximum_order'], stats['maximum_order'])
                    near['pair_evaluations'] += stats['pair_evaluations']
                    near['maximum_accepted_relative_difference'] = max(
                        near['maximum_accepted_relative_difference'], stats['maximum_accepted_relative_difference'])
                else:
                    d, k = _reference_pair_integrals(basis, r, c, medium, category,
                                                     config.order if category == 4 else config.singular_order)
                omega = 2*math.pi*medium.frequency
                for i, j, values in ((0, 0, k), (1, 1, k), (0, 1, 1j*omega*EPSILON_0*medium.epsilon_r*d),
                                     (1, 0, -1j*omega*MU_0*medium.mu_r*d)):
                    matrix.index_put_((ri+i*ne, ci+j*ne), values.reshape(-1), accumulate=True)
    return matrix, near


@pytest.mark.parametrize('shape', ['cube', 'thin_slab'])
@pytest.mark.parametrize('singular_order', [4, 8])
def test_dense_assembly_matches_previous_node_pair_traversal(shape, singular_order):
    from witwin.maxwell.mom.physics import compile_medium
    basis = _box_surface((1, 1, 1), (3, 3, 3)) if shape == 'cube' else _box_surface((1, 1, .08), (4, 4, 1))
    config = IntegrationConfig(order=4, singular_order=singular_order)
    media = (compile_medium(mw.Material(eps_r=1.), FREQUENCY, background=True),
             compile_medium(mw.Material(eps_r=3.5, sigma_e=.05), FREQUENCY))
    counts = torch.bincount(classify_all_pairs(basis.mesh).long(), minlength=5)
    assert bool((counts > 0).all())
    matrix, metadata = operator._assemble_scalar_regions(basis, media, config, kernel=operator.FREE_SPACE)
    reference, near = _reference_assembly(basis, media, config)
    assert relative(matrix, reference) < 1e-13
    for block in ((slice(0, basis.n_edges), slice(0, basis.n_edges)),
                  (slice(0, basis.n_edges), slice(basis.n_edges, None))):
        assert relative(matrix[block], reference[block]) < 1e-13
    observed = metadata['quadrature_error_estimate']['near_disjoint']
    assert observed['maximum_order'] == near['maximum_order'] > singular_order
    assert observed['pair_evaluations'] == near['pair_evaluations']
    assert observed['maximum_accepted_relative_difference'] == pytest.approx(
        near['maximum_accepted_relative_difference'], rel=1e-6)
