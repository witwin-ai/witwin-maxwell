"""Actual scalar PMCHWT storage/solver combinations and dual-device consistency."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.fields import decode
from witwin.maxwell.mom.linalg import StoppingRule, correct, solve_reference
from witwin.maxwell.mom.operator import IntegrationConfig, build_rhs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('failure', ['capacity', 'allocation'])
def test_dense_capacity_and_allocation_failure_allow_fresh_prepare(monkeypatch, failure):
    from witwin.maxwell.mom import runtime
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.25))],
                     sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=3, singular_order=4))
    calls = []
    original = runtime.assemble_system

    def fail_assembly(*args,**kwargs):
        calls.append(True)
        raise torch.cuda.OutOfMemoryError('injected assembly allocation failure')

    with monkeypatch.context() as patch:
        patch.setattr(runtime, 'assemble_system', fail_assembly)
        if failure == 'capacity':
            patch.setattr(torch.cuda, 'mem_get_info', lambda device: (1, 1))
            patch.setattr(torch.cuda, 'memory_reserved', lambda device: 0)
            patch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
        with pytest.raises(torch.cuda.OutOfMemoryError, match='capacity preflight|injected assembly'):
            mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()
        assert len(calls) == (0 if failure == 'capacity' else 1)
    assert runtime.assemble_system is original
    prepared = mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()
    receipt = prepared.solver.handle.metadata['memory_preflight']
    assert receipt['minimum_required_bytes'] <= receipt['available_bytes']
    assert receipt['workspace_included'] is False
    assert receipt['direct_factor_bytes'] == receipt['matrix_bytes']
    result = prepared.run()
    assert result.stats()['converged']


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA devices required')
@pytest.mark.parametrize('dtype', [torch.complex128, torch.complex64])
def test_actual_compact_dense_solver_matrix_and_cross_device(dtype):
    """Compare physical coefficients/fields, not just device labels or residual logs."""
    tol = 2e-7 if dtype == torch.complex128 else 3e-3
    solve_tol = 1e-10 if dtype == torch.complex128 else 2e-5
    snapshots = []
    orientations = [((0, 0, 1), (1, 0, 0)), ((0, 0, 1), (0, 1, 0)),
                    ((0, 0, -1), (1, 0, 0)), ((0, 0, -1), (0, 1, 0)),
                    ((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 1)),
                    ((0, 1, 0), (1, 0, 0)), ((0, 1, 0), (0, 0, 1))]
    for device in ('cuda:0', 'cuda:1'):
        with torch.cuda.device(device):
            scene = mw.Scene(device=device, structures=[mw.Structure(
                mw.Sphere(radius=.1), mw.Material(eps_r=2.25, sigma_e=.004))],
                sources=[mw.PlaneWave()])
            per_storage = []
            for storage in ('dense', 'compact_rwg_v1'):
                config = mw.MoMConfig(subdivisions=0, dtype=dtype, riesz_storage=storage,
                    integration=IntegrationConfig(order=3, singular_order=4))
                op = mw.Simulation.mom(scene, frequency=3e8, config=config).prepare().solver.handle
                sources = [mw.PlaneWave(direction=d, polarization=p) for d, p in orientations]
                rhs = torch.stack([build_rhs(op.basis, op.exterior, [s], order=3) for s in sources])
                oracle = torch.linalg.solve(op.matrix, rhs.mT).mT
                gram = op.basis.mass_matrix()
                metric = torch.block_diag(gram / abs(op.exterior.eta)**2, gram).to(dtype)
                points = torch.tensor([[.3, .2, .4]], device=device, dtype=gram.dtype)
                reference_fields = decode(op, oracle[0], points, 'exterior', 'scattered')
                solutions = []
                for method in ('direct', 'gmres', 'block_gmres'):
                    rule = StoppingRule(method=method, rtol=solve_tol, atol=0., restart=60,
                                        maxiter=180, stagnation_window=120,
                                        preconditioner='row_column_norm')
                    solution, log = (solve_reference(op, rhs, rule) if method == 'direct'
                                     else correct(op, rhs, torch.zeros_like(rhs), rule))
                    assert log['converged'] and not log.get('fallback_used', False), log
                    residual = op.apply(solution) - rhs
                    weighted = torch.linalg.solve(metric, residual.mT).mT
                    norm_r = (residual.conj() * weighted).real.sum(-1).clamp_min(0).sqrt()
                    weighted_b = torch.linalg.solve(metric, rhs.mT).mT
                    norm_b = (rhs.conj() * weighted_b).real.sum(-1).sqrt()
                    assert bool((norm_r / norm_b < tol).all())
                    assert float(torch.linalg.vector_norm(solution-oracle) / torch.linalg.vector_norm(oracle)) < tol
                    for field, reference in zip(decode(op, solution[0], points, 'exterior', 'scattered'), reference_fields):
                        assert float(torch.linalg.vector_norm(field-reference) / torch.linalg.vector_norm(reference)) < tol
                    solutions.append(solution.detach().cpu())
                x, y = oracle, rhs
                left, right = (op.apply(x).conj() * y).sum(), (x.conj() * op.adjoint(y)).sum()
                scale = torch.linalg.vector_norm(op.apply(x)) * torch.linalg.vector_norm(y)
                assert float(abs(left-right) / scale) < (1e-11 if dtype == torch.complex128 else 2e-5)
                per_storage.append([op.matrix.detach().cpu(), gram.detach().cpu(), rhs.detach().cpu(),
                                    *solutions, *(f.detach().cpu() for f in reference_fields)])
            for dense, compact in zip(*per_storage):
                assert float(torch.linalg.vector_norm(dense-compact) / torch.linalg.vector_norm(dense)) < tol
            snapshots.append(per_storage)
    for storage0, storage1 in zip(*snapshots):
        for gpu0, gpu1 in zip(storage0, storage1):
            assert float(torch.linalg.vector_norm(gpu0-gpu1) / torch.linalg.vector_norm(gpu0)) < tol


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('fraction', [.9, .99])
def test_legal_sliver_mesh_storage_residual_adjoint_and_queries(fraction):
    """Resolve legal low-quality panels without silently relaxing quadrature."""
    from witwin.maxwell.mom.mesh import SurfaceMesh, sphere_mesh
    mesh = sphere_mesh(radius=.1, subdivisions=0)
    adjacent = int(mesh.faces[(mesh.faces == 0).any(-1)][0, 1])
    assert adjacent != 0
    vertices = mesh.vertices.clone()
    vertices[0] = (1-fraction)*vertices[0] + fraction*vertices[adjacent]
    surface = SurfaceMesh(vertices, mesh.faces)
    edge_square_sum = (surface.triangles-torch.roll(surface.triangles, 1, 1)).square().sum((-1, -2))
    quality = 4*3**.5*surface.areas/edge_square_sum
    assert 0 < float(quality.min()) < (.16 if fraction == .9 else .016)
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(vertices=vertices, faces=mesh.faces,
        fill_mode='solid', recenter=False), mw.Material(eps_r=2.25, sigma_e=.004))], sources=[mw.PlaneWave()])
    observations = []
    for storage in ('dense', 'compact_rwg_v1'):
        prepared = mw.Simulation.mom(scene, frequency=3e8, config=mw.MoMConfig(riesz_storage=storage,
            integration=IntegrationConfig(order=4, singular_order=6))).prepare()
        result = prepared.run()
        op, rhs = prepared.solver.handle, prepared.solver.rhs
        x = result.surface_currents.scattered_coefficients
        assert result.stats()['converged']
        assert float(torch.linalg.vector_norm(op.apply(x)-rhs)/torch.linalg.vector_norm(rhs)) < 1e-10
        lhs, rhs_dot = torch.vdot(op.apply(x), rhs), torch.vdot(x, op.adjoint(rhs))
        assert float(abs(lhs-rhs_dot)/(torch.linalg.vector_norm(op.apply(x))*torch.linalg.vector_norm(rhs))) < 1e-12
        near = op.metadata['quadrature_error_estimate']['near_disjoint']
        assert op.metadata['triangle_pair_counts'][3] > 0
        assert near['pair_evaluations'] > 0 and near['maximum_order'] > 0
        assert near['maximum_accepted_relative_difference'] <= op.integration.near_tolerance
        assert near['maximum_order'] <= op.integration.near_max_order
        points = torch.tensor([[.3, .2, .4], [-.2, .3, .3]], device=vertices.device, dtype=vertices.dtype)
        fields = result.query_fields(points, region_id='exterior', field_kind='scattered')
        assert all(bool(torch.isfinite(field).all()) for field in fields)
        observations.append((x, *fields))
    for dense, compact in zip(*observations):
        torch.testing.assert_close(dense, compact, rtol=1e-9, atol=1e-12)
