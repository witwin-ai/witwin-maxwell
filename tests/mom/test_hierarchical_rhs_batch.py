"""Compressed backends prepare, solve and batch lifted RHS through the shared decode handle (CUDA)."""
import hashlib
from dataclasses import replace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.carrier import IncidentCarrier
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.fmm_config import DirectionalFMMConfig
from witwin.maxwell.mom.lifting import build_selected_lifted_rhs
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

BACKENDS = ('hmatrix', 'h2', 'direct_h2', 'fmm', 'directional')


def prepared(backend):
    # The scenes and solver settings of the existing public backend controls.
    bodies = [mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)), mw.Material(eps_r=2.)) for x in (-.5, .5)]
    if backend == 'hmatrix':
        bodies = [mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.))]
    scene = mw.Scene(structures=bodies, sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=2 if backend == 'hmatrix' else 3, singular_order=3),
        stopping=StoppingRule(method='gmres', preconditioner='row_column_norm' if backend in ('hmatrix', 'h2') else 'rwg_diagonal',
                              rtol=1e-12, atol=0, maxiter=240, restart=120),
        query_config=FieldQueryConfig(adaptive=False), riesz_storage='local_rwg_v1',
        operator_backend='fmm' if backend == 'directional' else backend,
        hierarchical=mw.HMatrixConfig(pivoting='adaptive') if backend in ('hmatrix', 'h2') else None)
    if backend == 'fmm':
        config = replace(config, fmm=mw.FMMConfig())
    if backend == 'directional':
        config = replace(config, integration=IntegrationConfig(order=6, singular_order=3, pair_batch=4,
                                                               far_pair_batch=16, near_max_order=32),
                         fmm=DirectionalFMMConfig(leaf_size=4, separation_ratio=.15, cone_subdivisions=(2**20,),
                                                  max_contexts=4096, tensor_workspace_budget_bytes=512*1024**2,
                                                  leaf_factor_kernel='original', leaf_factor_max_edges=16))
    return scene, config, mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()


@pytest.mark.parametrize('backend', BACKENDS)
def test_compressed_backend_prepare_solve_and_batched_rhs(backend):
    scene, config, simulation = prepared(backend)
    solver = simulation.solver
    handle = solver.handle
    assert handle.metadata['rhs_identity'] == hashlib.sha256(
        (handle.metadata['operator_identity']+':'+handle.incident_carrier.identity).encode()).hexdigest()
    # The directional DAG solve is covered by its public control; preparation
    # (which builds the lifted RHS) and batching are the subject here.
    if backend != 'directional':
        result = simulation.run()
        assert result.stats()['converged'], result.stats()
    other = mw.PlaneWave(direction=(0., .6, .8), polarization=(1., 0., 0.),
                         source_time=mw.CW(frequency=3e8, amplitude=.5, phase=.3))
    groups = ((scene.sources[0],), (other,), (scene.sources[0], other))
    if backend == 'directional':
        # The directional DAG preparation dominates; keep the batch minimal.
        groups = groups[:2]
    rhs, decoders = handle.build_rhs_batch(groups)
    scale = torch.linalg.vector_norm(solver.rhs)
    assert float(torch.linalg.vector_norm(rhs[0]-solver.rhs)/scale) < 1e-12
    # Coherent groups superpose up to each carrier's own near-quadrature acceptance.
    if len(groups) == 3:
        assert float(torch.linalg.vector_norm(rhs[2]-rhs[0]-rhs[1])/scale) < 1e-6
    system = handle.system
    for row, group, decoder in zip(rhs, groups, decoders):
        expected = IncidentCarrier.compile(group, handle.exterior)
        assert decoder.metadata['incident_carrier_identity'] == expected.identity
        assert decoder.metadata['rhs_identity'] == hashlib.sha256(
            (handle.metadata['operator_identity']+':'+expected.identity).encode()).hexdigest()
        for index, (body, local) in enumerate(zip(system.bodies, decoder.body_handles)):
            if backend == 'directional' and index:
                break
            values, record = build_selected_lifted_rhs(local, torch.arange(body.basis.n_edges, device=row.device))
            n = body.basis.n_edges
            batched = torch.cat((row[system.J_slices[index]], row[system.M_slices[index]]))
            assert float(torch.linalg.vector_norm(batched-values)/torch.linalg.vector_norm(values)) < 1e-12
            assert decoder.metadata['rhs_bodies'][index]['version'] == record['version'] == 'selected_analytic_lifted_rhs_v1'
            assert decoder.metadata['rhs_bodies'][index]['rhs_source_batch'] == len(groups)
            assert record['maximum_classification_pairs'] <= max(config.integration.pair_batch,
                                                                 config.integration.far_pair_batch)
    if backend != 'directional':
        currents, log = handle.solve(rhs[1], config.stopping)
        assert log['converged'], log


def test_batched_rhs_fails_closed_on_periodic_lattices():
    lattice = mw.MoMPeriodicConfig(vectors_m=((1., 0., 0.),), reciprocal_shells=4, quadrature_order=128)
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.))],
                     sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, periodic=lattice, integration=IntegrationConfig(order=2, singular_order=3),
                          query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1')
    handle = mw.Simulation.mom(scene, frequency=3e8, config=config).prepare().solver.handle
    with pytest.raises(NotImplementedError, match='periodic'):
        handle.build_rhs_batch(((scene.sources[0],),))
