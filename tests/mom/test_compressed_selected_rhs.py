"""Compressed scalar runtimes reuse bounded original analytic RHS integration."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.runtime import SystemDecodeHandle, SystemOperator
from witwin.maxwell.mom.lifting import build_lifted_rhs
from witwin.maxwell.mom.operator_error import measure_original_global_residual

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def compressed_config(backend, integration):
    stopping_parameters = {'preconditioner': 'none'} if backend in ('direct_h2', 'fmm') else {}
    return mw.MoMConfig(operator_backend=backend, integration=integration,
        riesz_storage='local_rwg_v1', stopping=StoppingRule(method='gmres', fallback='none',
                                                       **stopping_parameters))


@pytest.fixture(scope='module')
def original_rhs():
    vertices = torch.tensor([[0., 0., 0.], [.025, 0., 0.], [0., .025, 0.],
        [0., 0., .025]], device='cuda', dtype=torch.float64)
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device='cuda')
    bodies = [(i, str(i), RWGBasis(SurfaceMesh(vertices+vertices.new_tensor([x, 0., 0.]), faces)),
        compile_medium(mw.Material(eps_r=2.+i, mu_r=1.4+.2*i, sigma_e=.01, sigma_m=.02), 3e8))
        for i, x in enumerate((-.3, .3))]
    integration = IntegrationConfig(order=3, singular_order=4, pair_batch=7, far_pair_batch=11)
    system = SurfaceSystem.create(bodies)
    exterior = compile_medium(mw.Material(), 3e8)
    sources = (mw.PlaneWave(direction=(.6, 0., .8), polarization=(0., 1., 0.)),
        mw.PlaneWave(direction=(.6, 0., -.8), polarization=(0., 1., 0.),
                     source_time=mw.CW(frequency=3e8, amplitude=.4)))
    handle = SystemDecodeHandle(system, exterior, sources, mw.MoMConfig(integration=integration), {})
    reference = tuple(build_lifted_rhs(local)[0] for local in handle.body_handles)
    return system, exterior, sources, integration, reference


def diagnostic_handle(original, backend):
    system, exterior, sources, integration, _ = original
    handle = SystemDecodeHandle(system, exterior, sources, compressed_config(backend, integration), {})
    handle.shape = (2*system.total_edges,)*2
    handle.device, handle.dtype = system.device, torch.complex128
    return handle


def test_complete_selected_native_rhs_preserves_magnetic_multibody_rows_and_sources(original_rhs, monkeypatch, tmp_path):
    import witwin.maxwell.mom.lifting as lifting
    def forbidden(*args, **kwargs):
        raise AssertionError('compressed scalar RHS classified all global face pairs')
    monkeypatch.setattr(lifting, 'classify_all_pairs', forbidden)
    handle = diagnostic_handle(original_rhs, 'fmm')
    actual = SystemOperator.build_rhs(handle)
    expected = torch.empty_like(actual)
    for i, local in enumerate(original_rhs[-1]):
        n = handle.system.bodies[i].basis.n_edges
        expected[handle.system.J_slices[i]], expected[handle.system.M_slices[i]] = local[:n], local[n:]
    relative = float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
    assert relative < 2e-12 and bool(torch.isfinite(actual).all())
    for record, body in zip(handle.metadata['rhs_bodies'], handle.system.bodies):
        assert record['version'] == 'selected_analytic_lifted_rhs_v1'
        assert record['selected_edges'] == body.basis.n_edges
        assert record['maximum_classification_pairs'] <= 11
        assert sum(record['triangle_pair_counts']) == len(body.mesh.faces)**2
        assert not record['global_pair_classification_retained'] and not record['global_rhs_error_certified']
    (tmp_path/'rhs.json').write_text(json.dumps(dict(relative_error=relative,
        rhs_bodies=handle.metadata['rhs_bodies'], physical_qualification=False), indent=2)+'\n')


@pytest.mark.parametrize('backend', ['hmatrix', 'h2', 'direct_h2', 'fmm'])
def test_all_free_space_compressed_backends_route_complete_local_selections(original_rhs, monkeypatch, backend):
    import witwin.maxwell.mom.lifting as lifting
    import witwin.maxwell.mom.runtime as runtime
    handle = diagnostic_handle(original_rhs, backend)
    calls = []
    def selected(local, edges):
        i = handle.body_handles.index(local)
        assert torch.equal(edges, torch.arange(local.basis.n_edges, device=handle.device))
        calls.append(i)
        return original_rhs[-1][i], dict(version='cached_original_dispatch_control', global_rhs_error_certified=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('compressed scalar routing reached global RHS')
    monkeypatch.setattr(lifting, 'build_selected_lifted_rhs', selected)
    monkeypatch.setattr(runtime, 'build_lifted_rhs', forbidden)
    actual = SystemOperator.build_rhs(handle)
    assert actual.shape == (24,) and calls == [0, 1]
    assert handle.metadata['operator_backend'] == backend


def test_public_shared_compact_solve_uses_bounded_rhs_and_passes_complete_original_residual(monkeypatch, tmp_path):
    import witwin.maxwell.mom.lifting as lifting
    def forbidden(*args, **kwargs):
        raise AssertionError('public compressed preparation allocated global RHS pair classifications')
    monkeypatch.setattr(lifting, 'classify_all_pairs', forbidden)
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)),
        mw.Material(eps_r=2.+i, mu_r=1.2+.1*i)) for i, x in enumerate((-.2, .2))],
        sources=[mw.PlaneWave()])
    fmm = mw.FMMConfig(order=6, max_order=10, kernel_error_budget=mw.KernelErrorBudget(1e-6, 1e-5),
        kernel_bound_rule='direct_first_wave_min_v1', table_storage='compact_shared_displacement_v1',
        shift_storage='separable_v1', near_storage='scalar_reciprocal_dk_v1', leaf_size=8,
        separation_ratio=.25, leaf_factor_kernel='primary', leaf_factor_max_edges=16)
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=4, singular_order=3),
        operator_backend='fmm', fmm=fmm, riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', fallback='none',
                             rtol=1e-12, atol=0., maxiter=240, restart=120))
    prepared = mw.Simulation.mom(scene, frequency=9e7, config=config).prepare()
    result = prepared.run()
    assert result.stats()['converged'] and result.stats()['error_budget']['checked']
    solver = prepared.solver
    assert all(r['version'] == 'selected_analytic_lifted_rhs_v1'
               and not r['global_pair_classification_retained'] for r in solver.handle.metadata['rhs_bodies'])
    assert any(r['rhs_near_quadrature']['pair_evaluations'] > 0
               for r in solver.handle.metadata['rhs_bodies'])
    assert all(r['rhs_near_quadrature']['maximum_accepted_relative_difference'] <= 1e-5
               for r in solver.handle.metadata['rhs_bodies'])
    receipt = measure_original_global_residual(solver.handle.provider,
        result.surface_currents.scattered_coefficients, solver.rhs, edge_batch=19, source_batch=17)
    assert receipt['all_original_rows_measured'] and receipt['all_rhs_passed']
    assert receipt['relative_tolerance'] == 1e-10 and receipt['absolute_tolerance'] == 0.
    assert not receipt['physical_qualification']
    (tmp_path/'public-residual.json').write_text(json.dumps(receipt, indent=2)+'\n')
